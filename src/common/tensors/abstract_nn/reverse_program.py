"""Reverse proposals for retained :class:`FusedProgram` boundaries.

This is deliberately a differentiable, local inverse rather than a claim that
arbitrary tensor operations are algebraically invertible.  Every terminal
forward value is exposed as a desired-output parameter.  AbstractTensor's
canonical backward registry then supplies a VJP which proposes new values for
the original forward feeds; saved forward values remain explicit incidentals.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, Mapping

from ..abstraction import AbstractTensor as AT, tensor_identity
from ..autograd import autograd
from ..fused_ir import FusedProgram, Meta, OpStep
from .fused_program import ProgramRunner, build_fused_program, capture_backward_program


def retain_uncaptured_outputs(program: FusedProgram) -> FusedProgram:
    """Expose every terminal step result before output-reachability pruning.

    A caller must apply this to the unpruned program: a dependency branch that
    has already been removed cannot be recovered from the remaining IR.
    Existing output names and ids are preserved verbatim.
    """

    consumed = {
        input_id
        for step in program.steps
        for input_id in step.input_ids
    }
    outputs = dict(program.outputs)
    declared = set(outputs.values())
    for step in program.steps:
        if step.result_id in consumed or step.result_id in declared:
            continue
        base = f"uncaptured_{step.step_id}_{step.op_name}"
        name = base
        suffix = 2
        while name in outputs:
            name = f"{base}_{suffix}"
            suffix += 1
        outputs[name] = step.result_id
        declared.add(step.result_id)

    return FusedProgram(
        version=program.version,
        feeds=set(program.feeds),
        steps=list(program.steps),
        outputs=outputs,
        state_in=None if program.state_in is None else set(program.state_in),
        meta=None if program.meta is None else dict(program.meta),
        extras=None if program.extras is None else dict(program.extras),
    )


@dataclass(frozen=True)
class ReverseProgramResult:
    """The two products of reverse replay."""

    proposed_inputs: Dict[str, AT]
    incidentals: Dict[int, AT]


@dataclass(frozen=True)
class ReverseProgramCapture:
    """Executable reverse IR plus its output parameters and incidentals."""

    retained_forward: FusedProgram
    program: FusedProgram
    feed_values: Dict[int, AT]
    output_parameters: Dict[str, int]
    proposed_inputs: Dict[str, int]
    incidental_feed_ids: tuple[int, ...]
    missing_backward: tuple[str, ...]
    tape: Any

    def run(
        self,
        feeds: Mapping[int, AT] | None = None,
        *,
        training: bool = False,
    ) -> ReverseProgramResult:
        """Replay with captured values, optionally overriding any feed."""

        values = dict(self.feed_values)
        if feeds:
            values.update(feeds)
        previous_tape = autograd.tape
        autograd.tape = self.tape
        try:
            outputs = ProgramRunner(self.program)(values, training=training)
        finally:
            autograd.tape = previous_tape
        return ReverseProgramResult(
            proposed_inputs={name: outputs[name] for name in self.proposed_inputs},
            incidentals={feed_id: values[feed_id] for feed_id in self.incidental_feed_ids},
        )


def _feed_name(program: FusedProgram, feed_id: int) -> str:
    origins = (program.extras or {}).get("capture_feed_origins", {})
    origin = origins.get(feed_id, origins.get(str(feed_id), {})) if isinstance(origins, dict) else {}
    if isinstance(origin, dict) and origin.get("binding_name"):
        return str(origin["binding_name"])
    return f"feed_{feed_id}"


def _fresh_id(program: FusedProgram) -> int:
    ids = set(program.feeds) | set(program.outputs.values())
    for step in program.steps:
        ids.update(step.input_ids)
        ids.add(step.result_id)
    return max(ids, default=0) + 1


def capture_reverse_fused_program(
    forward_program: FusedProgram,
    forward_values: Mapping[int, AT],
    target_outputs: Mapping[str, AT],
    *,
    step_size: float = 1.0,
    wrt_feed_ids: Iterable[int] | None = None,
    backward_overrides: Dict[str, Callable[..., Any]] | None = None,
    allow_missing: bool = False,
) -> ReverseProgramCapture:
    """Capture a local reverse proposal for an unpruned forward program.

    ``target_outputs`` must provide one desired value for every declared or
    retained terminal output.  The resulting program treats those desired
    values as parameters, walks their residuals backward through canonical
    AbstractTensor rules, and returns ``current_input - step_size * gradient``
    for each original forward feed.  All other reverse feeds are reported as
    incidentals: values that must be known for that proposal to be valid.
    """

    if step_size < 0:
        raise ValueError("step_size must be non-negative")
    retained = retain_uncaptured_outputs(forward_program)
    expected = set(retained.outputs)
    supplied = set(target_outputs)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(
            f"target_outputs must match every retained output; missing={missing}, extra={extra}"
        )

    absent_outputs = sorted(set(retained.outputs.values()) - set(forward_values))
    if absent_outputs:
        raise KeyError(f"forward_values is missing output ids: {absent_outputs}")
    output_tensors = {
        name: forward_values[value_id]
        for name, value_id in retained.outputs.items()
    }
    tapes = {
        tape
        for value in output_tensors.values()
        if (tape := getattr(value, "_tape", None)) is not None
    }
    if len(tapes) != 1:
        raise ValueError("retained outputs must belong to one forward GradTape")
    forward_tape = next(iter(tapes))

    # Scalars and other saved operands may not be retained by the weak tensor
    # reference table, so recover them from operation contexts as well.
    live_values: Dict[int, AT] = dict(getattr(forward_tape, "_tensor_refs", {}))
    live_values.update(forward_values)
    for node in getattr(forward_tape, "_nodes", {}).values():
        for value in node.ctx.get("inputs", ()):
            if isinstance(value, AT):
                live_values[tensor_identity(value)] = value
        value = node.ctx.get("result")
        if isinstance(value, AT):
            live_values[tensor_identity(value)] = value
    absent_values = sorted(set(retained.feeds) - set(live_values))
    if absent_values:
        raise KeyError(f"forward values are unavailable for feed ids: {absent_values}")
    if wrt_feed_ids is None:
        wrt_ids = sorted(
            feed_id
            for feed_id in retained.feeds
            if bool(getattr(live_values[feed_id], "requires_grad", False))
        )
    else:
        wrt_ids = list(dict.fromkeys(wrt_feed_ids))
        invalid_wrt = sorted(set(wrt_ids) - set(retained.feeds))
        if invalid_wrt:
            raise ValueError(f"wrt_feed_ids are not forward feeds: {invalid_wrt}")
    if not wrt_ids:
        raise ValueError("no differentiable forward feeds were selected for reverse capture")

    previous_tape = autograd.tape
    autograd.tape = forward_tape
    objective_baseline = set(forward_tape.graph.nodes)
    parameter_values: Dict[int, AT] = {}
    output_parameters: Dict[str, int] = {}
    objective: AT | None = None
    try:
        for name, actual in output_tensors.items():
            target = target_outputs[name].detach()
            target._tape = forward_tape
            forward_tape.create_tensor_node(target)
            if tuple(actual.shape) != tuple(target.shape):
                raise ValueError(
                    f"target output {name!r} has shape {tuple(target.shape)}, "
                    f"expected {tuple(actual.shape)}"
                )
            target_id = tensor_identity(target)
            parameter_values[target_id] = target
            output_parameters[name] = target_id
            residual = actual - target
            term = (residual * residual).sum() * 0.5
            objective = term if objective is None else objective + term

        if objective is None:  # FusedProgram permits this, but no inverse does.
            raise ValueError("forward program has no outputs to reverse")
        objective_nodes = set(forward_tape.graph.nodes) - objective_baseline
        wrt = tuple(live_values[feed_id] for feed_id in wrt_ids)
        backward = capture_backward_program(
            objective,
            wrt,
            backward_overrides=backward_overrides,
            allow_missing=allow_missing,
            output_prefix="reverse_gradient",
        )
        objective_boundary = set(objective_nodes)
        for node_id in tuple(objective_nodes):
            objective_boundary.update(forward_tape.graph.predecessors(node_id))
        objective_program = build_fused_program(
            forward_tape.graph.subgraph(objective_boundary).copy(),
            outputs={"reverse_objective": tensor_identity(objective)},
        )
    finally:
        autograd.tape = previous_tape

    # Backward capture normally turns saved forward values into feeds.  Prefix
    # the target-dependent residual operations and remove their results from
    # that feed boundary so target changes remain live at replay time.
    combined_steps = list(objective_program.steps) + list(backward.program.steps)
    combined_steps = [replace(step, step_id=index) for index, step in enumerate(combined_steps)]
    produced = {step.result_id for step in combined_steps}
    combined_meta = dict(objective_program.meta or {})
    combined_meta.update(backward.program.meta or {})
    program = FusedProgram(
        version=max(objective_program.version, backward.program.version),
        feeds=(set(objective_program.feeds) | set(backward.program.feeds)) - produced,
        steps=combined_steps,
        outputs=dict(backward.program.outputs),
        state_in=backward.program.state_in,
        meta=combined_meta,
        extras=backward.program.extras,
    )
    feeds = set(program.feeds) | set(retained.feeds)
    feed_values = dict(backward.feed_values)
    feed_values.update(parameter_values)
    feed_values.update({feed_id: live_values[feed_id] for feed_id in retained.feeds})
    feed_values.update({tensor_identity(value): value for value in output_tensors.values()})
    steps = list(program.steps)
    outputs: Dict[str, int] = {}
    meta = {} if program.meta is None else dict(program.meta)
    next_id = _fresh_id(program)

    for index, feed_id in enumerate(wrt_ids):
        proposal_name = f"proposed_{_feed_name(retained, feed_id)}"
        gradient_id = program.outputs.get(f"reverse_gradient_{index}")
        result_id = feed_id
        if gradient_id is not None and step_size != 0:
            scaled_id = gradient_id
            if step_size != 1:
                scaled_id = next_id
                next_id += 1
                steps.append(OpStep(
                    step_id=len(steps), op_name="mul", input_ids=[gradient_id],
                    attrs={"right_scalar": float(step_size)}, result_id=scaled_id,
                ))
                meta[scaled_id] = replace(meta[gradient_id]) if gradient_id in meta else Meta()
            result_id = next_id
            next_id += 1
            steps.append(OpStep(
                step_id=len(steps), op_name="sub", input_ids=[feed_id, scaled_id],
                result_id=result_id,
            ))
            meta[result_id] = replace(meta[feed_id]) if feed_id in meta else Meta()
        outputs[proposal_name] = result_id

    reverse_program = FusedProgram(
        version=program.version,
        feeds=feeds,
        steps=steps,
        outputs=outputs,
        state_in=program.state_in,
        meta=meta,
        extras=program.extras,
    )
    target_ids = set(output_parameters.values())
    incidentals = tuple(sorted(feeds - target_ids))
    return ReverseProgramCapture(
        retained_forward=retained,
        program=reverse_program,
        feed_values=feed_values,
        output_parameters=output_parameters,
        proposed_inputs=dict(outputs),
        incidental_feed_ids=incidentals,
        missing_backward=backward.missing_backward,
        tape=forward_tape,
    )


__all__ = [
    "ReverseProgramCapture",
    "ReverseProgramResult",
    "capture_reverse_fused_program",
    "retain_uncaptured_outputs",
]

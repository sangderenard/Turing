"""Render Turing's compiler-owned control IR for accelerated targets.

The numerical body is deliberately absent from this module.  Scheduled-region
markers in :mod:`src.compiler.control_source` are composition sites: callers
provide target-specific region bodies, while this module reproduces the
planner's sequence, loop, state-machine, validation, publication, and nested
call structure for C, LLVM SSA, or GLSL.

This does not replace the established deployment emitters.  It codifies one
auditable target matrix for the control side of the compiler IR so future
accelerated backends can share the same structural contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Iterable, Mapping, Sequence

from ....compiler.control_source import (
    CallBlock,
    ControlBlock,
    ControlProgram,
    ControlTarget,
    LoopBlock,
    ParallelDeployment,
    RegionCode,
    SequenceBlock,
    StateMachineTick,
    StatementBlock,
    StreamPublishBlock,
    ValidationBlock,
    render_control_block,
)


class AcceleratedControlTarget(str, Enum):
    C = "c"
    LLVM_SSA = "llvm_ssa"
    GLSL = "glsl"
    # A Fortran launch environment.  Compatible whenever the control structure
    # is loops, sequences and state ticks over already-lowered region bodies --
    # which is what the planner produces.  Its value is that Fortran arrays
    # cannot alias, so the region bodies need no aliasing assertions.
    FORTRAN = "fortran"


@dataclass(frozen=True)
class AcceleratedControlSource:
    target: AcceleratedControlTarget
    function_name: str
    source: str
    region_indices: tuple[int, ...]


def _loop_inductions(block: ControlBlock) -> set[str]:
    """Every loop induction variable in a control tree.

    Fortran has no in-statement declaration, so a `do` variable must be
    declared in the subroutine body before use.
    """

    found: set[str] = set()
    if isinstance(block, LoopBlock):
        found.add(str(block.induction))
        found |= _loop_inductions(block.body)
    elif isinstance(block, SequenceBlock):
        for child in block.blocks:
            found |= _loop_inductions(child)
    elif isinstance(block, StateMachineTick):
        for _value, body in block.cases:
            found |= _loop_inductions(body)
    elif isinstance(block, ParallelDeployment):
        for lane in block.lanes:
            found |= _loop_inductions(lane)
    elif isinstance(block, CallBlock):
        found |= _loop_inductions(block.callee)
    return found


def _region_index(block: StatementBlock) -> int | None:
    if len(block.lines) != 1:
        return None
    match = re.fullmatch(r"__scheduled_region_(\d+)__", block.lines[0])
    return None if match is None else int(match.group(1))


def _region_codes(
    program: ControlProgram,
    target: ControlTarget,
    region_bodies: Mapping[int, Sequence[str]],
) -> tuple[RegionCode, ...]:
    expected = tuple(program.region_indices)
    missing = set(expected) - set(region_bodies)
    extra = set(region_bodies) - set(expected)
    if missing or extra:
        raise ValueError(
            "control region bodies do not match the program: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return tuple(
        RegionCode(
            region_index=index,
            target=target,
            body=StatementBlock(tuple(region_bodies[index])),
        )
        for index in expected
    )


def _render_c_or_glsl(
    program: ControlProgram,
    target: AcceleratedControlTarget,
    *,
    function_name: str,
    region_bodies: Mapping[int, Sequence[str]],
) -> AcceleratedControlSource:
    control_target = {
        AcceleratedControlTarget.C: ControlTarget.C,
        AcceleratedControlTarget.GLSL: ControlTarget.GLSL,
        AcceleratedControlTarget.FORTRAN: ControlTarget.FORTRAN,
    }[target]
    selected = {
        region.region_index: region.body
        for region in _region_codes(program, control_target, region_bodies)
    }
    consumed: list[int] = []

    def substitute(block: ControlBlock) -> ControlBlock:
        if isinstance(block, StatementBlock):
            index = _region_index(block)
            if index is None:
                return block
            consumed.append(index)
            return selected[index]
        if isinstance(block, SequenceBlock):
            return SequenceBlock(tuple(substitute(child) for child in block.blocks))
        if isinstance(block, LoopBlock):
            return LoopBlock(
                block.induction,
                block.start,
                block.stop,
                block.step,
                substitute(block.body),
                block.carried_aliases,
                block.parallel_iterations,
                block.dispatch_shell,
                block.recursion_region_id,
                block.schedule_preference,
            )
        if isinstance(block, StateMachineTick):
            return StateMachineTick(
                block.state,
                tuple((value, substitute(body)) for value, body in block.cases),
            )
        if isinstance(block, ParallelDeployment):
            return ParallelDeployment(
                tuple(substitute(lane) for lane in block.lanes),
                block.schedule_preference,
            )
        if isinstance(block, CallBlock):
            return CallBlock(
                block.callsite_id,
                substitute(block.callee),
                block.argument_bindings,
                block.result_bindings,
            )
        if isinstance(block, (ValidationBlock, StreamPublishBlock)):
            return block
        raise TypeError(f"unknown compiler control block {type(block).__name__}")

    composed = ControlProgram(
        root=substitute(program.root),
        region_indices=program.region_indices,
        uniforms=program.uniforms,
        value_aliases=program.value_aliases,
        iterable_bindings=program.iterable_bindings,
        static_iterable_bindings=program.static_iterable_bindings,
        collection_bindings=program.collection_bindings,
        closure_iterable_bindings=program.closure_iterable_bindings,
    )
    if tuple(consumed) != tuple(program.region_indices):
        raise ValueError(
            "control body traversal differs from the planned region order: "
            f"expected={program.region_indices!r}, consumed={tuple(consumed)!r}"
        )
    lines = render_control_block(composed.root, control_target)
    if target is AcceleratedControlTarget.FORTRAN:
        # Fortran declares loop variables up front and closes with a named
        # `end subroutine` rather than a brace.  bind(C) keeps the launch
        # environment callable through the same shell ABI as every other
        # target.
        induction = sorted(_loop_inductions(composed.root))
        declarations = (
            [f"    integer :: {', '.join(induction)}"] if induction else []
        )
        source = "\n".join(
            (
                f'subroutine {function_name}() '
                f'bind(C, name="{function_name}")',
                "    use, intrinsic :: iso_c_binding",
                "    implicit none",
                *declarations,
                *(f"    {line}" if line else "" for line in lines),
                f"end subroutine {function_name}",
                "",
            )
        )
    else:
        prefix = "void"
        source = "\n".join(
            (
                (
                    f"{prefix} {function_name}(void) {{"
                    if target is AcceleratedControlTarget.C
                    else f"{prefix} {function_name}() {{"
                ),
                *(f"    {line}" if line else "" for line in lines),
                "}",
                "",
            )
        )
    return AcceleratedControlSource(
        target=target,
        function_name=function_name,
        source=source,
        region_indices=tuple(program.region_indices),
    )


class _LLVMControlRenderer:
    def __init__(
        self,
        program: ControlProgram,
        function_name: str,
        region_bodies: Mapping[int, Sequence[str]],
    ):
        self.program = program
        self.function_name = function_name
        self.region_bodies = {
            int(index): tuple(lines)
            for index, lines in region_bodies.items()
        }
        expected = set(program.region_indices)
        if set(self.region_bodies) != expected:
            raise ValueError(
                "control region bodies do not match the program: "
                f"missing={sorted(expected - set(self.region_bodies))}, "
                f"extra={sorted(set(self.region_bodies) - expected)}"
            )
        self.blocks: dict[str, list[str]] = {}
        self.order: list[str] = []
        self.counter = 0
        self.i64_arguments: list[str] = []
        self.i1_arguments: list[str] = []
        self.region_declarations: set[int] = set()
        self.needs_validation = False
        self.needs_stream = False

    def fresh(self, stem: str) -> str:
        self.counter += 1
        return f"{stem}.{self.counter}"

    def block(self, label: str) -> list[str]:
        if label not in self.blocks:
            self.blocks[label] = []
            self.order.append(label)
        return self.blocks[label]

    @staticmethod
    def _identifier(value: str) -> str:
        cleaned = re.sub(r"\W", "_", value)
        if not cleaned or cleaned[0].isdigit():
            cleaned = "control_" + cleaned
        return cleaned

    def i64(self, expression: str) -> str:
        expression = str(expression)
        if re.fullmatch(r"[-+]?\d+", expression):
            return expression
        name = self._identifier(expression)
        if name not in self.i64_arguments:
            self.i64_arguments.append(name)
        return f"%{name}"

    def i1_value(self, value_id: int) -> str:
        name = f"value_{int(value_id)}"
        if name not in self.i1_arguments:
            self.i1_arguments.append(name)
        return f"%{name}"

    def emit_sequence(
        self,
        blocks: Iterable[ControlBlock],
        entry: str,
        exit_label: str,
    ) -> None:
        children = tuple(blocks)
        if not children:
            self.block(entry).append(f"br label %{exit_label}")
            return
        current = entry
        for index, child in enumerate(children):
            next_label = (
                exit_label
                if index == len(children) - 1
                else self.fresh("sequence")
            )
            self.emit(child, current, next_label)
            current = next_label

    def emit(self, block: ControlBlock, entry: str, exit_label: str) -> None:
        lines = self.block(entry)
        if isinstance(block, StatementBlock):
            index = _region_index(block)
            if index is None:
                for line in block.lines:
                    lines.append(str(line))
            else:
                self.region_declarations.add(index)
                for line in self.region_bodies[index]:
                    lines.append(str(line))
            lines.append(f"br label %{exit_label}")
            return
        if isinstance(block, SequenceBlock):
            self.emit_sequence(block.blocks, entry, exit_label)
            return
        if isinstance(block, LoopBlock):
            header = self.fresh("loop.header")
            body = self.fresh("loop.body")
            latch = self.fresh("loop.latch")
            loop_number = self.counter
            induction = self._identifier(block.induction)
            lines.append(f"br label %{header}")
            header_lines = self.block(header)
            header_lines.extend(
                (
                    f"%{induction}.{loop_number} = phi i64 "
                    f"[ {self.i64(block.start)}, %{entry} ], "
                    f"[ %{induction}.next.{loop_number}, %{latch} ]",
                    f"%loop.condition.{loop_number} = icmp slt i64 "
                    f"%{induction}.{loop_number}, {self.i64(block.stop)}",
                    f"br i1 %loop.condition.{loop_number}, "
                    f"label %{body}, label %{exit_label}",
                )
            )
            self.emit(block.body, body, latch)
            self.block(latch).extend(
                (
                    f"%{induction}.next.{loop_number} = add i64 "
                    f"%{induction}.{loop_number}, {self.i64(block.step)}",
                    f"br label %{header}",
                )
            )
            return
        if isinstance(block, StateMachineTick):
            state = self.i64(block.state)
            case_labels = [
                (value, self.fresh("state.case"))
                for value, _body in block.cases
            ]
            switch = [f"switch i64 {state}, label %{exit_label} ["]
            switch.extend(
                f"  i64 {self.i64(value)}, label %{label}"
                for (value, label) in case_labels
            )
            switch.append("]")
            lines.extend(switch)
            for (_value, body), (_case_value, label) in zip(
                block.cases, case_labels
            ):
                self.emit(body, label, exit_label)
            return
        if isinstance(block, ParallelDeployment):
            self.emit_sequence(block.lanes, entry, exit_label)
            return
        if isinstance(block, CallBlock):
            self.emit(block.callee, entry, exit_label)
            return
        if isinstance(block, ValidationBlock):
            self.needs_validation = True
            failure = self.fresh("validation.failure")
            expected = "true" if block.expect_true else "false"
            predicate = self.i1_value(block.predicate_value_id)
            lines.extend(
                (
                    f"%validation.bad.{self.counter} = icmp ne i1 "
                    f"{predicate}, {expected}",
                    f"br i1 %validation.bad.{self.counter}, "
                    f"label %{failure}, label %{exit_label}",
                )
            )
            self.block(failure).extend(
                (
                    f"call void @turing_validation_error("
                    f"i32 {int(block.error_code)})",
                    f"br label %{exit_label}",
                )
            )
            return
        if isinstance(block, StreamPublishBlock):
            self.needs_stream = True
            count = -1 if block.count_value_id is None else int(
                block.count_value_id
            )
            final = "true" if block.final else "false"
            call = (
                f"call void @turing_stream_publish("
                f"i32 {int(block.stream_id)}, i64 {int(block.value_id)}, "
                f"i64 {count}, i1 {final})"
            )
            if block.predicate_value_id is None:
                lines.extend((call, f"br label %{exit_label}"))
            else:
                publish = self.fresh("stream.publish")
                lines.append(
                    f"br i1 {self.i1_value(block.predicate_value_id)}, "
                    f"label %{publish}, label %{exit_label}"
                )
                self.block(publish).extend((call, f"br label %{exit_label}"))
            return
        raise TypeError(f"unknown compiler control block {type(block).__name__}")

    def render(self) -> AcceleratedControlSource:
        entry = "entry"
        exit_label = "control.exit"
        self.emit(self.program.root, entry, exit_label)
        self.block(exit_label).append("ret void")
        parameters = [
            *(f"i64 %{name}" for name in self.i64_arguments),
            *(f"i1 %{name}" for name in self.i1_arguments),
        ]
        declarations = []
        if self.needs_validation:
            declarations.append("declare void @turing_validation_error(i32)")
        if self.needs_stream:
            declarations.append(
                "declare void @turing_stream_publish(i32, i64, i64, i1)"
            )
        declarations.extend(
            f"declare void @turing_region_{index}()"
            for index in sorted(self.region_declarations)
        )
        body_lines = []
        for label in self.order:
            body_lines.append(f"{label}:")
            body_lines.extend(f"  {line}" for line in self.blocks[label])
        source = "\n".join(
            (
                'source_filename = "turing.control-ir"',
                *declarations,
                "",
                f"define void @{self.function_name}({', '.join(parameters)}) {{",
                *body_lines,
                "}",
                "",
            )
        )
        return AcceleratedControlSource(
            target=AcceleratedControlTarget.LLVM_SSA,
            function_name=self.function_name,
            source=source,
            region_indices=tuple(self.program.region_indices),
        )


def reduce_control_ir(
    program: ControlProgram,
    target: AcceleratedControlTarget,
    *,
    function_name: str = "turing_control",
    region_bodies: Mapping[int, Sequence[str]] | None = None,
) -> AcceleratedControlSource:
    """Render one compiler control program without touching numerical IR."""

    if not function_name.isidentifier():
        raise ValueError(f"invalid accelerated control function {function_name!r}")
    bodies = dict(region_bodies or {})
    if not bodies:
        if target is AcceleratedControlTarget.LLVM_SSA:
            bodies = {
                index: (f"call void @turing_region_{index}()",)
                for index in program.region_indices
            }
        elif target is AcceleratedControlTarget.FORTRAN:
            bodies = {
                index: (f"call turing_region_{index}()",)
                for index in program.region_indices
            }
        else:
            bodies = {
                index: (f"turing_region_{index}();",)
                for index in program.region_indices
            }
    if target in {
        AcceleratedControlTarget.C,
        AcceleratedControlTarget.GLSL,
        AcceleratedControlTarget.FORTRAN,
    }:
        return _render_c_or_glsl(
            program,
            target,
            function_name=function_name,
            region_bodies=bodies,
        )
    if target is AcceleratedControlTarget.LLVM_SSA:
        return _LLVMControlRenderer(
            program, function_name, bodies
        ).render()
    raise TypeError(f"unknown accelerated control target {target!r}")


def reduce_control_ir_all_targets(
    program: ControlProgram,
    *,
    function_name: str = "turing_control",
    region_bodies: Mapping[
        AcceleratedControlTarget, Mapping[int, Sequence[str]]
    ] | None = None,
) -> Mapping[AcceleratedControlTarget, AcceleratedControlSource]:
    """Render the same compiler-owned structure for all accelerated targets."""

    bodies = dict(region_bodies or {})
    return {
        target: reduce_control_ir(
            program,
            target,
            function_name=function_name,
            region_bodies=bodies.get(target),
        )
        for target in AcceleratedControlTarget
    }


__all__ = [
    "AcceleratedControlSource",
    "AcceleratedControlTarget",
    "reduce_control_ir",
    "reduce_control_ir_all_targets",
]

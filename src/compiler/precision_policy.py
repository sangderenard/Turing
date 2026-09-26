"""Compile-time precision policy for the AbstractTensor stage.

Optional.  When the compiler produces a law's AbstractTensor source
(``vehicle_python_compilation.symbolic_abstract_tensor_source``), a
``PrecisionPolicy`` may be supplied.  The compiler then decides ONCE, at
compile time, which values of the law need to be computed wide, and emits
exactly that: ``Precision.of`` where a wide section takes a base-width
operand, and one ``.collapse()`` -- a single rounding -- where base-width code
or the return consumes it.  Nothing is decided per operator at run time; the
emitted program simply is the program with its precision sections in it, and
``ir_identities.apply_precision_pipeline`` lowers those sections as it lowers
any authored ``Precision`` code.

How the plan is decided (``plan_precision``): the law is materialized twice
-- base width, and with every eligible value wide -- with every value
captured, and both run on the policy's sample inputs.  A value's error is
its base result's distance from its wide result, in ulps of the base dtype.
Values past ``policy.ulps`` are the culprits; the operators that produced
them are ranked in the plan's ledger.  The wide section is each culprit and
the cone of eligible values upstream of it: bits already lost in an earlier
rounding cannot be recovered by widening only the step that exposes the loss
(a cancelling subtraction), so the section starts where the loss began.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

#: The default automatic-search ceiling.  This is a cost policy, not an ABI
#: or arithmetic limit: authored Precision[n] sections may use any positive
#: width, and callers may explicitly ask the planner to search beyond this
#: conservative default.
MAX_LIMBS = 4

#: SSA operations a wide section may contain.  Native and eager are the same
#: set: every operation the Precision wrapper evaluates wide is one the
#: compiled precision pipeline lowers.
WIDE_OPERATIONS_BY_TARGET = {
    "native": frozenset({"Add", "Sub", "Mul", "Div", "Neg", "Pow", "Sqrt", "Exp", "Log"}),
    "eager": frozenset({"Add", "Sub", "Mul", "Div", "Neg", "Pow", "Sqrt", "Exp", "Log"}),
}
WIDE_OPERATIONS = WIDE_OPERATIONS_BY_TARGET["eager"]


@dataclass(frozen=True)
class PrecisionPolicy:
    """The configurable standard, and the inputs it is measured on.

    ``ulps``     the largest error, in ulps of the base dtype, a value may
                 carry before its section is computed wide -- and the error
                 the returned values must meet once it is
    ``limbs``    the narrowest width tried for a wide section (2 =
                 double-double); planning escalates to ``max_limbs`` when
                 that width does not bring the outputs within ``ulps``
    ``samples``  argument name -> representative input values
    """

    samples: Mapping[str, Any]
    ulps: float = 1.0
    limbs: int = 2
    max_limbs: int = MAX_LIMBS
    target: str = "native"

    def __post_init__(self):
        if self.ulps < 0.5:
            raise ValueError("a standard below half an ulp cannot be met by any rounding")
        if int(self.limbs) < 2:
            raise ValueError("a wide section needs at least two limbs")
        if self.target not in WIDE_OPERATIONS_BY_TARGET:
            raise ValueError(f"target is one of {sorted(WIDE_OPERATIONS_BY_TARGET)}")
        if int(self.max_limbs) < int(self.limbs):
            raise ValueError("max_limbs must be greater than or equal to limbs")


#: Standards of error acceptability, in ulps of binary64, strictest first.
#: ``exact`` asks for the correctly rounded value; ``faithful`` for one of the
#: two doubles bracketing it; the rest are named tolerances a law may declare.
PRECISION_TIERS = {
    "exact": 0.5,
    "faithful": 1.0,
    "tight": 4.0,
    "engineering": float(2 ** 20),      # ~1e-10 relative
    "visual": float(2 ** 29),           # ~6e-8 relative, single-precision-like
}


def precision_policies(samples: Mapping[str, Any], *, limbs: int = 2,
                       max_limbs: int = MAX_LIMBS, target: str = "native") -> dict:
    """One PrecisionPolicy per tier of PRECISION_TIERS, on the same samples."""
    return {name: PrecisionPolicy(samples=samples, ulps=ulps, limbs=limbs, max_limbs=max_limbs,
                                  target=target)
            for name, ulps in PRECISION_TIERS.items()}


@dataclass
class PrecisionPlan:
    """Which values are computed wide, and what the measurement found."""

    limbs: int
    wide_ids: frozenset
    value_ulps: dict = field(default_factory=dict)      # result id -> base error in ulps
    ledger: tuple = ()                                   # (operation, max ulps, culprit count)
    output_ulps: float = 0.0     # the planned program's worst output error vs the reference
    met: bool = True             # output_ulps within the standard
    at_ceiling: bool = False     # the plan used the caller's max_limbs policy bound
    tried: tuple = ()            # (limbs, output ulps) for every width tried

    def receipt(self) -> dict:
        return {"schema": "precision-plan-v2", "limbs": int(self.limbs),
                "wide_values": sorted(int(i) for i in self.wide_ids),
                "ledger": [list(row) for row in self.ledger],
                "output_ulps": float(self.output_ulps), "met": bool(self.met),
                "at_ceiling": bool(self.at_ceiling),
                "tried": [list(row) for row in self.tried]}


def _instructions(function):
    blocks = list((getattr(function, "blocks", {}) or {}).values())
    if len(blocks) != 1:
        raise NotImplementedError("precision planning is defined for single-block bodies")
    return list(blocks[0].instrs)


def _ulps(base, reference) -> float:
    base = np.asarray(base, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    spacing = np.spacing(np.abs(reference))
    spacing = np.where(spacing > 0.0, spacing, np.finfo(np.float64).tiny)
    distance = np.abs(base - reference)
    finite = np.isfinite(distance)
    return float(np.max(distance[finite] / spacing[finite])) if np.any(finite) else 0.0


def _instrument(statements):
    """Capture every ``tN`` a statement list assigns, by result id."""
    out = []
    for statement in statements:
        out.append(statement)
        if not isinstance(statement, ast.Assign):
            continue
        target = statement.targets[0]
        names = ([target] if isinstance(target, ast.Name)
                 else list(target.elts) if isinstance(target, ast.Tuple) else [])
        for node in names:
            if isinstance(node, ast.Name) and node.id.startswith("t") and node.id[1:].isdigit():
                out.append(ast.parse(f"__capture__[{int(node.id[1:])}] = {node.id}").body[0])
    return out


def _captured(function, parameter_names, samples, plan):
    """Run the law once as the stage would emit it, capturing every value."""
    from ..common.tensors import AbstractTensor
    from ..common.tensors.extended_precision import Precision
    from .ssa_python_materializer import materialize_function_body, materialize_precision_sections

    sections = []
    if plan is not None and plan.wide_ids:
        sections, statements, _ = materialize_precision_sections(
            function, parameter_names=parameter_names, precision_plan=plan,
            section_name="__section__")
    else:
        statements, _ = materialize_function_body(
            function, parameter_names=parameter_names, tensor_vocabulary=True)
    for section in sections:
        section.body = _instrument(section.body)
    law = ast.FunctionDef(
        name="__law__",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in parameter_names],
                           vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=_instrument(statements) or [ast.Pass()], decorator_list=[], returns=None)
    module = ast.fix_missing_locations(ast.Module(body=[*sections, law], type_ignores=[]))
    capture: dict = {}
    namespace = {"AbstractTensor": AbstractTensor, "Precision": Precision,
                 "math": __import__("math"), "__capture__": capture}
    exec(compile(module, "<precision-plan>", "exec"), namespace)
    namespace["__law__"](*(AbstractTensor.tensor(np.asarray(samples[n], dtype=np.float64))
                           for n in parameter_names))
    out = {}
    for value_id, value in capture.items():
        if isinstance(value, Precision):
            value = value.collapse()
        out[value_id] = np.asarray(value.tolist() if hasattr(value, "tolist") else value,
                                   dtype=np.float64)
    return out


def _returned_ids(instructions):
    for instruction in instructions:
        if str(instruction.op) in {"Ret", "ret", "Return", "return"}:
            return tuple(int(value.id) for value in instruction.args)
    return ()


def plan_precision(function, policy: PrecisionPolicy, parameter_names) -> PrecisionPlan:
    """Decide, once, which values of ``function`` are computed wide, and how wide.

    The reference is the whole law at ``policy.max_limbs``.  Culprits are the
    values whose base result is further from it than the standard; the section
    is each culprit and its eligible upstream cone.  Widths are tried from
    ``policy.limbs`` up to ``policy.max_limbs``, and the narrowest one whose
    returned values all come within the standard is kept.  A plan at the
    ceiling is reported as such: there is no wider binary64 reference to hold
    it against.
    """
    parameter_names = tuple(parameter_names)
    missing = set(parameter_names) - set(policy.samples)
    if missing:
        raise ValueError(f"precision policy has no samples for {sorted(missing)}")
    instructions = _instructions(function)
    producer = {int(i.res.id): i for i in instructions if getattr(i, "res", None) is not None}
    operations = WIDE_OPERATIONS_BY_TARGET[policy.target]
    eligible = frozenset(value_id for value_id, i in producer.items()
                         if str(i.op) in operations)
    returned = _returned_ids(instructions)
    ceiling = int(policy.max_limbs)
    base = _captured(function, parameter_names, policy.samples, None)
    # The reference is the whole law as wide as Precision can evaluate it --
    # every operation the eager wrapper widens, at the ceiling -- so a native
    # plan is judged against the truth, not against its own target's limits.
    reference_eligible = frozenset(value_id for value_id, i in producer.items()
                                   if str(i.op) in WIDE_OPERATIONS_BY_TARGET["eager"])
    reference = _captured(function, parameter_names, policy.samples,
                          PrecisionPlan(limbs=ceiling, wide_ids=reference_eligible))

    value_ulps = {value_id: _ulps(base[value_id], reference[value_id])
                  for value_id in eligible if value_id in base and value_id in reference}
    culprits = {value_id for value_id, e in value_ulps.items() if e > float(policy.ulps)}

    section, pending = set(), list(culprits)
    while pending:                                # the culprit and its eligible upstream cone
        value_id = pending.pop()
        if value_id in section or value_id not in eligible:
            continue
        section.add(value_id)
        for argument in producer[value_id].args:
            if int(argument.id) in producer:
                pending.append(int(argument.id))

    worst: dict = {}
    for value_id in culprits:
        op = str(producer[value_id].op)
        peak, count = worst.get(op, (0.0, 0))
        worst[op] = (max(peak, value_ulps[value_id]), count + 1)
    ledger = tuple(sorted(((op, peak, count) for op, (peak, count) in worst.items()),
                          key=lambda row: -row[1]))

    def output_error(captured):
        return max((_ulps(captured[i], reference[i]) for i in returned
                    if i in captured and i in reference), default=0.0)

    if not section:
        error = output_error(base)
        return PrecisionPlan(limbs=int(policy.limbs), wide_ids=frozenset(), value_ulps=value_ulps,
                             ledger=ledger, output_ulps=error, met=error <= float(policy.ulps),
                             tried=((1, error),))
    tried = []
    for limbs in range(int(policy.limbs), ceiling + 1):
        candidate = PrecisionPlan(limbs=limbs, wide_ids=frozenset(section))
        error = output_error(_captured(function, parameter_names, policy.samples, candidate))
        tried.append((limbs, error))
        if error <= float(policy.ulps) or limbs == ceiling:
            return PrecisionPlan(limbs=limbs, wide_ids=frozenset(section), value_ulps=value_ulps,
                                 ledger=ledger, output_ulps=error,
                                 met=error <= float(policy.ulps),
                                 at_ceiling=limbs == ceiling, tried=tuple(tried))
    raise AssertionError("unreachable: the loop returns at the ceiling")

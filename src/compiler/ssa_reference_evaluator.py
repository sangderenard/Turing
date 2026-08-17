"""Execute repository SSA directly, as a reference.

Every backend claims to preserve the meaning of the SSA it renders. Nothing
in this tree could check that claim, so a wrong number left two candidate
owners -- the lowering that produced the SSA, or the emission that rendered
it -- and no way to tell them apart without reading both. That routing
question has been the expensive one in every hard defect here.

This evaluator answers it. It runs the SSA itself, on the same inputs the
ABI binder hands a compiled artifact, so:

* if it reproduces the **artifact's** numbers, the SSA already says what the
  artifact does, and the defect is upstream in lowering/planning;
* if it reproduces the **authored oracle's** numbers, the SSA is right and
  emission changed the meaning.

Deliberately naive. It is a reference, so it favours the obvious reading of
every instruction over any cleverness: no fixed points, no caching, no
reordering, no vectorisation. If this and a backend disagree, the backend
is the thing that needs explaining.

Unsupported instructions raise rather than guessing. A reference that
silently invents a value for an op it does not know is worse than no
reference at all -- that failure mode has already cost this tree a day.

CALIBRATED, and what it now routes
----------------------------------
Verified in ``tests/test_ssa_reference_evaluator.py``:

* on a pure single-block function it reproduces both the authored SymPy
  equations and the compiled LLVM artifact to 1e-9;
* on a synthetic traversal -- loop-carried phi, region call, aggregate
  projection, neighbour address by wrapping arithmetic, array store -- it
  produces the hand-computable answer;
* on the REAL fluid traversal it reproduces the authored SymPy equations
  EXACTLY (max absolute difference 0.0).

So it may now be cited for routing, and its first result is that the fluid
traversal's SSA carries the authored meaning while the compiled LLVM
artifact does not: ssa-vs-oracle is 0.0 and llvm-vs-oracle is 6.66e-03.
The defect is in EMISSION, not in lowering or planning.

The bug that had made this look like an evaluator failure is worth
remembering, because it was silent and it was mine: ``dt`` carries no
per-argument accounting, so a binder that matched it by dtype-and-rank
found two float64 scalar candidates, correctly refused the ambiguity --
and then let it fall through to a scratch fill that bound it to 0.0. Every
other value stayed plausible while every timestep did nothing, which is
indistinguishable from a compiler that drops its flux terms. It is now
identified the way the compiled runtime binder identifies it, through the
callee formal it feeds, by name.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Any, Mapping

import numpy as np


class SSAEvaluationError(RuntimeError):
    """An instruction this evaluator cannot execute faithfully."""


@dataclass
class _Pointer:
    """An address: a container plus the index path taken into it.

    Kept explicit rather than collapsed to a Python reference, because the
    SSA distinguishes "the array" from "a place inside the array", and a
    Store must write through to storage the caller shares.
    """

    container: Any
    path: tuple[Any, ...] = ()

    def read(self) -> Any:
        target = self.container
        if not self.path:
            return target[0] if isinstance(target, list) else target
        if isinstance(target, list):          # aggregate (call result)
            return target[int(self.path[0])]
        array = np.asarray(target)
        return array[self._index(array)]

    def write(self, value: Any) -> None:
        target = self.container
        if isinstance(target, list):
            target[int(self.path[0]) if self.path else 0] = value
            return
        array = np.asarray(target)
        array[self._index(array)] = value

    def _index(self, array: np.ndarray) -> Any:
        path = tuple(int(item) for item in self.path)
        if array.ndim == 0:
            return ()
        if len(path) == 1 and array.ndim > 1:
            # A flat offset into a multidimensional arena.
            return np.unravel_index(path[0], array.shape)
        return path[: array.ndim]


@dataclass
class EvaluationResult:
    returned: tuple[Any, ...]
    values: dict[int, Any] = _field(default_factory=dict)
    steps: int = 0


# The opcode VOCABULARY is not restated here. `ssa_llvm_backend` already
# owns the scalar likeness table -- SSA opcode -> target template -- and a
# second, hand-written list of "ops the evaluator knows" would be exactly
# the drift-prone duplicate this tree keeps paying for. The tables below
# supply only SEMANTICS, keyed to that vocabulary, and `_audit_vocabulary`
# refuses to let the two disagree.
from .hierarchical_plan import (  # noqa: E402
    TENSOR_OPERATION_SCALAR_SPELLING,
)
from .ssa_llvm_backend import _BINARY as _LIKENESS_BINARY  # noqa: E402
from .ssa_llvm_backend import _UNARY as _LIKENESS_UNARY  # noqa: E402

_BINARY = {
    "Add": lambda a, b: a + b,
    "Sub": lambda a, b: a - b,
    "Mul": lambda a, b: a * b,
    "Div": lambda a, b: a / b,
    "Max": lambda a, b: np.maximum(a, b),
    "Min": lambda a, b: np.minimum(a, b),
    "Lt": lambda a, b: a < b,
    "Le": lambda a, b: a <= b,
    "Gt": lambda a, b: a > b,
    "Ge": lambda a, b: a >= b,
    "ULt": lambda a, b: np.uint64(a) < np.uint64(b),
    "ULe": lambda a, b: np.uint64(a) <= np.uint64(b),
    "Eq": lambda a, b: a == b,
    "Ne": lambda a, b: a != b,
    "And": lambda a, b: np.logical_and(a, b),
    "Or": lambda a, b: np.logical_or(a, b),
    "Xor": lambda a, b: np.logical_xor(a, b),
    "LAnd": lambda a, b: np.logical_and(a, b),
    "LOr": lambda a, b: np.logical_or(a, b),
    "BitAnd": lambda a, b: np.bitwise_and(np.int64(a), np.int64(b)),
    "BitOr": lambda a, b: np.bitwise_or(np.int64(a), np.int64(b)),
    "BitXor": lambda a, b: np.bitwise_xor(np.int64(a), np.int64(b)),
    "Shl": lambda a, b: np.left_shift(np.int64(a), np.int64(b)),
    "Shr": lambda a, b: np.right_shift(np.int64(a), np.int64(b)),
    # Floored, matching the likeness table's explicit note that Python's %
    # is floored for floats too and that a bare frem is C sign semantics.
    "Mod": lambda a, b: np.mod(a, b),
    "FloorDiv": lambda a, b: np.floor(np.divide(a, b)),
    "Pow": lambda a, b: np.power(a, b),
}

_UNARY = {
    "Neg": lambda a: -a,
    "Abs": np.abs,
    "Sqrt": np.sqrt,
    "Exp": np.exp,
    "Log": np.log,
    "Sin": np.sin,
    "Cos": np.cos,
    "Floor": np.floor,
    "Ceil": np.ceil,
    "Trunc": np.trunc,
    "Round": np.round,
    "Not": np.logical_not,
    "LNot": np.logical_not,
    "Invert": lambda a: np.bitwise_not(np.int64(a)),
    "SIToFP": lambda a: np.float64(a),
    "SiToFp": lambda a: np.float64(a),
    "UiToFp": lambda a: np.float64(a),
    "FPToSI": lambda a: np.int32(a),
    "FpToSi": lambda a: np.int32(a),
    "FpToUi": lambda a: np.uint32(a),
    "SExt": lambda a: np.int64(a),
    "ZExt": lambda a: np.uint64(a),
}


def _audit_vocabulary() -> tuple[frozenset[str], frozenset[str]]:
    """Compare these semantics against the authoritative likeness table.

    Returns ``(invented, unimplemented)``. ``invented`` must always be
    empty: an opcode this evaluator claims to know but the compiler's own
    table does not is a vocabulary this file made up, and it would silently
    diverge from every backend. ``unimplemented`` is allowed and is simply
    the honest edge of the reference -- those opcodes raise when reached.
    """
    ours = set(_BINARY) | set(_UNARY)
    theirs = set(_LIKENESS_BINARY) | set(_LIKENESS_UNARY)
    return frozenset(ours - theirs), frozenset(theirs - ours)


_INVENTED, _UNIMPLEMENTED = _audit_vocabulary()
if _INVENTED:
    raise ImportError(
        "ssa_reference_evaluator defines semantics for opcodes absent from "
        f"the compiler's own likeness table: {sorted(_INVENTED)}. The table "
        "in ssa_llvm_backend owns the vocabulary; this file may only supply "
        "meanings for what it already lists."
    )


def _declared_formal_names(function: Any) -> dict[str, int]:
    """Formal name -> position, from whichever record the function carries.

    A callee states its own port names; which key it uses depends on how
    it was produced, so all three spellings are read here rather than one
    being assumed present.
    """
    metadata = getattr(function, "metadata", None) or {}
    declared = tuple(metadata.get("argument_names") or ())
    if declared:
        return {str(name): index for index, name in enumerate(declared)}
    formals = [int(argument.id) for argument in getattr(function, "args", ())]
    by_id: dict[int, str] = {}
    for key in ("parameter_names", "value_names"):
        for name, value_id in (metadata.get(key) or ()):
            by_id.setdefault(int(value_id), str(name))
    return {
        by_id[value_id]: position
        for position, value_id in enumerate(formals)
        if value_id in by_id
    }


def bind_program_abi_arguments(
    function: Any,
    *,
    record: Any = None,
    record_parameter: str = "state",
    named: Mapping[str, Any] | None = None,
    scratch: bool = True,
    functions: Mapping[str, Any] | None = None,
) -> tuple[dict[int, Any], tuple[int, ...]]:
    """Bind a function's formals the way the compiled ABI binder does.

    Returns ``(arguments, unbound)``. Binding is by DECLARED IDENTITY --
    ``program_abi_parameter``/``program_abi_field`` accounting, then
    ``value_names`` -- never by position, because a formal list's order is
    an emission detail and pairing by position is how this tree has
    repeatedly bound the wrong value to the right-looking slot.

    ``scratch`` fills the remaining formals with zeros. Those are cells a
    callee writes before anything reads them (a linked function's outputs
    used as in-place storage). They are reported in ``unbound`` regardless,
    so a caller can refuse rather than quietly accept a zero that was never
    meant to be read -- an unbound formal is not the same claim as a zero.
    """
    arguments: dict[int, Any] = {}
    unbound: list[int] = []
    names = {
        str(label): int(value)
        for label, value in (function.metadata.get("value_names") or ())
    }
    by_name = {value: label for label, value in names.items()}
    supplied = dict(named or {})

    for argument in function.args:
        value_id = int(argument.id)
        accounting = argument.accounting or {}
        field = accounting.get("program_abi_field")
        parameter = accounting.get("program_abi_parameter")
        if (
            record is not None
            and parameter == record_parameter
            and field
            and hasattr(record, str(field))
        ):
            held = getattr(record, str(field))
            arguments[value_id] = (
                np.asarray(held, dtype=float) if np.ndim(held) else float(held)
            )
            continue
        label = by_name.get(value_id)
        if label is not None and label in supplied:
            arguments[value_id] = supplied[label]
            continue
        unbound.append(value_id)

    # A declared scalar parameter (dt is the standing example) reaches the
    # frame through parameter_value_abi and may carry NO per-argument
    # accounting of its own. It is still identifiable without guessing:
    # the functions this one calls name their own formals, so the operand
    # feeding a callee formal called "dt" IS this function's dt.
    #
    # This is the rule the compiled runtime binder already uses -- see
    # `_named_stencil_operand` in symbolic_fluid_native_runtime, which
    # reaches for exactly this when dt has no record accounting. Matching
    # by dtype-and-rank instead was tried here and is actively dangerous:
    # two float64 scalars were both candidates, the match was correctly
    # refused as ambiguous, and dt then fell through to the scratch fill
    # and was silently bound to 0.0 -- which does not crash, it simply
    # makes every timestep do nothing while every other value stays
    # plausible.
    for parameter_name in supplied:
        if names.get(parameter_name) in arguments:
            continue
        found: int | None = None
        for block in function.blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) not in {"Call", "call"}:
                    continue
                callee = functions.get(
                    str((instruction.attributes or {}).get("callee") or "")
                ) if functions else None
                if callee is None:
                    continue
                formal_names = _declared_formal_names(callee)
                position = formal_names.get(str(parameter_name))
                if position is None or position >= len(instruction.args):
                    continue
                candidate = int(instruction.args[position].id)
                if candidate in arguments and candidate not in unbound:
                    continue
                found = candidate
                break
            if found is not None:
                break
        if found is not None:
            arguments[found] = supplied[parameter_name]
            if found in unbound:
                unbound.remove(found)

    if scratch:
        for value_id in list(unbound):
            argument = next(
                a for a in function.args if int(a.id) == value_id
            )
            shape = tuple(argument.shape or ())
            dtype = str(argument.dtype or "float64")
            if shape:
                arguments[value_id] = np.zeros(
                    int(np.prod(shape)),
                    dtype=np.int64 if "int" in dtype else float,
                )
            elif dtype in {"unknown", "ptr"}:
                arguments[value_id] = np.zeros(8, dtype=np.int64)
            else:
                arguments[value_id] = 0.0
    return arguments, tuple(unbound)


def _cast_to(payload: Any, dtype: str) -> Any:
    """Produce ``payload`` in ``dtype``, as the backends' casts do.

    Both backends render a cast as "the operand in the RESULT's type" and
    neither consults the operand's own type, so this does not either.
    """
    name = str(dtype).lower()
    if name in {"bool", "i1"}:
        return np.asarray(payload).astype(bool)
    if name in {"int", "int32", "i32"}:
        return np.asarray(payload).astype(np.int32)
    if name in {"int64", "i64", "long"}:
        return np.asarray(payload).astype(np.int64)
    return np.asarray(payload).astype(np.float64)


class SSAReferenceEvaluator:
    """Executes one repository-SSA module."""

    def __init__(self, module: Any, *, step_limit: int = 5_000_000) -> None:
        self.module = module
        self.functions = dict(getattr(module, "functions", {}) or {})
        self.step_limit = int(step_limit)
        self.steps = 0

    # -- public -----------------------------------------------------------

    def run(
        self,
        function_name: str,
        arguments: Mapping[int, Any],
    ) -> EvaluationResult:
        """Execute ``function_name`` with ``arguments`` keyed by SSA value id.

        Array arguments are NOT copied: the SSA mutates caller-owned storage
        through Store, exactly as the compiled ABI does, so the caller sees
        those writes and can compare them against an artifact's.
        """
        function = self.functions.get(function_name)
        if function is None:
            raise SSAEvaluationError(f"no function {function_name!r} in module")
        values: dict[int, Any] = {}
        for value in function.args:
            key = int(value.id)
            if key in arguments:
                values[key] = arguments[key]
        returned = self._execute(function, values)
        return EvaluationResult(
            returned=returned, values=values, steps=self.steps,
        )

    # -- internals --------------------------------------------------------

    def _execute(self, function: Any, values: dict[int, Any]) -> tuple:
        blocks = dict(function.blocks)
        order = list(blocks)
        current = order[0]
        previous: str | None = None
        returned: tuple = ()

        while current is not None:
            block = blocks[current]
            next_block: str | None = None
            for instruction in block.instrs:
                self.steps += 1
                if self.steps > self.step_limit:
                    raise SSAEvaluationError(
                        f"step limit {self.step_limit} exceeded in "
                        f"{function.name}; the SSA may not terminate"
                    )
                operation = str(instruction.op)

                if operation in {"Br", "br"}:
                    next_block = str(instruction.attributes.get("target"))
                    break
                if operation in {"CondBr", "condbr"}:
                    predicate = self._operand(values, instruction.args[0])
                    next_block = str(
                        instruction.attributes["true_target"]
                        if bool(np.asarray(predicate).reshape(-1)[0])
                        else instruction.attributes["false_target"]
                    )
                    break
                if operation in {"Ret", "ret", "Return", "return"}:
                    returned = tuple(
                        self._operand(values, argument)
                        for argument in instruction.args
                    )
                    next_block = None
                    break

                self._step(function, instruction, values, previous)

            else:
                successors = list(getattr(block, "successors", ()) or ())
                next_block = successors[0] if len(successors) == 1 else None

            previous, current = current, next_block
        return returned

    @staticmethod
    def _operation_name(instruction: Any) -> str:
        """The instruction's operation, resolved the way backends resolve it.

        Two conventions meet here and both are already established in the
        tree, so neither is reinvented:

        * ``attributes["tensor_operation"] or op`` is how every backend
          reads an instruction's operation (see ssa_fortran_backend, which
          spells exactly this in several passes);
        * a tensor operation acting on scalars is respelled by the planner
          through ``TENSOR_OPERATION_SCALAR_SPELLING``, and one acting on
          arrays keeps its lowercase name. The two spellings are the same
          operation at different ranks, so reading them through that one
          table is what keeps this evaluator from drifting away from the
          planner's own idea of what an op means.

        NumPy supplies the rank difference for free: the same semantics
        applied to arrays broadcast elementwise, which is precisely what
        the tensor likeness table's ``binary_double``/``unary_double``
        kernels do.
        """
        operation = str(
            (instruction.attributes or {}).get("tensor_operation")
            or instruction.op
        )
        return TENSOR_OPERATION_SCALAR_SPELLING.get(
            operation.casefold(), operation,
        )

    def _step(
        self,
        function: Any,
        instruction: Any,
        values: dict[int, Any],
        previous: str | None,
    ) -> None:
        operation = self._operation_name(instruction)
        result = instruction.res

        if operation in {"Const", "const"}:
            attributes = instruction.attributes
            payload = attributes.get("constant")
            if payload is None:
                payload = attributes.get("value")
            if payload is None and "values" in attributes:
                payload = attributes.get("values")
            values[int(result.id)] = payload
            return

        if operation in {"Phi", "phi"}:
            incoming = tuple(instruction.attributes.get("incoming_blocks") or ())
            if previous is None or previous not in incoming:
                raise SSAEvaluationError(
                    f"phi {int(result.id)} reached from {previous!r}, which is "
                    f"not among {incoming!r}"
                )
            chosen = instruction.args[incoming.index(previous)]
            values[int(result.id)] = self._operand(values, chosen)
            return

        if operation in {"GetElementPtr", "getelementptr"}:
            base = self._value(values, instruction.args[0])
            aggregate_index = instruction.attributes.get("aggregate_index")
            if aggregate_index is not None:
                values[int(result.id)] = _Pointer(base, (int(aggregate_index),))
                return
            path = tuple(
                int(np.asarray(self._operand(values, argument)).reshape(-1)[0])
                for argument in instruction.args[1:]
            )
            container = base.container if isinstance(base, _Pointer) else base
            values[int(result.id)] = _Pointer(container, path)
            return

        if operation in {"Load", "load"}:
            pointer = self._value(values, instruction.args[0])
            values[int(result.id)] = (
                pointer.read() if isinstance(pointer, _Pointer) else pointer
            )
            return

        if operation in {"Store", "store"}:
            payload = self._operand(values, instruction.args[0])
            pointer = self._value(values, instruction.args[1])
            if not isinstance(pointer, _Pointer):
                raise SSAEvaluationError(
                    f"Store target {int(instruction.args[1].id)} is not an "
                    "address"
                )
            pointer.write(payload)
            return

        if operation in {"Call", "call"}:
            self._call(instruction, values)
            return

        if operation in {"Cast", "CastLike", "cast_like"} and instruction.args:
            # Both backends render a cast as "produce the operand in the
            # RESULT's type" -- ssa_llvm_backend loads the operand as
            # `_value_llvm_type(result)`, ssa_fortran_backend converts to
            # the declared result kind. Neither consults the operand's own
            # type, so neither does this.
            payload = self._operand(values, instruction.args[0])
            values[int(result.id)] = _cast_to(
                payload,
                str(
                    (instruction.attributes or {}).get("target_dtype")
                    or getattr(result, "dtype", None)
                    or "float64"
                ),
            )
            return

        if operation in {"Select", "where"} and len(instruction.args) == 3:
            # Select(mask, when_true, when_false), with the same truthiness
            # conversion every target applies to a numeric mask.
            mask = self._operand(values, instruction.args[0])
            when_true = self._operand(values, instruction.args[1])
            when_false = self._operand(values, instruction.args[2])
            values[int(result.id)] = np.where(
                np.asarray(mask).astype(bool), when_true, when_false,
            )
            return

        if operation in _BINARY:
            left = self._operand(values, instruction.args[0])
            right = self._operand(values, instruction.args[1])
            values[int(result.id)] = _BINARY[operation](left, right)
            return

        if operation in _UNARY:
            values[int(result.id)] = _UNARY[operation](
                self._operand(values, instruction.args[0])
            )
            return

        raise SSAEvaluationError(
            f"{function.name}: no reference semantics for {operation!r}. "
            "Add it deliberately rather than letting the evaluator guess."
        )

    def _call(self, instruction: Any, values: dict[int, Any]) -> None:
        callee_name = str(instruction.attributes.get("callee") or "")
        callee = self.functions.get(callee_name)
        if callee is None:
            raise SSAEvaluationError(f"call to unknown function {callee_name!r}")

        # Formals bind positionally to the caller's operands. Arrays bind by
        # reference so a Store inside the callee is visible to the caller,
        # which is what the compiled out-param convention also does.
        inner: dict[int, Any] = {}
        for formal, actual in zip(callee.args, instruction.args):
            inner[int(formal.id)] = self._value(values, actual)

        returned = self._execute(callee, inner)

        output_ids = tuple(map(int, instruction.attributes.get("output_ids", ())))
        if output_ids:
            # Two different callee shapes publish results, and confusing
            # them silently reads the wrong values:
            #
            # * a planner REGION is carved out of the caller and keeps the
            #   caller's numbering, so its outputs are values in its own
            #   body carrying the caller's ids;
            # * a linked FUNCTION has its own numbering entirely and
            #   publishes through Ret. Its Ret args correspond to
            #   `output_ids` POSITIONALLY.
            #
            # Reading `output_ids` out of a linked function's namespace is
            # the classic same-number-different-space error: for the fluid
            # step, 10 of its 11 caller ids happen to exist inside the
            # callee as unrelated values, so the read succeeds and returns
            # nonsense.
            # Prefer the callee's DECLARED output contract over the order
            # its Ret happens to list. `named_outputs` states (name, id)
            # for each published result, so reading through it removes the
            # assumption that Ret order is output order -- an assumption
            # about an emission detail, which is the kind this tree keeps
            # being wrong about. Ret is the fallback, for a callee that
            # declares nothing.
            declared = tuple(callee.metadata.get("named_outputs") or ())
            if declared and len(declared) == len(output_ids):
                published = [
                    inner.get(int(value_id)) for _name, value_id in declared
                ]
            elif returned and len(returned) == len(output_ids):
                published = list(returned)
            else:
                published = [inner.get(value_id) for value_id in output_ids]
            if instruction.res is not None:
                values[int(instruction.res.id)] = list(published)
            for position, value_id in enumerate(output_ids):
                if published[position] is not None:
                    # OVERWRITE, never setdefault. These carry the caller's
                    # ids, so on the second and later loop iterations they
                    # are already present -- keeping the first value would
                    # freeze every carried result at its first-iteration
                    # value while the loop appeared to run.
                    values[value_id] = published[position]
            return
        if instruction.res is not None:
            values[int(instruction.res.id)] = (
                returned[0] if len(returned) == 1 else list(returned)
            )

    def _value(self, values: dict[int, Any], value: Any) -> Any:
        """The stored object, address included -- no dereference."""
        key = int(value.id)
        if key in values:
            return values[key]
        raise SSAEvaluationError(
            f"value {key} was read before it was defined; the SSA reached a "
            "use its own definition does not dominate"
        )

    def _operand(self, values: dict[int, Any], value: Any) -> Any:
        """The stored object as an arithmetic operand, dereferenced.

        The repository IR does NOT insert a Load before every use of an
        address: the backends dereference implicitly, because there every
        value lives in a slot and an operand is fetched with
        ``load <ty>, ptr <slot>`` whether or not the SSA said Load. An
        address reaching an Add is therefore ordinary and means "the value
        at that address", not a pointer arithmetic error.

        Reproducing that implicit load here is what keeps this a reference
        for the SSA as the compiler actually interprets it, rather than for
        a stricter IR nobody emits.
        """
        stored = self._value(values, value)
        return stored.read() if isinstance(stored, _Pointer) else stored


__all__ = [
    "SSAEvaluationError",
    "SSAReferenceEvaluator",
    "EvaluationResult",
]

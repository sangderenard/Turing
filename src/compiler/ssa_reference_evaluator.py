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

STATUS: NOT YET TRUSTWORTHY FOR ROUTING. Do not cite its numbers.
----------------------------------------------------------------
It executes the whole advance closure (~22k steps, no shortfall) and its
vocabulary is derived from and audited against the compiler's own scalar
likeness table. But on the fluid program it produces values that match
NEITHER the artifact NOR the authored oracle, which means the remaining
error is in this file, not a finding about the program.

The known gap, and the most likely cause: **this evaluator applies SCALAR
semantics to every instruction.** `ssa_llvm_backend` carries TWO likeness
tables -- the scalar one imported below, and a separate tensor table
(``SSA tensor operation -> authored kernel symbol``). Instructions in this
very module carry tensor attributes (``tensor_operation``, ``tensor``,
``tensor_candidate``): region_0 alone has 21 ``Cast`` plus ``Max``,
``Min`` and ``Log`` so marked. An op that means "elementwise over an
arena" evaluated as though it meant "one scalar" is exactly the kind of
silent disagreement this evaluator exists to detect, and it currently
cannot detect it in itself.

Before this is used to route anything, wire in the tensor table and make a
tensor-attributed instruction either evaluate with tensor semantics or
raise. Until then its output is a work in progress, and the honest reading
of a disagreement is "the evaluator is wrong", not "the compiler is wrong".
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

    def _step(
        self,
        function: Any,
        instruction: Any,
        values: dict[int, Any],
        previous: str | None,
    ) -> None:
        operation = str(instruction.op)
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
            # A region publishes its outputs as values in its OWN body that
            # carry the caller's ids; the caller then projects them out of an
            # aggregate. Reproduce that shape exactly.
            aggregate = [inner.get(value_id) for value_id in output_ids]
            if instruction.res is not None:
                values[int(instruction.res.id)] = aggregate
            for value_id in output_ids:
                if value_id in inner:
                    # OVERWRITE, never setdefault. A region's outputs carry
                    # the caller's own ids, so on the second and later loop
                    # iterations those ids are already present -- keeping the
                    # first value would freeze every carried result at its
                    # first-iteration value while the loop appeared to run.
                    values[value_id] = inner[value_id]
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

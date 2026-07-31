"""Emit Fortran 2008 from Turing SSA.

Fortran is worth targeting for one specific reason, and it is not nostalgia:
**dummy arguments may not alias.**  The standard guarantees it, so a Fortran
compiler vectorizes array loops that an equivalent C or LLVM function cannot,
because there the compiler must assume ``a``, ``b`` and ``out`` might overlap.
The LLVM path in this repository recovers the same freedom only by asserting
``noalias`` explicitly; Fortran gets it from the language.

The emitter targets ``iso_c_binding`` with ``bind(C)``, so generated subroutines
share the calling convention the C and LLVM backends already use and can be
dropped into the same shell ABI.  Arrays are declared ``contiguous`` to state
stride-1 access, which is what allows unit-stride vector loads.

SSA maps onto Fortran cleanly:

* every ``SSAValue`` becomes a local scalar or array temporary;
* a ``BasicBlock`` becomes a labelled section;
* control edges become ``goto``, which Fortran still has and which mirrors
  ``br``/``condbr`` exactly;
* ``Phi`` is eliminated the standard way — assign into the phi's variable in
  each predecessor before branching, rather than inventing a parallel construct.

No Fortran compiler is required to *emit*.  Compilation is a separate, optional
step so the emitter stays usable on machines without gfortran, which is the
common case here.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue

# Fortran intrinsic (or expression template) for each SSA operation.  ``{0}``
# and ``{1}`` are the operand expressions.  Anything absent is reported as an
# unsupported operation rather than guessed at.
_BINARY: dict[str, str] = {
    "Add": "({0} + {1})",
    "Sub": "({0} - {1})",
    "Mul": "({0} * {1})",
    "Div": "({0} / {1})",
    "Pow": "({0} ** {1})",
    "Mod": "modulo({0}, {1})",
    "FloorDiv": "real(floor({0} / {1}), c_double)",
    "Eq": "({0} == {1})",
    "Ne": "({0} /= {1})",
    "Lt": "({0} < {1})",
    "Le": "({0} <= {1})",
    "Gt": "({0} > {1})",
    "Ge": "({0} >= {1})",
    "LAnd": "({0} .and. {1})",
    "LOr": "({0} .or. {1})",
    "And": "iand({0}, {1})",
    "Or": "ior({0}, {1})",
    "Xor": "ieor({0}, {1})",
    "Shl": "shiftl({0}, {1})",
    "Shr": "shiftr({0}, {1})",
    "MatMul": "matmul({0}, {1})",
    # Canonical lowercase spellings used by FusedProgram steps.
    "add": "({0} + {1})",
    "sub": "({0} - {1})",
    "mul": "({0} * {1})",
    "truediv": "({0} / {1})",
    "pow": "({0} ** {1})",
    "maximum": "max({0}, {1})",
    "minimum": "min({0}, {1})",
    "matmul": "matmul({0}, {1})",
    "mod": "modulo({0}, {1})",
    "floordiv": "real(floor({0} / {1}), c_double)",
    "less": "({0} < {1})",
    "less_equal": "({0} <= {1})",
    "greater": "({0} > {1})",
    "greater_equal": "({0} >= {1})",
    "equal": "({0} == {1})",
    "not_equal": "({0} /= {1})",
    "logical_and": "({0} .and. {1})",
    "logical_or": "({0} .or. {1})",
}

_UNARY: dict[str, str] = {
    "Neg": "(-{0})",
    "Abs": "abs({0})",
    "Not": "not({0})",
    "LNot": "(.not. {0})",
    # Numeric conversions. These arrive named after their LLVM opcodes,
    # because the precompile lowering shares an instruction vocabulary with
    # the LLVM backend, but each is an ordinary Fortran conversion intrinsic.
    # SExt/ZExt widen an integer, which in Fortran is just the integer kind
    # the target is already declared with.
    "SExt": "int({0}, c_int)",
    "ZExt": "int({0}, c_int)",
    "Trunc": "int({0}, c_int)",
    "FpToSi": "int({0}, c_int)",
    "FpToUi": "int({0}, c_int)",
    "SiToFp": "real({0}, c_double)",
    "UiToFp": "real({0}, c_double)",
    "FpExt": "real({0}, c_double)",
    "FpTrunc": "real({0}, c_double)",
    "neg": "(-{0})",
    "abs": "abs({0})",
    "sqrt": "sqrt({0})",
    "exp": "exp({0})",
    "log": "log({0})",
    "sin": "sin({0})",
    "cos": "cos({0})",
    "tan": "tan({0})",
    "asin": "asin({0})",
    "acos": "acos({0})",
    "atan": "atan({0})",
    "sinh": "sinh({0})",
    "cosh": "cosh({0})",
    "tanh": "tanh({0})",
    # FLOOR/CEILING/NINT return INTEGER in Fortran, where the numpy
    # equivalents return a float. Keeping the recorded program's type means
    # converting back, which also stops these from poisoning every intrinsic
    # downstream that then sees mixed INTEGER and REAL operands. Fortran
    # assignment converts on the way into an integer variable, so this is
    # safe even when the result is declared integer.
    "floor": "real(floor({0}), c_double)",
    "ceil": "real(ceiling({0}), c_double)",
    "round": "real(nint({0}), c_double)",
    "sign": "sign(1.0_c_double, {0})",
    "logical_not": "(.not. {0})",
    "asinh": "asinh({0})",
    "acosh": "acosh({0})",
    "atanh": "atanh({0})",
    "trunc": "aint({0})",
    "copy": "{0}",
    # Every other backend in the torture matrix reports a comparison as a
    # plain 0.0/1.0 double, not a native boolean -- callers compare outputs
    # with assert_allclose, not a boolean buffer.  Match that at the
    # boundary rather than exporting a LOGICAL dummy no caller expects.
    "bool_to_float64": "merge(1.0_c_double, 0.0_c_double, {0})",
}

# Reductions collapse an array to a scalar; they are emitted as whole-array
# intrinsics rather than explicit loops so the compiler picks the schedule.
_REDUCTION: dict[str, str] = {
    "sum": "sum({0})",
    "prod": "product({0})",
    "max": "maxval({0})",
    "min": "minval({0})",
    "mean": "(sum({0}) / real(size({0}), c_double))",
    "all": "all({0})",
    "any": "any({0})",
}

# Binary operators whose result shape is meant to differ from their operands',
# so conforming the operands to the result would be wrong.
_SHAPE_CHANGING_BINARY = frozenset({"MatMul", "matmul"})

# Operators that take LOGICAL operands natively, so a boolean reaching them
# must be left alone rather than converted to a number.
_LOGICAL_BINARY = frozenset(
    {
        "LAnd", "LOr", "logical_and", "logical_or",
        "And", "Or", "Xor",
        "Eq", "Ne", "equal", "not_equal",
    }
)
# Unary operations that take a LOGICAL operand, so it must not be converted
# to a number on the way in.
_LOGICAL_UNARY = frozenset({"LNot", "Not", "logical_not", "bool_to_float64"})

# Of those, the ones that also *produce* LOGICAL. bool_to_float64 is the
# conversion itself: it consumes a mask and yields a number, so treating its
# result as logical would convert what was just converted.
_LOGICAL_RESULT_UNARY = frozenset({"LNot", "Not", "logical_not"})

# Comparisons yield LOGICAL; everything else here yields a number.
_COMPARISON = frozenset(
    {
        "Eq", "Ne", "Lt", "Le", "Gt", "Ge",
        "equal", "not_equal", "less", "less_equal", "greater",
        "greater_equal",
    }
)

# Operations whose Fortran template requires REAL operands whatever the
# result is declared to be.
# Operations that rearrange values without computing new ones, so the type of
# the result is the type of what went in.
_SHAPE_ONLY = frozenset(
    {"slice", "reshape", "view", "permute", "stack", "concat", "scatter"}
)

_REAL_OPERAND = frozenset(
    {"sign", "floor", "ceil", "round", "trunc", "sqrt", "exp", "log"}
)

_INTEGER_DTYPES = frozenset(
    {"int", "int8", "int16", "int32", "int64", "i32", "i64", "bool", "logical"}
)

_DTYPE_KIND: dict[str, str] = {
    "float64": "real(c_double)",
    "float32": "real(c_float)",
    "double": "real(c_double)",
    "float": "real(c_float)",
    "int64": "integer(c_int64_t)",
    "int32": "integer(c_int32_t)",
    "int": "integer(c_int32_t)",
    "bool": "logical(c_bool)",
}

DEFAULT_DTYPE = "float64"


class FortranEmissionError(ValueError):
    """Raised when an SSA construct has no honest Fortran spelling."""


@dataclass(frozen=True)
class FortranShortfall:
    """One SSA operation the emitter cannot express."""

    op: str
    block: str
    reason: str

    def format(self) -> str:
        return f"{self.op} [{self.block}]: {self.reason}"


@dataclass
class FortranSubroutine:
    """Generated Fortran for one SSA function."""

    name: str
    source: str
    shortfalls: tuple[FortranShortfall, ...] = ()
    # The extent parameters this subroutine declares, in argument order, so a
    # caller in the same module can pass exactly what it expects.
    extent_names: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.shortfalls


def _declaration(value: SSAValue, *, elemental: bool) -> str:
    kind = _DTYPE_KIND.get(value.dtype or DEFAULT_DTYPE)
    if kind is None:
        raise FortranEmissionError(f"no Fortran kind for dtype {value.dtype!r}")
    return kind if elemental else kind


def _name(value: SSAValue) -> str:
    return f"t{value.id}"


def _is_array(value: SSAValue) -> bool:
    return bool(value.shape)


def _element_count(shape: tuple[int, ...]) -> int:
    total = 1
    for size in shape:
        total *= int(size)
    return total


def dimension_extents(values: Iterable[SSAValue]) -> dict[int, str]:
    """Map each distinct array dimension size across ``values`` to a Fortran
    extent parameter name.

    One name per distinct size, not one name per array: two arrays that
    happen to share a dimension size reuse the same parameter, and a matmul
    chain's differing row/inner/column counts each get their own. This is
    the single source of truth for extent naming -- the emitter uses it to
    declare arrays, and any caller building a shim that must pass matching
    extent arguments (fortran_jit_backend.py) uses the exact same function
    so the two never disagree on names or order.
    """

    sizes: dict[int, str] = {}
    for value in values:
        for size in value.shape:
            size = int(size)
            if size not in sizes:
                sizes[size] = f"extent_{size}"
    return sizes


def _broadcast(
    expression: str,
    shape: tuple[int, ...],
    result_shape: tuple[int, ...],
) -> str | None:
    """Expand ``expression`` from ``shape`` to ``result_shape``.

    numpy broadcasting is fully described by the two shapes: dimensions align
    from the right, and any operand dimension of extent one repeats to meet
    the result.  Fortran spells that repetition ``SPREAD``, which inserts a
    dimension rather than stretching one -- so each extent-one dimension is
    first indexed away (removing it) and then spread back at its own
    position with the result's extent.  Working left to right keeps every
    position valid as dimensions are reinserted.
    """

    rank = len(result_shape)
    if len(shape) > rank:
        return None
    # Align from the right, the way numpy does.
    aligned = (1,) * (rank - len(shape)) + tuple(int(size) for size in shape)
    expanding: list[int] = []
    for position, (size, target) in enumerate(zip(aligned, result_shape)):
        if size == int(target):
            continue
        if size != 1:
            return None
        expanding.append(position)
    if not expanding:
        return None

    # Only the operand's *own* extent-one dimensions need indexing away.
    # The dimensions alignment prepended do not exist on it, and SPREAD
    # introduces them; reshaping to the aligned rank first and then indexing
    # those positions back out would be a no-op -- and an illegal one, since
    # the result of RESHAPE is an expression, which Fortran will not
    # subscript.
    offset = rank - len(shape)
    subscripts = [
        "1" if (offset + position) in expanding else ":"
        for position in range(len(shape))
    ]
    result = (
        f"{expression}({', '.join(subscripts)})" if subscripts else expression
    )
    for position in expanding:
        result = (
            f"spread({result}, dim={position + 1}, "
            f"ncopies={int(result_shape[position])})"
        )
    return result


def _array_literal(values: Sequence[Any], shape: tuple[int, ...]) -> str:
    """A Fortran array constructor for an SSA array constant.

    Fortran expresses this natively: ``[a, b, c]`` is an array constructor,
    and ``reshape`` gives it a rank.  A constant whose elements are all equal
    needs neither -- Fortran broadcasts a scalar across a whole array on
    assignment, which is both the shortest source and the form a compiler
    folds best, so an all-``.false.`` mask of 124416 elements stays one token
    instead of 124416.
    """

    elements = tuple(values)
    if not elements:
        raise FortranEmissionError("cannot express an empty array constant")
    if len(set(elements)) == 1:
        return _literal(elements[0])
    constructor = "[" + ", ".join(_literal(element) for element in elements) + "]"
    if len(shape) <= 1:
        return constructor
    extents = ", ".join(str(int(size)) for size in shape)
    return f"reshape({constructor}, [{extents}])"


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return ".true._c_bool" if value else ".false._c_bool"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(float(value))
        if "e" in text or "E" in text:
            mantissa, _, exponent = text.partition("e")
            if "." not in mantissa:
                mantissa += ".0"
            return f"{mantissa}e{exponent}_c_double"
        if "." not in text:
            text += ".0"
        return f"{text}_c_double"
    raise FortranEmissionError(f"cannot express literal {value!r} in Fortran")


class _FunctionEmitter:
    """Translate one SSA :class:`Function` into a Fortran subroutine."""

    def __init__(
        self,
        function: Function,
        *,
        dtype: str = DEFAULT_DTYPE,
        outputs: Sequence[SSAValue] = (),
        callee_extents: Mapping[str, Sequence[str]] | None = None,
    ):
        self.function = function
        self.dtype = dtype
        self.outputs = tuple(outputs)
        self.callee_extents = dict(callee_extents or {})
        self.shortfalls: list[FortranShortfall] = []
        self._branch_targets: set[str] = set()
        # Single-use temporaries are substituted into their consumer so one SSA
        # chain becomes one Fortran array expression.  Emitting a statement per
        # step would materialise an array temporary per step, turning a fused
        # traversal into N passes over memory — which is precisely the cost the
        # whole-array form exists to avoid.
        self._inlined: dict[int, str] = {}
        self._use_sites: dict[int, list[tuple[str, int]]] = {}
        self._locals: dict[int, SSAValue] = {}
        self._phi_targets: dict[str, list[tuple[str, SSAValue, SSAValue]]] = {}
        self._loop_variables: list[str] = []
        # Values emitted as part of a multi-instruction group (a region call,
        # an indexed store) rather than one at a time.
        self._consumed: set[int] = set()
        self._address_producers: dict[int, tuple[SSAValue, SSAValue]] = {}
        self._producers: dict[int, Instr] = {}

    # -- expression construction ------------------------------------------
    def _operand(self, value: SSAValue) -> str:
        inlined = self._inlined.get(value.id)
        return inlined if inlined is not None else _name(value)

    def _collect_use_sites(self) -> None:
        for block in self.function.blocks.values():
            for index, instr in enumerate(block.instrs):
                for argument in instr.args:
                    self._use_sites.setdefault(argument.id, []).append(
                        (block.name, index)
                    )

    def _may_inline(self, instr: Instr, block: BasicBlock) -> bool:
        """Whether this result can be folded into its consumer.

        Only a value used exactly once, inside the same block, and not visible
        outside the subroutine may be substituted.  Crossing a block boundary
        would move the computation across control flow.
        """

        result = instr.res
        if result.id in self._bound_ids:
            return False
        if instr.op in ("Phi", "phi"):
            return False
        uses = self._use_sites.get(result.id, ())
        if len(uses) != 1:
            return False
        if uses[0][0] != block.name:
            return False
        return not self._is_subscripted_by(result, uses[0])

    def _is_subscripted_by(
        self, value: SSAValue, use: tuple[str, int]
    ) -> bool:
        """Whether this value's one consumer will subscript it.

        Fortran subscripts an array designator, never an arbitrary
        expression: ``(a + b)(1, :)`` is a syntax error.  So a value that its
        consumer has to index -- to take a section, or to conform a shape by
        indexing an extent-one dimension away -- must stay a named temporary
        even though it is used once.  Inlining it would be a folding
        optimisation that produces source no compiler accepts.
        """

        block_name, position = use
        block = self.function.blocks.get(block_name)
        if block is None or position >= len(block.instrs):
            return False
        consumer = block.instrs[position]
        operation = (
            consumer.attributes.get("tensor_operation") or consumer.op
        )
        if operation in ("slice", "scatter", "cumsum"):
            return True
        if consumer.res is None:
            return False
        # An elementwise operand of a different shape is conformed by
        # indexing (a broadcast) rather than by an intrinsic call.
        if operation in _BINARY and operation not in _SHAPE_CHANGING_BINARY:
            shape = tuple(value.shape)
            result_shape = tuple(consumer.res.shape)
            if shape and shape != result_shape:
                return True
        return False

    def _structural(
        self, instr: Instr, args: list[str], op: str | None = None
    ) -> str | None:
        """Shape-and-layout ops, as native Fortran array expressions.

        These are not elementwise, so they are absent from the intrinsic
        tables; Fortran still says all of them directly -- array sections,
        array constructors, ``reshape``. Anything whose Fortran form would
        depend on a memory layout this emitter cannot confirm returns None and
        is reported as a shortfall rather than guessed at.
        """

        # The precompile lowering wraps ops in Handler.Call and records the
        # canonical name under "tensor_operation"; the tape JIT builds the
        # instruction with that name as its opcode directly. Both name the
        # same operation, so both resolve here rather than in two emitters.
        operation = instr.attributes.get("tensor_operation") or op or instr.op
        if instr.res is None:
            return None
        attributes = instr.attributes
        shape = instr.res.shape
        rank = len(shape)

        if operation in _REDUCTION and "dim" in attributes and len(args) == 1:
            # Fortran reduces along one dimension natively. Arrays are
            # declared in SSA dimension order, so the axis needs no
            # translation; sum(a, dim=k) drops that dimension, and keepdim
            # asks for it back as an extent of one.
            source_rank = len(instr.args[0].shape)
            axis = (int(attributes["dim"]) % source_rank) + 1
            reduced = _REDUCTION[operation].format(
                f"{args[0]}, dim={axis}"
            )
            if len(shape) == source_rank:
                extents = ", ".join(str(int(size)) for size in shape)
                return f"reshape({reduced}, [{extents}])"
            return reduced

        if operation in ("reshape", "view") and len(args) == 1:
            # A reshape is defined by row-major element order, but Fortran
            # traverses column-major, so both ends need stating.
            #
            # Destination: ORDER=[n..1] makes the last dimension vary
            # fastest, which is the row-major fill.
            #
            # Source: a rank>1 operand's Fortran element order is
            # column-major, which is *not* the order the reshape reads it
            # in. Reversing its dimensions first yields an array whose
            # column-major traversal is the operand's row-major traversal.
            # For a rank-1 operand the two coincide and this collapses away.
            source = args[0]
            source_shape = tuple(instr.args[0].shape)
            source_rank = len(source_shape)
            if source_rank > 1:
                reversed_extents = ", ".join(
                    str(int(size)) for size in reversed(source_shape)
                )
                source_order = ", ".join(
                    str(value) for value in range(source_rank, 0, -1)
                )
                source = (
                    f"reshape({source}, [{reversed_extents}], "
                    f"order=[{source_order}])"
                )
            if rank <= 1:
                return f"reshape({source}, [{_element_count(shape)}])"
            extents = ", ".join(str(int(size)) for size in shape)
            order = ", ".join(str(value) for value in range(rank, 0, -1))
            return f"reshape({source}, [{extents}], order=[{order}])"

        if operation in ("zeros", "full"):
            # A scalar broadcasts across a whole array on assignment.
            fill = attributes.get("fill_value", 0)
            if instr.res.dtype in ("float64", "float32", "double"):
                fill = float(fill)
            return _literal(fill)

        if operation == "arange":
            start = int(attributes.get("start", 0))
            step = int(attributes.get("step", 1))
            end = int(attributes["end"])
            index = self._loop_variable()
            # An implied-do array constructor: the Fortran form of arange,
            # with no materialised element list regardless of length.
            return (
                f"[({index}, {index} = {start}, {end - 1}, {step})]"
            )

        if operation == "slice" and attributes.get("slice_kind") == "index_select":
            # A gather. Fortran applies a vector subscript along one
            # dimension directly, so no loop and no temporary are needed.
            if len(args) != 2:
                return None
            source = instr.args[0]
            source_rank = len(source.shape)
            dim = int(attributes.get("dim", 0)) % source_rank
            subscripts = [":"] * source_rank
            # SSA indices are 0-based; Fortran subscripts start at 1.
            subscripts[dim] = f"{args[1]} + 1"
            return f"{args[0]}({', '.join(subscripts)})"

        if operation == "slice":
            if attributes.get("slice_kind") != "axis" or len(args) != 1:
                return None
            source = instr.args[0]
            source_rank = len(source.shape)
            dim = int(attributes.get("dim", 0))
            start = int(attributes.get("start", 0))
            step = int(attributes.get("step", 1))
            count = int(attributes.get("count", 1))
            # Arrays are declared in SSA dimension order (see dims() in
            # emit()), so Fortran subscript k+1 is SSA dim k.
            axis = (dim % source_rank) + 1
            subscripts = [":"] * source_rank
            if count == 1 and rank == source_rank - 1:
                # A single index drops the rank, matching the result shape.
                subscripts[axis - 1] = str(start + 1)
            else:
                stop = start + count * step
                subscripts[axis - 1] = (
                    f"{start + 1}:{stop}:{step}" if step != 1
                    else f"{start + 1}:{stop}"
                )
            return f"{args[0]}({', '.join(subscripts)})"

        if operation == "permute":
            dims = list(attributes.get("dims") or ())
            if len(dims) != rank or sorted(dims) != list(range(rank)):
                return None
            # RESHAPE fills the result so that its dimension ORDER(k) takes
            # the source's k-th dimension; for a permutation that makes ORDER
            # the inverse of the requested dims. Arrays are declared in SSA
            # dimension order, so this is a plain 1-based inverse with no
            # reversal.
            inverse = [0] * rank
            for position, source_dim in enumerate(dims):
                inverse[source_dim] = position + 1
            order = ", ".join(str(value) for value in inverse)
            extents = ", ".join(str(int(size)) for size in shape)
            return f"reshape({args[0]}, [{extents}], order=[{order}])"

        return None

    def _is_logical(self, value: SSAValue) -> bool:
        """Whether this value's *emitted* expression is LOGICAL.

        The declared dtype alone is not enough. A value can be recorded as
        ``bool`` and still be produced by arithmetic -- numpy makes no
        distinction, so ``mask + 0`` keeps the boolean dtype while plainly
        being a number. Emitting MERGE against that gives a non-LOGICAL mask,
        which Fortran rejects. An inlined value's real type is decided by the
        instruction that produced it.
        """

        producer = self._producers.get(value.id)
        if producer is not None and value.id in self._inlined:
            operation = (
                producer.attributes.get("tensor_operation") or producer.op
            )
            if operation in _SHAPE_ONLY and producer.args:
                return self._is_logical(producer.args[0])
            if (
                operation in _COMPARISON
                or operation in _LOGICAL_BINARY
                or operation in _LOGICAL_RESULT_UNARY
            ):
                return True
            # Produced by arithmetic: a number, whatever the dtype claims.
            return False
        if str(getattr(value, "dtype", "") or "") not in ("bool", "logical"):
            return False
        if producer is None or value.id not in self._inlined:
            # A declared LOGICAL variable, or an argument: the declaration is
            # the truth.
            return True
        operation = (
            producer.attributes.get("tensor_operation") or producer.op
        )
        if operation in _SHAPE_ONLY and producer.args:
            # Rearranging a mask does not stop it being a mask.
            return self._is_logical(producer.args[0])
        return (
            operation in _COMPARISON
            or operation in _LOGICAL_BINARY
            or operation in _LOGICAL_RESULT_UNARY
        )

    def _instruction_is_logical(self, instr: Instr) -> bool:
        """Whether this instruction's expression evaluates to LOGICAL.

        Asked of the instruction rather than its result, because the result
        is about to be assigned to a declared variable and so is never
        inlined -- the value-level test would look at the declaration and
        answer with what we are trying to check.
        """

        operation = instr.attributes.get("tensor_operation") or instr.op
        if operation in _SHAPE_ONLY and instr.args:
            return self._is_logical(instr.args[0])
        return (
            operation in _COMPARISON
            or operation in _LOGICAL_BINARY
            or operation in _LOGICAL_RESULT_UNARY
        )

    def _collect_producers(self) -> None:
        for block in self.function.blocks.values():
            for instr in block.instrs:
                if instr.res is not None:
                    self._producers[instr.res.id] = instr

    def _numeric(self, instr: Instr, args: list[str]) -> list[str]:
        """Give a LOGICAL operand a numeric value where one is required.

        A comparison produces LOGICAL, and the recorded program then uses it
        in arithmetic the way numpy does, where ``True`` is 1.  Fortran keeps
        the types apart and rejects LOGICAL at every numeric intrinsic and
        operator, so the conversion has to be written out -- ``MERGE`` is how
        Fortran says it, and it is what this emitter already uses for a
        boolean leaving through a real buffer.
        """

        operation = instr.attributes.get("tensor_operation") or instr.op
        if operation in _REAL_OPERAND:
            # These templates name a REAL constant of their own (sign) or
            # feed a REAL-only intrinsic, so an integer operand is a kind
            # mismatch regardless of what the result is declared as.
            target_real = True
        elif operation in _COMPARISON:
            # A comparison's result is LOGICAL, which says nothing about how
            # its operands should promote -- they promote to each other.
            target_real = any(
                str(getattr(value, "dtype", None) or self.dtype)
                not in _INTEGER_DTYPES
                for value in instr.args
            )
        else:
            result_dtype = str(getattr(instr.res, "dtype", None) or self.dtype)
            target_real = result_dtype not in _INTEGER_DTYPES
        converted = list(args)
        for position, value in enumerate(instr.args):
            if position >= len(converted):
                continue
            if self._is_logical(value):
                converted[position] = _UNARY["bool_to_float64"].format(
                    converted[position]
                )
                continue
            # numpy promotes mixed integer/real operands silently; Fortran
            # rejects them, so the promotion is written out.
            dtype = str(getattr(value, "dtype", None) or self.dtype)
            operand_real = dtype not in _INTEGER_DTYPES
            if operand_real == target_real:
                continue
            converted[position] = (
                f"real({converted[position]}, c_double)"
                if target_real
                else f"int({converted[position]}, c_int)"
            )
        return converted

    def _conform(self, instr: Instr, args: list[str]) -> list[str] | None:
        """Make an elementwise op's operands conform to its result shape.

        numpy broadcasts, so the recorded program freely combines a
        ``(2304,)`` with a ``(1, 48, 48)`` or a one-element array with a
        whole field.  Fortran's whole-array operators require identical
        shapes, with exactly one exception: a scalar combines with any array.
        Emitting the operands unchanged produces "Incompatible ranks", so
        each is restated at the result's shape.
        """

        if instr.res is None:
            return args
        result_shape = tuple(instr.res.shape)
        result_count = _element_count(result_shape)
        conformed: list[str] = []
        for position, expression in enumerate(args):
            if position >= len(instr.args):
                # A scalar recorded as an attribute rather than an SSA value
                # (right_scalar/left_scalar) was appended to args. It is
                # already a Fortran scalar and conforms to anything.
                conformed.append(expression)
                continue
            value = instr.args[position]
            shape = tuple(value.shape)
            if shape == result_shape or not shape:
                conformed.append(expression)
                continue
            count = _element_count(shape)
            if count == result_count:
                # Same elements, different rank: restate the shape. Both
                # sides are already in this emitter's dimension order, so
                # this is a pure re-description, not a reordering.
                if not result_shape:
                    conformed.append(f"{expression}({', '.join(['1'] * len(shape))})")
                    continue
                extents = ", ".join(str(int(size)) for size in result_shape)
                conformed.append(f"reshape({expression}, [{extents}])")
                continue
            if count == 1:
                # A one-element array acting as a scalar: index it, so
                # Fortran's scalar-to-array broadcast applies.
                subscripts = ", ".join(["1"] * len(shape))
                conformed.append(f"{expression}({subscripts})")
                continue
            broadcast = _broadcast(expression, shape, result_shape)
            if broadcast is None:
                return None
            conformed.append(broadcast)
        return conformed

    def _region_call(
        self, block: BasicBlock, index: int
    ) -> list[str] | None:
        """A scheduled region's call, as one Fortran ``call`` statement.

        The lowering states a region call the way a pointer-based backend
        consumes it: return an aggregate, then walk into it per output.

            %agg = Call region(...)      result_convention='ssa.aggregate'
            %c   = Const <k>
            %ptr = GetElementPtr %agg, %c
            %out = Load %ptr

        Fortran has no pointer arithmetic and returns through arguments, so
        ``GetElementPtr`` has no standalone meaning here -- only the group
        does. Recognising the group turns all of it into one call whose
        outputs are actual arguments, which is what the callee already
        declares (its own ``intent(out)`` dummies).
        """

        instr = block.instrs[index]
        if instr.op not in ("Call", "call"):
            return None
        if instr.attributes.get("result_convention") != "ssa.aggregate":
            return None
        callee = instr.attributes.get("callee")
        if not callee:
            return None
        aggregate = instr.res
        outputs: dict[int, SSAValue] = {}
        consumed: list[SSAValue] = []
        if aggregate is not None:
            addresses: dict[int, int] = {}
            for follower in block.instrs[index + 1:]:
                if follower.res is None:
                    continue
                if follower.op == "GetElementPtr" and follower.args and (
                    follower.args[0].id == aggregate.id
                ):
                    position = follower.attributes.get("aggregate_index")
                    if position is None:
                        return None
                    addresses[follower.res.id] = int(position)
                    consumed.append(follower.res)
                elif follower.op == "Load" and follower.args and (
                    follower.args[0].id in addresses
                ):
                    outputs[addresses[follower.args[0].id]] = follower.res
                    consumed.append(follower.res)
            if not outputs:
                return None
            consumed.append(aggregate)

        ordered_outputs = [outputs[key] for key in sorted(outputs)]
        for value in ordered_outputs:
            self._locals[value.id] = value
        for value in consumed:
            self._consumed.add(value.id)

        # The callee is emitted by this same module, so its argument order is
        # known: its own extents first, then feeds, then outputs. The extents
        # must be the ones it actually declares -- rederiving them from the
        # call site gives a different set whenever the callee has interior
        # arrays whose sizes never appear in its signature.
        declared = self.callee_extents.get(str(callee))
        if declared is not None:
            extents = list(declared)
        else:
            call_values = [*instr.args, *ordered_outputs]
            extents = sorted(dimension_extents(call_values).values())
        arguments = [
            *extents,
            *(self._operand(value) for value in instr.args),
            *(_name(value) for value in ordered_outputs),
        ]
        return [f"    call {callee}({', '.join(arguments)})"]

    def _indexed_store(self, instr: Instr) -> list[str] | None:
        """``collection[i] = value``, without materialising an address.

        A ``GetElementPtr``/``Store`` pair is how a pointer-based backend
        writes one iteration's value into a resident collection.  Fortran
        indexes the array directly, so the address never becomes a value and
        the pair collapses into one assignment.
        """

        if instr.op != "Store" or len(instr.args) != 2:
            return None
        source, address = instr.args
        producer = self._address_producers.get(address.id)
        if producer is None:
            return None
        collection, position = producer
        self._consumed.add(address.id)
        # SSA induction values are 0-based; Fortran subscripts start at 1.
        return [
            f"    {self._operand(collection)}({self._operand(position)} + 1)"
            f" = {self._operand(source)}"
        ]

    def _collect_address_producers(self) -> None:
        """Index every ``GetElementPtr`` that addresses a collection slot."""

        for block in self.function.blocks.values():
            for instr in block.instrs:
                if instr.op != "GetElementPtr" or instr.res is None:
                    continue
                if instr.attributes.get("binding") != "collection_publication":
                    continue
                if len(instr.args) != 2:
                    continue
                self._address_producers[instr.res.id] = (
                    instr.args[0],
                    instr.args[1],
                )

    def _loop_variable(self) -> str:
        """An integer loop index, declared alongside the locals."""

        name = f"i_loop{len(self._loop_variables)}"
        self._loop_variables.append(name)
        return name

    def _statements(self, instr: Instr) -> list[str] | None:
        """Ops that are a Fortran *statement* rather than one expression.

        A running sum or an indexed assignment has no single-expression
        intrinsic form, so returning None from ``_expression`` and reporting a
        shortfall would be wrong -- Fortran expresses both directly, just as
        statements. Everything a single array expression can say stays in
        ``_expression``.
        """

        operation = instr.attributes.get("tensor_operation") or instr.op
        if (
            operation not in ("cumsum", "scatter", "stack", "concat")
            or instr.res is None
        ):
            return None
        target = _name(instr.res)

        if operation == "concat":
            # Concatenation writes each source into its own run of the joined
            # dimension. Fortran has no general concat intrinsic, but a
            # section assignment per source says it exactly, at any rank.
            rank = len(instr.res.shape)
            dim = int(instr.attributes.get("dim", 0)) % rank
            statements = []
            offset = 0
            for argument in instr.args:
                extent = int(argument.shape[dim]) if argument.shape else 1
                subscripts = [":"] * rank
                subscripts[dim] = f"{offset + 1}:{offset + extent}"
                statements.append(
                    f"    {target}({', '.join(subscripts)}) = "
                    f"{self._operand(argument)}"
                )
                offset += extent
            return statements

        if operation == "stack":
            # Stacking adds a new dimension, so each source fills one slice of
            # it -- a sequence of section assignments, which is exactly what
            # Fortran writes. Arrays are declared in SSA dimension order, so
            # the new axis sits where the op says it does.
            rank = len(instr.res.shape)
            dim = int(instr.attributes.get("dim", 0)) % rank
            statements = []
            for position, argument in enumerate(instr.args, start=1):
                subscripts = [":"] * rank
                subscripts[dim] = str(position)
                statements.append(
                    f"    {target}({', '.join(subscripts)}) = "
                    f"{self._operand(argument)}"
                )
            return statements

        if operation == "cumsum":
            source = self._operand(instr.args[0])
            rank = len(instr.res.shape)
            dim = int(instr.attributes.get("dim", 0)) % rank
            # Arrays are declared in SSA dimension order (see dims() in
            # emit()), so Fortran subscript k+1 is SSA dim k.
            axis = dim + 1
            index = self._loop_variable()
            def section(position: str) -> str:
                subscripts = [":"] * rank
                subscripts[axis - 1] = position
                return f"{target}({', '.join(subscripts)})"
            return [
                f"    {target} = {source}",
                f"    do {index} = 2, size({target}, {axis})",
                f"      {section(index)} = {section(index)} + "
                f"{section(f'{index} - 1')}",
                "    end do",
            ]

        # scatter: copy, then assign through the index vector. Fortran applies
        # a vector subscript elementwise, so no loop is needed.
        if len(instr.args) != 3:
            return None
        base, indices, values = (self._operand(a) for a in instr.args)
        if len(instr.res.shape) != 1:
            return None
        return [
            f"    {target} = {base}",
            # SSA indices are 0-based; Fortran subscripts start at 1.
            f"    {target}({indices} + 1) = {values}",
        ]

    def _expression(self, instr: Instr) -> str | None:
        op = instr.op
        if op in ("Call", "call"):
            # precompile_to_ssa.lower_fused_program_to_ssa wraps almost
            # every tensor op in Handler.Call (it names the C/LLVM kernel
            # symbol as "callee" for those backends), but it preserves the
            # original canonical op name under "tensor_operation" precisely
            # so a target that doesn't dispatch by callee symbol -- this one
            # -- can still recognise the operation. Without this, every
            # instruction coming from that lowering path reports as an
            # unsupported "Call" shortfall, even ops this emitter already
            # knows how to express (add, sin, sum, matmul, ...).
            tensor_operation = instr.attributes.get("tensor_operation")
            if tensor_operation is not None:
                op = str(tensor_operation)
        args = [self._operand(a) for a in instr.args]
        constant = instr.attributes.get("constant", None)

        if op in ("Const", "const"):
            if constant is None and "values" in instr.attributes:
                # An array constant carries its elements under "values", not
                # the scalar "constant" key.  Reading only "constant" here
                # yielded None and reported "cannot express literal None",
                # which named the missing key rather than the real content.
                return _array_literal(
                    instr.attributes["values"], instr.res.shape
                )
            if constant is None and instr.attributes.get("value") is not None:
                # Control-flow scalars (loop bounds, strides) are recorded
                # under "value" rather than "constant".  This is checked after
                # "values" on purpose: an array constant carries both keys,
                # with a vestigial "value" of None, so testing it first would
                # discard the real elements.
                return _literal(instr.attributes["value"])
            return _literal(constant)

        structural = self._structural(instr, args, op)
        if structural is not None:
            return structural

        # Scalar operands recorded as attributes rather than SSA values.
        right = instr.attributes.get("right_scalar")
        left = instr.attributes.get("left_scalar")
        if right is not None and len(args) == 1:
            args = [args[0], _literal(right)]
        elif left is not None and len(args) == 1:
            args = [_literal(left), args[0]]

        if op in _REDUCTION and len(args) == 1:
            if op == "sum" and self._is_logical(instr.args[0]):
                # Summing a mask counts its true elements, which Fortran
                # states directly; SUM rejects a LOGICAL argument outright.
                return f"count({args[0]})"
            args = self._numeric(instr, args)
            return _REDUCTION[op].format(*args)
        if op in _BINARY and len(args) == 2:
            template = _BINARY[op]
            # Shape first, then type. Conforming subscripts an operand to
            # index away an extent-one dimension, and Fortran will not
            # subscript the result of an intrinsic -- so converting first
            # would produce real(x, c_double)(1), which is a syntax error.
            if op not in _SHAPE_CHANGING_BINARY:
                # Only elementwise operators require conforming shapes.
                # matmul's operands are meant to differ from its result.
                conformed = self._conform(instr, args)
                if conformed is None:
                    return None
                args = conformed
            if op not in _LOGICAL_BINARY:
                args = self._numeric(instr, args)
            if instr.attributes.get("reverse"):
                args = [args[1], args[0]]
            # A comparison landing in a numeric variable is converted at the
            # assignment, and one inlined into a numeric context is converted
            # by _numeric on its consumer. Converting here as well would
            # wrap what was already wrapped.
            return template.format(*args)
        if op in _UNARY and len(args) == 1:
            if op not in _LOGICAL_UNARY:
                args = self._numeric(instr, args)
            return _UNARY[op].format(*args)
        if op in ("Select", "where") and len(args) == 3:
            return f"merge({args[1]}, {args[2]}, {args[0]})"
        return None

    # -- statement emission ------------------------------------------------
    def _emit_block(self, block: BasicBlock, body: list[str]) -> None:
        body.append(f"    ! block {block.name}")
        # Only emit a statement label where something actually branches to it;
        # an unreferenced label is a compiler warning and pure noise.
        if block.name in self._branch_targets:
            body.append(f"{self._label(block.name)} continue")
        for index, instr in enumerate(block.instrs):
            if instr.op in ("Phi", "phi"):
                # Phi is realised by the predecessors, not here.
                continue
            if instr.op in ("Br", "br"):
                target = str(instr.attributes.get("target", ""))
                self._emit_phi_copies(block.name, target, body)
                body.append(f"    goto {self._label(target)}")
                continue
            if instr.op in ("CondBr", "condbr"):
                # Both spellings occur: the control lowering emits
                # true_target/false_target, hand-built SSA uses true/false.
                # An unrecognised name silently resolved to the first block,
                # so every branch went to the same wrong label.
                true_target = str(
                    instr.attributes.get("true")
                    or instr.attributes.get("true_target")
                    or ""
                )
                false_target = str(
                    instr.attributes.get("false")
                    or instr.attributes.get("false_target")
                    or ""
                )
                condition = self._operand(instr.args[0])
                body.append(f"    if ({condition}) then")
                self._emit_phi_copies(block.name, true_target, body, indent=6)
                body.append(f"      goto {self._label(true_target)}")
                body.append("    else")
                self._emit_phi_copies(block.name, false_target, body, indent=6)
                body.append(f"      goto {self._label(false_target)}")
                body.append("    end if")
                continue
            if instr.op in ("Ret", "ret", "Return", "return"):
                body.append("    return")
                continue

            if instr.res is not None and instr.res.id in self._consumed:
                # Already emitted as part of a multi-instruction group.
                continue
            if instr.res is not None and instr.res.id in self._address_producers:
                # The address is folded into the subscript of the assignment
                # its Store becomes, so it never needs to exist as a value.
                continue

            group = self._region_call(block, index)
            if group is not None:
                body.extend(group)
                continue

            store = self._indexed_store(instr)
            if store is not None:
                body.extend(store)
                continue

            statements = self._statements(instr)
            if statements is not None:
                self._locals[instr.res.id] = instr.res
                body.extend(statements)
                continue

            expression = self._expression(instr)
            if expression is None:
                self.shortfalls.append(
                    FortranShortfall(
                        instr.op,
                        block.name,
                        "no Fortran intrinsic or expression is registered",
                    )
                )
                body.append(f"    ! UNSUPPORTED {instr.op}")
                continue
            if self._may_inline(instr, block):
                self._inlined[instr.res.id] = expression
                continue
            self._locals[instr.res.id] = instr.res
            if (
                self._instruction_is_logical(instr)
                and str(instr.res.dtype or self.dtype)
                not in ("bool", "logical")
            ):
                # A mask reaching a numeric variable. Fortran will not
                # convert LOGICAL on assignment the way it converts between
                # numeric kinds, so it is written out.
                expression = _UNARY["bool_to_float64"].format(expression)
            elif (
                not self._instruction_is_logical(instr)
                and str(instr.res.dtype or self.dtype) in ("bool", "logical")
            ):
                # And the reverse: a number recorded as bool, reaching a
                # variable declared LOGICAL. Non-zero is true, which is the
                # rule numpy applied when it produced the value.
                expression = f"(({expression}) /= 0)"
            body.append(f"    {_name(instr.res)} = {expression}")

    def _emit_phi_copies(
        self,
        source: str,
        target: str,
        body: list[str],
        *,
        indent: int = 4,
    ) -> None:
        pad = " " * indent
        for predecessor, result, incoming in self._phi_targets.get(target, ()):
            if predecessor != source:
                continue
            body.append(f"{pad}{_name(result)} = {_name(incoming)}")

    def _label(self, block_name: str) -> int:
        ordered = list(self.function.blocks)
        return 1000 + (ordered.index(block_name) if block_name in ordered else 0)

    def _collect_branch_targets(self) -> None:
        for block in self.function.blocks.values():
            for instr in block.instrs:
                if instr.op in ("Br", "br"):
                    self._branch_targets.add(
                        str(instr.attributes.get("target", ""))
                    )
                elif instr.op in ("CondBr", "condbr"):
                    # Must recognise the same spellings _emit_block does, or
                    # a branch is emitted to a label that is never placed.
                    self._branch_targets.add(
                        str(
                            instr.attributes.get("true")
                            or instr.attributes.get("true_target")
                            or ""
                        )
                    )
                    self._branch_targets.add(
                        str(
                            instr.attributes.get("false")
                            or instr.attributes.get("false_target")
                            or ""
                        )
                    )

    def _collect_phis(self) -> None:
        for block in self.function.blocks.values():
            for instr in block.instrs:
                if instr.op not in ("Phi", "phi"):
                    continue
                incoming = instr.attributes.get("incoming") or ()
                self._locals[instr.res.id] = instr.res
                entries = self._phi_targets.setdefault(block.name, [])
                for predecessor, value in incoming:
                    entries.append((str(predecessor), instr.res, value))

    # -- assembly ----------------------------------------------------------
    def emit(self) -> FortranSubroutine:
        self._bound_ids = {a.id for a in self.function.args} | {
            value.id for value in self.outputs
        }
        self._collect_branch_targets()
        self._collect_use_sites()
        self._collect_phis()
        self._collect_address_producers()
        self._collect_producers()
        body: list[str] = []
        for block in self.function.blocks.values():
            self._emit_block(block, body)

        # A bind(C) procedure may not take assumed-shape (``x(:)``) dummies:
        # those need a descriptor the C caller has no way to build before
        # Fortran 2018 / TS 29113.  Arrays are therefore explicit-shape over
        # extents passed as leading arguments, which is what a C caller would
        # have to supply anyway and lets the compiler see the trip count.
        #
        # Every array keeps its own shape here -- one extent parameter per
        # distinct dimension size actually present, not one extent shared by
        # every array in the function.  A matmul's (rows, inner) x
        # (inner, cols) -> (rows, cols) genuinely has three different sizes;
        # forcing them through one shared extent either rejects it outright
        # or -- worse -- silently declares mismatched-size arrays under one
        # extent, which compiles cleanly and corrupts memory at runtime.
        all_values = (
            *self.function.args,
            *self.outputs,
            *self._locals.values(),
        )
        arrays_present = any(_is_array(value) for value in all_values)
        dim_extents = dimension_extents(all_values) if arrays_present else {}

        def dims(value: SSAValue) -> str:
            return ", ".join(dim_extents[int(size)] for size in value.shape)

        extent_names = sorted(dim_extents.values())
        self.extent_names = tuple(extent_names)
        arguments = list(extent_names)
        arguments.extend(_name(a) for a in self.function.args)
        arguments.extend(_name(value) for value in self.outputs)

        declarations: list[str] = [
            f"    integer(c_int), intent(in), value :: {extent}"
            for extent in extent_names
        ]
        for argument in self.function.args:
            kind = _DTYPE_KIND.get(argument.dtype or self.dtype, "real(c_double)")
            if _is_array(argument):
                declarations.append(
                    f"    {kind}, intent(in) :: {_name(argument)}({dims(argument)})"
                )
            else:
                declarations.append(
                    f"    {kind}, intent(in), value :: {_name(argument)}"
                )
        for value in self.outputs:
            kind = _DTYPE_KIND.get(value.dtype or self.dtype, "real(c_double)")
            if _is_array(value):
                declarations.append(
                    f"    {kind}, intent(out) :: {_name(value)}({dims(value)})"
                )
            else:
                declarations.append(
                    f"    {kind}, intent(out) :: {_name(value)}"
                )

        bound = {a.id for a in self.function.args} | {
            value.id for value in self.outputs
        }
        for value in self._locals.values():
            if value.id in bound:
                continue
            kind = _DTYPE_KIND.get(value.dtype or self.dtype, "real(c_double)")
            if _is_array(value):
                # An automatic array sized by its own extents: no allocate,
                # no heap.
                declarations.append(
                    f"    {kind} :: {_name(value)}({dims(value)})"
                )
            else:
                declarations.append(f"    {kind} :: {_name(value)}")

        declarations.extend(
            f"    integer(c_int) :: {index}" for index in self._loop_variables
        )

        name = self.function.name
        lines = [
            f"  subroutine {name}({', '.join(arguments)}) bind(C, name=\"{name}\")",
            "    use, intrinsic :: iso_c_binding",
            "    implicit none",
            *declarations,
            "",
            *body,
            f"  end subroutine {name}",
        ]
        return FortranSubroutine(
            name,
            "\n".join(lines),
            tuple(self.shortfalls),
            tuple(extent_names),
        )


def emit_function(
    function: Function,
    *,
    dtype: str = DEFAULT_DTYPE,
    outputs: Sequence[SSAValue] = (),
    callee_extents: Mapping[str, Sequence[str]] | None = None,
) -> FortranSubroutine:
    """Translate one SSA function into a bind(C) Fortran subroutine.

    ``outputs`` names the SSA values that leave the subroutine.  SSA itself
    records only arguments, so results would otherwise be emitted as dead
    locals; naming them promotes them to ``intent(out)`` parameters.

    ``callee_extents`` maps a called subroutine's name to the extent
    parameters it declares, so a call passes exactly the extents that
    subroutine expects rather than extents rederived at the call site.
    """

    return _FunctionEmitter(
        function,
        dtype=dtype,
        outputs=outputs,
        callee_extents=callee_extents,
    ).emit()


@dataclass
class FortranModule:
    """A complete Fortran module wrapping one or more SSA functions."""

    name: str
    source: str
    subroutines: tuple[FortranSubroutine, ...] = ()

    @property
    def shortfalls(self) -> tuple[FortranShortfall, ...]:
        return tuple(s for sub in self.subroutines for s in sub.shortfalls)

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def write(self, directory: str | Path) -> Path:
        path = Path(directory) / f"{self.name}.f90"
        path.write_text(self.source, encoding="utf-8")
        return path


def emit_module(
    module: IRModule | Mapping[str, Function],
    *,
    name: str = "turing_ssa",
    dtype: str = DEFAULT_DTYPE,
    outputs: Mapping[str, Sequence[SSAValue]] | None = None,
) -> FortranModule:
    """Translate an SSA module into one Fortran module.

    ``outputs`` maps a function name to the SSA values it returns.
    """

    functions = (
        module.functions if isinstance(module, IRModule) else dict(module)
    )
    named_outputs = dict(outputs or {})
    # Two passes: a subroutine that calls another must pass exactly the
    # extents that one declares, and those are only known once it has been
    # emitted. The first pass is discarded apart from its signatures.
    callee_extents = {
        function_name: emit_function(
            function,
            dtype=dtype,
            outputs=named_outputs.get(function_name, ()),
        ).extent_names
        for function_name, function in functions.items()
    }
    subroutines = tuple(
        emit_function(
            function,
            dtype=dtype,
            outputs=named_outputs.get(function_name, ()),
            callee_extents=callee_extents,
        )
        for function_name, function in functions.items()
    )
    lines = [
        f"module {name}",
        "  use, intrinsic :: iso_c_binding",
        "  implicit none",
        "contains",
        "",
        *[sub.source for sub in subroutines],
        "",
        f"end module {name}",
    ]
    return FortranModule(name, "\n".join(lines) + "\n", subroutines)


# Toolchains that are commonly installed but left off PATH on Windows.  A
# compiler that exists is worth finding: without one the compile-and-run path
# silently disappears and emitted code goes unverified.
_KNOWN_COMPILER_PATHS = (
    r"C:\msys64\mingw64\bin\gfortran.exe",
    r"C:\MinGW\bin\gfortran.exe",
    r"C:\w64devkit\bin\gfortran.exe",
    r"C:\ProgramData\chocolatey\bin\gfortran.exe",
)


def fortran_compiler() -> str | None:
    """Return an available Fortran compiler, or ``None``.

    Emission never requires one.  This exists so callers can decide whether the
    compile-and-run path is available rather than discovering it by failure.
    ``TURING_FC`` overrides the search.
    """

    import os

    override = os.environ.get("TURING_FC")
    if override and Path(override).exists():
        return override
    for candidate in ("gfortran", "ifx", "ifort", "flang", "lfortran"):
        found = shutil.which(candidate)
        if found:
            return found
    for candidate in _KNOWN_COMPILER_PATHS:
        if Path(candidate).exists():
            return candidate
    return None


def compile_module(
    module: FortranModule,
    *,
    directory: str | Path | None = None,
    extra_flags: Sequence[str] = ("-O3", "-march=native", "-funroll-loops"),
) -> Path:
    """Compile a generated module to a shared library.

    Raises ``FortranEmissionError`` when no compiler is present; callers that
    only need source should not call this.
    """

    compiler = fortran_compiler()
    if compiler is None:
        raise FortranEmissionError(
            "no Fortran compiler found; emission does not require one, "
            "but compile_module does"
        )
    import os
    import sys

    workdir = Path(directory or tempfile.mkdtemp(prefix="turing_fortran_"))
    workdir.mkdir(parents=True, exist_ok=True)
    source = module.write(workdir)
    suffix = ".dll" if sys.platform == "win32" else ".so"
    library = workdir / f"{module.name}{suffix}"
    command = [
        compiler,
        "-shared",
        *(() if sys.platform == "win32" else ("-fPIC",)),
        *extra_flags,
        str(source),
        "-o",
        str(library),
    ]

    # gfortran spawns f951, which loads libgmp/libmpfr from the toolchain's own
    # bin directory.  When the compiler is invoked by absolute path and that
    # directory is not on PATH, f951 fails to load and gfortran exits non-zero
    # with no diagnostic at all.  Put it on PATH for the child.
    environment = dict(os.environ)
    compiler_bin = str(Path(compiler).parent)
    environment["PATH"] = compiler_bin + os.pathsep + environment.get("PATH", "")

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(workdir),
        env=environment,
    )
    if completed.returncode != 0:
        raise FortranEmissionError(
            f"Fortran compilation failed:\n{completed.stderr}"
        )
    return library


__all__ = [
    "DEFAULT_DTYPE",
    "FortranEmissionError",
    "FortranModule",
    "FortranShortfall",
    "FortranSubroutine",
    "compile_module",
    "dimension_extents",
    "emit_function",
    "emit_module",
    "fortran_compiler",
]

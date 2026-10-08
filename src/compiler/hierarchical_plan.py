"""Hierarchical, printable planning objects.

Text is a view, never the authority.  Each logical line is a typed item and
each nested scope is an explicit closure with an explicit capture set.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any, Mapping

from .monotonic_ids import GLOBAL_MONOTONIC_IDS
from ..transmogrifier.ssa import Instr, SSAValue


@dataclass(frozen=True)
class PlanLine:
    opcode: str
    inputs: tuple[int, ...] = ()
    outputs: tuple[int, ...] = ()
    attributes: tuple[tuple[str, Any], ...] = ()
    input_roles: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        opcode: str,
        *,
        inputs=(),
        outputs=(),
        attributes: Mapping[str, Any] | None = None,
        input_roles=(),
    ) -> "PlanLine":
        return cls(
            str(opcode),
            tuple(int(value) for value in inputs),
            tuple(int(value) for value in outputs),
            tuple(sorted((attributes or {}).items())),
            tuple(str(role) for role in input_roles),
        )


@dataclass(frozen=True)
class PlanClosure:
    name: str
    captures: tuple[int, ...]
    items: tuple["PlanItem", ...]
    closure_id: int = -1
    # Shape (and dtype) of the region's values, carried from the process graph's
    # per-node domain so the lowered SSA values are the arrays they are, not
    # shapeless scalars. ``(value_id, shape, dtype)`` per value.
    value_shapes: tuple[tuple[int, tuple[int, ...], str], ...] = ()
    # Dynamic spans have declared rank while their concrete extents remain
    # ordinary runtime SSA. Keep that fact separate from the static shape so
    # an empty tuple cannot silently turn a span into rank-zero arithmetic.
    value_ranks: tuple[tuple[int, int], ...] = ()
    # A captured value made by a shape-only operation outside the region
    # (``t1.unsqueeze(-1)``) is fed the storage of its source; the formal is a
    # VIEW of that storage with its own extents.  ``(value_id,
    # storage_value_id, view_shape, operation)`` per such capture: the formal
    # declares ``ssa_storage_view`` so call-metadata propagation keeps the
    # view's shape instead of restamping the storage owner's.
    value_views: tuple[tuple[int, int, tuple[int, ...], str], ...] = ()


@dataclass(frozen=True)
class PlanCall:
    """A typed call edge between two planned closures.

    The call owns its argument/result correlation.  Backend lowering may
    inline the callee, retain a native call, or split deployment, but it must
    never rediscover this relationship from source names or runtime values.
    """

    callsite_id: int
    callee: PlanClosure
    argument_value_ids: tuple[int, ...] = ()
    result_value_ids: tuple[int, ...] = ()
    argument_bindings: tuple[tuple[int, int], ...] = ()
    result_bindings: tuple[tuple[int, int], ...] = ()
    enclosing_loop_ids: tuple[int, ...] = ()


PlanItem = PlanLine | PlanClosure | PlanCall


#: How a tensor operation is spelled once it is known to act on scalars.
#:
#: The planner applies this ONLY when the result and every operand have an
#: empty shape; a genuinely tensor-shaped operation keeps its lowercase
#: name and is resolved through the tensor likeness table instead. The two
#: spellings are therefore the same operation at different ranks, which is
#: why anything interpreting repository SSA -- a backend, or the reference
#: evaluator -- must read them through this one table rather than keep a
#: private copy that can drift from it.
TENSOR_OPERATION_SCALAR_SPELLING: dict[str, str] = {
    "add": "Add", "sub": "Sub", "mul": "Mul",
    "truediv": "Div", "div": "Div", "floordiv": "FloorDiv",
    "mod": "Mod", "pow": "Pow", "neg": "Neg", "abs": "Abs",
    "equal": "Eq", "not_equal": "Ne", "less": "Lt",
    "less_equal": "Le", "greater": "Gt", "greater_equal": "Ge",
    "logical_and": "LAnd", "logical_or": "LOr",
    "logical_not": "LNot", "maximum": "Max", "minimum": "Min",
    "bitand": "BitAnd", "bitor": "BitOr", "bitxor": "BitXor",
    "shl": "Shl", "shr": "Shr", "invert": "Invert",
    "sqrt": "Sqrt", "exp": "Exp", "log": "Log",
    "isfinite": "IsFinite", "isnan": "IsNaN", "isinf": "IsInf",
}

#: Operations whose result is a truth value rather than a number.
#:
#: This is a fact about the vocabulary, so it lives with the vocabulary and
#: not inside whichever backend noticed it first. A relation lowered as
#: float64 tells every consumer the wrong thing, and the consumer cannot
#: recover it: the LLVM template for `Lt` emits `fcmp`, which yields i1
#: whatever the SSA claims, so a value declared double and rendered i1
#: disagrees with itself and the verifier rejects the first instruction
#: that consumes it. Fortran and SPIR-V have the same exposure.
PREDICATE_OPERATIONS: frozenset[str] = frozenset({
    "Eq", "Ne", "Lt", "Le", "Gt", "Ge", "ULt", "ULe",
    "LAnd", "LOr", "LNot", "LXor", "IsFinite", "IsNaN", "IsInf",
})
_PREDICATE_OPERATION_SPELLINGS = frozenset(
    operation.casefold() for operation in PREDICATE_OPERATIONS
)


def is_predicate_operation(operation: Any) -> bool:
    """Whether an SSA spelling produces truth, independent of casing."""

    return str(operation).casefold() in _PREDICATE_OPERATION_SPELLINGS


def plan_value_id_watermark(plan: PlanClosure) -> int:
    """First value id above every id the plan (recursively) mentions."""

    watermark = 0
    for value_id, _shape, _dtype in plan.value_shapes:
        watermark = max(watermark, int(value_id) + 1)
    for item in plan.items:
        if isinstance(item, PlanLine):
            for value_id in (*item.inputs, *item.outputs):
                watermark = max(watermark, int(value_id) + 1)
        elif isinstance(item, PlanClosure):
            watermark = max(watermark, plan_value_id_watermark(item))
        elif isinstance(item, PlanCall):
            for value_id in (
                *item.argument_value_ids, *item.result_value_ids,
                *(caller for caller, _callee in item.argument_bindings),
                *(caller for _callee, caller in item.result_bindings),
            ):
                watermark = max(watermark, int(value_id) + 1)
    return watermark


def expand_plan_regions(
    plan: PlanClosure, *, first_free_value_id: int = 0,
    function_scope: str = "<plan>",
    lexical_read_scope: tuple | None = None,
) -> dict[tuple[str, int], tuple[Instr, ...]]:
    """Lower every top-level ``region_*`` closure of ``plan`` exactly once.

    A region's lowering may synthesize temporaries an authored graph never
    named (the binary ``Max`` chain of a variadic ``max(a, b, c)``).  Those
    temporaries live in the SAME value-id space as every other id of the
    function, so they are allocated from one watermark that starts above
    every id the function already has in play (``first_free_value_id``:
    the plan alone is not the function -- control-only graph values such as
    an ``is not None`` test are never plan lines) and advances past each
    region's synthesized ids: no two regions coin the same temporary, and
    no temporary lands on an authored id (region 2's ``max`` chain coined
    50 while region 5's authored ``Div`` was 50, and the emitted function
    defined one result twice).  Every consumer of a region's instructions
    must read this one expansion so all stages agree on which ids the
    region defines.
    """

    watermark = max(int(first_free_value_id), plan_value_id_watermark(plan))
    expanded: dict[tuple[str, int], tuple[Instr, ...]] = {}
    for item in plan.items:
        if not (
            isinstance(item, PlanClosure) and item.name.startswith("region_")
        ):
            continue
        instructions = plan_region_to_ssa_instrs(
            item,
            first_free_value_id=watermark,
            function_scope=function_scope,
            lexical_read_scope=lexical_read_scope,
        )
        for instruction in instructions:
            for value in (
                *instruction.args,
                *((instruction.res,) if instruction.res is not None else ()),
            ):
                watermark = max(watermark, int(value.id) + 1)
        expanded[(str(item.name), int(item.closure_id))] = instructions
    return expanded


def copy_region_instructions(
    instructions: tuple[Instr, ...],
) -> list[Instr]:
    """Fresh instruction and value objects carrying the same identities.

    Stages retype a region's occurrences in place; each stage works on its
    own copies so one stage's view never leaks into another's.
    """

    return [
        replace(
            instruction,
            args=[replace(value) for value in instruction.args],
            res=(
                None if instruction.res is None else replace(instruction.res)
            ),
            arg_roles=list(instruction.arg_roles),
            attributes=dict(instruction.attributes),
        )
        for instruction in instructions
    ]


def plan_region_to_ssa_instrs(
    region: PlanClosure, *, first_free_value_id: int = 0,
    function_scope: str = "<plan>",
    lexical_read_scope: tuple | None = None,
) -> tuple[Instr, ...]:
    """Lower one planner-owned flat region to repository SSA instructions.

    Each SSA value carries the shape and dtype recorded for it on the region
    (``value_shapes``, from the process graph's per-node domain), so a value is
    the array it is and array ops lower as array ops rather than scalars.
    """

    shape_of = {
        int(value_id): tuple(int(dimension) for dimension in shape)
        for value_id, shape, _dtype in region.value_shapes
    }
    dtype_of = {
        int(value_id): dtype for value_id, _shape, dtype in region.value_shapes
    }
    rank_of = {
        int(value_id): int(rank) for value_id, rank in region.value_ranks
    }
    view_of = {
        int(value_id): (int(storage), tuple(int(e) for e in shape), str(operation))
        for value_id, storage, shape, operation in region.value_views
    }
    # The graph domain is deliberately permissive and often records scalar
    # control values with its default numerical dtype.  Operator semantics are
    # authoritative where they are stricter: comparisons/logical operations
    # produce predicates, and address indices/extents are integers.  Retain
    # these contracts in repository SSA rather than asking a target emitter to
    # reverse-engineer them from syntax.
    predicate_ops = {
        "eq", "equal", "ne", "not_equal", "lt", "less", "le",
        "less_equal", "gt", "greater", "ge", "greater_equal",
        "land", "logical_and", "lor", "logical_or", "lnot",
        "logical_not", "is", "is_not", "contains", "not_contains",
    } | _PREDICATE_OPERATION_SPELLINGS
    integer_result_ops = {
        "len", "length", "extent", "bitlength", "bit_length",
    }
    scalar_cast_dtypes = {
        "float": "float64",
        # Python ``int`` is not the backend's C ``int``.  Repository SSA uses
        # the widest portable signed scalar ABI for authored Python integers;
        # wrappers retain authored fallback for values outside that domain.
        "int": "int64",
        "bool": "bool",
    }

    # The operation owns its result domain. Commit that fact at the point
    # repository SSA values are born so later region copies and target
    # emitters consume one shared decision instead of re-inferring truth
    # types from whichever spelling (``IsFinite`` or ``isfinite``) survived.
    from .identity_concordance import current_identity_book
    predicate_type_page = current_identity_book().page(
        "operator_result_type_concordance"
    )
    for item in region.items:
        if not (
            isinstance(item, PlanLine)
            and item.outputs
            and str(item.opcode).casefold() in predicate_ops
        ):
            continue
        row = (
            str(function_scope), int(region.closure_id), str(region.name),
            int(item.outputs[0]),
        )
        claim = (str(item.opcode).casefold(), "bool")
        incumbent = predicate_type_page.latest(row)
        if incumbent is None:
            predicate_type_page.set(row, 0, claim)
        elif incumbent != claim:
            raise ValueError(
                "operator result-type concordance disagreement: "
                f"row={row!r}, incumbent={incumbent!r}, candidate={claim!r}"
            )

    def semantic_input_ids(item: PlanLine) -> tuple[int, ...]:
        if len(item.input_roles) != len(item.inputs):
            return tuple(map(int, item.inputs))
        return tuple(
            int(value_id)
            for value_id, role in zip(
                item.inputs, item.input_roles, strict=True,
            )
            if str(role).casefold() not in {
                "callee", "func", "function", "definition",
                "operator", "operator_reference",
            }
        )

    # An authored Python scalar literal is a weak operand: it names a value,
    # not an element width, so beside a typed float operand it does not widen
    # the result (``float32 * 2.0`` is float32, as in the eager lane).
    weak_literal_ids = {
        int(item.outputs[0])
        for item in region.items
        if isinstance(item, PlanLine)
        and item.outputs
        and str(item.opcode).casefold() in {"const", "constant"}
        and isinstance(
            dict(item.attributes).get("value"), (bool, int, float)
        )
    }
    def promoted_float_dtype(value_ids: tuple[int, ...]) -> str | None:
        """The widest float width the non-literal operands declare, if any.

        The declared promotion rule is the precision layer's: the wider
        element type wins (``topological_reducer._widest_element``).  Python's
        ``float`` is binary64 here, not that table's ``float32`` alias.
        """

        from ..common.tensors.topological_reducer import _widest_element

        return _widest_element(*(
            {"float": "float64"}.get(spelling, spelling)
            for value_id in value_ids
            if int(value_id) not in weak_literal_ids
            for spelling in (str(dtype_of.get(int(value_id)) or ""),)
        ))

    def promoted_numeric_dtype(value_ids: tuple[int, ...]) -> str | None:
        candidates = {
            str(dtype_of.get(int(value_id)) or "")
            for value_id in value_ids
        }
        if candidates.intersection({
            "float", "float16", "float32", "float64", "double",
            "f16", "f32", "f64",
        }):
            return promoted_float_dtype(value_ids) or "float64"
        if candidates.intersection({"int64", "i64"}):
            return "int64"
        if candidates.intersection({"int", "int32", "i32"}):
            return "int"
        if candidates == {"bool"}:
            return "bool"
        return None

    # Refine permissive graph-domain defaults using authored scalar operator
    # semantics before constructing any SSAValue.  Iterate because a chain
    # such as int(record[index]) -> min -> shift must carry the corrected
    # integer width through every intermediate, independent of node order.
    dtype_preserving_ops = {
        "add", "sub", "mul", "floordiv", "mod", "pow", "neg", "abs",
        "min", "max", "minimum", "maximum", "bitand", "bitor",
        "bitxor", "shl", "shr", "invert",
    }
    projection_ops = {
        "indexed", "getitem", "get_item", "subscript", "load",
    }
    for _ in range(max(1, len(region.items))):
        changed = False
        for item in region.items:
            if not isinstance(item, PlanLine) or not item.outputs:
                continue
            output_id = int(item.outputs[0])
            opcode = str(item.opcode).casefold()
            inferred: str | None = None
            declared_columns = tuple(
                dict(item.attributes).get("sequence_column_dtypes") or ()
            )
            if declared_columns and str(declared_columns[0]) not in {
                "", "None", "unknown",
            }:
                # The line produces a sequence HANDLE whose row contract is
                # declared on it (a record field's table storage: the
                # reducer stamps ``sequence_column_dtypes`` from the class
                # field contract on the field's GetAttr and its seeded
                # pre-branch state).  The handle's storage dtype is its
                # column-0 dtype, as ``_sequence_descriptor`` and the
                # sequence-program entry type an arena.  Without this the
                # handle fell to the ``float64`` default below: orbital
                # ``step_with_dt_control_used``, region 80, ``getattr
                # unresolved_report`` (contract ``token: int64``) -> 518
                # typed float64, overriding the int64 arena the builder had
                # declared, while every callee formal for the same arena
                # stayed int64 -- "10 incompatible final physical call
                # inputs; storage types are immutable".
                inferred = str(declared_columns[0])
            elif opcode in {"const", "constant"}:
                literal = dict(item.attributes).get("value")
                inferred = (
                    "bool" if isinstance(literal, bool)
                    else "int64" if isinstance(literal, int)
                    else "float64" if isinstance(literal, float)
                    else None
                )
            elif opcode in predicate_ops:
                inferred = "bool"
            elif opcode in integer_result_ops:
                inferred = "int64"
            elif opcode in scalar_cast_dtypes:
                inferred = scalar_cast_dtypes[opcode]
            elif opcode in {"truediv", "div"}:
                # True division is floating: the operands' widest declared
                # float width, float64 when none is declared.
                inferred = (
                    promoted_float_dtype(semantic_input_ids(item))
                    or "float64"
                )
            elif opcode in dtype_preserving_ops:
                inferred = promoted_numeric_dtype(semantic_input_ids(item))
            elif opcode in projection_ops:
                sources = semantic_input_ids(item)
                inferred = (
                    None if not sources else dtype_of.get(int(sources[0]))
                )
            if inferred is not None and dtype_of.get(output_id) != inferred:
                dtype_of[output_id] = inferred
                changed = True
        if not changed:
            break

    for item in region.items:
        if not isinstance(item, PlanLine):
            continue
        opcode = str(item.opcode).casefold()
        if item.outputs and opcode in {"const", "constant"}:
            literal = dict(item.attributes).get("value")
            if isinstance(literal, bool):
                dtype_of[int(item.outputs[0])] = "bool"
            elif isinstance(literal, int):
                dtype_of[int(item.outputs[0])] = "int64"
            elif isinstance(literal, float):
                dtype_of[int(item.outputs[0])] = "float64"
        if item.outputs and opcode in predicate_ops:
            dtype_of[int(item.outputs[0])] = "bool"
        elif item.outputs and opcode in integer_result_ops:
            dtype_of[int(item.outputs[0])] = "int64"
        elif item.outputs and opcode in scalar_cast_dtypes:
            dtype_of[int(item.outputs[0])] = scalar_cast_dtypes[opcode]
        if opcode == "getelementptr":
            # Only repository address arithmetic requires integer indices.
            # High-level Indexed/IndexedStore may be a dictionary lookup whose
            # key retains any authored type and is lowered through a table.
            index_inputs = item.inputs[1:]
            for value_id in index_inputs:
                dtype_of[int(value_id)] = "int64"

    def value(value_id: int) -> SSAValue:
        value_id = int(value_id)
        existing = values.get(value_id)
        if existing is not None:
            return existing
        made = SSAValue(
            value_id,
            dtype=dtype_of.get(value_id, "float64"),
            shape=shape_of.get(value_id, ()),
            accounting={
                "program_abi_rank": rank_of[value_id],
                "program_abi_storage": "span",
            } if rank_of.get(value_id, 0) > len(
                shape_of.get(value_id, ())
            ) else {},
        )
        view = view_of.get(value_id)
        if view is not None and tuple(made.shape) == view[1]:
            made.accounting = {
                **dict(made.accounting or {}),
                "ssa_storage_view": {
                    "storage_value_id": view[0],
                    "view_shape": view[1],
                    "operation": view[2],
                },
            }
        values[value_id] = made
        return made

    values: dict[int, SSAValue] = {}
    def fresh_like(result: SSAValue) -> SSAValue:
        from .concordance_declarations import CANONICAL_VALUE

        source = (
            None if lexical_read_scope is None else
            current_identity_book().latest_ref(
                CANONICAL_VALUE, (tuple(lexical_read_scope), int(result.id)),
            )
        )
        if source is None:
            value_id = GLOBAL_MONOTONIC_IDS.mint()
        else:
            from .concordance_declarations import FRESH_LIKE, PLANNER_HIERARCHY
            from .precompile_to_ssa import _mint_ssa_id

            value_id = _mint_ssa_id(
                current_identity_book(), function_scope, FRESH_LIKE, (source,),
                dtype=result.dtype, shape=tuple(result.shape),
                stage=PLANNER_HIERARCHY,
            )
        made = SSAValue(
            value_id,
            dtype=result.dtype,
            shape=tuple(result.shape),
            accounting=dict(result.accounting or {}),
        )
        values[value_id] = made
        return made

    def fork_operand_read(
        item: PlanLine, source_position: int, consumer: SSAValue,
        role: str, operand: SSAValue,
    ) -> None:
        """Carry one authored operand occurrence into its binary expansion.

        ``total = min(total, upper, lower)`` in a retained loop used to
        mint the first Min without the read of ``total`` at its new ``left``
        position.  The region feed then named a consumer with no lexical
        binding row.  A fold/clamp operand keeps the exact PlanLine read
        through an OperandFork, even when the emitted consumer is synthetic.
        """
        if lexical_read_scope is None or len(item.inputs) != len(item.input_roles):
            return
        from ..common.tensors.topological_reducer import _operand_positions
        from .concordance_declarations import (
            CONSUMER_OPERAND, IDENTITY_TRANSITION, LEXICAL_READ_BINDING,
            OPERAND_FORK, OPERAND_POSITION, OperandFork,
        )
        from .identity_concordance import Derived, Mode, Novel

        scope = tuple(lexical_read_scope)
        positions = tuple(
            (source_role, ordinal)
            for source_role, ordinal, _parent in _operand_positions(
                zip(item.inputs, item.input_roles)
            )
            if str(source_role).casefold() not in {
                "callee", "func", "function", "definition",
                "operator", "operator_reference",
            }
        )
        source_role, ordinal = positions[source_position]
        source_row = (scope, int(item.outputs[0]), source_role, ordinal)
        book = current_identity_book()
        read = book.latest_ref(LEXICAL_READ_BINDING, source_row)
        if read is None:
            return
        fact = book.page(LEXICAL_READ_BINDING).latest(source_row)
        if not isinstance(fact, str):
            return
        row = (scope, int(consumer.id), role, 0)
        fork = book.post(
            IDENTITY_TRANSITION, row,
            OperandFork("plan_binary_scalar_expansion", *source_row[1:]),
            stage=OPERAND_POSITION,
            provenance=Novel(OPERAND_FORK, (read,)), mode=Mode.REVISE,
        )
        binding = book.post(
            LEXICAL_READ_BINDING, row, fact, stage=OPERAND_POSITION,
            provenance=Derived((fork, read)), mode=Mode.REVISE,
        )
        operand_row = (scope, int(consumer.id), int(operand.id))
        previous = book.page(CONSUMER_OPERAND).latest(operand_row) or ()
        book.post(
            CONSUMER_OPERAND, operand_row,
            tuple(dict.fromkeys((*previous, (role, 0)))),
            stage=OPERAND_POSITION, provenance=Derived((binding, fork)),
            mode=Mode.REVISE,
        )

    instructions = []
    for item in region.items:
        if not isinstance(item, PlanLine):
            raise ValueError(
                f"{region.name!r} is not a flat operator region"
            )
        if len(item.outputs) > 1:
            raise ValueError(
                f"{item.opcode!r} may publish at most one SSA result"
            )
        result = (
            value(int(item.outputs[0])) if item.outputs else None
        )
        # A Python identity replacement is already the call's graph-native
        # meaning.  Its callable/definition input is provenance, not runtime
        # data.  Keep this distinction explicit through argument roles instead
        # of teaching each backend that ``float`` (or every future identity)
        # happens to carry a CPython function object in operand zero.
        paired_inputs = tuple(zip(item.inputs, item.input_roles))
        if len(item.input_roles) == len(item.inputs):
            semantic_inputs = semantic_input_ids(item)
            semantic_roles = tuple(
                str(role)
                for _value_id, role in paired_inputs
                if str(role).casefold() not in {
                    "callee", "func", "function", "definition",
                    "operator", "operator_reference",
                }
            )
        else:
            semantic_inputs = tuple(int(value_id) for value_id in item.inputs)
            semantic_roles = tuple(str(role) for role in item.input_roles)

        opcode = str(item.opcode)
        attributes = dict(item.attributes)
        if opcode.casefold() in scalar_cast_dtypes:
            attributes.setdefault("source_operator", opcode)
            attributes.setdefault(
                "target_dtype", scalar_cast_dtypes[opcode.casefold()]
            )
            opcode = "Cast"
        elif opcode.casefold() == "tensor":
            # AbstractTensor.tensor(x) is the general ensure-type idiom.  Once
            # x reaches typed repository SSA, normalization is represented by
            # a same-value cast carrying the schema promise; target code does
            # not reconstruct or invoke a Python object.
            attributes.setdefault("source_operator", opcode)
            attributes.setdefault("target_dtype", dtype_of.get(
                int(item.outputs[0]) if item.outputs else -1, "float64"
            ))
            opcode = "Cast"
        elif opcode == "const":
            # A plan line the planner folded to a literal (``0.5 * step`` of
            # an unrolled loop, ``structural_specialization``) is the SSA
            # constant, spelled ``Const`` like every other constant the
            # lowering emits.  The lowercase graph op reached the LLVM
            # emitter as "operation has no repository LLVM emission".
            opcode = "Const"

        scalar_spelling = TENSOR_OPERATION_SCALAR_SPELLING
        is_scalar = (
            result is not None
            and max(
                len(tuple(result.shape)),
                int((result.accounting or {}).get("program_abi_rank", 0)),
            ) == 0
            and all(
                max(
                    len(tuple(value(value_id).shape)),
                    int((value(value_id).accounting or {}).get(
                        "program_abi_rank", 0
                    )),
                ) == 0
                for value_id in semantic_inputs
            )
        )
        semantic_opcode = str(
            attributes.get("tensor_operation")
            or attributes.get("tensor_candidate")
            or opcode
        )
        if is_scalar and semantic_opcode.casefold() in scalar_spelling:
            opcode = scalar_spelling[semantic_opcode.casefold()]

        # Python's variadic min/max and the tensor clamp convenience are
        # ordinary binary SSA folds.  Decomposing them here keeps evaluation
        # order and data dependencies visible, and gives every backend the
        # same primitive program instead of four bespoke builtin handlers.
        fold_opcode = {"max": "Max", "min": "Min"}.get(
            semantic_opcode.casefold()
        )
        if is_scalar and fold_opcode is not None and len(semantic_inputs) >= 2:
            scalar_attributes = {
                key: value
                for key, value in attributes.items()
                if key not in {
                    "callee", "lowered_from", "tensor",
                    "tensor_candidate", "tensor_operation",
                }
            }
            operands = [value(value_id) for value_id in semantic_inputs]
            accumulator = operands[0]
            for position, operand in enumerate(operands[1:], 1):
                fold_result = (
                    result if position == len(operands) - 1
                    else fresh_like(result)
                )
                instructions.append(Instr(
                    fold_opcode,
                    [accumulator, operand],
                    fold_result,
                    arg_roles=["left", "right"],
                    attributes={
                        **scalar_attributes,
                        "source_operator": semantic_opcode,
                    },
                ))
                if position == 1:
                    fork_operand_read(item, 0, fold_result, "left", accumulator)
                fork_operand_read(item, position, fold_result, "right", operand)
                accumulator = fold_result
            continue
        if is_scalar and opcode.casefold() == "clamp" and len(semantic_inputs) == 3:
            operand, lower, upper = (
                value(value_id) for value_id in semantic_inputs
            )
            bounded_below = fresh_like(result)
            instructions.append(Instr(
                "Max", [operand, lower], bounded_below,
                arg_roles=["operand", "lower"],
                attributes={**attributes, "source_operator": opcode},
            ))
            fork_operand_read(item, 0, bounded_below, "operand", operand)
            fork_operand_read(item, 1, bounded_below, "lower", lower)
            instructions.append(Instr(
                "Min", [bounded_below, upper], result,
                arg_roles=["operand", "upper"],
                attributes={**attributes, "source_operator": opcode},
            ))
            fork_operand_read(item, 2, result, "upper", upper)
            continue

        instructions.append(Instr(
            opcode,
            [value(value_id) for value_id in semantic_inputs],
            result,
            arg_roles=list(semantic_roles),
            attributes=attributes,
        ))
    return tuple(instructions)


@dataclass(frozen=True)
class HierarchyValueTable:
    """Collision-free IDs for values whose local IDs live in many shells.

    ``scope`` is the hierarchy scope ``assign_hierarchy_ids`` minted for
    this table: its ``hierarchy_value`` rows ``(scope, closure, local)`` on
    the book hold the same correlations, each DERIVED from the local
    value's identity cell (plan 80, A2.5).  A scope tuple, never a Ref, so
    the table stays picklable as before.
    """

    correlations: tuple[tuple[int, int, int], ...]
    scope: tuple | None = None

    @cached_property
    def _global_ids(self) -> dict[tuple[int, int], int]:
        """Index the immutable correlation table once.

        Hierarchical composition asks for the same endpoint mappings many
        times while nesting calls, control blocks, bindings and shader
        storage.  Scanning ``correlations`` for every request makes that
        otherwise-linear compiler stage quadratic in the number of scoped
        values.  The tuple remains the canonical, printable representation;
        this index is only its exact lookup form and does not derive IDs from
        names, observed values, or runtime state.
        """

        return {
            (int(scope), int(local)): int(global_id)
            for scope, local, global_id in self.correlations
        }

    def global_id(self, closure_id: int, local_value_id: int) -> int:
        return self._global_ids[(int(closure_id), int(local_value_id))]


@dataclass(frozen=True)
class HierarchyIdentityReduction:
    """Result of removing semantically transparent call closures."""

    root: PlanClosure
    collapsed_callsites: tuple[int, ...]
    rounds: int


def reduce_hierarchy_identities(
    root: PlanClosure,
    identity_closure_ids: set[int] | frozenset[int],
) -> HierarchyIdentityReduction:
    """Remove post-planning call boundaries proven to be SSA identities.

    Identity discovery happens only after hierarchy construction, when argument
    and result bindings are explicit.  Value unification therefore remains the
    authority: this pass removes the now-redundant closure/control boundary but
    never invents an alias from source names or observed runtime values.

    The rewrite is a fixed point so future identities that erase enclosing
    structural reasons can expose further collapses without changing callers.
    """

    identities = frozenset(int(value) for value in identity_closure_ids)
    collapsed: list[int] = []
    rounds = 0
    current = root
    while True:
        changed = False

        def rewrite(closure: PlanClosure) -> PlanClosure:
            nonlocal changed
            items = []
            for item in closure.items:
                if isinstance(item, PlanCall):
                    callee = rewrite(item.callee)
                    if int(callee.closure_id) in identities:
                        collapsed.append(int(item.callsite_id))
                        changed = True
                        continue
                    items.append(PlanCall(
                        item.callsite_id,
                        callee,
                        item.argument_value_ids,
                        item.result_value_ids,
                        item.argument_bindings,
                        item.result_bindings,
                        item.enclosing_loop_ids,
                    ))
                elif isinstance(item, PlanClosure):
                    items.append(rewrite(item))
                else:
                    items.append(item)
            return PlanClosure(
                closure.name,
                closure.captures,
                tuple(items),
                closure.closure_id,
                closure.value_shapes,
                closure.value_ranks,
                closure.value_views,
            )

        updated = rewrite(current)
        if not changed:
            break
        current = updated
        rounds += 1
    return HierarchyIdentityReduction(
        current,
        tuple(dict.fromkeys(collapsed)),
        rounds,
    )


def assign_hierarchy_ids(
    root: PlanClosure,
    previous: HierarchyValueTable | None = None,
    *,
    shell: Any = None,
) -> tuple[PlanClosure, HierarchyValueTable]:
    """Assign dense IDs from deterministic scoped-identity token ordering.

    ``(closure_id, local_id)`` is a scoped source address, not a second
    runtime identity.  The returned global ID is the one semantic identity
    used by every later compiler stage.  ``previous`` is accepted for API
    compatibility but never influences the result: unchanged plan structure
    always produces the same dense IDs without a cache or dispenser.

    ``shell`` is the deployment whose plan ``root`` is.  With it, the
    correlation is put on the book (plan 80, A2.5): a hierarchy scope is
    minted, one ``hierarchy_value`` row per ``(closure, local)`` key is
    posted DERIVED from the local value's identity cell in its function's
    graph (a region closure's values are its enclosing function's; a
    ``PlanCall``'s callee is ``shell.callsite_function_shells[callsite]``),
    and one ``hierarchy_global_value`` row per equivalence class DERIVED
    from its members' cells.  Global ids are dense ``enumerate`` positions
    consumed dense: DERIVED, not minted.
    """

    next_closure = 0

    def number(closure: PlanClosure) -> PlanClosure:
        nonlocal next_closure
        closure_id = next_closure
        next_closure += 1
        items = tuple(
            PlanCall(
                item.callsite_id,
                number(item.callee),
                item.argument_value_ids,
                item.result_value_ids,
                item.argument_bindings,
                item.result_bindings,
                item.enclosing_loop_ids,
            )
            if isinstance(item, PlanCall)
            else number(item)
            if isinstance(item, PlanClosure)
            else item
            for item in closure.items
        )
        return PlanClosure(
            closure.name,
            closure.captures,
            items,
            closure_id,
            closure.value_shapes,
            closure.value_ranks,
            closure.value_views,
        )

    planned = number(root)
    keys: set[tuple[int, int]] = set()
    unions: list[
        tuple[tuple[int, int], tuple[int, int]]
    ] = []

    def collect(closure: PlanClosure) -> None:
        closure_id = int(closure.closure_id)
        for local_id in closure.captures:
            keys.add((closure_id, int(local_id)))
        for item in closure.items:
            if isinstance(item, PlanLine):
                for local_id in (*item.inputs, *item.outputs):
                    keys.add((closure_id, int(local_id)))
            elif isinstance(item, PlanCall):
                child_id = int(item.callee.closure_id)
                for local_id in (
                    *item.argument_value_ids,
                    *item.result_value_ids,
                ):
                    keys.add((closure_id, int(local_id)))
                for caller, callee in item.argument_bindings:
                    left = (closure_id, int(caller))
                    right = (child_id, int(callee))
                    keys.update((left, right))
                    unions.append((left, right))
                for callee, caller in item.result_bindings:
                    left = (child_id, int(callee))
                    right = (closure_id, int(caller))
                    keys.update((left, right))
                    unions.append((left, right))
                collect(item.callee)
            elif isinstance(item, PlanClosure):
                collect(item)

    collect(planned)
    parents = {key: key for key in keys}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for left, right in unions:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    del previous
    equivalence_classes: dict[
        tuple[int, int], list[tuple[int, int]]
    ] = {}
    for key in keys:
        equivalence_classes.setdefault(find(key), []).append(key)
    class_tokens = {
        root_key: tuple(sorted(members))
        for root_key, members in equivalence_classes.items()
    }
    root_ids = {
        root_key: global_id
        for global_id, root_key in enumerate(
            sorted(class_tokens, key=lambda item: class_tokens[item])
        )
    }
    correlations = []
    for closure_id, local_id in sorted(keys):
        root_key = find((closure_id, local_id))
        global_id = root_ids[root_key]
        correlations.append((closure_id, local_id, global_id))
    scope = None
    if shell is not None:
        scope = _post_hierarchy_values(planned, shell, correlations, find)
    return planned, HierarchyValueTable(tuple(correlations), scope)


def _closure_graphs(planned: PlanClosure, shell: Any) -> dict[int, Any]:
    """closure id -> the process graph whose values that closure names."""

    graphs: dict[int, Any] = {}

    def visit(closure: PlanClosure, owner: Any) -> None:
        graph = getattr(owner, "process_graph", None)
        if graph is not None:
            graphs[int(closure.closure_id)] = graph
        children = getattr(owner, "callsite_function_shells", {}) or {}
        for item in closure.items:
            if isinstance(item, PlanCall):
                child = children.get(int(item.callsite_id))
                visit(item.callee, child if child is not None else owner)
            elif isinstance(item, PlanClosure):
                # A region closure's values are its function's.
                visit(item, owner)

    visit(planned, shell)
    return graphs


def _post_hierarchy_values(
    planned: PlanClosure, shell: Any, correlations: list, find: Any,
) -> tuple:
    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import (
        HIERARCHY_GLOBAL_VALUE, HIERARCHY_VALUE, PLANNER_HIERARCHY,
        PLANNER_SCOPE, SYNTHESIZED_NO_SOURCE,
    )
    from .identity_concordance import (
        Derived, Mode, Unsourced, current_identity_book,
    )

    book = current_identity_book()
    scope = book.mint_scope("hierarchy", PLANNER_SCOPE)
    graphs = _closure_graphs(planned, shell)
    members: dict[int, list] = {}
    for closure_id, local_id, global_id in correlations:
        row = (scope, int(closure_id), int(local_id))
        graph = graphs.get(int(closure_id))
        cell = None
        if graph is not None and int(local_id) in graph.G:
            try:
                cell = node_identity_cell(graph, int(local_id))
            except ValueError:
                cell = None
        if cell is None:
            # A planner-minted value with no row of its own (a projection
            # leaf, an expanded identity body): the correlation is recorded
            # and the audit lists which writer still mints without a row
            # (plan 80, R4).
            ref = book.post(
                HIERARCHY_VALUE, row, int(global_id), stage=PLANNER_HIERARCHY,
                provenance=Unsourced(SYNTHESIZED_NO_SOURCE), mode=Mode.CONCORD,
            )
        else:
            ref = book.post(
                HIERARCHY_VALUE, row, int(global_id), stage=PLANNER_HIERARCHY,
                provenance=Derived((cell,)), mode=Mode.CONCORD,
            )
        members.setdefault(int(global_id), []).append(ref)
    for global_id, refs in members.items():
        book.post(
            HIERARCHY_GLOBAL_VALUE, (scope, int(global_id)), tuple(refs),
            stage=PLANNER_HIERARCHY, provenance=Derived(tuple(refs)),
            mode=Mode.CONCORD,
        )
    return scope


def render_plan_ascii(root: PlanClosure) -> str:
    """Render a stable tree view without changing the planning object."""

    lines: list[str] = []

    def visit(item: PlanItem, prefix: str, last: bool) -> None:
        branch = "`- " if last else "|- "
        if isinstance(item, PlanCall):
            arguments = ",".join(map(str, item.argument_value_ids)) or "-"
            results = ",".join(map(str, item.result_value_ids)) or "-"
            argument_bindings = ",".join(
                f"{caller}->{callee}"
                for caller, callee in item.argument_bindings
            ) or "-"
            result_bindings = ",".join(
                f"{callee}->{caller}"
                for callee, caller in item.result_bindings
            ) or "-"
            lines.append(
                f"{prefix}{branch}call #{item.callsite_id} "
                f"args=[{arguments}] results=[{results}] "
                f"arg-bind=[{argument_bindings}] "
                f"result-bind=[{result_bindings}]"
            )
            child_prefix = prefix + ("   " if last else "|  ")
            visit(item.callee, child_prefix, True)
            return
        if isinstance(item, PlanClosure):
            captures = ",".join(map(str, item.captures)) or "-"
            lines.append(
                f"{prefix}{branch}closure {item.name} "
                f"id={item.closure_id} captures=[{captures}]"
            )
            child_prefix = prefix + ("   " if last else "|  ")
            for index, child in enumerate(item.items):
                visit(child, child_prefix, index == len(item.items) - 1)
            return
        inputs = ",".join(map(str, item.inputs)) or "-"
        outputs = ",".join(map(str, item.outputs)) or "-"
        roles = ",".join(item.input_roles) or "-"
        attributes = " ".join(
            f"{name}={value!r}" for name, value in item.attributes
        )
        suffix = f" {attributes}" if attributes else ""
        lines.append(
            f"{prefix}{branch}{item.opcode} in=[{inputs}] roles=[{roles}] "
            f"out=[{outputs}]"
            f"{suffix}"
        )

    visit(root, "", True)
    return "\n".join(lines)


__all__ = [
    "PlanCall",
    "PlanClosure",
    "PlanItem",
    "PlanLine",
    "HierarchyValueTable",
    "HierarchyIdentityReduction",
    "assign_hierarchy_ids",
    "reduce_hierarchy_identities",
    "render_plan_ascii",
    "plan_region_to_ssa_instrs",
]

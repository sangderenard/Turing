"""Backend-neutral lowering of subscript ops to SSA address primitives.

``d[i]`` and ``d[i] = v`` arrive as the ops ``Indexed(base, index...) -> res``
and ``IndexedStore(base, index..., value) -> res``. Once every selector is a
scalar, both are an *address* into ``base`` followed by a load or a store. This
module rewrites that proven scalar case to ``GetElementPtr`` plus ``Load`` /
``Store`` -- the same address vocabulary the repository SSA and every backend
already speak. Slice-bearing operations remain semantic until tensor/layout
settlement can choose a view, gather, or scatter with authoritative extents.

A store mutates ``base`` in place, so its result value (the "returned array")
aliases ``base``: uses of the result are rewritten to ``base`` rather than
carrying a fresh SSA name for the same storage.
"""
from __future__ import annotations

import dataclasses

from .monotonic_ids import GLOBAL_MONOTONIC_IDS
from ..transmogrifier.ssa import Instr, SSAValue

_GATHER = ("Indexed", "gather")
_SCATTER = ("IndexedStore", "index_set")


def lower_indexing_to_ssa_addressing(functions) -> None:
    """Rewrite scalar ``Indexed``/``IndexedStore`` ops to address primitives.

    Slice-bearing operations remain intact for tensor/layout settlement.

    ``functions`` is a mapping of name -> repository SSA ``Function``.
    """

    def fresh() -> SSAValue:
        return SSAValue(GLOBAL_MONOTONIC_IDS.mint())

    for function in functions.values():

        constants = {
            int(instruction.res.id): instruction.attributes.get(
                "constant", instruction.attributes.get("value")
            )
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Const", "const"}
            and instruction.res is not None
        }

        def carries_slice_selector(instruction: Instr) -> bool:
            canonical_op = str(
                instruction.attributes.get("tensor_operation")
                or instruction.attributes.get("tensor")
                or instruction.op
            )
            if canonical_op in _GATHER:
                selectors = instruction.args[1:]
            elif canonical_op in _SCATTER:
                selectors = instruction.args[1:-1]
            else:
                return False
            # ``...`` is a full-extent selector, not an address: it resolves
            # against the base's rank exactly as a slice does.
            return any(
                isinstance(constants.get(int(selector.id)), slice)
                or constants.get(int(selector.id)) is Ellipsis
                for selector in selectors
            )

        # base value each store's result aliases, so later uses read the same
        # storage the store mutated in place.
        aliases: dict[int, SSAValue] = {}
        for block in function.blocks.values():
            rewritten = []
            for instruction in block.instrs:
                canonical_op = str(
                    instruction.attributes.get("tensor_operation")
                    or instruction.attributes.get("tensor")
                    or instruction.op
                )
                if carries_slice_selector(instruction):
                    # A Python slice (or ``...``) is not an address operand. Keep the
                    # semantic indexing operation until tensor/layout
                    # settlement can normalize its retained axes into a
                    # contiguous view, gather/copy, or scatter/update. Only
                    # all-scalar selectors are universal GEP arithmetic.
                    rewritten.append(instruction)
                elif canonical_op in _GATHER and len(instruction.args) >= 2:
                    base, *indices = instruction.args
                    address = fresh()
                    lowered_attributes = {
                        **dict(instruction.attributes or {}),
                        "lowered_from": str(canonical_op),
                    }
                    rewritten.append(
                        Instr(
                            "GetElementPtr", [base, *indices], address,
                            attributes=lowered_attributes,
                        )
                    )
                    rewritten.append(Instr(
                        "Load", [address], instruction.res,
                        attributes=lowered_attributes,
                    ))
                elif canonical_op in _SCATTER and len(instruction.args) >= 3:
                    base = instruction.args[0]
                    value = instruction.args[-1]
                    indices = instruction.args[1:-1]
                    address = fresh()
                    rewritten.append(
                        Instr("GetElementPtr", [base, *indices], address)
                    )
                    rewritten.append(Instr("Store", [value, address], None))
                    if instruction.res is not None:
                        aliases[int(instruction.res.id)] = base
                else:
                    rewritten.append(instruction)
            block.instrs = rewritten

        if not aliases:
            continue

        # A store's result is its mutated base: point every later use at
        # base. Bases CHAIN: when stores to one array are sequential in a
        # block -- which is what every unrolled loop and every size-baked
        # kernel produces -- store #2's base is store #1's result, so the
        # map holds 189 -> 146 -> 2. Resolving one level leaves a use
        # pointing at 146, a version id no instruction defines: the SSA is
        # then use-before-def and the evaluator (honestly) refuses while
        # some emitters (dishonestly) emitted it. Resolve every alias to
        # its ROOT storage.
        def root(value: SSAValue) -> SSAValue:
            seen: set[int] = set()
            while int(value.id) in aliases and int(value.id) not in seen:
                seen.add(int(value.id))
                value = aliases[int(value.id)]
            return value

        for block in function.blocks.values():
            for index, instruction in enumerate(block.instrs):
                if any(int(a.id) in aliases for a in instruction.args):
                    block.instrs[index] = dataclasses.replace(
                        instruction,
                        args=[root(a) if int(a.id) in aliases else a
                              for a in instruction.args],
                    )

    _propagate_scalar_dtypes(functions)


#: Scalar dtypes a ``Const`` may declare; a declared one is kept as stated.
_DECLARED_LITERAL_DTYPES = frozenset({
    "bool",
    "int", "int8", "int16", "int32", "int64", "i32", "i64",
    "float", "float16", "float32", "float64", "double", "f32", "f64",
})


def _propagate_scalar_dtypes(functions) -> None:
    """Settle scalar contracts after structural indexing is expanded.

    ``Indexed`` knows the element dtype of its base, while the universal
    ``GetElementPtr`` temporary intentionally does not.  Preserve that fact
    across the rewrite, then carry casts and integer arithmetic through the
    one function's SSA namespace.  Caller/callee agreement is handled by the
    explicit call signature; a bare integer ID is never used to conflate two
    function-local values.  This is type accounting only: IDs and instruction
    order are unchanged.
    """

    integers = {"int", "int8", "int16", "int32", "int64", "i32", "i64"}
    floats = {"float", "float16", "float32", "float64", "double", "f32", "f64"}
    preserving = {
        "Add", "Sub", "Mul", "FloorDiv", "Mod", "Pow", "Min", "Max",
        "BitAnd", "BitOr", "BitXor", "Shl", "Shr", "Neg", "Abs",
    }
    predicates = {
        "Eq", "Ne", "Lt", "Le", "Gt", "Ge", "ULt", "ULe",
        "LAnd", "LOr", "LNot", "LXor",
    }

    from .concordance_declarations import (
        CONST_DECLARED_WIDTH_NOT_ON_BOOK, INFERRED_INTEGER_WIDEN,
        IntegerWidthDecision, IntegerWidthFact, SCALAR_DTYPE_SETTLEMENT,
        SCALAR_INTEGER_WIDTH, SSA_VALUE, WIDENED_VALUE_NOT_ON_BOOK,
    )
    from .identity_concordance import (
        Derived, Mode, Novel, Unsourced, current_identity_book,
    )
    from .ssa_record_return_state import function_scope_of

    narrow_integers = {"int", "int8", "int16", "int32", "i32", "i64"}
    book = current_identity_book()

    from ..common.tensors.accelerator_backends.llvm_repository_ssa import (
        post_repository_kernel_identity,
    )

    for function in functions.values():
        scope = function_scope_of(function)
        # An imported repository kernel's values get their identity rows on
        # this book (its definition root and per-value rows) before any
        # decision here reads them.
        post_repository_kernel_identity(function, book)

        def value_cell(value_id: int):
            """The value's ``ssa_value`` cell and its fact, or (None, None)."""
            ref = book.latest_ref(SSA_VALUE, (scope, int(value_id)))
            if ref is None:
                return None, None
            return ref, book.pages[SSA_VALUE.name].cells.get(
                (ref.row, ref.column)
            )

        # The keep/widen statement for each result the integer-width rule
        # applied to, posted on ``scalar_integer_width`` once the fixed point
        # settles: (decision, source dtype, settled dtype, source cell).
        width_decisions: dict[int, tuple] = {}
        values: dict[int, list[SSAValue]] = {}
        instructions = []
        for value in function.args:
            values.setdefault(int(value.id), []).append(value)
        for block in function.blocks.values():
            for instruction in block.instrs:
                instructions.append(instruction)
                for value in (
                    *instruction.args,
                    *((instruction.res,) if instruction.res is not None else ()),
                ):
                    values.setdefault(int(value.id), []).append(value)

        dtype_of: dict[int, str] = {}
        for value_id, occurrences in values.items():
            declared = [str(value.dtype) for value in occurrences if value.dtype]
            if "int64" in declared or "i64" in declared:
                dtype_of[value_id] = "int64"
            elif declared:
                dtype_of[value_id] = declared[0]

        address_base: dict[int, int] = {}
        for instruction in instructions:
            if (
                instruction.op == "GetElementPtr"
                and instruction.res is not None
                and instruction.args
            ):
                address_base[int(instruction.res.id)] = int(
                    instruction.args[0].id
                )

        # A Python scalar literal is a weak operand: it names a value, not an
        # element width, so beside a typed float operand it does not widen the
        # result (``float32 * 2.0`` is float32, as in the eager lane).
        weak_literal_ids = {
            int(instruction.res.id)
            for instruction in instructions
            if instruction.op == "Const"
            and instruction.res is not None
            and isinstance(
                instruction.attributes.get("value"), (bool, int, float)
            )
        }

        def declared_float(instruction) -> str | None:
            """The widest float width the non-literal operands declare.

            The declared promotion rule is the precision layer's: the wider
            element type wins (``topological_reducer._widest_element``).
            Python's ``float`` is binary64 here, not that table's alias.
            """

            from ..common.tensors.topological_reducer import _widest_element

            return _widest_element(*(
                {"float": "float64"}.get(spelling, spelling)
                for value in instruction.args
                if int(value.id) not in weak_literal_ids
                for spelling in (dtype_of.get(int(value.id)),)
            ))

        for _ in range(max(1, len(instructions))):
            changed = False
            for instruction in instructions:
                if instruction.res is None:
                    continue
                result_id = int(instruction.res.id)
                operand_dtypes = tuple(
                    dtype_of.get(int(value.id)) for value in instruction.args
                )
                inferred = None
                if instruction.op == "Const":
                    literal = instruction.attributes.get("value")
                    declared = str(instruction.res.dtype or "")
                    inferred = (
                        # A typed pointer literal (including null) is an
                        # address value, not an integer inferred from its
                        # Python spelling. Preserve that physical contract.
                        declared if "ptr" in declared.casefold()
                        # A declared scalar dtype is the literal's identity;
                        # the Python spelling is only evidence when nothing
                        # was declared.  ``Metrics.advanced_dt: float |
                        # None = None`` mints its absent payload as a
                        # float64 ``Const 0``; retyping it int64 from the
                        # ``0`` left the callee returning an int64 slot the
                        # caller's declared field projection reads as float64.
                        else declared if declared in _DECLARED_LITERAL_DTYPES
                        else "bool" if isinstance(literal, bool)
                        else "int64" if isinstance(literal, int)
                        else "float64" if isinstance(literal, float)
                        else None
                    )
                elif instruction.op == "Cast":
                    inferred = instruction.attributes.get("target_dtype")
                elif instruction.op == "Load" and instruction.args:
                    declared = str(instruction.res.dtype or "")
                    if declared and "ptr" not in declared.casefold():
                        # LLVM opaque pointers do not state a pointee type.
                        # The Load result does, as does the original Indexed
                        # result retained by address lowering; never replace
                        # that explicit scalar contract with ``ptr``.
                        inferred = declared
                    else:
                        base_id = address_base.get(
                            int(instruction.args[0].id)
                        )
                        candidate = (
                            None if base_id is None
                            else dtype_of.get(base_id)
                        )
                        inferred = (
                            None
                            if candidate is not None
                            and "ptr" in str(candidate).casefold()
                            else candidate
                        )
                elif instruction.op == "BitLength":
                    inferred = "int64"
                elif instruction.op in predicates:
                    inferred = "bool"
                elif instruction.op in preserving and operand_dtypes and all(
                    candidate is not None for candidate in operand_dtypes
                ):
                    if any(candidate in floats for candidate in operand_dtypes):
                        inferred = declared_float(instruction) or "float64"
                    elif all(
                        candidate in integers for candidate in operand_dtypes
                    ):
                        inferred = "int64"
                elif instruction.op in {"Div", "Sqrt", "Exp", "Log"}:
                    inferred = declared_float(instruction) or "float64"
                # A Const's DECLARED integer width is its identity (rule
                # above) and survives; only an inferred integer widens to
                # int64.  tensor_ssa_lowering mints kernel shape vectors as
                # declared int32 (``int_vector``), the width broadcast_double
                # reads; widening them here stored [2, 2] as i64 which the
                # kernel read as i32 [2, 0]: zero output extent, the
                # broadcast temporary never written, and the linear
                # forward/loss/backward motion ran NaN from ``x @ W + b``.
                #
                # The decision is a posted row (lane A, 2026-10-03), one per
                # (function scope, result) on ``scalar_integer_width``:
                # KEPT_DECLARED is DERIVED from the Const's ``ssa_value``
                # cell, whose fact states the declared width (the minter
                # posted it: ``tensor_ssa_lowering``'s ``int_vector``, the
                # control builder's literals).  A Const the book never saw
                # (a repository kernel's ``llvm_literal``, imported with no
                # ``ssa_value`` rows) keeps its declared width as before but
                # the row is ``Unsourced(CONST_DECLARED_WIDTH_NOT_ON_BOOK)``:
                # the audit's worklist names the importer that owes the cell.
                # WIDENED is NOVEL(INFERRED_INTEGER_WIDEN) from the value's
                # own ``ssa_value`` cell.  The book and the instruction
                # disagreeing about a declared width is a missing edge: raise.
                if inferred in narrow_integers:
                    declared = str(instruction.res.dtype or "")
                    if (
                        instruction.op == "Const"
                        and declared in _DECLARED_LITERAL_DTYPES
                        and inferred == declared
                    ):
                        declared_cell, declared_fact = value_cell(result_id)
                        if declared_cell is not None and str(
                            getattr(declared_fact, "dtype", None)
                        ) != declared:
                            raise ValueError(
                                "declared integer width disagreement for "
                                f"{(scope, result_id)!r}: instruction="
                                f"{declared!r}, book={declared_fact!r}"
                            )
                        width_decisions[result_id] = (
                            IntegerWidthDecision.KEPT_DECLARED,
                            declared, declared, declared_cell,
                        )
                    else:
                        width_decisions[result_id] = (
                            IntegerWidthDecision.WIDENED,
                            str(inferred), "int64", value_cell(result_id)[0],
                        )
                        inferred = "int64"
                if inferred is not None and dtype_of.get(result_id) != inferred:
                    dtype_of[result_id] = str(inferred)
                    changed = True
            if not changed:
                break

        for result_id, (
            decision, source_dtype, dtype, source_cell,
        ) in width_decisions.items():
            fact = IntegerWidthFact(decision, source_dtype, dtype)
            if source_cell is None:
                provenance = Unsourced(
                    CONST_DECLARED_WIDTH_NOT_ON_BOOK
                    if decision is IntegerWidthDecision.KEPT_DECLARED
                    else WIDENED_VALUE_NOT_ON_BOOK
                )
            elif decision is IntegerWidthDecision.KEPT_DECLARED:
                provenance = Derived((source_cell,))
            else:
                provenance = Novel(INFERRED_INTEGER_WIDEN, (source_cell,))
            book.post(
                SCALAR_INTEGER_WIDTH, (scope, int(result_id)), fact,
                stage=SCALAR_DTYPE_SETTLEMENT, provenance=provenance,
                mode=Mode.CONCORD,
            )

        for value_id, dtype in dtype_of.items():
            for value in values.get(value_id, ()):
                value.dtype = dtype

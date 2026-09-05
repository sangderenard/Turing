"""Backend-neutral arithmetic identities over repository SSA.

The first inhabitant of the seam `SSA_IDENTITY_CATALOGUE.md` names: every
`ssa_*` backend consumes the same ``IRModule``, so an identity applied here is
inherited by all seven, including the six with no ``-O2`` behind them. This
module starts with the single highest-value item, constant-exponent ``Pow``
strength reduction (catalogue section 2.1) -- on the reference fluid kernel the
24 ``Pow`` calls with compile-time-constant exponents are the entire measured
cost.

The exact/inexact split is a policy boundary, not an implementation detail:

* exact (fires by default): ``x**2`` -> ``Mul(x, x)`` and ``x**-1`` ->
  ``Div(1, x)``. Both are bit-identical to a correctly rounded ``pow``.
* inexact (fires only when asked): ``x**0.5`` -> ``Sqrt(x)`` (differs at
  ``-0.0``/``-inf``), ``x**-2`` -> ``Div(1, Mul(x, x))`` and ``x**-0.5`` ->
  ``Div(1, Sqrt(x))`` (one extra rounding each). Until a value can carry a
  proven ``positive`` fact (catalogue section 5), these change bits and must
  be opted into, so their numeric consequence is a measured delta rather than
  a silent default.

``x**1``, ``x**0`` and integer exponents beyond ``2`` are deliberately absent:
``x**1`` does not occur in any measured workload and needs use-rewriting,
``x**0`` differs at ``NaN``, and ``x**3`` as ``x*x*x`` rounds twice where a
correctly rounded ``pow`` rounds once.
"""
from __future__ import annotations

import contextlib as _contextlib
import dataclasses

from ..transmogrifier.ssa import Instr, SSAValue

_POW = ("Pow", "pow")

# Exponents whose reduction is bit-identical to a correctly rounded pow.
_EXACT_EXPONENTS = (2.0, -1.0)
# Exponents whose reduction changes bits; opt-in only (see module docstring).
_INEXACT_EXPONENTS = (0.5, -0.5, -2.0)


def _module_watermark(functions) -> int:
    """One past the largest SSA id anywhere in the module.

    Same discipline as ``ir_indexing``: a planner region shares its caller's
    value space, so fresh ids must clear the MODULE's maximum, not one
    function's.
    """

    highest = -1
    for function in functions.values():
        for value in function.args:
            highest = max(highest, int(value.id))
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.res is not None:
                    highest = max(highest, int(instruction.res.id))
                for argument in instruction.args:
                    highest = max(highest, int(argument.id))
    return highest + 1


def _constant_value(instruction) -> float | None:
    """The numeric payload of a ``Const``, or ``None`` if it has none."""

    payload = instruction.attributes.get("constant")
    if payload is None:
        payload = instruction.attributes.get("values")
    if payload is None and "value" in instruction.attributes:
        payload = instruction.attributes.get("value")
    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        return None
    return float(payload)


def reduce_constant_exponent_pow(functions, inexact: bool | None = None) -> dict:
    """Rewrite ``Pow(x, Const c)`` to cheaper exact primitives, in place.

    ``functions`` is a mapping of name -> repository SSA ``Function``.
    ``inexact=True`` additionally fires the bit-changing set; the default asks
    the active :mod:`work_contract` (which still honors the legacy
    ``TURING_POW_INEXACT`` variable as an override), so no flag threads
    through the compile entry points. Returns a count per rewrite kind, for
    logging and for tests.
    """

    if inexact is None:
        from .work_contract import active_contract

        inexact = active_contract().inexact_identities
    allowed = _EXACT_EXPONENTS + (_INEXACT_EXPONENTS if inexact else ())

    next_id = _module_watermark(functions)

    def fresh() -> SSAValue:
        nonlocal next_id
        value = SSAValue(next_id, dtype="float64")
        next_id += 1
        return value

    counts = {"pow_to_mul": 0, "pow_to_reciprocal": 0, "pow_to_sqrt": 0}

    for function in functions.values():
        # Exponent constants live in the same function as their Pow; ids are
        # only comparable within one function's value space.
        constants: dict[int, float] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op == "Const" and instruction.res is not None:
                    value = _constant_value(instruction)
                    if value is not None:
                        constants[int(instruction.res.id)] = value

        # One shared literal 1.0 per function. Minted lazily, but INSERTED only
        # after every block is rewritten: inserting into the entry block while
        # it is itself being rebuilt would be overwritten by its new list.
        one_value: SSAValue | None = None

        def unit() -> SSAValue:
            nonlocal one_value
            if one_value is None:
                one_value = fresh()
            return one_value

        def stripped(instruction) -> dict:
            # A backend that prefers `tensor_operation` over `op` must not
            # keep reading "Pow" off the rewritten instruction.
            return {
                key: value
                for key, value in instruction.attributes.items()
                if key not in ("tensor_operation", "tensor")
            }

        for block in function.blocks.values():
            rewritten = []
            for instruction in block.instrs:
                canonical_op = str(
                    instruction.attributes.get("tensor_operation")
                    or instruction.attributes.get("tensor")
                    or instruction.op
                )
                exponent = None
                if (
                    canonical_op in _POW
                    and len(instruction.args) == 2
                    and instruction.res is not None
                ):
                    exponent = constants.get(int(instruction.args[1].id))
                if exponent is None or exponent not in allowed:
                    rewritten.append(instruction)
                    continue

                base = instruction.args[0]
                attributes = stripped(instruction)
                if exponent == 2.0:
                    rewritten.append(dataclasses.replace(
                        instruction, op="Mul", args=[base, base],
                        attributes=attributes,
                    ))
                    counts["pow_to_mul"] += 1
                elif exponent == -1.0:
                    rewritten.append(dataclasses.replace(
                        instruction, op="Div", args=[unit(), base],
                        attributes=attributes,
                    ))
                    counts["pow_to_reciprocal"] += 1
                elif exponent == 0.5:
                    rewritten.append(dataclasses.replace(
                        instruction, op="Sqrt", args=[base],
                        attributes=attributes,
                    ))
                    counts["pow_to_sqrt"] += 1
                elif exponent == -0.5:
                    root = fresh()
                    rewritten.append(
                        Instr("Sqrt", [base], root, attributes=dict(attributes))
                    )
                    rewritten.append(dataclasses.replace(
                        instruction, op="Div", args=[unit(), root],
                        attributes=attributes,
                    ))
                    counts["pow_to_reciprocal"] += 1
                elif exponent == -2.0:
                    square = fresh()
                    rewritten.append(
                        Instr("Mul", [base, base], square,
                              attributes=dict(attributes))
                    )
                    rewritten.append(dataclasses.replace(
                        instruction, op="Div", args=[unit(), square],
                        attributes=attributes,
                    ))
                    counts["pow_to_reciprocal"] += 1
            block.instrs = rewritten

        if one_value is not None:
            # Entry block is first in insertion order and dominates the rest.
            entry = next(iter(function.blocks.values()))
            entry.instrs.insert(
                0, Instr("Const", [], one_value, attributes={"constant": 1.0})
            )

    return counts


# Operations a region may contain and still be removable when nothing reads
# its results: value construction and arithmetic only -- no stores, no
# calls, no table traffic.
_PURE_REGION_OPS = frozenset({
    "Const", "Neg", "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Abs",
    "Max", "Min", "Eq", "Ne", "Lt", "Le", "Gt", "Ge", "FloorDiv", "Mod",
    "Shl", "Shr", "bitand", "bitor", "bitxor", "invert", "LAnd", "LOr",
    "Cast", "range", "string_token", "Ret",
})


def drop_dead_pure_structural_instructions(functions) -> int:
    """Remove unconsumed frontend structural values with no side effects."""

    removed = 0
    for function in functions.values():
        protected: set[int] = set()
        for key in (
            "source_output_value_ids", "required_source_value_ids"
        ):
            protected.update(map(int, function.metadata.get(key) or ()))
        for named_output in function.metadata.get("named_outputs") or ():
            try:
                protected.add(int(named_output[1]))
            except (TypeError, ValueError, IndexError):
                continue
        changed = True
        while changed:
            changed = False
            consumed = {
                int(argument.id)
                for block in function.blocks.values()
                for instruction in block.instrs
                for argument in instruction.args
            }
            for block in function.blocks.values():
                retained = []
                for instruction in block.instrs:
                    result_id = (
                        None if instruction.res is None
                        else int(instruction.res.id)
                    )
                    is_structural = (
                        instruction.attributes.get("structural_operation")
                        is not None
                    )
                    if (
                        result_id is not None
                        and result_id not in consumed
                        and result_id not in protected
                        # Const is intrinsically pure.  Frontend-only values
                        # such as a legalized slice token do not always carry
                        # a structural_operation tag, but once unconsumed they
                        # must disappear here rather than reach a numerical
                        # backend as an impossible literal.
                        and (is_structural or instruction.op == "Const")
                        and instruction.op in _PURE_REGION_OPS
                    ):
                        removed += 1
                        changed = True
                        continue
                    retained.append(instruction)
                block.instrs = retained
    return removed


def drop_dead_pure_region_calls(functions) -> int:
    """Remove aggregate region-call groups whose projections nobody reads.

    Catalogue section 2.2's first load-bearing inhabitant, motivated by
    completeness rather than speed: the planner occasionally carves a
    value's construction into its own region, and the caller's loop
    machinery then never reads the projected results (the materialized
    ``range`` of a comprehension is the observed case, in ``re``'s
    ``_mk_bitmap``). The dead group still emits, so a backend without a
    spelling for the region's payload reports a shortfall for code nothing
    runs.

    Removal is conservative: the callee body must be pure (every op in
    ``_PURE_REGION_OPS``), the aggregate and every projection pointer must
    be consumed only inside the group, and every projected ``Load`` result
    must be unconsumed and absent from the caller's declared outputs.
    Returns the number of removed call groups.
    """

    removed_total = 0
    removed_callees: set[str] = set()
    for function in functions.values():
        protected: set[int] = set()
        for key in (
            "source_output_value_ids",
            # Source-linked calls and structural record construction are
            # installed after region carving.  Their exact authored feeds are
            # a liveness contract even if a late control rewrite has not yet
            # placed the consuming instruction beside the region projection.
            "required_source_value_ids",
        ):
            protected.update(
                int(value_id)
                for value_id in (function.metadata.get(key) or ())
            )
        for name_value in (function.metadata.get("named_outputs") or ()):
            try:
                protected.add(int(name_value[1]))
            except (TypeError, ValueError, IndexError):
                continue
        # Loop-carried ports are consumed through metadata, not necessarily
        # as instruction operands; a projected value serving as a port must
        # never be swept.
        for port_id, port_value in dict(
            function.metadata.get("carried_port_values") or {}
        ).items():
            protected.add(int(port_id))
            try:
                protected.add(int(port_value.id))
            except (AttributeError, TypeError, ValueError):
                continue

        consumers: dict[int, int] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                for argument in instruction.args:
                    identity = int(argument.id)
                    consumers[identity] = consumers.get(identity, 0) + 1
                if instruction.op in ("Phi", "phi"):
                    # Phi incoming values may live only in the attribute
                    # record, not in args; the backends' latch copies are
                    # generated from exactly that record, so it consumes.
                    for _predecessor, value in (
                        instruction.attributes.get("incoming") or ()
                    ):
                        try:
                            identity = int(value.id)
                        except (AttributeError, TypeError, ValueError):
                            continue
                        consumers[identity] = consumers.get(identity, 0) + 1

        for block in function.blocks.values():
            drop: set[int] = set()
            for index, instruction in enumerate(block.instrs):
                if (
                    instruction.op not in ("Call", "call")
                    or instruction.attributes.get("result_convention")
                    != "ssa.aggregate"
                    or instruction.res is None
                ):
                    continue
                callee = functions.get(
                    str(instruction.attributes.get("callee") or "")
                )
                if callee is None or any(
                    body_instruction.op not in _PURE_REGION_OPS
                    for body_block in callee.blocks.values()
                    for body_instruction in body_block.instrs
                ):
                    continue
                aggregate_id = int(instruction.res.id)
                group = [index]
                pointer_ids: set[int] = set()
                load_results: list = []
                index_const_ids: set[int] = set()
                for follow_index in range(index + 1, len(block.instrs)):
                    follower = block.instrs[follow_index]
                    if follower.res is None:
                        continue
                    if follower.op == "GetElementPtr" and follower.args and (
                        int(follower.args[0].id) == aggregate_id
                    ):
                        group.append(follow_index)
                        pointer_ids.add(int(follower.res.id))
                        for operand in follower.args[1:]:
                            index_const_ids.add(int(operand.id))
                    elif follower.op == "Load" and follower.args and (
                        int(follower.args[0].id) in pointer_ids
                    ):
                        group.append(follow_index)
                        load_results.append(follower.res)
                group_indices = set(group)
                # Group-internal consumption of the aggregate and pointers.
                internal: dict[int, int] = {}
                for member_index in group:
                    for operand in block.instrs[member_index].args:
                        identity = int(operand.id)
                        internal[identity] = internal.get(identity, 0) + 1
                if consumers.get(aggregate_id, 0) != internal.get(
                    aggregate_id, 0
                ):
                    continue
                if any(
                    consumers.get(pointer, 0) != internal.get(pointer, 0)
                    for pointer in pointer_ids
                ):
                    continue
                if any(
                    consumers.get(int(value.id), 0) > 0
                    or int(value.id) in protected
                    for value in load_results
                ):
                    continue
                drop.update(group_indices)
                # An index constant consumed only by the removed GEPs goes
                # with them; one shared elsewhere stays.
                for const_index, candidate in enumerate(block.instrs):
                    if (
                        candidate.op == "Const"
                        and candidate.res is not None
                        and int(candidate.res.id) in index_const_ids
                        and consumers.get(int(candidate.res.id), 0)
                        == internal.get(int(candidate.res.id), 0)
                    ):
                        drop.add(const_index)
                removed_total += 1
                removed_callees.add(
                    str(instruction.attributes.get("callee") or "")
                )
            if drop:
                block.instrs = [
                    instruction
                    for index, instruction in enumerate(block.instrs)
                    if index not in drop
                ]
    # A planned region whose last call site was removed would still be emitted
    # as an export root.  The planner can also leave a structural constant in
    # an independently minted region after the owner has materialized the same
    # deterministic source value directly (``b"\0asm"`` in build_module).
    # Such an uncalled region is not a compilation unit when its own explicit
    # source-integral contract publishes no outputs.  Sweep precisely that
    # case and receipt it on the owner; authored functions, effectful regions,
    # and independently useful regions with outputs all remain untouched.
    still_called = {
        str(instruction.attributes.get("callee") or "")
        for function in functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op in ("Call", "call")
    }
    if removed_callees:
        for callee_name in removed_callees:
            if (
                callee_name in functions
                and callee_name not in still_called
                and "__planned_region_" in callee_name
            ):
                del functions[callee_name]
    for region_name, region in tuple(functions.items()):
        integral = dict(region.metadata.get("source_region_integral") or {})
        if (
            "__planned_region_" not in str(region_name)
            or region_name in still_called
            or "output_value_ids" not in integral
            or tuple(integral.get("output_value_ids") or ())
            or any(
                instruction.op not in _PURE_REGION_OPS
                for block in region.blocks.values()
                for instruction in block.instrs
            )
        ):
            continue
        owner_name = str(integral.get("owner") or "")
        owner = functions.get(owner_name)
        if owner is not None:
            receipts = list(owner.metadata.get(
                "discarded_outputless_source_regions", ()
            ))
            receipts.append({
                "function": str(region_name),
                "identity_token_chain": tuple(
                    integral.get("identity_token_chain") or ()
                ),
                "reason": "uncalled-pure-no-published-outputs",
            })
            owner.metadata["discarded_outputless_source_regions"] = tuple(
                receipts
            )
        del functions[region_name]
        removed_total += 1
    return removed_total



#: Limb elements narrower than the binary64 default. Carrying one of these
#: retypes the value, because on a lane with no f64 the two statements
#: cannot be allowed to disagree.
_NARROW_LIMB_ELEMENTS = frozenset({"float32", "f32", "single"})

_FLOAT64_SPELLINGS = frozenset({"float64", "double", "f64"})
_INT64_SPELLINGS = frozenset({"int64", "i64", "long"})


def narrow_float64_to_float32(module) -> int:
    """Retype every remaining binary64 value as binary32, once it is safe.

    The GPU lanes have no f64 AT ALL -- WGSL does not define the type --
    so a float64 value reaching them is not a precision choice, it is an
    impossibility. The choice that IS available is the one the ladder was
    built for: two float32 limbs in place of one float64, which carries
    more significand than the f64 it replaces rather than less.

    The precision expansion already types its own limbs from the declared
    element, so after it runs what is left at float64 is the surrounding
    scaffolding -- formals, the loads through them, and the exact integer
    floors of range reduction, none of which hold a limbed quantity. This
    narrows that scaffolding so the module speaks one element throughout.

    REFUSED when any precision section declares an element other than
    float32, because then the float64 values are load-bearing and
    narrowing them would be exactly the silent precision loss the whole
    pipeline exists to prevent. A module with no sections at all is also
    refused: nothing there is carrying precision in limbs, so narrowing it
    would simply be throwing digits away where the caller asked for none
    of this.

    Returns how many values were retyped.
    """

    receipt = (getattr(module, "metadata", {}) or {}).get(
        PRECISION_PIPELINE_METADATA
    ) or {}
    contracts = tuple(receipt.get("section_contracts", ()))
    if not contracts:
        raise ValueError(
            "narrow_float64_to_float32 refuses a module with no precision "
            "sections: there are no limbs carrying the precision that "
            "narrowing would rely on, so this would only lose digits"
        )
    declared = {
        str(contract.get("element") or "float64") for contract in contracts
    }
    unsupported = {
        element for element in declared
        if element not in _NARROW_LIMB_ELEMENTS
    }
    if unsupported:
        raise ValueError(
            "narrow_float64_to_float32 refuses a module whose sections "
            f"declare {sorted(unsupported)}: those limbs ARE float64 and "
            "narrowing them would silently halve the precision the caller "
            "declared"
        )

    seen: set[int] = set()
    narrowed = 0

    def retype(value) -> None:
        nonlocal narrowed
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        dtype = str(getattr(value, "dtype", None) or "")
        if dtype in _FLOAT64_SPELLINGS:
            value.dtype = "float32"
            narrowed += 1
        elif dtype in _INT64_SPELLINGS:
            # WGSL has no 64-bit integer either. These are the index and
            # quarter-turn counts around the arithmetic, not quantities --
            # a field addressed by an i32 index is bounded by the same
            # limit its buffers already are.
            value.dtype = "int32"
            narrowed += 1

    for function in (getattr(module, "functions", module) or {}).values():
        for formal in getattr(function, "args", ()):
            retype(formal)
        for block in getattr(function, "blocks", {}).values():
            for instruction in getattr(block, "instrs", ()):
                retype(getattr(instruction, "res", None))
                for argument in getattr(instruction, "args", ()):
                    retype(argument)
    return narrowed


#: Operations whose result is a truth value rather than a quantity.
_PREDICATE_NAMES = frozenset({
    "Lt", "Le", "Gt", "Ge", "Eq", "Ne",
    "lt", "le", "gt", "ge", "eq", "ne",
})



def _is_predicate_derived(instruction, definitions, predicates,
                          depth: int = 4) -> bool:
    """Whether this value descends from a comparison within a few steps.

    Shallow on purpose: a mask is formed close to the comparison that
    made it, and a deep search would start refusing genuine arithmetic
    that merely happens to sit downstream of a branch condition.
    """

    if depth <= 0:
        return False
    for argument in getattr(instruction, "args", ()):
        identifier = int(argument.id)
        if identifier in predicates:
            return True
        producer = definitions.get(identifier)
        if producer is not None and _is_predicate_derived(
            producer, definitions, predicates, depth - 1
        ):
            return True
    return False


def carry_precision_through_ssa(functions) -> int:
    """Make a precision result self-describing, then rename what reads it.

    Not an identity -- nothing here is rewritten into a cheaper equivalent.
    It lives beside them because it consumes repository SSA under the same
    ``(functions) -> count`` shape, and because every identity written for
    the precision operations needs this to have run first: until it does,
    precision stops at the first undeclared temporary and a chain is only
    ever visible as isolated operations.

    Ingestion can name an operation ``precision_*`` only where the source
    DECLARED an operand's width, because it walks the AST and no SSA value
    exists yet to ask. The result of such an operation is a genuine limbed
    quantity, but it is emitted as a bare scalar -- ``float64``, shape
    ``()`` -- so the next operation cannot tell it apart from ordinary
    arithmetic and lowers as ordinary arithmetic.

    Here the values exist, so the property can live on the value the way it
    lives on an ``AbstractTensor``: the limb count becomes a channel in the
    LAST dimension, which is the same layout the tensor uses and the same
    one a backend will stride. Shape alone is ambiguous -- a trailing axis
    of 2 could be a real tensor axis -- so ``accounting`` records that the
    axis IS limbs. The shape is the layout; the accounting is the fact.

    Propagation then stops being inference and becomes a read: an ordinary
    operation one of whose operands carries the limb fact is a precision
    operation, and is renamed to say so. Iterated to a fixed point, since
    renaming one instruction makes its result the evidence for the next.
    """

    from ..common.tensors.topological_reducer import (
        PRECISION_CLOSED_OPERATIONS, PRECISION_SINGULAR_NAMES,
    )

    singular = dict(PRECISION_SINGULAR_NAMES)
    planted = {name: operation for operation, name in singular.items()}
    changed_total = 0

    def limbs_of(value) -> int:
        try:
            record = value.accounting or {}
        except AttributeError:
            return 1
        return max(int(record.get("precision_limbs") or 1), 1)

    def carry(value, limbs: int, element: str | None) -> bool:
        """Give one result value the limb channel and the fact."""

        if value is None or limbs <= 1 or limbs_of(value) == limbs:
            return False
        value.accounting["precision_limbs"] = int(limbs)
        resolved = str(element or value.dtype or "") or None
        value.accounting["precision_element"] = resolved
        # The element is what the limbs ARE, so it is the value's dtype and
        # not merely a note beside it. Leaving the two to disagree is
        # invisible on a lane that has both widths and fatal on one that
        # does not: a float32 expansion kept saying float64, and the
        # WebGPU emitter -- where there IS no f64 -- refused sixty-seven
        # values whose limbs were already 24-bit. Only a NARROWING is
        # written here; the default element is the dtype the value already
        # carries, so a binary64 program is untouched.
        if resolved in _NARROW_LIMB_ELEMENTS and value.dtype != resolved:
            value.dtype = resolved
        existing = tuple(value.shape or ())
        # Idempotent: a re-run must not append a second channel.
        if not existing or existing[-1] != int(limbs):
            value.shape = (*existing, int(limbs))
        return True

    for function in functions.values():
        while True:
            changed = 0
            for block in function.blocks.values():
                for instruction in block.instrs:
                    operation = str(instruction.op)
                    attributes = instruction.attributes
                    if operation in planted:
                        limbs = max(
                            int(attributes.get("precision_limbs") or 1), 1
                        )
                        element = attributes.get("precision_element")
                    elif operation in PRECISION_CLOSED_OPERATIONS:
                        # An ordinary operation reading a limbed operand.
                        # Most limbs and highest precision decide, exactly
                        # as they do at ingestion.
                        limbs = max(
                            (limbs_of(value) for value in instruction.args),
                            default=1,
                        )
                        if limbs <= 1:
                            continue
                        element = next(
                            (
                                (value.accounting or {}).get(
                                    "precision_element"
                                )
                                for value in instruction.args
                                if limbs_of(value) > 1
                            ),
                            None,
                        )
                        instruction.op = singular[operation]
                        attributes["precision_limbs"] = int(limbs)
                        attributes["precision_element"] = element
                        attributes["lowered_from"] = operation
                        changed += 1
                    else:
                        continue
                    if carry(instruction.res, limbs, element):
                        changed += 1
            # The backward half of the same fact: a value CONSUMED at width
            # n must be PRODUCED at width n. Forward renaming alone spreads
            # only from limbed values outward, which strands any plain
            # subchain that merely feeds a precision operation -- the even
            # cores' ``s = z * z`` computed both operands unlimbed, so the
            # squaring stayed ordinary, the loads behind ``z`` were never
            # recognised as limb-strided, and a two-limb kernel silently
            # read the wrong elements. Renaming the defining instruction
            # here makes its operands evidence for the next forward sweep,
            # so the recognition walks all the way back to the loads.
            definitions = {
                int(instruction.res.id): instruction
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            }
            predicates = {
                int(instruction.res.id)
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
                and str(instruction.op) in _PREDICATE_NAMES
            }
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if str(instruction.op) not in planted:
                        continue
                    limbs = max(int(
                        instruction.attributes.get("precision_limbs") or 1
                    ), 1)
                    if limbs <= 1:
                        continue
                    element = instruction.attributes.get("precision_element")
                    for argument in instruction.args:
                        if limbs_of(argument) > 1:
                            continue
                        producer = definitions.get(int(argument.id))
                        if (
                            producer is None
                            or str(producer.op)
                            not in PRECISION_CLOSED_OPERATIONS
                        ):
                            continue
                        # A value built from a COMPARISON is a mask: it is
                        # zero or one, exactly representable in one limb,
                        # and there is nothing for a second limb to hold.
                        # Widening it anyway is not merely wasted work --
                        # it puts an error-free transformation on a
                        # boolean, and the LLVM lane then tracks the same
                        # register as both i1 and double and emits a
                        # two_product residual over a predicate. Measured,
                        # a quadrant blend that the C lane computed
                        # exactly came back off by half on LLVM. The
                        # forward pass never reached these because a mask
                        # has no limbed operand; only this backward half
                        # can, so the guard belongs here.
                        if _is_predicate_derived(
                            producer, definitions, predicates
                        ):
                            continue
                        producer.attributes["precision_limbs"] = int(limbs)
                        producer.attributes["precision_element"] = element
                        producer.attributes["lowered_from"] = str(producer.op)
                        producer.op = singular[str(producer.op)]
                        changed += 1
            if not changed:
                break
            changed_total += changed
    return changed_total


@dataclasses.dataclass(frozen=True)
class PrecisionIdentity:
    """One reduction the precision operations admit, and why it is true.

    ``requires`` is what must exist before it may fire -- a proven fact on a
    value, or a kernel the bank can serve. An empty tuple is unconditional.
    ``exact`` means the rewrite changes no bit of the result; an inexact one
    would need the work contract's licence, and there are none here on
    purpose: every entry below is a theorem, not a trade.
    """

    name: str
    matches: tuple[str, ...]
    exact: bool
    requires: tuple[str, ...]
    effect: str
    justification: str


#: The identity table for the precision reduction phase.
#:
#: Ordered by what they save, cheapest evidence first. An expansion operation
#: is expensive -- a two-limb product is seven flops before renormalisation --
#: so the identities proving one ISN'T NEEDED are worth more than any that
#: make one faster, and they come first.
PRECISION_IDENTITIES: tuple[PrecisionIdentity, ...] = (
    PrecisionIdentity(
        name="negation_is_per_limb",
        matches=("precision_neg",),
        exact=True,
        requires=(),
        effect="negate each limb; no error-free transformation at all",
        justification=(
            "Negation is exact in every binary floating-point format: it "
            "flips a sign bit and cannot round. An expansion is the sum of "
            "its limbs, so negating each limb negates the sum exactly and "
            "preserves nonoverlapping-ness. precision_neg is therefore "
            "never an expansion operation, only a channel-wise sign flip."
        ),
    ),
    PrecisionIdentity(
        name="subtraction_is_addition_of_negation",
        matches=("precision_sub",),
        exact=True,
        requires=("negation_is_per_limb",),
        effect="rewrite sub(a, b) to add(a, neg(b))",
        justification=(
            "Exact because the negation it introduces is exact. Its value "
            "is not speed -- it is that a destination implements four "
            "operations instead of five, and the subtraction path cannot "
            "drift from the addition path because there is only one."
        ),
    ),
    PrecisionIdentity(
        name="exact_identity_element",
        matches=("precision_add", "precision_mul"),
        exact=True,
        requires=("constant_operand", "operand_special_value_fact"),
        effect="add(x, 0) -> x; mul(x, 1) -> x",
        justification=(
            "Only against a literal exact zero or one AND proved operand "
            "special-value facts. A value that merely "
            "ROUNDS to zero or one is not an identity element: adding a "
            "denormal changes the low limb even when it cannot change the "
            "high one, and that low limb is the entire point of the "
            "representation. Signed zero and NaN payload behaviour also "
            "disqualify a blind use rewrite: even x + (+0.0) changes -0.0."
        ),
    ),
    PrecisionIdentity(
        name="scaling_by_power_of_two",
        matches=("precision_mul",),
        exact=True,
        requires=("constant_operand", "exponent_range_fact"),
        effect="scale every limb by the power of two; no transformation",
        justification=(
            "Within a proved normal exponent range, multiplication by a "
            "power of two only shifts the exponent, so it is exact for every "
            "limb independently and preserves the "
            "nonoverlapping property that makes the limbs a valid "
            "expansion. This is what makes Cody-Waite argument reduction "
            "affordable. Without that range fact, overflow and gradual "
            "underflow make the shortcut false and it must not fire."
        ),
    ),
    PrecisionIdentity(
        name="sterbenz_cancellation",
        matches=("precision_sub",),
        exact=True,
        requires=("operand_range_fact",),
        effect="collapse to an ordinary Sub",
        justification=(
            "Sterbenz's lemma: if b/2 <= a <= 2b then a - b is exactly "
            "representable, so the dual is provably zero and computing it "
            "is computing zero the long way. This is the identity that took "
            "expm1 to 1.75 ulp with no core at all. It needs a proven range "
            "on both operands -- catalogue section 5's fact slot -- and must "
            "never fire on a guess, because where it is wrong it is wrong "
            "silently and by the whole residual."
        ),
    ),
    PrecisionIdentity(
        name="single_renormalisation_per_chain",
        matches=("precision_add", "precision_sub"),
        exact=True,
        requires=("chain_internal_consumers", "fixed_width_chain_proof"),
        effect="renormalise once at the end of a chain, not per operation",
        justification=(
            "Renormalisation itself preserves an expansion's exact sum, but "
            "a FIXED-width chain may discard different trailing terms when "
            "the intermediate renormalisation moves. The shortcut therefore "
            "requires a proof that the chain's retained width contains every "
            "intermediate term; internal-consumer shape alone is not proof."
        ),
    ),
    PrecisionIdentity(
        name="exact_accumulation_over_long_chain",
        matches=("precision_add",),
        exact=True,
        requires=("chain_internal_consumers", "kernel:superaccumulator"),
        effect="accumulate into a fixed-point superaccumulator",
        justification=(
            "Past some chain length, not renormalising at all beats "
            "renormalising once: an accumulator wide enough to span the "
            "exponent range absorbs each term in constant time with no "
            "rounding whatsoever, and is read out once. It is exact for ANY "
            "number of terms and any conditioning, which a fixed-width "
            "expansion is not -- so this is also the only entry here that "
            "improves accuracy rather than merely preserving it."
        ),
    ),
    PrecisionIdentity(
        name="two_product_kernel",
        matches=("precision_mul",),
        exact=True,
        requires=("kernel:two_product",),
        effect="call the banked two_product kernel for the dual",
        justification=(
            "The dual of a product is Dekker's splitting -- six multiplies "
            "and several adds -- or, where the hardware has a fused "
            "multiply-add, the single instruction fma(a, b, -p). Both are "
            "exact and they agree bit for bit, so this is variant selection "
            "rather than a licence to be inexact. Making it a BANKED KERNEL "
            "rather than a backend rule is what makes that claim checkable: "
            "the bank verifies every variant against the spec's oracle "
            "before serving it, so the fused variant is proven equal to the "
            "split one rather than trusted because a capability flag was set."
        ),
    ),
)


#: Identities true over the reals and false over floating point.
#:
#: Recorded because a reduction phase that does not know them will find them:
#: each is a rewrite some optimizer already performs, and each destroys the
#: representation silently. This is the table's other half -- knowing what
#: must NOT reduce is the same knowledge as knowing what may.
PRECISION_REFUSALS: tuple[PrecisionIdentity, ...] = (
    PrecisionIdentity(
        name="cancellation_of_added_term",
        matches=("precision_add", "precision_sub"),
        exact=False,
        requires=("never",),
        effect="REFUSED: (a + b) - b is not a",
        justification=(
            "The whole representation exists because a + b loses bits that "
            "b cannot give back. This rewrite is exactly the assumption "
            "that no bits were lost, which is the assumption the limbs "
            "exist to deny."
        ),
    ),
    PrecisionIdentity(
        name="reassociation_of_the_dual",
        matches=("precision_add", "precision_sub", "precision_mul"),
        exact=False,
        requires=("never",),
        effect="REFUSED: the error term must keep its written association",
        justification=(
            "A dual expression is algebraically zero -- that is what makes "
            "it the error term -- so a reassociating simplifier that sees "
            "it whole can fold it away entirely. Knuth's and Dekker's "
            "proofs hold for one specific order of operations and for no "
            "other; the primal may be reassociated freely, the dual may not "
            "be touched at all. "
            "MEASURED, because the stronger claim is easy to assume and was "
            "asserted here before it was checked: gfortran at -ffast-math "
            "did NOT fold a raw two_sum residual, on inputs that genuinely "
            "lose bits. Lowering one SSA value per statement, through named "
            "temporaries loaded from and stored to arrays, appears to be "
            "why -- no simplifier ever sees the residual as one expression. "
            "So this refusal is a standing hazard rather than an observed "
            "failure, and the structure is carrying more of the protection "
            "than the isolation flags are. Treat a destination that inlines "
            "more aggressively as untested, not as safe."
        ),
    ),
    PrecisionIdentity(
        name="contraction_across_the_primal",
        matches=("precision_add", "precision_sub"),
        exact=False,
        requires=("never",),
        effect="REFUSED: the primal must not fuse with its own producers",
        justification=(
            "two_sum requires s to be the correctly rounded sum of exactly "
            "its two operands. If an operand was itself a product and the "
            "multiply-add contracts, s becomes the rounded value of a "
            "different expression while the dual computes its residual "
            "against the materialised operand, and the pair no longer sums "
            "to what it claims. Reassociation of the primal is safe; "
            "contraction across it is not, and the distinction is exactly "
            "one flag on one instruction."
        ),
    ),
    PrecisionIdentity(
        name="distribution_over_addition",
        matches=("precision_mul",),
        exact=False,
        requires=("never",),
        effect="REFUSED: a * (b + c) is not a * b + a * c",
        justification=(
            "Two roundings on the right against one on the left, and the "
            "limb counts differ: the distributed form needs a wider "
            "expansion to hold the same value. Over the reals it is free; "
            "here it changes both the result and the storage it requires."
        ),
    ),
)


def reduce_precision_operations(functions) -> dict:
    """Fire the identities whose evidence is already in the SSA.

    Every rewrite here is exact, so none consults the work contract. Most
    do not restructure anything: they record HOW an operation is to be
    realised -- per-limb, scaled, renormalised or not -- as a fact on the
    instruction. That is the honest shape while destinations are still
    unwritten, because the decision is the identity's to make and the
    spelling is theirs, and it keeps the reduction phase from having to
    know either.

    Absent on purpose: ``exact_identity_element``. Dropping ``add(x, 0)``
    means pointing every consumer at ``x``, and this module deliberately
    avoids use-rewriting -- it is why ``x**1`` is missing from the ``Pow``
    reduction. It needs the same machinery that entry needs, and should
    arrive with it rather than growing a private copy here.

    ``scaling_by_power_of_two`` and ``sterbenz_cancellation`` wait on proven
    range facts; delayed renormalisation waits on a fixed-width chain proof;
    the two kernel entries wait on the bank. Pattern shape is never promoted
    into numerical evidence here.

    Returns a count per identity name.
    """

    from math import frexp

    from ..common.tensors.topological_reducer import PRECISION_SINGULAR_NAMES

    add_name = PRECISION_SINGULAR_NAMES["Add"]
    sub_name = PRECISION_SINGULAR_NAMES["Sub"]
    mul_name = PRECISION_SINGULAR_NAMES["Mul"]
    neg_name = PRECISION_SINGULAR_NAMES["neg"]

    counts = {
        "negation_is_per_limb": 0,
        "subtraction_is_addition_of_negation": 0,
        "scaling_by_power_of_two": 0,
        "single_renormalisation_per_chain": 0,
    }
    next_id = _module_watermark(functions)

    def fresh(like) -> SSAValue:
        nonlocal next_id
        value = SSAValue(
            next_id, dtype=like.dtype, shape=tuple(like.shape or ()),
            device=like.device, accounting=dict(like.accounting or {}),
        )
        next_id += 1
        return value

    def is_power_of_two(value: float) -> bool:
        # Exactly a power of two, and finite: frexp puts every such value
        # at a mantissa of exactly 0.5. Zero and the specials are excluded
        # because scaling by them is not an exponent shift.
        if not value or value != value or value in (float("inf"), -float("inf")):
            return False
        return frexp(abs(value))[0] == 0.5

    for function in functions.values():
        # -- negation is a sign flip, never an expansion -------------------
        for block in function.blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) == neg_name:
                    instruction.attributes["precision_form"] = "per_limb"
                    counts["negation_is_per_limb"] += 1

        # -- subtraction is addition of a negation -------------------------
        # The result value is reused, so no consumer needs rewriting: only
        # a negation is inserted ahead of the operation that keeps the id.
        for block in function.blocks.values():
            if not any(
                str(instruction.op) == sub_name
                for instruction in block.instrs
            ):
                continue
            rewritten = []
            for instruction in block.instrs:
                if str(instruction.op) != sub_name or len(instruction.args) != 2:
                    rewritten.append(instruction)
                    continue
                left, right = instruction.args
                negated = fresh(right)
                rewritten.append(Instr(
                    neg_name, [right], negated,
                    attributes={
                        "precision_form": "per_limb",
                        "precision_limbs": instruction.attributes.get(
                            "precision_limbs"
                        ),
                        "precision_element": instruction.attributes.get(
                            "precision_element"
                        ),
                        "lowered_from": sub_name,
                    },
                ))
                instruction.op = add_name
                instruction.args = [left, negated]
                instruction.attributes["lowered_from"] = sub_name
                rewritten.append(instruction)
                counts["negation_is_per_limb"] += 1
                counts["subtraction_is_addition_of_negation"] += 1
            block.instrs = rewritten

        # -- scaling by a power of two is exact per limb -------------------
        constants: dict[int, float] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op == "Const" and instruction.res is not None:
                    value = _constant_value(instruction)
                    if value is not None:
                        constants[int(instruction.res.id)] = value
        for block in function.blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) != mul_name:
                    continue
                for operand in instruction.args:
                    scale = constants.get(int(operand.id))
                    if scale is None or not is_power_of_two(scale):
                        continue
                    if not instruction.attributes.get(
                        "precision_scale_exponent_range_proven"
                    ):
                        continue
                    instruction.attributes["precision_form"] = "scale_per_limb"
                    instruction.attributes["precision_scale"] = scale
                    counts["scaling_by_power_of_two"] += 1
                    break

        # -- one renormalisation per chain ---------------------------------
        # An intermediate expansion needs renormalising only if something
        # READS it. Anything reached through metadata counts as a reader:
        # a loop-carried port or a declared output is consumed through a
        # channel no scan of operands can see.
        protected: set[int] = set()
        for key in (
            "source_output_value_ids", "required_source_value_ids",
        ):
            protected.update(
                int(value_id)
                for value_id in (function.metadata.get(key) or ())
            )
        for name_value in (function.metadata.get("named_outputs") or ()):
            try:
                protected.add(int(name_value[1]))
            except (TypeError, ValueError, IndexError):
                continue
        for port_id, port_value in dict(
            function.metadata.get("carried_port_values") or {}
        ).items():
            protected.add(int(port_id))
            try:
                protected.add(int(port_value.id))
            except (AttributeError, TypeError, ValueError):
                continue

        readers: dict[int, list] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                for argument in instruction.args:
                    readers.setdefault(int(argument.id), []).append(instruction)
                if instruction.op in ("Phi", "phi"):
                    for _predecessor, value in (
                        instruction.attributes.get("incoming") or ()
                    ):
                        try:
                            readers.setdefault(int(value.id), []).append(
                                instruction
                            )
                        except (AttributeError, TypeError, ValueError):
                            continue

        precision_ops = frozenset(PRECISION_SINGULAR_NAMES.values())
        for block in function.blocks.values():
            for instruction in block.instrs:
                if (
                    str(instruction.op) not in precision_ops
                    or instruction.res is None
                ):
                    continue
                result = int(instruction.res.id)
                consumers = readers.get(result, ())
                # A renormalisation is needed where a value ESCAPES, not
                # between every pair of operations.
                #
                # The rule used to require the single consumer to be another
                # ADD, which is the shape a sum chain has and the shape a
                # Horner never has: there every add feeds a multiply, so the
                # identity matched nothing on the most common expression in
                # the whole signal pack -- 0 of 8 identities fired on a sine
                # core. Any precision operation can consume an unrenormalised
                # pair; what cannot is anything else, because outside the
                # section the pair is just two doubles whose sum is the
                # value.
                internal = (
                    result not in protected
                    and len(consumers) == 1
                    and str(consumers[0].op) in precision_ops
                    and bool(instruction.attributes.get(
                        "precision_fixed_width_chain_proven"
                    ))
                )
                instruction.attributes["precision_renormalise"] = not internal
                if internal:
                    counts["single_renormalisation_per_chain"] += 1
    return counts


#: The fused multiply-add, as repository SSA owns it.
#:
#: ``Fma(x, y, z)`` is ``x * y + z`` under EXACTLY ONE rounding. That single
#: rounding is the entire content of the operation: a separate multiply and
#: add round twice, so the two are different functions, not the same function
#: at different speeds. Every destination has a spelling -- C's ``fma``,
#: GLSL's and WGSL's ``fma``, LLVM's ``llvm.fma`` -- so this is a real
#: operation to be implemented per backend rather than a hint dropped for an
#: optimizer to notice.
#:
#: Naming it here rather than leaving it to LLVM's ``contract`` flag buys
#: determinism: a flag PERMITS fusion, so whether it happens depends on the
#: toolchain, the named target, and the optimizer's mood, and the six
#: backends with no ``-O2`` behind them never see it at all. An instruction
#: either is an Fma or is not.
FMA = "Fma"


def contract_multiply_add_to_fma(functions, licensed: bool | None = None) -> dict:
    """Fuse ``Mul`` feeding an ``Add``/``Sub`` into :data:`FMA`, in place.

    INEXACT in general, and gated accordingly -- the default asks the active
    work contract's ``contract_multiply_add``, the same switch the LLVM
    backend consults. Fusing removes a rounding, which USUALLY improves the
    result and always changes it, so it is a licensed change of bits rather
    than a free win.

    The exception is the reason this exists. Where the multiply and the
    subtraction form an error term -- ``a * b - fl(a * b)`` -- the fused
    form is not merely more accurate but EXACTLY the residual Dekker's
    splitting computes the long way, six multiplies and several adds
    replaced by one instruction. There the rewrite is exact, and the
    licence is beside the point; it needs the dual to be marked as such,
    which is deferred work, so today it rides the general licence.

    Only fires where the product is consumed by that one operation.
    Otherwise the multiply must be kept for its other readers and fusing
    computes it twice -- once fused, once not, at two different roundings,
    which is worse than not fusing at all.

    Returns a count per rewrite kind.
    """

    if licensed is None:
        from .work_contract import active_contract

        licensed = bool(active_contract().contract_multiply_add)
    counts = {"add_to_fma": 0, "sub_to_fma": 0}
    if not licensed:
        return counts

    next_id = _module_watermark(functions)

    for function in functions.values():
        protected: set[int] = set()
        for key in ("source_output_value_ids", "required_source_value_ids"):
            protected.update(
                int(value_id)
                for value_id in (function.metadata.get(key) or ())
            )
        for name_value in (function.metadata.get("named_outputs") or ()):
            try:
                protected.add(int(name_value[1]))
            except (TypeError, ValueError, IndexError):
                continue

        readers: dict[int, int] = {}
        products = {}
        fused: set[int] = set()
        for block in function.blocks.values():
            for instruction in block.instrs:
                for argument in instruction.args:
                    identity = int(argument.id)
                    readers[identity] = readers.get(identity, 0) + 1
                if instruction.op in ("Phi", "phi"):
                    for _predecessor, value in (
                        instruction.attributes.get("incoming") or ()
                    ):
                        try:
                            identity = int(value.id)
                        except (AttributeError, TypeError, ValueError):
                            continue
                        readers[identity] = readers.get(identity, 0) + 1
                if (
                    str(instruction.op) == "Mul"
                    and len(instruction.args) == 2
                    and instruction.res is not None
                ):
                    products[int(instruction.res.id)] = instruction

        for block in function.blocks.values():
            rewritten = []
            for instruction in block.instrs:
                operation = str(instruction.op)
                if operation not in ("Add", "Sub") or len(instruction.args) != 2:
                    rewritten.append(instruction)
                    continue
                left, right = instruction.args
                # `Sub(c, Mul(a, b))` is deliberately not matched: it is
                # `c - a * b`, which needs a negated multiplicand to reach
                # this form, and that is a second rewrite wearing the first
                # one's name.
                product = products.get(int(left.id))
                addend = right
                if product is None and operation == "Add":
                    product = products.get(int(right.id))
                    addend = left
                if product is None:
                    rewritten.append(instruction)
                    continue
                carrier = int(product.res.id)
                if readers.get(carrier, 0) != 1 or carrier in protected:
                    rewritten.append(instruction)
                    continue

                factors = list(product.args)
                if operation == "Sub":
                    negated = SSAValue(
                        next_id, dtype=addend.dtype,
                        shape=tuple(addend.shape or ()),
                        device=addend.device,
                        accounting=dict(addend.accounting or {}),
                    )
                    next_id += 1
                    rewritten.append(Instr(
                        "Neg", [addend], negated,
                        attributes={"lowered_from": "Sub"},
                    ))
                    addend = negated
                    counts["sub_to_fma"] += 1
                else:
                    counts["add_to_fma"] += 1
                instruction.op = FMA
                instruction.args = [*factors, addend]
                instruction.attributes["lowered_from"] = operation
                instruction.attributes["roundings"] = 1
                rewritten.append(instruction)
                fused.add(carrier)
                products.pop(carrier, None)
            block.instrs = rewritten

        # The fused multiplies are dead -- their one reader absorbed them.
        # Established by recounting what the rewritten code actually reads,
        # rather than by reasoning about what should now be unreferenced.
        if fused:
            live: set[int] = set()
            for block in function.blocks.values():
                for instruction in block.instrs:
                    for argument in instruction.args:
                        live.add(int(argument.id))
                    if instruction.op in ("Phi", "phi"):
                        for _predecessor, value in (
                            instruction.attributes.get("incoming") or ()
                        ):
                            try:
                                live.add(int(value.id))
                            except (AttributeError, TypeError, ValueError):
                                continue
            for block in function.blocks.values():
                block.instrs = [
                    each for each in block.instrs
                    if not (
                        each.res is not None
                        and int(each.res.id) in fused
                        and int(each.res.id) not in live
                        and int(each.res.id) not in protected
                    )
                ]
    return counts


# ---------------------------------------------------------------------------
# The API between SSA Fma and precision sections.
#
# A precision section is a stretch of SSA whose operations carry limbs. An
# `Fma` inside one is not the same request as an `Fma` in ordinary code: in
# ordinary code the single rounding is an improvement a contract licenses,
# and emitting a multiply and an add instead is merely a slower, slightly
# different answer. Inside a precision section the single rounding IS the
# operation -- `a * b - fl(a * b)` under two roundings is not a worse
# residual, it is zero -- so the same instruction carries an obligation the
# ordinary one does not.
#
# This is what the two sides have to agree on, stated once here so each
# backend can later be tuned against it rather than each inventing its own
# reading. Nothing below emits or lowers anything; it is the description a
# destination is measured against.
# ---------------------------------------------------------------------------

#: The section's `Fma` instructions are MANDATORY, not licensed.
#:
#: Deliberately not "Fma rounds once" -- that is what `Fma` means, it is true
#: in ordinary code as well, and a destination that cannot spell the op
#: shortfalls through the ordinary mechanism without any help from here.
#: What is section-specific is the opposite of the usual policy: an `Fma`
#: elsewhere APPEARS ONLY WHEN `contract_multiply_add` licenses it and must
#: not appear under `prove` at all, whereas here it is the residual itself,
#: so it has to be emitted whatever the contract says. A destination that
#: drops it because the contract forbids contraction has not been
#: conservative; it has computed zero.
FMA_MANDATORY = "fma.mandatory_regardless_of_contract"

#: A destination must not reassociate or contract within the section, except
#: at instructions the section itself declares as `Fma`. Both halves matter
#: and they pull opposite ways: the error terms are algebraically zero, so
#: any reassociating simplifier deletes them, while the declared Fma is a
#: contraction that must happen. "Optimize nothing here" and "optimize
#: everything here" are both wrong; the section says which instructions are
#: which.
SECTION_ISOLATION = "section.no_reassociation_except_declared_fma"

#: A destination may stage the section's lanes concurrently, or serially.
#: Unlike the other two this is permission, never obligation -- serial is
#: always a correct answer -- so a backend that ignores it is conforming,
#: not failing.
LANE_STAGING = "lanes.optional_concurrency"

#: Every obligation a precision section can place on a destination.
PRECISION_SECTION_OBLIGATIONS = (
    FMA_MANDATORY, SECTION_ISOLATION, LANE_STAGING,
)


#: The two exact spellings of ``a * b == p + e``, and who needs which.
#:
#: ``"fma"`` is the two-instruction form: ``e = fma(a, b, -p)``. It is the
#: cheap spelling, and the one that makes FMA_MANDATORY appear on a section's
#: contract, because the single rounding IS the residual.
#:
#: ``"split"`` is Dekker's spelling through Veltkamp splitting -- seventeen
#: ordinary operations, no fma anywhere. Same exact residual; the theorem
#: needs only IEEE mul/add/sub. Choosing it is how a destination with no
#: single-rounding fma (WebAssembly; WGSL and GLSL, whose ``fma()`` is
#: permitted to round twice) hosts a section HONESTLY: the obligation is
#: discharged by never being incurred, and the price is instruction count,
#: not accuracy. The eager twin lives at
#: ``common.tensors.extended_precision.two_product`` and the two must stay
#: instruction-for-instruction alike.
TWO_PRODUCT_FLAVORS = ("fma", "split")


#: The flavour the completed-module seam lowers with when its caller has no
#: way to say. The seam sits inside the source compiler, many layers below
#: anyone choosing a destination -- mirroring how ``active_contract()``
#: reaches the emitters -- so the choice travels the same way: a scoped
#: ambient value, defaulting to the cheap spelling.
_ACTIVE_TWO_PRODUCT_FLAVOR = "fma"


def active_two_product_flavor() -> str:
    """The flavour ``apply_precision_pipeline`` uses when not told."""

    return _ACTIVE_TWO_PRODUCT_FLAVOR


@_contextlib.contextmanager
def two_product_flavor_scope(flavor: str):
    """Lower every module built inside this scope with ``flavor``."""

    global _ACTIVE_TWO_PRODUCT_FLAVOR
    if flavor not in TWO_PRODUCT_FLAVORS:
        raise ValueError(
            f"unknown two_product flavour {flavor!r}; "
            f"expected one of {TWO_PRODUCT_FLAVORS}"
        )
    held = _ACTIVE_TWO_PRODUCT_FLAVOR
    _ACTIVE_TWO_PRODUCT_FLAVOR = str(flavor)
    try:
        yield
    finally:
        _ACTIVE_TWO_PRODUCT_FLAVOR = held


def _veltkamp_constant(element) -> float:
    """The splitting constant for a limb element.

    Sourced from ``extended_precision.LIMB_ELEMENTS`` so the compiler and
    the eager twin can never disagree about it. Splitting with the wrong
    constant does not degrade the residual, it invalidates the theorem --
    so an unknown element is a refusal, never a default. An ABSENT element
    is the binary64 default the whole scalar path assumes.
    """

    from ..common.tensors.extended_precision import limb_element_facts

    return float(limb_element_facts(element)["split"])


def _limb_width_ceiling(element) -> int:
    """The widest section the repository lowering accepts for an element."""

    from ..common.tensors.extended_precision import limb_element_facts

    return int(limb_element_facts(element)["max_limbs"])


#: What each destination actually delivers, as opposed to what it emits.
#:
#: The four lanes present a UNIFIED FRONT: every one of them accepts `Fma`
#: and produces something, so a program compiles on all four and no caller
#: has to write four versions. This table is what keeps that front from
#: becoming a lie. Emitting is not meeting: WebAssembly has no fma
#: instruction and expands into a multiply and an add, which rounds twice,
#: so it does not declare FMA_MANDATORY -- and a precision section asking
#: for one is refused HERE, before emission, rather than discovered later
#: in a residual that came back zero.
#:
#: Ordinary code is unaffected by that refusal. Code that merely wanted the
#: accuracy of an fma gets the arithmetic on every lane; only code whose
#: correctness depends on the single rounding is turned away, and only where
#: the single rounding is not available.
BACKEND_PRECISION_CAPABILITIES: dict[str, tuple[str, ...]] = {
    # C99 `fma`, and `#pragma STDC FP_CONTRACT OFF`. Both in the language.
    "c": (FMA_MANDATORY, SECTION_ISOLATION),
    # @llvm.fma.f64 is the operation, not a licence to fuse. Isolation is
    # withholding the fast-math flags, which are already per-instruction.
    "llvm": (FMA_MANDATORY, SECTION_ISOLATION),
    # IEEE_FMA is F2018 and the lane already uses IEEE_ARITHMETIC. Fortran
    # forbids reassociating a parenthesised expression -- and every emitted
    # binary is parenthesised -- while contraction, the one rewrite the
    # language leaves to the processor, is withdrawn per-artifact by the
    # toolchain: emit_module marks a module that carries sections and
    # aggressive_fortran_flags adds -ffp-contract=off for exactly those.
    # An explicit ieee_fma call is an intrinsic invocation, not a
    # contraction, so it survives the flag. Both halves together are the
    # whole of isolation, so the claim is now whole too.
    "fortran": (FMA_MANDATORY, SECTION_ISOLATION),
    # No fma instruction exists -- a section that CONTAINS one is refused,
    # because the expansion rounds twice and the residual comes back zero.
    # Isolation, though, wasm delivers by construction: its floating point
    # is deterministic IEEE 754 and an engine has no licence to reassociate
    # or contract anything. So a section lowered with the "split" flavour of
    # two_product -- no Fma to spell -- runs here honestly, just longer.
    "wasm": (SECTION_ISOLATION,),
    # SPIR-V spells the single rounding as GLSL.std.450 Fma and withdraws
    # contraction per-instruction with the NoContraction decoration; both
    # are in the assembler. Float64 limbs are fine: OpCapability Float64 is
    # emitted on demand, and the f32-only restriction covers only the
    # extended-instruction transcendentals a limb section never uses.
    "spirv": (FMA_MANDATORY, SECTION_ISOLATION),
    # WGSL's fma() is explicitly permitted to round twice, so FMA_MANDATORY
    # is not claimed and never will be -- a section containing an Fma is
    # still refused here, which is what the fma_value_ids gate in
    # precision_backend_shortfalls already decides on its own.
    #
    # SECTION_ISOLATION is claimed for the "split" flavour, which is the
    # route this comment always named. Split two_product spells no fma at
    # all: Veltkamp halves at 4097 (2**12 + 1), and for a 24-bit f32
    # significand the 12-bit half products are exact, so the residual is
    # recovered by subtraction rather than by a fused rounding. That
    # removes the failure this lane was guarding against -- an fma-shaped
    # residual coming back algebraically zero -- because there is no fused
    # operation left to double-round.
    #
    # What is NOT settled by the language is reassociation: two_sum's
    # `(a - (s - b')) + (b - b')` is only an error-free transformation if
    # it is evaluated as written. So this claim is carried by MEASUREMENT
    # rather than by a spec argument -- tools/probe_webgpu_precision.py
    # scores the emitted shader against exact rational truth in a real
    # browser, and a lane that reassociated would show it immediately as a
    # width that stops buying precision.
    "webgpu": (SECTION_ISOLATION,),
    # Desktop GLSL compute (GL 4.3 + GL_ARB_gpu_shader5): the `precise`
    # qualifier is exactly contraction-and-reassociation control -- values
    # so qualified must be computed as written, and an fma() call on
    # precise operands is required to be the single-rounding operation.
    # The lane's emitter qualifies every section temporary `precise` and
    # spells Fma as fma(), which is what makes both claims true. This key
    # is the DESKTOP lane; the browser fragment lane is "webgl" below.
    "glsl": (FMA_MANDATORY, SECTION_ISOLATION),
    # GLSL ES 3.0 fragment shaders: f32 only, no fma(), `precise` is
    # desktop 4.40+. Nothing claimable.
    "webgl": (),
}


def unmet_precision_obligations(contract, backend: str) -> tuple[str, ...]:
    """What ``backend`` cannot honour for one section. Empty means it may."""

    return contract.unmet_by(
        BACKEND_PRECISION_CAPABILITIES.get(str(backend).casefold(), ())
    )


@dataclasses.dataclass(frozen=True)
class PrecisionSectionContract:
    """What one precision section presents, and what it requires of a host.

    Derived from the instruction stream rather than declared by an author,
    so it cannot drift from the code it describes.
    """

    function: str
    #: Result ids of the section's instructions, in program order.
    value_ids: tuple[int, ...]
    #: The precision operation names present.
    operations: tuple[str, ...]
    #: Result ids of explicit `Fma` instructions, or abstract multiplication
    #: operations whose exact lowering necessarily introduces one.
    fma_value_ids: tuple[int, ...]
    #: Widest limb count and element type in the section.
    limbs: int
    element: str | None
    obligations: tuple[str, ...]

    def as_record(self) -> dict:
        return {
            "function": self.function,
            "value_ids": list(self.value_ids),
            "operations": list(self.operations),
            "fma_value_ids": list(self.fma_value_ids),
            "limbs": self.limbs,
            "element": self.element,
            "obligations": list(self.obligations),
        }

    def unmet_by(self, capabilities) -> tuple[str, ...]:
        """Obligations a destination declaring ``capabilities`` cannot meet.

        Empty means the section may be lowered there. Anything else is a
        shortfall to report BEFORE emitting, not a degradation to discover
        in the numbers afterwards -- the failures here do not announce
        themselves, they return plausible numbers whose residual is zero.
        ``LANE_STAGING`` never appears: it is permission, so no destination
        can fail it. ``FMA_MANDATORY`` appears only for a section that
        actually contains one.
        """

        declared = frozenset(str(each) for each in capabilities)
        required = set(self.obligations) - {LANE_STAGING}
        if not self.fma_value_ids:
            required.discard(FMA_MANDATORY)
        return tuple(sorted(required - declared))


def precision_section_contracts(
    functions, two_product_flavor: str = "fma",
) -> tuple[PrecisionSectionContract, ...]:
    """Read the contract each precision section places on a destination.

    A section is the maximal run of consecutive instructions in one block
    that carry limbs, plus any `Fma` sitting among them. Adjacency is the
    right criterion here rather than dataflow reachability: what a
    destination must refrain from reoptimizing is a contiguous stretch of
    emitted code, and an `Fma` between two limbed operations is inside the
    stretch whether or not it consumes one of them.

    ``two_product_flavor`` must match what ``lower_precision_operations``
    will actually emit, since the contract is recorded before the lowering
    runs and describes the instructions that WILL exist.
    """

    from ..common.tensors.topological_reducer import PRECISION_SINGULAR_NAMES

    if two_product_flavor not in TWO_PRODUCT_FLAVORS:
        raise ValueError(
            f"unknown two_product flavour {two_product_flavor!r}; "
            f"expected one of {TWO_PRODUCT_FLAVORS}"
        )
    precision_ops = frozenset(PRECISION_SINGULAR_NAMES.values())
    # Under the "fma" flavour, exact lowering of a multiplication or a
    # division necessarily introduces an Fma, so those operations carry the
    # obligation before the instruction exists. Under "split" they lower to
    # ordinary arithmetic and incur nothing; only an EXPLICIT Fma in the
    # stream still demands the real single rounding, because no expansion
    # can substitute for a request the author spelled out.
    precision_fma_ops = {
        PRECISION_SINGULAR_NAMES["Mul"],
        PRECISION_SINGULAR_NAMES["Div"],
    } if two_product_flavor == "fma" else set()
    contracts: list[PrecisionSectionContract] = []

    for name, function in functions.items():
        for block in function.blocks.values():
            runs: list[list] = []
            current: list = []
            for instruction in block.instrs:
                if str(instruction.op) in precision_ops or (
                    current and str(instruction.op) == FMA
                ):
                    current.append(instruction)
                    continue
                if current:
                    runs.append(current)
                    current = []
            if current:
                runs.append(current)

            for run in runs:
                if not any(str(each.op) in precision_ops for each in run):
                    continue
                limbs, element = 1, None
                for each in run:
                    limbs = max(
                        limbs, int(each.attributes.get("precision_limbs") or 1)
                    )
                    # Taken independently of the limb count: an operation can
                    # carry a width without yet knowing its element type, and
                    # reading the element off whichever instruction happened
                    # to be widest loses one that a neighbour does know.
                    #
                    # The RESULT VALUE is asked before the instruction. An
                    # operation named at ingestion keeps whatever element its
                    # attributes were stamped with -- usually none, since the
                    # declaration gives limbs and no type -- while the carry
                    # pass resolves the element onto the value from its dtype.
                    # Reading only the attributes reports None for a section
                    # whose element is perfectly well known.
                    if element is None:
                        result = each.res
                        element = (
                            (result.accounting or {}).get("precision_element")
                            or (result.dtype if result is not None else None)
                            if result is not None else None
                        ) or each.attributes.get("precision_element")
                contracts.append(PrecisionSectionContract(
                    function=str(name),
                    value_ids=tuple(
                        int(each.res.id) for each in run
                        if each.res is not None
                    ),
                    operations=tuple(sorted({str(each.op) for each in run})),
                    fma_value_ids=tuple(
                        int(each.res.id) for each in run
                        if (
                            str(each.op) == FMA
                            or str(each.op) in precision_fma_ops
                        ) and each.res is not None
                    ),
                    limbs=limbs,
                    element=None if element is None else str(element),
                    obligations=PRECISION_SECTION_OBLIGATIONS,
                ))
    return tuple(contracts)


#: Stamped on every instruction inside a precision section, and the one thing
#: about the section that must reach a destination.
PRECISION_SECTION_ATTRIBUTE = "precision_section"


def mark_precision_sections(functions) -> int:
    """Stamp each precision section's instructions so emission can see it.

    The OPERATOR does not survive: `precision_mul` and its siblings exist to
    be recognised, propagated along, and reduced against, and are then
    expanded into an error-free transformation before any destination is
    reached. That is deliberate -- it is why no backend implements them.

    But the expansion is plain Add/Sub/Mul that looks like everybody else's,
    and an optimiser has no way to tell a Dekker split from the same
    expression written by someone who would be glad to see it folded. So the
    section boundary has to survive in the operator's place, and this is what
    carries it.

    Stamping the instruction rather than recording a range is what makes it
    survive: an expansion that inherits its source's attributes stays marked
    without anything having to re-derive where the section went, and a pass
    that moves an instruction cannot silently move it out of its section.

    Returns the number of instructions marked.
    """

    wanted: dict[str, set[int]] = {}
    for contract in precision_section_contracts(functions):
        wanted.setdefault(contract.function, set()).update(contract.value_ids)

    marked = 0
    for name, function in functions.items():
        inside = wanted.get(str(name))
        if not inside:
            continue
        for block in function.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.res is not None
                    and int(instruction.res.id) in inside
                    and not instruction.attributes.get(
                        PRECISION_SECTION_ATTRIBUTE
                    )
                ):
                    instruction.attributes[PRECISION_SECTION_ATTRIBUTE] = True
                    marked += 1
    return marked


def lower_precision_operations(
    functions, two_product_flavor: str = "fma",
) -> dict:
    """Expand every `precision_*` into arithmetic a destination already has.

    A precision value becomes ONE SSA VALUE PER LIMB, not one value with a
    limb axis. That is the representation the working kernels use --
    `two_product(a, b, p, e, n)` writes its two limbs to two arrays -- and it
    is the one that survives, because each limb is then an ordinary scalar
    that every backend can already hold, load and store. The channel-shaped
    alternative needs a destination to understand an aggregate before it can
    do arithmetic on one.

    After this runs no `precision_*` remains, which is the point: the
    operators exist to be recognised and reduced against, and their identity
    is not meant to reach a backend. The section attribute is stamped on
    every instruction produced here so the boundary survives where the
    operator does not.

    Expansions are the same width-N algorithms as the eager tensor path:
    Knuth ``two_sum`` distillation for addition, every limb-pair through an
    error-free ``two_product`` for multiplication, and digit-at-a-time
    expansion division. Widths two through four are retained end to end.
    Returns a count per operation.

    ``two_product_flavor`` selects the exact-product spelling (see
    ``TWO_PRODUCT_FLAVORS``): ``"fma"`` emits the two-instruction residual,
    ``"split"`` emits Dekker's fma-free seventeen-operation form for
    destinations that cannot deliver a single rounding.
    """

    from ..common.tensors.topological_reducer import PRECISION_SINGULAR_NAMES

    if two_product_flavor not in TWO_PRODUCT_FLAVORS:
        raise ValueError(
            f"unknown two_product flavour {two_product_flavor!r}; "
            f"expected one of {TWO_PRODUCT_FLAVORS}"
        )

    add_name = PRECISION_SINGULAR_NAMES["Add"]
    sub_name = PRECISION_SINGULAR_NAMES["Sub"]
    mul_name = PRECISION_SINGULAR_NAMES["Mul"]
    div_name = PRECISION_SINGULAR_NAMES["Div"]
    neg_name = PRECISION_SINGULAR_NAMES["neg"]
    expandable = {add_name, sub_name, mul_name, div_name, neg_name}

    counts = {name: 0 for name in expandable}
    next_id = _module_watermark(functions)
    original_formals = {
        str(name): tuple(function.args) for name, function in functions.items()
    }

    # -- the boundary, brought into line with the interior ----------------
    #
    # A precision value is n scalars everywhere inside a function, but a
    # declared parameter still arrived as ONE. The two representations
    # disagreeing at the boundary is what let a four-limb coefficient be
    # undeliverable while the arithmetic that would consume it already
    # worked, so a parameter used at width n becomes n formals: the
    # declared one carries the high limb and the rest are appended.
    #
    # Appending rather than interleaving keeps every existing position
    # valid, so a caller that does not know about limbs is unaffected.
    #
    # Width is read from the operations rather than from a declaration.
    # The declaration is gone by now -- it lives in the AST -- but an
    # operation records the width it was recognised at, and a parameter
    # feeding a width-n operation is a width-n parameter. That also keeps
    # this honest about parameters that are declared and never used at
    # precision: they stay one scalar, because nothing needs more.
    # -- precision arrays: channel-strided limbs ---------------------------
    #
    # An ARRAY of precision values is one ordinary array whose element i,
    # limb k lives at flat index i * limbs + k. One SSA value, so it can be
    # passed and RETURNED like any other array -- which is what the scalar
    # representation cannot do, and why every precision result until now had
    # to collapse into a single double on the way out.
    #
    # The arithmetic is unchanged: limbs are loaded into scalars, operated on
    # per limb, and stored back. Only the accessors widen. That keeps the one
    # representation every backend can already hold and avoids asking any of
    # them to understand an aggregate.
    #
    # An array is recognised by USE, not by shape -- inside a region a passed
    # array has shape () and is told apart only by being the base of a
    # GetElementPtr. A base whose Load feeds a precision operation is a
    # precision array; so is one whose Store carries a limbed value.
    precision_arrays: dict[str, dict[int, int]] = {}
    for name, function in functions.items():
        bases: dict[int, int] = {}
        for block in function.blocks.values():
            pointers: dict[int, int] = {}
            for instruction in block.instrs:
                if (
                    str(instruction.op) == "GetElementPtr"
                    and instruction.args and instruction.res is not None
                ):
                    pointers[int(instruction.res.id)] = int(
                        instruction.args[0].id
                    )
            loaded: dict[int, int] = {}
            for instruction in block.instrs:
                if (
                    str(instruction.op) == "Load" and instruction.args
                    and instruction.res is not None
                    and int(instruction.args[0].id) in pointers
                ):
                    loaded[int(instruction.res.id)] = pointers[
                        int(instruction.args[0].id)
                    ]
            produced: dict[int, int] = {}
            for instruction in block.instrs:
                width = max(
                    int(instruction.attributes.get("precision_limbs") or 1), 1
                )
                if str(instruction.op) in expandable and width > 1:
                    for argument in instruction.args:
                        base = loaded.get(int(argument.id))
                        if base is not None:
                            bases[base] = max(bases.get(base, 1), width)
                    if instruction.res is not None:
                        produced[int(instruction.res.id)] = width
            # The destination side. An array that is only ever WRITTEN --
            # every output array is -- has no Load to be recognised by, and
            # would otherwise collapse its limbs on the way out, which is the
            # single thing that has capped every precision result so far.
            for instruction in block.instrs:
                if (
                    str(instruction.op) == "Store"
                    and len(instruction.args) >= 2
                    and int(instruction.args[0].id) in produced
                    and int(instruction.args[1].id) in pointers
                ):
                    base = pointers[int(instruction.args[1].id)]
                    bases[base] = max(
                        bases.get(base, 1),
                        produced[int(instruction.args[0].id)],
                    )
        if bases:
            precision_arrays[str(name)] = bases

    parameter_widths: dict[str, dict[int, int]] = {}
    for name, function in functions.items():
        formal_ids = {int(value.id) for value in function.args}
        widths: dict[int, int] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) not in expandable:
                    continue
                width = max(
                    int(instruction.attributes.get("precision_limbs") or 1), 1
                )
                for argument in instruction.args:
                    if int(argument.id) in formal_ids and width > 1:
                        widths[int(argument.id)] = max(
                            widths.get(int(argument.id), 1), width
                        )
        if widths:
            parameter_widths[str(name)] = widths

    # A callee whose formals grew needs its callers to supply the new ones,
    # and a caller can only do that if ITS matching formal grew too. Pushing
    # the requirement back through call sites to a fixed point is what makes
    # the change survive the wrapper/region pair the planner emits.
    changed = True
    while changed:
        changed = False
        for name, function in functions.items():
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.op not in ("Call", "call"):
                        continue
                    callee = str(instruction.attributes.get("callee") or "")
                    wanted = parameter_widths.get(callee)
                    if not wanted:
                        continue
                    target = functions.get(callee)
                    if target is None:
                        continue
                    formals = original_formals.get(callee, tuple(target.args))
                    caller_formals = {int(v.id) for v in function.args}
                    here = parameter_widths.setdefault(str(name), {})
                    for position, formal in enumerate(formals):
                        width = wanted.get(int(formal.id), 1)
                        if width <= 1 or position >= len(instruction.args):
                            continue
                        actual = int(instruction.args[position].id)
                        if actual in caller_formals and here.get(actual, 1) < width:
                            here[actual] = width
                            changed = True

    seeded: dict[str, list[list]] = {}
    for name, function in functions.items():
        widths = parameter_widths.get(str(name)) or {}
        if not widths:
            continue
        rows: list[list] = []
        for formal in list(function.args):
            width = widths.get(int(formal.id), 1)
            if width <= 1:
                continue
            row = [formal]
            for _index in range(1, width):
                extra = SSAValue(
                    next_id, dtype=formal.dtype, shape=(),
                    device=formal.device,
                )
                next_id += 1
                function.args.append(extra)
                row.append(extra)
            rows.append(row)
        seeded[str(name)] = rows

    for function_name, function in functions.items():
        # value id -> its limbs, high first. Absent means an ordinary value,
        # which is simply a one-limb expansion whose low limb is zero.
        limbs: dict[int, list] = {}
        lowered_limb_records: dict[int, tuple[int, ...]] = {}
        for row in seeded.get(str(function_name), ()):  # declared parameters
            limbs[int(row[0].id)] = list(row)
            lowered_limb_records[int(row[0].id)] = tuple(
                int(value.id) for value in row
            )

        for block in function.blocks.values():
            emitted: list = []

            # The element the section being expanded declared, when it is
            # narrower than the value it is expanding. An expansion mints
            # many values and every one of them is a LIMB, so the element
            # is what they are -- inheriting the pre-expansion dtype makes
            # a float32 expansion claim to be float64, which is wrong in
            # two ways at once. It refuses to emit on a lane that has no
            # f64 at all, and, worse, ``splitter_for`` reads the same
            # dtype to choose Veltkamp's constant: a float32 limb split at
            # 2**27 + 1 instead of 4097 does not halve a 24-bit
            # significand, so the half products stop being exact and the
            # residual is quietly wrong rather than loudly absent.
            section_element: list = [None]

            def fresh(like):
                nonlocal next_id
                declared = section_element[0]
                dtype = (
                    declared if declared in _NARROW_LIMB_ELEMENTS
                    else like.dtype
                )
                value = SSAValue(
                    next_id, dtype=dtype, shape=(), device=like.device,
                )
                next_id += 1
                return value

            def put(op, args, like):
                result = fresh(like)
                emitted.append(Instr(op, list(args), result, attributes={
                    PRECISION_SECTION_ATTRIBUTE: True,
                    "lowered_from": "precision",
                }))
                return result

            def parts(value):
                return limbs.get(int(value.id), [value])

            def zero(like):
                """Materialise an exact low limb for scalar promotion.

                Repeating the high limb would represent ``x + x`` rather
                than ``x``.  A plain scalar is the expansion ``[x, +0, ...]``;
                IEEE positive zero is exact in every supported element
                format and therefore needs no numerical precondition.
                """

                result = fresh(like)
                emitted.append(Instr("Const", [], result, attributes={
                    "constant": 0.0,
                    PRECISION_SECTION_ATTRIBUTE: True,
                    "lowered_from": "precision.scalar_promotion",
                }))
                return result

            def two_sum(a, b):
                s = put("Add", (a, b), a)
                bb = put("Sub", (s, a), a)
                left = put("Sub", (a, put("Sub", (s, bb), a)), a)
                right = put("Sub", (b, bb), a)
                return s, put("Add", (left, right), a)

            splitters: dict[float, object] = {}

            def splitter_for(value):
                """One Const per block per element, not one per product."""

                magnitude = _veltkamp_constant(value.dtype)
                held = splitters.get(magnitude)
                if held is None:
                    held = fresh(value)
                    emitted.append(Instr("Const", [], held, attributes={
                        "constant": float(magnitude),
                        PRECISION_SECTION_ATTRIBUTE: True,
                        "lowered_from": "precision.split_constant",
                    }))
                    splitters[magnitude] = held
                return held

            def two_product(a, b):
                product = put("Mul", (a, b), a)
                if two_product_flavor == "fma":
                    negated = put("Neg", (product,), a)
                    return product, put(FMA, (a, b, negated), a)
                # Dekker through Veltkamp splitting: the same exact residual
                # with no fma anywhere, for destinations that cannot deliver
                # a single rounding. The eager twin is
                # ``extended_precision.two_product`` / ``_split``; the
                # operation order here mirrors it exactly so the two stay
                # provably the same algorithm.
                constant = splitter_for(a)

                def split(value):
                    scaled = put("Mul", (value, constant), value)
                    high = put(
                        "Sub",
                        (scaled, put("Sub", (scaled, value), value)),
                        value,
                    )
                    return high, put("Sub", (value, high), value)

                a_high, a_low = split(a)
                b_high, b_low = split(b)
                error = put(
                    "Sub", (put("Mul", (a_high, b_high), a), product), a
                )
                error = put(
                    "Add", (error, put("Mul", (a_high, b_low), a)), a
                )
                error = put(
                    "Add", (error, put("Mul", (a_low, b_high), a)), a
                )
                return product, put(
                    "Add", (error, put("Mul", (a_low, b_low), a)), a
                )

            def expanded(value, width: int):
                held = list(parts(value))
                if len(held) < width:
                    held.extend(zero(value) for _ in range(width - len(held)))
                return held[:width]

            def renormalise(terms, width: int):
                """Distil arbitrary exact terms to ``width`` SSA limbs."""

                rest = list(terms)
                if not rest:
                    raise ValueError("cannot renormalise an empty expansion")
                kept = []
                for _index in range(int(width)):
                    if not rest:
                        break
                    carry = rest[-1]
                    tail = [None] * (len(rest) - 1)
                    for position in range(len(rest) - 2, -1, -1):
                        carry, tail[position] = two_sum(
                            rest[position], carry
                        )
                    kept.append(carry)
                    rest = tail
                while len(kept) < int(width):
                    kept.append(zero(kept[-1]))
                for leftover in rest:
                    kept[-1] = put("Add", (kept[-1], leftover), kept[-1])
                return kept

            def add_expansions(left, right, width: int):
                return renormalise([*left, *right], width)

            def multiply_expansions(left, right, width: int):
                products = []
                for left_limb in left:
                    for right_limb in right:
                        high, low = two_product(left_limb, right_limb)
                        products.extend((high, low))
                return renormalise(products, width)

            def lead(expansion):
                return (
                    expansion[0] if len(expansion) == 1
                    else put("Add", (expansion[0], expansion[1]), expansion[0])
                )

            def divide_expansions(left, right, width: int):
                remainder = list(left)
                quotient = []
                for _index in range(int(width) + 1):
                    digit = put("Div", (lead(remainder), lead(right)), left[0])
                    quotient.append(digit)
                    product = multiply_expansions(
                        [digit], right, int(width) + 2
                    )
                    remainder = add_expansions(
                        remainder,
                        [put("Neg", (term,), term) for term in product],
                        int(width) + 2,
                    )
                return renormalise(quotient, width)

            def collapse(parts_of, original):
                """Define the ORIGINAL value as the sum of its limbs.

                The expanded operation drops out of the stream, so nothing
                would define the value it used to produce -- and a declared
                output, a buffer order or a Ret still names that id. An
                undefined output buffer reads as zero, which is a plausible
                number and therefore the worst possible failure.

                Reusing the id rather than rewriting every consumer means
                metadata that names values, and not just instruction
                operands, keeps working without this pass having to know
                what all of it is.
                """

                total = parts_of[0]
                for each in parts_of[1:-1]:
                    total = put("Add", (total, each), total)
                emitted.append(Instr(
                    "Add", [total, parts_of[-1]], original,
                    attributes={
                        PRECISION_SECTION_ATTRIBUTE: True,
                        "lowered_from": "precision.collapse",
                    },
                ))
                # The limb channel the carry pass put on this value is now
                # STALE and actively harmful. It described a value whose
                # limbs lived on a trailing axis; lowering has just made
                # them separate scalars, and the value left here holds one
                # number. A backend that believes the shape builds an ABI
                # around an array -- Fortran demanded an integer extent
                # parameter per lowered value and the call wrote through a
                # null pointer. LLVM ignored the shape and worked, which is
                # exactly how a disagreement like this stays hidden.
                original.shape = tuple(original.shape or ())[:-1]
                if original.accounting:
                    original.accounting.pop("precision_limbs", None)
                    original.accounting.pop("precision_element", None)

            arrays = precision_arrays.get(str(function_name)) or {}
            # pointer value id -> (base array id, index value)
            addressed: dict[int, tuple] = {}

            def constant(number: int):
                nonlocal next_id
                value = SSAValue(next_id, dtype="int", shape=())
                next_id += 1
                emitted.append(Instr("Const", [], value, attributes={
                    "constant": int(number),
                    PRECISION_SECTION_ATTRIBUTE: True,
                }))
                return value

            def channel(base, index, width: int, position: int):
                """Address element ``index`` limb ``position``, stride ``width``.

                Flat index ``index * width + position``. Said with Mul, Add
                and Const, which every destination already supplies, rather
                than with a new operator seven backends would each have to
                implement.
                """

                scaled = put("Mul", (index, constant(width)), index)
                flat = (
                    scaled if position == 0
                    else put("Add", (scaled, constant(position)), index)
                )
                return put("GetElementPtr", (base, flat), index)

            for instruction in block.instrs:
                operation = str(instruction.op)
                section_element[0] = instruction.attributes.get(
                    "precision_element"
                )

                if operation in ("Call", "call"):
                    callee = str(instruction.attributes.get("callee") or "")
                    rows = seeded.get(callee)
                    if rows and not instruction.attributes.get(
                        "precision_actuals_bound"
                    ):
                        target_formals = original_formals.get(callee, ())
                        by_head = {int(row[0].id): row for row in rows}
                        original_actuals = list(instruction.args)
                        additions = []
                        for position, formal in enumerate(target_formals):
                            row = by_head.get(int(formal.id))
                            if row is None or position >= len(original_actuals):
                                continue
                            actual = original_actuals[position]
                            supplied = parts(actual)
                            for index in range(1, len(row)):
                                additions.append(
                                    supplied[index]
                                    if index < len(supplied)
                                    else zero(actual)
                                )
                        instruction.args.extend(additions)
                        instruction.attributes["precision_actuals_bound"] = True
                        instruction.attributes[
                            "precision_actual_count"
                        ] = len(additions)
                    emitted.append(instruction)
                    continue

                if (
                    operation == "GetElementPtr" and len(instruction.args) >= 2
                    and instruction.res is not None
                ):
                    addressed[int(instruction.res.id)] = (
                        instruction.args[0], instruction.args[1],
                    )
                    emitted.append(instruction)
                    continue

                if (
                    operation == "Load" and instruction.args
                    and instruction.res is not None
                ):
                    where = addressed.get(int(instruction.args[0].id))
                    width = arrays.get(int(where[0].id)) if where else None
                    if width and width > 1:
                        base, index = where
                        loaded_limbs = [
                            put("Load", (channel(base, index, width, position),),
                                instruction.res)
                            for position in range(width)
                        ]
                        limbs[int(instruction.res.id)] = loaded_limbs
                        lowered_limb_records[int(instruction.res.id)] = tuple(
                            int(value.id) for value in loaded_limbs
                        )
                        # The original value must still be DEFINED. Not every
                        # consumer of a load from a precision array is a
                        # precision operation -- a shell computing `s = z * z`
                        # from an undeclared local reads it as ordinary
                        # arithmetic -- and such a consumer still names the id
                        # this expansion just removed the definition of.
                        # Leaving it dangling produced NaN, because the
                        # instruction read whatever the slot happened to hold.
                        collapse(loaded_limbs, instruction.res)
                        continue
                    emitted.append(instruction)
                    continue

                if operation == "Store" and len(instruction.args) >= 2:
                    where = addressed.get(int(instruction.args[1].id))
                    width = arrays.get(int(where[0].id)) if where else None
                    stored = limbs.get(int(instruction.args[0].id))
                    if width and width > 1 and stored:
                        base, index = where
                        for position in range(width):
                            emitted.append(Instr(
                                "Store",
                                [
                                    stored[position] if position < len(stored)
                                    else zero(stored[0]),
                                    channel(base, index, width, position),
                                ],
                                None,
                                attributes={
                                    PRECISION_SECTION_ATTRIBUTE: True,
                                    "lowered_from": "precision.store",
                                },
                            ))
                        continue
                    emitted.append(instruction)
                    continue

                if operation not in expandable or instruction.res is None:
                    emitted.append(instruction)
                    continue

                if operation == neg_name:
                    # Exact per limb: a sign flip cannot round.
                    width = max(int(
                        instruction.attributes.get("precision_limbs") or 2
                    ), 2)
                    source = expanded(instruction.args[0], width)
                    negated = [put("Neg", (each,), each) for each in source]
                    limbs[int(instruction.res.id)] = negated
                    lowered_limb_records[int(instruction.res.id)] = tuple(
                        int(value.id) for value in negated
                    )
                    collapse(negated, instruction.res)
                    counts[operation] += 1
                    continue

                width = max(int(
                    instruction.attributes.get("precision_limbs") or 2
                ), 2)
                left = expanded(instruction.args[0], width)
                right = expanded(instruction.args[1], width)
                if operation == sub_name:
                    right = [put("Neg", (each,), each) for each in right]

                if operation == mul_name:
                    result_limbs = multiply_expansions(left, right, width)
                elif operation == div_name:
                    result_limbs = divide_expansions(left, right, width)
                else:
                    result_limbs = add_expansions(left, right, width)

                limbs[int(instruction.res.id)] = result_limbs
                lowered_limb_records[int(instruction.res.id)] = tuple(
                    int(value.id) for value in result_limbs
                )
                collapse(result_limbs, instruction.res)
                counts[operation] += 1

            block.instrs = emitted

        if lowered_limb_records:
            function.metadata["precision_lowered_values"] = tuple(
                (value_id, limb_ids)
                for value_id, limb_ids in sorted(lowered_limb_records.items())
            )

    return counts


PRECISION_PIPELINE_METADATA = "precision_pipeline"


def _refresh_precision_call_records(module) -> int:
    """Extend canonical call records after precision grows a scalar ABI.

    The source linker owns every pre-existing binding.  Precision lowering
    only appends formals and actuals, so this refresh preserves those records
    byte-for-byte and adds exact caller-value bindings for the appended limb
    positions.  No name matching or positional reconstruction of old state is
    permitted here.
    """

    refreshed = 0
    call_table = dict(getattr(module, "call_table", {}) or {})
    for caller_name, records in tuple(call_table.items()):
        caller = module.functions.get(str(caller_name))
        if caller is None:
            continue
        instructions = tuple(
            instruction
            for block in caller.blocks.values()
            for instruction in block.instrs
            if instruction.op in ("Call", "call")
        )
        updated = []
        for record in records:
            call = next((
                instruction for instruction in instructions
                if str(instruction.attributes.get("callee") or "")
                == str(record.callee_symbol or "")
                and (
                    instruction.attributes.get("plan_callsite_id") is None
                    or int(instruction.attributes["plan_callsite_id"])
                    == int(record.callsite_id)
                )
            ), None)
            callee = module.functions.get(str(record.callee_symbol or ""))
            if call is None or callee is None:
                updated.append(record)
                continue

            existing_arguments = {
                (int(actual), int(formal))
                for actual, formal in record.argument_bindings
            }
            existing_frames = {
                int(formal): (str(kind), source)
                for formal, kind, source in record.frame_bindings
            }
            additions = 0
            for actual, formal in zip(call.args, callee.args):
                binding = (int(actual.id), int(formal.id))
                if binding not in existing_arguments:
                    existing_arguments.add(binding)
                    additions += 1
                existing_frames.setdefault(
                    int(formal.id), ("caller_value", int(actual.id))
                )
            if not additions:
                updated.append(record)
                continue
            storage = tuple(dict.fromkeys((
                *map(int, record.callee_storage_value_ids),
                *(int(value.id) for value in callee.args),
            )))
            bound = set(existing_frames)
            updated.append(dataclasses.replace(
                record,
                argument_bindings=tuple(sorted(existing_arguments)),
                callee_storage_value_ids=storage,
                frame_bindings=tuple(
                    (formal, kind, source)
                    for formal, (kind, source) in sorted(existing_frames.items())
                ),
                unresolved_frame_value_ids=tuple(
                    value for value in record.unresolved_frame_value_ids
                    if int(value) not in bound
                ),
            ))
            refreshed += 1
        call_table[str(caller_name)] = tuple(updated)
    module.call_table = call_table
    return refreshed


def apply_precision_pipeline(
    module, two_product_flavor: str | None = None,
) -> dict:
    """Run the complete exact precision lowering transaction on ``module``.

    This is the sole production entry point.  It records the pre-lowering
    section contracts before their ``precision_*`` operators disappear,
    performs only catalogue identities marked exact, expands the arithmetic,
    repairs the canonical call ABI, and refuses to return if an abstract
    precision operator survived.  A durable receipt makes the transaction
    idempotent and gives emitters the obligations they must validate.

    ``two_product_flavor`` (see ``TWO_PRODUCT_FLAVORS``) chooses the exact
    product spelling and thereby which obligations the sections record:
    ``"fma"`` for destinations with a single-rounding fma, ``"split"`` for
    those without one.  ``None`` defers to ``active_two_product_flavor()``,
    which is how a caller above the completed-module seam chooses without
    threading an argument through the source compiler.  A module is lowered
    in ONE flavour; asking again with a different one is refused rather
    than answered with the wrong receipt, because the instructions already
    emitted cannot change.
    """

    from ..common.tensors.topological_reducer import PRECISION_SINGULAR_NAMES

    if two_product_flavor is None:
        two_product_flavor = active_two_product_flavor()
    if two_product_flavor not in TWO_PRODUCT_FLAVORS:
        raise ValueError(
            f"unknown two_product flavour {two_product_flavor!r}; "
            f"expected one of {TWO_PRODUCT_FLAVORS}"
        )
    functions = module.functions
    prior = (getattr(module, "metadata", {}) or {}).get(
        PRECISION_PIPELINE_METADATA
    )
    if prior and prior.get("status") == "lowered":
        held = str(prior.get("two_product_flavor") or "fma")
        if held != two_product_flavor:
            raise ValueError(
                f"module already lowered with two_product flavour {held!r}; "
                f"cannot re-lower as {two_product_flavor!r} -- lower a fresh "
                "module for the other flavour"
            )
        return prior

    precision_ops = frozenset(PRECISION_SINGULAR_NAMES.values())
    before = sum(
        str(instruction.op) in precision_ops
        for function in functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    if not before:
        return {
            "schema": "precision-pipeline-v1",
            "status": "not-present",
            "exact_only": True,
            "source_operations": 0,
            "section_contracts": [],
        }

    unsupported_seeds = tuple(
        (str(name), int(instruction.attributes.get("precision_limbs") or 1))
        for name, function in functions.items()
        for block in function.blocks.values()
        for instruction in block.instrs
        if str(instruction.op) in precision_ops
        and int(instruction.attributes.get("precision_limbs") or 1)
        > _limb_width_ceiling(
            instruction.attributes.get("precision_element")
        )
    )
    if unsupported_seeds:
        raise ValueError(
            "repository SSA precision lowering refuses sections wider than "
            "the element's proven ceiling (binary64: four limbs, binary32: "
            "eight) instead of duplicating or discarding limbs: "
            + repr(unsupported_seeds)
        )

    carried = carry_precision_through_ssa(functions)
    identity_counts = reduce_precision_operations(functions)
    contracts = precision_section_contracts(functions, two_product_flavor)
    unsupported_widths = tuple(
        (contract.function, contract.limbs)
        for contract in contracts
        if int(contract.limbs) > _limb_width_ceiling(contract.element)
    )
    if unsupported_widths:
        raise ValueError(
            "repository SSA precision lowering refuses sections wider than "
            "the element's proven ceiling (binary64: four limbs, binary32: "
            "eight) instead of duplicating or discarding limbs: "
            + repr(unsupported_widths)
        )
    marked = mark_precision_sections(functions)
    lowered = lower_precision_operations(functions, two_product_flavor)
    calls_refreshed = _refresh_precision_call_records(module)
    remaining = tuple(
        (str(name), int(instruction.res.id) if instruction.res else None,
         str(instruction.op))
        for name, function in functions.items()
        for block in function.blocks.values()
        for instruction in block.instrs
        if str(instruction.op) in precision_ops
    )
    if remaining:
        raise ValueError(
            "precision lowering left abstract operators: " + repr(remaining)
        )

    receipt = {
        "schema": "precision-pipeline-v1",
        "status": "lowered",
        "exact_only": True,
        "two_product_flavor": two_product_flavor,
        "source_operations": int(before),
        "carried_facts": int(carried),
        "identity_counts": dict(identity_counts),
        "identity_catalogue": [
            {
                "name": identity.name,
                "exact": bool(identity.exact),
                "requires": list(identity.requires),
                "fired": int(identity_counts.get(identity.name, 0)),
            }
            for identity in PRECISION_IDENTITIES
        ],
        "refused_identities": [identity.name for identity in PRECISION_REFUSALS],
        "section_contracts": [contract.as_record() for contract in contracts],
        "marked_instructions": int(marked),
        "lowered_operations": dict(lowered),
        "call_records_refreshed": int(calls_refreshed),
    }
    module.metadata[PRECISION_PIPELINE_METADATA] = receipt
    return receipt


def precision_backend_shortfalls(
    module, backend: str, function_names=None,
) -> tuple[dict, ...]:
    """Return persisted precision obligations ``backend`` cannot deliver."""

    receipt = (getattr(module, "metadata", {}) or {}).get(
        PRECISION_PIPELINE_METADATA
    ) or {}
    selected = None if function_names is None else {
        str(name) for name in function_names
    }
    capabilities = frozenset(BACKEND_PRECISION_CAPABILITIES.get(
        str(backend).casefold(), ()
    ))
    shortfalls = []
    for contract in receipt.get("section_contracts", ()):
        if selected is not None and str(contract.get("function")) not in selected:
            continue
        required = set(map(str, contract.get("obligations", ()))) - {
            LANE_STAGING
        }
        if not contract.get("fma_value_ids"):
            required.discard(FMA_MANDATORY)
        missing = tuple(sorted(required - capabilities))
        if missing:
            shortfalls.append({
                "function": str(contract.get("function")),
                "value_ids": tuple(map(int, contract.get("value_ids", ()))),
                "missing": missing,
            })
    return tuple(shortfalls)


def inline_host_linear_source_regions(functions) -> tuple[dict, tuple[dict, ...]]:
    """Return a host-lane view with safe one-call source regions spliced in.

    The repository keeps planned regions separate because shader deployment
    needs a dispatchable subgraph.  A native host lane has the opposite cost:
    an outputless loop-body region becomes one subroutine call per element.
    This identity does not alter the repository module.  It creates replacement
    caller/block/instruction containers only where every legality condition is
    proved, and returns an ordinary mapping for the host emitter.

    Eligible means: a compile-complementary source region, exactly one callsite,
    one straight-line block, no published outputs, no nested call/control, an
    exact formal/actual arity match, and no produced-ID collision in the caller.
    Anything less stays a call.  The receipt states every splice explicitly.
    """

    view = dict(functions)
    calls_by_callee: dict[str, list[tuple[str, str, int, object]]] = {}
    for caller_name, caller in functions.items():
        for block_name, block in caller.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if instruction.op not in ("Call", "call"):
                    continue
                callee = str(instruction.attributes.get("callee") or "")
                if callee:
                    calls_by_callee.setdefault(callee, []).append((
                        str(caller_name), str(block_name), index, instruction,
                    ))

    receipts = []
    forbidden = {
        "Call", "call", "Ret", "ret", "Return", "return", "Br", "br",
        "Branch", "branch", "CondBr", "condbr", "Phi", "phi",
    }
    replacements: dict[tuple[str, str, int], list] = {}
    removable = set()
    for callee_name, occurrences in calls_by_callee.items():
        callee = functions.get(callee_name)
        if callee is None or len(occurrences) != 1:
            continue
        integral = dict(callee.metadata.get("source_region_integral") or {})
        # PUBLISHED OUTPUTS ARE FINE. The splice clones each instruction
        # with its own result value, ids and all, so a value the region
        # published is produced in the caller at the same id the caller
        # already refers to -- there is nothing left to publish through.
        # The one hazard that would make this unsafe, an id in the region
        # colliding with one the caller already produces, is checked
        # separately below and still rejects the splice.
        #
        # Refusing them was costing a real destination: a two-limb result
        # publishes two values, and WGSL has no multi-value helper, so the
        # GPU lane could not take a wide kernel at all while the region
        # stayed a call.
        if not integral:
            continue
        if set(callee.blocks) != {"entry"}:
            continue
        body = callee.blocks["entry"]
        if any(str(instruction.op) in forbidden for instruction in body.instrs):
            continue
        caller_name, block_name, index, call = occurrences[0]
        if len(call.args) != len(callee.args):
            continue
        # An AGGREGATE result is not a reason to refuse. The call hands
        # back a span the caller immediately takes apart, and every read
        # of it carries `source_output_id` naming the value that slot
        # stands for -- so once the producing instructions live in the
        # caller, the span and the reads of it are both redundant and are
        # dropped below. Refusing this kept the region alive as a
        # separate function, and a value produced inside it could only
        # leave through the published set: the limbs of a wide result are
        # not in that set, so a two-limb kernel published its collapsed
        # head and the field it fed was single precision between steps.
        aggregate = (
            int(call.res.id) if call.res is not None
            and str(call.attributes.get("result_convention") or "")
            == "ssa.aggregate" else None
        )
        if call.res is not None and aggregate is None:
            continue
        produced = {
            int(instruction.res.id)
            for instruction in body.instrs if instruction.res is not None
        }
        formal_ids = {int(value.id) for value in callee.args}
        available = set(formal_ids)
        valid = True
        for instruction in body.instrs:
            if any(int(argument.id) not in available for argument in instruction.args):
                valid = False
                break
            if instruction.res is not None:
                available.add(int(instruction.res.id))
        if not valid:
            continue
        caller = functions[caller_name]
        # Instructions that only take the aggregate apart. They are keyed
        # by position so they can be deleted, and they are excluded from
        # the occupancy test below: a Load whose result IS the id the
        # spliced body produces is not a collision, it is the same value
        # arriving by a route that is about to disappear.
        aggregate_reads: dict[tuple, tuple] = {}
        if aggregate is not None:
            pointers = {aggregate}
            for caller_block_name, caller_block in caller.blocks.items():
                for position, instruction in enumerate(caller_block.instrs):
                    operation = str(instruction.op)
                    reads_aggregate = any(
                        int(argument.id) in pointers
                        for argument in instruction.args
                    )
                    if not reads_aggregate:
                        continue
                    if operation not in (
                        "GetElementPtr", "getelementptr", "Load", "load",
                    ):
                        aggregate_reads.clear()
                        break
                    # Only a GetElementPtr yields another POINTER into the
                    # span. A Load yields the value itself, and whatever
                    # consumes that -- the Ret, most importantly -- is an
                    # ordinary use and not an illegal reader of the
                    # aggregate. Treating a Load result as a pointer made
                    # the function's own return look like a violation and
                    # aborted every splice.
                    if instruction.res is not None and operation in (
                        "GetElementPtr", "getelementptr",
                    ):
                        pointers.add(int(instruction.res.id))
                    aggregate_reads[
                        (caller_name, str(caller_block_name), position)
                    ] = ()
                else:
                    continue
                break
            # Every read must be accounted for, and each published slot
            # must actually be produced by the body, or the caller would
            # be left referring to something no longer defined.
            named = {
                int(instruction.attributes.get("source_output_id"))
                for caller_block in caller.blocks.values()
                for instruction in caller_block.instrs
                if instruction.attributes.get("source_output_id") is not None
                and str(instruction.op) in ("Load", "load")
            }
            if not aggregate_reads or not named <= produced:
                continue
        read_results = {
            int(caller.blocks[key[1]].instrs[key[2]].res.id)
            for key in aggregate_reads
            if caller.blocks[key[1]].instrs[key[2]].res is not None
        }
        occupied = {
            int(value.id) for value in caller.args
        } | {
            int(instruction.res.id)
            for caller_block in caller.blocks.values()
            for instruction in caller_block.instrs
            if instruction.res is not None and instruction is not call
            and int(instruction.res.id) not in read_results
        }
        if produced & occupied:
            continue
        substitution = {
            int(formal.id): actual
            for formal, actual in zip(callee.args, call.args)
        }
        cloned = []
        for instruction in body.instrs:
            cloned.append(Instr(
                instruction.op,
                [substitution.get(int(argument.id), argument)
                 for argument in instruction.args],
                instruction.res,
                arg_roles=list(instruction.arg_roles),
                attributes={
                    **dict(instruction.attributes),
                    "inlined_source_region": str(callee_name),
                },
                source_span=(
                    None if instruction.source_span is None
                    else dict(instruction.source_span)
                ),
            ))
        replacements[(caller_name, block_name, index)] = cloned
        replacements.update(aggregate_reads)
        removable.add(callee_name)
        receipts.append({
            "caller": caller_name,
            "block": block_name,
            "callee": callee_name,
            "instruction_count": len(cloned),
            "identity_token_chain": list(
                integral.get("identity_token_chain") or ()
            ),
        })

    for caller_name in {key[0] for key in replacements}:
        caller = functions[caller_name]
        blocks = {}
        for block_name, block in caller.blocks.items():
            instructions = []
            for index, instruction in enumerate(block.instrs):
                instructions.extend(replacements.get(
                    (caller_name, str(block_name), index), (instruction,)
                ))
            blocks[block_name] = dataclasses.replace(
                block, instrs=instructions
            )
        metadata = dict(caller.metadata)
        own_receipts = tuple(
            receipt for receipt in receipts
            if receipt["caller"] == caller_name
        )
        metadata["host_linear_region_inlining"] = own_receipts
        # The limb rows move WITH the instructions. A value's row says
        # which SSA values are its limbs, and splicing relocates the
        # instructions that produce them into this caller -- so leaving
        # the rows behind in a function that no longer exists loses the
        # only record that a result is wide. Measured, the returned value
        # of a two-limb kernel had a row of (22418, 22437) inside the
        # region and none at all after inlining, so the low limb was
        # simply never published and the field it fed was single
        # precision between steps however many limbs the arithmetic used.
        rows = dict(metadata.get("precision_lowered_values") or ())
        for receipt in own_receipts:
            spliced = functions.get(str(receipt["callee"]))
            if spliced is None:
                continue
            # The caller's own rows WIN. A region shares its caller's
            # formals by identity, so it records rows for them too --
            # naming its own low-limb values, which are not the caller's.
            # Overwriting cost the caller the names of eight formals and
            # left them unbound at the boundary.
            for value_id, limb_ids in (
                spliced.metadata.get("precision_lowered_values") or ()
            ):
                rows.setdefault(value_id, limb_ids)
        if rows:
            metadata["precision_lowered_values"] = tuple(sorted(rows.items()))
        view[caller_name] = dataclasses.replace(
            caller, blocks=blocks, metadata=metadata
        )
    for callee_name in removable:
        view.pop(callee_name, None)
    return view, tuple(receipts)

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


# --------------------------------------------------------------------------
# Redundancy: one definition per distinct value, one computation per
# distinct expression. Catalogue sections 3 and 2.3.


#: Operations whose result depends only on their operands -- no store, no
#: call, no table traffic -- and which may therefore be shared when two of
#: them compute the same thing.
_PURE_FOR_SHARING = frozenset({
    "Neg", "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Abs",
    "Max", "Min", "Eq", "Ne", "Lt", "Le", "Gt", "Ge", "FloorDiv", "Mod",
    "Cast",
})


def _protected_value_ids(function) -> set[int]:
    """Ids other machinery names BY NUMBER, whose definition must survive.

    Sharing rewrites uses, so a value named only as an operand is free to be
    redirected. A value named in metadata is not: nothing rewrites the
    metadata, so removing its definition strands the reference.
    """

    protected: set[int] = set()
    for key in ("source_output_value_ids", "required_source_value_ids"):
        protected.update(
            int(value_id) for value_id in (function.metadata.get(key) or ())
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
    return protected


def _resolve(replacement: dict, value):
    """Follow a replacement chain to the surviving definition."""

    identifier = getattr(value, "id", None)
    seen = 0
    while identifier is not None and identifier in replacement:
        value = replacement[identifier]
        identifier = getattr(value, "id", None)
        seen += 1
        if seen > len(replacement):
            break
    return value


def _rewrite_uses(function, replacement: dict) -> None:
    if not replacement:
        return
    for block in function.blocks.values():
        rewritten = []
        for instruction in block.instrs:
            arguments = list(instruction.args or ())
            changed = False
            for index, argument in enumerate(arguments):
                settled = _resolve(replacement, argument)
                if settled is not argument:
                    arguments[index] = settled
                    changed = True
            rewritten.append(
                dataclasses.replace(instruction, args=arguments)
                if changed else instruction
            )
        block.instrs = rewritten


def deduplicate_constants(functions) -> int:
    """One ``Const`` definition per distinct value, per block.

    Worth having for itself -- the measured kernels carry 28 and 74 ``Const``
    instructions for 17 and 31 distinct values -- but worth much more for what
    it UNBLOCKS. Two structurally identical expressions are invisible to
    common-subexpression elimination while their equal constants are separate
    values: ``Mul(v, %8)`` and ``Mul(v, %12)`` do not match when ``%8`` and
    ``%12`` both hold 134217729.0. Measured on the same kernels, CSE alone
    finds ONE duplicate; run after this, it finds thirteen and fifty-six.

    So the two belong in this order, and the ordering is the point rather
    than an implementation detail.

    Scoped per block: a block's own earlier definition dominates every use
    the removed one dominated, so no dominance analysis is required to know
    the rewrite is legal.
    """

    removed = 0
    for function in functions.values():
        protected = _protected_value_ids(function)
        replacement: dict = {}
        for block in function.blocks.values():
            canonical: dict = {}
            kept = []
            for instruction in block.instrs:
                value = (_constant_value(instruction)
                         if instruction.op in ("Const", "const") else None)
                if value is None or instruction.res is None:
                    kept.append(instruction)
                    continue
                key = (repr(value), instruction.res.dtype,
                       tuple(instruction.res.shape or ()))
                if key in canonical and int(instruction.res.id) not in protected:
                    replacement[int(instruction.res.id)] = canonical[key]
                    removed += 1
                    continue
                canonical.setdefault(key, instruction.res)
                kept.append(instruction)
            block.instrs = kept
        _rewrite_uses(function, replacement)
    return removed


def eliminate_common_subexpressions(functions) -> int:
    """One computation per distinct pure expression, per block.

    Exactly result-preserving: the surviving instruction computes the same
    operation on the same operands, so every use sees the bit pattern it saw
    before. This is not a numerics policy question and needs no contract.

    Error-free transformations make this unusually profitable. ``two_product``
    splits BOTH its operands, so squaring a value splits it twice, and a
    double-double Horner re-splits the same argument at every step. Measured
    on the sine cores after constant deduplication: 13 of 182 instructions in
    the mixed kernel and 56 of 652 in the full double-double one.

    Scoped per block for the same dominance reason as the constants, which
    also keeps it away from loop-carried values: a definition hoisted across
    a back edge would change which iteration's value a use sees, and nothing
    here moves an instruction.
    """

    removed = 0
    for function in functions.values():
        protected = _protected_value_ids(function)
        replacement: dict = {}
        for block in function.blocks.values():
            canonical: dict = {}
            kept = []
            for instruction in block.instrs:
                if (instruction.op not in _PURE_FOR_SHARING
                        or instruction.res is None):
                    kept.append(instruction)
                    continue
                arguments = [_resolve(replacement, argument)
                             for argument in (instruction.args or ())]
                if arguments != list(instruction.args or ()):
                    instruction = dataclasses.replace(
                        instruction, args=arguments)
                key = (
                    instruction.op,
                    tuple(getattr(argument, "id", repr(argument))
                          for argument in arguments),
                    repr(sorted((instruction.attributes or {}).items(),
                                key=str)),
                )
                if key in canonical and int(instruction.res.id) not in protected:
                    replacement[int(instruction.res.id)] = canonical[key]
                    removed += 1
                    continue
                canonical.setdefault(key, instruction.res)
                kept.append(instruction)
            block.instrs = kept
        _rewrite_uses(function, replacement)
    return removed

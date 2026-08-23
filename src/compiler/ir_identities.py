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
        value.accounting["precision_element"] = str(
            element or value.dtype or ""
        ) or None
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
        requires=("constant_operand",),
        effect="add(x, 0) -> x; mul(x, 1) -> x",
        justification=(
            "Only against a literal exact zero or one. A value that merely "
            "ROUNDS to zero or one is not an identity element: adding a "
            "denormal changes the low limb even when it cannot change the "
            "high one, and that low limb is the entire point of the "
            "representation. Signed zero also disqualifies -- x + (-0.0) is "
            "x except when x is -0.0 -- so the literal must be +0.0."
        ),
    ),
    PrecisionIdentity(
        name="scaling_by_power_of_two",
        matches=("precision_mul",),
        exact=True,
        requires=("constant_operand",),
        effect="scale every limb by the power of two; no transformation",
        justification=(
            "Multiplication by a power of two only shifts the exponent, so "
            "it is exact for every limb independently and preserves the "
            "nonoverlapping property that makes the limbs a valid "
            "expansion. This is what makes Cody-Waite argument reduction "
            "affordable: its range-reduction scalings cost nothing."
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
        requires=("chain_internal_consumers",),
        effect="renormalise once at the end of a chain, not per operation",
        justification=(
            "renorm(renorm(a + b) + c) == renorm(a + b + c). Renormalisation "
            "redistributes limbs into nonoverlapping form; it does not "
            "change the value the expansion represents, so an intermediate "
            "one is needed only if something READS the intermediate. A "
            "chain of n additions therefore needs one renormalisation, not "
            "n. This is the largest win available here, and it became "
            "visible only once precision propagated along a chain -- before "
            "that every operation looked isolated and each one paid."
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
            "be touched at all.

"
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

    ``sterbenz_cancellation`` waits on a proven range (catalogue section
    5); the two kernel entries wait on the bank.

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

        for block in function.blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) != add_name or instruction.res is None:
                    continue
                result = int(instruction.res.id)
                consumers = readers.get(result, ())
                internal = (
                    result not in protected
                    and len(consumers) == 1
                    and str(consumers[0].op) == add_name
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
    # forbids reassociating a parenthesised expression, which is most of
    # isolation, but offers no way to withdraw contraction specifically --
    # so the obligation is not claimed rather than half-claimed.
    "fortran": (FMA_MANDATORY,),
    # No fma instruction exists. It emits, it does not deliver.
    "wasm": (),
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
    #: Result ids of instructions that are `Fma` and must stay one rounding.
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


def precision_section_contracts(functions) -> tuple[PrecisionSectionContract, ...]:
    """Read the contract each precision section places on a destination.

    A section is the maximal run of consecutive instructions in one block
    that carry limbs, plus any `Fma` sitting among them. Adjacency is the
    right criterion here rather than dataflow reachability: what a
    destination must refrain from reoptimizing is a contiguous stretch of
    emitted code, and an `Fma` between two limbed operations is inside the
    stretch whether or not it consumes one of them.
    """

    from ..common.tensors.topological_reducer import PRECISION_SINGULAR_NAMES

    precision_ops = frozenset(PRECISION_SINGULAR_NAMES.values())
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
                        if str(each.op) == FMA and each.res is not None
                    ),
                    limbs=limbs,
                    element=None if element is None else str(element),
                    obligations=PRECISION_SECTION_OBLIGATIONS,
                ))
    return tuple(contracts)

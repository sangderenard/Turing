"""Call argument harmonization: call argument identities, record field demand propagation, harmonised call argument shapes (relocated from fortran_c_shell)"""

from __future__ import annotations
from typing import (
    Any,
    Mapping,
)

from .monotonic_ids import GLOBAL_MONOTONIC_IDS
from .frame_call_linking import _linked_caller_member


def _call_argument_identity(
    owner_function: Any, value: Any, parameter_names_by_id: Mapping[int, str],
) -> tuple[str, Any] | None:
    """The one stable identity a call argument or formal carries, if any.

    Three sources, checked in order, all already computed elsewhere in this
    module for other reasons -- this does not invent new facts, it is the
    first place that reads them uniformly regardless of what KIND of value
    is being asked about:

    - a materialized record field carries its own name (``program_abi_field``,
      e.g. ``"error_channels.keys"``, ``"hard_failure"``);
    - a frame-storage slot appended to complete a callee's grown signature
      carries the id of the formal it was propagated for
      (``propagated_formal_id``);
    - an ordinary value that IS one of its own function's declared
      parameters (state, dt, a plain scalar -- forwarded straight through,
      not derived from a record at all) carries that parameter's own
      source-level name, via the function's ``parameter_names`` metadata.

    Anything reaching none of these three is a genuinely anonymous
    intermediate value used directly as an argument; there is no fact
    anywhere recording what it is, and this function says so honestly
    (``None``) rather than falling back to guessing from position.
    """

    accounting = dict(value.accounting or {})
    field = accounting.get("program_abi_field")
    if field is not None:
        return ("field", str(field))
    propagated = accounting.get("propagated_formal_id")
    if propagated is not None:
        return ("propagated", int(propagated))
    name = parameter_names_by_id.get(int(value.id))
    if name is not None:
        return ("parameter", str(name))
    return None


def _propagate_record_field_demand(
    all_functions: Mapping[str, Any],
    call_records: Mapping[str, Any] | None = None,
    all_record_tables: Mapping[str, Any] | None = None,
    all_sequence_tables: Mapping[str, Any] | None = None,
) -> None:
    """Grow a caller's formals to include every record field a callee it
    calls needs, transitively, before anything tries to align call shapes.

    A field is materialized into a function's OWN formal list only for
    fields that function's OWN body directly references (the ``if not
    candidates: continue`` skip in the nested-record materializer).  That
    is correct in isolation and wrong across a call: a function that only
    FORWARDS a whole record (``_propose_dt_pen(metrics, targets, dx,
    distribution)`` from inside ``step_with_dt_control_used``, never
    touching ``targets.cfl`` itself) never gets that field materialized,
    even though its callee needs it expanded.  There is no "whole record
    pointer" to extract the field from at this level once fields ARE
    materialized -- each becomes its own independent physical value, the
    same way :func:`_prune_unused_callee_formals`'s frame-storage
    completion (``linked_call_frame_storage``) grows a caller for a
    propagated storage slot it does not itself use, just for a different
    kind of need: a record field instead of frame storage.  That existing
    completion runs too late to help here -- at the very end of this
    module's lowering, well after the call-shape unification this feeds
    already has to have run -- so this is the same technique, run early.

    Growing is the only thing this does: it only ever APPENDS a new
    formal, copying the callee's own field metadata verbatim (with
    ``program_abi_field_written`` reset to ``False`` -- the caller is
    forwarding, never writing, whatever it does not already write itself),
    to both the caller's formal list and this one call's argument list.
    Nothing is removed, reordered, or guessed at; the CORRESPONDING
    reorder, once every needed field genuinely exists somewhere in the
    caller, is :func:`_harmonize_call_argument_shapes`'s job, called right
    after this.  Iterated to a fixed point exactly like frame-storage
    completion, so demand reaches as far up the call graph as it needs to.
    """

    from ..transmogrifier.ssa import SSAValue

    call_records = call_records or {}
    all_record_tables = all_record_tables or {}
    all_sequence_tables = all_sequence_tables or {}

    def linked_member(caller_symbol, caller, callee_symbol, callee, instruction, formal):
        """The caller member a callee field formal is, by recorded ids."""

        callsite_id = instruction.attributes.get("plan_callsite_id")
        record = next((
            candidate for candidate in call_records.get(str(caller_symbol), ())
            if candidate.callsite_id == callsite_id
            and str(candidate.callee_symbol) == str(callee_symbol)
        ), None) if callsite_id is not None else None
        if record is None:
            return None
        caller_values = {int(value.id): value for value in caller.args}
        caller_values.update({
            int(item.res.id): item.res
            for block in caller.blocks.values() for item in block.instrs
            if item.res is not None
        })

        member_id = _linked_caller_member(
            str(caller_symbol), record, int(formal.id),
            all_record_tables.get(str(callee_symbol)),
            all_record_tables.get(str(caller_symbol)),
            all_sequence_tables.get(str(callee_symbol)),
            all_sequence_tables.get(str(caller_symbol)),
            {int(value.id) for value in caller.args},
        )
        return None if member_id is None else caller_values.get(int(member_id))

    grew = True
    while grew:
        grew = False
        for caller_symbol, caller in all_functions.items():
            caller_fields = {
                ((formal.accounting or {}).get("program_abi_record"),
                 (formal.accounting or {}).get("program_abi_field")): formal
                for formal in caller.args
                if (formal.accounting or {}).get("program_abi_field") is not None
            }
            for block in caller.blocks.values():
                for instruction in block.instrs:
                    if (
                        instruction.op not in {"Call", "call"}
                        or instruction.attributes.get("tensor_operation")
                    ):
                        continue
                    callee = all_functions.get(str(
                        instruction.attributes.get("callee") or ""
                    ))
                    if callee is None:
                        continue
                    supplied = {
                        ((argument.accounting or {}).get("program_abi_record"),
                         (argument.accounting or {}).get("program_abi_field"))
                        for argument in instruction.args
                        if (argument.accounting or {}).get(
                            "program_abi_field"
                        ) is not None
                    }
                    declared = instruction.attributes.get("callee_input_ids")
                    if declared is not None and len(declared) != len(instruction.args):
                        raise ValueError("native call input receipt and operands disagree")
                    for formal in callee.args:
                        formal_accounting = dict(formal.accounting or {})
                        field = formal_accounting.get("program_abi_field")
                        if field is None:
                            continue
                        key = (formal_accounting.get("program_abi_record"), field)
                        if (int(formal.id) in declared if declared is not None else key in supplied):
                            continue
                        existing = linked_member(
                            caller_symbol, caller,
                            instruction.attributes.get("callee"), callee,
                            instruction, formal,
                        )

                        if existing is not None:
                            # Already materialized somewhere in the caller
                            # (a different call, or its own body) -- this
                            # call just needs to carry the SAME value, not
                            # a new one.
                            instruction.args.append(existing)
                        else:
                            propagated = SSAValue(
                                GLOBAL_MONOTONIC_IDS.mint(),
                                dtype=formal.dtype,
                                shape=tuple(formal.shape or ()),
                                device=formal.device,
                                accounting={
                                    **formal_accounting,
                                    "program_abi_field_written": False,
                                },
                            )
                            caller.args.append(propagated)
                            caller_fields[key] = propagated
                            instruction.args.append(propagated)
                        if declared is not None:
                            declared = (*declared, int(formal.id))
                            instruction.attributes["callee_input_ids"] = declared
                        supplied.add(key)
                        grew = True


def _harmonize_call_argument_shapes(all_functions: Mapping[str, Any]) -> None:
    """Make every call site into one specialized callee agree on argument
    shape, before any dtype/rank fact is unified over them.

    Two callees sharing a record-typed value (or a caller's own forwarded
    parameter) each materialize their OWN formal list independently, from
    their OWN body's references -- one skips a field the other keeps, and
    the same logical value ends up at different positions.  The call-type
    unification loop that runs after this one trusts plain position; a
    mispaired (actual, formal) can never converge there, because the pair
    names two different things to begin with (this is the exact defect
    diagnosed on the managed tire build across 2026-09-03/04, down to the
    specific value: an ``error_channels.keys`` array bound against a
    ``hard_failure`` scalar formal, purely from an unrelated argument-count
    drift earlier in the same call's argument list).

    This realigns each call's argument list to its callee's formal order by
    IDENTITY (:func:`_call_argument_identity`), not position -- the callee's
    own formal list is never touched, so nothing a DIFFERENT caller depends
    on can be removed.  A call is only ever changed when every one of its
    identified arguments has an exact, unambiguous counterpart among the
    callee's identified formals; anything left unidentified on either side
    must already agree in count, in its own existing relative order, or the
    call is named precisely and left for the caller (a human) to resolve --
    never silently paired by leftover position, which is how this defect
    was produced in the first place.
    """

    calls_by_callee: dict[str, list[tuple[Any, Any]]] = {}
    for caller in all_functions.values():
        for block in caller.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op not in {"Call", "call"}
                    or instruction.attributes.get("tensor_operation")
                ):
                    continue
                callee_name = str(instruction.attributes.get("callee") or "")
                if callee_name not in all_functions:
                    continue
                calls_by_callee.setdefault(callee_name, []).append(
                    (caller, instruction)
                )

    parameter_names_cache: dict[str, dict[int, str]] = {}

    def parameter_names_of(function: Any) -> dict[int, str]:
        cached = parameter_names_cache.get(function.name)
        if cached is not None:
            return cached
        names = {
            int(value_id): str(name)
            for name, value_id in (
                (function.metadata or {}).get("parameter_names", ())
            )
        }
        parameter_names_cache[function.name] = names
        return names

    for callee_name, sites in calls_by_callee.items():
        callee = all_functions[callee_name]
        callee_names = parameter_names_of(callee)
        formal_identities = [
            _call_argument_identity(callee, formal, callee_names)
            for formal in callee.args
        ]
        formal_positions: dict[tuple[str, Any], int] = {}
        ambiguous_formal = False
        for position, identity in enumerate(formal_identities):
            if identity is None:
                continue
            if identity in formal_positions:
                ambiguous_formal = True
                break
            formal_positions[identity] = position
        unnamed_formal_count = sum(
            1 for identity in formal_identities if identity is None
        )
        if unnamed_formal_count == len(formal_identities) and not any(
            instruction.attributes.get("callee_input_ids") is not None
            for _caller, instruction in sites
        ):
            # No formal on this callee carries any identity at all -- not a
            # record consumer, not one of an authored function's own named
            # parameters.  This is a compiler-synthesized positional
            # primitive (a generated sequence-insert helper, for example),
            # for which position IS the whole and correct contract; there is
            # nothing here for identity to improve on, so leave it exactly
            # as it already works.
            continue
        for caller, instruction in sites:
            declared = instruction.attributes.get("callee_input_ids")
            if declared is not None:
                if len(declared) != len(instruction.args) or len(set(declared)) != len(declared):
                    raise ValueError(f"invalid native input binding receipt for {callee_name}")
                bound = dict(zip(declared, instruction.args))
                formal_ids = tuple(int(formal.id) for formal in callee.args)
                if all(value_id in bound for value_id in formal_ids):
                    instruction.args = [bound[value_id] for value_id in formal_ids]
                    instruction.attributes["callee_input_ids"] = formal_ids
                # An incomplete exact binding cannot be repaired from names.
                continue
            if ambiguous_formal:
                continue
            caller_names = parameter_names_of(caller)
            actual_identities = [
                _call_argument_identity(caller, actual, caller_names)
                for actual in instruction.args
            ]
            if actual_identities == formal_identities:
                continue
            actual_positions: dict[tuple[str, Any], int] = {}
            ambiguous_actual = False
            for position, identity in enumerate(actual_identities):
                if identity is None:
                    continue
                if identity in actual_positions:
                    ambiguous_actual = True
                    break
                actual_positions[identity] = position
            if ambiguous_actual:
                continue
            if set(actual_positions) != set(formal_positions):
                # An identified field/parameter appears on one side and
                # not the other: this call cannot supply what the callee
                # needs, or supplies something it has no formal for.  Not
                # a position problem this pass can fix by realigning.
                continue
            unnamed_actual_positions = [
                position for position, identity in enumerate(actual_identities)
                if identity is None
            ]
            if len(unnamed_actual_positions) != unnamed_formal_count:
                # The identified portion matches perfectly; the remainder
                # on each side has no identity at all to align by, and
                # there are different numbers of them.  Naming this
                # precisely, instead of leaving it to surface later as an
                # unrelated dtype conflict between two named fields three
                # positions further down the list.
                raise ValueError(
                    "call argument shapes cannot be harmonized: "
                    f"{caller.name!r} -> {callee_name!r} supplies "
                    f"{len(unnamed_actual_positions)} argument(s) with no "
                    "identity (not a record field, not a propagated frame "
                    "slot, not one of the caller's own named parameters), "
                    f"the callee has {unnamed_formal_count} such formal "
                    "position(s); an unidentified remainder must already "
                    "agree in count to be aligned safely; unidentified "
                    "actuals="
                    f"{[(int(instruction.args[p].id), str(instruction.args[p].dtype), sorted((instruction.args[p].accounting or {}).keys())) for p in unnamed_actual_positions]!r} "
                    "unidentified formals="
                    f"{[(int(formal.id), str(formal.dtype), sorted((formal.accounting or {}).keys())) for formal, identity in zip(callee.args, formal_identities) if identity is None]!r} "
                    "formal consumers="
                    f"{[(int(formal.id), [(str(block_name), str(consumer.op), str(consumer.attributes.get('callee') or consumer.attributes.get('binding') or '')) for block_name, block in callee.blocks.items() for consumer in block.instrs if any(int(argument.id) == int(formal.id) for argument in consumer.args)]) for formal, identity in zip(callee.args, formal_identities) if identity is None]!r} "
                    "call actuals="
                    f"{[(int(a.id), str(a.dtype), identity) for a, identity in zip(instruction.args, actual_identities)]!r} "
                    "callee formals="
                    f"{[(int(f.id), str(f.dtype), identity) for f, identity in zip(callee.args, formal_identities)]!r} "
                    "call attributes="
                    f"{sorted(instruction.attributes.keys())!r}"
                )
            realigned = [None] * len(callee.args)
            for identity, formal_position in formal_positions.items():
                realigned[formal_position] = instruction.args[
                    actual_positions[identity]
                ]
            for unnamed_formal_position, unnamed_actual_position in zip(
                (
                    position for position, identity in enumerate(formal_identities)
                    if identity is None
                ),
                unnamed_actual_positions,
            ):
                realigned[unnamed_formal_position] = instruction.args[
                    unnamed_actual_position
                ]
            assert all(value is not None for value in realigned)
            instruction.args = realigned

"""Function alias concordance: read, publish and apply concorded function aliases, output identities and record-return Phi inputs (relocated from fortran_c_shell)"""

from __future__ import annotations
import networkx as nx
from typing import (
    Any,
    Mapping,
    Sequence,
)

from .frame_identity_book import (
    _frame_cells,
    _frame_post,
    _frame_value_cell,
)


_TEMPORAL_LOOP_ARGUMENT_BINDINGS = frozenset({
    "loop_carried_seed",
    "loop_carried",
    "loop_continue_carried",
    "loop_latch_carried",
    "loop_result_port",
})


def _loop_rebound_value_ids(function: Any) -> frozenset:
    """Caller ids a loop rebinds, so a bare id cannot name them.

    Empty when the function declares no loop scope, which is the common
    case and costs one page lookup.
    """

    try:
        from .identity_concordance import (
            current_identity_book, loop_scope_declarations,
        )

        return frozenset(
            int(rebind["outer"])
            for declaration in loop_scope_declarations(
                current_identity_book(), str(function.name),
            )
            for rebind in declaration["rebinds"]
        )
    except Exception:
        return frozenset()


def _concordant_function_aliases(
    function: Any, *, include_output_identities: bool = False,
) -> dict[int, int]:
    """Read every exact alias receipt through the shared concordance.

    Function metadata is the durable snapshot used after compilation.  The
    planning page is the live authority used while regions and call frames
    are still changing.  Late compiler passes must consult both or a fact
    proved after region construction can disappear at precisely that stage.
    ``resolved_concordant_alias_bindings`` compares terminal residents, so
    equivalent one-hop and transitive spellings agree while genuinely
    different storage still refuses.
    """

    from .identity_concordance import (
        authored_function_name,
        concordant_shape_transformation_state,
        descriptor_from_shape_transformation_state,
        record_shape_transformation,
        current_identity_book,
        resolved_concordant_alias_bindings,
    )

    metadata = dict(getattr(function, "metadata", {}) or {})
    aliases = resolved_concordant_alias_bindings(
        str(function.name),
        dict(metadata.get("value_aliases", {}) or {}),
    )
    if not include_output_identities:
        return aliases

    page = current_identity_book().page("output_identity_concordance")
    durable = dict(metadata.get("output_identity_aliases", ()) or ())
    concorded = page.alias_bindings(str(function.name))
    if durable != concorded:
        raise ValueError(
            "output identity concordance disagreement for "
            f"{function.name!r}: metadata={durable!r}, page={concorded!r}"
        )

    # Output history is a later semantic stage than storage planning.  An
    # authored spelling may first resolve to provisional storage and later be
    # selected as a function's merged output.  The output page therefore
    # supersedes the same source row instead of pretending the two stages are
    # simultaneous physical-alias claims.
    combined = {**aliases, **concorded}

    def terminal(value_id: int) -> int:
        current = int(value_id)
        path: list[int] = []
        while current in combined and int(combined[current]) != current:
            if current in path:
                raise ValueError(
                    "cyclic output identity concordance for "
                    f"{function.name!r}: {tuple((*path, current))}"
                )
            path.append(current)
            current = int(combined[current])
        return current

    return {alias: terminal(alias) for alias in combined}


def _publish_concorded_output_identities(
    function: Any, bindings: Mapping[int, int],
) -> dict[int, int]:
    """Publish semantic output history without making it a storage alias.

    A return-merge result and each value arriving on its predecessor edges are
    related by source output history, but are not interchangeable SSA
    occurrences. Keeping that relation on its own page prevents an ordinary
    physical alias settlement from turning ``Phi(a, b)`` into
    ``Phi(phi, phi)`` while still giving record/loop consumers one concorded
    output-identity authority.
    """

    from .concordance_declarations import OUTPUT_IDENTITY, OUTPUT_IDENTITY_STAGE
    from .identity_concordance import Mode, current_identity_book
    from .precompile_to_ssa import _post_derived_or_raw
    from .ssa_record_return_state import identity_cells

    book = current_identity_book()
    page = book.page(OUTPUT_IDENTITY)
    committed = dict(function.metadata.get("output_identity_aliases", ()) or ())
    for alias_id, result_id in bindings.items():
        alias_id, result_id = int(alias_id), int(result_id)
        incumbent = page.latest((str(function.name), alias_id))
        if incumbent is not None and int(incumbent) != result_id:
            raise ValueError(
                "output identity concordance disagreement for "
                f"{function.name!r} value {alias_id}: page says "
                f"{incumbent}, new result says {result_id}"
            )
        durable = committed.get(alias_id)
        if durable is not None and int(durable) != result_id:
            raise ValueError(
                "output identity concordance disagreement for "
                f"{function.name!r} value {alias_id}: metadata says "
                f"{durable}, new result says {result_id}"
            )
        if incumbent is None:
            # The output history row derives from the two values it relates:
            # the alias's and the result's identity cells (raw, as before,
            # when neither value has a row).
            _post_derived_or_raw(
                book, OUTPUT_IDENTITY, (str(function.name), alias_id), result_id,
                identity_cells(function, alias_id, result_id),
                stage=OUTPUT_IDENTITY_STAGE, mode=Mode.REVISE,
            )
        committed[alias_id] = result_id
    function.metadata["output_identity_aliases"] = tuple(sorted(
        committed.items()
    ))
    return committed


def _concord_record_return_phi_inputs(
    function_name: str, instruction: Any, candidates: Sequence[Any],
    *, selection_cells: Sequence[Any] = (), function: Any = None,
) -> list[Any]:
    """Keep a return-field Phi's predecessor identities distinct from itself.

    Record materialization is revisited during the call-frame fixed point. Once
    the merged record descriptor points at the field Phi results, reading that
    descriptor back as though it were an incoming record proposes the Phi's
    own result as its operand. The incumbent predecessor is the stronger SSA
    fact. Record both the proposed candidate and retained selection so every
    revisit consumes the same concorded decision.

    ``selection_cells`` (one per position, None where there is none) are the
    ``record_return_field_selection`` cells the candidates were chosen by.  A
    changed choice whose selection cell is stamped after the row's prior
    entry is a revision with a cause and is recorded with that cell as its
    source; a change with no changed source is still a disagreement.

    ``function`` (the SSA function the Phi belongs to) lets each row post
    DERIVED(the selection cell, the chosen value's ``ssa_value`` cell); a
    call without it, or a row with neither cell, writes raw as before.
    """

    from .concordance_declarations import (
        RECORD_RETURN_PHI_INPUT, RECORD_RETURN_VERSION,
    )
    from .identity_concordance import (
        ConcordanceRefusal, Derived, Mode, Ref, current_identity_book,
    )
    from .ssa_record_return_state import identity_cells

    result = instruction.res
    if result is None:
        return list(candidates)
    incumbents = list(instruction.args)
    if len(incumbents) != len(candidates):
        return list(candidates)
    book = current_identity_book()
    page = book.page(RECORD_RETURN_PHI_INPUT)
    selection_cells = tuple(selection_cells) + (None,) * (
        len(candidates) - len(tuple(selection_cells))
    )
    selected = []
    predecessors = tuple((instruction.attributes or {}).get(
        "incoming_blocks", ()
    ))
    field_name = str((instruction.attributes or {}).get(
        "record_field", ""
    ))
    for position, (incumbent, candidate) in enumerate(zip(
        incumbents, candidates,
    )):
        if int(candidate.id) == int(result.id):
            if int(incumbent.id) == int(result.id):
                raise ValueError(
                    "record return Phi has no non-self predecessor identity: "
                    f"function={function_name!r}, result={int(result.id)}, "
                    f"field={field_name!r}, position={position}"
                )
            chosen = incumbent
            reason = "merged_descriptor_self_candidate_rejected"
        else:
            chosen = candidate
            reason = "incoming_record_field"
        row = (
            str(function_name), int(result.id), field_name, int(position),
            str(predecessors[position]) if position < len(predecessors) else "",
        )
        selection = selection_cells[position]
        fact = (int(candidate.id), int(chosen.id), reason) + (
            (selection.key,) if isinstance(selection, Ref) else ()
        )
        entries = page.history(row)
        prior = entries[-1][1] if entries else None
        if prior is not None and int(tuple(prior)[1]) != int(chosen.id):
            # A different choice is a legal revision only when the selection
            # cell it derives from changed after the prior entry was written.
            prior_stamp = page.stamps.get((row, entries[-1][0]), -1)
            caused = (
                isinstance(selection, Ref)
                and book.stamp_of(selection) > prior_stamp
            )
            if not caused:
                raise ValueError(
                    "record return Phi input concordance disagreement: "
                    f"row={row!r}, prior={prior!r}, new={fact!r}"
                )
        if prior == fact:
            # The same decision again: nothing to revise.
            selected.append(chosen)
            continue
        cells = () if function is None else identity_cells(
            function, selection, chosen,
        )
        posted = False
        if cells:
            try:
                book.post(
                    RECORD_RETURN_PHI_INPUT, row, fact,
                    stage=RECORD_RETURN_VERSION, provenance=Derived(cells),
                    mode=Mode.REVISE,
                )
                posted = True
            except ConcordanceRefusal:
                # The candidate or reason changed with no changed source
                # (the choice itself did not, or the check above raised):
                # recorded raw, so the audit lists it.
                posted = False
        if not posted:
            page.set(
                row, max(page.columns, default=-1) + 1, fact,
            )
        selected.append(chosen)
    return selected


def _apply_concorded_function_aliases(
    function: Any, aliases: Mapping[int, int],
) -> tuple[tuple[Any, ...], ...]:
    """Apply physical aliases only where the selected resident dominates.

    Planning identity is timeless; SSA occurrences are not. A return merge can
    be the final resident of an authored value without being available in the
    loop or predecessor edges that produce it. Every attempted substitution is
    recorded, including dominance abstentions.
    """

    from .identity_concordance import current_identity_book

    block_names = tuple(function.blocks)
    if not block_names:
        return ()
    entry = "entry" if "entry" in function.blocks else block_names[0]
    cfg = nx.DiGraph()
    cfg.add_nodes_from(block_names)
    for block_name, block in function.blocks.items():
        cfg.add_edges_from(
            (str(block_name), str(successor))
            for successor in block.successors
            if successor in function.blocks
        )
    reachable = set(nx.descendants(cfg, entry)) | {entry}
    immediate = nx.immediate_dominators(cfg.subgraph(reachable), entry)
    immediate[entry] = entry

    def block_dominates(owner: str, target: str) -> bool:
        current = str(target)
        while current in immediate:
            if current == str(owner):
                return True
            parent = immediate[current]
            if parent == current:
                break
            current = parent
        return False

    formals = {int(value.id) for value in function.args}
    values = {int(value.id): value for value in function.args}
    definitions: dict[int, list[tuple[str, int, Any]]] = {}
    for block_name, block in function.blocks.items():
        for instruction_index, instruction in enumerate(block.instrs):
            if instruction.res is None:
                continue
            value_id = int(instruction.res.id)
            values.setdefault(value_id, instruction.res)
            definitions.setdefault(value_id, []).append((
                str(block_name), int(instruction_index), instruction,
            ))

    def available(
        value_id: int, target_block: str, use_index: int, phi_edge: bool,
    ) -> bool:
        value_id = int(value_id)
        if value_id in formals:
            return True
        locations = definitions.get(value_id, ())
        if len(locations) != 1:
            return False
        owner, definition_index, _instruction = locations[0]
        if not block_dominates(owner, target_block):
            return False
        if owner != target_block:
            return True
        return phi_edge or definition_index < use_index

    def terminal(value_id: int) -> int:
        current = int(value_id)
        seen: set[int] = set()
        while current in aliases and current not in seen:
            seen.add(current)
            current = int(aliases[current])
        return current

    page = current_identity_book().page("alias_application_concordance")
    receipts = []
    for block_name, block in function.blocks.items():
        for instruction_index, instruction in enumerate(block.instrs):
            if (
                instruction.attributes.get("binding")
                in _TEMPORAL_LOOP_ARGUMENT_BINDINGS
            ):
                continue
            incoming = tuple((instruction.attributes or {}).get(
                "incoming_blocks", ()
            )) if str(instruction.op).casefold() == "phi" else ()
            rewritten = list(instruction.args)
            for position, argument in enumerate(tuple(instruction.args)):
                resident_id = terminal(int(argument.id))
                resident = values.get(resident_id)
                if resident is None or resident_id == int(argument.id):
                    continue
                target_block = (
                    str(incoming[position])
                    if position < len(incoming) else str(block_name)
                )
                phi_edge = position < len(incoming)
                can_apply = available(
                    resident_id, target_block, int(instruction_index), phi_edge,
                )
                selected = resident if can_apply else argument
                reason = (
                    "resident_dominates_use"
                    if can_apply else "resident_not_available_at_use"
                )
                row = (
                    str(function.name), str(block_name), int(instruction_index),
                    int(position),
                )
                fact = (
                    int(argument.id), resident_id, int(selected.id), reason,
                    target_block,
                )
                prior = page.latest(row)
                if prior is not None and tuple(prior) != fact:
                    raise ValueError(
                        "alias application concordance disagreement: "
                        f"row={row!r}, prior={prior!r}, new={fact!r}"
                    )
                if prior is None:
                    page.set(
                        row, max(page.columns, default=-1) + 1, fact,
                    )
                rewritten[position] = selected
                receipts.append((*row[1:], *fact))
            instruction.args = rewritten
            # An id-carrying reference follows the same alias.  A Phi's
            # ``initial_value_id`` names the version that stood before its
            # merge; the planning alias that retired that spelling has a
            # resident, and the Phi must name THAT.  Left alone it names an id
            # the function no longer defines (N=2 orbital dt system: the
            # conditional-carried Phis of ``step_with_dt_control_used`` and
            # ``_apply_energy_sidechain``).  It is a use one past the
            # operands, and its row is the operands' page.
            initial = (instruction.attributes or {}).get("initial_value_id")
            if initial is not None and instruction.res is not None:
                resident_id = terminal(int(initial))
                if (
                    resident_id != int(initial)
                    and values.get(resident_id) is not None
                ):
                    row = (
                        str(function.name), str(block_name),
                        int(instruction_index), len(instruction.args),
                    )
                    fact = (
                        int(initial), resident_id, resident_id,
                        "initial_names_resident", str(block_name),
                    )
                    prior = page.latest(row)
                    if prior is not None and tuple(prior) != fact:
                        raise ValueError(
                            "alias application concordance disagreement: "
                            f"row={row!r}, prior={prior!r}, new={fact!r}"
                        )
                    if prior is None:
                        page.set(
                            row, max(page.columns, default=-1) + 1, fact,
                        )
                    instruction.attributes["initial_value_id"] = int(
                        page.latest(row)[2]
                    )
                    receipts.append((*row[1:], *fact))
    return tuple(receipts)


def _publish_concordant_function_aliases(
    function: Any, bindings: Mapping[int, int], *, provenance: str,
    sources: Mapping[int, tuple] | None = None,
) -> dict[int, int]:
    """Advance the live page and durable alias snapshot together.

    ``bindings`` are later, explicitly sourced evidence. When such evidence
    refines an existing provisional resident, record the transition before
    advancing it; callers may not silently overwrite a planning row.
    ``sources`` names, per alias id, the book cells the binding derives
    from (step 7); ``provenance`` stays the role string of the shape half.
    """

    from .identity_concordance import (
        concordant_shape_transformation_state,
        current_identity_book,
        descriptor_from_shape_transformation_state,
        record_shape_transformation,
        resolved_concordant_alias_bindings,
        shape_scope_of,
    )

    from .concordance_declarations import (
        PLANNING_ALIAS_TRANSITION, PLANNING_RESIDENCY, PLANNING_VALUE,
    )
    from .identity_concordance import Mode

    aliases = _concordant_function_aliases(function)
    page = current_identity_book().page("planning_value_concordance")
    transition_page = current_identity_book().page(
        "planning_alias_transition_concordance"
    )
    # Step 7: ``sources`` maps an alias id to the cells its binding derives
    # from; an alias with none is posted Unsourced (the caller named no
    # cell), never dropped and never a bare write.
    sources = {} if sources is None else dict(sources)

    def alias_cells(alias_id: int, resident_id: int) -> tuple:
        return _frame_cells(
            *sources.get(int(alias_id), ()),
            _frame_value_cell(function, alias_id),
            _frame_value_cell(function, resident_id),
        )

    advanced = dict(aliases)
    for alias_id, resident_id in bindings.items():
        alias_id, resident_id = int(alias_id), int(resident_id)
        resident_id = int(aliases.get(resident_id, resident_id))
        incumbent = int(aliases.get(alias_id, alias_id))
        if incumbent != resident_id and alias_id in aliases:
            fact = (incumbent, resident_id, str(provenance))
            row = (str(function.name), alias_id)
            prior = transition_page.latest(row)
            if prior is not None and tuple(prior) != fact:
                prior_target = int(tuple(prior)[1])
                if prior_target != incumbent:
                    raise ValueError(
                        "planning alias transition disagreement for "
                        f"{function.name!r} value {alias_id}: prior={prior!r}, "
                        f"new={fact!r}"
                    )
            _frame_post(
                PLANNING_ALIAS_TRANSITION, row, fact,
                stage=PLANNING_RESIDENCY, mode=Mode.REVISE,
                cells=alias_cells(alias_id, resident_id),
            )
        advanced[alias_id] = resident_id

    # Resolve chains after the later evidence has superseded its exact source
    # rows. The page itself is updated below; including its stale rows here
    # would misrepresent a temporal refinement as simultaneous disagreement.
    aliases = resolved_concordant_alias_bindings(
        str(function.name), advanced,
        page=type(page)(page.name),
    )
    for alias_id, resident_id in aliases.items():
        alias_id = int(alias_id)
        resident_id = int(resident_id)
        if page.latest((str(function.name), alias_id)) != resident_id:
            _frame_post(
                PLANNING_VALUE, (str(function.name), alias_id), resident_id,
                stage=PLANNING_RESIDENCY, mode=Mode.REVISE,
                cells=alias_cells(alias_id, resident_id),
            )
        authored_scope = shape_scope_of(function)
        source_state = concordant_shape_transformation_state(
            authored_scope, alias_id,
        )
        if source_state is not None:
            record_shape_transformation(
                authored_scope, alias_id,
                authored_scope, resident_id,
                stage="planning_value_concordance",
                operation="alias_residency",
                source_state=source_state,
                target_state=source_state,
                role=str(provenance),
            )
            descriptor = descriptor_from_shape_transformation_state(
                source_state
            )
            row_shape = (
                None if descriptor is None
                else descriptor.get("sequence_row_shape")
            )
            if row_shape is not None:
                from .identity_concordance import commit_sequence_row_layout

                commit_sequence_row_layout(
                    authored_scope, resident_id,
                    (tuple(map(int, row_shape)),),
                    ((descriptor or {}).get("dtype"),),
                    source=(
                        "planning concordance transformation "
                        f"{alias_id}->{resident_id}"
                    ),
                )
    if aliases:
        function.metadata["value_aliases"] = aliases
    return aliases


def _concord_unbound_variant_rows(function: Any) -> dict[int, int]:
    """Advance provisional row carriers to an exact linked resident.

    Variant-aware region planning can precede source-call linking. It then
    creates a row carrier whose ``unbound_variant_source_id`` names the
    semantic aggregate. If linking later advances that source to a physical
    span, the row carrier is a view of the same resident, not another ABI
    object. Publish that second edge through the ordinary concordance so
    dominance-aware alias settlement and signature pruning remain the sole
    mutation transaction.
    """

    aliases = _concordant_function_aliases(function)

    def terminal(value_id: int) -> int:
        current = int(value_id)
        seen: set[int] = set()
        while current in aliases and current not in seen:
            seen.add(current)
            current = int(aliases[current])
        return current

    values_by_id = {
        int(argument.id): argument for argument in function.args
    }
    values_by_id.update({
        int(instruction.res.id): instruction.res
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    })
    physical_ids = set(values_by_id)
    bindings = {}
    for argument in function.args:
        source_id = (argument.accounting or {}).get(
            "unbound_variant_source_id"
        )
        if source_id is None:
            continue
        resident_id = terminal(int(source_id))
        if resident_id == int(source_id):
            # Planned-region calls preserve their source feed identities even
            # after call linking replaces an aggregate result by its exact
            # physical span. A provisional variant row and the linked span
            # can therefore coexist as actuals for the same feed id. Accept
            # the span only when every non-provisional physical occurrence
            # agrees on one resident; multiple candidates are a real variant
            # and deliberately remain distinct.
            candidates = {
                int(actual.id)
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.op in {"Call", "call"}
                for position, feed_id in enumerate(tuple(
                    instruction.attributes.get("feed_ids") or ()
                ))
                if int(feed_id) == int(source_id)
                and position < len(instruction.args)
                for actual in (instruction.args[position],)
                if int(actual.id) not in {
                    int(argument.id), int(source_id),
                }
                and (
                    (actual.accounting or {}).get(
                        "linked_call_frame_storage"
                    ) is not None
                    or (
                        (actual.accounting or {}).get(
                            "program_abi_storage"
                        ) == "span"
                    )
                )
            }
            if len(candidates) == 1:
                resident_id = next(iter(candidates))
        if (
            resident_id != int(source_id)
            and resident_id != int(argument.id)
            and resident_id in physical_ids
        ):
            bindings[int(argument.id)] = resident_id
    if not bindings:
        return aliases
    return _publish_concordant_function_aliases(
        function,
        bindings,
        provenance="linked_variant_row_resident",
    )


def _concordant_function_resident(function: Any, value_id: int) -> int:
    """Return the terminal physical resident for one function-local value."""

    value_id = int(value_id)
    return int(_concordant_function_aliases(function).get(value_id, value_id))

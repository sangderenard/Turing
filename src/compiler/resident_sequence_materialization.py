"""Resident sequence materialisation: loop-carried storage aliases, literal and append mutations, capacity bounds (relocated from fortran_c_shell)."""

from __future__ import annotations

import ast
import dataclasses
import os
import sys
from dataclasses import replace
from typing import Any, Iterable, Mapping


def _source_contract_rows(graph_obj: Any, page: Any) -> tuple[tuple, ...]:
    """Enumerate page rows in every exact source scope carried by the graph."""

    metadata = graph_obj.graph
    scopes = []
    for name in (
        "source_glyph_sequence_scope",
        "source_aggregate_layout_scope",
        "source_numeric_scope",
        "authored_function_identity",
        "source_numeric_base_scope",
    ):
        scope = metadata.get(name)
        if scope is not None and scope not in scopes:
            scopes.append(scope)
    if not scopes:
        raise ValueError(
            f"graph has no exact source scope for concordance page {page.name}"
        )
    return tuple(dict.fromkeys(
        tuple(row)
        for scope in scopes
        for row in page.scope_rows(scope)
    ))


def _source_glyph_sequence_materializations(
    graph_obj: Any, *, with_cells: bool = False,
    compiled_validation_diagnostic_ids: Iterable[int] = (),
):
    """Read exact JoinedStr recipes into incumbent resident-sequence work.

    The reducer owns component order, glyph identities, conversions and value
    edges on ``source_glyph_sequence``.  This consumer never re-walks the AST.
    Literal components become fixed one-column token sequences; formatted
    components are resident sequences produced by their source-linked helper
    calls.  The JoinedStr destination is cleared and extended in recipe order
    through the ordinary sequence ABI.

    ``compiled_validation_diagnostic_ids`` are exact JoinedStr values used
    only as the message argument of a terminal raise guard already represented
    by the control program's ``ValidationBlock``.  Native validation consumes
    the predicate and exception identity; authored fallback retains the Python
    diagnostic.  Those recipes therefore have no resident consumer and are not
    materialised.  Unknown formatted values anywhere else remain a refusal.

    ``with_cells`` adds a fourth result: sequence id -> the recipe cells each
    declaration was read from, so a declaration's contract can derive from
    them (a compiler-authored sequence has no graph cell of its own).
    """

    from .concordance_declarations import (
        CONTROL_PROGRAM_BUILD,
        GLYPH_SEQUENCE_STEP,
        SOURCE_GLYPH_SEQUENCE,
        SourceGlyphSequence,
    )
    from .control_source import ControlSequenceMutation
    from .identity_concordance import Derived, Mode, current_identity_book

    book = current_identity_book()
    compiled_validation_diagnostic_ids = frozenset(map(
        int, compiled_validation_diagnostic_ids,
    ))
    declarations: list[tuple[int, str, int, bool]] = []
    initializations: list[tuple[int, str, int]] = []
    mutations: list[ControlSequenceMutation] = []
    step_counts: dict[tuple, int] = {}
    declaration_cells: dict[int, list] = {}

    def declare(cell, entry):
        """One declaration, remembered with the recipe cell it was read
        from."""

        declarations.append(entry)
        cells = declaration_cells.setdefault(int(entry[0]), [])
        if cell not in cells:
            cells.append(cell)

    def glyph_step(
        destination_id, operator, arguments, *, argument_kind, cell,
    ):
        """One composition step as a mutation owning its own control block.

        The clear and each extend of one destination share an effect node;
        each is a ``glyph_sequence_step`` row derived from the recipe cell,
        and that cell owns the step's block.
        """

        destination_id = int(destination_id)
        arguments = tuple(map(int, arguments))
        count_key = (cell.row[0], destination_id)
        ordinal = step_counts.get(count_key, 0)
        step_counts[count_key] = ordinal + 1
        row = (cell.row[0], destination_id, ordinal)
        fact = (operator, arguments, argument_kind)
        step_cell = book.latest_ref(GLYPH_SEQUENCE_STEP, row)
        if step_cell is None or book.page(GLYPH_SEQUENCE_STEP).latest(row) != fact:
            step_cell = book.post(
                GLYPH_SEQUENCE_STEP, row, fact, stage=CONTROL_PROGRAM_BUILD,
                provenance=Derived((cell,)), mode=Mode.REVISE,
            )
        return ControlSequenceMutation(
            destination_id, operator, arguments, destination_id,
            policy="duplicates", argument_kind=argument_kind,
            source_contract_cell=cell, step_cell=step_cell,
        )

    nodes_by_value = {
        int(data.get("value_id", node_id)): (int(node_id), data)
        for node_id, data in graph_obj.nodes(data=True)
    }
    page = book.page(SOURCE_GLYPH_SEQUENCE)
    for row in _source_contract_rows(graph_obj, page):
        if len(row) != 2 or int(row[1]) not in nodes_by_value:
            continue
        node_id, data = nodes_by_value[int(row[1])]
        cell = book.latest_ref(SOURCE_GLYPH_SEQUENCE, tuple(row))
        if cell is None:
            raise ValueError(
                "glyph-sequence graph view has no concordance source: "
                f"value={int(data.get('value_id', node_id))} row={row!r}"
            )
        fact = book.page(SOURCE_GLYPH_SEQUENCE).cell(cell.row, cell.column)
        if not isinstance(fact, SourceGlyphSequence):
            raise TypeError(
                f"source glyph sequence has invalid fact {fact!r} at {row!r}"
            )
        joined_id = int(row[1])
        if joined_id in compiled_validation_diagnostic_ids:
            continue
        literal_alternate = (
            len(fact.components) == 1
            and str(fact.components[0].kind) == "literal"
            and int(fact.components[0].component_value_id) == joined_id
        )
        if literal_alternate:
            if (
                len(fact.components) != 1
                or str(fact.components[0].kind) != "literal"
            ):
                raise ValueError(
                    "literal glyph alternate has no exact literal recipe: "
                    f"value={joined_id} row={row!r}"
                )
            declare(cell, (joined_id, "duplicates", 1, False))
            initializations.append((
                joined_id,
                "literal_glyphs="
                f"{tuple(map(int, fact.components[0].glyph_tokens))!r}",
                1,
            ))
            continue
        declare(cell, (joined_id, "duplicates", 1, True))
        initializations.append((joined_id, "duplicates", 1))
        mutations.append(glyph_step(
            joined_id, "clear", (),
            argument_kind="glyph_sequence_clear", cell=cell,
        ))
        def materialize_component(component):
            if component.union_alternatives:
                if component.operand_value_id is None:
                    raise ValueError(
                        "value-union glyph component has no semantic operand: "
                        f"value={int(component.component_value_id)}"
                    )
                destination_id = int(component.component_value_id)
                arm_sequence_ids = tuple(
                    int(arm.sequence_value_id)
                    for arm in component.union_alternatives
                )
                if len(arm_sequence_ids) != len(set(arm_sequence_ids)):
                    raise ValueError(
                        "value-union glyph alternatives share an arm recipe: "
                        f"value={destination_id}"
                    )
                declare(cell, (destination_id, "duplicates", 1, True))
                initializations.append((destination_id, "duplicates", 1))
                mutations.append(ControlSequenceMutation(
                    destination_id,
                    "glyph_union_select",
                    (int(component.operand_value_id), *arm_sequence_ids),
                    destination_id,
                    policy="duplicates",
                    argument_kind="value_union_glyph_projection",
                    source_contract_cell=cell,
                ))
                return destination_id
            if component.projection_components:
                destination_id = int(component.component_value_id)
                declare(cell, (destination_id, "duplicates", 1, True))
                initializations.append((destination_id, "duplicates", 1))
                mutations.append(glyph_step(
                    destination_id, "clear", (),
                    argument_kind="glyph_projection_clear", cell=cell,
                ))
                for child in component.projection_components:
                    child_id = materialize_component(child)
                    mutations.append(glyph_step(
                        destination_id, "extend", (child_id,),
                        argument_kind="glyph_projection_component",
                        cell=cell,
                    ))
                return destination_id
            if (
                str(component.kind) == "formatted"
                and component.formatted_sequence_value_id is None
                and str(component.projection_kind) not in {
                    "string_token", "enum_name",
                }
            ):
                raise ValueError(
                    "formatted glyph component has no resident producer: "
                    f"value={int(component.component_value_id)} "
                    f"projection={component.projection_kind!r}"
                )
            component_id = int(
                component.formatted_sequence_value_id
                if component.formatted_sequence_value_id is not None
                else component.component_value_id
            )
            declare(cell, (
                component_id, "duplicates", 1,
                str(component.kind) != "literal",
            ))
            if str(component.kind) == "literal":
                rows = tuple((int(token),) for token in component.glyph_tokens)
                initializations.append((
                    component_id,
                    f"literal_glyphs={tuple(row[0] for row in rows)!r}",
                    1,
                ))
            return component_id

        for component in fact.components:
            component_id = materialize_component(component)
            mutations.append(glyph_step(
                joined_id, "extend", (component_id,),
                argument_kind="glyph_sequence_component", cell=cell,
            ))
    result = (
        tuple(dict.fromkeys(declarations)),
        tuple(dict.fromkeys(initializations)),
        tuple(mutations),
    )
    if with_cells:
        return (*result, {
            sequence_id: tuple(cells)
            for sequence_id, cells in declaration_cells.items()
        })
    return result


def _glyph_component_resident_sequence_ids(component: Any) -> tuple[int, ...]:
    """Return exact resident recipes owned by one glyph component."""

    nested = tuple(
        value_id
        for child in tuple(
            getattr(component, "projection_components", ()) or ()
        )
        for value_id in _glyph_component_resident_sequence_ids(child)
    )
    alternatives = tuple(
        int(arm.sequence_value_id)
        for arm in tuple(
            getattr(component, "union_alternatives", ()) or ()
        )
    )
    formatted = getattr(component, "formatted_sequence_value_id", None)
    format_spec = getattr(component, "format_spec_value_id", None)
    return tuple(dict.fromkeys((
        *nested,
        *alternatives,
        *(() if formatted is None else (int(formatted),)),
        *(() if format_spec is None else (int(format_spec),)),
    )))


def _compiled_validation_diagnostic_sequence_ids(
    graph_obj: Any,
) -> frozenset[int]:
    """Read glyph values owned only by compiled validation diagnostics.

    The planner records those values as identity-bearing cells on each
    ``ValidationBlock``'s ``control_block`` concordance fact.  Join those cells
    back to this graph's identities; do not rediscover ownership from AST or
    graph shape in the materializer.
    """

    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import CONTROL_BLOCK, ControlBlockKind
    from .identity_concordance import current_identity_book

    scope = graph_obj.graph.get("lexical_read_scope")
    if scope is None:
        return frozenset()
    book = current_identity_book()
    page = book.page(CONTROL_BLOCK)
    diagnostic_cells = {
        cell
        for row in page.scope_rows(tuple(scope))
        for fact in (page.latest(row),)
        if getattr(fact, "kind", None) is ControlBlockKind.VALIDATION
        for cell in tuple(getattr(fact, "diagnostics", ()))
    }
    if not diagnostic_cells:
        return frozenset()
    diagnostic_ids = {
        int(data.get("value_id", node_id))
        for node_id, data in graph_obj.nodes(data=True)
        if node_identity_cell(graph_obj, int(node_id)) in diagnostic_cells
        and (data.get("attributes") or {}).get(
            "source_glyph_sequence_row"
        ) is not None
    }

    # A formatted component is a compiler-authored source-linked call whose
    # result is referenced by the JoinedStr recipe through concordance, not a
    # graph consumer edge.  When the JoinedStr is validation-only, those call
    # results are validation-only too.  Return the complete recipe closure so
    # call planning cannot strand formatter calls after the loop that owned
    # their operands (for example ``position`` in ``_validate_row``).
    source_page = book.page("source_glyph_sequence")

    # id_space.compose's diagnostic row 29 owns a formatted component
    # whose exact format_spec_value_id is JoinedStr 8 ("#x"). The previous
    # closure omitted that edge, so 8 became an unbounded FIXED arena leased
    # to the caller, although native validation never consumes its glyphs.
    # Follow resident recipe edges, and recurse only into graph values that
    # carry their own source_glyph_sequence row. Formatter result sequences
    # (compose's value 36) are marked diagnostic too, but cannot be asked to
    # own a JoinedStr row merely because they belong to that same closure.
    glyph_rows = {
        int(data.get("value_id", node_id)): tuple(row)
        for node_id, data in graph_obj.nodes(data=True)
        for row in ((data.get("attributes") or {}).get(
            "source_glyph_sequence_row"
        ),)
        if row is not None
    }
    pending = list(diagnostic_ids)
    visited = set()
    while pending:
        value_id = pending.pop()
        if value_id in visited:
            continue
        visited.add(value_id)
        row = glyph_rows.get(value_id)
        fact = None if row is None else source_page.latest(row)
        if fact is None:
            raise ValueError(
                "compiled validation diagnostic has no glyph concordance: "
                f"value={value_id} row={row!r}"
            )
        members = tuple(
            member
            for component in tuple(getattr(fact, "components", ()) or ())
            for member in _glyph_component_resident_sequence_ids(component)
        )
        diagnostic_ids.update(members)
        pending.extend(member for member in members if member in glyph_rows)
    return frozenset(diagnostic_ids)


def _string_token_glyph_materializations(graph_obj: Any, string_table: Any):
    """Materialize exact StringTable token spellings as a keyed SSA table."""

    from .concordance_declarations import (
        CELL_SET,
        CONTROL_PROGRAM_BUILD,
        SOURCE_GLYPH_SEQUENCE,
        STRING_TABLE_ENTRY,
        STRING_GLYPH_VOCABULARY,
        STRING_GLYPH_VOCABULARY_MINT,
        SOURCE_ENUM_GLYPH_TABLE,
        SourceEnumGlyphTable,
        SourceGlyphSequence,
    )
    from .control_source import ControlSequenceMutation
    from .identity_concordance import (
        NEW, ConcordanceRefusal, Derived, Mode, Novel, current_identity_book,
    )
    from .string_table import string_token
    from .glyph_sequence_format import quoted_string_glyph_tokens

    book = current_identity_book()
    recipes = []
    enum_recipes = []
    live_value_ids = {
        int(data.get("value_id", node_id))
        for node_id, data in graph_obj.nodes(data=True)
    }
    glyph_page = book.page(SOURCE_GLYPH_SEQUENCE)
    for row in _source_contract_rows(graph_obj, glyph_page):
        if len(row) != 2 or int(row[1]) not in live_value_ids:
            continue
        cell = book.latest_ref(SOURCE_GLYPH_SEQUENCE, tuple(row))
        if cell is None:
            raise ValueError(f"glyph recipe row has no latest cell: {row!r}")
        fact = glyph_page.cell(cell.row, cell.column)
        if not isinstance(fact, SourceGlyphSequence):
            raise TypeError(f"invalid source glyph recipe {fact!r}")
        def component_tree(component):
            yield component
            for child in component.projection_components:
                yield from component_tree(child)

        for root_component in fact.components:
            for component in component_tree(root_component):
                if (
                    str(component.kind) == "formatted"
                    and str(component.projection_kind) in {
                        "string_token", "enum_name",
                    }
                    and component.formatted_sequence_value_id is None
                    and component.operand_value_id is not None
                ):
                    target = (
                        enum_recipes
                        if str(component.projection_kind) == "enum_name"
                        else recipes
                    )
                    target.append((cell, component))
    if not recipes and not enum_recipes:
        return (), (), ()

    source_scope = tuple((recipes or enum_recipes)[0][0].row[0])

    def mint_vocabulary(rows, cells):
        """Mint one vocabulary table id through the book: a NOVEL
        ``string_glyph_vocabulary`` row on ``(source_scope, NEW)`` under
        ``string_glyph_vocabulary_mint`` whose operand is the one cell in
        ``cells`` (several become one ``cell_set`` row DERIVED from all of
        them, as ``_mint_ssa_id`` does).  The id is born with its mint edge,
        where the SSA composition audit reads it; a ``GLOBAL_MONOTONIC_IDS``
        mint here recorded nothing and the function that declared the table
        was refused as a dependency (``unsourced-identity``).  Returns
        ``(vocabulary id, its cell)``."""
        cells = tuple(dict.fromkeys(cell for cell in cells if cell is not None))
        if len(cells) == 1:
            operand = cells[0]
        elif cells:
            ordinal = book.page(CELL_SET).scope_row_count(source_scope) + 1
            operand = book.post(
                CELL_SET, (source_scope, ordinal),
                tuple(cell.key for cell in cells),
                stage=CONTROL_PROGRAM_BUILD, provenance=Derived(cells),
                mode=Mode.CONCORD,
            )
        else:
            raise ConcordanceRefusal(
                f"string glyph vocabulary under {source_scope!r} names no "
                "operand cell"
            )
        cell = book.post(
            STRING_GLYPH_VOCABULARY, (source_scope, NEW), tuple(rows),
            stage=CONTROL_PROGRAM_BUILD,
            provenance=Novel(STRING_GLYPH_VOCABULARY_MINT, (operand,)),
            mode=Mode.CONCORD,
        )
        return int(cell.row[1]), cell

    declarations = []
    initializations = []
    mutations = []
    entry_page = book.page(STRING_TABLE_ENTRY)
    grouped = {}
    for recipe_cell, component in recipes:
        mode = "repr" if int(component.conversion) == ord("r") else "raw"
        transform = component.projection_transform
        if transform not in {None, "strip"}:
            raise ValueError(
                "unknown string-token glyph transform: "
                f"value={int(component.operand_value_id)} "
                f"transform={transform!r}"
            )
        token_value_id = int(
            component.projection_source_value_id
            if component.projection_source_value_id is not None
            else component.operand_value_id
        )
        if transform == "strip" and component.projection_source_value_id is None:
            raise ValueError(
                "str.strip glyph projection has no exact source token: "
                f"value={int(component.operand_value_id)}"
            )
        grouped.setdefault((mode, transform), []).append((
            recipe_cell, component, int(token_value_id),
        ))
    for (mode, transform), mode_recipes in grouped.items():
        vocabulary_rows = []
        entry_cells = []
        for row in sorted(entry_page.rows(), key=lambda item: int(item[0])):
            entry_cell = book.latest_ref(STRING_TABLE_ENTRY, tuple(row))
            if entry_cell is None:
                raise ValueError(
                    f"StringTable entry has no concordance cell: {row!r}"
                )
            token = int(row[0])
            word = entry_page.cell(entry_cell.row, entry_cell.column)
            if not isinstance(word, str):
                raise TypeError(
                    f"StringTable concordance entry is not text: {row!r}"
                )
            entry_cells.append(entry_cell)
            projected_word = word.strip() if transform == "strip" else word
            glyphs = (
                tuple(quoted_string_glyph_tokens(projected_word))
                if mode == "repr"
                else tuple(
                    string_token(character) for character in projected_word
                )
            )
            vocabulary_rows.append((int(token), -1, len(glyphs)))
            vocabulary_rows.extend(
                (int(token), int(ordinal), int(glyph))
                for ordinal, glyph in enumerate(glyphs)
            )
        # The vocabulary is read from every StringTable entry this compile
        # interned and built for the glyph recipes that project a token in
        # this mode: minted from both, so a compile that interned no word
        # still names the recipes the (empty) vocabulary serves.
        vocabulary_id, vocabulary_cell = mint_vocabulary(vocabulary_rows, (
            *entry_cells,
            *(
                recipe_cell
                for recipe_cell, _component, _token_value_id
                in mode_recipes
            ),
        ))
        declarations.append((vocabulary_id, "unique", 3, False))
        initializations.append((
            vocabulary_id,
            f"literal_table={tuple(vocabulary_rows)!r}", 3,
        ))
        mutations.extend(
            ControlSequenceMutation(
                int(component.component_value_id),
                "glyph_token_lookup_copy",
                (int(token_value_id), vocabulary_id),
                int(component.component_value_id),
                policy="duplicates",
                argument_kind=(
                    "string_table_strip_glyph_projection"
                    if transform == "strip"
                    else "string_table_glyph_projection"
                ),
                source_contract_cell=recipe_cell,
                vocabulary_contract_cell=vocabulary_cell,
            )
            for recipe_cell, component, token_value_id in mode_recipes
        )
    enum_page = book.page(SOURCE_ENUM_GLYPH_TABLE)
    enum_grouped = {}
    for recipe_cell, component in enum_recipes:
        if not component.record_identity:
            raise ValueError(
                "Enum name glyph projection has no declared Enum identity"
            )
        enum_cell = component.enum_table_cell
        if enum_cell is None:
            raise ValueError(
                "Enum name glyph projection has no exact source table cell: "
                f"{component.record_identity!r}"
            )
        # Compilation-unit books cross the spill/pickle boundary before this
        # reader runs.  A Ref's Page is a rehydrated schema value there, not
        # necessarily the registry singleton.  Resolve the retained row
        # through the active book and require the complete cell key to agree;
        # the same rule guards rehydrated STRING_GLYPH_VOCABULARY refs during
        # SSA lowering.  Unit 041 carried the exact RowFieldKind table cell,
        # but an ``is`` test rejected its reconstructed Page object.
        canonical_enum_cell = book.latest_ref(
            SOURCE_ENUM_GLYPH_TABLE, tuple(enum_cell.row),
        )
        if (
            canonical_enum_cell is None
            or enum_cell.key != canonical_enum_cell.key
        ):
            raise ValueError(
                "Enum name glyph projection names a non-Enum source cell: "
                f"{enum_cell.key!r}"
            )
        mode = "repr" if int(component.conversion) == ord("r") else "raw"
        enum_grouped.setdefault((mode, canonical_enum_cell), []).append((
            recipe_cell, component,
        ))
    for (mode, enum_cell), mode_recipes in (
        enum_grouped.items()
    ):
        enum_fact = enum_page.cell(enum_cell.row, enum_cell.column)
        if not isinstance(enum_fact, SourceEnumGlyphTable):
            raise TypeError(f"invalid source Enum glyph table {enum_fact!r}")
        enum_identities = {
            str(component.record_identity)
            for _recipe_cell, component in mode_recipes
        }
        if enum_identities != {str(enum_fact.enum_identity)}:
            raise ValueError(
                "Enum name glyph projection disagrees with its exact source "
                f"table: component={tuple(sorted(enum_identities))!r} "
                f"table={enum_fact.enum_identity!r}"
            )
        vocabulary_rows = []
        for member in enum_fact.members:
            member_name = str(member.member_identity)
            glyphs = (
                tuple(quoted_string_glyph_tokens(member_name))
                if mode == "repr"
                else tuple(string_token(character) for character in member_name)
            )
            vocabulary_rows.append((
                int(member.runtime_value), -1, len(glyphs),
            ))
            vocabulary_rows.extend(
                (
                    int(member.runtime_value), int(ordinal), int(glyph),
                )
                for ordinal, glyph in enumerate(glyphs)
            )
        vocabulary_id, vocabulary_cell = mint_vocabulary(
            vocabulary_rows, (enum_cell,),
        )
        declarations.append((vocabulary_id, "unique", 3, False))
        initializations.append((
            vocabulary_id,
            f"literal_table={tuple(vocabulary_rows)!r}", 3,
        ))
        mutations.extend(
            ControlSequenceMutation(
                int(component.component_value_id),
                "glyph_token_lookup_copy",
                (int(component.operand_value_id), vocabulary_id),
                int(component.component_value_id),
                policy="duplicates",
                argument_kind="enum_name_glyph_projection",
                source_contract_cell=recipe_cell,
                vocabulary_contract_cell=vocabulary_cell,
            )
            for recipe_cell, component in mode_recipes
        )
    return tuple(declarations), tuple(initializations), tuple(mutations)


def _source_optional_glyph_materializations(graph_obj: Any):
    """Declare payload arenas only for exact optional-glyph carry cells."""

    from .concordance_declarations import (
        SOURCE_OPTIONAL_GLYPH_CARRY,
        SourceOptionalGlyphCarry,
    )
    from .identity_concordance import current_identity_book

    book = current_identity_book()
    declarations = []
    initializations = []
    cells = []
    nodes_by_value = {
        int(data.get("value_id", node_id)): (int(node_id), data)
        for node_id, data in graph_obj.nodes(data=True)
    }
    page = book.page(SOURCE_OPTIONAL_GLYPH_CARRY)
    for row in _source_contract_rows(graph_obj, page):
        if len(row) != 2 or int(row[1]) not in nodes_by_value:
            continue
        node_id, data = nodes_by_value[int(row[1])]
        cell = book.latest_ref(SOURCE_OPTIONAL_GLYPH_CARRY, tuple(row))
        if cell is None:
            raise ValueError(
                "optional glyph graph view has no concordance source: "
                f"value={int(data.get('value_id', node_id))} row={row!r}"
            )
        fact = book.page(SOURCE_OPTIONAL_GLYPH_CARRY).cell(
            cell.row, cell.column,
        )
        if not isinstance(fact, SourceOptionalGlyphCarry):
            raise TypeError(
                f"source optional glyph carry has invalid fact {fact!r}"
            )
        payload_id = int(fact.payload_sequence_value_id)
        if payload_id != int(data.get("value_id", node_id)):
            raise ValueError(
                "optional glyph payload disagrees with its graph identity: "
                f"payload={payload_id} node={data.get('value_id', node_id)}"
            )
        declarations.append((payload_id, "duplicates", 1, True))
        initializations.append((payload_id, "duplicates", 1))
        cells.append(cell)
    return (
        tuple(dict.fromkeys(declarations)),
        tuple(dict.fromkeys(initializations)),
        tuple(dict.fromkeys(cells)),
    )


def _loop_carried_initial_ids(graph_obj) -> frozenset[int]:
    """Entry-arm identities of every loop's carried bindings.

    The reducer records ``loop_carried_bindings[name] = (initial, updated)``
    on each loop.  Inside the loop the graph names a carried value by its
    INITIAL producer; that producer (often a literal) is the entry arm of the
    loop's Phi, not an invariant.  Every consumer that would read such a node
    as a constant reads this record instead of the literal.
    """

    return frozenset(
        int(initial)
        for _loop_id, data in graph_obj.nodes(data=True)
        for initial, _updated in (
            (data.get("attributes") or {}).get(
                "loop_carried_bindings", {}
            ) or {}
        ).values()
    )


def _loop_carried_storage_aliases(graph_obj) -> dict[int, int]:
    """Graph-derived storage aliases for loop-carried in-place mutation.

    A loop that mutates an array through ``IndexedStore`` leaves a
    LOOPRESULT node as the array's post-loop identity. That identity is
    the SAME STORAGE as the array it mutated -- in-place is the point --
    but the planner schedules later consumers against the loopresult id,
    and a lowering that cannot link it materializes a fresh, unconnected
    FORMAL: the second of two sequential loops over one array then reads
    the original buffer and the first loop's stores silently vanish (the
    sequential same-array store defect, pinned in
    ``test_compiled_linalg.py``).

    The chase is deliberately narrow, one level per node, composed by the
    builder's existing alias-chain walk (``external_value``):

    * a ``loopresult``/``loopexit`` node aliases its preferred-role parent
      ONLY when that parent is itself an ``IndexedStore`` version or
      another loop identity -- scalar carried results (a ``total``) are
      genuine values with carried-port machinery of their own and are
      never touched;
    * an ``IndexedStore`` node aliases its ``base`` -- the store versions
      resident memory, it does not mint storage (the same rule
      ``ir_indexing`` applies inside a function, extended to the graph
      so it holds ACROSS loops).
    """

    aliases: dict[int, int] = {}
    chainable = {"indexedstore", "loopresult", "loopexit"}
    store_versions: set[int] = set()
    loop_versions: dict[int, int] = {}

    def node_kind(node_id) -> str:
        data = graph_obj.nodes.get(node_id) or {}
        return str(data.get("op") or data.get("type") or "").casefold()

    def value_id_of(node_id) -> int:
        data = graph_obj.nodes.get(node_id) or {}
        return int(data.get("value_id", node_id))

    for node_id, data in graph_obj.nodes(data=True):
        kind = str(data.get("op") or data.get("type") or "").casefold()
        parents = tuple(data.get("parents") or ())
        if kind == "indexedstore":
            store_versions.add(value_id_of(node_id))
            base = next(
                (parent for parent, role in parents
                 if str(role) == "base" and parent in graph_obj),
                None,
            )
            if base is not None and node_kind(base) == "indexed":
                # ``g[i][1] = v`` writes a row of the list ``g``: its version
                # is the outer list's storage, not the row view's.
                outer = next(
                    (parent for parent, role in graph_obj.nodes[base].get("parents") or ()
                     if str(role) == "base" and parent in graph_obj
                     and ((graph_obj.nodes[parent].get("attributes") or {}).get(
                         "aggregate_kind") == "list"
                         or (graph_obj.nodes[parent].get("attributes") or {}).get(
                             "row_store_of_list"))),
                    None,
                )
                if outer is not None:
                    base = outer
            if base is not None:
                aliases[value_id_of(node_id)] = value_id_of(base)
            continue
        if kind in {"loopresult", "loopexit"}:
            for preferred_role in (
                "updated", "value", "body", "initial", "orelse"
            ):
                parent = next(
                    (parent for parent, role in parents
                     if str(role) == preferred_role
                     and parent in graph_obj),
                    None,
                )
                if parent is None:
                    continue
                if node_kind(parent) in chainable:
                    loop_versions[value_id_of(node_id)] = value_id_of(parent)
                break
    # second=total makes LoopResult(second) point at LoopResult(total), but
    # those scalar exit values have different seeds on a zero-trip path.
    # A loop identity alone does not prove shared storage. Propagate the
    # actual IndexedStore proof backwards through the preferred-parent graph
    # in O(V+E); only its proven cone belongs in this storage-alias ledger.
    from collections import deque
    dependents: dict[int, list[int]] = {}
    for source, target in loop_versions.items():
        dependents.setdefault(target, []).append(source)
    pending = deque(store_versions)
    proven = set(store_versions)
    while pending:
        for source in dependents.get(pending.popleft(), ()):
            if source not in proven:
                proven.add(source)
                pending.append(source)
    aliases.update({
        source: target for source, target in loop_versions.items()
        if source in proven
    })
    def root(value):
        seen = set()
        while value in aliases and value not in seen:
            seen.add(value)
            value = aliases[value]
        return value

    changed = True
    while changed:
        changed = False
        for node_id, data in graph_obj.nodes(data=True):
            # Program: ``d = {1: n, 2: 5}; for i in range(3): if i == n:
            # d[2] = i`` then ``d[2]`` (corpus c670).  Wrong record: the
            # loop's result names the body's ``if`` merge, which is no
            # store version, so the post-loop read of ``d`` was claimed by
            # no table and lowered as a positional row read (0 for 5).
            # Rule: a loop result whose carried value is a merge already
            # proven to be one dict's storage is that dict's storage.
            if node_kind(node_id) in {"loopresult", "loopexit"}:
                carried = next((
                    parent for preferred_role in ("updated", "value")
                    for parent, role in data.get("parents") or ()
                    if str(role) == preferred_role and parent in graph_obj
                ), None)
                if carried is None or node_kind(carried) != "phi":
                    continue
                resident = root(value_id_of(carried))
                if (graph_obj.nodes.get(resident, {}).get("attributes") or {}).get("aggregate_kind") != "dict":
                    continue
                value = value_id_of(node_id)
                if value != resident and aliases.get(value) != resident:
                    aliases[value] = resident
                    changed = True
                continue
            if node_kind(node_id) != "phi":
                continue
            parents = [root(value_id_of(parent)) for parent, role in data.get("parents") or ()
                       if role in {"body", "orelse"} and parent in graph_obj]
            if len(parents) != 2 or parents[0] != parents[1]:
                continue
            resident = parents[0]
            if (graph_obj.nodes.get(resident, {}).get("attributes") or {}).get("aggregate_kind") != "dict":
                continue
            value = value_id_of(node_id)
            if value != resident and aliases.get(value) != resident:
                aliases[value] = resident
                changed = True
    aliases = {source: root(target) for source, target in aliases.items()}
    # Never allow a cycle to reach the builder's chase (it guards with a
    # seen-set, but a self-alias is meaningless regardless).
    return {
        source: target for source, target in aliases.items()
        if source != target
    }


def _promote_conditional_sequence_aliases(
    control: Any,
    graph_obj: Any,
    *,
    call_result_kinds: Mapping[int, str] | None = None,
    structural_aliases: Mapping[int, tuple[int, str]] | None = None,
):
    """Turn branch-carried aggregate versions into resident arena carries.

    Ordinary conditional planning records every authored assignment in the
    scalar-shaped ``carried_aliases`` tuple.  Once source/call analysis proves
    those values are sequences, retain one initial arena, copy the selected
    branch value into it, and correlate the post-branch SSA spelling with that
    arena.  This is an IR correction, not a receipt-only relabeling.
    """

    from .control_source import (
        CallBlock, ConditionalBlock, ControlSequenceMutation, LoopBlock,
        SequenceBlock, SequenceQueryBlock, WhileBlock,
    )

    copyable_sequence_kinds = {"list", "bytes", "bytearray"}
    # ``unique`` is the physical policy recorded by the sequence-contract
    # concordance for mapping/set storage. Late control planning may no
    # longer have the frontend's ``dict``/``set`` spelling, but it still has
    # that exact committed contract; consume the durable policy instead of
    # reconstructing a source kind from graph-node attributes.
    in_place_sequence_kinds = {"dict", "set", "unique"}
    sequence_kinds = copyable_sequence_kinds | in_place_sequence_kinds
    resident_by_value: dict[int, int] = {}
    kind_by_value: dict[int, str] = {}
    for node_id, data in graph_obj.nodes(data=True):
        attributes = data.get("attributes") or {}
        kind = str(attributes.get("aggregate_kind") or "")
        if kind not in sequence_kinds:
            continue
        value_id = int(data.get("value_id", node_id))
        resident_by_value[value_id] = value_id
        kind_by_value[value_id] = kind
    for value_id, kind in dict(call_result_kinds or {}).items():
        if str(kind) in sequence_kinds:
            resident_by_value[int(value_id)] = int(value_id)
            kind_by_value[int(value_id)] = str(kind)
    for value_id, (resident_id, kind) in dict(
        structural_aliases or {}
    ).items():
        if str(kind) not in sequence_kinds:
            continue
        resident_by_value[int(value_id)] = int(resident_id)
        kind_by_value[int(value_id)] = str(kind)
        kind_by_value.setdefault(int(resident_id), str(kind))

    # In-place stores and their Phi/loop-result spellings are storage
    # versions, not scalar values.  Resolve them to their original arena before
    # inspecting conditional carries, otherwise the branch becomes an opaque
    # scalar Phi and every later row projection leaks into the public ABI.
    storage_aliases = _loop_carried_storage_aliases(graph_obj)
    changed = True
    while changed:
        changed = False
        for alias_id, source_id in storage_aliases.items():
            resident_id = resident_by_value.get(int(source_id))
            kind = kind_by_value.get(int(source_id))
            if resident_id is None or kind not in sequence_kinds:
                continue
            if resident_by_value.get(int(alias_id)) != int(resident_id):
                resident_by_value[int(alias_id)] = int(resident_id)
                kind_by_value[int(alias_id)] = str(kind)
                changed = True
        for node_id, data in graph_obj.nodes(data=True):
            if str(data.get("op") or data.get("type") or "").casefold() not in {
                "phi", "loopresult", "loopexit",
            }:
                continue
            value_id = int(data.get("value_id", node_id))
            resolved_parents = {
                (
                    int(resident_by_value[parent_value]),
                    str(kind_by_value[parent_value]),
                )
                for parent, role in data.get("parents") or ()
                if str(role) not in {"control", "test"}
                and parent in graph_obj
                for parent_value in (
                    int(graph_obj.nodes[parent].get("value_id", parent)),
                )
                if parent_value in resident_by_value
                and parent_value in kind_by_value
            }
            if len(resolved_parents) != 1:
                continue
            resident_id, kind = next(iter(resolved_parents))
            if resident_by_value.get(value_id) != resident_id:
                resident_by_value[value_id] = resident_id
                kind_by_value[value_id] = kind
                changed = True

    promoted_aliases: dict[int, tuple[int, str]] = {}
    destination_ids: set[int] = set()
    rebirth_ports: dict[int, int] = {}

    node_by_value = {
        int(data.get("value_id", node_id)): int(node_id)
        for node_id, data in graph_obj.nodes(data=True)
    }

    def refuse_shared_branch_rebinding(initial_id, arm_ids, merged_id):
        merged = graph_obj.nodes.get(node_by_value.get(int(merged_id)), {})
        name = (merged.get("attributes") or {}).get("binding_name")
        if name is None:
            return
        identities = dict(graph_obj.graph.get("identity_table") or {})

        def holders(value_id):
            return sorted(
                str(other) for other, history in identities.items()
                if str(other) != str(name)
                and int(value_id) in set(map(int, history))
            ) + sorted(
                f"node {int(reader)}"
                for reader, reader_data in graph_obj.nodes(data=True)
                for parent, role in reader_data.get("parents") or ()
                if int(parent) == node_by_value.get(int(value_id))
                and str(role).startswith("elts")
            )

        shared = holders(initial_id)
        if not shared:
            from .glsl_deployment_strategy import _mutated_sequence_node_ids

            changed = _mutated_sequence_node_ids(
                graph_obj, include_merge_aliases=False,
            )
            if node_by_value.get(int(merged_id)) in changed:
                shared = [
                    holder for arm_id in arm_ids
                    if int(arm_id) != int(initial_id)
                    for holder in holders(arm_id)
                ]
        if shared:
            raise ValueError(
                f"list {str(name)!r} is rebound on a branch while "
                f"{', '.join(dict.fromkeys(shared))} also holds it; a list "
                "rebound on a branch is carried by copying into its initial "
                "storage, which only one name may own"
            )

    def promote(block):
        if isinstance(block, ConditionalBlock):
            scalar = []
            # ``carried_field_cells`` is index-aligned with
            # ``carried_aliases`` (the control builder zips them).  A
            # promoted alias leaves ``carried_aliases``, so its field cells
            # leave with it, in lockstep (as the retained-values projection
            # in control_source does).  Dropping only the alias shifted every
            # later alias onto its predecessor's cells: ``rejected = False``
            # after ``m.unresolved_report = list(lines)`` was lowered as the
            # field merge of ``unresolved_report`` (dt_controller
            # step_with_dt_control_used; ConcordanceRefusal on
            # ssa_field_version in ``_carried_field_arm``).
            scalar_field_cells = []
            field_cells = tuple(block.carried_field_cells) + (None,) * (
                len(block.carried_aliases) - len(block.carried_field_cells)
            )
            sequences = list(block.carried_sequence_aliases)
            for carried, cells in zip(block.carried_aliases, field_cells):
                true_id, false_id, initial_id, merged_id = map(int, carried)
                initial_resident = resident_by_value.get(initial_id)
                initial_kind = kind_by_value.get(initial_id)
                true_resident = resident_by_value.get(true_id)
                false_resident = resident_by_value.get(false_id)
                true_kind = kind_by_value.get(true_id)
                false_kind = kind_by_value.get(false_id)
                if (
                    initial_resident is None
                    or initial_kind not in sequence_kinds
                    or true_resident is None
                    or false_resident is None
                    or true_kind != initial_kind
                    or false_kind != initial_kind
                    # Mapping/set residents carry multi-column table state.
                    # They may leave the scalar Phi only when both branches
                    # already name the same in-place storage; unlike a list
                    # value, they cannot be represented by the generic
                    # single-column ``replace`` mutation below.
                    or (
                        initial_kind in in_place_sequence_kinds
                        and not (
                            int(true_resident) == int(initial_resident)
                            and int(false_resident) == int(initial_resident)
                        )
                    )
                ):
                    scalar.append(carried)
                    scalar_field_cells.append(cells)
                    continue
                # Program: ``rep = [9]; keep = rep; if k > 2: rep = [k, 1,
                # 2]`` then ``len(keep)`` (scratch c12).  The copy overwrites
                # the initial list, which is the binding's own storage only
                # while no other name or container holds it (``keep`` must
                # still read ``[9]``), as for a list rebound in a loop.  An
                # arm another name holds is copied, not shared, so a later
                # change through the merge would not reach that name.
                if initial_kind in copyable_sequence_kinds:
                    refuse_shared_branch_rebinding(
                        initial_id, (true_id, false_id), merged_id,
                    )
                destination = int(initial_resident)
                sequences.append((
                    int(true_resident), int(false_resident),
                    destination, merged_id,
                ))
                destination_ids.add(destination)
                resident_by_value[merged_id] = destination
                kind_by_value[merged_id] = initial_kind
                promoted_aliases[merged_id] = (destination, initial_kind)
            # Program: ``if n > 2: t = [n, 1, 2] else: t = [7, n]`` then
            # ``len(t)``, ``total(t)`` (corpus c654).  Wrong record: a list
            # first bound in both arms, to displays of different lengths,
            # published a scalar result merge of two handles and no member
            # layout, so the merge had no birth (it was refused by name as
            # "different shapes").  Python's list length is dynamic; that is
            # no reason to refuse.  Rule: the merge is its own storage and
            # each arm copies its display into it (the ``replace`` a
            # carried sequence alias lowers), so its length is the taken
            # arm's and its capacity the longest arm's (the arms' own
            # bounds, surveyed as a copy source).
            results = []
            handles = list(block.sequence_handle_results)
            for result in block.result_aliases:
                true_id, false_id, merged_id = map(int, result)
                if not _list_merged_from_branches(
                    graph_obj, node_by_value.get(merged_id, -1),
                ):
                    results.append(result)
                    continue
                true_resident = resident_by_value.get(true_id)
                false_resident = resident_by_value.get(false_id)
                if true_resident is None or false_resident is None:
                    results.append(result)
                    continue
                # Program: ``a = [n, 1]; t = a if n > 2 else [7, n, 2];
                # return t`` (c743).  The returned merge IS the function's
                # result list, whose storage the caller passes in.  When
                # every arm is a private display of this function, both die
                # at the return, so the taken arm's rows copied into the
                # result are Python's object exactly.  Any other arm
                # (c787-c789: a parameter list) is the caller's object, so
                # the merge stays a handle merge and the function returns
                # its arm predicate; the caller selects among its actuals.
                node_id = node_by_value.get(merged_id, -1)
                if (
                    _returned_list_merge(graph_obj, node_id)
                    and _arms_private_displays(graph_obj, node_id)
                ):
                    destination = merged_id
                    sequences.append((
                        int(true_resident), int(false_resident),
                        destination, merged_id,
                    ))
                    destination_ids.add(destination)
                    continue
                # Program: ``a = [n, 1]; if n > 2: t = a else: t = [7, n,
                # 2]`` then ``a[0] = a[0] + 5; return t[0] + len(t)``
                # (c737).  Wrong record: the merge was its own storage,
                # each arm copied into it, so a change through ``a`` after
                # the branch never reached ``t`` (silent), though Python's
                # ``t`` IS ``a`` on one path.  Rule: the merge owns no
                # rows.  It is a handle: the alias keeps the arms' resident
                # storage (not a version spelling) and lowering selects the
                # taken arm's descriptor members with one Phi each
                # (``_lower_sequence_handle_merge``).  The component
                # {arms, merge} grows together (``_settle_linked_dynamic_
                # sequences``).
                results.append((
                    int(true_resident), int(false_resident), merged_id,
                ))
                handles.append(merged_id)
            return replace(
                block,
                body=promote(block.body),
                orelse=(
                    None if block.orelse is None else promote(block.orelse)
                ),
                result_aliases=tuple(results),
                sequence_handle_results=tuple(handles),
                carried_aliases=tuple(scalar),
                carried_field_cells=tuple(scalar_field_cells),
                carried_sequence_aliases=tuple(sequences),
            )
        if isinstance(block, SequenceBlock):
            return replace(block, blocks=tuple(promote(x) for x in block.blocks))
        if isinstance(block, LoopBlock):
            promoted_body = promote(block.body)
            scalar_carries = []
            sequence_updates: set[tuple[int, int]] = set()
            rebirths = []
            port_by_carry = {
                (int(updated), int(initial)): int(port)
                for port, initial, updated in block.result_ports
            }
            loop_node = (
                graph_obj.nodes.get(int(block.source_loop_node_id), {})
                if block.source_loop_node_id is not None else {}
            )
            carried_names = {
                (int(updated), int(initial)): str(name)
                for name, (initial, updated) in dict(
                    (loop_node.get("attributes") or {}).get(
                        "loop_carried_bindings"
                    ) or {}
                ).items()
            }
            identities = dict(graph_obj.graph.get("identity_table") or {})
            for updated_id, initial_id in block.carried_aliases:
                updated_id = int(updated_id)
                initial_id = int(initial_id)
                updated_resident = resident_by_value.get(updated_id)
                initial_resident = resident_by_value.get(initial_id)
                # Program: ``xs = [0, 0]; for i in range(4): t += xs[0];
                # xs = [i + n, i]`` then ``len(xs)`` (corpus c246).  Wrong
                # record: the rebound list kept a scalar carry, so only its
                # handle went through the loop Phi and the exit list's length
                # was never written (``len`` read 0, silently).  Rule: a list
                # the body rebinds to another list is carried like a branch
                # rebinding -- the body's list is copied into the initial
                # arena at the end of each iteration (one ``replace``).
                if (
                    updated_resident is not None
                    and initial_resident is not None
                    and updated_resident != initial_resident
                    and kind_by_value.get(updated_id) in copyable_sequence_kinds
                    and kind_by_value.get(initial_id)
                    == kind_by_value.get(updated_id)
                    and (updated_id, initial_id) in carried_names
                    and (updated_id, initial_id) in port_by_carry
                ):
                    # The copy overwrites the initial list.  That is the
                    # binding's own storage only while no other name or
                    # container holds either list: ``ys = xs`` before the
                    # loop must still read ``[7, 8]`` afterwards.
                    name = carried_names[(updated_id, initial_id)]
                    holders = sorted(
                        str(other)
                        for other, history in identities.items()
                        if str(other) != name
                        and {initial_id, updated_id} & set(map(int, history))
                    ) + sorted(
                        f"node {int(reader)}"
                        for reader, reader_data in graph_obj.nodes(data=True)
                        for parent, role in reader_data.get("parents") or ()
                        if int(parent) in {initial_id, updated_id}
                        and str(role).startswith("elts")
                    )
                    if holders:
                        raise ValueError(
                            f"list {name!r} is rebound inside a retained "
                            f"loop while {', '.join(holders)} also holds "
                            "it; a rebound loop list is carried by copying "
                            "into its initial storage, which only one name "
                            "may own"
                        )
                    rebirths.append(ControlSequenceMutation(
                        int(initial_resident), "replace",
                        (int(updated_resident),),
                        port_by_carry[(updated_id, initial_id)],
                        policy="duplicates",
                    ))
                    sequence_updates.add((updated_id, initial_id))
                    destination_ids.add(int(initial_resident))
                    continue
                if (
                    updated_resident is None
                    or initial_resident is None
                    or updated_resident != initial_resident
                    or kind_by_value.get(updated_id) not in sequence_kinds
                    or kind_by_value.get(initial_id) not in sequence_kinds
                ):
                    scalar_carries.append((updated_id, initial_id))
                    continue
                sequence_updates.add((updated_id, initial_id))
                kind = str(kind_by_value[initial_id])
                promoted_aliases[updated_id] = (
                    int(initial_resident), kind,
                )
                destination_ids.add(int(initial_resident))
            scalar_ports = []
            for port_id, initial_id, updated_id in block.result_ports:
                key = (int(updated_id), int(initial_id))
                if key not in sequence_updates:
                    scalar_ports.append((
                        int(port_id), int(initial_id), int(updated_id)
                    ))
                    continue
                resident_id = int(resident_by_value[int(initial_id)])
                kind = str(kind_by_value[int(initial_id)])
                resident_by_value[int(port_id)] = resident_id
                kind_by_value[int(port_id)] = kind
                promoted_aliases[int(port_id)] = (resident_id, kind)
                if any(
                    int(mutation.effect_node_id) == int(port_id)
                    for mutation in rebirths
                ):
                    rebirth_ports[int(port_id)] = resident_id
            return replace(
                block,
                body=promoted_body,
                carried_aliases=tuple(scalar_carries),
                result_ports=tuple(scalar_ports),
                sequence_mutations=(*block.sequence_mutations, *rebirths),
            )
        if isinstance(block, WhileBlock):
            return replace(
                block,
                condition=promote(block.condition),
                body=promote(block.body),
            )
        if isinstance(block, CallBlock):
            return replace(block, callee=promote(block.callee))
        return block

    promoted = replace(control, root=promote(control.root))
    if rebirth_ports:
        # A query of a rebound loop list's exit port (``len(xs)`` after the
        # loop) reads the initial arena the body's list was copied into;
        # the port has no storage of its own.
        def retarget(item):
            if isinstance(item, SequenceQueryBlock):
                resident = rebirth_ports.get(int(item.sequence_value_id))
                return item if resident is None else replace(
                    item, sequence_value_id=int(resident),
                )
            if isinstance(item, tuple):
                entries = tuple(retarget(entry) for entry in item)
                return item if all(
                    new is old for new, old in zip(entries, item)
                ) else entries
            if (dataclasses.is_dataclass(item) and not isinstance(item, type)
                    and type(item).__module__ == ConditionalBlock.__module__):
                changes = {}
                for field in dataclasses.fields(item):
                    value = getattr(item, field.name)
                    if isinstance(value, tuple) or dataclasses.is_dataclass(value):
                        retargeted = retarget(value)
                        if retargeted is not value:
                            changes[field.name] = retargeted
                return replace(item, **changes) if changes else item
            return item

        promoted = replace(promoted, root=retarget(promoted.root))
    return promoted, promoted_aliases, tuple(sorted(destination_ids))


def _retarget_forwarded_sequence_effects(
    control: Any, forwarded: Mapping[int, tuple[int, str]],
) -> Any:
    """Name the argument's storage in every effect on a forwarded call result.

    Rule 8: a call whose result is its argument's own list owns no storage,
    so an effect on the result (``t.append(5)``, ``len(t)``) is an effect on
    the argument's sequence.  The retained mutation records were written
    before the call's callee was specialized for its constant actuals
    (``pick(a, 5)`` returns ``a`` only in that copy), so the control program
    may still name the result: this retargets those effects onto the one
    owner, the actual's sequence, once the specialized forwarding is proven.
    """

    from .control_source import (
        ConditionalBlock, ControlSequenceMutation, SequenceQueryBlock,
    )

    resident = {
        int(value_id): int(target)
        for value_id, (target, _kind) in dict(forwarded).items()
        if int(value_id) != int(target)
    }
    if not resident:
        return control

    def retarget(item):
        if isinstance(item, (ControlSequenceMutation, SequenceQueryBlock)):
            target = resident.get(int(item.sequence_value_id))
            if target is not None:
                item = replace(item, sequence_value_id=target)
            if isinstance(item, SequenceQueryBlock):
                return item
        if isinstance(item, tuple):
            entries = tuple(retarget(entry) for entry in item)
            return item if all(
                new is old for new, old in zip(entries, item)
            ) else entries
        if (dataclasses.is_dataclass(item) and not isinstance(item, type)
                and type(item).__module__ == ConditionalBlock.__module__):
            changes = {}
            for field in dataclasses.fields(item):
                value = getattr(item, field.name)
                if isinstance(value, tuple) or dataclasses.is_dataclass(value):
                    retargeted = retarget(value)
                    if retargeted is not value:
                        changes[field.name] = retargeted
            return replace(item, **changes) if changes else item
        return item

    return replace(control, root=retarget(control.root))


def _constant_struct_pack_materializations(graph_obj: Any):
    """Fold fully source-static ``struct.pack`` calls to byte-sequence facts."""

    import struct

    materializations = []
    for node_id, data in sorted(graph_obj.nodes(data=True)):
        attributes = data.get("attributes") or {}
        if attributes.get("extraction_identity") != "_struct.pack":
            continue
        arguments = tuple(
            int(parent)
            for parent, role in sorted(
                data.get("parents") or (),
                key=lambda item: int(str(item[1]).split(":", 1)[1])
                if str(item[1]).startswith("arg:") else 1 << 30,
            )
            if str(role).startswith("arg:") and int(parent) in graph_obj
        )
        values = []
        for argument in arguments:
            argument_data = graph_obj.nodes[argument]
            argument_attributes = argument_data.get("attributes") or {}
            if "value" in argument_attributes:
                values.append(argument_attributes["value"])
            elif argument_data.get("constant") is not None:
                values.append(argument_data["constant"])
            else:
                break
        if len(values) != len(arguments) or not values:
            continue
        try:
            payload = struct.pack(*values)
        except (struct.error, TypeError, ValueError):
            continue
        value_id = int(data.get("value_id", node_id))
        updated = dict(attributes)
        updated.update({
            "producer_kind": "constant_sequence",
            "aggregate_kind": "bytes",
            "sequence_key_columns": (),
            "sequence_column_count": 1,
            "sequence_writable": False,
            "constant_bytes": bytes(payload),
        })
        data["attributes"] = updated
        materializations.append((
            value_id,
            bytes(payload),
            int(node_id),
            "_struct.pack",
        ))
    return tuple(materializations)


def _constant_byte_literal_materializations(graph_obj: Any):
    """Publish authored byte literals through the resident sequence ABI."""

    materializations = []
    for node_id, data in sorted(graph_obj.nodes(data=True)):
        attributes = data.get("attributes") or {}
        literal = attributes.get("value", data.get("constant"))
        if not isinstance(literal, bytes):
            continue
        value_id = int(data.get("value_id", node_id))
        updated = dict(attributes)
        updated.update({
            "producer_kind": "constant_sequence",
            "aggregate_kind": "bytes",
            "sequence_key_columns": (),
            "sequence_column_count": 1,
            "sequence_writable": False,
            "constant_bytes": bytes(literal),
        })
        data["attributes"] = updated
        materializations.append((
            value_id, bytes(literal), int(node_id), None,
        ))
    return tuple(materializations)


def _source_sequence_snapshots(graph_obj: Any):
    """Sequence conversion captures contents and length in separate storage."""
    from .control_source import ControlSequenceMutation
    mutations = []
    for node_id, data in graph_obj.nodes(data=True):
        attrs = data.get("attributes") or {}
        if (attrs.get("aggregate_kind") not in {"tuple", "list"}
                or attrs.get("producer_kind") != "aggregate_materialization"):
            continue
        sources = [int(parent) for parent, role in data.get("parents") or ()
                   if str(role) == "arg:0"]
        if len(sources) != 1:
            continue
        source = sources[0]
        if (graph_obj.nodes.get(source, {}).get("attributes") or {}).get("aggregate_kind") not in {"list", "tuple"}:
            continue
        mutations.append(ControlSequenceMutation(
            int(node_id), "replace", (source,), int(node_id), policy="duplicates",
        ))
    return tuple(mutations)


def _source_sequence_slices(graph_obj: Any):
    """A unit-stride list slice is a new sequence filled from its source.

    Program: ``xs = [3, n, 4]; ys = xs[:2]; return ys[1]``.  Wrong record:
    the slice node declared its own arena but nothing wrote it, so ``ys[1]``
    read an unwritten row (0.0 for n).  Rule: ``dst = src[lower:upper]`` is
    the mutation ``slice``: reset ``dst``, then append the clipped rows of
    ``src``.  ``argument_kind`` is ``slice:<lower>:<upper>``, each bound
    empty (absent), ``v<id>`` (a value, also listed in the arguments) or
    ``c<int>`` (a literal, as a folded ``slice`` selector carries it).  A
    stepped or non-integer slice is not admitted here and stays loud.
    """
    from .control_source import ControlSequenceMutation

    def value_id(node):
        return int(graph_obj.nodes[node].get("value_id", node))

    def literal_of(node):
        data = graph_obj.nodes[node]
        if str(data.get("type") or data.get("op")) not in {"Constant", "const"}:
            return False, None
        return True, (data.get("attributes") or {}).get("value")

    def integer_or_none(value):
        return value is None or (isinstance(value, int) and not isinstance(value, bool))

    mutations = []
    storage_aliases = _loop_carried_storage_aliases(graph_obj)
    nodes_by_value: dict[int, list[int]] = {}
    for node_id, data in graph_obj.nodes(data=True):
        nodes_by_value.setdefault(
            int(data.get("value_id", node_id)), [],
        ).append(int(node_id))

    for node_id, data in graph_obj.nodes(data=True):
        attrs = data.get("attributes") or {}
        if (attrs.get("aggregate_kind") != "list"
                or attrs.get("producer_kind") != "sequence_slice"):
            continue
        roles = {str(role): int(parent) for parent, role in data.get("parents") or ()}
        source, selector = roles.get("base"), roles.get("index")
        if source is None or selector is None or source not in graph_obj or selector not in graph_obj:
            continue
        source_value_id = value_id(source)
        storage_value_id = source_value_id
        seen_storage_ids = set()
        cyclic_storage_alias = False
        while storage_value_id in storage_aliases:
            if storage_value_id in seen_storage_ids:
                cyclic_storage_alias = True
                break
            seen_storage_ids.add(storage_value_id)
            storage_value_id = int(storage_aliases[storage_value_id])
        if cyclic_storage_alias:
            continue
        storage_nodes = nodes_by_value.get(storage_value_id, ())
        if len(storage_nodes) != 1:
            continue
        storage_node = graph_obj.nodes[storage_nodes[0]]
        if (storage_node.get("attributes") or {}).get("aggregate_kind") != "list":
            continue
        selector_data = graph_obj.nodes[selector]
        folded, selector_literal = literal_of(selector)
        # A slice after an in-place loop update can name an IndexedStore or
        # LoopResult as its base.  The exact graph storage-alias chain maps
        # that version to the resident sequence descriptor; use that identity
        # for the slice operation so its row type and storage come from the
        # same existing sequence contract.
        arguments, spellings = [storage_value_id], []
        if folded:
            if not (isinstance(selector_literal, slice) and selector_literal.step is None
                    and integer_or_none(selector_literal.start)
                    and integer_or_none(selector_literal.stop)):
                continue
            spellings = ["" if bound is None else f"c{int(bound)}"
                         for bound in (selector_literal.start, selector_literal.stop)]
        else:
            if str(selector_data.get("type") or selector_data.get("op")) != "Slice":
                continue
            bounds = {str(role): int(parent) for parent, role in selector_data.get("parents") or ()}
            if set(bounds) - {"lower", "upper", "step"}:
                continue
            if "step" in bounds and literal_of(bounds["step"]) != (True, None):
                continue
            for role in ("lower", "upper"):
                if role not in bounds:
                    spellings.append("")
                    continue
                is_literal, literal = literal_of(bounds[role])
                if is_literal and not integer_or_none(literal):
                    break
                if is_literal:
                    spellings.append("" if literal is None else f"c{int(literal)}")
                else:
                    arguments.append(value_id(bounds[role]))
                    spellings.append(f"v{value_id(bounds[role])}")
            else:
                pass
            if len(spellings) != 2:
                continue
        mutations.append(ControlSequenceMutation(
            value_id(node_id), "slice", tuple(arguments),
            int(node_id), policy="duplicates",
            argument_kind="slice:" + ":".join(spellings),
        ))
    return tuple(mutations)


def _source_mapping_mutations(graph_obj: Any):
    """Retain copies and writes to graph-owned dictionary materializers."""
    from .control_source import ControlSequenceMutation

    aliases = _loop_carried_storage_aliases(graph_obj)
    # The recursive metadata copier writes a caller-owned memo before its
    # child calls. Requiring a dict materializer hoisted that IndexedStore
    # into a numeric prelude before the branch's object constructor existed.
    # The existing keyed Input ABI proves this is the same lexical mapping
    # effect; its descriptor remains caller-owned, never a new local dict.
    keyed_inputs = {int(node_id) for node_id, data in graph_obj.nodes(data=True)
        if str(data.get('type') or data.get('op') or '').casefold() == 'input'
        and (graph_obj.graph.get('linked_value_abi') or {}).get(
            int(data.get('value_id', node_id)), {}).get('storage') == 'keyed'}
    materializers = {}
    for node_id, data in graph_obj.nodes(data=True):
        attrs = data.get("attributes") or {}
        if (attrs.get("aggregate_kind") == "dict"
                and attrs.get("producer_kind") == "aggregate_materialization"
                and not attrs.get("binding_name")):
            materializers[int(node_id)] = data
    mutations = []
    for node_id, data in materializers.items():
        arguments = [int(parent) for parent, role in data.get("parents") or ()
                     if str(role).startswith("arg:")]
        if len(arguments) != 1:
            continue
        source = arguments[0]
        source_data = graph_obj.nodes.get(source, {})
        expression = source_data.get("expr_obj")
        if isinstance(expression, ast.BoolOp) and isinstance(expression.op, ast.Or):
            source = next((int(parent) for parent, role in source_data.get("parents") or ()
                           if role == "value:0"), source)
        source = int(aliases.get(source, source))
        if (graph_obj.nodes.get(source, {}).get("attributes") or {}).get("aggregate_kind") != "dict":
            continue
        mutations.append(ControlSequenceMutation(
            int(data.get("value_id", node_id)), "replace", (source,),
            int(data.get("value_id", node_id)), policy="unique",
        ))
    for node_id, data in graph_obj.nodes(data=True):
        if str(data.get("type") or data.get("op")).lower() != "indexedstore":
            continue
        roles = {str(role): int(parent) for parent, role in data.get("parents") or ()}
        base = roles.get("base")
        base = aliases.get(base, base)
        if base not in (materializers.keys() | keyed_inputs) or "index" not in roles or "value" not in roles:
            continue
        mutations.append(ControlSequenceMutation(
            int(graph_obj.nodes[base].get("value_id", base)), "update",
            (roles["index"], roles["value"]), int(data.get("value_id", node_id)),
            policy="unique", argument_kind="mapping_items",
        ))
    return tuple(mutations)


def _identity_return_aliases(
    graph_obj: Any, function_table: Any, callsite_graphs: Any = None,
) -> dict[int, int]:
    """Prove caller aliases using callee return slots that return a formal.

    ``callsite_graphs`` maps a call node to the callee graph the planner
    specialized for THAT call.  A call whose constant actuals decide the
    callee's branch (``pick(a, 5)``) returns the formal in its specialized
    copy while the shared callee's slot is a merge; the call's own copy is
    the callee it runs, so it is the one read here.
    """
    from ..common.tensors.topological_reducer import _indexed_store_source

    if function_table is None:
        return {}
    callsite_graphs = {} if callsite_graphs is None else callsite_graphs
    result = {}
    for call_id, data in graph_obj.nodes(data=True):
        reference = (data.get("attributes") or {}).get("callee_ref")
        if reference is None:
            continue
        child = callsite_graphs.get(int(call_id))
        if child is None:
            child = getattr(function_table.entry(reference), "graph", None)
            child = getattr(child, "G", None)
        if child is None:
            continue
        sites = tuple((child.graph.get("return_slot_values") or {}).values())
        if not sites or len({len(site) for site in sites}) != 1:
            continue
        parameters = tuple(child.graph.get("function_parameters") or ())
        actuals = {str(role): int(parent) for parent, role in data.get("parents") or ()}
        # Program: ``def f(v): r = 0; while v < 100: (if v > 3: r = v; break);
        # v = v + 1; return r`` (and a direct tail call, which normalizes to
        # that loop).  Wrong record: the return slot names the parameter's
        # graph node, because ``r = v`` and the name ``v`` after a loop both
        # resolve to the pre-loop node; the call was folded into its actual
        # and the caller read the seed.  Rule: a parameter a loop carries or
        # a break binds is not the returned identity -- its node names only
        # the seed, so it is never a proved identity return.
        loop_rebound = set()
        for _loop_id, loop_data in child.nodes(data=True):
            loop_attrs = loop_data.get("attributes") or {}
            for pair in (loop_attrs.get("loop_carried_bindings") or {}).values():
                loop_rebound.add(int(pair[0]))
            for pair in (loop_attrs.get("loop_break_bindings") or {}).values():
                loop_rebound.update(int(item) for item in pair)
        for index, slots in enumerate(zip(*sites)):
            # ``a[0] = x; return a`` returns the store version of ``a``: the
            # list it writes (its ``base`` chain) is the returned object.
            slots = tuple(
                None if slot is None else _indexed_store_source(child, int(slot))
                for slot in slots
            )
            if len(set(slots)) != 1 or slots[0] is None:
                continue
            if int(slots[0]) in loop_rebound:
                continue
            formal = child.nodes.get(int(slots[0]), {})
            attrs = formal.get("attributes") or {}
            name = attrs.get("binding_name")
            if attrs.get("binding_kind") != "parameter" or name not in parameters:
                continue
            actual = actuals.get(f"arg:{parameters.index(name)}")
            if actual is None:
                continue
            if len(sites[0]) == 1:
                result[int(call_id)] = actual
            # A subscript of the call result selects a return SLOT only when
            # the callee returns a tuple/list display of slots.  ``def
            # same(a): return a`` returns one value: ``same(xs)[0]`` is
            # ELEMENT zero of that value, never the whole forwarded actual.
            if not set((child.graph.get("return_container_kinds") or {}).values()) & {"tuple", "list"}:
                continue
            for node_id, projection in graph_obj.nodes(data=True):
                if str(projection.get("type") or projection.get("op") or "").casefold() in {"indexedstore", "subscriptstore", "setitem"}:
                    continue
                roles = {str(role): int(parent) for parent, role in projection.get("parents") or ()}
                if roles.get("base") != int(call_id):
                    continue
                key = (graph_obj.nodes.get(roles.get("index"), {}).get("attributes") or {}).get("value")
                if key == index:
                    result[int(node_id)] = actual
    return result


def _returned_tuple_member_kinds(
    graph_obj: Any, function_table: Any,
) -> dict[int, str]:
    """Collection kind of each unpacked member of a tuple-returning call.

    Program: ``def mk(n): xs = [n, 1]; return xs, n + 1`` then
    ``xs, k = mk(n); xs.append(5); return len(xs)`` (c822).  Wrong record:
    the member projection ``call[0]`` carries no ``aggregate_kind`` (only the
    callee's return slot knows it is a list), so the in-place method on it
    was no mutation record and the append vanished (silent wrong value).
    Rule: a constant-index projection of a call result is the collection the
    callee returns at that position, whenever every return site agrees.
    """

    if function_table is None:
        return {}
    kinds = {}
    for call_id, data in graph_obj.nodes(data=True):
        reference = (data.get("attributes") or {}).get("callee_ref")
        if reference is None:
            continue
        child = getattr(
            getattr(function_table.entry(reference), "graph", None), "G", None,
        )
        if child is None:
            continue
        sites = tuple((child.graph.get("return_slot_values") or {}).values())
        if not sites or len({len(site) for site in sites}) != 1:
            continue
        member_kinds = []
        for slots in zip(*sites):
            site_kinds = {
                None if slot is None else _receiver_collection_kind(
                    child.nodes.get(int(slot), {})
                )
                for slot in slots
            }
            member_kinds.append(
                site_kinds.pop() if len(site_kinds) == 1 else None
            )
        for node_id, projection in graph_obj.nodes(data=True):
            if str(projection.get("type") or projection.get("op") or "").casefold() in {
                "indexedstore", "subscriptstore", "setitem",
            }:
                continue
            roles = {
                str(role): int(parent)
                for parent, role in projection.get("parents") or ()
            }
            if roles.get("base") != int(call_id) or "index" not in roles:
                continue
            key = (graph_obj.nodes.get(roles["index"], {}).get("attributes") or {}).get("value")
            if (
                isinstance(key, int) and not isinstance(key, bool)
                and 0 <= key < len(member_kinds)
                and member_kinds[key] in {"list", "bytes", "bytearray"}
            ):
                kinds[int(projection.get("value_id", node_id))] = member_kinds[key]
    return kinds


def _returned_formal_sequence_aliases(
    graph_obj: Any, function_table: Any, *,
    call_result_kinds: Mapping[int, str] | None = None,
    callsite_graphs: Any = None,
) -> dict[int, tuple[int, str]]:
    """Rule 8 for a whole sequence result.

    A call whose result is its caller argument's own list (``def mk(a):
    return a``) is that list: the planner's identity forwarding of a
    returned formal, read for a sequence actual.  The call owns no storage
    of its own, so it names the actual's arena and kind.
    """

    forwarded = {}
    for call_id, actual_id in _identity_return_aliases(
        graph_obj, function_table, callsite_graphs,
    ).items():
        call_data = graph_obj.nodes.get(int(call_id), {})
        # A tuple member ``b`` of ``b, k = f(a, n)`` (``return a, n``) is the
        # call's authored projection: it names the actual's storage as the
        # whole-result call does.
        if (
            (call_data.get("attributes") or {}).get("callee_ref") is None
            and not (call_data.get("attributes") or {}).get(
                "authored_call_result_projection"
            )
        ):
            continue
        actual_data = graph_obj.nodes.get(int(actual_id), {})
        value_id = int(actual_data.get("value_id", actual_id))
        kind = str(
            (actual_data.get("attributes") or {}).get("aggregate_kind")
            or dict(call_result_kinds or {}).get(value_id, "")
        )
        if kind in {"list", "bytes", "bytearray"}:
            forwarded[int(call_data.get("value_id", call_id))] = (value_id, kind)
    # ``u = same(same(a))``: the actual of one forwarding call may itself be
    # a forwarded result.  The list is still ``a``'s one storage.
    for result_id, (actual_id, kind) in tuple(forwarded.items()):
        seen = {int(result_id)}
        while int(actual_id) in forwarded and int(actual_id) not in seen:
            seen.add(int(actual_id))
            actual_id = forwarded[int(actual_id)][0]
        forwarded[result_id] = (int(actual_id), kind)
    return forwarded


def _static_mapping_capacity_bounds(
    graph_obj: Any,
    *,
    nonmutating_call_ids: Iterable[int] = (),
    linked_bounded_ids: Iterable[int] = (),
) -> dict[int, int]:
    """Bound local keyed arenas by their complete constant-key universe.

    Repeated writes, including writes in loops, cannot increase this bound.
    Copies inherit the source universe. Dynamic keys and external sources
    deliberately have no bound here; they need a runtime capacity contract.
    """
    aliases = _loop_carried_storage_aliases(graph_obj)
    nonmutating_calls = set(map(int, nonmutating_call_ids))
    # A dict display passed to callees that add no keys (``total({1: n})``):
    # every escape is an exact argument binding into a component no function
    # grows (``_settle_linked_dynamic_sequences``), so its keys bound it.
    linked_bounded = set(map(int, linked_bounded_ids))
    keys, copies, unknown = {}, {}, set()
    set_universes: set[int] = set()
    for node_id, data in graph_obj.nodes(data=True):
        expression = data.get("expr_obj")
        attributes = data.get("attributes") or {}
        if (attributes.get("aggregate_kind") == "dict"
                and attributes.get("materialization_kind") == "unrolled_loop"):
            universe = keys.setdefault(int(node_id), set())
            for row_id in attributes.get("materialized_value_ids", ()):
                row = graph_obj.nodes.get(int(row_id), {})
                leaves = tuple(int(parent) for parent, role in row.get("parents", ()) if role == "elts")
                key_data = graph_obj.nodes.get(leaves[0], {}) if leaves else {}
                try:
                    if "value" in (key_data.get("attributes") or {}):
                        key = key_data["attributes"]["value"]
                    else:
                        key = ast.literal_eval(key_data.get("expr_obj"))
                    universe.add(key)
                except (ValueError, TypeError):
                    unknown.add(int(node_id))
        if isinstance(expression, ast.Dict):
            try:
                keys[int(node_id)] = {ast.literal_eval(key) for key in expression.keys}
            except (ValueError, TypeError):
                unknown.add(int(node_id))
        # Program: ``ss = {4, 6}; n in ss`` (scratch c_n3).  Wrong record: a
        # set display had neither a bound nor the dynamic contract, so its
        # arena was refused ("no proven capacity").  Rule: a set display is
        # a keyed arena whose literal members are its key universe, exactly
        # as a dict display's keys are; any other writer unbounds it below.
        if isinstance(expression, ast.Set):
            set_universes.add(int(node_id))
            try:
                keys[int(node_id)] = {
                    ast.literal_eval(element) for element in expression.elts
                }
            except (ValueError, TypeError):
                unknown.add(int(node_id))
        elif (
            attributes.get("aggregate_kind") in {"set", "frozenset"}
            and str(data.get("type")) in {"Constant", "Const", "const"}
            and isinstance(attributes.get("value"), (set, frozenset, list, tuple))
        ):
            # The structural fold's literal of an unchanged set display
            # (scratch c_n3): the same key universe, read from the literal.
            set_universes.add(int(node_id))
            try:
                keys[int(node_id)] = set(attributes["value"])
            except TypeError:
                unknown.add(int(node_id))
    # Program: ``d[(1, n, 2)] = 5; d[(1, n + 1, 2)] = 6`` (string core
    # ``key = (domain, hi, lo, count)``).  Wrong record: a tuple subscript's
    # store names one ``index`` parent per component, and the role map kept
    # only the last one, so both stores read as the literal key ``2``; the
    # universe bound was one row and the second insert reported full
    # (validation 75).  Rule: a store's key is all its index components,
    # in order; it is literal only when every component is.
    def literal_store_key(index_ids):
        components = tuple(
            ast.literal_eval(graph_obj.nodes[int(index)].get("expr_obj"))
            for index in index_ids
        )
        return components[0] if len(components) == 1 else components

    store_index_ids = {}
    for store_id, store in graph_obj.nodes(data=True):
        if str(store.get("type") or store.get("op") or "").casefold() == "indexedstore":
            store_index_ids[int(store.get("value_id", store_id))] = tuple(
                int(parent) for parent, role in store.get("parents") or ()
                if str(role) == "index"
            )
    for mutation in _source_mapping_mutations(graph_obj):
        destination = int(mutation.sequence_value_id)
        if mutation.operator == "replace":
            copies[destination] = int(mutation.argument_value_ids[0])
            keys.setdefault(destination, set())
        elif mutation.operator == "update":
            store_key = store_index_ids.get(int(mutation.effect_node_id))
            for key_id in mutation.argument_value_ids[::2]:
                try:
                    key = literal_store_key(store_key or (key_id,))
                    keys.setdefault(destination, set()).add(key)
                except (KeyError, ValueError, TypeError):
                    unknown.add(destination)
    # Program: ``s = {1, 2}; for i in ...: s.add(i * 3)`` (scratch c_s2).
    # Wrong record: the add's node is a method call of the recorded
    # mutation, which the call scan below does not match, so the display's
    # two-member universe stood as the bound and every later add was
    # refused into a dropped status.  Rule: a retained insertion into a
    # keyed universe adds its literal key; any other key unbounds it.
    # Only a set universe is adjusted here (a dict's writers are the store
    # and call scans below); with none, the records are not read -- a bare
    # graph with no source scope has none to read.
    for mutation in (
        _sequence_append_call_mutations(graph_obj) if set_universes else ()
    ):
        destination = int(aliases.get(
            int(mutation.sequence_value_id), int(mutation.sequence_value_id),
        ))
        if destination not in keys or mutation.operator == "clear":
            continue
        try:
            literal_keys = [
                ast.literal_eval(graph_obj.nodes[int(argument)].get("expr_obj"))
                for argument in mutation.argument_value_ids
            ]
        except (KeyError, ValueError, TypeError):
            unknown.add(destination)
            continue
        if mutation.operator != "add" or len(literal_keys) != 1:
            unknown.add(destination)
            continue
        keys[destination].add(literal_keys[0])
    # ``d.update(<dict display>)`` stores the display's rows into ``d`` and
    # only reads the display (corpus c214): the receiver's universe gains
    # the rows' literal keys, as ``d[k] = v`` does, and neither is
    # unbounded by the call scan below.
    display_updates = set()
    for mutation in _dict_update_mutations(graph_obj):
        if mutation.argument_kind != "mapping_items":
            continue
        display_updates.add(int(mutation.effect_node_id))
        destination = int(aliases.get(
            int(mutation.sequence_value_id), int(mutation.sequence_value_id),
        ))
        if destination not in keys:
            continue
        for key_id in mutation.argument_value_ids[::2]:
            try:
                keys[destination].add(ast.literal_eval(
                    graph_obj.nodes[int(key_id)].get("expr_obj")
                ))
            except (KeyError, ValueError, TypeError):
                unknown.add(destination)
    for _node_id, data in graph_obj.nodes(data=True):
        roles = {str(role): int(parent) for parent, role in data.get("parents") or ()}
        operation = str(data.get("type") or data.get("op") or "").casefold()
        if int(_node_id) in display_updates:
            continue
        if operation == "indexedstore":
            destination = int(aliases.get(roles.get("base"), roles.get("base", -1)))
            if destination in keys:
                try:
                    key = literal_store_key(tuple(
                        int(parent) for parent, role in data.get("parents") or ()
                        if str(role) == "index"
                    ) or (roles["index"],))
                    keys[destination].add(key)
                except (KeyError, ValueError, TypeError):
                    unknown.add(destination)
        elif (operation in {"call", "plancall", "update", "setdefault"}
              and int(_node_id) not in nonmutating_calls
              and int(_node_id) not in copies
              # ``k in d`` reaches the graph as a Call over the local dict
              # literal; the builtin dict containment query never inserts a
              # key, so it does not unbound the constant-key universe (it
              # did: ``{1: 10, 2: 20, 3: 30}`` lost its bound and its arena
              # was allocated with one row).
              and not (
                  isinstance(data.get("expr_obj"), ast.Compare)
                  and all(
                      isinstance(op, (ast.In, ast.NotIn))
                      for op in data["expr_obj"].ops
                  )
              )
              # A dict's own method that cannot insert a key (``clear``
              # empties it; the readers read it) leaves the constant-key
              # universe a bound, as ``in`` does.  ``cells.clear()`` had
              # unbounded ``{1: 10, 2: 20}``.
              and not (
                  isinstance(data.get("expr_obj"), ast.Call)
                  and isinstance(data["expr_obj"].func, ast.Attribute)
                  and data["expr_obj"].func.attr in {
                      "clear", "get", "keys", "values", "items", "copy",
                  }
                  and not data["expr_obj"].args[1:]
              )
              # A builtin that reads or copies its argument (``len(d)``,
              # the escape rule's ``_NONESCAPING_COLLECTION_CALLS``) never
              # inserts into it either.
              and not (
                  isinstance(data.get("expr_obj"), ast.Call)
                  and _collection_call_identity(data)
                  in _NONESCAPING_COLLECTION_CALLS
              )):
            for role, parent in roles.items():
                destination = int(aliases.get(parent, parent))
                if destination in linked_bounded and role.startswith("arg:"):
                    continue
                if destination in keys:
                    unknown.add(destination)
    # Program: ``d = {1: 5, 2: n}; del d[1]; d[1] = 4`` (scratch s42/t2).
    # Wrong record: a keyed delete only clears its row's live flag, so the
    # later store of the same key appended a THIRD row, past the two-key
    # universe bound; the store's capacity status was dropped and
    # ``len(d)`` was 1 for 2 (clean audit).  Rule: each keyed delete that
    # runs at most once may cost one more resident row; a delete a loop
    # holds may run any number of times, so it unbounds the arena.
    holders: dict[int, int] = {}
    for node_id, data in graph_obj.nodes(data=True):
        for parent, role in data.get("parents") or ():
            if str(role) in {"body", "orelse"}:
                holders[int(parent)] = int(node_id)
    deleted_rows: dict[int, int] = {}
    for node_id, data in graph_obj.nodes(data=True):
        if str(data.get("type") or data.get("op") or "").casefold() != "delitem":
            continue
        roles = {str(role): int(parent) for parent, role in data.get("parents") or ()}
        destination = int(aliases.get(roles.get("base"), roles.get("base", -1)))
        if destination not in keys:
            continue
        holder, seen, looped = holders.get(int(node_id)), set(), False
        while holder is not None and holder not in seen:
            seen.add(holder)
            if isinstance(
                graph_obj.nodes.get(holder, {}).get("expr_obj"),
                (ast.For, ast.AsyncFor, ast.While),
            ):
                looped = True
                break
            holder = holders.get(holder)
        if looped:
            unknown.add(destination)
        else:
            deleted_rows[destination] = deleted_rows.get(destination, 0) + 1
    changed = True
    while changed:
        changed = False
        for destination, source in copies.items():
            if source not in keys or source in unknown:
                if destination not in unknown:
                    unknown.add(destination)
                    changed = True
            else:
                previous = len(keys[destination])
                keys[destination].update(keys[source])
                changed |= len(keys[destination]) != previous
    # A copy holds its source's rows, the deleted ones included.
    for destination, source in copies.items():
        deleted_rows[destination] = (
            deleted_rows.get(destination, 0) + deleted_rows.get(source, 0)
        )
    return {node_id: len(universe) + deleted_rows.get(node_id, 0)
            for node_id, universe in keys.items()
            if node_id not in unknown}


def _control_sequence_capacity_bounds(control: Any, sequence_ids: Iterable[int]) -> dict[int, int]:
    """Interval transfer over the existing lexical control tree, not a replay.

    In the controller-shaped repro, failures grows at most once in a while
    whose every path breaks, and lines starts with one row then grows at most
    len(failures). Treating every loop as unbounded discarded both proofs;
    the native frame defaulted to one row and append silently hit capacity.
    Branches join by max; bounded loops multiply growth. Unknown backedges or
    a loop mutating its own extent invalidate the affected bound, not guess.
    """
    import re
    from .control_source import (CallBlock, ConditionalBlock, LoopBlock,
        LoopControlBlock, SequenceBlock, SequenceMutationBlock, WhileBlock)

    initial = {int(value_id): 0 for value_id in sequence_ids}
    peaks = dict(initial)

    def join(left, right):
        return {key: None if left[key] is None or right[key] is None
                else max(left[key], right[key]) for key in initial}

    def effects(block):
        if block is None:
            return ()
        result = list(getattr(block, "sequence_mutations", ()))
        if isinstance(block, SequenceMutationBlock):
            result.append(block.mutation)
        for child in getattr(block, "blocks", ()):
            result.extend(effects(child))
        for arm in ("condition", "body", "orelse", "callee"):
            result.extend(effects(getattr(block, arm, None)))
        return tuple(result)

    def outcomes(block):
        if isinstance(block, LoopControlBlock):
            return ({block.action} if block.predicate_value_id is None
                    and block.predicate_expression is None else {"normal", block.action})
        if isinstance(block, SequenceBlock):
            result = {"normal"}
            for child in block.blocks:
                if "normal" not in result:
                    break
                result = (result - {"normal"}) | outcomes(child)
            return result
        if isinstance(block, ConditionalBlock):
            return outcomes(block.body) | outcomes(block.orelse)
        return {"normal"}

    def extent(text, state):
        def expression(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                return int(node.value)
            if isinstance(node, ast.Name):
                match = re.fullmatch(r"__iterable_extent_(\d+)__", node.id)
                return state.get(int(match[1])) if match else None
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                left, right = expression(node.left), expression(node.right)
                return None if left is None or right is None else left + right
            return None
        try:
            return expression(ast.parse(str(text), mode="eval").body)
        except (SyntaxError, ValueError, TypeError):
            return None

    def mutate(mutation, state):
        sequence_id = int(mutation.sequence_value_id)
        if sequence_id not in state:
            return state
        state = dict(state)
        current = state[sequence_id]
        if mutation.operator == "clear" and mutation.predicate_expression is None:
            state[sequence_id] = 0
        elif mutation.operator == "append" and current is not None:
            state[sequence_id] = current + 1
        elif mutation.operator not in {"clear", "pop", "remove"}:
            state[sequence_id] = None
        peaks[sequence_id] = (None if peaks[sequence_id] is None or state[sequence_id] is None
                              else max(peaks[sequence_id], state[sequence_id]))
        return state

    def visit(block, state):
        if block is None:
            return state
        if isinstance(block, SequenceMutationBlock):
            return mutate(block.mutation, state)
        if isinstance(block, SequenceBlock):
            for child in block.blocks:
                state = visit(child, state)
                if outcomes(child) <= {"break", "return", "continue"}:
                    break
            return state
        if isinstance(block, ConditionalBlock):
            return join(visit(block.body, dict(state)), visit(block.orelse, dict(state)))
        if isinstance(block, CallBlock):
            return visit(block.callee, state)
        if isinstance(block, (LoopBlock, WhileBlock)):
            writes = {int(m.sequence_value_id) for m in effects(block)} & set(initial)
            count = None
            if isinstance(block, WhileBlock):
                body_outcomes = outcomes(SequenceBlock((block.body, *block.terminal_controls)))
                if body_outcomes <= {"break", "return"}:
                    count = 1
            else:
                start, stop, step = (extent(text, state) for text in (block.start, block.stop, block.step))
                reads = {int(value_id) for value_id in re.findall(
                    r"__iterable_extent_(\d+)__", str(block.stop))}
                if (not reads & writes and start is not None and stop is not None
                        and step is not None and step > 0):
                    count = len(range(start, stop, step))
            if count is None:
                state = dict(state)
                for key in writes:
                    state[key] = peaks[key] = None
                return state
            if count == 0:
                return state
            before = dict(state)
            state = visit(getattr(block, "condition", None), dict(state))
            state = visit(block.body, state)
            for mutation in block.sequence_mutations:
                state = mutate(mutation, state)
            if count > 1:
                # Nested bounds must be invariant over the outer body. A
                # written extent could grow each iteration; do not extrapolate
                # the first iteration's work in that case.
                def variable_extent(candidate):
                    if isinstance(candidate, LoopBlock) and writes & {
                        int(v) for v in re.findall(r"__iterable_extent_(\d+)__", str(candidate.stop))}:
                        return True
                    return any(variable_extent(child) for child in getattr(candidate, "blocks", ())) or any(
                        variable_extent(getattr(candidate, arm)) for arm in ("body", "orelse", "callee")
                        if getattr(candidate, arm, None) is not None)
                unsafe = variable_extent(block.body)
                for key in writes:
                    if unsafe or before[key] is None or state[key] is None or peaks[key] is None:
                        state[key] = peaks[key] = None
                    else:
                        growth = max(0, state[key] - before[key], peaks[key] - before[key])
                        state[key] = before[key] + count * growth
                        peaks[key] = max(peaks[key], state[key])
            return join(before, state)
        return state

    visit(control.root, initial)
    return {key: value for key, value in peaks.items() if value is not None}


#: Builtins that read (or copy) a collection argument without retaining or
#: mutating it: the argument's rows after the call are its rows before it.
_NONESCAPING_COLLECTION_CALLS = frozenset({
    "builtins.tuple", "builtins.len", "builtins.bool", "builtins.enumerate",
    "builtins.bytes", "builtins.bytearray", "builtins.list", "builtins.set",
    "builtins.frozenset", "builtins.dict", "builtins.sorted", "builtins.sum",
    "builtins.min", "builtins.max", "builtins.any", "builtins.all",
    "builtins.zip", "builtins.reversed", "builtins.isinstance",
    "builtins.id", "builtins.str", "builtins.repr",
})


_BUILTIN_COLLECTION_KINDS = frozenset({
    "list", "array", "dict", "set", "frozenset", "tuple", "str", "bytes", "bytearray",
})


def _collection_call_identity(data: Mapping[str, Any]) -> Any:
    """A call node's builtin identity, read as the escape rule reads it."""

    attributes = data.get("attributes") or {}
    identity = attributes.get("extraction_identity")
    reference = attributes.get("static_python_reference")
    if (
        identity is None
        and isinstance(reference, str)
        and "." not in reference
    ):
        identity = f"builtins.{reference}"
    return identity


def _call_escaped_argument_ids(graph_obj: Any) -> frozenset[int]:
    """Values passed as an argument into a call that may retain or mutate it.

    One definition for every reader: the capacity survey (an escaped list's
    rows are not bounded by its own appends) and the structural fold (an
    escaped collection's initializer is not its runtime contents).
    """

    return frozenset(
        argument for _call, argument in _call_escaped_argument_edges(graph_obj)
    )


def _call_escaped_argument_edges(graph_obj: Any) -> frozenset[tuple[int, int]]:
    """``(call node, argument)`` for every escape ``_call_escaped_argument_ids``
    names: the same definition, keeping which call each value escapes into."""

    escaped: set[tuple[int, int]] = set()
    for _node_id, data in graph_obj.nodes(data=True):
        expression = data.get("expr_obj")
        if not isinstance(expression, ast.Call):
            continue
        # Before extraction resolves the call, its static symbol names the
        # builtin (``bytes([11])``).
        identity = _collection_call_identity(data)
        if identity in _NONESCAPING_COLLECTION_CALLS:
            continue
        # Program: ``scopes.setdefault(row[0], {}).setdefault(row, None)``
        # (identity_spill._reindex_in_place).  Wrong record: the ``{}`` and
        # ``[]`` defaults of a builtin collection's own method were counted
        # as escapes, so the structural fold kept them as unknown-dtype
        # sequences the compiler's table operations never read.  Rule: a
        # method of a builtin collection receiver is the compiler's own
        # table operation on resident storage, not a call into code that
        # may retain or mutate its argument.
        if isinstance(expression.func, ast.Attribute):
            receiver = next((
                parent for parent, role in data.get("parents") or ()
                if str(role) == "operand"
            ), None)
            if receiver is not None and (
                (graph_obj.nodes.get(receiver, {}).get("attributes") or {}).get(
                    "aggregate_kind"
                )
            ) in _BUILTIN_COLLECTION_KINDS:
                continue
        for parent, role in data.get("parents") or ():
            if str(role).startswith("arg:"):
                escaped.add((int(_node_id), int(parent)))
    return frozenset(escaped)


#: Methods of a mutable builtin collection that read their receiver without
#: changing it.  Every other method of a list/dict/set/bytearray receiver
#: (``pop``, ``sort``, ``reverse``, ``insert``, ``remove``, ``update`` ...,
#: and any method this table does not name) changes the receiver's contents.
_RECEIVER_READING_METHODS = frozenset({
    "count", "index", "copy", "get", "keys", "values", "items",
    "issubset", "issuperset", "isdisjoint", "union", "intersection",
    "difference", "symmetric_difference", "fromkeys", "hex", "decode",
    "startswith", "endswith", "find", "rfind",
})

_MUTABLE_COLLECTION_KINDS = frozenset({"list", "dict", "set", "bytearray", "array"})


def _receiver_collection_kind(data: Any) -> str | None:
    """A receiver node's builtin collection kind: its ``aggregate_kind``,
    else the builtin class its call result is declared to be.

    Program: ``t = pick(n); t.append(9)`` with ``pick -> list[int]``
    (corpus c216).  Wrong record: a call result carries no
    ``aggregate_kind``, only ``result_class_ref='builtins.list'``, so its
    in-place methods were no mutation record and the append vanished (the
    caller read past the length: garbage).  Rule: a call result whose
    declared class is a builtin collection is that collection.
    """

    attributes = (data or {}).get("attributes") or {}
    kind = attributes.get("aggregate_kind")
    if kind is not None:
        return str(kind)
    declared = str(attributes.get("result_class_ref") or "")
    if declared.startswith("builtins.") and declared[9:] in (
        _MUTABLE_COLLECTION_KINDS | {"bytes", "tuple", "frozenset"}
    ):
        return declared[9:]
    return None


def _method_mutated_receiver_ids(graph_obj: Any) -> frozenset[int]:
    """Mutable collections whose own method call changes their contents.

    The receivers of ``_method_mutation_edges``.
    """

    return frozenset(
        receiver for _call, receiver in _method_mutation_edges(graph_obj)
    )


def _method_mutation_edges(graph_obj: Any) -> frozenset[tuple[int, int]]:
    """``(call node, receiver)`` for each method call that mutates its
    mutable collection receiver.

    Program: ``xs = [5, n, 2]; xs.pop(0); return xs[0] * 10 + xs[1]``
    (scratch c143).  Wrong record: the structural fold counted only retained
    mutation records, escaped arguments and indexed stores as mutation, so
    ``xs[0]`` folded to the display's 5 after ``pop`` removed it (51, not
    12; likewise ``sort``/``reverse``/``insert``).  Rule: a method call on a
    mutable builtin collection receiver mutates that receiver unless the
    method only reads it (``_RECEIVER_READING_METHODS``).
    """

    mutated: set[tuple[int, int]] = set()
    for node_id, data in graph_obj.nodes(data=True):
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
        ) or expression.func.attr in _RECEIVER_READING_METHODS:
            continue
        for parent, role in data.get("parents") or ():
            if str(role) != "operand" or parent not in graph_obj:
                continue
            if (
                _receiver_collection_kind(graph_obj.nodes[parent])
                in _MUTABLE_COLLECTION_KINDS
                or isinstance(
                    graph_obj.nodes[parent].get("expr_obj"),
                    (ast.List, ast.Set, ast.Dict),
                )
            ):
                mutated.add((int(node_id), int(parent)))
    return frozenset(mutated)


def _static_sequence_capacity_bounds(
    graph_obj: Any,
    *,
    nonmutating_call_ids: Iterable[int] = (),
    control: Any = None,
) -> dict[int, int]:
    return _static_sequence_capacity_survey(
        graph_obj, nonmutating_call_ids=nonmutating_call_ids, control=control,
    )[0]


def _static_sequence_capacity_survey(
    graph_obj: Any,
    *,
    nonmutating_call_ids: Iterable[int] = (),
    control: Any = None,
    linked_bounded_ids: Iterable[int] = (),
) -> tuple[dict[int, int], frozenset[int]]:
    """Proven row bounds, and the local lists no bound covers.

    The second set is the authored local lists (``xs = []``) whose rows the
    survey cannot bound -- appended inside a loop, or appended by an
    operator other than ``append`` -- and which never escape into a call.
    Their storage is the function's own, so their capacity is the dynamic
    contract (heap owner, checked geometric growth), not a fixed extent.
    """
    bounds = _static_mapping_capacity_bounds(
        graph_obj,
        nonmutating_call_ids=nonmutating_call_ids,
        linked_bounded_ids=linked_bounded_ids,
    )
    local_lists = {int(node_id): len(data["expr_obj"].elts)
                   for node_id, data in graph_obj.nodes(data=True)
                   if isinstance(data.get("expr_obj"), ast.List)}
    # Program: ``ys = [x * n for x in xs]; ys.append(n)`` with ``xs`` a
    # folded literal, then ``len(ys) * 100 + sum(ys)`` (c266).  Wrong
    # record: the unrolled comprehension is born here like a display
    # (``_list_literal_mutations``) but was not a local list, so its later
    # append was never counted; the arena took the planner's extent (3,),
    # the append found it full and its status was dropped (306 for 407;
    # clean audit).  Rule: an unrolled list comprehension is a local list
    # of its materialized rows.
    local_lists.update({
        int(data.get("value_id", node_id)): len(tuple(
            attributes.get("materialized_value_ids", ())
        ))
        for node_id, data in graph_obj.nodes(data=True)
        for attributes in ((data.get("attributes") or {}),)
        if attributes.get("materialization_kind") == "unrolled_loop"
        and attributes.get("aggregate_kind") in {"list", "set"}
    })
    # Program: ``xs = [1, 5, 9]; for x in xs: ...`` (c156).  Wrong record:
    # the structural fold turns an unchanged list display into a Constant
    # whose literal still names its members, and no bound covered it, so C
    # refused its private arena ("no proven capacity").  Rule: a folded list
    # display is a lexical birth, one row per member, like the display it
    # replaced; later growth is counted below exactly as for a display.
    local_lists.update({
        int(data.get("value_id", node_id)): len(attributes["value"])
        for node_id, data in graph_obj.nodes(data=True)
        for attributes in ((data.get("attributes") or {}),)
        if str(data.get("type")) in {"Constant", "Const", "const"}
        and attributes.get("aggregate_kind") == "list"
        and isinstance(attributes.get("value"), (list, tuple))
        and int(data.get("value_id", node_id)) not in local_lists
    })
    # A list merged from equal-length arm displays (an ``aggregate_phi``)
    # is born at the merge from its member Phis (``_list_literal_
    # mutations``), so it is a local list of that many rows.
    local_lists.update({
        int(node_id): len(attributes["aggregate_leaf_value_ids"])
        for node_id, data in graph_obj.nodes(data=True)
        for attributes in (data.get("attributes") or {},)
        if attributes.get("producer_kind") == "aggregate_phi"
        and attributes.get("aggregate_kind") in {"list", "set"}
        and attributes.get("aggregate_leaf_value_ids")
    })
    unknown = set()
    escaped = set()
    loops = [data["expr_obj"] for _, data in graph_obj.nodes(data=True)
             if isinstance(data.get("expr_obj"), (ast.For, ast.While))]
    for mutation in _sequence_append_call_mutations(graph_obj):
        sequence_id = int(mutation.sequence_value_id)
        if sequence_id not in local_lists or mutation.operator == "clear":
            continue
        expression = graph_obj.nodes.get(int(mutation.effect_node_id), {}).get("expr_obj")
        line = getattr(expression, "lineno", None)
        if (mutation.operator != "append" or line is None
                or any(loop.lineno <= line <= loop.end_lineno for loop in loops)):
            unknown.add(sequence_id)
        else:
            local_lists[sequence_id] += 1
    # ``linked_bounded_ids``: escaped lists every escape of which is an exact
    # argument binding into a component no function grows
    # (``_settle_linked_dynamic_sequences``).  The callees share this storage
    # and add no rows, so its own rows bound it.
    linked_bounded = set(map(int, linked_bounded_ids))
    for parent in _call_escaped_argument_ids(graph_obj):
        if parent in local_lists and parent not in linked_bounded:
            unknown.add(parent)
            escaped.add(parent)
    bounds.update({key: count for key, count in local_lists.items() if key not in unknown})
    # Program: ``t = (n + 1, n + 2, n * 3); len(t)`` (c260).  Wrong record:
    # a tuple display had resident storage but no birth and no bound, so
    # ``len(t)`` read the frame's zeroed length cell (0 for 3; clean audit).
    # Rule: a tuple display is born like a list display, and nothing grows
    # a tuple, so its member count bounds it even when it escapes.
    for node_id, data in graph_obj.nodes(data=True):
        expression = data.get("expr_obj")
        if (isinstance(expression, ast.Tuple)
                and isinstance(expression.ctx, ast.Load)
                and not any(isinstance(element, ast.Starred)
                            for element in expression.elts)):
            bounds.setdefault(int(node_id), len(expression.elts))
    # Program: ``a = [1, 2]; t = a if n > 2 else [3, 4]`` then ``len(a)``
    # (corpus c271).  Wrong record: the merge (an ``aggregate_phi``) was
    # bounded, but each folded literal arm is a resident arena of its own
    # (it is the phi's member), and no arm had a bound, so C refused it as
    # a private sequence with no capacity.  Rule: an arm of a list merge
    # that is a folded literal and that nothing grows is bounded by its
    # literal length.
    grown_arms = {
        int(mutation.sequence_value_id)
        for mutation in _sequence_append_call_mutations(graph_obj)
    }
    for _node_id, data in graph_obj.nodes(data=True):
        attributes = data.get("attributes") or {}
        if (attributes.get("producer_kind") != "aggregate_phi"
                or attributes.get("aggregate_kind") != "list"):
            continue
        for parent, role in data.get("parents") or ():
            if str(role) not in {"body", "orelse"} or int(parent) in grown_arms:
                continue
            arm = graph_obj.nodes.get(int(parent), {}).get("attributes") or {}
            if (arm.get("aggregate_kind") == "list"
                    and isinstance(arm.get("value"), (list, tuple))):
                bounds.setdefault(int(parent), len(arm["value"]))
    if control is not None:
        bounds.update({key: max(count, bounds.get(key, 0)) for key, count in _control_sequence_capacity_bounds(
            control, local_lists,
        ).items() if key not in escaped})
    # Program: ``ys = list(xs); ys.append(n); len(ys)`` (fuzzer p3_37).
    # Wrong record: the snapshot took its source's bound and its own later
    # appends were never counted, so the append found the arena full, its
    # status was dropped and ``len(ys)`` returned 3 for 4.  Rule: a
    # snapshot's bound is its source's plus its own straight-line appends;
    # any other growth leaves it unbounded (the dynamic contract), exactly
    # as a local list's.
    # A slice holds at most its source's rows, so it takes the snapshot
    # bound: its source's plus its own straight-line appends.
    snapshots = (*_source_sequence_snapshots(graph_obj),
                 *_source_sequence_slices(graph_obj))
    snapshot_effects = {int(mutation.effect_node_id) for mutation in snapshots}
    snapshot_ids = {int(mutation.sequence_value_id) for mutation in snapshots}
    snapshot_growth = {sequence_id: 0 for sequence_id in snapshot_ids}
    unbounded_snapshots = set()
    for mutation in _sequence_append_call_mutations(graph_obj):
        sequence_id = int(mutation.sequence_value_id)
        if (sequence_id not in snapshot_ids or mutation.operator == "clear"
                or int(mutation.effect_node_id) in snapshot_effects):
            continue
        expression = graph_obj.nodes.get(int(mutation.effect_node_id), {}).get("expr_obj")
        line = getattr(expression, "lineno", None)
        if (mutation.operator != "append" or line is None
                or any(loop.lineno <= line <= loop.end_lineno for loop in loops)):
            unbounded_snapshots.add(sequence_id)
        else:
            snapshot_growth[sequence_id] += 1
    escaped_snapshots = {
        parent for parent in _call_escaped_argument_ids(graph_obj)
        if parent in snapshot_ids
    }
    for mutation in snapshots:
        source = int(mutation.argument_value_ids[0])
        destination = int(mutation.sequence_value_id)
        if destination in unbounded_snapshots or destination in escaped_snapshots:
            bounds.pop(destination, None)
            continue
        if source in bounds:
            bounds[destination] = bounds[source] + snapshot_growth[destination]
        elif str(
            (graph_obj.nodes.get(source) or {}).get("type") or ""
        ).casefold() == "input":
            # Program: ``def add(self, values, b, e):
            # self.cover.extend(values[b:e])`` (token orchard core).  Wrong
            # record: the slice's source is a caller's list (a parameter),
            # which no survey here bounds, so the slice was neither bounded
            # nor unbounded and C/LLVM refused it as a private sequence with
            # no proven capacity.  Rule: a snapshot of a parameter holds the
            # caller's run-time rows (at most ``len(values)``, clipped to
            # ``e - b`` for a slice); it takes the dynamic contract, whose
            # append reserves exactly the rows it copies.
            unbounded_snapshots.add(destination)
    # Program: ``xs = [5, 7] + [n, 7]; len(xs)`` (fuzzer p3_54).  Wrong
    # record: nothing bounded the concatenation, so its arena took the
    # planner's elementwise-add extent (2,) and the second operand's rows
    # were written past it.  Rule: a list concatenation holds exactly its
    # operands' rows; its bound is the sum of their bounds (a folded display
    # is a constant of its literal length).
    def sequence_bound(node_id):
        if node_id in bounds:
            return bounds[node_id]
        literal = (graph_obj.nodes.get(node_id, {}).get("attributes") or {}).get("value")
        if isinstance(literal, (list, tuple)):
            return len(literal)
        return None

    # Program: ``xs = [1, 2] + [n, 8]; ys = [n * 2] + xs; len(ys)`` (c263).
    # Wrong record: a concatenation operand that is itself a concatenation
    # (an ``Add`` with no aggregate kind) was not a list here, so ``ys`` had
    # no bound, took the planner's elementwise extent (2,) and ``len(ys)``
    # read 1 for 5 (clean audit).  Rule: a bounded concatenation is a list
    # operand like any other; settle the bounds to a fixed point.
    concatenations: set[int] = set()
    changed = True
    while changed:
        changed = False
        for node_id, data in graph_obj.nodes(data=True):
            if str(data.get("type") or data.get("op") or "") not in {"Add", "add"}:
                continue
            operands = [int(parent) for parent, role in data.get("parents") or ()
                        if str(role) in {"lhs", "rhs"} and int(parent) in graph_obj]
            if len(operands) != 2 or not all(
                (graph_obj.nodes[operand].get("attributes") or {}).get("aggregate_kind") == "list"
                or int(operand) in concatenations
                for operand in operands
            ):
                continue
            operand_bounds = [sequence_bound(operand) for operand in operands]
            value_id = int(data.get("value_id", node_id))
            if None not in operand_bounds and value_id not in bounds:
                bounds[value_id] = sum(operand_bounds)
                concatenations.add(int(node_id))
                changed = True
    # Program: ``d = {}; for i in range(n % 4 + 1): d[i % 2] = i`` (fuzzer
    # p1_27).  Wrong record: a local dict whose key universe the survey
    # cannot bound was neither bounded nor unbounded, so C refused it as a
    # private sequence with no capacity contract.  Rule: a dict born here
    # (a display) with no proven key universe that never escapes is the
    # function's own storage and takes the dynamic contract, as a list does.
    local_dicts = {
        int(node_id) for node_id, data in graph_obj.nodes(data=True)
        if isinstance(data.get("expr_obj"), (ast.Dict, ast.Set))
    }
    # Program: ``ys = {x * n for x in [4, 0, 8]}; ys.add(n + 5)`` (scratch
    # cs_a).  Wrong record: an unrolled set comprehension is born here like
    # a set display (``_list_literal_mutations``: one clear, one ``add``
    # per row) but was no local set, so once ``add`` grew it no rule gave
    # it a capacity and C refused it.  Rule: an unrolled set comprehension
    # that a later effect grows is a local set exactly as a set display is.
    # One nothing grows keeps its rows' exact descriptor (scratch cs_f:
    # ``{x % 2 + n for x in [1, 3, 5]}`` read 902 for 101 on the dynamic
    # contract).
    grown_sets = {int(mutation.sequence_value_id)
                  for mutation in _sequence_append_call_mutations(graph_obj)
                  if mutation.operator != "clear"}
    local_dicts.update(
        int(data.get("value_id", node_id))
        for node_id, data in graph_obj.nodes(data=True)
        for attributes in ((data.get("attributes") or {}),)
        if attributes.get("materialization_kind") == "unrolled_loop"
        and attributes.get("aggregate_kind") == "set"
        and int(data.get("value_id", node_id)) in grown_sets
    )
    escaped_dicts = {
        parent for parent in _call_escaped_argument_ids(graph_obj)
        if parent in local_dicts
    }
    # Program: ``xs = [2] * (n % 4 + 1)`` (fuzzer p3_54).  Wrong record: a
    # replication with a runtime count took the literal's one-row extent as
    # its capacity, so the fill ran out of rows.  Rule: a replication born
    # here whose count is not a literal is unbounded (the dynamic contract;
    # its fill reserves the count); a literal count bounds it exactly.
    mutated = {int(mutation.sequence_value_id)
               for mutation in _sequence_append_call_mutations(graph_obj)}
    local_replications = set()
    for node_id, data in graph_obj.nodes(data=True):
        if (data.get("attributes") or {}).get("producer_kind") != "sequence_replication":
            continue
        operands = [int(parent) for parent, role in data.get("parents") or ()
                    if str(role) in {"lhs", "rhs"} and int(parent) in graph_obj]
        rows = [operand for operand in operands
                if (graph_obj.nodes[operand].get("attributes") or {}).get(
                    "aggregate_kind") in {"list", "tuple"}]
        counts = [operand for operand in operands if operand not in rows]
        if len(rows) != 1 or len(counts) != 1:
            continue
        count_literal = (graph_obj.nodes[counts[0]].get("attributes") or {}).get("value")
        leaves = (graph_obj.nodes[rows[0]].get("attributes") or {}).get(
            "aggregate_leaf_value_ids", ())
        value_id = int(data.get("value_id", node_id))
        if (isinstance(count_literal, int) and not isinstance(count_literal, bool)
                and value_id not in mutated and value_id not in bounds):
            bounds[value_id] = max(0, int(count_literal)) * len(tuple(leaves))
        elif value_id not in bounds:
            local_replications.add(value_id)
    escaped_replications = {
        parent for parent in _call_escaped_argument_ids(graph_obj)
        if parent in local_replications
    }
    # Program: ``rep = [9]; if k > 2: rep = [k, 1, 2]`` then ``len(rep)``
    # (corpus c440).  Wrong record: a list rebound on a branch is carried by
    # copying the arm's list into the initial list's storage
    # (``carried_sequence_aliases``), but that copy is no mutation this
    # survey reads, so ``[]`` kept the bound of its own display, the copy
    # found the arena full, and ``len(rep)`` read 0 (silently).  Rule: the
    # storage holds at most the larger of its own rows and each copied
    # arm's; a copied arm with no bound, or later growth of the storage,
    # leaves it to the dynamic contract.
    copies: dict[int, set[int]] = {}

    def carried_copies(block):
        if block is None:
            return
        for alias in getattr(block, "carried_sequence_aliases", ()) or ():
            for source in (int(alias[0]), int(alias[1])):
                if source != int(alias[2]):
                    copies.setdefault(int(alias[2]), set()).add(source)
        for child in getattr(block, "blocks", ()) or ():
            carried_copies(child)
        for arm in ("condition", "body", "orelse", "callee"):
            carried_copies(getattr(block, arm, None))

    if control is not None:
        carried_copies(control.root)
    grown = {
        int(mutation.sequence_value_id)
        for mutation in _sequence_append_call_mutations(graph_obj)
        if mutation.operator != "clear"
    }
    copies_unbounded: set[int] = set()
    settled = False
    while not settled:
        settled = True
        for destination, sources in copies.items():
            own = sequence_bound(destination)
            # A list merged from displays of different lengths owns no rows
            # of its own: its arms' copies are all it holds (c654).
            if own is None and _list_merged_from_branches(
                graph_obj, destination,
            ):
                own = 0
            if own is None:
                continue
            source_bounds = [sequence_bound(source) for source in sources]
            if destination in grown or None in source_bounds:
                bounds.pop(destination, None)
                copies_unbounded.add(destination)
                continue
            widest = max(own, *source_bounds)
            if widest != bounds.get(destination):
                bounds[destination] = widest
                settled = False
    unbounded = frozenset(
        key for key in local_lists if key not in bounds and key not in escaped
    ) | frozenset(
        key for key in unbounded_snapshots if key not in escaped_snapshots
    ) | frozenset(
        key for key in local_replications if key not in escaped_replications
    ) | frozenset(
        key for key in local_dicts
        if key not in bounds and key not in escaped_dicts
    ) | frozenset(
        key for key in copies_unbounded
        if key not in _call_escaped_argument_ids(graph_obj)
    )
    # c632: the exact imported array constructor was concorded to a local
    # sequence materializer before extraction.  Its empty allocation owns
    # storage even when the next operation stores it into a record field;
    # the field read is a borrowed view and must not mint another owner.
    owned_empty_arrays = frozenset(
        int(data.get("value_id", node_id))
        for node_id, data in graph_obj.nodes(data=True)
        for attributes in (data.get("attributes") or {},)
        if attributes.get("aggregate_kind") == "array"
        and attributes.get("producer_kind") == "sequence_materialization"
        and tuple(role for _parent, role in data.get("parents") or () if role != "callee")
        == ("arg:0",)
    )
    # Vocabulary.__init__ hands its empty lookup dictionaries to exact
    # declared record slots. That handoff transfers the constructor's owned
    # storage; treating it as an escaped borrowed argument left every later
    # setdefault at capacity zero. Keep the allocation in its native context,
    # as for an empty typed array, and let the established heap contract wait
    # for its proved column layouts before binding pointer cells.
    from .concordance_declarations import CLASS_FIELD_DECLARATION
    from .identity_concordance import Ref

    owned_record_initializers = {
        int(parent)
        for _node_id, target in graph_obj.nodes(data=True)
        if str(target.get("type") or target.get("op") or "").casefold() == "setattr"
        and isinstance((target.get("attributes") or {}).get("field_declaration_cell"), Ref)
        and (target.get("attributes") or {})["field_declaration_cell"].page
            == CLASS_FIELD_DECLARATION
        for parent, role in target.get("parents") or ()
        if str(role) == "value"
    }
    owned_record_dicts = frozenset(
        int(data.get("value_id", node_id))
        for node_id, data in graph_obj.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Dict)
        and not data["expr_obj"].keys
        and int(node_id) in owned_record_initializers
    )
    return bounds, unbounded | owned_empty_arrays | owned_record_dicts

def _source_extraction_identity(
    graph_obj: Any,
    node_id: int,
) -> str | None:
    """Return the exact extraction identity retained by one source graph node."""

    attributes = (
        graph_obj.nodes.get(int(node_id), {}).get("attributes") or {}
    )
    identity = attributes.get("extraction_identity")
    return None if identity is None else str(identity)
def _linked_sequence_propagation_kind(
    mapped_arguments: Iterable[Any],
) -> str | None:
    """Return the exact ABI provenance that lets sequence storage escape."""

    accounting = tuple(
        dict(argument.accounting or {}) for argument in mapped_arguments
    )
    if any(item.get("program_abi_parameter") for item in accounting):
        return "program_abi_parameter"
    returned_record_identities = {
        (
            item.get("program_abi_record"),
            item.get("program_abi_keyed_owner")
            or str(item.get("program_abi_field")).split(".", 1)[0],
            item.get("returned_record_storage"),
        )
        for item in accounting
        if item.get("program_abi_record") is not None
        and item.get("program_abi_field") is not None
        and item.get("returned_record_storage") is not None
    }
    if accounting and len(returned_record_identities) == 1:
        return "exact_returned_record_storage"
    return None


def source_sort_contract(graph_obj: Any, call_node_id: int):
    """Return the exact source cell and fact for ``list.sort``/``sorted``.

    Direct row projections and resident mapping lookups are expressible in
    repository SSA without retaining a Python callable.  Every other keyword
    form remains explicitly unsupported rather than being silently ignored.
    """

    from .concordance_declarations import (
        SOURCE_SORT_CONTRACT, SourceSortContract,
    )
    from .identity_concordance import current_identity_book

    data = graph_obj.nodes.get(int(call_node_id), {})
    row = (data.get("attributes") or {}).get("source_sort_contract_row")
    if row is None:
        raise ValueError(
            f"sort call has no source contract row: value={call_node_id}"
        )
    cell = current_identity_book().latest_ref(
        SOURCE_SORT_CONTRACT, tuple(row)
    )
    if cell is None:
        raise ValueError(f"sort source contract has no latest cell: {row!r}")
    fact = current_identity_book().page(SOURCE_SORT_CONTRACT).cell(
        cell.row, cell.column,
    )
    if not isinstance(fact, SourceSortContract):
        raise ValueError(
            f"sort graph view has no exact source contract: {row!r}"
        )
    return fact, cell


def _discarded_dict_pop(graph_obj: Any, node_id: int, storage_aliases: Mapping[int, int]):
    """``(dict node, argument nodes, argument kind)`` for a dict ``pop``
    statement whose result nothing reads, else None.

    Program: ``d = {1: 10, 2: 20}; d.pop(1, None); d[3] = n`` then
    ``len(d)`` (corpus c708).  Wrong record: a dict ``pop`` statement had no
    mutation record outside a retained loop's state effects, so it lowered
    to an empty region and the key stayed (3 for 2; clean audit).  Rule: a
    dict ``pop`` whose result nothing reads is a keyed delete of its key, as
    the loop lowering already states it: lookup-and-delete, an absent key
    tolerated only when a default is given.  A statement after a store in
    the same body names the store's version of the dict: the same storage,
    by the storage aliases.
    """

    data = graph_obj.nodes[node_id]
    expression = data.get("expr_obj")
    if not (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and expression.func.attr == "pop"
    ):
        return None
    parents = tuple(data.get("parents") or ())
    receiver = next((
        int(parent) for parent, role in parents
        if str(role) == "operand" and int(parent) in graph_obj
    ), None)
    arguments = tuple(
        int(parent) for parent, role in parents
        if str(role).startswith("arg:") and int(parent) in graph_obj
    )
    if receiver is None or len(arguments) not in {1, 2}:
        return None
    receiver = int(storage_aliases.get(receiver, receiver))
    if (graph_obj.nodes[receiver].get("attributes") or {}).get(
        "aggregate_kind"
    ) != "dict":
        return None
    for successor in graph_obj.successors(node_id):
        # A statement's only consumer is the control construct that holds it
        # as a ``body``/``orelse`` member.
        if str(graph_obj.nodes[successor].get("type") or "") not in {
            "For", "While", "If"
        } or any(
            str(role) not in {"body", "orelse"}
            for parent, role in (
                graph_obj.nodes[successor].get("parents") or ()
            )
            if int(parent) == int(node_id)
        ):
            return None
    return receiver, arguments, (
        "mapping_pop_default_none" if len(arguments) == 2 else "mapping_pop"
    )


def _sequence_append_call_mutations(graph_obj: Any):
    """Recover authored resident sequence calls as lexical effects."""

    from .control_source import ControlExpression, ControlSequenceMutation
    from .string_table import string_token

    def retained_argument_expressions(
        record: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        expression = record.get("expression")
        if not isinstance(expression, ast.Call) or len(expression.args) != 1:
            return ()
        argument = expression.args[0]
        if not isinstance(argument, ast.Constant) or not isinstance(
            argument.value, (str, bool, int, float)
        ):
            return ()
        literal = (
            string_token(argument.value)
            if isinstance(argument.value, str)
            else argument.value
        )
        # Structural folding may remove the literal's graph node after this
        # source-effect snapshot. Keep its value on the mutation itself so
        # control lowering emits a local constant rather than inventing an
        # unexplained function formal for the vanished node id.
        return (ControlExpression("const", literal=literal),)

    sequence_kinds = {"list", "set", "bytes", "bytearray", "array"}
    mutations = []
    from .concordance_declarations import (
        SOURCE_SORT_CONTRACT, SourceSortContract,
    )
    from .identity_concordance import current_identity_book
    live_value_ids = {
        int(data.get("value_id", node_id))
        for node_id, data in graph_obj.nodes(data=True)
    }
    sort_page = current_identity_book().page(SOURCE_SORT_CONTRACT)
    for row in _source_contract_rows(graph_obj, sort_page):
        if len(row) != 2 or int(row[1]) not in live_value_ids:
            continue
        fact = sort_page.latest(row)
        if not isinstance(fact, SourceSortContract):
            raise TypeError(f"invalid source sort contract {fact!r}")
        if fact.destination_value_id is None:
            raise ValueError(f"source sort contract has no destination: {row!r}")
        arguments = tuple(map(int, fact.argument_value_ids))
        argument_kind = str(fact.argument_kind)
        if str(fact.operation) == "sorted":
            if fact.source_value_id is None:
                raise ValueError(f"sorted contract has no source: {row!r}")
            arguments = (int(fact.source_value_id), *arguments)
            argument_kind = f"sorted_{argument_kind}"
        mutations.append(ControlSequenceMutation(
            sequence_value_id=int(fact.destination_value_id),
            operator=str(fact.operation),
            argument_value_ids=arguments,
            effect_node_id=int(row[1]),
            policy=str(fact.policy),
            argument_kind=argument_kind,
            sort_key_column=fact.key_column,
            sort_reverse=bool(fact.reverse),
            source_contract_cell=current_identity_book().latest_ref(
                SOURCE_SORT_CONTRACT, row,
            ),
        ))
    for node_id, record in sorted(
        (graph_obj.graph.get("source_sequence_mutation_records") or {}).items(),
        key=lambda item: int(item[0]),
    ):
        operation = str(record["operator"])
        if operation == "sort":
            continue
        argument_kind = str(record.get("argument_kind", "value"))
        argument_ids = tuple(map(int, record["argument_value_ids"]))
        key_column = None
        reverse = False
        mutations.append(ControlSequenceMutation(
            sequence_value_id=int(record["sequence_value_id"]),
            operator=operation,
            argument_value_ids=argument_ids,
            effect_node_id=int(node_id),
            policy=record.get("policy"),
            argument_kind=argument_kind,
            argument_expressions=retained_argument_expressions(record),
            extraction_identity=_source_extraction_identity(
                graph_obj, node_id,
            ),
            sort_key_column=key_column,
            sort_reverse=reverse,
        ))
    mutations.extend(_source_mapping_mutations(graph_obj))
    mutations.extend(_source_sequence_snapshots(graph_obj))
    mutations.extend(_source_sequence_slices(graph_obj))
    recorded_effect_ids = {
        int(mutation.effect_node_id) for mutation in mutations
    }
    storage_aliases = _loop_carried_storage_aliases(graph_obj)
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr in {"append", "add", "clear", "pop"}
        ):
            continue
        if int(node_id) in recorded_effect_ids:
            continue
        parents = tuple(data.get("parents") or ())
        destination_node = next((
            int(parent) for parent, role in parents
            if str(role) == "operand" and int(parent) in graph_obj
        ), None)
        arguments = tuple(
            int(parent) for parent, role in parents
            if str(role).startswith("arg:") and int(parent) in graph_obj
        )
        operation = str(expression.func.attr)
        arity = 0 if operation == "clear" else 1
        discarded_pop = (
            _discarded_dict_pop(graph_obj, int(node_id), storage_aliases)
            if operation == "pop" else None
        )
        if operation == "pop":
            if discarded_pop is None:
                continue
            destination_node, arguments, pop_argument_kind = discarded_pop
            arity = len(arguments)
        destination = (
            graph_obj.nodes[destination_node]
            if destination_node is not None else {}
        )
        attributes = destination.get("attributes") or {}
        pops_dict = discarded_pop is not None
        if destination_node is None or len(arguments) != arity:
            continue
        # Program: ``cells = {1: 10, 2: 20}; if n: cells.clear();
        # return len(cells)``.  Wrong record: a dict's clear was not a
        # mutation at all (dicts were outside ``sequence_kinds``), so the
        # table kept its rows.  Rule: clearing a dict is the same resident
        # clear as a list's, on the dict's unique-keyed table.
        clears_dict = (
            operation == "clear" and attributes.get("aggregate_kind") == "dict"
        )
        if (
            attributes.get("aggregate_kind") not in sequence_kinds
            and not clears_dict and not pops_dict
        ):
            continue
        argument_kind = "value"
        if pops_dict:
            argument_kind = pop_argument_kind
        argument_ids = tuple(
            int(graph_obj.nodes[argument].get("value_id", argument))
            for argument in arguments
        )
        key_column = None
        reverse = False
        mutations.append(ControlSequenceMutation(
            sequence_value_id=int(destination.get(
                "value_id", destination_node
            )),
            operator=operation,
            argument_value_ids=argument_ids,
            effect_node_id=int(data.get("value_id", node_id)),
            policy=(
                "unique"
                if attributes.get("aggregate_kind") in {"set", "dict"}
                else "duplicates"
            ),
            argument_kind=argument_kind,
            extraction_identity=_source_extraction_identity(
                graph_obj, node_id,
            ),
            sort_key_column=key_column,
            sort_reverse=reverse,
        ))
    nodes_by_value = {
        int(data.get("value_id", node_id)): data
        for node_id, data in graph_obj.nodes(data=True)
    }
    from .concordance_declarations import (
        SOURCE_AGGREGATE_LAYOUT,
        SourceAggregateLayout,
    )
    aggregate_scope = graph_obj.graph.get("source_aggregate_layout_scope")
    aggregate_page = current_identity_book().page(SOURCE_AGGREGATE_LAYOUT)

    def aggregate_members(value_id: int) -> tuple[int, ...]:
        """Return the reducer-owned immediate row layout for one tuple.

        Graph attributes are a projection/debug view and can disappear at a
        copy boundary.  SOURCE_AGGREGATE_LAYOUT is the immutable authority for
        member order, including a tuple selected by conditional control.  Keep
        nested resident sequences as one member: their descriptor handle is a
        row column, not a request to flatten runtime-length storage.
        """

        if aggregate_scope is None:
            return ()
        layout = aggregate_page.latest((aggregate_scope, int(value_id)))
        if not isinstance(layout, SourceAggregateLayout):
            return ()
        if str(layout.aggregate_kind) != "tuple":
            return ()
        return tuple(int(member.value_id) for member in layout.members)

    expanded_mutations = []
    for mutation in mutations:
        arguments: list[int] = []
        for value_id in mutation.argument_value_ids:
            attributes = (
                nodes_by_value.get(int(value_id), {}).get("attributes") or {}
            )
            leaves = tuple(map(
                int, attributes.get("aggregate_leaf_value_ids", ())
            ))
            if not leaves:
                leaves = aggregate_members(int(value_id))
            abi = (graph_obj.graph.get('linked_value_abi') or {}).get(int(value_id), {})
            if (attributes.get("aggregate_kind") == "tuple" and leaves
                    and not mutation.argument_kind.startswith((
                        "mapping_", "sort_key_",
                    ))
                    and not (abi.get('storage') == 'reference' and abi.get('dtype') == 'opaque_ref')):
                arguments.extend(leaves)
            else:
                arguments.append(int(value_id))
        # Program: ``xs = [5, n, 2]; xs.insert(0, 8)`` (scratch c142).
        # Wrong record: insert's (index, value) arguments were read as a
        # two-column row, and the shell declared ``xs`` two columns wide.
        # Rule: only an operator whose arguments ARE the row (append, add,
        # remove, discard) writes a row; any other operator keeps its kind.
        row_operator = str(mutation.operator) in {
            "append", "add", "remove", "discard",
        }
        expanded_mutations.append(replace(
            mutation,
            argument_value_ids=tuple(arguments),
            argument_kind=(
                "row" if row_operator and len(arguments) > 1
                and not mutation.argument_kind.startswith((
                    "mapping_", "sort_key_",
                ))
                else mutation.argument_kind
            ),
        ))
    return tuple(expanded_mutations)


def _joined_byte_sequence_ids(
    graph_obj: Any,
    *,
    call_result_kinds: Mapping[int, str],
    declared_sequence_ids: Iterable[int],
    structural_byte_sequence_ids: Iterable[int] = (),
    transformed_call_source_ids: Iterable[int] = (),
) -> tuple[int, ...]:
    """Find resident ``list[bytes]`` identities needing dual ABI views.

    The authored outer list owns an element count, while its eventual empty
    byte join owns a flattened byte stream. Parameters already express these
    as ``row_count``/``join_bytes`` source transforms; this recovers the same
    representation for locals without inspecting Python values.
    """

    declared = set(map(int, declared_sequence_ids))
    node_by_value = {
        int(data.get("value_id", node_id)): data
        for node_id, data in graph_obj.nodes(data=True)
    }
    byte_values = {
        int(value_id)
        for value_id, data in node_by_value.items()
        if str((data.get("attributes") or {}).get("aggregate_kind"))
        in {"bytes", "bytearray"}
    }
    byte_values.update(
        int(value_id)
        for value_id, kind in call_result_kinds.items()
        if str(kind) in {"bytes", "bytearray"}
    )
    byte_values.update(map(int, structural_byte_sequence_ids))
    seeds: set[int] = set(map(int, transformed_call_source_ids))
    for value_id, data in node_by_value.items():
        attributes = data.get("attributes") or {}
        if attributes.get("aggregate_kind") != "list":
            continue
        leaves = tuple(map(
            int, attributes.get("aggregate_leaf_value_ids", ())
        ))
        if not leaves:
            leaves = tuple(
                int(graph_obj.nodes[parent].get("value_id", parent))
                for parent, role in (data.get("parents") or ())
                if str(role).startswith("elts") and parent in graph_obj
            )
        if leaves and all(int(leaf) in byte_values for leaf in leaves):
            seeds.add(int(value_id))
    append_mutations = _sequence_append_call_mutations(graph_obj)
    for mutation in append_mutations:
        if (
            mutation.operator == "append"
            and len(mutation.argument_value_ids) == 1
            and int(mutation.argument_value_ids[0]) in byte_values
        ):
            seeds.add(int(mutation.sequence_value_id))

    if os.environ.get("TURING_DEBUG_JOINED_SEQUENCE"):
        print(
            "DEBUGJOINED "
            f"fn={graph_obj.graph.get('function_name')} "
            f"bytes={tuple(sorted(byte_values))!r} "
            f"seeds={tuple(sorted(seeds))!r} "
            f"mutations={tuple((int(item.sequence_value_id), tuple(map(int, item.argument_value_ids))) for item in append_mutations)!r} "
            f"transformed={tuple((int(value_id), str((node_by_value.get(int(value_id), {}).get('attributes') or {}).get('binding_name')), ast.dump(node_by_value.get(int(value_id), {}).get('expr_obj'), include_attributes=False) if isinstance(node_by_value.get(int(value_id), {}).get('expr_obj'), ast.AST) else None) for value_id in sorted(map(int, transformed_call_source_ids)))!r} "
            f"identities={tuple((str(name), tuple(map(int, history))) for name, history in (graph_obj.graph.get('identity_table') or {}).items() if str(name) in {'types', 'entries'})!r} "
            f"declared={tuple(sorted(declared))!r}",
            file=sys.stderr,
        )

    mutation_destinations = {
        int(item.sequence_value_id) for item in append_mutations
    }
    joined = {
        int(value_id)
        for value_id in seeds
        if (
            int(value_id) in declared
            or int(value_id) in mutation_destinations
            or (
                node_by_value.get(int(value_id), {}).get("attributes") or {}
            ).get("aggregate_kind") == "list"
            or isinstance(
                (
                    (graph_obj.nodes.get(int(value_id), {}).get("attributes")
                     or {}).get(
                        "value",
                        graph_obj.nodes.get(int(value_id), {}).get("constant"),
                    )
                ),
                list,
            )
        )
    }
    for history in (graph_obj.graph.get("identity_table") or {}).values():
        identities = set(map(int, history))
        if identities & seeds:
            joined.update(identities & declared)
    return tuple(sorted(joined))


def _scalar_constant_member(row: Mapping[str, Any]) -> Any:
    """The literal a folded aggregate member carries, or None.

    Only a Constant node's own recorded literal counts; an aggregate member
    (a nested tuple/list) is not a scalar literal.
    """

    if str(row.get("type")) != "Constant":
        return None
    attributes = row.get("attributes") or {}
    if attributes.get("aggregate_kind") is not None:
        return None
    expression = row.get("expr_obj")
    if isinstance(expression, ast.Constant):
        return expression.value
    literal = attributes.get("value", row.get("constant"))
    return literal if isinstance(literal, (bool, int, float, str)) else None


def _list_merged_from_branches(graph_obj: Any, node_id: int) -> bool:
    """Whether a list merge is first bound on two branches (a handle merge).

    Both arms are lists and the merge is first bound there (no initial
    list).  Python's merged name is the taken arm's own list whatever the
    arms' lengths (displays of equal or different length, a parameter,
    another local list), so every such merge is a handle onto the taken
    arm's storage and publishes no member layout (corpus c733, c734,
    c870-c874).
    """

    data = graph_obj.nodes.get(int(node_id)) or {}
    attributes = data.get("attributes") or {}
    # ``aggregate_phi`` is the statement/name merge, ``aggregate_merge`` the
    # conditional expression's (``t = a if c else [3, 4, 5]``): the same
    # aggregate semantics (reducer, expression-form conditional selection).
    if (
        attributes.get("producer_kind") not in {"aggregate_phi", "aggregate_merge"}
        or attributes.get("aggregate_kind") != "list"
        or attributes.get("initial_value_id") is not None
        or attributes.get("aggregate_leaf_value_ids") is not None
    ):
        return False
    arms = tuple(
        graph_obj.nodes[parent]
        for parent, role in data.get("parents") or ()
        if str(role) in {"body", "orelse"} and parent in graph_obj
    )
    return len(arms) == 2 and all(
        (arm.get("attributes") or {}).get("aggregate_kind") == "list"
        for arm in arms
    )


def _returned_list_merge(graph_obj: Any, node_id: int) -> bool:
    """Whether the merge's value is one of the function's returned values."""

    value_id = int((graph_obj.nodes.get(int(node_id)) or {}).get("value_id", node_id))
    return value_id in {
        int(returned)
        for name in tuple(graph_obj.graph.get("function_outputs") or ())
        for returned in tuple(
            (graph_obj.graph.get("identity_table") or {}).get(str(name), ())
        )
    }


def _arms_private_displays(graph_obj: Any, node_id: int) -> bool:
    """Whether every arm of a list merge is a display only the merge reads."""

    data = graph_obj.nodes[int(node_id)]
    arms = tuple(
        int(parent) for parent, role in data.get("parents") or ()
        if str(role) in {"body", "orelse"} and int(parent) in graph_obj
    )
    for arm in arms:
        attributes = graph_obj.nodes[arm].get("attributes") or {}
        if (
            attributes.get("producer_kind") != "aggregate"
            or attributes.get("aggregate_leaf_value_ids") is None
        ):
            return False
        for reader, reader_data in graph_obj.nodes(data=True):
            if int(reader) != int(node_id) and any(
                int(parent) == arm
                for parent, _role in reader_data.get("parents") or ()
            ):
                return False
    return True


def _returned_copy_merge(graph_obj: Any, node_id: int) -> bool:
    """A returned unequal-arm list merge whose arms are private displays.

    It owns the result storage and each arm is copied into it; every other
    unequal-arm merge is a handle onto the taken arm's storage.
    """

    return (
        _list_merged_from_branches(graph_obj, int(node_id))
        and _returned_list_merge(graph_obj, int(node_id))
        and _arms_private_displays(graph_obj, int(node_id))
    )


def _handle_merge_storage_roots(graph_obj: Any) -> dict[int, int]:
    """One storage key for a list merged from branches and each of its arms.

    A list merge is a handle onto the taken arm's own storage
    (``_lower_sequence_handle_merge``), so the merge and its arms name ONE
    storage: a call that changes ``t`` changes ``a`` where ``t`` is ``a``.
    Program: ``t = a if n > 2 else [7, n, 2]`` as an if/else, ``grow(t, k)``
    then ``len(a) * 100 + a[0]`` (c739).  Wrong record: readers and effects
    were keyed by the receiver's own value id, so the reducer fused the
    reads of ``a`` into a region placed before the call (silent).  Rule: the
    key of every member of a handle component is one value id (the smallest),
    read from the graph's own merge edges.
    """

    parent: dict[int, int] = {}

    def find(value_id):
        parent.setdefault(value_id, value_id)
        while parent[value_id] != value_id:
            parent[value_id] = parent[parent[value_id]]
            value_id = parent[value_id]
        return value_id

    for node_id, data in graph_obj.nodes(data=True):
        if not _list_merged_from_branches(graph_obj, int(node_id)):
            continue
        if _returned_copy_merge(graph_obj, int(node_id)):
            continue
        merged = int(data.get("value_id", node_id))
        for arm, role in data.get("parents") or ():
            if str(role) not in {"body", "orelse"} or arm not in graph_obj:
                continue
            left = find(int(graph_obj.nodes[arm].get("value_id", arm)))
            right = find(merged)
            if left != right:
                parent[max(left, right)] = min(left, right)
    roots = {value_id: find(value_id) for value_id in list(parent)}
    return roots


def _list_rebound_on_one_branch(graph_obj: Any, node_id: int) -> bool:
    """Whether a list merge is one branch rebinding its initial list.

    Program: the tire's ``metrics.unresolved_report = list(lines)`` on one
    branch of ``step_with_dt_control_used`` (corpus c440).  Such a merge
    (an ``aggregate_phi`` whose other arm is its initial list) is carried
    by the conditional, which copies the arm into the initial list's
    storage (``carried_sequence_aliases``); the merge names that storage.
    Both arms rebinding, or an initial empty display (whose rows have no
    element type before the copy), is not this shape.
    """

    data = graph_obj.nodes.get(int(node_id)) or {}
    attributes = data.get("attributes") or {}
    initial = attributes.get("initial_value_id")
    if (
        attributes.get("producer_kind") != "aggregate_phi"
        or attributes.get("aggregate_kind") != "list"
        or initial is None
    ):
        return False
    arms = tuple(
        int(graph_obj.nodes[parent].get("value_id", parent))
        for parent, role in data.get("parents") or ()
        if str(role) in {"body", "orelse"} and parent in graph_obj
    )
    if len(arms) != 2 or arms.count(int(initial)) != 1:
        return False
    initial_nodes = [
        candidate for candidate, candidate_data in graph_obj.nodes(data=True)
        if int(candidate_data.get("value_id", candidate)) == int(initial)
    ]
    if len(initial_nodes) != 1:
        return False
    initial_node = graph_obj.nodes[initial_nodes[0]]
    literal = initial_node.get("constant")
    if literal is None:
        literal = (initial_node.get("attributes") or {}).get("value")
    expression = initial_node.get("expr_obj")
    return not (
        isinstance(literal, list) and not literal
        or isinstance(expression, ast.List) and not expression.elts
    )


def _list_literal_mutations(graph_obj: Any, sequence_ids: Iterable[int]) -> tuple[Any, ...]:
    """Publish ordinary list rows at their exact authored construction site.

    ``lines=[10.0]; lines.append(20.0)`` retained List->Constant edges but
    received only the frame's zero-length initializer. A list birth is a
    lexical clear followed by its ordered rows, just like a dict birth. The
    accepted joined-byte and opaque-object representations retain their own
    materializers; callers pass only ordinary declared sequence identities.
    """
    from .control_source import ControlExpression, ControlSequenceMutation
    from .concordance_declarations import (
        SOURCE_AGGREGATE_LAYOUT,
        SourceAggregateLayout,
    )
    from .identity_concordance import current_identity_book
    from .string_table import string_token

    accepted = set(map(int, sequence_ids))
    # The rows of a list of lists are born at the outer display and own no
    # declaration of their own (``list_row_displays``).
    for outer_id, rows in list_row_displays(graph_obj).items():
        if outer_id in accepted:
            accepted.update(row for row, _node in rows)
    aggregate_scope = graph_obj.graph.get("source_aggregate_layout_scope")

    def layout_members_of(layout):
        return (
            isinstance(layout, SourceAggregateLayout)
            and str(layout.aggregate_kind) in {"list", "set"}
            and bool(layout.members)
        )

    def refuse_shared_merge_arms(merged_id, merged):
        from .glsl_deployment_strategy import _mutated_sequence_node_ids

        changed = _mutated_sequence_node_ids(graph_obj)
        changed_in_place = _mutated_sequence_node_ids(
            graph_obj, include_merge_aliases=False,
        )
        arms = tuple(
            int(parent) for parent, role in merged.get("parents") or ()
            if str(role) in {"body", "orelse"} and int(parent) in graph_obj
        )
        for arm in arms:
            readers = tuple(
                int(reader) for reader, reader_data in graph_obj.nodes(data=True)
                if int(reader) != merged_id
                and any(int(parent) == arm for parent, _role in reader_data.get("parents") or ())
            )
            if arm in changed_in_place or (readers and arm in changed):
                raise ValueError(
                    f"collection {merged_id} merges the collection {arm} "
                    "from a branch, which changes or is shared after its "
                    "display; a list or set bound on two branches is born "
                    "from its arms' members only while each arm is its own "
                    "unchanged display"
                )
    aggregate_page = current_identity_book().page(SOURCE_AGGREGATE_LAYOUT)
    mutations = []
    nodes_by_value: dict[int, list[int]] = {}
    for candidate_id, candidate in graph_obj.nodes(data=True):
        nodes_by_value.setdefault(
            int(candidate.get("value_id", candidate_id)), []
        ).append(candidate_id)
    mutated_ids = {
        int(record["sequence_value_id"])
        for record in (
            graph_obj.graph.get("source_sequence_mutation_records") or {}
        ).values()
    }
    output_ids = {
        int(value_id)
        for name in tuple(graph_obj.graph.get("function_outputs") or ())
        for value_id in tuple(
            (graph_obj.graph.get("identity_table") or {}).get(str(name), ())
        )
    }
    # Unread: no successor except other unread collections, its identity
    # held by this node alone, never mutated and never an output.  A
    # replication read by nothing leaves its row display unread too.
    unread_candidates = {
        int(node_id)
        for node_id, data in graph_obj.nodes(data=True)
        if (data.get("attributes") or {}).get("aggregate_kind") in {"list", "set"}
        and nodes_by_value.get(int(data.get("value_id", node_id))) == [node_id]
        and int(data.get("value_id", node_id)) not in mutated_ids
        and int(data.get("value_id", node_id)) not in output_ids
    }
    unread_collections: set[int] = set()
    grew = True
    while grew:
        grew = False
        for candidate in unread_candidates - unread_collections:
            readers = {int(reader) for reader in graph_obj.successors(candidate)}
            readers.update(
                int(child) for child, _role in
                graph_obj.nodes[candidate].get("children") or ()
            )
            if readers <= unread_collections:
                unread_collections.add(candidate)
                grew = True
    for node_id, data in graph_obj.nodes(data=True):
        sequence_id = int(data.get("value_id", node_id))
        expression = data.get("expr_obj")
        attributes = data.get("attributes") or {}
        # Program: ``xs = [1, 4, 9]; ys = [x > n for x in xs]; any(ys)``
        # (corpus c240).  Wrong record: the comprehension over a folded
        # literal evaporated into a ListComp materializer whose rows are
        # ``materialized_value_ids``, but only displays were born, so the
        # resident arena kept its zero length and ``any``/``sum``/``len``
        # read an empty list (silently; the audit saw an operand with no
        # producer).  Rule: an unrolled list or set comprehension is born
        # like a display -- one lexical clear, then one append (``add``)
        # per materialized row, in iteration order -- as the unrolled dict
        # comprehension already is.
        # Program: ``xs = [1, n] * 3`` then ``xs[3]`` (corpus c249, lanes
        # plan C9); also ``[n] * 3``.  Wrong record: only a replication of
        # one literal member has a birth (the ``fill`` initialization in
        # ``fortran_c_shell``); any other row had none, so the product
        # lowered as a numerical ``Mul`` of the row's handle and its reads
        # went past it (silently; clean audit).  A birth by repeated member
        # appends would give one member producer several control blocks,
        # which the control book cannot own (as ``[n, n]`` is refused
        # below).  Rule: refuse such a replication by name.
        # Program: the tire's ``vector_cross`` (``left[..., 1]``; corpus
        # c500), whose indexing builds ``[slice(None)] * rank``.  Wrong
        # record: the index folded, so nothing reads the replication or
        # its row display ``[slice(None)]``; the replication was still
        # refused for its row operand (when folded too), and the display
        # was still born, appending a slice selector as a row of no native
        # dtype (C emission refused it, later crashed on dtype ``none``).
        # Rule: an unread collection has no birth to lower or refuse.
        if int(node_id) in unread_collections:
            continue
        if attributes.get("producer_kind") == "sequence_replication":
            operands = [
                int(parent) for parent, role in data.get("parents") or ()
                if str(role) in {"lhs", "rhs"} and int(parent) in graph_obj
            ]
            row_ids = [
                operand for operand in operands
                if (graph_obj.nodes[operand].get("attributes") or {}).get(
                    "aggregate_kind") in {"list", "tuple"}
            ]
            leaves = tuple(
                int(leaf)
                for row_id in row_ids
                for leaf in (graph_obj.nodes[row_id].get("attributes")
                             or {}).get("aggregate_leaf_value_ids", ())
            )
            # One member, literal or runtime, is born by the fill (``[n] *
            # 3``, corpus c334/c335: the fill reads the member's value).
            # Only a wider row has no birth.
            if len(row_ids) != 1 or len(leaves) != 1 or len(
                nodes_by_value.get(leaves[0], ())
            ) != 1:
                raise ValueError(
                    f"list {sequence_id} replicates a row other than one "
                    "member; a replication is lowered only as the fill of "
                    "one member"
                )
        if (sequence_id in accepted
                and attributes.get("materialization_kind") == "unrolled_loop"
                and attributes.get("aggregate_kind") in {"list", "set"}):
            is_set = attributes.get("aggregate_kind") == "set"
            rows = []
            for ordinal, row_id in enumerate(
                attributes.get("materialized_value_ids", ())
            ):
                producers = nodes_by_value.get(int(row_id), ())
                if len(producers) != 1:
                    raise ValueError(
                        f"comprehension {sequence_id} row {row_id} lacks one "
                        "exact producer"
                    )
                row = graph_obj.nodes[producers[0]]
                if (row.get("attributes") or {}).get(
                    "aggregate_kind"
                ) is not None:
                    raise ValueError(
                        f"comprehension {sequence_id} row {row_id} is an "
                        "aggregate; unrolled aggregate rows have no birth"
                    )
                literal = _scalar_constant_member(row)
                expressions = ()
                if isinstance(literal, (bool, int, float, str)):
                    literal = (
                        string_token(literal) if isinstance(literal, str)
                        else literal
                    )
                    expressions = (ControlExpression("const", literal=literal),)
                rows.append(ControlSequenceMutation(
                    sequence_value_id=sequence_id,
                    operator="add" if is_set else "append",
                    argument_value_ids=(int(row_id),),
                    effect_node_id=int(producers[0]),
                    policy="unique" if is_set else "duplicates",
                    argument_kind="value",
                    argument_expressions=expressions,
                    display_member=(int(node_id), int(ordinal)),
                ))
            mutations.append(ControlSequenceMutation(
                sequence_value_id=sequence_id, operator="clear",
                argument_value_ids=(), effect_node_id=int(node_id),
                policy="unique" if is_set else "duplicates",
            ))
            mutations.extend(rows)
            continue
        if (sequence_id not in accepted
                or attributes.get("producer_kind") not in {"aggregate", "aggregate_phi"}
                or attributes.get("materialization_kind") is not None):
            continue
        layout = (
            aggregate_page.latest((aggregate_scope, sequence_id))
            if aggregate_scope is not None else None
        )
        if attributes.get("producer_kind") == "aggregate_phi":
            # Program: ``if n > 2: t = [n, 1] else: t = [5, 6]`` then
            # ``return len(t) * 10 + t[0]`` (corpus c271).  Wrong record: a
            # list merged from its arms had member Phis but no birth, so
            # ``len(t)`` read a never-born arena (0) and appends after the
            # merge were lost (silent).  Rule: the merged list is its own
            # storage, born at the merge from its member Phis.  That copy is
            # the list only while every arm is a display nothing else reads
            # or changes; otherwise refuse by name.  A set merged the same
            # way (``s = {1, n} if n > 0 else {2, 3}``, corpus c274) had no
            # capacity and was refused at emission; its birth is the same,
            # under the unique ``add`` policy.
            if not layout_members_of(layout):
                # Program: the tire's ``metrics.unresolved_report =
                # list(lines)`` on one branch of
                # ``step_with_dt_control_used`` (corpus c440).  A list
                # rebound on one branch, the other keeping its initial
                # list, is carried by the conditional: it copies the arm
                # into the initial list's storage (``carried_sequence_
                # aliases``), and the merge names that storage.  Both arms
                # rebinding, or an initial empty display (whose rows have
                # no element type before the copy), stay refused.
                rebound_on_one_branch = _list_rebound_on_one_branch(
                    graph_obj, int(node_id),
                )
                if _list_merged_from_branches(graph_obj, int(node_id)):
                    # No birth here: the merge is a handle onto the taken
                    # arm's own storage, or (returned over private displays)
                    # the result storage each arm is copied into
                    # (``_promote_conditional_sequence_aliases``,
                    # ``_lower_sequence_handle_merge``).
                    continue
                if attributes.get("aggregate_kind") == "list" and not (
                    rebound_on_one_branch
                ):
                    # Program: ``def pick(n): if n > 2: r = [n] else: r =
                    # []; return r`` (scratch c972).  Arms of different
                    # lengths publish no member layout, and nothing else
                    # bore the merge, so the caller read an empty list.
                    raise ValueError(
                        f"list {int(node_id)} is bound to lists of different "
                        "shapes on two branches; a list merged from branches "
                        "is lowered only from equal-length displays"
                    )
                continue
            refuse_shared_merge_arms(int(node_id), data)
        # Program: ``s = {1, 2}; for i in ...: s.add(i * 3); len(s)``
        # (scratch c_s2).  Wrong record: only list displays were born, so a
        # set display's members were never inserted and ``len(s)`` counted
        # the loop's adds alone (silently; clean audit).  Rule: a set
        # display's birth is the same lexical clear followed by one ``add``
        # per member, in order, under the unique policy ``s.add`` uses.  A
        # set initialized from a compile-time table keeps that initializer.
        is_set = (
            isinstance(expression, ast.Set)
            or attributes.get("aggregate_kind") in {"set", "frozenset"}
        )
        if is_set and attributes.get("compile_time_sequence_items") is not None:
            continue
        layout_members = (
            tuple(int(member.value_id) for member in layout.members)
            if isinstance(layout, SourceAggregateLayout)
            and str(layout.aggregate_kind) in {"list", "tuple", "set", "frozenset"}
            else ()
        )
        # Program: ``xs = [5, 7, 11]; return xs[n]`` (c36_literal_index_param).
        # Wrong record: the structural fold turned the all-constant List into
        # a Constant node (no AST, no ``elts`` edges), and this birth keyed on
        # ``isinstance(expr_obj, ast.List)``, so the arena kept its zero
        # length and every ``xs[n]`` read 0.  Rule: the list birth is the
        # SOURCE_AGGREGATE_LAYOUT record's ordered members; the AST, when it
        # survives, only refines a member's spelling.
        if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
            if not expression.elts:
                continue
            elements = tuple(expression.elts)
        elif layout_members:
            elements = (None,) * len(layout_members)
        else:
            continue
        parents = tuple((int(parent), graph_obj.nodes[parent])
                        for parent, role in data.get("parents", ())
                        if str(role).startswith("elts") and parent in graph_obj)
        if layout_members and len(layout_members) != len(elements):
            raise ValueError(
                f"List {sequence_id} aggregate layout has "
                f"{len(layout_members)} member(s), source has "
                f"{len(elements)}"
            )
        rows = []
        for ordinal, element in enumerate(elements):
            member_id = (
                layout_members[ordinal] if layout_members else None
            )
            if element is None:
                exact = tuple(
                    (int(parent), graph_obj.nodes[parent])
                    for parent in nodes_by_value.get(int(member_id), ())
                )
            else:
                exact = tuple(
                    (parent, row) for parent, row in parents
                    if (
                        int(row.get("value_id", parent)) == int(member_id)
                        if member_id is not None
                        else row.get("expr_obj") is element
                    )
                )
            # Program: ``a = [n, 7, n, 7]; a[1] = n + 1`` (corpus c725).
            # Wrong record: one value used for two members has one producer
            # node and two ``elts`` edges, so both edges matched the second
            # occurrence's ordinal and the list was refused as lacking one
            # exact producer.  Rule: the k-th member that names a value is
            # that value's k-th edge, in source order; a value with fewer
            # edges than members is still refused.
            if (
                member_id is not None and element is not None
                and len(exact) > 1
                and not isinstance(element, ast.Starred)
            ):
                earlier = sum(
                    1 for position in range(ordinal)
                    if int(layout_members[position]) == int(member_id)
                )
                if earlier < len(exact):
                    exact = (exact[earlier],)
            if len(exact) != 1 or isinstance(element, ast.Starred):
                raise ValueError(
                    f"{'Set' if is_set else 'List'} {sequence_id} lacks one "
                    "exact element producer"
                )
            parent, row = exact[0]
            value_id = int(row.get("value_id", parent))
            row_attributes = row.get("attributes") or {}
            # Program: ``g = [[1, 2], [3, n]]; g[i][1]`` (corpus c732).  A
            # list row is a child sequence of its own, born by its own
            # display; the outer list stores the row's handle in the
            # child-table pool and copies the row's members into it
            # (``child_sequence`` append).  Set, dict and tuple rows have
            # no pool row layout and stay refused by name.
            if row_attributes.get("aggregate_kind") == "list" and not is_set:
                rows.append(ControlSequenceMutation(
                    sequence_value_id=sequence_id, operator="append",
                    argument_value_ids=(value_id,), effect_node_id=parent,
                    policy="duplicates", argument_kind="child_sequence",
                    display_member=(int(node_id), int(ordinal)),
                ))
                continue
            if row_attributes.get("aggregate_kind") in {
                "set", "frozenset", "dict",
            }:
                raise ValueError(
                    f"{'Set' if is_set else 'List'} {sequence_id} member "
                    f"{ordinal} is a {row_attributes['aggregate_kind']} "
                    "display; only list rows have a resident birth "
                    "(nested collections are not lowered)"
                )
            leaves = tuple(map(int, row_attributes.get(
                "aggregate_leaf_value_ids", ())))
            is_tuple = (
                isinstance(element, ast.Tuple) if element is not None
                else row_attributes.get("aggregate_kind") == "tuple"
            )
            arguments = (leaves if is_tuple and leaves else (value_id,))
            expressions = ()
            literal = (
                element.value if isinstance(element, ast.Constant)
                else _scalar_constant_member(row) if element is None
                else None
            )
            if isinstance(literal, (bool, int, float, str)):
                literal = string_token(literal) if isinstance(literal, str) else literal
                expressions = (ControlExpression("const", literal=literal),)
            rows.append(ControlSequenceMutation(
                sequence_value_id=sequence_id,
                operator="add" if is_set else "append",
                argument_value_ids=arguments, effect_node_id=parent,
                policy="unique" if is_set else "duplicates",
                argument_kind="row" if len(arguments) > 1 else "value",
                argument_expressions=expressions,
                display_member=(int(node_id), int(ordinal)),
            ))
        mutations.append(ControlSequenceMutation(
            sequence_value_id=sequence_id, operator="clear", argument_value_ids=(),
            effect_node_id=int(node_id),
            policy="unique" if is_set else "duplicates",
        ))
        mutations.extend(rows)
    return _list_rows_born_by_outer_display(graph_obj, tuple(mutations))


def list_row_displays(graph_obj: Any) -> dict[int, tuple[tuple[int, int], ...]]:
    """Each list display whose members are all list displays, by value id.

    The value is its rows in member order as ``(row value id, row node)``.
    Such a list is a nested table: its one column holds row handles and each
    row is a slice of the child-table pool.  A display with any member that
    is not a list display is not one (and is refused at birth if it mixes).
    """

    result: dict[int, tuple[tuple[int, int], ...]] = {}
    for node_id, data in graph_obj.nodes(data=True):
        attributes = data.get("attributes") or {}
        if attributes.get("aggregate_kind") != "list":
            continue
        members = tuple(
            int(parent) for parent, role in data.get("parents") or ()
            if str(role).startswith("elts") and parent in graph_obj
        )
        if not members:
            # An all-constant list of lists folds to one constant that keeps
            # its rows' value ids as its aggregate leaves.
            members = tuple(
                producer
                for leaf in attributes.get("aggregate_leaf_value_ids") or ()
                for producer in (next((
                    int(candidate) for candidate, candidate_data
                    in graph_obj.nodes(data=True)
                    if int(candidate_data.get("value_id", candidate))
                    == int(leaf)
                ), None),)
                if producer is not None
            )
            if len(members) != len(
                attributes.get("aggregate_leaf_value_ids") or ()
            ):
                continue
        if not members or not all(
            (graph_obj.nodes[member].get("attributes") or {}).get(
                "aggregate_kind"
            ) == "list"
            for member in members
        ):
            continue
        result[int(data.get("value_id", node_id))] = tuple(
            (int(graph_obj.nodes[member].get("value_id", member)), member)
            for member in members
        )
    return result


def dict_row_displays(graph_obj: Any) -> dict[int, tuple[tuple[int, int], ...]]:
    """Each dict display whose values are all list displays, by value id.

    The value is its rows in entry order as ``(row value id, row node)``.
    Such a mapping's value column holds child-pool row handles, as a
    ``setdefault(key, [])`` receiver's does.
    """

    result: dict[int, tuple[tuple[int, int], ...]] = {}
    for node_id, data in graph_obj.nodes(data=True):
        attributes = data.get("attributes") or {}
        if not (
            attributes.get("aggregate_kind") == "dict"
            and attributes.get("mapping_value_aggregate_kind") == "list"
            and isinstance(data.get("expr_obj"), ast.Dict)
        ):
            continue
        rows = tuple(
            (int(graph_obj.nodes[parent].get("value_id", parent)), int(parent))
            for parent, role in data.get("parents") or ()
            if str(role).startswith("values") and parent in graph_obj
        )
        if rows:
            result[int(data.get("value_id", node_id))] = rows
    return result


def list_row_pool_extents(graph_obj: Any) -> dict[int, tuple[int, int]]:
    """``(rows, stride)`` of each list of lists' child-table pool.

    The pool holds one fixed-stride slice per row: ``rows`` is the outer
    display's member count and ``stride`` the longest row display.  A row
    whose display length the source does not state is refused by name: its
    slice would have no proven extent.
    """

    extents: dict[int, tuple[int, int]] = {}
    for outer, rows in (
        *list_row_displays(graph_obj).items(),
        *dict_row_displays(graph_obj).items(),
    ):
        stride = 0
        for row, node in rows:
            data = graph_obj.nodes[node]
            attributes = data.get("attributes") or {}
            members = [
                parent for parent, role in data.get("parents") or ()
                if str(role).startswith("elts")
            ]
            count = (
                len(members) if members
                else len(attributes["aggregate_leaf_value_ids"])
                if attributes.get("aggregate_leaf_value_ids")
                else len(attributes["compile_time_sequence_items"])
                if attributes.get("compile_time_sequence_items") is not None
                else len(attributes["value"])
                if isinstance(attributes.get("value"), (list, tuple))
                else None
            )
            if count is None:
                raise ValueError(
                    f"list {row} is a row of list {outer} and its display "
                    "length is not stated by the source; a row slice has "
                    "no proven extent"
                )
            stride = max(stride, count)
        extents[int(outer)] = (len(rows), max(1, stride))
    return extents


def list_row_reads(graph_obj: Any) -> tuple[int, ...]:
    """Value ids of ``g[i]`` for each list of lists ``g``: a row read.

    A row read is a positional load of a row handle from the outer list's
    one column and selects the row's pool slice; the element reads that
    index it are positional loads from that slice.
    """

    outers = set(list_row_displays(graph_obj))
    # A dict display of lists: ``d[k]`` is its keyed lookup, the row a
    # selected pool slice.  (Iterating the dict yields keys, not rows.)
    mapping_outers = set(dict_row_displays(graph_obj))
    # ``g[i][1] = v`` is a new version of ``g`` (a loop may carry it): reads
    # of any version are reads of the same list of lists.
    aliases = _loop_carried_storage_aliases(graph_obj)
    for version in tuple(aliases):
        root = int(version)
        for _ in range(len(aliases) + 1):
            if root not in aliases:
                break
            root = int(aliases[root])
        if root in outers:
            outers.add(int(version))
        if root in mapping_outers:
            mapping_outers.add(int(version))
    reads = []
    for node_id, data in graph_obj.nodes(data=True):
        if str(data.get("op") or data.get("type") or "").casefold() != "indexed":
            continue
        base = next((
            int(parent) for parent, role in data.get("parents") or ()
            if str(role) == "base" and parent in graph_obj
        ), None)
        if base is not None and int(
            graph_obj.nodes[base].get("value_id", base)
        ) in (outers | mapping_outers):
            reads.append(int(data.get("value_id", node_id)))
    # ``for r in g``: the loop target is a row of the list of lists.
    for node_id, data in graph_obj.nodes(data=True):
        if str(data.get("op") or data.get("type") or "").casefold() != "for":
            continue
        iterable = next((
            int(parent) for parent, role in data.get("parents") or ()
            if str(role) == "iterable" and parent in graph_obj
        ), None)
        if iterable is None or int(
            graph_obj.nodes[iterable].get("value_id", iterable)
        ) not in outers:
            continue
        loop = data.get("expr_obj")
        if not (isinstance(loop, ast.For) and isinstance(loop.target, ast.Name)):
            continue
        reads.extend(
            int(child_data.get("value_id", child))
            for child, child_data in graph_obj.nodes(data=True)
            if str(child_data.get("op") or child_data.get("type") or "").casefold() == "input"
            and (child_data.get("attributes") or {}).get("binding_kind") == "loop"
            and (child_data.get("attributes") or {}).get("binding_name") == loop.target.id
        )
    return tuple(sorted(set(reads)))


def _list_rows_born_by_outer_display(
    graph_obj: Any, mutations: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Move each list row's birth to the list display that owns it.

    Program: ``g = [[1, 2], [3, n]]; g[i][1]`` (corpus c732).  Wrong record:
    the inner display's own ``clear`` and the outer's row append both belong
    to the inner display's node, which holds one control block per kind, so
    the two facts met on one block row and the program was refused with an
    uncaused revision naming neither (corpus c375).  Rule: a list whose rows
    are lists owns the whole birth.  At the outer display, in order: the
    outer's ``clear``, then per row one ``child_row`` append (the row's
    handle allocated in the child-table pool and appended to the outer),
    then that row's member appends against the row's pool slice.  The inner
    display keeps no birth of its own, so it is only ever consumed as a row:
    a row that is read, changed, or listed twice elsewhere is a shared list
    and is refused by name.
    """

    from dataclasses import replace
    from .control_source import ControlSequenceMutation

    births = [
        item for item in mutations
        if str(item.argument_kind) == "child_sequence"
    ]
    if not births:
        return mutations
    outer_nodes = {
        int(item.display_member[0]): int(item.sequence_value_id)
        for item in births
    }
    rows_of: dict[int, list[tuple[int, int, int]]] = {}
    for item in births:
        rows_of.setdefault(int(item.sequence_value_id), []).append((
            int(item.display_member[1]), int(item.argument_value_ids[0]),
            int(item.effect_node_id),
        ))
    row_slot: dict[int, tuple[int, int, int, int]] = {}
    node_of_outer = {outer: node for node, outer in outer_nodes.items()}
    outer_nodes_by_value = node_of_outer
    for outer, members in rows_of.items():
        members.sort()
        for position, (_ordinal, inner, inner_node) in enumerate(members):
            # A row is its pool slice: reads and stores through it
            # (``g[1][i]`` folds to the row itself) address that slice, and
            # nothing else may take the list as a value.
            escapes = [
                consumer for consumer in graph_obj.successors(inner_node)
                if int(consumer) != outer_nodes_by_value[outer]
                and str(
                    graph_obj.nodes[consumer].get("op")
                    or graph_obj.nodes[consumer].get("type") or ""
                ).casefold() not in {"indexed", "indexedstore", "setitem"}
            ]
            if inner in row_slot or escapes:
                raise ValueError(
                    f"list {inner} is a row of list {outer} and is also "
                    "listed twice or used as a whole list elsewhere; a "
                    "shared row has one pool slice and is not lowered"
                )
            row_slot[inner] = (outer, position, inner_node, len(members))
    mixed = [
        item for item in mutations
        if int(item.sequence_value_id) in rows_of
        and str(item.operator) in {"append", "add"}
        and str(item.argument_kind) != "child_sequence"
    ]
    if mixed:
        raise ValueError(
            f"list {int(mixed[0].sequence_value_id)} mixes list rows with "
            "other members; a list of lists is lowered only when every row "
            "is a list"
        )
    # Each row's member appends keep their producers as owners but execute
    # at the outer display, right after that row's allocation: the display
    # ordinal is (row position, 0 for the allocation, 1 + member ordinal).
    row_appends: dict[int, list[Any]] = {}
    for item in mutations:
        sequence_id = int(item.sequence_value_id)
        if sequence_id in row_slot and str(item.operator) != "clear":
            outer, position, _node, _count = row_slot[sequence_id]
            row_appends.setdefault(sequence_id, []).append(replace(
                item, display_member=(
                    node_of_outer[outer],
                    (position << 16) | (1 + int(item.display_member[1])),
                ),
            ))
    rewritten = []
    for item in mutations:
        sequence_id = int(item.sequence_value_id)
        if str(item.argument_kind) == "child_sequence" or (
            sequence_id in row_slot
        ):
            continue
        rewritten.append(item)
        if sequence_id in rows_of and str(item.operator) == "clear":
            for _ordinal, inner, inner_node in rows_of[sequence_id]:
                outer, position, _node, _count = row_slot[inner]
                rewritten.append(ControlSequenceMutation(
                    sequence_value_id=outer, operator="append",
                    argument_value_ids=(), effect_node_id=inner_node,
                    policy="duplicates", argument_kind="child_row",
                    display_member=(
                        node_of_outer[outer], position << 16,
                    ),
                ))
                rewritten.extend(row_appends.get(inner, ()))
    return tuple(rewritten)


def _joined_list_literal_mutations(
    graph_obj: Any, joined_sequence_ids: Iterable[int]
) -> tuple[Any, ...]:
    """Materialize each authored list element into its resident dual view."""

    from .control_source import ControlSequenceMutation

    joined = set(map(int, joined_sequence_ids))
    mutations = []
    emitted: set[tuple[int, tuple[int, int]]] = set()

    def append_mutation(
        sequence_id: int, effect_node_id: int, value_id: int,
        display_member: tuple[int, int],
    ) -> None:
        # One append per display member: ``[b, b]`` is two appends of one
        # value, so the key is the member, not (producer, value).
        key = (int(sequence_id), tuple(map(int, display_member)))
        if key in emitted:
            return
        emitted.add(key)
        mutations.append(ControlSequenceMutation(
            sequence_value_id=int(sequence_id),
            operator="append",
            argument_value_ids=(int(value_id),),
            effect_node_id=int(effect_node_id),
            policy="duplicates",
            argument_kind="joined_literal_element",
            display_member=tuple(map(int, display_member)),
        ))

    nodes = tuple(graph_obj.nodes(data=True))

    def expression_node(expression: ast.AST) -> tuple[int, int] | None:
        exact = [
            (int(node_id), int(data.get("value_id", node_id)))
            for node_id, data in nodes
            if data.get("expr_obj") is expression
        ]
        if len(exact) == 1:
            return exact[0]
        position = (
            int(getattr(expression, "lineno", -1) or -1),
            int(getattr(expression, "col_offset", -1) or -1),
            int(getattr(expression, "end_lineno", -1) or -1),
            int(getattr(expression, "end_col_offset", -1) or -1),
            type(expression),
        )
        positioned = [
            (int(node_id), int(data.get("value_id", node_id)))
            for node_id, data in nodes
            for candidate in (data.get("expr_obj"),)
            if isinstance(candidate, ast.AST)
            and (
                int(getattr(candidate, "lineno", -2) or -2),
                int(getattr(candidate, "col_offset", -2) or -2),
                int(getattr(candidate, "end_lineno", -2) or -2),
                int(getattr(candidate, "end_col_offset", -2) or -2),
                type(candidate),
            ) == position
        ]
        return positioned[0] if len(positioned) == 1 else None

    for node_id, data in sorted(
        nodes, key=lambda item: int(item[0])
    ):
        sequence_id = int(data.get("value_id", node_id))
        if sequence_id not in joined:
            continue
        expression = data.get("expr_obj")
        if not isinstance(expression, ast.List):
            continue
        elements = sorted(
            [
                (
                int(parent), str(role),
                int(graph_obj.nodes[parent].get("value_id", parent)),
                )
                for parent, role in (data.get("parents") or ())
                if str(role).startswith("elts") and parent in graph_obj
            ],
            key=lambda item: (
                int(item[1].split(":", 1)[1])
                if ":" in item[1] and item[1].split(":", 1)[1].isdigit()
                else item[0]
            ),
        )
        for ordinal, (parent, _role, value_id) in enumerate(elements):
            append_mutation(
                sequence_id, parent, value_id, (int(node_id), ordinal),
            )

    # The source realizer may specialize an inline dynamic list into an empty
    # structural resident while retaining the authored list only on the Call
    # that consumes it (``_vector([uleb(index)])``). Its elements are still
    # ordinary ProcessGraph nodes. Correlate those exact AST objects/source
    # spans back to their deterministic value identities and initialize the
    # resident before the consuming call.
    for call_node_id, data in sorted(nodes, key=lambda item: int(item[0])):
        expression = data.get("expr_obj")
        if not isinstance(expression, ast.Call) or not expression.args:
            continue
        literal = expression.args[0]
        if not isinstance(literal, ast.List):
            continue
        sequence_ids = tuple(dict.fromkeys(
            int(graph_obj.nodes[parent].get("value_id", parent))
            for parent, role in (data.get("parents") or ())
            if str(role) == "arg:0" and parent in graph_obj
        ))
        if len(sequence_ids) != 1 or sequence_ids[0] not in joined:
            continue
        if any(key[0] == sequence_ids[0] for key in emitted):
            # Already materialized from its own List node above.
            continue
        for ordinal, element in enumerate(literal.elts):
            correlated = expression_node(element)
            if correlated is None:
                continue
            effect_node_id, value_id = correlated
            # The literal has no node of its own here: its members execute
            # at the consuming call, in member order.
            append_mutation(
                sequence_ids[0], effect_node_id, value_id,
                (int(call_node_id), ordinal),
            )
    return tuple(mutations)


def _dict_update_mutations(graph_obj: Any) -> tuple[Any, ...]:
    """``d.update(...)`` on a dict receiver, as table mutations.

    Program: ``d = {1: 10, 2: 20}; d.update({1: 5}); return d[1] + n``
    (corpus c214).  Wrong record: ``update`` on a dict was no mutation
    record of any kind, so its rows were never stored and the later lookup
    read the initializer's 10 (silent; clean audit).  Rule:
    ``d.update(<dict display>)`` is one keyed store per display row, in
    order, at the call (the table store ``d[k] = v`` lowers to).  Its rows
    are the display's own key and value producers, evaluated before the
    call as Python evaluates the argument.  Any other ``update`` (another
    mapping, keywords, ``**``) is posted as a ``mapping_source`` update,
    which the table lowering refuses by name.
    """

    from .control_source import ControlSequenceMutation

    calls = [
        (node_id, data)
        for node_id, data in sorted(graph_obj.nodes(data=True), key=lambda item: int(item[0]))
        if isinstance(data.get("expr_obj"), ast.Call)
        and isinstance(data["expr_obj"].func, ast.Attribute)
        and data["expr_obj"].func.attr == "update"
    ]
    if not calls:
        return ()
    # A display changed after its birth no longer holds its display rows
    # (``e = {2: n}; e[2] = 9; d.update(e)``): its update is a mapping
    # source, by the one definition of "changed".
    from .glsl_deployment_strategy import _mutated_sequence_node_ids

    changed = _mutated_sequence_node_ids(graph_obj)
    mutations = []
    for node_id, data in calls:
        expression = data.get("expr_obj")
        roles = [
            (str(role), int(parent)) for parent, role in data.get("parents") or ()
            if parent in graph_obj
        ]
        receiver = next((parent for role, parent in roles if role == "operand"), None)
        if receiver is None or (
            graph_obj.nodes[receiver].get("attributes") or {}
        ).get("aggregate_kind") != "dict":
            continue
        sequence_id = int(graph_obj.nodes[receiver].get("value_id", receiver))
        sources = [parent for role, parent in roles if role.startswith("arg:")]
        source = graph_obj.nodes[sources[0]] if len(sources) == 1 else {}
        display = source.get("expr_obj")
        if not (
            len(sources) == 1 and len(expression.args) == 1
            and not expression.keywords
            and isinstance(display, ast.Dict)
            and None not in display.keys
            and (source.get("attributes") or {}).get("aggregate_kind") == "dict"
            and int(sources[0]) not in changed
        ):
            mutations.append(ControlSequenceMutation(
                sequence_value_id=sequence_id, operator="update",
                argument_value_ids=tuple(
                    int(graph_obj.nodes[parent].get("value_id", parent))
                    for parent in sources
                ),
                effect_node_id=int(node_id), policy="unique",
                argument_kind="mapping_source",
            ))
            continue
        if not display.keys:
            continue
        source_parents = tuple(source.get("parents") or ())
        keys = tuple(
            int(graph_obj.nodes[parent].get("value_id", parent))
            for parent, role in source_parents
            if str(role).startswith("keys") and parent in graph_obj
        )
        values = tuple(
            int(graph_obj.nodes[parent].get("value_id", parent))
            for parent, role in source_parents
            if str(role).startswith("values") and parent in graph_obj
        )
        if not len(keys) == len(values) == len(display.keys):
            raise ValueError(
                f"Dictionary update {node_id} lacks one exact key and value "
                "producer per display row"
            )
        mutations.append(ControlSequenceMutation(
            sequence_value_id=sequence_id, operator="update",
            argument_value_ids=tuple(
                value_id for pair in zip(keys, values) for value_id in pair
            ),
            effect_node_id=int(node_id), policy="unique",
            argument_kind="mapping_items",
        ))
    return tuple(mutations)


def _dict_rows_born_by_display(
    graph_obj: Any, node_id: int, data: Mapping[str, Any],
    expression: ast.Dict, key_parents: tuple[tuple[int, int], ...],
) -> tuple[Any, ...]:
    """Birth of a dict display whose values are all list displays.

    Program: ``d = {1: [n, 2, 3]}; d[1][2]``.  Wrong record: the list value
    was stored as its own arena's id where the value column holds child-pool
    handles, so a read through it returned 0 (silent).  Rule: each entry is
    a ``setdefault`` of its key (the row's handle is allocated in the child
    pool, as for ``d.setdefault(k, []).append(...)``), then the row's member
    appends against that pool slice, in order, at the display.  A repeated
    key selects the same row and clears it first, as a later entry replaces
    an earlier one.
    """

    from .control_source import ControlExpression, ControlSequenceMutation
    from .string_table import string_token

    sequence_id = int(data.get("value_id", node_id))
    value_parents = tuple(
        (int(parent), int(graph_obj.nodes[parent].get("value_id", parent)))
        for parent, role in data.get("parents") or ()
        if str(role).startswith("values") and parent in graph_obj
    )
    if not (
        len(key_parents) == len(value_parents) == len(expression.keys)
        and None not in expression.keys
    ):
        raise ValueError(
            f"dict {sequence_id} lists its rows without one exact key and "
            "list producer per entry; a mapping of lists is lowered only "
            "from keyed list displays"
        )
    rows = {value_id for _node, value_id in value_parents}
    if len(rows) != len(value_parents):
        raise ValueError(
            f"dict {sequence_id} stores one list under two keys; a shared "
            "row has one pool slice and is not lowered"
        )
    births = _list_literal_mutations(graph_obj, rows)
    members_of: dict[int, list[Any]] = {}
    for item in births:
        if str(item.operator) != "clear":
            members_of.setdefault(int(item.sequence_value_id), []).append(item)
    mutations = []
    for position, (key_expression, (key_node, key_value), (row_node, row_value)) in enumerate(zip(
        expression.keys, key_parents, value_parents,
    )):
        key_literal = (
            key_expression.value
            if isinstance(key_expression, ast.Constant) else None
        )
        key_ir = None
        if isinstance(key_literal, (bool, int, float, str)):
            key_ir = ControlExpression(
                "const", value_id=key_value,
                literal=(
                    string_token(key_literal)
                    if isinstance(key_literal, str) else key_literal
                ),
            )
        mutations.append(ControlSequenceMutation(
            sequence_value_id=sequence_id, operator="setdefault",
            # The pool path reads no default (the row is born empty and
            # filled by the appends below); naming the row here would make
            # the setdefault consume what its own appends produce.
            argument_value_ids=(key_value, key_value),
            effect_node_id=row_node, policy="unique",
            argument_kind="mapping_child_row",
            argument_expressions=(key_ir, None),
            display_member=(int(node_id), position << 16),
        ))
        for item in members_of.get(row_value, ()):
            mutations.append(replace(item, display_member=(
                int(node_id),
                (position << 16) | (1 + int(item.display_member[1])),
            )))
    return tuple(mutations)


def _dict_literal_mutations(graph_obj: Any) -> tuple[Any, ...]:
    """Materialize authored dynamic dict rows at the literal's source site.

    Compile-time mappings use ``literal_table=`` initialization, but a dict
    whose values are computed SSA results cannot be initialized in the entry
    prelude.  Its AST already provides an exact ordered key/value relation;
    retain each row as an ordinary unique-sequence insertion at its key's
    lexical position inside the literal. Data scheduling separately ensures
    that the corresponding value producer completes first. String keys become
    the repository's content-addressed token, the same representation used by
    dynamic keyed lookups.
    """

    from .control_source import ControlExpression, ControlSequenceMutation
    from .string_table import string_token

    nodes = tuple(graph_obj.nodes(data=True))

    def expression_value(expression: ast.AST) -> tuple[int, int] | None:
        exact = tuple(
            (int(node_id), int(data.get("value_id", node_id)))
            for node_id, data in nodes
            if data.get("expr_obj") is expression
        )
        if len(exact) == 1:
            return exact[0]
        position = (
            int(getattr(expression, "lineno", -1) or -1),
            int(getattr(expression, "col_offset", -1) or -1),
            int(getattr(expression, "end_lineno", -1) or -1),
            int(getattr(expression, "end_col_offset", -1) or -1),
            type(expression),
        )
        positioned = tuple(
            (int(node_id), int(data.get("value_id", node_id)))
            for node_id, data in nodes
            for candidate in (data.get("expr_obj"),)
            if isinstance(candidate, ast.AST)
            and (
                int(getattr(candidate, "lineno", -2) or -2),
                int(getattr(candidate, "col_offset", -2) or -2),
                int(getattr(candidate, "end_lineno", -2) or -2),
                int(getattr(candidate, "end_col_offset", -2) or -2),
                type(candidate),
            ) == position
        )
        return positioned[0] if len(positioned) == 1 else None

    mutations = []
    for node_id, data in sorted(nodes, key=lambda item: int(item[0])):
        expression = data.get("expr_obj")
        attributes = data.get("attributes") or {}
        if (attributes.get("aggregate_kind") == "dict"
                and attributes.get("materialization_kind") == "unrolled_loop"):
            sequence_id = int(data.get("value_id", node_id))
            mutations.append(ControlSequenceMutation(
                sequence_value_id=sequence_id, operator="clear",
                argument_value_ids=(), effect_node_id=int(node_id), policy="unique",
            ))
            for row_id in attributes.get("materialized_value_ids", ()):
                row = graph_obj.nodes.get(int(row_id), {})
                leaves = tuple(int(parent) for parent, role in row.get("parents", ())
                               if role == "elts")
                if len(leaves) != 2:
                    raise ValueError(f"Dictionary row {row_id} must publish one key and value")
                mutations.append(ControlSequenceMutation(
                    sequence_value_id=sequence_id, operator="update",
                    argument_value_ids=tuple(int(graph_obj.nodes[parent].get("value_id", parent)) for parent in leaves),
                    effect_node_id=int(row_id), policy="unique",
                    argument_kind="mapping_items",
                ))
            continue
        if not (
            isinstance(expression, ast.Dict)
            and attributes.get("aggregate_kind") == "dict"
            and len(expression.keys) == len(expression.values)
        ):
            continue
        sequence_id = int(data.get("value_id", node_id))
        key_parents = tuple(
            (int(parent), int(graph_obj.nodes[parent].get("value_id", parent)))
            for parent, role in data.get("parents") or ()
            if str(role).startswith("keys") and parent in graph_obj
        )
        # Each execution of a literal constructs a fresh mapping. Keep one
        # lexical owner even for constant rows and for literals inside loops.
        mutations.append(ControlSequenceMutation(
            sequence_value_id=sequence_id, operator="clear",
            argument_value_ids=(), effect_node_id=int(node_id), policy="unique",
        ))
        if attributes.get("mapping_value_aggregate_kind") == "list":
            mutations.extend(_dict_rows_born_by_display(
                graph_obj, int(node_id), data, expression, key_parents,
            ))
            continue
        value_parents = tuple(
            (int(parent), int(graph_obj.nodes[parent].get("value_id", parent)))
            for parent, role in data.get("parents") or ()
            if str(role).startswith("values") and parent in graph_obj
        )
        for index, (key_expression, value_expression) in enumerate(zip(
            expression.keys, expression.values
        )):
            if key_expression is None:
                continue
            key = (
                key_parents[index]
                if len(key_parents) == len(expression.keys)
                else expression_value(key_expression)
            )
            value = (
                value_parents[index]
                if len(value_parents) == len(expression.values)
                else expression_value(value_expression)
            )
            if key is None or value is None:
                continue
            key_node_id, key_value_id = key
            _value_node_id, value_value_id = value
            key_literal = (
                key_expression.value
                if isinstance(key_expression, ast.Constant)
                else None
            )
            key_expression_ir = None
            if isinstance(key_literal, str):
                key_expression_ir = ControlExpression(
                    "const", value_id=key_value_id,
                    literal=string_token(key_literal),
                )
            elif isinstance(key_literal, (bool, int, float)):
                key_expression_ir = ControlExpression(
                    "const", value_id=key_value_id, literal=key_literal,
                )
            mutations.append(ControlSequenceMutation(
                sequence_value_id=sequence_id,
                operator="update",
                argument_value_ids=(key_value_id, value_value_id),
                # The stored value can name an SSA producer authored much
                # earlier than the literal. Its producer position is not the
                # store's effect position: using it moved the first row ahead
                # of the literal's clear. The key is an exact AST child of
                # this literal and gives every row a stable authored order.
                effect_node_id=key_node_id,
                policy="unique",
                argument_kind="mapping_items",
                argument_expressions=(key_expression_ir, None),
            ))
    mutations.extend(_dict_update_mutations(graph_obj))
    return tuple(mutations)

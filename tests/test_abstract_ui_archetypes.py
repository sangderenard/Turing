"""Archetypes instantiate structure, namespace, IntelliType, and edit receipts."""

import pytest

from src.compiler.abstract_ui_archetypes import (
    ArchetypeContext,
    ArchetypeLibrary,
    LivingDocument,
    box,
    buttons,
    class_context,
    class_panel_recipe,
    displays,
    front,
    inside,
    label,
    top_front,
)
from src.compiler.oop_schema import (
    ClassSchema,
    FieldSchema,
    MethodSchema,
    ParameterSchema,
)


def _pump_schema() -> ClassSchema:
    return ClassSchema(
        identity="demo.Pump",
        fields=(
            FieldSchema("pressure", "float", slot=0),
            FieldSchema("enabled", "bool", slot=1),
        ),
        methods=(
            MethodSchema("start", body_reference="demo.Pump.start"),
            MethodSchema(
                "set_pressure",
                (ParameterSchema("value", "float"),),
                body_reference="demo.Pump.set_pressure",
            ),
        ),
        origin_language="python",
    )


def _instantiate_panel(document=None, *, symbol_name="panel"):
    library = ArchetypeLibrary().define("class-panel", class_panel_recipe())
    context = ArchetypeContext(
        ("world", "foundry"),
        {"class": _pump_schema()},
        actor="builder",
        location="foundry-square",
    )
    return library.instantiate(
        "class-panel",
        document=document or LivingDocument.empty("living-program"),
        context=context,
        symbol_name=symbol_name,
    )


def test_motivating_panel_syntax_preserves_human_statement_order():
    recipe = (
        box.with_(class_context, inside)
        .with_(buttons.with_(label, front), front)
        .with_(displays, top_front)
        .connect(displays, class_context.members, "displays")
        .connect(buttons, class_context.methods, "invokes")
    )
    assert [statement.operation for statement in recipe.statements] == [
        "with", "with", "with", "connect", "connect",
    ]
    assert [statement.relationship for statement in recipe.statements] == [
        "inside", "front", "top-front", "displays", "invokes",
    ]


def test_instantiation_creates_nodes_edges_symbol_and_inferred_intellitype():
    result = _instantiate_panel()
    symbol = result.document.namespace.resolve("world.foundry.panel")
    assert symbol is result.symbol
    assert symbol.identity == result.root_identity
    assert result.document.revision == 1
    assert result.intellitype.root_archetype == "box"
    assert set(result.intellitype.capabilities) >= {
        "container", "interior", "action-host", "display-host",
    }
    assert {slot.placement for slot in result.intellitype.slots} >= {
        "inside", "front", "top-front",
    }
    assert {binding.relationship for binding in result.intellitype.bindings} == {
        "displays", "invokes",
    }


def test_class_member_and_method_connections_resolve_against_program_schema():
    result = _instantiate_panel()
    by_name = {node.name: node for node in result.edit.added_nodes}
    members = by_name["class.members"]
    methods = by_name["class.methods"]
    assert dict(members.properties)["value"] == ("pressure", "enabled")
    assert dict(methods.properties)["value"] == ("start", "set_pressure")

    edge_targets = {
        edge.relationship: edge.target
        for edge in result.edit.added_edges
        if edge.relationship in {"displays", "invokes"}
    }
    assert edge_targets == {
        "displays": members.identity,
        "invokes": methods.identity,
    }


def test_user_action_is_a_transactional_living_document_edit():
    original = LivingDocument.empty("living-program")
    result = _instantiate_panel(original)
    assert original.revision == 0
    assert not original.nodes
    assert result.edit.action == "instantiate-archetype"
    assert result.edit.actor == "builder"
    assert result.edit.location == "foundry-square"
    assert (result.edit.before_revision, result.edit.after_revision) == (0, 1)
    mapping = result.edit.to_mapping()
    assert mapping["schema"] == "abstract-ui-archetype-v0"
    assert mapping["published_symbols"][0]["path"] == (
        "world", "foundry", "panel",
    )


def test_namespace_refuses_accidental_redefinition_but_allows_named_instances():
    first = _instantiate_panel()
    with pytest.raises(ValueError, match="namespace symbol already exists"):
        _instantiate_panel(first.document)

    second = _instantiate_panel(first.document, symbol_name="secondary_panel")
    assert second.document.revision == 2
    assert second.document.namespace.resolve(
        "world.foundry.secondary_panel"
    ).identity == second.root_identity
    assert sum(
        node.identity == "demo.Pump/methods" for node in second.document.nodes
    ) == 1
    assert "demo.Pump/methods" not in {
        node.identity for node in second.edit.added_nodes
    }


def test_recipe_is_reusable_and_context_changes_namespace_and_program_binding():
    class Other:
        value: int

        def reset(self):
            return None

    library = ArchetypeLibrary().define("class-panel", class_panel_recipe())
    context = ArchetypeContext(
        ("world", "archive"), {"class": Other}, actor="visitor", location="hall",
    )
    result = library.instantiate(
        "class-panel",
        document=LivingDocument.empty("another-program"),
        context=context,
        symbol_name="other_panel",
    )
    assert result.symbol.qualified_name == "world.archive.other_panel"
    values = {
        node.name: dict(node.properties).get("value")
        for node in result.edit.added_nodes
    }
    assert values["class.members"] == ("value",)
    assert values["class.methods"] == ("reset",)

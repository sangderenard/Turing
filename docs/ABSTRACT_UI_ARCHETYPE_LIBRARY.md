# AbstractUI archetype library and living namespace

An `AbstractUI` metaphor interprets something that already exists. An
archetype is different: it is a reusable construction the user may instantiate
into the living program.

The initial implementation is
`src/compiler/abstract_ui_archetypes.py`.

## Motivating recipe

The original sketch becomes valid source-safe Python as:

```python
panel = (
    box.with_(class_context, inside)
    .with_(buttons.with_(label, front), front)
    .with_(displays, top_front)
    .connect(displays, class_context.members, "displays")
    .connect(buttons, class_context.methods, "invokes")
)
```

`class_context` retains the canonical spelling `class`; `top_front` retains
`top-front`. The recipe records each line in appearance order.

## Four layers

```text
UIIntention
    human meaning, possibly still open vocabulary

ArchetypeRecipe
    reusable construction: parts, placements, contextual references,
    and connections

ArchetypeInstantiation
    concrete LivingNodes and LivingEdges inserted into one document

IntelliType
    structural type inferred from the instantiated graph: capabilities,
    addressable slots, and contextual bindings
```

The archetype is not itself a panel object. Instantiating it creates one panel
identity and publishes that identity under a requested namespace path.

## Living edits

Instantiation is a transactional `LivingDocumentEdit`:

```text
actor
location
action = instantiate-archetype
before revision
after revision
added nodes
added edges
published symbols
recipe statements
```

The old document remains immutable. The returned document has a new revision.
Thus “the user builds a panel in the foundry” and “the program gains the symbol
`world.foundry.panel`” are two descriptions of the same edit.

The edit serializes without host objects and is intended to become a typed
sidecar on `DualIRShell`. It should not be hidden in generated HTML, a game
save, or a Python callback because all of those are merely representation
spaces observing the same living edit.

## Contextual references

An archetype may name facts supplied by the place where it is instantiated:

```text
class
class.members
class.methods
```

The current resolver accepts Python classes and `ClassSchema`-shaped objects.
It publishes reference nodes instead of copying the program members into the
archetype. Displays bind to the member set; buttons bind to the method set.

This is the beginning of IntelliType: the instantiated panel knows it is a
container with an interior, action host, display host, member binding, and
method binding. Those facts arise from its graph rather than from a handwritten
nominal `Panel` class.

## Identity and namespace rules

- Recipes are reusable and have library names.
- Instances receive deterministic identities from document and namespace.
- Instantiation publishes exactly one requested namespace symbol.
- Reusing an occupied symbol is rejected rather than overwritten.
- A second explicitly named instance is legal and produces different node and
  IntelliType identities.
- Context references retain the identity of the class/schema they reference.
- Placement, appearance, and metaphor never substitute for identity.

## Next boundary

The current edit is an `AbstractUI` sidecar. The next compiler step is to
correlate it with `DualIRShell.class_navigation`, `reference_tables`, and a
typed living-interface member. When an archetype action invokes a method or
publishes a field, the final action edge must resolve through those existing
tables rather than through string evaluation.


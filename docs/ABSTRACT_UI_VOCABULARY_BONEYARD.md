# AbstractUI vocabulary boneyard

`src/compiler/abstract_ui_vocabulary.py` is an aspirational but executable
language surface for `AbstractUI`. It intentionally feels more like an
optimistic Perl DSL than a finished compiler grammar: obvious human words are
callable, nest naturally, and produce immutable intention trees.

```python
world(
    region("foundry",
        building("pump house",
            interior(room("control room")),
            contains(gauge("pressure")),
        ),
    ),
)
```

Consequential flow can remain compact:

```python
when(player("operator").does(open_(door("east")))) \
    >> reveal(room("pump"))
```

The punctuation is structural:

```text
a >> b   then(a, b)
a & b    all_of(a, b)
a | b    any_of(a, b)
~a       not_(a)
```

Intentions deliberately have no Python truth value. A conditional must become
an explicit intention rather than accidentally executing during authorship.

## Why the boneyard is huge

The vocabulary ranges over the finite conceptual neighborhoods currently
expected of `AbstractUI`:

- context and projection;
- existence, identity, and containment;
- navigation, tracks, regions, buildings, and interiors;
- action, requirements, permissions, effects, and consequences;
- sequence, scenes, cutscenes, and choices;
- controlled narrative and understandable descriptions;
- programming-language and OOP archival structure;
- forms, CSS, layout, and style;
- images, shaders, sound, and resources;
- host events and contextual gesture recognition;
- accessibility and equivalent non-spatial routes;
- data, values, and live bindings.

This is not backend coverage. It is a searchable catalog of intentions from
which canonical subsets and conformance tables can be selected. Every word at
least produces inspectable data through `UIIntention.to_data()` and appears in
`vocabulary_manifest()`.

## Forgiving phrases

Declared words carry a domain and one-sentence intention. Fluent words that
have not yet been declared remain explicit `domain="open"` nodes:

```python
building("archive").remembers("every visitor", tenderly=True)
```

That permissiveness belongs only to the authoring/boneyard layer. A compiler
or backend must classify the word, preserve it as an honest shortfall, or
reject it in strict mode. It must never silently pretend to have implemented
the phrase.

## Relationship to the canonical graphs

The boneyard is upstream of the finite existence, navigation, and action graph
schemas. It lets the project discover vocabulary by use before freezing every
word into a backend contract:

```text
human intention phrase
    -> UIIntention tree
    -> vocabulary normalization and shortfalls
    -> existence/navigation/action graphs
    -> prosaic or world context projection
    -> browser/SDL/Pygame/accessibility host
```

Words admitted into the canonical graph layer must receive defined roles,
edge direction, identity behavior, and prosaic/world equivalence tests.

The first such experimental bridge is documented in
[`ABSTRACT_UI_INTROSPECTIVE_CLASS_MAP.md`](ABSTRACT_UI_INTROSPECTIVE_CLASS_MAP.md):
real Python and SSA-described classes become regions, buildings, rooms,
tracks, intention phrases, and explicit implied-code receipts.

# HTML interface containment graph

For the `AbstractUI` backend model, expanded root contract, canonical node
vocabulary, and independent CSS graph, see
[`INTERFACE_ROOT_HTML_CSS_GRAPH_VOCABULARY.md`](INTERFACE_ROOT_HTML_CSS_GRAPH_VOCABULARY.md).

HTML is a first-class source language for the translator, not merely the
wrapper produced by a web backend. `html_process_graph.py` normalizes a page
into the repository's dependency-graph convention.

## Neutral vocabulary

The first schema deliberately has only three node names:

| Node | Meaning |
|---|---|
| `InterfaceRoot` | Host or system root. The HTML document is one spelling. |
| `InterfaceContainer` | Any nested interface entity, including value-holding form elements. |
| `InterfaceContent` | Text, comment, or declaration content. |

Exact tags are vocabulary tokens on `InterfaceContainer`, not distinct graph
types. Consequently HTML `<div>`, Java scene groups, and a native window
container can share one conceptual node without losing their source spelling.
The vocabulary separately records a small capability such as `structure`,
`form`, `value`, `action`, or `resource`.

## Edge direction and order

ProcessGraph dependencies point from a required value into its consumer. The
markup graph uses the same rule:

```text
text/input -> label -> form -> div -> InterfaceRoot
```

Read forward, a container depends upon its contents. Read backward, the same
edge is ordinary containment. The root is therefore the final system
container, not an unrelated wrapper node.

Every occurrence has both:

- a deterministic preorder node identity;
- a structural `position` tuple based on authored sibling order.

Every containment edge also has an explicit `ordinal`. Appearance order does
not depend on hash iteration, labels, CSS classes, or generated identifiers.

## Initial HTML profile

The bounded vocabulary covers document structure, `div`, form controls, and
the resource elements needed to carry CSS or behavior:

```text
html head body title meta link style script div
form label input select option button textarea output
```

Unknown tags are preserved as generic containers and reported as vocabulary
shortfalls. Strict mode rejects them. Structurally mismatched or unclosed tags
are always rejected because ambiguous containment cannot produce a dependable
program graph.

## Deliberate next boundary: CSS and bindings

Containment is only one relationship. CSS must add selector-to-container and
property-dependency relations without changing the containment tree. Program
bindings and events likewise attach typed edges to containers rather than
becoming children.

The next neutral concepts should therefore be `InterfaceRule`,
`InterfaceSelector`, and `InterfaceProperty`, followed by value and event
binding edges. They should not be folded into `InterfaceContainer`; doing so
would confuse structural authority with styling or runtime data flow.

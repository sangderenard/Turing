# AbstractUI: roots and the canonical HTML/CSS graph vocabulary

**Status:** design document for the next interface-graph expansion. Names in
the implemented containment slice are marked **implemented**; the rest are
proposed canonical names to settle before expanding the parser.

The broad aspirational authoring surface is catalogued separately in
[`ABSTRACT_UI_VOCABULARY_BONEYARD.md`](ABSTRACT_UI_VOCABULARY_BONEYARD.md).
The boneyard is intentionally permissive; this document defines the stricter
canonical graph and backend contracts it must eventually lower into.

`AbstractUI` is the language-neutral interface operation surface, following
the repository's established `AbstractTensor` pattern. HTML and CSS are
compiler languages understood by `AbstractUI`: HTML describes an ordered
containment graph, while CSS describes a separate graph of selection,
conditions, declarations, and values. Registered `AbstractUI` backends realize
the same operation vocabulary with a browser, SDL, Pygame/PyOpenGL, a graph
recorder, or another host without changing its conceptual names.

## 1. What is the root?

There are two distinct roots, because a UI runtime can own more than one
window, document, embedded surface, or offscreen target.

### `InterfaceSystemRoot`

The absolute root of one running UI subsystem. It owns host lifetime and the
capabilities shared by all presentation surfaces:

- event acquisition and dispatch scheduling;
- clocks and frame scheduling;
- display enumeration and global pixel-density information;
- creation and destruction of presentation surfaces;
- graphics API/context selection;
- shared fonts, textures, cursors, and other host resources.

`InterfaceSystemRoot` is not an HTML element. It is the host dependency to
which a UI-enriched program is attached.

### `InterfaceRoot` — **implemented**

One presentable interface surface owned by `InterfaceSystemRoot`. It owns:

- one ordered document/container tree;
- zero or more stylesheet roots;
- viewport size, scale, and focus state;
- one event-routing boundary;
- a presentation target and optional graphics context;
- begin-frame/present-frame lifecycle.

The current HTML ingester creates an `InterfaceRoot` as its graph root. Until
the system-root layer is attached, that is a valid standalone page graph. In a
complete program the relationship becomes:

```text
document contents -> InterfaceRoot -> InterfaceSystemRoot
```

The system root depends on every surface it must operate and present.

### Concrete and reference root bindings

| Target | `InterfaceSystemRoot` realization | `InterfaceRoot` realization |
|---|---|---|
| C / C++ | initialized SDL subsystem, event pump, shared resource lifetime | `SDL_Window*`, presentation mode, and optional `SDL_GLContext` |
| Python | initialized Pygame runtime and event clock | Pygame display surface plus the current PyOpenGL context when requested |
| JavaScript | browser realm, event loop, and global `Window` capabilities | `Document` plus its document element; an iframe or shadow-root surface may provide another root |
| Prosaic annotation projection | annotated program/class archive | module/class/object document or form root |
| Interactive world projection | world and its simulation/runtime laws | region, zone, island, district, or map layer |

The binding is a backend fact, not a canonical node rename. We do not call
the neutral root `SDLWindow`, `PygameSurface`, or `Document`, because doing so
would make one host's spelling authoritative over the others.

The root contract must expose capabilities rather than promising one renderer.
An SDL root may use OpenGL, Vulkan, SDL's 2D renderer, or a CPU surface. A
Pygame root may use a software `Surface` or PyOpenGL. The browser decides its
own layout/compositing implementation while still presenting the same logical
surface contract.

## 2. `AbstractUI` follows the `AbstractTensor` backend pattern

`AbstractTensor` succeeds because callers author against one semantic
operation vocabulary while registered backend classes implement storage and
execution. `AbstractUI` should make the same separation:

```text
AbstractUI semantic operations
    ├── GraphUIOperations          records canonical HTML/CSS graphs
    ├── AnnotationUIOperations     compiles adorned OOP into plain documents/forms
    ├── WorldMapUIOperations       projects the same UI as an interactive world
    ├── BrowserUIOperations        realizes them as DOM and browser CSS
    ├── SDLUIOperations            realizes roots/events/presentation in C/C++
    └── PygameOpenGLUIOperations   realizes them through Pygame/PyOpenGL
```

The names are proposed, but the mechanism should mirror the real tensor
registry:

```python
UI_BACKEND_REGISTRY: dict[str, type[AbstractUI]] = {}

register_ui_backend("graph", GraphUIOperations)
register_ui_backend("annotations", AnnotationUIOperations)
register_ui_backend("world_map", WorldMapUIOperations)
register_ui_backend("browser", BrowserUIOperations)
register_ui_backend("sdl", SDLUIOperations)
register_ui_backend("pygame_opengl", PygameOpenGLUIOperations)

with AbstractUI.use_backend("graph"):
    ...
```

As with `AbstractTensor`, backends implement primitive hooks and public methods
compose them. A backend must not replace a missing primitive with a subtly
different host behavior. It implements the canonical contract or reports a
named shortfall.

The graph backend is important: it gives Python, JavaScript, C++, Java, and
HTML/CSS frontends a source-producing backend analogous to the SSA
`AbstractTensor` backend. It records operations without opening a window. The
browser/SDL/Pygame backends can then consume that same recorded structure or
execute equivalent operations eagerly.

The annotation and world-map backends are equally mandatory reference
backends. They establish the semantic range every canonical UI operation must
support: from ordinary, inspectable class documentation and forms to a spatial,
interactive world in which the same program consequences remain available.

### Initial `AbstractUI` primitive vocabulary

| Public operation | Backend primitive responsibility |
|---|---|
| `system_root(...)` | Initialize or correlate the host UI subsystem |
| `root(...)` | Create/correlate one window, document, or surface |
| `container(tag, ...)` | Create one neutral container with source vocabulary |
| `content(value, kind=...)` | Create text/comment/declaration content |
| `append(parent, child, ordinal=...)` | Establish ordered containment |
| `identity(node, alias=...)` | Publish stable identity and optional source alias |
| `group(name, members=...)` | Publish or update identity-set membership |
| `property(node, name, value)` | Publish authored or live property state |
| `state(node, name, value)` | Publish dynamic interaction state |
| `resource(kind, value)` | Publish a stylesheet, texture, font, canvas, etc. |
| `stylesheet(...)` | Attach an independently rooted CSS graph |
| `bind_value(...)` | Correlate program state with an interface property |
| `bind_action(...)` | Correlate a normalized event with a program action |
| `events(root)` | Receive normalized host events |
| `begin_frame(root)` | Begin one presentation update |
| `present(root)` | Commit/present one update |

Structural methods should return `AbstractUI` values or typed handles carrying
canonical identities, just as tensor methods return `AbstractTensor` values
instead of leaking NumPy/Torch storage. Backend-native handles remain behind a
correlation method for explicit interop.

This does not mean every CSS declaration invokes an eager host call. A graph
backend records nodes; a browser backend may batch DOM/CSS changes; a native
backend may reduce computed properties before layout. The shared fact is the
operation's meaning and identity, not its execution timing.

### Two required semantic projection backends

These two backends are contract witnesses. A proposed UI or OOP annotation is
not sufficiently canonical until both can express it without discarding its
identity, state, actions, consequences, or accessible description.

#### `AnnotationUIOperations`: prosaic compilation of annotations

This backend translates deliberately adorned OOP structure into the least
surprising document/form expression. It is the reference for inspectability,
automation, archival clarity, and assistive access.

| OOP declaration | Prosaic `AbstractUI` expression |
|---|---|
| module/package/namespace | document or top-level section |
| class | identified section/form and reusable container template |
| constructed object | one class-template instance with a stable identity |
| contained object field | nested section or linked object reference |
| scalar/property field | labeled input, output, or read-only value |
| enum/choice field | labeled select/options |
| Boolean field | checkbox/toggle with explicit state text |
| method | button/action entry with parameters expressed as form fields |
| constructor | creation form |
| event/callback | described event endpoint and subscription state |
| docstring/comment annotation | visible explanatory content |
| type/range/unit/default annotation | form constraints and value metadata |
| permission/effect annotation | access notice and consequence description |

“Prosaic” is a strength: this backend does not invent spatial metaphors. It
renders exactly what the annotations claim, in declaration order, using basic
containers and form elements. It must remain usable even when every decorative
or game-oriented backend is unavailable.

#### `WorldMapUIOperations`: interactive spatial expression

This backend translates the same canonical graph into navigable map and game
elements. It is not a visualization detached from the application: interacting
with the world dispatches the same `InterfaceActionBinding` and mutates the
same program state as the prosaic form.

| Canonical concept | World-map expression |
|---|---|
| `InterfaceSystemRoot` | world and its simulation/runtime laws |
| `InterfaceRoot` / document | region, zone, island, district, or map layer |
| structural container | place, parcel, room, enclosure, or nested area |
| container with an interior | building with enterable interior |
| form/container of related actions | facility, workshop, office, console room |
| `InterfaceContent` | sign, inscription, dialogue, placard, or narrated description |
| `InterfaceIdentity` | persistent world-entity identity and map address |
| `InterfaceGroup` | district, faction, category, overlay, or discoverable collection |
| value property/control | instrument, gauge, lever, dial, inventory item, or terminal field |
| action binding | door, switch, tool use, dialogue choice, or interaction prompt |
| state | visible condition, occupancy, light, animation, marker, or status overlay |
| resource | terrain, sprite, mesh, texture, sound, shader, or referenced artifact |
| stylesheet | zoning, theme, material, atmosphere, placement, and presentation rules |
| selector group | population/query selecting world entities |
| event route | proximity, focus, collision, activation, dialogue, or direct command route |

A document becomes a region because both are named scopes containing other
identified scopes. A container becomes a building only when its declared
capabilities say it has an interior; the backend must not guess that every
`div` is architecture. Less adorned containers receive deterministic fallback
archetypes such as region, place, object, or terminal.

### Accessibility and consequential equivalence

The two reference backends must be correlated identity-for-identity:

- every world entity exposes a semantic label, role, state, description, and
  available action vocabulary;
- every spatial action has a prosaic document/form route unless explicitly
  restricted by the program's declared permissions;
- both projections dispatch the same typed action identity;
- both read and update the same bound program values;
- consequences are published back through canonical state/property nodes, so
  the document and world remain mutually current;
- navigation in the world has a non-spatial hierarchy/path representation;
- decoration may enrich expression but cannot hide an otherwise authorized
  operation from the prosaic accessibility projection.

This makes accessibility structural rather than a late textual overlay. The
ordinary annotation backend is always a complete semantic route through the
program; the world backend is a richer kinetic route through the same graph.

### OOP adornment is a published schema, not arbitrary reflection

Rich OOP environments already expect investment in classes, ownership, and
hierarchical conceptual archives. `AbstractUI` should make that investment
portable through a bounded annotation vocabulary.

Initial neutral annotation concepts should include:

```text
ui.root          ui.document       ui.region
ui.container     ui.interior       ui.building
ui.identity      ui.group          ui.label
ui.description   ui.order          ui.visible_when
ui.value         ui.input          ui.output
ui.range         ui.unit           ui.choice
ui.action        ui.event          ui.consequence
ui.permission    ui.resource       ui.style
ui.world_role    ui.landmark       ui.portal
```

Language frontends may spell these as Python decorators/type metadata, C++
attributes, Java annotations, JavaScript metadata, or external schema. They
must lower to the same annotation records before either reference backend sees
them.

Annotations declare meaning and correlation; they should not contain opaque
host callbacks or backend-native layout objects. A class may be richly adorned,
but every adornment must be serializable, inspectable, and either understood or
reported as a named backend shortfall.

## 3. Graphs remain separate

The interface description is not one undifferentiated graph. It is a family
of correlated graphs with stable cross-references:

```text
HTML containment graph
    publishes identities, groups, properties, and states
                 ^
                 | selector/match references
CSS style graph  +----> resolved-style graph ----> layout/paint

program graph ---- value/event bindings ----> HTML identities

all surface outputs ------------------------> InterfaceRoot
```

Containment alone determines structural ownership. A selector match, event
listener, or style declaration must never make a node a structural child.

## 4. Shared invariants

1. Every source occurrence has a stable compiler identity.
2. Containment children retain authored order and explicit ordinals.
3. Source spelling is retained even after mapping to a neutral node name.
4. Node identity, HTML `id`, CSS class membership, and structural position are
   different facts.
5. Cross-graph references target identities or groups, never display labels.
6. Unknown vocabulary is preserved with a shortfall or rejected in strict
   mode; it is never silently assigned invented behavior.
7. Host handles (`SDL_Window*`, Python objects, JavaScript object references)
   live in backend correlation tables, not serialized canonical nodes.
8. CSS cascade order cannot reorder HTML containment.
9. Paint stacking cannot redefine structural ownership or event ancestry.
10. Derived match and computed-style nodes record their source facts so a
    translation can explain why a value won.
11. Prosaic and world-map projections preserve the same action and state
    identities.
12. Every authorized world interaction has a non-spatial accessible route.
13. OOP adornments are typed canonical records, not backend callback objects.

## 5. Canonical HTML/interface node names

These are conceptual nodes. Exact HTML tags remain vocabulary tokens.

### Structural nodes

| Canonical node | Status | Meaning |
|---|---|---|
| `InterfaceSystemRoot` | proposed | Absolute host UI/runtime owner |
| `InterfaceRoot` | implemented | One document, window, or presentation surface |
| `InterfaceContainer` | implemented | Any nested interface entity, including form controls |
| `InterfaceContent` | implemented | Authored text, comment, or declaration content |

All HTML elements map to `InterfaceContainer`. An `<input>` is still a
container: it contains live value, validity, focus, and interaction state even
though HTML forbids markup children. “Void” is a source grammar constraint,
not evidence that the neutral object has no state.

### Addressing and membership nodes

| Canonical node | Meaning |
|---|---|
| `InterfaceIdentity` | Unique address of exactly one root, container, or content occurrence |
| `InterfaceGroup` | Named or derived set of identities, such as a CSS class or tag group |
| `InterfaceProperty` | Authored attribute or live element property associated with an identity |
| `InterfaceState` | Dynamic state fact such as focus, checked, disabled, hover, or validity |

Every interface occurrence receives an internal `InterfaceIdentity`, whether
or not HTML supplies an `id` attribute. HTML `id="gain"` adds the source alias
`gain` to that identity. It does not create identity itself.

Groups allow CSS and program operations to name sets without copying element
nodes:

- `.knob` resolves to a named class `InterfaceGroup`;
- `input` resolves to a tag `InterfaceGroup`;
- `[disabled]` resolves through `InterfaceProperty` membership;
- `:focus` and `:checked` resolve through dynamic `InterfaceState` membership.

### Resource and behavior references

| Canonical node | Meaning |
|---|---|
| `InterfaceResource` | Stylesheet, image, font, canvas target, texture, shader output, or other referenced resource |
| `InterfaceValueBinding` | Correlation between a program value and an interface property/state |
| `InterfaceActionBinding` | Correlation between an interface event and a program action |

Bindings belong to the wider interface/program correlation layer. They are
listed here because HTML form elements publish their endpoints, but they are
not containment children.

## 6. HTML vocabulary tokens

The first compiler profile should stay small. Semantic HTML tags may be
accepted as aliases later, but they need not expand the canonical node set.

### Level 0: document and basic composition

```text
html head body title meta link style script div
```

### Level 0: forms and ordinary interaction

```text
form label input select option button textarea output
```

### Level 1: useful form and rendering extensions

```text
fieldset legend datalist progress meter canvas img
```

For every token, the HTML table should declare:

- neutral node name (`InterfaceContainer`);
- capability set (`structure`, `form`, `value`, `action`, `resource`, etc.);
- void/non-void source grammar;
- legal property vocabulary;
- default event vocabulary;
- default state vocabulary;
- whether text/element content is legal;
- backend lowering shortfalls.

This is analogous to an operator table: exact source spelling selects neutral
meaning and a checked lowering contract.

## 7. HTML containment and publication edges

Canonical dependency direction follows `ProcessGraph`: a dependency points
into its consumer.

| Edge role | Direction | Meaning |
|---|---|---|
| `content` | content/container -> containing container/root | Container depends on ordered content |
| `identifies` | identity -> interface occurrence | Occurrence is addressed by identity |
| `member` | identity -> group | Group contains that identity |
| `property` | property -> identity | Identity publishes authored/live property |
| `state` | state -> identity | Identity currently publishes dynamic state |
| `resource` | resource -> identity/root | Identity or root consumes a resource |
| `surface` | `InterfaceRoot` -> `InterfaceSystemRoot` | System owns and operates the surface |

`content` is structural. The other edge roles publish facts about structure
without becoming containment.

## 8. CSS is its own graph

CSS has its own root, nodes, ordering, and derived products. Its source graph
must be serializable even before any HTML document is available.

### Canonical CSS node names

| Canonical node | Meaning |
|---|---|
| `StyleSheetRoot` | Ordered root of one stylesheet or imported style unit |
| `StyleRule` | One selector list paired with an ordered declaration block |
| `StyleSelector` | A selector expression producing a set of interface identities |
| `StylePredicate` | One tag, identity, group, property, state, or universal test |
| `StyleRelationship` | Descendant, child, adjacent-sibling, or general-sibling relation between predicates |
| `StyleDeclaration` | One authored property/value pair, including importance and source ordinal |
| `StyleValue` | Literal, dimension, color, keyword, list, or composed CSS value |
| `StyleVariable` | Definition or reference of a custom property |
| `StyleFunction` | `calc`, `var`, `min`, `max`, `clamp`, color function, transform function, etc. |
| `StyleCondition` | Media/support/container/layer condition guarding rules |
| `StyleMatch` | Derived evidence that one rule matches one interface identity |
| `ComputedProperty` | Derived winning value for one identity/property pair |

At-rules are not all one semantic operation. Imports add stylesheet
dependencies; media/support rules add `StyleCondition`; layers affect cascade
rank; keyframes will later add an animation graph. The source parser may keep
an `at_rule` spelling, but normalization should lower each supported form into
the appropriate conceptual node.

### CSS graph edges

| Edge role | Direction | Meaning |
|---|---|---|
| `predicate` | identity/group/property/state -> selector | Selector depends on interface facts |
| `relationship` | predicate/selector -> selector | Compound or relational selector dependency |
| `selects` | selector -> rule | Rule depends on its target-set expression |
| `declares` | declaration -> rule | Rule depends on an authored declaration |
| `value` | value/function/variable -> declaration | Declaration depends on its value expression |
| `argument` | value -> function | CSS function depends on ordered arguments |
| `variable` | variable definition -> variable reference/value | Custom-property dependency |
| `condition` | condition -> rule | Rule is active only when the condition holds |
| `rule` | rule/import -> stylesheet root | Stylesheet depends on ordered rules/imports |
| `stylesheet` | stylesheet root -> interface root | Surface consumes the stylesheet |
| `matches` | rule + identity -> style match | Derived match records both causes |
| `candidate` | style match/declaration -> computed property | Cascade candidate for identity/property |
| `inherits` | parent computed property -> child computed property | Explicit inherited-value dependency |
| `resolves` | computed property -> identity | Identity receives the winning style value |

Because CSS is independently rooted, it can be parsed, transformed, compared,
or translated without an HTML graph. Selector resolution begins only after a
stylesheet is attached to an `InterfaceRoot` with published identities and
groups.

## 9. Example: a class selector and form identity

Source:

```html
<div class="knob">
  <label for="gain">Gain</label>
  <input id="gain" type="range">
</div>
```

```css
.knob input:focus { outline-width: 2px; }
```

Normalized relationships:

```text
InterfaceIdentity(input#gain) -> InterfaceGroup(tag:input)
InterfaceIdentity(div)        -> InterfaceGroup(class:knob)
InterfaceState(focus)         -> InterfaceIdentity(input#gain)

InterfaceGroup(class:knob) --predicate--> StyleSelector
InterfaceGroup(tag:input)  --predicate--> StyleSelector
InterfaceState(focus)      --predicate--> StyleSelector
StyleRelationship(descendant) ----------> StyleSelector

StyleValue(2px) -> StyleDeclaration(outline-width) -> StyleRule
StyleSelector -------------------------------------> StyleRule
StyleRule -----------------------------------------> StyleSheetRoot
StyleSheetRoot ------------------------------------> InterfaceRoot
```

When the input is focused, resolution adds a `StyleMatch`; cascade evaluation
then produces `ComputedProperty(input#gain, outline-width) = 2px` with links
back to the declaration, rule, selector, and focus state that caused it.

## 10. Cascade and deterministic order

CSS source order is necessary but not sufficient. Each declaration candidate
must carry a deterministic cascade key containing at least:

```text
origin
layer order
importance
selector specificity
scope/proximity when supported
stylesheet attachment order
rule source order
declaration source order
```

Authored order remains the final tie-breaker. It does not replace specificity,
importance, or inheritance, and none of those values changes containment
order.

`ComputedProperty` should retain the complete winning key and rejected
candidates. That makes CSS translation and debugging inspectable instead of
leaving backend-specific style engines as unexplained authorities.

## 11. Layout, paint, and events are consumers

HTML and CSS do not need to prescribe one layout implementation. Resolved
properties feed later graphs:

```text
HTML containment + ComputedProperty
              -> layout constraints/boxes
              -> paint order/resources
              -> host presentation
```

- `BrowserUIOperations` may emit HTML/CSS and let the browser perform these
  derived stages.
- `SDLUIOperations` and `PygameOpenGLUIOperations` may interpret the same computed properties into box,
  text, clipping, and paint operations.
- A WASM/native layout solver may calculate expensive window placement or
  spatial indexing while still consuming the same resolved graph.

Event ancestry derives from HTML containment. Hit regions derive from layout.
Paint stacking derives from computed style. These facts meet during dispatch
but remain separately inspectable.

## 12. Required `AbstractUI` root-backend contract

Every registered root backend must provide equivalent operations, even when its
host implements them differently:

```text
initialize system
create/destroy interface root
query viewport and pixel scale
poll or receive normalized events
make presentation/graphics context current
begin frame
present frame
request next frame
resolve shared resource
shutdown system
```

Browser `present frame` may be implicit and `request next frame` maps to the
browser scheduler. SDL/Pygame usually present explicitly. The canonical
contract records the semantic operation; backend tables record its spelling.

Normalized root events initially need:

```text
close resize scale-change focus blur
pointer-move pointer-down pointer-up wheel
key-down key-up text-input
frame tick
```

Host-specific event payloads may be retained as provenance, but portable
handlers consume the normalized vocabulary.

### Required backend-table columns

The `AbstractUI` backend conformance table must not merely record whether a
method exists. For every primitive and annotation concept it records:

```text
canonical name
semantic inputs and outputs
graph node/edge lowering
prosaic annotation lowering
world-map lowering
browser lowering
SDL lowering
Pygame/PyOpenGL lowering
accessibility route
state/action identity preservation
known shortfall and reason
```

The annotation and world-map columns are compulsory. A new operation is not
universally admitted merely because HTML or one native backend can draw it.

## 13. Proposed implementation sequence

1. Add `InterfaceSystemRoot` without changing the current standalone
   `InterfaceRoot` HTML behavior.
2. Define the `AbstractUI` operation/annotation tables with compulsory
   annotation and world-map backend columns.
3. Implement `GraphUIOperations` plus the prosaic `AnnotationUIOperations` as
   the first executable conformance pair.
4. Materialize `InterfaceIdentity`, `InterfaceGroup`, and
   `InterfaceProperty` during HTML ingestion.
5. Expand the bounded HTML vocabulary table with legal capabilities and
   states rather than adding tag-specific canonical node classes.
6. Add a CSS parser that produces `StyleSheetRoot` through `StyleCondition`
   without resolving selectors.
7. Add attachment and selector resolution, producing `StyleMatch` nodes.
8. Add deterministic cascade reduction into `ComputedProperty` nodes.
9. Implement `WorldMapUIOperations` against the same identities, properties,
   states, and actions; test equivalence against the annotation backend.
10. Implement and register browser, SDL C/C++, and Pygame/PyOpenGL
   `AbstractUI` backends against the same primitive/root contract.
11. Only then connect backend-specific layout/paint optimizations.

## 14. Decisions this document makes

- The absolute root is `InterfaceSystemRoot`; one presentation surface is
  `InterfaceRoot`.
- `AbstractUI` is the public semantic operation surface and owns a backend
  registry patterned after `AbstractTensor`.
- `AnnotationUIOperations` and `WorldMapUIOperations` are mandatory semantic
  reference backends and compulsory columns in backend conformance tables.
- SDL, Pygame/PyOpenGL, and the browser are backend realizations, not canonical
  node names.
- All HTML elements remain `InterfaceContainer` nodes selected by vocabulary
  tokens and capabilities.
- CSS has an independently rooted graph.
- CSS crosses into HTML through stable identities, groups, properties, and
  states.
- OOP adornments lower through a shared typed annotation schema before either
  reference backend interprets them.
- The prosaic projection is the reliable accessible route; the world projection
  preserves the same consequential flow through spatial interaction.
- Selector matches and computed properties are derived nodes with provenance.
- Containment, style, layout, paint, and event routing remain distinct typed
  relationships.

## 15. Open decisions before code expansion

- Whether `InterfaceIdentity` is always a material graph node or a typed
  reference-table entry projected as a node on demand.
- Whether a stylesheet import remains a second `StyleSheetRoot` connected by
  an import edge or is flattened with retained source provenance.
- The exact supported selector and CSS property profile for the first native
  SDL/Pygame backend.
- Whether multiple Pygame/SDL windows are required in the first backend
  or merely permitted by the canonical system/root distinction.
- Where computed layout boxes live: a new layout graph, `hierarchy_plan`, or a
  typed interface member added beside the existing dual IR.
- Which minimal world archetype vocabulary is sufficient to demonstrate
  region/building/interior/action equivalence without turning the canonical
  graph into a game-engine ontology.
- How consequence/effect annotations correlate with `ControlProgram` effects
  and permissions without duplicating their authority.

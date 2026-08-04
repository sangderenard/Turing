# Shader component ABI

`turing.shader-component.v1` is the runtime contract shared by Python-emitted
WGSL compute, desktop GLSL 4.30 compute, and GLSL ES 3.00 fragment components.
Shader language syntax is a decoration; shells link logical ports rather than
parsing declarations from generated source.

## Component record

Every component declares:

- a stable component identity, language, stage, and entry point;
- contiguous logical port slots, each with role, dtype, value identity,
  backend binding, and transport decoration;
- backend decorations such as workgroup size, arena binding, and execution
  model; and
- an eight-word sentinel header.

The logical ordering is always feeds, outputs, then uniforms. Consequently a
GLSL sampler, a GLSL SSBO arena slot, and a WGSL storage binding can describe
the same component port even though their backend binding syntax differs.

## Sentinel words

```text
magic version endian generation ready error port_count checksum
```

The fixed magic, ABI version, byte-order word, port count, and checksum are
validated before dispatch/draw. Generation prevents stale output from a prior
dispatch being accepted as current output. Ready and error are published after
completion. This header travels with local cached artifacts and online
messages; the receiver applies the same validation in either case.

## External links

External links have two independent classifications:

- scope: `system-local` or `online-cross-program`;
- transport: `shared-arena`, `compiled-artifact`, or `online-message`.

Online scope requires an endpoint and online-message transport. Local links
cannot silently become network links. Each link connects exactly one typed
output port to one typed feed port and lowers to an explicit SSA `Call` whose
attributes retain scope, transport, endpoint, alias policy, component/slot
identities, and sentinel generation rule.

Feedback links must be non-aliasing versioned boundaries. The multi-shell
planner omits declared feedback edges from the within-generation topological
order; an undeclared cycle fails closed.

## Hierarchical shells

The existing `PlanClosure`/`PlanCall` hierarchy remains the authority for
nested call structure. `validate_hierarchical_component_plan` consumes its
explicit argument and result bindings and proves that each endpoint maps to
exactly one canonical component port. It never rediscovers relationships from
source names, observed values, or generated shader text.

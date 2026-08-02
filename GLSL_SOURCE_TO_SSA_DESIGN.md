# GLSL source to existing SSA

## Observed shader surfaces

The Pluck demo is not one shader dialect or one program.  Its inline sources
include GLSL 3.30 vertex/fragment programs for body, plate, line, per-vertex
colour, marching, sensor blit, and HUD rendering, plus GLSL 4.30 compute
programs for ray work, sensor accumulation, and decay.  The important split is
therefore expression semantics versus stage/storage ABI.

The live spectral text demo does not define a parallel text-specific shader
language.  It launches `exposure_render_demo.py`, which supplies the shared
optical renderer with the repository shader directory.  Font/layout work is
upstream scene construction; the photographed result uses the same optical
shader family.  Text-specific translation belongs in layout/string processing,
not in a new numerical SSA opcode set.

## Boundary

```text
GLSL spelling
    -> lexical source table
    -> existing Handler instructions only
    -> existing SSA analysis / ProcessGraph conversion
    -> existing backend emitter
```

The lexical table is bidirectional so an existing Handler can be printed with
one deterministic GLSL spelling.  Aliases are accepted only at the source
edge.  They are not retained in SSA.

Composite GLSL conveniences lower as graphs of existing instructions:

- `min` / `max`: comparison plus `Select`
- `clamp`: `max`, then `min`
- `mix`: `x + (y - x) * a`
- `step`: comparison plus constants and `Select`
- `smoothstep`: subtract/divide, clamp, and cubic arithmetic
- `inversesqrt`: `1 / sqrt(x)`
- `distance`: subtract, then length

Existing ProcessGraph mathematical operations such as `sin`, `sqrt`, `dot`,
and `cross` use the existing SSA `Call` instruction with the canonical
ProcessGraph callee.  A source spelling with no exact current operation is not
wrapped in a fallback instruction; it becomes a normal lowering shortfall.

## WebGL specialization

WebGL 2 borrows common GLSL scalar expressions, but not the desktop compute
ABI.  SSA is handed to the existing WebGL fragment emitter, which owns texture
feeds, fragment coordinates, MRT outputs, and GLSL ES version/precision text.

Texture sampling and fragment derivatives (`texture`, `texelFetch`, `dFdx`,
`dFdy`, `fwidth`) are explicitly identified as currently unlowered.  The table
exists to make the diagnostic target-specific, not to create replacement SSA
operators.

## Current supported source slice

The first source reader intentionally accepts scalar, straight-line entry
functions:

- `uniform` and `in` scalar arguments
- declared `out` values and `return`
- scalar declarations and direct-name assignments
- unary/binary operators, compound assignments, calls, and ternaries
- scalar casts and the composite recipes above

The next useful slices, in order suggested by the Pluck/optical shaders, are:

1. scalar/vector component indexing and swizzles through existing addressing
   operations;
2. structured vector/matrix construction without pretending it is scalar;
3. structured control flow using existing branches, blocks, and Phi values;
4. texture/image/storage operations only where an exact existing operation is
   available;
5. stage inputs, SSBO/image layout, atomics, barriers, and invocation built-ins
   as backend ABI specialization rather than numerical operators.

Until each slice has an exact lowering, it stays visible in shortfall reports.

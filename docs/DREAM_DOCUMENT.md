# Dream documents

`turing.dream-document.v1` is a byte-scanned polyglot program container. A
single `.dream` file may contain Python, JavaScript, GLSL compute, GLSL
fragment, or future language arenas without requiring the complete file to be
valid source in any one language.

## Blocks

Ordinary language arena:

```text
/*@turing.segment.v1
id = setup
language = python
encoding = utf-8
inputs = configuration
outputs = state
@end*/
...payload...
/*@turing.end*/
```

Shader arena:

```text
/*@turing.shader.v1
id = update
language = glsl
stage = compute
inputs = state
outputs = colors
abi = turing.shader-component.v1
@end*/
...shader source...
/*@turing.end*/
```

A shader block is intrinsically an in-place device deployment. It does not
need a separate language-switch or launch directive. The runtime raises the
GPU-active indicator when entering the block and clears it on departure.

Parallel deployment is structure rather than locking syntax:

```text
/*@turing.parallel.v1
id = frame
members = cpu-step, gpu-colors
join = frame-barrier
@end*/
```

Members are submitted together and joined. The runtime does not manufacture a
shared-state lock; blocks communicate through declared ports or deliberately
share an arena supplied by their host.

## Loading

The byte scanner validates restricted-ASCII headers, declared encodings,
unique identities, optional SHA-256 hashes, parallel membership, and complete
framing before invoking a language parser. It projects every executable block
to a `turing.card-graph.v1` card with a content-hash cache key, sequential read
path, resident boundary connections, parallel deployment records, and block
metadata. `DreamRuntime` is the read head over that graph.

Inspect the initial simulator:

```text
python -m src.compiler.dream_document examples/reversible_chip_simulator.dream --inspect
```

Reference-run its Python blocks and simulated shader deployments:

```text
python -m src.compiler.dream_document examples/reversible_chip_simulator.dream --run-reference
```

A live OpenGL host replaces only the shader-deployer callback; it does not
rewrite the document or its graph.

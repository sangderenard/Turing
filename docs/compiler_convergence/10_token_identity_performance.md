# Task 10 — deterministic token-identity performance watch

## Observed code state

- `reduce_abstract_tensor_topology()` builds a structural token chain for every
  node.  Each chain includes a full `ast.dump()` of the node and full
  descriptions of every parent, then `lexicographical_topological_order()`
  compares those tuples.
- Equal base chains receive a local `version:N` according to deterministic
  graph insertion order.  Cross-process tests cover the present small example.
- `structural_context_tokens()` recursively sorts mappings and lexicalizes all
  leaf values.  `encode_identity_tokens()` is a reversible base-257 encoding
  used per token; identities remain vectors of those token IDs in the corpus.
- There is no large-program timing or memory guard for canonical relabeling.

## Work sequence

1. Instrument reduction phases without changing identity semantics: base
   description, tokenization, topological ordering, and relabeling.
2. Measure node count, token count/bytes, peak memory, and phase time on a
   ladder including BLAS kernels and `re._compile` if it reaches reduction.
3. If token construction dominates, intern repeated node/parent descriptions
   or compare compact deterministic token vectors while retaining the full
   reversible chain as compiler metadata.  Do not introduce a cache-dependent
   identity or hash-only collision path.
4. Add a scale regression threshold only after collecting a stable baseline;
   avoid machine-fragile wall-time assertions in ordinary CI.

## Acceptance

- Same source remains cross-process deterministic and fully explainable.
- Any compact comparison form round-trips to or indexes the retained token
  chain without changing dense IDs.
- Large-program reports separate token cost from the rest of topology
  reduction and can detect superlinear regressions.

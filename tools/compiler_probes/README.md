# Compiler probes and repair tools

These scripts were moved from the root of `build/` on 2026-09-24. They are
hand-authored diagnostic, replay, inspection, and historical repair programs,
not generated build products. Keeping them under an ignored build directory
made a safe cleanup impossible.

The relocation preserves their original convention:

- run them from the repository root;
- generated inputs and outputs continue to live under `build/`;
- imports resolve from the repository root;
- scripts that invoke another probe now use `tools/compiler_probes/`.

Several `_patch_*.py` files are historical source-rewrite tools. Inspect their
target and current applicability before running them; their presence here is a
record, not a recommendation to replay old mutations.


# Participating-medium catalog export verification

The user requested several vapors and dusts in the existing material catalog.
The canonical source remains the sibling spectral-analyzer/configs/materials
YAML library, not a new Turing registry. Six optional medium profiles were
added there: water, ethanol and acetone vapor; coarse quartz, calcite and
halite dust. Unknown molecular spectra remain unavailable rather than becoming
fog. Coarse-dust extinction is a labeled, explicit opt-in approximation.

Turing's existing material_library_export already preserves the definitions.
Only its regression count and source-preservation assertions changed here:
53 source records, 44 compiled editor surface records, 6 medium definitions,
9 existing unresolved color profiles. Both export tests pass with the installed
render Python, which has PyYAML. The first new warning assertion expected the
wrong substring; it was corrected to the actual no-inferred-opacity warning.
No exporter runtime changes, compiler changes, lowering, GPU jobs, or deployment
were performed for this catalog extension.

The real DiscreteFluid.sample_at publication check in the sibling renderer
also passes using the canonical quartz coefficient. Published sigma is
584.5610961914062 m^-1 versus the independent kernel sum 584.5610957580724;
state is unchanged and empty particles yield zero. This synthetic fixture is
not a real atmosphere or a completed bowl/liquid contact scenario.

Full schema, sources, limits and reproduction commands:
../spectral-analyzer/docs/20261111-participating-medium-catalog.md (from the
workspace root; the document is in the separate renderer repository).

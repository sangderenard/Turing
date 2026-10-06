"""One list of the files that implement the deployment-strategy stage.

``glsl_deployment_strategy.py`` is being compartmentalized into sibling
modules by pure relocation.  Three whole-file digests must cover every one of
those files, so each of them reads this single list instead of naming files
itself: ``site_bundle._BUNDLE_COMPILER_IMPLEMENTATION_FILES``,
``project_compilation_product.compiler_toolchain_fingerprint`` and
``glsl_backend._lowering_implementation_digest``.

Paths are repository-relative and POSIX-style.  Every module relocated out of
``glsl_deployment_strategy.py`` is added here in the commit that moves it.
"""

from __future__ import annotations

COMPILER_IMPLEMENTATION_FILES: tuple[str, ...] = (
    "src/compiler/glsl_deployment_strategy.py",
    "src/compiler/planner_concordance_posts.py",
    "src/compiler/formal_shape_ledger.py",
    "src/compiler/deployment_profiler.py",
    "src/compiler/region_scheduling.py",
    "src/compiler/call_argument_binding.py",
    "src/compiler/source_node_facts.py",
)

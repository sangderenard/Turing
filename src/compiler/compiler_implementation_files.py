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
    "src/compiler/receiver_resolution_prepasses.py",
    "src/compiler/planned_shell_tree.py",
    "src/compiler/planner_concordance_posts.py",
    "src/compiler/formal_shape_ledger.py",
    "src/compiler/deployment_profiler.py",
    "src/compiler/region_scheduling.py",
    "src/compiler/call_argument_binding.py",
    "src/compiler/source_node_facts.py",
    # relocated out of fortran_c_shell.py (split/shell-leaves)
    "src/compiler/declared_piece_signature.py",
    "src/compiler/ast_control_normalization.py",
    "src/compiler/native_link_audit.py",
    "src/compiler/native_shell_abi.py",
    "src/compiler/authored_parameter_abi.py",
    "src/compiler/frame_identity_book.py",
    "src/compiler/frame_storage_roles.py",
    "src/compiler/resident_sequence_materialization.py",
    "src/compiler/sequence_storage_binding.py",
)

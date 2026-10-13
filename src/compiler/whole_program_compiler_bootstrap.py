"""Versioned native partitions of the compiler's existing outer stages.

The abandoned v1 experiment tried to replace the compiler with one DLL rooted
at ``compile_program``. That erased the useful boundaries already proved by
the compiler and made one enormous Python-object interface the prerequisite
for every gain. This module now describes one composition instead:

* the Python coordinator and IdentityBook remain the semantic authority;
* existing native foundations are named, versioned boundary components;
* the whole outer compiler stage is submitted through the public partitioned
  source compiler used by ordinary compiled programs.

Every component version includes its authored source, its declared boundary
contract and the compiler toolchain fingerprint. Source changes therefore
select a new versioned product directory and can flow into later compiles;
stale native code is never silently selected for a changed source tree.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .native_compiler_accelerators import (
    BROADCAST_SHAPES, DEAD_METADATA_ORDER, DEPENDENCY_LEVELS, METADATA_COPY,
    READ_ONLY_MODULE, TOPOLOGICAL_ORDER,
)


BOOTSTRAP_SCHEMA = "turing.partitioned-compiler-bootstrap.v2"
STRATEGY = "versioned-native-boundary-composition"


@dataclass(frozen=True)
class CompilerBoundary:
    """One coarse compiler boundary or an existing native foundation."""

    name: str
    kind: str
    source: str
    entrypoint: str
    semantic: str
    consumes: str
    produces: str
    disposition: str = "available"

    def receipt(self, repository: Path, toolchain: Mapping[str, Any]) -> dict[str, Any]:
        source_path = repository / self.source
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        contract = {
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "entrypoint": self.entrypoint,
            "semantic": self.semantic,
            "consumes": self.consumes,
            "produces": self.produces,
            "disposition": self.disposition,
        }
        version = hashlib.sha256(json.dumps({
            "contract": contract,
            "source_sha256": source_sha256,
            "compiler_toolchain_sha256": toolchain["sha256"],
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return {
            **contract,
            "source_sha256": source_sha256,
            "compiler_toolchain_sha256": toolchain["sha256"],
            "version": version,
        }


# These are the boundaries that exist in the compiler, not a second registry.
# Semantics match native_compiler_accelerators exactly. The metadata copier is
# retained as negative evidence: it is versioned, but it is not an admitted
# component because its frequently crossed object boundary lost decisively.
BOUNDARIES = (
    CompilerBoundary(
        "graph-topology", "native-foundation",
        "src/compiler/graph_refinement_kernels.py", "topological_order",
        TOPOLOGICAL_ORDER,
        "one admitted integer graph", "one lexical order",
    ),
    CompilerBoundary(
        "graph-dependency-levels", "native-foundation",
        "src/compiler/graph_refinement_kernels.py", "dependency_levels",
        DEPENDENCY_LEVELS,
        "one admitted graph and order", "one dependency-level table",
    ),
    CompilerBoundary(
        "graph-dead-metadata", "native-foundation",
        "src/compiler/graph_refinement_kernels.py", "dead_metadata_order",
        DEAD_METADATA_ORDER,
        "one admitted metadata graph", "one ordered removal proposal",
    ),
    CompilerBoundary(
        "descriptor-broadcast", "native-foundation",
        "src/compiler/graph_refinement_kernels.py", "broadcast_shapes",
        BROADCAST_SHAPES,
        "one descriptor batch", "one shape proposal table",
    ),
    CompilerBoundary(
        "whole-module-storage-effects", "native-foundation",
        "src/compiler/graph_refinement_kernels.py", "read_only_storage_effects",
        READ_ONLY_MODULE,
        "one complete SSA module fact table",
        "read-only and pointer-storage fixed points",
    ),
    CompilerBoundary(
        "metadata-copy", "rejected-boundary",
        "src/compiler/process_graph_fusion.py", "isolate_metadata",
        METADATA_COPY,
        "frequent Python object batches", "copied Python object batches",
        disposition="rejected-hot-boundary",
    ),
    CompilerBoundary(
        "process-graph-lowering", "outer-partition",
        "src/common/tensors/accelerator_backends/aot_compile.py",
        "_lower_process_graph_to_compilation",
        "compiler.process_graph.lowering",
        "one complete built ProcessGraph plus its compile policy",
        "one AOTCompilation containing planned graph, control and region products",
        disposition="bootstrap-target",
    ),
    CompilerBoundary(
        "python-ast-ingestion", "frontend-partition",
        "src/transmogrifier/graph/graph_express2.py", "ProcessGraph.build_from_ast",
        "compiler.frontend.python_ast_ingestion",
        "one complete Python AST and extraction policy", "one built ProcessGraph",
        disposition="candidate-after-process-graph-lowering",
    ),
    CompilerBoundary(
        "sympy-ingestion", "frontend-partition",
        "src/compiler/symbolic_process_graph.py", "ingest_sympy_expressions",
        "compiler.frontend.sympy_ingestion",
        "one complete named SymPy expression set", "one canonical ProcessGraph",
        disposition="candidate-after-process-graph-lowering",
    ),
)


@dataclass(frozen=True)
class WholeProgramCompilerProduct:
    """A canonical partitioned product for one versioned outer partition."""

    boundary: Mapping[str, Any]
    product: Mapping[str, Any]
    manifest: Path


def compiler_boundary_receipts() -> tuple[dict[str, Any], ...]:
    from .project_compilation_product import compiler_toolchain_fingerprint

    repository = Path(__file__).resolve().parents[2]
    toolchain = compiler_toolchain_fingerprint()
    return tuple(boundary.receipt(repository, toolchain) for boundary in BOUNDARIES)


def plan_whole_program_compiler(
    directory: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
    **_legacy_options: Any,
) -> dict[str, Any]:
    """Publish the versioned composition without launching a compiler build."""

    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    boundaries = compiler_boundary_receipts()
    target = next(
        boundary for boundary in boundaries
        if boundary["name"] == "process-graph-lowering"
    )
    receipt = {
        "schema": BOOTSTRAP_SCHEMA,
        "stage": "planned",
        "strategy": STRATEGY,
        "semantic_authority": "python-coordinator-and-identity-book",
        "boundaries": list(boundaries),
        "next_partition": target,
        "source_update_policy": (
            "source-or-toolchain changes select a new versioned product; "
            "only receipt-verified matching products may activate"
        ),
    }
    path = destination / "plan.json"
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    if progress is not None:
        progress(
            "compiler-bootstrap: planned versioned boundary composition; "
            f"next={target['name']} version={target['version'][:12]}"
        )
    return receipt


def build_whole_program_compiler(
    directory: str | Path,
    *,
    jobs: int | None = None,
    source_compilation: str = "auto",
    max_total_resident_bytes: int | None = 10 << 30,
    minimum_total_resident_bytes: int = 2 << 30,
    system_reserve_bytes: int = 2 << 30,
    adaptive_memory_ceiling: bool = True,
    worker_resident_reservation_bytes: int = 2 << 30,
    max_worker_memory_bytes: int | None = 10 << 30,
    progress: Callable[[str], None] | None = None,
) -> WholeProgramCompilerProduct:
    """Bootstrap the complete ProcessGraph-to-AOT outer partition.

    Bootstrap uses the same public partitioned compiler entry as every other
    program. This function only selects the existing outer seam and gives the
    resulting product a source/toolchain-derived version directory; it owns no
    compilation, memory, timeout, or fallback policy of its own.
    """

    from .fortran_c_shell import lower_ast_source_to_ssa
    from .project_compilation_product import DEFAULT_PROJECT_EXTRACTION_CONTRACT

    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plan = plan_whole_program_compiler(destination, progress=progress)
    boundary = dict(plan["next_partition"])
    repository = Path(__file__).resolve().parents[2]
    product_root = (
        destination / "products" / boundary["name"] / boundary["version"]
    )

    source_path = repository / boundary["source"]
    source_module = ".".join(source_path.relative_to(repository).with_suffix("").parts)
    partitioned_options = {
        "max_total_resident_bytes": max_total_resident_bytes,
        "minimum_total_resident_bytes": minimum_total_resident_bytes,
        "system_reserve_bytes": system_reserve_bytes,
        "adaptive_memory_ceiling": adaptive_memory_ceiling,
        "worker_resident_reservation_bytes": worker_resident_reservation_bytes,
        "max_worker_memory_bytes": max_worker_memory_bytes,
    }
    if jobs is not None:
        partitioned_options["jobs"] = jobs
    lower_ast_source_to_ssa(
        source_path.read_text(encoding="utf-8"),
        boundary["entrypoint"],
        source_module=source_module,
        python_package=source_module.rpartition(".")[0],
        extraction_contract=DEFAULT_PROJECT_EXTRACTION_CONTRACT,
        name="compiler_bootstrap",
        source_compilation=source_compilation,
        compilation_directory=product_root,
        partitioned_options=partitioned_options,
        progress=progress,
    )
    product_manifest = product_root / "product" / "manifest.json"
    product = json.loads(product_manifest.read_text(encoding="utf-8"))
    manifest_value = {
        "schema": BOOTSTRAP_SCHEMA,
        "stage": "built",
        "strategy": STRATEGY,
        "boundary": boundary,
        "product": product,
        "product_directory": product_root.as_posix(),
    }
    manifest = destination / "manifest.json"
    manifest.write_text(
        json.dumps(manifest_value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return WholeProgramCompilerProduct(boundary, product, manifest)


__all__ = [
    "BOOTSTRAP_SCHEMA",
    "BOUNDARIES",
    "CompilerBoundary",
    "STRATEGY",
    "WholeProgramCompilerProduct",
    "build_whole_program_compiler",
    "compiler_boundary_receipts",
    "plan_whole_program_compiler",
]

"""Run the real recursive, disk-cached host extraction rooted at compile()."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import sys
from time import perf_counter

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from src.compiler.host_code_modules import (
    extract_host_code_library,
    materialize_host_code_library,
)


def main() -> None:
    started = perf_counter()
    library = extract_host_code_library(compile)
    if library is None:
        raise RuntimeError("compile() has no host-code identity")
    module = materialize_host_code_library(library)
    raw_categories = Counter(
        str(blocker.kind)
        for unit in library.units
        for blocker in unit.result.blockers
    )
    effective_categories = Counter(
        str(blocker.kind) for blocker in library.effective_blockers
    )
    report = {
        "root_cache_key": library.root_cache_key,
        "root_function": library.materialized_root_function,
        "unit_count": len(library.units),
        "function_count": len(module.functions),
        "dependency_occurrence_count": len(library.dependencies),
        "unresolved_dependency_occurrence_count": len(
            library.unresolved_dependencies
        ),
        "raw_blocker_occurrence_count": len(library.blockers),
        "raw_blocker_categories": dict(sorted(raw_categories.items())),
        "effective_blocker_occurrence_count": len(library.effective_blockers),
        "effective_blocker_categories": dict(sorted(effective_categories.items())),
        "machine_state_complete": library.machine_state_complete,
        "repository_ssa_complete": library.repository_ssa_complete,
        "elapsed_seconds": perf_counter() - started,
        "units": [
            {
                "cache_key": unit.cache_key,
                "provider": unit.identity.provider,
                "module_path": str(unit.identity.module_path),
                "symbol": unit.identity.symbol,
                "entry_rva": unit.identity.entry_rva,
                "cache_path": str(unit.cache_path),
                "cache_hit": unit.cache_hit,
                "function_count": len(unit.result.module.functions),
                "blocker_occurrence_count": len(unit.result.blockers),
            }
            for unit in library.units
        ],
        "dependency_occurrences": [
            {
                "source_cache_key": edge.source_cache_key,
                "external_identity": edge.external_identity,
                "target_cache_key": edge.target_cache_key,
                "resolution": edge.resolution,
                "source_address": edge.source_address,
            }
            for edge in library.dependencies
        ],
        "blocker_occurrences": [
            {
                "unit_cache_key": unit.cache_key,
                "occurrence": blocker.occurrence,
                "function_rva": blocker.function_rva,
                "function_name": blocker.function_name,
                "kind": blocker.kind,
                "address": blocker.address,
                "detail": blocker.detail,
                "external_identity": blocker.external_identity,
            }
            for unit in library.units
            for blocker in unit.result.blockers
        ],
        "effective_blocker_occurrences": [
            {
                "occurrence": blocker.occurrence,
                "function_rva": blocker.function_rva,
                "function_name": blocker.function_name,
                "kind": blocker.kind,
                "address": blocker.address,
                "detail": blocker.detail,
                "external_identity": blocker.external_identity,
            }
            for blocker in library.effective_blockers
        ],
    }
    output = Path("build/compile-host-library-report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({
        key: value for key, value in report.items()
        if key not in {
            "units", "dependency_occurrences", "blocker_occurrences",
            "effective_blocker_occurrences",
        }
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

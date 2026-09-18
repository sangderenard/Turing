"""Detailed spectral report for a saved lowered SSA module.

This is an analysis tool, not the colour-oriented ``spectral_view.py``.
It reads an already-lowered SSA pickle so it never confuses a source program
with one that silently failed to lower. It reports topology and control-loop
structure, then computes every selected region's spectrum through the existing
FFT or ``AT.eigh`` dispatch.

    python tools/spectral_graph_report.py
    python tools/spectral_graph_report.py --ssa build/run/control_repository_ssa.pkl
    python tools/spectral_graph_report.py --functions my_function --json report.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def newest_lowering() -> Path:
    """Return the newest standard lowered SSA artifact, or refuse clearly."""

    candidates = sorted(
        (ROOT / "build").glob("*/control_repository_ssa.pkl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            "no lowered SSA artifact under build/. Pass --ssa PATH after "
            "running a lowering that writes control_repository_ssa.pkl"
        )
    return candidates[0]


def _tensor_values(value: Any) -> list[float]:
    import numpy as np

    raw = value.data if hasattr(value, "data") else value
    return [float(item) for item in np.asarray(raw).reshape(-1)]


def _node_label(node: Any) -> str:
    return repr(node)


def _region_record(label: str, field: Any, node_order: tuple[Any, ...], *, loop: Any,
                   circulant_tol: float) -> dict[str, Any]:
    from src.compiler.spectral_graph_analysis import (
        analyze_region_spectrum,
        is_circulant,
        symmetric_adjacency,
    )

    included = set(node_order)
    adjacency, _index = symmetric_adjacency(node_order, tuple(
        edge for edge in field.edge_list()
        if edge[0] in included and edge[1] in included
    ))
    circulant = is_circulant(adjacency, tol=circulant_tol)
    record: dict[str, Any] = {
        "label": label,
        "node_count": len(node_order),
        "nodes": [_node_label(node) for node in node_order],
        "circulant": circulant,
        "method": None,
        "status": "analysed",
    }
    spectrum = analyze_region_spectrum(
        field, node_order, loop=loop, circulant_tol=circulant_tol,
    )
    record.update(
        status="analysed",
        method=spectrum.method,
        eigenvalues=sorted(_tensor_values(spectrum.eigenvalues)),
    )
    if len(node_order) >= 2:
        record["spectral_gap"] = spectrum.spectral_gap()
    return record


def build_report(module: Any, *, functions: Iterable[str] | None,
                 circulant_tol: float) -> dict[str, Any]:
    """Build a serialisable report without running influence propagation."""

    from src.compiler.influence_field import InfluenceContract, field_from_ssa
    from src.compiler.spectral_graph_analysis import natural_loop_regions

    selected = tuple(functions or ())
    field = field_from_ssa(
        module,
        InfluenceContract(enabled=True),
        functions=selected or None,
    )
    nodes = field.node_keys()
    edges = field.edge_list()
    roles = Counter(role for _source, _target, role in edges)
    loops = natural_loop_regions(field)

    regions = [
        _region_record(
            "whole graph", field, nodes, loop=None, circulant_tol=circulant_tol,
        )
    ]
    for index, loop in enumerate(loops, start=1):
        node_order = tuple(node for node in nodes if node in loop.nodes)
        region = _region_record(
            f"loop {index}", field, node_order, loop=loop, circulant_tol=circulant_tol,
        )
        region["header"] = _node_label(loop.header)
        region["latch"] = _node_label(loop.latch)
        regions.append(region)

    return {
        "schema": "turing-spectral-graph-report-v1",
        "functions": list(selected) if selected else "all",
        "topology": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "edge_roles": dict(sorted(roles.items())),
            "loop_count": len(loops),
        },
        "regions": regions,
    }


def print_report(report: dict[str, Any]) -> None:
    topology = report["topology"]
    print("SPECTRAL GRAPH REPORT")
    print(f"functions: {report['functions']}")
    print(f"topology: {topology['node_count']} nodes, {topology['edge_count']} edges")
    print(f"edge roles: {topology['edge_roles']}")
    print(f"natural loops: {topology['loop_count']}")
    for region in report["regions"]:
        print(f"\n{region['label']}: {region['node_count']} nodes; "
              f"circulant={region['circulant']}")
        print(f"  {region['method']}; spectral gap={region.get('spectral_gap', 'n/a')}")
        print(f"  eigenvalues: {region['eigenvalues']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssa", type=Path, default=None,
                        help="lowered control_repository_ssa.pkl (default: newest under build/)")
    parser.add_argument("--functions", default="",
                        help="comma-separated SSA function names (default: all)")
    parser.add_argument("--circulant-tol", type=float, default=1e-9)
    parser.add_argument("--json", type=Path, default=None,
                        help="write the full machine-readable report here")
    args = parser.parse_args()
    source = args.ssa or newest_lowering()
    if not source.is_file():
        parser.error(f"SSA artifact does not exist: {source}")
    with source.open("rb") as stream:
        module, _outputs, _exports = pickle.load(stream)
    functions = tuple(name.strip() for name in args.functions.split(",") if name.strip())
    report = build_report(
        module, functions=functions, circulant_tol=args.circulant_tol,
    )
    report["ssa_artifact"] = str(source)
    print_report(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

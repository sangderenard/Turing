"""A bounded same-Python eager/native experiment, not a replacement compiler.

Default: run the authored dependency_levels eagerly, with no compilation.
--compile: build it through the existing compiler accelerator API.
--native-payload: reuse an existing graph_kernel_native_worker test payload.
This is a trusted local pickle, not an arbitrary downloaded artifact.
"""
from __future__ import annotations

import argparse
import dataclasses
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import pickle
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.compiler.graph_refinement_kernels import dependency_levels

NAMES = ("n", "preference", "order", "parent_offsets", "parents",
         "child_offsets", "children", "asap", "levels")


def source_digest():
    return hashlib.sha256((ROOT / "src/compiler/graph_refinement_kernels.py").read_bytes()).hexdigest()


def save_payload(provider, directory):
    """Reuse the existing native-worker payload; pin this experiment's source."""
    artifact = dataclasses.replace(provider.metadata["artifact"], emission=None,
        _entry=None, _library=None, _validation_error_reset=None, _validation_error_take=None)
    path = directory / "dependency-levels.pkl"
    path.write_bytes(pickle.dumps((artifact, provider.metadata["abi"]["parameters"],
                                  ("dependency_graphs",), [])))
    path.with_suffix(".receipt.json").write_text(json.dumps({
        "source_sha256": source_digest(),
        "payload_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(Path(artifact.library_path).read_bytes()).hexdigest(),
        "source_compiler": provider.metadata["source_compiler"],
        "concordance": provider.metadata["concordance"],
    }, indent=2), encoding="utf-8")
    return path


def case(nodes):
    """Two independent chains with redundant two-hop edges: known levels."""
    if not 2 <= nodes <= 65536:
        raise ValueError("probe nodes must be in [2, 65536]")
    split = nodes // 2
    parents = [[] for _ in range(nodes)]
    children = [[] for _ in range(nodes)]
    expected = np.empty(nodes, dtype=np.int64)
    for first, stop in ((0, split), (split, nodes)):
        for node in range(first, stop):
            expected[node] = node - first
            for parent in range(max(first, node - 2), node):
                parents[node].append(parent)
                children[parent].append(node)

    def csr(rows):
        offsets, edges = [0], []
        for row in rows:
            edges.extend(row)
            offsets.append(len(edges))
        return np.asarray(offsets, dtype=np.int64), np.asarray(edges or [0], dtype=np.int64)

    parent_offsets, parent_edges = csr(parents)
    child_offsets, child_edges = csr(children)
    values = dict(n=np.array(nodes, dtype=np.int64), preference=np.array(0, dtype=np.int64),
                  order=np.arange(nodes, dtype=np.int64), parent_offsets=parent_offsets,
                  parents=parent_edges, child_offsets=child_offsets, children=child_edges,
                  asap=np.zeros(nodes, dtype=np.int64), levels=np.zeros(nodes, dtype=np.int64))
    return values, expected


def measure(nodes=256, repeats=8, *, artifact=None, parameters=None, workers=1):
    """Keep packing/setup separate from kernel time; each task owns its arena."""
    if not 1 <= repeats <= 100 or workers not in (1, 2):
        raise ValueError("repeats must be 1..100; workers must be 1 or 2")
    if artifact is not None:
        if not artifact.complete:
            raise ValueError("native artifact is incomplete")
        if "@turing_validation_error(" in artifact.llvm_ir:
            raise ValueError("parallel probe refuses shared validation globals")
        if set(parameters or {}) != set(NAMES):
            raise ValueError("native payload is not the authored dependency_levels ABI")
        artifact.entry()  # Load on coordinator; invocations share code only.
    rendezvous = threading.Barrier(workers) if workers > 1 else None

    def one(preference):
        started = time.perf_counter()
        values, expected = case(nodes)
        values["preference"][...] = preference
        execution = None
        if artifact is not None:
            from src.compiler.ssa_llvm_backend import prepare_artifact_execution
            execution = prepare_artifact_execution(artifact, {
                int(parameters[name]): values[name] for name in NAMES})
        setup = time.perf_counter() - started
        if rendezvous is not None:
            rendezvous.wait(timeout=5)
        started = time.perf_counter()
        for _ in range(repeats):
            values["asap"].fill(0)
            values["levels"].fill(0)
            if execution is None:
                count = dependency_levels(int(values["n"]), int(values["preference"]),
                                          *(values[name] for name in NAMES[2:]))
                assert count == nodes
            else:
                execution.run()
        elapsed = time.perf_counter() - started
        np.testing.assert_array_equal(values["asap"], expected)
        np.testing.assert_array_equal(values["levels"], expected)
        return {"preference": "alap" if preference else "asap", "nodes": nodes,
                "edges": int(values["parent_offsets"][-1]), "repeats": repeats,
                "setup_seconds": setup, "kernel_seconds": elapsed,
                "owned_input_scratch_bytes": sum(value.nbytes for value in values.values()),
                "worker_thread": threading.get_ident(),
                "output_sha256": hashlib.sha256(values["levels"].tobytes()).hexdigest(),
                "exact": True}

    started = time.perf_counter()
    if workers == 1:
        results = [one(preference) for preference in (0, 1)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, (0, 1)))
    return {"mode": "native" if artifact is not None else "eager", "workers": workers,
            "wall_seconds": time.perf_counter() - started, "cases": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--compile", type=Path, metavar="DIRECTORY")
    selection.add_argument("--native-payload", type=Path)
    args = parser.parse_args(argv)
    artifact, parameters = None, None
    payload = args.native_payload
    source_receipt = "eager authored source"
    from src.compiler.native_compiler_accelerators import DEPENDENCY_LEVELS, compiler_accelerators
    provider = compiler_accelerators.resolve(DEPENDENCY_LEVELS)
    if args.compile is not None:
        if args.compile.exists() and any(args.compile.iterdir()):
            raise ValueError("native probe compilation requires a fresh/empty directory")
        from src.compiler.native_compiler_accelerators import compile_and_register_dependency_levels_accelerator
        provider = compile_and_register_dependency_levels_accelerator(
            args.compile, progress=lambda message: None)
        payload = save_payload(provider, args.compile)
        source_receipt = "fresh public-source compile"
    if args.native_payload is not None:
        with args.native_payload.open("rb") as stream:
            artifact, parameters, names, _cases = pickle.load(stream)
        if names != ("dependency_graphs",):
            raise ValueError("expected an existing accepted dependency-level test payload")
        receipt_path = args.native_payload.with_suffix(".receipt.json")
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt["source_sha256"] != source_digest()
                    or receipt["payload_sha256"] != hashlib.sha256(args.native_payload.read_bytes()).hexdigest()
                    or receipt["library_sha256"] != hashlib.sha256(Path(artifact.library_path).read_bytes()).hexdigest()):
                raise ValueError("native probe source/payload/library receipt is stale")
            source_receipt = "source/payload/library digest matched"
        else:
            source_receipt = "explicit trusted test payload; no source digest receipt"
    elif provider.tier == "native":
        artifact = provider.metadata["artifact"]
        parameters = provider.metadata["abi"]["parameters"]
    result = measure(args.nodes, args.repeats, artifact=artifact, parameters=parameters,
                     workers=args.workers)
    result["authored_source"] = str(ROOT / "src/compiler/graph_refinement_kernels.py")
    result["native_library"] = str(artifact.library_path) if artifact is not None else None
    result["native_payload"] = str(payload) if payload is not None else None
    result["source_receipt"] = source_receipt
    print("GRAPH_KERNEL_MODE=" + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

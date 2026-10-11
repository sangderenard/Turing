"""Small public-entry lowering and chain-work counts; no trace or deadline.

--original-passes selects the frozen pre-worklist publication test oracle
and restores per-query publication at the read-only plan boundary.  The latter
retains one extra no-op first-read receipt check from the current boundary;
it is not a byte-for-byte historical whole-compiler performance baseline.
Stops after real SSA lowering, before LLVM emission or target execution.
"""
from pathlib import Path
import statistics
import sys
import time
import inspect
import os
import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, os.environ.get("TURING_PROBE_EXAMPLES", str(ROOT / "examples")))
from test_abi_identity_worklist import ORIGINAL, chain, run
from src.compiler import identity_concordance as concordance
from src.compiler import glsl_deployment_strategy as strategy
from src.compiler import fortran_c_shell as compiler
from tools.compiler_probes.probe_dt_unit_interfaces import main

if "--original-cone" in sys.argv:
    from descriptor_invalidation_full_scan import original_invalidation
    strategy._invalidate_tensor_descriptor_dependents = original_invalidation

original_mode = "--original-passes" in sys.argv
if original_mode:
    concordance.publish_program_abi_graph_identities = ORIGINAL
    descriptor = strategy._tensor_descriptor

    def fresh_publication(graph, node, seen=None):
        caller = inspect.currentframe().f_back
        if caller.f_code.co_name == "_operand_descriptor":
            caller = caller.f_back
        if caller.f_code.co_name in {"planned_descriptor", "fold_descriptor",
                                    "boundary_descriptor", "_propagate_callsite_tensor_specializations"}:
            seen = None
        return descriptor(graph, node, seen)

    strategy._tensor_descriptor = fresh_publication

publication = concordance.publish_program_abi_graph_identities
publication_calls = []
publication_callers = {}
metadata_work = {}
if "--metadata-work" in sys.argv:
    from src.compiler import process_graph_fusion as fusion
    extract = fusion.extract_clean_process_subgraph

    def observed_extract(graph, node_ids):
        nodes = tuple(node_ids)
        proper = bool(nodes) and len(set(nodes)) < len(graph.G)
        memo = set()

        def count(value):
            identity = id(value)
            if identity in memo:
                return 1
            if isinstance(value, (dict, list, tuple, set, frozenset)):
                memo.add(identity)
                if isinstance(value, dict):
                    return 1 + sum(count(key) + count(item) for key, item in value.items())
                return 1 + sum(count(item) for item in value)
            return 1

        for key, value in graph.G.graph.items():
            if proper and key in fusion._COMPARTMENT_SHARED_METADATA:
                continue
            if proper and key in fusion._COMPARTMENT_NODE_KEYED_METADATA:
                value = {node: value[node] for node in nodes if node in value}
            row = metadata_work.setdefault(("region" if proper else "whole", key), [0, 0])
            row[0] += 1
            row[1] += count(value)
        return extract(graph, nodes)

    fusion.extract_clean_process_subgraph = observed_extract
    strategy.extract_clean_process_subgraph = observed_extract


def measured_publication(graph, **kwargs):
    started = time.perf_counter()
    result = publication(graph, **kwargs)
    publication_calls.append((len(graph), time.perf_counter() - started))
    if "--caller-work" in sys.argv:
        caller = inspect.currentframe().f_back
        path = []
        for _ in range(4):
            if caller is None:
                break
            path.append((caller.f_code.co_name, caller.f_lineno))
            caller = caller.f_back
        row = publication_callers.setdefault(tuple(path), [0, 0])
        row[0] += 1
        row[1] += len(graph)
    return result


concordance.publish_program_abi_graph_identities = measured_publication
entry = compiler.lower_ast_source_to_ssa


class LoweringComplete(Exception):
    pass


def lower_only(*args, **kwargs):
    kwargs["progress"] = lambda _message: None
    publication_calls.clear()
    publication_callers.clear()
    metadata_work.clear()
    started = time.perf_counter()
    if "--profile-phases" in sys.argv:
        import cProfile
        import pstats
        profile = cProfile.Profile()
        result = profile.runcall(entry, *args, **kwargs)
        print("SMALL_LOWERING_PROFILE_CUMULATIVE", flush=True)
        pstats.Stats(profile).strip_dirs().sort_stats("cumulative").print_stats(25)
        print("SMALL_LOWERING_PROFILE_EXCLUSIVE", flush=True)
        pstats.Stats(profile).strip_dirs().sort_stats("tottime").print_stats(25)
        pstats.Stats(profile).strip_dirs().print_callers("io.open")
        pstats.Stats(profile).strip_dirs().print_callers("lzma.py:49")
    else:
        result = entry(*args, **kwargs)
    elapsed = time.perf_counter() - started
    proof = None
    if "--llvm-proof" in sys.argv:
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
        artifact = emit_ssa_function_to_llvm(result[0], result[2][0], entry_name="graph_scaling_proof")
        if not artifact.complete:
            raise RuntimeError(artifact.shortfalls)
        proof = hashlib.sha256(artifact.llvm_ir.encode("utf-8")).hexdigest()
    print({"mode": "original-passes" if original_mode else "worklists",
           "program": kwargs.get("name", "dt-proposal"),
           "lowering_seconds": elapsed,
           "abi_publication_calls": len(publication_calls),
           "abi_publication_seconds": sum(seconds for _, seconds in publication_calls),
           "abi_receipt_node_work": sum(nodes for nodes, _ in publication_calls),
           "functions": len(result[0].functions),
           "llvm_sha256": proof,
           "concordance": concordance.concordance_report(result[0]).splitlines()[0]}, flush=True)
    if publication_callers:
        print("PUBLICATION_CALLER_WORK", sorted(publication_callers.items(),
              key=lambda item: -item[1][1])[:15], flush=True)
    if metadata_work:
        print("METADATA_COPY_WORK", sorted(metadata_work.items(),
              key=lambda item: -item[1][1])[:25], flush=True)
    raise LoweringComplete()


compiler.lower_ast_source_to_ssa = lower_only
if "--arithmetic-scaling" in sys.argv:
    for size in (128, 256, 512):
        source = "def root(x):\n" + "".join(
            f"    y{n} = {'x' if n == 0 else 'y' + str(n - 1)} + {n + 1}.0\n"
            for n in range(size)
        ) + f"    return y{size - 1}\n"
        try:
            compiler.lower_ast_source_to_ssa(source, "root", name=f"arithmetic_{size}",
                extraction_contract=ROOT / "extraction_contracts/program_extraction.yaml")
        except LoweringComplete:
            pass
else:
    for _ in range(1 if "--one" in sys.argv else 3):
        try:
            main(ROOT / "build/serial-graph-scaling", proposal=True, serial=True)
        except LoweringComplete:
            pass

for size in (256, 512, 1024):
    samples = []
    for _ in range(3):
        started = time.perf_counter()
        _, visits, _ = run(publication, chain(size))
        samples.append(time.perf_counter() - started)
    print({"chain_nodes": size, "node_visits": visits,
           "median_seconds": statistics.median(samples)}, flush=True)

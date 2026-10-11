"""Crash-isolated runtime half of graph-kernel compiler micro-tests."""
from pathlib import Path
import ctypes
import faulthandler
import json
import os
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if os.name == "nt":
    ctypes.windll.kernel32.SetErrorMode(0x0002)  # No interactive crash reporter.
faulthandler.disable()  # Memory corruption must exit, not repeat stack dumps.
with open(sys.argv[1], "rb") as stream:
    artifact, parameters, names, arrays = pickle.load(stream)
from src.compiler.ssa_llvm_backend import prepare_artifact_execution

try:
    if names == ("graph_work_batch",):
        import time
        from types import SimpleNamespace
        import networkx as nx
        from src.compiler.graph_work import (
            RegionTopologyPool, bounded_schedule_regions, dependency_snapshot,
        )
        from src.compiler.identity_concordance import IdentityBook
        from src.compiler.native_compiler_accelerators import (
            TOPOLOGICAL_ORDER, DEPENDENCY_LEVELS, compiler_accelerators,
            _native_topological_runner, _native_dependency_levels_runner,
        )
        for semantic, runner, compiled, inputs in zip(
            (TOPOLOGICAL_ORDER, DEPENDENCY_LEVELS),
            (_native_topological_runner, _native_dependency_levels_runner), artifact, parameters,
        ):
            compiler_accelerators.register_native(semantic, runner(compiled, inputs),
                name=f"turing:{semantic}", library_path=compiled.library_path)
        pool = RegionTopologyPool(IdentityBook(), arrays["resident_regions"])
        results = []
        def admitted():
            for index, snapshot in enumerate(arrays["snapshots"]):
                pool.reserve()
                graph = nx.DiGraph()
                graph.add_nodes_from((node, {"parents": list(parents), "children": list(children)})
                                     for node, parents, children in snapshot.records)
                graph.add_edges_from(snapshot.edges)
                region = SimpleNamespace(G=graph, levels={},
                    scheduler=SimpleNamespace(G=graph, levels_asap={}, levels_alap={}))
                yield index, region, snapshot
        def finish(index, region, snapshot, proposal):
            results.append(proposal.levels)
            pool.add(region, reserved=True)
        started = time.perf_counter()
        try:
            bounded_schedule_regions(admitted(), workers=arrays["workers"],
                admission_limit=arrays["workers"], finish=finish,
                stats=pool.stats, executor=pool.worker_executor(arrays["workers"]))
            receipt = {"seconds": time.perf_counter() - started,
                       "stats": pool.summary(), "levels": results}
        finally:
            pool.close()
        print("GRAPH_WORK_BATCH=" + json.dumps(receipt), flush=True)
    elif names == ("storage_effect_modules",):
        from src.compiler.native_compiler_accelerators import _native_read_only_module_runner
        from src.compiler.ssa_llvm_backend import LLVMExecution
        original_run = LLVMExecution.run
        calls = [0]
        def counted(self):
            calls[0] += 1
            return original_run(self)
        LLVMExecution.run = counted
        run = _native_read_only_module_runner(artifact, parameters)
        from src.compiler.ssa_call_input_adapters import _infer_pointer_storage
        for case_index, case in enumerate(arrays):
            functions, expected = case[:2]
            before = calls[0]
            actual, storage = run(functions, include_storage_types=True)
            assert actual == expected, (case_index, [(key, actual[key], expected[key]) for key in actual if actual[key] != expected[key]])
            assert calls[0] - before == int(bool(expected)), "ONE native invocation per entire module"
            if len(case) == 3:
                _infer_pointer_storage(functions, storage)
                accounting = {(function.name, int(value.id)): dict(value.accounting or {})
                              for function in functions.values() for value in function.args}
                assert accounting == case[2], (case_index, storage,
                    [(key, accounting[key], case[2][key]) for key in accounting if accounting[key] != case[2][key]])
        print("STORAGE_EFFECT_NATIVE=" + json.dumps({"modules": len(arrays), "native_calls": calls[0],
                                                    "exact_all_values": True}), flush=True)
    elif names == ("storage_effect_serial",):
        from src.compiler.native_compiler_accelerators import (
            READ_ONLY_MODULE, compiler_accelerators, load_storage_effects_accelerator,
        )
        from src.compiler import fortran_c_shell
        from tools.compiler_probes.probe_dt_unit_interfaces import main as dt_probe
        import time
        installed = load_storage_effects_accelerator(arrays['installation'])
        assert installed is not None and installed.metadata['loaded_without_compilation']
        run = installed.run
        calls, values = [0], [0]
        def observed(functions, **kwargs):
            calls[0] += 1
            result = run(functions, **kwargs)
            values[0] += len(result[0] if kwargs.get('include_storage_types') else result)
            return result
        compiler_accelerators.register_native(READ_ONLY_MODULE, observed,
            name="turing:whole-module-storage-effects", library_path=artifact.library_path)
        canonical = fortran_c_shell.lower_ast_source_to_ssa
        def quiet(*args, **kwargs):
            assert args[1] != 'read_only_storage_effects', "compiler component must be reused, not rebuilt"
            kwargs['progress'] = lambda _message: None
            return canonical(*args, **kwargs)
        fortran_c_shell.lower_ast_source_to_ssa = quiet
        started = time.perf_counter()
        dt_probe(Path(sys.argv[1]).parent / "dt-serial", proposal=True, serial=True)
        print("STORAGE_EFFECT_REAL=" + json.dumps({"whole_module_calls": calls[0],
                                                 "value_facts": values[0],
                                                 "loaded_without_compilation": True,
                                                 "seconds": time.perf_counter() - started}), flush=True)
    elif names == ("dead_metadata_graphs",):
        from types import SimpleNamespace
        from src.compiler.native_compiler_accelerators import (
            DEAD_METADATA_ORDER, compiler_accelerators, _native_dead_metadata_runner,
        )
        from src.compiler.glsl_deployment_strategy import _remove_dead_metadata_nodes
        from src.compiler.ssa_llvm_backend import LLVMExecution
        calls = [0]
        original_run = LLVMExecution.run
        def counted(self):
            calls[0] += 1
            return original_run(self)
        LLVMExecution.run = counted
        compiler_accelerators.register_native(
            DEAD_METADATA_ORDER, _native_dead_metadata_runner(artifact, parameters),
            name="probe:whole-dead-metadata-worklist", library_path=artifact.library_path,
        )
        for source_graph, protected, expected in arrays:
            graph = SimpleNamespace(G=source_graph.copy())
            removed = []
            before = calls[0]
            def remove(node):
                removed.append(node)
                graph.G.remove_node(node)
            _remove_dead_metadata_nodes(graph, protected, remove)
            actual = (removed, tuple(graph.G.nodes), sorted(graph.G.edges))
            assert actual == expected, (actual, expected)
            assert calls[0] - before == int(bool(source_graph)), "one native call per whole graph"
        print("DEAD_METADATA_NATIVE=" + json.dumps({"graphs": len(arrays), "native_calls": calls[0],
                                                   "exact_removal_order": True}), flush=True)
    elif names == ("shape_pairs",):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        from src.compiler.native_compiler_accelerators import _native_broadcast_shapes_runner
        from src.compiler.ssa_llvm_backend import LLVMExecution
        run = _native_broadcast_shapes_runner(artifact, parameters)
        assert run(()) == ()
        # Neither non-integer extents nor wide integers may be narrowed by
        # native ABI packing. They remain with the same authored eager law.
        assert run((((2**100,), (1,)),)) == ((2**100,),)
        assert run((((2**63,), (1,)),)) == ((2**63,),)
        assert run((((2**64 - 1,), (1,)),)) == ((2**64 - 1,),)
        assert run((((1.5,), (2,)),)) == (None,)
        rendezvous = Barrier(2)
        original_run = LLVMExecution.run
        def together(self):
            rendezvous.wait(timeout=5)
            return original_run(self)
        LLVMExecution.run = together
        with ThreadPoolExecutor(max_workers=2) as pool:
            shapes = list(pool.map(run, (arrays, arrays)))
        assert shapes[0] == shapes[1]
        print("GRAPH_KERNEL_SHAPES=" + json.dumps(shapes[0]), flush=True)
    elif names == ("dependency_graphs",):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        from src.compiler.native_compiler_accelerators import _native_dependency_levels_runner
        from src.compiler.ssa_llvm_backend import LLVMExecution
        run = _native_dependency_levels_runner(artifact, parameters)
        rendezvous = Barrier(2)
        original_run = LLVMExecution.run
        def together(self):
            rendezvous.wait(timeout=5)
            return original_run(self)
        LLVMExecution.run = together
        with ThreadPoolExecutor(max_workers=2) as pool:
            levels = list(pool.map(lambda work: run(work[0], order=work[1], preference=work[2]), arrays))
        print("GRAPH_KERNEL_LEVELS=" + json.dumps(levels), flush=True)
    elif names == ("program_outputs",):
        execution = prepare_artifact_execution(artifact, arrays["inputs"]).run()
        print("GRAPH_PROGRAM_OUTPUTS=" + json.dumps([
            execution.buffers[identity].tolist() for identity in arrays["outputs"]
        ]), flush=True)
    elif names == ("topology_graphs",):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        from src.compiler.native_compiler_accelerators import _native_topological_runner
        from src.compiler.ssa_llvm_backend import LLVMExecution
        run = _native_topological_runner(artifact, parameters)
        rendezvous = Barrier(2)
        original_run = LLVMExecution.run
        def together(self):
            rendezvous.wait(timeout=5)
            return original_run(self)
        LLVMExecution.run = together
        with ThreadPoolExecutor(max_workers=2) as pool:
            orders = list(pool.map(lambda graph: run(graph, key=lambda node: node % 2), arrays))
        print("GRAPH_KERNEL_ORDERS=" + json.dumps(orders), flush=True)
    else:
        execution = prepare_artifact_execution(artifact, {
            int(parameters[name]): value for name, value in zip(names, arrays)
            if name in parameters
        })
        execution.run()
        print("GRAPH_KERNEL_ARRAYS=" + json.dumps([value.tolist() for value in arrays]), flush=True)
except BaseException as error:
    print(f"native graph worker refused: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    os._exit(1)

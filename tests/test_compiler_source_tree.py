from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.compiler import compiler_source_tree as trees
from src.compiler import compiler_bootstrap_runtime as runtime
from src.compiler import project_compilation_product as products


@pytest.fixture(autouse=True)
def isolated_cuts():
    trees.clear_source_cuts()
    runtime._ACTIVE_DEPLOYMENTS.clear()
    yield
    trees.clear_source_cuts()
    trees._INSTALLED.clear()
    runtime._ACTIVE_DEPLOYMENTS.clear()


SOURCE = '''def outer(x: float):
    def inner(y: float):
        return y * 2.0 + 1.0
    return inner(x) * 3.0
'''


def test_native_tree_installs_inner_then_replaces_it_with_outer(tmp_path, monkeypatch):
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
    source = tmp_path / "native_tree_example.py"
    source.write_text(SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    module = importlib.import_module("native_tree_example")
    original = module.outer
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "values": [{"function": name, "parameter": parameter, "storage": "scalar",
                    "dtype": "float64", "rank": 0, "shape": [], "python_type": "builtins.float"}
                   for name, parameter in (("outer", "x"), ("outer.<locals>.inner", "y"), ("root", "x"))],
        "records": {}, "bindings": [],
    })
    planned = products.compile_partitioned_source(
        SOURCE, tmp_path / "growth", plan_only=True, interface_tiles=True,
        extraction_contract=contract, source_options={"source_module": module.__name__},
    )
    registry = tmp_path / "registry.json"
    monkeypatch.setenv(runtime.COMPILER_BOOTSTRAP_REGISTRY_ENV, str(registry))
    manifest = products.compile_resolved_process_graph_plan(
        planned["resolved_process_graph"], planned["process_graph_unit_plan"],
        tmp_path / "growth/product", max_stage_units=1, jobs=1,
    )
    inner = manifest["units"][0]
    assert inner["qualified_names"] == ["outer.<locals>.inner"]
    child_root = tmp_path / "inner-product"
    result = products.publish_native_interface_product(
        source, module.__name__, tmp_path / "growth/product/units" / f'unit_{inner["unit_index"]:03d}', child_root,
    )
    assert result["automatic_native_verification"][0]["status"] == "verified", result
    runtime.publish_compiler_bootstrap_products([child_root], registry_path=registry)
    state = trees.initialize_registered_source_tree(registry)
    assert list(state["trees"][0]["selected_cuts"]) == ["outer.<locals>.inner"]
    deployed = runtime._ACTIVE_DEPLOYMENTS[(str(child_root), "outer.<locals>.inner")]
    before = deployed.__turing_native_route_counts__["native"]
    for value in (-3.0, 0.0, 21.0):
        assert module.outer(value) == original(value)
    assert deployed.__turing_native_route_counts__["native"] == before + 3
    assert module.outer.__globals__ is vars(module)
    assert module.outer.__code__.co_firstlineno == original.__code__.co_firstlineno
    rewritten, _bindings = trees.splice_source(SOURCE, module.__name__, entrypoint="outer")
    assert "y * 2.0" not in rewritten
    assert len(rewritten.splitlines()) == len(SOURCE.splitlines())
    changed = SOURCE.replace("y * 2.0 + 1.0", "y * 9.0 + 1.0")
    assert trees.splice_source(changed, module.__name__, entrypoint="outer")[0] == changed
    saved_outer = module.outer
    with monkeypatch.context() as scoped:
        scoped.setattr(trees, "initialize_nested_cuts", lambda _module: (_ for _ in ()).throw(ValueError("splice failed")))
        with pytest.raises(ValueError, match="splice failed"):
            trees.initialize_registered_source_tree(registry)
    assert module.outer is saved_outer
    assert module.outer(21.0) == 129.0
    # A restart imports Python, then its initializer reinstates the accepted cut.
    child = subprocess.run([sys.executable, "-c", '''
import sys
from src.compiler.compiler_source_tree import initialize_registered_source_tree
import native_tree_example as m
s = initialize_registered_source_tree(sys.argv[1])
assert m.outer(21.0) == 129.0
assert list(s['trees'][0]['selected_cuts']) == ['outer.<locals>.inner']
print('fresh-process native cut: passed')
''', str(registry)], capture_output=True, text=True, timeout=20)
    assert child.returncode == 0, child.stdout + child.stderr
    manifest = products.compile_resolved_process_graph_plan(
        planned["resolved_process_graph"], planned["process_graph_unit_plan"],
        tmp_path / "growth/product", max_stage_units=1, jobs=1,
    )
    outer = next(unit for unit in manifest["units"] if unit["qualified_names"] == ["outer"])
    outer_root = tmp_path / "outer-product"
    result = products.publish_native_interface_product(
        source, module.__name__, tmp_path / "growth/product/units" / f'unit_{outer["unit_index"]:03d}', outer_root,
    )
    assert result["automatic_native_verification"][0]["status"] == "verified", result
    runtime.publish_compiler_bootstrap_products([child_root, outer_root], registry_path=registry)
    state = trees.initialize_registered_source_tree(registry)
    assert list(state["trees"][0]["selected_cuts"]) == ["outer"]
    assert (str(child_root), "outer.<locals>.inner") not in runtime._ACTIVE_DEPLOYMENTS
    assert module.outer(21.0) == 129.0
    # Subsequent source lowering links the binary boundary, not its authored body.
    lowered, outputs, exports = lower_ast_source_to_ssa(
        "def root(x: float):\n    return compiled(x) + 1.0\n", "root",
        python_bindings={"compiled": module.outer}, progress_bars=False,
        extraction_contract=contract,
    )
    artifact = emit_ssa_function_to_llvm(lowered, exports[0])
    assert artifact.complete and artifact.linked_objects
    compiled = compile_artifact(artifact, directory=tmp_path / "caller")
    function = lowered.functions[exports[0]]
    execution = prepare_artifact_execution(compiled, {int(function.args[0].id): 21.0}).run()
    assert float(execution.buffers[int(outputs[exports[0]][0].id)].reshape(-1)[0]) == 130.0


def test_nested_initializer_refuses_captured_cells_without_declared_abi(tmp_path, monkeypatch):
    source = tmp_path / "captured_tree_example.py"
    source.write_text("def outer(x):\n    def inner(y):\n        return x + y\n    return inner(1)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("captured_tree_example")
    with pytest.raises(ValueError, match="closure-cell ABI"):
        trees.nested_authored_callable(module, "outer.<locals>.inner")


def test_unverified_source_cut_is_not_registered():
    with pytest.raises(ValueError, match="verified callable"):
        trees.register_cut("module", "leaf", lambda value: value, "unproved")

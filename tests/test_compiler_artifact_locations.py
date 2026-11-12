"""Path-only tests: never import compiler bootstrap or invoke a compiler."""
import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "compiler"
spec = importlib.util.spec_from_file_location("artifact_locations_under_test", SOURCE / "artifact_locations.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_default_preserves_repository_root(monkeypatch):
    monkeypatch.delenv("TURING_COMPILER_ARTIFACT_ROOT", raising=False)
    assert module.compiler_artifact_root() == ROOT


def test_site_override_is_external_absolute_path(tmp_path, monkeypatch):
    monkeypatch.setenv("TURING_COMPILER_ARTIFACT_ROOT", str(tmp_path))
    assert module.compiler_artifact_root() == tmp_path.resolve()
    monkeypatch.setenv("TURING_COMPILER_ARTIFACT_ROOT", "relative")
    with pytest.raises(ValueError, match="must be absolute"):
        module.compiler_artifact_root()


@pytest.mark.parametrize("name,suffix", [
    ("_metadata_copy_install_directory", "metadata_copy"),
    ("_storage_effect_install_directory", "storage_effects"),
])
def test_foundation_defaults_use_site_root_and_preserve_explicit_directory(tmp_path, monkeypatch, name, suffix):
    # Execute only these pure selectors, not module-level compiler imports.
    tree = ast.parse((SOURCE / "native_compiler_accelerators.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"Path": Path, "compiler_artifact_root": module.compiler_artifact_root}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "selector", "exec"), namespace)
    monkeypatch.setenv("TURING_COMPILER_ARTIFACT_ROOT", str(tmp_path))
    assert namespace[name]() == tmp_path / "build" / "compiler_foundations" / suffix
    explicit = tmp_path / "explicit"
    assert namespace[name](explicit) == explicit


@pytest.mark.parametrize("filename,function", [
    ("tensor_kernel_lanes.py", "eigen_object"),
    ("python_native_shell.py", "compile_ast_fortran_io"),
    ("graph_work.py", "ensure_graph_foundation"),
])
def test_native_output_selectors_use_relocatable_root(filename, function):
    tree = ast.parse((SOURCE / filename).read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    # The generated native shell function is identified by the existing directory assignment.
    candidates = [node for node in definitions if any(
        isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
        and item.func.id == "compiler_artifact_root" for item in ast.walk(node))]
    assert len(candidates) == 1
    if filename == "tensor_kernel_lanes.py":
        assert candidates[0].name == function
    assignment = next(node for node in ast.walk(candidates[0]) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "directory" for target in node.targets)
        and any(isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                and item.func.id == "compiler_artifact_root" for item in ast.walk(node.value)))
    namespace = {"compiler_artifact_root": lambda: Path("/site/job"), "entrypoint": "entry", "digest": "abc"}
    result = eval(compile(ast.Expression(assignment.value), "selector", "eval"), namespace)
    assert result.is_relative_to(Path("/site/job"))

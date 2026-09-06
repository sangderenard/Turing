"""Exercise the plain on-disk DualIRShell archive against real compiles.

Deliberately manual, not wired into compile_ast_aot itself (see
GRAPH_DESCRIPTION_LAYER_SURVEY.md) -- these tests turn it on for real
compiled programs so we can see whether it actually round-trips and how
large a real shell is on disk, without changing production compile
behavior.
"""

from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from src.common.tensors.accelerator_backends.shell_archive import (
    load_shell,
    save_shell,
    shell_archive_root,
)


def _compile(source: str, entrypoint: str, feeds: dict):
    return compile_ast_aot(source, entrypoint, feeds, precompile_only=True)


def test_a_real_compiled_shell_round_trips_through_the_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("TURING_ACCELERATOR_CACHE_DIR", str(tmp_path))
    source = "def add_one(x):\n    return x + 1\n"
    shell = _compile(source, "add_one", {"x": 41}).shell

    path = save_shell(shell, key="add_one")

    assert path.exists()
    assert path.parent == shell_archive_root()
    assert path.stat().st_size > 0

    reloaded = load_shell(path)
    assert reloaded.name == shell.name
    assert reloaded.compiled_shell_program is not None
    assert type(reloaded.class_navigation) is type(shell.class_navigation)
    assert type(reloaded.dependency_regions) is type(shell.dependency_regions)


def test_each_save_is_a_new_file_not_an_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("TURING_ACCELERATOR_CACHE_DIR", str(tmp_path))
    source = "def add_one(x):\n    return x + 1\n"
    shell = _compile(source, "add_one", {"x": 41}).shell

    first = save_shell(shell, key="add_one")
    second = save_shell(shell, key="add_one")

    assert first != second
    assert first.exists() and second.exists()


def test_shell_size_on_disk_for_a_real_multi_function_program(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TURING_ACCELERATOR_CACHE_DIR", str(tmp_path))
    source = """
def helper(x, y):
    total = x + y
    return total * 2

def add_one(x):
    return helper(x, 1)
"""
    shell = _compile(source, "add_one", {"x": 41}).shell
    path = save_shell(shell, key="add_one_multi_function")
    size = path.stat().st_size
    with capsys.disabled():
        print(f"\n  shell archive size (multi-function program): {size} bytes")
    assert size > 0

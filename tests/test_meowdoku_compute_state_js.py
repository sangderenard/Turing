"""Bounded Node checks of the real compute runtime's opt-in state handoff."""
from pathlib import Path
import ast
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_full_state_start_and_displayed_handoff():
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run(
        ["node", str(root / "tests" / "meowdoku_compute_state_check.mjs")],
        cwd=root, capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_published_compute_asset_import_closure(tmp_path):
    """Execute the exporter's actual static copy loops, not a hand-picked list."""
    root = Path(__file__).resolve().parents[1]
    here = root / "examples" / "meowdoku"
    parsed = ast.parse((here / "web_export.py").read_text(encoding="utf-8"))
    main = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    loops = [node for node in main.body if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple)
             and any(isinstance(item, ast.Constant) and item.value == marker
                     for item in node.iter.elts for marker in ("compute.js", "marble_object.js"))]
    assert len(loops) == 2
    program = ast.Module(body=loops, type_ignores=[])
    exec(compile(ast.fix_missing_locations(program), str(here / "web_export.py"), "exec"),
         {"HERE": here, "out": tmp_path, "shutil": shutil})
    # The existing server.install_shell step publishes this separate loader.
    shutil.copyfile(root / "examples/mobile_reference_app/web/turing-wasm-shell.js", tmp_path / "turing-wasm-shell.js")
    (tmp_path / "package.json").write_text('{"type":"module"}', encoding="utf-8")
    assert (tmp_path / "compute_state.js").read_bytes() == (here / "web/compute_state.js").read_bytes()
    done = subprocess.run(
        ["node", "--input-type=module", "-e", "await import(process.argv[1]);", (tmp_path / "compute.js").as_uri()],
        capture_output=True, text=True, timeout=20,
    )
    assert done.returncode == 0, done.stdout + done.stderr

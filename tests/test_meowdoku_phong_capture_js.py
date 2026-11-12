"""Completed-frame capture controller; real browser encoding is checked separately."""
from pathlib import Path
import shutil
import subprocess
import pytest


def test_phong_capture_controller():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the browser capture controller')
    script = Path(__file__).with_name('meowdoku_phong_capture_check.mjs')
    subprocess.run([node, str(script)], check=True, timeout=30, cwd=script.parent.parent)


def test_phong_capture_runs_after_final_composite_and_is_exported():
    root = Path(__file__).resolve().parents[1]
    studio = (root / 'examples/meowdoku/web/marble_studio.js').read_text(encoding='utf8')
    assert studio.index('phongCapture.frame()') > studio.index('studioDof.finish(dofCamera')
    assert 'phongCapture.fail(error)' in studio
    assert '"marble_phong_capture.js"' in (root / 'examples/meowdoku/web_export.py').read_text(encoding='utf8')
    html = (root / 'examples/meowdoku/web/marble_studio.html').read_text(encoding='utf8')
    for element in ['photo', 'start', 'stop', 'fps', 'status', 'png', 'video']:
        assert f'id="phong-capture-{element}"' in html

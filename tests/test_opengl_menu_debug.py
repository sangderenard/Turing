import pytest
from src.opengl_render.render_sim_coordinator import run_option, OPTIONS

@pytest.mark.parametrize('choice', list(OPTIONS.keys()))
def test_menu_runs_in_debug(choice):
    proc = run_option(choice, debug=True)
    assert proc.returncode == 0
    out = (proc.stdout or '').lower()
    assert 'points' in out or 'positions' in out
    assert 'dtype' in out  # metadata vector info

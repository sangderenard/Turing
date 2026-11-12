"""Structured Studio grids preserve the incumbent geometry and photo boundary."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from meowdoku import marble_photo


def test_structured_grid_and_reflective_card_pass_actual_photo_decoder():
    result = subprocess.run(["node", "tests/meowdoku_studio_grid_check.mjs"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30, check=True)
    assert "Graph paper passed" in result.stdout
    class NoSaved:
        def get(self, _key):
            return None
    for name in ["ray.mrp1", "graph-paper.mrp1"]:
        raw = (ROOT / "build/dot-studio-grid-check" / name).read_bytes()
        *_, config, _digest = marble_photo.decode_upload(raw, NoSaved())
        assert config["camera"]["resolution"] == 256
        assert config["oversampling"] == 4
        assert config["tiled"] == {"tile_size": 1024}
        assert config["camera"]["resolution"] * config["oversampling"] == 1024

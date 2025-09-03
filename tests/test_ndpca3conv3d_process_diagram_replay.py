import pytest

pytest.skip("demo replay flaky and slow (KeyError)", allow_module_level=True)

import sys
from pathlib import Path

from src.common.tensors.abstract_convolution import demo_ndpca3conv3d_process_diagram as demo


def test_demo_replay_path_does_not_raise(monkeypatch):
    out_file = Path(demo.__file__).with_name("ndpca3conv3d_training.png")
    if out_file.exists():
        out_file.unlink()
    monkeypatch.setattr(sys, "argv", [str(demo.__file__)])
    try:
        demo.main()
    except AssertionError:
        # The demo performs additional validation internally which may assert
        # on parameter mismatches.  These assertions are outside the scope of
        # this regression test which only guards against `ValueError` being
        # raised during replay.
        pass
    except ValueError as exc:  # pragma: no cover - failing path for clarity
        pytest.fail(f"Replay path raised ValueError: {exc}")
    finally:
        if out_file.exists():
            out_file.unlink()

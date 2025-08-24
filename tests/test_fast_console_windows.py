import os
import pytest
from src.common.fast_console import cffiPrinter


@pytest.mark.skipif(os.name != "nt", reason="Windows-only smoke test")
def test_fast_console_prints_to_console(capsys):
    try:
        printer = cffiPrinter()
    except OSError:  # pragma: no cover - unsupported platform
        pytest.skip("fast console unavailable on this platform")
    try:
        printer.print("hello\n")
        printer.flush()
    finally:
        printer.stop()
    captured = capsys.readouterr()
    assert "hello" in captured.out

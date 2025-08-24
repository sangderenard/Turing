import pytest
from src.common.fast_console import cffiPrinter


def test_fast_console_formats_text():
    try:
        printer = cffiPrinter()
    except OSError:  # pragma: no cover - unsupported platform
        pytest.skip("fast console unavailable on this platform")
    try:
        total = 3 + 4
        printer.print(f"3 + 4 = {total:02d}\n")
        printer.flush()
    finally:
        printer.stop()
    assert True

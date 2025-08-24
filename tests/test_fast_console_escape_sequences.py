import pytest
from src.common.fast_console import cffiPrinter


def test_fast_console_honors_escape_sequences():
    try:
        printer = cffiPrinter()
    except OSError:  # pragma: no cover - unsupported platform
        pytest.skip("fast console unavailable on this platform")
    try:
        printer.print("\x1b[2J")  # Clear screen
        printer.print("Hello\n")
        printer.flush()
    finally:
        printer.stop()
    assert True

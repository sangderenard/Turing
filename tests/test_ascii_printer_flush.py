import sys
import os
import pytest
from src.rendering.ascii_diff import ThreadedAsciiDiffPrinter


@pytest.mark.skipif(os.name != "nt", reason="fast console requires Windows")
def test_queue_join_prints_before_prompt(capfd):
    printer = ThreadedAsciiDiffPrinter()
    try:
        printer.enqueue("FRAME\n")
        printer.wait_until_empty()
        print("PROMPT")
    finally:
        printer.stop()
    out = capfd.readouterr().out
    assert out == "FRAME\nPROMPT\n"

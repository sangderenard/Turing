import sys
import os
import pytest
from src.rendering.ascii_diff import ThreadedAsciiDiffPrinter


@pytest.mark.skipif(os.name != "nt", reason="fast console requires Windows")
def test_queue_join_prints_before_prompt(capfd):
    printer = ThreadedAsciiDiffPrinter()
    try:
        q = printer.get_queue()
        q.put("FRAME\n")
        q.join()
        print("PROMPT")
    finally:
        printer.stop()
    out = capfd.readouterr().out
    assert out == "FRAME\nPROMPT\n"

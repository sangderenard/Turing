import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import transmogrifier


def test_root_exports():
    expected = {
        "Simulator",
        "BitTensorMemoryGraph",
        "NodeEntry",
        "EdgeEntry",
        "GraphSearch",
    }
    assert expected.issubset(set(transmogrifier.__all__))

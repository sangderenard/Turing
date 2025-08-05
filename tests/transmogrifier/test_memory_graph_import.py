import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from transmogrifier.graph.memory_graph import NodeEntry, EdgeEntry
import ctypes


def test_struct_sizes():
    assert ctypes.sizeof(NodeEntry) > 0
    assert ctypes.sizeof(EdgeEntry) > 0

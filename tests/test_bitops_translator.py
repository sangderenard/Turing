import pytest
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.compiler.bitops import BitOps

# basic sanity checks for translated operations
@pytest.mark.parametrize("a,b,width", [(0b1010, 0b0110, 4)])
def test_basic_bitops(a, b, width):
    ops = BitOps(bit_width=width, encoding="binary")
    assert ops.bit_and(a, b) == (a & b)
    assert ops.bit_or(a, b) == (a | b)
    assert ops.bit_xor(a, b) == (a ^ b)
    assert ops.bit_not(a) == ((~a) & ((1 << width) - 1))
    assert ops.bit_shift_left(a, 1) == ((a << 1) & ((1 << width) - 1))
    assert ops.bit_shift_right(a, 1) == (a >> 1)
    assert ops.bit_add(a, b) == ((a + b) & ((1 << width) - 1))
    assert ops.bit_sub(a, b) == ((a - b) & ((1 << width) - 1))
    assert ops.bit_mul(a, b) == ((a * b) & ((1 << width) - 1))
    assert ops.bit_div(a, b) == ((a // b) & ((1 << width) - 1))
    assert ops.bit_mod(a, b) == ((a % b) & ((1 << width) - 1))
    # provenance recorded
    assert ops.translator.graph.nodes
    nxg = ops.translator.graph.nx
    assert nxg is not None
    assert len(nxg.nodes) == len(ops.translator.graph.nodes)
    assert len(nxg.edges) == len(ops.translator.graph.edges)

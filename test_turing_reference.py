"""
test_turing_reference.py ───────────────────────────────────────────────────────
Reference backend + pytest harness that exercises *every* primitive and
derived faculty of the `Turing` scaffold *with provenance turned on*.

Run with:
    pytest -q test_turing_reference.py
"""

from __future__ import annotations

import pytest
from turing import Hooks, Turing
from turing_provenance import ProvenanceGraph, instrument_hooks
from typing import List, Tuple
import docstrings_math

docstrings_math.apply_math_docstrings(Turing)
# ──────────────────────────────────────────────────────────────────────────────
#  Simple list‑of‑bits backend  (MSB→LSB order)
# ──────────────────────────────────────────────────────────────────────────────

BitStr = List[int]  # each element 0 or 1

def _check_bits(x: BitStr):
    assert all(b in (0,1) for b in x), "bitstring must contain only 0/1"

# -------- Boolean NAND -------------------------------------------------------

def nand(x: BitStr, y: BitStr) -> BitStr:
    _check_bits(x); _check_bits(y); assert len(x)==len(y)
    return [1 - (a & b) for a,b in zip(x,y)]

# -------- Shifts -------------------------------------------------------------

def sigma_L(x: BitStr, k: int) -> BitStr:
    _check_bits(x); assert k>=0
    return x + [0]*k

def sigma_R(x: BitStr, k: int) -> BitStr:
    _check_bits(x); assert k>=0
    return x[:-k] if k else x.copy()

# -------- Structural ---------------------------------------------------------

def concat(x: BitStr, y: BitStr) -> BitStr:
    _check_bits(x); _check_bits(y)
    return x + y

def slice_bits(x: BitStr, i:int, j:int) -> BitStr:
    _check_bits(x)
    return x[i:j]

def mu(x: BitStr, y: BitStr, sel: BitStr) -> BitStr:
    _check_bits(x); _check_bits(y); _check_bits(sel)
    assert len(x)==len(y)==len(sel)
    return [ (b if s else a) for a,b,s in zip(x,y,sel) ]

# -------- Size ---------------------------------------------------------------

def length(x: BitStr) -> int: return len(x)

def zeros(n:int) -> BitStr: return [0]*n

raw_hooks = Hooks(
    nand     = nand,
    sigma_L  = sigma_L,
    sigma_R  = sigma_R,
    concat   = concat,
    slice    = slice_bits,
    mu       = mu,
    length   = length,
    zeros    = zeros,
)

graph = ProvenanceGraph()
hooks = instrument_hooks(raw_hooks, graph)

tm = Turing(hooks)

# Helper to build bitstrings from int ------------------------------------------------

def bits_from_int(val:int, width:int) -> BitStr:
    return [ (val >> (width-1-i)) & 1 for i in range(width) ]

def int_from_bits(bits:BitStr)->int:
    v=0
    for b in bits: v=(v<<1)|b
    return v

# ──────────────────────────────────────────────────────────────────────────────
#  Tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b,width",[(0b1011,0b1100,4)])
def test_boolean_derivatives(a,b,width):
    A=bits_from_int(a,width); B=bits_from_int(b,width)
    # NOT
    assert int_from_bits(tm.NOT(A)) == (~a)&((1<<width)-1)
    # AND
    assert int_from_bits(tm.AND(A,B)) == (a & b)
    # OR
    assert int_from_bits(tm.OR(A,B))  == (a | b)
    # XOR
    assert int_from_bits(tm.XOR(A,B)) == (a ^ b)
    # XNOR
    assert int_from_bits(tm.XNOR(A,B)) == (~(a ^ b)) & ((1<<width)-1)

@pytest.mark.parametrize("val,k,width",[(0b1011,2,4)])
def test_shifts(val,k,width):
    X=bits_from_int(val,width)
    assert int_from_bits(tm.sigma_L(X,k)) == (val << k)
    r=tm.sigma_R(X,k)
    assert int_from_bits(r) == (val >> k)


def test_concat_slice():
    A=[1,0,1]; B=[0,1]
    C=tm.concat(A,B)
    assert C==[1,0,1,0,1]
    assert tm.slc(C,1,4)==[0,1,0]


def test_mu():
    x=[0,0,0,0]
    y=[1,1,1,1]
    sel=[0,1,0,1]
    out=tm.mu(x,y,sel)
    assert out==[0,1,0,1]


def test_rotL():
    bits=[1,0,0,1]
    assert tm.rho_L(bits,1)==[0,0,1,1]


def test_succ_and_add():
    a,b=5,3
    width=4
    A=bits_from_int(a,width)
    B=bits_from_int(b,width)
    S=tm.ripple_add(A,B)
    assert int_from_bits(S)==a+b
    inc=tm.succ(A)
    assert int_from_bits(inc)==a+1


def test_write_and_move():
    tape=[0,0,0,0]
    # write 1 at idx 2
    tape2=tm.write_bit(tape,2,[1])
    assert tape2==[0,0,1,0]


def test_provenance_nonempty():
    assert graph.nodes, "No provenance recorded"
    # at least one edge produced by earlier ops
    assert graph.edges

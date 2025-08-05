from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional, Union

from .turing import Turing, Hooks
from .turing_provenance import ProvenanceGraph, instrument_hooks

# Simple list-of-bits backend reused from bitops_translator
BitStr = List[int]

def _check_bits(x: BitStr):
    assert all(b in (0, 1) for b in x)

def nand(x: BitStr, y: BitStr) -> BitStr:
    _check_bits(x); _check_bits(y); assert len(x) == len(y)
    return [1 - (a & b) for a, b in zip(x, y)]

def sigma_L(x: BitStr, k: int) -> BitStr:
    _check_bits(x); assert k >= 0
    return x + [0] * k

def sigma_R(x: BitStr, k: int) -> BitStr:
    _check_bits(x); assert k >= 0
    return x[:-k] if k else x.copy()

def concat(x: BitStr, y: BitStr) -> BitStr:
    _check_bits(x); _check_bits(y)
    return x + y

def slice_bits(x: BitStr, i: int, j: int) -> BitStr:
    _check_bits(x)
    return x[i:j]

def mu(x: BitStr, y: BitStr, sel: BitStr) -> BitStr:
    _check_bits(x); _check_bits(y); _check_bits(sel)
    assert len(x) == len(y) == len(sel)
    return [(b if s else a) for a, b, s in zip(x, y, sel)]

def length(x: BitStr) -> int:
    _check_bits(x)
    return len(x)

def zeros(n: int) -> BitStr:
    return [0] * n

@dataclass
class SSAView:
    node_id: int
    _bits: Optional[BitStr] = None
    _value_cache: Optional[int] = None

    @property
    def value(self) -> int:
        if self._value_cache is None and self._bits is not None:
            v = 0
            for b in self._bits:
                v = (v << 1) | b
            self._value_cache = v
        return int(self._value_cache or 0)

class ProvenanceTM:
    """Tiny Turing Machine exposing tape/head as SSA views."""

    def __init__(self, width: int):
        self.width = width
        raw = Hooks(
            nand=nand,
            sigma_L=sigma_L,
            sigma_R=sigma_R,
            concat=concat,
            slice=slice_bits,
            mu=mu,
            length=length,
            zeros=zeros,
        )
        self.graph = ProvenanceGraph()
        hooks = instrument_hooks(raw, self.graph)
        self.tm = Turing(hooks)
        self.tape_layers: List[List[BitStr]] = []
        self.head_layers: List[BitStr] = []
        self.state_layers: List[BitStr] = []
        self.state_map: Dict[str, int] = {}

    # ---------------- helpers ----------------
    def _node(self, obj: BitStr) -> int:
        return self.graph._producer.get(id(obj), -1)

    def _one(self) -> BitStr:
        return self.tm.NOT(self.tm.zeros(1))

    # ---------------- init --------------------
    def init_tape(self, data: Union[str, int, Iterable[int], bytes, bytearray, BitStr]) -> None:
        """Initialize tape from various pythonic representations.

        ``data`` may be provided as:

        * ``str`` of ``"0"``/``"1"`` characters
        * ``int`` interpreted as a binary word (MSB first)
        * ``bytes``/``bytearray`` expanded MSB→LSB
        * any iterable of ``0``/``1`` integers

        The tape is padded with zeros or truncated to ``self.width``.
        """
        bits: List[int]
        if isinstance(data, str):
            bits = [1 if ch == '1' else 0 for ch in data]
        elif isinstance(data, int):
            bstr = format(data, f"0{self.width}b")
            bits = [int(b) for b in bstr][-self.width:]
        elif isinstance(data, (bytes, bytearray)):
            bits = []
            for byte in data:
                for i in range(7, -1, -1):
                    bits.append((byte >> i) & 1)
        else:  # iterable of bits
            bits = [1 if int(b) else 0 for b in data]

        layer: List[BitStr] = []
        for b in bits[:self.width]:
            cell = self._one() if b else self.tm.zeros(1)
            layer.append(cell)
        while len(layer) < self.width:
            layer.append(self.tm.zeros(1))
        self.tape_layers = [layer]

    def init_head(self, pos: int) -> None:
        head = self.tm.zeros(self.width)
        if 0 <= pos < self.width:
            head = self.tm.write_bit(head, pos, self._one())
        self.head_layers = [head]

    def init_state(self, name: str) -> None:
        if name not in self.state_map:
            self.state_map[name] = len(self.state_map)
        idx = self.state_map[name]
        size = len(self.state_map)
        vec = self.tm.zeros(size)
        vec = self.tm.write_bit(vec, idx, self._one())
        self.state_layers = [vec]

    # ---------------- stepping ----------------
    def step(self) -> None:
        if self.tape_layers:
            self.tape_layers.append(self.tape_layers[-1][:])
        if self.head_layers:
            self.head_layers.append(self.head_layers[-1])
        if self.state_layers:
            self.state_layers.append(self.state_layers[-1])

    # ---------------- views -------------------
    def tape_view(self, t: int) -> List[SSAView]:
        layer = self.tape_layers[t]
        return [SSAView(self._node(cell), cell) for cell in layer]

    def head_view(self, t: int) -> SSAView:
        mask = self.head_layers[t]
        return SSAView(self._node(mask), mask)

    def state_view(self, t: int) -> SSAView:
        vec = self.state_layers[t]
        return SSAView(self._node(vec), vec)

    # ---------------- mutation via SSA --------
    def patch_cell(self, t: int, idx: int, new_val: int) -> None:
        while len(self.tape_layers) <= t + 1:
            self.tape_layers.append(self.tape_layers[-1][:])
        bit1 = self._one() if new_val else self.tm.zeros(1)
        old_cell = self.tape_layers[t][idx]
        new_cell = self.tm.write_bit(old_cell, 0, bit1)
        self.tape_layers[t + 1][idx] = new_cell


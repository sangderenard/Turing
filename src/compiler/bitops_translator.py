from __future__ import annotations

from ..turing_machine.turing import Hooks, Turing
from ..turing_machine.turing_provenance import ProvenanceGraph, instrument_hooks

BitStr = list  # simple list-of-bits backend

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

class BitOpsTranslator:
    """Translate integer operations into Turing primitives with provenance."""

    def __init__(self, bit_width: int):
        self.bit_width = bit_width
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

    # conversion helpers
    def bits_from_int(self, val: int) -> BitStr:
        return [(val >> (self.bit_width - 1 - i)) & 1 for i in range(self.bit_width)]

    def int_from_bits(self, bits: BitStr) -> int:
        v = 0
        for b in bits:
            v = (v << 1) | b
        return v

    def _is_zero(self, bits: BitStr) -> bool:
        acc = self.tm.slc(bits, 0, 1)
        for i in range(1, self.bit_width):
            acc = self.tm.OR(acc, self.tm.slc(bits, i, i + 1))
        return self.int_from_bits(acc) == 0

    def apply_bits(self, op: str, *operands: BitStr) -> BitStr:
        """Apply one derived integer operation to existing bit carriers.

        This is the authoritative derived-operation entry point for both
        concrete integer translation and ProcessGraph expansion. Primitive
        provenance is recorded by the instrumented Turing hooks.
        """
        if op == "bitand":
            return self.tm.AND(*operands)
        if op == "bitor":
            return self.tm.OR(*operands)
        if op == "bitxor":
            return self.tm.XOR(*operands)
        if op == "invert":
            return self.tm.NOT(operands[0])
        if op == "add":
            return self.tm.slc(
                self.tm.ripple_add(*operands), 1, self.bit_width + 1
            )
        if op == "sub":
            y_inv = self.tm.NOT(operands[1])
            y_neg = self.tm.slc(
                self.tm.succ(y_inv), 1, self.bit_width + 1
            )
            return self.tm.slc(
                self.tm.ripple_add(operands[0], y_neg),
                1,
                self.bit_width + 1,
            )
        if op == "mul":
            width = self.bit_width
            product = self.tm.zeros(width * 2)
            for index in range(width):
                selector_bit = self.tm.slc(
                    operands[1], width - 1 - index, width - index
                )
                shifted = self.tm.sigma_L(operands[0], index)
                padded = self.tm.concat(self.tm.zeros(width - index), shifted)
                selector = self._expand_bit(selector_bit, width * 2)
                addend = self.tm.mu(
                    self.tm.zeros(width * 2), padded, selector
                )
                product = self.tm.slc(
                    self.tm.ripple_add(product, addend),
                    1,
                    width * 2 + 1,
                )
            return self.tm.slc(product, width, width * 2)
        raise NotImplementedError(f"no Turing derived expansion for {op!r}")

    # pointwise logic
    def bit_and(self, x: int, y: int) -> int:
        X, Y = self.bits_from_int(x), self.bits_from_int(y)
        out = self.apply_bits("bitand", X, Y)
        return self.int_from_bits(out)

    def bit_or(self, x: int, y: int) -> int:
        X, Y = self.bits_from_int(x), self.bits_from_int(y)
        out = self.apply_bits("bitor", X, Y)
        return self.int_from_bits(out)

    def bit_xor(self, x: int, y: int) -> int:
        X, Y = self.bits_from_int(x), self.bits_from_int(y)
        out = self.apply_bits("bitxor", X, Y)
        return self.int_from_bits(out)

    def bit_not(self, x: int) -> int:
        X = self.bits_from_int(x)
        out = self.apply_bits("invert", X)
        return self.int_from_bits(out)

    # shifts
    def bit_shift_left(self, x: int, k: int) -> int:
        X = self.bits_from_int(x)
        shifted = self.tm.sigma_L(X, k)
        n = self.tm.length(shifted)
        truncated = self.tm.slc(shifted, n - self.bit_width, n)
        return self.int_from_bits(truncated)

    def bit_shift_right(self, x: int, k: int) -> int:
        X = self.bits_from_int(x)
        shifted = self.tm.sigma_R(X, k)
        pad = self.tm.zeros(self.bit_width - self.tm.length(shifted))
        padded = self.tm.concat(pad, shifted)
        return self.int_from_bits(padded)

    # arithmetic
    def bit_add(self, x: int, y: int) -> int:
        X, Y = self.bits_from_int(x), self.bits_from_int(y)
        return self.int_from_bits(self.apply_bits("add", X, Y))

    def bit_sub(self, x: int, y: int) -> int:
        X, Y = self.bits_from_int(x), self.bits_from_int(y)
        return self.int_from_bits(self.apply_bits("sub", X, Y))

    def _expand_bit(self, bit1: BitStr, n: int) -> BitStr:
        out = self.tm.zeros(n)
        for i in range(n):
            out = self.tm.write_bit(out, i, bit1)
        return out

    def bit_mul(self, x: int, y: int) -> int:
        X, Y = self.bits_from_int(x), self.bits_from_int(y)
        return self.int_from_bits(self.apply_bits("mul", X, Y))

    def bit_div(self, x: int, y: int) -> int:
        X, Y = self.bits_from_int(x), self.bits_from_int(y)
        if self._is_zero(Y):
            return self.int_from_bits(self.tm.zeros(self.bit_width))
        n = self.bit_width
        quotient = self.tm.zeros(n)
        remainder = self.tm.zeros(n)
        y_neg = self.tm.NOT(Y)
        y_neg = self.tm.succ(y_neg)
        y_neg = self.tm.slc(y_neg, 1, n + 1)
        for i in range(n):
            remainder = self.tm.sigma_L(remainder, 1)
            remainder = self.tm.slc(remainder, 1, n + 1)
            bit_i = self.tm.slc(X, i, i + 1)
            remainder = self.tm.write_bit(remainder, n - 1, bit_i)
            diff = self.tm.ripple_add(remainder, y_neg)
            carry = self.tm.slc(diff, 0, 1)
            rem_candidate = self.tm.slc(diff, 1, n + 1)
            sel = self._expand_bit(carry, n)
            remainder = self.tm.mu(remainder, rem_candidate, sel)
            quotient = self.tm.write_bit(quotient, i, carry)
        return self.int_from_bits(quotient)

    def bit_mod(self, x: int, y: int) -> int:
        X, Y = self.bits_from_int(x), self.bits_from_int(y)
        if self._is_zero(Y):
            return self.int_from_bits(self.tm.zeros(self.bit_width))
        n = self.bit_width
        remainder = self.tm.zeros(n)
        y_neg = self.tm.NOT(Y)
        y_neg = self.tm.succ(y_neg)
        y_neg = self.tm.slc(y_neg, 1, n + 1)
        for i in range(n):
            remainder = self.tm.sigma_L(remainder, 1)
            remainder = self.tm.slc(remainder, 1, n + 1)
            bit_i = self.tm.slc(X, i, i + 1)
            remainder = self.tm.write_bit(remainder, n - 1, bit_i)
            diff = self.tm.ripple_add(remainder, y_neg)
            carry = self.tm.slc(diff, 0, 1)
            rem_candidate = self.tm.slc(diff, 1, n + 1)
            sel = self._expand_bit(carry, n)
            remainder = self.tm.mu(remainder, rem_candidate, sel)
        return self.int_from_bits(remainder)

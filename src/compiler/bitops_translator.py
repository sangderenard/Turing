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

    # pointwise logic
    def bit_and(self, x: int, y: int) -> int:
        X, Y = self.bits_from_int(x), self.bits_from_int(y)
        out = self.tm.AND(X, Y)
        return self.int_from_bits(out)

    def bit_or(self, x: int, y: int) -> int:
        X, Y = self.bits_from_int(x), self.bits_from_int(y)
        out = self.tm.OR(X, Y)
        return self.int_from_bits(out)

    def bit_xor(self, x: int, y: int) -> int:
        X, Y = self.bits_from_int(x), self.bits_from_int(y)
        out = self.tm.XOR(X, Y)
        return self.int_from_bits(out)

    def bit_not(self, x: int) -> int:
        X = self.bits_from_int(x)
        out = self.tm.NOT(X)
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
        summed = self.tm.ripple_add(X, Y)
        truncated = self.tm.slc(summed, 1, self.bit_width + 1)
        return self.int_from_bits(truncated)

    def bit_sub(self, x: int, y: int) -> int:
        X, Y = self.bits_from_int(x), self.bits_from_int(y)
        y_inv = self.tm.NOT(Y)
        y_neg = self.tm.succ(y_inv)
        y_neg_trunc = self.tm.slc(y_neg, 1, self.bit_width + 1)
        diff = self.tm.ripple_add(X, y_neg_trunc)
        truncated = self.tm.slc(diff, 1, self.bit_width + 1)
        return self.int_from_bits(truncated)

    def _expand_bit(self, bit1: BitStr, n: int) -> BitStr:
        out = self.tm.zeros(n)
        for i in range(n):
            out = self.tm.write_bit(out, i, bit1)
        return out

    def bit_mul(self, x: int, y: int) -> int:
        X, Y = self.bits_from_int(x), self.bits_from_int(y)
        width = self.bit_width
        product = self.tm.zeros(width * 2)
        for i in range(width):
            sel_bit = self.tm.slc(Y, width - 1 - i, width - i)
            x_shift = self.tm.sigma_L(X, i)
            x_pad = self.tm.concat(self.tm.zeros(width - i), x_shift)
            sel = self._expand_bit(sel_bit, width * 2)
            addend = self.tm.mu(self.tm.zeros(width * 2), x_pad, sel)
            product = self.tm.ripple_add(product, addend)
            product = self.tm.slc(product, 1, width * 2 + 1)
        final = self.tm.slc(product, width, width * 2)
        return self.int_from_bits(final)

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

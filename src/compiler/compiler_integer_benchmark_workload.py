"""Exact unsigned scalar library boundaries for the product verifier."""
from __future__ import annotations


def uint32_equal(value: uint32, expected: uint32) -> bool:
    return value == expected


def uint64_equal(value: uint64, expected: uint64) -> bool:
    return value == expected

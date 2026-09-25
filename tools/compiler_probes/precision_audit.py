"""Audit: which basic AbstractTensor operations are limb-correct?

Every operation is scored against an EXACT Fraction reference computed from
the limbs, so the three outcomes are distinguished by measurement: correct,
refused (safe), or silently wrong (the dangerous one).
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from fractions import Fraction
import numpy as np
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.extended_precision import Precision

WIDTH = 2
VALUES = [0.1, 0.25, 1.5]

def exact(precision_value):
    rows = precision_value.to_float_lists()
    width = len(rows[0]) if isinstance(rows[0], list) else 1
    if width == 1 and not isinstance(rows[0], list):
        return [sum((Fraction(r) for r in rows), Fraction())]
    return [sum((Fraction(row[i]) for row in rows), Fraction()) for i in range(width)]

def make():
    return Precision.of(AbstractTensor.get_tensor(np.asarray(VALUES)), WIDTH)

def report(name, fn, expected_fn):
    try:
        produced = fn(make())
    except Exception as error:
        print("  %-14s REFUSED    %s" % (name, type(error).__name__))
        return
    try:
        if isinstance(produced, Precision):
            got = exact(produced)
        else:
            listed = produced.tolist() if hasattr(produced, "tolist") else produced
            if not isinstance(listed, list):
                listed = [listed]
            got = [Fraction(float(v)) for v in listed]
    except Exception as error:
        print("  %-14s ODD-SHAPE  %s" % (name, str(error)[:40]))
        return
    want = expected_fn([Fraction(v) for v in VALUES])
    if len(got) != len(want):
        print("  %-14s WRONG-LEN  got %d want %d" % (name, len(got), len(want)))
        return
    worst = max(abs(float(g - w)) for g, w in zip(got, want))
    verdict = "ok" if worst < 1e-30 else ("LOSSY %.1e" % worst if worst < 1e-10 else "WRONG %.3e" % worst)
    print("  %-14s %s" % (name, verdict))

print("ARITHMETIC (endorsed)")
report("add 1.0", lambda p: p + 1.0, lambda v: [x + 1 for x in v])
report("sub 1.0", lambda p: p - 1.0, lambda v: [x - 1 for x in v])
report("mul 3.0", lambda p: p * 3.0, lambda v: [x * 3 for x in v])
report("div 3.0", lambda p: p / 3.0, lambda v: [x / 3 for x in v])
report("neg", lambda p: -p, lambda v: [-x for x in v])
report("p*p", lambda p: p * p, lambda v: [x * x for x in v])

print("STRUCTURAL (must stride limbs)")
report("collapse", lambda p: p.collapse(), lambda v: v)
report("reshape", lambda p: Precision(p._value.reshape(3, WIDTH).reshape(-1), WIDTH), lambda v: v)

print("REDUCTIONS")
report("sum (collapsed)", lambda p: p.collapse().sum(), lambda v: [sum(v)])
report("mean (collapsed)", lambda p: p.collapse().mean(), lambda v: [sum(v) / len(v)])
report("sum (wide)", lambda p: p.sum(), lambda v: [sum(v)])
report("mean (wide)", lambda p: p.mean(), lambda v: [sum(v) / len(v)])

print("ELEMENTWISE BEYOND ARITHMETIC")
for name, call, ref in [
    ("abs", lambda p: abs(p), lambda v: [abs(x) for x in v]),
    ("sqrt", lambda p: p.sqrt(), None),
    ("floor", lambda p: p.floor(), lambda v: [Fraction(int(x)) for x in v]),
    ("exp", lambda p: p.exp(), None),
    ("sin", lambda p: p.sin(), None),
    ("pow2", lambda p: p ** 2, lambda v: [x * x for x in v]),
    ("sign", lambda p: p.sign(), lambda v: [Fraction(1) for _ in v]),
]:
    if ref is None:
        try:
            call(make()); print("  %-14s RETURNED (unverified reference)" % name)
        except Exception as e:
            print("  %-14s REFUSED    %s" % (name, type(e).__name__))
    else:
        report(name, call, ref)

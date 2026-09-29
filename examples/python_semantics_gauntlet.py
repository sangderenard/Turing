"""Python-semantics torture gauntlet for the source compiler.

Every case is legal, deterministic CPython whose answer depends on a rule a
naive lowering gets wrong: late binding, evaluation order, aliasing under
augmented assignment, exceptions that still mutate, ``finally`` swallowing a
return, class-scope name resolution, arbitrary-precision integers, floor
semantics on negatives, banker's rounding, generator protocols, descriptors,
metaclasses and reflected operators.

Nothing here is allowed to depend on implementation details that CPython does
not promise (object ids, small-int caching, hash randomization, timing).

``python_semantics_gauntlet()`` is the single entry: it returns one
``(case_name, value)`` row per case.  ``EXPECTED`` is the CPython answer for
the **first call in a fresh process**: the mutable default, the metaclass
registry and the ``__init_subclass__`` list are module state that survives
between calls, exactly as a compiled program's module state must.  A
compiled program is rated by comparing its rows to ``EXPECTED`` row by row;
a row the compiler refuses is a refusal, a row that differs is a miscompile.

Run directly (plain CPython, no compiler) to self-check the expectations:

    python examples/python_semantics_gauntlet.py
"""

from __future__ import annotations

import functools


# 1. Late-binding closures: every lambda sees the loop variable's last value
#    unless it was captured as a default.
def case_late_binding():
    late = [lambda: i for i in range(4)]
    early = [lambda i=i: i for i in range(4)]
    return tuple(f() for f in late) + tuple(f() for f in early)


# 2. A mutable default is created once and shared by every call.
def _accumulate(x, bucket=[]):
    bucket.append(x)
    return len(bucket)


def case_mutable_default():
    return (_accumulate(1), _accumulate(2), _accumulate(3, []), _accumulate(4))


# 3. for/else and while/else: ``else`` runs only when no ``break`` fired.
def case_loop_else():
    found = "none"
    for n in range(2, 10):
        if n * n == 49:
            found = n
            break
    else:
        found = "exhausted"
    k = 0
    while k < 3:
        k += 1
    else:
        k += 100
    return (found, k)


# 4. A chained comparison evaluates its middle operand exactly once.
def case_chained_once():
    calls = []

    def mid():
        calls.append(1)
        return 5

    return (1 < mid() < 10, 1 < mid() > 10, len(calls))


# 5. ``and``/``or`` return an operand, not a bool.
def case_short_circuit_values():
    return (0 or "", [] or [0], "a" and 3, None and 1 / 0, 0 or None or 7)


# 6. A walrus inside a comprehension binds in the enclosing function.
def case_walrus_leak():
    total = 0
    squares = [total := total + v for v in range(5)]
    return (squares, total)


# 7. Assignment targets bind left to right, after the right side is built.
def case_target_order():
    x = [0, 0, 0]
    i = 0
    i, x[i] = 1, 9
    a, b = 1, 2
    a, b = b, a + b
    first, *middle, last = range(6)
    (p, (q, *r)) = (1, (2, 3, 4))
    return (x, i, a, b, first, middle, last, p, q, r)


# 8. ``+=`` on a list mutates the shared object; ``= a +`` rebinds.  And
#    ``t[0] += [...]`` on a tuple raises *after* the list was extended.
def case_aliasing_augassign():
    a = [1]
    alias = a
    a += [2]
    b = [1]
    b_alias = b
    b = b + [2]
    t = ([1],)
    raised = False
    try:
        t[0] += [5]
    except TypeError:
        raised = True
    return (alias, b_alias, t, raised)


# 9. Arbitrary-precision ints, floor division/modulo toward -inf, banker's
#    rounding, and bool being an int.
def case_numeric_rules():
    return (
        2 ** 100 % 97,
        -7 // 2, -7 % 2, 7 // -2, 7 % -2,
        divmod(-17, 5),
        round(0.5), round(1.5), round(2.5), round(-2.5),
        True + True, sum([True, False, True]),
        (10 ** 30 + 1) - 10 ** 30,
        int("-0x1f", 16), 1 << 70 >> 68,
    )


# 10. Floats: representation error, signed zero, NaN never equal to itself.
def case_float_rules():
    nan = float("nan")
    return (
        0.1 + 0.2 == 0.3,
        repr(-0.0),
        nan == nan, nan != nan,
        1e308 * 10 == float("inf"),
        float.hex(0.5),
    )


# 11. ``return`` in ``finally`` swallows the in-flight exception and
#     overrides the ``try`` block's own return.
def _finally_wins():
    try:
        raise ValueError("lost")
    finally:
        return "finally"


def _finally_overrides():
    try:
        return "try"
    finally:
        return "finally-again"


def case_finally_semantics():
    log = []
    try:
        try:
            raise KeyError("inner")
        except KeyError as error:
            raise RuntimeError("outer") from error
    except RuntimeError as error:
        log.append(type(error.__cause__).__name__)
    try:
        pass
    except Exception:
        log.append("never")
    else:
        log.append("else-ran")
    return (_finally_wins(), _finally_overrides(), tuple(log))


# 12. Generators: send(), return value through ``yield from``, finally on
#     close().
def _echo():
    received = []
    try:
        while True:
            value = yield len(received)
            received.append(value)
    finally:
        received.append("closed")
        _echo.last = tuple(received)


def _inner():
    yield 1
    yield 2
    return "inner-result"


def _outer():
    result = yield from _inner()
    yield result


def case_generators():
    gen = _echo()
    first = next(gen)
    second = gen.send("a")
    third = gen.send("b")
    gen.close()
    return (first, second, third, _echo.last, tuple(_outer()))


# 13. Class-body names are invisible inside a comprehension in that body
#     (except the outermost iterable), and nonlocal rebinding.
def case_scopes():
    class Holder:
        base = 10
        values = [k for k in range(base, base + 3)]  # outermost iterable
        try:
            broken = [base for _ in range(1)]
        except NameError:
            broken = "NameError"

    counter = 0

    def bump():
        nonlocal counter
        counter += 5

    bump()
    bump()
    return (Holder.values, Holder.broken, counter)


# 14. Descriptors, __getattr__ fallback, __init_subclass__ and a metaclass
#     that intercepts construction.
class _Doubling:
    def __set_name__(self, owner, name):
        self.slot = "_" + name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.slot) * 2

    def __set__(self, instance, value):
        setattr(instance, self.slot, value)


class _Registry(type):
    made = []

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        _Registry.made.append(cls.__name__)
        return instance


class _Base(metaclass=_Registry):
    children = []

    def __init_subclass__(cls, tag="?", **kwargs):
        super().__init_subclass__(**kwargs)
        _Base.children.append((cls.__name__, tag))

    def __getattr__(self, name):
        return "missing:" + name


class _Leaf(_Base, tag="leaf"):
    width = _Doubling()

    def __init__(self, width):
        self.width = width


def case_object_model():
    leaf = _Leaf(21)
    dynamic = type("Dynamic", (_Leaf,), {"extra": 1})
    return (
        leaf.width, leaf.nothing_here, tuple(_Base.children),
        tuple(_Registry.made), dynamic.extra, dynamic.__mro__[1].__name__,
    )


# 15. Reflected operators and NotImplemented fallback; truthiness through
#     __len__ when __bool__ is absent.
class _Meters:
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        if isinstance(other, _Meters):
            return _Meters(self.value + other.value)
        return NotImplemented

    def __radd__(self, other):
        return _Meters(self.value + other)


class _Bag:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def case_operators():
    total = sum([_Meters(1), _Meters(2), _Meters(3)])  # starts at 0 + _Meters
    return (total.value, bool(_Bag(0)), bool(_Bag(3)), (5).__add__(1.5))


# 16. Dict ordering, unpack override order, setdefault, and a key equal to
#     True colliding with 1.
def case_dicts():
    base = {"a": 1, "b": 2}
    merged = {**base, "a": 9, **{"c": 3}}
    collide = {1: "int", True: "bool", 1.0: "float"}
    groups = {}
    for word in ("ant", "bee", "ape", "bat"):
        groups.setdefault(word[0], []).append(word)
    return (list(merged.items()), list(collide.items()), groups)


# 17. Parameters: positional-only, keyword-only, *args/**kwargs, and
#     decorators applied bottom-up.
def _trace(tag):
    def decorate(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return tag + "(" + function(*args, **kwargs) + ")"
        return wrapper
    return decorate


@_trace("outer")
@_trace("inner")
def _signature(a, b=2, /, c=3, *args, d, e=5, **kwargs):
    return f"{a},{b},{c},{args},{d},{e},{sorted(kwargs.items())}"


def case_parameters():
    return (_signature(1, d=4), _signature(1, 9, 8, 7, 6, d=0, z=1),
            _signature.__name__)


# 18. Strings: negative-step slices, repetition, nested format specs.
def case_strings():
    text = "gauntlet"
    width, places = 9, 3
    return (
        text[::-2], text[-3:1:-1], "ab" * 3, "-" * -2,
        f"{3.14159:{width}.{places}f}|", f"{255:#010b}", f"{'x':^5}",
        "a,b,,c".split(","), "  pad ".strip().upper(),
    )


# 19. Structural pattern matching with class patterns, guards and captures.
class _Point:
    __match_args__ = ("x", "y")

    def __init__(self, x, y):
        self.x, self.y = x, y


def _describe(subject):
    match subject:
        case _Point(0, 0):
            return "origin"
        case _Point(x, 0) if x > 0:
            return f"east {x}"
        case [first, *rest] if len(rest) == 2:
            return f"triple from {first}"
        case {"kind": "circle", "r": radius}:
            return f"circle {radius}"
        case str() | bytes():
            return "text"
        case _:
            return "other"


def case_pattern_matching():
    return tuple(_describe(s) for s in (
        _Point(0, 0), _Point(4, 0), _Point(-1, 0), [1, 2, 3],
        {"kind": "circle", "r": 2, "extra": 0}, b"x", 7,
    ))


# 20. Sorting is stable; reduce folds left; ``sorted`` of mixed-sign keys.
def case_ordering():
    rows = [("b", 2), ("a", 2), ("c", 1), ("d", 1)]
    by_count = sorted(rows, key=lambda row: row[1])
    folded = functools.reduce(lambda acc, v: acc * 10 + v, [1, 2, 3], 0)
    return (by_count, folded, sorted([-3, 2, -1], key=abs),
            max([], default="empty"))


CASES = (
    case_late_binding, case_mutable_default, case_loop_else,
    case_chained_once, case_short_circuit_values, case_walrus_leak,
    case_target_order, case_aliasing_augassign, case_numeric_rules,
    case_float_rules, case_finally_semantics, case_generators,
    case_scopes, case_object_model, case_operators, case_dicts,
    case_parameters, case_strings, case_pattern_matching, case_ordering,
)


def python_semantics_gauntlet():
    """Run every case once, in order; one ``(name, value)`` row per case."""

    return tuple((case.__name__, case()) for case in CASES)


# The CPython 3.11 answer for every case.
EXPECTED = {
    "case_late_binding": (3, 3, 3, 3, 0, 1, 2, 3),
    "case_mutable_default": (1, 2, 1, 3),
    "case_loop_else": (7, 103),
    "case_chained_once": (True, False, 2),
    "case_short_circuit_values": ("", [0], 3, None, 7),
    "case_walrus_leak": ([0, 1, 3, 6, 10], 10),
    "case_target_order": (
        [0, 9, 0], 1, 2, 3, 0, [1, 2, 3, 4], 5, 1, 2, [3, 4],
    ),
    "case_aliasing_augassign": ([1, 2], [1], ([1, 5],), True),
    "case_numeric_rules": (
        16, -4, 1, -4, -1, (-4, 3), 0, 2, 2, -2, 2, 2, 1, -31, 4,
    ),
    "case_float_rules": (
        False, "-0.0", False, True, True, "0x1.0000000000000p-1",
    ),
    "case_finally_semantics": (
        "finally", "finally-again", ("KeyError", "else-ran"),
    ),
    "case_generators": (
        0, 1, 2, ("a", "b", "closed"), (1, 2, "inner-result"),
    ),
    "case_scopes": ([10, 11, 12], "NameError", 10),
    "case_object_model": (
        42, "missing:nothing_here", (("_Leaf", "leaf"), ("Dynamic", "?")),
        ("_Leaf",), 1, "_Leaf",
    ),
    "case_operators": (6, False, True, NotImplemented),
    "case_dicts": (
        [("a", 9), ("b", 2), ("c", 3)],
        [(1, "float")],
        {"a": ["ant", "ape"], "b": ["bee", "bat"]},
    ),
    "case_parameters": (
        "outer(inner(1,2,3,(),4,5,[]))",
        "outer(inner(1,9,8,(7, 6),0,5,[('z', 1)]))",
        "_signature",
    ),
    "case_strings": (
        "tlna", "ltnu", "ababab", "", "    3.142|", "0b11111111", "  x  ",
        ["a", "b", "", "c"], "PAD",
    ),
    "case_pattern_matching": (
        "origin", "east 4", "other", "triple from 1", "circle 2", "text",
        "other",
    ),
    "case_ordering": (
        [("c", 1), ("d", 1), ("b", 2), ("a", 2)], 123, [-1, 2, -3], "empty",
    ),
}


if __name__ == "__main__":
    import pprint

    rows = python_semantics_gauntlet()
    if EXPECTED is None:
        pprint.pprint(dict(rows), width=100, sort_dicts=False)
    else:
        failures = [
            (name, value, EXPECTED.get(name))
            for name, value in rows
            if EXPECTED.get(name) != value
        ]
        for name, value, expected in failures:
            print(f"MISMATCH {name}: got {value!r}, expected {expected!r}")
        print(f"{len(rows) - len(failures)}/{len(rows)} cases match CPython")

"""Differential fuzzer: random programs from composable snippets, lowered to
C and compared with Python; failures clustered by signature.

usage: python tools/differential_fuzz.py OUTDIR SEED COUNT
(FUZZ_EXT=1/2/3/4 widen the grammar: extended builtins, then more control
and sequences, then helpers that mutate and read a caller's list, then
whole-collection loops, dict deletion, swaps and replication.
FUZZ_BACKENDS=llvm,fortran also runs each program through LLVM and/or
Fortran: a wrong value is WRONG, and a refusal of a program C ran is
LLVM-LOUD or FORTRAN-LOUD.)
Writes OUTDIR/<id>.py for every program and OUTDIR/results.jsonl.
"""
import sys, os, random, json, re, warnings, pathlib, tempfile, traceback
warnings.filterwarnings("ignore")
ROOT = os.environ.get("TURING_ROOT", str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.path.insert(0, ROOT)

# Each snippet: (setup lines using {v} fresh suffix, int expression it yields).
# {n} is the int parameter.  Every snippet is total for n in [-3, 12].
SNIPPETS = [
    # lists
    (["xs{v} = [{a}, n, {b}]"], "xs{v}[0] * 7 + xs{v}[2]"),
    (["xs{v} = []", "for i{v} in range(n % 6 + 1):", "    xs{v}.append(i{v} * {a})"], "len(xs{v}) * 10 + xs{v}[-1]"),
    (["xs{v} = [{a}, n, {b}, {c}]", "xs{v}.sort()"], "xs{v}[0] * 100 + xs{v}[3]"),
    (["xs{v} = sorted([n, {a}, {b}])"], "xs{v}[1]"),
    (["xs{v} = [{a}, {b}, n]", "p{v} = xs{v}.pop()"], "p{v} * 10 + len(xs{v})"),
    (["xs{v} = [{a}, {b}, n]", "xs{v}.insert(1, {c})"], "xs{v}[1] * 10 + xs{v}[2]"),
    (["xs{v} = [{a}, {b}, n]", "xs{v}.reverse()"], "xs{v}[0] - xs{v}[2]"),
    (["xs{v} = [{a}, n, {b}]", "xs{v}[1] = {c}", "xs{v}[0] += n"], "xs{v}[0] + xs{v}[1]"),
    (["xs{v} = [i{v} * {a} for i{v} in range(n % 5 + 1) if i{v} % 2 == 0]"], "len(xs{v}) * 100 + sum(xs{v})"),
    (["xs{v} = [n, {a}, {b}]"], "max(xs{v}) * 10 + min(xs{v})"),
    (["xs{v} = [{a}, n]", "xs{v}.extend([{b}, {c}])"], "len(xs{v}) * 10 + xs{v}[3]"),
    (["xs{v} = [{a}, {b}, {a}]", "xs{v}.remove({a})"], "len(xs{v}) * 10 + xs{v}[0]"),
    # dicts / sets / tuples
    (["d{v} = {{1: {a}, 2: {b}}}", "d{v}[3] = n"], "d{v}[1] + d{v}[3] * 10 + len(d{v})"),
    (["d{v} = {{}}", "for i{v} in range(n % 4 + 1):", "    d{v}[i{v} % 2] = i{v}"], "len(d{v}) * 10 + d{v}[0]"),
    (["s{v} = set()", "for i{v} in range(n % 5 + 2):", "    s{v}.add(i{v} % 3)"], "len(s{v}) * 10 + (1 if 2 in s{v} else 0)"),
    (["t{v} = (n, {a}, {b})", "q{v}, r{v}, w{v} = t{v}"], "q{v} + r{v} * w{v}"),
    (["t{v} = (n + {a}, n * {b}, {c})"], "len(t{v}) * 100 + t{v}[1]"),
    (["t{v} = (n > {a}, n < {b})", "c{v} = 0", "for x{v} in t{v}:", "    if x{v}:", "        c{v} += 1"], "c{v} * 10 + len(t{v})"),
    # control
    (["c{v} = 0", "for i{v} in range(n % 7 + 1):", "    if i{v} % 3 == 0:", "        continue", "    c{v} += i{v}"], "c{v}"),
    (["c{v} = -1", "for i{v} in range(10):", "    if i{v} * {a} > n + 5:", "        c{v} = i{v}", "        break"], "c{v}"),
    (["c{v} = 0", "for i{v} in range(5):", "    if i{v} > n:", "        break", "else:", "    c{v} = 99"], "c{v}"),
    (["k{v} = {a}", "w{v} = 0", "while k{v} > 0 and w{v} < 20:", "    k{v} -= 1", "    w{v} += n % 3 + 1"], "w{v} * 10 + k{v}"),
    (["c{v} = 1 if 0 < n < {a} + 5 else 2"], "c{v}"),
    (["c{v} = n or {a}"], "c{v}"),
    (["c{v} = {a} if n % 2 == 0 and n > 1 else {b}"], "c{v}"),
    (["c{v} = 0", "if n > {a}:", "    c{v} = 1", "elif n == {a}:", "    c{v} = 2", "else:", "    c{v} = 3"], "c{v}"),
    # arithmetic
    (["c{v} = n // 3 + n % 4 - (-n) // 2"], "c{v}"),
    (["c{v} = abs(n - {a}) + max(n, {b}) - min(n, {c})"], "c{v}"),
    (["c{v} = (n * {a}) ** 2 % 97"], "c{v}"),
    # any / all / sum generators
    (["c{v} = int(any(i{v} * {a} > n for i{v} in range(4)))"], "c{v}"),
    (["c{v} = sum(i{v} * n for i{v} in range(4))"], "c{v}"),
    # helper call
    (["c{v} = helper(n, {a})"], "c{v}"),
    (["xs{v} = build(n % 5)"], "len(xs{v}) * 10 + sum(xs{v})"),
]

EXTENDED = [
    (["xs{v} = [{a}, n, {b}]", "t{v} = 0", "for k{v}, x{v} in enumerate(xs{v}):", "    t{v} += k{v} * x{v}"], "t{v}"),
    (["xs{v} = [{a}, n, {b}]", "ys{v} = [n, {c}, {a}]", "t{v} = 0", "for p{v}, q{v} in zip(xs{v}, ys{v}):", "    t{v} += p{v} * q{v}"], "t{v}"),
    (["xs{v} = [{a}, {b}, {c}]"], "(1 if n in xs{v} else 0) + (2 if {a} in xs{v} else 0)"),
    (["xs{v} = [{a}, {b}, {a}, n]"], "xs{v}.count({a}) * 10 + xs{v}.index({b})"),
    (["f{v} = n * 0.5 + {a}", "g{v} = f{v} * 2.0"], "int(g{v})"),
    (["xs{v} = [{a}, {b}] + [n, {c}]"], "len(xs{v}) * 10 + xs{v}[2]"),
    (["d{v} = {{1: {a}}}", "d{v}[2] = n", "xs{v} = [{b}] + [n, {c}]"], "xs{v}[1] * 10 + len(xs{v}) + d{v}[2]"),
    (["xs{v} = [n] + [{a}, {b}] + [n * {c}]"], "len(xs{v}) * 10 + xs{v}[3]"),
    (["t{v} = 0", "for y{v} in [x{v} * n for x{v} in [{a}, {b}, {c}]]:", "    t{v} += y{v}"], "t{v}"),
    (["ys{v} = [x{v} * n for x{v} in [{a}, {b}]]", "ys{v}.append(n + {c})"], "len(ys{v}) * 100 + sum(ys{v}) + ys{v}[2] * 1000"),
    (["xs{v} = [{a}] * (n % 4 + 1)"], "len(xs{v}) * 10 + xs{v}[0]"),
    (["xs{v} = [{a}, n, {b}, {c}]", "del xs{v}[1]"], "len(xs{v}) * 10 + xs{v}[1]"),
    (["q{v}, r{v} = pair(n, {a})"], "q{v} * 10 + r{v}"),
    (["c{v} = first_over(n, {a})"], "c{v}"),
    # an early return beside a returned length of a list parameter (corpus
    # c372, plan C7)
    (["c{v} = neg_or_len([{a}, n, n - {b}])"], "c{v}"),
    (["xs{v} = []", "k{v} = n % 5 + 2", "while k{v} > 0:", "    xs{v}.append(k{v})", "    k{v} -= 2"], "len(xs{v}) * 10 + xs{v}[0]"),
    (["xs{v} = [{a}, n, {b}]", "for j{v} in range(len(xs{v})):", "    xs{v}[j{v}] = xs{v}[j{v}] * 2"], "xs{v}[0] + xs{v}[1] + xs{v}[2]"),
    (["xs{v} = [n, {a}, {b}]", "m{v} = xs{v}[0]", "for x{v} in xs{v}:", "    if x{v} > m{v}:", "        m{v} = x{v}"], "m{v}"),
    (["bs{v} = [n > {a}, n < {b}, True]"], "sum(1 for b{v} in bs{v} if b{v})"),
    (["xs{v} = [{a}, n, {b}]", "ys{v} = list(xs{v})", "ys{v}.append(n)"], "len(xs{v}) * 10 + len(ys{v})"),
    (["xs{v} = [{a}, n, {b}, {c}]"], "xs{v}[-1] * 10 + xs{v}[-2]"),
    # comprehensions over an all-constant (folded) list (corpus c240-c243)
    (["xs{v} = [{a}, {b}, {c}]", "ys{v} = [x{v} * n for x{v} in xs{v}]"], "sum(ys{v}) * 10 + len(ys{v})"),
    (["xs{v} = [{a}, {b}, {c}]"], "int(any([x{v} > n for x{v} in xs{v}])) * 10 + int(all([x{v} > n for x{v} in xs{v}]))"),
    (["xs{v} = [{a}, {b}, {c}]", "ys{v} = [x{v} - n for x{v} in xs{v}]"], "ys{v}[n % 3] * 10 + sum(ys{v})"),
    (["s{v} = {{(x{v} + n) % 3 for x{v} in [{a}, {b}, {c}]}}"], "len(s{v})"),
    # a read of a list before a later store, sort or pop of it, and a
    # returned read of a grown list (corpus c310-c319)
    (["xs{v} = [{a}, n]", "xs{v}.append({b})", "r{v} = xs{v}[1]", "xs{v}[1] = n + {c}"], "r{v} * 10 + xs{v}[1] + len(xs{v})"),
    (["xs{v} = [{a}, n, {b}]", "r{v} = xs{v}[0]", "xs{v}.sort()"], "r{v} * 100 + xs{v}[0]"),
    (["xs{v} = [{a}, n, {b}]", "r{v} = xs{v}[-1]", "xs{v}.pop(0)"], "r{v} * 10 + len(xs{v})"),
    # a set comprehension grown by add (corpus c285, c286)
    (["s{v} = {{(x{v} + n) % 3 for x{v} in [{a}, {b}, {c}]}}", "s{v}.add(n + {a})"], "len(s{v}) * 100 + sum(s{v})"),
    # an unrolled literal loop beside an in-place sort (corpus c244, plan C15)
    (["t{v} = 0", "for x{v} in [{a}, {b}]:", "    t{v} += x{v} * n", "xs{v} = [{c}, n, {a}]", "xs{v}.sort()"], "xs{v}[0] * 10 + xs{v}[2] + t{v}"),
    (["ys{v} = [x{v} * n for x{v} in [{a}, {b}]]", "xs{v} = [{c}, n]", "xs{v}.sort(reverse=True)"], "xs{v}[0] * 10 + sum(ys{v})"),
    # a display rebound inside an unrolled loop (corpus c245, plan C2)
    (["xs{v} = [{a}]", "for i{v} in range(3):", "    xs{v} = [i{v} + n, {b}, i{v}]"], "len(xs{v}) * 100 + xs{v}[0] * 10 + xs{v}[1]"),
    # a display rebound inside a retained loop that reads it (corpus c246)
    (["xs{v} = [{a}]", "t{v} = 0", "for i{v} in range(4):", "    t{v} += len(xs{v}) * 10 + xs{v}[-1]", "    xs{v} = [i{v}, n, {b}]"], "t{v} * 10 + len(xs{v})"),
    # ... and read by subscript after a loop that may run zero times (c247)
    (["xs{v} = [{a}, {b}]", "for i{v} in range(n):", "    xs{v} = [xs{v}[1] + i{v}, n, {c}]"], "xs{v}[0] * 10 + xs{v}[1] + len(xs{v})"),
    # an extend by an all-constant display beside a comprehension over a
    # folded list (corpus c248, plan C16)
    (["ys{v} = [{a}, n]", "ys{v}.extend([{b}, {c}])", "ws{v} = [z{v} - n for z{v} in [{a}, {c}]]"], "ys{v}[3] * 10 + sum(ws{v}) + len(ys{v})"),
    # len of a dict after a keyed delete, also after an unrolled loop
    # (corpus c370, plan C3)
    (["d{v} = {{1: {a}, 2: n, 3: {b}}}", "del d{v}[2]"], "len(d{v}) * 10 + n"),
    (["e{v} = {{}}", "for i{v} in range(4):", "    e{v}[i{v}] = i{v} + n", "del e{v}[{c} % 4]"], "len(e{v})"),
    # chained comparisons whose middle is a builtin conversion or measure,
    # one behind a guarding ``and`` (lanes plan A0)
    (["f{v} = n * 0.5 + {a}"], "1 if 0 <= int(f{v}) <= {b} + 3 else 2"),
    (["xs{v} = [{a}, n, {b}]"], "(1 if 1 < len(xs{v}) <= n else 0) + (2 if n != 0 and 0 < abs({a} // n) < 3 else 0)"),
    # Optional scalar returns: a statement merge with a None arm, and an
    # early ``return None`` guard, each tested for presence by the caller.
    (["o{v} = maybe_inc(n, {a})"], "(-1 if o{v} is None else o{v})"),
    (["o{v} = maybe_guard(n, {a})", "t{v} = 0", "if o{v} is not None:", "    t{v} = o{v} * 3"], "t{v}"),
    # A value-union return (``x if c else None``) tested by identity.
    (["o{v} = maybe_pick(n, {a})", "t{v} = 5", "if o{v} is not None:", "    t{v} = o{v} - 2"], "t{v}"),
    # The same union forwarded unchanged by a wrapper.
    (["o{v} = wrap_pick(n, {a})"], "(-3 if o{v} is None else o{v} + 1)"),
    # An IfExp whose arm is an operator over a union narrowed by the test
    # (corpus c364-c367, doc 34 U1).
    (["o{v} = maybe_pick(n, {a})", "q{v} = None if o{v} is None else o{v} * 3 - {b}"], "(-9 if q{v} is None else q{v})"),
    (["o{v} = bump_pick(n, {a})"], "(-4 if o{v} is None else o{v})"),
]
EXT_HELPERS = """def pair(x: int, y: int) -> tuple[int, int]:
    return x // y, x % y


def maybe_inc(x: int, y: int):
    if x > y:
        r = x + y
    else:
        r = None
    return r


def maybe_guard(x: int, y: int):
    if x < y:
        return None
    return x - y


def maybe_pick(x: int, y: int):
    return x * 2 if x != y else None


def wrap_pick(x: int, y: int):
    return maybe_pick(x + 1, y)


def bump_pick(x: int, y: int):
    s = maybe_pick(x, y)
    return None if s is None else s + 1


def first_over(x: int, y: int) -> int:
    for i in range(10):
        if i * y > x:
            return i
    return -1


def neg_or_len(xs: list[int]) -> int:
    for x in xs:
        if x < 0:
            return x
    return len(xs)


"""

EXTENDED2 = [
    (["f{v} = 0.0", "for i{v} in range(n % 4 + 1):", "    f{v} += i{v} * 0.5"], "int(f{v} * 4)"),
    (["xs{v} = [n, {a}, {b}]"], "total(xs{v})"),
    (["m{v} = 0", "for i{v} in range(3):", "    for j{v} in range(i{v} + 1):", "        m{v} += i{v} * j{v} + n"], "m{v}"),
    (["xs{v} = []", "for i{v} in range(4):", "    if i{v} % 2 == n % 2:", "        xs{v}.append(i{v})"], "len(xs{v}) * 10 + (xs{v}[0] if xs{v} else -1)"),
    (["c{v} = 0", "while True:", "    c{v} += 1", "    if c{v} > n % 5:", "        break"], "c{v}"),
    (["xs{v} = [n, {a}, {b}]", "ys{v} = [x{v} * 2 for x{v} in xs{v} if x{v} > {c}]"], "len(ys{v}) * 100 + sum(ys{v})"),
    (["q{v} = divmod(n + 7, {a})"], "q{v}[0] * 10 + q{v}[1]"),
    (["xs{v} = sorted([n, {a}, {b}], reverse=True)"], "xs{v}[0] * 10 + xs{v}[2]"),
    (["xs{v} = [n, {a}, {b}]", "xs{v}.sort(reverse=True)"], "xs{v}[0] - xs{v}[2]"),
    (["k{v} = n", "s{v} = 0", "while k{v} > 0:", "    s{v} += k{v} % 10", "    k{v} //= 10"], "s{v}"),
    (["xs{v} = [{a}, {b}]", "xs{v} += [n]"], "len(xs{v}) * 10 + xs{v}[-1]"),
    (["b{v} = n > {a} or n < -{b}"], "int(b{v}) + (3 if not b{v} else 0)"),
    # a callee growing its own list and dict (A6, capacity of hoisted storage)
    (["c{v} = localgrow(n + {a})"], "c{v}"),
    # a callee returning a list display (A6)
    (["xs{v} = pairlist(n + {a})"], "len(xs{v}) * 10 + xs{v}[1]"),
    (["xs{v} = pairlist(n)", "xs{v}.append({b})"], "len(xs{v}) * 10 + xs{v}[2] + xs{v}[0] * 100"),
    # a list bound to equal-length displays on two branches (c271)
    (["if n > {a}:", "    t{v} = [n, {b}]", "else:", "    t{v} = [{c}, n]", "t{v}.append({a})"], "len(t{v}) * 100 + t{v}[0] * 10 + t{v}[2]"),
    (["t{v} = [n, {b}] if n % 2 == 0 else [{c}, {a}]"], "len(t{v}) * 10 + t{v}[1]"),
    # a set bound to displays on two branches (c274)
    (["s{v} = {{n, {a}}} if n > {b} else {{{c}, 2}}", "s{v}.add({a})"], "len(s{v}) * 10 + int({c} in s{v})"),
    # concatenation of displays then a dict display (p16_29, region order)
    (["xs{v} = [{a}, -1] + [n, {b}]", "d{v} = {{1: {c}}}"], "xs{v}[2] * 10 + xs{v}[0] + d{v}[1]"),
    # dict.update from a display stores its rows (A3)
    (["d{v} = {{1: {a}, 2: n}}", "d{v}.update({{1: n + {b}, 3: {c}}})"], "d{v}[1] * 10 + d{v}[2] + len(d{v}) * 100 + d{v}[3]"),
    # a dict counted through get() in a loop, read after it (c330)
    (["d{v} = {{}}", "for i{v} in range(n % 4 + {a}):", "    d{v}[i{v} % 2] = d{v}.get(i{v} % 2, {b}) + {c}"], "d{v}[0] + len(d{v}) * 100"),
    (["d{v} = {{0: n}}", "d{v}[1] = d{v}.get(0, {b}) + {a}"], "d{v}[1] * 10 + len(d{v})"),
]
EXTENDED3 = [
    (["xs{v} = [n, {a}]", "setfirst(xs{v}, {b})"], "len(xs{v}) * 100 + xs{v}[0] * 10 + xs{v}[1]"),
    (["xs{v} = [n, {a}]", "push(xs{v}, {b})"], "len(xs{v}) * 100 + xs{v}[-1]"),
    (["xs{v} = [{a}, n]", "grow(xs{v}, n % 3 + 1)"], "len(xs{v}) * 100 + xs{v}[-1] + xs{v}[0]"),
    (["xs{v} = [n, {a}, {b}]", "xs{v}.append({c})"], "head(xs{v}) * 10 + width(xs{v})"),
    (["xs{v} = [n, {a}]", "swap(xs{v})"], "xs{v}[0] * 10 + xs{v}[1]"),
    (["xs{v} = []", "push(xs{v}, n)", "push(xs{v}, {a})", "setfirst(xs{v}, {b})"], "len(xs{v}) * 100 + xs{v}[0] + xs{v}[1]"),
    (["xs{v} = [n, {a}]", "ys{v} = [{b}]", "grow(ys{v}, 2)"], "head(xs{v}) + width(ys{v}) * 10 + ys{v}[-1]"),
    (["xs{v} = [n, {a}, {b}]"], "pick(xs{v}, n % 3) * 10 + width(xs{v})"),
    # a set grown by a helper (corpus c288, c289)
    (["s{v} = {{n, {a}}}", "addto(s{v}, {b})", "addto(s{v}, n)"], "len(s{v}) * 100 + sum(s{v})"),
    # backend parity (lane E): a static int and bool tuple read in a loop
    # under a condition, a resident sum/count/index/any, a sort of a
    # display with a parameter, an f-string measured in a callee
    (["r{v} = 0", "for x{v} in ({a}, {b}, {c}):", "    if x{v} <= n:", "        r{v} += x{v}"], "r{v}"),
    (["r{v} = 0", "for f{v} in (True, False, n > {a}):", "    if f{v}:", "        r{v} += 1"], "r{v}"),
    (["xs{v} = [{a}, n, {a}, {b}]"], "sum(xs{v}) * 100 + xs{v}.count({a}) * 10 + xs{v}.index(n)"),
    (["xs{v} = [{a}, n, {b}]", "xs{v}.remove(n)"], "len(xs{v}) * 10 + xs{v}[1]"),
    (["c{v} = label(n + {a})"], "c{v}"),
    # list slices (lane E): a folded slice, a slice of a display holding a
    # parameter, a slice with a runtime lower bound, a sum of a folded list
    (["xs{v} = [{a}, {b}, {c}, {a}]", "ys{v} = xs{v}[1:3] + [n]"], "len(ys{v}) * 100 + ys{v}[0] * 10 + ys{v}[2]"),
    (["xs{v} = [{a}, n, {b}]", "ys{v} = xs{v}[-2:]"], "ys{v}[0] * 10 + len(ys{v})"),
    (["xs{v} = [{a}, n, {b}, {c}]", "ys{v} = xs{v}[n % 3:]"], "len(ys{v}) * 10 + ys{v}[0]"),
    (["xs{v} = [{a}, {b}, {c}]"], "sum(xs{v}) + n"),
    # loop targets over an unrolled comprehension and over zip (lane E)
    (["t{v} = 0", "for y{v} in [x{v} * n for x{v} in [{a}, {b}]]:", "    t{v} += y{v}"], "t{v}"),
    (["t{v} = 0", "for p{v}, q{v} in zip([{a}, n], [n, {b}]):", "    t{v} += p{v} * {c} + q{v}"], "t{v}"),
    # max/min/sum over a resident list, Fortran reduces over the live
    # length (lane E)
    (["xs{v} = [n, {a}, n * {b}]", "xs{v}.append({c} - n)"], "max(xs{v}) * 1000 + min(xs{v}) * 10 + sum(xs{v}, n)"),
    (["xs{v} = [n, {a}]", "xs{v}.append(n - {b})"], "int(any(xs{v})) * 10 + int(all(xs{v}))"),
    # a callee's scalar result beside a callee swap, Fortran (lane E)
    (["xs{v} = [n, {a}]", "h{v} = head(xs{v})", "swap(xs{v})", "swap(xs{v})", "swap(xs{v})"], "h{v} * 100 + xs{v}[0] * 10 + xs{v}[1]"),
    # dict.get with a default past float64's exact integers (lane E)
    (["d{v} = {{1: {a}, 2: n}}"], "d{v}.get({b} + 3, 36028797018963969) + d{v}.get(2, {c})"),
    # a callee reads the length of a caller list grown in a loop (lane E)
    (["xs{v} = [n, {a}]", "for i{v} in range(n % 3 + 1):", "    xs{v}.append(i{v} + {b})"], "width(xs{v}) * 100 + head(xs{v})"),
    # a dict that accumulates into itself through get (lane E)
    (["d{v} = {{}}", "for i{v} in range(n % 4 + 1):", "    d{v}[i{v} % 2] = d{v}.get(i{v} % 2, {a}) + 36028797018963969", "t{v} = 0", "for k{v} in range(3):", "    if k{v} in d{v}:", "        t{v} = t{v} + d{v}[k{v}]"], "t{v}"),
    # a pop from the front inside the growing loop, then iteration (lane E)
    (["xs{v} = []", "for i{v} in range(n % 5 + 2):", "    xs{v}.append(i{v} + {a})", "    if i{v} % 2 == 1:", "        xs{v}.pop(0)", "t{v} = 0", "for x{v} in xs{v}:", "    t{v} = t{v} * 10 + x{v}"], "t{v}"),
    # an optional tuple result unpacked after a None test (lane E)
    (["f{v} = maybe_pair(n, {a}) if n != {b} else None", "r{v} = -1", "if f{v} is not None:", "    p{v}, q{v} = f{v}", "    r{v} = p{v} * 10 + q{v}"], "r{v}"),
    # a callee that returns an optional scalar, read after a None test (lane E)
    (["o{v} = maybe_twice(n, {a})", "t{v} = {b}", "if o{v} is not None:", "    t{v} = o{v} - {c}"], "t{v} * 10 + n"),
    # reductions over a populated list literal (lane E)
    ([], "sum([{a}, {b}, n]) * 10 + min([{c}, n, {a}])"),
    # pop at an index and del on a fixed list (lane E)
    (["xs{v} = [{a}, n, {b}, {c}]", "a{v} = xs{v}.pop(1)", "del xs{v}[0]"], "a{v} * 100 + xs{v}[0] * 10 + xs{v}[1]"),
    # a callee list bound to branch displays (c271, c331)
    (["c{v} = branchpair(n - {a})"], "c{v}"),
    # enumerate over a list holding a large int (lane E)
    (["xs{v} = [{a}, n + 36028797018963969, {b}]", "t{v} = 0", "for k{v}, x{v} in enumerate(xs{v}):", "    t{v} += k{v} * x{v}"], "t{v}"),
    # tuple comprehension targets over enumerate and zip (lane E)
    (["xs{v} = [{a}, n, {b}]", "ys{v} = [k{v} * x{v} for k{v}, x{v} in enumerate(xs{v})]", "zs{v} = [p{v} * q{v} for p{v}, q{v} in zip(xs{v}, ys{v})]"], "sum(ys{v}) + sum(zs{v})"),
    # enumerate counting from a start (lane E)
    (["t{v} = 0", "for k{v}, x{v} in enumerate([{a}, n, {b}], n % 4 + {c}):", "    t{v} = t{v} * 10 + k{v} * x{v}"], "t{v}"),
    # a callee filters a list literal in its loop, so it keeps live flags (c197)
    ([], "positives([{a}, -{b}, {c}]) * 10 + n"),
]
EXT3_HELPERS = """def setfirst(xs: list[int], v: int) -> None:
    xs[0] = v


def push(xs: list[int], v: int) -> None:
    xs.append(v)


def grow(xs: list[int], k: int) -> None:
    for i in range(k):
        xs.append(i)


def head(xs: list[int]) -> int:
    return xs[0]


def width(xs: list[int]) -> int:
    return len(xs)


def swap(xs: list[int]) -> None:
    t = xs[0]
    xs[0] = xs[1]
    xs[1] = t


def pick(xs: list[int], i: int) -> int:
    return xs[i]


def addto(s: set[int], v: int) -> None:
    s.add(v)


def label(x: int) -> int:
    return len(f"value={int(x)}")


def maybe_pair(x: int, y: int) -> tuple[int, int] | None:
    if x > y:
        return None
    return (x + 1, y * 2)


def maybe_twice(x: int, y: int):
    return x * 2 if x != y else None


def positives(xs: list[int]) -> int:
    t = 0
    for x in xs:
        if x > 0:
            t += x
    return t


def branchpair(x: int) -> int:
    if x > 2:
        t = [x, 1]
    else:
        t = [5, 6]
    t.append(7)
    return len(t) * 10 + t[0] + t[2]


"""
EXT2_HELPERS = """def total(xs: list[int]) -> int:
    t = 0
    for x in xs:
        t += x
    return t


def pairlist(x: int) -> list[int]:
    return [x, x + 1]


def localgrow(x: int) -> int:
    xs = [7, 5]
    xs.append(x)
    d = {1: x, 2: 5}
    d[3] = x * 2
    return xs[0] + xs[2] * 10 + d[1] * 100 + d[3]


"""

HELPERS = '''def helper(x: int, y: int) -> int:
    if x > y:
        return x - y
    return y * 2


def build(m: int) -> list[int]:
    out = []
    for i in range(m):
        out.append(i * i)
    return out


'''


# FUZZ_EXT=4 (silent-wrong lane): whole-collection loops, dict iteration
# and deletion, tuple swaps, nested stores, slices then appends, list
# replication, aliasing and sets.  Seeds 80-83 found the families of corpus
# c333-c338.
EXTENDED4 = [
    (["xs{v} = [n, {a}]", "r{v} = xs{v}[0]", "swap(xs{v})"], "r{v} * 100 + head(xs{v}) * 10 + xs{v}[1]"),
    (["d{v} = {{0: n}}", "d{v}[1] = d{v}.get(0, -1) + {a}", "t{v} = 0", "for y{v} in [x{v} * n for x{v} in [{b}, 0]]:", "    t{v} += y{v}", "xs{v} = [n, {c}]", "h{v} = head(xs{v})", "swap(xs{v})"], "d{v}[1] * 10 + len(d{v}) + t{v} + h{v} * 100 + xs{v}[0] * 10 + xs{v}[1]"),
    (["xs{v} = [n, {a}, {b}, {c}]", "s{v} = 0", "while xs{v}:", "    s{v} = s{v} * 3 + xs{v}.pop()"], "s{v}"),
    (["xs{v} = [{a}, n, {b}]", "r{v} = xs{v}[-1]", "xs{v}.pop(0)", "s{v} = 0", "for k{v}, x{v} in enumerate([{c}, n], n % 3 + {a}):", "    s{v} = s{v} * 10 + k{v} * x{v}"], "r{v} * 10 + len(xs{v}) + s{v}"),
    (["s{v} = 0", "t{v} = (n, {a}, n + {b})", "for x{v} in t{v}:", "    if x{v} % 2 == 0:", "        continue", "    s{v} = s{v} * 3 + x{v}"], "s{v} * 10 + t{v}[1]"),
    (["d{v} = {{1: n, 2: {a}}}", "s{v} = 0", "for k{v} in d{v}:", "    s{v} += k{v} * d{v}[k{v}]"], "s{v}"),
    (["d{v} = {{1: n, 3: {a}}}", "s{v} = 0", "for k{v}, w{v} in d{v}.items():", "    s{v} += k{v} * 10 + w{v}"], "s{v}"),
    (["p{v}, q{v} = n, {a}", "p{v}, q{v} = q{v}, p{v} + q{v}"], "p{v} * 100 + q{v}"),
    (["m{v} = [[n, {a}], [{b}, {c}]]", "m{v}[1][0] += n"], "m{v}[0][1] * 10 + m{v}[1][0]"),
    (["xs{v} = [n, {a}, {b}, {c}]", "ys{v} = xs{v}[1:3]", "ys{v}.append(n)"], "len(ys{v}) * 100 + ys{v}[0] + ys{v}[-1]"),
    (["s{v} = 0", "for i{v}, x{v} in enumerate([n, {a}, {b}]):", "    s{v} += i{v} * x{v}"], "s{v}"),
    (["s{v} = 0", "for x{v}, y{v} in zip([n, {a}], [{b}, {c}]):", "    s{v} += x{v} * y{v}"], "s{v}"),
    (["xs{v} = []", "k{v} = n % 5", "while k{v} > 0:", "    xs{v}.insert(0, k{v})", "    k{v} -= 1"], "len(xs{v}) * 10 + (xs{v}[0] if xs{v} else -1)"),
    (["d{v} = {{1: n, 2: {a}}}", "del d{v}[1]"], "len(d{v}) * 10 + int(1 in d{v}) + d{v}[2]"),
    (["xs{v} = [{b}, n, {a}]", "xs{v}.sort()", "xs{v}.remove(n)"], "xs{v}[0] * 10 + xs{v}[1]"),
    (["xs{v} = [n, {a}]", "ys{v} = list(xs{v})", "ys{v}.append({b})", "xs{v}[0] = {c}"], "xs{v}[0] + ys{v}[0] * 10 + len(ys{v}) * 100"),
    (["xs{v} = [n, {b}, {c}]"], "int(any(x{v} > {a} for x{v} in xs{v})) + 2 * int(all(x{v} > 0 for x{v} in xs{v}))"),
    (["xs{v} = [n, {a}, {b}]", "ys{v} = xs{v}", "ys{v}.append({c})"], "len(xs{v}) * 10 + xs{v}[-1]"),
    (["xs{v} = [n, {a}]", "for i{v} in range(2):", "    xs{v}.append(xs{v}[-1] + xs{v}[-2])"], "xs{v}[-1] * 10 + len(xs{v})"),
    (["d{v} = {{}}", "for x{v} in [n, {a}, n]:", "    d{v}[x{v}] = d{v}.get(x{v}, 0) + 1"], "d{v}[n] * 10 + len(d{v})"),
    (["xs{v} = [n, {a}, {b}]", "t{v} = 0", "for x{v} in xs{v}:", "    if x{v} > {c}:", "        continue", "    t{v} += x{v}"], "t{v}"),
    (["s{v} = {{n, {a}}}", "s{v}.discard(n)", "s{v}.add({b})"], "len(s{v}) * 10 + int(n in s{v})"),
    (["xs{v} = [n, {a}, {b}]", "xs{v}[0], xs{v}[2] = xs{v}[2], xs{v}[0]"], "xs{v}[0] * 100 + xs{v}[2]"),
    (["xs{v} = [n] * 3", "xs{v}[1] = {a}"], "sum(xs{v}) * 10 + xs{v}[1]"),
]


def make_program(rng):
    pool = SNIPPETS + (EXTENDED if os.environ.get("FUZZ_EXT") else []) + (
        EXTENDED2 if os.environ.get("FUZZ_EXT") in {"2", "3", "4"} else []) + (
        EXTENDED3 if os.environ.get("FUZZ_EXT") in {"3", "4"} else []) + (
        EXTENDED4 if os.environ.get("FUZZ_EXT") == "4" else [])
    picks = rng.sample(range(len(pool)), rng.randint(1, 3))
    body, exprs = [], []
    for v, index in enumerate(picks):
        lines, expr = pool[index]
        values = {"v": v, "a": rng.randint(1, 6), "b": rng.randint(-3, 9), "c": rng.randint(0, 8)}
        body += ["    " + line.format(**values) for line in lines]
        exprs.append(expr.format(**values))
    text = HELPERS + (EXT_HELPERS if os.environ.get("FUZZ_EXT") else "") + (
        EXT2_HELPERS if os.environ.get("FUZZ_EXT") in {"2", "3", "4"} else "") + (
        EXT3_HELPERS if os.environ.get("FUZZ_EXT") in {"3", "4"} else "") + "def root(n: int) -> int:\n" + "\n".join(body) + "\n    return " + " + ".join(f"({e})" for e in exprs) + "\n"
    return text, tuple(picks)


def signature(text):
    text = re.sub(r"\b\d+\b", "#", text)
    text = re.sub(r"'[^']*'", "'_'", text)
    return text[:110]


def run_one(text):
    import numpy as np
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_to_c
    from src.compiler.identity_concordance import concordance_report
    ns = {}
    exec(compile(text, "<fuzz>", "exec"), ns)
    inputs = [-3, 0, 2, 5, 9, 12]
    expected = {}
    for n in inputs:
        try:
            expected[n] = ns["root"](n)
        except Exception as error:  # noqa: BLE001
            expected[n] = ("raise", type(error).__name__)
    try:
        module, _, exports = lower_ast_source_to_ssa(text, "root", name="fz", extraction_contract=ROOT + "/extraction_contracts/program_extraction.yaml", progress=lambda _: None)
    except Exception as error:  # noqa: BLE001
        return "LOUD", signature(f"lower {type(error).__name__}: {error}")
    report = concordance_report(module, limit=3)
    findings = re.search(r"(\d+) finding", report)
    finding_count = int(findings.group(1)) if findings else 0
    root = module.functions[exports[0]]
    params = dict(root.metadata["parameter_names"])
    from src.compiler.project_compilation_product import _unexplained_root_argument_ids
    unexplained = _unexplained_root_argument_ids(root)
    if unexplained:
        # The native emission product refuses this root (public inputs that
        # are compiler intermediates); emit_ssa_to_c alone does not check.
        return "LOUD", signature(f"product refuses unexplained root inputs {unexplained}")
    outs = [i.args for b in root.blocks.values() for i in b.instrs if i.op == "Ret"][0]
    art = emit_ssa_to_c(module, exports[0])
    if not art.complete:
        return "LOUD", signature(f"emit {art.shortfalls[0].operation}: {art.shortfalls[0].reason}")
    art.compile(pathlib.Path(tempfile.mkdtemp()) / "native")
    wrong, typed = [], []
    for n in inputs:
        want = expected[n]
        try:
            ex = art.prepare_execution({params["n"]: np.array([n])} if "n" in params else {}).run()
            got = [ex.buffers[v.id].reshape(-1) for v in outs]
        except Exception as error:  # noqa: BLE001
            if isinstance(want, tuple):
                continue
            wrong.append(f"n={n} raised {type(error).__name__} want {want}")
            continue
        if isinstance(want, tuple):
            wrong.append(f"n={n} returned {got[0].tolist()} want {want[1]}")
            continue
        value = got[0][0]
        if float(value) != float(want):
            wrong.append(f"n={n} got {value} want {want}")
        elif got[0].dtype.kind not in "iub":
            typed.append(n)
    if not wrong and "llvm" in os.environ.get("FUZZ_BACKENDS", ""):
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
        )
        llvm = emit_ssa_function_to_llvm(module, exports[0])
        if not llvm.complete:
            return "LLVM-LOUD", signature(f"llvm {llvm.shortfalls[0].operation}: {llvm.shortfalls[0].reason}")
        try:
            llvm = compile_artifact(llvm)
        except Exception as error:  # noqa: BLE001
            return "LLVM-LOUD", signature(f"llvm compile {type(error).__name__}: {error}")
        for n in inputs:
            want = expected[n]
            if isinstance(want, tuple):
                continue
            ex = prepare_artifact_execution(llvm, {params["n"]: np.array([n])} if "n" in params else {}).run()
            value = np.asarray(ex.buffers[outs[0].id]).reshape(-1)[0]
            if float(value) != float(want):
                wrong.append(f"llvm n={n} got {value} want {want}")
    if not wrong and "fortran" in os.environ.get("FUZZ_BACKENDS", ""):
        from src.compiler.ssa_fortran_backend import FortranCoreNative, emit_module
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        try:
            fortran = emit_module(module, progress=lambda _: None)
        except Exception as error:  # noqa: BLE001
            return "FORTRAN-LOUD", signature(f"fortran {type(error).__name__}: {error}")
        if not fortran.complete:
            return "FORTRAN-LOUD", signature(f"fortran {fortran.shortfalls[0].format()}")
        try:
            built = compile_fortran_module_c_shell(
                fortran, {}, pathlib.Path(tempfile.mkdtemp()), library=True,
                entrypoint=exports[0], name=exports[0][:48],
                # FUZZ_FORTRAN_STANDALONE=0 links libgfortran dynamically
                # (a Linux libgfortran.a is not position independent).
                standalone=os.environ.get("FUZZ_FORTRAN_STANDALONE", "1") != "0",
            )
            record = next(e for e in fortran.api.entry_points if str(e.name) == exports[0])
            native = FortranCoreNative(built.executable_path, record, fortran.api)
        except Exception as error:  # noqa: BLE001
            return "FORTRAN-LOUD", signature(f"fortran compile {type(error).__name__}: {error}")
        for n in inputs:
            want = expected[n]
            if isinstance(want, tuple):
                continue
            try:
                ex = native.prepare_execution({params["n"]: np.array([n])} if "n" in params else {}).run()
            except Exception as error:  # noqa: BLE001
                return "FORTRAN-LOUD", signature(f"fortran run {type(error).__name__}: {error}")
            value = np.asarray(ex.buffers[outs[0].id]).reshape(-1)[0]
            if float(value) != float(want):
                wrong.append(f"fortran n={n} got {value} want {want}")
    if wrong:
        return "WRONG", "; ".join(wrong[:2]) + f" | findings={finding_count}"
    if finding_count:
        return "FINDING", signature(report.splitlines()[1] if len(report.splitlines()) > 1 else report)
    if typed:
        return "TYPE", "int returned as float"
    return "PASS", ""


def isolated(text, limit=300):
    """run_one in a forked child: a native segfault or hang is a result."""
    import multiprocessing as mp
    context = mp.get_context("fork")
    parent, child = context.Pipe(duplex=False)

    def target():
        try:
            child.send(run_one(text))
        except Exception as error:  # noqa: BLE001
            child.send(("CRASH", signature(f"{type(error).__name__}: {error}")))

    process = context.Process(target=target)
    process.start()
    process.join(limit)
    if process.is_alive():
        process.kill()
        process.join()
        return "HANG", f"exceeded {limit}s"
    if parent.poll():
        return parent.recv()
    return "SEGV", f"native exit code {process.exitcode}"


def main():
    out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(sys.argv[2]))
    with open(out / "results.jsonl", "a") as sink:
        for index in range(int(sys.argv[3])):
            text, picks = make_program(rng)
            name = f"p{sys.argv[2]}_{index}"
            (out / f"{name}.py").write_text(text)
            status, detail = isolated(text)
            sink.write(json.dumps({"name": name, "picks": picks, "status": status, "detail": detail}) + "\n")
            sink.flush()
            print(name, status, detail[:120], flush=True)


if __name__ == "__main__":
    main()

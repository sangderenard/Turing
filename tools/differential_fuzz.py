"""Differential fuzzer: random programs from composable snippets, lowered to
C and compared with Python; failures clustered by signature.

usage: python tools/differential_fuzz.py OUTDIR SEED COUNT
(FUZZ_EXT=1/2/3 widen the grammar: extended builtins, then more control
and sequences, then helpers that mutate and read a caller's list.)
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
    # an unrolled literal loop beside an in-place sort (corpus c244, plan C15)
    (["t{v} = 0", "for x{v} in [{a}, {b}]:", "    t{v} += x{v} * n", "xs{v} = [{c}, n, {a}]", "xs{v}.sort()"], "xs{v}[0] * 10 + xs{v}[2] + t{v}"),
    (["ys{v} = [x{v} * n for x{v} in [{a}, {b}]]", "xs{v} = [{c}, n]", "xs{v}.sort(reverse=True)"], "xs{v}[0] * 10 + sum(ys{v})"),
    # a display rebound inside an unrolled loop (corpus c245, plan C2)
    (["xs{v} = [{a}]", "for i{v} in range(3):", "    xs{v} = [i{v} + n, {b}, i{v}]"], "len(xs{v}) * 100 + xs{v}[0] * 10 + xs{v}[1]"),
    # a display rebound inside a retained loop that reads it (corpus c246)
    (["xs{v} = [{a}]", "t{v} = 0", "for i{v} in range(4):", "    t{v} += len(xs{v}) * 10 + xs{v}[-1]", "    xs{v} = [i{v}, n, {b}]"], "t{v} * 10 + len(xs{v})"),
    # chained comparisons whose middle is a builtin conversion or measure,
    # one behind a guarding ``and`` (lanes plan A0)
    (["f{v} = n * 0.5 + {a}"], "1 if 0 <= int(f{v}) <= {b} + 3 else 2"),
    (["xs{v} = [{a}, n, {b}]"], "(1 if 1 < len(xs{v}) <= n else 0) + (2 if n != 0 and 0 < abs({a} // n) < 3 else 0)"),
]
EXT_HELPERS = """def pair(x: int, y: int) -> tuple[int, int]:
    return x // y, x % y


def first_over(x: int, y: int) -> int:
    for i in range(10):
        if i * y > x:
            return i
    return -1


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


def make_program(rng):
    pool = SNIPPETS + (EXTENDED if os.environ.get("FUZZ_EXT") else []) + (
        EXTENDED2 if os.environ.get("FUZZ_EXT") in {"2", "3"} else []) + (
        EXTENDED3 if os.environ.get("FUZZ_EXT") == "3" else [])
    picks = rng.sample(range(len(pool)), rng.randint(1, 3))
    body, exprs = [], []
    for v, index in enumerate(picks):
        lines, expr = pool[index]
        values = {"v": v, "a": rng.randint(1, 6), "b": rng.randint(-3, 9), "c": rng.randint(0, 8)}
        body += ["    " + line.format(**values) for line in lines]
        exprs.append(expr.format(**values))
    text = HELPERS + (EXT_HELPERS if os.environ.get("FUZZ_EXT") else "") + (
        EXT2_HELPERS if os.environ.get("FUZZ_EXT") in {"2", "3"} else "") + (
        EXT3_HELPERS if os.environ.get("FUZZ_EXT") == "3" else "") + "def root(n: int) -> int:\n" + "\n".join(body) + "\n    return " + " + ".join(f"({e})" for e in exprs) + "\n"
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

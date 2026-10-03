"""The orbital transfer show-off set, compiled as written through the sanctioned lane.

Program: ``Orbit.stable_orbit_transfer_solution(Orbit.symbolic_orbit('1'),
Orbit.symbolic_orbit('2'))`` (``src/transmogrifier/orbital.py`` ->
``OrbitalTransfer.symbolic_transfer_spline`` in ``orbital_transfer.py``) --
the same call the graph_express2 suite builds into a ProcessGraph.  Two
centers ``mu_1, c_1`` / ``mu_2, c_2`` (``c_i`` are 3x1 MatrixSymbols), position
``r(s)`` as undefined Functions of arc length ``s``, the equation of motion
``d2r/ds2 = F_grav1 + F_grav2 + F_extra``, the total energy, the force-cost
``Integral(sqrt(F.F), (s, 0, L))`` and the two boundary conditions.

NOTHING is substituted.  The physics module is the truth; the compiler must
lower it as written.  The only thing this probe adds is what the sanctioned
entry's API requires of any program: each output is a named scalar Symbol.
Every Equality ``lhs = rhs`` of the set contributes its lhs and its rhs
components as named outputs (``<entry>_lhs_<k>``, ``<entry>_rhs_<k>``); a
scalar expression is one output named after its dict key.  One law per dict
entry, so a construct that fails is attributed to its own entry.  The raw
Equalities are also offered to ``compile_sympy_equations`` unnamed, and that
refusal is recorded too.

Route per law: ``compile_sympy_equations`` -> ``piece_from_law``
(``symbolic_abstract_tensor_source`` -> ``lower_ast_source_to_ssa`` ->
LLVM emission + compile) -> C emission (``emit_ssa_module_to_c``) + compile
-> native run against a sympy reference.  The reference evaluates the very
outputs the compiler declared (``declared_symbolic_outputs``) with concrete
functions chosen for ``r(s)`` and ``F(s)`` (reference side only), each
declared MatrixSymbol element input read from its own column
(``matrix_element_inputs``), ``doit()`` and ``lambdify``.  A law that reaches the
book prints the process-graph / book / emission-unit counts and one unit's
chain back to its ``source_span`` through ``book.edges_into``.

The probe exits nonzero while any law fails to lower.  Each failure prints
the construct, the stage, the raising frame and the message verbatim.

Work list: ``WORK_ITEMS`` below (the user's order, 2026-10-02), and
``LAW_BLOCKERS`` -- the work items each law needs before it can pass.
``tests/test_orbital_transfer_compile.py`` marks exactly those laws
``xfail(strict=True)`` with reasons built from ``WORK_ITEMS``, and asserts
that the union of the blockers IS the work list, so a fix that lands flips
its laws to XPASS and forces both tables to be edited.  Item 1 (matrices
and complicated lhs) is resolved: a matrix-valued Equality declares one
output per component and a non-name lhs its residual
(``symbolic_equation_compiler.declared_symbolic_outputs``); a MatrixElement
of a MatrixSymbol is an input column (``symbolic_process_graph``).  Item 3
(externals) is resolved: each undefined Function is a declared external
the compiled program calls through its slot table, filled at load from
``host_externals`` (``src/compiler/external_functions.py``).
History and verbatim messages:
``docs/concordance_census/CONTINUATION_orbital_probe.md``,
``CONTINUATION_orbital_step1_matrices.md`` and
``CONTINUATION_orbital_item3_externals.md``.

    python -u tools/compiler_probes/probe_orbital_transfer.py
"""
from __future__ import annotations

import collections
import pathlib
import sys
import traceback
import warnings

import numpy as np
import sympy as sp

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "compiler_probes"))

BUILD = REPO / "build" / "orbital_transfer"
BATCH = 4
TOLERANCE = 1e-12

# The user's work items for this set (2026-10-02), the ones still open.
# All resolved (2026-10-03): 1 matrices and complicated lhs; 2 Greek names
# (the mu_i laws pass as written, no sanitation needed); 3 external
# functions as runtime-slot externals; 4 derivatives of externals as their
# declared derivative externals and the Integral as its declared quadrature
# lowered natively (``CONTINUATION_orbital_item3_externals.md``).
WORK_ITEMS: dict[int, str] = {}

# law -> the open work items it needs before it can pass.  A law not listed
# must pass; every law of the set passes.
LAW_BLOCKERS: dict[str, tuple[int, ...]] = {}


def blocker_reason(items) -> str:
    """The xfail reason for a law blocked by ``items``, from ``WORK_ITEMS``."""
    return " | ".join(f"work item {item}: {WORK_ITEMS[item]}" for item in items)


# -- the program -----------------------------------------------------------

def build_program() -> dict:
    """The set exactly as the physics module builds it (graph_express2's call)."""
    from src.transmogrifier.orbital import Orbit

    return Orbit.stable_orbit_transfer_solution(
        Orbit.symbolic_orbit("1"), Orbit.symbolic_orbit("2"))


def _components(value) -> list:
    if isinstance(value, (sp.MatrixBase, sp.MatrixExpr)):
        rows, cols = value.shape
        return [value[i, j] for i in range(rows) for j in range(cols)]
    return [value]


def named_laws(program: dict) -> dict[str, tuple[sp.Equality, ...]]:
    """One law per dict entry; outputs named, expressions untouched."""
    laws: dict[str, tuple[sp.Equality, ...]] = {}
    for key, value in program.items():
        if isinstance(value, dict):
            continue      # force_components: the EOM's own terms, already in it
        if isinstance(value, sp.Equality):
            # one law per side, so a construct on one side (the lhs
            # Derivative) does not mask the other side's (the rhs
            # MatrixElement)
            for side, part in (("lhs", value.lhs), ("rhs", value.rhs)):
                laws[f"{key}_{side}"] = tuple(
                    sp.Eq(sp.Symbol(f"{key}_{side}_{index}"), element, evaluate=False)
                    for index, element in enumerate(_components(part), start=1))
        else:
            laws[key] = (sp.Eq(sp.Symbol(key), value, evaluate=False),)
    return laws


def benchmark_laws(program: dict) -> dict[str, tuple[sp.Equality, ...]]:
    """The raw Equalities of the set, unnamed as the physics module returns
    them (``orbital_transfer_raw``), then one law per entry and side."""
    raw = tuple(value for value in program.values() if isinstance(value, sp.Equality))
    return {"orbital_transfer_raw": raw, **named_laws(program)}


# -- failure capture -------------------------------------------------------

Failure = collections.namedtuple("Failure", "law stage construct error frame")


def _raising_frame(error: BaseException) -> str:
    frames = traceback.extract_tb(error.__traceback__)
    inside = [frame for frame in frames if "src" in pathlib.Path(frame.filename).parts]
    frame = (inside or frames)[-1]
    try:
        where = pathlib.Path(frame.filename).resolve().relative_to(REPO).as_posix()
    except ValueError:
        where = frame.filename
    return f"{where}:{frame.lineno} {frame.name}"


def _construct_of(equations) -> str:
    from sympy.core.function import AppliedUndef
    from sympy.matrices.expressions.matexpr import MatrixElement

    kinds = set()
    for equation in equations:
        for side in (equation.lhs, equation.rhs) if not isinstance(equation.lhs, sp.Symbol) \
                else (equation.rhs,):
            for node in sp.preorder_traversal(side):
                if isinstance(node, sp.Derivative):
                    kinds.add("Derivative")
                elif isinstance(node, sp.Integral):
                    kinds.add("Integral")
                elif isinstance(node, MatrixElement):
                    kinds.add("MatrixElement")
                elif isinstance(node, (sp.MatrixBase, sp.MatrixExpr)):
                    kinds.add("Matrix")
                elif isinstance(node, AppliedUndef):
                    kinds.add("undefined Function")
                elif isinstance(node, sp.Symbol) and not node.name.isascii():
                    kinds.add("non-ASCII symbol name")
    return ", ".join(sorted(kinds)) or "scalar"


def _record(failures, law, stage, equations, error):
    failure = Failure(law, stage, _construct_of(equations),
                      f"{type(error).__name__}: {error}", _raising_frame(error))
    failures.append(failure)
    print(f"FAIL {law:24} stage={stage}")
    print(f"       constructs: {failure.construct}")
    print(f"       raised at:  {failure.frame}")
    print(f"       {failure.error[:1600]}")
    return failure


# -- reference -------------------------------------------------------------

def reference_bindings(program: dict):
    """Concrete functions for the reference evaluation only."""
    s = sp.Symbol("s", real=True)
    return {
        sp.Function("r1"): sp.Lambda(s, 3 + sp.cos(s / 5)),
        sp.Function("r2"): sp.Lambda(s, 2 * sp.sin(s / 7)),
        sp.Function("r3"): sp.Lambda(s, s / 11),
        sp.Function("F1"): sp.Lambda(s, sp.Rational(1, 10) + s / 100),
        sp.Function("F2"): sp.Lambda(s, sp.Rational(-1, 20)),
        sp.Function("F3"): sp.Lambda(s, s**2 / 1000),
    }


def external_derivatives() -> dict:
    """The host's declared derivative of each position external: velocity
    ``v_i = dr_i/ds`` and acceleration ``a_i = dv_i/ds``.  The set takes
    Derivatives of ``r_i`` only; the craft supplies its velocity and
    acceleration as externals of their own."""
    out = {}
    for index in (1, 2, 3):
        r, v, a = (sp.Function(f"{kind}{index}") for kind in ("r", "v", "a"))
        out[r] = v
        out[v] = a
    return out


def host_externals(concrete) -> dict:
    """The host's runtime implementations of the set's externals.

    The same concrete functions the reference uses, as numpy callables: what
    the craft's r()/F() seam supplies at runtime, bound into the compiled
    program's slot table at load (``bind_external_slots``).  Each declared
    derivative external is the derivative of the host's own function (the
    host knows its velocity)."""
    bodies = {str(function.__name__): body for function, body in concrete.items()}
    for function, derivative in external_derivatives().items():
        body = bodies.get(str(function.__name__))
        if body is not None:
            (variable,) = body.variables
            bodies[str(derivative.__name__)] = sp.Lambda(variable, sp.diff(body.expr, variable))
    out = {}
    for name, body in bodies.items():
        function = sp.lambdify(body.variables, body.expr, modules="numpy")
        out[name] = (lambda f: lambda *args: np.broadcast_to(
            np.asarray(f(*args), dtype=np.float64), np.shape(args[0])))(function)
    return out


def reference_columns(names, batch=BATCH) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261002)
    return {name: rng.uniform(0.5, 2.0, batch) for name in names}


def declared_quadrature(integral):
    """``integral`` as the compiler's declared rule: weights times the
    integrand at the nodes mapped onto each finite axis."""
    from src.compiler.symbolic_process_graph import _gauss_legendre_rule

    rule = _gauss_legendre_rule()
    body = integral.function
    for variable, lower, upper in integral.limits:
        half, mid = (upper - lower) / 2, (upper + lower) / 2
        body = sp.Add(*(weight * half * body.xreplace({variable: mid + half * node})
                        for node, weight in rule))
    return body


def reference_values(compilation, law, columns, concrete) -> dict[str, np.ndarray]:
    """Each output the compiler declared, evaluated by sympy.

    The outputs are read from the compiler's own declaration
    (``declared_symbolic_outputs``: assignment rhs, or residual lhs - rhs),
    and each MatrixSymbol element it declared as an input column is read
    from that column (``matrix_element_inputs``)."""
    from src.compiler.symbolic_equation_compiler import declared_symbolic_outputs

    elements = {
        sp.MatrixSymbol(matrix, *shape)[tuple(index)]: sp.Symbol(column)
        for column, matrix, shape, index
        in compilation.function.metadata["matrix_element_inputs"]}
    out = {}
    for row in declared_symbolic_outputs(compilation.equations, law):
        expr = row.expression.xreplace(elements)
        for function, body in concrete.items():
            expr = expr.replace(function, body)
        expr = expr.doit()
        # An Integral SymPy cannot integrate in closed form is compiled as its
        # DECLARED lowering, the Gauss-Legendre rule of
        # ``symbolic_process_graph.lower_integral_declaration``; the reference
        # applies that same rule, so the check is of the lowering, not of the
        # rule's own truncation error.
        expr = expr.replace(lambda node: isinstance(node, sp.Integral), declared_quadrature)
        symbols = sorted(expr.free_symbols, key=lambda symbol: symbol.name)
        missing = [str(symbol) for symbol in symbols if str(symbol) not in columns]
        if missing:
            raise KeyError(f"reference for {row.name}: no column for {missing}")
        function = sp.lambdify(symbols, expr, modules="numpy")
        value = function(*(columns[str(symbol)] for symbol in symbols))
        out[row.name] = np.broadcast_to(np.asarray(value, np.float64), (BATCH,)).copy()
    return out


# -- book readouts ---------------------------------------------------------

def book_report(module, entry, graphs) -> int:
    import probe_emission_chain as chain
    from src.compiler.concordance_declarations import EMISSION_UNIT, Backend, EmittedUnit

    book = module.metadata["identity_book"]
    cells = sum(len(page.cells) for page in book.pages.values())
    units = chain.rows_of(book, EMISSION_UNIT)
    unsourced = collections.Counter(
        reason.name for page, _row, reason, _stage in book.unsourced_rows()
        if getattr(page, "name", page) == EMISSION_UNIT.name)
    nodes = len(graphs[-1].G) if graphs else 0
    print(f"       process graph nodes={nodes}  book pages={len(book.pages)} cells={cells}  "
          f"emission units={len(units)} sourced={len(units) - sum(unsourced.values())} "
          f"unsourced={sum(unsourced.values())} {dict(unsourced)}")
    for backend in (Backend.C_MODULE, Backend.LLVM_MODULE):
        rows = [row for row in units if row[1] is backend and row[0] == entry]
        statements = [book.latest_ref(EMISSION_UNIT, row) for row in rows
                      if getattr(chain.fact_of(book, book.latest_ref(EMISSION_UNIT, row)),
                                 "kind", None) is not None
                      and chain.fact_of(book, book.latest_ref(EMISSION_UNIT, row)).kind.name
                      == "STATEMENT"]
        if not statements:
            print(f"       [{backend.name}] no STATEMENT unit in {entry}")
            continue
        token = statements[-1]
        fact = chain.fact_of(book, token)
        print(f"       [{backend.name}] unit {fact.spelling!r} in {entry}: {fact.text.strip()!r}")
        pages, hops = chain.walk(book, token, show=True)
        spans = pages.get("source_span", ())
        print(f"       [{backend.name}] chain hops={hops} source_span rows={len(spans)}")
        if not spans:
            return 1
    return 0


# -- one law ---------------------------------------------------------------

def run_law(law, equations, failures, concrete, sink=None):
    """Returns the piece (lowered and emitted) or None."""
    from src.compiler.external_functions import bind_external_slots
    from src.compiler.native_package import piece_from_law
    from src.compiler.symbolic_equation_compiler import compile_sympy_equations
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c

    try:
        compilation = compile_sympy_equations(
            list(equations), name=law, external_derivatives=external_derivatives())
    except Exception as error:  # noqa: BLE001 -- recorded verbatim
        _record(failures, law, "compile_sympy_equations", equations, error)
        return None
    arguments = tuple(compilation.function.metadata["argument_names"])
    print(f"ok   {law:24} compile_sympy_equations: arguments={arguments} "
          f"instructions={len(compilation.instructions)} "
          f"process graph nodes={len(compilation.process_graph.G)}")
    directory = BUILD / law
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            piece = piece_from_law(compilation, law, BATCH, directory=directory,
                                   resolved_process_graph_sink=sink)
    except Exception as error:  # noqa: BLE001
        _record(failures, law, "piece_from_law", equations, error)
        return None
    print(f"ok   {law:24} piece_from_law: entry={piece.entry} (LLVM emitted, compiled)")
    try:
        c_artifact = emit_ssa_module_to_c(piece.module, piece.entry)
        if not c_artifact.complete:
            raise RuntimeError(f"C emission shortfalls: {c_artifact.shortfalls[:3]}")
        c_artifact = c_artifact.compile(directory / "c")
    except Exception as error:  # noqa: BLE001
        _record(failures, law, "emit_ssa_module_to_c", equations, error)
        return piece
    columns = reference_columns(arguments)
    try:
        expected = reference_values(compilation, law, columns, concrete)
        externals = host_externals(concrete)
        if piece.artifact.external_slots:
            bind_external_slots(piece.artifact, externals)
        if c_artifact.external_slots:
            bind_external_slots(c_artifact, externals)
        llvm = dict(zip(piece.output_names, piece(*(columns[name] for name in arguments))))
        execution = c_artifact.prepare_execution(
            {value_id: columns[name] for name, value_id in zip(arguments, piece.argument_ids)})
        execution.run()
        worst = 0.0
        for name, want in expected.items():
            for lane, got in (("llvm", llvm[name]),
                              ("c", execution.buffers[piece.output_ids[name]]
                               if name in piece.output_ids
                               else np.full(BATCH, piece.constant_outputs[name]))):
                error = float(np.max(np.abs(np.asarray(got) - want) / np.maximum(np.abs(want), 1e-300)))
                worst = max(worst, error)
        if worst > TOLERANCE:
            raise AssertionError(f"native vs sympy max relative error {worst:.3e} > {TOLERANCE}")
        print(f"ok   {law:24} native (C, LLVM) vs sympy reference: max relative error {worst:.3e}")
    except Exception as error:  # noqa: BLE001
        _record(failures, law, "native_vs_reference", equations, error)
    return piece


def lower_for_viewer(process_graph_sink):
    """``view_identity_concordance --probe orbital``: the first law of the set
    that reaches the book, lowered exactly as this probe lowers it.  Returns
    (module, root symbol); refuses with the recorded failures when none does."""
    program = build_program()
    concrete = reference_bindings(program)
    failures: list = []
    for law, equations in benchmark_laws(program).items():
        piece = run_law(law, equations, failures, concrete,
                        sink=process_graph_sink)
        if piece is not None:
            return piece.module, piece.entry
    raise SystemExit("--probe orbital: no law of the orbital transfer set lowers yet:\n" + "\n".join(
        f"  {f.law}: {f.stage}: {f.error[:200]}" for f in failures))


def main() -> int:
    # The set's symbol names are not all cp1252 (GREEK SMALL LETTER MU).
    sys.stdout.reconfigure(errors="backslashreplace")
    program = build_program()
    concrete = reference_bindings(program)
    laws = benchmark_laws(program)
    failures: list = []
    print("program entries:", ", ".join(program))
    print("laws:", {law: [str(eq.lhs) for eq in eqs] for law, eqs in laws.items()})

    reached = 0
    failed_laws: set[str] = set()
    for law, equations in laws.items():
        graphs: list = []
        before = len(failures)
        piece = run_law(law, equations, failures, concrete, sink=graphs.append)
        if piece is not None:
            reached += 1
            if book_report(piece.module, piece.entry, graphs):
                print(f"FAIL {law:24} chain did not reach a source_span row")
                failures.append(Failure(law, "chain", "", "no source_span", ""))
        if len(failures) > before:
            failed_laws.add(law)

    print()
    print(f"laws reaching the book: {reached}/{len(laws)}; failures: {len(failures)}")
    by_stage = collections.Counter((f.stage, f.frame.split(' ')[-1]) for f in failures)
    for (stage, where), count in sorted(by_stage.items()):
        print(f"  {count} x {stage} raised in {where}")
    print()
    print("work list (open items; items 1-4 resolved):")
    for item, text in WORK_ITEMS.items():
        blocked = [law for law, items in LAW_BLOCKERS.items() if item in items]
        print(f"  {item}. {text}")
        print(f"     laws: {', '.join(blocked)}")
    drift = 0
    for law in laws:
        expected_fail = law in LAW_BLOCKERS
        failed = law in failed_laws
        verdict = "pass"
        if failed and expected_fail:
            verdict = "fail, expected: work items " + ", ".join(
                str(item) for item in LAW_BLOCKERS[law])
        if failed != expected_fail:
            drift += 1
            verdict = ("UNEXPECTED FAIL" if failed
                       else "XPASS -- remove its LAW_BLOCKERS entry")
        print(f"  {law:24} {verdict}")
    print(f"work-list drift: {drift}")
    return 1 if failures or drift else 0


if __name__ == "__main__":
    sys.exit(main())

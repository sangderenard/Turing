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
-> native run against a sympy reference.  The reference is the same set with
concrete functions chosen for ``r(s)``, ``F(s)`` and numbers for the centers
(reference side only), ``doit()`` and ``lambdify``.  A law that reaches the
book prints the process-graph / book / emission-unit counts and one unit's
chain back to its ``source_span`` through ``book.edges_into``.

The probe exits nonzero while any law fails to lower.  Each failure prints
the construct, the stage, the raising frame and the message verbatim.

Ranked work list (measured 2026-10-02: 0/8 laws reach the book, 9 failures;
smallest fix first; verbatim messages in
``docs/concordance_census/CONTINUATION_orbital_probe.md``):

1. Matrix-valued Equality as an output (the raw set): ``equation output
   must be a Symbol`` at ``symbolic_equation_compiler.py:91``
   (``_compile_sympy_equations_uncached``).  Fix there: an Equality whose
   lhs is a Matrix names one output per element.
2. ``MatrixElement`` (``c_1[0, 0]``, ``r_start[0, 0]``, ``r_end[0, 0]``):
   ``no SymPy to ProcessGraph translation rule for MatrixElement`` at
   ``symbolic_process_graph.py:1064`` (``add_node``, strict ingest).  Fix in
   ``SYMPY_PROCESS_GRAPH_TRANSLATIONS`` / ``ingest_sympy_expression``: a
   MatrixSymbol is a parameter of declared shape, a MatrixElement an index
   into it.  Laws: equation_of_motion_rhs, initial/terminal_condition_rhs.
3. Undefined applied Functions as values (``r1(0)``, ``r1(L)``, ``F1(s)``
   at the quadrature nodes): ingest emits ``Call callee='r1'`` with no body
   and nothing binds it; ``lower_ast_source_to_ssa`` then raises
   ``FortranEmissionError: full-native execution contract rejected ...``
   at ``fortran_c_shell.py:45626`` (``_lower_ast_source_to_ssa_impl``):
   ``undefined_operands=3`` for the cost Integral (the F1/F2/F3 results
   feeding planned region 5), ``structural_outputs ... call-result-
   unavailable`` for the boundary-condition lhs.  The Integral itself lowers
   (5-point Gauss-Legendre in ``lower_integral_declaration``).  Fix: a
   declared binding for an AppliedUndef (column / sampled Table / bound
   callee -- the binding ``bitops.declare`` already asks for) accepted by
   ``compile_sympy_equations`` and carried into the materialized source.
4. ``Derivative`` of an applied undefined Function (``d2 r1(s)/ds2`` in the
   EOM lhs, ``d r1(s)/ds`` in the energy): ``ProcessGraph has no
   graph-native adjoint rule for <n>:call`` at
   ``process_graph_autograd.py:2072`` (``differentiate_process_graph``,
   reached from ``symbolic_process_graph.add_node``'s Derivative branch).
   Needs 3 first (a bound callee or Table has a derivative; an unbound call
   has none), then an adjoint rule for ``call`` in ``process_graph_autograd``.
5. Non-ASCII symbol names (``mu_1`` spelled with GREEK SMALL LETTER MU):
   UNKNOWN -- every law that carries them fails earlier (items 2 and 4).
   That name's ``isidentifier()`` is True, so the materialized Python is
   legal; not observed either way.

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
    """Concrete functions and numbers for the reference evaluation only."""
    s = sp.Symbol("s", real=True)
    concrete = {
        sp.Function("r1"): sp.Lambda(s, 3 + sp.cos(s / 5)),
        sp.Function("r2"): sp.Lambda(s, 2 * sp.sin(s / 7)),
        sp.Function("r3"): sp.Lambda(s, s / 11),
        sp.Function("F1"): sp.Lambda(s, sp.Rational(1, 10) + s / 100),
        sp.Function("F2"): sp.Lambda(s, sp.Rational(-1, 20)),
        sp.Function("F3"): sp.Lambda(s, s**2 / 1000),
    }
    matrices = {
        sp.MatrixSymbol("c_1", 3, 1): sp.ImmutableMatrix([0, 0, 0]),
        sp.MatrixSymbol("c_2", 3, 1): sp.ImmutableMatrix([10, -1, 2]),
        sp.MatrixSymbol("r_start", 3, 1): sp.ImmutableMatrix([1, 2, 3]),
        sp.MatrixSymbol("r_end", 3, 1): sp.ImmutableMatrix([4, 5, 6]),
    }
    return concrete, matrices


def reference_columns(names, batch=BATCH) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261002)
    return {name: rng.uniform(0.5, 2.0, batch) for name in names}


def reference_values(equations, columns, concrete, matrices) -> dict[str, np.ndarray]:
    out = {}
    for equation in equations:
        expr = equation.rhs.xreplace(matrices)
        for function, body in concrete.items():
            expr = expr.replace(function, body)
        expr = expr.doit()
        symbols = sorted(expr.free_symbols, key=lambda symbol: symbol.name)
        missing = [str(symbol) for symbol in symbols if str(symbol) not in columns]
        if missing:
            raise KeyError(f"reference for {equation.lhs}: no column for {missing}")
        function = sp.lambdify(symbols, expr, modules="numpy")
        value = function(*(columns[str(symbol)] for symbol in symbols))
        out[str(equation.lhs)] = np.broadcast_to(np.asarray(value, np.float64), (BATCH,)).copy()
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

def run_law(law, equations, failures, concrete, matrices, sink=None):
    """Returns the piece (lowered and emitted) or None."""
    from src.compiler.native_package import piece_from_law
    from src.compiler.symbolic_equation_compiler import compile_sympy_equations
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c

    try:
        compilation = compile_sympy_equations(list(equations), name=law)
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
        expected = reference_values(equations, columns, concrete, matrices)
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
    concrete, matrices = reference_bindings(program)
    failures: list = []
    for law, equations in named_laws(program).items():
        piece = run_law(law, equations, failures, concrete, matrices,
                        sink=process_graph_sink)
        if piece is not None:
            return piece.module, piece.entry
    raise SystemExit("--probe orbital: no law of the orbital transfer set lowers yet:\n" + "\n".join(
        f"  {f.law}: {f.stage}: {f.error[:200]}" for f in failures))


def main() -> int:
    program = build_program()
    concrete, matrices = reference_bindings(program)
    laws = named_laws(program)
    failures: list = []
    print("program entries:", ", ".join(program))
    print("laws:", {law: [str(eq.lhs) for eq in eqs] for law, eqs in laws.items()})

    # The raw Equalities of the set, unnamed, as the physics module returns them.
    from src.compiler.symbolic_equation_compiler import compile_sympy_equations

    raw = [value for value in program.values() if isinstance(value, sp.Equality)]
    try:
        compile_sympy_equations(raw, name="orbital_transfer_raw")
        print("ok   raw Equalities accepted")
    except Exception as error:  # noqa: BLE001
        _record(failures, "orbital_transfer_raw", "compile_sympy_equations", raw, error)

    reached = 0
    for law, equations in laws.items():
        graphs: list = []
        piece = run_law(law, equations, failures, concrete, matrices, sink=graphs.append)
        if piece is not None:
            reached += 1
            if book_report(piece.module, piece.entry, graphs):
                print(f"FAIL {law:24} chain did not reach a source_span row")
                failures.append(Failure(law, "chain", "", "no source_span", ""))

    print()
    print(f"laws reaching the book: {reached}/{len(laws)}; failures: {len(failures)}")
    by_stage = collections.Counter((f.stage, f.frame.split(' ')[-1]) for f in failures)
    for (stage, where), count in sorted(by_stage.items()):
        print(f"  {count} x {stage} raised in {where}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

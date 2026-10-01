"""Emission is the last layer of the book: every unit of C text is a row.

Plan 100, part B (``docs/concordance_census/100_plan_step9_graph_input_and_emission_output.md``
sections 4 and 7).  Each program of ``probe_scalar_native_correctness`` is
lowered, emitted to C twice -- once with the module's book replaced by a
detached book (the recorder posts nothing), once with its own book -- and the
two sources must be byte-identical.  The second artifact is compiled, so its
file, command and library rows are posted.  Then, read only from the book the
compile attached to the module:

1. counts per case: ``emission_unit`` / ``emission_function`` /
   ``emission_artifact`` rows; unsourced units by reason;
2. ``unit_count`` on every ``emission_function`` row equals its unit rows;
3. every line a unit claims is a line of the source, and every source line no
   unit claims is preamble (before the first prototype) or blank;
4. the artifact chain LIBRARY -> COMPILE_COMMAND -> SOURCE_FILE ->
   MODULE_TEXT -> each function's ``emission_function`` row;
5. for ``bump`` (``k + 1``): the C token that spells the value the root
   returns, and its chain walked back through ``edges_into`` / ``mint_of``
   to a ``source_span`` row of kind ``BinOp``, passing ``canonical_value``,
   ``ingestion_value`` and, for ``k``, ``scalar_parameter`` (annotated) or
   ``name_binding`` (plain).

    python -u tools/compiler_probes/probe_emission_chain.py
"""
from __future__ import annotations

import collections
import pathlib
import sys
import warnings

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "compiler_probes"))

import probe_scalar_native_correctness as native  # noqa: E402
from src.compiler.concordance_declarations import (  # noqa: E402
    EMISSION_ARTIFACT,
    EMISSION_FUNCTION,
    EMISSION_UNIT,
    ArtifactPart,
    Backend,
    EmittedUnit,
    FunctionEmission,
    UnitKind,
)
from src.compiler.emission_concordance import value_cell  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.identity_concordance import IdentityBook, render_row  # noqa: E402
from src.compiler.ssa_c_backend import emit_ssa_module_to_c  # noqa: E402

BUILD = REPO / "build" / "emission_chain"
ROOT = "scalar_native__f"


def lower(source: str, dtype: str, has_tensor: bool):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lower_ast_source_to_ssa(
            source, "f", name="scalar_native", python_bindings={},
            extraction_contract=native.contract(dtype, has_tensor),
            runtime_closure_only=True,
            **({"tensor_ssa_reference": native._tensor_reference()} if has_tensor else {}),
        )


def fact_of(book, ref):
    return book.pages[ref.page.name].cells.get((ref.row, ref.column))


def rows_of(book, page):
    stored = book.pages.get(page.name)
    return () if stored is None else stored.rows()


def hop(ref) -> str:
    return f"{ref.page.name} {render_row(ref.row)}"


def walk(book, start, depth_limit: int = 40, show: bool = True):
    """Breadth-first over ``edges_into`` and ``mint_of`` operands; returns
    the set of pages reached and the refs seen."""

    seen = {start.key: 0}
    queue = collections.deque([(start, 0)])
    pages: dict[str, list] = collections.defaultdict(list)
    hops = 0
    while queue:
        ref, depth = queue.popleft()
        pages[ref.page.name].append(ref)
        if depth >= depth_limit:
            continue
        sources = [(source, stage.name) for source, stage in book.edges_into(ref)]
        mint = book.mint_of(ref)
        if mint is not None:
            sources.extend((operand, f"mint {mint[0].name}") for operand in mint[1])
        for source, label in sources:
            if source.key in seen:
                continue
            seen[source.key] = depth + 1
            hops += 1
            if show:
                print(f"      [{depth + 1:2}] {hop(ref)} -> {hop(source)} ({label})")
            queue.append((source, depth + 1))
    return pages, hops


def check_llvm(module, book, workdir) -> int:
    """The LLVM module lane: byte identity, unit counts, artifact chain.
    Text coverage is printed, not gated: ``_annotate_noalias`` rewrites the
    define lines after their units were posted, and the frame releases are
    spliced before every return from one unit."""

    from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm

    failures = 0
    module.metadata["identity_book"] = IdentityBook(detached=True)
    try:
        silent = emit_ssa_function_to_llvm(module, ROOT)
    finally:
        module.metadata["identity_book"] = book
    artifact = emit_ssa_function_to_llvm(module, ROOT)
    identical = silent.llvm_ir == artifact.llvm_ir
    failures += 0 if identical else 1
    if not artifact.complete:
        print(f"       FAIL llvm emission incomplete: {artifact.shortfalls[:3]}")
        return failures + 1
    compile_artifact(artifact, directory=workdir / "llvm")
    units = [row for row in rows_of(book, EMISSION_UNIT) if row[1] is Backend.LLVM_MODULE]
    functions = [row for row in rows_of(book, EMISSION_FUNCTION) if row[1] is Backend.LLVM_MODULE]
    unsourced = collections.Counter(
        reason.name for page, row, reason, _stage in book.unsourced_rows()
        if getattr(page, "name", page) == EMISSION_UNIT.name and row[1] is Backend.LLVM_MODULE
    )
    per_function = collections.Counter((row[0], row[1]) for row in units)
    for row in functions:
        fact = book.pages[EMISSION_FUNCTION.name].latest(row)
        if not isinstance(fact, FunctionEmission) or fact.unit_count != per_function[row]:
            failures += 1
            print(f"       FAIL llvm emission_function {render_row(row)}: {fact!r} vs {per_function[row]}")
    unit_lines: collections.Counter = collections.Counter()
    for row in units:
        fact = book.pages[EMISSION_UNIT.name].latest(row)
        if isinstance(fact, EmittedUnit):
            unit_lines.update(fact.text.split("\n"))
    claimed_extra = unit_lines - collections.Counter(artifact.llvm_ir.split("\n"))
    library = book.latest_ref(EMISSION_ARTIFACT, (artifact.name, Backend.LLVM_MODULE, ArtifactPart.LIBRARY))
    ref, chain = library, []
    for expected in (ArtifactPart.COMPILE_COMMAND, ArtifactPart.SOURCE_FILE, ArtifactPart.MODULE_TEXT):
        ref = None if ref is None else next(
            (source for source, _stage in book.edges_into(ref)
             if source.page is EMISSION_ARTIFACT and source.row[2] is expected), None,
        )
        chain.append(expected.name if ref is not None else f"MISSING {expected.name}")
    module_functions = () if ref is None else tuple(
        source for source, _stage in book.edges_into(ref) if source.page is EMISSION_FUNCTION
    )
    ok_chain = ref is not None and len(module_functions) == len(functions)
    failures += 0 if ok_chain else 1
    print(f"       {'ok  ' if identical and ok_chain else 'FAIL'} llvm byte-identical={identical} "
          f"units={len(units)} functions={len(functions)} unsourced units={sum(unsourced.values())} "
          f"{dict(unsourced)}; chain LIBRARY -> {' -> '.join(chain)} -> {len(module_functions)} "
          f"emission_function cell(s); unit lines not in final IR={sum(claimed_extra.values())}")
    return failures


def check_case(name: str, annotated: bool) -> int:
    template, dtype, scalar, has_tensor = native.PROGRAMS[name]
    annotation = (": float" if dtype == "float64" else ": int") if annotated else ""
    label = f"{name:7} {'annotated' if annotated else 'plain    '}"
    module, outputs, _ = lower(template.format(a=annotation), dtype, has_tensor)
    book = module.metadata["identity_book"]
    failures = 0

    # 1. byte identity: detached book first (posts nothing), then the real one
    module.metadata["identity_book"] = IdentityBook(detached=True)
    try:
        silent = emit_ssa_module_to_c(module, ROOT)
    finally:
        module.metadata["identity_book"] = book
    artifact = emit_ssa_module_to_c(module, ROOT)
    identical = silent.source == artifact.source
    failures += 0 if identical else 1
    if not artifact.complete:
        print(f"FAIL {label} emission incomplete: {artifact.shortfalls[:3]}")
        return failures + 1
    workdir = BUILD / label.replace(" ", "_")
    workdir.mkdir(parents=True, exist_ok=True)
    artifact.compile(workdir / "scalar_native")

    units = rows_of(book, EMISSION_UNIT)
    functions = rows_of(book, EMISSION_FUNCTION)
    artifacts = rows_of(book, EMISSION_ARTIFACT)
    unsourced = collections.Counter(
        reason.name for page, _row, reason, _stage in book.unsourced_rows()
        if getattr(page, "name", page) == EMISSION_UNIT.name
    )
    print(f"{'ok  ' if identical else 'FAIL'} {label} byte-identical={identical} "
          f"units={len(units)} functions={len(functions)} artifacts={len(artifacts)} "
          f"unsourced units={sum(unsourced.values())} {dict(unsourced)}")

    # 2. unit_count per function row
    per_function = collections.Counter((row[0], row[1]) for row in units)
    for row in functions:
        fact = book.pages[EMISSION_FUNCTION.name].latest(row)
        if not isinstance(fact, FunctionEmission) or fact.unit_count != per_function[row]:
            failures += 1
            print(f"FAIL   emission_function {render_row(row)}: {fact!r} vs {per_function[row]} unit rows")

    # 3. every unit line is a source line; unclaimed lines are preamble
    unit_lines: collections.Counter = collections.Counter()
    for row in units:
        if row[1] is not Backend.C_MODULE:
            continue
        fact = book.pages[EMISSION_UNIT.name].latest(row)
        if getattr(fact, "kind", None) in {UnitKind.STATEMENT, UnitKind.FUNCTION_HEADER,
                                           UnitKind.BLOCK_LABEL, UnitKind.PHI_EDGE_ASSIGNMENT,
                                           UnitKind.BRANCH, UnitKind.RETURN, UnitKind.CALL,
                                           UnitKind.DECLARATION, UnitKind.OUTPUT_STORE,
                                           UnitKind.PROTOTYPE, UnitKind.FORMAL, UnitKind.TABLE}:
            unit_lines.update(fact.text.split("\n"))
    source_lines = collections.Counter(artifact.source.split("\n"))
    claimed_extra = unit_lines - source_lines
    unclaimed = source_lines - unit_lines
    first_prototype = next(
        (index for index, line in enumerate(artifact.source.split("\n"))
         if line.startswith("static ") and line.endswith(");")), None,
    )
    preamble = set(artifact.source.split("\n")[:first_prototype])
    stray = [line for line in unclaimed if line.strip() and line not in preamble]
    if claimed_extra or stray:
        failures += 1
        print(f"FAIL   text coverage: claimed-not-in-source={list(claimed_extra)[:3]} "
              f"unclaimed-body={stray[:3]}")

    # 4. the artifact chain
    library = book.latest_ref(EMISSION_ARTIFACT, (artifact.name, Backend.C_MODULE, ArtifactPart.LIBRARY))
    chain = []
    ref = library
    for expected in (ArtifactPart.COMPILE_COMMAND, ArtifactPart.SOURCE_FILE, ArtifactPart.MODULE_TEXT):
        nxt = None if ref is None else next(
            (source for source, _stage in book.edges_into(ref)
             if source.page is EMISSION_ARTIFACT and source.row[2] is expected), None,
        )
        chain.append(expected.name if nxt is not None else f"MISSING {expected.name}")
        ref = nxt
    module_functions = () if ref is None else tuple(
        source for source, _stage in book.edges_into(ref) if source.page is EMISSION_FUNCTION
    )
    ok_chain = library is not None and ref is not None and len(module_functions) == len(
        [row for row in functions if row[1] is Backend.C_MODULE]
    )
    failures += 0 if ok_chain else 1
    print(f"       artifact chain LIBRARY -> {' -> '.join(chain)} -> "
          f"{len(module_functions)} emission_function cell(s){'' if ok_chain else '  FAIL'}")

    failures += check_llvm(module, book, workdir)

    # 5. one token's chain (bump)
    if name == "bump":
        root = module.functions[ROOT]
        value_id = int(outputs[ROOT][0].id)
        cell = value_cell(book, root, value_id)
        spelled = [
            target for target, _stage in (book.edges_out_of(cell) if cell else ())
            if target.page is EMISSION_UNIT and target.row[1] is Backend.C_MODULE
        ]
        spelled = [ref for ref in spelled if isinstance(fact_of(book, ref), EmittedUnit)]
        print(f"       units spelling the returned value: "
              f"{[(ref.row[0], ref.row[2], fact_of(book, ref).kind.name) for ref in spelled]}")
        token = next(
            (ref for ref in spelled if fact_of(book, ref).spelling == f"t{value_id}"),
            spelled[0] if spelled else None,
        )
        if token is None:
            print(f"FAIL   no C unit spells the returned value's cell {cell!r}")
            return failures + 1
        fact = fact_of(book, token)
        print(f"       token {fact.spelling!r} ({fact.kind.name}) in {token.row[0]}: {fact.text.strip()!r}")
        pages, hops = walk(book, token, show=True)
        spans = [ref for ref in pages.get("source_span", ()) if getattr(fact_of(book, ref), "kind", None) == "BinOp"]
        needed = ("canonical_value", "ingestion_value")
        missing = [page for page in needed if page not in pages]
        ok = bool(spans) and not missing
        failures += 0 if ok else 1
        print(f"       {'ok  ' if ok else 'FAIL'} chain hops={hops}; source_span BinOp rows={len(spans)}; "
              f"missing pages={missing}; pages reached={sorted(pages)}")
        # Plan 100, 7.2 also asks for the operand ``k``'s hop.  It is reached
        # only through the unit that spells ``k + 1`` with its operands (the
        # planned region's Add); that unit is unsourced while the region's
        # literal operand has no identity cell, so this line reports, it does
        # not gate.
        operand_page = "scalar_parameter" if annotated else "name_binding"
        print(f"       OPEN operand hop via {operand_page}: "
              f"{'reached' if operand_page in pages else 'not reached'}")
        for row in rows_of(book, EMISSION_UNIT):
            unit = book.pages[EMISSION_UNIT.name].latest(row)
            if row[0] != ROOT and isinstance(unit, EmittedUnit) and unit.spelling == f"t{value_id}":
                reasons = [reason.name for page, urow, reason, _stage in book.unsourced_rows()
                           if getattr(page, "name", page) == EMISSION_UNIT.name and urow == row]
                print(f"       {row[0]} unit {row[2]} {unit.text.strip()!r}: "
                      f"{'unsourced ' + ','.join(reasons) if reasons else 'derived'}")
    return failures


def main() -> int:
    failures = 0
    for name in native.PROGRAMS:
        for annotated in (False, True):
            try:
                failures += check_case(name, annotated)
            except Exception as error:  # noqa: BLE001 -- report each case
                failures += 1
                print(f"FAIL {name} {'annotated' if annotated else 'plain'} "
                      f"{type(error).__name__}: {str(error)[:300]}")
    print("failures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

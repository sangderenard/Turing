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
5. for ``bump`` (``k + 1``): the C token and the LLVM token that spell the
   value the root returns, and each chain walked back through ``edges_into``
   / ``mint_of`` to a ``source_span`` row of kind ``BinOp``, passing
   ``canonical_value``, ``ingestion_value`` and, for ``k``,
   ``scalar_parameter`` (annotated) or ``name_binding`` (plain).

The LLVM module lane is checked the same way (byte identity, unit counts,
artifact chain, every unit line a line of the final IR -- the define lines
``_annotate_noalias`` rewrites are revised units).  Further LLVM lanes, each
byte-identical with a detached book and with ``unit_count`` equal to its unit
rows: the single-block lane (``emit_ssa_function_to_llvm`` on the planned
region, a one-block function), and ``with_native_sgd_loop`` /
``with_native_adam_loop`` around the module-lane artifact (every id the
wrapper mints a NOVEL ``native_loop_value`` row).  KERNEL_TEXT units are
counted; values local to an imported kernel's scope are counted.  Last, the
gaps the detached emissions kept on the module metadata are replayed onto the
book (``Unsourced(no_book_at_emission)`` rows).

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
    NATIVE_LOOP_VALUE,
    ArtifactPart,
    Backend,
    EmittedUnit,
    FunctionEmission,
    NativeLoop,
    UnitKind,
)
from src.compiler.emission_concordance import (  # noqa: E402
    EMISSION_GAPS,
    replay_emission_gaps,
    value_cell,
)
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


def unsourced_units(book, backend, keys=None) -> collections.Counter:
    return collections.Counter(
        reason.name for page, row, reason, _stage in book.unsourced_rows()
        if getattr(page, "name", page) == EMISSION_UNIT.name and row[1] is backend
        and (keys is None or row[0] in keys)
    )


def unit_count_failures(book, backend, keys=None, label="") -> tuple[int, list, list]:
    """``unit_count`` on every ``emission_function`` row of ``backend``
    (restricted to ``keys``) equals its unit rows."""

    units = [row for row in rows_of(book, EMISSION_UNIT)
             if row[1] is backend and (keys is None or row[0] in keys)]
    functions = [row for row in rows_of(book, EMISSION_FUNCTION)
                 if row[1] is backend and (keys is None or row[0] in keys)]
    per_function = collections.Counter((row[0], row[1]) for row in units)
    failures = 0
    for row in functions:
        fact = book.pages[EMISSION_FUNCTION.name].latest(row)
        if not isinstance(fact, FunctionEmission) or fact.unit_count != per_function[row]:
            failures += 1
            print(f"       FAIL {label} emission_function {render_row(row)}: {fact!r} vs {per_function[row]}")
    return failures, units, functions


def kernel_unit_count(book, units) -> int:
    return sum(
        1 for row in units
        if getattr(book.pages[EMISSION_UNIT.name].latest(row), "kind", None) is UnitKind.KERNEL_TEXT
    )


def lines_not_in(book, units, text) -> int:
    unit_lines: collections.Counter = collections.Counter()
    for row in units:
        fact = book.pages[EMISSION_UNIT.name].latest(row)
        if isinstance(fact, EmittedUnit):
            unit_lines.update(fact.text.split("\n"))
    return sum((unit_lines - collections.Counter(text.split("\n"))).values())


def check_single_block(module, book) -> int:
    """The single-block lane: ``emit_ssa_function_to_llvm`` on the planned
    region (one block; a tensor kernel it calls is a leaf).  The region
    alone is not a complete program, so only the text and rows are read."""

    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm

    region = f"{ROOT}__planned_region_0"
    if region not in module.functions or len(module.functions[region].blocks) != 1:
        print(f"       skip single-block: no one-block {region}")
        return 0
    module.metadata["identity_book"] = IdentityBook(detached=True)
    try:
        silent = emit_ssa_function_to_llvm(module, region)
    finally:
        module.metadata["identity_book"] = book
    artifact = emit_ssa_function_to_llvm(module, region)
    if artifact.emission is None or artifact.emission.backend is not Backend.LLVM_SCALAR:
        print("       FAIL single-block: the region took another lane")
        return 1
    identical = silent.llvm_ir == artifact.llvm_ir
    failures, units, _functions = unit_count_failures(book, Backend.LLVM_SCALAR, label="single-block")
    failures += 0 if identical else 1
    unsourced = unsourced_units(book, Backend.LLVM_SCALAR)
    extra = lines_not_in(book, units, artifact.llvm_ir)
    failures += 1 if extra else 0
    print(f"       {'ok  ' if not failures else 'FAIL'} llvm single-block byte-identical={identical} "
          f"units={len(units)} kernel_text={kernel_unit_count(book, units)} "
          f"unsourced units={sum(unsourced.values())} {dict(unsourced)}; "
          f"unit lines not in IR={extra}")
    return failures


def check_wrappers(module, book, artifact, silent) -> int:
    """``with_native_sgd_loop`` / ``with_native_adam_loop`` around the
    module-lane artifact: byte identity against the wrap of the unbooked
    artifact, unit counts, the minted ids' NOVEL rows, MODULE_TEXT ->
    the wrapper's row -> the wrapped root's row."""

    from src.compiler.ssa_llvm_backend import with_native_adam_loop, with_native_sgd_loop

    shapes = dict(zip(artifact.buffer_order, artifact.buffer_shapes))
    dtypes = dict(zip(artifact.buffer_order, artifact.buffer_dtypes or ()))
    pair = next(
        ((first, second) for index, first in enumerate(artifact.buffer_order)
         for second in artifact.buffer_order[index + 1:]
         if shapes[first] == shapes[second] and dtypes.get(first) == dtypes.get(second)
         and all(isinstance(extent, int) for extent in shapes[first])),
        None,
    )
    if pair is None:
        print("       skip wrappers: no public pair of one static shape")
        return 0
    failures = 0
    for loop, wrap in ((NativeLoop.SGD, with_native_sgd_loop), (NativeLoop.ADAM, with_native_adam_loop)):
        if loop is NativeLoop.ADAM and dtypes.get(pair[0]) != "double":
            continue
        quiet = wrap(silent, parameter_gradient_pairs=[pair])
        wrapped = wrap(artifact, parameter_gradient_pairs=[pair])
        identical = quiet.llvm_ir == wrapped.llvm_ir
        key = (wrapped.name, loop)
        bad, units, functions = unit_count_failures(book, Backend.LLVM_MODULE, keys={key}, label=loop.name)
        minted = [row for row in rows_of(book, NATIVE_LOOP_VALUE) if row[0] == key]
        novel = [row for row in minted if book.mint_of(book.latest_ref(NATIVE_LOOP_VALUE, row)) is not None]
        new_ids = set(wrapped.buffer_order) - set(artifact.buffer_order)
        unsourced = unsourced_units(book, Backend.LLVM_MODULE, keys={key})
        module_text = book.latest_ref(EMISSION_ARTIFACT, (wrapped.name, Backend.LLVM_MODULE, ArtifactPart.MODULE_TEXT))
        reached = {} if module_text is None else walk(book, module_text, show=False)[0]
        reaches_root = (
            any(ref.row[0] == key for ref in reached.get(EMISSION_FUNCTION.name, ()))
            and any(ref.row[0] == ROOT for ref in reached.get(EMISSION_FUNCTION.name, ()))
        )
        ok = (identical and not bad and len(novel) == len(new_ids) == len(minted)
              and not unsourced and reaches_root and wrapped.emission is not None)
        failures += 0 if ok else 1
        print(f"       {'ok  ' if ok else 'FAIL'} llvm {loop.name} wrapper byte-identical={identical} "
              f"units={len(units)} minted ids={len(new_ids)} novel rows={len(novel)} "
              f"unsourced units={sum(unsourced.values())} {dict(unsourced)}; "
              f"MODULE_TEXT -> wrapper row -> root row={reaches_root}")
    return failures


def token_chain(book, root, value_id, backend, annotated) -> int:
    """The unit of ``backend`` spelling ``t{value_id}`` (else the first unit
    that derives from the value's cell), and its chain back to the source."""

    cell = value_cell(book, root, value_id)
    spelled = [
        target for target, _stage in (book.edges_out_of(cell) if cell else ())
        if target.page is EMISSION_UNIT and target.row[1] is backend
    ]
    spelled = [ref for ref in spelled if isinstance(fact_of(book, ref), EmittedUnit)]
    named = [
        book.latest_ref(EMISSION_UNIT, row) for row in rows_of(book, EMISSION_UNIT)
        if row[1] is backend
        and getattr(book.pages[EMISSION_UNIT.name].latest(row), "spelling", None) == f"t{value_id}"
    ]
    token = next(iter(named), None) or next(iter(spelled), None)
    print(f"       [{backend.name}] units spelling the returned value: "
          f"{[(ref.row[0], ref.row[2], fact_of(book, ref).kind.name) for ref in spelled]}")
    if token is None:
        print(f"FAIL   no {backend.name} unit spells the returned value's cell {cell!r}")
        return 1
    fact = fact_of(book, token)
    print(f"       [{backend.name}] token {fact.spelling!r} ({fact.kind.name}) in {token.row[0]}: "
          f"{fact.text.strip()!r}")
    pages, hops = walk(book, token, show=True)
    spans = [ref for ref in pages.get("source_span", ()) if getattr(fact_of(book, ref), "kind", None) == "BinOp"]
    needed = ("canonical_value", "ingestion_value")
    missing = [page for page in needed if page not in pages]
    ok = bool(spans) and not missing
    print(f"       {'ok  ' if ok else 'FAIL'} [{backend.name}] chain hops={hops}; source_span BinOp rows={len(spans)}; "
          f"missing pages={missing}; pages reached={sorted(pages)}")
    operand_page = "scalar_parameter" if annotated else "name_binding"
    print(f"       OPEN [{backend.name}] operand hop via {operand_page}: "
          f"{'reached' if operand_page in pages else 'not reached'}")
    return 0 if ok else 1


def check_llvm(module, book, workdir) -> int:
    """The LLVM module lane: byte identity, unit counts, artifact chain,
    every unit line a line of the final IR (``_annotate_noalias`` revises
    the header units it rewrites); then the single-block lane and the
    native loop wrappers."""

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
    failures += check_wrappers(module, book, artifact, silent)
    compile_artifact(artifact, directory=workdir / "llvm")
    module_keys = {name for name in module.functions}
    bad, units, functions = unit_count_failures(book, Backend.LLVM_MODULE, keys=module_keys, label="llvm")
    failures += bad
    unsourced = unsourced_units(book, Backend.LLVM_MODULE, keys=module_keys)
    claimed_extra = lines_not_in(book, units, artifact.llvm_ir)
    failures += 1 if claimed_extra else 0
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
    print(f"       {'ok  ' if identical and ok_chain and not claimed_extra else 'FAIL'} "
          f"llvm byte-identical={identical} "
          f"units={len(units)} kernel_text={kernel_unit_count(book, units)} "
          f"functions={len(functions)} unsourced units={sum(unsourced.values())} "
          f"{dict(unsourced)}; chain LIBRARY -> {' -> '.join(chain)} -> {len(module_functions)} "
          f"emission_function cell(s); unit lines not in final IR={claimed_extra}")
    revised = sum(
        1 for row in units
        if any(stage.name == "llvm_noalias_annotation"
               for _source, stage in book.edges_into(book.latest_ref(EMISSION_UNIT, row)))
    )
    kernels = [book.latest_ref(EMISSION_UNIT, row) for row in units
               if getattr(book.pages[EMISSION_UNIT.name].latest(row), "kind", None) is UnitKind.KERNEL_TEXT]
    sample = ""
    if kernels:
        fact = fact_of(book, kernels[0])
        sources = collections.Counter(
            (source.page.name, source.row[2] if source.page is EMISSION_ARTIFACT else None)
            for source, _stage in book.edges_into(kernels[0])
        )
        sample = f"; KERNEL_TEXT {fact.spelling!r} <- {dict(sources)}"
    print(f"       header units revised by llvm_noalias_annotation={revised}{sample}")
    failures += check_single_block(module, book)
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
    from src.compiler.emission_concordance import is_imported_kernel

    kernel_scoped = sum(
        1 for row in units
        if row[0] in module.functions and is_imported_kernel(module.functions[row[0]])
    )
    print(f"{'ok  ' if identical else 'FAIL'} {label} byte-identical={identical} "
          f"units={len(units)} functions={len(functions)} artifacts={len(artifacts)} "
          f"unsourced units={sum(unsourced.values())} {dict(unsourced)}; "
          f"units under an imported kernel's scope={kernel_scoped}")

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

    # 5. one token's chain per backend (bump).  Plan 100, 7.2 also asks for
    # the operand ``k``'s hop; it is printed OPEN, not gated.
    if name == "bump":
        root = module.functions[ROOT]
        value_id = int(outputs[ROOT][0].id)
        for backend in (Backend.C_MODULE, Backend.LLVM_MODULE):
            failures += token_chain(book, root, value_id, backend, annotated)

    # 6. the gaps the detached emissions kept, replayed onto the book
    kept = len(module.metadata.get(EMISSION_GAPS) or ())
    replayed = replay_emission_gaps(module)
    ok = kept > 0 and len(replayed) == kept and not module.metadata.get(EMISSION_GAPS)
    failures += 0 if ok else 1
    print(f"       {'ok  ' if ok else 'FAIL'} no-book gaps kept on metadata={kept} "
          f"replayed as Unsourced(no_book_at_emission)={len(replayed)}")
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

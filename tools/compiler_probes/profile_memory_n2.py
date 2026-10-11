"""Memory profile of the N-piece orbital dt lowering (the user runs it).

    python -u tools/compiler_probes/profile_memory_n2.py [N|all] --backend llvm
        [--trace] [--budget-bytes B] [--save-ssa]

TURING_PROBE_EXTERNAL_ROOT can select the original examples/engine_toy roots
while compiler imports stay in this worktree. Lowers the first N
pieces of the orbital craft in link mode, as ``lowered_system`` does (a piece
that is missing or stale is rebuilt through ``equation_piece``, only those N),
and prints, inline:

* the stage table: every ``memory_regulation`` stage boundary and every
  progress message that moved the resident set by 5% or more, with the
  resident set and (``--trace``) the tracemalloc total;
* at the PEAK resident set seen at a progress message: the top 15 object
  types by estimated size, the identity book's largest pages, the retained
  planning memos, and (``--trace``) the top 10 tracemalloc sites;
* the sha256 of the emitted C or LLVM module, so a regulated run
  (``--budget-bytes``) can be compared with the unregulated one.

It compiles; it is not run by the agents that edit the compiler.
"""
from __future__ import annotations

import gc
import os
import hashlib
import sys
import tempfile
import time
import tracemalloc
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = Path(os.environ.get("TURING_PROBE_EXTERNAL_ROOT", str(ROOT)))
for _entry in (EXTERNAL_ROOT.parent / "engine_toy", EXTERNAL_ROOT / "examples", ROOT):
    sys.path.insert(0, str(_entry))

from src.compiler.shell_telemetry import process_rss_bytes  # noqa: E402

MIB = 2 ** 20
STARTED = time.time()


def estimated_histogram(limit: int = 15) -> list[tuple[str, int, int]]:
    sizes: Counter = Counter()
    counts: Counter = Counter()
    for obj in gc.get_objects():
        name = type(obj).__name__
        try:
            sizes[name] += sys.getsizeof(obj)
        except TypeError:
            continue
        counts[name] += 1
    return [(name, counts[name], size) for name, size in sizes.most_common(limit)]


def book_report() -> list[str]:
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler import region_scheduling as scheduling
    from src.compiler.identity_concordance import current_identity_book

    book = current_identity_book()
    # (cell_count reads a spilled page's size from its segment; the plain
    # ``len(page.cells)`` would read the page back and measure nothing.)
    pages = sorted(
        ((page.cell_count(), name)
         for name, page in dict.items(book.pages)),
        reverse=True,
    )
    lines = [f"  identity book: {sum(n for n, _ in pages):,} cells "
             f"({book.resident_cell_count():,} in RAM, "
             f"{len(book.spilled_pages())} pages spilled), "
             f"{len(book.scope_projections):,} projected scopes, "
             f"{len(book.__dict__.get('_node_universes', {})):,} interned node sets"]
    lines += [f"    {count:>10,}  {name}" for count, name in pages[:10]]
    lines.append(
        "  planning memos: "
        f"callsite shell types {len(strategy._CALLSITE_SHELL_TYPE_CACHE):,}, "
        f"child signature results "
        f"{sum(len(v[1]) for v in strategy._CHILD_SIGNATURE_RESULTS.values()):,}, "
        f"polymorphic scans {len(strategy._POLYMORPHIC_FORMAL_SCANS):,}, "
        f"dependency levels {len(scheduling._DEPENDENCY_LEVEL_CACHE):,}"
    )
    return lines


def main(argv: list[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Compile N orbital rules through OUR public compiler; no target execution or deadline.")
    parser.add_argument("N", nargs="?", default="2", help="positive rule count, or all")
    parser.add_argument("--backend", choices=("c", "llvm"), default="c")
    parser.add_argument("--source-compilation", choices=("serial", "units"), default="serial")
    parser.add_argument("--optimization", choices=("O0", "O1", "O2", "O3"), default="O3")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--save-ssa", action="store_true", help="retain the public lowerer's SSA before backend emission, for failure analysis")
    parser.add_argument("--budget-bytes", type=int)
    parser.add_argument("--digest-only", action="store_true", help="retained compatibility option")
    options = parser.parse_args(argv[1:])
    os.chdir(ROOT)
    try:
        pieces_wanted = None if options.N == "all" else int(options.N)
    except ValueError:
        parser.error("N must be positive, or all")
    if pieces_wanted is not None and pieces_wanted < 1:
        parser.error("N must be positive, or all")
    tracing = "--trace" in argv
    budget = options.budget_bytes
    if tracing:
        tracemalloc.start(10)

    import dataclasses

    from src.compiler import memory_regulation
    from src.compiler.work_contract import active_contract, set_active_contract

    if budget is not None:
        set_active_contract(dataclasses.replace(
            active_contract(), memory_budget_bytes=budget,
        ))

    table: list[tuple[float, str, int, int]] = []

    def traced() -> int:
        return tracemalloc.get_traced_memory()[0] if tracing else 0

    original_boundary = memory_regulation.MemoryRegulator.boundary

    def boundary(self, label):
        resident = original_boundary(self, label)
        table.append((time.time() - STARTED, f"[stage] {label}", resident, traced()))
        return resident

    memory_regulation.MemoryRegulator.boundary = boundary

    peak = {"rss": 0, "label": "", "report": []}

    def progress(message: str) -> None:
        resident = process_rss_bytes()
        last = table[-1][2] if table else 0
        if not last or abs(resident - last) >= 0.05 * last:
            table.append((time.time() - STARTED, str(message)[:100], resident, traced()))
        if resident > peak["rss"] * 1.05:
            peak["rss"], peak["label"] = resident, str(message)
            report = [f"  object types by estimated size at: {message[:90]}"]
            report += [f"    {size / MIB:>9.1f} MiB  {count:>11,}  {name}"
                       for name, count, size in estimated_histogram()]
            report += book_report()
            if tracing:
                snapshot = tracemalloc.take_snapshot()
                report.append("  tracemalloc top sites:")
                for stat in snapshot.statistics("lineno")[:10]:
                    report.append(f"    {stat.size / MIB:>9.1f} MiB  {stat}")
            peak["report"] = report

    import llvm_dt_system as lds
    from orbital_craft_machine import craft_machine_equations, orbital_craft
    from orbital_jumper import ORBITAL_CHANNEL_NAMES
    import honorary_engine_equation_catalogue as honorary
    from src.compiler.native_law_kernels import LLVMPiece, piece_staleness

    craft = orbital_craft()
    paths = []
    rules = list(craft_machine_equations(craft, 1, green_coast=True))
    if pieces_wanted is not None and pieces_wanted > len(rules):
        parser.error(f"N={pieces_wanted} exceeds this craft's {len(rules)} rules")
    selected = rules if pieces_wanted is None else rules[:pieces_wanted]
    print(f"RULE_SUBMISSION count={len(selected)} available={len(rules)} backend={options.backend} "
          f"source_compilation={options.source_compilation} optimization={options.optimization} "
          f"trace={tracing} compiler_root={ROOT}", flush=True)
    for name, equations in selected:
        path = (honorary._LLVM_LAW_CACHE_DIR / name / "b1"
                / honorary._llvm_piece_cache_key(name, 1, tuple(equations))
                / f"{name}.piece")
        if not path.is_file() or piece_staleness(LLVMPiece.load(path)):
            progress(f"profile: building piece {name}")
            honorary.equation_piece(name, equations, batch=1)
        paths.append(path)

    (ROOT / "build").mkdir(exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=f"profile_rules_{len(selected)}_{options.backend}_", dir=ROOT / "build"))
    print("RULE_BUILD_DIRECTORY=" + str(build), flush=True)
    progress("profile: lowered_system begin")
    from src.compiler import fortran_c_shell
    ordinary_lower = fortran_c_shell.lower_ast_source_to_ssa
    def capture_ssa(*args, **kwargs):
        result = ordinary_lower(*args, **kwargs)
        import pickle
        destination = build / "repository-ssa.pkl"
        try:
            with destination.open("wb") as stream:
                pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as error:
            print(f"SSA_CAPTURE_FAILED {type(error).__name__}: {error}; compile continues", flush=True)
        else:
            print(f"SSA_CAPTURE {destination} bytes={destination.stat().st_size}", flush=True)
        return result
    if options.save_ssa:
        fortran_c_shell.lower_ast_source_to_ssa = capture_ssa
    try:
        system = lds.lowered_system(
            paths, directory=build, piece_mode="link",
            channel_names=ORBITAL_CHANNEL_NAMES, progress=progress,
            backend=lds.LLVM_BACKEND if options.backend == "llvm" else lds.C_BACKEND,
            source_compilation=options.source_compilation, optimization=options.optimization,
        )
        progress("profile: lowered_system done")
    finally:
        fortran_c_shell.lower_ast_source_to_ssa = ordinary_lower
        print(f"\n{'t(s)':>8} {'RSS MiB':>9} {'traced MiB':>11}  stage")
        for when, label, resident, trace in table:
            print(f"{when:8.1f} {resident / MIB:9.0f} {trace / MIB:11.0f}  {label}")
        print(f"\npeak resident set at a progress message: {peak['rss'] / MIB:.0f} MiB "
              f"({peak['label'][:90]})")
        print("\n".join(peak["report"]))
        print(f"PROBE_ELAPSED {time.time() - STARTED:.3f}s BUILD_DIRECTORY {build}", flush=True)
    source = build / f"{system.artifact.name}{'.ll' if options.backend == 'llvm' else '.c'}"
    if source.is_file():
        print(f"\nemitted {options.backend} {source.name}: sha256 "
              f"{hashlib.sha256(source.read_bytes()).hexdigest()}")
    print(f"{len(system.module.functions)} functions")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

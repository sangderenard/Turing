"""Memory profile of the N-piece orbital dt lowering (the user runs it).

    python -u tools/compiler_probes/profile_memory_n2.py [N] [--trace]
        [--budget-bytes B] [--digest-only]

PYTHONPATH must include C:\\dev\\Powershell\\engine_toy.  Lowers the first N
pieces of the orbital craft in link mode, as ``lowered_system`` does (a piece
that is missing or stale is rebuilt through ``equation_piece``, only those N),
and prints, inline:

* the stage table: every ``memory_regulation`` stage boundary and every
  progress message that moved the resident set by 5% or more, with the
  resident set and (``--trace``) the tracemalloc total;
* at the PEAK resident set seen at a progress message: the top 15 object
  types by estimated size, the identity book's largest pages, the retained
  planning memos, and (``--trace``) the top 10 tracemalloc sites;
* the sha256 of the emitted C module, so a regulated run
  (``--budget-bytes``) can be compared with the unregulated one.

It compiles; it is not run by the agents that edit the compiler.
"""
from __future__ import annotations

import gc
import hashlib
import sys
import tempfile
import time
import tracemalloc
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _entry in (ROOT.parent / "engine_toy", ROOT / "examples", ROOT):
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
    pieces_wanted = int(next((a for a in argv[1:] if a.isdigit()), 2))
    tracing = "--trace" in argv
    budget = None
    if "--budget-bytes" in argv:
        budget = int(argv[argv.index("--budget-bytes") + 1])
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
    for name, equations in list(
            craft_machine_equations(craft, 1, green_coast=True))[:pieces_wanted]:
        path = (honorary._LLVM_LAW_CACHE_DIR / name / "b1"
                / honorary._llvm_piece_cache_key(name, 1, tuple(equations))
                / f"{name}.piece")
        if not path.is_file() or piece_staleness(LLVMPiece.load(path)):
            progress(f"profile: building piece {name}")
            honorary.equation_piece(name, equations, batch=1)
        paths.append(path)

    build = Path(tempfile.gettempdir()) / "profile_memory_n2"
    progress("profile: lowered_system begin")
    system = lds.lowered_system(
        paths, directory=build, piece_mode="link",
        channel_names=ORBITAL_CHANNEL_NAMES, progress=progress,
    )
    progress("profile: lowered_system done")

    print(f"\n{'t(s)':>8} {'RSS MiB':>9} {'traced MiB':>11}  stage")
    for when, label, resident, trace in table:
        print(f"{when:8.1f} {resident / MIB:9.0f} {trace / MIB:11.0f}  {label}")
    print(f"\npeak resident set at a progress message: {peak['rss'] / MIB:.0f} MiB "
          f"({peak['label'][:90]})")
    print("\n".join(peak["report"]))
    source = build / f"{system.artifact.name}.c"
    if source.is_file():
        print(f"\nemitted C {source.name}: sha256 "
              f"{hashlib.sha256(source.read_bytes()).hexdigest()}")
    print(f"{len(system.module.functions)} functions")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

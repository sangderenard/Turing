"""Memory profile of the N-piece orbital dt lowering (the user runs it).

    python -u tools/compiler_probes/profile_memory_n2.py [N|all] --backend llvm
        [--trace] [--budget-bytes B] [--save-ssa]

TURING_PROBE_EXTERNAL_ROOT can select the original examples/engine_toy roots
while compiler imports stay in this worktree. Lowers the first N
pieces of the orbital craft in link mode. Missing or stale laws are supplied
as a parallel source list; the complete linked system follows. Both use the
canonical eleven-stage display, with auto inside each source. With --trace it
also prints, inline:

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
             f"({book.resident_cell_count():,} uncompressed cells in RAM, "
             f"{len(book.spilled_pages())} pages spilled), "
             f"{len(book.scope_projections):,} projected scopes, "
             f"{len(book.__dict__.get('_node_universes', {})):,} interned node sets"]
    lines += [f"    {count:>10,}  {name}" for count, name in pages[:10]]
    if book.compressed_pages():
        from src.compiler.identity_compression import compression_stats, disk_tile_stats

        raw, resident = compression_stats(book)
        disk_tiles, disk_bytes = disk_tile_stats(book)
        lines.append(f"  compressed IN RAM: {len(book.compressed_pages())} pages, "
                     f"{resident / MIB:.2f} MiB retained buffers/indexes, "
                     f"{raw / MIB:.2f} MiB tile serialization, "
                     f"disk_tiles={disk_tiles} disk_MiB={disk_bytes / MIB:.2f}, "
                     f"budget_met={book.__dict__.get('_compression_budget_met')}")
    live = book.__dict__.get("_live_tile_report")
    if live is not None:
        lines.append(f"  LIVE graph/control/deployment/SSA/book disk tiles={live['disk_tiles']:,} "
                     f"disk_MiB={live['disk_bytes'] / MIB:.2f} "
                     f"decoded tile limit={live['hot_limit']} tile_rows={live['tile_rows']}")
    lines.append(
        "  planning memos: "
        f"callsite shell types {len(strategy._CALLSITE_SHELL_TYPE_CACHE):,}, "
        f"child signature results "
        f"{sum(len(v[1]) for v in strategy._CHILD_SIGNATURE_RESULTS.values()):,}, "
        f"polymorphic scans {len(strategy._POLYMORPHIC_FORMAL_SCANS):,}, "
        f"dependency levels {len(scheduling._DEPENDENCY_LEVEL_CACHE):,}"
    )
    return lines


def _run_profile(argv: list[str], scope) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Compile N orbital rules through OUR public compiler; no target execution or deadline.")
    parser.add_argument("N", nargs="?", default="2", help="positive rule count, or all")
    parser.add_argument("--backend", choices=("c", "llvm"), default="c")
    parser.add_argument("--source-compilation", choices=("auto", "serial", "parallel", "units"), default="auto")
    parser.add_argument("--source-schedule", choices=("serial", "parallel"), default="parallel")
    parser.add_argument("--jobs", type=int, default=None, help="optional maximum source-list workers; inner auto remains implicit")
    parser.add_argument("--min-workers", "--minimum-workers", type=int, default=12,
                        help="minimum admitted workers when that many independent jobs are ready (default: 12)")
    parser.add_argument("--max-total-gb", type=float, default=10.0)
    parser.add_argument("--max-worker-gb", type=float, default=10.0)
    parser.add_argument("--system-reserve-gb", type=float, default=2.0)
    parser.add_argument("--worker-reserve-gb", type=float, default=2.0)
    parser.add_argument("--thread-working-mib", type=int, default=128)
    parser.add_argument("--worker-elastic-available-percent", type=int, default=75)
    parser.add_argument("--optimization", choices=("O0", "O1", "O2", "O3"), default="O3")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--save-ssa", action="store_true", help="retain the public lowerer's SSA before backend emission, for failure analysis")
    parser.add_argument("--budget-bytes", type=int)
    parser.add_argument("--ram-tiles", action="store_true", help="live graph/control/deployment/SSA/book tiles; stronger disk tier above --budget-bytes")
    parser.add_argument("--digest-only", action="store_true", help="retained compatibility option")
    options = parser.parse_args(argv[1:])
    schedule = options.source_schedule if options.source_compilation == "auto" else (
        "parallel" if options.source_compilation == "units" else options.source_compilation)
    worker_options = {"max_total_resident_bytes": int(options.max_total_gb * (1 << 30)),
        "max_worker_memory_bytes": int(options.max_worker_gb * (1 << 30)),
        "system_reserve_bytes": int(options.system_reserve_gb * (1 << 30)),
        "worker_resident_reservation_bytes": int(options.worker_reserve_gb * (1 << 30)),
        "thread_working_bytes": options.thread_working_mib << 20,
        "minimum_worker_jobs": options.min_workers,
        "worker_elastic_available_percent": options.worker_elastic_available_percent}
    if options.jobs is not None:
        worker_options["jobs"] = options.jobs
    os.chdir(ROOT)
    try:
        pieces_wanted = None if options.N == "all" else int(options.N)
    except ValueError:
        parser.error("N must be positive, or all")
    if pieces_wanted is not None and pieces_wanted < 1:
        parser.error("N must be positive, or all")
    tracing = "--trace" in argv
    budget = options.budget_bytes
    if budget is not None:
        worker_options["max_worker_memory_bytes"] = budget
    if tracing:
        tracemalloc.start(10)

    import dataclasses

    from src.compiler import memory_regulation
    from src.compiler.work_contract import active_contract, scoped_work_contract

    scope.enter_context(scoped_work_contract(dataclasses.replace(
            active_contract(),
            memory_budget_bytes=budget if budget is not None else worker_options["max_worker_memory_bytes"],
            compress_cold_pages=options.ram_tiles or active_contract().compress_cold_pages,
            spill_cold_pages=False if options.ram_tiles else active_contract().spill_cold_pages,
    )))

    table: list[tuple[float, str, int, int]] = []

    def traced() -> int:
        return tracemalloc.get_traced_memory()[0] if tracing else 0

    original_boundary = memory_regulation.MemoryRegulator.boundary

    def boundary(self, label):
        resident = original_boundary(self, label)
        table.append((time.time() - STARTED, f"[stage] {label}", resident, traced()))
        return resident

    memory_regulation.MemoryRegulator.boundary = boundary
    scope.callback(setattr, memory_regulation.MemoryRegulator, "boundary", original_boundary)

    peak = {"rss": 0, "label": "", "report": []}

    def progress(message: str) -> None:
        from src.compiler.shell_telemetry import active_compile_progress
        display = active_compile_progress()
        if display is not None:
            display.log(message)
        resident = process_rss_bytes()
        last = table[-1][2] if table else 0
        if not last or abs(resident - last) >= 0.05 * last:
            table.append((time.time() - STARTED, str(message)[:100], resident, traced()))
        if resident > peak["rss"] * 1.05:
            peak["rss"], peak["label"] = resident, str(message)
            if not tracing:
                return
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
    from src.compiler.shell_telemetry import compile_rule_batch
    scope.enter_context(compile_rule_batch(len(selected), names=[name for name, _ in selected]))
    storage_contract = active_contract()
    print(f"RULE_SUBMISSION count={len(selected)} available={len(rules)} backend={options.backend} "
          f"source_compilation={options.source_compilation} optimization={options.optimization} "
          f"trace={tracing} compiler_root={ROOT} "
          f"ram_tiles={storage_contract.compress_cold_pages} "
          f"tile_disk={storage_contract.compressed_tile_disk} "
          f"memory_budget_bytes={storage_contract.memory_budget_bytes}", flush=True)
    (ROOT / "build").mkdir(exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=f"profile_rules_{len(selected)}_{options.backend}_", dir=ROOT / "build"))
    print("RULE_BUILD_DIRECTORY=" + str(build), flush=True)
    from src.compiler.build_product_retention import (
        compiler_product_lease,
        prune_stale_orbital_profile_products,
    )
    scope.enter_context(compiler_product_lease(build))
    removed_products = prune_stale_orbital_profile_products(
        ROOT / "build", build,
    )
    if removed_products:
        print(
            "RULE_BUILD_RETENTION removed="
            f"{len(removed_products)} active={build.name}",
            flush=True,
        )
    paths = compile_rule_pieces(selected, build, source_compilation=schedule,
                               jobs=options.jobs, optimization=options.optimization, progress=progress,
                               partitioned_options=worker_options)
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
            source_compilation=schedule, optimization=options.optimization,
            partitioned_options=worker_options,
        )
        progress("profile: lowered_system done")
    finally:
        fortran_c_shell.lower_ast_source_to_ssa = ordinary_lower
        if tracing:
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


def compile_rule_pieces(rules, directory, *, source_compilation="auto", jobs=None,
                        optimization="O3", progress=None, partitioned_options=None):
    """Prepare cached inputs, then submit every missing law as one source program."""
    import honorary_engine_equation_catalogue as honorary
    from src.compiler.native_law_kernels import LLVMPiece, piece_staleness, post_piece_book
    from src.compiler.shell_telemetry import active_rule_batch_progress
    from src.compiler.native_package import pieces_from_equations
    paths, missing = [], {}
    for name, equations in rules:
        key = honorary._llvm_piece_cache_key(name, 1, tuple(equations))
        path = honorary._LLVM_LAW_CACHE_DIR / name / "b1" / key / f"{name}.piece"
        paths.append(path)
        if not path.is_file() or piece_staleness(LLVMPiece.load(path)):
            missing[name] = tuple(equations)
        else:
            batch = active_rule_batch_progress()
            if batch is not None:
                batch.complete(name, 0.0)
    if missing:
        pieces = pieces_from_equations(
            missing, 1, directory=Path(directory) / "laws",
            source_compilation="parallel" if source_compilation == "units" else source_compilation,
            partitioned_options={**dict(partitioned_options or {}), **({} if jobs is None else {"jobs": jobs})},
            optimization=optimization, progress=progress,
        )
        for (name, equations), path in zip(rules, paths):
            if name not in pieces:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with honorary._PieceLock(path.parent / f"{name}.lock"):
                honorary._save_piece_index(pieces[name], path)
                post_piece_book(path.parent, name, 1,
                                honorary._llvm_piece_cache_key(name, 1, tuple(equations)),
                                built=pieces[name])
    return paths


def main(argv):
    from contextlib import ExitStack
    from src.compiler.shell_telemetry import compile_progress
    with ExitStack() as scope:
        scope.enter_context(compile_progress(None, collect_profile=True))
        return _run_profile(argv, scope)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""Small real book benchmark; run against separate baseline and branch roots.

python tools/compiler_probes/benchmark_dense_book.py --root . --cells 10000

Timing and allocation measurement are separate. No source lowering or native
execution is replaced by this storage workload. Facts form a sourced chain.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cells', type=int, default=10000)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve()))
    from src.compiler.identity_concordance import (
        IdentityBook, Registry, RowField, RowFieldKind, Novel, Derived, Mode, Ref,
        render_identity_book,
    )
    from src.compiler.work_contract import active_contract, scoped_work_contract
    from src.compiler import compiler_bootstrap_runtime
    describe_backend = getattr(compiler_bootstrap_runtime, 'provenance_core_state', None)
    backend = describe_backend() if describe_backend is not None else {'backend': 'python'}
    contract = active_contract()

    def populate():
        registry = Registry()
        page = registry.declare_page('bench_value', (RowField('value', RowFieldKind.VALUE_ID),), int)
        stage = registry.declare_stage('bench')
        root = registry.declare_transform('bench_root', 0)
        book = IdentityBook(registry=registry)
        previous = book.post(page, (0,), 0, stage=stage, provenance=Novel(root, ()), mode=Mode.CONCORD)
        for i in range(1, args.cells):
            previous = book.post(page, (i,), i, stage=stage, provenance=Derived((previous,)), mode=Mode.CONCORD)
        return book, page

    with scoped_work_contract(contract):
        walls, cpus, reads = [], [], []
        for _ in range(args.repeats):
            gc.collect()
            wall, cpu = time.perf_counter(), time.process_time()
            book, page = populate()
            walls.append(time.perf_counter() - wall)
            cpus.append(time.process_time() - cpu)
            wall = time.perf_counter()
            checksum = sum(book.page(page).latest((i,)) for i in range(args.cells))
            reads.append(time.perf_counter() - wall)
            assert checksum == args.cells * (args.cells - 1) // 2
            del book, page
        gc.collect()
        tracemalloc.start()
        book, page = populate()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        sample = Ref(page, (1,), 0)
        hash(sample)
        ref_bytes = sys.getsizeof(sample)
        if hasattr(sample, '__dict__'):
            ref_bytes += sys.getsizeof(sample.__dict__)
        hashed = [Ref(page, (i,), 0) for i in range(args.cells)]
        for item in hashed:
            hash(item)
        hash_times = []
        for _ in range(args.repeats):
            wall = time.perf_counter()
            sum(map(hash, hashed))
            hash_times.append(time.perf_counter() - wall)
        # One coarse operation: all provenance/history searches, ordering,
        # row rendering and string assembly stay inside the module boundary.
        format_times = []
        for _ in range(args.repeats):
            wall = time.perf_counter()
            formatted = render_identity_book(book)
            format_times.append(time.perf_counter() - wall)
        formatted_bytes = len(formatted.encode('utf-8'))
        del formatted
        print(json.dumps({
            'root': str(args.root.resolve()), 'storage': backend['backend'],
            'facts': args.cells, 'materialized_cells': sum(p.cell_count() for p in book.pages.values()),
            'repeats': args.repeats, 'post_wall_median_s': statistics.median(walls),
            'post_cpu_median_s': statistics.median(cpus),
            'latest_read_wall_median_s': statistics.median(reads),
            'traced_retained_bytes': current, 'traced_peak_bytes': peak,
            'reference_bytes_each_including_dict': ref_bytes,
            'cached_hash_wall_median_s': statistics.median(hash_times),
            'whole_book_format_wall_median_s': statistics.median(format_times),
            'formatted_bytes': formatted_bytes,
        }, sort_keys=True))


if __name__ == '__main__':
    main()

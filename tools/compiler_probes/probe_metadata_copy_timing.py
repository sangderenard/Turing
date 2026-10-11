"""Whole authored copier: eager vs installed native, ALL boundary cost included."""
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.compiler.native_compiler_accelerators import (
    compile_and_register_metadata_copy_accelerator, metadata_copy_source,
)


def main():
    namespace = {}
    exec(metadata_copy_source(), namespace)
    original = namespace['metadata_copy_program']
    started = time.perf_counter()
    provider = compile_and_register_metadata_copy_accelerator()
    print('BUILD_OR_LOAD_SECONDS', time.perf_counter() - started, provider.library_path, flush=True)
    for count in (1, 4, 16, 64):
        leaf = object()
        shared = [leaf]
        values = tuple({'row': row, 'left': shared, 'right': shared} for row in range(count))
        def eager():
            return original(values, {})
        def native():
            return provider.run(values, copier=lambda value: (_ for _ in ()).throw(
                AssertionError('unexpected Python fallback')))
        timings = {}
        for label, run in (('python', eager), ('native_with_boundary', native)):
            samples = []
            for _ in range(5):
                started = time.perf_counter()
                result = run()
                samples.append(time.perf_counter() - started)
                assert result == values and result is not values
                assert result[0]['left'] is result[-1]['right'] and result[0]['left'] is not shared
                assert result[0]['left'][0] is leaf
            timings[label] = statistics.median(samples)
        print('ROWS', count, 'SECONDS', timings, 'NATIVE_SPEEDUP',
              timings['python'] / timings['native_with_boundary'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

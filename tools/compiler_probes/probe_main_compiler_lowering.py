"""Submit the unchanged complete main compiler source to its public entry.

No extracted helper, substitute entry, target execution, foreign compiler or
compiler edits. The parent streams output without imposing a time limit.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import hashlib
    import json
    import time
    import traceback
    from src.compiler import fortran_c_shell
    from src.compiler.extraction_contract import ExtractionContract

    path = Path(fortran_c_shell.__file__).resolve()
    source = path.read_text(encoding='utf-8')
    contract = ExtractionContract(ROOT / 'extraction_contracts/program_extraction.yaml')
    contract = contract.with_execution_file(ROOT / 'extraction_contracts/vehicle_full_native_execution.yaml')
    entrypoint = 'lower_ast_source_to_ssa'
    print('MAIN_COMPILER_SUBMISSION=' + json.dumps({
        'source': str(path), 'entrypoint': entrypoint,
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'source_lines': len(source.splitlines()), 'complete_module_source': True,
        'execution': 'full native, no Python callbacks', 'diagnostic_deadline_seconds': None,
    }), flush=True)
    last_progress = [None]
    def progress(message):
        last_progress[0] = str(message)
        if '[graph-build #' in str(message):
            counter = str(message).split('#', 1)[1].split(']', 1)[0]
            if counter != '1' and not counter.endswith('0000'):
                return
        # The public entry redirects discovery's incidental stdout. Progress
        # must still reach the user's actual stream, not that private capture.
        print('[main-self-lower] ' + str(message), file=sys.__stdout__, flush=True)
    started = time.perf_counter()
    try:
        module, outputs, exports = fortran_c_shell.lower_ast_source_to_ssa(
            source, entrypoint, name='compiler_main_self_lower',
            python_bindings=dict(vars(fortran_c_shell)), runtime_closure_only=True,
            extraction_contract=contract, progress=progress, progress_bars=False,
            identity_log_level='summary',
        )
    except BaseException as error:
        print('MAIN_COMPILER_BLOCKER=' + json.dumps({
            'seconds': time.perf_counter() - started, 'type': type(error).__name__,
            'message': str(error), 'last_progress': last_progress[0],
            'frames': [{'file': frame.filename, 'line': frame.lineno, 'function': frame.name,
                        'source': frame.line} for frame in traceback.extract_tb(error.__traceback__)],
        }), flush=True)
        traceback.print_exception(error, file=sys.stdout)
        return 2
    print('MAIN_COMPILER_LOWERED=' + json.dumps({
        'seconds': time.perf_counter() - started, 'functions': len(module.functions),
        'exports': exports, 'outputs': outputs,
    }, default=str), flush=True)
    from src.compiler.identity_concordance import concordance_report
    print(concordance_report(module), flush=True)
    return 0


if __name__ == '__main__':
    if '--child' in sys.argv:
        raise SystemExit(main())
    result = subprocess.run([sys.executable, '-u', __file__, '--child'])
    raise SystemExit(result.returncode)

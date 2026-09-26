"""Save fresh managed-controller source and SSA checkpoints without compiling C."""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from joblib.externals import cloudpickle
    from src.compiler import fortran_c_shell as shell
    from src.compiler.project_compilation_product import _dump_resolved_process_graph
    from src.compiler.vehicle_python_compilation import lower_balloon_tire_managed_python_ssa
    from src.compiler.ssa_self_check import run_all

    def save(name, value, serializer=pickle):
        destination = args.output / name
        temporary = destination.with_name(destination.name + '.tmp')
        with temporary.open('wb') as stream:
            serializer.dump(value, stream)
        temporary.replace(destination)
        print(f'Checkpoint saved: {destination}', flush=True)

    original_lower = shell.lower_ast_source_to_ssa
    original_link = shell._class_surface_ssa_program

    def capture_source(*positional, **keywords):
        def sink(graph):
            destination = args.output / 'resolved-process-graph.pkl'
            temporary = destination.with_name(destination.name + '.tmp')
            with temporary.open('wb') as stream:
                _dump_resolved_process_graph(graph, stream)
            temporary.replace(destination)
            print(f'Checkpoint saved: {destination}', flush=True)
        return original_lower(*positional, **{**keywords, 'resolved_process_graph_sink': sink})

    def capture_link(*positional, **keywords):
        save('pre-frame-link.pkl', (positional, {
            key: value for key, value in keywords.items() if key != 'progress'
        }), cloudpickle)
        result = original_link(*positional, **keywords)
        save('repository-ssa.pkl', result)
        findings = run_all(result[0])
        for finding in findings:
            print(finding, flush=True)
        print(f'Structural check findings: {len(findings)}', flush=True)
        return result

    shell.lower_ast_source_to_ssa = capture_source
    shell._class_surface_ssa_program = capture_link
    try:
        lower_balloon_tire_managed_python_ssa(progress=lambda message: print(message, flush=True))
    finally:
        shell.lower_ast_source_to_ssa = original_lower
        shell._class_surface_ssa_program = original_link


if __name__ == '__main__':
    main()

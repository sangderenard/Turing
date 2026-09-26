"""Initialize and drive the compiler graph-translation training network."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.tensors.abstract_nn.graph_translation_network import (
    CompilerTransformationRoute,
    GraphTranslationNetwork,
    TransformationWeightMatrix,
)
from src.common.tensors.abstract_nn.training_data_store import (
    CompilerTrainingDatabase,
)


DEFAULT_FORMS = (
    "source", "python_ast", "process_graph", "ssa", "llvm", "fortran",
    "c_shell", "trace_manifest", "native_trace", "shell_telemetry",
    "spectral_dye",
)

DEFAULT_ROUTES = (
    CompilerTransformationRoute("source", "python_ast", "parse_python"),
    CompilerTransformationRoute("source", "process_graph", "compile_ast_aot"),
    CompilerTransformationRoute("process_graph", "ssa", "lower_repository_ssa"),
    CompilerTransformationRoute("ssa", "llvm", "emit_llvm"),
    CompilerTransformationRoute("ssa", "fortran", "emit_fortran"),
    CompilerTransformationRoute("ssa", "c_shell", "emit_c_shell"),
    CompilerTransformationRoute("process_graph", "trace_manifest", "instrument_trace"),
    CompilerTransformationRoute("trace_manifest", "native_trace", "run_native_trace"),
    CompilerTransformationRoute("native_trace", "spectral_dye", "analyse_trace_dye"),
    CompilerTransformationRoute("shell_telemetry", "spectral_dye", "analyse_trace_dye"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path,
        default=ROOT / "build" / "compiler_training.sqlite3",
    )
    parser.add_argument("--vocabulary-size", type=int, default=65536)
    parser.add_argument("--program-id", type=int)
    parser.add_argument("--densify", action="store_true")
    parser.add_argument(
        "--work", action="store_true",
        help="execute registered pending compiler-teacher commands",
    )
    args = parser.parse_args()
    if args.vocabulary_size < 1:
        parser.error("--vocabulary-size must be positive")
    if args.densify and args.program_id is None:
        parser.error("--densify requires --program-id")

    with CompilerTrainingDatabase(args.database) as database:
        weights = TransformationWeightMatrix(DEFAULT_FORMS)
        weights.populate_stubs(args.vocabulary_size)
        weights.persist(database)
        network = GraphTranslationNetwork(database, weights, DEFAULT_ROUTES)
        requests = (
            network.densify(args.program_id)
            if args.densify else ()
        )
        completed = ()
        if args.work:
            from src.common.tensors.abstract_nn.compiler_teacher_worker import (
                CompilerTeacherWorker,
            )
            completed = CompilerTeacherWorker(database).run_pending()
        counts = {
            table: database.connection.execute(
                f"SELECT count(*) FROM {table}"
            ).fetchone()[0]
            for table in (
                "programs", "views", "token_events", "transformations",
                "compiler_commands", "weight_sets",
            )
        }
    print(" ".join(f"{name}={count}" for name, count in counts.items()))
    if requests:
        print("queued " + ", ".join(
            f"{request.source_form}->{request.target_form}:{request.command_name}"
            for request in requests
        ))
    if completed:
        print("completed " + ", ".join(
            f"{view.form}:{view.id}" for view in completed
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

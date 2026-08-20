"""Compile, profile, trace, and render one authored Python entrypoint.

    python tools/compile_spectral_dye_trace.py examples/demo.py main
    python tools/compile_spectral_dye_trace.py examples/demo.py main --feeds inputs.json --frames 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _feeds(argument: str) -> dict:
    if not argument:
        return {}
    candidate = Path(argument)
    text = candidate.read_text(encoding="utf-8") if candidate.is_file() else argument
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("--feeds must be a JSON object or a file containing one")
    return value


def _names(spec: str) -> tuple[str, ...]:
    """Parse authored target names, never transient compiler identifiers."""

    return tuple(dict.fromkeys(token for token in spec.replace(",", " ").split() if token))


def _trace_payload(stderr: str) -> dict:
    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if line.startswith('{') and '"trace"' in line:
            return json.loads(line)
    raise RuntimeError("native shell completed without emitting its trace JSON")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="authored Python source")
    parser.add_argument("entry", help="function to compile and run")
    parser.add_argument("--feeds", default="", help="JSON object or JSON file for runtime inputs")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--target", default="",
                        help="authored names to select through the trace manifest, e.g. energy,total")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--training-db", type=Path,
        default=ROOT / "build" / "compiler_training.sqlite3",
        help="SQLite corpus receiving source, graph, backend, and trace views",
    )
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"source file does not exist: {args.source}")
    if args.frames < 1:
        parser.error("--frames must be positive")
    try:
        feeds = _feeds(args.feeds)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
    from src.compiler.fortran_c_shell import compile_ast_fortran_c_shell
    from src.compiler.influence_field import InfluenceContract, field_from_process_graph
    from src.compiler.shell_telemetry import TelemetryChannel
    from src.compiler.spectral_trace_dye import analyse_trace_dye
    from spectral_dye_trace import _render_timeline

    output = args.out or ROOT / "build" / f"spectral-dye-{args.source.stem}-{args.entry}"
    output.mkdir(parents=True, exist_ok=True)
    source = args.source.read_text(encoding="utf-8")
    channel = TelemetryChannel(name=f"spectral-dye:{args.entry}")
    with channel.timed("aot compile", path=str(args.source), entry=args.entry):
        compilation = compile_ast_aot(
            source, args.entry, feeds, trace=True, profiling=True,
            precompile_only=True, backend="c",
        )
    manifest = dict((compilation.map_ir or {}).get("trace") or {})
    if not manifest:
        raise RuntimeError("trace-enabled AOT compilation produced no trace manifest")
    manifest_path = output / "trace_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with channel.timed("native shell compile", path=str(args.source), entry=args.entry):
        executable = compile_ast_fortran_c_shell(
            source, args.entry, feeds, output, compilation=compilation, trace=True,
        )
    with channel.timed("native shell run", path=str(executable.executable_path), frames=args.frames):
        completed = executable.run(frames=args.frames)
    native_trace = _trace_payload(completed.stderr)
    for launch in native_trace["trace"].get("launches", ()):
        channel.trace("native region", region=int(launch["region"]), path=args.entry,
                      nanoseconds=int(launch.get("shell_ns", 0)), status=int(launch.get("status", 0)))
    telemetry_path = output / "shell_telemetry.json"
    telemetry_path.write_text(channel.to_json() + "\n", encoding="utf-8")
    trace_path = output / "native_trace.json"
    trace_path.write_text(json.dumps(native_trace, indent=2) + "\n", encoding="utf-8")

    field = field_from_process_graph(compilation.deployment.process_graph, InfluenceContract(enabled=True))
    field.propagate()
    report = analyse_trace_dye(
        native_trace, manifest, field, level="ssa", target_names=_names(args.target),
    )
    report_path = output / "spectral_dye_trace.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    image_path = output / "spectral_dye_trace.png"
    _render_timeline(image_path, report)
    from src.common.tensors.abstract_nn.token_lexicon import (
        CompilerTokenLexicon, lexical_tokens, structural_context_tokens,
    )
    from src.common.tensors.abstract_nn.training_data_store import (
        CompilerTrainingDatabase, put_reduced_graph_view,
    )
    lexicon = CompilerTokenLexicon.load(
        ROOT / "build" / "compiler_token_lexicon.json"
    )
    with CompilerTrainingDatabase(args.training_db) as database:
        program_id = database.put_program(
            source, args.entry,
            metadata={"source_path": str(args.source), "demo": "spectral_dye"},
        )
        source_tokens = lexical_tokens(source)
        source_view = database.put_view(
            program_id, "source", {"text": source}, source_tokens,
            token_ids=tuple(lexicon.token_id(token) for token in source_tokens),
            lexicon_revision=lexicon.revision,
            generator="compile_spectral_dye_trace",
        )
        graph_view = put_reduced_graph_view(
            database, program_id, compilation.deployment.process_graph,
            lexicon_revision=lexicon.revision,
        )
        database.link_views(
            program_id, source_view.id, graph_view.id, "reduce_process_graph",
            weight_key=f"source->process_graph@{lexicon.revision}",
            compiler_command={"command": "compile_ast_aot", "entry": args.entry},
        )
        previous_view = graph_view
        for form, path, payload in (
            ("fortran", output / f"{args.entry}.f90", None),
            ("c_shell", output / f"{args.entry}.c", None),
            ("trace_manifest", manifest_path, manifest),
            ("native_trace", trace_path, native_trace),
            ("shell_telemetry", telemetry_path, json.loads(channel.to_json())),
            ("spectral_dye", report_path, report),
        ):
            if payload is None:
                if not path.is_file():
                    continue
                text_payload = path.read_text(encoding="utf-8")
                payload = {"text": text_payload}
                tokens = lexical_tokens(text_payload)
            else:
                tokens = structural_context_tokens(payload)
            view = database.put_view(
                program_id, form, payload, tokens,
                token_ids=tuple(lexicon.token_id(token) for token in tokens),
                lexicon_revision=lexicon.revision,
                generator="compile_spectral_dye_trace",
            )
            database.link_views(
                program_id, previous_view.id, view.id, f"produce_{form}",
                weight_key=f"{previous_view.form}->{form}@{lexicon.revision}",
                compiler_command={"command": f"produce_{form}", "entry": args.entry},
            )
            previous_view = view
    print(f"wrote {manifest_path}\nwrote {telemetry_path}\nwrote {trace_path}")
    print(f"wrote {report_path}\nwrote {image_path}")
    print(f"training corpus -> {args.training_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

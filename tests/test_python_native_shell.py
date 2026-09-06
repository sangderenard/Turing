from src.compiler.compiled_program_api import CompiledProgramAPI, EntryPoint, Parameter
from src.compiler.python_native_shell import (
    NativeFilePortHandler, _option_parser,
    emit_python_native_shell, profile_frame_arrivals,
)
from src.compiler.shell_io import (
    ShellIOBinding,
    ShellIOManifest,
    ShellIORequest,
    SystemPort,
    ShellOption,
    attach_shell_io,
)


def test_python_launcher_is_generated_only_from_compiled_descriptor(tmp_path):
    manifest = ShellIOManifest(
        (ShellIORequest.create("display_double_buffer"),),
        bindings=(ShellIOBinding("display.back", "frame", "t9"),),
        options=(ShellOption("width", "int", 320),),
    )
    api = attach_shell_io(CompiledProgramAPI(
        "demo", "fortran", "frame", (EntryPoint(
            "frame", "frame", "control", (Parameter(
                "t9", "output", "float32", "float", "c_float", "reference",
                shape=(320,), source_name="pixels",
            ),),
        ),),
    ), manifest)

    generated = emit_python_native_shell(api, tmp_path / "demo.dll")

    assert "Mandelbrot" not in generated.source
    assert "cursor" not in generated.source.casefold()
    assert "launch_native_shell(COMPILED_API, LIBRARY_PATH)" in generated.source
    assert '"resource": "display.back"' in generated.source
    assert "sys.path.insert(0," in generated.source
    compile(generated.source, "generated_native_shell.py", "exec")


class _ProfileRuntime:
    def __init__(self):
        self.calls = []

    def execute(self, elapsed):
        self.calls.append(("execute", elapsed))

    def frame(self, elapsed):
        self.calls.append(("frame", elapsed))


def test_arrival_profile_can_stop_before_display_transfer():
    runtime = _ProfileRuntime()
    timestamps = iter((1_000_000, 3_000_000, 6_000_000))

    profile = profile_frame_arrivals(
        runtime,
        arrivals=3,
        warmup=1,
        ignore_transfer=True,
        clock_ns=lambda: next(timestamps),
    )

    assert [call[0] for call in runtime.calls] == ["execute"] * 4
    assert profile.gaps_ms == (2.0, 3.0)
    mapping = profile.to_mapping()
    assert mapping["measurement_boundary"] == "execution_completion"
    assert mapping["transfer_excluded"] is True


def test_arrival_profile_includes_display_acquisition_by_default():
    runtime = _ProfileRuntime()
    timestamps = iter((1_000_000, 2_000_000))

    profile = profile_frame_arrivals(
        runtime,
        arrivals=2,
        warmup=0,
        clock_ns=lambda: next(timestamps),
    )

    assert [call[0] for call in runtime.calls] == ["frame", "frame"]
    assert profile.to_mapping()["measurement_boundary"] == (
        "display_plane_arrival"
    )


def test_native_file_port_cli_loads_byte_exact_data_and_length(tmp_path):
    subject = tmp_path / "subject.exe"
    subject.write_bytes(b"MZ\x00\xff")
    manifest = ShellIOManifest(
        (ShellIORequest.create("files"),),
        system_ports=(SystemPort.create(
            "subject-binary", "file", "input", entry_point="load_subject",
            fields={"data": "subject_bytes", "length": "subject_length"},
        ),),
    ).to_mapping()

    options = vars(_option_parser(manifest).parse_args([
        "--file-subject-binary", str(subject), "--headless",
    ]))
    handler = NativeFilePortHandler(manifest, options)

    assert handler.resource("subject-binary", "data").tolist() == [77, 90, 0, 255]
    assert handler.resource("subject-binary", "length") == 4

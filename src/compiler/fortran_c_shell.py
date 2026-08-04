"""Package an emitted ``bind(C)`` Fortran module in a native C shell.

The C translation unit contains only the generic profiled launch boundary,
buffer ownership, declared state feedback, and diagnostics.  Program logic
remains in the :class:`~src.compiler.ssa_fortran_backend.FortranModule` that
the ordinary AST/Control/SSA pipeline emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

import numpy as np

from ..common.tensors.accelerator_backends.profiled_c_shell import _C_SOURCE
from .ssa_fortran_backend import FortranEmissionError, fortran_compiler


_NUMPY_DTYPES = {
    "bool": np.dtype("bool"),
    "logical": np.dtype("bool"),
    "float": np.dtype("float32"),
    "float32": np.dtype("float32"),
    "f32": np.dtype("float32"),
    "double": np.dtype("float64"),
    "float64": np.dtype("float64"),
    "f64": np.dtype("float64"),
    "int": np.dtype("int32"),
    "int32": np.dtype("int32"),
    "i32": np.dtype("int32"),
    "int64": np.dtype("int64"),
    "i64": np.dtype("int64"),
}


@dataclass(frozen=True)
class FortranCShellExecutable:
    directory: Path
    executable_path: Path
    fortran_source_path: Path
    c_source_path: Path
    api_path: Path
    initial_state_path: Path
    final_outputs_path: Path
    entrypoint: str

    def run(
        self,
        *,
        frames: int = 1,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if frames < 0:
            raise ValueError("native C shell frame count cannot be negative")
        environment = dict(os.environ)
        compiler = fortran_compiler()
        if compiler is not None:
            environment["PATH"] = (
                str(Path(compiler).parent)
                + os.pathsep
                + environment.get("PATH", "")
            )
        return subprocess.run(
            [str(self.executable_path), str(frames)],
            cwd=str(self.directory),
            env=environment,
            capture_output=capture_output,
            text=True,
            check=True,
        )


def _identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
    if not result or result[0].isdigit():
        result = "turing_" + result
    return result


def _entrypoint(module: Any, name: str | None = None) -> Any:
    selected = name or module.api.entry
    if selected is None:
        raise ValueError("Fortran module has no selected entry point")
    return module.api.entry_point(str(selected))


def _extent_values(
    entry: Any,
    overrides: Mapping[str, int] | None,
) -> dict[str, int]:
    values = {
        parameter.name: int(parameter.name.rsplit("_", 1)[-1])
        for parameter in entry.parameters
        if parameter.role == "extent"
    }
    for name, value in dict(overrides or {}).items():
        if name not in values:
            raise ValueError(f"unknown Fortran extent override {name!r}")
        if int(value) < 1:
            raise ValueError(f"Fortran extent {name!r} must be positive")
        values[name] = int(value)
    return values


def _element_count(parameter: Any, extents: Mapping[str, int]) -> int:
    count = 1
    for extent in tuple(parameter.shape or ()):
        count *= int(extents.get(f"extent_{int(extent)}", extent))
    return max(count, 1)


def _source_name(parameter: Any) -> str:
    return str(parameter.source_name or parameter.name)


def _c_string(value: str) -> str:
    return json.dumps(str(value))


def _display_configuration(module: Any, entry: Any) -> dict[str, Any] | None:
    """Resolve an optional declarative display request from the shared IO ABI."""

    metadata = dict(getattr(module.api, "metadata", {}) or {})
    shell_io = metadata.get("shell_io") or {}
    requirements = shell_io.get("requirements") or {}
    requests = [
        request for request in requirements.get("requests", ())
        if request.get("capability") == "display_double_buffer"
    ]
    if not requests:
        return None
    if len(requests) != 1:
        raise ValueError("C shell requires one display_double_buffer request")
    attributes = dict(requests[0].get("attributes") or {})
    pixel_format = str(attributes.get("pixel_format", "rgb_f64_planar"))
    if pixel_format != "rgb_f64_planar":
        raise ValueError(
            "native C shell currently supports display pixel format "
            "'rgb_f64_planar'; got " + repr(pixel_format)
        )
    width = int(attributes.get("width", 0))
    height = int(attributes.get("height", 0))
    if width < 1 or height < 1:
        raise ValueError("native display request needs positive width and height")
    bindings = {
        str(binding.get("resource")): str(binding.get("parameter"))
        for binding in requirements.get("bindings", ())
        if str(binding.get("entry_point")) == str(entry.name)
        and str(binding.get("resource", "")).startswith("display.")
    }
    missing = {f"display.{channel}" for channel in ("red", "green", "blue")} - set(bindings)
    if missing:
        raise ValueError(
            "rgb_f64_planar display lacks bindings: "
            + ", ".join(sorted(missing))
        )
    parameters = {parameter.name: parameter for parameter in entry.parameters}
    channels = []
    for resource in ("display.red", "display.green", "display.blue"):
        parameter = parameters.get(bindings[resource])
        if parameter is None or parameter.role != "output":
            raise ValueError(f"{resource} must bind an output ABI parameter")
        if str(parameter.c_type) != "double":
            raise ValueError(f"{resource} must bind a float64 output")
        channels.append(parameter.name)
    return {
        "width": width,
        "height": height,
        "title": str(attributes.get("title", "Turing native display")),
        "channels": tuple(channels),
    }


def emit_fortran_c_shell_source(
    module: Any,
    *,
    entrypoint: str | None = None,
    state_feedback: Mapping[str, str] | None = None,
    extent_overrides: Mapping[str, int] | None = None,
    initial_state_filename: str = "initial-state.bin",
    final_outputs_filename: str = "final-outputs.bin",
) -> str:
    """Emit a standalone C main around one described Fortran entry point."""

    entry = _entrypoint(module, entrypoint)
    parameters = tuple(entry.parameters)
    extents = _extent_values(entry, extent_overrides)
    values = tuple(item for item in parameters if item.role != "extent")
    inputs = tuple(item for item in values if item.role == "input")
    outputs = tuple(item for item in values if item.role == "output")
    unsupported = tuple(
        item for item in values if item.role not in {"input", "output"}
    )
    if unsupported:
        raise ValueError(
            "C shell cannot allocate parameter roles "
            + ", ".join(sorted({item.role for item in unsupported}))
        )
    slot_by_name = {
        _source_name(parameter): index
        for index, parameter in enumerate(values)
    }
    slot_by_parameter = {
        parameter.name: index for index, parameter in enumerate(values)
    }
    display = _display_configuration(module, entry)
    if display is not None:
        expected_pixels = int(display["width"]) * int(display["height"])
        for parameter_name in display["channels"]:
            parameter = next(
                item for item in values if item.name == parameter_name
            )
            if _element_count(parameter, extents) != expected_pixels:
                raise ValueError(
                    f"display channel {parameter_name!r} has "
                    f"{_element_count(parameter, extents)} elements; expected "
                    f"{expected_pixels}"
                )
    feedback = dict(state_feedback or {})
    missing_feedback = {
        name
        for pair in feedback.items()
        for name in pair
        if name not in slot_by_name
    }
    if missing_feedback:
        raise ValueError(
            "state feedback references absent Fortran ABI names: "
            + ", ".join(sorted(missing_feedback))
        )

    prototype_arguments = []
    call_arguments = []
    value_index = 0
    for parameter in parameters:
        c_type = str(parameter.c_type)
        if parameter.role == "extent":
            prototype_arguments.append(c_type)
            call_arguments.append(str(extents[parameter.name]))
            continue
        pointer = parameter.passing == "reference"
        prototype_arguments.append(c_type + (" *" if pointer else ""))
        slot = f"slots[{value_index}]"
        call_arguments.append(
            f"({c_type} *){slot}" if pointer
            else f"*(({c_type} *){slot})"
        )
        value_index += 1

    allocation_lines = []
    input_read_lines = []
    for index, parameter in enumerate(values):
        c_type = str(parameter.c_type)
        count = _element_count(parameter, extents)
        allocation_lines.extend((
            f"    slots[{index}] = calloc({count}, sizeof({c_type}));",
            f"    if (!slots[{index}]) return 3;",
        ))
        if parameter.role == "input":
            input_read_lines.extend((
                f"    if (fread(slots[{index}], sizeof({c_type}), {count}, state) "
                f"!= {count}) {{",
                f"        fprintf(stderr, \"short initial state at {_c_string(_source_name(parameter))[1:-1]}\\n\");",
                "        return 4;",
                "    }",
            ))

    feedback_lines = []
    for input_name, output_name in feedback.items():
        input_slot = slot_by_name[input_name]
        output_slot = slot_by_name[output_name]
        input_parameter = values[input_slot]
        output_parameter = values[output_slot]
        if (
            input_parameter.c_type != output_parameter.c_type
            or _element_count(input_parameter, extents)
            != _element_count(output_parameter, extents)
        ):
            raise ValueError(
                f"state feedback {input_name!r}->{output_name!r} has "
                "incompatible storage"
            )
        feedback_lines.append(
            f"        memcpy(slots[{input_slot}], slots[{output_slot}], "
            f"{_element_count(input_parameter, extents)} * sizeof({input_parameter.c_type}));"
        )

    output_lines = []
    output_write_lines = []
    for output_index, parameter in enumerate(outputs):
        slot = slot_by_name[_source_name(parameter)]
        count = _element_count(parameter, extents)
        separator = "" if output_index == 0 else ","
        output_lines.extend((
            f"    {{ double sum = 0.0; size_t i;",
            f"      for (i = 0; i < {count}; ++i) sum += (({parameter.c_type} *)slots[{slot}])[i];",
            f"      printf(\"{separator}\\\"{_source_name(parameter)}\\\":{{\\\"first\\\":%.17g,\\\"sum\\\":%.17g}}\",",
            f"             (double)(({parameter.c_type} *)slots[{slot}])[0], sum); }}",
        ))
        output_write_lines.append(
            f"    fwrite(slots[{slot}], sizeof({parameter.c_type}), {count}, outputs_file);"
        )

    display_source = ""
    display_open_lines: list[str] = []
    display_loop_condition = "frame < frames"
    display_message_lines: list[str] = []
    display_present_lines: list[str] = []
    display_close_lines: list[str] = []
    default_frames = "1"
    if display is not None:
        red_slot, green_slot, blue_slot = (
            slot_by_parameter[name] for name in display["channels"]
        )
        width = int(display["width"])
        height = int(display["height"])
        title = _c_string(display["title"])
        default_frames = "0"
        display_loop_condition = "turing_display_running && (frames == 0 || frame < frames)"
        display_source = r'''
#if !defined(_WIN32)
#error "The dependency-free native display adapter currently requires Win32"
#else
static HWND turing_display_window = NULL;
static int turing_display_running = 1;
static uint32_t *turing_display_pixels = NULL;
static int turing_display_width = 0;
static int turing_display_height = 0;

static LRESULT CALLBACK turing_display_proc(
    HWND window, UINT message, WPARAM wparam, LPARAM lparam
) {
    (void)wparam;
    (void)lparam;
    if (message == WM_CLOSE) {
        DestroyWindow(window);
        return 0;
    }
    if (message == WM_DESTROY) {
        turing_display_running = 0;
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(window, message, wparam, lparam);
}

static void turing_display_set_utf8_title(HWND window, const char *title) {
    int length = MultiByteToWideChar(CP_UTF8, 0, title, -1, NULL, 0);
    wchar_t *wide;
    if (length < 1) return;
    wide = (wchar_t *)calloc((size_t)length, sizeof(wchar_t));
    if (wide == NULL) return;
    if (MultiByteToWideChar(CP_UTF8, 0, title, -1, wide, length)) {
        SetWindowTextW(window, wide);
    }
    free(wide);
}

static int turing_display_open(int width, int height, const char *title) {
    WNDCLASSA window_class = {0};
    RECT rectangle = {0, 0, width, height};
    HINSTANCE instance = GetModuleHandleA(NULL);
    window_class.lpfnWndProc = turing_display_proc;
    window_class.hInstance = instance;
    window_class.lpszClassName = "TuringNativeDisplay";
    window_class.hCursor = LoadCursor(NULL, IDC_ARROW);
    if (!RegisterClassA(&window_class) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        return 0;
    }
    AdjustWindowRect(&rectangle, WS_OVERLAPPEDWINDOW, FALSE);
    turing_display_window = CreateWindowExA(
        0, window_class.lpszClassName, "", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rectangle.right - rectangle.left, rectangle.bottom - rectangle.top,
        NULL, NULL, instance, NULL
    );
    if (turing_display_window == NULL) return 0;
    turing_display_set_utf8_title(turing_display_window, title);
    turing_display_pixels = (uint32_t *)calloc(
        (size_t)width * (size_t)height, sizeof(uint32_t)
    );
    if (turing_display_pixels == NULL) return 0;
    turing_display_width = width;
    turing_display_height = height;
    return 1;
}

static void turing_display_messages(void) {
    MSG message;
    while (PeekMessageA(&message, NULL, 0, 0, PM_REMOVE)) {
        if (message.message == WM_QUIT) turing_display_running = 0;
        TranslateMessage(&message);
        DispatchMessageA(&message);
    }
}

static unsigned int turing_display_channel(double value) {
    if (value <= 0.0) return 0;
    if (value >= 255.0) return 255;
    return (unsigned int)(value + 0.5);
}

static void turing_display_present(
    const double *red, const double *green, const double *blue
) {
    BITMAPINFO information = {0};
    RECT client;
    HDC device;
    size_t index;
    size_t count = (size_t)turing_display_width * (size_t)turing_display_height;
    for (index = 0; index < count; ++index) {
        unsigned int r = turing_display_channel(red[index]);
        unsigned int g = turing_display_channel(green[index]);
        unsigned int b = turing_display_channel(blue[index]);
        turing_display_pixels[index] = b | (g << 8) | (r << 16);
    }
    information.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    information.bmiHeader.biWidth = turing_display_width;
    information.bmiHeader.biHeight = -turing_display_height;
    information.bmiHeader.biPlanes = 1;
    information.bmiHeader.biBitCount = 32;
    information.bmiHeader.biCompression = BI_RGB;
    GetClientRect(turing_display_window, &client);
    device = GetDC(turing_display_window);
    StretchDIBits(
        device, 0, 0, client.right, client.bottom,
        0, 0, turing_display_width, turing_display_height,
        turing_display_pixels, &information, DIB_RGB_COLORS, SRCCOPY
    );
    ReleaseDC(turing_display_window, device);
}

static void turing_display_close(void) {
    free(turing_display_pixels);
    turing_display_pixels = NULL;
    if (turing_display_window != NULL && IsWindow(turing_display_window)) {
        DestroyWindow(turing_display_window);
    }
    turing_display_window = NULL;
}
#endif
'''
        display_open_lines = [
            f"    if (!turing_display_open({width}, {height}, {title})) return 7;",
        ]
        display_message_lines = [
            "        turing_display_messages();",
            "        if (!turing_display_running) break;",
        ]
        display_present_lines = [
            "        turing_display_present(",
            f"            (const double *)slots[{red_slot}],",
            f"            (const double *)slots[{green_slot}],",
            f"            (const double *)slots[{blue_slot}]);",
            "        turing_display_messages();",
        ]
        display_close_lines = ["    turing_display_close();"]

    source = "\n".join((
        _C_SOURCE,
        "",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "",
        display_source,
        f"extern void {entry.symbol}({', '.join(prototype_arguments)});",
        "",
        "static int turing_fortran_compute(void *context, unsigned long long *device_ns) {",
        "    void **slots = (void **)context;",
        "    *device_ns = 0;",
        f"    {entry.symbol}({', '.join(call_arguments)});",
        "    return 1;",
        "}",
        "",
        "int main(int argc, char **argv) {",
        f"    int frames = argc > 1 ? atoi(argv[1]) : {default_frames};",
        f"    void *slots[{len(values)}] = {{0}};",
        "    TuringLaunchProfile profile = {0};",
        "    TuringLaunchStats stats = {0};",
        "    int frame;",
        f"    FILE *state = fopen({_c_string(initial_state_filename)}, \"rb\");",
        "    if (frames < 0) return 2;",
        "    if (!state) { perror(\"initial state\"); return 2; }",
        *allocation_lines,
        *input_read_lines,
        "    fclose(state);",
        *display_open_lines,
        "    turing_launch_stats_reset(&stats);",
        f"    for (frame = 0; {display_loop_condition}; ++frame) {{",
        *display_message_lines,
        "        if (turing_profiled_launch_ex(turing_fortran_compute, slots,",
        "                &profile, &stats, NULL, NULL, 3) != 1) return 5;",
        *feedback_lines,
        *display_present_lines,
        "    }",
        *display_close_lines,
        "    printf(\"{\\\"status\\\":%d,\\\"frames\\\":%d,\\\"shell_ns_total\\\":%llu,\\\"outputs\\\":{\",",
        "           profile.status, frame, stats.shell_ns_total);",
        *output_lines,
        "    printf(\"}}\\n\");",
        f"    {{ FILE *outputs_file = fopen({_c_string(final_outputs_filename)}, \"wb\");",
        "      if (!outputs_file) { perror(\"final outputs\"); return 6; }",
        *output_write_lines,
        "      fclose(outputs_file); }",
        f"    for (frame = 0; frame < {len(values)}; ++frame) free(slots[frame]);",
        "    return 0;",
        "}",
        "",
    ))
    return source


def compile_fortran_module_c_shell(
    module: Any,
    inputs: Mapping[str, Any],
    directory: str | Path,
    *,
    entrypoint: str | None = None,
    state_feedback: Mapping[str, str] | None = None,
    extent_overrides: Mapping[str, int] | None = None,
    name: str = "turing_fortran_c_shell",
) -> FortranCShellExecutable:
    """Compile generated Fortran plus the generic profiled C main."""

    compiler = fortran_compiler()
    if compiler is None:
        raise FortranEmissionError("no Fortran compiler found")
    compiler = str(Path(compiler).resolve())
    gcc = str(Path(compiler).with_name("gcc.exe" if os.name == "nt" else "gcc"))
    if not Path(gcc).is_file():
        raise FortranEmissionError(f"C compiler beside gfortran is missing: {gcc}")
    output = Path(directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    entry = _entrypoint(module, entrypoint)
    extents = _extent_values(entry, extent_overrides)
    values = tuple(item for item in entry.parameters if item.role != "extent")
    input_parameters = tuple(item for item in values if item.role == "input")
    missing = {
        _source_name(parameter)
        for parameter in input_parameters
        if _source_name(parameter) not in inputs
    }
    if missing:
        raise ValueError("missing C-shell inputs: " + ", ".join(sorted(missing)))

    state_bytes = bytearray()
    for parameter in input_parameters:
        source_name = _source_name(parameter)
        dtype = _NUMPY_DTYPES.get(str(parameter.dtype).casefold())
        if dtype is None:
            raise ValueError(f"unsupported C-shell dtype {parameter.dtype!r}")
        value = np.asarray(inputs[source_name], dtype=dtype)
        expected = _element_count(parameter, extents)
        if value.size != expected:
            raise ValueError(
                f"input {source_name!r} has {value.size} elements; "
                f"compiled ABI requires {expected}"
            )
        state_bytes.extend(np.ascontiguousarray(value).tobytes())

    fortran_path = output / f"{name}.f90"
    c_path = output / f"{name}.c"
    api_path = output / f"{name}.api.yaml"
    state_path = output / "initial-state.bin"
    final_outputs_path = output / "final-outputs.bin"
    fortran_path.write_text(module.source, encoding="utf-8")
    c_path.write_text(
        emit_fortran_c_shell_source(
            module,
            entrypoint=entry.name,
            state_feedback=state_feedback,
            extent_overrides=extents,
            initial_state_filename=state_path.name,
            final_outputs_filename=final_outputs_path.name,
        ),
        encoding="utf-8",
    )
    module.api.write(api_path)
    state_path.write_bytes(bytes(state_bytes))

    suffix = ".exe" if os.name == "nt" else ""
    executable = output / f"{name}{suffix}"
    fortran_object = output / f"{name}.fortran.o"
    c_object = output / f"{name}.shell.o"
    environment = dict(os.environ)
    environment["PATH"] = (
        str(Path(compiler).parent) + os.pathsep + environment.get("PATH", "")
    )
    commands = (
        [compiler, "-O3", "-c", str(fortran_path), "-o", str(fortran_object)],
        [gcc, "-O3", "-std=c11", "-c", str(c_path), "-o", str(c_object)],
        [
            compiler, str(c_object), str(fortran_object), "-o", str(executable),
            *(
                ["-mwindows", "-lgdi32", "-luser32"]
                if _display_configuration(module, entry) else []
            ),
        ],
    )
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=str(output),
            env=environment,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise FortranEmissionError(
                "native Fortran/C-shell compilation failed:\n"
                + (completed.stderr or completed.stdout)
            )
    return FortranCShellExecutable(
        directory=output,
        executable_path=executable,
        fortran_source_path=fortran_path,
        c_source_path=c_path,
        api_path=api_path,
        initial_state_path=state_path,
        final_outputs_path=final_outputs_path,
        entrypoint=entry.name,
    )


__all__ = [
    "FortranCShellExecutable",
    "compile_fortran_module_c_shell",
    "emit_fortran_c_shell_source",
]

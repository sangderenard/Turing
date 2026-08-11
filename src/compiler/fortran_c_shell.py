"""Package an emitted ``bind(C)`` Fortran module in a native C shell.

The C translation unit contains only the generic profiled launch boundary,
buffer ownership, declared state feedback, and diagnostics.  Program logic
remains in the :class:`~src.compiler.ssa_fortran_backend.FortranModule` that
the ordinary AST/Control/SSA pipeline emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping
from typing import Callable

import numpy as np

from ..common.tensors.accelerator_backends.profiled_c_shell import _C_SOURCE
from .fortran_toolchain import (
    aggressive_c_flags,
    aggressive_fortran_flags,
    standalone_fortran_link_flags,
    standalone_runtime_shim_sources,
)
from .ssa_fortran_backend import FortranEmissionError, fortran_compiler


_NUMPY_DTYPES = {
    "uint8": np.dtype("uint8"),
    "u8": np.dtype("uint8"),
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
        files: Mapping[str, str | Path] | None = None,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if frames < 0:
            raise ValueError("native C shell frame count cannot be negative")
        arguments = [str(self.executable_path), str(frames)]
        for name, path in sorted(dict(files or {}).items()):
            arguments.extend(("--file-" + _identifier(name).replace("_", "-"), str(Path(path).resolve())))
        return subprocess.run(
            arguments,
            cwd=str(self.directory),
            env=dict(os.environ),
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


def _fortran_storage_index(
    parameter: Any,
    extents: Mapping[str, int],
    linear_index: str,
) -> str:
    """Map one C-row-major logical index to Fortran array storage.

    The API shape is semantic and remains in Python/NumPy dimension order.
    A ``bind(C)`` Fortran dummy with that shape stores its first dimension
    fastest, so the outer shell must perform this boundary permutation once.
    Resident feedback arenas stay in Fortran order and require no copies.
    """

    shape = tuple(
        int(extents.get(f"extent_{int(size)}", size))
        for size in tuple(parameter.shape or ())
    )
    if len(shape) <= 1:
        return linear_index
    terms = []
    for dimension, size in enumerate(shape):
        c_stride = 1
        for following in shape[dimension + 1:]:
            c_stride *= int(following)
        fortran_stride = 1
        for preceding in shape[:dimension]:
            fortran_stride *= int(preceding)
        coordinate = (
            f"(({linear_index}) / {c_stride}) % {size}"
            if c_stride != 1
            else f"({linear_index}) % {size}"
        )
        terms.append(
            coordinate
            if fortran_stride == 1
            else f"({coordinate}) * {fortran_stride}"
        )
    return " + ".join(terms)


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
        "frame_delay_ms": max(0, int(attributes.get("frame_delay_ms", 0))),
    }


def _system_file_configurations(module: Any, entry: Any) -> tuple[dict[str, Any], ...]:
    metadata = dict(getattr(module.api, "metadata", {}) or {})
    requirements = dict((metadata.get("shell_io") or {}).get("requirements") or {})
    parameters = {parameter.name: parameter for parameter in entry.parameters}
    configurations = []
    for port in requirements.get("system_ports", ()):
        if port.get("kind") != "file" or port.get("direction") not in {
            "input", "bidirectional",
        }:
            continue
        if str(port.get("entry_point")) != str(entry.name):
            continue
        fields = {
            str(field.get("name")): str(field.get("parameter"))
            for field in port.get("fields", ())
        }
        if set(fields) < {"data", "length"}:
            raise ValueError(f"native file port {port.get('name')!r} lacks data/length fields")
        data = parameters.get(fields["data"])
        length = parameters.get(fields["length"])
        if data is None or length is None:
            raise ValueError(f"native file port {port.get('name')!r} has unknown parameters")
        if str(data.c_type) != "uint8_t" or data.passing != "reference":
            raise ValueError("native file data parameter must be a uint8 reference")
        if str(length.c_type) not in {"int32_t", "int64_t"}:
            raise ValueError("native file length parameter must be int32 or int64")
        attributes = dict(port.get("attributes") or {})
        capacity = int(attributes.get("maximum_bytes", _element_count(data, _extent_values(entry, None))))
        if capacity < 1:
            raise ValueError("native input file capacity must be positive")
        configurations.append({
            "name": str(port["name"]),
            "flag": "--file-" + _identifier(str(port["name"])).replace("_", "-"),
            "data": data,
            "length": length,
            "capacity": capacity,
            "optional": bool(port.get("optional")),
        })
    return tuple(configurations)


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
    file_ports = _system_file_configurations(module, entry)
    system_parameters = {
        parameter.name
        for port in file_ports
        for parameter in (port["data"], port["length"])
    }
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
        file_port = next((port for port in file_ports if port["data"].name == parameter.name), None)
        count = int(file_port["capacity"]) if file_port else _element_count(parameter, extents)
        allocation_lines.extend((
            f"    slots[{index}] = calloc({count}, sizeof({c_type}));",
            f"    if (!slots[{index}]) return 3;",
        ))
        if parameter.role == "input" and parameter.name not in system_parameters:
            if len(tuple(parameter.shape or ())) <= 1:
                input_read_lines.extend((
                    f"    if (fread(slots[{index}], sizeof({c_type}), {count}, state) "
                    f"!= {count}) {{",
                    f"        fprintf(stderr, \"short initial state at {_c_string(_source_name(parameter))[1:-1]}\\n\");",
                    "        return 4;",
                    "    }",
                ))
            else:
                storage_index = _fortran_storage_index(
                    parameter, extents, "logical_index"
                )
                input_read_lines.extend((
                    "    { size_t logical_index;",
                    f"      for (logical_index = 0; logical_index < {count}; ++logical_index) {{",
                    f"        {c_type} element;",
                    f"        if (fread(&element, sizeof({c_type}), 1, state) != 1) {{",
                    f"          fprintf(stderr, \"short initial state at {_c_string(_source_name(parameter))[1:-1]}\\n\");",
                    "          return 4;",
                    "        }",
                    f"        (({c_type} *)slots[{index}])[{storage_index}] = element;",
                    "      }",
                    "    }",
                ))

    file_load_lines = []
    for port in file_ports:
        data_slot = slot_by_parameter[port["data"].name]
        length_slot = slot_by_parameter[port["length"].name]
        variable = _identifier("file_" + port["name"])
        file_load_lines.extend((
            f"    const char *{variable} = turing_argument_value(argc, argv, {_c_string(port['flag'])});",
            *(
                (f"    if ({variable} == NULL) {{ fprintf(stderr, \"missing {port['flag']}\\n\"); return 8; }}",)
                if not port["optional"] else ()
            ),
            f"    if ({variable} != NULL) {{",
            "        size_t loaded_bytes = 0;",
            f"        if (!turing_read_file({variable}, (uint8_t *)slots[{data_slot}], {port['capacity']}, &loaded_bytes)) return 9;",
            f"        *(({port['length'].c_type} *)slots[{length_slot}]) = ({port['length'].c_type})loaded_bytes;",
            "    }",
        ))

    feedback_lines = []
    feedback_finalize_lines = []
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
        swap = (
            f"{{ void *feedback_arena = slots[{input_slot}]; "
            f"slots[{input_slot}] = slots[{output_slot}]; "
            f"slots[{output_slot}] = feedback_arena; }}"
        )
        feedback_lines.append(f"        {swap}")
        # After the last frame the latest value is in the input address. Swap
        # once more so the public output name still denotes the final result
        # for serialization and caller inspection.
        feedback_finalize_lines.append(f"    {swap}")

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
        if len(tuple(parameter.shape or ())) <= 1:
            output_write_lines.append(
                f"    fwrite(slots[{slot}], sizeof({parameter.c_type}), {count}, outputs_file);"
            )
        else:
            storage_index = _fortran_storage_index(
                parameter, extents, "logical_index"
            )
            output_write_lines.extend((
                "    { size_t logical_index;",
                f"      for (logical_index = 0; logical_index < {count}; ++logical_index) {{",
                f"        const {parameter.c_type} *element = &(({parameter.c_type} *)slots[{slot}])[{storage_index}];",
                f"        fwrite(element, sizeof({parameter.c_type}), 1, outputs_file);",
                "      }",
                "    }",
            ))

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
        if int(display["frame_delay_ms"]) > 0:
            display_present_lines.append(
                f"        Sleep({int(display['frame_delay_ms'])});"
            )
        display_close_lines = ["    turing_display_close();"]

    source = "\n".join((
        _C_SOURCE,
        "",
        "#include <stdbool.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "",
        r'''#if defined(_WIN32)
/* GCC 16's MinGW static libgfortran uses the POSIX strndup entry point, while
 * an older Windows CRT does not export it. Keep the standalone runtime archive
 * resolvable without introducing another redistributable DLL. The definition is
 * WEAK: a newer mingw CRT (or the toolchain's own CRT shim) that supplies a
 * strong strndup overrides this one, so linking both never multiply-defines the
 * symbol; when nothing else provides it, this fills the reference. */
__attribute__((weak)) char *strndup(const char *source, size_t maximum) {
    size_t length = 0;
    char *copy;
    while (length < maximum && source[length] != '\0') ++length;
    copy = (char *)malloc(length + 1);
    if (copy == NULL) return NULL;
    memcpy(copy, source, length);
    copy[length] = '\0';
    return copy;
}
#endif
''',
        r'''static FILE *turing_open_artifact(
    const char *executable, const char *filename, const char *mode
) {
    char path[4096];
    const char *slash = strrchr(executable, '/');
    const char *backslash = strrchr(executable, '\\');
    const char *separator = slash;
    size_t directory_length;
    if (backslash != NULL && (separator == NULL || backslash > separator)) {
        separator = backslash;
    }
    if (separator == NULL) return fopen(filename, mode);
    directory_length = (size_t)(separator - executable + 1);
    if (directory_length + strlen(filename) + 1 > sizeof(path)) return NULL;
    memcpy(path, executable, directory_length);
    strcpy(path + directory_length, filename);
    return fopen(path, mode);
}
''',

        *(r'''static const char *turing_argument_value(int argc, char **argv, const char *flag) {
    int index;
    for (index = 2; index + 1 < argc; ++index) {
        if (strcmp(argv[index], flag) == 0) return argv[index + 1];
    }
    return NULL;
}

static int turing_read_file(
    const char *path, uint8_t *destination, size_t capacity, size_t *length
) {
    FILE *file = fopen(path, "rb");
    long size;
    if (file == NULL) { perror(path); return 0; }
    if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file); return 0;
    }
    if ((unsigned long long)size > (unsigned long long)capacity) {
        fprintf(stderr, "input file exceeds compiled port capacity: %s\n", path);
        fclose(file); return 0;
    }
    if (fread(destination, 1, (size_t)size, file) != (size_t)size) {
        fclose(file); return 0;
    }
    fclose(file);
    *length = (size_t)size;
    return 1;
}
''' if file_ports else "",),
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
        f"    FILE *state = turing_open_artifact(argv[0], {_c_string(initial_state_filename)}, \"rb\");",
        "    if (frames < 0) return 2;",
        "    if (!state) { perror(\"initial state\"); return 2; }",
        *allocation_lines,
        *file_load_lines,
        *input_read_lines,
        "    fclose(state);",
        *display_open_lines,
        "    turing_launch_stats_reset(&stats);",
        f"    for (frame = 0; {display_loop_condition}; ++frame) {{",
        *display_message_lines,
        "        if (turing_profiled_launch_ex(turing_fortran_compute, slots,",
        "                &profile, &stats, NULL, NULL, 3) != 1) return 5;",
        *display_present_lines,
        *feedback_lines,
        "    }",
        *display_close_lines,
        *feedback_finalize_lines,
        "    printf(\"{\\\"status\\\":%d,\\\"frames\\\":%d,\\\"shell_ns_total\\\":%llu,\\\"outputs\\\":{\",",
        "           profile.status, frame, stats.shell_ns_total);",
        *output_lines,
        "    printf(\"}}\\n\");",
        f"    {{ FILE *outputs_file = turing_open_artifact(argv[0], {_c_string(final_outputs_filename)}, \"wb\");",
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
    standalone: bool = True,
    library: bool = False,
) -> FortranCShellExecutable:
    """Compile generated Fortran plus the generic profiled C main.

    ``library=True`` instead builds a SHARED LIBRARY (.dll/.so) from just the
    Fortran module -- the compiled section exported for other programs to link
    against, "recognize without lowering". It skips the C-shell main and all of
    the runtime input/state machinery (a DLL of a section has no run harness and
    no initial state), so a parameterful section compiles without feeds.
    """

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
    file_ports = _system_file_configurations(module, entry)
    system_parameters = {
        parameter.name
        for port in file_ports
        for parameter in (port["data"], port["length"])
    }
    state_bytes = bytearray()
    if not library:
        missing = {
            _source_name(parameter)
            for parameter in input_parameters
            if parameter.name not in system_parameters
            if _source_name(parameter) not in inputs
        }
        if missing:
            raise ValueError(
                "missing C-shell inputs: " + ", ".join(sorted(missing))
            )

        for parameter in input_parameters:
            if parameter.name in system_parameters:
                continue
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

    if library:
        suffix = ".dll" if os.name == "nt" else ".so"
    else:
        suffix = ".exe" if os.name == "nt" else ""
    executable = output / f"{name}{suffix}"
    fortran_object = output / f"{name}.fortran.o"
    c_object = output / f"{name}.shell.o"
    environment = dict(os.environ)
    environment["PATH"] = (
        str(Path(compiler).parent) + os.pathsep + environment.get("PATH", "")
    )
    fortran_flags = aggressive_fortran_flags(compiler)
    c_flags = aggressive_c_flags(compiler)
    try:
        link_flags = (
            standalone_fortran_link_flags(compiler)
            if standalone else ("-flto",)
        )
    except ValueError as error:
        raise FortranEmissionError(str(error)) from error
    if library:
        # A shared library of the section: compile the Fortran module and link
        # it -shared, exporting its symbols. No C-shell main, no runtime input.
        commands = (
            [compiler, *fortran_flags, "-c", str(fortran_path), "-o", str(fortran_object)],
            [
                compiler, "-shared", "-o", str(executable), str(fortran_object),
                *standalone_runtime_shim_sources(compiler, output, standalone),
            ],
        )
    else:
        commands = (
            [compiler, *fortran_flags, "-c", str(fortran_path), "-o", str(fortran_object)],
            [gcc, *c_flags, "-std=c11", "-c", str(c_path), "-o", str(c_object)],
            [
                compiler, str(c_object), str(fortran_object),
                *standalone_runtime_shim_sources(compiler, output, standalone),
                "-o", str(executable),
                *link_flags,
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


def _field_slot_ops(graph_obj: Any):
    """Recover a method's instance-field accesses as slot loads and stores.

    A class's field layout is declared once (``class_table[class]['fields']``),
    giving each field a fixed slot. ``self`` is that field arena. This reads the
    process graph's field-op nodes and returns, for one method:

    * ``self_value_id`` -- the value id of the ``self`` arena, or ``None``;
    * ``field_ops`` -- ``(kind, value_id, slot)`` for every field access in the
      graph's own schedule order, ``kind`` being ``"read"`` (a ``GetAttr``, whose
      ``value_id`` is the result the method already consumes) or ``"write"`` (a
      ``setattr``, whose ``value_id`` is the stored source). Keeping reads and
      writes in one ordered list preserves their interleaving, so a store and a
      later read of one slot stay in the order the source wrote them;
    * ``field_count`` -- the arena length, so ``self`` is a sized array.
    """

    class_table = dict(graph_obj.graph.get("class_table") or {})
    owner = graph_obj.graph.get("method_owner")
    record = (
        class_table.get(owner)
        if owner in class_table
        else (next(iter(class_table.values())) if len(class_table) == 1 else None)
    )
    fields = tuple((record or {}).get("fields") or ())
    slot_of = {name: index for index, name in enumerate(fields)}

    identity = dict(graph_obj.graph.get("identity_table") or {})
    self_history = identity.get("self") or ()
    self_value_id = int(self_history[-1]) if self_history else None

    # Order field ops by SOURCE order (node id), not the data-dependency
    # schedule. Memory ordering between a write and a later read of the same
    # field is a real dependency the AST wrote but the graph does not carry as a
    # data edge, so a topological sort is free to float the read ahead of the
    # write. Nodes are created in source order, so their ids preserve the order
    # the programmer wrote -- which is the order the memory operations must run.
    field_ops: list[tuple[str, int, int]] = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        node_type = data.get("op") or data.get("type")
        attribute = (data.get("attributes") or {}).get("attribute")
        if attribute is None or attribute not in slot_of:
            continue
        if node_type == "GetAttr":
            result_id = data.get("value_id", node_id)
            field_ops.append(("read", int(result_id), slot_of[attribute]))
        elif node_type in ("setattr", "SetAttr"):
            source_parent = next(
                (
                    parent
                    for parent, role in (data.get("parents") or ())
                    if str(role) == "value"
                ),
                None,
            )
            if source_parent is None:
                continue
            source_id = graph_obj.nodes[source_parent].get(
                "value_id", source_parent
            )
            field_ops.append(("write", int(source_id), slot_of[attribute]))
    return self_value_id, tuple(field_ops), len(fields)


def _emit_class_surface_module(compilation: Any, artifact_name: str):
    """Emit every planned method of a whole object as one ``bind(C)`` library.

    This is the whole-object emission path and it performs NO numeric
    projection.  Each method lowers its own control program plus the operator
    regions the planner already carved out -- straight through
    ``lower_control_sections_to_ssa`` -- so a method with no numeric region (a
    void constructor) and a method with one (a ``mul``) lower the same way, and
    neither builds or validates a ``FusedProgram``.  Every method becomes its
    own linkable export; nothing is folded into a single entry and nothing is
    pruned.

    Returns ``(FortranModule, export_symbols)`` or ``(None, ())`` when the
    deployment exposes no planned methods (so the caller can fall through).
    """

    from ..transmogrifier.ssa import IRModule
    from .glsl_deployment_strategy import _walk_planned_shells
    from .precompile_to_ssa import lower_control_sections_to_ssa
    from .ssa_fortran_backend import emit_module
    from .string_table import StringTable

    # One table for the whole object: every method's string constants tokenize
    # into it, and it persists token -> word for reverse lookup.
    string_table = StringTable()

    all_functions: dict[str, Any] = {}
    section_outputs: dict[str, tuple[Any, ...]] = {}
    export_symbols: list[str] = []
    for shell in _walk_planned_shells(compilation.deployment):
        graph = getattr(shell, "process_graph", None)
        graph_obj = graph.G if graph is not None else None
        function_name = (
            graph_obj.graph.get("function_name") if graph_obj is not None else None
        )
        if function_name is None:
            continue
        control = getattr(shell, "shell_control_program", None)
        if control is None:
            continue
        symbol = f"{artifact_name}__{function_name}"
        # Instance fields flow through the object's field arena: ``self`` is a
        # slot array, a field read is a load from its slot, a field write a
        # store. In whole-program precompile mode the field-op region is never
        # built (gated behind ``not precompile_only``), so recover the field ops
        # from the process graph and hand them to the lowerer as slot access.
        self_id, field_ops, field_count = _field_slot_ops(graph_obj)
        module_ir, shortfalls, shell_section_outputs = (
            lower_control_sections_to_ssa(
                control,
                hierarchy_plan=getattr(shell, "hierarchy_plan", None),
                control_name=symbol,
                identity_table=dict(graph_obj.graph.get("identity_table") or {}),
                function_outputs=tuple(
                    graph_obj.graph.get("function_outputs") or ()
                ),
                function_parameters=tuple(
                    graph_obj.graph.get("function_parameters") or ()
                ),
                self_value_id=self_id,
                field_ops=field_ops,
                field_count=field_count,
                string_table=string_table,
            )
        )
        if shortfalls:
            raise FortranEmissionError(
                f"method {symbol!r} has operators without an SSA handler: "
                + "; ".join(f"{item.name} ({item.reason})" for item in shortfalls)
            )
        all_functions.update(module_ir.functions)
        section_outputs.update(shell_section_outputs)
        export_symbols.append(symbol)
    if not export_symbols:
        return None, ()
    # Persist token -> word so the emitted object's words are reversible.
    try:
        string_table.save()
    except Exception:  # noqa: BLE001 -- reverse-lookup cache, never fatal
        pass

    def emit_outputs(name: str, function: Any) -> tuple[Any, ...]:
        # A flat operator region has no explicit return: its outputs come from
        # the lowerer as ``intent(out)`` dummies the target appends. A control
        # function names its outputs with a return instruction.
        if name in section_outputs:
            return section_outputs[name]
        returns = tuple(
            instruction.args
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret", "Return", "return"}
        )
        return tuple(returns[-1]) if returns else ()

    emitted = emit_module(
        IRModule(all_functions),
        name=f"{artifact_name}_fortran",
        outputs={
            name: emit_outputs(name, function)
            for name, function in all_functions.items()
        },
        # A library exports its whole surface: keep and export every method and
        # region function, not just the ones one nominal entry reaches.
        extra_roots=tuple(all_functions),
    )
    if not emitted.complete:
        raise FortranEmissionError(
            "class surface could not emit hierarchical object program: "
            + "; ".join(item.format() for item in emitted.shortfalls)
        )
    return emitted, tuple(export_symbols)


def compile_ast_fortran_c_shell(
    source: str,
    entrypoint: str,
    feeds: Mapping[str, Any],
    directory: str | Path,
    *,
    python_bindings: Mapping[str, Any] | None = None,
    output_names: tuple[str, ...] | list[str] | None = None,
    state_feedback: Mapping[str, str] | None = None,
    display: Mapping[str, Any] | None = None,
    name: str | None = None,
    standalone: bool = True,
    progress: Callable[[str], None] | None = None,
    checkpoint: bool | str | Path = False,
    mutable_parameters: tuple[str, ...] | list[str] | set[str] = (),
    retain_card_program: bool = True,
    compilation: Any | None = None,
    library: bool = False,
    dependency_seeds: tuple[str, ...] = (),
    retain: Any = (),
) -> FortranCShellExecutable:
    """Compile Python AST through the registered Fortran target and C shell.

    ``library=True`` builds a shared library (.dll/.so) of the compiled section
    -- the section exported for other programs to link against -- instead of a
    standalone C-shell executable. See ``compile_fortran_module_c_shell``.

    This is the application-neutral native entrypoint.  It accepts authored
    Python, runs the ordinary ProcessGraph/AOT compiler, projects that
    compiler's public numerical program, and only then selects Fortran.
    Dotted aggregate feed names such as ``state.u`` are resolved from the
    caller's object without flattening or copying its arena in Python source.

    ``compilation`` lets a caller that already ran the whole-program no-bake
    ``compile_ast_aot`` (e.g. to first release the backend-neutral dual-IR
    checkpoint) hand that exact ``AOTCompilation`` in, so the Fortran shell
    runs the already-produced dual IR instead of compiling the program a
    second time.
    """

    from ..common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
        project_public_numerical_program,
    )
    from .machine_targets import get_target
    from .shell_io import (
        ShellIOBinding,
        ShellIOCapability,
        ShellIOManifest,
        ShellIORequest,
        attach_shell_io,
    )

    compilation = compilation if compilation is not None else compile_ast_aot(
        source,
        entrypoint,
        dict(feeds),
        backend="c",
        precompile_only=True,
        bake_mode="whole_program",
        python_bindings=dict(python_bindings or {}),
        progress=progress,
        checkpoint=checkpoint,
        mutable_parameters=tuple(mutable_parameters),
        dependency_seeds=tuple(dependency_seeds),
        retain=retain,
    )
    hierarchical_outputs = dict(compilation.public_output_value_ids)
    hierarchical_inputs = dict(compilation.public_input_value_ids)
    if output_names is not None and hierarchical_outputs:
        names = tuple(map(str, output_names))
        if set(hierarchical_outputs) <= set(names):
            pass
        elif len(names) != len(hierarchical_outputs):
            raise ValueError(
                f"received {len(names)} output names for "
                f"{len(hierarchical_outputs)} hierarchical outputs"
            )
        else:
            hierarchical_outputs = {
                output_name: value_id
                for output_name, value_id in zip(
                    names, hierarchical_outputs.values()
                )
            }

    artifact_name = str(name or entrypoint)
    module = None

    # Whole-object library build: emit every planned method as its own export
    # via the non-numeric control-sections path. A class has no program-level
    # return surface, so it never reaches the numeric emission below -- and it
    # must not, because that path projects and validates a numerical program the
    # object does not have. This early return skips all of the single-entry
    # native-input/card machinery, which does not apply to a multi-method
    # library.
    if library:
        class_module, export_symbols = _emit_class_surface_module(
            compilation, artifact_name
        )
        if class_module is not None:
            if progress is not None:
                progress(
                    f"emitted object surface {artifact_name}: "
                    f"exports {list(export_symbols)}"
                )
            return compile_fortran_module_c_shell(
                class_module,
                {},
                directory,
                entrypoint=export_symbols[0],
                name=artifact_name,
                standalone=standalone,
                library=True,
            )

    if hierarchical_outputs and compilation.region_programs:
        from ..transmogrifier.ssa import IRModule
        from .precompile_to_ssa import lower_precompile_and_control_to_ssa
        from .precompile_ssa_validator import (
            validate_precompile_ssa_compatibility,
        )
        from .ssa_fortran_backend import emit_module

        identity_table = {
            **dict(compilation.identity_table),
            **{
                source_name: (int(value_id),)
                for source_name, value_id in hierarchical_inputs.items()
            },
            **{
                source_name: (int(value_id),)
                for source_name, value_id in hierarchical_outputs.items()
            },
        }
        numerical_seed = next(
            (
                candidate
                for candidate in compilation.region_programs.values()
                if validate_precompile_ssa_compatibility(
                    candidate
                ).valid_precompile
            ),
            None,
        )
        if numerical_seed is None:
            raise FortranEmissionError(
                "hierarchical AST program has no structurally valid "
                "numerical region"
            )
        lowering = lower_precompile_and_control_to_ssa(
            numerical_seed,
            compilation.shell_control_program,
            region_programs=dict(compilation.region_programs),
            hierarchy_plan=getattr(compilation, "hierarchy_plan", None),
            numerical_name=f"{artifact_name}_discovery",
            control_name=artifact_name,
            identity_table=identity_table,
            function_outputs=tuple(hierarchical_outputs),
            function_parameters=tuple(hierarchical_inputs),
        )
        if lowering.shortfalls or not lowering.validation.valid_precompile:
            raise FortranEmissionError(
                lowering.shortfall_report()
                + f"; format_issues={lowering.validation.format_issues!r}"
            )
        # A single-entry build exports just its entry (plus its numeric
        # regions). A library build of a whole class exports EVERY function the
        # lowering produced -- each method is a linkable export -- so an app can
        # link any of them, not only the nominal entry.
        functions = {
            function_name: function
            for function_name, function in lowering.module.functions.items()
            if library
            or function_name == artifact_name
            or function_name.startswith("numerical_region_")
        }

        def returned_values(function: Any) -> tuple[Any, ...]:
            returns = tuple(
                instruction.args
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.op in {"Ret", "ret", "Return", "return"}
            )
            return tuple(returns[-1]) if returns else ()

        module = emit_module(
            IRModule(functions),
            name=f"{artifact_name}_fortran",
            outputs={
                function_name: returned_values(function)
                for function_name, function in functions.items()
            },
            # A library exports its whole surface: keep and export every section
            # function, not just the ones the entry reaches.
            extra_roots=tuple(functions) if library else (),
        )
        if not module.complete:
            raise FortranEmissionError(
                "Fortran target could not emit hierarchical AST program: "
                + "; ".join(item.format() for item in module.shortfalls)
            )

    program = project_public_numerical_program(compilation)
    if module is None and output_names is not None:
        names = tuple(map(str, output_names))
        if len(names) != len(program.outputs):
            metadata = program.meta or {}
            output_summary = tuple(
                (
                    output_name,
                    tuple(getattr(metadata.get(value_id), "shape", ()) or ()),
                )
                for output_name, value_id in program.outputs.items()
            )
            declared = {
                output_name: tuple(
                    compilation.identity_table.get(output_name, ())
                )
                for output_name in compilation.function_outputs
            }
            available = {
                *map(int, program.feeds),
                *(int(step.result_id) for step in program.steps),
                *map(int, program.outputs.values()),
            }
            raise ValueError(
                f"received {len(names)} output names for "
                f"{len(program.outputs)} compiled outputs; "
                f"declared={declared!r}; "
                f"declared_available={{{', '.join(f'{key!r}: {tuple(value in available for value in values)!r}' for key, values in declared.items())}}}; "
                f"first={output_summary[:16]!r}; last={output_summary[-16:]!r}"
            )
        program = replace(
            program,
            outputs={
                output_name: value_id
                for output_name, value_id in zip(
                    names, program.outputs.values()
                )
            },
        )
    if module is None:
        emitted = get_target("fortran").emit(program, name=artifact_name)
        if not emitted.complete or emitted.module is None:
            raise FortranEmissionError(
                "Fortran target could not emit compiled AST program: "
                + "; ".join(emitted.shortfalls)
            )
        module = emitted.module

    # Hierarchical lowering can promote a region-private feed into the public
    # control ABI after ``parameter_names`` was recorded.  Its SSA identity is
    # still present in the graph identity table, so restore the authored feed
    # name here.  Stateful shells can then declare feedback by program name
    # instead of depending on an unstable ``t<ID>`` spelling.
    feed_names_by_value_id: dict[int, str] = {}
    ambiguous_feed_ids: set[int] = set()
    candidate_feed_names = {
        str(name)
        for name in compilation.identity_table
        if str(name).split(".", 1)[0] in feeds
    }
    candidate_feed_names.update(map(str, feeds))
    for feed_name in candidate_feed_names:
        for value_id in compilation.identity_table.get(feed_name, ()):
            value_id = int(value_id)
            previous = feed_names_by_value_id.get(value_id)
            if previous is not None and previous != str(feed_name):
                ambiguous_feed_ids.add(value_id)
            else:
                feed_names_by_value_id[value_id] = str(feed_name)
    if feed_names_by_value_id:
        entry_points = []
        for described_entry in module.api.entry_points:
            described_parameters = []
            for parameter in described_entry.parameters:
                source_name = parameter.source_name
                if source_name is None and parameter.name.startswith("t"):
                    try:
                        value_id = int(parameter.name[1:])
                    except ValueError:
                        value_id = -1
                    if value_id not in ambiguous_feed_ids:
                        source_name = feed_names_by_value_id.get(value_id)
                described_parameters.append(
                    replace(parameter, source_name=source_name)
                )
            entry_points.append(replace(
                described_entry,
                parameters=tuple(described_parameters),
            ))
        module = replace(
            module,
            api=replace(module.api, entry_points=tuple(entry_points)),
        )
    if display is not None:
        options = dict(display)
        channels = tuple(
            map(str, options.pop("channels", ("red", "green", "blue")))
        )
        if channels != ("red", "green", "blue"):
            raise ValueError(
                "native rgb_f64_planar display requires red, green, blue"
            )
        manifest = ShellIOManifest(
            requests=(ShellIORequest.create(
                ShellIOCapability.DISPLAY,
                attributes={
                    "pixel_format": "rgb_f64_planar",
                    **options,
                },
            ),),
            bindings=tuple(
                ShellIOBinding(
                    f"display.{channel}", artifact_name, channel
                )
                for channel in channels
            ),
        )
        module = replace(module, api=attach_shell_io(module.api, manifest))

    def resolve_source_name(source_name: str) -> Any:
        if source_name in feeds:
            return feeds[source_name]
        root, *attributes = source_name.split(".")
        if root not in feeds:
            raise KeyError(source_name)
        value = feeds[root]
        for attribute in attributes:
            value = getattr(value, attribute)
        return value

    native_inputs: dict[str, Any] = {}
    public_input_names_by_id = {
        int(value_id): str(source_name)
        for source_name, value_id in (
            compilation.public_input_value_ids or {}
        ).items()
    }

    def resolve_compiled_value(value_id: int) -> Any:
        visited = set()
        current = int(value_id)
        while current not in visited:
            visited.add(current)
            if current in compilation.region_feed_values:
                return compilation.region_feed_values[current]
            source_name = public_input_names_by_id.get(current)
            if source_name is not None:
                return resolve_source_name(source_name)
            alias = compilation.hierarchical_value_aliases.get(current)
            if alias is None:
                break
            current = int(alias)
        raise KeyError(value_id)

    # A library build has no run harness and no initial state, so it needs no
    # concrete input values -- the section's parameters stay symbolic library
    # arguments. Skip resolving native inputs entirely.
    entry = module.api.entry_point(artifact_name)
    if not library:
        for parameter in entry.parameters:
            if parameter.role != "input":
                continue
            source_name = str(parameter.source_name or parameter.name)
            try:
                native_inputs[source_name] = resolve_source_name(source_name)
            except (AttributeError, KeyError) as error:
                if parameter.name.startswith("t"):
                    try:
                        value_id = int(parameter.name[1:])
                    except ValueError:
                        value_id = -1
                    try:
                        native_inputs[source_name] = resolve_compiled_value(
                            value_id
                        )
                    except (AttributeError, KeyError):
                        pass
                    else:
                        continue
                raise ValueError(
                    f"compiled input {source_name!r} ({parameter.shape!r}) "
                    "has no value in feeds or the captured region cache; "
                    f"endpoint={compilation.hierarchical_value_diagnostics.get(value_id)!r}"
                ) from error
    resolved_feedback = dict(state_feedback or {})
    abi_source_names = {
        str(parameter.source_name or parameter.name)
        for parameter in entry.parameters
        if parameter.role != "extent"
    }

    def canonical_hierarchy_value(value_id: int) -> int:
        visited = set()
        current = int(value_id)
        while current not in visited:
            visited.add(current)
            alias = compilation.hierarchical_value_aliases.get(current)
            if alias is None:
                return current
            current = int(alias)
        return current

    for input_name, output_name in tuple(resolved_feedback.items()):
        if output_name in abi_source_names or input_name not in abi_source_names:
            continue
        input_id = compilation.public_input_value_ids.get(input_name)
        output_id = compilation.public_output_value_ids.get(output_name)
        if output_id is None:
            history = tuple(compilation.identity_table.get(output_name, ()))
            output_id = history[-1] if history else None
        if (
            input_id is not None
            and output_id is not None
            and canonical_hierarchy_value(int(input_id))
            == canonical_hierarchy_value(int(output_id))
        ):
            # The declared function output is the same preallocated arena as
            # its input.  Fortran correctly publishes one inout ABI parameter
            # rather than allocating a copy-only output.  Point feedback at
            # that shared slot so the C shell preserves the alias contract.
            resolved_feedback[input_name] = input_name
    if retain_card_program:
        from .parametric_card_program import build_parametric_card_program

        card_public_inputs = dict(hierarchical_inputs)
        if not card_public_inputs:
            for parameter in entry.parameters:
                if parameter.role != "input" or not parameter.name.startswith("t"):
                    continue
                try:
                    value_id = int(parameter.name[1:])
                except ValueError:
                    continue
                card_public_inputs[
                    str(parameter.source_name or parameter.name)
                ] = value_id
        card_public_outputs = dict(hierarchical_outputs)
        if not card_public_outputs:
            card_public_outputs = {
                str(output_name): int(value_id)
                for output_name, value_id in program.outputs.items()
            }
        card_program = build_parametric_card_program(
            compilation,
            feedback=resolved_feedback,
            public_inputs=card_public_inputs,
            public_outputs=card_public_outputs,
        )
        module = replace(
            module,
            api=replace(
                module.api,
                metadata={
                    **dict(module.api.metadata or {}),
                    "card_program": card_program.to_mapping(),
                },
            ),
        )
    return compile_fortran_module_c_shell(
        module,
        native_inputs,
        directory,
        entrypoint=artifact_name,
        state_feedback=resolved_feedback,
        name=artifact_name,
        standalone=standalone,
        library=library,
    )


__all__ = [
    "FortranCShellExecutable",
    "compile_fortran_module_c_shell",
    "compile_ast_fortran_c_shell",
    "emit_fortran_c_shell_source",
]

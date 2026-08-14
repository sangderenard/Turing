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
import ast
import inspect
import importlib
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
    if not tuple(parameter.shape or ()) and parameter.extent is not None:
        return max(int(extents[str(parameter.extent)]), 1)
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
    # Pack runtime dependencies into the contract at compile time. The
    # producer knows them here -- the compiler's own bin directory is where a
    # gfortran-built library's support DLLs live -- and a consumer must never
    # rediscover them by loader archaeology (nodus boundary error register,
    # E15: a missing libgfortran presented as a silent LoadLibrary failure
    # that mimicked an ABI bug).
    api = module.api
    runtime_dependencies = []
    if os.name == "nt":
        toolchain_bin = Path(compiler).parent
        for dll_name in (
            "libgfortran-5.dll",
            "libquadmath-0.dll",
            "libgcc_s_seh-1.dll",
            "libwinpthread-1.dll",
        ):
            candidate = toolchain_bin / dll_name
            if candidate.exists():
                runtime_dependencies.append(
                    {"name": dll_name, "path": candidate.as_posix()}
                )
    if runtime_dependencies:
        metadata = dict(api.metadata)
        metadata["runtime_dependencies"] = runtime_dependencies
        api = replace(api, metadata=metadata)
    api.write(api_path)
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


def _field_slot_ops(
    graph_obj: Any,
    *,
    retained_storage_identities: frozenset[str] = frozenset(),
):
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
    const_sources: dict[int, Any] = {}
    sequence_initializations: list[tuple[int, str, int]] = []
    sequence_declarations: list[tuple[int, str, int, bool]] = []
    sequence_memberships: list[tuple[int, int, int, bool]] = []
    table_lookups: list[tuple[int, int | tuple[int, ...], int]] = []
    table_stores: list[tuple[int, int | tuple[int, ...], int, int]] = []
    table_deletions: list[
        tuple[int, int | tuple[int, ...], int | None, str]
    ] = []
    retained_sequence_ids: set[int] = set()
    nested_sequence_ids: set[int] = set()
    nested_record_fields: dict[int, tuple[str, int]] = {}
    field_aliases = dict(graph_obj.graph.get("class_field_aliases") or {})
    field_aggregate_kinds = dict(
        graph_obj.graph.get("class_field_aggregate_kinds") or {}
    )
    field_value_aggregate_kinds = dict(
        graph_obj.graph.get("class_field_value_aggregate_kinds") or {}
    )

    def canonical_field(name: str) -> str:
        seen: set[str] = set()
        current = str(name)
        while current in field_aliases and current not in seen:
            seen.add(current)
            current = str(field_aliases[current])
        return current

    # A method may mention only an alias (NetworkX ``_succ``) while its record
    # storage is authored under the canonical field (``_adj``). Correlate all
    # such GetAttr occurrences to one resident sequence ID; this is storage
    # aliasing, not value copying or object dispatch.
    field_sequence_ids: dict[str, int] = {}
    lexical_sequence_ids: dict[tuple[str, str], int] = {}
    inferred_nested_table_bases: dict[int, int] = {}
    aggregate_reads: list[tuple[str, str, int]] = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if (data.get("op") or data.get("type")) != "GetAttr":
            continue
        attribute = (data.get("attributes") or {}).get("attribute")
        if field_aggregate_kinds.get(str(attribute)) not in {
            "list", "set", "dict", "tuple", "bytes", "bytearray"
        }:
            continue
        aggregate_reads.append((
            str(attribute), canonical_field(str(attribute)),
            int(data.get("value_id", node_id)),
        ))
    for attribute, canonical, result_id in aggregate_reads:
        if attribute == canonical:
            field_sequence_ids[canonical] = result_id
    for _attribute, canonical, result_id in aggregate_reads:
        field_sequence_ids.setdefault(canonical, result_id)
    # Runtime aggregates captured from a lexical/module binding are resident
    # storage exactly like aggregate fields, but they have no record slot.
    # Correlate all normalized occurrences by their authored binding identity;
    # never inspect or serialize the bound Python collection's contents.
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        attributes = data.get("attributes") or {}
        aggregate_kind = attributes.get("aggregate_kind")
        binding_name = attributes.get("binding_name")
        binding_kind = attributes.get("binding_kind")
        if (
            aggregate_kind not in {
                "list", "set", "dict", "tuple", "bytes", "bytearray"
            }
            or binding_name is None
            or binding_kind not in {"parameter", "closure", "external"}
        ):
            continue
        identity_key = (str(binding_kind), str(binding_name))
        sequence_id = lexical_sequence_ids.setdefault(
            identity_key, int(data.get("value_id", node_id))
        )
        sequence_declarations.append((
            sequence_id,
            "unique" if aggregate_kind in {"set", "dict"} else "duplicates",
            2 if aggregate_kind == "dict" else 1,
            aggregate_kind not in {"tuple", "bytes"},
        ))
        storage_identity = f"{binding_kind}.{binding_name}"
        if storage_identity in retained_storage_identities:
            retained_sequence_ids.add(sequence_id)

    # An unannotated parameter can still state a complete aggregate contract
    # through authored operations.  ``key in p`` plus ``p[key]`` whose result
    # is iterated proves a keyed table whose values are child sequences.  This
    # derives storage from graph structure; it does not inspect parameter
    # contents or infer from a method's spelling.
    iterated_value_ids = {
        int(graph_obj.nodes[parent].get("value_id", parent))
        for node_id in graph_obj.nodes()
        for data in (graph_obj.nodes[node_id],)
        if (data.get("op") or data.get("type")) == "For"
        for parent, role in (data.get("parents") or ())
        if str(role) == "iterable" and parent in graph_obj
    }
    inferred_by_identity: dict[tuple[str, str], list[int]] = {}
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if (data.get("op") or data.get("type")) != "Indexed":
            continue
        result_id = int(data.get("value_id", node_id))
        if result_id not in iterated_value_ids:
            continue
        base_nodes = tuple(
            int(parent) for parent, role in (data.get("parents") or ())
            if str(role) == "base" and parent in graph_obj
        )
        if len(base_nodes) != 1:
            continue
        base_data = graph_obj.nodes[base_nodes[0]]
        base_attributes = base_data.get("attributes") or {}
        if base_attributes.get("binding_kind") != "parameter":
            continue
        binding_name = base_attributes.get("binding_name")
        if binding_name is None:
            continue
        inferred_by_identity.setdefault(
            ("parameter", str(binding_name)), []
        ).append(int(base_data.get("value_id", base_nodes[0])))
    for identity_key, observed_ids in inferred_by_identity.items():
        history = tuple(map(int, identity.get(identity_key[1], ())))
        sequence_id = int(history[0] if history else observed_ids[0])
        lexical_sequence_ids.setdefault(identity_key, sequence_id)
        sequence_declarations.append((sequence_id, "unique", 2, False))
        nested_sequence_ids.add(sequence_id)
        for value_id in (*history, *observed_ids):
            inferred_nested_table_bases[int(value_id)] = sequence_id

    # Locally constructed aggregates are resident storage too.  Previously the
    # record extractor declared only parameters, captures and object fields;
    # a local ``out = []`` or ``charmap = bytearray(256)`` therefore reached a
    # planned region as a shapeless scalar unless append/add happened to create
    # a descriptor as a side effect.  The source graph already owns the exact
    # producer identity, storage policy and writability, so publish that same
    # identity in the method sequence table.  Tuple values remain record rows
    # and are handled by projected-row/record lowering rather than pretending
    # the heterogeneous record itself is one homogeneous arena.
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        attributes = data.get("attributes") or {}
        aggregate_kind = attributes.get("aggregate_kind")
        if aggregate_kind not in {
            "list", "set", "dict", "bytes", "bytearray"
        }:
            continue
        sequence_id = int(data.get("value_id", node_id))
        sequence_declarations.append((
            sequence_id,
            "unique" if aggregate_kind in {"set", "dict"} else "duplicates",
            2 if aggregate_kind == "dict" else 1,
            aggregate_kind != "bytes",
        ))

    def table_sequence(base_id: int) -> tuple[int | None, str | None]:
        if base_id not in graph_obj:
            return None, None
        base_data = graph_obj.nodes[base_id]
        inferred_sequence_id = inferred_nested_table_bases.get(
            int(base_data.get("value_id", base_id))
        )
        if inferred_sequence_id is not None:
            return inferred_sequence_id, "parameter.inferred_nested_table"
        attributes = base_data.get("attributes") or {}
        field_name = attributes.get("attribute")
        if field_aggregate_kinds.get(str(field_name)) == "dict":
            canonical = canonical_field(str(field_name))
            return (
                field_sequence_ids[canonical],
                f"{owner}.{canonical}",
            )
        if attributes.get("aggregate_kind") != "dict":
            return None, None
        binding_name = attributes.get("binding_name")
        binding_kind = attributes.get("binding_kind")
        identity_key = (str(binding_kind), str(binding_name))
        if binding_name is None or identity_key not in lexical_sequence_ids:
            return None, None
        return (
            lexical_sequence_ids[identity_key],
            f"{binding_kind}.{binding_name}",
        )

    def authored_index_values(data: Any) -> int | tuple[int, ...] | None:
        values = tuple(
            int(graph_obj.nodes[parent].get("value_id", parent))
            for parent, role in (data.get("parents") or ())
            if str(role) == "index" and parent in graph_obj
        )
        if not values:
            return None
        return values[0] if len(values) == 1 else values

    def deletes_first_live_key(data: Any) -> bool:
        expression = data.get("expr_obj")
        if not isinstance(expression, ast.Subscript):
            return False
        key = expression.slice
        return (
            isinstance(key, ast.Call)
            and isinstance(key.func, ast.Name)
            and key.func.id == "next"
            and len(key.args) == 1
            and isinstance(key.args[0], ast.Call)
            and isinstance(key.args[0].func, ast.Name)
            and key.args[0].func.id == "iter"
            and len(key.args[0].args) == 1
            and ast.dump(key.args[0].args[0], include_attributes=False)
            == ast.dump(expression.value, include_attributes=False)
        )
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        node_type = data.get("op") or data.get("type")
        attribute = (data.get("attributes") or {}).get("attribute")
        canonical_attribute = (
            canonical_field(str(attribute)) if attribute is not None else None
        )
        if canonical_attribute is None or canonical_attribute not in slot_of:
            continue
        if node_type == "GetAttr":
            result_id = data.get("value_id", node_id)
            field_ops.append((
                "read", int(result_id), slot_of[canonical_attribute]
            ))
            aggregate_kind = field_aggregate_kinds.get(str(attribute))
            if aggregate_kind in {
                "list", "set", "dict", "tuple", "bytes", "bytearray"
            }:
                sequence_id = field_sequence_ids[canonical_field(str(attribute))]
                if int(result_id) == sequence_id:
                    sequence_declarations.append((
                        sequence_id,
                        "unique" if aggregate_kind in {"set", "dict"} else "duplicates",
                        2 if aggregate_kind == "dict" else 1,
                        aggregate_kind not in {"tuple", "bytes"},
                    ))
                storage_identity = f"{owner}.{canonical_field(str(attribute))}"
                if storage_identity in retained_storage_identities:
                    retained_sequence_ids.add(sequence_id)
                if field_value_aggregate_kinds.get(str(attribute)) == "dict":
                    nested_sequence_ids.add(sequence_id)
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
            source_data = graph_obj.nodes[source_parent]
            source_id = source_data.get("value_id", source_parent)
            field_ops.append((
                "write", int(source_id), slot_of[canonical_attribute]
            ))
            source_attributes = source_data.get("attributes") or {}
            nested_class_identity = source_attributes.get("class_ref")
            if nested_class_identity is not None:
                nested_record_fields[slot_of[canonical_attribute]] = (
                    str(nested_class_identity), int(source_id)
                )
            aggregate_kind = (
                source_attributes.get("aggregate_kind")
                or field_aggregate_kinds.get(str(attribute))
            )
            if (
                aggregate_kind in {"list", "set", "dict", "bytearray"}
                and attribute not in field_aliases
            ):
                sequence_initializations.append((
                    int(source_id),
                    "unique" if aggregate_kind in {"set", "dict"} else "duplicates",
                    2 if aggregate_kind == "dict" else 1,
                ))
                storage_identity = f"{owner}.{canonical_field(str(attribute))}"
                if storage_identity in retained_storage_identities:
                    retained_sequence_ids.add(int(source_id))
                if field_value_aggregate_kinds.get(str(attribute)) == "dict":
                    nested_sequence_ids.add(int(source_id))
            # A constant field write (``self.x = None`` / ``5`` / ``"s"``) has
            # no producer in the control body, so carry the constant value; the
            # injection materialises it before the store (None becomes the
            # absence sentinel via the tokenizer).
            if (source_data.get("op") or source_data.get("type")) in (
                "const",
                "Constant",
            ):
                attrs = source_data.get("attributes") or {}
                const_sources[int(source_id)] = attrs.get(
                    "value", source_data.get("constant")
                )
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Compare)
            and len(expression.ops) == 1
            and isinstance(expression.ops[0], (ast.In, ast.NotIn))
        ):
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        query_id = by_role.get("lhs")
        field_id = by_role.get("rhs")
        if query_id is None or field_id is None or field_id not in graph_obj:
            continue
        field_data = graph_obj.nodes[field_id]
        field_attributes = field_data.get("attributes") or {}
        field_name = field_attributes.get("attribute")
        sequence_id = None
        if field_aggregate_kinds.get(str(field_name)) in {"dict", "set"}:
            sequence_id = field_sequence_ids[canonical_field(str(field_name))]
        elif field_attributes.get("aggregate_kind") in {"dict", "set"}:
            identity_key = (
                str(field_attributes.get("binding_kind")),
                str(field_attributes.get("binding_name")),
            )
            sequence_id = lexical_sequence_ids.get(identity_key)
        elif int(field_data.get("value_id", field_id)) in (
            inferred_nested_table_bases
        ):
            sequence_id = inferred_nested_table_bases[
                int(field_data.get("value_id", field_id))
            ]
        if sequence_id is None:
            continue
        sequence_memberships.append((
            int(data.get("value_id", node_id)),
            int(graph_obj.nodes[query_id].get("value_id", query_id)),
            int(sequence_id),
            isinstance(expression.ops[0], ast.NotIn),
        ))
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if (data.get("op") or data.get("type")) != "Indexed":
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        base_id = by_role.get("base")
        key_id = authored_index_values(data)
        if base_id is None or key_id is None or base_id not in graph_obj:
            continue
        sequence_id, _storage_identity = table_sequence(base_id)
        if sequence_id is None:
            continue
        table_lookups.append((
            int(data.get("value_id", node_id)),
            key_id,
            int(sequence_id),
        ))
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if (data.get("op") or data.get("type")) != "IndexedStore":
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        base_id = by_role.get("base")
        key_id = authored_index_values(data)
        value_id = by_role.get("value")
        if None in (base_id, key_id, value_id) or base_id not in graph_obj:
            continue
        sequence_id, _storage_identity = table_sequence(base_id)
        if sequence_id is None:
            continue
        table_stores.append((
            int(data.get("value_id", node_id)),
            key_id,
            int(graph_obj.nodes[value_id].get("value_id", value_id)),
            int(sequence_id),
        ))
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if str(data.get("op") or data.get("type")).lower() != "delitem":
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        base_id = by_role.get("base")
        key_id = (() if deletes_first_live_key(data)
                  else authored_index_values(data))
        if base_id is None or key_id is None or base_id not in graph_obj:
            continue
        sequence_value_id, storage_identity = table_sequence(base_id)
        if storage_identity is None:
            storage_identity = f"nested-table-value:{base_id}"
        table_deletions.append((
            int(data.get("value_id", node_id)),
            key_id,
            sequence_value_id,
            storage_identity,
        ))
    # Runtime sequence replication (``[x] * count``) is a fill of resident
    # caller storage, not numerical multiplication and not a Python literal.
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        attributes = data.get("attributes") or {}
        if attributes.get("producer_kind") != "sequence_replication":
            continue
        parents = tuple(data.get("parents") or ())
        sequence_parents = tuple(
            int(parent) for parent, role in parents
            if str(role) in {"lhs", "rhs"}
            and parent in graph_obj
            and (graph_obj.nodes[parent].get("attributes") or {}).get(
                "aggregate_kind"
            ) in {"list", "tuple"}
        )
        fill_ids = tuple(
            int(leaf_id)
            for parent in sequence_parents
            for leaf_id in (
                graph_obj.nodes[parent].get("attributes") or {}
            ).get("aggregate_leaf_value_ids", ())
        )
        count_ids = tuple(
            int(parent) for parent, role in parents
            if str(role) in {"lhs", "rhs"}
            and int(parent) not in sequence_parents
        )
        if len(fill_ids) != 1 or len(count_ids) != 1:
            continue
        fill_data = graph_obj.nodes.get(fill_ids[0], {})
        literal = (fill_data.get("attributes") or {}).get(
            "value", fill_data.get("constant")
        )
        if literal is not None and not isinstance(literal, (bool, int, float)):
            continue
        result_id = int(data.get("value_id", node_id))
        sequence_initializations.append((
            result_id,
            f"fill={literal!r};count={count_ids[0]}",
            1,
        ))
    key_width_by_sequence: dict[int, int] = {}
    for _effect_or_result, key_ids, sequence_id, *_rest in (
        *table_lookups, *table_stores, *table_deletions
    ):
        if sequence_id is None:
            continue
        key_width_by_sequence[int(sequence_id)] = max(
            key_width_by_sequence.get(int(sequence_id), 1),
            len(key_ids) if isinstance(key_ids, tuple) else 1,
        )
    sequence_declarations = [
        (
            sequence_id,
            policy,
            (
                max(
                    column_count,
                    key_width_by_sequence.get(sequence_id, 1) + 1,
                )
                if policy == "unique" and column_count > 1
                else column_count
            ),
            writable,
        )
        for sequence_id, policy, column_count, writable
        in sequence_declarations
    ]
    return (
        self_value_id,
        tuple(field_ops),
        const_sources,
        len(fields),
        fields,
        owner,
        tuple(dict.fromkeys(sequence_initializations)),
        tuple(
            (slot_of[alias], slot_of[target])
            for alias, target in field_aliases.items()
            if alias in slot_of and target in slot_of
        ),
        tuple(dict.fromkeys(sequence_declarations)),
        tuple(dict.fromkeys(sequence_memberships)),
        tuple(dict.fromkeys(table_lookups)),
        tuple(dict.fromkeys(table_stores)),
        tuple(dict.fromkeys(table_deletions)),
        tuple(sorted(retained_sequence_ids)),
        tuple(sorted(nested_sequence_ids)),
        tuple(sorted(
            (slot, identity, value_id)
            for slot, (identity, value_id) in nested_record_fields.items()
        )),
    )


def _sequence_augassign_ops(graph_obj: Any) -> tuple[tuple[int, int, int], ...]:
    """Return proven ``sequence += sequence`` storage mutations.

    The graph's identity history correlates each lexical spelling with its
    successive SSA versions.  Resolve the AugAssign destination back to the
    resident aggregate producer bearing that spelling, and do the same for the
    source.  No operation-name inference is involved: both endpoints must be
    graph nodes already carrying an aggregate storage contract.
    """

    sequence_kinds = {"list", "set", "dict", "bytes", "bytearray"}
    identity = dict(graph_obj.graph.get("identity_table") or {})
    resident_by_value: dict[int, int] = {}
    for _name, history in identity.items():
        residents = tuple(
            int(value_id)
            for value_id in history
            if int(value_id) in graph_obj
            and (graph_obj.nodes[int(value_id)].get("attributes") or {}).get(
                "aggregate_kind"
            ) in sequence_kinds
        )
        if not residents:
            continue
        resident = residents[0]
        for value_id in history:
            resident_by_value[int(value_id)] = resident

    # Conditional/loop joins preserve the authored storage identity when every
    # incoming value already denotes that same resident arena.  Carry that
    # alias through Phi/LoopResult nodes to later AugAssign occurrences; never
    # choose an arbitrary branch when the residents differ or are unresolved.
    changed = True
    while changed:
        changed = False
        for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
            if int(node_id) in resident_by_value:
                continue
            data = graph_obj.nodes[node_id]
            if str(data.get("type") or data.get("op")).lower() not in {
                "phi", "loopresult"
            }:
                continue
            incoming = {
                resident_by_value[int(parent)]
                for parent, role in (data.get("parents") or ())
                if str(role) in {"body", "orelse", "initial", "updated", "value"}
                and int(parent) in resident_by_value
            }
            unresolved = any(
                str(role) in {"body", "orelse", "initial", "updated", "value"}
                and int(parent) not in resident_by_value
                for parent, role in (data.get("parents") or ())
            )
            if len(incoming) == 1 and not unresolved:
                resident_by_value[int(node_id)] = incoming.pop()
                changed = True

    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.AugAssign)
            and isinstance(expression.op, ast.Add)
            and isinstance(expression.target, ast.Name)
        ):
            continue
        parents = {
            str(role): int(parent) for parent, role in (data.get("parents") or ())
        }
        result_id = int(data.get("value_id", node_id))
        destination_id = resident_by_value.get(parents.get("lhs", -1))
        source_id = resident_by_value.get(parents.get("rhs", -1))
        if destination_id is None or source_id is None:
            continue
        operations.append((result_id, destination_id, source_id))
    return tuple(operations)


def _sequence_append_fill_ops(
    graph_obj: Any,
) -> tuple[
    tuple[int, int, int | float | bool | None, int, int], ...
]:
    """Return exact ``resident += literal_sequence * runtime_count`` rows."""

    sequence_kinds = {"list", "bytes", "bytearray"}
    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.AugAssign)
            and isinstance(expression.op, ast.Add)
        ):
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        lhs_id = by_role.get("lhs")
        rhs_id = by_role.get("rhs")
        if lhs_id is None or rhs_id is None:
            continue
        lhs = graph_obj.nodes.get(lhs_id, {})
        rhs = graph_obj.nodes.get(rhs_id, {})
        if (lhs.get("attributes") or {}).get(
            "aggregate_kind"
        ) not in sequence_kinds:
            continue
        if str(rhs.get("type") or rhs.get("op")) not in {"Mul", "Mult"}:
            continue
        operands = {
            str(role): int(parent)
            for parent, role in (rhs.get("parents") or ())
        }
        literal_id = operands.get("lhs")
        count_id = operands.get("rhs")
        if literal_id is None or count_id is None:
            continue
        literal_data = graph_obj.nodes.get(literal_id, {})
        literal = (literal_data.get("attributes") or {}).get(
            "value", literal_data.get("constant")
        )
        if isinstance(literal, (bytes, bytearray)) and len(literal) == 1:
            literal = int(literal[0])
        elif not isinstance(literal, (bool, int, float)) and literal is not None:
            continue
        operations.append((
            int(data.get("value_id", node_id)),
            int(lhs.get("value_id", lhs_id)),
            literal,
            int(graph_obj.nodes[count_id].get("value_id", count_id)),
            int(rhs.get("value_id", rhs_id)),
        ))
    return tuple(operations)


def _sequence_append_slice_ops(
    graph_obj: Any,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    """Return exact ``resident += resident[lower:upper]`` mutations.

    Only an authored unit-stride slice with explicit lower and upper values is
    admitted here.  The SSA helper performs Python-compatible bound clipping;
    other slice forms remain in the ordinary lowering ledger.
    """

    sequence_kinds = {"list", "bytes", "bytearray"}
    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.AugAssign)
            and isinstance(expression.op, ast.Add)
        ):
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        lhs_id = by_role.get("lhs")
        rhs_id = by_role.get("rhs")
        if lhs_id is None or rhs_id is None:
            continue
        lhs = graph_obj.nodes.get(lhs_id, {})
        rhs = graph_obj.nodes.get(rhs_id, {})
        if (lhs.get("attributes") or {}).get(
            "aggregate_kind"
        ) not in sequence_kinds:
            continue
        if str(rhs.get("type") or rhs.get("op")) != "Indexed":
            continue
        indexed = {
            str(role): int(parent)
            for parent, role in (rhs.get("parents") or ())
        }
        source_id = indexed.get("base")
        slice_id = indexed.get("index")
        if source_id is None or slice_id is None:
            continue
        source = graph_obj.nodes.get(source_id, {})
        slice_data = graph_obj.nodes.get(slice_id, {})
        if (source.get("attributes") or {}).get(
            "aggregate_kind"
        ) not in sequence_kinds:
            continue
        if str(slice_data.get("type") or slice_data.get("op")) != "Slice":
            continue
        bounds = {
            str(role): int(parent)
            for parent, role in (slice_data.get("parents") or ())
        }
        lower_id = bounds.get("lower")
        upper_id = bounds.get("upper")
        if lower_id is None or upper_id is None or "step" in bounds:
            continue
        operations.append((
            int(data.get("value_id", node_id)),
            int(lhs.get("value_id", lhs_id)),
            int(source.get("value_id", source_id)),
            int(graph_obj.nodes[lower_id].get("value_id", lower_id)),
            int(graph_obj.nodes[upper_id].get("value_id", upper_id)),
            int(rhs.get("value_id", rhs_id)),
        ))
    return tuple(operations)


def _sequence_bit_pack_ops(
    graph_obj: Any,
) -> tuple[tuple[int, int, int, tuple[int, ...]], ...]:
    """Recognize complete translate/reverse/fixed-bit-word materialization.

    This is the structural meaning of ``_mk_bitmap``: a 0/1 byte arena is
    packed little-endian into fixed-width words.  Recognition follows the AST
    dataflow (translate -> reverse slice -> fixed slice -> int(base=2) ->
    list comprehension), never the function name.
    """

    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        result = graph_obj.nodes[node_id]
        if (result.get("attributes") or {}).get(
            "aggregate_kind"
        ) != "list":
            continue
        expression = result.get("expr_obj")
        if not isinstance(expression, ast.ListComp) or len(expression.generators) != 1:
            continue
        element_parents = tuple(
            int(parent) for parent, role in (result.get("parents") or ())
            if str(role) == "elt" and parent in graph_obj
        )
        if len(element_parents) != 1:
            continue
        conversion = graph_obj.nodes[element_parents[0]]
        conversion_expression = conversion.get("expr_obj")
        if not (
            isinstance(conversion_expression, ast.Call)
            and len(conversion_expression.args) == 2
            and isinstance(conversion_expression.args[1], ast.Constant)
            and conversion_expression.args[1].value == 2
        ):
            continue
        conversion_roles = {
            str(role): int(parent)
            for parent, role in (conversion.get("parents") or ())
        }
        slice_value_id = conversion_roles.get("arg:0")
        if slice_value_id is None or slice_value_id not in graph_obj:
            continue
        sliced = graph_obj.nodes[slice_value_id]
        sliced_roles = {
            str(role): int(parent)
            for parent, role in (sliced.get("parents") or ())
        }
        reversed_id = sliced_roles.get("base")
        inner_slice_id = sliced_roles.get("index")
        if reversed_id is None or inner_slice_id is None:
            continue
        reversed_value = graph_obj.nodes[reversed_id]
        reversed_roles = {
            str(role): int(parent)
            for parent, role in (reversed_value.get("parents") or ())
        }
        translated_id = reversed_roles.get("base")
        reverse_slice_id = reversed_roles.get("index")
        if translated_id is None or reverse_slice_id is None:
            continue
        translated = graph_obj.nodes[translated_id]
        translated_roles = {
            str(role): int(parent)
            for parent, role in (translated.get("parents") or ())
        }
        source_id = translated_roles.get("operand")
        if source_id is None:
            continue
        source = graph_obj.nodes[source_id]
        if (source.get("attributes") or {}).get(
            "aggregate_kind"
        ) not in {"bytes", "bytearray", "list"}:
            continue
        reverse_slice = graph_obj.nodes[reverse_slice_id]
        reverse_step = tuple(
            int(parent) for parent, role in (reverse_slice.get("parents") or ())
            if str(role) == "step"
        )
        if len(reverse_step) != 1:
            continue
        step_expression = graph_obj.nodes[reverse_step[0]].get("expr_obj")
        if not (
            isinstance(step_expression, ast.UnaryOp)
            and isinstance(step_expression.op, ast.USub)
            and isinstance(step_expression.operand, ast.Constant)
            and step_expression.operand.value == 1
        ):
            continue
        inner_slice = graph_obj.nodes[inner_slice_id]
        lower_ids = tuple(
            int(parent) for parent, role in (inner_slice.get("parents") or ())
            if str(role) == "lower"
        )
        if len(lower_ids) != 1:
            continue
        lower = graph_obj.nodes[lower_ids[0]]
        lower_roles = {
            str(role): int(parent)
            for parent, role in (lower.get("parents") or ())
        }
        width_node_id = lower_roles.get("rhs")
        if width_node_id is None:
            continue
        width_id = int(graph_obj.nodes[width_node_id].get(
            "value_id", width_node_id
        ))
        consumed = tuple(dict.fromkeys((
            int(translated.get("value_id", translated_id)),
            int(reversed_value.get("value_id", reversed_id)),
            int(lower.get("value_id", lower_ids[0])),
            int(sliced.get("value_id", slice_value_id)),
            int(result.get("value_id", node_id)),
        )))
        operations.append((
            int(result.get("value_id", node_id)),
            int(source.get("value_id", source_id)),
            width_id,
            consumed,
        ))
    return tuple(operations)


def _sequence_prepend_concat_ops(
    graph_obj: Any,
) -> tuple[tuple[int, int, int, int, int], ...]:
    """Recognize ``[scalar] + sequence`` consumed by ``base[0:0] = ...``."""

    sequence_kinds = {"list", "bytearray"}
    resident_by_value: dict[int, int] = {}
    for history in (graph_obj.graph.get("identity_table") or {}).values():
        residents = tuple(
            int(value_id) for value_id in history
            if int(value_id) in graph_obj
            and (graph_obj.nodes[int(value_id)].get("attributes") or {}).get(
                "aggregate_kind"
            ) in sequence_kinds
        )
        if not residents:
            continue
        resident = residents[0]
        for value_id in history:
            resident_by_value[int(value_id)] = resident
    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        store = graph_obj.nodes[node_id]
        if str(store.get("type") or store.get("op")) != "IndexedStore":
            continue
        roles = {
            str(role): int(parent)
            for parent, role in (store.get("parents") or ())
        }
        base_id = roles.get("base")
        index_id = roles.get("index")
        value_id = roles.get("value")
        if None in (base_id, index_id, value_id):
            continue
        base = graph_obj.nodes[base_id]
        index = graph_obj.nodes[index_id]
        concatenation = graph_obj.nodes[value_id]
        resident_id = resident_by_value.get(
            int(base.get("value_id", base_id))
        )
        if resident_id is None:
            continue
        bounds = {
            str(role): int(parent)
            for parent, role in (index.get("parents") or ())
        }
        if set(bounds) != {"lower", "upper"}:
            continue
        bound_values = []
        for bound_id in bounds.values():
            bound_data = graph_obj.nodes[bound_id]
            bound_values.append((bound_data.get("attributes") or {}).get(
                "value", bound_data.get("constant")
            ))
        if bound_values != [0, 0]:
            continue
        if str(concatenation.get("type") or concatenation.get("op")) != "Add":
            continue
        concat_roles = {
            str(role): int(parent)
            for parent, role in (concatenation.get("parents") or ())
        }
        singleton_id = concat_roles.get("lhs")
        tail_id = concat_roles.get("rhs")
        if singleton_id is None or tail_id is None:
            continue
        singleton = graph_obj.nodes[singleton_id]
        leaves = tuple((singleton.get("attributes") or {}).get(
            "aggregate_leaf_value_ids", ()
        ))
        if (
            (singleton.get("attributes") or {}).get("aggregate_kind") != "list"
            or len(leaves) != 1
        ):
            continue
        operations.append((
            int(store.get("value_id", node_id)),
            int(resident_id),
            int(leaves[0]),
            int(concatenation.get("value_id", value_id)),
            int(graph_obj.nodes[tail_id].get("value_id", tail_id)),
        ))
    return tuple(operations)


def _sequence_prepend_packed_call_ops(
    graph_obj: Any,
) -> tuple[tuple[int, int, int, int, int, int, int], ...]:
    """Correlate prefix splice with the pursued byte-packing call edge."""

    sequence_kinds = {"list", "bytes", "bytearray"}
    resident_by_value: dict[int, int] = {}
    for history in (graph_obj.graph.get("identity_table") or {}).values():
        residents = tuple(
            int(value_id) for value_id in history
            if int(value_id) in graph_obj
            and (graph_obj.nodes[int(value_id)].get("attributes") or {}).get(
                "aggregate_kind"
            ) in sequence_kinds
        )
        if not residents:
            continue
        resident = residents[0]
        for value_id in history:
            resident_by_value[int(value_id)] = resident
    operations = []
    for (
        store_result_id, destination_id, prefix_id, concat_result_id,
        tail_result_id,
    ) in _sequence_prepend_concat_ops(graph_obj):
        tail_node = next((
            (node_id, data)
            for node_id, data in graph_obj.nodes(data=True)
            if int(data.get("value_id", node_id)) == int(tail_result_id)
        ), None)
        if tail_node is None:
            continue
        _tail_node_id, tail = tail_node
        attributes = tail.get("attributes") or {}
        if attributes.get("callee_ref") is None:
            continue
        arguments = tuple(
            int(graph_obj.nodes[parent].get("value_id", parent))
            for parent, role in (tail.get("parents") or ())
            if str(role).startswith(("arg:", "kw:"))
        )
        if len(arguments) != 1:
            continue
        source_id = resident_by_value.get(int(arguments[0]))
        if source_id is None:
            continue
        operations.append((
            int(store_result_id), int(destination_id), int(prefix_id),
            int(source_id), int(concat_result_id), int(tail_result_id),
            int(_tail_node_id),
        ))
    return tuple(dict.fromkeys(operations))


def _sequence_inplace_bit_pack_call_ops(
    graph: Any,
) -> tuple[tuple[int, int, int, int, int], ...]:
    """Return pursued calls whose callee is a structural bit-pack program."""

    graph_obj = graph.G
    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        return ()
    sequence_kinds = {"list", "bytes", "bytearray"}
    resident_by_value: dict[int, int] = {}
    for history in (graph_obj.graph.get("identity_table") or {}).values():
        residents = tuple(
            int(value_id) for value_id in history
            if int(value_id) in graph_obj
            and (graph_obj.nodes[int(value_id)].get("attributes") or {}).get(
                "aggregate_kind"
            ) in sequence_kinds
        )
        if not residents:
            continue
        resident = residents[0]
        for value_id in history:
            resident_by_value[int(value_id)] = resident
    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        call = graph_obj.nodes[node_id]
        attributes = call.get("attributes") or {}
        reference = attributes.get("callee_ref")
        if reference is None:
            continue
        try:
            callee = function_table.entry(int(reference)).graph
        except (KeyError, TypeError, ValueError):
            continue
        if callee is None:
            continue
        contracts = _sequence_bit_pack_ops(callee.G)
        if len(contracts) != 1:
            continue
        _callee_destination, _callee_source, _width_id, _consumed = contracts[0]
        arguments = tuple(
            int(graph_obj.nodes[parent].get("value_id", parent))
            for parent, role in (call.get("parents") or ())
            if str(role).startswith(("arg:", "kw:"))
        )
        if len(arguments) != 1:
            continue
        resident_id = resident_by_value.get(int(arguments[0]))
        if resident_id is None:
            continue
        entry = function_table.entry(int(reference))
        callable_object = getattr(entry, "python_callable", None)
        if callable_object is None and "." in str(entry.qualified_name):
            parts = str(entry.qualified_name).split(".")
            for split in range(len(parts) - 1, 0, -1):
                try:
                    candidate = importlib.import_module(".".join(parts[:split]))
                except ImportError:
                    continue
                try:
                    for attribute in parts[split:]:
                        candidate = getattr(candidate, attribute)
                except AttributeError:
                    continue
                callable_object = candidate
                break
        if callable_object is None:
            continue
        try:
            signature = inspect.signature(callable_object)
        except (TypeError, ValueError):
            continue
        parameters = tuple(signature.parameters.values())
        if len(parameters) < 2:
            continue
        width_default = parameters[1].default
        if not isinstance(width_default, int) or width_default <= 0:
            continue
        operations.append((
            int(call.get("value_id", node_id)),
            int(resident_id),
            int(width_default),
            int(reference),
            int(node_id),
        ))
    return tuple(dict.fromkeys(operations))


def _nested_row_projection_ops(
    graph_obj: Any, control: Any,
) -> tuple[tuple[int, int, int, str], ...]:
    """Find fixed-column reads from a projected/destructured loop row."""

    aliases_by_value: dict[int, frozenset[int]] = {}
    for history in (graph_obj.graph.get("identity_table") or {}).values():
        values = frozenset(map(int, history))
        for value_id in values:
            aliases_by_value[int(value_id)] = values
    operations = []
    for _iterable, target_id, induction, _projection in (
        getattr(control, "projected_iterable_bindings", ())
    ):
        aliases = aliases_by_value.get(
            int(target_id), frozenset((int(target_id),))
        )
        for node_id, data in graph_obj.nodes(data=True):
            if str(data.get("type")) not in {"Indexed", "indexed"}:
                continue
            roles = {
                str(role): int(parent)
                for parent, role in (data.get("parents") or ())
            }
            base = roles.get("base")
            index = roles.get("index")
            if base not in aliases or index is None or index not in graph_obj:
                continue
            index_data = graph_obj.nodes[index]
            expression = index_data.get("expr_obj")
            literal = (
                expression.value
                if isinstance(expression, ast.Constant)
                else (index_data.get("attributes") or {}).get(
                    "value", index_data.get("value")
                )
            )
            if not isinstance(literal, int) or isinstance(literal, bool):
                continue
            operations.append((
                int(target_id), int(literal),
                int(data.get("value_id", node_id)), str(induction),
            ))
    return tuple(dict.fromkeys(operations))


def _class_surface_ssa_program(
    compilation: Any,
    artifact_name: str,
    *,
    tensor_ssa_reference: Any = None,
):
    """Lower every planned method of a whole object to one reusable SSA unit.

    This is the whole-object emission path and it performs NO numeric
    projection.  Each method lowers its own control program plus the operator
    regions the planner already carved out -- straight through
    ``lower_control_sections_to_ssa`` -- so a method with no numeric region (a
    void constructor) and a method with one (a ``mul``) lower the same way, and
    neither builds or validates a ``FusedProgram``.  Every method becomes its
    own linkable export; nothing is folded into a single entry and nothing is
    pruned.

    This boundary deliberately precedes Fortran emission so the fully lowered
    whole-object SSA can be inspected and verified before target emission.
    """

    from ..transmogrifier.ssa import (
        IRModule, SSAMachineControlTable, SSAMachineIndirectTable,
    )
    from .glsl_deployment_strategy import _walk_planned_shells
    from .precompile_to_ssa import lower_control_sections_to_ssa
    from .string_table import StringTable

    # One table for the whole object: every method's string constants tokenize
    # into it, and it persists token -> word for reverse lookup.
    string_table = StringTable()

    all_functions: dict[str, Any] = {}
    all_tensor_tables: dict[str, Any] = {}
    all_sequence_tables: dict[str, Any] = {}
    all_record_tables: dict[str, Any] = {}
    machine_control_links: list[Any] = []
    machine_indirect_links: list[Any] = []
    pending_call_records: list[tuple[str, Any, Any, Any, Any]] = []
    class_table = None
    source_function_table = getattr(
        getattr(compilation.deployment, "process_graph", None),
        "function_table",
        None,
    )
    function_symbols: dict[int, str] = {}
    section_outputs: dict[str, tuple[Any, ...]] = {}
    export_symbols: list[str] = []
    lowering_failures: list[tuple[str, Any]] = []
    planned_shells = tuple(_walk_planned_shells(compilation.deployment))
    source_name_references: dict[str, set[int]] = {}
    for planned_shell in planned_shells:
        planned_graph = getattr(
            getattr(planned_shell, "process_graph", None), "G", None
        )
        if planned_graph is None:
            continue
        planned_name = planned_graph.graph.get("function_name")
        planned_reference = planned_graph.graph.get("function_ref")
        if planned_name is not None and planned_reference is not None:
            source_name_references.setdefault(
                str(planned_name), set()
            ).add(int(planned_reference))
    # Decompiled host modules share repository IR containers and FunctionTable
    # ownership. Merge them before source methods so calls link to exact roots;
    # the explicit completeness fact below distinguishes legalized repository
    # SSA from retained machine-state dialect inside those containers.
    if source_function_table is not None:
        for entry in source_function_table:
            host_module = entry.metadata.get("host_ssa_module")
            host_root = entry.metadata.get("host_ssa_root")
            if host_module is None or host_root is None:
                continue
            all_functions.update(host_module.functions)
            function_symbols[int(entry.reference.address)] = str(host_root)
            all_tensor_tables.update(getattr(host_module, "tensor_tables", {}))
            all_sequence_tables.update(getattr(host_module, "sequence_tables", {}))
            all_record_tables.update(getattr(host_module, "record_tables", {}))
            machine_control_links.extend(
                getattr(
                    getattr(host_module, "machine_control_table", None),
                    "links", (),
                )
            )
            machine_indirect_links.extend(
                getattr(
                    getattr(host_module, "machine_indirect_table", None),
                    "links", (),
                )
            )
            host_blockers = tuple(entry.metadata.get("host_ssa_blockers", ()))
            host_hard_blockers = tuple(entry.metadata.get(
                "host_ssa_hard_blockers", host_blockers,
            ))
            host_legalization_shortfalls = tuple(entry.metadata.get(
                "host_ssa_legalization_shortfalls", (),
            ))
            host_unresolved_dependencies = tuple(entry.metadata.get(
                "host_ssa_unresolved_dependencies", (),
            ))
            host_repository_ssa_complete = bool(entry.metadata.get(
                "host_repository_ssa_complete", False,
            ))
            host_machine_state_complete = bool(entry.metadata.get(
                "host_machine_state_complete", False,
            ))
            host_native_module = entry.metadata.get("host_native_module")
            host_root_function = all_functions.get(str(host_root))
            if host_root_function is not None:
                from .native_code_retention import (
                    WINDOWS_AMD64_NATIVE_LINKER,
                    select_host_implementation,
                )
                implementation_decision = select_host_implementation(
                    repository_ssa_complete=host_repository_ssa_complete,
                    machine_state_ssa_complete=host_machine_state_complete,
                    retained_native_module=host_native_module,
                    target=WINDOWS_AMD64_NATIVE_LINKER,
                )
                host_root_function.metadata.update({
                    "host_ssa_complete": (
                        host_repository_ssa_complete
                    ),
                    "host_machine_state_complete": (
                        host_machine_state_complete
                    ),
                    "host_ssa_blockers": host_blockers,
                    "host_ssa_hard_blockers": host_hard_blockers,
                    "host_ssa_legalization_shortfalls": (
                        host_legalization_shortfalls
                    ),
                    "host_ssa_unresolved_dependencies": (
                        host_unresolved_dependencies
                    ),
                    "host_ssa_cache_key": entry.metadata.get(
                        "host_ssa_cache_key"
                    ),
                    "host_native_module": host_native_module,
                    "implementation_variants": entry.metadata.get(
                        "implementation_variants", ("repository-ssa",)
                    ),
                    "implementation_decision": implementation_decision,
                    "selected_implementation": implementation_decision.implementation.value,
                    "implementation_deployable": implementation_decision.deployable,
                })
    retained_storage_identities: set[str] = set()
    for shell in planned_shells:
        graph = getattr(shell, "process_graph", None)
        graph_obj = graph.G if graph is not None else None
        if graph_obj is None:
            continue
        field_contract = _field_slot_ops(graph_obj)
        for _effect, _key, sequence_id, storage_identity in field_contract[12]:
            if sequence_id is not None:
                retained_storage_identities.add(str(storage_identity))
    for shell in planned_shells:
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
        # Some precompile-only shells retain the flat region schedule even
        # though branch compartments were already proven during partitioning.
        # Reapply the backend-neutral ordinary-conditional overlay here before
        # repository SSA sees the program.  Existing loop/while control is the
        # outer control and strict region containment nests each conditional
        # at its real lexical marker.
        from .control_source import overlay_scheduled_control
        from .glsl_deployment_strategy import (
            _ordinary_conditional_control_programs,
        )
        conditional_controls = _ordinary_conditional_control_programs(
            graph,
            control.region_indices,
            getattr(shell, "dispatch_subgraphs", ()),
        )
        from .control_source import SequenceBlock, StatementBlock
        lowered_conditional_count = 0
        if conditional_controls:
            from .control_source import ConditionalBlock

            def conditional_of(program):
                return next((
                    block for block in program.root.blocks
                    if isinstance(block, ConditionalBlock)
                ), None)

            conditional_blocks = tuple(map(
                conditional_of, conditional_controls
            ))
            source_expressions = {
                int(block.source_node_id): graph_obj.nodes[
                    int(block.source_node_id)
                ].get("expr_obj")
                for block in conditional_blocks
                if block is not None and block.source_node_id is not None
            }
            parent_by_child: dict[int, int] = {}
            for child_index, child in enumerate(conditional_blocks, start=1):
                if child is None or child.source_node_id is None:
                    continue
                candidates = []
                for parent_index, parent in enumerate(
                    conditional_blocks, start=1
                ):
                    if (
                        parent is None
                        or parent_index == child_index
                        or parent.source_node_id is None
                    ):
                        continue
                    expression = source_expressions.get(
                        int(parent.source_node_id)
                    )
                    if expression is None:
                        continue
                    descendants = {
                        id(member) for statement in (
                            *expression.body, *expression.orelse
                        ) for member in ast.walk(statement)
                    }
                    child_expression = source_expressions.get(
                        int(child.source_node_id)
                    )
                    if child_expression is not None and id(child_expression) in descendants:
                        span = int(getattr(
                            expression, "end_lineno", expression.lineno
                        )) - int(expression.lineno)
                        candidates.append((span, parent_index))
                if candidates:
                    parent_by_child[child_index] = min(candidates)[1]
            # Index zero is the already-scheduled outer control program;
            # conditional programs begin at one.  The AST supplies exact
            # lexical containment, including equal-region nesting such as an
            # ``if`` whose entire arm is another ``if``.  Only maximal source
            # conditionals attach directly to the schedule root; every nested
            # conditional attaches to its nearest lexical conditional.
            direct_children: dict[int, list[int]] = {}
            for child_index in range(1, len(conditional_controls) + 1):
                direct_children.setdefault(
                    parent_by_child.get(child_index, 0), []
                ).append(child_index)
            control = overlay_scheduled_control(
                control.region_indices,
                (control, *conditional_controls),
                known_nesting={
                    parent: tuple(children)
                    for parent, children in direct_children.items()
                },
            )
            def marker_counts(block, counts):
                from .control_source import (
                    CallBlock, ConditionalBlock, LoopBlock,
                    ParallelDeployment, SequenceBlock, StateMachineTick,
                    StatementBlock, WhileBlock,
                )
                if isinstance(block, StatementBlock):
                    if (
                        len(block.lines) == 1
                        and block.lines[0].startswith("__scheduled_region_")
                    ):
                        index = int(
                            block.lines[0][len("__scheduled_region_"):-2]
                        )
                        counts[index] = counts.get(index, 0) + 1
                elif isinstance(block, SequenceBlock):
                    for child in block.blocks:
                        marker_counts(child, counts)
                elif isinstance(block, ConditionalBlock):
                    marker_counts(block.body, counts)
                    if block.orelse is not None:
                        marker_counts(block.orelse, counts)
                elif isinstance(block, (LoopBlock,)):
                    marker_counts(block.body, counts)
                elif isinstance(block, WhileBlock):
                    marker_counts(block.condition, counts)
                    marker_counts(block.body, counts)
                elif isinstance(block, StateMachineTick):
                    for _case, body in block.cases:
                        marker_counts(body, counts)
                    if block.default is not None:
                        marker_counts(block.default, counts)
                elif isinstance(block, ParallelDeployment):
                    for lane in block.lanes:
                        marker_counts(lane, counts)
                elif isinstance(block, CallBlock):
                    marker_counts(block.callee, counts)
            counts = {}
            marker_counts(control.root, counts)
            duplicates = {
                region: count for region, count in counts.items()
                if count != 1
            }
            if duplicates:
                raise FortranEmissionError(
                    "conditional control duplicated scheduled regions: "
                    f"{duplicates!r}"
                )
            lowered_conditional_count = len(conditional_controls)
        function_reference = graph_obj.graph.get("function_ref")
        qualified_name = None
        if function_reference is not None and source_function_table is not None:
            try:
                qualified_name = source_function_table.entry(
                    int(function_reference)
                ).qualified_name
            except (KeyError, TypeError, ValueError):
                qualified_name = None
        symbol_source = (
            qualified_name
            if len(source_name_references.get(str(function_name), ())) > 1
            else function_name
        )
        symbol_suffix = str(symbol_source).replace(".", "__")
        symbol = f"{artifact_name}__{symbol_suffix}"
        if function_reference is not None:
            function_symbols[int(function_reference)] = symbol
        # Instance fields flow through the object's field arena: ``self`` is a
        # slot array, a field read is a load from its slot, a field write a
        # store. In whole-program precompile mode the field-op region is never
        # built (gated behind ``not precompile_only``), so recover the field ops
        # from the process graph and hand them to the lowerer as slot access.
        self_id, field_ops, const_sources, field_count, field_names, record_identity, sequence_initializations, field_aliases, sequence_declarations, sequence_memberships, table_lookups, table_stores, table_deletions, retained_sequence_ids, nested_sequence_ids, nested_record_fields = _field_slot_ops(
            graph_obj,
            retained_storage_identities=frozenset(retained_storage_identities),
        )
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
                field_const_sources=const_sources,
                field_count=field_count,
                field_names=field_names,
                record_identity=record_identity,
                sequence_initializations=sequence_initializations,
                field_aliases=field_aliases,
                sequence_declarations=sequence_declarations,
                sequence_memberships=sequence_memberships,
                table_lookups=table_lookups,
                table_stores=table_stores,
                table_deletions=table_deletions,
                retained_sequence_ids=retained_sequence_ids,
                nested_sequence_ids=nested_sequence_ids,
                nested_record_fields=nested_record_fields,
                sequence_augassigns=_sequence_augassign_ops(graph_obj),
                sequence_append_fills=_sequence_append_fill_ops(graph_obj),
                sequence_append_slices=_sequence_append_slice_ops(graph_obj),
                sequence_bit_packs=_sequence_bit_pack_ops(graph_obj),
                sequence_prepends=_sequence_prepend_concat_ops(graph_obj),
                sequence_prepend_packed_calls=(
                    _sequence_prepend_packed_call_ops(graph_obj)
                ),
                sequence_inplace_bit_pack_calls=(
                    _sequence_inplace_bit_pack_call_ops(graph)
                ),
                nested_row_projections=_nested_row_projection_ops(
                    graph_obj, control
                ),
                string_table=string_table,
                tensor_ssa_reference=tensor_ssa_reference,
            )
        )
        if shortfalls:
            lowering_failures.extend((symbol, item) for item in shortfalls)
        all_functions.update(module_ir.functions)
        lowered_control = module_ir.functions.get(symbol)
        if lowered_control is not None:
            source_output_value_ids = tuple(dict.fromkeys(
                int(value_id)
                for name in tuple(
                    graph_obj.graph.get("function_outputs") or ()
                )
                for value_id in tuple(
                    (graph_obj.graph.get("identity_table") or {}).get(
                        str(name), ()
                    )
                )
            ))
            lowered_control.metadata.update({
                "source_conditional_count": len(conditional_controls),
                "lowered_conditional_count": lowered_conditional_count,
                "source_output_value_ids": source_output_value_ids,
            })
        all_tensor_tables.update(
            getattr(module_ir, "tensor_tables", {})
        )
        all_sequence_tables.update(
            getattr(module_ir, "sequence_tables", {})
        )
        all_record_tables.update(
            getattr(module_ir, "record_tables", {})
        )
        from .hierarchical_plan import PlanCall
        pending_call_records.extend(
            (symbol, item, graph_obj, module_ir, shell)
            for item in getattr(shell, "hierarchy_plan", ()).items
            if isinstance(item, PlanCall)
        )
        # The whole-object module must retain the same class/member records
        # used to resolve the method closure.  They are an ABI description of
        # field slots and function references, not runtime object dispatch.
        if class_table is None and compilation.class_navigation is not None:
            from .precompile_to_ssa import lower_class_navigation_to_ssa

            class_table = lower_class_navigation_to_ssa(
                compilation.class_navigation
            ).class_table
        section_outputs.update(shell_section_outputs)
        export_symbols.append(symbol)
    if lowering_failures:
        raise FortranEmissionError(
            "whole-object methods have operators without an SSA handler: "
            + "; ".join(
                f"{symbol}::{item.location}::{item.name} ({item.reason})"
                for symbol, item in lowering_failures
            )
        )
    if not export_symbols:
        return None, {}, ()
    if class_table is not None and function_symbols:
        class_table = replace(
            class_table,
            classes=tuple(
                replace(
                    record,
                    methods=tuple(
                        replace(
                            method,
                            function_name=function_symbols.get(
                                int(method.function_reference)
                            ),
                        )
                        for method in record.methods
                    ),
                )
                for record in class_table.classes
            ),
        )
    # Persist token -> word so the emitted object's words are reversible.
    try:
        string_table.save()
    except Exception:  # noqa: BLE001 -- reverse-lookup cache, never fatal
        pass

    def emit_outputs(name: str, function: Any) -> tuple[Any, ...]:
        # A flat operator region has no explicit return: its outputs come from
        # the lowerer as ``intent(out)`` dummies the target appends. A control
        # function names its outputs with a return instruction.
        if name in section_outputs and section_outputs[name]:
            return section_outputs[name]
        returns = tuple(
            instruction.args
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret", "Return", "return"}
        )
        return tuple(returns[-1]) if returns else ()

    from ..transmogrifier.ssa import (
        SSAChildTablePoolDescriptor,
        SSACallRecord,
        SSARecordDescriptor,
        SSARecordFieldDescriptor,
        SSARecordFieldStorage,
        SSARecordInstancePoolDescriptor,
        SSARecordInstancePoolField,
        SSARecordTable,
        SSASequenceDescriptor,
        SSASequenceTable,
        SSAValue,
    )

    # Class construction is raw caller-owned storage plus an authored
    # constructor call.  The frontend already preserves every ``Class(...)``
    # node with its exact ``class_ref`` and every method call binds its receiver
    # value through PlanCall.  Materialize that missing correlation here, at
    # the same whole-program call-frame boundary that links ordinary calls.
    #
    # Each constructor occurrence receives a distinct set of arena ids.  The
    # stable field ``storage_identity`` (for example ``Store.table``) tells us
    # which callee field it implements; the receiver value tells us *which
    # instance*.  Consequently two Store() calls never become global/shared
    # storage, and no Python object or runtime dispatcher is introduced.
    constructor_calls: list[SSACallRecord] = []
    constructor_anchors: dict[tuple[str, int], int | None] = {}
    constructor_instance_pools: dict[
        tuple[str, int], dict[str, Any]
    ] = {}
    source_graphs_by_symbol = {
        f"{artifact_name}__{graph.graph.get('function_name')}": graph
        for shell in planned_shells
        for graph in (
            getattr(getattr(shell, "process_graph", None), "G", None),
        )
        if graph is not None and graph.graph.get("function_name") is not None
    }

    def function_values(function: Any) -> dict[int, Any]:
        values = {int(value.id): value for value in function.args}
        values.update({
            int(instruction.res.id): instruction.res
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        })
        return values

    def clone_value(source: Any, value_id: int, *, accounting=None):
        return SSAValue(
            int(value_id),
            dtype=getattr(source, "dtype", None),
            shape=tuple(getattr(source, "shape", ()) or ()),
            device=getattr(source, "device", None),
            accounting={
                **dict(getattr(source, "accounting", {}) or {}),
                **dict(accounting or {}),
            },
        )

    def map_child_pool(pool: Any, remap: Mapping[int, int]):
        if pool is None:
            return None
        return SSAChildTablePoolDescriptor(
            handle_column=int(pool.handle_column),
            column_value_ids=tuple(
                remap[int(value_id)] for value_id in pool.column_value_ids
            ),
            length_value_id=remap[int(pool.length_value_id)],
            capacity_value_id=remap[int(pool.capacity_value_id)],
            row_stride_value_id=remap[int(pool.row_stride_value_id)],
            status_value_id=(
                None if pool.status_value_id is None
                else remap[int(pool.status_value_id)]
            ),
            live_flags_value_id=(
                None if pool.live_flags_value_id is None
                else remap[int(pool.live_flags_value_id)]
            ),
            column_dtypes=tuple(pool.column_dtypes),
            key_columns=tuple(pool.key_columns),
            writable=bool(pool.writable),
        )

    def loop_constructor_requires_instance_pool(
        graph: Any, receiver_id: int, enclosing_loop_ids: tuple[int, ...]
    ) -> bool:
        if not enclosing_loop_ids:
            return False
        output_ids = {
            int(value_id)
            for name in tuple(graph.graph.get("function_outputs") or ())
            for value_id in tuple(
                (graph.graph.get("identity_table") or {}).get(str(name), ())
            )
        }
        if int(receiver_id) in output_ids:
            return True
        # Passing the record as data stores/returns/aliases its identity beyond
        # the current iteration. Attribute lookup is only an address derivation
        # and does not itself extend the record lifetime; all other consumer
        # roles require a per-iteration instance row until proven otherwise.
        for successor in graph.successors(int(receiver_id)):
            successor_data = graph.nodes[successor]
            roles = {
                str(role) for parent, role in (
                    successor_data.get("parents") or ()
                ) if int(parent) == int(receiver_id)
            }
            if (
                (successor_data.get("op") or successor_data.get("type"))
                == "GetAttr"
                and roles <= {"value", "base", "object", "operand"}
            ):
                continue
            if roles:
                return True
        return False

    if class_table is not None:
        class_definitions = {
            str(record.identity): record for record in class_table.classes
        }
        caller_contexts = {}
        for caller_symbol, _item, caller_graph, _module, caller_shell in (
            pending_call_records
        ):
            caller_contexts.setdefault(
                str(caller_symbol), (caller_graph, caller_shell)
            )
        # A function with construction but no ordinary method/function call has
        # no pending PlanCall record.  Include every planned shell as well.
        for shell in planned_shells:
            graph = getattr(getattr(shell, "process_graph", None), "G", None)
            function_name = (
                None if graph is None else graph.graph.get("function_name")
            )
            if function_name is not None:
                caller_contexts.setdefault(
                    f"{artifact_name}__{function_name}", (graph, shell)
                )

        constructor_symbol_by_class = {
            str(definition.identity): str(method.function_name)
            for definition in class_definitions.values()
            for method in definition.methods
            if method.name in {"__new__", "__init__"}
            and method.function_name is not None
        }

        # Materialize constructor-owned frames dependency-first.  If
        # ``Outer.__init__`` constructs ``Inner``, the Inner frame must already
        # have been copied into Outer before a caller copies Outer.  This is a
        # local topological order over the already-pursued constructor calls,
        # not another ingestion/lowering pipeline.
        ordered_contexts = []
        visiting = set()
        visited = set()

        def visit_constructor_context(symbol: str) -> None:
            if symbol in visited or symbol in visiting:
                return
            visiting.add(symbol)
            context = caller_contexts.get(symbol)
            if context is not None:
                graph, _shell = context
                dependencies = tuple(dict.fromkeys(
                    constructor_symbol_by_class[str(class_ref)]
                    for _node_id, data in graph.nodes(data=True)
                    for class_ref in ((data.get("attributes") or {}).get(
                        "class_ref"
                    ),)
                    if class_ref is not None
                    and str(class_ref) in constructor_symbol_by_class
                ))
                for dependency in dependencies:
                    visit_constructor_context(str(dependency))
                ordered_contexts.append((symbol, context))
            visiting.remove(symbol)
            visited.add(symbol)

        for caller_symbol in caller_contexts:
            visit_constructor_context(str(caller_symbol))

        for caller_symbol, (caller_graph, caller_shell) in ordered_contexts:
            caller = all_functions.get(caller_symbol)
            if caller is None:
                continue
            available = function_values(caller)
            graph_ids = {
                int(data.get("value_id", node_id))
                for node_id, data in caller_graph.nodes(data=True)
            }
            next_value_id = 1 + max((*available, *graph_ids), default=0)
            caller_records = all_record_tables.setdefault(
                caller_symbol, SSARecordTable()
            )
            caller_sequences = all_sequence_tables.setdefault(
                caller_symbol, SSASequenceTable()
            )
            for node_id, node_data in sorted(
                caller_graph.nodes(data=True), key=lambda item: int(item[0])
            ):
                attributes = node_data.get("attributes") or {}
                class_identity = attributes.get("class_ref")
                if class_identity is None:
                    continue
                class_definition = class_definitions.get(str(class_identity))
                if class_definition is None:
                    continue
                constructor_method = next((
                    method for name in ("__new__", "__init__")
                    for method in class_definition.methods
                    if method.name == name and method.function_name is not None
                ), None)
                if constructor_method is None:
                    continue
                constructor_symbol = str(constructor_method.function_name)
                constructor = all_functions.get(constructor_symbol)
                constructor_table = all_record_tables.get(constructor_symbol)
                constructor_sequences = all_sequence_tables.get(
                    constructor_symbol
                )
                if (
                    constructor is None
                    or constructor_table is None
                    or not constructor_table.records
                ):
                    continue
                templates = tuple(
                    record for record in constructor_table.records.values()
                    if record.identity == str(class_identity)
                )
                if len(templates) != 1:
                    continue
                template = templates[0]
                constructor_records_by_id = {
                    int(record.record_id): record
                    for record in constructor_table.records.values()
                }

                def nested_record_closure(root: Any) -> tuple[Any, ...]:
                    """Return the authored record/storage tree rooted at ``root``.

                    Nested records are still ordinary repository-SSA record
                    descriptors.  Following their ids here makes construction
                    copy the complete caller-owned storage frame rather than
                    reducing a nested record field to one opaque handle.
                    """

                    ordered = []
                    pending = [root]
                    seen = set()
                    while pending:
                        record = pending.pop()
                        record_id = int(record.record_id)
                        if record_id in seen:
                            continue
                        seen.add(record_id)
                        ordered.append(record)
                        for field in reversed(record.fields):
                            if field.record_id is None:
                                continue
                            nested = constructor_records_by_id.get(
                                int(field.record_id)
                            )
                            if nested is not None:
                                pending.append(nested)
                    return tuple(ordered)

                template_records = nested_record_closure(template)
                receiver_id = int(node_data.get("value_id", node_id))
                if receiver_id in caller_records.records:
                    continue
                constructor_values = function_values(constructor)
                constructor_self_id = int(template.record_id)
                self_is_field_storage = any(
                    constructor_self_id in tuple(map(int, field.value_ids))
                    for field in template.fields
                )
                remap: dict[int, int] = {}
                if not self_is_field_storage:
                    remap[constructor_self_id] = receiver_id
                constructor_graph = source_graphs_by_symbol.get(
                    constructor_symbol
                )
                constructor_parameter_ids: set[int] = set()
                if constructor_graph is not None:
                    identities = constructor_graph.graph.get(
                        "identity_table"
                    ) or {}
                    parameter_names = tuple(
                        constructor_graph.graph.get("function_parameters") or ()
                    )
                    positional_names = tuple(
                        name for name in parameter_names if name != "self"
                    )
                    for parent, role in node_data.get("parents") or ():
                        role = str(role)
                        if role in {"callee", "func", "definition"}:
                            continue
                        argument_suffix = (
                            role[4:] if role.startswith("arg:") else role[3:]
                            if role.startswith("arg") else ""
                        )
                        if argument_suffix.isdigit():
                            index = int(argument_suffix)
                            name = (
                                positional_names[index]
                                if index < len(positional_names) else None
                            )
                        elif role.startswith("kw:"):
                            name = role.split(":", 1)[1]
                        else:
                            name = None
                        history = tuple(identities.get(name, ()))
                        if not history:
                            continue
                        parameter_id = int(history[0])
                        remap[parameter_id] = int(
                            caller_graph.nodes[parent].get("value_id", parent)
                        )
                        constructor_parameter_ids.add(parameter_id)
                referenced_ids = tuple(dict.fromkeys(
                    int(value_id)
                    for record in template_records
                    for field in record.fields
                    for value_id in (
                        *field.value_ids,
                        *((field.record_id,) if field.record_id is not None else ()),
                    )
                ))
                for old_id in referenced_ids:
                    if old_id in remap:
                        continue
                    remap[old_id] = next_value_id
                    next_value_id += 1

                # Constructor field writes and subsequent reads can carry
                # separate local sequence descriptors for one record slot.
                # Correlate all such authored field-op views to the canonical
                # record field before building the call frame.
                constructor_field_sequence_ids = set()
                constructor_field_sequence_ids_by_name = {}
                if constructor_graph is not None:
                    constructor_field_contract = _field_slot_ops(
                        constructor_graph
                    )
                    constructor_field_names = tuple(
                        constructor_field_contract[4]
                    )
                    for _kind, value_id, slot in constructor_field_contract[1]:
                        constructor_field_sequence_ids.add(int(value_id))
                        if 0 <= int(slot) < len(constructor_field_names):
                            constructor_field_sequence_ids_by_name.setdefault(
                                str(constructor_field_names[int(slot)]), set()
                            ).add(int(value_id))
                if constructor_sequences is not None:
                    for field in template.fields:
                        if field.sequence_id is None:
                            continue
                        canonical = constructor_sequences.by_id(
                            field.sequence_id
                        )
                        if canonical is None:
                            continue
                        canonical_ids = (
                            *canonical.column_value_ids,
                            canonical.length_address_id,
                            canonical.capacity_value_id,
                            *((canonical.status_address_id,)
                              if canonical.status_address_id is not None else ()),
                            *((canonical.live_flags_value_id,)
                              if canonical.live_flags_value_id is not None else ()),
                        )
                        resident_ids = tuple(remap[int(value_id)]
                                             for value_id in canonical_ids)
                        for local in constructor_sequences.sequences.values():
                            if (
                                int(local.sequence_id)
                                not in constructor_field_sequence_ids_by_name.get(
                                    str(field.name), set()
                                )
                                or len(local.column_value_ids)
                                != len(canonical.column_value_ids)
                                or tuple(local.key_columns)
                                != tuple(canonical.key_columns)
                            ):
                                continue
                            local_ids = (
                                *local.column_value_ids,
                                local.length_address_id,
                                local.capacity_value_id,
                                *((local.status_address_id,)
                                  if local.status_address_id is not None else ()),
                                *((local.live_flags_value_id,)
                                  if local.live_flags_value_id is not None else ()),
                            )
                            if len(local_ids) == len(resident_ids):
                                remap.update(zip(
                                    map(int, local_ids), resident_ids
                                ))

                new_arguments = []
                for old_id, new_id in remap.items():
                    if (
                        old_id == constructor_self_id
                        and old_id not in referenced_ids
                    ):
                        continue
                    if new_id in available:
                        continue
                    source = constructor_values.get(old_id, SSAValue(old_id))
                    value = clone_value(source, new_id, accounting={
                        "record_instance": str(class_identity),
                        "constructor_callsite_id": int(node_id),
                    })
                    caller.args.append(value)
                    available[new_id] = value
                    new_arguments.append(value)

                mapped_records = {}
                for record_template in reversed(template_records):
                    mapped_fields = []
                    for field in record_template.fields:
                        mapped_sequence_id = None
                        if field.sequence_id is not None:
                            source_sequence = (
                                None if constructor_sequences is None
                                else constructor_sequences.by_id(field.sequence_id)
                            )
                            if source_sequence is None:
                                continue
                            pool_ids = ()
                            if source_sequence.child_table_pool is not None:
                                pool = source_sequence.child_table_pool
                                pool_ids = (
                                    *pool.column_value_ids,
                                    pool.length_value_id,
                                    pool.capacity_value_id,
                                    pool.row_stride_value_id,
                                    *((pool.status_value_id,)
                                      if pool.status_value_id is not None else ()),
                                    *((pool.live_flags_value_id,)
                                      if pool.live_flags_value_id is not None else ()),
                                )
                            for old_id in map(int, pool_ids):
                                if old_id not in remap:
                                    remap[old_id] = next_value_id
                                    source = constructor_values.get(
                                        old_id, SSAValue(old_id)
                                    )
                                    value = clone_value(
                                        source,
                                        next_value_id,
                                        accounting={
                                            "record_instance": str(class_identity),
                                            "constructor_callsite_id": int(node_id),
                                        },
                                    )
                                    caller.args.append(value)
                                    available[next_value_id] = value
                                    next_value_id += 1
                            mapped_sequence_id = remap[
                                int(source_sequence.sequence_id)
                            ]
                            caller_sequences.register(SSASequenceDescriptor(
                                sequence_id=mapped_sequence_id,
                                column_value_ids=tuple(
                                    remap[int(value_id)]
                                    for value_id in source_sequence.column_value_ids
                                ),
                                length_address_id=remap[
                                    int(source_sequence.length_address_id)
                                ],
                                capacity_value_id=remap[
                                    int(source_sequence.capacity_value_id)
                                ],
                                status_address_id=(
                                    None
                                    if source_sequence.status_address_id is None
                                    else remap[int(source_sequence.status_address_id)]
                                ),
                                column_dtypes=tuple(source_sequence.column_dtypes),
                                key_columns=tuple(source_sequence.key_columns),
                                live_flags_value_id=(
                                    None
                                    if source_sequence.live_flags_value_id is None
                                    else remap[int(source_sequence.live_flags_value_id)]
                                ),
                                capacity_policy=source_sequence.capacity_policy,
                                writable=bool(source_sequence.writable),
                                child_table_pool=map_child_pool(
                                    source_sequence.child_table_pool, remap
                                ),
                            ))
                        mapped_fields.append(SSARecordFieldDescriptor(
                            name=field.name,
                            storage=field.storage,
                            storage_identity=field.storage_identity,
                            value_ids=tuple(
                                remap[int(value_id)]
                                for value_id in field.value_ids
                            ),
                            sequence_id=mapped_sequence_id,
                            record_id=(
                                None
                                if field.record_id is None
                                else remap[int(field.record_id)]
                            ),
                            offset=field.offset,
                            dtype=field.dtype,
                            writable=field.writable,
                        ))
                    mapped_record_id = (
                        receiver_id
                        if int(record_template.record_id) == constructor_self_id
                        else remap[int(record_template.record_id)]
                    )
                    mapped_record = SSARecordDescriptor(
                        mapped_record_id,
                        str(record_template.identity),
                        tuple(mapped_fields),
                    )
                    caller_records.register(mapped_record)
                    mapped_records[int(record_template.record_id)] = mapped_record
                mapped_fields = list(
                    mapped_records[int(template.record_id)].fields
                )
                constructor_bindings = tuple(
                    (
                        old_id,
                        (
                            "caller_value"
                            if old_id in constructor_parameter_ids
                            else "caller_storage"
                        ),
                        new_id,
                    )
                    for old_id, new_id in remap.items()
                    if old_id in {int(value.id) for value in constructor.args}
                )
                unresolved = tuple(
                    int(value.id) for value in constructor.args
                    if int(value.id) not in remap
                )
                enclosing_loop_ids = tuple(
                    int(plan.loop.node_id)
                    for plan in sorted(
                        (
                            plan for plan in caller_shell.loop_plans
                            if int(node_id) in plan.loop.body_nodes
                        ),
                        key=lambda plan: -len(plan.loop.body_nodes),
                    )
                )
                requires_instance_pool = (
                    loop_constructor_requires_instance_pool(
                        caller_graph, receiver_id, enclosing_loop_ids
                    )
                )
                if requires_instance_pool:
                    destination_ids = tuple(dict.fromkeys(
                        int(caller_graph.nodes[parent].get("value_id", parent))
                        for successor in caller_graph.successors(receiver_id)
                        for successor_data in (caller_graph.nodes[successor],)
                        if str(successor_data.get("op") or "").lower()
                        in {"append", "add"}
                        for parent, role in (
                            successor_data.get("parents") or ()
                        )
                        if str(role) == "operand"
                    ))
                    mapped_leaf_fields = tuple(
                        field
                        for record in mapped_records.values()
                        for field in record.fields
                        if field.storage is not SSARecordFieldStorage.RECORD
                    )
                    pooled_fields = tuple(
                        field for field in mapped_leaf_fields
                        if field.sequence_id is not None
                    )
                    scalar_fields = tuple(
                        field for field in mapped_leaf_fields
                        if field.storage is SSARecordFieldStorage.SCALAR
                    )
                    poolable_fields = (*pooled_fields, *scalar_fields)
                    if (
                        len(destination_ids) == 1
                        and len(poolable_fields) == len(mapped_leaf_fields)
                        and poolable_fields
                    ):
                        destination = caller_sequences.by_id(
                            destination_ids[0]
                        )
                        field_pools = []
                        pool_specs = []
                        for pooled_field in pooled_fields:
                            field_sequence = caller_sequences.by_id(
                                pooled_field.sequence_id
                            )
                            template_field = next(
                                field
                                for record in template_records
                                for field in record.fields
                                if field.storage_identity
                                == pooled_field.storage_identity
                            )
                            callee_sequence = constructor_sequences.by_id(
                                template_field.sequence_id
                            )
                            if field_sequence is None or callee_sequence is None:
                                pool_specs = []
                                break
                            row_stride_id = next_value_id
                            next_value_id += 1
                            row_stride = SSAValue(
                                row_stride_id,
                                dtype="int",
                                accounting={
                                    "record_instance_pool_stride": (
                                        str(pooled_field.storage_identity)
                                    ),
                                    "constructor_callsite_id": int(node_id),
                                },
                            )
                            caller.args.append(row_stride)
                            available[row_stride_id] = row_stride
                            pool = SSAChildTablePoolDescriptor(
                                handle_column=0,
                                column_value_ids=tuple(
                                    field_sequence.column_value_ids
                                ),
                                length_value_id=int(
                                    field_sequence.length_address_id
                                ),
                                capacity_value_id=int(
                                    field_sequence.capacity_value_id
                                ),
                                row_stride_value_id=row_stride_id,
                                status_value_id=(
                                    None
                                    if field_sequence.status_address_id is None
                                    else int(field_sequence.status_address_id)
                                ),
                                live_flags_value_id=(
                                    None
                                    if field_sequence.live_flags_value_id is None
                                    else int(field_sequence.live_flags_value_id)
                                ),
                                column_dtypes=tuple(
                                    field_sequence.column_dtypes
                                ),
                                key_columns=tuple(
                                    field_sequence.key_columns
                                ),
                                writable=bool(field_sequence.writable),
                            )
                            field_pools.append(SSARecordInstancePoolField(
                                storage_identity=str(
                                    pooled_field.storage_identity
                                ),
                                storage=SSARecordFieldStorage.SEQUENCE,
                                sequence_pool=pool,
                            ))
                            pool_specs.append({
                                "pool": pool,
                                "callee_field": template_field,
                                "callee_sequence": callee_sequence,
                            })
                        scalar_specs = []
                        scalar_source_ids = tuple(dict.fromkeys(
                            int(value_id)
                            for field in scalar_fields
                            for value_id in field.value_ids
                        ))
                        if len(scalar_source_ids) > 1:
                            pool_specs = []
                        elif scalar_fields:
                            scalar_source_id = scalar_source_ids[0]
                            scalar_stride_id = next_value_id
                            next_value_id += 1
                            scalar_stride = SSAValue(
                                scalar_stride_id,
                                dtype="int",
                                accounting={
                                    "record_instance_pool_scalar_stride": (
                                        str(class_identity)
                                    ),
                                    "constructor_callsite_id": int(node_id),
                                },
                            )
                            caller.args.append(scalar_stride)
                            available[scalar_stride_id] = scalar_stride
                            for scalar_field in scalar_fields:
                                template_field = next(
                                    field
                                    for record in template_records
                                    for field in record.fields
                                    if field.storage_identity
                                    == scalar_field.storage_identity
                                )
                                field_pools.append(SSARecordInstancePoolField(
                                    storage_identity=str(
                                        scalar_field.storage_identity
                                    ),
                                    storage=SSARecordFieldStorage.SCALAR,
                                    scalar_value_id=scalar_source_id,
                                    scalar_stride_value_id=scalar_stride_id,
                                    scalar_offset=int(
                                        scalar_field.offset or 0
                                    ),
                                ))
                                scalar_specs.append({
                                    "arena_value_id": scalar_source_id,
                                    "stride_value_id": scalar_stride_id,
                                    "offset": int(scalar_field.offset or 0),
                                    "callee_value_ids": tuple(map(
                                        int, template_field.value_ids
                                    )),
                                })
                        if destination is not None and pool_specs:
                            # Preserve the historical one-field projection for
                            # existing nested-table consumers. Multi-field
                            # records use the record-level grouping below.
                            caller_sequences.sequences[
                                int(destination.sequence_id)
                            ] = replace(
                                destination,
                                child_table_pool=(
                                    pool_specs[0]["pool"]
                                    if len(pool_specs) == 1 else None
                                ),
                            )
                            record_pool = SSARecordInstancePoolDescriptor(
                                int(destination.sequence_id),
                                tuple(field_pools),
                            )
                            caller_records.records[receiver_id] = replace(
                                caller_records.records[receiver_id],
                                instance_pool=record_pool,
                            )
                            constructor_instance_pools[(
                                caller_symbol, int(node_id)
                            )] = {
                                "receiver_id": receiver_id,
                                "destination_sequence_id": int(
                                    destination.sequence_id
                                ),
                                "fields": tuple(pool_specs),
                                "scalar_fields": tuple(scalar_specs),
                            }
                            requires_instance_pool = False
                constructor_calls.append(SSACallRecord(
                    caller=caller_symbol,
                    callsite_id=int(node_id),
                    callee_reference=int(constructor_method.function_reference),
                    callee_name=str(constructor_method.name),
                    callee_symbol=constructor_symbol,
                    argument_bindings=((receiver_id, constructor_self_id),),
                    enclosing_loop_ids=enclosing_loop_ids,
                    callee_storage_value_ids=tuple(
                        int(value.id) for value in constructor.args
                    ),
                    frame_bindings=constructor_bindings,
                    unresolved_frame_value_ids=unresolved,
                    decomposition=(
                        "requires_loop_instance_pool"
                        if requires_instance_pool else None
                    ),
                ))
                later_values = sorted(
                    int(data.get("value_id", other_id))
                    for other_id, data in caller_graph.nodes(data=True)
                    if int(other_id) > int(node_id)
                )
                constructor_anchors[(caller_symbol, int(node_id))] = (
                    later_values[0] if later_values else None
                )

    call_records: dict[str, list[SSACallRecord]] = {}
    call_anchor_value_ids: dict[tuple[str, int], int | None] = {}
    seen_calls: set[tuple[str, int, int | None]] = set()
    for caller_symbol, planned_call, caller_graph, caller_module, caller_shell in (
        pending_call_records
    ):
        call_data = caller_graph.nodes.get(int(planned_call.callsite_id), {})
        attributes = call_data.get("attributes") or {}
        reference = attributes.get(
            "callee_ref",
            attributes.get("method_ref", attributes.get("constructor_ref")),
        )
        call_key = (
            str(caller_symbol), int(planned_call.callsite_id),
            None if reference is None else int(reference),
        )
        if call_key in seen_calls:
            continue
        seen_calls.add(call_key)
        callee_symbol = (
            None if reference is None
            else function_symbols.get(int(reference))
        )
        callee_function = (
            None if callee_symbol is None
            else all_functions.get(callee_symbol)
        )
        child_shell = getattr(caller_shell, "callsite_function_shells", {}).get(
            int(planned_call.callsite_id)
        )
        child_graph = getattr(
            getattr(child_shell, "process_graph", None), "G", None
        )
        caller_aliases: dict[int, int] = {}
        for history in (caller_graph.graph.get("identity_table") or {}).values():
            canonical = next((
                int(value_id) for value_id in reversed(history)
                if any(
                    int(value.id) == int(value_id)
                    for value in all_functions[caller_symbol].args
                )
                or any(
                    instruction.res is not None
                    and int(instruction.res.id) == int(value_id)
                    for block in all_functions[caller_symbol].blocks.values()
                    for instruction in block.instrs
                )
            ), None)
            if canonical is not None:
                for value_id in history:
                    caller_aliases[int(value_id)] = int(canonical)
        exact_bindings = {
            int(callee): caller_aliases.get(int(caller), int(caller))
            for caller, callee in planned_call.argument_bindings
        }
        identity_aliases: dict[int, int] = {}
        if child_graph is not None:
            for history in (child_graph.graph.get("identity_table") or {}).values():
                bound = next((
                    exact_bindings[int(value_id)]
                    for value_id in history
                    if int(value_id) in exact_bindings
                ), None)
                if bound is not None:
                    for value_id in history:
                        identity_aliases[int(value_id)] = int(bound)
        default_literals: dict[int, Any] = {}
        if child_graph is not None and source_function_table is not None:
            child_reference = child_graph.graph.get("function_ref")
            try:
                child_entry = source_function_table.entry(int(child_reference))
            except (KeyError, TypeError, ValueError):
                child_entry = None
            callable_object = (
                None if child_entry is None
                else getattr(child_entry, "python_callable", None)
            )
            if (
                callable_object is None
                and child_entry is not None
                and "." in str(child_entry.qualified_name)
            ):
                parts = str(child_entry.qualified_name).split(".")
                for split in range(len(parts) - 1, 0, -1):
                    try:
                        candidate = importlib.import_module(
                            ".".join(parts[:split])
                        )
                    except ImportError:
                        continue
                    try:
                        for attribute in parts[split:]:
                            candidate = getattr(candidate, attribute)
                    except AttributeError:
                        continue
                    callable_object = candidate
                    break
            if callable_object is not None:
                try:
                    signature = inspect.signature(callable_object)
                except (TypeError, ValueError):
                    signature = None
                if signature is not None:
                    identities = child_graph.graph.get("identity_table") or {}
                    for parameter in signature.parameters.values():
                        if parameter.default is inspect.Parameter.empty:
                            continue
                        history = tuple(identities.get(parameter.name, ()))
                        for value_id in history:
                            default_literals[int(value_id)] = parameter.default
        frame_bindings = []
        unresolved_frame = []
        receiver_record = None
        callee_record = None
        if callee_symbol is not None:
            callee_records = all_record_tables.get(callee_symbol)
            candidates = (
                () if callee_records is None
                else tuple(callee_records.records.values())
            )
            if len(candidates) == 1:
                callee_record = candidates[0]
                bound_receiver = exact_bindings.get(int(callee_record.record_id))
                caller_records = all_record_tables.get(caller_symbol)
                if bound_receiver is not None and caller_records is not None:
                    receiver_record = caller_records.records.get(
                        int(bound_receiver)
                    )
        storage_bindings = {}
        if receiver_record is not None and callee_record is not None:
            caller_fields = {
                field.storage_identity: field
                for field in receiver_record.fields
            }
            for field in callee_record.fields:
                caller_field = caller_fields.get(field.storage_identity)
                if (
                    caller_field is None
                    or len(caller_field.value_ids) != len(field.value_ids)
                ):
                    continue
                storage_bindings.update(zip(
                    map(int, field.value_ids),
                    map(int, caller_field.value_ids),
                ))
            # Repeated GetAttr/SetAttr occurrences may have distinct local
            # sequence ids even though the record table correctly identifies
            # one physical field.  Bind every descriptor proven to be another
            # view of that field.  The proof is structural: its sequence id is
            # one of the authored field-op value ids for the same slot and its
            # row contract matches the canonical descriptor.  This preserves
            # every occurrence while giving them one caller-owned arena.
            callee_sequence_table = all_sequence_tables.get(callee_symbol)
            caller_sequence_table = all_sequence_tables.get(caller_symbol)
            callee_shell = getattr(
                caller_shell, "callsite_function_shells", {}
            ).get(int(planned_call.callsite_id))
            callee_graph = getattr(
                getattr(callee_shell, "process_graph", None), "G", None
            )
            field_value_ids_by_identity = {}
            if callee_graph is not None:
                for node_id, data in callee_graph.nodes(data=True):
                    record_field = (data.get("attributes") or {}).get(
                        "record_field"
                    )
                    if not record_field or len(record_field) != 2:
                        continue
                    field_value_ids_by_identity.setdefault(
                        f"{record_field[0]}.{record_field[1]}", set()
                    ).add(int(data.get("value_id", node_id)))
            if (
                callee_sequence_table is not None
                and caller_sequence_table is not None
            ):
                canonical_pairs = []
                for field in callee_record.fields:
                    if field.sequence_id is None:
                        continue
                    caller_field = caller_fields.get(field.storage_identity)
                    if caller_field is None or caller_field.sequence_id is None:
                        continue
                    canonical_pairs.append((
                        field.storage_identity,
                        callee_sequence_table.by_id(field.sequence_id),
                        caller_sequence_table.by_id(caller_field.sequence_id),
                    ))
                for local in callee_sequence_table.sequences.values():
                    for storage_identity, canonical, resident in canonical_pairs:
                        if int(local.sequence_id) not in (
                            field_value_ids_by_identity.get(
                                str(storage_identity), set()
                            )
                        ):
                            continue
                        if canonical is None or resident is None:
                            continue
                        if (
                            len(local.column_value_ids)
                            != len(canonical.column_value_ids)
                            or tuple(local.key_columns)
                            != tuple(canonical.key_columns)
                            or bool(local.writable) != bool(canonical.writable)
                        ):
                            continue
                        local_ids = (
                            *local.column_value_ids,
                            local.length_address_id,
                            local.capacity_value_id,
                            *((local.status_address_id,)
                              if local.status_address_id is not None else ()),
                            *((local.live_flags_value_id,)
                              if local.live_flags_value_id is not None else ()),
                        )
                        resident_ids = (
                            *resident.column_value_ids,
                            resident.length_address_id,
                            resident.capacity_value_id,
                            *((resident.status_address_id,)
                              if resident.status_address_id is not None else ()),
                            *((resident.live_flags_value_id,)
                              if resident.live_flags_value_id is not None else ()),
                        )
                        if len(local_ids) == len(resident_ids):
                            storage_bindings.update(zip(
                                map(int, local_ids), map(int, resident_ids)
                            ))
                        break
        for value in (() if callee_function is None else callee_function.args):
            value_id = int(value.id)
            if value_id in storage_bindings:
                frame_bindings.append((
                    value_id, "caller_storage", storage_bindings[value_id]
                ))
            elif value_id in exact_bindings:
                frame_bindings.append((
                    value_id, "caller_value", exact_bindings[value_id]
                ))
            elif value_id in identity_aliases:
                frame_bindings.append((
                    value_id, "caller_alias", identity_aliases[value_id]
                ))
            elif value_id in default_literals:
                frame_bindings.append((
                    value_id, "default_literal", default_literals[value_id]
                ))
            else:
                unresolved_frame.append(value_id)
        decompositions = tuple(
            instruction
            for block in all_functions[caller_symbol].blocks.values()
            for instruction in block.instrs
            if instruction.attributes.get("decomposed_plan_call")
            and instruction.attributes.get("plan_callsite_id") is not None
            and int(instruction.attributes["plan_callsite_id"])
            == int(planned_call.callsite_id)
            and (
                reference is None
                or instruction.attributes.get("callee_reference") is None
                or reference is not None
                and int(instruction.attributes.get("callee_reference"))
                == int(reference)
            )
        )
        resolution = "decomposed" if decompositions else "unresolved"
        call_records.setdefault(caller_symbol, []).append(SSACallRecord(
            caller=caller_symbol,
            callsite_id=int(planned_call.callsite_id),
            callee_reference=(None if reference is None else int(reference)),
            callee_name=str(planned_call.callee.name),
            callee_symbol=callee_symbol,
            argument_bindings=tuple(planned_call.argument_bindings),
            result_bindings=tuple(planned_call.result_bindings),
            enclosing_loop_ids=tuple(planned_call.enclosing_loop_ids),
            callee_storage_value_ids=(
                () if callee_function is None
                else tuple(int(value.id) for value in callee_function.args)
            ),
            frame_bindings=tuple(frame_bindings),
            unresolved_frame_value_ids=tuple(unresolved_frame),
            resolution=resolution,
            decomposition=(
                None if not decompositions
                else str(decompositions[0].attributes.get(
                    "ssa_sequence_operation"
                ))
            ),
        ))
        call_expression = call_data.get("expr_obj")
        call_position = (
            int(getattr(call_expression, "end_lineno", 0) or 0),
            int(getattr(call_expression, "end_col_offset", 0) or 0),
        )
        caller_result_ids = {
            int(instruction.res.id)
            for block in all_functions[caller_symbol].blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        }
        anchors = []
        for node_id, node_data in caller_graph.nodes(data=True):
            expression = node_data.get("expr_obj")
            value_id = int(node_data.get("value_id", node_id))
            if expression is None or value_id not in caller_result_ids:
                continue
            position = (
                int(getattr(expression, "lineno", 0) or 0),
                int(getattr(expression, "col_offset", 0) or 0),
            )
            if position > call_position:
                anchors.append((*position, value_id))
        call_anchor_value_ids[(
            str(caller_symbol), int(planned_call.callsite_id)
        )] = min(anchors)[2] if anchors else None

    for record in constructor_calls:
        existing = call_records.setdefault(record.caller, [])
        duplicate_index = next((
            index for index, candidate in enumerate(existing)
            if int(candidate.callsite_id) == int(record.callsite_id)
            and candidate.callee_symbol == record.callee_symbol
        ), None)
        if duplicate_index is None:
            existing.append(record)
        else:
            candidate = existing[duplicate_index]
            # The hierarchy-owned constructor occurrence supplies lexical loop
            # ownership and explicit self/argument bindings; the record-ABI
            # occurrence supplies the complete caller-storage frame.  Merge
            # those two views into one execution record.
            existing[duplicate_index] = replace(
                record,
                argument_bindings=(
                    candidate.argument_bindings or record.argument_bindings
                ),
                result_bindings=(
                    candidate.result_bindings or record.result_bindings
                ),
                enclosing_loop_ids=(
                    candidate.enclosing_loop_ids or record.enclosing_loop_ids
                ),
            )
        call_anchor_value_ids[(record.caller, record.callsite_id)] = (
            constructor_anchors.get((record.caller, record.callsite_id))
        )

    # A method call can appear twice in the hierarchy catalogue: once as the
    # callable/attribute shell (no frame bindings) and once as the execution
    # occurrence (complete PlanCall bindings).  They are not two executions.
    # Prefer the complete frame for an identical caller/callee binding shape;
    # keep genuinely distinct complete occurrences and keep an incomplete one
    # only when no complete record can supersede it.
    for caller_symbol, records in tuple(call_records.items()):
        # Node/value ids retain authored expression order in a ProcessGraph.
        # Constructors were recovered after ordinary PlanCalls, so restore the
        # shared source order before fixed-point materialization.  Inserting
        # each call ahead of the same Ret then preserves ``construct; method``
        # rather than accidentally reversing them.
        records.sort(key=lambda record: int(record.callsite_id))
        complete_keys = {
            (record.callee_reference, record.callee_symbol)
            for record in records
            if not record.unresolved_frame_value_ids
        }
        call_records[caller_symbol] = [
            record for record in records
            if not (
                record.unresolved_frame_value_ids
                and not record.argument_bindings
                and not record.result_bindings
                and (record.callee_reference, record.callee_symbol)
                in complete_keys
            )
        ]

    # Materialize the first ordinary repository-SSA call frames.  Eligibility
    # is contract based: every callee argument is explained, exactly one
    # planner result is bound, the callee's authored conditional catalogue is
    # fully lowered, and the callee itself has no unresolved source calls.
    # Anything outside that proof remains an unresolved call record.
    from ..transmogrifier.ssa import Instr, SSAValue

    # Calls form a dependency graph, not a declaration-order list.  Resolve it
    # to a fixed point: once every authored call in a leaf is materialized, its
    # callers become eligible in the next round, continuing outward through an
    # arbitrarily deep source closure.  A single pass strands whichever caller
    # happened to be visited before its callee and falsely reports complete
    # source as unresolved at emission.
    changed = True
    while changed:
        changed = False
        callee_callers = {
            caller: tuple(records) for caller, records in call_records.items()
        }
        for caller_symbol, records in tuple(call_records.items()):
            caller = all_functions[caller_symbol]
            values = {int(value.id): value for value in caller.args}
            values.update({
                int(instruction.res.id): instruction.res
                for block in caller.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            })
            next_value_id = 1 + max(values, default=0)
            rebuilt_records = []
            for record in records:
                was_unresolved = record.resolution == "unresolved"
                callee = (
                    None if record.callee_symbol is None
                    else all_functions.get(record.callee_symbol)
                )
                result_binding = (
                    record.result_bindings[0]
                    if len(record.result_bindings) == 1 else None
                )
                callee_records = callee_callers.get(
                    str(record.callee_symbol), ()
                )
                callee_outputs = (
                    () if callee is None
                    else emit_outputs(record.callee_symbol, callee)
                )
                returns_value = (
                    len(record.result_bindings) == 1
                    and len(callee_outputs) == 1
                )
                returns_void = (
                    not record.result_bindings and not callee_outputs
                )
                eligible = (
                    was_unresolved
                    and callee is not None
                    and not record.unresolved_frame_value_ids
                    and (returns_value or returns_void)
                    and (
                        not record.enclosing_loop_ids
                        or record.callee_name in {"__init__", "__new__"}
                    )
                    and record.decomposition != "requires_loop_instance_pool"
                    and not any(
                        item.resolution == "unresolved"
                        for item in callee_records
                    )
                    and int(callee.metadata.get(
                        "source_conditional_count", 0
                    )) == int(callee.metadata.get(
                        "lowered_conditional_count", 0
                    ))
                )
                binding_by_callee = {
                    int(value_id): (str(kind), source)
                    for value_id, kind, source in record.frame_bindings
                }
                if eligible:
                    call_arguments = []
                    constants = []
                    instance_pool = constructor_instance_pools.get((
                        str(caller_symbol), int(record.callsite_id)
                    ))
                    pooled_argument_ids = {}
                    pooled_setup = []
                    if instance_pool is not None:
                        target_loop_id = int(record.enclosing_loop_ids[-1])
                        induction = next((
                            instruction.res
                            for block in caller.blocks.values()
                            for instruction in block.instrs
                            if instruction.op == "Phi"
                            and instruction.res is not None
                            and instruction.attributes.get("source_name")
                            == f"iteration_{target_loop_id}"
                        ), None)
                        if induction is None:
                            eligible = False
                        else:
                            destination_sequence_id = int(
                                instance_pool["destination_sequence_id"]
                            )
                            for block in caller.blocks.values():
                                for instruction in block.instrs:
                                    if (
                                        instruction.attributes.get(
                                            "ssa_sequence_operation"
                                        ) in {"append", "add"}
                                        and int(instruction.attributes.get(
                                            "sequence_id", -1
                                        )) == destination_sequence_id
                                        and instruction.args
                                        and int(instruction.args[-1].id)
                                        == int(instance_pool["receiver_id"])
                                    ):
                                        # A list/set of records stores the
                                        # pool row handle. Replace only the
                                        # authored inserted-value operand;
                                        # another ABI argument may legally
                                        # share its local numeric id.
                                        instruction.args[-1] = induction
                            for field_spec in instance_pool["fields"]:
                                pool = field_spec["pool"]
                                callee_sequence = field_spec["callee_sequence"]
                                row_offset = SSAValue(
                                    next_value_id, dtype="int"
                                )
                                next_value_id += 1
                                pooled_setup.append(Instr(
                                    "Mul",
                                    [
                                        induction,
                                        values[int(pool.row_stride_value_id)],
                                    ],
                                    row_offset,
                                    attributes={
                                        "binding": "record_instance_pool_row"
                                    },
                                ))
                                pointer_sources = {
                                    **{
                                        int(callee_id): (
                                            int(source_id), row_offset
                                        )
                                        for callee_id, source_id in zip(
                                            callee_sequence.column_value_ids,
                                            pool.column_value_ids,
                                        )
                                    },
                                    int(callee_sequence.length_address_id): (
                                        int(pool.length_value_id), induction
                                    ),
                                    **(
                                        {
                                            int(callee_sequence.status_address_id): (
                                                int(pool.status_value_id),
                                                induction,
                                            )
                                        }
                                        if (
                                            callee_sequence.status_address_id
                                            is not None
                                            and pool.status_value_id is not None
                                        ) else {}
                                    ),
                                    **(
                                        {
                                            int(callee_sequence.live_flags_value_id): (
                                                int(pool.live_flags_value_id),
                                                row_offset,
                                            )
                                        }
                                        if (
                                            callee_sequence.live_flags_value_id
                                            is not None
                                            and pool.live_flags_value_id is not None
                                        ) else {}
                                    ),
                                }
                                pooled_argument_ids[
                                    int(callee_sequence.capacity_value_id)
                                ] = values[int(pool.row_stride_value_id)]
                                for callee_id, (source_id, offset) in (
                                    pointer_sources.items()
                                ):
                                    pointer = SSAValue(
                                        next_value_id,
                                        dtype=values[int(source_id)].dtype,
                                        accounting={
                                            "record_instance_pool_pointer": True
                                        },
                                    )
                                    next_value_id += 1
                                    pooled_setup.append(Instr(
                                        "GetElementPtr",
                                        [values[int(source_id)], offset],
                                        pointer,
                                        attributes={
                                            "binding": "record_instance_pool"
                                        },
                                    ))
                                    pooled_argument_ids[callee_id] = pointer
                            for scalar_spec in instance_pool.get(
                                "scalar_fields", ()
                            ):
                                scalar_base = SSAValue(
                                    next_value_id, dtype="int"
                                )
                                next_value_id += 1
                                pooled_setup.append(Instr(
                                    "Mul",
                                    [
                                        induction,
                                        values[int(
                                            scalar_spec["stride_value_id"]
                                        )],
                                    ],
                                    scalar_base,
                                    attributes={
                                        "binding": (
                                            "record_instance_pool_scalar_row"
                                        )
                                    },
                                ))
                                scalar_index = scalar_base
                                if int(scalar_spec["offset"]):
                                    offset_value = SSAValue(
                                        next_value_id, dtype="int"
                                    )
                                    next_value_id += 1
                                    pooled_setup.append(Instr(
                                        "Const", [], offset_value,
                                        attributes={
                                            "value": int(
                                                scalar_spec["offset"]
                                            )
                                        },
                                    ))
                                    scalar_index = SSAValue(
                                        next_value_id, dtype="int"
                                    )
                                    next_value_id += 1
                                    pooled_setup.append(Instr(
                                        "Add",
                                        [scalar_base, offset_value],
                                        scalar_index,
                                        attributes={
                                            "binding": (
                                                "record_instance_pool_scalar_offset"
                                            )
                                        },
                                    ))
                                pointer = SSAValue(
                                    next_value_id,
                                    dtype=values[int(
                                        scalar_spec["arena_value_id"]
                                    )].dtype,
                                    accounting={
                                        "record_instance_pool_pointer": True
                                    },
                                )
                                next_value_id += 1
                                pooled_setup.append(Instr(
                                    "GetElementPtr",
                                    [
                                        values[int(
                                            scalar_spec["arena_value_id"]
                                        )],
                                        scalar_index,
                                    ],
                                    pointer,
                                    attributes={
                                        "binding": (
                                            "record_instance_pool_scalar"
                                        )
                                    },
                                ))
                                for callee_id in scalar_spec[
                                    "callee_value_ids"
                                ]:
                                    pooled_argument_ids[int(callee_id)] = pointer
                    for argument in callee.args:
                        if int(argument.id) in pooled_argument_ids:
                            call_arguments.append(
                                pooled_argument_ids[int(argument.id)]
                            )
                            continue
                        kind, source = binding_by_callee[int(argument.id)]
                        if kind in {
                            "caller_value", "caller_alias", "caller_storage"
                        }:
                            value = values.get(int(source))
                            if value is None:
                                eligible = False
                                break
                            call_arguments.append(value)
                        elif kind == "default_literal":
                            value = SSAValue(
                                next_value_id,
                                dtype=argument.dtype,
                                shape=argument.shape,
                            )
                            next_value_id += 1
                            constants.append(Instr(
                                "Const", [], value,
                                attributes={"value": source},
                            ))
                            call_arguments.append(value)
                        else:
                            eligible = False
                            break
                if eligible:
                    if returns_value:
                        _callee_result_id, caller_result_id = result_binding
                        callee_output = callee_outputs[0]
                        result = values.get(int(caller_result_id), SSAValue(
                            int(caller_result_id),
                            dtype=callee_output.dtype,
                            shape=callee_output.shape,
                        ))
                    else:
                        caller_result_id = None
                        result = None
                    native_call = Instr(
                        "Call", call_arguments, result,
                        attributes={
                            "callee": record.callee_symbol,
                            "source_linked": True,
                            "plan_callsite_id": record.callsite_id,
                            "callee_reference": record.callee_reference,
                        },
                    )
                    inserted = False
                    if returns_value:
                        for block in caller.blocks.values():
                            for index, instruction in enumerate(block.instrs):
                                if any(
                                    int(value.id) == int(caller_result_id)
                                    for value in instruction.args
                                ):
                                    block.instrs[index:index] = [
                                        *constants, native_call
                                    ]
                                    inserted = True
                                    break
                            if inserted:
                                break
                    else:
                        if record.enclosing_loop_ids:
                            target_loop_id = int(record.enclosing_loop_ids[-1])
                            header_name = next((
                                block.name
                                for block in caller.blocks.values()
                                if any(
                                    instruction.op == "Phi"
                                    and instruction.attributes.get("source_name")
                                    == f"iteration_{target_loop_id}"
                                    for instruction in block.instrs
                                )
                            ), None)
                            body_name = None
                            if header_name is not None:
                                header = caller.blocks[header_name]
                                branch = next((
                                    instruction for instruction in header.instrs
                                    if instruction.op == "CondBr"
                                ), None)
                                if branch is not None:
                                    body_name = branch.attributes.get(
                                        "true_target"
                                    )
                            body = caller.blocks.get(str(body_name))
                            if body is not None and instance_pool is not None:
                                destination_sequence_id = int(
                                    instance_pool["destination_sequence_id"]
                                )
                                insertion_index = next((
                                    index
                                    for index, instruction in enumerate(
                                        body.instrs
                                    )
                                    if instruction.attributes.get(
                                        "ssa_sequence_operation"
                                    ) in {"append", "add"}
                                    and int(instruction.attributes.get(
                                        "sequence_id", -1
                                    )) == destination_sequence_id
                                ), None)
                                if insertion_index is not None:
                                    # The constructor is the authored value
                                    # expression of this append/add. Initialize
                                    # its pool row before publishing the handle
                                    # into the containing sequence.
                                    body.instrs[insertion_index:insertion_index] = [
                                        *constants, *pooled_setup, native_call
                                    ]
                                    inserted = True
                            if (
                                not inserted
                                and
                                body is not None
                                and body.instrs
                                and body.instrs[-1].op in {
                                    "Br", "br", "Branch", "branch"
                                }
                            ):
                                body.instrs[-1:-1] = [
                                    *constants, *pooled_setup, native_call
                                ]
                                inserted = True
                        anchor_value_id = call_anchor_value_ids.get((
                            str(caller_symbol), int(record.callsite_id)
                        ))
                        if not inserted and anchor_value_id is not None:
                            for block in caller.blocks.values():
                                for index, instruction in enumerate(block.instrs):
                                    if (
                                        instruction.res is not None
                                        and int(instruction.res.id)
                                        == int(anchor_value_id)
                                    ):
                                        block.instrs[index:index] = [
                                            *constants, native_call
                                        ]
                                        inserted = True
                                        break
                                if inserted:
                                    break
                        if not inserted and not record.enclosing_loop_ids:
                            for block in caller.blocks.values():
                                if (
                                    block.instrs
                                    and block.instrs[-1].op in {
                                        "Ret", "ret", "Return", "return"
                                    }
                                ):
                                    block.instrs[-1:-1] = [
                                        *constants, native_call
                                    ]
                                    inserted = True
                                    break
                    if (
                        not inserted
                        and returns_value
                        and int(caller_result_id) in set(map(
                            int,
                            caller.metadata.get(
                                "source_output_value_ids", ()
                            ),
                        ))
                    ):
                        # A function whose body is solely ``return callee(...)``
                        # has no ordinary consumer instruction to anchor the
                        # call: control lowering emitted an empty Ret because
                        # the PlanCall is linked afterward.  The source output
                        # ledger is the exact authored control position, so put
                        # the call immediately before that terminator and make
                        # the terminator return its result.
                        for block in caller.blocks.values():
                            if (
                                block.instrs
                                and block.instrs[-1].op in {
                                    "Ret", "ret", "Return", "return"
                                }
                                and not block.instrs[-1].args
                            ):
                                block.instrs[-1:-1] = [
                                    *constants, native_call
                                ]
                                block.instrs[-1].args = [result]
                                inserted = True
                                break
                    if inserted:
                        if returns_value:
                            caller.args = [
                                value for value in caller.args
                                if int(value.id) != int(caller_result_id)
                            ]
                            values[int(caller_result_id)] = result
                        record = replace(record, resolution="native_call")
                        changed = True
                rebuilt_records.append(record)
            call_records[caller_symbol] = rebuilt_records

    # A constructed-record result is a compile-time correlation once every
    # consumer has been rewritten to its physical field arenas or pool handle.
    # Remove only the shapeless conceptual receiver argument; a sequence
    # capacity or other physical ABI value may legitimately share the same
    # source-local numeric id and must remain.
    for function_name, record_table in all_record_tables.items():
        function = all_functions.get(function_name)
        if function is None:
            continue
        record_ids = set(map(int, record_table.records))
        consumed_ids = {
            int(argument.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        }
        function.args = [
            argument for argument in function.args
            if not (
                int(argument.id) in record_ids
                and int(argument.id) not in consumed_ids
                and argument.dtype is None
                and not argument.shape
                and not argument.accounting
            )
        ]
        unique_arguments = {}
        for argument in function.args:
            existing = unique_arguments.get(int(argument.id))
            if existing is None:
                unique_arguments[int(argument.id)] = argument
                continue
            existing_physical = bool(
                existing.dtype is not None
                or existing.shape
                or existing.accounting
            )
            argument_physical = bool(
                argument.dtype is not None
                or argument.shape
                or argument.accounting
            )
            if argument_physical and not existing_physical:
                unique_arguments[int(argument.id)] = argument
        function.args = list(unique_arguments.values())

    return (
        IRModule(
            all_functions,
            **(
                {"function_table": source_function_table}
                if source_function_table is not None else {}
            ),
            **({"class_table": class_table} if class_table is not None else {}),
            tensor_tables=all_tensor_tables,
            sequence_tables=all_sequence_tables,
            record_tables=all_record_tables,
            call_table={
                caller: tuple(records)
                for caller, records in call_records.items()
            },
            machine_control_table=(
                SSAMachineControlTable(tuple(machine_control_links))
            ),
            machine_indirect_table=(
                SSAMachineIndirectTable(tuple(machine_indirect_links))
            ),
        ),
        {
            name: emit_outputs(name, function)
            for name, function in all_functions.items()
        },
        tuple(export_symbols),
    )


def _emit_class_surface_module(
    compilation: Any,
    artifact_name: str,
    *,
    tensor_ssa_reference: Any = None,
):
    """Emit the reusable whole-object SSA program as one Fortran module."""

    from .ssa_fortran_backend import emit_module

    ssa_module, outputs, export_symbols = _class_surface_ssa_program(
        compilation,
        artifact_name,
        tensor_ssa_reference=tensor_ssa_reference,
    )
    if ssa_module is None:
        return None, ()
    emitted = emit_module(
        ssa_module,
        name=f"{artifact_name}_fortran",
        outputs=outputs,
        # A library exports its whole surface: keep and export every method and
        # region function, not just the ones one nominal entry reaches.
        extra_roots=tuple(ssa_module.functions),
    )
    if not emitted.complete:
        raise FortranEmissionError(
            "class surface could not emit hierarchical object program: "
            + "; ".join(item.format() for item in emitted.shortfalls)
        )
    return emitted, export_symbols


def lower_ast_source_to_ssa(
    source: str,
    entrypoint: str,
    *,
    python_bindings: Mapping[str, Any] | None = None,
    dependency_seeds: tuple[str, ...] = (),
    retain: Any = (),
    tensor_code_references: Mapping[str, Callable[..., Any]] | None = None,
    tensor_ssa_reference: Any = None,
    name: str | None = None,
    runtime_closure_only: bool = True,
    progress: Callable[[str], None] | None = None,
):
    """Ingest one complete authored program and lower it directly to SSA.

    This is the explicit non-projecting compiler entry point.  It preserves
    source control, ordinary arithmetic, tensor operations, calls, memory and
    returns through ProcessGraph planning and repository SSA.  It never
    captures, constructs, validates, or projects a numerical ``FusedProgram``
    and it does not execute the submitted program.

    The returned tuple is ``(IRModule, outputs, exports)``.  Target emission is
    deliberately separate so callers can inspect the complete SSA program
    before choosing Fortran or another backend.
    """

    import contextlib
    import io
    from types import SimpleNamespace

    from ..common.tensors.accelerator_backends.aot_compile import (
        _source_dependency_is_not_tensor_primitive,
    )
    from ..common.tensors.topological_reducer import (
        reduce_abstract_tensor_topology,
    )
    from ..transmogrifier.graph.graph_express2 import ProcessGraph
    from .glsl_deployment_strategy import strategize_shell_deployment
    from .shell_reference_tables import build_class_navigation_table

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    tree = ast.parse(source)
    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = dict(python_bindings or {})
    report("ssa-source: building complete ProcessGraph source closure")
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            tree,
            resolve_unresolved_parents=True,
            parent_include=_source_dependency_is_not_tensor_primitive,
            pursuit_roots=(
                tuple(dict.fromkeys((entrypoint, *dependency_seeds)))
                if runtime_closure_only else None
            ),
            tensor_code_references=dict(tensor_code_references or {}),
            retain=retain,
            progress=report,
        )
    graph.G.graph["compile_targets"] = tuple(dict.fromkeys((
        str(entrypoint), *map(str, dependency_seeds),
    )))
    report("ssa-source: reducing source topology")
    reduce_abstract_tensor_topology(graph)
    report("ssa-source: planning complete control/operator graph")
    deployment_type = strategize_shell_deployment(
        graph,
        backend="fortran",
        runtime_closure_only=runtime_closure_only,
    )
    deployment = deployment_type(profiling=False, shell_language="glsl")
    deployment.compile_process_graph(prepare_ephemerals=False)
    deployment.prepare_graph_precompile(
        progress=report,
        structural_ssa_only=True,
    )
    compilation = SimpleNamespace(
        deployment=deployment,
        class_navigation=build_class_navigation_table(graph),
    )
    report("ssa-source: lowering full planned source to repository SSA")
    return _class_surface_ssa_program(
        compilation,
        _identifier(str(name or entrypoint)),
        tensor_ssa_reference=tensor_ssa_reference,
    )


lower_ast_source_to_ssa.__canonical_source_compiler__ = True


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
    tensor_code_references: Mapping[str, Callable[..., Any]] | None = None,
    tensor_ssa_reference: Any = None,
    runtime_closure_only: bool = False,
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

    from .compiler_entrypoints import warn_legacy_source_compiler

    warn_legacy_source_compiler("compile_ast_fortran_c_shell")

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
        tensor_code_references=tensor_code_references,
        require_planned_shells=library,
        runtime_closure_only=runtime_closure_only,
        # A whole-object library lowers every method's complete local
        # ControlProgram/operator graph directly.  Captured-region hierarchy
        # projection is a numerical optimization with an independent marker
        # catalogue; allowing it to run here can discard structural call
        # placement before the direct SSA lowerer sees the program.
        project_captured_hierarchy=not library,
    )
    hierarchical_outputs = dict(compilation.public_output_value_ids)
    hierarchical_inputs = dict(compilation.public_input_value_ids)
    # ``public_output_value_ids`` contains only hierarchy terminals that the
    # numerical capture marked device-resident.  A source return can instead
    # be a structural/control value (a callee result or an object-field load)
    # and is still a real compiled-program output.  Recover every declared
    # return from the authoritative identity history so the complete
    # ControlProgram/region SSA path emits it; never force such a return
    # through ``project_public_numerical_program`` merely because capture did
    # not classify it as a numerical terminal.
    for output_name in getattr(compilation, "function_outputs", ()):
        history = tuple(
            getattr(compilation, "identity_table", {}).get(output_name, ())
        )
        if history:
            hierarchical_outputs.setdefault(
                str(output_name), int(history[-1])
            )
    # A later field read after a source-ordered write consumes the stored
    # value even when the ProcessGraph has only the receiver edge on GetAttr.
    # Recover that memory dependency explicitly for public returns.  This is
    # the same field-slot ordering rule used by whole-object lowering, applied
    # to an ordinary function receiving a record parameter.
    source_process_graph = getattr(
        getattr(compilation, "deployment", None), "process_graph", None
    )
    source_function_table = getattr(source_process_graph, "function_table", None)
    if source_function_table is not None:
        try:
            source_entry = source_function_table.entry(entrypoint)
        except KeyError:
            source_entry = None
        function_graph = getattr(getattr(source_entry, "graph", None), "G", None)
        if function_graph is not None:
            for output_name, output_id in tuple(hierarchical_outputs.items()):
                if int(output_id) not in function_graph:
                    continue
                output_data = function_graph.nodes[int(output_id)]
                if output_data.get("type") != "GetAttr":
                    continue
                attribute = (output_data.get("attributes") or {}).get("attribute")
                receiver = next((
                    int(parent)
                    for parent, role in (output_data.get("parents") or ())
                    if str(role) in {"value", "object"}
                ), None)
                stored_value = None
                for node_id in sorted(
                    function_graph.nodes, key=lambda value: int(value)
                ):
                    if int(node_id) >= int(output_id):
                        break
                    data = function_graph.nodes[node_id]
                    if (
                        data.get("type") not in {"SetAttr", "setattr"}
                        or (data.get("attributes") or {}).get("attribute")
                        != attribute
                    ):
                        continue
                    parents = tuple(data.get("parents") or ())
                    target = next((
                        int(parent) for parent, role in parents
                        if str(role) in {"object", "value"}
                    ), None)
                    value = next((
                        int(parent) for parent, role in parents
                        if str(role) == "value"
                    ), None)
                    if target == receiver and value is not None:
                        stored_value = value
                if stored_value is not None:
                    hierarchical_outputs[output_name] = int(stored_value)
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

    # ``entrypoint`` is a Python-qualified source name and may contain dots
    # (for example ``ProcessGraph.build_from_ast``).  The artifact name is
    # also the prefix for every emitted Fortran procedure and C symbol, where
    # dots and the other Python punctuation are illegal.  Sanitize once at
    # this boundary so module names, intra-module calls, API symbols, and file
    # names all use the same spelling.
    artifact_name = _identifier(str(name or entrypoint))
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
            compilation,
            artifact_name,
            tensor_ssa_reference=tensor_ssa_reference,
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
        raise FortranEmissionError(
            "whole-object library compilation produced no planned method "
            "surface; refusing to substitute a numerical projection"
        )

    if hierarchical_outputs and compilation.region_programs:
        from .hierarchical_plan import PlanCall, PlanClosure
        from .precompile_to_ssa import lower_control_sections_to_ssa
        from .ssa_fortran_backend import emit_module

        runtime_value_meta: dict[int, tuple[tuple[int, ...], str]] = {}
        for source_name, value_id in hierarchical_inputs.items():
            root, *attributes = str(source_name).split(".")
            if root not in feeds:
                continue
            runtime_value = feeds[root]
            try:
                for attribute in attributes:
                    runtime_value = getattr(runtime_value, attribute)
                array = np.asarray(runtime_value)
            except (AttributeError, TypeError, ValueError):
                continue
            runtime_value_meta[int(value_id)] = (
                tuple(map(int, array.shape)), str(array.dtype)
            )

        def apply_runtime_value_meta(closure: Any) -> Any:
            if not isinstance(closure, PlanClosure):
                return closure
            shape_records = {
                int(value_id): (tuple(shape), str(dtype))
                for value_id, shape, dtype in closure.value_shapes
            }
            for value_id in (
                *closure.captures,
                *(
                    value_id
                    for item in closure.items
                    if hasattr(item, "inputs")
                    for value_id in (*item.inputs, *item.outputs)
                ),
            ):
                if int(value_id) in runtime_value_meta:
                    shape_records[int(value_id)] = runtime_value_meta[int(value_id)]
            shape_preserving = {
                "Add", "Sub", "Mul", "Div", "Pow", "Mod",
                "add", "sub", "mul", "div", "pow", "mod",
            }
            changed = True
            while changed:
                changed = False
                for item in closure.items:
                    if (
                        getattr(item, "opcode", None) not in shape_preserving
                        or not getattr(item, "outputs", ())
                    ):
                        continue
                    shaped_inputs = [
                        shape_records[int(value_id)]
                        for value_id in getattr(item, "inputs", ())
                        if int(value_id) in shape_records
                        and shape_records[int(value_id)][0]
                    ]
                    if not shaped_inputs:
                        continue
                    propagated = max(
                        shaped_inputs,
                        key=lambda record: (len(record[0]), record[0]),
                    )
                    for value_id in item.outputs:
                        previous = shape_records.get(int(value_id))
                        if previous != propagated:
                            shape_records[int(value_id)] = propagated
                            changed = True
            rebuilt_items = tuple(
                apply_runtime_value_meta(item)
                if isinstance(item, PlanClosure)
                else replace(item, callee=apply_runtime_value_meta(item.callee))
                if isinstance(item, PlanCall)
                else item
                for item in closure.items
            )
            return replace(
                closure,
                items=rebuilt_items,
                value_shapes=tuple(
                    (value_id, shape, dtype)
                    for value_id, (shape, dtype) in shape_records.items()
                ),
            )

        hierarchy_plan = apply_runtime_value_meta(
            getattr(compilation, "hierarchy_plan", None)
        )

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
        lowered_module, lowering_shortfalls, lowered_outputs = (
            lower_control_sections_to_ssa(
                compilation.shell_control_program,
                hierarchy_plan=hierarchy_plan,
                control_name=artifact_name,
                identity_table=identity_table,
                function_outputs=tuple(hierarchical_outputs),
                function_parameters=tuple(hierarchical_inputs),
                tensor_ssa_reference=tensor_ssa_reference,
            )
        )
        if lowering_shortfalls:
            raise FortranEmissionError(
                "complete hierarchical AST program has SSA shortfalls: "
                + "; ".join(
                    f"{item.name} ({item.reason})"
                    for item in lowering_shortfalls
                )
            )
        module = emit_module(
            lowered_module,
            name=f"{artifact_name}_fortran",
            outputs=lowered_outputs,
            extra_roots=tuple(lowered_module.functions) if library else (),
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
    "lower_ast_source_to_ssa",
]

"""Package an emitted ``bind(C)`` Fortran module in a native C shell.

The C translation unit contains only the generic profiled launch boundary,
buffer ownership, declared state feedback, and diagnostics.  Program logic
remains in the :class:`~src.compiler.ssa_fortran_backend.FortranModule` that
the ordinary AST/Control/SSA pipeline emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import copy
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import re
import subprocess
import ast
import inspect
import importlib
import hashlib
import textwrap
from typing import Any, Iterable, Mapping
from typing import Callable

import numpy as np

from ..common.tensors.accelerator_backends.profiled_c_shell import (
    _C_SOURCE, _C_TRACE_SOURCE,
)
from ..transmogrifier.graph.edge_roles import (
    keyword_argument_name,
    ordered_arguments,
    positional_argument_index,
)
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
        stream_frames: bool = False,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if frames < 0:
            raise ValueError("native C shell frame count cannot be negative")
        arguments = [str(self.executable_path), str(frames)]
        if stream_frames:
            arguments.append("--stream-frames")
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


def _record_receipts_for_function(
    program_abi: Mapping[str, Any],
    function_name: str,
    parameters: Iterable[str],
    *,
    method_owner: str | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Select explicit record bindings plus an exact method receiver schema.

    The receipt form is used after the extraction contract object has been
    reduced to serializable graph metadata.  Keep its selection semantics in
    step with :meth:`ProgramABIContract.records_for_function`: an unannotated
    ``self`` is safely identifiable from ``method_owner`` only when exactly
    one declared record has that class identity.
    """

    parameter_names = set(map(str, parameters))
    records = dict(program_abi.get("records") or {})
    selected: dict[str, Mapping[str, Any]] = {}
    for binding in tuple(program_abi.get("bindings") or ()):
        parameter = str(binding.get("parameter") or "")
        record_name = str(binding.get("record") or "")
        if (
            parameter in parameter_names
            and fnmatchcase(str(function_name), str(binding.get("function") or ""))
            and record_name in records
        ):
            selected[parameter] = records[record_name]
    if (
        method_owner is not None
        and "self" in parameter_names
        and "self" not in selected
    ):
        owner = str(method_owner)
        candidates = tuple(
            record
            for name, record in records.items()
            if (
                str(name) == owner
                or str(record.get("identity") or "") == owner
                or str(record.get("identity") or "").rsplit(".", 1)[-1]
                == owner
            )
        )
        if len(candidates) == 1:
            selected["self"] = candidates[0]
    return selected


def _entrypoint(module: Any, name: str | None = None) -> Any:
    selected = name or module.api.entry
    if selected is None:
        raise ValueError("Fortran module has no selected entry point")
    return module.api.entry_point(str(selected))


def _extent_values(
    entry: Any,
    overrides: Mapping[str, int] | None,
) -> dict[str, int]:
    values: dict[str, int] = {}
    unresolved: set[str] = set()
    for parameter in entry.parameters:
        if parameter.role != "extent":
            continue
        name = str(parameter.name)
        fixed = re.fullmatch(r"extent_([1-9][0-9]*)", name)
        if fixed is None:
            unresolved.add(name)
        else:
            values[name] = int(fixed.group(1))
    for name, value in dict(overrides or {}).items():
        if name not in values and name not in unresolved:
            raise ValueError(f"unknown Fortran extent override {name!r}")
        if int(value) < 1:
            raise ValueError(f"Fortran extent {name!r} must be positive")
        values[name] = int(value)
        unresolved.discard(name)
    if unresolved:
        names = ", ".join(sorted(unresolved))
        raise ValueError(
            "shape-dynamic Fortran extents require explicit positive "
            f"extent_overrides: {names}"
        )
    return values


def _element_count(parameter: Any, extents: Mapping[str, int]) -> int:
    dynamic_dimensions = tuple(getattr(parameter, "extents", ()) or ())
    if dynamic_dimensions:
        count = 1
        for dimension in dynamic_dimensions:
            count *= int(extents[str(dimension)])
        return max(count, 1)
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

    dynamic_dimensions = tuple(getattr(parameter, "extents", ()) or ())
    shape = (
        tuple(int(extents[str(name)]) for name in dynamic_dimensions)
        if dynamic_dimensions
        else tuple(
            int(extents.get(f"extent_{int(size)}", size))
            for size in tuple(parameter.shape or ())
        )
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
    trace: bool = False,
    trace_capacity: int = 4096,
) -> str:
    """Emit a standalone C main around one described Fortran entry point.

    ``trace`` compiles the launch digest IN. It is a compile-time decision,
    not a runtime flag: with it off the ring, its logger and the hook that
    would call it are absent from the binary entirely, so a launch pays
    nothing for a facility it was not built with. With it on, every launch
    writes one fixed-size record -- sequence, shell ns, device ns, region,
    status -- into a ring the executable owns, and main drains it at the
    end. Nothing crosses a language boundary while the program runs.
    """

    entry = _entrypoint(module, entrypoint)
    parameters = tuple(entry.parameters)
    extents = _extent_values(entry, extent_overrides)
    values = tuple(item for item in parameters if item.role != "extent")
    inputs = tuple(item for item in values if item.role in {"input", "inout"})
    outputs = tuple(item for item in values if item.role in {"output", "inout"})
    unsupported = tuple(
        item
        for item in values
        if item.role not in {"input", "inout", "workspace", "output"}
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
        if parameter.role in {"input", "inout"} and parameter.name not in system_parameters:
            if len(
                tuple(getattr(parameter, "extents", ()) or parameter.shape or ())
            ) <= 1:
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
    frame_output_lines = []
    output_write_lines = []
    for output_index, parameter in enumerate(outputs):
        # Several native parameters may retain the same authored source_name
        # after whole-program linking.  Publication must address this exact
        # output parameter, not whichever same-source argument happened to be
        # last in the signature.
        slot = slot_by_parameter[parameter.name]
        count = _element_count(parameter, extents)
        separator = "" if output_index == 0 else ","
        output_lines.extend((
            f"    {{ double sum = 0.0; size_t i;",
            f"      for (i = 0; i < {count}; ++i) sum += (({parameter.c_type} *)slots[{slot}])[i];",
            f"      printf(\"{separator}\\\"{_source_name(parameter)}\\\":{{\\\"first\\\":%.17g,\\\"sum\\\":%.17g}}\",",
            f"             (double)(({parameter.c_type} *)slots[{slot}])[0], sum); }}",
        ))
        frame_output_lines.extend((
            f"        {{ double sum = 0.0; size_t i;",
            f"          for (i = 0; i < {count}; ++i) sum += (({parameter.c_type} *)slots[{slot}])[i];",
            f"          printf(\"{separator}\\\"{_source_name(parameter)}\\\":{{\\\"first\\\":%.17g,\\\"sum\\\":%.17g}}\",",
            f"                 (double)(({parameter.c_type} *)slots[{slot}])[0], sum); }}",
        ))
        if len(
            tuple(getattr(parameter, "extents", ()) or parameter.shape or ())
        ) <= 1:
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
        # The macro has to precede the base source: the launch hook inside
        # it is guarded by `#if TURING_TRACE`, so defining it afterwards
        # would compile the ring in while leaving the hook that feeds it
        # compiled out -- a digest that is present and permanently empty.
        f"#define TURING_TRACE {1 if trace else 0}",
        _C_SOURCE,
        # `_C_TRACE_SOURCE` defines the ring types itself. The companion
        # `_C_TRACE_DECLARATIONS` exists for cffi's cdef, where the types
        # must be announced without bodies; pasting it into a real
        # translation unit redefines every struct and forward-references
        # TuringLaunchProfile before the base source declares it.
        _C_TRACE_SOURCE if trace else "",
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
        "    int stream_frames = 0;",
        f"    void *slots[{len(values)}] = {{0}};",
        "    TuringLaunchProfile profile = {0};",
        "    TuringLaunchStats stats = {0};",
        *((
            "    TuringTraceRing trace_ring = {0};",
            f"    TuringTraceRecord trace_storage[{trace_capacity}];",
            "    TuringTraceSite trace_site = {0};",
        ) if trace else ()),
        "    int frame;",
        "    { int argument_index;",
        "      for (argument_index = 2; argument_index < argc; ++argument_index)",
        "        if (strcmp(argv[argument_index], \"--stream-frames\") == 0) stream_frames = 1; }",
        f"    FILE *state = turing_open_artifact(argv[0], {_c_string(initial_state_filename)}, \"rb\");",
        "    if (frames < 0) return 2;",
        "    if (!state) { perror(\"initial state\"); return 2; }",
        *allocation_lines,
        *file_load_lines,
        *input_read_lines,
        "    fclose(state);",
        *display_open_lines,
        "    turing_launch_stats_reset(&stats);",
        *((
            f"    turing_trace_ring_reset(&trace_ring, trace_storage, {trace_capacity});",
            "    trace_site.ring = &trace_ring;",
            "    trace_site.region = 0;",
        ) if trace else ()),
        f"    for (frame = 0; {display_loop_condition}; ++frame) {{",
        *display_message_lines,
        "        if (turing_profiled_launch_ex(turing_fortran_compute, slots,",
        (
            "                &profile, &stats, turing_trace_logger_address(),"
            " &trace_site, 3) != 1) return 5;"
            if trace else
            "                &profile, &stats, NULL, NULL, 3) != 1) return 5;"
        ),
        *display_present_lines,
        *feedback_lines,
        "        if (stream_frames) {",
        "            printf(\"{\\\"event\\\":\\\"frame\\\",\\\"frame\\\":%d,\\\"outputs\\\":{\", frame + 1);",
        *frame_output_lines,
        "            printf(\"}}\\n\");",
        "            fflush(stdout);",
        "        }",
        "    }",
        *display_close_lines,
        *feedback_finalize_lines,
        *((
            "    { unsigned long long available ="
            " turing_trace_available(&trace_ring);",
            "      unsigned long long lost = turing_trace_lost(&trace_ring);",
            "      unsigned long long index;",
            "      fprintf(stderr,"
            " \"{\\\"trace\\\":{\\\"records\\\":%llu,"
            "\\\"lost\\\":%llu,\\\"launches\\\":[\", available, lost);",
            "      for (index = 0; index < available; ++index) {",
            "        const TuringTraceRecord *record ="
            " &trace_ring.records[index % trace_ring.capacity];",
            "        fprintf(stderr, \"%s{\\\"seq\\\":%llu,"
            "\\\"shell_ns\\\":%llu,\\\"device_ns\\\":%llu,"
            "\\\"region\\\":%d,\\\"status\\\":%d}\","
            " index ? \",\" : \"\", record->sequence, record->shell_ns,"
            " record->device_ns, record->region, record->status);",
            "      }",
            "      fprintf(stderr, \"]}}\"); fputc(10, stderr); }",
        ) if trace else ()),
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
    trace: bool = False,
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
    input_parameters = tuple(
        item for item in values if item.role in {"input", "inout"}
    )
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
            trace=trace,
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
    keyed_table_fields: frozenset[str] = frozenset(),
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

    def node_operation(data: Mapping[str, Any]) -> str:
        """Canonical ProcessGraph operation spelling for field analysis.

        Authored graph nodes carry the semantic class in ``type`` and often a
        lower-case executable spelling in ``op``.  Preferring ``op`` without
        normalizing it made ``type=GetAttr, op=getattr`` invisible to the
        whole-object resolver, exactly where object fields should become
        native record storage.
        """

        return str(data.get("op") or data.get("type") or "").casefold()

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
        if node_operation(data) != "getattr":
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
    # A contract-declared keyed field is a lookup table too, but it is a
    # program-ABI record field, not a class-field aggregate, so it must not
    # enter ``field_sequence_ids`` (that registry engages the object-field
    # arena machinery).  Same identity convention: the GetAttr's own value id
    # names the table, one canonical id per field.
    keyed_field_sequence_ids: dict[str, int] = {}
    if keyed_table_fields:
        for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
            data = graph_obj.nodes[node_id]
            if node_operation(data) != "getattr":
                continue
            attribute = str(
                (data.get("attributes") or {}).get("attribute") or ""
            )
            if attribute not in keyed_table_fields:
                continue
            keyed_field_sequence_ids.setdefault(
                attribute, int(data.get("value_id", node_id))
            )
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
        if node_operation(data) != "indexed":
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
        if str(field_name) in keyed_field_sequence_ids:
            return (
                keyed_field_sequence_ids[str(field_name)],
                f"keyed.{field_name}",
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
        node_type = node_operation(data)
        attribute = (data.get("attributes") or {}).get("attribute")
        canonical_attribute = (
            canonical_field(str(attribute)) if attribute is not None else None
        )
        if canonical_attribute is None or canonical_attribute not in slot_of:
            continue
        if node_type == "getattr":
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
        elif node_type == "setattr":
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
            if node_operation(source_data) == "staticreference":
                reference_identity = source_attributes.get(
                    "static_python_reference"
                )
                if reference_identity is not None:
                    from .string_table import string_token

                    identity = str(reference_identity)
                    const_sources[int(source_id)] = {
                        "ssa_reference_identity": identity,
                        "reference_kind": "static-python",
                        "reference_handle": string_token(
                            "\x00turing.reference.static-python\x00" + identity
                        ),
                        "host_resident": True,
                    }
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
            if node_operation(source_data) in {"const", "constant"}:
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
        if node_operation(data) != "indexed":
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
    # ``d.get(key, default)`` is the same lookup ``d[key]`` is -- the key's
    # token walked against the table -- differing only in what the absent
    # branch yields.  Recognising only ``indexed`` left ``get`` unclaimed, so
    # its result crossed every backend as a producerless argument.  The
    # authored default rides beside the lookup by result id.
    table_lookup_defaults: dict[int, Any] = {}
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if node_operation(data) != "get":
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        base_id = by_role.get("operand") or by_role.get("value")
        key_id = by_role.get("arg:0")
        if base_id is None or key_id is None or base_id not in graph_obj:
            continue
        sequence_id, _storage_identity = table_sequence(base_id)
        if sequence_id is None:
            continue
        result_id = int(data.get("value_id", node_id))
        key_id = int(graph_obj.nodes[key_id].get("value_id", key_id))
        table_lookups.append((result_id, key_id, int(sequence_id)))
        default_node = by_role.get("arg:1")
        if default_node is not None and default_node in graph_obj:
            default_data = graph_obj.nodes[default_node]
            literal = default_data.get("constant")
            if literal is None:
                literal = (
                    default_data.get("attributes") or {}
                ).get("value")
            if isinstance(literal, (int, float)) and not isinstance(
                literal, bool
            ):
                table_lookup_defaults[result_id] = float(literal)
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if node_operation(data) != "indexedstore":
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
    # A record-field dict is a table exactly as a local one is, but declare it
    # only where a table operation actually addresses it: a declaration
    # materializes anonymous descriptor storage into the frame, and doing that
    # in functions that merely ITERATE the mapping displaced their public-span
    # correlation for every unrelated rank-2 field.
    field_table_ids = {
        int(sequence_id)
        for sequence_id in (
            *field_sequence_ids.values(),
            *keyed_field_sequence_ids.values(),
        )
    }
    referenced_table_ids = {
        int(sequence_id)
        for _result, _query, sequence_id in table_lookups
    } | {
        int(sequence_id)
        for _effect, _key, _value, sequence_id in table_stores
    }
    declared_ids = {
        int(sequence_id) for sequence_id, *_rest in sequence_declarations
    }
    sequence_declarations.extend(
        (int(sequence_id), "unique", 2, False)
        for sequence_id in sorted(field_table_ids & referenced_table_ids)
        if int(sequence_id) not in declared_ids
    )
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
        dict(table_lookup_defaults),
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
    from .precompile_to_ssa import (
        lower_control_sections_to_ssa,
        resolve_sequence_schemas,
    )
    from .string_table import StringTable

    # One table for the whole object: every method's string constants tokenize
    # into it, and it persists token -> word for reverse lookup.
    string_table = StringTable()

    all_functions: dict[str, Any] = {}
    all_tensor_tables: dict[str, Any] = {}
    all_sequence_tables: dict[str, Any] = {}
    all_record_tables: dict[str, Any] = {}
    all_reference_tables: dict[str, Any] = {}
    machine_control_links: list[Any] = []
    machine_indirect_links: list[Any] = []
    pending_call_records: list[tuple[str, Any, Any, Any, Any]] = []

    def dependency_closure(graph: Any, seeds: Any) -> set[int]:
        """Follow the ProcessGraph's already-authored value dependencies."""

        retained = set(map(int, seeds))
        stack = list(retained)
        while stack:
            value_id = stack.pop()
            data = graph.nodes.get(int(value_id), {})
            for parent, role in data.get("parents") or ():
                parent = int(parent)
                if str(role) == "callee" or parent in retained:
                    continue
                retained.add(parent)
                stack.append(parent)
        return retained
    class_table = None
    source_function_table = getattr(
        getattr(compilation.deployment, "process_graph", None),
        "function_table",
        None,
    )
    deployment_graph = getattr(
        getattr(compilation.deployment, "process_graph", None), "G", None
    )
    program_abi = (
        {}
        if deployment_graph is None
        else dict(deployment_graph.graph.get("program_abi") or {})
    )
    function_symbols: dict[int, str] = {}
    shell_symbols: dict[int, str] = {}
    section_outputs: dict[str, tuple[Any, ...]] = {}
    export_symbols: list[str] = []
    lowering_failures: list[tuple[str, Any]] = []
    planned_shells = tuple(_walk_planned_shells(
        compilation.deployment,
        include_function_registry=not bool(getattr(
            compilation.deployment, "runtime_closure_only", False
        )),
    ))
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
            all_reference_tables.update(
                getattr(host_module, "reference_tables", {})
            )
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
    # A sequence's numeric id is a global identity across the whole program
    # (it traces back to one shared ProcessGraph node's value_id), but each
    # shell (method) below lowers with its own local, independently-inferred
    # view of any sequence it touches. Two shells touching the same sequence
    # can therefore disagree not just on element dtype but on the sequence's
    # actual shape (how many storage cells it has) -- a real memory-layout
    # bug, not a cosmetic one. Survey every shell's raw declarations here,
    # before any shell's lowering commits to a shape, and resolve one
    # structural schema per sequence_id that every shell's lowering below
    # will be handed and required to agree with. See
    # ResolvedSequenceSchema in precompile_to_ssa.py for the full rationale.
    keyed_table_fields = frozenset(
        str(_field_name)
        for _record in dict(program_abi.get("records") or {}).values()
        for _field_name, _field in dict(_record.get("fields") or {}).items()
        if str(_field.get("storage") or "") == "keyed"
    )
    shell_sequence_evidence: list[dict[str, Any]] = []
    for shell in planned_shells:
        graph = getattr(shell, "process_graph", None)
        graph_obj = graph.G if graph is not None else None
        if graph_obj is None:
            continue
        (
            _self_id, _field_ops, _const_sources, _field_count, _field_names,
            _record_identity, sequence_initializations, _field_aliases,
            sequence_declarations, _sequence_memberships, _table_lookups,
            _table_lookup_defaults, _table_stores, table_deletions,
            retained_sequence_ids, nested_sequence_ids, _nested_record_fields,
        ) = _field_slot_ops(
            graph_obj,
            retained_storage_identities=frozenset(retained_storage_identities),
            keyed_table_fields=keyed_table_fields,
        )
        shell_sequence_evidence.append({
            "sequence_declarations": sequence_declarations,
            "sequence_initializations": sequence_initializations,
            "table_deletions": table_deletions,
            "retained_sequence_ids": retained_sequence_ids,
            "nested_sequence_ids": nested_sequence_ids,
        })
    resolved_sequence_schemas, sequence_schema_shortfalls = (
        resolve_sequence_schemas(
            shell_sequence_evidence, location="sequence-schema-survey",
        )
    )
    if sequence_schema_shortfalls:
        lowering_failures.extend(
            ("<sequence-schema-survey>", item)
            for item in sequence_schema_shortfalls
        )
    for shell in planned_shells:
        graph = getattr(shell, "process_graph", None)
        graph_obj = graph.G if graph is not None else None
        function_name = (
            graph_obj.graph.get("function_name") if graph_obj is not None else None
        )
        if function_name is None:
            continue
        if program_abi and not graph_obj.graph.get("parameter_record_abi"):
            selected = _record_receipts_for_function(
                program_abi,
                str(function_name),
                graph_obj.graph.get("function_parameters") or (),
                method_owner=graph_obj.graph.get("method_owner"),
            )
            if selected:
                graph_obj.graph["parameter_record_abi"] = selected
        if program_abi and not graph_obj.graph.get("parameter_value_abi"):
            parameters = set(map(
                str, graph_obj.graph.get("function_parameters") or ()
            ))
            selected_values = {}
            for binding in tuple(program_abi.get("values") or ()):
                parameter = str(binding.get("parameter") or "")
                if (
                    parameter in parameters
                    and fnmatchcase(
                        str(function_name), str(binding.get("function") or "")
                    )
                ):
                    selected_values[parameter] = dict(binding)
            if selected_values:
                graph_obj.graph["parameter_value_abi"] = selected_values
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

            # Several conditions in one ``if/elif`` cascade can share a
            # scheduled predicate region.  Each isolated conditional view
            # names that prefix, but nesting all views would execute it once
            # per level.  Hoist only repeated top-level predicate prefixes
            # back to the flat schedule; branch regions remain owned by their
            # lexical conditional and are embedded below as usual.
            prefix_counts: dict[int, int] = {}
            prefixes_by_control: list[tuple[int, ...]] = []
            for conditional_control in conditional_controls:
                prefixes = []
                for block in conditional_control.root.blocks:
                    if isinstance(block, ConditionalBlock):
                        break
                    if (
                        isinstance(block, StatementBlock)
                        and len(block.lines) == 1
                        and block.lines[0].startswith("__scheduled_region_")
                    ):
                        region = int(
                            block.lines[0][len("__scheduled_region_"):-2]
                        )
                        prefixes.append(region)
                        prefix_counts[region] = prefix_counts.get(region, 0) + 1
                prefixes_by_control.append(tuple(prefixes))
            shared_prefixes = {
                region for region, count in prefix_counts.items() if count > 1
            }
            if shared_prefixes:
                conditional_controls = tuple(
                    replace(
                        conditional_control,
                        root=SequenceBlock(tuple(
                            block
                            for block in conditional_control.root.blocks
                            if not (
                                isinstance(block, StatementBlock)
                                and len(block.lines) == 1
                                and block.lines[0].startswith(
                                    "__scheduled_region_"
                                )
                                and int(block.lines[0][
                                    len("__scheduled_region_"):-2
                                ]) in shared_prefixes
                            )
                        )),
                        region_indices=tuple(
                            region
                            for region in conditional_control.region_indices
                            if int(region) not in shared_prefixes
                        ),
                    )
                    for conditional_control in conditional_controls
                )

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
                    "conditional control duplicated scheduled regions in "
                    f"{function_name!r}: {duplicates!r}"
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
        specialization_contract = (
            graph_obj.graph.get("planner_specializations") or {},
            graph_obj.graph.get("planner_tensor_descriptors") or {},
        )
        if any(specialization_contract):
            specialization_digest = hashlib.sha256(
                repr(specialization_contract).encode("utf-8")
            ).hexdigest()[:12]
            symbol_suffix = (
                f"{symbol_suffix}__specialized_{specialization_digest}"
            )
        symbol = f"{artifact_name}__{symbol_suffix}"
        shell_symbols[id(shell)] = symbol
        if function_reference is not None:
            function_symbols[int(function_reference)] = symbol
        # Instance fields flow through the object's field arena: ``self`` is a
        # slot array, a field read is a load from its slot, a field write a
        # store. In whole-program precompile mode the field-op region is never
        # built (gated behind ``not precompile_only``), so recover the field ops
        # from the process graph and hand them to the lowerer as slot access.
        self_id, field_ops, const_sources, field_count, field_names, record_identity, sequence_initializations, field_aliases, sequence_declarations, sequence_memberships, table_lookups, table_lookup_defaults, table_stores, table_deletions, retained_sequence_ids, nested_sequence_ids, nested_record_fields = _field_slot_ops(
            graph_obj,
            retained_storage_identities=frozenset(retained_storage_identities),
            # A contract-declared keyed field is a lookup table, but it is a
            # program-ABI record field, NOT a class-field aggregate: seeding
            # it into class_field_aggregate_kinds engaged the object-field
            # arena machinery in every frame and displaced public-span
            # correlation for unrelated fields.  This channel reaches only
            # table recognition.
            keyed_table_fields=frozenset(
                str(_field_name)
                for _record in dict(program_abi.get("records") or {}).values()
                for _field_name, _field in dict(
                    _record.get("fields") or {}
                ).items()
                if str(_field.get("storage") or "") == "keyed"
            ),
        )
        from .hierarchical_plan import PlanCall

        local_plan_calls = tuple(
            item
            for item in getattr(shell, "hierarchy_plan", ()).items
            if isinstance(item, PlanCall)
        )
        identities = graph_obj.graph.get("identity_table") or {}
        region_output_value_ids = {
            int(region_index): tuple(map(
                int, subgraph.G.graph.get("deployment_outputs", ()),
            ))
            for region_index, subgraph in enumerate(
                getattr(shell, "dispatch_subgraphs", ())
            )
        }
        parameter_facts = {
            **dict(graph_obj.graph.get("parameter_defaults") or {}),
            **dict(graph_obj.graph.get("planner_specializations") or {}),
        }
        parameter_value_dtypes = {}
        def scalar_fact_dtype(fact):
            if isinstance(fact, bool):
                return "bool"
            if isinstance(fact, int):
                return "int"
            if isinstance(fact, float):
                return "float64"
            return None

        for parameter_name in tuple(
            graph_obj.graph.get("function_parameters") or ()
        ):
            history = tuple(identities.get(str(parameter_name), ()))
            fact_dtype = scalar_fact_dtype(
                parameter_facts.get(str(parameter_name))
            )
            if history and fact_dtype is not None:
                parameter_value_dtypes[int(history[0])] = fact_dtype
        for value_id, data in graph_obj.nodes(data=True):
            if str(data.get("type") or "").casefold() != "input":
                continue
            parameter_name = str(
                (data.get("attributes") or {}).get("binding_name") or ""
            )
            fact = parameter_facts.get(parameter_name)
            fact_dtype = scalar_fact_dtype(fact)
            if fact_dtype is not None:
                parameter_value_dtypes[int(value_id)] = fact_dtype
        # A free-function record assignment already has an exact dependency
        # edge: SetAttr(object=<parameter>, value=<producer>).  Preserve that
        # producer identity across the region/control call boundary so the
        # target can bind it directly to the caller's field storage.  Method
        # ``self`` writes use the class field-slot arena handled by
        # ``field_ops`` above and must not be duplicated here.
        parameter_record_write_value_ids: list[int] = []
        declared_record_parameters = {
            str(parameter_name)
            for parameter_name in dict(
                graph_obj.graph.get("parameter_record_abi") or {}
            )
            if str(parameter_name) != "self"
        }
        record_parameter_value_ids = {
            int(value_id)
            for parameter_name in declared_record_parameters
            for value_id in identities.get(parameter_name, ())
        }
        if record_parameter_value_ids:
            for _node_id, data in graph_obj.nodes(data=True):
                if str(
                    data.get("op") or data.get("type") or ""
                ).casefold() != "setattr":
                    continue
                parents = tuple(data.get("parents") or ())
                object_value_ids = {
                    int(graph_obj.nodes[parent].get("value_id", parent))
                    for parent, role in parents
                    if str(role) in {"object", "base", "receiver"}
                    and parent in graph_obj
                }
                if not object_value_ids.intersection(
                    record_parameter_value_ids
                ):
                    continue
                parameter_record_write_value_ids.extend(
                    int(graph_obj.nodes[parent].get("value_id", parent))
                    for parent, role in parents
                    if str(role) == "value" and parent in graph_obj
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
                value_dtypes=parameter_value_dtypes,
                region_output_value_ids=region_output_value_ids,
                record_field_write_value_ids=tuple(dict.fromkeys(
                    parameter_record_write_value_ids
                )),
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
                table_lookup_defaults=table_lookup_defaults,
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
                resolved_sequence_schemas=resolved_sequence_schemas,
            )
        )
        if shortfalls:
            lowering_failures.extend((symbol, item) for item in shortfalls)
        all_functions.update(module_ir.functions)
        lowered_control = module_ir.functions.get(symbol)
        if lowered_control is not None:
            source_output_value_ids = tuple(dict.fromkeys(
                int(history[-1])
                for name in tuple(
                    graph_obj.graph.get("function_outputs") or ()
                )
                for history in (tuple(
                    (graph_obj.graph.get("identity_table") or {}).get(
                        str(name), ()
                    )
                ),)
                if history
            ))
            lowered_control.metadata.update({
                "source_conditional_count": len(conditional_controls),
                "lowered_conditional_count": lowered_conditional_count,
                "source_output_value_ids": source_output_value_ids,
                "parameter_record_abi": copy.deepcopy(
                    graph_obj.graph.get("parameter_record_abi") or {}
                ),
                "parameter_value_abi": copy.deepcopy(
                    graph_obj.graph.get("parameter_value_abi") or {}
                ),
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
        all_reference_tables.update(
            getattr(module_ir, "reference_tables", {})
        )
        pending_call_records.extend(
            (symbol, item, graph_obj, module_ir, shell)
            for item in local_plan_calls
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
        returns = tuple(
            instruction.args
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret", "Return", "return"}
        )
        if returns:
            return tuple(returns[-1])
        if name in section_outputs and section_outputs[name]:
            return section_outputs[name]
        return ()

    from ..transmogrifier.ssa import (
        Instr,
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

    source_graphs_by_symbol = {
        shell_symbols.get(
            id(shell),
            f"{artifact_name}__{graph.graph.get('function_name')}",
        ): graph
        for shell in planned_shells
        for graph in (
            getattr(getattr(shell, "process_graph", None), "G", None),
        )
        if graph is not None and graph.graph.get("function_name") is not None
    }

    abi_records = dict(program_abi.get("records") or {})

    def abi_record_for_call(data: Mapping[str, Any]):
        if str(data.get("type") or data.get("op") or "").casefold() != "call":
            return None
        attributes = dict(data.get("attributes") or {})
        candidates = tuple(filter(None, (
            attributes.get("class_ref"),
            attributes.get("static_python_reference"),
        )))
        for record_name, record in abi_records.items():
            identity = str(record.get("identity") or record_name)
            if any(
                str(candidate) in {str(record_name), identity}
                or identity.endswith("." + str(candidate))
                for candidate in candidates
            ):
                return str(record_name), record
        return None

    # A value retained for a later source-linked call can be produced inside a
    # numerical region even though it is not a public function result.  The
    # control lowerer cannot see pending PlanCall consumers yet, so extend the
    # region aggregate from both the function's explicit source-output ledger
    # and every pending PlanCall feed before call linking.  A call is an
    # authored consumer just as surely as Ret is; omitting those feeds lets the
    # region lowerer prune values which the later call-frame linker then cannot
    # bind.  These projections are ordinary local SSA values; callers may
    # remove the hidden name from their final Ret afterward.
    pending_call_feed_ids: dict[str, set[int]] = {}
    for caller_symbol, planned_call, _graph, _module, _shell in (
        pending_call_records
    ):
        pending_call_feed_ids.setdefault(str(caller_symbol), set()).update(
            int(caller_id)
            for caller_id, _callee_id in planned_call.argument_bindings
        )
    # Schema-declared record literals are structural consumers even when the
    # external/dataclass constructor has no pursued method shell. Preserve
    # their authored positional/keyword feeds through numerical partitioning.
    for caller_symbol, graph in source_graphs_by_symbol.items():
        for _node_id, data in graph.nodes(data=True):
            if abi_record_for_call(data) is None:
                continue
            pending_call_feed_ids.setdefault(str(caller_symbol), set()).update(
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role) != "callee"
            )
    for caller_symbol, caller in all_functions.items():
        pending_call_feed_ids.setdefault(str(caller_symbol), set()).update(
            map(int, caller.metadata.get("source_output_value_ids", ()))
        )
    # Public structural expressions (BoolOp, tuple/record construction, field
    # publication) are not themselves numerical regions. Retain their exact
    # operand closure so every numerical ancestor survives until structural
    # SSA reconstruction; stopping at the public node alone can prune the
    # second comparison in ``a and b`` or a constructor keyword constant.
    for caller_symbol, seeds in tuple(pending_call_feed_ids.items()):
        graph = source_graphs_by_symbol.get(str(caller_symbol))
        if graph is None:
            continue
        pending_call_feed_ids[str(caller_symbol)] = dependency_closure(
            graph, seeds
        )
    for caller_name, caller in all_functions.items():
        desired_ids = tuple(dict.fromkeys((
            *map(int, caller.metadata.get("source_output_value_ids", ())),
            *sorted(pending_call_feed_ids.get(str(caller_name), ())),
        )))
        if not desired_ids:
            continue
        # A region-call's OUT-params are pointers passed as ARGS, not as
        # instruction.res -- the call writes through them directly, the
        # same in-place mechanism a loop-carried phi's own update uses.
        # Checking only `caller.args` and `instr.res` treats an id already
        # satisfied this way as still "desired", so the aggregate-unpack
        # materialization below built a SECOND, competing producer for it
        # (a GetElementPtr+Load pair reading a bogus address) -- and
        # whichever one rendered last in the backend's id-keyed pointer
        # cache silently won, clobbering the call's real, correct write
        # with garbage. Any id already referenced as an operand anywhere in
        # this function already has a valid SSAValue for it in scope and
        # must not be re-materialized.
        available = {
            int(value.id) for value in caller.args
        } | {
            int(instruction.res.id)
            for block in caller.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        } | {
            int(argument.id)
            for block in caller.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        }
        next_projection_id = 1 + max(available, default=0)
        for desired_id in desired_ids:
            if desired_id in available:
                continue
            producer = None
            for block in caller.blocks.values():
                for index, instruction in enumerate(block.instrs):
                    if (
                        instruction.op not in {"Call", "call"}
                        or instruction.res is None
                        or instruction.attributes.get("region_index") is None
                        or instruction.attributes.get("source_linked")
                        or instruction.attributes.get("result_convention")
                        != "ssa.aggregate"
                    ):
                        continue
                    callee_symbol = str(
                        instruction.attributes.get("callee") or ""
                    )
                    callee = all_functions.get(callee_symbol)
                    if callee is None:
                        continue
                    produced_value = next((
                        candidate.res
                        for callee_block in callee.blocks.values()
                        for candidate in callee_block.instrs
                        if candidate.res is not None
                        and int(candidate.res.id) == desired_id
                    ), None)
                    if produced_value is not None:
                        producer = (
                            block, index, instruction,
                            callee_symbol, produced_value,
                        )
                        break
                if producer is not None:
                    break
            if producer is None:
                source = (
                    source_graphs_by_symbol.get(str(caller_name)).nodes.get(
                        int(desired_id), {}
                    )
                    if source_graphs_by_symbol.get(str(caller_name)) is not None
                    else {}
                )
                unresolved = list(caller.metadata.get(
                    "unresolved_required_source_values", ()
                ))
                unresolved.append((
                    int(desired_id),
                    str(source.get("op") or source.get("type") or ""),
                    tuple(
                        (int(parent), str(role))
                        for parent, role in source.get("parents") or ()
                    ),
                ))
                caller.metadata["unresolved_required_source_values"] = tuple(
                    dict.fromkeys(unresolved)
                )
                continue
            block, call_index, call, callee_symbol, produced_value = producer
            declared = list(map(
                int, call.attributes.get("output_ids", ())
            ))
            if desired_id in declared:
                continue
            output_index = len(declared)
            declared.append(desired_id)
            call.attributes["output_ids"] = tuple(declared)
            index_value = SSAValue(next_projection_id, dtype="int")
            next_projection_id += 1
            address = SSAValue(next_projection_id, dtype="ptr")
            next_projection_id += 1
            result = SSAValue(
                desired_id,
                dtype=produced_value.dtype,
                shape=tuple(produced_value.shape or ()),
                device=produced_value.device,
                accounting=dict(produced_value.accounting),
            )
            block.instrs[call_index + 1:call_index + 1] = [
                Instr("Const", [], index_value, attributes={"value": output_index}),
                Instr(
                    "GetElementPtr", [call.res, index_value], address,
                    attributes={
                        "aggregate_index": output_index,
                        "source_output_id": desired_id,
                    },
                ),
                Instr(
                    "Load", [address], result,
                    attributes={
                        "aggregate_index": output_index,
                        "source_output_id": desired_id,
                    },
                ),
            ]
            existing_outputs = tuple(section_outputs.get(callee_symbol, ()))
            section_outputs[callee_symbol] = (
                existing_outputs
                if desired_id in {int(value.id) for value in existing_outputs}
                else (*existing_outputs, produced_value)
            )
            available.add(desired_id)

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
    def function_values(function: Any) -> dict[int, Any]:
        values = {int(value.id): value for value in function.args}
        values.update({
            int(instruction.res.id): instruction.res
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        })
        return values

    def recover_structural_source_outputs(
        symbol: str, graph: Any
    ) -> None:
        """Publish source results whose producer is structural, not numeric."""

        function = all_functions.get(symbol)
        if function is None:
            return
        returns = [
            instruction
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret", "Return", "return"}
        ]
        if not returns:
            return
        terminator = returns[-1]
        published = {int(value.id) for value in terminator.args}
        values = function_values(function)
        insertions = []
        structural_shortfalls = []
        # Selection chains need their own intermediate results; a graph node id
        # only exists for the authored expression itself. These stay inside the
        # function's own SSA numbering -- graph node keys are Python object
        # identities and are not part of that space.
        next_structural_id = 1 + max((
            *values,
            *(
                int(data["value_id"])
                for _node_id, data in graph.nodes(data=True)
                if isinstance(data.get("value_id"), int)
            ),
        ), default=0)

        def structural_boolop_value(value_id: int, data, canonical: str):
            """Lower ``and``/``or`` as the operand selection Python defines.

            ``a or b`` evaluates to ``a`` when ``a`` is truthy and otherwise to
            ``b``; it is not the boolean ``a | b``. The two agree only when
            both operands are already boolean, which is the ordinary condition
            case, so that keeps its cheap logical opcode. Any other operand --
            a dict, a reference, a number -- must keep its own value and type,
            so it lowers to ``Select``, whose mask every backend already
            resolves through the same truthiness rule it uses elsewhere.
            Returns ``None`` to let the caller fall through to the boolean
            opcode.
            """

            ordered = []
            for parent, role in data.get("parents") or ():
                role = str(role)
                if role == "callee":
                    continue
                # BoolOp operands are ordered edges; `or` is not commutative
                # over values, so the authored order is the exact order.
                index = (
                    int(role.split(":", 1)[1])
                    if role.startswith("value:") and role.split(":", 1)[1].isdigit()
                    else len(ordered)
                )
                ordered.append((index, int(parent)))
            ordered.sort()
            if len(ordered) < 2:
                return None
            operands = []
            for _index, parent in ordered:
                operand = ensure_structural_value(parent)
                if operand is None:
                    return None
                operands.append(operand)

            def destroys_value(operand) -> bool:
                # A declared container or reference has no boolean form at all:
                # combining it yields a truth value and the dict, span, or
                # record it named is gone. A declared non-boolean scalar keeps
                # its own value too. An operand whose type is still unknown is
                # left to the boolean opcode rather than guessed at here.
                storage = str(
                    (operand.accounting or {}).get("program_abi_storage") or ""
                )
                if storage in {"reference", "span", "record", "keyed"}:
                    return True
                dtype = str(getattr(operand, "dtype", "") or "").casefold()
                return bool(dtype) and dtype not in {"bool", "unknown"}

            if not any(destroys_value(operand) for operand in operands):
                return None

            nonlocal next_structural_id
            current = operands[0]
            for position, operand in enumerate(operands[1:], start=1):
                last = position == len(operands) - 1
                if last:
                    result_id = value_id
                else:
                    result_id = next_structural_id
                    next_structural_id += 1
                dtype = current.dtype or operand.dtype
                result = SSAValue(int(result_id), dtype=dtype)
                # Select(mask, when_true, when_false). `or` keeps the left
                # operand when it is truthy; `and` keeps the right one.
                arguments = (
                    [current, current, operand]
                    if canonical == "logical_or"
                    else [current, operand, current]
                )
                insertions.append(Instr(
                    "Select", arguments, result,
                    attributes={
                        "structural_operation": "boolop",
                        "semantic_family": canonical,
                        "short_circuit_selection": True,
                    },
                ))
                values[int(result_id)] = result
                current = result
            return current

        def ensure_structural_value(value_id: int):
            """Lower a missing direct expression from its exact graph edges."""

            value_id = int(value_id)
            if value_id in values:
                return values[value_id]
            data = graph.nodes.get(value_id, {})
            operation = str(
                data.get("op") or data.get("type") or ""
            ).casefold()
            attributes = dict(data.get("attributes") or {})
            if operation in {"constant", "const"}:
                expression = data.get("expr_obj")
                if (
                    "value" not in attributes
                    and "constant" not in data
                    and not isinstance(expression, ast.Constant)
                ):
                    structural_shortfalls.append((value_id, operation, "constant-value"))
                    return None
                literal = attributes.get(
                    "value",
                    data.get(
                        "constant",
                        expression.value
                        if isinstance(expression, ast.Constant) else None,
                    ),
                )
                dtype = (
                    "bool" if isinstance(literal, bool)
                    else "int64" if isinstance(literal, int)
                    else "float64" if isinstance(literal, float)
                    else None
                )
                result = SSAValue(value_id, dtype=dtype)
                insertions.append(Instr(
                    "Const", [], result, attributes={"value": literal},
                ))
                values[value_id] = result
                return result
            if operation in {"loopresult", "loopexit", "identity"}:
                parents = tuple(data.get("parents") or ())
                for preferred_role in (
                    "updated", "value", "body", "initial", "orelse"
                ):
                    for parent, role in parents:
                        if str(role) != preferred_role:
                            continue
                        result = ensure_structural_value(int(parent))
                        if result is not None:
                            values[value_id] = result
                            return result
                structural_shortfalls.append((
                    value_id, operation, "carried-value"
                ))
                return None
            canonical = {
                "add": "add", "sub": "sub", "mul": "mul",
                "div": "truediv", "truediv": "truediv",
                "greater": "greater", "gt": "greater",
                "less": "less", "lt": "less",
                "greaterequal": "greater_equal",
                "greater_equal": "greater_equal",
                "lessequal": "less_equal", "less_equal": "less_equal",
                "equal": "equal", "eq": "equal",
                "notequal": "not_equal", "not_equal": "not_equal",
            }.get(operation)
            expression = data.get("expr_obj")
            if operation == "boolop":
                canonical = (
                    "logical_and"
                    if isinstance(getattr(expression, "op", None), ast.And)
                    else "logical_or"
                    if isinstance(getattr(expression, "op", None), ast.Or)
                    else None
                )
                if canonical is not None:
                    selected = structural_boolop_value(
                        value_id, data, canonical,
                    )
                    if selected is not None:
                        return selected
            if canonical is None:
                structural_shortfalls.append((value_id, operation, "operator"))
                return None
            from .ssa_numeric_operators import TENSOR_SSA_OPERATOR_BY_NAME

            row = TENSOR_SSA_OPERATOR_BY_NAME.get(canonical)
            if row is None or not row.is_direct:
                structural_shortfalls.append((value_id, operation, "direct-handler"))
                return None
            arguments = []
            for parent, role in data.get("parents") or ():
                if str(role) == "callee":
                    continue
                argument = ensure_structural_value(int(parent))
                if argument is None:
                    structural_shortfalls.append((
                        value_id, operation, f"operand:{int(parent)}"
                    ))
                    return None
                arguments.append(argument)
            if len(arguments) < 1:
                structural_shortfalls.append((value_id, operation, "arity"))
                return None
            dtype = (
                "bool" if canonical in {
                    "logical_and", "logical_or", "equal", "not_equal",
                    "less", "less_equal", "greater", "greater_equal",
                } else arguments[0].dtype
            )
            result = SSAValue(value_id, dtype=dtype)
            insertions.append(Instr(
                row.handler.value,
                arguments,
                result,
                attributes={
                    "structural_operation": operation,
                    "semantic_family": canonical,
                },
            ))
            values[value_id] = result
            return result

        def literal_value(node_id: int) -> Any:
            data = graph.nodes.get(int(node_id), {})
            attributes = data.get("attributes") or {}
            if "value" in attributes:
                return copy.deepcopy(attributes["value"])
            if "constant" in data:
                return copy.deepcopy(data["constant"])
            expression = data.get("expr_obj")
            if expression is not None:
                try:
                    return ast.literal_eval(expression)
                except (TypeError, ValueError):
                    pass
            operation = str(
                data.get("op") or data.get("type") or ""
            ).casefold()
            if operation in {"list", "tuple"}:
                return [
                    literal_value(int(parent))
                    for parent, role in data.get("parents") or ()
                    if str(role) == "elts"
                ]
            raise ValueError(f"node {node_id} is not a literal")

        def authored_output_literal(output_name: str) -> Any:
            if source_function_table is None:
                raise ValueError(output_name)
            function_reference = graph.graph.get("function_ref")
            try:
                entry = source_function_table.entry(int(function_reference))
            except (KeyError, TypeError, ValueError):
                raise ValueError(output_name) from None
            callable_object = getattr(entry, "python_callable", None)
            if callable_object is None:
                raise ValueError(output_name)
            try:
                tree = ast.parse(textwrap.dedent(
                    inspect.getsource(callable_object)
                ))
            except (OSError, TypeError, IndentationError, SyntaxError):
                raise ValueError(output_name) from None
            for node in ast.walk(tree):
                target = None
                value = None
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        tuple(node.targets)
                        if isinstance(node, ast.Assign)
                        else (node.target,)
                    )
                    if any(
                        isinstance(candidate, ast.Name)
                        and candidate.id == output_name
                        for candidate in targets
                    ):
                        target = output_name
                        value = node.value
                if target is None or value is None:
                    continue
                candidate = (
                    value.args[0]
                    if isinstance(value, ast.Call) and value.args
                    else value
                )
                try:
                    return ast.literal_eval(candidate)
                except (TypeError, ValueError):
                    continue
            raise ValueError(output_name)

        identities = graph.graph.get("identity_table") or {}
        record_table = all_record_tables.get(symbol)
        returned_record_layouts = []
        for output_name in tuple(graph.graph.get("function_outputs") or ()):
            history = tuple(identities.get(str(output_name), ()))
            if not history:
                continue
            output_id = int(history[-1])
            if output_id in published:
                continue
            data = graph.nodes.get(output_id, {})
            record = (
                None if record_table is None
                else record_table.records.get(output_id)
            )
            if record is not None:
                matched = abi_record_for_call(data)
                if matched is not None:
                    _record_name, contract_record = matched
                    existing_fields = {field.name for field in record.fields}
                    keyword_values = {
                        str(role).split(":", 1)[1]: int(parent)
                        for parent, role in data.get("parents") or ()
                        if str(role).startswith("kw:")
                    }
                    appended_fields = []
                    for field_name, field in dict(
                        contract_record.get("fields") or {}
                    ).items():
                        if field_name in existing_fields:
                            continue
                        source_id = keyword_values.get(str(field_name))
                        source = (
                            None if source_id is None
                            else ensure_structural_value(source_id)
                        )
                        if source is None:
                            continue
                        if str(field["storage"]) == "keyed":
                            # Three physical slots, correlated from the mapping
                            # literal's own key/value edges -- see the
                            # constructor-literal path.
                            continue
                        storage = {
                            "scalar": SSARecordFieldStorage.SCALAR,
                            "span": SSARecordFieldStorage.SPAN,
                            "reference": SSARecordFieldStorage.REFERENCE,
                            "record": SSARecordFieldStorage.RECORD,
                        }[str(field["storage"])]
                        if storage is SSARecordFieldStorage.RECORD:
                            continue
                        appended_fields.append(SSARecordFieldDescriptor(
                            str(field_name), storage,
                            storage_identity=(
                                f"{contract_record['identity']}.{field_name}"
                            ),
                            value_ids=(int(source.id),),
                            dtype=field.get("dtype"),
                            writable=bool(field.get("mutable", False)),
                        ))
                    if appended_fields:
                        record = replace(
                            record, fields=(*record.fields, *appended_fields)
                        )
                        record_table.records[output_id] = record
                layout = tuple(
                    int(value_id)
                    for field in record.fields
                    for value_id in field.value_ids
                    if int(value_id) in values
                )
                for value_id in layout:
                    if value_id not in published:
                        terminator.args.append(values[value_id])
                        published.add(value_id)
                returned_record_layouts.append((output_id, layout))
                continue
            reconstructed = ensure_structural_value(output_id)
            if reconstructed is not None:
                terminator.args.append(reconstructed)
                published.add(output_id)
                continue
            if output_id in values:
                terminator.args.append(values[output_id])
                published.add(output_id)
                continue
            attributes = data.get("attributes") or {}
            operation = str(
                data.get("op") or data.get("type") or ""
            ).casefold()
            result = None
            instruction = None
            if (
                operation == "input"
                and str(output_name) not in set(map(
                    str, graph.graph.get("function_parameters") or ()
                ))
            ):
                try:
                    literal = authored_output_literal(str(output_name))
                except ValueError:
                    literal = None
                if literal is not None:
                    array = np.asarray(literal)
                    result = SSAValue(
                        output_id,
                        dtype=str(array.dtype),
                        shape=tuple(map(int, array.shape)),
                        accounting={
                            "authored_output_literal": str(output_name)
                        },
                    )
                    instruction = Instr(
                        "Const", [], result,
                        attributes={
                            "value": literal,
                            "values": literal,
                            "tensor_operation": "tensor_from_list",
                        },
                    )
                else:
                    tensor = data.get("tensor") or {}
                    result = SSAValue(
                        output_id,
                        dtype=tensor.get("dtype"),
                        shape=tuple(tensor.get("shape") or ()),
                        accounting={
                            "externalized_source_output": str(output_name)
                        },
                    )
                    function.args.append(result)
                    values[output_id] = result
                    terminator.args.append(result)
                    published.add(output_id)
                    continue
            elif operation == "_tensor_from_list":
                data_parent = next((
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role) == "arg:0"
                ), None)
                if data_parent is not None:
                    try:
                        literal = literal_value(data_parent)
                    except ValueError:
                        literal = None
                    if literal is not None:
                        array = np.asarray(literal)
                        result = SSAValue(
                            output_id,
                            dtype=str(array.dtype),
                            shape=tuple(map(int, array.shape)),
                            accounting={
                                "tensor_constructor": "tensor_from_list",
                                "requires_grad": bool(attributes.get(
                                    "requires_grad", False
                                )),
                            },
                        )
                        instruction = Instr(
                            "Const", [], result,
                            attributes={
                                "value": literal,
                                "values": literal,
                                "tensor_operation": "tensor_from_list",
                            },
                        )
            elif (
                operation == "call"
                and attributes.get("static_python_reference") == "id"
            ):
                # `id(x)` takes one argument: argument ZERO, not whichever
                # positional edge the parent set yields first.
                arguments = ordered_arguments(data.get("parents") or ())
                source_id = int(arguments[0]) if arguments else None
                if source_id is not None and source_id in values:
                    result = SSAValue(
                        output_id,
                        dtype="int64",
                        accounting={"tensor_identity": "stable-handle"},
                    )
                    instruction = Instr(
                        "Cast", [values[source_id]], result,
                        attributes={
                            "tensor_operation": "tensor_identity",
                            "reference_cast": True,
                        },
                    )
            elif operation == "boolop":
                operands = [
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role).startswith("value:")
                    and int(parent) in values
                ]
                expression = data.get("expr_obj")
                opcode = (
                    "And" if isinstance(getattr(expression, "op", None), ast.And)
                    else "Or" if isinstance(
                        getattr(expression, "op", None), ast.Or
                    ) else None
                )
                if opcode is not None and len(operands) >= 2:
                    current = values[operands[0]]
                    for operand_id in operands[1:]:
                        is_last = operand_id == operands[-1]
                        combined = SSAValue(
                            output_id if is_last else max(values) + 1,
                            dtype="bool",
                        )
                        insertions.append(Instr(
                            opcode, [current, values[operand_id]], combined,
                            attributes={"structural_operation": "boolop"},
                        ))
                        values[int(combined.id)] = combined
                        current = combined
                    result = current
                    # The instruction sequence was already appended above.
                    instruction = None
                    terminator.args.append(result)
                    published.add(output_id)
            elif operation == "ifexp":
                parameter_id = next((
                    int(value_id)
                    for name in tuple(
                        graph.graph.get("function_parameters") or ()
                    )
                    for value_id in tuple(identities.get(str(name), ()))[:1]
                ), None)
                parameter = values.get(parameter_id)
                shape = tuple(getattr(parameter, "shape", ()) or ())
                if shape and all(int(extent) >= 0 for extent in shape):
                    result = SSAValue(output_id, dtype="int64")
                    instruction = Instr(
                        "Const", [], result,
                        attributes={
                            "value": int(np.prod(shape, dtype=np.int64)),
                            "structural_operation": "nested_count",
                        },
                    )
            if output_id in published:
                continue
            if instruction is None or result is None:
                continue
            insertions.append(instruction)
            values[output_id] = result
            terminator.args.append(result)
            published.add(output_id)
        # Structural expressions can also be private feeds to a later
        # source-linked call.  They need the same exact reconstruction as a
        # public Ret value, but must not be added to Ret merely because a call
        # consumes them.  This is common for authored boolean combinations
        # passed as keyword arguments.
        for required_id in sorted(pending_call_feed_ids.get(symbol, ())):
            if int(required_id) in values:
                continue
            operation = str(
                graph.nodes.get(int(required_id), {}).get("op")
                or graph.nodes.get(int(required_id), {}).get("type")
                or ""
            ).casefold()
            if operation in {
                "boolop", "constant", "const", "loopresult", "loopexit",
                "identity", "add", "sub", "mul", "div", "truediv",
                "greater", "gt", "less", "lt", "greaterequal",
                "greater_equal", "lessequal", "less_equal", "equal", "eq",
                "notequal", "not_equal",
            }:
                ensure_structural_value(int(required_id))
        if insertions:
            for block in function.blocks.values():
                if terminator in block.instrs:
                    index = block.instrs.index(terminator)
                    block.instrs[index:index] = insertions
                    break
            function.metadata["recovered_structural_outputs"] = tuple(
                int(instruction.res.id) for instruction in insertions
            )
        if returned_record_layouts:
            function.metadata["record_return_layouts"] = tuple(
                returned_record_layouts
            )
        if structural_shortfalls:
            function.metadata["structural_output_shortfalls"] = tuple(
                dict.fromkeys(structural_shortfalls)
            )

    # Determine the least record surface required by the whole source-linked
    # call graph.  A function that only forwards ``state`` has no local
    # GetAttr node, but it must still carry exactly the fields read by its
    # descendants.  Propagating these names backward through PlanCall's exact
    # argument bindings avoids both Python object handles and the rejected
    # alternative of expanding every schema field at every call boundary.
    record_parameter_specs: dict[tuple[str, str], Mapping[str, Any]] = {}
    record_parameter_by_value: dict[str, dict[int, tuple[str, str]]] = {}
    record_field_demands: dict[tuple[str, str], set[str]] = {}
    for source_symbol, source_graph in source_graphs_by_symbol.items():
        identities = source_graph.graph.get("identity_table") or {}
        declared = dict(
            source_graph.graph.get("parameter_record_abi") or {}
        )
        by_value = record_parameter_by_value.setdefault(source_symbol, {})
        for parameter_name, record in declared.items():
            key = (str(source_symbol), str(parameter_name))
            record_parameter_specs[key] = record
            record_field_demands.setdefault(key, set())
            parameter_ids = set(map(
                int, identities.get(str(parameter_name), ())
            ))
            for value_id in parameter_ids:
                by_value[int(value_id)] = key
            declared_fields = set(map(
                str, dict(record.get("fields") or {})
            ))
            for node_id, data in source_graph.nodes(data=True):
                if str(
                    data.get("type") or data.get("op") or ""
                ).casefold() != "getattr":
                    continue
                attribute = str(
                    (data.get("attributes") or {}).get("attribute") or ""
                )
                if attribute not in declared_fields:
                    continue
                if any(
                    int(parent) in parameter_ids
                    and str(role) in {"value", "object", "base"}
                    for parent, role in data.get("parents") or ()
                ):
                    record_field_demands[key].add(attribute)

    record_forwarding_edges = []
    for caller_symbol, planned_call, caller_graph, _module, caller_shell in (
        pending_call_records
    ):
        call_data = caller_graph.nodes.get(
            int(planned_call.callsite_id), {}
        )
        attributes = call_data.get("attributes") or {}
        reference = attributes.get(
            "callee_ref",
            attributes.get("method_ref", attributes.get("constructor_ref")),
        )
        child_shell = getattr(
            caller_shell, "callsite_function_shells", {}
        ).get(int(planned_call.callsite_id))
        callee_symbol = (
            shell_symbols.get(id(child_shell))
            if child_shell is not None else None
        ) or (
            None if reference is None
            else function_symbols.get(int(reference))
        )
        if callee_symbol is None:
            continue
        caller_by_value = record_parameter_by_value.get(
            str(caller_symbol), {}
        )
        callee_by_value = record_parameter_by_value.get(
            str(callee_symbol), {}
        )
        for caller_id, callee_id in planned_call.argument_bindings:
            caller_key = caller_by_value.get(int(caller_id))
            callee_key = callee_by_value.get(int(callee_id))
            if caller_key is None or callee_key is None:
                continue
            caller_record = record_parameter_specs[caller_key]
            callee_record = record_parameter_specs[callee_key]
            if str(caller_record.get("identity")) != str(
                callee_record.get("identity")
            ):
                continue
            record_forwarding_edges.append((caller_key, callee_key))

    changed = True
    while changed:
        changed = False
        for caller_key, callee_key in record_forwarding_edges:
            missing = (
                record_field_demands[callee_key]
                - record_field_demands[caller_key]
            )
            if missing:
                record_field_demands[caller_key].update(missing)
                changed = True

    def materialize_parameter_record_abi(symbol: str, graph: Any) -> None:
        """Make contract-declared record fields ordinary physical SSA inputs.

        Read-only scalar views are passed by value and spans are passed as
        arenas. A scalar field actually written by the function remains
        unresolved until its reference/slot ABI is available; passing that
        field by value would silently destroy state updates across a call.
        """

        function = all_functions.get(symbol)
        if function is None:
            return
        declared_records = dict(graph.graph.get("parameter_record_abi") or {})
        if not declared_records:
            return
        identities = graph.graph.get("identity_table") or {}
        values = function_values(function)
        next_physical_id = 1 + max((
            *values,
            *(int(data.get("value_id", node_id))
              for node_id, data in graph.nodes(data=True)),
        ), default=0)
        table = all_record_tables.setdefault(symbol, SSARecordTable())
        for parameter_name, record in declared_records.items():
            demanded_fields = record_field_demands.get(
                (str(symbol), str(parameter_name)), set()
            )
            parameter_ids = set(map(
                int, identities.get(str(parameter_name), ())
            ))
            if not parameter_ids:
                continue
            record_id = next((
                int(value.id) for value in function.args
                if int(value.id) in parameter_ids
            ), min(parameter_ids))
            written_fields = {
                str((data.get("attributes") or {}).get("attribute"))
                for _node_id, data in graph.nodes(data=True)
                if str(data.get("type") or data.get("op") or "").casefold()
                == "setattr"
                and (data.get("attributes") or {}).get("attribute")
                is not None
                and any(
                    int(parent) in parameter_ids
                    and str(role) in {"value", "object", "base", "receiver"}
                    for parent, role in data.get("parents") or ()
                )
            }
            write_source_ids_by_field: dict[str, tuple[int, ...]] = {}
            for _node_id, data in graph.nodes(data=True):
                if (
                    str(data.get("type") or data.get("op") or "").casefold()
                    != "setattr"
                ):
                    continue
                attributes = data.get("attributes") or {}
                field_name = attributes.get("attribute")
                if field_name is None or not any(
                    int(parent) in parameter_ids
                    and str(role) in {"value", "object", "base", "receiver"}
                    for parent, role in data.get("parents") or ()
                ):
                    continue
                sources = tuple(
                    int(graph.nodes[parent].get("value_id", parent))
                    for parent, role in data.get("parents") or ()
                    if str(role) == "value" and parent in graph
                )
                if sources:
                    write_source_ids_by_field[str(field_name)] = tuple(
                        dict.fromkeys((
                            *write_source_ids_by_field.get(
                                str(field_name), ()
                            ),
                            *sources,
                        ))
                    )
            fields = []
            for field_name, field in dict(record.get("fields") or {}).items():
                storage = str(field.get("storage") or "")
                mutable = bool(field.get("mutable", False))
                candidate_ids = tuple(dict.fromkeys((
                    *(
                        int(data.get("value_id", node_id))
                        for node_id, data in graph.nodes(data=True)
                        if str(
                            data.get("type") or data.get("op") or ""
                        ).casefold() == "getattr"
                        and str((
                            data.get("attributes") or {}
                        ).get("attribute")) == str(field_name)
                        and any(
                            int(parent) in parameter_ids
                            and str(role) in {"value", "object", "base"}
                            for parent, role in data.get("parents") or ()
                        )
                    ),
                    *write_source_ids_by_field.get(str(field_name), ()),
                )))
                if not candidate_ids:
                    if str(field_name) not in demanded_fields:
                        continue
                    candidate_ids = (next_physical_id,)
                    next_physical_id += 1
                if storage == "keyed":
                    # Materialize once per function. A second pass over the
                    # same symbol sees different attribute occurrences, so
                    # re-running would append a second set of slots and leave
                    # the mapping naming the first -- ids that no longer
                    # correspond to anything in this frame.
                    already = next((
                        int(existing)
                        for value_id in candidate_ids
                        if (existing := (
                            (values.get(int(value_id)) or SSAValue(-1)
                             ).accounting or {}
                        ).get("program_abi_keyed_length")) is not None
                        and int(existing) in values
                    ), None)
                    if already is not None:
                        continue
                    # A mapping keyed by words is not one opaque handle. It is
                    # a length plus two parallel vectors: the keys as the
                    # repository's universal string tokens, and the values.
                    # Because the token is content-addressed, a constant key
                    # and a name hashed at run time select the same slot, so
                    # this shape serves a fixed key set and a dynamic one
                    # identically. The mapping's own value keeps its identity
                    # and names the three slots, so the consumers that still
                    # read it can be resolved against them.
                    parts = {
                        "length": ("scalar", "int64", 0),
                        "keys": ("span", "int64", 1),
                        "values": ("span", str(field.get("dtype") or "float64"), 1),
                    }
                    part_ids: dict[str, int] = {}
                    for part_name, (
                        part_storage, part_dtype, part_rank
                    ) in parts.items():
                        part_id = next_physical_id
                        next_physical_id += 1
                        part_ids[part_name] = part_id
                        part_value = SSAValue(
                            part_id,
                            dtype=part_dtype,
                            accounting={
                                "program_abi_record": str(record["identity"]),
                                "program_abi_parameter": str(parameter_name),
                                "program_abi_field": f"{field_name}.{part_name}",
                                "program_abi_storage": part_storage,
                                "program_abi_rank": part_rank,
                                "program_abi_mutable": mutable,
                                "program_abi_field_written": False,
                                "program_abi_keyed_owner": str(field_name),
                                "program_abi_keyed_part": part_name,
                            },
                        )
                        function.args.append(part_value)
                        values[part_id] = part_value
                        fields.append(SSARecordFieldDescriptor(
                            f"{field_name}.{part_name}",
                            SSARecordFieldStorage.SCALAR
                            if part_storage == "scalar"
                            else SSARecordFieldStorage.SPAN,
                            storage_identity=(
                                f"{record['identity']}.{field_name}.{part_name}"
                            ),
                            value_ids=(part_id,),
                            dtype=part_dtype,
                            writable=bool(mutable),
                        ))
                    for value_id in candidate_ids:
                        mapping = values.get(int(value_id))
                        if mapping is None:
                            # The mapping's own occurrence is still a physical
                            # input: consumers that have not yet been resolved
                            # against the three slots continue to name it, and
                            # it is what carries the slot correlation.
                            mapping = SSAValue(int(value_id))
                            function.args.append(mapping)
                            values[int(value_id)] = mapping
                        mapping.accounting = {
                            **dict(mapping.accounting or {}),
                            "program_abi_record": str(record["identity"]),
                            "program_abi_parameter": str(parameter_name),
                            "program_abi_field": str(field_name),
                            "program_abi_storage": "keyed",
                            "program_abi_keyed_length": part_ids["length"],
                            "program_abi_keyed_keys": part_ids["keys"],
                            "program_abi_keyed_values": part_ids["values"],
                        }
                    continue
                field_written = str(field_name) in written_fields
                dtype = field.get("dtype")
                rank = int(field.get("rank", 0))
                physical_ids = []
                for value_id in candidate_ids:
                    value = values.get(value_id)
                    if value is None:
                        value = SSAValue(
                            value_id,
                            dtype=None if dtype is None else str(dtype),
                            accounting={
                                "program_abi_record": str(record["identity"]),
                                "program_abi_parameter": str(parameter_name),
                                "program_abi_field": str(field_name),
                                "program_abi_storage": storage,
                                "program_abi_rank": rank,
                                "program_abi_mutable": mutable,
                                "program_abi_field_written": field_written,
                            },
                        )
                        function.args.append(value)
                        values[value_id] = value
                    else:
                        if value.dtype in {None, "unknown"} and dtype is not None:
                            value.dtype = str(dtype)
                        value.accounting = {
                            **dict(value.accounting or {}),
                            "program_abi_record": str(record["identity"]),
                            "program_abi_parameter": str(parameter_name),
                            "program_abi_field": str(field_name),
                            "program_abi_storage": storage,
                            "program_abi_rank": rank,
                            "program_abi_mutable": mutable,
                            "program_abi_field_written": field_written,
                        }
                    physical_ids.append(value_id)
                descriptor_storage = {
                    "scalar": SSARecordFieldStorage.SCALAR,
                    "span": SSARecordFieldStorage.SPAN,
                    "record": SSARecordFieldStorage.RECORD,
                    "reference": SSARecordFieldStorage.REFERENCE,
                }[storage]
                if descriptor_storage is SSARecordFieldStorage.RECORD:
                    # Nested-record correlation needs its own exact descriptor
                    # id; do not manufacture one from an attribute occurrence.
                    continue
                fields.append(SSARecordFieldDescriptor(
                    str(field_name),
                    descriptor_storage,
                    storage_identity=f"{record['identity']}.{field_name}",
                    value_ids=tuple(physical_ids),
                    dtype=None if dtype is None else str(dtype),
                    writable=bool(mutable and field_written),
                ))
            if fields and record_id not in table.records:
                table.register(SSARecordDescriptor(
                    record_id, str(record["identity"]), tuple(fields),
                ))
        if not table.records:
            all_record_tables.pop(symbol, None)

    def materialize_program_abi_record_literals(symbol: str, graph: Any) -> None:
        """Lower schema-known constructor calls to field correlations.

        A dataclass-shaped boundary does not require executing its Python
        constructor. The authored argument edges already contain every field
        value; the program ABI supplies field order, defaults, storage and
        dtype. The result is an SSARecordDescriptor plus ordinary field SSA,
        with no opaque object id and no invented constructor operator.
        """

        function = all_functions.get(symbol)
        if function is None or not abi_records:
            return
        values = function_values(function)
        next_value_id = 1 + max((
            *values,
            *(int(node_id) for node_id in graph.nodes),
        ), default=0)
        constants = []
        table = all_record_tables.setdefault(symbol, SSARecordTable())
        layouts = []
        for node_id, data in graph.nodes(data=True):
            matched = abi_record_for_call(data)
            if matched is None:
                continue
            _record_name, record = matched
            record_id = int(data.get("value_id", node_id))
            if record_id in table.records:
                continue
            field_contracts = tuple(
                dict(record.get("fields") or {}).items()
            )
            keyword_values = {
                str(role).split(":", 1)[1]: int(parent)
                for parent, role in data.get("parents") or ()
                if str(role).startswith("kw:")
            }
            # Ordered by the role's declared index, not by the order the
            # parent set happens to yield, and matching both ProcessGraph
            # spellings. This list is indexed positionally just below, so
            # taking the set's order would bind field N to whichever
            # argument iteration happened to reach first.
            positional_values = [
                int(parent)
                for parent in ordered_arguments(data.get("parents") or ())
            ]
            fields = []
            physical_layout = []
            for index, (field_name, field) in enumerate(field_contracts):
                value_id = keyword_values.get(str(field_name))
                if value_id is None and index < len(positional_values):
                    value_id = positional_values[index]
                if value_id is None and "default" in field:
                    default = field.get("default")
                    # Optional None needs a tagged optional ABI, which this
                    # scalar/span contract does not yet claim. Leave it absent
                    # rather than encoding a false floating-point zero.
                    if default is not None:
                        value_id = next_value_id
                        next_value_id += 1
                        value = SSAValue(
                            value_id, dtype=field.get("dtype"),
                            accounting={
                                "program_abi_default": str(field_name),
                                "program_abi_record": str(record["identity"]),
                            },
                        )
                        constants.append(Instr(
                            "Const", [], value, attributes={"value": default},
                        ))
                        values[value_id] = value
                if value_id is not None and int(value_id) not in values:
                    source = graph.nodes.get(int(value_id), {})
                    source_attributes = dict(source.get("attributes") or {})
                    if str(
                        source.get("type") or source.get("op") or ""
                    ).casefold() in {"constant", "const"} and (
                        "value" in source_attributes or "constant" in source
                    ):
                        literal = source_attributes.get(
                            "value", source.get("constant")
                        )
                        value = SSAValue(
                            int(value_id), dtype=field.get("dtype"),
                            accounting={
                                "program_abi_constructor_literal": str(
                                    field_name
                                ),
                                "program_abi_record": str(record["identity"]),
                            },
                        )
                        constants.append(Instr(
                            "Const", [], value, attributes={"value": literal},
                        ))
                        values[int(value_id)] = value
                if value_id is None or int(value_id) not in values:
                    continue
                value_id = int(value_id)
                value = values[value_id]
                dtype = field.get("dtype")
                if value.dtype in {None, "unknown"} and dtype is not None:
                    value.dtype = str(dtype)
                if str(field["storage"]) == "keyed":
                    # The constructor argument here is a mapping literal, which
                    # is three physical slots (length, key tokens, values), not
                    # one. Correlating it needs the literal's own key/value
                    # edges, exactly as a nested record needs its own descriptor
                    # id; manufacturing a single slot from this occurrence would
                    # state a layout the record does not have.
                    continue
                storage = {
                    "scalar": SSARecordFieldStorage.SCALAR,
                    "span": SSARecordFieldStorage.SPAN,
                    "reference": SSARecordFieldStorage.REFERENCE,
                    "record": SSARecordFieldStorage.RECORD,
                }[str(field["storage"])]
                if storage is SSARecordFieldStorage.RECORD:
                    continue
                fields.append(SSARecordFieldDescriptor(
                    str(field_name), storage,
                    storage_identity=f"{record['identity']}.{field_name}",
                    value_ids=(value_id,),
                    dtype=None if dtype is None else str(dtype),
                    writable=bool(field.get("mutable", False)),
                ))
                physical_layout.append(value_id)
            if fields:
                table.register(SSARecordDescriptor(
                    record_id, str(record["identity"]), tuple(fields),
                ))
                layouts.append((record_id, tuple(physical_layout)))
        if constants:
            for block in function.blocks.values():
                if block.instrs and block.instrs[-1].op in {
                    "Ret", "ret", "Return", "return"
                }:
                    block.instrs[-1:-1] = constants
                    break
        if layouts:
            function.metadata["record_return_layouts"] = tuple(layouts)
        if not table.records:
            all_record_tables.pop(symbol, None)

    def resolve_keyed_mapping_iterables(symbol: str, graph: Any) -> None:
        """Bind ``d.items()``/``.keys()``/``.values()`` to the mapping's slots.

        A keyed mapping is already a length and two parallel vectors, and the
        loop lowering already walks an iterable as parallel columns: column 0
        is the iterable itself and each further column is an appended source
        carrying ``projected_row_source_id``/``projected_row_column``. Those
        columns *are* the mapping's key and value vectors, so nothing new is
        built here -- only recognised. Left unrecognised they stay anonymous
        storage with no length to iterate and no slot to read, which is what
        made every consumer of a mapping unresolvable at every backend.

        Both ends of the association are exact. The reducer states the method
        as the node's own operation with the mapping as its operand, and the
        column index is carried on the appended source, so neither the mapping
        nor the column is inferred from a name or a position.
        """

        function = all_functions.get(symbol)
        if function is None:
            return
        slots_by_mapping: dict[int, dict[str, int]] = {}
        for value in function.args:
            accounting = value.accounting or {}
            if accounting.get("program_abi_storage") != "keyed":
                continue
            parts = {
                part: accounting.get(f"program_abi_keyed_{part}")
                for part in ("length", "keys", "values")
            }
            if any(slot is None for slot in parts.values()):
                continue        # unresolved in this frame; leave it alone
            slots_by_mapping[int(value.id)] = {
                part: int(slot) for part, slot in parts.items()
            }
        # The mapping identity's slot correlation is frame-local and may have
        # been dropped, while the parts themselves still name their owner.
        # Lookups rebind through the parts, so their presence alone keeps
        # this pass alive.
        has_keyed_parts = any(
            (value.accounting or {}).get("program_abi_keyed_owner")
            is not None
            for value in function.args
        )
        if not slots_by_mapping and not has_keyed_parts:
            return

        # method -> the slot each successive destructured column selects
        columns_by_method = {
            "items": ("keys", "values"),
            "keys": ("keys",),
            "values": ("values",),
        }
        replacements: dict[int, int] = {}
        for node_id, data in graph.nodes(data=True):
            method = str(
                data.get("type") or data.get("op") or ""
            ).casefold()
            columns = columns_by_method.get(method)
            if columns is None:
                continue
            owner = next((
                int(graph.nodes[parent].get("value_id", parent))
                for parent, role in data.get("parents") or ()
                if str(role) in {"operand", "value", "object", "base"}
                and parent in graph
            ), None)
            slots = (
                None if owner is None else slots_by_mapping.get(int(owner))
            )
            if slots is None:
                continue
            iterable_id = int(data.get("value_id", node_id))
            replacements[iterable_id] = slots[columns[0]]
            for value in function.args:
                accounting = value.accounting or {}
                source = accounting.get("projected_row_source_id")
                if source is None or int(source) != iterable_id:
                    continue
                column = int(accounting.get("projected_row_column") or 0)
                if column < len(columns):
                    replacements[int(value.id)] = slots[columns[column]]
            # The iterable's extent is the mapping's own declared length.
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if (
                        instruction.op == "Call"
                        and instruction.attributes.get("tensor_operation")
                        == "extent"
                        and instruction.args
                        and int(instruction.args[0].id) == iterable_id
                        and instruction.res is not None
                    ):
                        replacements[int(instruction.res.id)] = slots["length"]
        # A table lookup on a keyed mapping walks the same declared vectors.
        # The descriptor was built during lowering from anonymous storage --
        # (keys, values, length, capacity) fresh arguments -- because the
        # slots only exist after record materialization.  Bind them here.
        # The mapping identity's own accounting may be frame-local-dropped,
        # so the parts are found by their owner/part markers and the lookup's
        # field by its GetAttr node in the source graph.  A caller-supplied
        # mapping is always exactly full, so capacity IS the length; the
        # status cell stays an ordinary frame-allocated scalar.
        parts_by_owner: dict[str, dict[str, int]] = {}
        for value in function.args:
            accounting = value.accounting or {}
            owner = accounting.get("program_abi_keyed_owner")
            part = accounting.get("program_abi_keyed_part")
            if owner is None or part is None:
                continue
            parts_by_owner.setdefault(str(owner), {})[str(part)] = int(
                value.id
            )
        field_of_sequence: dict[int, str] = {}
        for node_id, data in graph.nodes(data=True):
            attribute = (data.get("attributes") or {}).get("attribute")
            if attribute is None:
                continue
            field_of_sequence[int(data.get("value_id", node_id))] = str(
                attribute
            )
        helper_argument_dtypes = (
            ("int64", None), ("float64", None), ("int", (1,)),
            ("int", None), ("int", (1,)), ("int64", None),
        )
        for block in function.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op != "Call"
                    or instruction.attributes.get("ssa_sequence_operation")
                    != "lookup"
                    or len(instruction.args) < 6
                ):
                    continue
                sequence_id = int(
                    instruction.attributes.get("sequence_id", -1)
                )
                owner = field_of_sequence.get(sequence_id)
                if owner is None:
                    continue
                # The slots this lookup must walk may not exist in this frame
                # yet -- for a mapping produced by a call, the linker imports
                # them later.  Stamp the owner now, while the source graph is
                # at hand; the storage is bound after call-frame linking.
                instruction.attributes["keyed_lookup_owner"] = str(owner)
        if not replacements:
            return

        values = function_values(function)
        resolved = {
            source: values[target]
            for source, target in replacements.items()
            if target in values
        }
        for block in function.blocks.values():
            kept = []
            for instruction in block.instrs:
                if (
                    instruction.res is not None
                    and int(instruction.res.id) in resolved
                ):
                    continue        # its value is the slot now
                instruction.args = [
                    resolved.get(int(argument.id), argument)
                    for argument in instruction.args
                ]
                kept.append(instruction)
            block.instrs = kept
        function.args = [
            value for value in function.args
            if int(value.id) not in resolved
        ]

    for source_symbol, source_graph in source_graphs_by_symbol.items():
        materialize_parameter_record_abi(source_symbol, source_graph)

    for source_symbol, source_graph in source_graphs_by_symbol.items():
        resolve_keyed_mapping_iterables(source_symbol, source_graph)

    for source_symbol, source_graph in source_graphs_by_symbol.items():
        materialize_program_abi_record_literals(source_symbol, source_graph)

    for source_symbol, source_graph in source_graphs_by_symbol.items():
        recover_structural_source_outputs(source_symbol, source_graph)

    for function in all_functions.values():
        source_output_ids = tuple(map(
            int, function.metadata.get("source_output_value_ids", ())
        ))
        if not source_output_ids:
            continue
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op not in {"Ret", "ret", "Return", "return"}:
                    continue
                by_id = {
                    int(argument.id): argument
                    for argument in instruction.args
                }
                # A carried reduction returns through its LoopResult port,
                # but the value standing at that port is the carried PHI,
                # which has its own id.  The builder exports port -> phi in
                # metadata; selection by raw layout id must follow it, or
                # the component resolves to the port's unwritten field cell
                # and every carried maximum publishes zero.
                for argument in instruction.args:
                    for port_id in (
                        (argument.accounting or {}).get("carried_port_ids")
                        or ()
                    ):
                        by_id.setdefault(int(port_id), argument)
                for port_id, port_value in dict(
                    function.metadata.get("carried_port_values") or {}
                ).items():
                    # The port map is the AUTHORITY: a stale component
                    # object carrying the port's id may already sit in the
                    # Ret from earlier expansion, and it names the unwritten
                    # field cell.
                    by_id[int(port_id)] = port_value
                record_layouts = dict(
                    function.metadata.get("record_return_layouts", ())
                )
                expanded_source_ids = tuple(
                    expanded
                    for value_id in source_output_ids
                    for expanded in record_layouts.get(value_id, (value_id,))
                )
                selected = [
                    by_id[value_id]
                    for value_id in expanded_source_ids
                    if value_id in by_id
                ]
                # Some branch histories intentionally publish a predecessor
                # id (the zmap fallback is one); preserve those when the final
                # identity has no materialized spelling. Otherwise discard
                # incidental control/region outputs from the public ABI.
                if selected:
                    instruction.args = selected

    # Parameter-record ABI expansion replaces a conceptual Python receiver
    # with its physical fields. Remove an unconsumed shapeless receiver before
    # call-frame linking; waiting until final cleanup leaves an otherwise
    # complete record-to-record call asking for a nonexistent object handle.
    for function_name, function in all_functions.items():
        record_table = all_record_tables.get(function_name)
        record_ids = set(
            () if record_table is None else map(int, record_table.records)
        )
        source_graph = source_graphs_by_symbol.get(function_name)
        if source_graph is not None:
            identities = source_graph.graph.get("identity_table") or {}
            for parameter_name in (
                source_graph.graph.get("parameter_record_abi") or {}
            ):
                record_ids.update(map(
                    int, identities.get(str(parameter_name), ())
                ))
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
        # the current iteration. Attribute lookup and a receiver-bound method
        # invocation are address/use operations only; neither extends the
        # instance lifetime. Only an actual data escape requires a row pool.
        for successor in graph.successors(int(receiver_id)):
            successor_data = graph.nodes[successor]
            successor_attributes = successor_data.get("attributes") or {}
            roles = {
                str(role) for parent, role in (
                    successor_data.get("parents") or ()
                ) if int(parent) == int(receiver_id)
            }
            successor_operation = str(
                successor_data.get("op")
                or successor_data.get("type")
                or ""
            ).casefold()
            if (
                successor_operation == "getattr"
                and roles <= {"value", "base", "object", "operand"}
            ):
                continue
            if (
                successor_attributes.get("method_ref") is not None
                and roles <= {"value", "base", "object", "operand"}
            ):
                continue
            if roles:
                return True
        return False

    def lexical_loop_ids(
        graph: Any, node_id: int, candidate_ids: Iterable[int]
    ) -> tuple[int, ...]:
        """Discard dependency-closure loop ownership outside lexical spans."""

        node_expression = graph.nodes.get(int(node_id), {}).get("expr_obj")
        node_line = getattr(node_expression, "lineno", None)
        if node_line is None:
            return tuple(map(int, candidate_ids))
        retained = []
        for loop_id in candidate_ids:
            loop_expression = graph.nodes.get(int(loop_id), {}).get("expr_obj")
            loop_line = getattr(loop_expression, "lineno", None)
            loop_end = getattr(loop_expression, "end_lineno", None)
            if loop_line is None or loop_end is None:
                retained.append(int(loop_id))
                continue
            if int(loop_line) <= int(node_line) <= int(loop_end):
                retained.append(int(loop_id))
        return tuple(retained)

    if class_table is not None:
        class_definitions = {
            str(record.identity): record for record in class_table.classes
        }
        class_alias_candidates: dict[str, list[Any]] = {}
        for definition in class_definitions.values():
            class_alias_candidates.setdefault(
                str(definition.identity).rsplit(".", 1)[-1], []
            ).append(definition)
        class_aliases = {
            alias: candidates[0]
            for alias, candidates in class_alias_candidates.items()
            if len(candidates) == 1
        }

        def resolve_class_definition(identity: Any) -> Any:
            """Resolve frontend short class refs without guessing collisions."""

            text = str(identity)
            return class_definitions.get(text) or class_aliases.get(text)

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

        constructor_symbol_by_class = {}
        for definition in class_definitions.values():
            method = next((
                method for name in ("__new__", "__init__")
                for method in definition.methods
                if method.name == name and method.function_name is not None
            ), None)
            if method is None:
                continue
            constructor_symbol_by_class[str(definition.identity)] = str(
                method.function_name
            )
            short_identity = str(definition.identity).rsplit(".", 1)[-1]
            if class_aliases.get(short_identity) is definition:
                constructor_symbol_by_class[short_identity] = str(
                    method.function_name
                )

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
                # StaticReference nodes deliberately carry the same class_ref
                # so member navigation can resolve them, but only the Call is
                # an instance allocation site with caller-owned storage.
                node_operation = str(
                    node_data.get("op") or node_data.get("type") or ""
                ).casefold()
                if class_identity is None or node_operation != "call":
                    continue
                class_definition = resolve_class_definition(class_identity)
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
                class_storage_identities = {
                    str(class_identity),
                    str(class_definition.identity),
                    str(class_definition.identity).rsplit(".", 1)[-1],
                }
                templates = tuple(
                    record for record in constructor_table.records.values()
                    if record.identity in class_storage_identities
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
                planned_constructor_call = max(
                    (
                        item
                        for pending_caller, item, _graph, _module, _shell
                        in pending_call_records
                        if str(pending_caller) == str(caller_symbol)
                        and int(item.callsite_id) == int(node_id)
                    ),
                    key=lambda item: len(item.argument_bindings),
                    default=None,
                )
                if planned_constructor_call is not None:
                    for caller_id, callee_id in (
                        planned_constructor_call.argument_bindings
                    ):
                        remap[int(callee_id)] = int(caller_id)
                        constructor_parameter_ids.add(int(callee_id))
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
                        index = positional_argument_index(role)
                        keyword = keyword_argument_name(role)
                        if index is not None:
                            name = (
                                positional_names[index]
                                if index < len(positional_names) else None
                            )
                        elif keyword is not None:
                            name = keyword
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
                # A constructor's repository-SSA signature is its complete
                # physical frame, not merely its authored parameters and
                # record fields. Region scratch and descriptor slots are also
                # caller-owned storage, so allocate every remaining frame id
                # instead of leaving the object intrinsically host-bound.
                for value in constructor.args:
                    old_id = int(value.id)
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
                enclosing_loop_ids = lexical_loop_ids(
                    caller_graph, int(node_id), enclosing_loop_ids
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
    result_storage_bindings_by_call: dict[
        tuple[str, int], dict[int, int]
    ] = {}
    call_anchor_value_ids: dict[tuple[str, int], int | None] = {}
    seen_calls: set[tuple[str, int, int | None]] = set()
    for caller_symbol, planned_call, caller_graph, caller_module, caller_shell in (
        pending_call_records
    ):
        call_data = caller_graph.nodes.get(int(planned_call.callsite_id), {})
        call_operation = str(
            call_data.get("op") or call_data.get("type") or ""
        ).casefold()
        if isinstance(call_data.get("expr_obj"), ast.Attribute):
            # A bound-method selector may carry the same method_ref as the
            # authored Call that consumes it.  It is a navigation value, not
            # a second invocation with an empty argument frame.
            continue
        if call_operation == "staticreference":
            # A class StaticReference is a navigable definition handle, not a
            # runtime constructor execution.  The real Call node carries the
            # same constructor_ref and is materialized above.
            continue
        if (str(caller_symbol), int(planned_call.callsite_id)) in (
            constructor_anchors
        ):
            # The record-ABI constructor occurrence is authoritative.  A
            # PlanCall for a specialized view of the same source Call would
            # otherwise become a second execution with a partial frame.
            continue
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
        child_shell = getattr(caller_shell, "callsite_function_shells", {}).get(
            int(planned_call.callsite_id)
        )
        callee_symbol = (
            shell_symbols.get(id(child_shell))
            if child_shell is not None else None
        ) or (
            None if reference is None
            else function_symbols.get(int(reference))
        )
        callee_function = (
            None if callee_symbol is None
            else all_functions.get(callee_symbol)
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
        if child_graph is not None and callee_function is not None:
            for value in callee_function.args:
                value_id = int(value.id)
                node = child_graph.nodes.get(value_id)
                if node is None or str(node.get("type")) not in {
                    "Constant", "Const", "const",
                }:
                    continue
                node_attributes = node.get("attributes") or {}
                if "value" in node_attributes:
                    default_literals[value_id] = copy.deepcopy(
                        node_attributes["value"]
                    )
                elif "constant" in node:
                    default_literals[value_id] = copy.deepcopy(
                        node["constant"]
                    )
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
                            node = child_graph.nodes.get(int(value_id), {})
                            # Name history also contains every later SSA
                            # assignment to the parameter's spelling.  A
                            # Python default belongs only to the authored
                            # parameter Input; marking a local reassignment as
                            # the default turns real dataflow into ``None`` at
                            # every caller.
                            if (
                                str(node.get("type") or node.get("op") or "")
                                .casefold() != "input"
                            ):
                                continue
                            default_literals[int(value_id)] = parameter.default
        frame_bindings = []
        unresolved_frame = []
        receiver_record = None
        callee_record = None
        result_storage_bindings: dict[int, int] = {}
        result_storage_bindings_by_call[(
            str(caller_symbol), int(planned_call.callsite_id)
        )] = result_storage_bindings
        result_record_bindings: dict[int, int] = {}
        if callee_symbol is not None and callee_function is not None:
            callee_result_records = all_record_tables.get(callee_symbol)
            callee_result_sequences = all_sequence_tables.get(callee_symbol)
            caller_result_records = all_record_tables.setdefault(
                caller_symbol, SSARecordTable()
            )
            caller_result_sequences = all_sequence_tables.setdefault(
                caller_symbol, SSASequenceTable()
            )
            caller_values = function_values(all_functions[caller_symbol])
            caller_graph_ids = {
                int(data.get("value_id", node_id))
                for node_id, data in caller_graph.nodes(data=True)
            }
            next_result_storage_id = 1 + max(
                (*caller_values, *caller_graph_ids), default=0
            )

            def allocate_result_storage(old_id: int) -> int:
                nonlocal next_result_storage_id
                old_id = int(old_id)
                if old_id in result_storage_bindings:
                    return result_storage_bindings[old_id]
                new_id = next_result_storage_id
                next_result_storage_id += 1
                source = function_values(callee_function).get(
                    old_id, SSAValue(old_id)
                )
                value = clone_value(source, new_id, accounting={
                    "returned_record_storage": str(callee_symbol),
                    "callsite_id": int(planned_call.callsite_id),
                })
                all_functions[caller_symbol].args.append(value)
                caller_values[new_id] = value
                result_storage_bindings[old_id] = new_id
                return new_id

            for callee_result_id, caller_result_id in (
                planned_call.result_bindings
            ):
                if callee_result_records is None:
                    continue
                root = callee_result_records.records.get(
                    int(callee_result_id)
                )
                if root is None or int(caller_result_id) in (
                    caller_result_records.records
                ):
                    continue
                source_records = {
                    int(record.record_id): record
                    for record in callee_result_records.records.values()
                }
                pending_records = [root]
                record_order = []
                seen_record_ids = set()
                while pending_records:
                    record = pending_records.pop()
                    record_id = int(record.record_id)
                    if record_id in seen_record_ids:
                        continue
                    seen_record_ids.add(record_id)
                    record_order.append(record)
                    for field in record.fields:
                        nested = (
                            None if field.record_id is None
                            else source_records.get(int(field.record_id))
                        )
                        if nested is not None:
                            pending_records.append(nested)
                result_record_bindings[int(root.record_id)] = int(
                    caller_result_id
                )
                for record in record_order:
                    if int(record.record_id) == int(root.record_id):
                        continue
                    result_record_bindings[int(record.record_id)] = (
                        next_result_storage_id
                    )
                    next_result_storage_id += 1
                for record in reversed(record_order):
                    mapped_fields = []
                    for field in record.fields:
                        mapped_sequence_id = None
                        if field.sequence_id is not None:
                            sequence = (
                                None if callee_result_sequences is None
                                else callee_result_sequences.by_id(
                                    int(field.sequence_id)
                                )
                            )
                            if sequence is not None:
                                sequence_ids = (
                                    *sequence.column_value_ids,
                                    sequence.length_address_id,
                                    sequence.capacity_value_id,
                                    *((sequence.status_address_id,)
                                      if sequence.status_address_id is not None
                                      else ()),
                                    *((sequence.live_flags_value_id,)
                                      if sequence.live_flags_value_id is not None
                                      else ()),
                                )
                                pool = sequence.child_table_pool
                                if pool is not None:
                                    sequence_ids = (
                                        *sequence_ids,
                                        *pool.column_value_ids,
                                        pool.length_value_id,
                                        pool.capacity_value_id,
                                        pool.row_stride_value_id,
                                        *((pool.status_value_id,)
                                          if pool.status_value_id is not None
                                          else ()),
                                        *((pool.live_flags_value_id,)
                                          if pool.live_flags_value_id is not None
                                          else ()),
                                    )
                                for value_id in sequence_ids:
                                    allocate_result_storage(int(value_id))
                                mapped_sequence_id = allocate_result_storage(
                                    int(sequence.sequence_id)
                                )
                                caller_result_sequences.register(
                                    SSASequenceDescriptor(
                                        sequence_id=mapped_sequence_id,
                                        column_value_ids=tuple(
                                            result_storage_bindings[int(value_id)]
                                            for value_id in sequence.column_value_ids
                                        ),
                                        length_address_id=result_storage_bindings[
                                            int(sequence.length_address_id)
                                        ],
                                        capacity_value_id=result_storage_bindings[
                                            int(sequence.capacity_value_id)
                                        ],
                                        status_address_id=(
                                            None
                                            if sequence.status_address_id is None
                                            else result_storage_bindings[int(
                                                sequence.status_address_id
                                            )]
                                        ),
                                        column_dtypes=tuple(
                                            sequence.column_dtypes
                                        ),
                                        key_columns=tuple(sequence.key_columns),
                                        live_flags_value_id=(
                                            None
                                            if sequence.live_flags_value_id is None
                                            else result_storage_bindings[int(
                                                sequence.live_flags_value_id
                                            )]
                                        ),
                                        capacity_policy=sequence.capacity_policy,
                                        writable=bool(sequence.writable),
                                        child_table_pool=map_child_pool(
                                            sequence.child_table_pool,
                                            result_storage_bindings,
                                        ),
                                    )
                                )
                        for value_id in field.value_ids:
                            allocate_result_storage(int(value_id))
                        mapped_fields.append(SSARecordFieldDescriptor(
                            name=field.name,
                            storage=field.storage,
                            storage_identity=field.storage_identity,
                            value_ids=tuple(
                                result_storage_bindings[int(value_id)]
                                for value_id in field.value_ids
                            ),
                            sequence_id=mapped_sequence_id,
                            record_id=(
                                None if field.record_id is None
                                else result_record_bindings[int(field.record_id)]
                            ),
                            offset=field.offset,
                            dtype=field.dtype,
                            writable=field.writable,
                        ))
                    mapped_record_id = result_record_bindings[
                        int(record.record_id)
                    ]
                    caller_result_records.register(SSARecordDescriptor(
                        mapped_record_id,
                        str(record.identity),
                        tuple(mapped_fields),
                    ))
        bound_record_pairs = []
        if callee_symbol is not None:
            callee_records = all_record_tables.get(callee_symbol)
            candidates = (
                () if callee_records is None
                else tuple(callee_records.records.values())
            )
            caller_records = all_record_tables.get(caller_symbol)
            if caller_records is not None:
                for candidate in candidates:
                    bound_receiver = exact_bindings.get(
                        int(candidate.record_id)
                    )
                    if bound_receiver is None:
                        continue
                    bound_record = caller_records.records.get(
                        int(bound_receiver)
                    )
                    if bound_record is not None:
                        bound_record_pairs.append((bound_record, candidate))
            if bound_record_pairs:
                receiver_record, callee_record = bound_record_pairs[0]
        storage_bindings = dict(result_storage_bindings)
        for bound_record, candidate in bound_record_pairs:
            caller_fields = {
                field.storage_identity: field
                for field in bound_record.fields
            }
            for field in candidate.fields:
                caller_field = caller_fields.get(field.storage_identity)
                if (
                    caller_field is None
                    or not caller_field.value_ids
                    or not field.value_ids
                ):
                    continue
                # Every descriptor member has this exact physical storage
                # identity. Multiple GetAttr occurrences are views, not
                # independent ABI arenas, so bind them all to the caller's
                # canonical field slot.
                caller_storage = int(caller_field.value_ids[0])
                storage_bindings.update(
                    (int(value_id), caller_storage)
                    for value_id in field.value_ids
                )
        if receiver_record is not None and callee_record is not None:
            caller_fields = {
                field.storage_identity: field
                for field in receiver_record.fields
            }
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
        # Snapshot the physical frame. For a recursive call caller and callee
        # are the same Function object, and propagating a missing storage slot
        # appends to caller.args. Iterating the live list would therefore grow
        # the sequence forever.
        for value in (
            () if callee_function is None else tuple(callee_function.args)
        ):
            value_id = int(value.id)
            if value_id in storage_bindings:
                frame_bindings.append((
                    value_id, "caller_storage", storage_bindings[value_id]
                ))
            elif value_id in exact_bindings:
                caller_value_id = int(exact_bindings[value_id])
                caller_node = caller_graph.nodes.get(caller_value_id)
                caller_attributes = (
                    {} if caller_node is None
                    else caller_node.get("attributes") or {}
                )
                if (
                    caller_node is not None
                    and str(caller_node.get("type")) in {
                        "Constant", "Const", "const",
                    }
                    and (
                        "value" in caller_attributes
                        or "constant" in caller_node
                    )
                ):
                    frame_bindings.append((
                        value_id,
                        "caller_literal",
                        copy.deepcopy(
                            caller_attributes.get(
                                "value", caller_node.get("constant")
                            )
                        ),
                    ))
                elif (
                    caller_node is not None
                    and str(
                        caller_node.get("type")
                        or caller_node.get("op")
                        or ""
                    ).casefold() == "staticreference"
                    and caller_attributes.get("function_ref") is not None
                ):
                    from ..transmogrifier.function_table import (
                        FunctionReference,
                    )

                    frame_bindings.append((
                        value_id,
                        "caller_literal",
                        FunctionReference(int(
                            caller_attributes["function_ref"]
                        )),
                    ))
                else:
                    frame_bindings.append((
                        value_id, "caller_value", caller_value_id
                    ))
            elif value_id in identity_aliases:
                frame_bindings.append((
                    value_id, "caller_alias", identity_aliases[value_id]
                ))
            elif value_id in default_literals:
                frame_bindings.append((
                    value_id, "default_literal", default_literals[value_id]
                ))
            elif "record_instance" in dict(value.accounting or {}):
                # Storage introduced while constructing an object remains
                # part of that function's physical ABI even when it is not a
                # published field of the returned record. Propagate those
                # descriptor/scratch slots through an ordinary wrapper call
                # instead of reintroducing a host-object boundary.
                frame_bindings.append((
                    value_id,
                    "caller_storage",
                    allocate_result_storage(value_id),
                ))
            else:
                # Every remaining callee argument is still a concrete member
                # of the repository-SSA physical frame.  It is neither an
                # authored argument, a default, nor correlated record storage,
                # so give the caller a distinct storage slot and propagate it
                # outward.  Leaving these as "unresolved" made list/tensor
                # descriptors, loop scratch, hook tables, and tape mechanics
                # look like opaque Python dependencies even though their full
                # contents were already present in the callee signature.
                frame_bindings.append((
                    value_id,
                    "caller_storage",
                    allocate_result_storage(value_id),
                ))
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
        normalized_loop_ids = lexical_loop_ids(
            caller_graph,
            int(planned_call.callsite_id),
            tuple(planned_call.enclosing_loop_ids),
        )
        recursive_region = (
            str(caller_symbol) == str(callee_symbol)
            and bool(normalized_loop_ids)
            and bool(all_functions[caller_symbol].metadata.get(
                "recursion_table"
            ))
        )
        resolution = (
            "decomposed"
            if decompositions or recursive_region
            else "unresolved"
        )
        call_records.setdefault(caller_symbol, []).append(SSACallRecord(
            caller=caller_symbol,
            callsite_id=int(planned_call.callsite_id),
            callee_reference=(None if reference is None else int(reference)),
            callee_name=str(planned_call.callee.name),
            callee_symbol=callee_symbol,
            argument_bindings=tuple(planned_call.argument_bindings),
            result_bindings=tuple(planned_call.result_bindings),
            enclosing_loop_ids=normalized_loop_ids,
            callee_storage_value_ids=(
                () if callee_function is None
                else tuple(int(value.id) for value in callee_function.args)
            ),
            frame_bindings=tuple(frame_bindings),
            unresolved_frame_value_ids=tuple(unresolved_frame),
            resolution=resolution,
            decomposition=(
                "recursion_region"
                if recursive_region
                else None if not decompositions
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

    # Specialize the two authored recursive fallbacks to the repository
    # operations they define. Native tensor targets implement zero-fill and
    # element count directly; retaining their Python list recursion as another
    # runtime call would duplicate that mechanism and, historically, left an
    # empty control shell with an unresolved self-call.
    from ..common.tensors.backward_registry import eps as backward_epsilon

    for caller_symbol, records in tuple(call_records.items()):
        caller = all_functions[caller_symbol]
        caller_graph = source_graphs_by_symbol.get(caller_symbol)
        caller_values = function_values(caller)
        rebuilt = []
        for record in records:
            if (
                record.resolution == "unresolved"
                and str(record.caller) == str(record.callee_symbol)
                and record.callee_name in {"zmap", "_count"}
            ):
                rebuilt.append(replace(
                    record,
                    resolution="decomposed",
                    decomposition=(
                        "fill_zero" if record.callee_name == "zmap"
                        else "tensor_numel"
                    ),
                ))
                continue
            if (
                record.resolution == "unresolved"
                and record.callee_name == "eps"
                and len(record.result_bindings) == 1
            ):
                _callee_result_id, caller_result_id = record.result_bindings[0]
                result = caller_values.get(
                    int(caller_result_id),
                    SSAValue(int(caller_result_id), dtype="float64"),
                )
                # ``backward_registry.eps`` is an authored scalar helper.
                # The source-call placeholder may have inherited a caller's
                # tensor descriptor before decomposition; none of that shape
                # belongs to the constant which replaces the call.
                result.dtype = "float64"
                result.shape = ()
                result.accounting = {
                    **dict(result.accounting or {}),
                    "physical_dtype": "float64",
                }
                intrinsic = Instr(
                    "Const", [], result,
                    attributes={
                        "value": float(backward_epsilon()),
                        "structural_operation": "backward_epsilon",
                    },
                )
                inserted = False
                for block in caller.blocks.values():
                    for index, instruction in enumerate(block.instrs):
                        if any(
                            int(argument.id) == int(caller_result_id)
                            for argument in instruction.args
                        ):
                            block.instrs[index:index] = [intrinsic]
                            inserted = True
                            break
                    if inserted:
                        break
                if not inserted:
                    for block in caller.blocks.values():
                        if block.instrs and block.instrs[-1].op in {
                            "Ret", "ret", "Return", "return"
                        }:
                            block.instrs[-1:-1] = [intrinsic]
                            inserted = True
                            break
                if inserted:
                    caller.args = [
                        value for value in caller.args
                        if int(value.id) != int(caller_result_id)
                    ]
                    caller_values[int(caller_result_id)] = result
                    rebuilt.append(replace(
                        record,
                        resolution="decomposed",
                        decomposition="backward_epsilon",
                    ))
                    continue
            if (
                record.resolution == "unresolved"
                and record.callee_name == "_count"
                and len(record.result_bindings) == 1
            ):
                source_id = next((
                    int(caller_id)
                    for caller_id, callee_id in record.argument_bindings
                    if int(callee_id) == 0
                ), None)
                _callee_result_id, caller_result_id = (
                    record.result_bindings[0]
                )
                source = caller_values.get(source_id)
                shape = tuple(getattr(source, "shape", ()) or ())
                if not shape and caller_graph is not None and source_id is not None:
                    def inherited_shape(
                        value_id: int, seen: frozenset[int] = frozenset()
                    ) -> tuple[Any, ...]:
                        value_id = int(value_id)
                        if value_id in seen:
                            return ()
                        resident = caller_values.get(value_id)
                        resident_shape = tuple(
                            getattr(resident, "shape", ()) or ()
                        )
                        if resident_shape:
                            return resident_shape
                        source_data = caller_graph.nodes.get(value_id, {})
                        tensor = source_data.get("tensor") or {}
                        tensor_shape = tuple(tensor.get("shape") or ())
                        if tensor_shape:
                            return tensor_shape
                        for parent, role in source_data.get("parents") or ():
                            if str(role) in {
                                "operand", "value", "base", "arg:0", "lhs"
                            }:
                                parent_shape = inherited_shape(
                                    int(parent), seen | {value_id}
                                )
                                if parent_shape:
                                    return parent_shape
                        return ()

                    shape = inherited_shape(int(source_id))
                if source is None and source_id is not None:
                    source = SSAValue(
                        int(source_id),
                        shape=tuple(shape),
                        accounting={
                            "externalized_intrinsic_source": "tensor_numel"
                        },
                    )
                    caller.args.append(source)
                    caller_values[int(source_id)] = source
                if source is not None:
                    result = SSAValue(int(caller_result_id), dtype="int64")
                    intrinsic = (
                        Instr(
                            "Const", [], result,
                            attributes={
                                "value": int(np.prod(shape, dtype=np.int64)),
                                "structural_operation": "tensor_numel",
                            },
                        )
                        if shape and all(int(extent) >= 0 for extent in shape)
                        else Instr(
                            "extent", [source], result,
                            attributes={
                                "tensor_operation": "extent",
                                "extent_kind": "numel",
                                "dim": -1,
                                "structural_operation": "tensor_numel",
                            },
                        )
                    )
                    inserted = False
                    for block in caller.blocks.values():
                        for index, instruction in enumerate(block.instrs):
                            if any(
                                int(argument.id) == int(caller_result_id)
                                for argument in instruction.args
                            ):
                                block.instrs[index:index] = [intrinsic]
                                inserted = True
                                break
                        if inserted:
                            break
                    if not inserted:
                        for block in caller.blocks.values():
                            if block.instrs and block.instrs[-1].op in {
                                "Ret", "ret", "Return", "return"
                            }:
                                block.instrs[-1:-1] = [intrinsic]
                                inserted = True
                                break
                    if inserted:
                        caller.args = [
                            value for value in caller.args
                            if int(value.id) != int(caller_result_id)
                        ]
                        caller_values[int(caller_result_id)] = result
                        rebuilt.append(replace(
                            record,
                            resolution="decomposed",
                            decomposition="tensor_numel",
                        ))
                        continue
            rebuilt.append(record)
        call_records[caller_symbol] = rebuilt

    # A parameter specialized to a compile-time literal is not a runtime
    # scalar once the specialized body no longer consumes it.  This includes
    # higher-order FunctionTable references and ordinary immutable/default
    # literals such as ``None``. Once every incoming call proves the same
    # category, erase the dead argument from the physical frame and from those
    # bindings. No sentinel value is introduced; genuinely dynamic optionals
    # remain arguments and still require a tagged ABI.
    from ..transmogrifier.function_table import FunctionReference

    incoming_by_callee: dict[str, list[tuple[str, int]]] = {}
    for caller_symbol, records in call_records.items():
        for index, record in enumerate(records):
            incoming_by_callee.setdefault(
                str(record.callee_symbol), []
            ).append((str(caller_symbol), index))
    for callee_symbol, incoming in incoming_by_callee.items():
        callee = all_functions.get(callee_symbol)
        if callee is None or not incoming:
            continue
        consumed = {
            int(argument.id)
            for block in callee.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        }
        removable = set()
        for argument in callee.args:
            argument_id = int(argument.id)
            if argument_id in consumed:
                continue
            bindings = []
            complete = True
            for caller_symbol, record_index in incoming:
                record = call_records[caller_symbol][record_index]
                binding = next((
                    (kind, source)
                    for callee_id, kind, source in record.frame_bindings
                    if int(callee_id) == argument_id
                ), None)
                if binding is None:
                    complete = False
                    break
                bindings.append(binding)
            if complete and bindings:
                function_references = all(
                    kind == "caller_literal"
                    and isinstance(source, FunctionReference)
                    for kind, source in bindings
                )
                literal_values = all(
                    kind in {"caller_literal", "default_literal"}
                    and not isinstance(source, FunctionReference)
                    for kind, source in bindings
                )
                if literal_values:
                    first = bindings[0][1]
                    try:
                        literal_values = all(
                            source == first for _kind, source in bindings[1:]
                        )
                    except (TypeError, ValueError):
                        literal_values = False
                if function_references or literal_values:
                    removable.add(argument_id)
        if not removable:
            continue
        callee.args = [
            argument for argument in callee.args
            if int(argument.id) not in removable
        ]
        for caller_symbol, record_index in incoming:
            record = call_records[caller_symbol][record_index]
            call_records[caller_symbol][record_index] = replace(
                record,
                frame_bindings=tuple(
                    binding for binding in record.frame_bindings
                    if int(binding[0]) not in removable
                ),
                callee_storage_value_ids=tuple(
                    value_id for value_id in record.callee_storage_value_ids
                    if int(value_id) not in removable
                ),
            )

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
    # Structural-result discovery may conservatively allocate caller-owned
    # storage before numerical aggregate lowering proves that the aggregate is
    # returned through SSA instead.  Such storage is not part of the authored
    # ABI.  Remove only allocations with this exact provenance when no
    # instruction consumes them; live record arenas remain ordinary arguments.
    for function in all_functions.values():
        consumed_ids = {
            int(argument.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        }
        function.args = [
            argument for argument in function.args
            if not (
                (argument.accounting or {}).get("returned_record_storage")
                and int(argument.id) not in consumed_ids
            )
        ]
    changed = True
    while changed:
        changed = False
        callee_callers = {
            caller: tuple(records) for caller, records in call_records.items()
        }
        for caller_symbol, records in tuple(call_records.items()):
            caller = all_functions[caller_symbol]
            caller_graph = source_graphs_by_symbol.get(caller_symbol)
            values = {int(value.id): value for value in caller.args}
            # Ids genuinely produced by an existing instruction, as opposed to
            # a shapeless placeholder some other record's processing may have
            # `setdefault`-ed into `values` this same round. A record's own
            # authored callsite id (below, `returns_aggregate`) is not
            # guaranteed disjoint from some unrelated value's id drawn from a
            # different numbering source (e.g. a required-source-value
            # resolved earlier via aggregate unpacking) -- reusing an
            # already-produced id for a new, unrelated result would give two
            # different instructions the same SSA identity.
            produced_ids = {
                int(instruction.res.id)
                for block in caller.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            }
            values.update({
                int(instruction.res.id): instruction.res
                for block in caller.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            })
            for pending_record in records:
                if len(pending_record.result_bindings) == 1:
                    _callee_id, caller_id = pending_record.result_bindings[0]
                    caller_tensor = (
                        caller_graph.nodes.get(int(caller_id), {}).get("tensor")
                        or {}
                    )
                    values.setdefault(int(caller_id), SSAValue(
                        int(caller_id),
                        dtype=caller_tensor.get("dtype"),
                        shape=tuple(caller_tensor.get("shape") or ()),
                    ))
                elif len(pending_record.result_bindings) > 1:
                    values.setdefault(
                        int(pending_record.callsite_id),
                        SSAValue(
                            int(pending_record.callsite_id),
                            accounting={
                                "ssa_aggregate_outputs": tuple(
                                    int(caller_id)
                                    for _callee_id, caller_id
                                    in pending_record.result_bindings
                                )
                            },
                        ),
                    )
            next_value_id = 1 + max(values, default=0)
            rebuilt_records = []

            def resolve_call_feed(
                source_id: int, prelude: list[Instr]
            ) -> SSAValue | None:
                """Resolve structural call feeds at their invocation site."""

                nonlocal next_value_id
                source_id = int(source_id)
                if source_id in values:
                    return values[source_id]
                if caller_graph is None:
                    return None
                data = caller_graph.nodes.get(source_id, {})
                operation = str(
                    data.get("op") or data.get("type") or ""
                ).casefold()
                if operation == "getattr":
                    attribute = str((data.get("attributes") or {}).get(
                        "attribute", ""
                    ))
                    receiver_id = next((
                        int(parent)
                        for parent, role in data.get("parents") or ()
                        if str(role) in {
                            "value", "object", "base", "operand"
                        }
                    ), None)
                    table = all_record_tables.get(caller_symbol)
                    record = (
                        None if table is None or receiver_id is None
                        else table.records.get(receiver_id)
                    )
                    field = (
                        None if record is None else next((
                            field for field in record.fields
                            if str(field.name) == attribute
                        ), None)
                    )
                    if field is not None and len(field.value_ids) == 1:
                        return values.get(int(field.value_ids[0]))
                    return None
                if operation == "boolop":
                    operands = []
                    for parent, role in data.get("parents") or ():
                        if not str(role).startswith("value:"):
                            continue
                        operand = resolve_call_feed(int(parent), prelude)
                        if operand is None:
                            return None
                        operands.append(operand)
                    expression = data.get("expr_obj")
                    opcode = (
                        "LAnd"
                        if isinstance(getattr(expression, "op", None), ast.And)
                        else "LOr"
                        if isinstance(getattr(expression, "op", None), ast.Or)
                        else None
                    )
                    if opcode is None or len(operands) < 2:
                        return None
                    current = operands[0]
                    for index, operand in enumerate(operands[1:], start=1):
                        is_last = index == len(operands) - 1
                        result_id = source_id if is_last else next_value_id
                        if not is_last:
                            next_value_id += 1
                        result = SSAValue(
                            result_id,
                            dtype="bool",
                        )
                        prelude.append(Instr(
                            opcode, [current, operand], result,
                            attributes={
                                "structural_operation": "boolop",
                                "call_feed": True,
                            },
                        ))
                        current = result
                    values[source_id] = current
                    return current
                return None

            def pending_result_id(pending):
                if len(pending.result_bindings) == 1:
                    return int(pending.result_bindings[0][1])
                if len(pending.result_bindings) > 1:
                    return int(pending.callsite_id)
                return None

            def downstream_anchor(value_id, seen=frozenset()):
                value_id = int(value_id)
                if value_id in seen:
                    return None
                for candidate in records:
                    if not any(
                        str(kind) in {
                            "caller_value", "caller_alias", "caller_storage"
                        }
                        and int(source) == value_id
                        for _callee_id, kind, source in candidate.frame_bindings
                    ):
                        continue
                    candidate_result = pending_result_id(candidate)
                    if candidate_result is None:
                        continue
                    if any(
                        int(argument.id) == candidate_result
                        for block in caller.blocks.values()
                        for instruction in block.instrs
                        for argument in instruction.args
                    ):
                        return candidate_result
                    nested = downstream_anchor(
                        candidate_result, seen | {value_id}
                    )
                    if nested is not None:
                        return nested
                return None

            def source_loop_blocks(loop_id: int) -> frozenset[str]:
                """Return the CFG compartment owned by one authored loop.

                ``SSACallRecord.enclosing_loop_ids`` and the loop-header Phi
                both carry the reducer's source ProcessGraph identity.  Walk
                only the true/body side and stop at the header and false/exit
                edge; this preserves nested-loop blocks without inferring
                lexical ownership from block names or dictionary order.
                """

                loop_id = int(loop_id)
                header = next((
                    block
                    for block in caller.blocks.values()
                    if any(
                        instruction.op == "Phi"
                        and (
                            instruction.attributes.get(
                                "source_loop_node_id"
                            ) == loop_id
                            or instruction.attributes.get("source_name")
                            == f"iteration_{loop_id}"
                        )
                        for instruction in block.instrs
                    )
                ), None)
                branch = (
                    None if header is None else next((
                        instruction
                        for instruction in header.instrs
                        if instruction.op == "CondBr"
                    ), None)
                )
                if branch is None:
                    return frozenset()
                body_name = str(branch.attributes.get("true_target"))
                exit_name = str(branch.attributes.get("false_target"))
                owned = set()
                pending = [body_name]
                while pending:
                    block_name = pending.pop()
                    if (
                        block_name in owned
                        or block_name in {header.name, exit_name}
                        or block_name not in caller.blocks
                    ):
                        continue
                    owned.add(block_name)
                    pending.extend(caller.blocks[block_name].successors)
                return frozenset(owned)

            def insert_at_loop_anchor(
                record: SSACallRecord,
                sequence: list[Instr],
            ) -> bool:
                if not record.enclosing_loop_ids:
                    return False
                owned = source_loop_blocks(record.enclosing_loop_ids[-1])
                if not owned:
                    return False
                anchor_value_id = call_anchor_value_ids.get((
                    str(caller_symbol), int(record.callsite_id)
                ))
                if anchor_value_id is None:
                    return False
                produced_ids = {
                    int(instruction.res.id)
                    for instruction in sequence
                    if instruction.res is not None
                }
                for instruction in sequence:
                    produced_ids.update(
                        int(output_id) for output_id in (
                            (instruction.attributes or {}).get(
                                "output_ids"
                            ) or ()
                        )
                    )
                for block_name, block in caller.blocks.items():
                    if block_name not in owned:
                        continue
                    for index, instruction in enumerate(block.instrs):
                        if (
                            instruction.res is not None
                            and int(instruction.res.id)
                            == int(anchor_value_id)
                        ):
                            # The anchor names WHERE the authored program
                            # placed this call; region reordering may have
                            # legally moved a consumer of the call's outputs
                            # ahead of that anchor.  A call may never follow
                            # a consumer of its own results, so clamp to the
                            # earliest such consumer.
                            index = min(index, next((
                                consumer_index
                                for consumer_index, candidate in enumerate(
                                    block.instrs[:index]
                                )
                                if any(
                                    int(argument.id) in produced_ids
                                    for argument in candidate.args
                                )
                            ), index))
                            block.instrs[index:index] = sequence
                            return True
                return False

            for record in records:
                result_storage_bindings = (
                    result_storage_bindings_by_call.setdefault(
                        (str(record.caller), int(record.callsite_id)), {}
                    )
                )
                callee = (
                    None if record.callee_symbol is None
                    else all_functions.get(record.callee_symbol)
                )
                if callee is not None:
                    current_frame_ids = {
                        int(argument.id) for argument in callee.args
                    }
                    refreshed_frame_bindings = [
                        binding for binding in record.frame_bindings
                        if int(binding[0]) in current_frame_ids
                    ]
                    bound_frame_ids = {
                        int(binding[0]) for binding in refreshed_frame_bindings
                    }
                    # Linking a callee can grow its physical frame: storage
                    # required by a newly materialized nested call becomes an
                    # ordinary callee argument.  Call records are discovered
                    # before that fixed point, so extend the caller frame here
                    # instead of permanently stranding an otherwise complete
                    # call behind a stale argument snapshot.
                    if (
                        str(record.caller) != str(record.callee_symbol)
                    ):
                        for argument in callee.args:
                            argument_id = int(argument.id)
                            if argument_id in bound_frame_ids:
                                continue
                            caller_storage = clone_value(
                                argument,
                                next_value_id,
                                accounting={
                                    "linked_call_frame_storage": str(
                                        record.callee_symbol
                                    ),
                                    "callsite_id": int(record.callsite_id),
                                },
                            )
                            next_value_id += 1
                            caller.args.append(caller_storage)
                            values[int(caller_storage.id)] = caller_storage
                            refreshed_frame_bindings.append((
                                argument_id,
                                "caller_storage",
                                int(caller_storage.id),
                            ))
                            bound_frame_ids.add(argument_id)
                            changed = True
                    record = replace(
                        record,
                        callee_storage_value_ids=tuple(
                            int(argument.id) for argument in callee.args
                        ),
                        frame_bindings=tuple(refreshed_frame_bindings),
                        unresolved_frame_value_ids=tuple(
                            int(value_id)
                            for value_id in record.unresolved_frame_value_ids
                            if int(value_id) in current_frame_ids
                        ),
                    )
                was_unresolved = record.resolution == "unresolved"
                callee_records = callee_callers.get(
                    str(record.callee_symbol), ()
                )
                callee_outputs = (
                    () if callee is None
                    else emit_outputs(record.callee_symbol, callee)
                )
                callee_aggregate_outputs = (
                    tuple((callee_outputs[0].accounting or {}).get(
                        "ssa_aggregate_outputs", ()
                    ))
                    if len(callee_outputs) == 1 else ()
                )
                callee_record_table = all_record_tables.get(
                    str(record.callee_symbol)
                )
                caller_record_table = all_record_tables.get(
                    str(record.caller)
                )
                if (
                    callee_record_table is not None
                    and caller_record_table is None
                ):
                    caller_record_table = all_record_tables.setdefault(
                        str(record.caller), SSARecordTable()
                    )
                # A callee record can itself become physical during an inner
                # call-linking round. Materialize the corresponding caller
                # record at that moment rather than requiring it to have
                # existed during initial call discovery.
                if (
                    callee_record_table is not None
                    and caller_record_table is not None
                ):
                    for callee_id, caller_id in record.result_bindings:
                        callee_result_record = (
                            callee_record_table.records.get(int(callee_id))
                        )
                        if (
                            callee_result_record is None
                            or int(caller_id) in caller_record_table.records
                            or any(
                                field.sequence_id is not None
                                or field.record_id is not None
                                for field in (
                                    () if callee_result_record is None
                                    else callee_result_record.fields
                                )
                            )
                        ):
                            continue
                        live_result_map: dict[int, int] = {}
                        mapped_fields = []
                        callee_values = function_values(callee)
                        for field in callee_result_record.fields:
                            mapped_ids = []
                            for callee_value_id in map(int, field.value_ids):
                                caller_value_id = live_result_map.get(
                                    callee_value_id
                                )
                                if caller_value_id is None:
                                    caller_value_id = next_value_id
                                    next_value_id += 1
                                    source = callee_values.get(
                                        callee_value_id,
                                        SSAValue(
                                            callee_value_id,
                                            dtype=field.dtype,
                                        ),
                                    )
                                    value = clone_value(
                                        source,
                                        caller_value_id,
                                        accounting={
                                            "returned_record_storage": str(
                                                record.callee_symbol
                                            ),
                                            "callsite_id": int(
                                                record.callsite_id
                                            ),
                                            "late_record_surface": True,
                                        },
                                    )
                                    caller.args.append(value)
                                    values[caller_value_id] = value
                                    live_result_map[callee_value_id] = (
                                        caller_value_id
                                    )
                                mapped_ids.append(caller_value_id)
                            mapped_fields.append(SSARecordFieldDescriptor(
                                field.name,
                                field.storage,
                                storage_identity=field.storage_identity,
                                value_ids=tuple(mapped_ids),
                                sequence_id=field.sequence_id,
                                record_id=field.record_id,
                                offset=field.offset,
                                dtype=field.dtype,
                                writable=field.writable,
                            ))
                        caller_record_table.register(SSARecordDescriptor(
                            int(caller_id),
                            str(callee_result_record.identity),
                            tuple(mapped_fields),
                        ))
                        result_storage_bindings.update(live_result_map)
                # Record surfaces can become physical after initial call
                # discovery (for example, once a schema constructor's
                # defaulted fields and loop-carried values are recovered).
                # Refresh the result map from the live record tables on every
                # linking fixed-point pass. Stable storage identities, not a
                # stale discovery-time snapshot or source-local numeric ids,
                # prove which caller slot receives each callee field.
                if (
                    callee_record_table is not None
                    and caller_record_table is not None
                ):
                    for callee_id, caller_id in record.result_bindings:
                        callee_result_record = (
                            callee_record_table.records.get(int(callee_id))
                        )
                        caller_result_record = (
                            caller_record_table.records.get(int(caller_id))
                        )
                        if (
                            callee_result_record is None
                            or caller_result_record is None
                        ):
                            continue
                        caller_fields = {
                            str(field.storage_identity): field
                            for field in caller_result_record.fields
                        }
                        for callee_field in callee_result_record.fields:
                            caller_field = caller_fields.get(str(
                                callee_field.storage_identity
                            ))
                            if (
                                caller_field is None
                                or len(callee_field.value_ids)
                                != len(caller_field.value_ids)
                            ):
                                continue
                            result_storage_bindings.update(zip(
                                map(int, callee_field.value_ids),
                                map(int, caller_field.value_ids),
                            ))
                record_return_layouts = dict(
                    () if callee is None else callee.metadata.get(
                        "record_return_layouts", ()
                    )
                )
                live_record_result_map = {
                    int(field_id): int(result_storage_bindings[field_id])
                    for callee_id, _caller_id in record.result_bindings
                    for field_id in record_return_layouts.get(
                        int(callee_id), ()
                    )
                    if int(field_id) in result_storage_bindings
                }
                live_result_slots = set(live_record_result_map.values())
                if live_result_slots and callee is not None:
                    callee_values = {
                        int(argument.id): argument for argument in callee.args
                    }
                    refreshed_bindings = []
                    for callee_id, kind, source in record.frame_bindings:
                        source_id = (
                            int(source)
                            if str(kind) in {
                                "caller_value", "caller_alias", "caller_storage"
                            }
                            else None
                        )
                        if (
                            str(kind) == "caller_storage"
                            and source_id in live_result_slots
                            and live_record_result_map.get(int(callee_id))
                            != source_id
                        ):
                            argument = callee_values.get(
                                int(callee_id), SSAValue(int(callee_id))
                            )
                            replacement = clone_value(
                                argument,
                                next_value_id,
                                accounting={
                                    "linked_call_frame_storage": str(
                                        record.callee_symbol
                                    ),
                                    "callsite_id": int(record.callsite_id),
                                    "split_from_result_storage": source_id,
                                },
                            )
                            next_value_id += 1
                            caller.args.append(replacement)
                            values[int(replacement.id)] = replacement
                            refreshed_bindings.append((
                                int(callee_id), "caller_storage",
                                int(replacement.id),
                            ))
                            changed = True
                        else:
                            refreshed_bindings.append((callee_id, kind, source))
                    record = replace(
                        record, frame_bindings=tuple(refreshed_bindings)
                    )
                if callee is not None:
                    callee_values = {
                        int(argument.id): argument for argument in callee.args
                    }
                    storage_identity_by_value = {}
                    if callee_record_table is not None:
                        for descriptor in callee_record_table.records.values():
                            for field in descriptor.fields:
                                for value_id in field.value_ids:
                                    storage_identity_by_value[int(value_id)] = (
                                        str(field.storage_identity)
                                    )
                    owner_by_slot = {}
                    slot_by_owner = {}
                    distinct_bindings = []
                    for callee_id, kind, source in record.frame_bindings:
                        if str(kind) != "caller_storage":
                            distinct_bindings.append((callee_id, kind, source))
                            continue
                        source_id = int(source)
                        storage_identity = storage_identity_by_value.get(
                            int(callee_id)
                        )
                        owner = (
                            ("record", storage_identity)
                            if storage_identity is not None
                            else ("value", int(callee_id))
                        )
                        first_owner = owner_by_slot.setdefault(source_id, owner)
                        if first_owner == owner:
                            distinct_bindings.append((callee_id, kind, source))
                            continue
                        replacement_id = slot_by_owner.get((source_id, owner))
                        if replacement_id is None:
                            argument = callee_values.get(
                                int(callee_id), SSAValue(int(callee_id))
                            )
                            replacement = clone_value(
                                argument,
                                next_value_id,
                                accounting={
                                    "linked_call_frame_storage": str(
                                        record.callee_symbol
                                    ),
                                    "callsite_id": int(record.callsite_id),
                                    "split_from_unproven_alias": source_id,
                                },
                            )
                            next_value_id += 1
                            caller.args.append(replacement)
                            values[int(replacement.id)] = replacement
                            replacement_id = int(replacement.id)
                            slot_by_owner[(source_id, owner)] = replacement_id
                            changed = True
                        distinct_bindings.append((
                            int(callee_id), "caller_storage", replacement_id,
                        ))
                    record = replace(
                        record, frame_bindings=tuple(distinct_bindings)
                    )
                physical_result_bindings = []
                for callee_id, caller_id in record.result_bindings:
                    layout = tuple(record_return_layouts.get(
                        int(callee_id), ()
                    ))
                    if layout:
                        physical_result_bindings.extend(
                            (int(field_id), int(result_storage_bindings[field_id]))
                            for field_id in layout
                            if int(field_id) in result_storage_bindings
                        )
                    else:
                        physical_result_bindings.append((
                            int(callee_id), int(caller_id)
                        ))
                physical_result_bindings = tuple(physical_result_bindings)
                result_binding = (
                    physical_result_bindings[0]
                    if len(physical_result_bindings) == 1 else None
                )
                returns_structural_record = (
                    bool(record.result_bindings)
                    and callee_record_table is not None
                    and caller_record_table is not None
                    and all(
                        int(callee_id) in callee_record_table.records
                        and int(caller_id) in caller_record_table.records
                        for callee_id, caller_id in record.result_bindings
                    )
                )
                forwarded_aggregate = (
                    not record.result_bindings
                    and len(callee_outputs) > 1
                    and int(record.callsite_id) in set(map(
                        int,
                        caller.metadata.get("source_output_value_ids", ()),
                    ))
                )
                bound_aggregate_outputs = ()
                if (
                    len(record.result_bindings) == 1
                    and len(callee_aggregate_outputs) > 1
                ):
                    aggregate_id = int(record.result_bindings[0][1])
                    candidate_graphs = []
                    if caller_graph is not None:
                        candidate_graphs.append(caller_graph)
                    candidate_graphs.extend(
                        graph for graph in source_graphs_by_symbol.values()
                        if graph is not caller_graph
                    )
                    for aggregate_graph in candidate_graphs:
                        if aggregate_id not in aggregate_graph:
                            continue
                        aggregate_node = aggregate_graph.nodes[aggregate_id]
                        aggregate_attributes = (
                            aggregate_node.get("attributes") or {}
                        )
                        callee_ref = aggregate_attributes.get("callee_ref")
                        if (
                            record.callee_reference is not None
                            and callee_ref is not None
                            and int(callee_ref)
                            != int(record.callee_reference)
                        ):
                            continue
                        projections = []
                        projection_ids = set(map(
                            int, aggregate_graph.successors(aggregate_id)
                        ))
                        projection_ids.update(
                            int(child_id)
                            for child_id, _role
                            in aggregate_node.get(
                                "children", ()
                            )
                        )
                        projection_ids.update(
                            int(node_id)
                            for node_id, data
                            in aggregate_graph.nodes(data=True)
                            if any(
                                int(parent_id) == aggregate_id
                                and str(role) == "base"
                                for parent_id, role in data.get("parents", ())
                            )
                        )
                        for projection_id in projection_ids:
                            projection = aggregate_graph.nodes[projection_id]
                            if str(
                                projection.get("op")
                                or projection.get("type")
                                or ""
                            ).casefold() != "indexed":
                                continue
                            projection_index = (
                                projection.get("attributes") or {}
                            ).get("gradient_result_index")
                            if projection_index is None:
                                continue
                            projections.append((
                                int(projection_index), int(projection_id)
                            ))
                        projections.sort()
                        if tuple(
                            index for index, _node_id in projections
                        ) == tuple(range(len(callee_aggregate_outputs))):
                            bound_aggregate_outputs = tuple(
                                node_id for _index, node_id in projections
                            )
                            break
                    if not bound_aggregate_outputs:
                        downstream_projections = {
                            tuple(map(
                                int,
                                instruction.attributes.get("output_ids", ()),
                            ))
                            for block in caller.blocks.values()
                            for instruction in block.instrs
                            if instruction.op in {"Call", "call"}
                            and any(
                                int(argument.id) == aggregate_id
                                for argument in instruction.args
                            )
                            and len(tuple(
                                instruction.attributes.get("output_ids", ())
                            )) == len(callee_aggregate_outputs)
                        }
                        if len(downstream_projections) == 1:
                            bound_aggregate_outputs = next(iter(
                                downstream_projections
                            ))
                # Source output identities describe authored intent, but they
                # are not a native call result until lowering has retained a
                # physical SSA output.  Treating metadata alone as a result
                # made call linking index an empty ``callee_outputs`` tuple and
                # silently crossed precisely the unresolved object/tensor
                # boundary this table exists to preserve.
                returns_value = (
                    len(physical_result_bindings) == 1
                    and len(callee_outputs) == 1
                    and not callee_aggregate_outputs
                )
                returns_bound_aggregate = (
                    len(physical_result_bindings) == 1
                    and len(callee_aggregate_outputs) > 1
                    and len(bound_aggregate_outputs)
                    == len(callee_aggregate_outputs)
                )
                returns_aggregate = (
                    len(physical_result_bindings) > 1
                    and len(callee_outputs) == len(physical_result_bindings)
                ) or forwarded_aggregate
                returns_physical_result = (
                    returns_value
                    or returns_bound_aggregate
                    or returns_aggregate
                )
                returns_void = (
                    not record.result_bindings and not callee_outputs
                )
                eligible = (
                    was_unresolved
                    and callee is not None
                    and not record.unresolved_frame_value_ids
                    and (
                        returns_value
                        or returns_bound_aggregate
                        or returns_aggregate
                        or returns_void
                        or returns_structural_record
                    )
                    and record.decomposition != "requires_loop_instance_pool"
                    # Repository SSA calls bind linkable symbols, not
                    # recursively materialized function bodies. Requiring all
                    # of a callee's own calls to be resolved first imposes a
                    # false topological order and deadlocks every legitimate
                    # recursive/SCC call graph (tape traversal is one). Each
                    # occurrence is validated by its own frame and result
                    # contract; the module completeness audit catches any
                    # genuinely unresolved member after this fixed point.
                    and int(callee.metadata.get(
                        "source_conditional_count", 0
                    )) == int(callee.metadata.get(
                        "lowered_conditional_count", 0
                    ))
                )
                eligibility_reasons = tuple(filter(None, (
                    "not_pending" if not was_unresolved else None,
                    "missing_callee" if callee is None else None,
                    "unresolved_frame" if record.unresolved_frame_value_ids else None,
                    "unmaterialized_result" if not (
                        returns_value
                        or returns_bound_aggregate
                        or returns_aggregate
                        or returns_void
                        or returns_structural_record
                    ) else None,
                    "loop_instance_pool_required" if (
                        record.decomposition == "requires_loop_instance_pool"
                    ) else None,
                    "conditional_surface_incomplete" if (
                        callee is not None
                        and int(callee.metadata.get("source_conditional_count", 0))
                        != int(callee.metadata.get("lowered_conditional_count", 0))
                    ) else None,
                )))
                call_argument_failure = None
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
                        binding = binding_by_callee.get(int(argument.id))
                        if binding is None:
                            eligible = False
                            break
                        kind, source = binding
                        if kind in {
                            "caller_value", "caller_alias", "caller_storage"
                        }:
                            value = resolve_call_feed(int(source), constants)
                            if value is None and kind == "caller_storage":
                                # A structural-record cleanup may remove a
                                # shapeless argument whose numeric id happens
                                # to alias a physical frame slot.  The binding
                                # still proves that slot belongs to this call,
                                # so restore the callee-shaped storage value
                                # rather than treating it as a Python input.
                                value = clone_value(
                                    argument,
                                    int(source),
                                    accounting={
                                        "linked_call_frame_storage": str(
                                            record.callee_symbol
                                        ),
                                        "callsite_id": int(
                                            record.callsite_id
                                        ),
                                    },
                                )
                                caller.args.append(value)
                                values[int(source)] = value
                            if value is None:
                                call_argument_failure = (
                                    f"missing_{kind}:{int(source)}"
                                )
                                eligible = False
                                break
                            call_arguments.append(value)
                        elif kind in {"default_literal", "caller_literal"}:
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
                            call_argument_failure = f"unsupported_binding:{kind}"
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
                        # The caller-side placeholder may predate source-call
                        # linking and therefore carry no useful type.  A
                        # resolved call's physical result contract is the
                        # callee output itself; copy it onto the retained SSA
                        # value instead of letting backend defaults silently
                        # turn predicates into floating-point values.
                        result.dtype = callee_output.dtype
                        result.shape = tuple(callee_output.shape)
                        result.device = callee_output.device
                        result.accounting = {
                            **dict(result.accounting or {}),
                            **dict(callee_output.accounting or {}),
                        }
                    elif returns_bound_aggregate:
                        _callee_result_id, caller_result_id = result_binding
                        result = values.get(
                            int(caller_result_id),
                            SSAValue(int(caller_result_id)),
                        )
                        result.accounting = {
                            **dict(result.accounting or {}),
                            "ssa_aggregate_outputs": bound_aggregate_outputs,
                        }
                    elif returns_aggregate:
                        caller_result_id = int(record.callsite_id)
                        if caller_result_id in produced_ids:
                            # The call-site's own AST node id coincides with
                            # a value some OTHER, already-existing
                            # instruction already produces -- drawn from an
                            # unrelated numbering source (e.g. a
                            # required-source-value pulled out of a
                            # different call's aggregate output via
                            # `source_output_id`). Adopting it here would
                            # give two different instructions the same SSA
                            # identity, which is exactly the class of bug
                            # the freshening pass later in this function
                            # cannot safely repair (it renames a colliding
                            # `.res` in place but never rewrites the other
                            # instructions that already reference the old
                            # id by number). Allocate a genuinely fresh id
                            # for this call's own aggregate result instead.
                            caller_result_id = next_value_id
                            next_value_id += 1
                        result = values.get(
                            caller_result_id,
                            SSAValue(
                                caller_result_id,
                                accounting={
                                    "ssa_aggregate_outputs": tuple(
                                        (
                                            int(caller_id)
                                            for _callee_id, caller_id
                                            in physical_result_bindings
                                        ) if physical_result_bindings else (
                                            int(value.id)
                                            for value in callee_outputs
                                        )
                                    )
                                },
                            ),
                        )
                        produced_ids.add(caller_result_id)
                    else:
                        caller_result_id = (
                            int(record.result_bindings[0][1])
                            if returns_structural_record else None
                        )
                        result = None
                    native_call = Instr(
                        "Call", call_arguments, result,
                        attributes={
                            "callee": record.callee_symbol,
                            "source_linked": True,
                            "plan_callsite_id": record.callsite_id,
                            "callee_reference": record.callee_reference,
                            **({
                                "result_convention": "ssa.aggregate",
                                "output_ids": tuple(
                                    bound_aggregate_outputs
                                    if returns_bound_aggregate else (
                                        int(caller_id)
                                        for _callee_id, caller_id
                                        in physical_result_bindings
                                    )
                                ),
                            } if (
                                returns_bound_aggregate or returns_aggregate
                            ) else {}),
                        },
                    )
                    aggregate_unpack = []
                    if returns_aggregate and result is not None:
                        callee_outputs_by_id = {
                            int(value.id): value for value in callee_outputs
                        }
                        for output_index, (callee_id, caller_id) in enumerate(
                            physical_result_bindings
                        ):
                            index_value = SSAValue(next_value_id, dtype="int")
                            next_value_id += 1
                            address = SSAValue(next_value_id, dtype="ptr")
                            next_value_id += 1
                            caller_node = caller_graph.nodes.get(
                                int(caller_id), {}
                            )
                            caller_tensor = caller_node.get("tensor") or {}
                            output = values.get(
                                int(caller_id),
                                SSAValue(
                                    int(caller_id),
                                    dtype=caller_tensor.get("dtype"),
                                    shape=tuple(
                                        caller_tensor.get("shape") or ()
                                    ),
                                ),
                            )
                            callee_output = callee_outputs_by_id.get(
                                int(callee_id)
                            )
                            if callee_output is not None:
                                # PlanCall result bindings are the exact type
                                # correlation.  The caller graph describes
                                # semantic source shape, but the callee's
                                # physical output owns the repository-SSA ABI.
                                output.dtype = callee_output.dtype
                                output.shape = tuple(callee_output.shape)
                                output.device = callee_output.device
                                output.accounting = {
                                    **dict(output.accounting or {}),
                                    **dict(callee_output.accounting or {}),
                                    "ssa_call_result_from": (
                                        str(record.callee_symbol),
                                        int(callee_id),
                                    ),
                                }
                            aggregate_unpack.extend((
                                Instr(
                                    "Const", [], index_value,
                                    attributes={"value": int(output_index)},
                                ),
                                Instr(
                                    "GetElementPtr",
                                    [result, index_value],
                                    address,
                                    attributes={
                                        "aggregate_index": int(output_index)
                                    },
                                ),
                                Instr(
                                    "Load", [address], output,
                                    attributes={
                                        "aggregate_index": int(output_index),
                                        "source_output_id": int(caller_id),
                                    },
                                ),
                            ))
                            values[int(caller_id)] = output
                    native_sequence = [
                        *constants, native_call, *aggregate_unpack
                    ]
                    # A source-linked call inside a loop is scheduled by the
                    # reducer's lexical call anchor within that exact loop
                    # compartment.  Its eventual result consumer may live at
                    # loop exit or function Ret and is therefore not a valid
                    # execution anchor.
                    inserted = insert_at_loop_anchor(
                        record, native_sequence
                    )
                    if returns_physical_result:
                        consumed_result_ids = (
                            {
                                int(caller_id)
                                for _callee_id, caller_id
                                in physical_result_bindings
                            } | {int(caller_result_id)}
                            if returns_aggregate
                            else {int(caller_result_id)}
                        )
                        if not inserted:
                            for block in caller.blocks.values():
                                for index, instruction in enumerate(block.instrs):
                                    if any(
                                        int(value.id) in consumed_result_ids
                                        for value in instruction.args
                                    ):
                                        block.instrs[index:index] = native_sequence
                                        inserted = True
                                        break
                                if inserted:
                                    break
                    else:
                        if not inserted and record.enclosing_loop_ids:
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
                    source_output_ids = tuple(map(
                        int,
                        caller.metadata.get("source_output_value_ids", ()),
                    ))
                    produced_results = {
                        int(caller_id): values[int(caller_id)]
                        for _callee_id, caller_id in record.result_bindings
                        if int(caller_id) in values
                    }
                    caller_record_table = all_record_tables.get(
                        caller_symbol
                    )
                    if caller_record_table is not None:
                        for source_output_id in source_output_ids:
                            if source_output_id in (
                                caller_record_table.records
                            ):
                                produced_results.setdefault(
                                    source_output_id,
                                    SSAValue(
                                        source_output_id,
                                        accounting={
                                            "structural_record_result": True,
                                            "callsite_id": int(
                                                record.callsite_id
                                            ),
                                        },
                                    ),
                                )
                    if (
                        returns_value
                        or returns_bound_aggregate
                        or forwarded_aggregate
                    ) and caller_result_id is not None and result is not None:
                        produced_results[int(caller_result_id)] = result
                    authored_results = {
                        value_id: produced_results[value_id]
                        for value_id in source_output_ids
                        if value_id in produced_results
                    }
                    if authored_results:
                        # A function whose body is solely ``return callee(...)``
                        # has no ordinary consumer instruction to anchor the
                        # call: control lowering emitted an empty Ret because
                        # PlanCall is linked afterward.  The same applies to an
                        # unpacked multi-result call: materializing the Call and
                        # its aggregate projections does not retroactively add
                        # those projections to Ret.  The source-output ledger is
                        # the exact authored order, so publish every produced
                        # result there whether the call already found an
                        # ordinary insertion anchor or must use Ret itself.
                        for block in caller.blocks.values():
                            if (
                                block.instrs
                                and block.instrs[-1].op in {
                                    "Ret", "ret", "Return", "return"
                                }
                            ):
                                if not inserted:
                                    block.instrs[-1:-1] = native_sequence
                                returned = {
                                    int(argument.id): argument
                                    for argument in block.instrs[-1].args
                                }
                                returned.update(authored_results)
                                block.instrs[-1].args = [
                                    returned[value_id]
                                    for value_id in source_output_ids
                                    if value_id in returned
                                ]
                                inserted = True
                                break
                    if (
                        not inserted
                        and returns_physical_result
                        and caller_result_id is not None
                    ):
                        # A source-call chain can have no materialized direct
                        # consumer yet: the next PlanCall is itself pending.
                        # Anchor the producer at the first downstream result
                        # that the scheduled SSA already consumes. This keeps
                        # dependency order without turning an intermediate
                        # source-call result into a host ABI argument.
                        anchor = downstream_anchor(int(caller_result_id))
                        if anchor is not None:
                            for block in caller.blocks.values():
                                for index, instruction in enumerate(block.instrs):
                                    if any(
                                        int(argument.id) == int(anchor)
                                        for argument in instruction.args
                                    ):
                                        block.instrs[index:index] = [
                                            *constants, native_call
                                        ]
                                        inserted = True
                                        break
                                if inserted:
                                    break
                    if (
                        not inserted
                        and returns_physical_result
                        and record.enclosing_loop_ids
                    ):
                        target_loop_id = int(record.enclosing_loop_ids[-1])
                        header = next((
                            block
                            for block in caller.blocks.values()
                            if any(
                                instruction.op == "Phi"
                                and instruction.attributes.get("source_name")
                                == f"iteration_{target_loop_id}"
                                for instruction in block.instrs
                            )
                        ), None)
                        branch = (
                            None if header is None else next((
                                instruction for instruction in header.instrs
                                if instruction.op == "CondBr"
                            ), None)
                        )
                        body = (
                            None if branch is None else caller.blocks.get(str(
                                branch.attributes.get("true_target")
                            ))
                        )
                        if body is not None:
                            insertion_index = next((
                                index
                                for index, instruction in enumerate(body.instrs)
                                if instruction.attributes.get(
                                    "ssa_sequence_operation"
                                ) in {"append", "add", "store"}
                            ), None)
                            if insertion_index is None and body.instrs:
                                insertion_index = (
                                    len(body.instrs) - 1
                                    if body.instrs[-1].op in {
                                        "Br", "br", "Branch", "branch"
                                    }
                                    else len(body.instrs)
                                )
                            if insertion_index is not None:
                                body.instrs[insertion_index:insertion_index] = [
                                    *native_sequence
                                ]
                                inserted = True
                        if not inserted:
                            preceding_calls = []
                            for candidate_block in caller.blocks.values():
                                for candidate in candidate_block.instrs:
                                    candidate_callsite = candidate.attributes.get(
                                        "plan_callsite_id"
                                    )
                                    if (
                                        candidate.op in {"Call", "call"}
                                        and candidate_callsite is not None
                                        and int(candidate_callsite)
                                        < int(record.callsite_id)
                                    ):
                                        preceding_calls.append((
                                            int(candidate_callsite),
                                            candidate_block,
                                        ))
                            if preceding_calls:
                                _callsite, candidate_block = max(
                                    preceding_calls,
                                    key=lambda item: item[0],
                                )
                                insertion_index = len(candidate_block.instrs)
                                if (
                                    candidate_block.instrs
                                    and candidate_block.instrs[-1].op in {
                                        "Br", "br", "Branch", "branch",
                                        "Ret", "ret", "Return", "return",
                                    }
                                ):
                                    insertion_index -= 1
                                candidate_block.instrs[
                                    insertion_index:insertion_index
                                ] = native_sequence
                                inserted = True
                    if (
                        not inserted
                        and returns_physical_result
                        and not record.enclosing_loop_ids
                    ):
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
                    if inserted:
                        if returns_physical_result:
                            caller.args = [
                                value for value in caller.args
                                if int(value.id) != int(caller_result_id)
                            ]
                            values[int(caller_result_id)] = result
                        record = replace(record, resolution="native_call")
                        diagnostics = dict(caller.metadata.get(
                            "unresolved_call_diagnostics", {}
                        ))
                        diagnostics.pop(int(record.callsite_id), None)
                        if diagnostics:
                            caller.metadata[
                                "unresolved_call_diagnostics"
                            ] = diagnostics
                        else:
                            caller.metadata.pop(
                                "unresolved_call_diagnostics", None
                            )
                        changed = True
                if record.resolution == "unresolved":
                    diagnostics = dict(caller.metadata.get(
                        "unresolved_call_diagnostics", {}
                    ))
                    diagnostics[int(record.callsite_id)] = {
                        "callee": str(record.callee_symbol),
                        "reasons": (
                            *eligibility_reasons,
                            *((call_argument_failure,)
                              if call_argument_failure else ()),
                            *(
                                ("insertion_point_missing",)
                                if eligible else ()
                            ),
                        ),
                        "callee_output_count": len(callee_outputs),
                        "physical_result_count": len(
                            physical_result_bindings
                        ),
                        "semantic_result_count": len(
                            record.result_bindings
                        ),
                        "returns_structural_record": bool(
                            returns_structural_record
                        ),
                    }
                    caller.metadata["unresolved_call_diagnostics"] = diagnostics
                rebuilt_records.append(record)
            call_records[caller_symbol] = rebuilt_records

        # A call resolved in this round may have created physical fields for
        # a record-valued public result. Expand that Ret before the next round
        # so callers observe the callee's new aggregate surface as part of the
        # same dependency fixed point.
        for function_name, record_table in all_record_tables.items():
            function = all_functions.get(function_name)
            if function is None:
                continue
            current_values = function_values(function)
            layouts = dict(function.metadata.get(
                "record_return_layouts", ()
            ))
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.op not in {
                        "Ret", "ret", "Return", "return"
                    }:
                        continue
                    expanded = []
                    changed_return = False
                    for argument in instruction.args:
                        returned_record = record_table.records.get(
                            int(argument.id)
                        )
                        if returned_record is None:
                            expanded.append(argument)
                            continue
                        layout = tuple(
                            int(value_id)
                            for field in returned_record.fields
                            for value_id in field.value_ids
                            if int(value_id) in current_values
                        )
                        if not layout:
                            expanded.append(argument)
                            continue
                        carried = dict(
                            function.metadata.get("carried_port_values")
                            or {}
                        )
                        expanded.extend(
                            carried.get(
                                int(value_id), current_values[value_id]
                            )
                            for value_id in layout
                        )
                        layouts[int(returned_record.record_id)] = layout
                        changed_return = True
                    if changed_return:
                        instruction.args = expanded
                        changed = True
            if layouts:
                function.metadata["record_return_layouts"] = tuple(
                    layouts.items()
                )

    # Object/call-frame discovery precedes some ordinary SSA-producing passes.
    # Both phases allocate monotonically within the values visible at the
    # time, so a synthetic projection/index can otherwise reuse an authored
    # argument or output id materialized later.  This is target-neutral: two
    # distinct SSAValue objects may not own the same integer identity.
    #
    # Preserve arguments and authored/named outputs. Freshen only the other
    # result objects; every operand holds the object itself, so its edges,
    # ordering, types, and scheduling remain unchanged and no call/record ABI
    # identity needs rewriting.
    for function_name, function in all_functions.items():
        instructions = [
            instruction
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        ]
        occupied = {
            int(argument.id) for argument in function.args
        } | {
            int(instruction.res.id) for instruction in instructions
        }
        canonical: dict[int, Any] = {}
        for argument in function.args:
            canonical.setdefault(int(argument.id), argument)
        for output in emit_outputs(function_name, function):
            canonical.setdefault(int(output.id), output)
        for instruction in instructions:
            result_id = int(instruction.res.id)
            if int(instruction.attributes.get(
                "source_output_id", -1
            )) == result_id:
                canonical.setdefault(result_id, instruction.res)
        for instruction in instructions:
            canonical.setdefault(int(instruction.res.id), instruction.res)

        next_value_id = 1 + max(occupied, default=0)
        freshened: dict[int, int] = {}
        seen_objects: set[int] = set()
        for instruction in instructions:
            result = instruction.res
            object_id = id(result)
            if object_id in seen_objects:
                continue
            seen_objects.add(object_id)
            old_id = int(result.id)
            if canonical[old_id] is result:
                continue
            while next_value_id in occupied:
                next_value_id += 1
            result.id = next_value_id
            occupied.add(next_value_id)
            freshened[old_id] = next_value_id
            next_value_id += 1
        if freshened:
            function.metadata["freshened_synthetic_value_ids"] = tuple(
                sorted(freshened.items())
            )

    # A native Call is an equality constraint between each caller operand and
    # its callee parameter.  Settle that constraint in repository SSA so every
    # backend receives the same dtype and dynamic-rank facts.  An explicit
    # ABI/physical type on the formal remains authoritative; otherwise the
    # authored caller occurrence replaces a default/unaccounted formal type.
    changed_call_types = True
    while changed_call_types:
        changed_call_types = False
        for caller in all_functions.values():
            for block in caller.blocks.values():
                for instruction in block.instrs:
                    if (
                        instruction.op not in {"Call", "call"}
                        or instruction.attributes.get("tensor_operation")
                    ):
                        continue
                    callee = all_functions.get(str(
                        instruction.attributes.get("callee") or ""
                    ))
                    if callee is None:
                        continue
                    for actual, formal in zip(instruction.args, callee.args):
                        actual_rank = max(
                            len(tuple(actual.shape or ())),
                            int((actual.accounting or {}).get(
                                "program_abi_rank", 0
                            )),
                            int((actual.accounting or {}).get(
                                "ssa_call_rank", 0
                            )),
                        )
                        formal_rank = max(
                            len(tuple(formal.shape or ())),
                            int((formal.accounting or {}).get(
                                "program_abi_rank", 0
                            )),
                            int((formal.accounting or {}).get(
                                "ssa_call_rank", 0
                            )),
                        )
                        call_rank = max(actual_rank, formal_rank)
                        for value, rank in (
                            (actual, actual_rank), (formal, formal_rank)
                        ):
                            if rank == call_rank or call_rank == 0:
                                continue
                            value.accounting = {
                                **dict(value.accounting or {}),
                                "ssa_call_rank": call_rank,
                            }
                            changed_call_types = True

                        actual_dtype = str(actual.dtype or "")
                        formal_dtype = str(formal.dtype or "")
                        formal_accounting = dict(formal.accounting or {})
                        formal_is_physical = bool(
                            formal_accounting.get("physical_dtype")
                            or formal_accounting.get("program_abi_storage")
                        )
                        formal_is_contracted = bool(
                            formal_is_physical
                            or formal_accounting.get("ssa_call_dtype")
                        )
                        actual_is_exact_result = bool(
                            (actual.accounting or {}).get(
                                "ssa_call_result_from"
                            )
                        )
                        actual_is_link_storage = bool(
                            (actual.accounting or {}).get(
                                "returned_record_storage"
                            )
                            or (actual.accounting or {}).get(
                                "linked_call_frame_storage"
                            )
                        )
                        if (
                            actual_is_exact_result
                            and not formal_is_physical
                            and actual_dtype
                            and actual_dtype != "unknown"
                            and formal_dtype != actual_dtype
                        ):
                            # A PlanCall result binding correlates the callee's
                            # physical output with this caller value exactly.
                            # It outranks a dtype previously inferred onto the
                            # consumer formal, but never an explicit physical
                            # or program-ABI declaration.
                            formal.dtype = actual.dtype
                            formal.accounting = {
                                **formal_accounting,
                                "ssa_call_dtype": actual_dtype,
                                "ssa_call_result_source": tuple(
                                    (actual.accounting or {})[
                                        "ssa_call_result_from"
                                    ]
                                ),
                            }
                            changed_call_types = True
                        elif (
                            formal_is_contracted
                            and actual_is_link_storage
                            and formal_dtype
                            and formal_dtype != "unknown"
                            and actual_dtype != formal_dtype
                        ):
                            actual.dtype = formal.dtype
                            actual.accounting = {
                                **dict(actual.accounting or {}),
                                "ssa_call_dtype": formal_dtype,
                            }
                            changed_call_types = True
                        elif (
                            actual_dtype
                            and actual_dtype != "unknown"
                            and actual_dtype != formal_dtype
                            and not formal_is_contracted
                        ):
                            formal.dtype = actual.dtype
                            formal.accounting = {
                                **formal_accounting,
                                "ssa_call_dtype": actual_dtype,
                            }
                            changed_call_types = True
                        elif (
                            formal_dtype
                            and formal_dtype != "unknown"
                            and actual_dtype in {"", "unknown"}
                        ):
                            actual.dtype = formal.dtype
                            changed_call_types = True

    # Argument equality reaches its fixed point before result projection.
    # Projecting results inside that bidirectional loop lets provisional
    # consumer types feed back into their own producers and oscillate.  The
    # callee output ABI is now settled, so copy it outward once through the
    # exact aggregate result bindings, then update immediate consumer formals.
    exact_result_values: set[int] = set()
    for caller in all_functions.values():
        caller_values = function_values(caller)
        for block in caller.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op not in {"Call", "call"}
                    or instruction.attributes.get("result_convention")
                    != "ssa.aggregate"
                ):
                    continue
                callee = all_functions.get(str(
                    instruction.attributes.get("callee") or ""
                ))
                output_ids = tuple(map(
                    int, instruction.attributes.get("output_ids", ())
                ))
                if callee is None or not output_ids:
                    continue
                callee_outputs = tuple(emit_outputs(callee.name, callee))
                if len(output_ids) != len(callee_outputs):
                    continue
                for caller_id, callee_output in zip(output_ids, callee_outputs):
                    caller_output = caller_values.get(caller_id)
                    if caller_output is None:
                        continue
                    caller_output.dtype = callee_output.dtype
                    caller_output.shape = tuple(callee_output.shape)
                    caller_output.device = callee_output.device
                    caller_output.accounting = {
                        **dict(caller_output.accounting or {}),
                        "ssa_call_result_from": (
                            str(callee.name), int(callee_output.id)
                        ),
                    }
                    exact_result_values.add(id(caller_output))
    for caller in all_functions.values():
        for block in caller.blocks.values():
            for instruction in block.instrs:
                if instruction.op not in {"Call", "call"}:
                    continue
                callee = all_functions.get(str(
                    instruction.attributes.get("callee") or ""
                ))
                if callee is None:
                    continue
                for actual, formal in zip(instruction.args, callee.args):
                    if id(actual) not in exact_result_values:
                        continue
                    formal_accounting = dict(formal.accounting or {})
                    if (
                        formal_accounting.get("physical_dtype")
                        or formal_accounting.get("program_abi_storage")
                    ):
                        continue
                    formal.dtype = actual.dtype
                    formal.shape = tuple(actual.shape)
                    formal.device = actual.device
                    formal.accounting = {
                        **formal_accounting,
                        "ssa_call_dtype": str(actual.dtype or "unknown"),
                        "ssa_call_result_source": tuple(
                            (actual.accounting or {})[
                                "ssa_call_result_from"
                            ]
                        ),
                    }

    # Linking a callee's own calls can expand its physical argument frame
    # after an incoming native Call was materialized in an earlier fixed-point
    # round. Refresh those already-emitted call operands from the final call
    # records so callsite and callee ABIs cannot drift by dependency order.
    for caller_symbol, records in call_records.items():
        caller = all_functions.get(caller_symbol)
        if caller is None:
            continue
        caller_values = function_values(caller)
        caller_graph = source_graphs_by_symbol.get(caller_symbol)
        caller_records = all_record_tables.get(caller_symbol)

        def final_frame_value(source_id: int) -> SSAValue | None:
            source_id = int(source_id)
            value = caller_values.get(source_id)
            if value is not None or caller_graph is None:
                return value
            data = caller_graph.nodes.get(source_id, {})
            if str(
                data.get("op") or data.get("type") or ""
            ).casefold() != "getattr":
                return None
            receiver_id = next((
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role) in {"value", "object", "base", "operand"}
            ), None)
            record = (
                None
                if caller_records is None or receiver_id is None
                else caller_records.records.get(receiver_id)
            )
            attribute = str((data.get("attributes") or {}).get(
                "attribute", ""
            ))
            field = (
                None if record is None else next((
                    field for field in record.fields
                    if str(field.name) == attribute
                ), None)
            )
            if field is None or len(field.value_ids) != 1:
                return None
            return caller_values.get(int(field.value_ids[0]))

        next_value_id = 1 + max(caller_values, default=0)
        for record in records:
            if record.resolution != "native_call":
                continue
            callee = all_functions.get(str(record.callee_symbol))
            if callee is None:
                continue
            binding_by_callee = {
                int(value_id): (str(kind), source)
                for value_id, kind, source in record.frame_bindings
            }
            call_site = next((
                (block, index, instruction)
                for block in caller.blocks.values()
                for index, instruction in enumerate(block.instrs)
                if instruction.op in {"Call", "call"}
                and instruction.attributes.get("source_linked")
                and instruction.attributes.get("plan_callsite_id") is not None
                and int(instruction.attributes["plan_callsite_id"])
                == int(record.callsite_id)
                and str(instruction.attributes.get("callee"))
                == str(record.callee_symbol)
            ), None)
            if call_site is None:
                continue
            block, index, instruction = call_site
            refreshed = []
            constants = []
            complete = True
            for argument in callee.args:
                binding = binding_by_callee.get(int(argument.id))
                if binding is None:
                    complete = False
                    break
                kind, source = binding
                if kind in {
                    "caller_value", "caller_alias", "caller_storage"
                }:
                    value = final_frame_value(int(source))
                    if value is None:
                        complete = False
                        break
                    exact_result_source = (argument.accounting or {}).get(
                        "ssa_call_result_from"
                    )
                    value_accounting = dict(value.accounting or {})
                    if (
                        exact_result_source
                        and not value_accounting.get("physical_dtype")
                        and not value_accounting.get("program_abi_storage")
                    ):
                        # The finalized frame binding is the planner's exact
                        # identity correlation, not a new type-inference
                        # opportunity.  Refreshing a late-created record slot
                        # must carry the already-settled callee result ABI with
                        # it; otherwise a provisional scalar default survives
                        # only because this operand was materialized after the
                        # call-type fixed point.
                        value.dtype = argument.dtype
                        value.shape = tuple(argument.shape)
                        value.device = argument.device
                        value.accounting = {
                            **value_accounting,
                            "ssa_call_result_from": tuple(
                                exact_result_source
                            ),
                            "ssa_call_dtype": str(
                                argument.dtype or "unknown"
                            ),
                        }
                    refreshed.append(value)
                    continue
                if kind in {"default_literal", "caller_literal"}:
                    if isinstance(source, FunctionReference):
                        complete = False
                        break
                    value = SSAValue(
                        next_value_id,
                        dtype=argument.dtype,
                        shape=argument.shape,
                    )
                    next_value_id += 1
                    constants.append(Instr(
                        "Const", [], value, attributes={"value": source},
                    ))
                    caller_values[int(value.id)] = value
                    refreshed.append(value)
                    continue
                complete = False
                break
            if not complete:
                continue
            if constants:
                block.instrs[index:index] = constants
            instruction.args = refreshed

    # Native call linking can create a caller-owned record descriptor and its
    # aggregate-unpack values after the earlier source-output recovery pass.
    # Finalize every such public record now so Ret exposes physical fields,
    # never the conceptual Python record handle.
    for function_name, function in all_functions.items():
        available = set(function_values(function))
        graph = source_graphs_by_symbol.get(function_name)
        record_table = all_record_tables.get(function_name)
        if graph is not None and record_table is not None:
            for node_id, data in graph.nodes(data=True):
                if str(
                    data.get("op") or data.get("type") or ""
                ).casefold() != "getattr":
                    continue
                receiver_id = next((
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role) in {
                        "value", "object", "base", "operand"
                    }
                ), None)
                record = (
                    None if receiver_id is None
                    else record_table.records.get(receiver_id)
                )
                attribute = str((data.get("attributes") or {}).get(
                    "attribute", ""
                ))
                field = (
                    None if record is None else next((
                        field for field in record.fields
                        if str(field.name) == attribute
                    ), None)
                )
                if (
                    field is not None
                    and field.value_ids
                    and all(int(value_id) in available
                            for value_id in field.value_ids)
                ):
                    available.add(int(node_id))
        shortfalls = tuple(
            row for row in function.metadata.get(
                "structural_output_shortfalls", ()
            )
            if int(row[0]) not in available
        )
        if shortfalls:
            function.metadata["structural_output_shortfalls"] = shortfalls
        else:
            function.metadata.pop("structural_output_shortfalls", None)
        unresolved_required = tuple(
            row for row in function.metadata.get(
                "unresolved_required_source_values", ()
            )
            if int(row[0]) not in available
        )
        if unresolved_required:
            function.metadata[
                "unresolved_required_source_values"
            ] = unresolved_required
        else:
            function.metadata.pop(
                "unresolved_required_source_values", None
            )
    for function_name, record_table in all_record_tables.items():
        function = all_functions.get(function_name)
        if function is None:
            continue
        values = function_values(function)
        layouts = dict(function.metadata.get("record_return_layouts", ()))
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op not in {"Ret", "ret", "Return", "return"}:
                    continue
                expanded = []
                changed_return = False
                for argument in instruction.args:
                    record = record_table.records.get(int(argument.id))
                    if record is None:
                        expanded.append(argument)
                        continue
                    layout = tuple(
                        int(value_id)
                        for field in record.fields
                        for value_id in field.value_ids
                        if int(value_id) in values
                    )
                    if not layout:
                        expanded.append(argument)
                        continue
                    # A component standing at a LoopResult port means the
                    # carried phi; the raw field value is the port's
                    # unwritten slot.
                    carried = dict(
                        function.metadata.get("carried_port_values") or {}
                    )
                    expanded.extend(
                        carried.get(int(value_id), values[value_id])
                        for value_id in layout
                    )
                    layouts[int(record.record_id)] = layout
                    changed_return = True
                if changed_return:
                    instruction.args = expanded
        if layouts:
            function.metadata["record_return_layouts"] = tuple(
                layouts.items()
            )

    # A constructed-record result is a compile-time correlation once every
    # consumer has been rewritten to its physical field arenas or pool handle.
    # Remove only the shapeless conceptual receiver argument; a sequence
    # capacity or other physical ABI value may legitimately share the same
    # source-local numeric id and must remain.
    for function_name, function in all_functions.items():
        record_table = all_record_tables.get(function_name)
        record_ids = set(
            () if record_table is None else map(int, record_table.records)
        )
        source_graph = source_graphs_by_symbol.get(function_name)
        if source_graph is not None:
            identities = source_graph.graph.get("identity_table") or {}
            for parameter_name in (
                source_graph.graph.get("parameter_record_abi") or {}
            ):
                record_ids.update(map(
                    int, identities.get(str(parameter_name), ())
                ))
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

    # The cleanup above deliberately removes conceptual record handles from
    # final physical signatures.  Some incoming Calls were refreshed before
    # that cleanup, so reconcile them once more from the exact call-frame
    # contract.  This is not positional trimming: each surviving formal is
    # rebound by its callee-local SSA identity, preserving ordinary SSA flow
    # when structural OOP values disappear from the native ABI.
    for caller_symbol, records in call_records.items():
        caller = all_functions.get(caller_symbol)
        if caller is None:
            continue
        caller_values = function_values(caller)
        caller_graph = source_graphs_by_symbol.get(caller_symbol)
        caller_record_table = all_record_tables.get(caller_symbol)
        next_value_id = 1 + max(caller_values, default=0)

        def cleaned_frame_value(source_id: int) -> SSAValue | None:
            source_id = int(source_id)
            value = caller_values.get(source_id)
            if value is not None or caller_graph is None:
                return value
            data = caller_graph.nodes.get(source_id, {})
            if str(
                data.get("op") or data.get("type") or ""
            ).casefold() != "getattr":
                return None
            receiver_id = next((
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role) in {"value", "object", "base", "operand"}
            ), None)
            descriptor = (
                None
                if caller_record_table is None or receiver_id is None
                else caller_record_table.records.get(receiver_id)
            )
            attribute = str((data.get("attributes") or {}).get(
                "attribute", ""
            ))
            field = (
                None if descriptor is None else next((
                    item for item in descriptor.fields
                    if str(item.name) == attribute
                ), None)
            )
            if field is None or len(field.value_ids) != 1:
                return None
            return caller_values.get(int(field.value_ids[0]))

        for record in records:
            if record.resolution != "native_call":
                continue
            callee = all_functions.get(str(record.callee_symbol))
            if callee is None:
                continue
            binding_by_callee = {
                int(value_id): (str(kind), source)
                for value_id, kind, source in record.frame_bindings
            }
            call_site = next((
                (block, index, instruction)
                for block in caller.blocks.values()
                for index, instruction in enumerate(block.instrs)
                if instruction.op in {"Call", "call"}
                and instruction.attributes.get("source_linked")
                and instruction.attributes.get("plan_callsite_id") is not None
                and int(instruction.attributes["plan_callsite_id"])
                == int(record.callsite_id)
                and str(instruction.attributes.get("callee"))
                == str(record.callee_symbol)
            ), None)
            if call_site is None:
                continue
            block, index, instruction = call_site
            refreshed = []
            constants = []
            for argument in callee.args:
                binding = binding_by_callee.get(int(argument.id))
                if binding is None:
                    refreshed = []
                    break
                kind, source = binding
                if kind in {
                    "caller_value", "caller_alias", "caller_storage"
                }:
                    value = cleaned_frame_value(int(source))
                    if value is None:
                        refreshed = []
                        break
                    refreshed.append(value)
                elif kind in {"default_literal", "caller_literal"}:
                    if isinstance(source, FunctionReference):
                        refreshed = []
                        break
                    value = SSAValue(
                        next_value_id,
                        dtype=argument.dtype,
                        shape=argument.shape,
                    )
                    next_value_id += 1
                    constants.append(Instr(
                        "Const", [], value, attributes={"value": source},
                    ))
                    caller_values[int(value.id)] = value
                    refreshed.append(value)
                else:
                    refreshed = []
                    break
            if len(refreshed) != len(callee.args):
                continue
            if constants:
                block.instrs[index:index] = constants
            instruction.args = refreshed

    # A table lookup on a keyed mapping walks the mapping's own declared
    # vectors.  Its descriptor was built during lowering from anonymous
    # storage -- (keys, values, length, capacity) fresh arguments -- because
    # the slots exist only after record materialization and call-frame
    # linking.  Every frame is linked now, so bind them: keys/values/length
    # are the owner's parts, and a caller-supplied mapping is always exactly
    # full, so capacity IS the length -- the same value fills both formal
    # positions.  Both formals must therefore agree with that one value's
    # real width (int64, matching keys/query below), not the generic
    # scalar-arena default: declaring either as int32 while the caller's
    # actual keyed-field length is int64 is a real Fortran ABI mismatch, not
    # a cosmetic one, since a shared value can only have one true width.
    # The status cell stays an ordinary frame-allocated scalar.
    _keyed_helper_dtypes = (
        ("int64", None), ("float64", None), ("int64", (1,)),
        ("int64", None), ("int", (1,)), ("int64", None),
    )
    for function in all_functions.values():
        parts_by_owner: dict[str, dict[str, Any]] = {}
        for value in function.args:
            accounting = value.accounting or {}
            owner_name = accounting.get("program_abi_keyed_owner")
            part_name = accounting.get("program_abi_keyed_part")
            if owner_name is None or part_name is None:
                continue
            parts_by_owner.setdefault(str(owner_name), {})[
                str(part_name)
            ] = value
        if not parts_by_owner:
            continue
        replaced_storage_ids: set[int] = set()
        for block in function.blocks.values():
            for instruction in block.instrs:
                owner_name = instruction.attributes.get("keyed_lookup_owner")
                if owner_name is None or len(instruction.args) < 6:
                    continue
                parts = parts_by_owner.get(str(owner_name))
                if parts is None or any(
                    name not in parts
                    for name in ("length", "keys", "values")
                ):
                    continue
                replaced_storage_ids.update(
                    int(argument.id) for argument in instruction.args[:4]
                )
                instruction.args[0] = parts["keys"]
                instruction.args[1] = parts["values"]
                instruction.args[2] = parts["length"]
                instruction.args[3] = parts["length"]
                helper = all_functions.get(
                    str(instruction.attributes.get("callee") or "")
                )
                if helper is not None:
                    typed: dict[int, str] = {}
                    for argument, (dtype, shape) in zip(
                        helper.args, _keyed_helper_dtypes
                    ):
                        if argument.dtype in {None, "unknown", "None"}:
                            argument.dtype = dtype
                        if shape is not None and not tuple(
                            argument.shape or ()
                        ):
                            argument.shape = shape
                        typed[int(argument.id)] = str(argument.dtype)
                    # The body holds its own SSAValue instances for the same
                    # ids; retype them too, and give each Load the element
                    # type of the span it reads.
                    span_element = {
                        int(helper.args[0].id): "int64",
                        int(helper.args[1].id): "float64",
                    }
                    address_element: dict[int, str] = {}
                    for helper_block in helper.blocks.values():
                        for helper_instruction in helper_block.instrs:
                            for value in (
                                *helper_instruction.args,
                                *((helper_instruction.res,)
                                  if helper_instruction.res is not None
                                  else ()),
                            ):
                                refined = typed.get(int(value.id))
                                if refined is not None and value.dtype in {
                                    None, "unknown", "None",
                                }:
                                    value.dtype = refined
                            if (
                                helper_instruction.op == "GetElementPtr"
                                and helper_instruction.res is not None
                                and helper_instruction.args
                            ):
                                element = span_element.get(
                                    int(helper_instruction.args[0].id)
                                )
                                if element is not None:
                                    address_element[
                                        int(helper_instruction.res.id)
                                    ] = element
                            if (
                                helper_instruction.op == "Load"
                                and helper_instruction.res is not None
                                and helper_instruction.args
                                and helper_instruction.res.dtype in {
                                    None, "unknown", "None",
                                }
                            ):
                                element = address_element.get(
                                    int(helper_instruction.args[0].id)
                                )
                                if element is not None:
                                    helper_instruction.res.dtype = element
        if not replaced_storage_ids:
            continue
        still_consumed = {
            int(argument.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        }
        dropped_positions = [
            position
            for position, value in enumerate(function.args)
            if int(value.id) in replaced_storage_ids
            and int(value.id) not in still_consumed
        ]
        if not dropped_positions:
            continue
        original_arity = len(function.args)
        function.args = [
            value
            for position, value in enumerate(function.args)
            if position not in set(dropped_positions)
        ]
        # A formal exists only together with the operand every caller feeds
        # it.  Dropping the formal alone leaves each call site one operand
        # too long, and the public-span origin walk skips calls whose arity
        # disagrees -- silently severing every span reached through this
        # function for every caller above it.
        function_symbol = next(
            (
                candidate_symbol
                for candidate_symbol, candidate in all_functions.items()
                if candidate is function
            ),
            None,
        )
        if function_symbol is None:
            continue
        for caller in all_functions.values():
            for block in caller.blocks.values():
                for instruction in block.instrs:
                    if (
                        instruction.op != "Call"
                        or str(
                            instruction.attributes.get("callee") or ""
                        ) != function_symbol
                        or len(instruction.args) != original_arity
                    ):
                        continue
                    instruction.args = [
                        argument
                        for position, argument in enumerate(instruction.args)
                        if position not in set(dropped_positions)
                    ]

    # A declared record field keeps its storage identity across the call frame.
    # The contract states `height` as a rank-2 span, but a callee's formal
    # parameter was built before that contract was materialized, so it arrived
    # as an untyped scalar and every address into it became unresolvable. The
    # binding the caller already computed is the exact carrier: walk each call's
    # argument positions and give the callee's parameter the same field
    # identity. Nothing is inferred from names, and no field is invented -- an
    # argument only inherits what its own caller was already declared to hold.
    for caller in all_functions.values():
        for block in caller.blocks.values():
            for instruction in block.instrs:
                if instruction.op != "Call":
                    continue
                callee = all_functions.get(
                    str(instruction.attributes.get("callee") or "")
                )
                if callee is None or len(callee.args) != len(instruction.args):
                    continue
                for fed, formal in zip(instruction.args, callee.args):
                    accounting = dict(fed.accounting or {})
                    if not accounting.get("program_abi_storage"):
                        continue
                    if (formal.accounting or {}).get("program_abi_storage"):
                        continue
                    # A keyed mapping's slot ids name values in the caller's
                    # own frame. The callee materializes its own slots from the
                    # same contract, so carrying these across would point at
                    # whatever happens to hold those ids there.
                    for frame_local in (
                        "program_abi_keyed_length",
                        "program_abi_keyed_keys",
                        "program_abi_keyed_values",
                    ):
                        accounting.pop(frame_local, None)
                    formal.accounting = {
                        **dict(formal.accounting or {}), **accounting,
                    }
                    if formal.dtype in {None, "unknown"} and fed.dtype:
                        formal.dtype = fed.dtype
                    # The declared rank travels in the field identity, not in
                    # `shape`. Only the rank is known here -- the extents are
                    # measured from the real buffer at call time -- and `shape`
                    # is the repository's *static* element-count contract, so
                    # naming symbolic axes there would corrupt every buffer
                    # size and block copy derived from it.

    # A keyed mapping's slot ids name values in one frame. Several passes copy
    # field accounting between frames, so verify the correlation still resolves
    # where it is stated and drop it where it does not. A mapping that names no
    # slots is simply unresolved here -- honest, and refusable by a backend --
    # whereas one naming ids this frame never defined would silently address
    # whatever else happens to hold them.
    for function in all_functions.values():
        frame_values = {int(value.id) for value in function.args}
        frame_values.update(
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        )
        for value in function.args:
            accounting = dict(value.accounting or {})
            if accounting.get("program_abi_storage") != "keyed":
                continue
            slots = [
                accounting.get("program_abi_keyed_length"),
                accounting.get("program_abi_keyed_keys"),
                accounting.get("program_abi_keyed_values"),
            ]
            if all(
                slot is not None and int(slot) in frame_values
                for slot in slots
            ):
                continue
            for frame_local in (
                "program_abi_keyed_length",
                "program_abi_keyed_keys",
                "program_abi_keyed_values",
            ):
                accounting.pop(frame_local, None)
            value.accounting = accounting

    # Literal construction is pure. Source ingestion intentionally retains
    # strings, empty tuples, debug labels, and optional markers long enough
    # for structural planning. After call frames and public returns are fixed,
    # an unconsumed literal is dead program metadata. Remove it once here so
    # all backends receive the same instruction stream instead of inventing
    # four different nonnumeric-constant policies.
    for function in all_functions.values():
        consumed_ids = {
            int(argument.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        }
        for block in function.blocks.values():
            block.instrs = [
                instruction
                for instruction in block.instrs
                if not (
                    instruction.op in {"Const", "const"}
                    and not instruction.args
                    and "value" in instruction.attributes
                    and instruction.res is not None
                    and int(instruction.res.id) not in consumed_ids
                )
            ]

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
            reference_tables=all_reference_tables,
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


def lower_class_surface_to_ssa(
    compilation: Any,
    artifact_name: str,
    *,
    tensor_ssa_reference: Any = None,
):
    """Public target-neutral entry to the repository whole-object lowering.

    The implementation historically lived beside the Fortran shell because
    that was its first consumer.  Its product is nevertheless an ordinary
    :class:`IRModule`: class definitions, record/sequence/reference tables,
    method functions, and explicit call records.  Generic compiler and
    visualization paths use this entry rather than discarding that object
    geometry at the numerical-precompile boundary.
    """

    return _class_surface_ssa_program(
        compilation,
        artifact_name,
        tensor_ssa_reference=tensor_ssa_reference,
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
    entrypoint: str | None = None,
    *,
    python_bindings: Mapping[str, Any] | None = None,
    dependency_seeds: tuple[str, ...] = (),
    retain: Any = (),
    tensor_code_references: Mapping[str, Callable[..., Any]] | None = None,
    tensor_ssa_reference: Any = None,
    name: str | None = None,
    runtime_closure_only: bool = True,
    progress: Callable[[str], None] | None = None,
    boundary_namespace: Any = None,
    source_language: str = "python",
    extraction_contract: Any = None,
    linked_process_graphs: Mapping[str, Any] | None = None,
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
    extraction_policy = extraction_contract
    if extraction_policy is not None:
        from .extraction_contract import ExtractionContract
        if isinstance(extraction_policy, (str, os.PathLike)):
            extraction_policy = ExtractionContract(extraction_policy)
        elif not hasattr(extraction_policy, "decide"):
            raise TypeError(
                "extraction_contract must be a path or ExtractionContract"
            )
    # No selected root is the canonical whole-source mode.  It deliberately
    # disables runtime-closure pruning so module statements, every authored
    # definition, and their configured dependency domains remain eligible.
    compile_targets = (
        () if entrypoint is None else (str(entrypoint), *map(str, dependency_seeds))
    )
    whole_source = entrypoint is None
    graph = ProcessGraph(
        materialize_memory=False,
        boundary_namespace=boundary_namespace,
        source_language=source_language,
    )
    linked_process_graphs = {
        str(function_name): function_graph
        for function_name, function_graph in dict(
            linked_process_graphs or {}
        ).items()
    }
    if linked_process_graphs:
        from .process_graph_function_linking import link_process_graph_functions

        report("ssa-source: registering authored ProcessGraph functions")
        link_process_graph_functions(graph, linked_process_graphs)
    graph.python_bindings = dict(python_bindings or {})
    report("ssa-source: building complete ProcessGraph source closure")
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            tree,
            resolve_unresolved_parents=True,
            parent_include=(
                extraction_policy
                if extraction_policy is not None
                else _source_dependency_is_not_tensor_primitive
            ),
            pursuit_roots=(
                tuple(dict.fromkeys(compile_targets))
                if runtime_closure_only and not whole_source else None
            ),
            tensor_code_references=dict(tensor_code_references or {}),
            retain=retain,
            progress=report,
        )
    if extraction_policy is not None and (
        extraction_policy.program_abi.records
        or extraction_policy.program_abi.values
    ):
        # Type the physical Python/native boundary before topology reduction.
        # This is declarative ABI information only: it does not instantiate a
        # Python object, infer a convenient shape, or authorize new source
        # pursuit. Every pursued function receives only the record bindings
        # whose function/parameter rules explicitly match the contract.
        for entry in graph.function_table:
            function_graph = getattr(getattr(entry, "graph", None), "G", None)
            if function_graph is None:
                continue
            function_name = str(
                function_graph.graph.get("function_name") or entry.name
            )
            records = extraction_policy.program_abi.records_for_function(
                function_name,
                method_owner=function_graph.graph.get("method_owner"),
                parameters=function_graph.graph.get("function_parameters") or (),
            )
            parameters = set(map(
                str, function_graph.graph.get("function_parameters") or ()
            ))
            selected = {
                parameter: record.receipt()
                for parameter, record in records.items()
                if parameter in parameters
            }
            if selected:
                function_graph.graph["parameter_record_abi"] = selected
            values = extraction_policy.program_abi.values_for_function(
                function_name
            )
            selected_values = {
                parameter: binding.receipt()
                for parameter, binding in values.items()
                if parameter in parameters
            }
            if selected_values:
                function_graph.graph["parameter_value_abi"] = selected_values
        graph.G.graph["program_abi"] = extraction_policy.program_abi.receipt()
    graph.G.graph["compile_targets"] = tuple(dict.fromkeys(compile_targets))
    report("ssa-source: reducing source topology")
    reduce_abstract_tensor_topology(graph)
    if extraction_policy is not None and (
        extraction_policy.program_abi.records
        or extraction_policy.program_abi.values
    ):
        # Reduction extracts fresh per-function graphs from the complete
        # source graph. Reattach the declarative ABI to those canonical
        # graphs before structural specialization and hierarchy planning;
        # attaching it only to the pre-reduction discovery graphs leaves
        # method receivers untyped during exactly the pass that decides
        # schema guards and optional-field branches.
        for entry in graph.function_table:
            function_graph = getattr(getattr(entry, "graph", None), "G", None)
            if function_graph is None:
                continue
            function_name = str(
                function_graph.graph.get("function_name") or entry.name
            )
            parameters = set(map(
                str, function_graph.graph.get("function_parameters") or ()
            ))
            records = extraction_policy.program_abi.records_for_function(
                function_name,
                method_owner=function_graph.graph.get("method_owner"),
                parameters=parameters,
            )
            selected = {
                parameter: record.receipt()
                for parameter, record in records.items()
                if parameter in parameters
            }
            if selected:
                function_graph.graph["parameter_record_abi"] = selected
            values = extraction_policy.program_abi.values_for_function(
                function_name
            )
            selected_values = {
                parameter: binding.receipt()
                for parameter, binding in values.items()
                if parameter in parameters
            }
            if selected_values:
                function_graph.graph["parameter_value_abi"] = selected_values
    if linked_process_graphs:
        # Reduction may rewrite call nodes and dependency graphs. Reapply the
        # idempotent function-table link before planning so the direct SSA path
        # never substitutes Python capture or a FusedProgram for the authored
        # cross-language function.
        from .process_graph_function_linking import link_process_graph_functions

        report("ssa-source: resolving authored ProcessGraph calls")
        link_process_graph_functions(graph, linked_process_graphs)
    report("ssa-source: planning complete control/operator graph")
    deployment_type = strategize_shell_deployment(
        graph,
        backend="fortran",
        runtime_closure_only=(runtime_closure_only and not whole_source),
    )
    deployment = deployment_type(profiling=False, shell_language="glsl")
    deployment.prepare_complete_catalogue = whole_source
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
        _identifier(str(name or entrypoint or "whole_source")),
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

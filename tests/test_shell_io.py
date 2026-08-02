import pytest

from src.compiler.shell_io import (
    NATIVE_PROCESS_SHELL,
    WEB_JAVASCRIPT_SHELL,
    ShellIOABI,
    ShellIOBinding,
    ShellIOCapability,
    ShellIOManifest,
    ShellIORequest,
    ShellOption,
    attach_shell_io,
    plan_shell_stack,
)
from src.compiler.compiled_program_api import (
    CompiledProgramAPI, EntryPoint, Parameter,
)


def _interactive_manifest():
    return ShellIOManifest((
        ShellIORequest.create("keyboard"),
        ShellIORequest.create("pointer"),
        ShellIORequest.create("files"),
        ShellIORequest.create("display_double_buffer", optional=True),
    ))


def test_wasm_is_wrapped_by_web_javascript_for_requested_io():
    stack = plan_shell_stack("wasm", _interactive_manifest(), (
        WEB_JAVASCRIPT_SHELL,
    ))

    assert [wrapper.name for wrapper in stack.wrappers] == ["web_javascript"]
    assert stack.outer_kind == "web_page"
    assert ShellIOCapability.FILES in stack.provided
    assert stack.optional_available == {ShellIOCapability.DISPLAY}


def test_fortran_is_wrapped_without_teaching_fortran_about_io():
    stack = plan_shell_stack("fortran", _interactive_manifest(), (
        NATIVE_PROCESS_SHELL,
    ))

    assert [wrapper.name for wrapper in stack.wrappers] == ["native_process"]
    assert stack.outer_kind == "native_process"


def test_missing_required_shell_io_is_an_explicit_failure():
    with pytest.raises(ValueError, match="no shell stack"):
        plan_shell_stack("wasm", _interactive_manifest(), ())


def test_shared_abi_has_event_file_and_optional_double_buffer_mailboxes():
    mapping = ShellIOABI().to_mapping()

    assert mapping["input_events"]["record_bytes"] == 32
    assert mapping["file_requests"]["header_fields"][1:3] == [
        "read_index", "write_index",
    ]
    assert "read" in mapping["files"]["operations"]
    assert mapping["display"]["optional"] is True
    assert mapping["records"]["input_event_i32"][:3] == [
        "kind", "code", "value",
    ]


def test_shell_io_travels_in_the_existing_compiled_api_descriptor():
    api = CompiledProgramAPI("demo", "wasm", "run", metadata={"kept": 1})

    attached = attach_shell_io(api, _interactive_manifest())

    assert api.metadata == {"kept": 1}
    assert attached.metadata["kept"] == 1
    shell_io = attached.to_mapping()["metadata"]["shell_io"]
    assert shell_io["requirements"]["schema"] == (
        "turing-shell-io-requirements"
    )
    assert shell_io["abi"]["schema"] == "turing-shell-io-abi"


def test_manifest_serializes_parameter_bindings_and_generated_options():
    manifest = ShellIOManifest(
        (ShellIORequest.create("display_double_buffer"),),
        bindings=(ShellIOBinding("display.back", "frame", "t9"),),
        options=(ShellOption("width", "int", 640, "display width"),),
    )

    mapping = manifest.to_mapping()

    assert mapping["bindings"] == [{
        "resource": "display.back", "entry_point": "frame", "parameter": "t9",
    }]
    assert mapping["options"] == [{
        "name": "width", "type": "int", "default": 640,
        "help": "display width",
    }]

    specialized = manifest.specialize_options({"width": 320})
    assert specialized.to_mapping()["options"][0]["default"] == 320
    assert manifest.to_mapping()["options"][0]["default"] == 640


def test_attaching_io_resolves_source_name_to_fortran_abi_parameter():
    output = Parameter(
        "t77", "output", "float32", "float", "c_float", "reference",
        shape=(16,), source_name="pixels",
    )
    api = CompiledProgramAPI(
        "demo", "fortran", "frame",
        (EntryPoint("frame", "frame", "control", (output,)),),
    )
    manifest = ShellIOManifest(
        (ShellIORequest.create("display_double_buffer"),),
        bindings=(ShellIOBinding("display.back", "frame", "pixels"),),
    )

    attached = attach_shell_io(api, manifest).to_mapping()

    assert attached["metadata"]["shell_io"]["requirements"]["bindings"] == [{
        "resource": "display.back", "entry_point": "frame", "parameter": "t77",
    }]

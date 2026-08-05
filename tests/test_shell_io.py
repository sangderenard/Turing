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
    ExternalReferenceDomain,
    SystemPort,
    VirtualFileSystemContract,
    VirtualMount,
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
    assert mapping["external_references"]["web_domain"] == "bundle"
    assert mapping["external_references"]["operations"] == [
        "resolve", "call", "release",
    ]
    assert mapping["records"]["external_request_i32"][2] == "reference_id"


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


def test_file_parameter_port_resolves_data_and_length_through_compiled_api():
    parameters = (
        Parameter("t4", "input", "u8", "uint8_t", "c_uint8", "reference", source_name="binary_bytes"),
        Parameter("t5", "input", "i64", "int64_t", "c_int64", "value", source_name="binary_length"),
    )
    api = CompiledProgramAPI(
        "machine", "wasm", "load_subject",
        (EntryPoint("load_subject", "load_subject", "control", parameters),),
    )
    manifest = ShellIOManifest(
        (ShellIORequest.create("files"),),
        system_ports=(SystemPort.create(
            "subject", "file", "input", entry_point="load_subject",
            fields={"data": "binary_bytes", "length": "binary_length"},
            attributes={"accept": ".exe,.dll,application/octet-stream"},
        ),),
    )

    attached = attach_shell_io(api, manifest).to_mapping()
    port = attached["metadata"]["shell_io"]["requirements"]["system_ports"][0]

    assert port["kind"] == "file"
    assert port["fields"] == [
        {"name": "data", "parameter": "t4"},
        {"name": "length", "parameter": "t5"},
    ]


def test_web_bundle_references_are_distinct_from_native_host_references():
    web_manifest = ShellIOManifest(
        (ShellIORequest.create("bundle_references"),),
        system_ports=(SystemPort.create(
            "decoder", "external_reference", "call",
            external_domain=ExternalReferenceDomain.BUNDLE,
            attributes={"bundle": "machine-decoder", "export": "decode"},
        ),),
    )
    assert plan_shell_stack("wasm", web_manifest, (WEB_JAVASCRIPT_SHELL,)).outer_kind == "web_page"

    host_manifest = ShellIOManifest(
        (ShellIORequest.create("host_references"),),
        system_ports=(SystemPort.create(
            "kernel32", "external_reference", "call",
            external_domain=ExternalReferenceDomain.HOST_SYSTEM,
            attributes={"library": "kernel32", "symbol": "ReadFile"},
        ),),
    )
    with pytest.raises(ValueError, match="no shell stack"):
        plan_shell_stack("wasm", host_manifest, (WEB_JAVASCRIPT_SHELL,))
    assert plan_shell_stack("fortran", host_manifest, (NATIVE_PROCESS_SHELL,)).outer_kind == "native_process"


def test_virtual_filesystem_mounts_are_shell_specific_and_serialized():
    web = ShellIOManifest(
        (ShellIORequest.create("files"),),
        virtual_filesystem=VirtualFileSystemContract(mounts=(
            VirtualMount.create("/", "memory", access="read_write"),
            VirtualMount.create("/programs", "bundle", source="program-bundle"),
        )),
    )
    stack = plan_shell_stack("wasm", web, (WEB_JAVASCRIPT_SHELL,))
    assert stack.outer_kind == "web_page"
    mapping = web.to_mapping()["virtual_filesystem"]
    assert mapping["current_directory"] == "/"
    assert mapping["mounts"][1]["kind"] == "bundle"

    native_only = ShellIOManifest(
        (ShellIORequest.create("files"),),
        virtual_filesystem=VirtualFileSystemContract(mounts=(
            VirtualMount.create("/", "memory", access="read_write"),
            VirtualMount.create("/host", "host_directory", source="C:\\sandbox"),
        )),
    )
    with pytest.raises(ValueError, match="no shell stack"):
        plan_shell_stack("wasm", native_only, (WEB_JAVASCRIPT_SHELL,))
    assert plan_shell_stack(
        "fortran", native_only, (NATIVE_PROCESS_SHELL,),
    ).outer_kind == "native_process"


def test_file_broker_declares_namespace_and_journal_operations():
    files = ShellIOABI().to_mapping()["files"]
    assert files["namespace"] == "utf8-posix-absolute"
    assert files["effects"] == "ordered-journal"
    assert {"list", "rename", "chdir", "flush"} <= set(files["operations"])

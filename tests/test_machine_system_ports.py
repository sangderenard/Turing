from src.compiler.machine_execution import (
    MachineExecutionState,
    MachineExternalCallRequest,
    MachineExternalReference,
)
from src.compiler.machine_system_ports import deterministic_windows_bootstrap_port
from src.compiler.amd64_machine_semantics import PagedByteMemory


def _request(library: str, symbol: str, *, pointer: int = 0x7000):
    return MachineExternalCallRequest(
        request_id=5,
        reference=MachineExternalReference(1, 0xFFFF800000000000, "guest-binary", library, symbol),
        instruction_address=0x1000,
        return_address=0x1002,
        arguments=(pointer, 0, 0, 0),
        stack_pointer=0x7FF8,
    )


def test_bootstrap_port_is_exact_allowlist_and_returns_deterministic_effects():
    port = deterministic_windows_bootstrap_port(file_time=123, process_id=77)
    state = MachineExecutionState(pc=0)

    time = port.handle(
        _request("API-MS-WIN-CORE-SYSINFO-L1-1-0.DLL", "GetSystemTimeAsFileTime"),
        state,
    )
    process = port.handle(
        _request("api-ms-win-core-processthreads-l1-1-0.dll", "GetCurrentProcessId"),
        state,
    )

    assert time is not None
    assert time.memory_writes[0].data == (123).to_bytes(8, "little")
    assert process is not None and process.result == 77
    module = port.handle(
        _request("api-ms-win-core-libraryloader-l1-2-0.dll", "GetModuleHandleW", pointer=0),
        state,
    )
    assert module is not None and module.result == 0x140000000
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000).map_bytes(
        0x7000, "KERNEL32.DLL\0".encode("utf-16le"),
    )
    named_module = port.handle(
        _request("api-ms-win-core-libraryloader-l1-2-0.dll", "GetModuleHandleW"),
        MachineExecutionState(pc=0, memory=memory),
    )
    assert named_module is not None and named_module.result == 0xFFFF900000000001
    memory = memory.map_zeroes(0x140000000, 0x1000)
    memory = memory.map_bytes(0x140000100, b"SetThreadUILanguage\x00")
    proc_request = MachineExternalCallRequest(
        14,
        MachineExternalReference(
            8, 0, "guest-binary",
            "api-ms-win-core-libraryloader-l1-2-0.dll", "GetProcAddress",
        ),
        0, 0, (0xFFFF900000000001, 0x140000100, 0, 0), 0,
    )
    proc = port.handle(proc_request, MachineExecutionState(pc=0, memory=memory))
    assert proc is not None and proc.resolution is not None
    assert (proc.resolution.library, proc.resolution.symbol) == (
        "kernel32.dll", "SetThreadUILanguage",
    )
    assert port.handle(_request("kernel32.dll", "CreateProcessW"), state) is None


def test_virtual_system_state_effects_are_explicit_in_completion():
    port = deterministic_windows_bootstrap_port()
    request = _request(
        "api-ms-win-core-errorhandling-l1-1-0.dll",
        "SetUnhandledExceptionFilter",
        pointer=0x1234,
    )
    completion = port.handle(request, MachineExecutionState(pc=0))
    assert completion is not None
    assert completion.result == 0
    assert completion.system_writes[0].key == "windows.unhandled_exception_filter"
    assert completion.system_writes[0].value == 0x1234


def test_initterm_becomes_bounded_ordered_guest_dispatch_plan():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    memory = memory.write_unsigned(0x7000, 64, 0x3000)
    memory = memory.write_unsigned(0x7008, 64, 0)
    memory = memory.write_unsigned(0x7010, 64, 0x5000)
    request = MachineExternalCallRequest(
        request_id=8,
        reference=MachineExternalReference(2, 0, "guest-binary", "msvcrt.dll", "_initterm"),
        instruction_address=0,
        return_address=0,
        arguments=(0x7000, 0x7018, 0, 0),
        stack_pointer=0,
    )

    completion = deterministic_windows_bootstrap_port().handle(
        request, MachineExecutionState(pc=0, memory=memory),
    )

    assert completion is not None
    assert completion.guest_calls == (0x3000, 0x5000)


def test_getmainargs_builds_shell_arguments_in_preallocated_guest_arena():
    memory = PagedByteMemory.empty().map_zeroes(0x6000, 0x3000)
    state = MachineExecutionState(
        pc=0,
        memory=memory,
        system_state={
            "windows.system_arena_cursor": 0x6000,
            "windows.system_arena_limit": 0x8000,
        },
    )
    request = MachineExternalCallRequest(
        request_id=9,
        reference=MachineExternalReference(3, 0, "guest-binary", "msvcrt.dll", "__getmainargs"),
        instruction_address=0,
        return_address=0,
        arguments=(0x8000, 0x8008, 0x8010, 0),
        stack_pointer=0,
    )
    completion = deterministic_windows_bootstrap_port(
        arguments=("cmd.exe", "/c", "echo hello"),
        environment=("A=B",),
    ).handle(request, state)

    assert completion is not None
    effects = {effect.address: effect.data for effect in completion.memory_writes}
    assert effects[0x8000] == (3).to_bytes(4, "little")
    argv_address = int.from_bytes(effects[0x8008], "little")
    first_argument = int.from_bytes(effects[argv_address][:8], "little")
    assert effects[first_argument] == b"cmd.exe\x00"


def test_onexit_registration_is_reversible_system_state_not_host_mutation():
    port = deterministic_windows_bootstrap_port()
    request = MachineExternalCallRequest(
        request_id=10,
        reference=MachineExternalReference(4, 0, "guest-binary", "msvcrt.dll", "_onexit"),
        instruction_address=0,
        return_address=0,
        arguments=(0x140001000, 0, 0, 0),
        stack_pointer=0,
    )
    completion = port.handle(request, MachineExecutionState(pc=0))

    assert completion is not None and completion.result == 0x140001000
    assert [(item.key, item.value) for item in completion.system_writes] == [
        ("windows.onexit.0", 0x140001000),
        ("windows.onexit.count", 1),
    ]


def test_crt_memory_capabilities_return_explicit_bounded_effects():
    port = deterministic_windows_bootstrap_port()
    memory = PagedByteMemory.empty().map_zeroes(0x6000, 0x2000)
    memory = memory.map_bytes(0x6100, b"source")
    state = MachineExecutionState(pc=0, memory=memory)
    copy_request = MachineExternalCallRequest(
        11, MachineExternalReference(5, 0, "guest-binary", "msvcrt.dll", "memcpy"),
        0, 0, (0x6200, 0x6100, 6, 0), 0,
    )
    fill_request = MachineExternalCallRequest(
        12, MachineExternalReference(6, 0, "guest-binary", "msvcrt.dll", "memset"),
        0, 0, (0x6300, 0xAB, 4, 0), 0,
    )

    copied = port.handle(copy_request, state)
    filled = port.handle(fill_request, state)

    assert copied is not None and copied.memory_writes[0].data == b"source"
    assert filled is not None and filled.memory_writes[0].data == b"\xAB" * 4


def test_open_thread_allocates_only_virtual_handle_state():
    port = deterministic_windows_bootstrap_port(thread_id=7)
    request = MachineExternalCallRequest(
        13,
        MachineExternalReference(
            7, 0, "guest-binary",
            "api-ms-win-core-processthreads-l1-1-0.dll", "OpenThread",
        ),
        0, 0, (0x1FFFFF, 0, 7, 0), 0,
    )
    completion = port.handle(request, MachineExecutionState(pc=0))

    assert completion is not None and completion.result == 0x100
    effects = {item.key: item.value for item in completion.system_writes}
    assert effects["windows.handle.256.kind"] == 1
    assert effects["windows.handle.256.id"] == 7


def test_dynamically_resolved_kernel_export_uses_same_capability_gate():
    port = deterministic_windows_bootstrap_port(ui_language=0x0409)
    request = MachineExternalCallRequest(
        15,
        MachineExternalReference(9, 0, "guest-binary", "kernel32.dll", "SetThreadUILanguage"),
        0, 0, (0, 0, 0, 0), 0,
    )
    completion = port.handle(request, MachineExecutionState(pc=0))
    assert completion is not None and completion.result == 0x0409
    assert completion.system_writes[0].key == "windows.thread_ui_language"


def test_heap_policy_changes_virtual_state_without_host_allocator_access():
    port = deterministic_windows_bootstrap_port()
    request = MachineExternalCallRequest(
        16,
        MachineExternalReference(
            10, 0, "guest-binary", "api-ms-win-core-heap-l1-1-0.dll",
            "HeapSetInformation",
        ),
        0, 0, (0, 1, 0, 0), 0,
    )
    completion = port.handle(request, MachineExecutionState(pc=0))
    assert completion is not None and completion.result == 1
    assert completion.system_writes[0].key == "windows.heap.0.information_class.1"


def test_process_heap_allocations_use_preallocated_reversible_arena():
    port = deterministic_windows_bootstrap_port()
    request = MachineExternalCallRequest(
        26, MachineExternalReference(
            20, 0, "guest-binary", "api-ms-win-core-heap-l1-1-0.dll",
            "HeapAlloc",
        ), 0, 0, (0x300, 0x8, 32, 0), 0,
    )
    state = MachineExecutionState(pc=0, system_state={
        "windows.system_arena_cursor": 0x7003,
        "windows.system_arena_limit": 0x8000,
    })
    completion = port.handle(request, state)
    assert completion is not None and completion.result == 0x7010
    assert completion.memory_writes[0].data == bytes(32)
    effects = {item.key: item.value for item in completion.system_writes}
    assert effects["windows.heap.allocation.28688.size"] == 32
    assert effects["windows.heap.allocation.28688.active"] == 1


def test_crt_time_uses_virtual_file_time_epoch():
    unix_epoch_file_time = 116_444_736_000_000_000
    port = deterministic_windows_bootstrap_port(
        file_time=unix_epoch_file_time + 42 * 10_000_000,
    )
    request = MachineExternalCallRequest(
        27, MachineExternalReference(
            21, 0, "guest-binary", "msvcrt.dll", "time",
        ), 0, 0, (0x7000, 0, 0, 0), 0,
    )
    completion = port.handle(request, MachineExecutionState(pc=0))
    assert completion is not None and completion.result == 42
    assert completion.memory_writes[0].data == (42).to_bytes(8, "little")


def test_registry_policy_is_empty_and_never_reads_host_registry():
    port = deterministic_windows_bootstrap_port()
    request = MachineExternalCallRequest(
        17,
        MachineExternalReference(
            11, 0, "guest-binary", "api-ms-win-core-registry-l1-1-0.dll",
            "RegOpenKeyExW",
        ),
        0, 0,
        (0xFFFFFFFF80000001, 0x7000, 0, 0x20019),
        0,
        stack_arguments=(0x7100,),
    )
    completion = port.handle(request, MachineExecutionState(pc=0))
    assert completion is not None and completion.result == 2
    assert completion.memory_writes == ()


def test_console_encoding_is_explicit_virtual_configuration():
    port = deterministic_windows_bootstrap_port(output_code_page=65001)
    request = MachineExternalCallRequest(
        18,
        MachineExternalReference(
            12, 0, "guest-binary", "api-ms-win-core-console-l1-1-0.dll",
            "GetConsoleOutputCP",
        ),
        0, 0, (0, 0, 0, 0), 0,
    )
    completion = port.handle(request, MachineExecutionState(pc=0))
    assert completion is not None and completion.result == 65001


def test_utf8_code_page_info_is_written_to_guest_structure():
    port = deterministic_windows_bootstrap_port()
    request = MachineExternalCallRequest(
        19,
        MachineExternalReference(
            13, 0, "guest-binary", "api-ms-win-core-localization-l1-2-0.dll",
            "GetCPInfo",
        ),
        0, 0, (65001, 0x7000, 0, 0), 0,
    )
    completion = port.handle(request, MachineExecutionState(pc=0))
    assert completion is not None and completion.result == 1
    assert len(completion.memory_writes[0].data) == 20
    assert completion.memory_writes[0].data[:4] == (4).to_bytes(4, "little")


def test_critical_section_state_is_virtual_and_recursion_aware():
    port = deterministic_windows_bootstrap_port(thread_id=3)
    init = MachineExternalCallRequest(
        20, MachineExternalReference(
            14, 0, "guest-binary", "api-ms-win-core-synch-l1-1-0.dll",
            "InitializeCriticalSection",
        ), 0, 0, (0x7000, 0, 0, 0), 0,
    )
    initialized = port.handle(init, MachineExecutionState(pc=0))
    assert initialized is not None and len(initialized.memory_writes[0].data) == 40
    state_values = {item.key: item.value for item in initialized.system_writes}
    enter = MachineExternalCallRequest(
        21, MachineExternalReference(
            15, 0, "guest-binary", "api-ms-win-core-synch-l1-1-0.dll",
            "EnterCriticalSection",
        ), 0, 0, (0x7000, 0, 0, 0), 0,
    )
    acquired = port.handle(enter, MachineExecutionState(pc=0, system_state=state_values))
    assert acquired is not None
    effects = {item.key: item.value for item in acquired.system_writes}
    assert effects["windows.critical_section.28672.owner"] == 3
    assert effects["windows.critical_section.28672.recursion"] == 1


def test_console_control_handler_is_registered_in_reversible_state():
    port = deterministic_windows_bootstrap_port()
    request = MachineExternalCallRequest(
        22, MachineExternalReference(
            16, 0, "guest-binary", "api-ms-win-core-console-l1-1-0.dll",
            "SetConsoleCtrlHandler",
        ), 0, 0, (0x140001000, 1, 0, 0), 0,
    )
    completion = port.handle(request, MachineExecutionState(pc=0))
    assert completion is not None and completion.result == 1
    assert completion.system_writes[0].key == (
        "windows.console_control_handler.5368713216"
    )


def test_crt_standard_descriptors_map_to_virtual_console_handles():
    port = deterministic_windows_bootstrap_port()
    request = MachineExternalCallRequest(
        23, MachineExternalReference(
            17, 0, "guest-binary", "msvcrt.dll", "_get_osfhandle",
        ), 0, 0, (1, 0, 0, 0), 0,
    )
    completion = port.handle(request, MachineExecutionState(pc=0))
    assert completion is not None and completion.result == 0x201


def test_console_mode_is_guest_visible_and_handle_scoped():
    port = deterministic_windows_bootstrap_port()
    request = MachineExternalCallRequest(
        24, MachineExternalReference(
            18, 0, "guest-binary", "api-ms-win-core-console-l1-1-0.dll",
            "GetConsoleMode",
        ), 0, 0, (0x201, 0x7000, 0, 0), 0,
    )
    completion = port.handle(request, MachineExecutionState(pc=0))
    assert completion is not None and completion.result == 1
    assert completion.memory_writes[0].data == (3).to_bytes(4, "little")


def test_environment_block_comes_only_from_declared_virtual_environment():
    port = deterministic_windows_bootstrap_port(environment=("A=B", "C=D"))
    request = MachineExternalCallRequest(
        25, MachineExternalReference(
            19, 0, "guest-binary",
            "api-ms-win-core-processenvironment-l1-1-0.dll",
            "GetEnvironmentStringsW",
        ), 0, 0, (0, 0, 0, 0), 0,
    )
    state = MachineExecutionState(pc=0, system_state={
        "windows.system_arena_cursor": 0x7000,
        "windows.system_arena_limit": 0x8000,
    })
    completion = port.handle(request, state)
    assert completion is not None and completion.result == 0x7000
    assert completion.memory_writes[0].data.decode("utf-16le") == "A=B\0C=D\0\0"

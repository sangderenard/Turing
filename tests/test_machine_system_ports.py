from dataclasses import replace
import struct

from src.compiler.machine_execution import (
    MachineExecutionState,
    MachineExternalCallRequest,
    MachineExternalDeviceWrite,
    MachineExternalReference,
    MachineExternalStateWrite,
)
from src.compiler.machine_system_ports import deterministic_windows_bootstrap_port
from src.compiler.amd64_machine_semantics import PagedByteMemory
from src.compiler.shell_io import VirtualFileSystemContract, VirtualMount
from src.compiler.virtual_filesystem import VirtualFileSystemState
from src.compiler.virtual_process import VirtualProgramRegistry, VirtualProgramResult


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
    relocated_module = port.handle(
        _request(
            "api-ms-win-core-libraryloader-l1-2-0.dll",
            "GetModuleHandleW",
            pointer=0,
        ),
        MachineExecutionState(
            pc=0,
            system_state={"windows.loader.image_base": 0x150000000},
        ),
    )
    assert relocated_module is not None
    assert relocated_module.result == 0x150000000

    assert port.supports(_request(
        "api-ms-win-core-processthreads-l1-1-0.dll", "GetCurrentProcessId",
    ))
    assert not port.supports(_request("kernel32.dll", "DefinitelyNotAllowed"))
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


def test_error_modes_and_thread_last_error_are_reversible_state():
    port = deterministic_windows_bootstrap_port(thread_id=7)
    state = MachineExecutionState(pc=0, system_state={
        "windows.error_mode": 1,
        "windows.thread.7.last_error": 5,
    })
    mode = port.handle(_request(
        "api-ms-win-core-errorhandling-l1-1-0.dll", "SetErrorMode", pointer=3,
    ), state)
    last = port.handle(_request(
        "api-ms-win-core-errorhandling-l1-1-0.dll", "GetLastError",
    ), state)
    assert mode is not None and mode.result == 1
    assert mode.system_writes[0].value == 3
    assert last is not None and last.result == 5


def test_executable_search_current_directory_policy_uses_guest_environment():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    memory = memory.map_bytes(0x7000, "tool.exe\0dir\\tool.exe\0".encode("utf-16le"))
    port = deterministic_windows_bootstrap_port()
    request = MachineExternalCallRequest(
        45, MachineExternalReference(
            39, 0, "guest-binary", "api-ms-win-core-processenvironment-l1-2-0.dll",
            "NeedCurrentDirectoryForExePathW",
        ), 0, 0, (0x7000, 0, 0, 0), 0,
    )
    assert port.handle(request, MachineExecutionState(pc=0, memory=memory)).result == 1
    disabled = MachineExecutionState(
        pc=0, memory=memory,
        environment_state={"NoDefaultCurrentDirectoryInExePath": ""},
    )
    assert port.handle(request, disabled).result == 0
    with_slash = replace(request, arguments=(0x7012, 0, 0, 0))
    assert port.handle(with_slash, disabled).result == 1


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


def test_crt_exit_runs_callbacks_lifo_then_requests_process_termination():
    request = MachineExternalCallRequest(
        101, MachineExternalReference(41, 0, "guest-binary", "msvcrt.dll", "exit"),
        0, 0, (37, 0, 0, 0), 0,
    )
    completion = deterministic_windows_bootstrap_port().handle(
        request,
        MachineExecutionState(pc=0, system_state={
            "windows.onexit.count": 3,
            "windows.onexit.0": 0x1000,
            "windows.onexit.1": 0x2000,
            "windows.onexit.2": 0x3000,
        }),
    )
    assert completion is not None
    assert completion.guest_calls == (0x3000, 0x2000, 0x1000)
    assert completion.terminate and completion.exit_code == 37


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


def test_heap_realloc_copies_payload_and_retires_old_virtual_allocation():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x2000)
    memory = memory.map_bytes(0x7100, b"payload!")
    state = MachineExecutionState(pc=0, memory=memory, system_state={
        "windows.system_arena_cursor": 0x7201,
        "windows.system_arena_limit": 0x9000,
        "windows.heap.allocation.28928.size": 8,
        "windows.heap.allocation.28928.active": 1,
    })
    request = MachineExternalCallRequest(
        44, MachineExternalReference(
            38, 0, "guest-binary", "api-ms-win-core-heap-l1-1-0.dll", "HeapReAlloc",
        ), 0, 0, (0x300, 0, 0x7100, 16), 0,
    )
    completion = deterministic_windows_bootstrap_port().handle(request, state)
    assert completion is not None and completion.result == 0x7210
    assert completion.memory_writes[-1].data == b"payload!"
    effects = {item.key: item.value for item in completion.system_writes}
    assert effects["windows.heap.allocation.28928.active"] == 0
    assert effects["windows.heap.allocation.29200.active"] == 1


def test_heap_reuses_released_virtual_blocks_before_exhausting_system_arena():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    state = MachineExecutionState(pc=0, memory=memory, system_state={
        "windows.system_arena_cursor": 0x7FF0,
        "windows.system_arena_limit": 0x8000,
        "windows.heap.allocation.28928.size": 128,
        "windows.heap.allocation.28928.active": 0,
    })
    request = MachineExternalCallRequest(
        301, MachineExternalReference(
            201, 0, "guest-binary", "api-ms-win-core-heap-l1-1-0.dll", "HeapAlloc",
        ), 0, 0, (0x300, 0x8, 96, 0), 0,
    )
    completion = deterministic_windows_bootstrap_port().handle(request, state)

    assert completion is not None and completion.result == 0x7100
    assert completion.memory_writes[0].data == bytes(96)
    assert {item.key: item.value for item in completion.system_writes} == {
        "windows.heap.allocation.28928.size": 96,
        "windows.heap.allocation.28928.active": 1,
    }


def test_crt_malloc_is_an_adapter_over_the_same_virtual_heap():
    port = deterministic_windows_bootstrap_port()
    request = MachineExternalCallRequest(
        31, MachineExternalReference(
            25, 0, "guest-binary", "msvcrt.dll", "malloc",
        ), 0, 0, (64, 0, 0, 0), 0,
    )
    state = MachineExecutionState(pc=0, system_state={
        "windows.system_arena_cursor": 0x7001,
        "windows.system_arena_limit": 0x8000,
    })
    completion = port.handle(request, state)
    assert completion is not None and completion.result == 0x7010
    assert any(
        item.key == "windows.heap.allocation.28688.size" and item.value == 64
        for item in completion.system_writes
    )


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


def test_crt_random_state_is_explicit_and_replayable():
    port = deterministic_windows_bootstrap_port()
    srand_request = MachineExternalCallRequest(
        28, MachineExternalReference(22, 0, "guest-binary", "msvcrt.dll", "srand"),
        0, 0, (7, 0, 0, 0), 0,
    )
    seeded = port.handle(srand_request, MachineExecutionState(pc=0))
    assert seeded is not None
    state = MachineExecutionState(
        pc=0,
        system_state={item.key: item.value for item in seeded.system_writes},
    )
    rand_request = MachineExternalCallRequest(
        29, MachineExternalReference(23, 0, "guest-binary", "msvcrt.dll", "rand"),
        0, 0, (0, 0, 0, 0), 0,
    )
    first = port.handle(rand_request, state)
    replay = port.handle(rand_request, state)
    assert first == replay


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


def test_locale_identity_comes_from_declared_virtual_ui_language():
    port = deterministic_windows_bootstrap_port(ui_language=0x0411)
    lcid = port.handle(_request(
        "api-ms-win-core-localization-l1-2-0.dll", "GetUserDefaultLCID",
    ), MachineExecutionState(pc=0))
    lang = port.handle(_request(
        "api-ms-win-core-localization-l1-2-0.dll", "GetSystemDefaultLangID",
    ), MachineExecutionState(pc=0))
    assert lcid is not None and lcid.result == 0x0411
    assert lang is not None and lang.result == 0x0411


def test_locale_info_uses_declared_table_and_win32_character_counts():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    state = MachineExecutionState(pc=0, memory=memory)
    port = deterministic_windows_bootstrap_port(ui_language=0x0409)
    request = MachineExternalCallRequest(
        49, MachineExternalReference(
            43, 0, "guest-binary", "api-ms-win-core-localization-l1-2-0.dll",
            "GetLocaleInfoW",
        ), 0, 0, (0x0409, 0x1E, 0x7000, 8), 0,
    )
    value = port.handle(request, state)
    size = port.handle(replace(request, arguments=(0x0409, 0x1E, 0, 0)), state)
    assert value is not None and value.result == 2
    assert value.memory_writes[0].data.decode("utf-16le") == ":\0"
    assert size is not None and size.result == 2


def test_crt_standard_descriptors_map_to_virtual_console_handles():
    port = deterministic_windows_bootstrap_port()
    request = MachineExternalCallRequest(
        23, MachineExternalReference(
            17, 0, "guest-binary", "msvcrt.dll", "_get_osfhandle",
        ), 0, 0, (1, 0, 0, 0), 0,
    )
    completion = port.handle(request, MachineExecutionState(pc=0))
    assert completion is not None and completion.result == 0x201

    std_handle = port.handle(MachineExternalCallRequest(
        231, MachineExternalReference(
            171, 0, "guest-binary",
            "api-ms-win-core-processenvironment-l1-1-0.dll", "GetStdHandle",
        ), 0, 0, (0xFFFFFFF5, 0, 0, 0), 0,
    ), MachineExecutionState(pc=0))
    assert std_handle is not None and std_handle.result == 0x201


def test_msvcrt_setjmp_captures_the_amd64_jump_buffer_contract():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    registers = tuple(0x100 + index for index in range(16))
    vectors = tuple((index << 120) | index for index in range(16))
    state = MachineExecutionState(
        pc=0, registers=registers, vector_registers=vectors, memory=memory,
        system_state={"amd64.mxcsr": 0x9FC0, "amd64.fpcsr": 0x037F},
    )
    request = MachineExternalCallRequest(
        50, MachineExternalReference(
            44, 0, "guest-binary", "msvcrt.dll", "_setjmp",
        ), 0x140001000, 0x140001005,
        (0x7000, 0x7FFEFFF0, 0, 0), 0x7FFEFFE0,
    )
    completion = deterministic_windows_bootstrap_port().handle(request, state)

    assert completion is not None and completion.result == 0
    payload = completion.memory_writes[0].data
    assert len(payload) == 256
    qword = lambda offset: int.from_bytes(payload[offset:offset + 8], "little")
    assert tuple(qword(offset) for offset in range(0, 88, 8)) == (
        0x7FFEFFF0, registers[3], 0x7FFEFFE8, registers[5],
        registers[6], registers[7], registers[12], registers[13],
        registers[14], registers[15], 0x140001005,
    )
    assert int.from_bytes(payload[88:92], "little") == 0x9FC0
    assert int.from_bytes(payload[92:94], "little") == 0x037F
    assert int.from_bytes(payload[96:112], "little") == vectors[6]
    assert int.from_bytes(payload[240:256], "little") == vectors[15]


def _local_unwind_fixture(*, scope_count=1, target=0x140002090):
    base = 0x140000000
    memory = PagedByteMemory.empty().map_zeroes(base, 0x5000)
    memory = memory.map_bytes(base, b"MZ")
    memory = memory.write_unsigned(base + 0x3C, 32, 0x80)
    memory = memory.map_bytes(base + 0x80, b"PE\0\0")
    memory = memory.write_unsigned(base + 0x80 + 20, 16, 240)
    optional = base + 0x80 + 24
    memory = memory.write_unsigned(optional, 16, 0x20B)
    memory = memory.write_unsigned(optional + 56, 32, 0x5000)
    memory = memory.write_unsigned(optional + 108, 32, 16)
    memory = memory.write_unsigned(optional + 112 + 3 * 8, 32, 0x1000)
    memory = memory.write_unsigned(optional + 112 + 3 * 8 + 4, 32, 12)
    memory = memory.map_bytes(
        base + 0x1000,
        struct.pack("<III", 0x2000, 0x2100, 0x1100),
    )
    # V1, EHANDLER|UHANDLER, no unwind codes; language handler then C scope data.
    memory = memory.map_bytes(
        base + 0x1100,
        bytes((1 | (3 << 3), 0, 0, 0))
        + struct.pack("<II", 0x1800, scope_count)
        + struct.pack("<IIII", 0x2000, 0x2080, 0x3000, 0),
    )
    frame = 0x7FFEFFF0
    request = MachineExternalCallRequest(
        51, MachineExternalReference(
            45, 0, "guest-binary", "msvcrt.dll", "_local_unwind",
        ), base + 0x4000, base + 0x2050,
        (frame, target, 0, 0), 0x7FFEFFE0,
    )
    state = MachineExecutionState(
        pc=0, memory=memory,
        system_state={"windows.loader.image_base": base},
    )
    return request, state


def test_msvcrt_local_unwind_dispatches_one_bounded_guest_finally_scope():
    request, state = _local_unwind_fixture()
    completion = deterministic_windows_bootstrap_port().handle(request, state)

    assert completion is not None
    assert completion.guest_calls == (0x140003000,)
    assert tuple((item.register, item.value) for item in completion.register_writes) == (
        (1, 1), (2, 0x7FFEFFF0),
    )


def test_msvcrt_local_unwind_fails_closed_on_unbounded_scope_metadata():
    request, state = _local_unwind_fixture(scope_count=1025)
    assert deterministic_windows_bootstrap_port().handle(request, state) is None


def test_msvcrt_local_unwind_completes_without_callback_when_target_stays_in_scope():
    request, state = _local_unwind_fixture(target=0x140002070)
    completion = deterministic_windows_bootstrap_port().handle(request, state)
    assert completion is not None and completion.guest_calls == ()
    assert completion.register_writes == ()


def test_msvcrt_vsnwprintf_formats_wide_guest_arguments_without_host_crt():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    memory = memory.map_bytes(0x7200, "%s %08X %I64d\r\n\0".encode("utf-16le"))
    memory = memory.map_bytes(0x7400, "hello\0".encode("utf-16le"))
    memory = memory.map_bytes(
        0x7600,
        (0x7400).to_bytes(8, "little")
        + (0x2A).to_bytes(8, "little")
        + ((-17) & ((1 << 64) - 1)).to_bytes(8, "little"),
    )
    state = MachineExecutionState(pc=0, memory=memory)
    request = MachineExternalCallRequest(
        54, MachineExternalReference(
            48, 0, "guest-binary", "msvcrt.dll", "_vsnwprintf",
        ), 0, 0, (0x7000, 128, 0x7200, 0x7600), 0,
    )
    completion = deterministic_windows_bootstrap_port().handle(request, state)

    expected = "hello 0000002A -17\r\n"
    assert completion is not None and completion.result == len(expected)
    assert completion.memory_writes[0].data.decode("utf-16le") == expected + "\0"


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


def test_srw_shared_lock_state_is_explicit_reversible_and_excludes_writers():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    state = MachineExecutionState(pc=0, memory=memory)
    port = deterministic_windows_bootstrap_port(thread_id=9)

    def call(symbol, active):
        return port.handle(MachineExternalCallRequest(
            240, MachineExternalReference(
                180, 0, "guest-binary", "api-ms-win-core-synch-l1-1-0.dll", symbol,
            ), 0, 0, (0x7000, 0, 0, 0), 0,
        ), active)

    acquired = call("AcquireSRWLockShared", state)
    assert acquired is not None and acquired.result == 1
    state = replace(state, system_state={
        effect.key: effect.value for effect in acquired.system_writes
    })
    denied = call("TryAcquireSRWLockExclusive", state)
    released = call("ReleaseSRWLockShared", state)
    assert denied is not None and denied.result == 0
    assert released is not None
    assert released.memory_writes[0].data == bytes(8)
    assert {effect.key: effect.value for effect in released.system_writes}[
        "windows.srw_lock.28672.readers"
    ] == 0


def test_console_title_and_wide_output_become_reversible_device_effects():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    memory = memory.map_bytes(0x7000, "hello\0".encode("utf-16le"))
    state = MachineExecutionState(
        pc=0, memory=memory,
        text_state={"windows.console.title": "virtual cmd"},
    )
    port = deterministic_windows_bootstrap_port()
    title = port.handle(MachineExternalCallRequest(
        51, MachineExternalReference(
            45, 0, "guest-binary", "api-ms-win-core-console-l2-2-0.dll",
            "GetConsoleTitleW",
        ), 0, 0, (0x7200, 32, 0, 0), 0,
    ), state)
    output = port.handle(MachineExternalCallRequest(
        52, MachineExternalReference(
            46, 0, "guest-binary", "api-ms-win-core-console-l1-1-0.dll",
            "WriteConsoleW",
        ), 0, 0, (0x201, 0x7000, 5, 0x7400), 0,
    ), state)
    assert title is not None and title.result == len("virtual cmd")
    assert title.memory_writes[0].data.decode("utf-16le") == "virtual cmd\0"
    assert output is not None and output.result == 1
    assert output.memory_writes[0].data == (5).to_bytes(4, "little")
    assert output.device_writes[0].device == "console.output"
    assert output.device_writes[0].data == b"hello"


def test_console_input_executable_name_is_reversible_text_state():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    memory = memory.map_bytes(0x7000, "cmd.exe\0".encode("utf-16le"))
    port = deterministic_windows_bootstrap_port()
    set_request = MachineExternalCallRequest(
        250, MachineExternalReference(
            190, 0, "guest-binary", "kernel32.dll", "SetConsoleInputExeNameW",
        ), 0, 0, (0x7000, 0, 0, 0), 0,
    )
    completion = port.handle(set_request, MachineExecutionState(pc=0, memory=memory))
    assert completion is not None and completion.result == 1
    assert completion.text_writes[0].key == "windows.console.input_exe_name"
    assert completion.text_writes[0].value == "cmd.exe"

    get_request = MachineExternalCallRequest(
        251, MachineExternalReference(
            191, 0, "guest-binary", "kernel32.dll", "GetConsoleInputExeNameW",
        ), 0, 0, (0x7200, 32, 0, 0), 0,
    )
    read = port.handle(
        get_request,
        MachineExecutionState(
            pc=0, memory=memory,
            text_state={"windows.console.input_exe_name": "cmd.exe"},
        ),
    )
    assert read is not None and read.result == 7
    assert read.memory_writes[0].data.decode("utf-16le") == "cmd.exe\0"


def test_read_file_blocks_until_reversible_console_input_is_available():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    request = MachineExternalCallRequest(
        252, MachineExternalReference(
            192, 0, "guest-binary", "api-ms-win-core-file-l1-1-0.dll", "ReadFile",
        ), 0, 0, (0x200, 0x7000, 32, 0x7040), 0, stack_arguments=(0,),
    )
    port = deterministic_windows_bootstrap_port()
    assert port.handle(request, MachineExecutionState(pc=0, memory=memory)) is None

    completion = port.handle(
        request,
        MachineExecutionState(
            pc=0, memory=memory,
            device_state={"console.input": b"echo hello\r\nremaining"},
        ),
    )
    assert completion is not None and completion.result == 1
    assert completion.memory_writes[0].data == b"echo hello\r\nremaining"
    assert completion.memory_writes[1].data == (21).to_bytes(4, "little")
    assert completion.device_writes == (
        MachineExternalDeviceWrite("console.input", b"", append=False),
    )


def test_read_console_w_consumes_utf8_device_input_as_utf16_guest_text():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    request = MachineExternalCallRequest(
        253, MachineExternalReference(
            193, 0, "guest-binary", "api-ms-win-core-console-l1-1-0.dll", "ReadConsoleW",
        ), 0, 0, (0x200, 0x7000, 4, 0x7040), 0,
    )
    completion = deterministic_windows_bootstrap_port().handle(
        request,
        MachineExecutionState(
            pc=0, memory=memory,
            device_state={"console.input": "héllo".encode("utf-8")},
        ),
    )
    assert completion is not None and completion.result == 1
    assert completion.memory_writes[0].data.decode("utf-16le") == "héll"
    assert completion.memory_writes[1].data == (4).to_bytes(4, "little")
    assert completion.device_writes[0].data == b"o"


def test_console_screen_information_is_derived_from_reversible_terminal_state():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    request = MachineExternalCallRequest(
        254, MachineExternalReference(
            194, 0, "guest-binary", "api-ms-win-core-console-l2-1-0.dll",
            "GetConsoleScreenBufferInfo",
        ), 0, 0, (0x201, 0x7000, 0, 0), 0,
    )
    completion = deterministic_windows_bootstrap_port().handle(
        request,
        MachineExecutionState(
            pc=0, memory=memory,
            device_state={"console.output": b"first\r\nnext"},
        ),
    )
    assert completion is not None and completion.result == 1
    values = struct.unpack("<hhhhHhhhhhh", completion.memory_writes[0].data)
    assert values[:5] == (80, 32766, 4, 1, 7)
    assert values[-2:] == (80, 25)


def test_create_process_resolves_only_to_registered_card_executor(tmp_path):
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x2000)
    memory = memory.map_bytes(0x7000, "reduce.exe\0".encode("utf-16le"))
    memory = memory.map_bytes(0x7100, 'reduce.exe "two words"\0'.encode("utf-16le"))
    registry = VirtualProgramRegistry()
    registry.register(
        "/c/tools/reduce.exe",
        bundle_reference="bundle:reduce@sha256:abc",
        executor_reference="card-set:reduce:v1",
        executor=lambda invocation: VirtualProgramResult(
            9, (" ".join(invocation.arguments) + "\r\n").encode(), execution_units=4,
        ),
    )
    port = deterministic_windows_bootstrap_port(
        program_registry=registry, current_directory=r"C:\tools",
    )
    request = MachineExternalCallRequest(
        255, MachineExternalReference(
            195, 0, "guest-binary", "kernel32.dll", "CreateProcessW",
        ), 0, 0, (0x7000, 0x7100, 0, 0), 0,
        stack_arguments=(0, 0, 0, 0, 0, 0x7800),
    )

    completion = port.handle(request, MachineExecutionState(pc=0, memory=memory))

    assert completion is not None and completion.result == 1
    process_handle, _, _, _ = struct.unpack("<QQII", completion.memory_writes[0].data)
    assert process_handle == 0x40000000000000FF
    assert completion.device_writes[0].data == b"two words\r\n"
    assert completion.deployments[0].kind == "card-set-executor"
    assert completion.deployments[0].resolved_reference == "bundle:reduce@sha256:abc"
    assert completion.deployments[0].executor_reference == "card-set:reduce:v1"
    deployment = completion.deployments[0]
    assert deployment.child_tape_schema == "turing.virtual-child-process-tape.v1"
    assert deployment.child_tape_reference == (
        f"child-tape:sha256:{deployment.child_tape_digest}"
    )
    exported = registry.export_child_tapes(tmp_path / "children")
    assert len(exported) == 1
    assert exported[0].stem == deployment.child_tape_digest
    assert b'"kind":"standard_output"' in exported[0].read_bytes()


def test_proc_thread_attribute_list_is_reversible_opaque_guest_state():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x2000)
    memory = memory.map_bytes(0x7100, (48).to_bytes(8, "little"))
    memory = memory.map_bytes(0x7200, (0x201).to_bytes(8, "little"))
    port = deterministic_windows_bootstrap_port()
    initialized = port.handle(MachineExternalCallRequest(
        256, MachineExternalReference(
            196, 0, "guest-binary", "api-ms-win-core-processthreads-l1-1-0.dll",
            "InitializeProcThreadAttributeList",
        ), 0, 0, (0x7000, 1, 0, 0x7100), 0,
    ), MachineExecutionState(pc=0, memory=memory))
    assert initialized is not None and initialized.result == 1
    assert initialized.memory_writes[1].address == 0x7000
    assert len(initialized.memory_writes[1].data) == 48

    state = MachineExecutionState(
        pc=0, memory=memory,
        system_state={
            "windows.proc_thread_attributes.28672.initialized": 1,
            "windows.proc_thread_attributes.28672.size": 48,
            "windows.proc_thread_attributes.28672.capacity": 1,
        },
    )
    updated = port.handle(MachineExternalCallRequest(
        257, MachineExternalReference(
            197, 0, "guest-binary", "api-ms-win-core-processthreads-l1-1-0.dll",
            "UpdateProcThreadAttribute",
        ), 0, 0, (0x7000, 0, 0x00020002, 0x7200), 0,
        stack_arguments=(8, 0, 0),
    ), state)
    assert updated is not None and updated.result == 1
    assert updated.text_writes[0].value == "0102000000000000"

    query = port.handle(MachineExternalCallRequest(
        258, MachineExternalReference(
            198, 0, "guest-binary", "api-ms-win-core-processthreads-l1-1-0.dll",
            "InitializeProcThreadAttributeList",
        ), 0, 0, (0, 1, 0, 0x7100), 0,
    ), MachineExecutionState(pc=0, memory=memory))
    assert query is not None and query.result == 0
    assert query.system_writes[0].value == 122


def test_startup_info_exposes_only_virtual_standard_handles():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    completion = deterministic_windows_bootstrap_port().handle(
        MachineExternalCallRequest(
            259, MachineExternalReference(
                199, 0, "guest-binary",
                "api-ms-win-core-processthreads-l1-1-0.dll", "GetStartupInfoW",
            ), 0, 0, (0x7000, 0, 0, 0), 0,
        ),
        MachineExecutionState(pc=0, memory=memory),
    )
    assert completion is not None
    payload = completion.memory_writes[0].data
    assert len(payload) == 104
    assert int.from_bytes(payload[0:4], "little") == 104
    assert int.from_bytes(payload[60:64], "little") == 0x100
    assert struct.unpack_from("<QQQ", payload, 80) == (0x200, 0x201, 0x202)


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


def test_command_line_is_windows_quoted_and_stored_in_guest_arena():
    port = deterministic_windows_bootstrap_port(
        arguments=("cmd.exe", "/c", "echo hello", 'a"b'),
    )
    request = MachineExternalCallRequest(
        30, MachineExternalReference(
            24, 0, "guest-binary",
            "api-ms-win-core-processenvironment-l1-1-0.dll", "GetCommandLineW",
        ), 0, 0, (0, 0, 0, 0), 0,
    )
    state = MachineExecutionState(pc=0, system_state={
        "windows.system_arena_cursor": 0x7000,
        "windows.system_arena_limit": 0x8000,
    })
    completion = port.handle(request, state)
    assert completion is not None and completion.result == 0x7000
    value = completion.memory_writes[0].data.decode("utf-16le").rstrip("\0")
    assert value == 'cmd.exe /c "echo hello" "a\\"b"'


def test_current_directory_is_shell_supplied_with_win32_length_contract():
    port = deterministic_windows_bootstrap_port(current_directory="C:\\work")
    request = MachineExternalCallRequest(
        32, MachineExternalReference(
            26, 0, "guest-binary",
            "api-ms-win-core-processenvironment-l1-1-0.dll",
            "GetCurrentDirectoryW",
        ), 0, 0, (32, 0x7000, 0, 0), 0,
    )
    completion = port.handle(request, MachineExecutionState(pc=0))
    assert completion is not None and completion.result == len("C:\\work")
    assert completion.memory_writes[0].data.decode("utf-16le") == "C:\\work\0"


def test_directory_and_module_filename_are_derived_from_virtual_namespace():
    filesystem = VirtualFileSystemState.create(
        VirtualFileSystemContract(
            current_directory="/c/work",
            mounts=(VirtualMount.create("/", "memory", access="read_write"),),
        ),
        files={"/c/windows/system32/cmd.exe": b"MZ"},
    )
    state = MachineExecutionState(pc=0, virtual_filesystem=filesystem)
    port = deterministic_windows_bootstrap_port(
        current_directory="Z:\\ignored",
        module_virtual_path="/c/windows/system32/cmd.exe",
    )
    cwd_request = MachineExternalCallRequest(
        33, MachineExternalReference(
            27, 0, "guest-binary",
            "api-ms-win-core-processenvironment-l1-1-0.dll", "GetCurrentDirectoryW",
        ), 0, 0, (64, 0x7000, 0, 0), 0,
    )
    module_request = MachineExternalCallRequest(
        34, MachineExternalReference(
            28, 0, "guest-binary",
            "api-ms-win-core-libraryloader-l1-2-0.dll", "GetModuleFileNameW",
        ), 0, 0, (0, 0x7200, 128, 0), 0,
    )
    cwd = port.handle(cwd_request, state)
    module = port.handle(module_request, state)
    assert cwd is not None
    assert cwd.memory_writes[0].data.decode("utf-16le") == "C:\\work\0"
    assert module is not None
    assert module.memory_writes[0].data.decode("utf-16le") == (
        "C:\\windows\\system32\\cmd.exe\0"
    )


def test_environment_variable_and_wide_compare_are_deterministic():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    memory = memory.map_bytes(0x7000, "Path\0".encode("utf-16le"))
    memory = memory.map_bytes(0x7100, "Alpha\0alpha\0".encode("utf-16le"))
    state = MachineExecutionState(pc=0, memory=memory)
    port = deterministic_windows_bootstrap_port(environment=("PATH=/tools",))
    env = port.handle(MachineExternalCallRequest(
        35, MachineExternalReference(
            29, 0, "guest-binary",
            "api-ms-win-core-processenvironment-l1-1-0.dll", "GetEnvironmentVariableW",
        ), 0, 0, (0x7000, 0x7200, 32, 0), 0,
    ), state)
    compare = port.handle(MachineExternalCallRequest(
        36, MachineExternalReference(30, 0, "guest-binary", "msvcrt.dll", "_wcsicmp"),
        0, 0, (0x7100, 0x710C, 0, 0), 0,
    ), state)
    assert env is not None and env.result == len("/tools")
    assert env.memory_writes[0].data.decode("utf-16le") == "/tools\0"
    assert compare is not None and compare.result == 0
    bounded_compare = port.handle(MachineExternalCallRequest(
        43, MachineExternalReference(37, 0, "guest-binary", "msvcrt.dll", "_wcsnicmp"),
        0, 0, (0x7100, 0x710C, 2, 0), 0,
    ), state)
    assert bounded_compare is not None and bounded_compare.result == 0

    delete = port.handle(MachineExternalCallRequest(
        37, MachineExternalReference(
            31, 0, "guest-binary",
            "api-ms-win-core-processenvironment-l1-1-0.dll", "SetEnvironmentVariableW",
        ), 0, 0, (0x7000, 0, 0, 0), 0,
    ), state)
    assert delete is not None and delete.result == 1
    assert delete.environment_writes[0].key == "Path"
    assert delete.environment_writes[0].value is None

    hidden = type(delete.environment_writes[0])("=C:", r"C:\work")
    assert hidden.key == "=C:"

    upper = port.handle(
        _request("msvcrt.dll", "towupper", pointer=ord("c")), state,
    )
    assert upper is not None and upper.result == ord("C")
    assert port.handle(
        _request("msvcrt.dll", "iswalpha", pointer=ord("C")), state,
    ).result == 1
    assert port.handle(
        _request("msvcrt.dll", "iswalpha", pointer=ord("7")), state,
    ).result == 0
    assert port.handle(
        _request("msvcrt.dll", "iswxdigit", pointer=ord("f")), state,
    ).result == 1


def test_file_path_capabilities_use_virtual_namespace_not_host_paths():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x2000)
    memory = memory.map_bytes(0x7000, "..\\bin\\cmd.exe\0".encode("utf-16le"))
    filesystem = VirtualFileSystemState.create(
        VirtualFileSystemContract(
            current_directory="/c/work/jobs",
            mounts=(VirtualMount.create("/", "memory", access="read_write"),),
        ),
        files={"/c/work/bin/cmd.exe": b"MZ"},
    )
    state = MachineExecutionState(
        pc=0, memory=memory, virtual_filesystem=filesystem,
    )
    port = deterministic_windows_bootstrap_port()
    full = port.handle(MachineExternalCallRequest(
        38, MachineExternalReference(
            32, 0, "guest-binary", "api-ms-win-core-file-l1-1-0.dll",
            "GetFullPathNameW",
        ), 0, 0, (0x7000, 256, 0x7400, 0x7600), 0,
    ), state)
    attributes = port.handle(MachineExternalCallRequest(
        39, MachineExternalReference(
            33, 0, "guest-binary", "api-ms-win-core-file-l1-1-0.dll",
            "GetFileAttributesW",
        ), 0, 0, (0x7000, 0, 0, 0), 0,
    ), state)
    assert full is not None and full.result == len(r"C:\work\bin\cmd.exe")
    assert full.memory_writes[0].data.decode("utf-16le") == (
        "C:\\work\\bin\\cmd.exe\0"
    )
    assert int.from_bytes(full.memory_writes[1].data, "little") == (
        0x7400 + len("C:\\work\\bin\\") * 2
    )
    assert attributes is not None and attributes.result == 0x80

    memory = memory.map_bytes(0x7800, "..\\bin\\*.exe\0".encode("utf-16le"))
    find_state = MachineExecutionState(
        pc=0, memory=memory, virtual_filesystem=filesystem,
    )
    found = port.handle(MachineExternalCallRequest(
        40, MachineExternalReference(
            34, 0, "guest-binary", "api-ms-win-core-file-l1-1-0.dll",
            "FindFirstFileW",
        ), 0, 0, (0x7800, 0x8000, 0, 0), 0,
    ), find_state)
    assert found is not None and found.result == 0x1000
    assert found.memory_writes[0].data[44:58].decode("utf-16le") == "cmd.exe"
    assert found.filesystem_effects[0].entries == ("/c/work/bin/cmd.exe",)
    found_ex = port.handle(MachineExternalCallRequest(
        48, MachineExternalReference(
            42, 0, "guest-binary", "api-ms-win-core-file-l1-1-0.dll",
            "FindFirstFileExW",
        ), 0, 0, (0x7800, 1, 0x8000, 0), 0,
        stack_arguments=(0, 2),
    ), find_state)
    assert found_ex is not None and found_ex.result == 0x1000

    missing_memory = memory.map_bytes(
        0x7C00, "C:\\work\\bin\\missing.*\0".encode("utf-16le"),
    )
    missing_state = replace(
        find_state, memory=missing_memory,
        system_state={"windows.thread.1.last_error": 5},
    )
    missing = port.handle(MachineExternalCallRequest(
        49, MachineExternalReference(
            43, 0, "guest-binary", "api-ms-win-core-file-l1-1-0.dll",
            "FindFirstFileExW",
        ), 0, 0, (0x7C00, 1, 0x8000, 0), 0,
        stack_arguments=(0, 2),
    ), missing_state)
    assert missing is not None and missing.result == (1 << 64) - 1
    assert missing.system_writes == (MachineExternalStateWrite(
        "windows.thread.1.last_error", 2,
    ),)

    memory = memory.map_bytes(0x7A00, "C:\\\0Z:\\\0".encode("utf-16le"))
    drive_state = MachineExecutionState(
        pc=0, memory=memory, virtual_filesystem=filesystem,
    )
    drive = port.handle(MachineExternalCallRequest(
        46, MachineExternalReference(
            40, 0, "guest-binary", "api-ms-win-core-file-l1-1-0.dll", "GetDriveTypeW",
        ), 0, 0, (0x7A00, 0, 0, 0), 0,
    ), drive_state)
    missing_drive = port.handle(replace(
        MachineExternalCallRequest(
            47, MachineExternalReference(
                41, 0, "guest-binary", "api-ms-win-core-file-l1-1-0.dll", "GetDriveTypeW",
            ), 0, 0, (0x7A00, 0, 0, 0), 0,
        ), arguments=(0x7A08, 0, 0, 0),
    ), drive_state)
    assert drive is not None and drive.result == 3
    assert missing_drive is not None and missing_drive.result == 1


def test_wide_string_crt_family_returns_guest_pointers_and_bounded_writes():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x1000)
    memory = memory.map_bytes(0x7000, "alpha/beta\0alp\0".encode("utf-16le"))
    state = MachineExecutionState(pc=0, memory=memory)
    port = deterministic_windows_bootstrap_port()
    found = port.handle(MachineExternalCallRequest(
        41, MachineExternalReference(35, 0, "guest-binary", "msvcrt.dll", "wcschr"),
        0, 0, (0x7000, ord("/"), 0, 0), 0,
    ), state)
    length = port.handle(
        _request("msvcrt.dll", "wcslen", pointer=0x7000), state,
    )
    copied = port.handle(MachineExternalCallRequest(
        42, MachineExternalReference(36, 0, "guest-binary", "msvcrt.dll", "wcsncpy"),
        0, 0, (0x7800, 0x7016, 5, 0), 0,
    ), state)
    assert found is not None and found.result == 0x7000 + len("alpha") * 2
    assert length is not None and length.result == len("alpha/beta")
    assert copied is not None and copied.result == 0x7800
    assert copied.memory_writes[0].data == "alp".encode("utf-16le") + bytes(4)


def test_setlocale_updates_reversible_text_state_and_guest_string():
    memory = PagedByteMemory.empty().map_zeroes(0x7000, 0x2000)
    memory = memory.map_bytes(0x7000, b".ACP\x00")
    state = MachineExecutionState(
        pc=0, memory=memory,
        system_state={
            "windows.system_arena_cursor": 0x7100,
            "windows.system_arena_limit": 0x9000,
        },
    )
    request = MachineExternalCallRequest(
        50, MachineExternalReference(44, 0, "guest-binary", "msvcrt.dll", "setlocale"),
        0, 0, (0, 0x7000, 0, 0), 0,
    )
    completion = deterministic_windows_bootstrap_port(input_code_page=65001).handle(
        request, state,
    )
    assert completion is not None and completion.result == 0x7100
    assert completion.memory_writes[0].data == b"English_United States.65001\x00"
    text = {item.key: item.value for item in completion.text_writes}
    assert text["msvcrt.locale.all"] == "English_United States.65001"
    assert text["msvcrt.locale.ctype"] == "English_United States.65001"

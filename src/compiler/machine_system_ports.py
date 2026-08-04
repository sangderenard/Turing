"""Capability-gated system ports for captured guest-binary references."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

from .machine_execution import (
    MachineExecutionState,
    MachineExternalCallCompletion,
    MachineExternalCallRequest,
    MachineExternalMemoryWrite,
    MachineExternalStateWrite,
    MachineExternalResolution,
)


ExternalCapabilityHandler = Callable[
    [MachineExternalCallRequest, MachineExecutionState],
    MachineExternalCallCompletion | None,
]


def _identity(request: MachineExternalCallRequest) -> tuple[str, str]:
    return request.reference.library.casefold(), request.reference.symbol.casefold()


@dataclass(frozen=True, slots=True)
class CapabilityGatedExternalPort:
    """Resolve only explicitly registered guest external capabilities."""

    handlers: Mapping[tuple[str, str], ExternalCapabilityHandler]

    @classmethod
    def build(
        cls,
        handlers: Mapping[tuple[str, str], ExternalCapabilityHandler],
    ) -> "CapabilityGatedExternalPort":
        normalized = {
            (library.casefold(), symbol.casefold()): handler
            for (library, symbol), handler in handlers.items()
        }
        return cls(MappingProxyType(normalized))

    def handle(
        self,
        request: MachineExternalCallRequest,
        state: MachineExecutionState,
    ) -> MachineExternalCallCompletion | None:
        handler = self.handlers.get(_identity(request))
        return None if handler is None else handler(request, state)


def _return_value(value: int) -> ExternalCapabilityHandler:
    def handler(request, state):
        return MachineExternalCallCompletion(request.request_id, result=value)
    return handler


def _write_u64(value: int) -> ExternalCapabilityHandler:
    def handler(request, state):
        return MachineExternalCallCompletion(
            request.request_id,
            result=1,
            memory_writes=(MachineExternalMemoryWrite(
                request.arguments[0],
                int(value & ((1 << 64) - 1)).to_bytes(8, "little"),
            ),),
        )
    return handler


def _read_utf16(memory, address: int, *, maximum_characters: int = 32768) -> str | None:
    units = bytearray()
    try:
        for index in range(maximum_characters):
            pair = bytes((memory[address + index * 2], memory[address + index * 2 + 1]))
            if pair == b"\x00\x00":
                return bytes(units).decode("utf-16le")
            units.extend(pair)
    except (KeyError, UnicodeDecodeError):
        return None
    return None


def _read_ascii(memory, address: int, *, maximum_characters: int = 32768) -> str | None:
    value = bytearray()
    try:
        for index in range(maximum_characters):
            byte = memory[address + index]
            if byte == 0:
                return bytes(value).decode("ascii")
            value.append(byte)
    except (KeyError, UnicodeDecodeError):
        return None
    return None


def _module_handle(
    module_base: int,
    module_handles: Mapping[str, int],
) -> ExternalCapabilityHandler:
    def handler(request, state):
        if request.arguments[0] != 0:
            name = _read_utf16(state.memory, request.arguments[0])
            if name is None:
                return None
            handle = module_handles.get(name.casefold(), 0)
            return MachineExternalCallCompletion(request.request_id, result=handle)
        return MachineExternalCallCompletion(
            request.request_id, result=module_base,
        )
    return handler


def _get_proc_address(module_handles: Mapping[str, int]) -> ExternalCapabilityHandler:
    libraries_by_handle = {handle: name for name, handle in module_handles.items()}

    def handler(request, state):
        handle, name_address = request.arguments[:2]
        library = libraries_by_handle.get(handle)
        if library is None or name_address <= 0xFFFF:
            return None
        symbol = _read_ascii(state.memory, name_address)
        if not symbol:
            return None
        return MachineExternalCallCompletion(
            request.request_id,
            resolution=MachineExternalResolution(library, symbol),
        )
    return handler


def _set_unhandled_exception_filter(request, state):
    key = "windows.unhandled_exception_filter"
    previous = int(state.system_state.get(key, 0))
    return MachineExternalCallCompletion(
        request.request_id,
        result=previous,
        system_writes=(MachineExternalStateWrite(key, request.arguments[0]),),
    )


def _msvcrt_initterm(request, state):
    begin, end = request.arguments[:2]
    if end < begin or (end - begin) % 8 or (end - begin) // 8 > 4096:
        return None
    calls: list[int] = []
    for address in range(begin, end, 8):
        try:
            target = int.from_bytes(
                bytes(state.memory[address + index] for index in range(8)),
                "little",
            )
        except KeyError:
            return None
        if target:
            calls.append(target)
    return MachineExternalCallCompletion(
        request.request_id,
        guest_calls=tuple(calls),
    )


def _msvcrt_getmainargs(
    arguments: tuple[str, ...],
    environment: tuple[str, ...],
) -> ExternalCapabilityHandler:
    encoded_arguments = tuple(value.encode("mbcs") + b"\x00" for value in arguments)
    encoded_environment = tuple(value.encode("mbcs") + b"\x00" for value in environment)

    def handler(request, state):
        cursor = int(state.system_state.get("windows.system_arena_cursor", 0))
        limit = int(state.system_state.get("windows.system_arena_limit", 0))
        if not cursor or cursor >= limit:
            return None
        writes: list[MachineExternalMemoryWrite] = []

        def place_strings(values: tuple[bytes, ...]) -> tuple[int, ...]:
            nonlocal cursor
            pointers: list[int] = []
            for value in values:
                pointers.append(cursor)
                writes.append(MachineExternalMemoryWrite(cursor, value))
                cursor += len(value)
            return tuple(pointers)

        argv_pointers = place_strings(encoded_arguments)
        env_pointers = place_strings(encoded_environment)
        cursor = (cursor + 7) & -8
        argv_address = cursor
        argv_bytes = b"".join(pointer.to_bytes(8, "little") for pointer in argv_pointers) + bytes(8)
        writes.append(MachineExternalMemoryWrite(argv_address, argv_bytes))
        cursor += len(argv_bytes)
        environment_address = cursor
        environment_bytes = b"".join(pointer.to_bytes(8, "little") for pointer in env_pointers) + bytes(8)
        writes.append(MachineExternalMemoryWrite(environment_address, environment_bytes))
        cursor += len(environment_bytes)
        if cursor > limit:
            return None
        argc_pointer, argv_pointer, environment_pointer = request.arguments[:3]
        writes.extend((
            MachineExternalMemoryWrite(argc_pointer, len(arguments).to_bytes(4, "little")),
            MachineExternalMemoryWrite(argv_pointer, argv_address.to_bytes(8, "little")),
            MachineExternalMemoryWrite(environment_pointer, environment_address.to_bytes(8, "little")),
        ))
        return MachineExternalCallCompletion(
            request.request_id,
            result=0,
            memory_writes=tuple(writes),
            system_writes=(MachineExternalStateWrite(
                "windows.system_arena_cursor", cursor,
            ),),
        )
    return handler


def _msvcrt_onexit(request, state):
    count = int(state.system_state.get("windows.onexit.count", 0))
    callback = request.arguments[0]
    return MachineExternalCallCompletion(
        request.request_id,
        result=callback,
        system_writes=(
            MachineExternalStateWrite(f"windows.onexit.{count}", callback),
            MachineExternalStateWrite("windows.onexit.count", count + 1),
        ),
    )


def _msvcrt_atexit(request, state):
    completion = _msvcrt_onexit(request, state)
    return MachineExternalCallCompletion(
        completion.request_id,
        result=0,
        memory_writes=completion.memory_writes,
        system_writes=completion.system_writes,
        guest_calls=completion.guest_calls,
    )


def _bounded_count(request, *, maximum: int = 64 * 1024 * 1024) -> int | None:
    count = int(request.arguments[2])
    return count if 0 <= count <= maximum else None


def _msvcrt_memset(request, state):
    count = _bounded_count(request)
    if count is None:
        return None
    destination, value = request.arguments[:2]
    return MachineExternalCallCompletion(
        request.request_id,
        result=destination,
        memory_writes=(MachineExternalMemoryWrite(
            destination, bytes((value & 0xFF,)) * count,
        ),),
    )


def _msvcrt_memory_copy(request, state):
    count = _bounded_count(request)
    if count is None:
        return None
    destination, source = request.arguments[:2]
    try:
        data = bytes(state.memory[source + index] for index in range(count))
    except KeyError:
        return None
    return MachineExternalCallCompletion(
        request.request_id,
        result=destination,
        memory_writes=(MachineExternalMemoryWrite(destination, data),),
    )


def _msvcrt_memcmp(request, state):
    count = _bounded_count(request)
    if count is None:
        return None
    left, right = request.arguments[:2]
    try:
        for index in range(count):
            difference = state.memory[left + index] - state.memory[right + index]
            if difference:
                return MachineExternalCallCompletion(request.request_id, result=difference)
    except KeyError:
        return None
    return MachineExternalCallCompletion(request.request_id, result=0)


def _windows_open_thread(thread_id: int) -> ExternalCapabilityHandler:
    def handler(request, state):
        access, inherit, requested_thread = request.arguments[:3]
        if requested_thread != thread_id:
            return MachineExternalCallCompletion(request.request_id, result=0)
        handle = int(state.system_state.get("windows.next_handle", 0x100))
        return MachineExternalCallCompletion(
            request.request_id,
            result=handle,
            system_writes=(
                MachineExternalStateWrite("windows.next_handle", handle + 1),
                MachineExternalStateWrite(f"windows.handle.{handle}.kind", 1),
                MachineExternalStateWrite(f"windows.handle.{handle}.id", thread_id),
                MachineExternalStateWrite(f"windows.handle.{handle}.access", access),
                MachineExternalStateWrite(f"windows.handle.{handle}.inherit", inherit),
            ),
        )
    return handler


def _windows_close_handle(request, state):
    handle = request.arguments[0]
    key = f"windows.handle.{handle}.kind"
    if not state.system_state.get(key, 0):
        return MachineExternalCallCompletion(request.request_id, result=0)
    return MachineExternalCallCompletion(
        request.request_id,
        result=1,
        system_writes=(MachineExternalStateWrite(key, 0),),
    )


def _set_thread_ui_language(default_language: int) -> ExternalCapabilityHandler:
    def handler(request, state):
        requested = request.arguments[0] & 0xFFFF
        selected = requested or (default_language & 0xFFFF)
        return MachineExternalCallCompletion(
            request.request_id,
            result=selected,
            system_writes=(MachineExternalStateWrite(
                "windows.thread_ui_language", selected,
            ),),
        )
    return handler


def _heap_set_information(request, state):
    heap, information_class, information = request.arguments[:3]
    return MachineExternalCallCompletion(
        request.request_id,
        result=1,
        system_writes=(
            MachineExternalStateWrite(
                f"windows.heap.{heap}.information_class.{information_class}",
                information,
            ),
        ),
    )


def _get_process_heap(request, state):
    return MachineExternalCallCompletion(request.request_id, result=0x300)


def _msvcrt_time(file_time: int) -> ExternalCapabilityHandler:
    unix_time = file_time // 10_000_000 - 11_644_473_600

    def handler(request, state):
        output = request.arguments[0]
        writes = (
            (MachineExternalMemoryWrite(
                output, int(unix_time).to_bytes(8, "little", signed=True),
            ),)
            if output else ()
        )
        return MachineExternalCallCompletion(
            request.request_id,
            result=unix_time,
            memory_writes=writes,
        )
    return handler


def _msvcrt_srand(request, state):
    return MachineExternalCallCompletion(
        request.request_id,
        system_writes=(MachineExternalStateWrite(
            "msvcrt.rand_state", request.arguments[0] & 0xFFFFFFFF,
        ),),
    )


def _msvcrt_rand(request, state):
    seed = int(state.system_state.get("msvcrt.rand_state", 1)) & 0xFFFFFFFF
    seed = (seed * 214013 + 2531011) & 0xFFFFFFFF
    return MachineExternalCallCompletion(
        request.request_id,
        result=(seed >> 16) & 0x7FFF,
        system_writes=(MachineExternalStateWrite("msvcrt.rand_state", seed),),
    )


def _heap_alloc(request, state):
    heap, flags, size = request.arguments[:3]
    if heap != 0x300 or size > 256 * 1024 * 1024:
        return MachineExternalCallCompletion(request.request_id, result=0)
    cursor = (int(state.system_state.get("windows.system_arena_cursor", 0)) + 15) & -16
    limit = int(state.system_state.get("windows.system_arena_limit", 0))
    capacity = max(int(size), 1)
    if not cursor or cursor + capacity > limit:
        return MachineExternalCallCompletion(request.request_id, result=0)
    writes = (
        (MachineExternalMemoryWrite(cursor, bytes(capacity)),)
        if flags & 0x8 else ()
    )
    return MachineExternalCallCompletion(
        request.request_id,
        result=cursor,
        memory_writes=writes,
        system_writes=(
            MachineExternalStateWrite("windows.system_arena_cursor", cursor + capacity),
            MachineExternalStateWrite(f"windows.heap.allocation.{cursor}.size", size),
            MachineExternalStateWrite(f"windows.heap.allocation.{cursor}.active", 1),
        ),
    )


def _heap_free(request, state):
    heap, _, address = request.arguments[:3]
    active_key = f"windows.heap.allocation.{address}.active"
    valid = heap == 0x300 and bool(state.system_state.get(active_key, 0))
    return MachineExternalCallCompletion(
        request.request_id,
        result=int(valid),
        system_writes=(MachineExternalStateWrite(active_key, 0),) if valid else (),
    )


def _heap_size(request, state):
    heap, _, address = request.arguments[:3]
    active = state.system_state.get(f"windows.heap.allocation.{address}.active", 0)
    size = state.system_state.get(f"windows.heap.allocation.{address}.size", 0)
    return MachineExternalCallCompletion(
        request.request_id,
        result=int(size) if heap == 0x300 and active else (1 << 64) - 1,
    )


def _empty_registry_open_key(request, state):
    # ERROR_FILE_NOT_FOUND. The virtual registry is intentionally empty unless
    # a future shell capability supplies an explicit registry image.
    return MachineExternalCallCompletion(request.request_id, result=2)


def _get_cp_info(request, state):
    code_page, output = request.arguments[:2]
    maximum_character_size = 4 if code_page == 65001 else 1
    cp_info = (
        maximum_character_size.to_bytes(4, "little")
        + b"?\x00"
        + bytes(12)
        + bytes(2)  # native structure tail alignment
    )
    return MachineExternalCallCompletion(
        request.request_id,
        result=1,
        memory_writes=(MachineExternalMemoryWrite(output, cp_info),),
    )


def _initialize_critical_section(request, state):
    address = request.arguments[0]
    prefix = f"windows.critical_section.{address}"
    return MachineExternalCallCompletion(
        request.request_id,
        result=1,
        memory_writes=(MachineExternalMemoryWrite(address, bytes(40)),),
        system_writes=(
            MachineExternalStateWrite(f"{prefix}.initialized", 1),
            MachineExternalStateWrite(f"{prefix}.owner", 0),
            MachineExternalStateWrite(f"{prefix}.recursion", 0),
        ),
    )


def _enter_critical_section(thread_id: int) -> ExternalCapabilityHandler:
    def handler(request, state):
        address = request.arguments[0]
        prefix = f"windows.critical_section.{address}"
        if not state.system_state.get(f"{prefix}.initialized", 0):
            return None
        owner = int(state.system_state.get(f"{prefix}.owner", 0))
        if owner not in (0, thread_id):
            return None  # remains pending until the owning virtual thread leaves
        recursion = int(state.system_state.get(f"{prefix}.recursion", 0)) + 1
        return MachineExternalCallCompletion(
            request.request_id,
            result=1,
            system_writes=(
                MachineExternalStateWrite(f"{prefix}.owner", thread_id),
                MachineExternalStateWrite(f"{prefix}.recursion", recursion),
            ),
        )
    return handler


def _leave_critical_section(thread_id: int) -> ExternalCapabilityHandler:
    def handler(request, state):
        address = request.arguments[0]
        prefix = f"windows.critical_section.{address}"
        owner = int(state.system_state.get(f"{prefix}.owner", 0))
        recursion = int(state.system_state.get(f"{prefix}.recursion", 0))
        if owner != thread_id or recursion <= 0:
            return None
        recursion -= 1
        return MachineExternalCallCompletion(
            request.request_id,
            system_writes=(
                MachineExternalStateWrite(f"{prefix}.owner", thread_id if recursion else 0),
                MachineExternalStateWrite(f"{prefix}.recursion", recursion),
            ),
        )
    return handler


def _set_console_control_handler(request, state):
    callback, add = request.arguments[:2]
    key = f"windows.console_control_handler.{callback}"
    return MachineExternalCallCompletion(
        request.request_id,
        result=1,
        system_writes=(MachineExternalStateWrite(key, int(bool(add))),),
    )


def _get_osfhandle(request, state):
    descriptors = {0: 0x200, 1: 0x201, 2: 0x202}
    return MachineExternalCallCompletion(
        request.request_id,
        result=descriptors.get(request.arguments[0], (1 << 64) - 1),
    )


def _get_console_mode(request, state):
    handle, output = request.arguments[:2]
    defaults = {0x200: 0x0007, 0x201: 0x0003, 0x202: 0x0003}
    if handle not in defaults:
        return MachineExternalCallCompletion(request.request_id, result=0)
    mode = int(state.system_state.get(f"windows.console.{handle}.mode", defaults[handle]))
    return MachineExternalCallCompletion(
        request.request_id,
        result=1,
        memory_writes=(MachineExternalMemoryWrite(
            output, mode.to_bytes(4, "little"),
        ),),
    )


def _set_console_mode(request, state):
    handle, mode = request.arguments[:2]
    if handle not in (0x200, 0x201, 0x202):
        return MachineExternalCallCompletion(request.request_id, result=0)
    return MachineExternalCallCompletion(
        request.request_id,
        result=1,
        system_writes=(MachineExternalStateWrite(
            f"windows.console.{handle}.mode", mode & 0xFFFFFFFF,
        ),),
    )


def _get_file_type(request, state):
    return MachineExternalCallCompletion(
        request.request_id,
        result=2 if request.arguments[0] in (0x200, 0x201, 0x202) else 0,
    )


def _get_environment_strings(environment: tuple[str, ...]) -> ExternalCapabilityHandler:
    payload = (
        "\0".join(environment).encode("utf-16le") + b"\x00\x00\x00\x00"
        if environment else b"\x00\x00\x00\x00"
    )

    def handler(request, state):
        cursor = (int(state.system_state.get("windows.system_arena_cursor", 0)) + 1) & -2
        limit = int(state.system_state.get("windows.system_arena_limit", 0))
        if not cursor or cursor + len(payload) > limit:
            return None
        return MachineExternalCallCompletion(
            request.request_id,
            result=cursor,
            memory_writes=(MachineExternalMemoryWrite(cursor, payload),),
            system_writes=(
                MachineExternalStateWrite("windows.system_arena_cursor", cursor + len(payload)),
                MachineExternalStateWrite("windows.environment_strings", cursor),
            ),
        )
    return handler


def _free_environment_strings(request, state):
    return MachineExternalCallCompletion(
        request.request_id,
        result=int(request.arguments[0] == state.system_state.get("windows.environment_strings", 0)),
    )


def deterministic_windows_bootstrap_port(
    *,
    file_time: int = 132_537_600_000_000_000,
    process_id: int = 1,
    thread_id: int = 1,
    performance_counter: int = 0,
    performance_frequency: int = 10_000_000,
    tick_count: int = 0,
    module_base: int = 0x140000000,
    arguments: tuple[str, ...] = ("cmd.exe",),
    environment: tuple[str, ...] = (),
    module_handles: Mapping[str, int] | None = None,
    ui_language: int = 0x0409,
    input_code_page: int = 65001,
    output_code_page: int = 65001,
) -> CapabilityGatedExternalPort:
    """A small, reproducible Windows loader/startup capability set.

    This is deliberately not a general Win32 passthrough.  Values come from
    caller-owned virtual state, allowing replay and reverse execution to be
    exact. Unknown functions remain pending at the system-port boundary.
    """

    active_module_handles = {
        "kernel32.dll": 0xFFFF900000000001,
        "ntdll.dll": 0xFFFF900000000002,
        **{
            name.casefold(): int(handle)
            for name, handle in (module_handles or {}).items()
        },
    }
    api_sets = {
        "api-ms-win-core-sysinfo-l1-1-0.dll": {
            "GetSystemTimeAsFileTime": _write_u64(file_time),
            "GetTickCount": _return_value(tick_count & 0xFFFFFFFF),
            "GetTickCount64": _return_value(tick_count),
        },
        "api-ms-win-core-processthreads-l1-1-0.dll": {
            "GetCurrentProcessId": _return_value(process_id),
            "GetCurrentThreadId": _return_value(thread_id),
            "GetCurrentProcess": _return_value((1 << 64) - 1),
            "GetCurrentThread": _return_value((1 << 64) - 2),
            "OpenThread": _windows_open_thread(thread_id),
        },
        "api-ms-win-core-profile-l1-1-0.dll": {
            "QueryPerformanceCounter": _write_u64(performance_counter),
            "QueryPerformanceFrequency": _write_u64(performance_frequency),
        },
        "api-ms-win-core-libraryloader-l1-2-0.dll": {
            "GetModuleHandleW": _module_handle(
                module_base, MappingProxyType(active_module_handles),
            ),
            "GetProcAddress": _get_proc_address(
                MappingProxyType(active_module_handles),
            ),
        },
        "msvcrt.dll": {
            "__set_app_type": _return_value(0),
            "_initterm": _msvcrt_initterm,
            "__getmainargs": _msvcrt_getmainargs(
                tuple(arguments), tuple(environment),
            ),
            "_onexit": _msvcrt_onexit,
            "atexit": _msvcrt_atexit,
            "memset": _msvcrt_memset,
            "memcpy": _msvcrt_memory_copy,
            "memmove": _msvcrt_memory_copy,
            "memcmp": _msvcrt_memcmp,
            "_get_osfhandle": _get_osfhandle,
            "time": _msvcrt_time(file_time),
            "srand": _msvcrt_srand,
            "rand": _msvcrt_rand,
        },
        "api-ms-win-core-errorhandling-l1-1-0.dll": {
            "SetUnhandledExceptionFilter": _set_unhandled_exception_filter,
        },
        "api-ms-win-core-handle-l1-1-0.dll": {
            "CloseHandle": _windows_close_handle,
        },
        "api-ms-win-core-heap-l1-1-0.dll": {
            "HeapSetInformation": _heap_set_information,
            "GetProcessHeap": _get_process_heap,
            "HeapAlloc": _heap_alloc,
            "HeapFree": _heap_free,
            "HeapSize": _heap_size,
        },
        "api-ms-win-core-registry-l1-1-0.dll": {
            "RegOpenKeyExW": _empty_registry_open_key,
        },
        "api-ms-win-core-console-l1-1-0.dll": {
            "GetConsoleCP": _return_value(input_code_page),
            "GetConsoleOutputCP": _return_value(output_code_page),
            "SetConsoleCtrlHandler": _set_console_control_handler,
            "GetConsoleMode": _get_console_mode,
            "SetConsoleMode": _set_console_mode,
            "GetFileType": _get_file_type,
        },
        "api-ms-win-core-localization-l1-2-0.dll": {
            "GetCPInfo": _get_cp_info,
        },
        "api-ms-win-core-synch-l1-1-0.dll": {
            "InitializeCriticalSection": _initialize_critical_section,
            "EnterCriticalSection": _enter_critical_section(thread_id),
            "LeaveCriticalSection": _leave_critical_section(thread_id),
        },
        "api-ms-win-core-processenvironment-l1-1-0.dll": {
            "GetEnvironmentStringsW": _get_environment_strings(tuple(environment)),
            "FreeEnvironmentStringsW": _free_environment_strings,
        },
        "kernel32.dll": {
            "SetThreadUILanguage": _set_thread_ui_language(ui_language),
        },
    }
    handlers = {
        (library, symbol): handler
        for library, symbols in api_sets.items()
        for symbol, handler in symbols.items()
    }
    return CapabilityGatedExternalPort.build(handlers)


__all__ = [
    "CapabilityGatedExternalPort",
    "ExternalCapabilityHandler",
    "deterministic_windows_bootstrap_port",
]

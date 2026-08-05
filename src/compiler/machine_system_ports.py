"""Capability-gated system ports for captured guest-binary references."""

from __future__ import annotations

from dataclasses import dataclass, replace
import fnmatch
import struct
from types import MappingProxyType
from typing import Callable, Mapping

from .machine_execution import (
    MachineExecutionState,
    MachineExternalCallCompletion,
    MachineExternalCallRequest,
    MachineExternalDeployment,
    MachineExternalMemoryWrite,
    MachineExternalRegisterWrite,
    MachineExternalStateWrite,
    MachineExternalThreadSpawn,
    MachineExternalResolution,
    MachineExternalEnvironmentWrite,
    MachineExternalTextStateWrite,
    MachineExternalDeviceWrite,
)
from .virtual_filesystem import (
    VirtualFileEffect, normalize_virtual_path, virtual_path_to_windows,
)
from .virtual_process import VirtualProgramRegistry, split_windows_command_line


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

    def supports(self, request: MachineExternalCallRequest) -> bool:
        """Report capability admission separately from temporary readiness."""

        return _identity(request) in self.handlers

    def wait_kind(self, request: MachineExternalCallRequest) -> str | None:
        """Distinguish admitted blocking input from an unsupported call shape."""

        library, symbol = _identity(request)
        if symbol in {"readconsolew", "readconsolea"} and "console" in library:
            return "terminal_input"
        if symbol == "readfile" and "file" in library:
            return "terminal_input"
        if symbol == "waitforsingleobject":
            return "thread_wait"
        return None


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
            request.request_id,
            result=int(state.system_state.get(
                "windows.loader.image_base", module_base,
            )),
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


def _set_error_mode(request, state):
    key = "windows.error_mode"
    previous = int(state.system_state.get(key, 0))
    return MachineExternalCallCompletion(
        request.request_id, result=previous,
        system_writes=(MachineExternalStateWrite(key, request.arguments[0]),),
    )


def _get_error_mode(request, state):
    return MachineExternalCallCompletion(
        request.request_id, result=int(state.system_state.get("windows.error_mode", 0)),
    )


def _set_last_error(thread_id: int) -> ExternalCapabilityHandler:
    def handler(request, state):
        return MachineExternalCallCompletion(
            request.request_id,
            system_writes=(MachineExternalStateWrite(
                f"windows.thread.{thread_id}.last_error", request.arguments[0],
            ),),
        )
    return handler


def _get_last_error(thread_id: int) -> ExternalCapabilityHandler:
    def handler(request, state):
        return MachineExternalCallCompletion(
            request.request_id,
            result=int(state.system_state.get(f"windows.thread.{thread_id}.last_error", 0)),
        )
    return handler


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


def _msvcrt_exit(*, run_callbacks: bool = True) -> ExternalCapabilityHandler:
    def handler(request, state):
        callbacks = ()
        if run_callbacks:
            count = int(state.system_state.get("windows.onexit.count", 0))
            callbacks = tuple(
                int(state.system_state.get(f"windows.onexit.{index}", 0))
                for index in reversed(range(count))
                if int(state.system_state.get(f"windows.onexit.{index}", 0))
            )
        return MachineExternalCallCompletion(
            request.request_id,
            guest_calls=callbacks,
            terminate=True,
            exit_code=int(request.arguments[0]) & 0xFFFFFFFF,
        )
    return handler


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


def _ascii_wide_fold(value: str) -> str:
    return "".join(
        chr(ord(character) + 32) if "A" <= character <= "Z" else character
        for character in value
    )


def _msvcrt_wcsicmp(request, state, *, bounded: bool = False):
    left = _read_utf16(state.memory, request.arguments[0])
    right = _read_utf16(state.memory, request.arguments[1])
    if left is None or right is None:
        return None
    if bounded:
        count = int(request.arguments[2])
        left, right = left[:count], right[:count]
    left_folded, right_folded = _ascii_wide_fold(left), _ascii_wide_fold(right)
    result = (left_folded > right_folded) - (left_folded < right_folded)
    return MachineExternalCallCompletion(request.request_id, result=result)


def _msvcrt_wcsnicmp(request, state):
    return _msvcrt_wcsicmp(request, state, bounded=True)


def _msvcrt_wide_case(*, upper: bool) -> ExternalCapabilityHandler:
    def handler(request, state):
        value = int(request.arguments[0])
        if upper and ord("a") <= value <= ord("z"):
            result = value - 32
        elif not upper and ord("A") <= value <= ord("Z"):
            result = value + 32
        else:
            result = value
        return MachineExternalCallCompletion(request.request_id, result=result)
    return handler


def _msvcrt_wide_classifier(kind: str) -> ExternalCapabilityHandler:
    def handler(request, state):
        value = int(request.arguments[0])
        alpha = ord("A") <= value <= ord("Z") or ord("a") <= value <= ord("z")
        digit = ord("0") <= value <= ord("9")
        upper = ord("A") <= value <= ord("Z")
        lower = ord("a") <= value <= ord("z")
        space = value in (9, 10, 11, 12, 13, 32)
        control = value < 32 or value == 127
        printable = 32 <= value < 127
        punctuation = printable and not alpha and not digit and not space
        hexadecimal = digit or ord("A") <= value <= ord("F") or ord("a") <= value <= ord("f")
        predicates = {
            "iswalpha": alpha, "iswdigit": digit, "iswalnum": alpha or digit,
            "iswspace": space, "iswupper": upper, "iswlower": lower,
            "iswcntrl": control, "iswprint": printable, "iswgraph": printable and not space,
            "iswpunct": punctuation, "iswxdigit": hexadecimal,
        }
        return MachineExternalCallCompletion(
            request.request_id, result=int(predicates[kind]),
        )
    return handler


def _msvcrt_wide_search(*, reverse: bool = False) -> ExternalCapabilityHandler:
    def handler(request, state):
        address, target = request.arguments[:2]
        value = _read_utf16(state.memory, address)
        if value is None or target > 0xFFFF:
            return None
        character = chr(target)
        index = value.rfind(character) if reverse else value.find(character)
        if target == 0:
            index = len(value)
        return MachineExternalCallCompletion(
            request.request_id, result=0 if index < 0 else address + index * 2,
        )
    return handler


def _msvcrt_wide_length(request, state):
    value = _read_utf16(state.memory, request.arguments[0])
    return None if value is None else MachineExternalCallCompletion(
        request.request_id, result=len(value),
    )


def _msvcrt_wide_compare(*, bounded: bool = False) -> ExternalCapabilityHandler:
    def handler(request, state):
        left = _read_utf16(state.memory, request.arguments[0])
        right = _read_utf16(state.memory, request.arguments[1])
        if left is None or right is None:
            return None
        if bounded:
            count = int(request.arguments[2])
            left, right = left[:count], right[:count]
        result = (left > right) - (left < right)
        return MachineExternalCallCompletion(request.request_id, result=result)
    return handler


def _msvcrt_wide_copy(*, bounded: bool = False, concatenate: bool = False) -> ExternalCapabilityHandler:
    def handler(request, state):
        destination, source = request.arguments[:2]
        value = _read_utf16(state.memory, source)
        if value is None:
            return None
        prefix = ""
        if concatenate:
            prefix = _read_utf16(state.memory, destination)
            if prefix is None:
                return None
        if bounded:
            count = int(request.arguments[2])
            copied = value[:count]
            payload = copied.encode("utf-16le")
            if len(copied) < count:
                payload += bytes((count - len(copied)) * 2)
        else:
            payload = (prefix + value).encode("utf-16le") + b"\x00\x00"
        return MachineExternalCallCompletion(
            request.request_id, result=destination,
            memory_writes=(MachineExternalMemoryWrite(destination, payload),),
        )
    return handler


def _format_wide_printf(memory, format_string: str, next_argument, *, maximum_characters: int):
    """Format the deterministic integer/string subset of the MS wide CRT ABI."""

    output: list[str] = []
    index = 0

    def bounded_width(value: int) -> int:
        if abs(value) > maximum_characters:
            raise ValueError("wide printf field exceeds the declared output bound")
        return value

    while index < len(format_string):
        if format_string[index] != "%":
            output.append(format_string[index])
            index += 1
            continue
        index += 1
        if index < len(format_string) and format_string[index] == "%":
            output.append("%")
            index += 1
            continue
        flags = ""
        while index < len(format_string) and format_string[index] in "-+ 0#":
            if format_string[index] not in flags:
                flags += format_string[index]
            index += 1
        if index < len(format_string) and format_string[index] == "*":
            width = int(next_argument() & 0xFFFFFFFF)
            if width & 0x80000000:
                width -= 1 << 32
            width = bounded_width(width)
            index += 1
        else:
            begin = index
            while index < len(format_string) and format_string[index].isdigit():
                index += 1
            width = bounded_width(int(format_string[begin:index] or 0))
        precision = None
        if index < len(format_string) and format_string[index] == ".":
            index += 1
            if index < len(format_string) and format_string[index] == "*":
                precision = int(next_argument() & 0xFFFFFFFF)
                if precision & 0x80000000:
                    precision -= 1 << 32
                precision = None if precision < 0 else bounded_width(precision)
                index += 1
            else:
                begin = index
                while index < len(format_string) and format_string[index].isdigit():
                    index += 1
                precision = bounded_width(int(format_string[begin:index] or 0))
        length = ""
        for candidate in ("I64", "I32", "ll", "hh", "l", "h", "j", "z", "t", "L"):
            if format_string.startswith(candidate, index):
                length = candidate
                index += len(candidate)
                break
        if index >= len(format_string):
            return None
        conversion = format_string[index]
        index += 1
        if conversion not in "cCsSdiuoxXp":
            return None
        raw = int(next_argument())
        if conversion in "sS":
            narrow = (conversion == "S") != (length == "h")
            if raw == 0:
                field = "(null)"
            else:
                field = (
                    _read_ascii(memory, raw, maximum_characters=maximum_characters)
                    if narrow else
                    _read_utf16(memory, raw, maximum_characters=maximum_characters)
                )
                if field is None:
                    return None
            if precision is not None:
                field = field[:precision]
        elif conversion in "cC":
            field = chr(raw & (0xFF if conversion == "C" or length == "h" else 0xFFFF))
        else:
            bits = 64 if length in {"I64", "ll", "j", "z", "t"} or conversion == "p" else 32
            mask = (1 << bits) - 1
            value = raw & mask
            sign = ""
            if conversion in "di":
                if value & (1 << (bits - 1)):
                    value -= 1 << bits
                if value < 0:
                    sign, value = "-", -value
                elif "+" in flags:
                    sign = "+"
                elif " " in flags:
                    sign = " "
                base, uppercase = 10, False
            else:
                base = {"u": 10, "o": 8, "x": 16, "X": 16, "p": 16}[conversion]
                uppercase = conversion in "Xp"
            digits = (
                str(value) if base == 10 else
                (format(value, "o") if base == 8 else format(value, "X" if uppercase else "x"))
            )
            if precision == 0 and value == 0:
                digits = ""
            if precision is not None:
                digits = digits.rjust(precision, "0")
            prefix = sign
            if conversion == "p":
                digits = digits.rjust(16, "0")
            elif "#" in flags and value:
                prefix += "0" if base == 8 else ("0X" if uppercase else "0x")
            field = prefix + digits
            if "0" in flags and "-" not in flags and precision is None and width > len(field):
                field = prefix + digits.rjust(width - len(prefix), "0")
        if width < 0:
            flags += "-"
            width = -width
        if width > len(field):
            field = field.ljust(width) if "-" in flags else field.rjust(width)
        output.append(field)
        if sum(map(len, output)) > maximum_characters:
            raise ValueError("wide printf output exceeds the declared bound")
    return "".join(output)


def _msvcrt_vsnwprintf(maximum_characters: int) -> ExternalCapabilityHandler:
    def handler(request, state):
        destination, capacity, format_address, argument_list = request.arguments[:4]
        if capacity > maximum_characters:
            return None
        format_string = _read_utf16(
            state.memory, format_address, maximum_characters=maximum_characters,
        )
        if format_string is None:
            return None
        cursor = int(argument_list)

        def next_argument():
            nonlocal cursor
            value = int.from_bytes(
                bytes(state.memory[cursor + offset] for offset in range(8)), "little",
            )
            cursor += 8
            return value

        try:
            formatted = _format_wide_printf(
                state.memory, format_string, next_argument,
                maximum_characters=maximum_characters,
            )
        except (KeyError, UnicodeError, ValueError):
            return None
        if formatted is None:
            return None
        if len(formatted) >= capacity:
            payload = formatted[:capacity].encode("utf-16le")
            result = (1 << 64) - 1
        else:
            payload = formatted.encode("utf-16le") + b"\x00\x00"
            result = len(formatted)
        return MachineExternalCallCompletion(
            request.request_id, result=result,
            memory_writes=(MachineExternalMemoryWrite(destination, payload),),
        )
    return handler


def _msvcrt_setlocale(code_page: int) -> ExternalCapabilityHandler:
    category_names = {
        0: "all", 1: "collate", 2: "ctype", 3: "monetary",
        4: "numeric", 5: "time",
    }

    def handler(request, state):
        category, locale_address = request.arguments[:2]
        if category not in category_names:
            return MachineExternalCallCompletion(request.request_id, result=0)
        key = f"msvcrt.locale.{category_names[category]}"
        writes = ()
        if locale_address:
            requested = _read_ascii(state.memory, locale_address)
            if requested is None:
                return None
            if requested in {"C", "POSIX"}:
                selected = "C"
            elif requested == "" or requested.casefold() in {".acp", ".ocp", ".utf-8", ".utf8"}:
                selected = f"English_United States.{code_page}"
            elif requested.casefold().startswith("english_united states."):
                selected = requested
            else:
                return MachineExternalCallCompletion(request.request_id, result=0)
            keys = tuple(category_names.values()) if category == 0 else (category_names[category],)
            writes = tuple(
                MachineExternalTextStateWrite(f"msvcrt.locale.{name}", selected)
                for name in keys
            )
        else:
            selected = state.text_state.get(key, "C")
        payload = selected.encode("ascii") + b"\x00"
        cursor = (int(state.system_state.get("windows.system_arena_cursor", 0)) + 7) & -8
        limit = int(state.system_state.get("windows.system_arena_limit", 0))
        if not cursor or cursor + len(payload) > limit:
            return MachineExternalCallCompletion(request.request_id, result=0)
        return MachineExternalCallCompletion(
            request.request_id, result=cursor,
            memory_writes=(MachineExternalMemoryWrite(cursor, payload),),
            system_writes=(
                MachineExternalStateWrite("windows.system_arena_cursor", cursor + len(payload)),
                MachineExternalStateWrite(f"{key}.pointer", cursor),
            ),
            text_writes=writes,
        )
    return handler


def _msvcrt_setjmp(request, state):
    """Capture the MSVC AMD64 ``_JUMP_BUFFER`` in guest memory.

    The second argument is the establisher frame supplied by the x64 CRT call
    sequence.  ``Rsp`` records the caller-visible stack pointer, after the
    external CALL's return address has been consumed.  Floating-point control
    words live in reversible virtual system state until they become first-class
    architectural registers.
    """

    buffer = int(request.arguments[0])
    frame = int(request.arguments[1])
    registers = state.registers
    words = (
        frame,
        registers[3],                         # RBX
        (request.stack_pointer + 8) & ((1 << 64) - 1),
        registers[5],                         # RBP
        registers[6],                         # RSI
        registers[7],                         # RDI
        registers[12], registers[13], registers[14], registers[15],
        request.return_address,
    )
    payload = bytearray(256)
    for index, value in enumerate(words):
        payload[index * 8:index * 8 + 8] = int(value).to_bytes(8, "little")
    payload[88:92] = int(state.system_state.get(
        "amd64.mxcsr", 0x1F80,
    ) & 0xFFFFFFFF).to_bytes(4, "little")
    payload[92:94] = int(state.system_state.get(
        "amd64.fpcsr", 0x027F,
    ) & 0xFFFF).to_bytes(2, "little")
    for register in range(6, 16):
        value = int(state.vector_registers[register]) & ((1 << 128) - 1)
        offset = 96 + (register - 6) * 16
        payload[offset:offset + 16] = value.to_bytes(16, "little")
    return MachineExternalCallCompletion(
        request.request_id, result=0,
        memory_writes=(MachineExternalMemoryWrite(buffer, payload),),
    )


def _read_guest_unsigned(memory, address: int, width: int) -> int:
    """Read one bounded little-endian field without manufacturing mappings."""

    if width not in (2, 4, 8):
        raise ValueError("guest metadata field width is unsupported")
    return int.from_bytes(bytes(memory[int(address) + index] for index in range(width)), "little")


def _msvcrt_local_unwind(request, state):
    """Dispatch one bounded AMD64 C termination scope from guest PE metadata.

    MSVCRT's x64 ``_local_unwind(frame, target)`` is a thin ``RtlUnwind``
    wrapper.  This capability owns the common same-frame C ``__finally``
    shape used by the Windows command processor.  The PE exception directory,
    unwind record, and scope table are read from reversible guest memory.  Any
    chained, malformed, non-C, or multi-callback shape remains pending.
    """

    frame, target = (int(request.arguments[0]), int(request.arguments[1]))
    base = int(state.system_state.get("windows.loader.image_base", 0))
    control = int(request.return_address)
    memory = state.memory
    if not base or not frame or not (base <= control and base <= target):
        return None
    try:
        if bytes(memory[base + index] for index in range(2)) != b"MZ":
            return None
        pe_offset = _read_guest_unsigned(memory, base + 0x3C, 4)
        if not 0x40 <= pe_offset <= 0x100000:
            return None
        pe = base + pe_offset
        if bytes(memory[pe + index] for index in range(4)) != b"PE\0\0":
            return None
        optional_size = _read_guest_unsigned(memory, pe + 20, 2)
        optional = pe + 24
        if _read_guest_unsigned(memory, optional, 2) != 0x20B or optional_size < 144:
            return None
        image_size = _read_guest_unsigned(memory, optional + 56, 4)
        directory_count = _read_guest_unsigned(memory, optional + 108, 4)
        if not image_size or directory_count <= 3:
            return None
        exception_rva = _read_guest_unsigned(memory, optional + 112 + 3 * 8, 4)
        exception_size = _read_guest_unsigned(memory, optional + 112 + 3 * 8 + 4, 4)
        if (
            not exception_rva or not exception_size or exception_size % 12
            or exception_size > 12 * 65536
            or exception_rva + exception_size > image_size
        ):
            return None
        control_rva = control - base
        function = None
        for offset in range(0, exception_size, 12):
            record = base + exception_rva + offset
            begin = _read_guest_unsigned(memory, record, 4)
            end = _read_guest_unsigned(memory, record + 4, 4)
            unwind_rva = _read_guest_unsigned(memory, record + 8, 4)
            if begin <= control_rva < end:
                function = (begin, end, unwind_rva)
                break
        if function is None:
            return None
        _begin, _end, unwind_rva = function
        if not 0 < unwind_rva < image_size:
            return None
        unwind = base + unwind_rva
        version_flags = memory[unwind]
        version, flags = version_flags & 7, version_flags >> 3
        code_count = memory[unwind + 2]
        if version != 1 or flags & 4 or not flags & 3:
            return None
        handler_offset = 4 + ((code_count + 1) & ~1) * 2
        if handler_offset > 516:
            return None
        language_handler = _read_guest_unsigned(memory, unwind + handler_offset, 4)
        if not 0 < language_handler < image_size:
            return None
        scope = unwind + handler_offset + 4
        scope_count = _read_guest_unsigned(memory, scope, 4)
        if not 0 < scope_count <= 1024:
            return None
        target_rva = target - base
        callbacks = []
        for index in range(scope_count):
            entry = scope + 4 + index * 16
            begin = _read_guest_unsigned(memory, entry, 4)
            end = _read_guest_unsigned(memory, entry + 4, 4)
            handler = _read_guest_unsigned(memory, entry + 8, 4)
            jump_target = _read_guest_unsigned(memory, entry + 12, 4)
            if not (0 <= begin <= end <= image_size):
                return None
            if begin <= control_rva < end and not begin <= target_rva < end:
                # JumpTarget==0 is the documented C termination/__finally
                # record. Filter/__except records are not cleanup callbacks.
                if jump_target == 0:
                    if not 0 < handler < image_size:
                        return None
                    callbacks.append(base + handler)
        if len(callbacks) > 1:
            return None
    except (KeyError, IndexError, OverflowError, ValueError):
        return None
    return MachineExternalCallCompletion(
        request.request_id,
        register_writes=(
            MachineExternalRegisterWrite(1, 1),      # RCX: abnormal termination
            MachineExternalRegisterWrite(2, frame),  # RDX: establisher frame
        ) if callbacks else (),
        guest_calls=tuple(callbacks),
    )


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


def _windows_create_thread(request, state):
    attributes, stack_size, start_address, parameter = request.arguments
    creation_flags = int(request.stack_arguments[0]) if request.stack_arguments else 0
    thread_id_pointer = (
        int(request.stack_arguments[1]) if len(request.stack_arguments) > 1 else 0
    )
    # Security descriptors and suspended startup require additional modeled
    # objects/scheduler states; leave them pending rather than ignoring them.
    if attributes or not start_address or creation_flags:
        return None
    thread_id = int(state.system_state.get("windows.thread.next_id", 0x30000))
    handle = 0xFFFFA00000000000 + thread_id
    writes = (
        (MachineExternalMemoryWrite(
            thread_id_pointer, int(thread_id).to_bytes(4, "little"),
        ),)
        if thread_id_pointer else ()
    )
    return MachineExternalCallCompletion(
        request.request_id,
        result=handle,
        memory_writes=writes,
        system_writes=(
            MachineExternalStateWrite("windows.thread.next_id", thread_id + 1),
            MachineExternalStateWrite(f"windows.handle.{handle}.kind", 1),
            MachineExternalStateWrite(f"windows.handle.{handle}.id", thread_id),
        ),
        thread_spawns=(MachineExternalThreadSpawn(
            int(start_address), int(parameter), int(stack_size),
            creation_flags, thread_id, handle,
        ),),
    )


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
    capacity = max(int(size), 1)
    reusable = []
    prefix = "windows.heap.allocation."
    suffix = ".active"
    for key, active in state.system_state.items():
        if active or not key.startswith(prefix) or not key.endswith(suffix):
            continue
        try:
            address = int(key[len(prefix):-len(suffix)])
        except ValueError:
            continue
        available = int(state.system_state.get(f"{prefix}{address}.size", 0))
        if available >= capacity:
            reusable.append((available, address))
    if reusable:
        _available, address = min(reusable)
        writes = (
            (MachineExternalMemoryWrite(address, bytes(capacity)),)
            if flags & 0x8 else ()
        )
        return MachineExternalCallCompletion(
            request.request_id,
            result=address,
            memory_writes=writes,
            system_writes=(
                MachineExternalStateWrite(f"{prefix}{address}.size", size),
                MachineExternalStateWrite(f"{prefix}{address}.active", 1),
            ),
        )
    cursor = (int(state.system_state.get("windows.system_arena_cursor", 0)) + 15) & -16
    limit = int(state.system_state.get("windows.system_arena_limit", 0))
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


def _heap_realloc(request, state):
    heap, flags, address, size = request.arguments[:4]
    if address == 0:
        return _heap_alloc(replace(request, arguments=(heap, flags, size, 0)), state)
    active_key = f"windows.heap.allocation.{address}.active"
    old_size = int(state.system_state.get(f"windows.heap.allocation.{address}.size", 0))
    if heap != 0x300 or not state.system_state.get(active_key, 0):
        return MachineExternalCallCompletion(request.request_id, result=0)
    allocated = _heap_alloc(
        replace(request, arguments=(heap, flags, size, 0)), state,
    )
    if not allocated.result:
        return allocated
    copy_size = min(old_size, int(size))
    try:
        copied = bytes(state.memory[address + index] for index in range(copy_size))
    except KeyError:
        return MachineExternalCallCompletion(request.request_id, result=0)
    writes = tuple(allocated.memory_writes)
    if copied:
        writes = (*writes, MachineExternalMemoryWrite(allocated.result, copied))
    return MachineExternalCallCompletion(
        request.request_id, result=allocated.result,
        memory_writes=writes,
        system_writes=(*allocated.system_writes, MachineExternalStateWrite(active_key, 0)),
    )


def _global_alloc(request, state):
    flags, size = request.arguments[:2]
    return _heap_alloc(
        replace(request, arguments=(0x300, 0x8 if flags & 0x40 else 0, size, 0)),
        state,
    )


def _global_free(request, state):
    address = request.arguments[0]
    completion = _heap_free(
        replace(request, arguments=(0x300, 0, address, 0)), state,
    )
    return replace(completion, result=0 if completion.result else address)


def _global_size(request, state):
    return _heap_size(
        replace(request, arguments=(0x300, 0, request.arguments[0], 0)), state,
    )


def _msvcrt_malloc(request, state):
    return _heap_alloc(
        replace(request, arguments=(0x300, 0, request.arguments[0], 0)),
        state,
    )


def _msvcrt_calloc(request, state):
    count, element_size = request.arguments[:2]
    size = count * element_size
    if count and size // count != element_size:
        return MachineExternalCallCompletion(request.request_id, result=0)
    return _heap_alloc(
        replace(request, arguments=(0x300, 0x8, size, 0)),
        state,
    )


def _msvcrt_free(request, state):
    address = request.arguments[0]
    if address == 0:
        return MachineExternalCallCompletion(request.request_id)
    completion = _heap_free(
        replace(request, arguments=(0x300, 0, address, 0)), state,
    )
    return replace(completion, result=0)


def _msvcrt_msize(request, state):
    return _heap_size(
        replace(request, arguments=(0x300, 0, request.arguments[0], 0)),
        state,
    )


def _msvcrt_realloc(request, state):
    address, size = request.arguments[:2]
    if address == 0:
        return _heap_alloc(
            replace(request, arguments=(0x300, 0, size, 0)), state,
        )
    if size == 0:
        freed = _msvcrt_free(request, state)
        return replace(freed, result=0)
    old_size = int(state.system_state.get(f"windows.heap.allocation.{address}.size", 0))
    active = state.system_state.get(f"windows.heap.allocation.{address}.active", 0)
    if not active:
        return MachineExternalCallCompletion(request.request_id, result=0)
    allocated = _heap_alloc(
        replace(request, arguments=(0x300, 0, size, 0)), state,
    )
    if not allocated.result:
        return allocated
    copy_size = min(old_size, size)
    try:
        copied = bytes(state.memory[address + index] for index in range(copy_size))
    except KeyError:
        return MachineExternalCallCompletion(request.request_id, result=0)
    return replace(
        allocated,
        memory_writes=(*allocated.memory_writes, MachineExternalMemoryWrite(
            allocated.result, copied,
        )),
        system_writes=(*allocated.system_writes, MachineExternalStateWrite(
            f"windows.heap.allocation.{address}.active", 0,
        )),
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


def _srw_lock(thread_id: int, *, shared: bool, release: bool = False, attempt: bool = False) -> ExternalCapabilityHandler:
    """Model an opaque, zero-initialized SRW lock in reversible shell state."""

    def handler(request, state):
        address = int(request.arguments[0])
        prefix = f"windows.srw_lock.{address}"
        readers = int(state.system_state.get(f"{prefix}.readers", 0))
        owner = int(state.system_state.get(f"{prefix}.owner", 0))
        if release:
            if shared:
                if readers <= 0:
                    return None
                readers -= 1
            else:
                if owner != thread_id:
                    return None
                owner = 0
            result = 0
        elif shared:
            if owner:
                return MachineExternalCallCompletion(request.request_id, result=0) if attempt else None
            readers += 1
            result = 1
        else:
            if owner or readers:
                return MachineExternalCallCompletion(request.request_id, result=0) if attempt else None
            owner = thread_id
            result = 1
        opaque_word = 1 if owner else readers << 1
        return MachineExternalCallCompletion(
            request.request_id, result=result,
            memory_writes=(MachineExternalMemoryWrite(
                address, opaque_word.to_bytes(8, "little"),
            ),),
            system_writes=(
                MachineExternalStateWrite(f"{prefix}.readers", readers),
                MachineExternalStateWrite(f"{prefix}.owner", owner),
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


def _get_std_handle(request, state):
    selector = request.arguments[0] & 0xFFFFFFFF
    handles = {0xFFFFFFF6: 0x200, 0xFFFFFFF5: 0x201, 0xFFFFFFF4: 0x202}
    return MachineExternalCallCompletion(
        request.request_id, result=handles.get(selector, (1 << 64) - 1),
    )


def _get_console_title(request, state):
    output, capacity = request.arguments[:2]
    title = state.text_state.get("windows.console.title", "")
    if capacity == 0:
        return MachineExternalCallCompletion(request.request_id, result=0)
    copied = title[:max(0, int(capacity) - 1)]
    return MachineExternalCallCompletion(
        request.request_id, result=len(copied),
        memory_writes=(MachineExternalMemoryWrite(
            output, copied.encode("utf-16le") + b"\x00\x00",
        ),),
    )


def _get_console_screen_buffer_info(request, state):
    handle, output = request.arguments[:2]
    if handle not in (0x201, 0x202):
        return MachineExternalCallCompletion(request.request_id, result=0)
    terminal = state.device_state.get("console.output", b"")
    try:
        text = terminal.decode("utf-8")
    except UnicodeDecodeError:
        text = ""
    cursor_x = len(text.rsplit("\n", 1)[-1].rstrip("\r")) % 80
    cursor_y = text.count("\n")
    window_top = max(0, cursor_y - 24)
    payload = struct.pack(
        "<hhhhHhhhhhh",
        80, 32766, cursor_x, min(cursor_y, 32765), 0x0007,
        0, window_top, 79, window_top + 24, 80, 25,
    )
    return MachineExternalCallCompletion(
        request.request_id, result=1,
        memory_writes=(MachineExternalMemoryWrite(output, payload),),
    )


def _get_largest_console_window_size(request, state):
    handle = request.arguments[0]
    return MachineExternalCallCompletion(
        request.request_id,
        result=(80 | (25 << 16)) if handle in (0x201, 0x202) else 0,
    )


def _get_console_cursor_info(request, state):
    handle, output = request.arguments[:2]
    if handle not in (0x201, 0x202):
        return MachineExternalCallCompletion(request.request_id, result=0)
    return MachineExternalCallCompletion(
        request.request_id, result=1,
        memory_writes=(MachineExternalMemoryWrite(
            output, struct.pack("<II", 25, 1),
        ),),
    )


def _query_api_set_presence(request, state):
    """Report optional API-set extensions absent without consulting the host."""

    _namespace, present = request.arguments[:2]
    return MachineExternalCallCompletion(
        request.request_id, result=1,
        memory_writes=(MachineExternalMemoryWrite(present, b"\x00"),),
    )


def _rtl_create_unicode_string_from_ascii(request, state):
    descriptor, source = request.arguments[:2]
    value = _read_ascii(state.memory, source)
    if value is None:
        return MachineExternalCallCompletion(request.request_id, result=0)
    payload = value.encode("utf-16le") + b"\x00\x00"
    cursor = (int(state.system_state.get("windows.system_arena_cursor", 0)) + 1) & -2
    limit = int(state.system_state.get("windows.system_arena_limit", 0))
    if not cursor or cursor + len(payload) > limit or len(payload) > 0xFFFF:
        return MachineExternalCallCompletion(request.request_id, result=0)
    unicode_string = struct.pack(
        "<HH4xQ", len(payload) - 2, len(payload), cursor,
    )
    return MachineExternalCallCompletion(
        request.request_id, result=1,
        memory_writes=(
            MachineExternalMemoryWrite(cursor, payload),
            MachineExternalMemoryWrite(descriptor, unicode_string),
        ),
        system_writes=(MachineExternalStateWrite(
            "windows.system_arena_cursor", cursor + len(payload),
        ),),
    )


def _rtl_free_unicode_string(request, state):
    descriptor = request.arguments[0]
    return MachineExternalCallCompletion(
        request.request_id,
        memory_writes=(MachineExternalMemoryWrite(descriptor, bytes(16)),),
    )


def _rtl_nt_status_to_dos_error(request, state):
    status = request.arguments[0] & 0xFFFFFFFF
    return MachineExternalCallCompletion(
        request.request_id,
        result={0xC0000022: 5, 0xC000007C: 1008}.get(status, 317),
    )


def _create_virtual_process(
    registry: VirtualProgramRegistry,
    *,
    current_directory: str,
) -> ExternalCapabilityHandler:
    """Resolve CreateProcessW only to a declared Turing program executor."""

    def handler(request, state):
        application_pointer, command_pointer = request.arguments[:2]
        application = _read_utf16(state.memory, application_pointer) if application_pointer else None
        command_line = _read_utf16(state.memory, command_pointer) if command_pointer else ""
        if command_line is None:
            return MachineExternalCallCompletion(request.request_id, result=0)
        arguments = split_windows_command_line(command_line)
        requested = application or (arguments[0] if arguments else "")
        if not requested:
            return MachineExternalCallCompletion(request.request_id, result=0)
        child_arguments = arguments[1:] if arguments else ()
        process_information = (
            request.stack_arguments[5] if len(request.stack_arguments) > 5 else 0
        )
        if not process_information:
            return MachineExternalCallCompletion(request.request_id, result=0)
        filesystem = state.virtual_filesystem
        active_directory = (
            filesystem.current_directory if filesystem is not None else current_directory
        )
        path_entries = tuple(
            item for item in state.environment_state.get("PATH", "").split(";") if item
        )
        deployment = registry.launch(
            requested,
            child_arguments,
            deployment_id=request.request_id,
            current_directory=active_directory,
            path_search=path_entries,
            environment=state.environment_state,
            standard_input=state.device_state.get("console.input", b""),
        )
        if deployment is None:
            return None
        handle = 0x4000000000000000 | int(request.request_id)
        thread_handle = 0x5000000000000000 | int(request.request_id)
        process_id = 0x10000 + int(request.request_id)
        thread_id = 0x20000 + int(request.request_id)
        result = deployment.result
        device_writes = tuple(
            MachineExternalDeviceWrite("console.output", payload)
            for payload in (result.standard_output, result.standard_error)
            if payload
        )
        return MachineExternalCallCompletion(
            request.request_id,
            result=1,
            memory_writes=(MachineExternalMemoryWrite(
                process_information,
                struct.pack("<QQII", handle, thread_handle, process_id, thread_id),
            ),),
            system_writes=(
                MachineExternalStateWrite(f"windows.process.{handle}.exit_code", result.exit_code),
                MachineExternalStateWrite(f"windows.process.{handle}.complete", 1),
            ),
            text_writes=(
                MachineExternalTextStateWrite(
                    f"windows.process.{handle}.requested", deployment.invocation.requested_path,
                ),
                MachineExternalTextStateWrite(
                    f"windows.process.{handle}.bundle", deployment.program.bundle_reference,
                ),
                MachineExternalTextStateWrite(
                    f"windows.process.{handle}.executor", deployment.program.executor_reference,
                ),
            ),
            device_writes=device_writes,
            deployments=(MachineExternalDeployment(
                deployment.deployment_id,
                "card-set-executor",
                deployment.invocation.requested_path,
                deployment.program.bundle_reference,
                deployment.program.executor_reference,
                result.exit_code,
                result.execution_units,
                deployment.child_tape.schema,
                deployment.child_tape.digest,
                f"child-tape:sha256:{deployment.child_tape.digest}",
            ),),
        )
    return handler


def _wait_for_virtual_process(request, state):
    handle = request.arguments[0]
    complete = state.system_state.get(f"windows.process.{handle}.complete")
    return None if complete is None else MachineExternalCallCompletion(
        request.request_id, result=0,
    )


def _wait_for_virtual_object(request, state):
    handle = int(request.arguments[0])
    timeout = int(request.arguments[1] & 0xFFFFFFFF)
    def completed(result: int):
        return MachineExternalCallCompletion(
            request.request_id,
            result=result,
            system_writes=(MachineExternalStateWrite(
                "windows.thread.waiting_request", 0,
            ),),
        )
    kind = int(state.system_state.get(f"windows.handle.{handle}.kind", 0))
    if kind == 1:
        thread_id = int(state.system_state.get(f"windows.handle.{handle}.id", 0))
        if not thread_id:
            return completed(0xFFFFFFFF)
        active = int(state.system_state.get(f"windows.thread.{thread_id}.active", 0))
        if not active:
            return completed(0)
        return (
            completed(258)
            if timeout == 0 else None
        )
    complete = state.system_state.get(f"windows.process.{handle}.complete")
    if complete is not None:
        return completed(0 if complete else 258) if complete or timeout == 0 else None
    return completed(0xFFFFFFFF)


def _get_thread_exit_code(request, state):
    handle, output = request.arguments[:2]
    thread_id = int(state.system_state.get(f"windows.handle.{handle}.id", 0))
    if (
        int(state.system_state.get(f"windows.handle.{handle}.kind", 0)) != 1
        or not thread_id
    ):
        return MachineExternalCallCompletion(request.request_id, result=0)
    active = int(state.system_state.get(f"windows.thread.{thread_id}.active", 0))
    exit_code = (
        259 if active else int(state.system_state.get(
            f"windows.thread.{thread_id}.exit_code", 0,
        )) & 0xFFFFFFFF
    )
    return MachineExternalCallCompletion(
        request.request_id,
        result=1,
        memory_writes=(MachineExternalMemoryWrite(
            int(output), exit_code.to_bytes(4, "little"),
        ),),
    )


def _get_virtual_process_exit_code(request, state):
    handle, output = request.arguments[:2]
    exit_code = state.system_state.get(f"windows.process.{handle}.exit_code")
    if exit_code is None:
        return MachineExternalCallCompletion(request.request_id, result=0)
    return MachineExternalCallCompletion(
        request.request_id, result=1,
        memory_writes=(MachineExternalMemoryWrite(
            output, int(exit_code & 0xFFFFFFFF).to_bytes(4, "little"),
        ),),
    )


def _initialize_proc_thread_attribute_list(thread_id: int) -> ExternalCapabilityHandler:
    """Model the documented size-query/initialize contract for opaque lists."""

    def handler(request, state):
        address, count, flags, size_pointer = request.arguments[:4]
        if flags != 0 or not 0 < count <= 64 or not size_pointer:
            return None
        required = 32 + int(count) * 16
        size_write = MachineExternalMemoryWrite(
            size_pointer, required.to_bytes(8, "little"),
        )
        if not address:
            return MachineExternalCallCompletion(
                request.request_id, result=0,
                memory_writes=(size_write,),
                system_writes=(MachineExternalStateWrite(
                    f"windows.thread.{thread_id}.last_error", 122,
                ),),
            )
        try:
            supplied = int.from_bytes(
                bytes(state.memory[size_pointer + index] for index in range(8)),
                "little",
            )
            bytes(state.memory[address + index] for index in range(required))
        except KeyError:
            return None
        if supplied < required:
            return MachineExternalCallCompletion(
                request.request_id, result=0,
                memory_writes=(size_write,),
                system_writes=(MachineExternalStateWrite(
                    f"windows.thread.{thread_id}.last_error", 122,
                ),),
            )
        return MachineExternalCallCompletion(
            request.request_id, result=1,
            memory_writes=(
                size_write, MachineExternalMemoryWrite(address, bytes(required)),
            ),
            system_writes=(
                MachineExternalStateWrite(
                    f"windows.proc_thread_attributes.{address}.initialized", 1,
                ),
                MachineExternalStateWrite(
                    f"windows.proc_thread_attributes.{address}.size", required,
                ),
                MachineExternalStateWrite(
                    f"windows.proc_thread_attributes.{address}.capacity", int(count),
                ),
            ),
        )
    return handler


_SUPPORTED_PROC_THREAD_ATTRIBUTES = frozenset({
    0x00020000,  # PARENT_PROCESS
    0x00060001,  # internal additive EXTENDED_FLAGS used by CreateProcessW
    0x00020002,  # HANDLE_LIST
    0x00020007,  # MITIGATION_POLICY
    0x0002000D,  # JOB_LIST
    0x00020016,  # PSEUDOCONSOLE
    0x00020019,  # MACHINE_TYPE
})


def _update_proc_thread_attribute(request, state):
    address, flags, attribute, value_pointer = request.arguments[:4]
    value_size = request.stack_arguments[0] if request.stack_arguments else 0
    previous_value = request.stack_arguments[1] if len(request.stack_arguments) > 1 else 0
    return_size = request.stack_arguments[2] if len(request.stack_arguments) > 2 else 0
    prefix = f"windows.proc_thread_attributes.{address}"
    if (
        flags != 0 or previous_value or return_size
        or attribute not in _SUPPORTED_PROC_THREAD_ATTRIBUTES
        or not state.system_state.get(f"{prefix}.initialized")
        or not value_pointer or not 0 < value_size <= 4096
    ):
        return None
    try:
        payload = bytes(
            state.memory[value_pointer + index] for index in range(int(value_size))
        )
    except KeyError:
        return None
    key = f"{prefix}.attribute.{attribute}"
    return MachineExternalCallCompletion(
        request.request_id, result=1,
        system_writes=(
            MachineExternalStateWrite(f"{key}.value_pointer", value_pointer),
            MachineExternalStateWrite(f"{key}.size", value_size),
        ),
        text_writes=(MachineExternalTextStateWrite(
            f"{key}.value_hex", payload.hex(),
        ),),
    )


def _delete_proc_thread_attribute_list(request, state):
    address = request.arguments[0]
    prefix = f"windows.proc_thread_attributes.{address}"
    size = int(state.system_state.get(f"{prefix}.size", 0))
    if not state.system_state.get(f"{prefix}.initialized") or size <= 0:
        return None
    return MachineExternalCallCompletion(
        request.request_id,
        memory_writes=(MachineExternalMemoryWrite(address, bytes(size)),),
        system_writes=(MachineExternalStateWrite(f"{prefix}.initialized", 0),),
    )


def _get_startup_info(request, state):
    output = request.arguments[0]
    if not output:
        return None
    # Windows AMD64 STARTUPINFOW: pointer fields are naturally 8-byte aligned.
    payload = bytearray(104)
    payload[0:4] = (104).to_bytes(4, "little")
    payload[60:64] = (0x100).to_bytes(4, "little")  # STARTF_USESTDHANDLES
    payload[80:88] = (0x200).to_bytes(8, "little")
    payload[88:96] = (0x201).to_bytes(8, "little")
    payload[96:104] = (0x202).to_bytes(8, "little")
    return MachineExternalCallCompletion(
        request.request_id,
        memory_writes=(MachineExternalMemoryWrite(output, bytes(payload)),),
    )


def _set_console_title(request, state):
    title = _read_utf16(state.memory, request.arguments[0])
    if title is None:
        return MachineExternalCallCompletion(request.request_id, result=0)
    return MachineExternalCallCompletion(
        request.request_id, result=1,
        text_writes=(MachineExternalTextStateWrite("windows.console.title", title),),
    )


def _set_console_input_exe_name(request, state):
    value = _read_utf16(state.memory, request.arguments[0])
    if value is None:
        return MachineExternalCallCompletion(request.request_id, result=0)
    return MachineExternalCallCompletion(
        request.request_id, result=1,
        text_writes=(MachineExternalTextStateWrite(
            "windows.console.input_exe_name", value,
        ),),
    )


def _get_console_input_exe_name(request, state):
    output, capacity = request.arguments[:2]
    value = state.text_state.get("windows.console.input_exe_name", "cmd.exe")
    if capacity <= len(value):
        return MachineExternalCallCompletion(request.request_id, result=0)
    return MachineExternalCallCompletion(
        request.request_id, result=len(value),
        memory_writes=(MachineExternalMemoryWrite(
            output, value.encode("utf-16le") + b"\x00\x00",
        ),),
    )


def _write_console(*, wide: bool, maximum_bytes: int) -> ExternalCapabilityHandler:
    def handler(request, state):
        handle, source, count, written = request.arguments[:4]
        if handle not in (0x201, 0x202) or count > maximum_bytes:
            return MachineExternalCallCompletion(request.request_id, result=0)
        try:
            if wide:
                raw = bytes(state.memory[source + index] for index in range(int(count) * 2))
                payload = raw.decode("utf-16le").encode("utf-8")
            else:
                raw = bytes(state.memory[source + index] for index in range(int(count)))
                payload = raw.decode("utf-8").encode("utf-8")
        except (KeyError, UnicodeDecodeError):
            return MachineExternalCallCompletion(request.request_id, result=0)
        existing = state.device_state.get("console.output", b"")
        if len(existing) + len(payload) > maximum_bytes:
            return MachineExternalCallCompletion(request.request_id, result=0)
        writes = (
            (MachineExternalMemoryWrite(written, int(count).to_bytes(4, "little")),)
            if written else ()
        )
        return MachineExternalCallCompletion(
            request.request_id, result=1, memory_writes=writes,
            device_writes=(MachineExternalDeviceWrite("console.output", payload),),
        )
    return handler


def _read_console(*, wide: bool, maximum_bytes: int) -> ExternalCapabilityHandler:
    """Consume shell-injected terminal bytes or leave the guest call pending."""

    def handler(request, state):
        handle, destination, count, read = request.arguments[:4]
        if handle != 0x200 or count > maximum_bytes:
            return MachineExternalCallCompletion(request.request_id, result=0)
        available = bytes(state.device_state.get("console.input", b""))
        if not available:
            return None
        try:
            if wide:
                text = available.decode("utf-8")
                consumed_text = text[:int(count)]
                consumed = consumed_text.encode("utf-8")
                payload = consumed_text.encode("utf-16le")
                units = len(consumed_text)
            else:
                consumed = available[:int(count)]
                payload = consumed
                units = len(consumed)
        except UnicodeDecodeError:
            return MachineExternalCallCompletion(request.request_id, result=0)
        writes = [MachineExternalMemoryWrite(destination, payload)]
        if read:
            writes.append(MachineExternalMemoryWrite(read, units.to_bytes(4, "little")))
        return MachineExternalCallCompletion(
            request.request_id, result=1,
            memory_writes=tuple(writes),
            device_writes=(MachineExternalDeviceWrite(
                "console.input", available[len(consumed):], append=False,
            ),),
        )
    return handler


def _read_file(maximum_bytes: int) -> ExternalCapabilityHandler:
    """Implement the standard-input subset of ReadFile as a blocking port."""

    def handler(request, state):
        handle, destination, count, read = request.arguments[:4]
        overlapped = request.stack_arguments[0] if request.stack_arguments else 0
        if handle != 0x200 or overlapped or count > maximum_bytes:
            return MachineExternalCallCompletion(request.request_id, result=0)
        available = bytes(state.device_state.get("console.input", b""))
        if not available:
            return None
        consumed = available[:int(count)]
        writes = [MachineExternalMemoryWrite(destination, consumed)]
        if read:
            writes.append(MachineExternalMemoryWrite(
                read, len(consumed).to_bytes(4, "little"),
            ))
        return MachineExternalCallCompletion(
            request.request_id, result=1,
            memory_writes=tuple(writes),
            device_writes=(MachineExternalDeviceWrite(
                "console.input", available[len(consumed):], append=False,
            ),),
        )
    return handler


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


def _get_environment_variable(environment: tuple[str, ...]) -> ExternalCapabilityHandler:
    values = {
        item.split("=", 1)[0].casefold(): item.split("=", 1)[1]
        for item in environment if "=" in item and item.split("=", 1)[0]
    }

    def handler(request, state):
        active_values = dict(values)
        active_values.update({
            key.casefold(): value for key, value in state.environment_state.items()
        })
        name_address, output, capacity = request.arguments[:3]
        name = _read_utf16(state.memory, name_address)
        if name is None or name.casefold() not in active_values:
            return MachineExternalCallCompletion(request.request_id, result=0)
        value = active_values[name.casefold()]
        if capacity <= len(value):
            return MachineExternalCallCompletion(
                request.request_id, result=len(value) + 1,
            )
        return MachineExternalCallCompletion(
            request.request_id, result=len(value),
            memory_writes=(MachineExternalMemoryWrite(
                output, value.encode("utf-16le") + b"\x00\x00",
            ),),
        )
    return handler


def _set_environment_variable(request, state):
    name = _read_utf16(state.memory, request.arguments[0])
    if not name:
        return MachineExternalCallCompletion(request.request_id, result=0)
    value_address = request.arguments[1]
    value = None if value_address == 0 else _read_utf16(state.memory, value_address)
    if value_address and value is None:
        return MachineExternalCallCompletion(request.request_id, result=0)
    return MachineExternalCallCompletion(
        request.request_id, result=1,
        environment_writes=(MachineExternalEnvironmentWrite(name, value),),
    )


def _need_current_directory_for_exe_path(
    environment: tuple[str, ...],
) -> ExternalCapabilityHandler:
    configured_names = {
        item.split("=", 1)[0].casefold()
        for item in environment if "=" in item and not item.startswith("=")
    }

    def handler(request, state):
        executable = _read_utf16(state.memory, request.arguments[0])
        if executable is None:
            return MachineExternalCallCompletion(request.request_id, result=0)
        if "\\" in executable:
            result = 1
        else:
            names = (
                {key.casefold() for key in state.environment_state}
                if state.environment_state else configured_names
            )
            result = int("nodefaultcurrentdirectoryinexepath" not in names)
        return MachineExternalCallCompletion(request.request_id, result=result)
    return handler


def _quote_windows_argument(value: str) -> str:
    if value and not any(character in value for character in ' \t"'):
        return value
    result = ['"']
    backslashes = 0
    for character in value:
        if character == "\\":
            backslashes += 1
            continue
        if character == '"':
            result.append("\\" * (backslashes * 2 + 1))
            result.append('"')
        else:
            result.append("\\" * backslashes)
            result.append(character)
        backslashes = 0
    result.append("\\" * (backslashes * 2))
    result.append('"')
    return "".join(result)


def _get_command_line(arguments: tuple[str, ...]) -> ExternalCapabilityHandler:
    command_line = " ".join(_quote_windows_argument(value) for value in arguments)
    payload = command_line.encode("utf-16le") + b"\x00\x00"

    def handler(request, state):
        existing = int(state.system_state.get("windows.command_line_w", 0))
        if existing:
            return MachineExternalCallCompletion(request.request_id, result=existing)
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
                MachineExternalStateWrite("windows.command_line_w", cursor),
            ),
        )
    return handler


def _get_current_directory(current_directory: str) -> ExternalCapabilityHandler:
    def handler(request, state):
        value = (
            virtual_path_to_windows(state.virtual_filesystem.current_directory)
            if state.virtual_filesystem is not None else current_directory
        )
        value = value.rstrip("\\/") if len(value) > 3 else value
        payload = value.encode("utf-16le") + b"\x00\x00"
        character_count = len(value)
        capacity, output = request.arguments[:2]
        if capacity <= character_count:
            return MachineExternalCallCompletion(
                request.request_id, result=character_count + 1,
            )
        return MachineExternalCallCompletion(
            request.request_id,
            result=character_count,
            memory_writes=(MachineExternalMemoryWrite(output, payload),),
        )
    return handler


def _get_module_filename(module_virtual_path: str) -> ExternalCapabilityHandler:
    def handler(request, state):
        module, output, capacity = request.arguments[:3]
        if module != 0 or capacity == 0:
            return MachineExternalCallCompletion(request.request_id, result=0)
        if state.virtual_filesystem is not None:
            try:
                state.virtual_filesystem.stat(module_virtual_path)
            except FileNotFoundError:
                return MachineExternalCallCompletion(request.request_id, result=0)
        value = virtual_path_to_windows(module_virtual_path)
        encoded = value.encode("utf-16le")
        if len(value) >= capacity:
            data = encoded[:capacity * 2]
            result = capacity
        else:
            data = encoded + b"\x00\x00"
            result = len(value)
        return MachineExternalCallCompletion(
            request.request_id, result=result,
            memory_writes=(MachineExternalMemoryWrite(output, data),),
        )
    return handler


def _get_full_path_name(request, state):
    source, capacity, output, file_part_output = request.arguments[:4]
    value = _read_utf16(state.memory, source)
    if value is None or state.virtual_filesystem is None:
        return MachineExternalCallCompletion(request.request_id, result=0)
    virtual_path = normalize_virtual_path(
        value, state.virtual_filesystem.current_directory,
    )
    windows_path = virtual_path_to_windows(virtual_path)
    required = len(windows_path) + 1
    if capacity < required:
        return MachineExternalCallCompletion(request.request_id, result=required)
    writes = [MachineExternalMemoryWrite(
        output, windows_path.encode("utf-16le") + b"\x00\x00",
    )]
    if file_part_output:
        slash = windows_path.rfind("\\")
        file_part = output + (slash + 1) * 2 if slash + 1 < len(windows_path) else 0
        writes.append(MachineExternalMemoryWrite(
            file_part_output, int(file_part).to_bytes(8, "little"),
        ))
    return MachineExternalCallCompletion(
        request.request_id, result=len(windows_path), memory_writes=tuple(writes),
    )


def _get_file_attributes(request, state):
    path = _read_utf16(state.memory, request.arguments[0])
    if path is None or state.virtual_filesystem is None:
        return MachineExternalCallCompletion(request.request_id, result=0xFFFFFFFF)
    try:
        entry = state.virtual_filesystem.stat(path)
    except (FileNotFoundError, PermissionError):
        return MachineExternalCallCompletion(request.request_id, result=0xFFFFFFFF)
    attributes = 0x10 if entry.directory else 0x80
    return MachineExternalCallCompletion(request.request_id, result=attributes)


def _set_current_directory(request, state):
    path = _read_utf16(state.memory, request.arguments[0])
    if path is None or state.virtual_filesystem is None:
        return MachineExternalCallCompletion(request.request_id, result=0)
    try:
        entry = state.virtual_filesystem.stat(path)
    except (FileNotFoundError, PermissionError):
        return MachineExternalCallCompletion(request.request_id, result=0)
    if not entry.directory:
        return MachineExternalCallCompletion(request.request_id, result=0)
    return MachineExternalCallCompletion(
        request.request_id, result=1,
        filesystem_effects=(VirtualFileEffect("chdir", path),),
    )


def _create_directory(request, state):
    path = _read_utf16(state.memory, request.arguments[0])
    if path is None or state.virtual_filesystem is None:
        return MachineExternalCallCompletion(request.request_id, result=0)
    try:
        state.virtual_filesystem.stat(path)
        return MachineExternalCallCompletion(request.request_id, result=0)
    except FileNotFoundError:
        return MachineExternalCallCompletion(
            request.request_id, result=1,
            filesystem_effects=(VirtualFileEffect("mkdir", path),),
        )


def _delete_virtual_path(*, directory: bool) -> ExternalCapabilityHandler:
    def handler(request, state):
        path = _read_utf16(state.memory, request.arguments[0])
        if path is None or state.virtual_filesystem is None:
            return MachineExternalCallCompletion(request.request_id, result=0)
        try:
            entry = state.virtual_filesystem.stat(path)
        except (FileNotFoundError, PermissionError):
            return MachineExternalCallCompletion(request.request_id, result=0)
        if entry.directory != directory:
            return MachineExternalCallCompletion(request.request_id, result=0)
        return MachineExternalCallCompletion(
            request.request_id, result=1,
            filesystem_effects=(VirtualFileEffect("remove", path),),
        )
    return handler


def _windows_find_data(filesystem, path: str) -> bytes:
    entry = filesystem.stat(path)
    name = normalize_virtual_path(path).rsplit("/", 1)[-1]
    payload = bytearray(592)
    attributes = 0x10 if entry.directory else 0x80
    payload[0:4] = attributes.to_bytes(4, "little")
    size = len(entry.data)
    payload[28:32] = (size >> 32).to_bytes(4, "little")
    payload[32:36] = (size & 0xFFFFFFFF).to_bytes(4, "little")
    encoded_name = name.encode("utf-16le")[:518]
    payload[44:44 + len(encoded_name)] = encoded_name
    return bytes(payload)


def _find_first_file(request, state, *, thread_id: int = 1):
    pattern_value = _read_utf16(state.memory, request.arguments[0])
    output = request.arguments[1]
    filesystem = state.virtual_filesystem
    if pattern_value is None or filesystem is None:
        return MachineExternalCallCompletion(
            request.request_id, result=(1 << 64) - 1,
            system_writes=(MachineExternalStateWrite(
                f"windows.thread.{thread_id}.last_error", 2,
            ),),
        )
    pattern_path = normalize_virtual_path(pattern_value, filesystem.current_directory)
    directory, pattern = pattern_path.rsplit("/", 1)
    directory = directory or "/"
    try:
        candidates = filesystem.list(directory)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        candidates = ()
    matches = tuple(
        entry.path for entry in candidates
        if fnmatch.fnmatchcase(entry.path.rsplit("/", 1)[-1].casefold(), pattern.casefold())
    )
    if not matches:
        return MachineExternalCallCompletion(
            request.request_id, result=(1 << 64) - 1,
            system_writes=(MachineExternalStateWrite(
                f"windows.thread.{thread_id}.last_error", 2,
            ),),
        )
    handle = filesystem.next_handle
    return MachineExternalCallCompletion(
        request.request_id, result=handle,
        memory_writes=(MachineExternalMemoryWrite(
            output, _windows_find_data(filesystem, matches[0]),
        ),),
        filesystem_effects=(VirtualFileEffect(
            "open", pattern_path, handle=handle, entries=matches,
        ),),
    )


def _find_next_file(request, state, *, thread_id: int = 1):
    handle_value, output = request.arguments[:2]
    filesystem = state.virtual_filesystem
    handle = filesystem.handles.get(handle_value) if filesystem is not None else None
    next_position = handle.position + 1 if handle is not None else 0
    if handle is None or next_position >= len(handle.entries):
        return MachineExternalCallCompletion(
            request.request_id, result=0,
            system_writes=(MachineExternalStateWrite(
                f"windows.thread.{thread_id}.last_error", 18,
            ),),
        )
    return MachineExternalCallCompletion(
        request.request_id, result=1,
        memory_writes=(MachineExternalMemoryWrite(
            output, _windows_find_data(filesystem, handle.entries[next_position]),
        ),),
        filesystem_effects=(VirtualFileEffect(
            "advance", handle.path, handle=handle_value,
        ),),
    )


def _find_first_file_ex(request, state, *, thread_id: int = 1):
    filename, info_level, output, search_operation = request.arguments[:4]
    search_filter = request.stack_arguments[0] if request.stack_arguments else 0
    flags = request.stack_arguments[1] if len(request.stack_arguments) > 1 else 0
    if info_level not in (0, 1) or search_operation != 0 or search_filter != 0:
        return None
    # FIND_FIRST_EX_CASE_SENSITIVE (1) is not supported by the Windows-like,
    # case-insensitive VFS. LARGE_FETCH (2) changes only host batching.
    if flags & ~0x2:
        return None
    return _find_first_file(
        replace(request, arguments=(filename, output, 0, 0)), state,
        thread_id=thread_id,
    )


def _find_close(request, state, *, thread_id: int = 1):
    handle = request.arguments[0]
    filesystem = state.virtual_filesystem
    if filesystem is None or handle not in filesystem.handles:
        return MachineExternalCallCompletion(
            request.request_id, result=0,
            system_writes=(MachineExternalStateWrite(
                f"windows.thread.{thread_id}.last_error", 6,
            ),),
        )
    return MachineExternalCallCompletion(
        request.request_id, result=1,
        filesystem_effects=(VirtualFileEffect("close", "/", handle=handle),),
    )


def _get_drive_type(request, state):
    path = _read_utf16(state.memory, request.arguments[0])
    filesystem = state.virtual_filesystem
    if path is None or filesystem is None:
        return MachineExternalCallCompletion(request.request_id, result=1)
    normalized = normalize_virtual_path(path, filesystem.current_directory)
    parts = normalized.strip("/").split("/")
    root = "/" + parts[0] if parts and len(parts[0]) == 1 and parts[0].isalpha() else normalized
    try:
        entry = filesystem.stat(root)
    except (FileNotFoundError, PermissionError):
        return MachineExternalCallCompletion(request.request_id, result=1)
    # DRIVE_FIXED denotes stable random-access storage from the guest's view;
    # it does not claim the backing storage is a native fixed disk.
    return MachineExternalCallCompletion(
        request.request_id, result=3 if entry.directory else 1,
    )


def _get_locale_info(ui_language: int) -> ExternalCapabilityHandler:
    # Stable en-US-compatible NLS values used by command parsing. Additional
    # locales must be declared as data rather than inherited from the host.
    strings = {
        0x0000000E: ".", 0x0000000F: ",", 0x00000010: "3;0",
        0x00000011: "2", 0x00000012: "1", 0x00000013: "0123456789",
        0x00000014: "$", 0x00000015: "USD", 0x00000016: ".",
        0x00000017: ",", 0x00000018: "3;0", 0x00000019: "2",
        0x0000001A: "2", 0x0000001B: "0", 0x0000001C: "0",
        0x0000001D: "/", 0x0000001E: ":", 0x0000001F: "M/d/yyyy",
        0x00000020: "dddd, MMMM d, yyyy", 0x00000021: "0",
        0x00000022: "0", 0x00000023: "0", 0x00000024: "0",
        0x00000025: "0", 0x00000026: "0", 0x00000027: "0",
        0x00000028: "AM", 0x00000029: "PM",
        0x00001001: "English (United States)",
        0x00001002: "United States", 0x00001003: "h:mm:ss tt",
        0x00001005: "0", 0x00000059: "en", 0x0000005A: "US",
        0x0000005C: "en-US", 0x00000067: "eng", 0x00000068: "USA",
    }
    for index, value in enumerate((
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    )):
        strings[0x2A + index] = value
    for index, value in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")):
        strings[0x31 + index] = value
    for index, value in enumerate((
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    )):
        strings[0x38 + index] = value
    for index, value in enumerate((
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    )):
        strings[0x44 + index] = value

    def handler(request, state):
        locale, locale_type, output, capacity = request.arguments[:4]
        if locale not in (ui_language, 0x400, 0x800):
            return MachineExternalCallCompletion(request.request_id, result=0)
        return_number = bool(locale_type & 0x20000000)
        kind = locale_type & ~(0x80000000 | 0x40000000 | 0x20000000)
        value = strings.get(kind)
        if value is None:
            return None
        if return_number:
            try:
                payload = int(value).to_bytes(4, "little")
            except ValueError:
                return MachineExternalCallCompletion(request.request_id, result=0)
            required = 2
        else:
            payload = value.encode("utf-16le") + b"\x00\x00"
            required = len(value) + 1
        if capacity == 0:
            return MachineExternalCallCompletion(request.request_id, result=required)
        if capacity < required:
            return MachineExternalCallCompletion(request.request_id, result=0)
        return MachineExternalCallCompletion(
            request.request_id, result=required,
            memory_writes=(MachineExternalMemoryWrite(output, payload),),
        )
    return handler


def _format_cmd_system_message(request, state):
    flags, _source, message_id, _language = request.arguments[:4]
    output = request.stack_arguments[0] if request.stack_arguments else 0
    capacity = request.stack_arguments[1] if len(request.stack_arguments) > 1 else 0
    messages = {
        8: "Not enough memory resources are available to process this command.\r\n",
        9040: "Microsoft Windows [Version 10.0.19045.0]\r\n",
        9009: (
            "is not recognized as an internal or external command,\r\n"
            "operable program or batch file.\r\n"
        ),
    }
    value = messages.get(int(message_id))
    # FORMAT_MESSAGE_ALLOCATE_BUFFER and unknown message tables remain
    # explicit frontiers; this owns only deterministic cmd startup text.
    if value is None or flags & 0x100 or not output or capacity <= len(value):
        return None
    return MachineExternalCallCompletion(
        request.request_id, result=len(value),
        memory_writes=(MachineExternalMemoryWrite(
            output, value.encode("utf-16le") + b"\x00\x00",
        ),),
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
    current_directory: str = "C:\\",
    module_virtual_path: str = "/program/subject.exe",
    maximum_device_bytes: int = 4 * 1024 * 1024,
    program_registry: VirtualProgramRegistry | None = None,
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
            "GetVersion": _return_value(10 | (19045 << 16)),
        },
        "api-ms-win-core-processthreads-l1-1-0.dll": {
            "GetCurrentProcessId": _return_value(process_id),
            "GetCurrentThreadId": _return_value(thread_id),
            "GetCurrentProcess": _return_value((1 << 64) - 1),
            "GetCurrentThread": _return_value((1 << 64) - 2),
            "OpenThread": _windows_open_thread(thread_id),
            "CreateThread": _windows_create_thread,
            "GetExitCodeThread": _get_thread_exit_code,
            "GetStartupInfoW": _get_startup_info,
            "InitializeProcThreadAttributeList": _initialize_proc_thread_attribute_list(
                thread_id,
            ),
            "UpdateProcThreadAttribute": _update_proc_thread_attribute,
            "DeleteProcThreadAttributeList": _delete_proc_thread_attribute_list,
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
            "GetModuleFileNameW": _get_module_filename(module_virtual_path),
        },
        "msvcrt.dll": {
            "__set_app_type": _return_value(0),
            "_setjmp": _msvcrt_setjmp,
            "_local_unwind": _msvcrt_local_unwind,
            "_vsnwprintf": _msvcrt_vsnwprintf(maximum_device_bytes // 2),
            "_initterm": _msvcrt_initterm,
            "__getmainargs": _msvcrt_getmainargs(
                tuple(arguments), tuple(environment),
            ),
            "_onexit": _msvcrt_onexit,
            "atexit": _msvcrt_atexit,
            "exit": _msvcrt_exit(),
            "_exit": _msvcrt_exit(run_callbacks=False),
            "_Exit": _msvcrt_exit(run_callbacks=False),
            "memset": _msvcrt_memset,
            "memcpy": _msvcrt_memory_copy,
            "memmove": _msvcrt_memory_copy,
            "memcmp": _msvcrt_memcmp,
            "_wcsicmp": _msvcrt_wcsicmp,
            "_wcsnicmp": _msvcrt_wcsnicmp,
            "towupper": _msvcrt_wide_case(upper=True),
            "towlower": _msvcrt_wide_case(upper=False),
            "wcslen": _msvcrt_wide_length,
            "wcschr": _msvcrt_wide_search(),
            "wcsrchr": _msvcrt_wide_search(reverse=True),
            "wcscmp": _msvcrt_wide_compare(),
            "wcsncmp": _msvcrt_wide_compare(bounded=True),
            "wcscpy": _msvcrt_wide_copy(),
            "wcsncpy": _msvcrt_wide_copy(bounded=True),
            "wcscat": _msvcrt_wide_copy(concatenate=True),
            "setlocale": _msvcrt_setlocale(input_code_page),
            **{
                name: _msvcrt_wide_classifier(name) for name in (
                    "iswalpha", "iswdigit", "iswalnum", "iswspace",
                    "iswupper", "iswlower", "iswcntrl", "iswprint",
                    "iswgraph", "iswpunct", "iswxdigit",
                )
            },
            "_get_osfhandle": _get_osfhandle,
            "time": _msvcrt_time(file_time),
            "srand": _msvcrt_srand,
            "rand": _msvcrt_rand,
            "malloc": _msvcrt_malloc,
            "calloc": _msvcrt_calloc,
            "realloc": _msvcrt_realloc,
            "free": _msvcrt_free,
            "??3@YAXPEAX@Z": _msvcrt_free,
            "??_V@YAXPEAX@Z": _msvcrt_free,
            "_msize": _msvcrt_msize,
        },
        "api-ms-win-core-errorhandling-l1-1-0.dll": {
            "SetUnhandledExceptionFilter": _set_unhandled_exception_filter,
            "SetErrorMode": _set_error_mode,
            "GetErrorMode": _get_error_mode,
            "SetLastError": _set_last_error(thread_id),
            "GetLastError": _get_last_error(thread_id),
        },
        "api-ms-win-core-apiquery-l1-1-0.dll": {
            "ApiSetQueryApiSetPresence": _query_api_set_presence,
        },
        "api-ms-win-core-handle-l1-1-0.dll": {
            "CloseHandle": _windows_close_handle,
        },
        "api-ms-win-core-synch-l1-2-0.dll": {
            "WaitForSingleObject": _wait_for_virtual_object,
        },
        "api-ms-win-core-heap-l1-1-0.dll": {
            "HeapSetInformation": _heap_set_information,
            "GetProcessHeap": _get_process_heap,
            "HeapAlloc": _heap_alloc,
            "HeapFree": _heap_free,
            "HeapSize": _heap_size,
            "HeapReAlloc": _heap_realloc,
        },
        "api-ms-win-core-heap-l2-1-0.dll": {
            "GlobalAlloc": _global_alloc,
            "GlobalFree": _global_free,
            "GlobalLock": lambda request, state: MachineExternalCallCompletion(
                request.request_id, result=request.arguments[0],
            ),
            "GlobalUnlock": _return_value(1),
            "GlobalSize": _global_size,
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
            "GetStdHandle": _get_std_handle,
            "GetConsoleTitleW": _get_console_title,
            "SetConsoleTitleW": _set_console_title,
            "WriteConsoleW": _write_console(wide=True, maximum_bytes=maximum_device_bytes),
            "WriteConsoleA": _write_console(wide=False, maximum_bytes=maximum_device_bytes),
            "ReadConsoleW": _read_console(wide=True, maximum_bytes=maximum_device_bytes),
            "ReadConsoleA": _read_console(wide=False, maximum_bytes=maximum_device_bytes),
        },
        "api-ms-win-core-console-l2-2-0.dll": {
            "GetConsoleTitleW": _get_console_title,
            "SetConsoleTitleW": _set_console_title,
            "WriteConsoleW": _write_console(wide=True, maximum_bytes=maximum_device_bytes),
            "WriteConsoleA": _write_console(wide=False, maximum_bytes=maximum_device_bytes),
            "ReadConsoleW": _read_console(wide=True, maximum_bytes=maximum_device_bytes),
            "ReadConsoleA": _read_console(wide=False, maximum_bytes=maximum_device_bytes),
        },
        "api-ms-win-core-console-l2-1-0.dll": {
            "GetConsoleScreenBufferInfo": _get_console_screen_buffer_info,
            "GetLargestConsoleWindowSize": _get_largest_console_window_size,
            "GetConsoleCursorInfo": _get_console_cursor_info,
        },
        "api-ms-win-core-localization-l1-2-0.dll": {
            "GetCPInfo": _get_cp_info,
            "GetUserDefaultLCID": _return_value(ui_language),
            "GetSystemDefaultLCID": _return_value(ui_language),
            "GetUserDefaultLangID": _return_value(ui_language & 0xFFFF),
            "GetSystemDefaultLangID": _return_value(ui_language & 0xFFFF),
            "GetThreadLocale": _return_value(ui_language),
            "GetLocaleInfoW": _get_locale_info(ui_language),
            "FormatMessageW": _format_cmd_system_message,
        },
        "api-ms-win-core-synch-l1-1-0.dll": {
            "InitializeCriticalSection": _initialize_critical_section,
            "EnterCriticalSection": _enter_critical_section(thread_id),
            "LeaveCriticalSection": _leave_critical_section(thread_id),
            "AcquireSRWLockShared": _srw_lock(thread_id, shared=True),
            "ReleaseSRWLockShared": _srw_lock(thread_id, shared=True, release=True),
            "TryAcquireSRWLockShared": _srw_lock(thread_id, shared=True, attempt=True),
            "AcquireSRWLockExclusive": _srw_lock(thread_id, shared=False),
            "ReleaseSRWLockExclusive": _srw_lock(thread_id, shared=False, release=True),
            "TryAcquireSRWLockExclusive": _srw_lock(thread_id, shared=False, attempt=True),
        },
        "api-ms-win-core-processenvironment-l1-1-0.dll": {
            # Current Windows import libraries forward this process-wide
            # pseudo-handle query through the process-environment API set.
            "GetStdHandle": _get_std_handle,
            "GetEnvironmentStringsW": _get_environment_strings(tuple(environment)),
            "FreeEnvironmentStringsW": _free_environment_strings,
            "GetCommandLineW": _get_command_line(tuple(arguments)),
            "GetCurrentDirectoryW": _get_current_directory(current_directory),
            "GetEnvironmentVariableW": _get_environment_variable(tuple(environment)),
            "SetEnvironmentVariableW": _set_environment_variable,
            "SetCurrentDirectoryW": _set_current_directory,
        },
        "api-ms-win-core-processenvironment-l1-2-0.dll": {
            "NeedCurrentDirectoryForExePathW": _need_current_directory_for_exe_path(
                tuple(environment),
            ),
        },
        "api-ms-win-core-file-l1-1-0.dll": {
            "ReadFile": _read_file(maximum_device_bytes),
            "GetFullPathNameW": _get_full_path_name,
            "GetFileAttributesW": _get_file_attributes,
            "SetCurrentDirectoryW": _set_current_directory,
            "CreateDirectoryW": _create_directory,
            "RemoveDirectoryW": _delete_virtual_path(directory=True),
            "DeleteFileW": _delete_virtual_path(directory=False),
            "GetFileType": _get_file_type,
            "FindFirstFileW": lambda request, state: _find_first_file(
                request, state, thread_id=thread_id,
            ),
            "FindFirstFileExW": lambda request, state: _find_first_file_ex(
                request, state, thread_id=thread_id,
            ),
            "FindNextFileW": lambda request, state: _find_next_file(
                request, state, thread_id=thread_id,
            ),
            "FindClose": lambda request, state: _find_close(
                request, state, thread_id=thread_id,
            ),
            "GetDriveTypeW": _get_drive_type,
        },
        "kernel32.dll": {
            "SetThreadUILanguage": _set_thread_ui_language(ui_language),
            "SetConsoleInputExeNameW": _set_console_input_exe_name,
            "GetConsoleInputExeNameW": _get_console_input_exe_name,
            "IsDebuggerPresent": _return_value(0),
            "WaitForSingleObject": _wait_for_virtual_object,
            "GetExitCodeThread": _get_thread_exit_code,
            "CloseHandle": _windows_close_handle,
        },
        "ntdll.dll": {
            "RtlCreateUnicodeStringFromAsciiz": _rtl_create_unicode_string_from_ascii,
            "RtlFreeUnicodeString": _rtl_free_unicode_string,
            # Deterministic unprivileged shell: the virtual thread has no token
            # of its own (STATUS_NO_TOKEN).
            "NtOpenThreadToken": _return_value(0xC000007C),
            "NtOpenProcessToken": _return_value(0xC0000022),
            "RtlNtStatusToDosError": _rtl_nt_status_to_dos_error,
        },
    }
    if program_registry is not None:
        create_process = _create_virtual_process(
            program_registry, current_directory=current_directory,
        )
        api_sets["api-ms-win-core-processthreads-l1-1-0.dll"].update({
            "CreateProcessW": create_process,
            "GetExitCodeProcess": _get_virtual_process_exit_code,
        })
        api_sets.setdefault("api-ms-win-core-synch-l1-2-0.dll", {}).update({
            "WaitForSingleObject": _wait_for_virtual_object,
        })
        api_sets["kernel32.dll"].update({
            "CreateProcessW": create_process,
            "GetExitCodeProcess": _get_virtual_process_exit_code,
            "WaitForSingleObject": _wait_for_virtual_object,
        })
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

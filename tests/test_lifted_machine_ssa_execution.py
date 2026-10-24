"""Lifted machine code runs through the repository's SSA reference evaluator.

A lifted function declares ``machine_state_model``; its integer handlers are
fixed-width bit patterns, not the Python frontend's logical ``And`` or float
``Trunc``, and its memory is the machine semantics' ``PagedByteMemory``.
The libpython tests execute functions lifted from the shared library, over
that library's own mapped and relocated image, and compare them with the
same library's native exports.
"""

import ctypes
import random
import sys

import pytest

from src.compiler.amd64_machine_semantics import PagedByteMemory, map_elf_image
from src.compiler.machine_code_lifting import raise_binary_region_to_ssa
from src.compiler.ssa_reference_evaluator import (
    SSAEvaluationError, SSAReferenceEvaluator,
)
from src.transmogrifier.ssa import IRModule


def _lift(encoded, registers=("rdi", "rsi")):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="subject",
        full_vocabulary_report=True, cfg_decode=True,
        argument_registers=registers,
        argument_names=tuple(f"arg{index}" for index in range(len(registers))),
    )
    assert lifting.failed_vocabulary == ()
    assert lifting.function.metadata.get("machine_state_model")
    return IRModule({"subject": lifting.function})


_STACK_TOP = 0x7FFF0000


def _with_stack(memory):
    return memory.map_zeroes(_STACK_TOP - 0x10000, 0x11000)


def _run(module, function_name="subject", memory=None, **named):
    """Bind every live-in formal by its argument name, then run."""

    function = module.functions[function_name]
    names = function.metadata["argument_names"]
    if memory is None:
        memory = _with_stack(PagedByteMemory.empty())
    arguments = {}
    for name, formal in zip(names, function.args):
        if formal.dtype == "memory":
            arguments[int(formal.id)] = memory
        elif name == "__machine_rsp":
            arguments[int(formal.id)] = _STACK_TOP
        else:
            arguments[int(formal.id)] = named.get(name, 0)
    return SSAReferenceEvaluator(module).run(function_name, arguments).returned


def test_formals_carry_the_whole_64_bit_register():
    # lea rax, [rdi + rsi]; ret.  A formal typed int32 would drop the high
    # half of each argument.
    module = _lift(b"\x48\x8d\x04\x37\xc3")
    formals = module.functions["subject"].args
    assert [item.dtype for item in formals[:2]] == ["int64", "int64"]
    returned = _run(module, arg0=1 << 40, arg1=(1 << 64) - 1)
    assert returned[0] == (1 << 40) - 1


def test_lifted_integer_handlers_are_fixed_width_bit_patterns():
    # mov eax, edi; shr eax, 3; xor eax, esi; and eax, 0x7ff; ret
    # 32-bit logical shift, then bitwise (not logical) xor and and.
    module = _lift(b"\x89\xf8\xc1\xe8\x03\x31\xf0\x25\xff\x07\x00\x00\xc3")
    rng = random.Random(3)
    for _ in range(200):
        left, right = rng.getrandbits(64), rng.getrandbits(64)
        expected = (((left & 0xFFFFFFFF) >> 3) ^ (right & 0xFFFFFFFF)) & 0x7FF
        assert _run(module, arg0=left, arg1=right)[0] & 0xFFFFFFFF == expected


_DATA = 0x5000
_DATA_BYTES = bytes.fromhex("ff80fe8f01020304")


def _data_memory():
    return _with_stack(PagedByteMemory.empty()).map_bytes(_DATA, _DATA_BYTES)


def test_lifted_loads_read_the_mapped_memory_and_refuse_unmapped_addresses():
    # mov rax, [rdi]; ret
    module = _lift(b"\x48\x8b\x07\xc3")
    returned = _run(module, memory=_data_memory(), arg0=_DATA)
    assert returned[0] == int.from_bytes(_DATA_BYTES, "little")
    with pytest.raises(SSAEvaluationError, match="at 0x9000 .*unmapped memory"):
        _run(module, memory=_data_memory(), arg0=0x9000)


@pytest.mark.parametrize(("encoded", "expected"), (
    # movzx eax, word [rdi]
    (b"\x0f\xb7\x07", 0x80FF),
    # movzx eax, byte [rdi + 1]
    (b"\x0f\xb6\x47\x01", 0x80),
    # movsx eax, word [rdi]
    (b"\x0f\xbf\x07", 0xFFFF80FF),
    # movsx rax, byte [rdi + 1]
    (b"\x48\x0f\xbe\x47\x01", (1 << 64) - 0x80),
    # movsxd rax, dword [rdi]
    (b"\x48\x63\x07", (1 << 64) - 0x70017F01),
    # movsx rax, word [rdi + 4]
    (b"\x48\x0f\xbf\x47\x04", 0x0201),
))
def test_extending_loads_read_the_source_width_from_memory(encoded, expected):
    # The source width is the token name's last width; a load of the
    # destination width would also read the neighbouring bytes.
    module = _lift(encoded + b"\xc3")
    assert _run(module, memory=_data_memory(), arg0=_DATA)[0] == expected


def test_lifted_stores_thread_a_new_memory_value():
    # mov [rdi], rsi; mov rax, [rdi]; ret
    module = _lift(b"\x48\x89\x37\x48\x8b\x07\xc3")
    memory = _data_memory()
    assert _run(module, memory=memory, arg0=_DATA, arg1=0x1122)[0] == 0x1122
    assert memory.read_unsigned(_DATA, 64) == int.from_bytes(_DATA_BYTES, "little")
    with pytest.raises(SSAEvaluationError, match="Store .*unmapped memory"):
        _run(module, memory=memory, arg0=0x9000, arg1=1)


def test_transfer_out_of_the_lifted_function_is_refused_by_name():
    # jmp rel32 to code outside the lifted region.
    lifting = raise_binary_region_to_ssa(
        b"\xe9\x10\x00\x00\x00", maximum_file_size=5, size=5,
        base_address=0x1000, name="subject",
        full_vocabulary_report=True, cfg_decode=True,
        argument_registers=("rdi",), argument_names=("arg0",),
    )
    module = IRModule({"subject": lifting.function})
    with pytest.raises(SSAEvaluationError, match="machine address linking"):
        _run(module, arg0=1)


def _shared_libpython():
    import sysconfig
    from pathlib import Path

    directory = sysconfig.get_config_var("LIBDIR")
    name = sysconfig.get_config_var("INSTSONAME")
    if not directory or not name:
        return None
    path = Path(directory) / name
    return path if path.is_file() else None


_LINUX_LIBPYTHON = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="lifts and loads a Linux libpython",
)


@pytest.fixture(scope="module")
def libpython():
    from src.compiler.elf_ingestion import parse_elf_image

    path = _shared_libpython()
    if path is None:
        pytest.skip("no shared libpython beside this interpreter")
    image = parse_elf_image(path.read_bytes())
    memory, external = map_elf_image(PagedByteMemory.empty(), image)
    # A second, uninitialised copy of the library.  The functions called
    # here touch no interpreter state, so it is safe to run them natively.
    native = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    return path, image, _with_stack(memory), external, native


@_LINUX_LIBPYTHON
def test_mapped_libpython_image_matches_the_dynamic_loader(libpython):
    _path, image, memory, external, native = libpython
    rva = image.export_by_name("_PyTime_AsMicroseconds").rva
    bias = ctypes.cast(native._PyTime_AsMicroseconds, ctypes.c_void_p).value - rva
    relro = next(item for item in image.sections if item.name == ".data.rel.ro")
    checked = 0
    for record in image.relocation_records:
        if not relro.contains_rva(record.offset) or record.type != 8:
            continue
        loaded = ctypes.c_uint64.from_address(bias + record.offset).value
        assert memory.read_unsigned(record.offset, 64) == loaded - bias
        checked += 1
    assert checked > 1000
    # Slots the loader fills from other modules point at unmapped addresses.
    assert {item.relocation_type for item in external} >= {6, 7}
    with pytest.raises(KeyError):
        memory.read_unsigned(external[0].target_address, 8)
    with pytest.raises(KeyError):
        memory.read_unsigned(0, 8)


@_LINUX_LIBPYTHON
@pytest.mark.parametrize("symbol", (
    "_PyUnicode_ToLowercase", "_PyUnicode_IsAlpha", "_PyUnicode_ToDecimalDigit",
))
def test_lifted_unicode_table_lookups_match_the_native_export(libpython, symbol):
    from src.compiler.cpython_compile_ssa import lift_elf_export_to_ssa

    path, _image, memory, _external, native = libpython
    function = getattr(native, symbol)
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_uint32]
    lifted = lift_elf_export_to_ssa(path, root_symbol=symbol)
    assert lifted.blockers == ()
    rng = random.Random(symbol)
    points = (
        *range(0x300), 0x10FFFF, 0x110000, 0xFFFFFFFF,
        *(rng.randrange(0x110000) for _ in range(300)),
    )
    for point in points:
        returned = _run(
            lifted.module, lifted.root_function, memory=memory, arg0=point,
        )[0] & 0xFFFFFFFF
        signed = returned - (1 << 32) if returned >> 31 else returned
        assert signed == function(point), hex(point)


@_LINUX_LIBPYTHON
def test_lifted_pytime_as_microseconds_matches_the_native_export(libpython):
    from src.compiler.cpython_compile_ssa import lift_elf_export_to_ssa

    path, _image, _memory, _external, native = libpython
    native = native._PyTime_AsMicroseconds
    native.restype = ctypes.c_int64
    native.argtypes = [ctypes.c_int64, ctypes.c_int]

    lifted = lift_elf_export_to_ssa(path, root_symbol="_PyTime_AsMicroseconds")
    assert lifted.blockers == ()

    rng = random.Random(1)
    times = [
        0, 1, -1, 999, 1000, 1001, -999, -1000, -1001, 1500, -1500, 2500,
        -2500, 1 << 62, -(1 << 62), (1 << 63) - 1, -(1 << 63),
        *(rng.getrandbits(64) - (1 << 63) for _ in range(40)),
        *(rng.randrange(-10 ** 7, 10 ** 7) for _ in range(40)),
    ]
    for time in times:
        # _PyTime_ROUND_FLOOR, CEILING, HALF_EVEN, UP
        for mode in range(4):
            returned = _run(
                lifted.module, lifted.root_function,
                arg0=time & ((1 << 64) - 1), arg1=mode,
            )[0]
            signed = returned - (1 << 64) if returned >> 63 else returned
            assert signed == native(time, mode), (time, mode)

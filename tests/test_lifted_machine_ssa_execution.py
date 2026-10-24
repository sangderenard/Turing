"""Lifted machine code runs through the repository's SSA reference evaluator.

A lifted function declares ``machine_state_model``; its integer handlers are
fixed-width bit patterns, not the Python frontend's logical ``And`` or float
``Trunc``.  The final test executes a function lifted from libpython and
compares it with the same library's native export.
"""

import ctypes
import random
import sys

import pytest

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


def _run(module, function_name="subject", **named):
    """Bind every live-in formal by its argument name, then run."""

    function = module.functions[function_name]
    names = function.metadata["argument_names"]
    arguments = {}
    for name, formal in zip(names, function.args):
        if formal.dtype == "memory":
            continue
        if name == "__machine_rsp":
            arguments[int(formal.id)] = 0x7FFF0000
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


def test_lifted_memory_access_is_refused_by_name():
    # mov rax, [rdi]; ret
    module = _lift(b"\x48\x8b\x07\xc3")
    with pytest.raises(SSAEvaluationError, match="'Load' in lifted code"):
        _run(module, arg0=0x1000)


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


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="lifts and loads a Linux libpython",
)
def test_lifted_pytime_as_microseconds_matches_the_native_export():
    from src.compiler.cpython_compile_ssa import lift_elf_export_to_ssa

    path = _shared_libpython()
    if path is None:
        pytest.skip("no shared libpython beside this interpreter")
    # The same image is lifted and executed natively.  _PyTime_AsMicroseconds
    # touches no interpreter state, so a second copy of the library is safe.
    native = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)._PyTime_AsMicroseconds
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

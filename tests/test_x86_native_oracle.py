"""The amd64 reference semantics against the CPU that runs the tests.

Each case runs one instruction natively, from a register file and CF the
test chooses, and through ``default_effect_handlers``; registers, the
CF/PF/AF/ZF/SF/OF flags and XMM0-1 must agree.  This is the oracle the
hand-written semantics never had: the SBB overflow flag was wrong for
every form until this compared it with hardware.
"""

import ctypes
import mmap
import platform
import random
import struct
import sys

import pytest

from src.compiler.amd64_machine_semantics import default_effect_handlers
from src.compiler.machine_execution import MachineExecutionState
from src.compiler.machine_reference_vocabulary import X86ReferenceDecoder


pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in {"x86_64", "amd64"} or sys.platform == "win32",
    reason="runs instructions on an x86-64 POSIX host",
)


@pytest.fixture(scope="module")
def page():
    try:
        mapped = mmap.mmap(
            -1, 1 << 16,
            prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
        )
    except (OSError, AttributeError) as error:
        pytest.skip(f"no executable mapping: {error}")
    return mapped, ctypes.addressof(ctypes.c_char.from_buffer(mapped))


PAGE = 1 << 16
DATA = 0x8000  # regs in 0..15 *8, flags 128, saved rsp 136, xmm0 144, xmm1 160, outregs 256.., outflags 384, outxmm 392/408, frame 512
def rip(at_code, target): return struct.pack('<i', target - at_code)

def build(instr):
    code = bytearray()
    def emit(b): code.extend(b)
    def riprel(prefix_opcode, reg, target):
        # prefix_opcode bytes then modrm(00 reg 101) then disp32 relative to end
        rex = 0x48 | (0x04 if reg & 8 else 0)
        start = len(code)
        b = bytes([rex]) + prefix_opcode + bytes([((reg & 7) << 3) | 5])
        emit(b + rip(start + len(b) + 4, target))
    for r in (3, 5, 12, 13, 14, 15):  # push callee-saved
        emit(bytes([0x41, 0x50 + (r & 7)]) if r >= 8 else bytes([0x50 + r]))
    riprel(b'\x89', 4, DATA + 136)          # mov [saved], rsp
    start = len(code); b = b'\xff\x35'; emit(b + rip(start + 6, DATA + 128)); emit(b'\x9d')  # push [flags]; popfq
    for x, off in ((0, 144), (1, 160)):    # movdqu xmm, [rip]
        start = len(code); b = b'\xf3\x0f\x6f' + bytes([(x << 3) | 5]); emit(b + rip(start + len(b) + 4, DATA + off))
    for r in range(16):
        if r == 4: continue
        riprel(b'\x8b', r, DATA + 8 * r)    # mov r, [in]
    emit(instr)
    for r in range(16):
        riprel(b'\x89', r, DATA + 256 + 8 * r)  # mov [out], r (includes rsp)
    for x, off in ((0, 392), (1, 408)):
        start = len(code); b = b'\xf3\x0f\x7f' + bytes([(x << 3) | 5]); emit(b + rip(start + len(b) + 4, DATA + off))
    riprel(b'\x8b', 4, DATA + 136)          # mov rsp, [saved]
    emit(b'\x9c\x58')                        # pushfq; pop rax
    riprel(b'\x89', 0, DATA + 384)
    for r in reversed((3, 5, 12, 13, 14, 15)):
        emit(bytes([0x41, 0x58 + (r & 7)]) if r >= 8 else bytes([0x58 + r]))
    emit(b'\xc3')
    return bytes(code)

def native(buf, base, instr, regs, cf, xmm):
    code = build(instr)
    buf[0:len(code)] = code
    for r in range(16): buf[DATA + 8*r:DATA + 8*r + 8] = struct.pack('<Q', regs[r])
    buf[DATA+128:DATA+136] = struct.pack('<Q', 0x202 | (1 if cf else 0))
    for i, off in enumerate((144, 160)): buf[DATA+off:DATA+off+16] = xmm[i].to_bytes(16, 'little')
    ctypes.CFUNCTYPE(None)(base)()
    out = [struct.unpack('<Q', buf[DATA+256+8*r:DATA+264+8*r])[0] for r in range(16)]
    flags = struct.unpack('<Q', buf[DATA+384:DATA+392])[0]
    ox = [int.from_bytes(buf[DATA+off:DATA+off+16], 'little') for off in (392, 408)]
    return out, flags, ox

def model(instr, regs, cf, xmm, memory):
    ins = X86ReferenceDecoder().decode_cfg_report(instr + b'\xc3', base_address=0x1000).instructions[0]
    state = MachineExecutionState(pc=0x1000, registers=tuple(regs), vector_registers=tuple(xmm) + (0,)*14,
                                  flags=0x202 | (1 if cf else 0), memory=memory)
    out = default_effect_handlers()[int(ins.semantic)](state, ins)
    return list(out.registers), out.flags, list(out.vector_registers[:2]), ins.token.name

FLAGS = 0x1 | 0x4 | 0x10 | 0x40 | 0x80 | 0x800
def check(buf, base, instr, n=200, frame=False):
    rng = random.Random(7); bad = 0; tok = None
    for _ in range(n):
        regs = [rng.choice((0, 1, 0xffffffffffffffff, 0x8000000000000000, 0x7fffffff, 0x7fffffffffffffff, 0x7f, 0x80, rng.getrandbits(64), rng.getrandbits(8))) for _ in range(16)]
        cf = rng.random() < 0.5
        xmm = [rng.getrandbits(128), rng.getrandbits(128)]
        memory = {}
        if frame:
            saved = rng.getrandbits(64)
            regs[5] = base + DATA + 512
            buf[DATA+512:DATA+520] = struct.pack('<Q', saved)
            memory = {regs[5] + i: b for i, b in enumerate(struct.pack('<Q', saved))}
        regs[4] = 0  # rsp is not an input
        nr, nf, nx = native(buf, base, instr, regs, cf, xmm)
        mr, mf, mx, tok = model(instr, regs, cf, xmm, memory)
        keys = [r for r in range(16) if r != 4] + ([4] if frame else [])
        if any(nr[r] != mr[r] for r in keys) or (nf & FLAGS) != (mf & FLAGS) or nx != mx:
            bad += 1
    return tok, bad



@pytest.mark.parametrize(("encoded", "token"), (
    (b"\x48\x09\xf0", "OR_RM64_R64"),
    (b"\x08\xd0", "OR_RM8_R8"),
    (b"\x44\x08\xc2", "OR_RM8_R8"),
    (b"\x19\xc0", "SBB_RM32_R32"),
    (b"\x45\x19\xff", "SBB_RM32_R32"),
    (b"\x19\xd1", "SBB_RM32_R32"),
    (b"\x48\x19\xd1", "SBB_RM64_R64"),
    (b"\x1b\xc0", "SBB_R32_RM32"),
    (b"\x48\x1b\xc2", "SBB_R64_RM64"),
    (b"\x1a\xc2", "SBB_R8_RM8"),
    (b"\x66\x0f\xef\xc1", "PXOR_XMM_XMMM128"),
    (b"\xc9", "LEAVE"),
    (b"\x66\x0f\xd6\xc8", "MOVQ_XMMM64_XMM"),
    (b"\x48\x0f\x4c\xc2", "CMOVL_R64_RM64"),
    (b"\x83\xd8\xff", "SBB_RM32_IMM8"),
    (b"\x41\x83\xde\xff", "SBB_RM32_IMM8"),
    (b"\x48\x83\xd8\xff", "SBB_RM64_IMM8"),
    (b"\x48\x83\xda\x05", "SBB_RM64_IMM8"),
    (b"\x66\x0f\xfa\xc1", "PSUBD_XMM_XMMM128"),
    (b"\x66\x0f\xfe\xc1", "PADDD_XMM_XMMM128"),
    (b"\x0f\x9d\xc0", "SETGE_RM8"),
    (b"\x66\x2e\x0f\x1f\x84\x00\x00\x00\x00\x00", "NOP_RM_CS"),
    (b"\x48\x83\xd6\x00", "ADC_RM64_IMM8"),
    (b"\x48\x83\xd1\xff", "ADC_RM64_IMM8"),
    (b"\x83\xd2\x01", "ADC_RM32_IMM8"),
    (b"\x41\x83\xd0\x7f", "ADC_RM32_IMM8"),
    (b"\x66\x41\x0f\xbe\xcc", "MOVSX_R16_RM8"),
    (b"\x66\x0f\xbe\xc1", "MOVSX_R16_RM8"),
    (b"\x20\xc2", "AND_RM8_R8"),
    (b"\x44\x20\xc0", "AND_RM8_R8"),
    (b"\x48\x0f\xba\xea\x3f", "BTS_RM64_IMM8"),
    (b"\x48\x0f\xba\xe9\x05", "BTS_RM64_IMM8"),
    (b"\x48\x0f\xba\xf1\x3f", "BTR_RM64_IMM8"),
    (b"\x48\xf7\xc2\x00\x00\x00\x80", "TEST_RM64_IMM32"),
    (b"\x48\xf7\xc0\xff\x00\x00\x00", "TEST_RM64_IMM32"),
    (b"\x66\x41\x81\xfc\xff\x00", "CMP_RM16_IMM16"),
    (b"\x66\x81\xf9\x00\x80", "CMP_RM16_IMM16"),
))
def test_reference_semantics_agree_with_the_cpu(page, encoded, token):
    buf, base = page
    decoded, bad = check(buf, base, encoded, frame=token == "LEAVE")
    assert decoded == token
    assert bad == 0

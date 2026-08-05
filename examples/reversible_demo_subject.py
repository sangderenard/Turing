"""Build the redistributable AMD64 PE subject shipped with the web demo.

The fixture is intentionally authored as bytes rather than copied from a host
operating system.  Its only runtime-described function contains ``NOP; RET``;
that is enough to demonstrate fetch, forward execution, rewind, register/page
display, and a clean top-level return while keeping every published byte ours.
"""

from __future__ import annotations

import struct


def build_reversible_demo_subject() -> bytes:
    """Return a deterministic two-section PE32+ AMD64 executable image."""
    image = bytearray(0x800)
    image[:2] = b"MZ"
    struct.pack_into("<I", image, 0x3C, 0x80)
    image[0x80:0x84] = b"PE\0\0"
    coff = 0x84
    struct.pack_into("<HHIIIHH", image, coff, 0x8664, 2, 0, 0, 0, 0xF0, 0x22)
    optional = coff + 20
    struct.pack_into("<H", image, optional, 0x20B)
    struct.pack_into("<I", image, optional + 16, 0x1000)
    struct.pack_into("<Q", image, optional + 24, 0x140000000)
    struct.pack_into("<I", image, optional + 108, 16)
    struct.pack_into("<II", image, optional + 112 + 3 * 8, 0x2000, 12)
    sections = optional + 0xF0
    image[sections:sections + 8] = b".text\0\0\0"
    struct.pack_into("<IIII", image, sections + 8, 2, 0x1000, 0x200, 0x400)
    struct.pack_into("<I", image, sections + 36, 0x60000020)
    pdata = sections + 40
    image[pdata:pdata + 8] = b".pdata\0\0"
    struct.pack_into("<IIII", image, pdata + 8, 12, 0x2000, 0x200, 0x600)
    struct.pack_into("<I", image, pdata + 36, 0x40000040)
    image[0x400:0x402] = b"\x90\xc3"  # NOP; RET
    struct.pack_into("<III", image, 0x600, 0x1000, 0x1002, 0)
    return bytes(image)


if __name__ == "__main__":
    import sys

    sys.stdout.buffer.write(build_reversible_demo_subject())

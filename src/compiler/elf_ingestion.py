"""Read an ELF64 AMD64 image into the catalogue the binary lifter consumes.

The lifter was written against ``PEImage``: sections addressed by RVA, exact
function ranges from ``.pdata``, named exports, and import slots whose
contents belong to another module.  An ELF image states each of these
facts too, in its own records, and this reader translates record for
record rather than approximating any of them:

- The address namespace is the image's own virtual addresses with an image
  base of zero.  ``rva == vaddr`` for both ``ET_DYN`` (based at zero) and
  ``ET_EXEC`` (absolute), so ``address - image_base`` is unchanged.
- Function ranges are the ``.eh_frame`` FDEs (``pc_begin``/``pc_range``),
  the unwind table every x86-64 System V object carries, as ``.pdata`` is
  on Windows.  An exported symbol's ``st_size`` is not used to invent a
  range where no FDE exists.
- Exports are the defined, global or weak ``STT_FUNC``/``STT_GNU_IFUNC``
  symbols of ``.dynsym``: what another module may bind to.
- Import slots are the GOT entries named by ``R_X86_64_JUMP_SLOT`` and
  ``R_X86_64_GLOB_DAT`` relocations against undefined symbols.  An ELF
  import names a symbol, not a library; the library is the one a
  ``.gnu.version_r`` entry assigns to the symbol's version, else ``*``
  (any loaded module, in the loader's search order).

- Dynamic relocations are kept whole in ``relocation_records`` (type,
  slot, addend, symbol) so a loader of these bytes can fill each slot the
  way ``ld.so`` would.

Nothing here loads or executes the image.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct

from .binary_ingestion import BinaryFormatError, PERuntimeFunction


class ELFType(IntEnum):
    EXEC = 2
    DYN = 3


_EM_X86_64 = 62
_SHT_NOBITS = 8
_SHT_DYNSYM = 11
_SHT_RELA = 4
_SHT_GNU_VERNEED = 0x6FFFFFFE
_SHT_GNU_VERSYM = 0x6FFFFFFF
_SHF_EXECINSTR = 0x4
_STT_FUNC = 2
_STT_GNU_IFUNC = 10
_STB_GLOBAL = 1
_STB_WEAK = 2
_SHN_UNDEF = 0
_R_X86_64_GLOB_DAT = 6
_R_X86_64_JUMP_SLOT = 7


@dataclass(frozen=True, slots=True)
class ELFSection:
    name: str
    type: int
    flags: int
    virtual_address: int
    raw_offset: int
    raw_size: int
    link: int
    entry_size: int

    @property
    def executable(self) -> bool:
        return bool(self.flags & _SHF_EXECINSTR)

    @property
    def virtual_size(self) -> int:
        return self.raw_size

    def contains_rva(self, rva: int) -> bool:
        return (
            self.virtual_address != 0 and self.raw_size > 0
            and self.virtual_address <= rva < self.virtual_address + self.raw_size
        )

    def file_offset_for_rva(self, rva: int) -> int | None:
        if self.type == _SHT_NOBITS or not self.contains_rva(rva):
            return None
        return self.raw_offset + (rva - self.virtual_address)


@dataclass(frozen=True, slots=True)
class ELFExportSymbol:
    name: str
    rva: int
    size: int
    forwarder: None = None

    @property
    def display_name(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class ELFImportSymbol:
    """One GOT slot whose value the loader binds to another module."""

    library: str
    name: str
    iat_rva: int
    relocation_type: int

    @property
    def display_name(self) -> str:
        return f"{self.library}!{self.name}"


@dataclass(frozen=True, slots=True)
class ELFDynamicRelocation:
    """One ``Elf64_Rela`` record of the dynamic relocation tables.

    ``symbol_value`` is the image address of the symbol when this image
    defines it as data or a plain function, and None when the loader must
    supply it: an undefined symbol, or an ``STT_GNU_IFUNC`` whose slot holds
    what its resolver returns at load time.
    """

    type: int
    offset: int
    addend: int
    symbol_name: str
    symbol_value: int | None


@dataclass(frozen=True, slots=True)
class ELFImage:
    type: ELFType
    image_base: int
    entrypoint_rva: int
    soname: str | None
    needed: tuple[str, ...]
    sections: tuple[ELFSection, ...]
    runtime_functions: tuple[PERuntimeFunction, ...]
    exports: tuple[ELFExportSymbol, ...]
    imports: tuple[ELFImportSymbol, ...]
    relocations: tuple[tuple[int, int], ...]
    lowest_load_address: int
    encoded: bytes
    delay_imports: tuple[()] = ()
    relocation_records: tuple[ELFDynamicRelocation, ...] = ()

    @property
    def is_shared_library(self) -> bool:
        return self.type is ELFType.DYN

    def rva_for_loaded_address(self, address: int, mapped_base: int) -> int:
        """The image address of ``address`` in a copy mapped at ``mapped_base``.

        ``mapped_base`` is the start of the image's first mapping (what
        ``dladdr`` reports as ``dli_fbase``); the load bias is its distance
        from the lowest PT_LOAD page, zero for an ``ET_EXEC`` image.
        """

        return int(address) - (int(mapped_base) - self.lowest_load_address)

    def section_for_rva(self, rva: int) -> ELFSection | None:
        matches = tuple(
            section for section in self.sections if section.contains_rva(rva)
        )
        return matches[0] if len(matches) == 1 else None

    def file_offset_for_rva(self, rva: int) -> int | None:
        section = self.section_for_rva(rva)
        return None if section is None else section.file_offset_for_rva(rva)

    def runtime_function_for_rva(self, rva: int) -> PERuntimeFunction | None:
        functions = self.runtime_functions
        low, high = 0, len(functions)
        while low < high:
            middle = (low + high) // 2
            if functions[middle].begin_rva <= rva:
                low = middle + 1
            else:
                high = middle
        if low and functions[low - 1].contains_rva(rva):
            return functions[low - 1]
        return None

    def export_by_name(self, name: str) -> ELFExportSymbol | None:
        matches = tuple(item for item in self.exports if item.name == str(name))
        if len({item.rva for item in matches}) > 1:
            raise BinaryFormatError(f"conflicting ELF export name {name!r}")
        return matches[0] if matches else None


def _string(data: bytes, offset: int) -> str:
    end = data.find(b"\0", offset)
    if end < 0:
        raise BinaryFormatError("unterminated ELF string")
    return data[offset:end].decode("utf-8", "replace")


def _uleb(data: bytes, offset: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return value, offset


def _sleb(data: bytes, offset: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            if byte & 0x40:
                value -= 1 << shift
            return value, offset


def _encoded_pointer(
    data: bytes, offset: int, encoding: int, field_address: int,
    *, apply_base: bool = True,
) -> tuple[int, int]:
    """Read one DW_EH_PE pointer: low nibble is the format, high the base."""

    if encoding == 0xFF:
        raise BinaryFormatError("DW_EH_PE_omit pointer has no value")
    form = encoding & 0x0F
    if form == 0x00:
        value, end = struct.unpack_from("<Q", data, offset)[0], offset + 8
    elif form == 0x01:
        value, end = _uleb(data, offset)
    elif form == 0x02:
        value, end = struct.unpack_from("<H", data, offset)[0], offset + 2
    elif form == 0x03:
        value, end = struct.unpack_from("<I", data, offset)[0], offset + 4
    elif form == 0x04:
        value, end = struct.unpack_from("<Q", data, offset)[0], offset + 8
    elif form == 0x09:
        value, end = _sleb(data, offset)
    elif form == 0x0A:
        value, end = struct.unpack_from("<h", data, offset)[0], offset + 2
    elif form == 0x0B:
        value, end = struct.unpack_from("<i", data, offset)[0], offset + 4
    elif form == 0x0C:
        value, end = struct.unpack_from("<q", data, offset)[0], offset + 8
    else:
        raise BinaryFormatError(f"unsupported DW_EH_PE format {form:#x}")
    application = encoding & 0x70
    if apply_base and application == 0x10:
        value += field_address
    elif apply_base and application not in {0x00, 0x10}:
        raise BinaryFormatError(
            f"unsupported DW_EH_PE application {application:#x}"
        )
    if encoding & 0x80:
        raise BinaryFormatError("indirect DW_EH_PE pointer in an FDE range")
    return value & 0xFFFFFFFFFFFFFFFF, end


def _eh_frame_functions(
    data: bytes, section: ELFSection,
) -> tuple[PERuntimeFunction, ...]:
    """Every FDE range of ``.eh_frame``; ``unwind_info_rva`` is the FDE."""

    start = section.raw_offset
    end = start + section.raw_size
    cie_encodings: dict[int, int] = {}
    functions: list[PERuntimeFunction] = []
    offset = start
    while offset + 4 <= end:
        length, = struct.unpack_from("<I", data, offset)
        if length == 0:
            break
        header = 4
        if length == 0xFFFFFFFF:
            length, = struct.unpack_from("<Q", data, offset + 4)
            header = 12
        body = offset + header
        record_end = body + length
        if record_end > end:
            raise BinaryFormatError(".eh_frame record runs past its section")
        identifier, = struct.unpack_from("<I", data, body)
        cursor = body + 4
        if identifier == 0:
            version = data[cursor]
            cursor += 1
            augmentation = _string(data, cursor)
            cursor += len(augmentation.encode("utf-8")) + 1
            _code_alignment, cursor = _uleb(data, cursor)
            _data_alignment, cursor = _sleb(data, cursor)
            if version == 1:
                cursor += 1
            else:
                _return_register, cursor = _uleb(data, cursor)
            encoding = 0x00
            if augmentation.startswith("z"):
                _length, cursor = _uleb(data, cursor)
                for letter in augmentation[1:]:
                    if letter == "R":
                        encoding = data[cursor]
                        cursor += 1
                    elif letter == "P":
                        personality = data[cursor]
                        cursor += 1
                        _value, cursor = _encoded_pointer(
                            data, cursor, personality & 0x7F,
                            section.virtual_address + (cursor - start),
                            apply_base=False,
                        )
                    elif letter == "L":
                        cursor += 1
                    elif letter not in {"S", "B", "G"}:
                        raise BinaryFormatError(
                            f"unsupported CIE augmentation {augmentation!r}"
                        )
            cie_encodings[offset] = encoding
        else:
            cie = body - identifier
            if cie not in cie_encodings:
                raise BinaryFormatError(
                    f"FDE at {offset:#x} names no preceding CIE"
                )
            encoding = cie_encodings[cie]
            begin, cursor = _encoded_pointer(
                data, cursor, encoding,
                section.virtual_address + (cursor - start),
            )
            size, cursor = _encoded_pointer(
                data, cursor, encoding & 0x0F, 0, apply_base=False,
            )
            if size:
                functions.append(PERuntimeFunction(
                    int(begin), int(begin + size),
                    section.virtual_address + (offset - start),
                ))
        offset = record_end
    functions.sort(key=lambda item: item.begin_rva)
    for previous, current in zip(functions, functions[1:]):
        if current.begin_rva < previous.end_rva:
            raise BinaryFormatError(
                f"overlapping FDE ranges at {previous.begin_rva:#x} "
                f"and {current.begin_rva:#x}"
            )
    return tuple(functions)


def parse_elf_image(encoded: bytes) -> ELFImage:
    """Parse a little-endian ELF64 AMD64 executable or shared object."""

    data = bytes(encoded)
    if data[:4] != b"\x7fELF":
        raise BinaryFormatError("not an ELF image")
    if data[4] != 2 or data[5] != 1:
        raise BinaryFormatError("only little-endian ELF64 images are read")
    (
        file_type, machine, _version, entry, program_offset, section_offset,
        _flags, _header_size, program_entry_size, program_count,
        section_entry_size, section_count, section_names,
    ) = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
    load_addresses = []
    for index in range(program_count):
        (
            segment_type, _segment_flags, _offset, virtual_address,
            _physical, _file_size, _memory_size, alignment,
        ) = struct.unpack_from(
            "<IIQQQQQQ", data, program_offset + index * program_entry_size,
        )
        if segment_type == 1:
            load_addresses.append(
                virtual_address & ~(max(int(alignment), 1) - 1)
            )
    if machine != _EM_X86_64:
        raise BinaryFormatError(f"ELF machine {machine} is not x86-64")
    if file_type not in {int(ELFType.EXEC), int(ELFType.DYN)}:
        raise BinaryFormatError(f"ELF type {file_type} is not a loadable image")
    if section_offset == 0 or section_count == 0:
        raise BinaryFormatError("ELF image has no section headers")
    raw_sections = [
        struct.unpack_from(
            "<IIQQQQIIQQ", data, section_offset + index * section_entry_size,
        )
        for index in range(section_count)
    ]
    name_table = raw_sections[section_names][4]
    sections = tuple(
        ELFSection(
            _string(data, name_table + name), kind, flags, address, offset,
            size, link, entry_size,
        )
        for (
            name, kind, flags, address, offset, size, link, _info,
            _alignment, entry_size,
        ) in raw_sections
    )

    def by_name(name: str) -> ELFSection | None:
        return next((item for item in sections if item.name == name), None)

    dynsym_index = next(
        (index for index, item in enumerate(sections) if item.type == _SHT_DYNSYM),
        None,
    )
    symbols: list[tuple[str, int, int, int, int]] = []
    if dynsym_index is not None:
        table = sections[dynsym_index]
        strings = sections[table.link].raw_offset
        for position in range(
            table.raw_offset, table.raw_offset + table.raw_size, table.entry_size,
        ):
            name, info, _other, index, value, size = struct.unpack_from(
                "<IBBHQQ", data, position,
            )
            symbols.append((_string(data, strings + name), info, index, value, size))

    exports = tuple(
        ELFExportSymbol(name, int(value), int(size))
        for name, info, index, value, size in symbols
        if name and index != _SHN_UNDEF
        and (info >> 4) in {_STB_GLOBAL, _STB_WEAK}
        and (info & 0x0F) in {_STT_FUNC, _STT_GNU_IFUNC}
    )

    # Version needs: which library supplies each required version index.
    version_library: dict[int, str] = {}
    verneed = next(
        (item for item in sections if item.type == _SHT_GNU_VERNEED), None,
    )
    if verneed is not None:
        strings = sections[verneed.link].raw_offset
        cursor = verneed.raw_offset
        while True:
            _version, count, file_name, auxiliary, next_entry = struct.unpack_from(
                "<HHIII", data, cursor,
            )
            library = _string(data, strings + file_name)
            entry_cursor = cursor + auxiliary
            for _ in range(count):
                _hash, _flags, other, _name, next_auxiliary = struct.unpack_from(
                    "<IHHII", data, entry_cursor,
                )
                version_library[int(other) & 0x7FFF] = library
                entry_cursor += next_auxiliary
            if not next_entry:
                break
            cursor += next_entry
    versym = next(
        (item for item in sections if item.type == _SHT_GNU_VERSYM), None,
    )

    def symbol_library(index: int) -> str:
        if versym is None:
            return "*"
        version, = struct.unpack_from("<H", data, versym.raw_offset + 2 * index)
        return version_library.get(int(version) & 0x7FFF, "*")

    imports: list[ELFImportSymbol] = []
    dynamic_relocations: list[tuple[int, int]] = []
    relocation_records: list[ELFDynamicRelocation] = []
    for relocations in sections:
        if relocations.type != _SHT_RELA or relocations.link != dynsym_index:
            continue
        for position in range(
            relocations.raw_offset,
            relocations.raw_offset + relocations.raw_size,
            relocations.entry_size or 24,
        ):
            offset, info, addend = struct.unpack_from("<QQq", data, position)
            kind, symbol = info & 0xFFFFFFFF, info >> 32
            dynamic_relocations.append((int(kind), int(offset)))
            symbol_name, symbol_value = "", None
            if symbol:
                symbol_name, symbol_info, index, value, _size = symbols[symbol]
                if index != _SHN_UNDEF and (symbol_info & 0x0F) != _STT_GNU_IFUNC:
                    symbol_value = int(value)
            relocation_records.append(ELFDynamicRelocation(
                int(kind), int(offset), int(addend), symbol_name, symbol_value,
            ))
            if kind not in {_R_X86_64_JUMP_SLOT, _R_X86_64_GLOB_DAT} or not symbol:
                continue
            name, _info, index, _value, _size = symbols[symbol]
            if index != _SHN_UNDEF:
                # A slot bound to this image's own definition is internal
                # (interposable, but its implementation is in these bytes).
                continue
            imports.append(ELFImportSymbol(
                symbol_library(symbol), name, int(offset), int(kind),
            ))

    soname = None
    needed: list[str] = []
    dynamic = by_name(".dynamic")
    if dynamic is not None:
        strings = sections[dynamic.link].raw_offset
        for position in range(
            dynamic.raw_offset, dynamic.raw_offset + dynamic.raw_size, 16,
        ):
            tag, value = struct.unpack_from("<qQ", data, position)
            if tag == 0:
                break
            if tag == 1:
                needed.append(_string(data, strings + value))
            elif tag == 14:
                soname = _string(data, strings + value)

    eh_frame = by_name(".eh_frame")
    runtime_functions = (
        _eh_frame_functions(data, eh_frame) if eh_frame is not None else ()
    )
    return ELFImage(
        ELFType(file_type), 0, int(entry), soname, tuple(needed), sections,
        runtime_functions, exports, tuple(imports),
        tuple(dynamic_relocations), min(load_addresses, default=0), data,
        relocation_records=tuple(relocation_records),
    )


__all__ = [
    "ELFExportSymbol", "ELFImage", "ELFImportSymbol", "ELFSection",
    "ELFType", "parse_elf_image",
]

import builtins
import sys

import pytest

from src.compiler.machine_reference_vocabulary import (
    MachineSemanticToken,
    X86InstructionToken,
    X86ReferenceDecoder,
)


linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="reads the running interpreter's own ELF image",
)


def test_endbr64_is_a_landing_mark_and_endbr32_is_not():
    decoder = X86ReferenceDecoder()

    marked = decoder.decode_cfg_report(b"\xf3\x0f\x1e\xfa\xc3", base_address=0x1000)
    assert not marked.failures
    first = marked.instructions[0]
    assert X86InstructionToken(first.token) is X86InstructionToken.ENDBR64
    assert MachineSemanticToken(first.semantic) is MachineSemanticToken.NO_OPERATION
    assert len(first.encoded) == 4

    # ENDBR32 shares F3 0F 1E; only the ModRM byte FA is the 64-bit mark.
    other = decoder.decode_cfg_report(b"\xf3\x0f\x1e\xfb\xc3", base_address=0x1000)
    assert other.failures


def _running_interpreter_image():
    from src.compiler.host_code_modules import resolve_host_code_identity

    identity = resolve_host_code_identity(builtins.compile)
    assert identity is not None
    return identity


@linux_only
def test_compile_resolves_to_the_loaded_elf_image_not_a_windows_dll():
    from src.compiler.elf_ingestion import parse_elf_image

    identity = _running_interpreter_image()
    assert identity.provider == "cpython-elf"
    assert identity.module_path.is_file()
    assert identity.module_path.suffix.casefold() != ".dll"
    image = parse_elf_image(identity.module_path.read_bytes())
    exported = image.export_by_name("Py_CompileString")
    assert exported is not None
    # The export's bytes are bounded by an FDE that starts at it, as a
    # .pdata record bounds a PE export.
    owner = image.runtime_function_for_rva(exported.rva)
    assert owner is not None and owner.begin_rva == exported.rva
    assert image.file_offset_for_rva(exported.rva) is not None
    # Imports are GOT slots naming another module's symbol.
    assert any(item.name == "memcpy" for item in image.imports)
    assert all(image.runtime_function_for_rva(item.iat_rva) is None for item in image.imports)


@linux_only
def test_builtin_function_resolves_to_its_fde_entry():
    from src.compiler.elf_ingestion import parse_elf_image
    from src.compiler.host_code_modules import resolve_host_code_identity

    identity = resolve_host_code_identity(len)
    assert identity is not None
    assert identity.provider == "cpython-elf-pycfunction"
    assert identity.symbol == "builtins.len"
    image = parse_elf_image(identity.module_path.read_bytes())
    owner = image.runtime_function_for_rva(identity.entry_rva)
    assert owner is not None and owner.begin_rva == identity.entry_rva


@linux_only
def test_elf_export_lifts_with_system_v_arguments_and_is_retained_whole():
    from src.compiler.cpython_compile_ssa import lift_elf_export_to_ssa

    identity = _running_interpreter_image()
    result = lift_elf_export_to_ssa(
        identity.module_path, root_symbol="Py_IsInitialized",
    )

    assert result.root_function in result.module.functions
    assert result.machine_state_complete
    retained = result.retained_native_module
    assert retained is not None
    assert (retained.format, retained.operating_system, retained.abi) == (
        "elf-image", "linux", "sysv-amd64",
    )
    assert retained.encoded == identity.module_path.read_bytes()


@linux_only
def test_elf_import_calls_are_named_external_modules():
    from src.compiler.cpython_compile_ssa import lift_elf_export_to_ssa
    from src.compiler.host_code_modules import resolve_host_code_identity

    identity = resolve_host_code_identity(len)
    result = lift_elf_export_to_ssa(
        identity.module_path,
        root_symbol=identity.symbol,
        root_rva=identity.entry_rva,
        root_calling_convention=identity.calling_convention,
    )

    assert result.root_function in result.module.functions
    for link in result.module.machine_indirect_table.links:
        assert link.target_kind != "pe-import"
        if link.target_kind == "elf-import":
            assert "!" in link.external_identity
    # What does not lift is refused by name, never dropped.
    assert all(blocker.detail for blocker in result.blockers)

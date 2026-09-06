from pathlib import Path
import sys
from types import SimpleNamespace

from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr

from src.compiler.binary_ingestion import parse_pe_image
from src.compiler.native_code_retention import (
    HostImplementationKind, NativeRetentionMode, NativeTargetContext,
    PORTABLE_AMD64_MACHINE_SSA_VM,
    WINDOWS_AMD64_NATIVE_LINKER, WINDOWS_AMD64_NATIVE_LOADER,
    retain_pe_image, select_host_implementation,
)
from src.compiler import cpython_compile_ssa
from src.compiler.machine_code_lifting import BinaryToSSAResult, VocabularyStatistics
from src.compiler.machine_reference_vocabulary import (
    DecodedInstruction, MachineSemanticToken, RelativeAddressOperand,
    X86InstructionToken,
)


def _python_image():
    path = Path(sys.executable).resolve().parent / (
        f"python{sys.version_info.major}{sys.version_info.minor}.dll"
    )
    encoded = path.read_bytes()
    image, _ = parse_pe_image(encoded, maximum_file_size=len(encoded))
    return path, encoded, image


def test_complete_pe_image_is_retained_with_loader_context():
    path, encoded, image = _python_image()

    retained = retain_pe_image(image, source_identity=str(path))

    assert retained.encoded == encoded
    assert retained.format == "pe-image"
    assert retained.architecture == "amd64"
    assert retained.abi == "windows-x64"
    assert retained.exports["Py_CompileString"] == image.export_by_name("Py_CompileString").rva
    assert retained.sections
    assert retained.unwind_functions
    assert retained.digest


def test_retention_distinguishes_loader_linker_and_translation_targets():
    _path, _encoded, image = _python_image()
    retained = retain_pe_image(image)

    assert retained.retention_mode(WINDOWS_AMD64_NATIVE_LOADER) is NativeRetentionMode.LOADABLE_IMAGE
    # A final PE/DLL is not a relocatable COFF object accepted directly by a
    # native linker's object-input slot.
    assert retained.retention_mode(WINDOWS_AMD64_NATIVE_LINKER) is NativeRetentionMode.TRANSLATE
    assert retained.retention_mode(NativeTargetContext(
        "wasm32", "browser", "wasm", frozenset({"wasm"}),
    )) is NativeRetentionMode.TRANSLATE


def test_retained_image_writes_exact_original_bytes(tmp_path):
    _path, encoded, image = _python_image()
    retained = retain_pe_image(image)
    output = tmp_path / "python-retained.dll"

    retained.write(output)

    assert output.read_bytes() == encoded


def test_backend_selection_uses_native_only_when_artifact_context_matches():
    _path, _encoded, image = _python_image()
    retained = retain_pe_image(image)

    loader = select_host_implementation(
        repository_ssa_complete=False,
        retained_native_module=retained,
        target=WINDOWS_AMD64_NATIVE_LOADER,
    )
    linker = select_host_implementation(
        repository_ssa_complete=True,
        retained_native_module=retained,
        target=WINDOWS_AMD64_NATIVE_LINKER,
    )

    assert loader.implementation is HostImplementationKind.RETAINED_NATIVE_MODULE
    assert loader.native_mode is NativeRetentionMode.LOADABLE_IMAGE
    assert loader.deployable
    assert linker.implementation is HostImplementationKind.REPOSITORY_SSA
    assert linker.native_mode is NativeRetentionMode.TRANSLATE


def test_backend_selection_does_not_call_incomplete_ssa_a_fallback():
    _path, _encoded, image = _python_image()
    retained = retain_pe_image(image)
    wasm = NativeTargetContext(
        "wasm32", "browser", "wasm", frozenset({"wasm"}),
    )

    decision = select_host_implementation(
        repository_ssa_complete=False,
        retained_native_module=retained,
        target=wasm,
    )

    assert decision.implementation is HostImplementationKind.TRANSLATION_REQUIRED
    assert not decision.deployable
    assert decision.native_mode is NativeRetentionMode.TRANSLATE


def test_machine_state_ssa_is_a_distinct_deployable_implementation():
    target = NativeTargetContext(
        "wasm32", "browser", "wasm", frozenset({"wasm"}),
    )

    accepted = select_host_implementation(
        repository_ssa_complete=False,
        machine_state_ssa_complete=True,
        retained_native_module=None,
        target=PORTABLE_AMD64_MACHINE_SSA_VM,
    )
    rejected = select_host_implementation(
        repository_ssa_complete=False,
        machine_state_ssa_complete=True,
        retained_native_module=None,
        target=target,
    )

    assert accepted.implementation is HostImplementationKind.MACHINE_STATE_SSA
    assert accepted.deployable
    assert rejected.implementation is HostImplementationKind.TRANSLATION_REQUIRED


def test_dynamic_indirect_control_is_machine_complete_but_not_dependency_closed():
    from src.compiler.cpython_compile_ssa import (
        NativeCompileSSABlocker, NativeCompileSSAResult,
    )

    indirect = NativeCompileSSABlocker(
        1, 0x2000, "dispatch", "indirect-jump", 0x180002000,
        "native jump target depends on register machine state",
    )
    result = NativeCompileSSAResult(
        IRModule({}), "dispatch", "dispatch", Path(__file__),
        (0x2000,), (indirect,), None,
    )

    assert result.machine_state_complete
    assert result.machine_state_blockers == ()
    assert not result.dependency_context_complete
    assert not result.complete
    assert result.hard_blockers == (indirect,)


def test_cross_pdata_tail_edge_becomes_exact_machine_state_funclet(monkeypatch):
    image_base = 0x180000000
    owner = SimpleNamespace(
        begin_rva=0x2000, end_rva=0x2010,
    )
    image = SimpleNamespace(
        image_base=image_base,
        runtime_function_for_rva=lambda rva: (
            owner if owner.begin_rva <= rva < owner.end_rva else None
        ),
    )
    target = image_base + 0x2004
    source = Function(
        "source", [], {
            "entry": BasicBlock("entry", [Instr(
                "CondBr", [], None,
                attributes={
                    "machine_address": image_base + 0x1000,
                    "true_target": None,
                    "true_target_address": target,
                    "false_target": "ret",
                    "false_target_address": None,
                },
            )], ["ret"]),
            "ret": BasicBlock("ret", [Instr("Ret", [], None)], []),
        },
    )
    # The target is four bytes into its owner and begins a one-instruction
    # funclet. Remaining bytes are retained as unreachable, not interpreted.
    owner_bytes = b"\xcc\xcc\xcc\xcc\xc3" + b"\xcc" * 11
    monkeypatch.setattr(
        cpython_compile_ssa, "pe_runtime_function_region",
        lambda *_args, **_kwargs: (owner, 0, owner_bytes),
    )
    functions = {source.name: source}

    table, blockers = cpython_compile_ssa._link_machine_control_funclets(
        image, functions,
    )

    assert blockers == ()
    assert len(table.links) == 1
    link = table.links[0]
    assert link.target_address == target
    assert link.target_kind == "runtime-function-interior"
    assert link.target_function == "machine_funclet_00002004"
    funclet = functions[link.target_function]
    assert funclet.metadata["dialect"] == "turing.machine-state-ssa.amd64.v1"
    assert funclet.metadata["machine_funclet_entry_address"] == target
    assert funclet.metadata["machine_owner_rva"] == owner.begin_rva


def test_direct_call_worklist_preserves_alternate_entry_inside_same_owner():
    image_base = 0x180000000
    owner = SimpleNamespace(
        begin_rva=0x2000, end_rva=0x2100,
    )
    target_rva = 0x2040
    image = SimpleNamespace(
        image_base=image_base,
        runtime_function_for_rva=lambda rva: (
            owner if owner.begin_rva <= rva < owner.end_rva else None
        ),
    )
    decoded = DecodedInstruction(
        image_base + 0x2010,
        X86InstructionToken.CALL_REL32,
        MachineSemanticToken.DIRECT_RELATIVE_CALL,
        (RelativeAddressOperand(0x2B, 32, image_base + target_rva),),
        b"\xe8\x2b\x00\x00\x00",
    )
    lifting = SimpleNamespace(decoded=(decoded,))

    targets = cpython_compile_ssa._direct_function_targets(
        image, owner, lifting,
    )

    assert targets == ((target_rva, owner),)

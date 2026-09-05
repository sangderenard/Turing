from types import MappingProxyType
from types import SimpleNamespace

import pytest

from src.compiler.freestanding_amd64 import (
    FREESTANDING_AMD64_ABI, FREESTANDING_CAPABILITY_LIBRARY,
    TURING_FREESTANDING_AMD64_LOADER, validate_freestanding_amd64_image,
    validate_freestanding_amd64_program,
    retain_freestanding_amd64_program,
)
from src.compiler.machine_reference_vocabulary import (
    DecodedInstruction, MachineSemanticToken, X86InstructionToken,
)
from src.compiler.native_code_retention import (
    NativeRetentionMode, RetainedNativeModule,
)


def _module(*, imports=(), operating_system="turing", abi=FREESTANDING_AMD64_ABI):
    return RetainedNativeModule(
        "pe-image", "amd64", operating_system, abi, b"MZ", 0x140000000,
        0x1000, False, MappingProxyType({"entry": 0x1000}), tuple(imports),
        ((".text", 0x1000, 1, 0, 0x60000020),), (), (), "test",
    )


def test_freestanding_profile_accepts_only_its_eager_capability_abi():
    module = _module(imports=((
        FREESTANDING_CAPABILITY_LIBRARY, "turing_output_publish", 0x2000, False,
    ),))

    instruction = DecodedInstruction(
        0x140001000, X86InstructionToken.RET_NEAR,
        MachineSemanticToken.RETURN, (), b"\xc3",
    )
    validation = validate_freestanding_amd64_image(
        module, (instruction,), executable_coverage_complete=True,
    )

    assert validation.compatible
    assert module.retention_mode(
        TURING_FREESTANDING_AMD64_LOADER
    ) is NativeRetentionMode.LOADABLE_IMAGE


def test_windows_import_and_post_baseline_instruction_are_exact_shortfalls():
    module = _module(imports=(("kernel32.dll", "ExitProcess", 0x2000, False),))
    instruction = DecodedInstruction(
        0x140001000, X86InstructionToken.PCMPEQQ_XMM_XMMM128,
        MachineSemanticToken.VECTOR_COMPARE_EQUAL_QWORDS, (),
        bytes.fromhex("660f3829c0"),
    )

    validation = validate_freestanding_amd64_image(
        module, (instruction,), executable_coverage_complete=True,
    )

    assert not validation.compatible
    assert [item.kind for item in validation.shortfalls] == [
        "foreign-import", "instruction-level",
    ]
    assert [item.occurrence for item in validation.shortfalls] == [1, 2]


def test_structural_profile_without_complete_code_census_is_not_certified():
    validation = validate_freestanding_amd64_image(_module())

    assert not validation.compatible
    assert [item.kind for item in validation.shortfalls] == [
        "executable-coverage",
    ]


def test_program_graph_is_the_authority_for_executable_coverage():
    instruction = DecodedInstruction(
        0x140001000, X86InstructionToken.RET_NEAR,
        MachineSemanticToken.RETURN, (), b"\xc3",
    )
    report = type("Report", (), {"instructions": (instruction,)})()
    record = type("Record", (), {"report": report})()
    complete = type("Program", (), {
        "functions": (record,), "complete": True,
    })()
    incomplete = type("Program", (), {
        "functions": (record,), "complete": False,
    })()

    assert validate_freestanding_amd64_program(_module(), complete).compatible
    assert [
        item.kind for item in
        validate_freestanding_amd64_program(_module(), incomplete).shortfalls
    ] == ["executable-coverage"]


def test_validating_constructor_rejects_relabelled_windows_import(monkeypatch):
    retained = _module(
        imports=(("kernel32.dll", "ExitProcess", 0x2000, False),),
        operating_system="windows", abi="windows-x64",
    )
    program = SimpleNamespace(functions=(), complete=True)
    monkeypatch.setattr(
        "src.compiler.freestanding_amd64.retain_pe_image",
        lambda *_args, **_kwargs: retained,
    )

    with pytest.raises(ValueError, match="outside the freestanding capability ABI"):
        retain_freestanding_amd64_program(object(), program)

from __future__ import annotations

import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.native_code_retention import (
    ANDROID_ARM64_NATIVE_LINKER,
    IOS_ARM64_NATIVE_LINKER,
    X86_64_PORTABLE_EXECUTOR,
    HostImplementationKind,
    build_target_from_metadata,
    build_targets,
    resolve_build_target,
    select_host_implementation,
)
from src.compiler.ssa_llvm_backend import (
    LLVMFunctionArtifact,
    compile_artifact,
    compile_object_artifact,
    emit_ssa_function_to_llvm,
)


def test_ios_arm64_is_an_explicit_native_build_target():
    target = resolve_build_target("ios-arm64")

    assert target is IOS_ARM64_NATIVE_LINKER
    assert target.architecture == "arm64"
    assert target.operating_system == "ios"
    assert target.abi == "darwin-arm64"
    assert target.object_formats == frozenset({"macho-object"})
    assert target.llvm_triple == "aarch64-apple-ios15.0"
    assert target.toolchain_target == "aarch64-ios"
    assert target.sdk == "iphoneos"
    assert target.to_mapping()["schema"] == "turing.build-target.v1"


def test_mobile_and_portable_x86_targets_share_one_registry():
    targets = build_targets()

    assert targets["android-arm64-v8a"] is ANDROID_ARM64_NATIVE_LINKER
    assert targets["ios-arm64"] is IOS_ARM64_NATIVE_LINKER
    assert targets["x86-64-portable"] is X86_64_PORTABLE_EXECUTOR
    assert not X86_64_PORTABLE_EXECUTOR.object_formats
    assert X86_64_PORTABLE_EXECUTOR.machine_state_ssa_dialects == frozenset({
        "turing.machine-state-ssa.amd64.v1",
    })


def test_portable_x86_target_selects_machine_state_execution_not_native_binary():
    decision = select_host_implementation(
        repository_ssa_complete=False,
        machine_state_ssa_complete=True,
        retained_native_module=None,
        target="x86-64-portable",
    )

    assert decision.implementation is HostImplementationKind.MACHINE_STATE_SSA
    assert decision.deployable


def test_source_compiler_passes_ios_target_as_verified_module_contract():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def root(x: float):\n    return x + 1.0\n",
        "root",
        build_target="ios-arm64",
        extraction_contract=ExtractionContract(
            "extraction_contracts/program_extraction.yaml"
        ),
        progress_bars=False,
    )

    assert module.metadata["build_target"]["identity"] == "ios-arm64"
    assert build_target_from_metadata(module.metadata) is IOS_ARM64_NATIVE_LINKER
    artifact = emit_ssa_function_to_llvm(module, _exports[0])
    assert 'target triple = "aarch64-apple-ios15.0"' in artifact.llvm_ir


def test_backend_refuses_tampered_target_receipt():
    receipt = IOS_ARM64_NATIVE_LINKER.to_mapping()
    receipt["llvm_triple"] = "x86_64-unknown-linux-gnu"

    with pytest.raises(ValueError, match="disagrees on llvm_triple"):
        build_target_from_metadata({"build_target": receipt})


@pytest.mark.parametrize(("target", "magic"), (
    (IOS_ARM64_NATIVE_LINKER, bytes.fromhex("cffaedfe")),
    (ANDROID_ARM64_NATIVE_LINKER, b"\x7fELF"),
))
def test_mobile_targets_cross_compile_native_objects(tmp_path, target, magic):
    artifact = LLVMFunctionArtifact(
        "mobile_probe",
        (
            f'target triple = "{target.llvm_triple}"\n'
            "define void @mobile_probe(ptr %buffers, ptr %extents) { ret void }\n"
        ),
        (), (), (), (), build_target=target,
    )

    path = compile_object_artifact(artifact, directory=tmp_path / target.identity)

    assert path.suffix == ".o"
    assert path.read_bytes().startswith(magic)


def test_ios_final_link_is_reserved_for_the_xcode_sdk_host(tmp_path):
    artifact = LLVMFunctionArtifact(
        "ios_link_probe",
        (
            'target triple = "aarch64-apple-ios15.0"\n'
            "define void @ios_link_probe(ptr %buffers, ptr %extents) { ret void }\n"
        ),
        (), (), (), (), build_target=IOS_ARM64_NATIVE_LINKER,
    )

    with pytest.raises(ValueError, match="Xcode host"):
        compile_artifact(artifact, directory=tmp_path / "ios-link")


def test_unknown_target_is_refused_before_lowering():
    with pytest.raises(KeyError, match="unknown build target"):
        resolve_build_target("ios-sometimes-arm64")

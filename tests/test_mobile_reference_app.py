from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_wasm_backend import (
    emit_ssa_module_to_wasm_core,
    prepare_wasm_core_execution,
)


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "examples/mobile_reference_app"


def test_reference_app_uses_one_turing_program_and_main_output_shader():
    source = (APP / "program.py").read_text(encoding="utf-8")
    android = (APP / "android/main.cpp").read_text(encoding="utf-8")
    ios = (APP / "ios/main.mm").read_text(encoding="utf-8")

    assert "def mobile_main(" in source
    assert "turing_mobile_eval(" in android
    assert "turing_mobile_eval(" in ios
    assert (APP / "shaders/main_output.frag").is_file()


def test_android_is_a_native_activity_with_vulkan_requirement():
    manifest = (APP / "android/AndroidManifest.xml").read_text(encoding="utf-8")

    assert 'package="org.turingcompiler.mobile"' in manifest
    assert "android.app.NativeActivity" in manifest
    assert "android.hardware.vulkan.version" in manifest
    assert "android:minSdkVersion=\"24\"" in manifest


def test_ios_host_has_real_metal_shell_and_unsigned_ipa_boundary():
    host = (APP / "ios/main.mm").read_text(encoding="utf-8")
    packager = (APP / "ios/package_ios.py").read_text(encoding="utf-8")

    assert "MTKViewDelegate" in host
    assert "newLibraryWithSource" in host
    assert "mobile_main.o" in packager
    assert '"--sdk", "iphoneos", "clang++"' in packager
    assert "Signing and device installation are separate" in packager


def test_pwa_launches_public_compiler_entry_and_not_private_region():
    shell = (APP / "web/index.html").read_text(encoding="utf-8")
    binding = (APP / "web/turing-wasm-shell.js").read_text(encoding="utf-8")
    builder = (APP / "build_mobile.py").read_text(encoding="utf-8")

    assert "contract.entrypoint" in binding
    assert "TuringWasmProgram" in shell
    assert "emit_ssa_module_to_wasm_core" in builder
    assert "llvm-wasm32" not in builder
    assert "planned_region_0" not in shell


def test_independent_wasm_backend_runs_public_whole_source_entry():
    module, outputs, exports = lower_ast_source_to_ssa(
        (APP / "program.py").read_text(encoding="utf-8"),
        "mobile_main",
        build_target="android-arm64-v8a",
        extraction_contract=ExtractionContract(
            ROOT / "extraction_contracts/program_extraction.yaml"
        ),
        progress_bars=False,
    )
    entry = exports[0]
    artifact = emit_ssa_module_to_wasm_core(module, entry, entry_name=entry)

    assert artifact.complete, artifact.shortfalls
    assert artifact.name == entry
    input_ids = tuple(int(value.id) for value in module.functions[entry].args)
    output_ids = tuple(int(value.id) for value in outputs[entry])
    execution = prepare_wasm_core_execution(
        artifact,
        dict(zip(input_ids, (0.5, 0.25, 0.75))),
    )
    try:
        execution.run()
        assert [float(execution.buffers[value_id][0]) for value_id in output_ids] == pytest.approx(
            (0.425, 0.3625, 0.7375)
        )
    finally:
        execution.close()


def test_pwa_manifest_is_standalone():
    manifest = json.loads((APP / "web/manifest.webmanifest").read_text(encoding="utf-8"))

    assert manifest["display"] == "standalone"
    assert manifest["start_url"] == "./"

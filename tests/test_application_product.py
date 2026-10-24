from __future__ import annotations

import json

import pytest

from src.compiler.application_product import (
    ApplicationProduct,
    ExecutionProduct,
    PresentationProduct,
    ProductFile,
    SCHEMA,
    TargetProduct,
    android_vulkan_dependency,
    moltenvk_ios_dependency,
    publish_web_application_product,
)
from src.compiler.compiled_program_api import CompiledProgramAPI, EntryPoint


def test_application_product_references_existing_contracts_without_copying_ports(tmp_path):
    (tmp_path / "program.api.yaml").write_text("schema: turing-compiled-program-api-v1\n")
    (tmp_path / "program.wasm").write_bytes(b"\0asm\1\0\0\0")
    (tmp_path / "main.spv").write_bytes(b"\x03\x02#\x07")
    (tmp_path / "main.spirv.json").write_text(
        '{"schema":"turing.spirv-product.v1"}\n', encoding="utf-8",
    )
    api = ProductFile.from_root(tmp_path, "program.api.yaml")
    execution = ExecutionProduct(
        "contiguous-wasm", ProductFile.from_root(tmp_path, "program.wasm"),
    )
    presentation = PresentationProduct(
        "example.main-output",
        ProductFile.from_root(tmp_path, "main.spirv.json"),
        ProductFile.from_root(tmp_path, "main.spv"),
    )
    product = ApplicationProduct(
        "example.app", api, "whole_program",
        targets={
            "web": TargetProduct(execution),
            "android-arm64-v8a": TargetProduct(
                execution, presentation, (android_vulkan_dependency(),),
            ),
            "ios-arm64": TargetProduct(
                execution, presentation, (moltenvk_ios_dependency(),),
            ),
        },
        default_target="web",
    )

    mapping = product.to_mapping()
    assert mapping["schema"] == SCHEMA
    assert mapping["program"]["entrypoint"] == "whole_program"
    assert mapping["shell"] == {"program_api_metadata": "shell_io"}
    assert mapping["targets"]["android-arm64-v8a"]["presentation"][
        "component_id"
    ] == "example.main-output"
    assert mapping["targets"]["ios-arm64"]["runtime_dependencies"][0][
        "shader_input"
    ] == "spirv"
    # Logical ports/resources stay in their authoritative sidecars.
    assert "ports" not in json.dumps(mapping)
    assert "resources" not in json.dumps(mapping)


def test_product_files_refuse_paths_outside_the_product():
    with pytest.raises(ValueError, match="contained relative paths"):
        ProductFile("../program.wasm", "0" * 64)


def test_application_product_requires_a_real_default_target():
    file = ProductFile("program.api.yaml", "0" * 64)
    execution = ExecutionProduct("native", ProductFile("program.so", "1" * 64))
    with pytest.raises(ValueError, match="unknown default"):
        ApplicationProduct(
            "example", file, "run", targets={"android": TargetProduct(execution)},
            default_target="ios",
        )


def test_mobile_vulkan_dependencies_name_the_platform_owners():
    assert android_vulkan_dependency()["provider"] == "android-ndk"
    ios = moltenvk_ios_dependency()
    assert ios["kind"] == "static-xcframework"
    assert ios["product"].endswith("MoltenVK.xcframework")
    assert ios["archive"]["minimum_ios"] == "15.0"
    assert len(ios["archive"]["sha256"]) == 64
    with pytest.raises(ValueError, match="no pinned MoltenVK"):
        moltenvk_ios_dependency(version="unreviewed")


def test_web_publisher_extracts_execution_from_the_page_product(tmp_path):
    coordinator = tmp_path / "wasm" / "coordinator.wasm"
    coordinator.parent.mkdir()
    coordinator.write_bytes(b"\0asm\1\0\0\0")
    api = CompiledProgramAPI(
        module="example", language="wasm", entry="run",
        entry_points=(EntryPoint("run", "run_range", "control"),),
        metadata={"shell_io": {"requirements": {"version": 1}}},
    )

    reference = publish_web_application_product(
        tmp_path,
        identity="example",
        api=api,
        coordinator="wasm/coordinator.wasm",
        execution_manifest={"coordinator": {"url": "wasm/coordinator.wasm"}},
    )

    product = json.loads((tmp_path / reference.path).read_text())
    assert product["program"]["api"]["path"] == "application/program.api.yaml"
    assert product["targets"]["web"]["execution"]["kind"] == "wasm-coordinator"
    assert product["targets"]["web"]["execution"]["artifact"]["path"] == (
        "wasm/coordinator.wasm"
    )
    assert (tmp_path / "application/web-execution.json").is_file()


def test_web_publisher_requires_the_compiler_owned_entrypoint(tmp_path):
    api = CompiledProgramAPI(module="example", language="wasm", entry=None)

    with pytest.raises(ValueError, match="compiled-program entrypoint"):
        publish_web_application_product(
            tmp_path,
            identity="example",
            api=api,
            coordinator="wasm/coordinator.wasm",
            execution_manifest={},
        )

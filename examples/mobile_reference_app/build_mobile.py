"""Build the same Turing application for Android, iOS ARM64, and mobile web."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.application_product import (
    ApplicationProduct,
    ExecutionProduct,
    PresentationProduct,
    ProductFile,
    TargetProduct,
    android_vulkan_dependency,
    moltenvk_ios_dependency,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    compile_object_artifact,
    emit_ssa_function_to_llvm,
)
from src.compiler.ssa_wasm_backend import emit_ssa_module_to_wasm_core

ENTRY = "mobile_main"
PACKAGE = "org.turingcompiler.mobile"
SDK = ROOT / "build/mobile-dependencies/android-sdk"
NDK_VERSION = "30.0.16248370"
API = 24


def run(command: list[str | Path], *, cwd: Path | None = None) -> None:
    printable = " ".join(str(item) for item in command)
    print(f"[mobile-build] {printable}", flush=True)
    subprocess.run([str(item) for item in command], cwd=cwd, check=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def find_glslang() -> Path:
    found = shutil.which("glslangValidator")
    if found:
        return Path(found)
    candidates = sorted(Path("C:/VulkanSDK").glob("*/Bin/glslangValidator.exe"), reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError("glslangValidator is required (install the Vulkan SDK)")


def compile_source(target: str):
    source = (APP / "program.py").read_text(encoding="utf-8")
    module, outputs, exports = lower_ast_source_to_ssa(
        source,
        ENTRY,
        build_target=target,
        extraction_contract=ExtractionContract(
            ROOT / "extraction_contracts/program_extraction.yaml"
        ),
        progress_bars=False,
    )
    entry = exports[0]
    llvm = emit_ssa_function_to_llvm(module, entry)
    if not llvm.complete:
        raise RuntimeError(f"LLVM emission shortfalls: {llvm.shortfalls}")
    return module, outputs, entry, llvm


def compile_shaders(output: Path) -> tuple[Path, Path]:
    glslang = find_glslang()
    output.mkdir(parents=True, exist_ok=True)
    vertex = output / "fullscreen.vert.spv"
    fragment = output / "main_output.frag.spv"
    run([glslang, "-V", APP / "shaders/fullscreen.vert", "-o", vertex])
    run([glslang, "-V", APP / "shaders/main_output.frag", "-o", fragment])
    return vertex, fragment


def words(path: Path) -> tuple[int, ...]:
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"SPIR-V size is not word-aligned: {path}")
    return struct.unpack(f"<{len(data) // 4}I", data)


def shader_header(vertex: Path, fragment: Path, path: Path) -> None:
    def array(name: str, source: Path) -> str:
        values = words(source)
        body = ",".join(f"0x{value:08x}u" for value in values)
        return f"inline constexpr uint32_t {name}[] = {{{body}}};\ninline constexpr size_t {name}Size = sizeof({name});\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#pragma once\n#include <cstddef>\n#include <cstdint>\n" +
        array("kFullscreenVert", vertex) + array("kMainOutputFrag", fragment),
        encoding="utf-8",
    )


def abi_header(module, outputs, entry: str, artifact, path: Path) -> None:
    function = module.functions[entry]
    arguments = tuple(int(value.id) for value in function.args)
    returned = tuple(int(value.id) for value in outputs[entry])
    slot = {int(value_id): index for index, value_id in enumerate(artifact.buffer_order)}
    if len(arguments) != 3 or len(returned) != 3:
        raise ValueError("mobile_main must have three inputs and three outputs")
    if any(dtype != "double" for dtype in artifact.buffer_dtypes):
        raise TypeError(f"reference shell requires float64 buffers: {artifact.buffer_dtypes}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""#pragma once
#include <cstdint>
extern "C" void {entry}(void**, int32_t*);
struct TuringMobileColor {{ double red, green, blue; }};
inline TuringMobileColor turing_mobile_eval(double phase, double touch_x, double touch_y) {{
    double values[{len(artifact.buffer_order)}]{{}};
    values[{slot[arguments[0]]}] = phase;
    values[{slot[arguments[1]]}] = touch_x;
    values[{slot[arguments[2]]}] = touch_y;
    void* buffers[{len(artifact.buffer_order)}]{{}};
    int32_t extents[{len(artifact.buffer_order)}]{{}};
    for (int i = 0; i < {len(artifact.buffer_order)}; ++i) {{ buffers[i] = &values[i]; extents[i] = 1; }}
    {entry}(buffers, extents);
    return {{values[{slot[returned[0]]}], values[{slot[returned[1]]}], values[{slot[returned[2]]}]}};
}}
""", encoding="utf-8")


def program_abi(artifact, inputs: tuple[str, ...], outputs: tuple[str, ...]) -> dict:
    runtime = artifact.runtime_abi()
    runtime.update({"schema": "turing.mobile-program-abi.v1", "inputs": list(inputs),
                    "outputs": list(outputs)})
    return runtime


def write_native_product(directory: Path, artifact_path: Path, api_path: Path,
                         fragment: Path, target_name: str) -> None:
    deployment = directory / "main-output.deployment.json"
    write_json(deployment, {
        "schema": "turing.shader-deployment.v1",
        "component_id": "main-output",
        "language": "spirv",
        "stage": "fragment",
        "artifact": fragment.name,
        "native_api": "vulkan",
    })
    product = ApplicationProduct(
        "turing-mobile-reference",
        ProductFile.from_root(directory, api_path.name),
        str(json.loads(api_path.read_text(encoding="utf-8"))["entry"]),
        targets={target_name: TargetProduct(
            ExecutionProduct("native-library", ProductFile.from_root(directory, artifact_path.name)),
            PresentationProduct(
                "main-output",
                ProductFile.from_root(directory, deployment.name),
                ProductFile.from_root(directory, fragment.name),
            ),
            (android_vulkan_dependency(),) if target_name.startswith("android") else (),
        )},
        default_target=target_name,
    )
    product.write(directory / "application.json")


def build_android(output: Path) -> Path:
    ndk = SDK / f"ndk/{NDK_VERSION}"
    tools = SDK / "build-tools/37.0.0"
    platform = SDK / "platforms/android-37.0/android.jar"
    clang = ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin/clang.exe"
    clangxx = ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe"
    glue = ndk / "sources/android/native_app_glue"
    for required in (clang, clangxx, tools / "aapt.exe", tools / "apksigner.bat", platform):
        if not required.exists():
            raise FileNotFoundError(f"Android toolchain component missing: {required}")

    module, outputs, entry, artifact = compile_source("android-arm64-v8a")
    output.mkdir(parents=True, exist_ok=True)
    generated = output / "generated"
    artifact_ir = output / "mobile_main.ll"
    artifact_ir.write_text(artifact.llvm_ir, encoding="utf-8")
    write_json(output / "program.abi.json",
               program_abi(artifact, ("phase", "touch_x", "touch_y"), ("red", "green", "blue")))
    abi_header(module, outputs, entry, artifact, generated / "turing_program_abi.h")
    vertex, fragment = compile_shaders(output)
    shader_header(vertex, fragment, generated / "mobile_shaders.h")

    target = f"aarch64-linux-android{API}"
    common = [f"--target={target}", "-fPIC", "-O2"]
    program_o, main_o, glue_o = output / "program.o", output / "main.o", output / "native_app_glue.o"
    run([clang, *common, "-c", artifact_ir, "-o", program_o])
    run([clang, *common, "-c", glue / "android_native_app_glue.c", "-I", glue, "-o", glue_o])
    run([clangxx, *common, "-std=c++20", "-DVK_USE_PLATFORM_ANDROID_KHR",
         "-I", glue, "-I", output, "-c", APP / "android/main.cpp", "-o", main_o])
    library = output / "libturing_mobile.so"
    run([clangxx, f"--target={target}", "-shared", "-Wl,--no-undefined",
         "-Wl,-u,ANativeActivity_onCreate", "-Wl,-soname,libturing_mobile.so",
         main_o, glue_o, program_o, "-landroid", "-llog", "-lvulkan", "-o", library])
    write_native_product(output, library, output / "program.abi.json", fragment,
                         "android-arm64-v8a")

    unsigned = output / "unsigned.apk"
    aligned = output / "aligned.apk"
    final = output / "turing-mobile-debug.apk"
    run([tools / "aapt.exe", "package", "-f", "-M", APP / "android/AndroidManifest.xml",
         "-I", platform, "-F", unsigned])
    staging = output / "apk-staging"
    staged_lib = staging / "lib/arm64-v8a/libturing_mobile.so"
    staged_lib.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(library, staged_lib)
    run([tools / "aapt.exe", "add", unsigned, "lib/arm64-v8a/libturing_mobile.so"], cwd=staging)
    run([tools / "zipalign.exe", "-f", "-p", "4", unsigned, aligned])
    keytool = shutil.which("keytool")
    if not keytool:
        raise FileNotFoundError("keytool is required to create the debug signing key")
    keystore = ROOT / "build/mobile-dependencies/turing-mobile-debug.keystore"
    if not keystore.exists():
        run([keytool, "-genkeypair", "-keystore", keystore, "-storepass", "android",
             "-keypass", "android", "-alias", "androiddebugkey", "-keyalg", "RSA",
             "-keysize", "2048", "-validity", "10000", "-dname", "CN=Turing Mobile Debug"])
    run([tools / "apksigner.bat", "sign", "--ks", keystore, "--ks-pass", "pass:android",
         "--key-pass", "pass:android", "--out", final, aligned])
    run([tools / "apksigner.bat", "verify", "--verbose", "--print-certs", final])
    run([tools / "aapt.exe", "dump", "badging", final])
    return final


def build_web(output: Path) -> Path:
    module, outputs, entry, llvm = compile_source("android-arm64-v8a")
    wasm = emit_ssa_module_to_wasm_core(module, entry, entry_name=entry)
    if not wasm.complete:
        raise RuntimeError(f"independent WebAssembly emission shortfalls: {wasm.shortfalls}")
    output.mkdir(parents=True, exist_ok=True)
    wasm_path = output / "mobile_main.wasm"
    wasm_path.write_bytes(wasm.binary)
    for source in (APP / "web").iterdir():
        shutil.copy2(source, output / source.name)
    argument_ids = tuple(int(value.id) for value in module.functions[entry].args)
    output_ids = tuple(int(value.id) for value in outputs[entry])
    input_names = dict(zip(argument_ids, ("phase", "touch_x", "touch_y")))
    output_names = dict(zip(output_ids, ("red", "green", "blue")))
    descriptors = []
    buffer_offset = 8
    for kind, value_id in wasm.parameters:
        role = "output" if value_id in output_names else "input"
        descriptor = {
            "kind": kind,
            "value_id": value_id,
            "role": role,
            "name": output_names.get(value_id, input_names.get(value_id, f"value_{value_id}")),
        }
        if kind == "buffer":
            descriptor["offset"] = buffer_offset
            buffer_offset += 8
        descriptors.append(descriptor)
    contract = {
        "schema": "turing.mobile-wasm-abi.v1",
        "entrypoint": entry,
        "inputs": ["phase", "touch_x", "touch_y"],
        "outputs": ["red", "green", "blue"],
        "parameters": descriptors,
    }
    write_json(output / "program.abi.json", contract)
    product = {
        "schema": "turing.application-product.v1",
        "identity": "turing-mobile-reference",
        "program": {"api": "program.abi.json", "entrypoint": entry},
        "shell": {"kind": "pwa", "launch": "index.html"},
        "targets": {"web": {"execution": {"kind": "repository-ssa-wasm", "artifact": wasm_path.name},
                              "presentation": {"component_id": "main-output", "api": "webgl2"}}},
    }
    write_json(output / "application.json", product)
    return output / "index.html"


def build_ios_compiler_product(output: Path) -> Path:
    module, outputs, entry, artifact = compile_source("ios-arm64")
    output.mkdir(parents=True, exist_ok=True)
    (output / "mobile_main.ll").write_text(artifact.llvm_ir, encoding="utf-8")
    object_path = compile_object_artifact(artifact, directory=output)
    stable_object = output / "mobile_main.o"
    if object_path != stable_object:
        shutil.copy2(object_path, stable_object)
    abi_header(module, outputs, entry, artifact,
               output / "generated/turing_program_abi.h")
    vertex, fragment = compile_shaders(output)
    spirv_cross = find_glslang().with_name("spirv-cross.exe" if os.name == "nt" else "spirv-cross")
    if not spirv_cross.exists():
        raise FileNotFoundError("SPIRV-Cross is required to generate Metal source")
    for source, destination in (
        (vertex, output / "fullscreen.vert.metal"),
        (fragment, output / "main_output.frag.metal"),
    ):
        print(f"[mobile-build] {spirv_cross} {source} --msl --msl-version 20000", flush=True)
        converted = subprocess.run(
            [str(spirv_cross), str(source), "--msl", "--msl-version", "20000"],
            check=True, capture_output=True, text=True,
        )
        destination.write_text(converted.stdout, encoding="utf-8")
    write_json(output / "program.abi.json",
               program_abi(artifact, ("phase", "touch_x", "touch_y"), ("red", "green", "blue")))
    receipt = {
        "schema": "turing.ios-compiler-product.v1",
        "target": "ios-arm64",
        "entrypoint": entry,
        "object": stable_object.name,
        "object_sha256": hashlib.sha256(stable_object.read_bytes()).hexdigest(),
        "linked": False,
        "shader_translation": "SPIR-V -> MSL 2.0 via SPIRV-Cross",
        "native_host": "examples/mobile_reference_app/ios/main.mm",
        "vulkan_compatibility_dependency": moltenvk_ios_dependency(),
        "next_stage": "link the native host and this object against iPhoneOS SDK frameworks, then package/sign",
    }
    write_json(output / "compiler-product.json", receipt)
    return stable_object


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "build/mobile-reference")
    parser.add_argument("--android", action="store_true")
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--ios", action="store_true")
    args = parser.parse_args()
    selected = args.android or args.web or args.ios
    if args.android or not selected:
        print(f"Android APK: {build_android(args.output / 'android')}")
    if args.web or not selected:
        print(f"Web launch: {build_web(args.output / 'web')}")
    if args.ios or not selected:
        print(f"iOS ARM64 object: {build_ios_compiler_product(args.output / 'ios')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

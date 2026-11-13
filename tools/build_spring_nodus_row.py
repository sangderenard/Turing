"""Compile the SymPy spring law and stage its private-state Nodus row in user space.

The generated C++ row only owns topology/state and gathers neighbors.  The
Turing-compiled AbstractTensor law performs every physics step over one batch.
Build/load the emitted .cpp with Nodus's existing plugin builder.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.native_law_kernels import law_kernel
from src.compiler.spring_network_law import INPUT_NAMES, compile_spring_network_step


NETWORKS = 8
NODES = 256
AXES = 3
BATCH = NETWORKS * NODES * AXES


def stage(directory: Path, nodus_root: Path | None = None, nodus_build: Path | None = None,
          accept: bool = False, deploy_dir: Path | None = None) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    kernel = law_kernel(compile_spring_network_step(), "spring_network_step", BATCH, "llvm")
    artifact = kernel.artifact
    # The row template's buffers are positional.  Refuse any compiler ABI
    # drift instead of calling a changed kernel with stale pointer order.
    expected = (
        "anchor", "contact_enabled", "contact_height", "drag", "dt", "edge_0", "edge_1", "edge_2", "edge_3",
        "external_acceleration", "inverse_mass", "neighbor_0", "neighbor_1",
        "neighbor_2", "neighbor_3", "occupied", "pinned", "position",
        "rest_0", "rest_1", "rest_2", "rest_3", "stiffness", "velocity",
    )
    if tuple(kernel.argument_names) != expected or set(kernel.argument_names) != set(INPUT_NAMES):
        raise RuntimeError("spring row argument order changed")
    ids = tuple(kernel.argument_ids) + (kernel.output_ids["position_next"], kernel.output_ids["velocity_next"])
    if tuple(artifact.buffer_order) != ids or artifact.extent_order or len(ids) != 26:
        raise RuntimeError("spring row buffer ABI changed")
    if any(tuple(shape) != (BATCH,) for shape in artifact.buffer_shapes):
        raise RuntimeError("spring row batch shape changed")
    if any(dtype != "double" for dtype in artifact.buffer_dtypes):
        raise RuntimeError("spring row dtype changed")
    if artifact.library_path is None:
        raise RuntimeError("spring law was not linked")
    dll = directory / artifact.library_path.name
    shutil.copy2(artifact.library_path, dll)
    template = Path(__file__).resolve().parents[1] / "src" / "compiler" / "spring_network_nodus_row.cpp.in"
    source = directory / "spring_network_nodus_row.cpp"
    runtime_kernel = deploy_dir / dll.name if deploy_dir is not None else dll
    source.write_text(template.read_text(encoding="utf-8").replace("@DLL@", str(runtime_kernel).replace("\\", "\\\\"))
                      .replace("@ENTRY@", artifact.name), encoding="utf-8")
    receipt = {"schema": "nodus.spring-network-row.v1", "source": str(source),
               "kernel": str(dll), "entry": artifact.name, "networks": NETWORKS,
               "nodes_per_network": NODES, "axes": AXES, "batch": BATCH}
    if nodus_root is not None and nodus_build is not None:
        core = nodus_build / "Release" / "nodus_tensor_core.lib"
        if not (nodus_root / "include" / "tool_api.h").is_file() or not core.is_file():
            raise FileNotFoundError("Nodus headers or tensor-core import library are absent")
        cmake = directory / "CMakeLists.txt"
        acceptance = Path(__file__).resolve().parents[1] / "tests" / "spring_nodus_row_acceptance.cpp"
        demo = Path(__file__).resolve().parents[1] / "tests" / "spring_nodus_row_demo.cpp"
        host = Path(__file__).resolve().parents[1] / "src" / "compiler" / "spring_network_nodus_host.cpp"
        cmake.write_text("\n".join((
            "cmake_minimum_required(VERSION 3.20)",
            "project(spring_network_nodus_row LANGUAGES CXX)",
            f'add_library(spring_network_nodus_row SHARED "{source.as_posix()}")',
            f'target_include_directories(spring_network_nodus_row PRIVATE "{(nodus_root / "include").as_posix()}")',
            f'target_link_libraries(spring_network_nodus_row PRIVATE "{core.as_posix()}")',
            'set_target_properties(spring_network_nodus_row PROPERTIES CXX_STANDARD 17 '
            'RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" '
            'RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")',
            f'add_executable(spring_nodus_row_acceptance "{acceptance.as_posix()}")',
            f'target_include_directories(spring_nodus_row_acceptance PRIVATE "{(nodus_root / "include").as_posix()}")',
            f'target_link_libraries(spring_nodus_row_acceptance PRIVATE "{core.as_posix()}" ws2_32)',
            'set_target_properties(spring_nodus_row_acceptance PROPERTIES CXX_STANDARD 17 '
            'RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" '
            'RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")',
            f'add_executable(spring_nodus_row_demo "{demo.as_posix()}")',
            f'target_include_directories(spring_nodus_row_demo PRIVATE "{(nodus_root / "include").as_posix()}")',
            f'target_link_libraries(spring_nodus_row_demo PRIVATE "{core.as_posix()}" ws2_32)',
            'set_target_properties(spring_nodus_row_demo PROPERTIES CXX_STANDARD 17 '
            'RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" '
            'RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")',
            f'add_executable(spring_network_nodus_host "{host.as_posix()}")',
            f'target_include_directories(spring_network_nodus_host PRIVATE "{(nodus_root / "include").as_posix()}")',
            f'target_link_libraries(spring_network_nodus_host PRIVATE "{core.as_posix()}")',
            'set_target_properties(spring_network_nodus_host PROPERTIES CXX_STANDARD 17 '
            'RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" '
            'RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")',
            "",
        )), encoding="utf-8")
        build = directory / "build"
        subprocess.run(["cmake", "-S", str(directory), "-B", str(build)], check=True)
        subprocess.run(["cmake", "--build", str(build), "--config", "Release"], check=True)
        plugin = build / "bin" / "spring_network_nodus_row.dll"
        if not plugin.is_file():
            raise RuntimeError("Nodus spring row did not link")
        receipt["plugin"] = str(plugin)
        if deploy_dir is not None:
            deploy_dir.mkdir(parents=True, exist_ok=True)
            for product in (dll, plugin, build / "bin" / "spring_network_nodus_host.exe",
                            nodus_build / "Release" / "nodus_tensor_core.dll"):
                shutil.copy2(product, deploy_dir / product.name)
            receipt["deployed"] = str(deploy_dir)
        if accept:
            environment = os.environ.copy()
            environment["PATH"] = str(nodus_build / "Release") + os.pathsep + environment.get("PATH", "")
            result = build / "bin" / "spring_nodus_row_acceptance.exe"
            subprocess.run([str(result), str(plugin)], check=True, env=environment)
            receipt["acceptance"] = "passed"
    (directory / "spring_network_row.receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path, help="User-owned module directory")
    parser.add_argument("--nodus-root", type=Path)
    parser.add_argument("--nodus-build", type=Path)
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--deploy-dir", type=Path, help="copy linked runtime products to the service area; compilation remains in --output")
    args = parser.parse_args()
    if (args.nodus_root is None) != (args.nodus_build is None):
        parser.error("--nodus-root and --nodus-build must be supplied together")
    if args.accept and args.nodus_root is None:
        parser.error("--accept requires --nodus-root and --nodus-build")
    if args.deploy_dir and args.nodus_root is None:
        parser.error("--deploy-dir requires --nodus-root and --nodus-build")
    print(json.dumps(stage(args.output, args.nodus_root, args.nodus_build, args.accept, args.deploy_dir), indent=2))


if __name__ == "__main__":
    main()

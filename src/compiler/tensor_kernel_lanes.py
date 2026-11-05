"""Tensor-kernel lanes: who defines the repository's tensor kernel symbols.

The ``repository`` lane (the default) is the authored LLVM kernel module:
every kernel an emitted program calls is pulled in as its handwritten
definition.  The ``eigen`` lane answers the same symbols with Eigen: the
call is emitted against an external declaration with the authored
signature, and the compiled Eigen object joins the artifact's
``linked_objects`` (digest-checked by ``compile_artifact``).

The lane is chosen by the active work contract's ``tensor_kernel_lane``;
the ``eigen`` preset names it.  Eigen's GEMM is not the compensated inner
product, so a product built under this lane is a different numerical
contract, which is why the switch is a contract and not a flag.

The Eigen headers are found at ``TURING_EIGEN_INCLUDE`` or, by default,
in the workspace's fftfree build (``fftfree/build/_deps/eigen-src``).
A missing header tree is a named refusal.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

HONORED_LANES = ("repository", "eigen")

# Symbol -> the external declaration with the authored signature.
EIGEN_SYMBOLS: dict[str, str] = {
    "matmul_double": "declare void @matmul_double(ptr, ptr, ptr, i32, i32, i32)",
}

_SOURCE = (Path(__file__).resolve().parents[1] / "common" / "tensors"
           / "accelerator_backends" / "c_backend" / "eigen_kernels.cpp")
# x86-64-v3 (AVX2 + FMA): the workspace's Eigen headers predate this clang
# and their AVX-512 packet math does not compile under it (pfirst return
# type), while -march=native on an AVX-512 host selects it.  The AVX2 + FMA
# packet paths compile.
# -fno-threadsafe-statics: Eigen's function-local statics would otherwise
# call the C++ runtime's __cxa_guard_acquire/release, which the plain C link
# of an LLVM product does not carry.  The kernels run single-threaded per call.
_FLAGS = ("-O3", "-mcpu=x86_64_v3", "-fno-exceptions", "-fno-rtti", "-fno-threadsafe-statics",
          "-DEIGEN_NO_DEBUG", "-DEIGEN_DONT_PARALLELIZE")


def eigen_include() -> Path:
    explicit = os.environ.get("TURING_EIGEN_INCLUDE")
    candidates = [Path(explicit)] if explicit else []
    candidates.append(Path(__file__).resolve().parents[3] / "fftfree" / "build" / "_deps" / "eigen-src")
    for candidate in candidates:
        if (candidate / "Eigen" / "Core").is_file():
            return candidate
    raise FileNotFoundError(
        "the eigen tensor-kernel lane needs the Eigen headers: set "
        "TURING_EIGEN_INCLUDE to the directory holding Eigen/Core "
        f"(looked in {[str(c) for c in candidates]})")


def active_lane() -> str:
    from .work_contract import active_contract
    return str(getattr(active_contract(), "tensor_kernel_lane", "repository"))


def lane_declaration(symbol: str) -> str | None:
    """The external declaration that replaces ``symbol``'s authored
    definition under the active lane, or None (authored definition)."""
    if active_lane() != "eigen":
        return None
    return EIGEN_SYMBOLS.get(symbol)


def eigen_object() -> tuple[str, str]:
    """The compiled Eigen kernel object as ``(path, sha256)``, built once
    per source+flags+headers and cached under ``build/tensor_kernel_lanes``."""
    include = eigen_include()
    source = _SOURCE.read_text(encoding="utf-8")
    key = hashlib.sha256("\0".join((source, *_FLAGS, str(include))).encode("utf-8")).hexdigest()[:16]
    directory = Path(__file__).resolve().parents[2] / "build" / "tensor_kernel_lanes"
    directory.mkdir(parents=True, exist_ok=True)
    obj = directory / f"eigen_kernels_{key}.obj"
    if not obj.is_file():
        partial = obj.with_suffix(".partial.obj")
        command = [sys.executable, "-m", "ziglang", "c++", "-c", *_FLAGS,
                   "-I", str(include), str(_SOURCE), "-o", str(partial)]
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode or not partial.is_file():
            raise RuntimeError(f"eigen kernel object failed to build:\n{(done.stderr or done.stdout)[-3000:]}")
        partial.replace(obj)
    digest = hashlib.sha256(obj.read_bytes()).hexdigest()
    return str(obj), digest

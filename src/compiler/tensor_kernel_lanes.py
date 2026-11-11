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
from .artifact_locations import compiler_artifact_root

HONORED_LANES = ("repository", "eigen")

# Symbol -> the external declaration with the authored signature.
EIGEN_SYMBOLS: dict[str, str] = {
    "matmul_double": "declare void @matmul_double(ptr, ptr, ptr, i32, i32, i32)",
    # Same symbol and layout as the authored kernel; nested loops in place of
    # the per-element div/rem decode.
    "unfold2d_double": "declare void @unfold2d_double(ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)",
    # Lane-only whole operations (no authored repository definition):
    # softmax over the last axis of a (rows, cols) block, and an elementwise
    # binary over operands addressed by element strides (0 on a broadcast
    # axis), so no broadcast copy is materialized.
    "softmax_lastdim_double": "declare void @softmax_lastdim_double(ptr, ptr, i32, i32)",
    "broadcast_binary_double": "declare void @broadcast_binary_double(ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32)",
}

# The capability manifest: the operations each lane performs WHOLE, so the
# front end and the promotion identities keep them as one node instead of
# decomposing them.  An operation outside its lane's set is decomposed
# through the minimal tensor vocabulary as before.
#   matmul / batched_matmul  -- matmul_double / batched_matmul_indexed_double
#   softmax_lastdim          -- softmax_lastdim_double
#   broadcast_binary         -- broadcast_binary_double for BROADCAST_BINARY_OPCODES
LANE_CAPABILITIES: dict[str, frozenset[str]] = {
    "repository": frozenset({"matmul", "batched_matmul"}),
    "eigen": frozenset({"matmul", "batched_matmul", "softmax_lastdim", "broadcast_binary"}),
}
# Binary opcodes (CTensorOp numbering) broadcast_binary_double implements, with
# the repository's exact semantics: add, sub, mul, truediv, maximum
# (a > b ? a : b), minimum (a < b ? a : b).
BROADCAST_BINARY_OPCODES = frozenset({0, 1, 2, 3, 26, 27})
BROADCAST_MAX_RANK = 8


def lane_capabilities(lane: str | None = None) -> frozenset[str]:
    return LANE_CAPABILITIES[lane or active_lane()]


def lane_handles(operation: str, lane: str | None = None) -> bool:
    """Does the active lane perform ``operation`` whole?"""
    return operation in lane_capabilities(lane)

_SOURCE = (Path(__file__).resolve().parents[1] / "common" / "tensors"
           / "accelerator_backends" / "c_backend" / "eigen_kernels.cpp")
# Built for this host (-march=native: AVX-512 on the development machine).
# Released Eigen (3.4.0, 3.4.1, 5.0.0) does not compile its packet math under
# zig's clang 21 (``pfirst`` of a ``_mm_permute_ps`` result is not recognized
# as Packet4f); Eigen's development head does, so it is preferred below.
_BASE_FLAGS = ("-O3", "-fno-exceptions", "-fno-rtti",
               # Eigen's function-local statics would otherwise call the C++
               # runtime's __cxa_guard_acquire/release, which the plain C link
               # of an LLVM product does not carry.
               "-fno-threadsafe-statics",
               "-DEIGEN_NO_DEBUG", "-DEIGEN_DONT_PARALLELIZE")
# The numeric contract's ``float_arithmetic`` knob: ``fast`` licenses
# reassociation and FMA contraction while keeping inf/NaN honored.
_FLOAT_ARITHMETIC_FLAGS = {
    "fast": ("-ffast-math", "-fno-finite-math-only"),
    "strict": ("-ffp-contract=off",),
}


def _flags(include: Path) -> tuple[str, ...]:
    """Host target for headers whose packet math compiles under this clang
    (Eigen's development head); a released Eigen (the fftfree fallback) is
    limited to x86-64-v3 (AVX2 + FMA)."""
    from .ssa_call_input_adapters import float_arithmetic_policy
    target = ("-march=native",) if include.name == "eigen-master" else ("-mcpu=x86_64_v3",)
    return _BASE_FLAGS + target + _FLOAT_ARITHMETIC_FLAGS[float_arithmetic_policy()]


def eigen_include() -> Path:
    explicit = os.environ.get("TURING_EIGEN_INCLUDE")
    candidates = [Path(explicit)] if explicit else []
    candidates.append(Path(__file__).resolve().parents[2] / "build" / "third_party" / "eigen-master")
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
    flags = _flags(include)
    key = hashlib.sha256("\0".join((source, *flags, str(include))).encode("utf-8")).hexdigest()[:16]
    directory = compiler_artifact_root() / "build" / "tensor_kernel_lanes"
    directory.mkdir(parents=True, exist_ok=True)
    obj = directory / f"eigen_kernels_{key}.obj"
    if not obj.is_file():
        partial = obj.with_suffix(".partial.obj")
        command = [sys.executable, "-m", "ziglang", "c++", "-c", *flags,
                   "-I", str(include), str(_SOURCE), "-o", str(partial)]
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode or not partial.is_file():
            raise RuntimeError(f"eigen kernel object failed to build:\n{(done.stderr or done.stdout)[-3000:]}")
        partial.replace(obj)
    digest = hashlib.sha256(obj.read_bytes()).hexdigest()
    return str(obj), digest

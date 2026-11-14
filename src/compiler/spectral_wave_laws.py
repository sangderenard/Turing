"""Spectral T4 wave laws lowered through the repository's symbolic compiler.

Each law acts on columns of independent frequency bins, aperture pixels, or
Jones samples. The FFT, arena ownership, and optical path scheduling remain
with Spectral's T4 implementation. These equations match the arithmetic in
``spectral-analyzer/csrc/kernels/wave_t4.cpp``; domain selection is explicit so
an evanescent bin never evaluates the propagating square root (or vice versa).
"""

from __future__ import annotations

from pathlib import Path

import sympy as sp

from .native_package import piece_from_law
from .symbolic_equation_compiler import compile_sympy_equations


TAU = sp.Float("6.283185307179586476925286766559")


def _symbols(*names: str) -> tuple[sp.Symbol, ...]:
    return tuple(sp.Symbol(name, real=True) for name in names)


def wave_equations() -> dict[str, tuple[sp.Equality, ...]]:
    """The authored T4 arithmetic, split at the source's physical domains."""

    ar, ai, kx, ky, wavelength, distance, direction = _symbols(
        "ar", "ai", "kx", "ky", "wavelength", "distance", "direction")
    kz2 = (TAU / wavelength) ** 2 - kx ** 2 - ky ** 2
    phase = distance * sp.sqrt(kz2)
    transfer_re = sp.cos(phase)
    transfer_im = direction * sp.sin(phase)
    propagating = (
        sp.Eq(sp.Symbol("out_re"), ar * transfer_re - ai * transfer_im,
              evaluate=False),
        sp.Eq(sp.Symbol("out_im"), ar * transfer_im + ai * transfer_re,
              evaluate=False),
    )

    # The caller supplies positive -kz2 for bins selected by kz2 < 0.
    gap, = _symbols("gap")
    decay = sp.exp(-distance * sp.sqrt(gap))
    evanescent = (
        sp.Eq(sp.Symbol("out_re"), ar * decay, evaluate=False),
        sp.Eq(sp.Symbol("out_im"), ai * decay, evaluate=False),
    )

    real_index, background_index, imaginary_index = _symbols(
        "real_index", "background_index", "imaginary_index")
    phase = direction * (TAU / wavelength) * (
        real_index - background_index) * distance
    attenuation = sp.exp(-TAU * imaginary_index * distance / wavelength)
    material_re = attenuation * sp.cos(phase)
    material_im = attenuation * sp.sin(phase)
    aperture = (
        sp.Eq(sp.Symbol("out_re"), ar * material_re - ai * material_im,
              evaluate=False),
        sp.Eq(sp.Symbol("out_im"), ar * material_im + ai * material_re,
              evaluate=False),
    )

    sr, si, pr, pi, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im = _symbols(
        "sr", "si", "pr", "pi", "a_re", "a_im", "b_re", "b_im",
        "c_re", "c_im", "d_re", "d_im")
    jones = (
        sp.Eq(sp.Symbol("out_s_re"), a_re * sr - a_im * si + b_re * pr - b_im * pi,
              evaluate=False),
        sp.Eq(sp.Symbol("out_s_im"), a_re * si + a_im * sr + b_re * pi + b_im * pr,
              evaluate=False),
        sp.Eq(sp.Symbol("out_p_re"), c_re * sr - c_im * si + d_re * pr - d_im * pi,
              evaluate=False),
        sp.Eq(sp.Symbol("out_p_im"), c_re * si + c_im * sr + d_re * pi + d_im * pr,
              evaluate=False),
    )
    return {
        "angular_propagating": propagating,
        "angular_evanescent": evanescent,
        "aperture_material": aperture,
        "jones_interface": jones,
    }


def compile_wave_kernel(kind: str, batch: int, *, directory: Path | str):
    """Compile one named law as a native batch kernel via AbstractTensor/SSA."""

    equations = wave_equations()
    if kind not in equations:
        raise KeyError(kind)
    if batch < 1:
        raise ValueError("wave kernel batch must be positive")
    name = f"spectral_wave_{kind}"
    compilation = compile_sympy_equations(equations[kind], name=name,
                                          dtype="float64")
    return piece_from_law(compilation, name, batch, directory=directory)


def compile_wave_compute_shader(kind: str, *, local_size: int = 64):
    """Emit the same law's f32 scalar SSA as a deployable GPU shader.

    The f64 native pieces above preserve T4's transfer-table arithmetic. GPU
    transcendental operations require f32 in the current SPIR-V backend; the
    artifact reports any unsupported operation instead of narrowing itself.
    """

    from .ssa_spirv_backend import emit_deployable_compute_shader

    equations = wave_equations()
    if kind not in equations:
        raise KeyError(kind)
    name = f"spectral_wave_{kind}_f32"
    compiled = compile_sympy_equations(equations[kind], name=name,
                                       dtype="float32")
    values = {
        instruction.res.id: instruction.res
        for block in compiled.function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    }
    outputs = tuple(values[compiled.output_ids[str(eq.lhs)]]
                    for eq in equations[kind])
    return emit_deployable_compute_shader(
        compiled.function, outputs=outputs, dtype="float32",
        local_size=local_size, name=name, module=compiled.module)


def publish_wave_products(directory: Path | str) -> dict[str, Path]:
    """Publish the four deployable products with the existing shader records."""

    from .shader_deployment import publish_spirv_product

    root = Path(directory)
    products = {}
    for kind in wave_equations():
        product = publish_spirv_product(root / kind,
                                        compile_wave_compute_shader(kind))
        if product.module_path is None:
            raise RuntimeError(f"{kind}: no validated SPIR-V product")
        products[kind] = product.module_path
    return products


__all__ = ["TAU", "wave_equations", "compile_wave_kernel",
           "compile_wave_compute_shader", "publish_wave_products"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for name, path in publish_wave_products(args.output).items():
        print(f"{name}\t{path}")

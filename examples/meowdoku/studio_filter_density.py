"""Continuous filter density through the existing symbolic/SSA compiler.

No raster texture or alternative compiler. Authoring accepts a SymPy Lambda,
not an evaluated user-code string. The native result is an absorption-only
local filter law, not registration with a ray reader or a refracting volume.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import sympy as sp

from src.compiler.symbolic_equation_compiler import compile_sympy_equations, SymbolicPublication
from src.compiler.ssa_c_backend import emit_ssa_function_to_c
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
from src.compiler.ssa_webgpu_backend import emit_module as emit_webgpu

U, V = sp.symbols("u v")
SIGMA, PATH = sp.symbols("sigma_a_per_m path_m")


def bezier_density(degree=3):
    """Tensor-product Bernstein density over [0,1]^2, relative concentration.

    Nonnegative controls guarantee nonnegative density throughout the patch.
    The controls are runtime values; the polynomial itself is compile-static.
    """
    if type(degree) is not int or not 1 <= degree <= 3:
        raise ValueError("Density Bezier degree must be 1, 2 or 3")
    controls = tuple(sp.Symbol(f"density_{i}_{j}") for i in range(degree+1) for j in range(degree+1))
    basis = lambda x, i: sp.binomial(degree,i)*x**i*(1-x)**(degree-i)
    expression = sum(controls[i*(degree+1)+j]*basis(U,i)*basis(V,j)
                     for i in range(degree+1) for j in range(degree+1))
    return sp.Lambda((U,V),expression)


def _validate_lambda(density):
    if not isinstance(density,sp.Lambda) or density.variables != (U,V):
        raise ValueError("Density needs an authored symbolic Lambda((u,v), expression)")
    if density.expr.has(sp.Derivative,sp.Integral,sp.Sum,sp.Product) or sp.count_ops(density.expr)>1500:
        raise ValueError("Density must be a bounded explicit expression")
    if density.expr.free_symbols & {SIGMA,PATH}:
        raise ValueError("Density cannot redefine absorption coefficient or path length")
    if any(not str(s).isidentifier() for s in density.expr.free_symbols):
        raise ValueError("Density parameter names must be identifiers")
    return density


def compile_density(density=None):
    density=_validate_lambda(bezier_density() if density is None else density)
    rho=density.expr
    # Spectral OpticalMaterial.transmittance uses this same Beer-Lambert law;
    # sigma_a is supplied at the actual ray wavelength, in reciprocal metres.
    tau=SIGMA*PATH*rho
    equations=(sp.Eq(sp.Symbol("relative_density"),rho,evaluate=False),
               sp.Eq(sp.Symbol("optical_depth"),tau,evaluate=False),
               sp.Eq(sp.Symbol("transmittance"),sp.exp(-tau),evaluate=False))
    return compile_sympy_equations(equations,name="studio_filter_density",dtype="float32",publications=(
        SymbolicPublication("relative_density","optics.filter.relative_concentration",unit="1"),
        SymbolicPublication("optical_depth","optics.filter.optical_depth",unit="1"),
        SymbolicPublication("transmittance","optics.filter.power_transmittance",unit="1")))


def neutral_density_sigma(optical_density, thickness_m):
    """Wavelength-neutral alpha from base-10 ND at a reference path thickness.

    ND 1 means 10% power transmission. It is not concentration or opacity.
    Constant extinction k would NOT be neutral: alpha=4*pi*k/lambda varies.
    """
    optical_density=float(optical_density);thickness_m=float(thickness_m)
    if not math.isfinite(optical_density) or optical_density<0 or not math.isfinite(thickness_m) or thickness_m<=0:
        raise ValueError("ND must be nonnegative and reference thickness positive")
    return math.log(10)*optical_density/thickness_m


def sigma_from_extinction(k,wavelength_um):
    """Same units and relation as Spectral OpticalMaterial.transmittance."""
    k=float(k);wavelength_um=float(wavelength_um)
    if not math.isfinite(k) or k<0 or not math.isfinite(wavelength_um) or wavelength_um<=0:
        raise ValueError("Invalid extinction or wavelength")
    return 4*math.pi*k/(wavelength_um*1e-6)


def straight_filter_path(thickness_m, cosine, *, spatially_uniform=False):
    """Uniform index-matched slab path; no implicit grazing clamp.

    A varying 2D field is validated only at normal incidence here. At oblique
    incidence it needs its density integral along the actual in-material ray,
    including side exit and curved aperture boundaries. Reject, never pretend
    the entry-point density is that integral.
    """
    thickness_m=float(thickness_m);cosine=abs(float(cosine))
    if not math.isfinite(thickness_m) or thickness_m<0 or not math.isfinite(cosine) or not 0<cosine<=1:
        raise ValueError("Invalid filter path or grazing ray")
    if cosine!=1 and not spatially_uniform:
        raise ValueError("Oblique varying-density filters need an actual path integral")
    return thickness_m/cosine


@dataclass
class DensityProduct:
    native: object
    compilation: object
    manifest: dict

    def sample(self, values):
        values={str(k):float(v) for k,v in values.items()}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Filter inputs must be finite")
        for key in ("u","v"):
            if key in values and not 0<=values[key]<=1:
                raise ValueError("Filter point is outside its authored UV patch")
        if values.get("sigma_a_per_m",-1)<0 or values.get("path_m",-1)<0:
            raise ValueError("Absorption and in-material path must be nonnegative")
        if any(value<0 for key,value in values.items() if key.startswith("density_")):
            raise ValueError("Bezier density controls must be nonnegative")
        output=dict(zip(self.native.output_names,self.native.run(values)))
        if not all(math.isfinite(v) for v in output.values()) or output["relative_density"]<0 or output["optical_depth"]<0 or not 0<=output["transmittance"]<=1:
            raise ValueError("Compiled custom density produced an invalid absorbing filter")
        return output


def build_density(directory,density=None):
    """Compile and link the actual shared SSA C product; also emit LLVM/WGSL.

    Unsupported backend operations remain explicit errors. No fallback to
    SymPy evaluation or a hand-authored shader is used.
    """
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    density=_validate_lambda(bezier_density() if density is None else density)
    compilation=compile_density(density)
    native=emit_ssa_function_to_c(compilation.module,compilation.function.name)
    llvm=emit_ssa_function_to_llvm(compilation.module,compilation.function.name)
    outputs=compilation.function.blocks["entry"].instrs[-1].args
    wgsl=emit_webgpu(compilation.module,name="studio_filter_density",outputs={compilation.function.name:outputs},count=1)
    for name,artifact in (("C",native),("LLVM",llvm),("WGSL",wgsl)):
        if not artifact.complete:
            raise RuntimeError(f"Filter {name} emission incomplete: {artifact.shortfalls}")
    native.compile(directory)
    (directory/'studio_filter_density.ll').write_text(llvm.llvm_ir,encoding='utf-8')
    (directory/'studio_filter_density.wgsl').write_text(wgsl.source,encoding='utf-8')
    expression=sp.srepr(density)
    manifest={"schema":"turing.studio-filter-density.v1","expression":expression,
        "expression_sha256":hashlib.sha256(expression.encode()).hexdigest(),
        "compiler":"symbolic_equation_compiler.compile_sympy_equations -> repository SSA",
        "numeric":"float32","domain":"u,v in [0,1]", "density_unit":"relative concentration",
        "absorption_unit":"1/metre at the actual ray wavelength", "path_unit":"metres inside the medium",
        "law":"T=exp(-sigma_a(lambda)*relative_density(u,v)*path_m)",
        "inputs":native.input_names,"outputs":native.output_names,
        "optics":"absorption only; no interface Fresnel, refraction, scattering or diffraction",
        "ray_integration":"not registered with the production mixed-scene reader",
        "artifacts":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('studio_filter_density.*') if not p.name.endswith('.manifest.json')}}
    (directory/'studio_filter_density.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return DensityProduct(native,compilation,manifest)

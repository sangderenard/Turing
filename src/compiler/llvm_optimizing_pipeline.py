"""An optimizing LLVM compilation path for Turing's handwritten SSA kernels.

The existing JIT path parses the handwritten module, verifies it, and hands it
straight to MCJIT built with ``create_target_machine(opt=0)``.  No pass pipeline
runs, the target is the default triple with no CPU features, and the handwritten
kernels carry no aliasing or fast-math metadata.  The result is correct, generic,
scalar code: LLVM is never asked to optimize anything.

This module is the optimizing counterpart.  It is deliberately separate from
``llvm_jit_backend`` so the unoptimized path stays available as a differential
reference — when vectorized output disagrees with scalar output, you need both.

Three things are required to get vector code out of LLVM, and all three are
missing by default:

1. **A pass pipeline.**  ``PassBuilder`` with a speed level actually runs
   inlining, LICM, unrolling and the vectorizers.
2. **A specific target.**  The default triple implies a baseline CPU.  Naming the
   host unlocks its vector width (AVX/FMA here).
3. **Aliasing facts.**  Given ``ptr %a, ptr %b, ptr %out``, LLVM must assume the
   buffers overlap and will refuse to vectorize.  ``noalias`` is what Fortran
   gets from its language rules and C needs ``restrict`` for.

Fast-math is offered but defaults **off**: reassociation changes results, and
this repository verifies backends against each other numerically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_DEFINE = re.compile(r"^define\b[^{]*?@(?P<name>[\w.$]+)\s*\((?P<params>[^)]*)\)")
_FLOAT_OP = re.compile(
    r"(?P<head>=\s*)(?P<op>fadd|fsub|fmul|fdiv|frem)\s+(?!fast\b|nnan\b|ninf\b|reassoc\b|contract\b)"
)


@dataclass(frozen=True)
class OptimizationProfile:
    """How aggressively to compile, and which liberties are permitted."""

    speed_level: int = 3
    size_level: int = 0
    loop_vectorization: bool = True
    slp_vectorization: bool = True
    loop_unrolling: bool = True
    loop_interleaving: bool = True
    inlining_threshold: int | None = 275

    # Target selection.  The host is the right default for a JIT; a fixed
    # triple is the right default for reproducible artifacts.
    use_host_cpu: bool = True
    cpu: str = ""
    features: str = ""

    # IR hardening applied before optimization.
    annotate_noalias: bool = True
    fast_math: bool = False

    # Vector width the compiler should prefer, in bits.  ``None`` lets LLVM
    # decide, which is not always right: AMD Bulldozer-family cores (bdver1-4)
    # split a 256-bit AVX operation into two 128-bit halves internally, so wider
    # vectors buy no throughput and cost the split plus vzeroupper.  GCC's
    # -march=native encodes that tuning; LLVM's target machine does not apply it
    # here, so it is worth stating explicitly.
    prefer_vector_width: int | None = None

    @property
    def opt(self) -> int:
        return max(0, min(3, int(self.speed_level)))


REFERENCE_PROFILE = OptimizationProfile(
    speed_level=0,
    loop_vectorization=False,
    slp_vectorization=False,
    loop_unrolling=False,
    loop_interleaving=False,
    inlining_threshold=None,
    use_host_cpu=False,
    annotate_noalias=False,
    fast_math=False,
)


def _binding():
    """Return the llvmlite binding.

    llvmlite 0.47 initializes LLVM on import and raises if the old explicit
    ``initialize()`` calls are made, so this deliberately does not make them.
    """

    from llvmlite import binding as llvm

    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    return llvm


def annotate_pointer_parameters(llvm_ir: str) -> str:
    """Mark pointer parameters ``noalias`` so the vectorizer is permitted to run.

    The handwritten kernels take distinct operand and destination buffers.  That
    fact is true by construction of the calling convention but is invisible to
    LLVM, which must otherwise assume every store may clobber a later load.
    """

    lines = llvm_ir.splitlines()
    out: list[str] = []
    for line in lines:
        match = _DEFINE.match(line)
        if match is None:
            out.append(line)
            continue
        params = match.group("params")
        if not params.strip():
            out.append(line)
            continue
        rewritten = []
        for param in params.split(","):
            token = param.strip()
            if token.startswith("ptr") and "noalias" not in token:
                token = token.replace("ptr", "ptr noalias", 1)
            rewritten.append(token)
        line = (
            line[: match.start("params")]
            + ", ".join(rewritten)
            + line[match.end("params") :]
        )
        out.append(line)
    return "\n".join(out) + ("\n" if llvm_ir.endswith("\n") else "")


def apply_fast_math(llvm_ir: str) -> str:
    """Attach ``fast`` to floating point arithmetic.

    This permits reassociation and FMA contraction.  It changes results and is
    therefore never applied unless explicitly requested.
    """

    return _FLOAT_OP.sub(lambda m: f"{m.group('head')}{m.group('op')} fast ", llvm_ir)


_ATTRIBUTE_GROUP = 91


def apply_prefer_vector_width(llvm_ir: str, width: int) -> str:
    """Attach a ``prefer-vector-width`` function attribute to every definition.

    There is no target-machine knob for this in llvmlite, so it is expressed the
    way LLVM itself does: an attribute group referenced by each ``define``.
    """

    marker = f"#{_ATTRIBUTE_GROUP}"
    if marker in llvm_ir:
        return llvm_ir

    def attach(match: "re.Match[str]") -> str:
        return f"{match.group(0).rstrip()[:-1].rstrip()} {marker} {{"

    rewritten = re.sub(r"^define\b[^{]*\{", attach, llvm_ir, flags=re.MULTILINE)
    return (
        rewritten.rstrip("\n")
        + f'\n\nattributes {marker} = {{ "prefer-vector-width"="{width}" }}\n'
    )


def harden_ir(llvm_ir: str, profile: OptimizationProfile) -> str:
    """Apply the metadata LLVM needs before optimization can do anything."""

    if profile.annotate_noalias:
        llvm_ir = annotate_pointer_parameters(llvm_ir)
    if profile.fast_math:
        llvm_ir = apply_fast_math(llvm_ir)
    if profile.prefer_vector_width is not None:
        llvm_ir = apply_prefer_vector_width(
            llvm_ir, profile.prefer_vector_width
        )
    return llvm_ir


# CPU families whose floating point units are 128 bits wide, so a 256-bit AVX
# operation is split into two halves internally.  On these, preferring wider
# vectors buys no throughput and costs the split plus vzeroupper -- measured
# here at ~2.3x slower than 128-bit on bdver2.  GCC's -march=native applies this
# tuning; LLVM's target machine does not, so it must be stated.
_NARROW_VECTOR_CPUS = frozenset(
    {
        "bdver1",
        "bdver2",
        "bdver3",
        "bdver4",
        "btver1",
        "btver2",
        "znver1",
    }
)


def host_preferred_vector_width() -> int | None:
    """Vector width to prefer on this host, or ``None`` to let LLVM decide.

    Returning ``None`` is the right answer on Zen 2+ and modern Intel, which
    have full-width 256-bit units; forcing 128 there would give up half the
    throughput.  This is deliberately a narrow allow-list rather than a guess.
    """

    llvm = _binding()
    try:
        cpu = llvm.get_host_cpu_name()
    except Exception:
        return None
    return 128 if cpu in _NARROW_VECTOR_CPUS else None


def tuned_host_profile(**overrides: Any) -> OptimizationProfile:
    """The configuration measured to reach gfortran parity on this host."""

    settings: dict[str, Any] = {
        "speed_level": 3,
        "use_host_cpu": True,
        "annotate_noalias": True,
        "prefer_vector_width": host_preferred_vector_width(),
    }
    settings.update(overrides)
    return OptimizationProfile(**settings)


def target_machine(profile: OptimizationProfile, *, jit: bool = True):
    """Build a target machine that names a real CPU instead of the baseline."""

    llvm = _binding()
    target = llvm.Target.from_default_triple()
    if profile.use_host_cpu:
        cpu = profile.cpu or llvm.get_host_cpu_name()
        features = profile.features or llvm.get_host_cpu_features().flatten()
    else:
        cpu = profile.cpu
        features = profile.features
    return target.create_target_machine(
        cpu=cpu,
        features=features,
        opt=profile.opt,
        jit=jit,
    )


def run_pipeline(module, machine, profile: OptimizationProfile) -> None:
    llvm = _binding()
    pto = llvm.PipelineTuningOptions(
        speed_level=profile.opt,
        size_level=profile.size_level,
    )
    pto.loop_vectorization = profile.loop_vectorization
    pto.slp_vectorization = profile.slp_vectorization
    pto.loop_unrolling = profile.loop_unrolling
    pto.loop_interleaving = profile.loop_interleaving
    if profile.inlining_threshold is not None:
        pto.inlining_threshold = profile.inlining_threshold
    builder = llvm.PassBuilder(machine, pto)
    manager = builder.getModulePassManager()
    manager.run(module, builder)


@dataclass
class OptimizedModule:
    """The result of compiling one module, with evidence of what changed."""

    profile: OptimizationProfile
    source_ir: str
    hardened_ir: str
    optimized_ir: str
    assembly: str
    cpu: str
    features_enabled: int = 0
    _machine: Any = field(default=None, repr=False)

    @property
    def vector_instruction_count(self) -> int:
        """Count *packed* SIMD instructions in the emitted assembly.

        Counting ``xmm`` registers would overstate this badly: scalar SSE uses
        ``xmm`` too, so unoptimized scalar code (``vaddsd``, ``vmulsd``) appears
        vectorized when it is not.  Only the packed forms — ``addpd``, ``mulps``
        and friends, with an optional AVX ``v`` prefix — mean real SIMD.
        """

        return len(re.findall(r"\bv?[a-z]+[24]?p[sd]\b", self.assembly))

    @property
    def vectorized(self) -> bool:
        return self.vector_instruction_count > 0

    def summary(self) -> str:
        return (
            f"cpu={self.cpu} opt={self.profile.opt} "
            f"noalias={self.profile.annotate_noalias} "
            f"fastmath={self.profile.fast_math} "
            f"vector_regs={self.vector_instruction_count} "
            f"ir_lines={len(self.optimized_ir.splitlines())}"
        )


def optimize_ir(
    llvm_ir: str,
    profile: OptimizationProfile | None = None,
) -> OptimizedModule:
    """Harden, optimize and emit assembly for one LLVM module."""

    profile = profile or OptimizationProfile()
    llvm = _binding()
    hardened = harden_ir(llvm_ir, profile)
    machine = target_machine(profile)
    module = llvm.parse_assembly(hardened)
    module.verify()
    if profile.opt > 0:
        run_pipeline(module, machine, profile)
    assembly = machine.emit_assembly(module)
    cpu = llvm.get_host_cpu_name() if profile.use_host_cpu else (profile.cpu or "generic")
    return OptimizedModule(
        profile=profile,
        source_ir=llvm_ir,
        hardened_ir=hardened,
        optimized_ir=str(module),
        assembly=assembly,
        cpu=cpu,
        _machine=machine,
    )


class OptimizingJITProgram:
    """MCJIT program built through the optimizing pipeline."""

    def __init__(
        self,
        llvm_ir: str,
        *,
        profile: OptimizationProfile | None = None,
    ):
        profile = profile or OptimizationProfile()
        llvm = _binding()
        self.result = optimize_ir(llvm_ir, profile)
        machine = target_machine(profile)
        module = llvm.parse_assembly(self.result.hardened_ir)
        module.verify()
        if profile.opt > 0:
            run_pipeline(module, machine, profile)
        engine = llvm.create_mcjit_compiler(module, machine)
        engine.finalize_object()
        engine.run_static_constructors()
        self._engine = engine
        self._machine = machine
        self.profile = profile

    def address(self, symbol: str) -> int:
        found = int(self._engine.get_function_address(symbol))
        if not found:
            raise RuntimeError(f"optimizing JIT did not expose {symbol!r}")
        return found


def compare_profiles(
    llvm_ir: str,
    *,
    optimized: OptimizationProfile | None = None,
) -> tuple[OptimizedModule, OptimizedModule]:
    """Compile the same module unoptimized and optimized, for differential use."""

    return (
        optimize_ir(llvm_ir, REFERENCE_PROFILE),
        optimize_ir(llvm_ir, optimized or OptimizationProfile()),
    )


__all__ = [
    "OptimizationProfile",
    "OptimizedModule",
    "OptimizingJITProgram",
    "REFERENCE_PROFILE",
    "annotate_pointer_parameters",
    "apply_fast_math",
    "compare_profiles",
    "harden_ir",
    "optimize_ir",
    "host_preferred_vector_width",
    "run_pipeline",
    "target_machine",
    "tuned_host_profile",
]

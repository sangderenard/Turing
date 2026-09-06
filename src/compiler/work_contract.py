"""The work contract: what a compilation is for, stated once, as presets.

The pipeline has accumulated independent switches that all answer the same
question -- "how faithful must this artifact be, and to whom?" -- from
different corners:

* the identity policy (``ir_identities``: exact-only vs the bit-changing set),
* multiply-add contraction (``ssa_llvm_backend``: ``contract`` + host target),
* emission register reuse (the slot-keyed cache: evaporate redundant loads),
* the fusion level (``fusion_levels``: REGIONS vs FUSED honored today),
* the diagnostic channels (``watch``/``history``/``text_sink``: values that
  must stay observable in real storage).

Any one of these chosen alone is a local decision; together they are a
CONTRACT for the work, and the combinations that make sense are few. This
module names them. A preset is a complete, internally consistent answer, so a
caller states intent ("prove", "develop", "deploy", "fast") instead of
recalling which five switches cooperate.

The presets
-----------

``prove``    Conservative equality-proving form. Every value lives in and is
             re-read from its pool slot; no register reuse, no identity that
             changes bits, no contraction. This is the shape you diff two
             backends over, value by value.
``develop``  The default. In-place pool composition with same-block register
             reuse and the EXACT identity set only -- bit-identical to
             ``prove`` by construction, measured 6x faster on the fluid
             flagship. Diagnostics fully available (stores never evaporate).
``deploy``   ``develop`` plus the inexact identity set (sqrt family). Changes
             bits within documented bounds (the fluid's mass_err <= 1e-15
             gate held); still no contraction, so results are stable across
             hosts.
``fast``     Everything: inexact identities, multiply-add contraction, host
             target named. Bit-stability across machines is explicitly
             surrendered (fma availability differs by CPU). ~10x measured.

Diagnostics are compatible with every preset TODAY because the register
cache only evaporates loads -- every store still lands in its slot, so
``watch``/``history`` read truthful storage under all four. Any future mode
that evaporates STORES must consult the watch set here first; that is a
contract change, not a flag.

The full surface, audited 2026-08-19
------------------------------------

Beyond the three performance switches, the contract carries or names every
policy axis the pipeline already has a home for. Wired today (a field here
reaches a real consumer):

* ``compiler`` / ``compiler_flags`` — toolchain specification and flag
  passthrough, consumed by both zig cc invocations (LLVM and C shells).
* ``resolver_epsilon`` — precision bound for baked resolver tables
  (``bounded_constants.materialize_pi`` and series terms).
* ``loops`` — the loop-optimization subcontract. It centralizes the low-priority
  unroll threshold and register-block width; semantic recurrence/effect
  preservation remains a mandatory veto above those preferences.
* ``shaders`` — the shader-identity subcontract. The GEMM tiler reads it once
  and defaults to the optimized cooperative GLSL identity; the direct humble
  source lowering remains an explicit proof/profiling choice.
* ``extraction`` — the WHOLE ingestion/native-pursual policy, embedded: an
  ``ExtractionContract`` (or path to one). That object already decides
  python-call allowance (INGEST_PYTHON / PYTHON_HOST_CALL), native pursuit
  (USE_NATIVE), binary decompilation (DECOMPILE_MACHINE) and refusal
  (REJECT) per subject -- which is also where external-linking policy
  lives. ``None`` preserves the historical default: the gate is DISABLED
  (recorded hazard: an absent contract silently permits everything;
  2026-08-17 crash). ``lower_ast_source_to_ssa`` consults this when its own
  ``extraction_contract`` argument is None, so per-call still wins.

Declared with a single honored value (asking for anything else refuses,
same doctrine as ``fusion_levels`` -- these become real when their layer
is wired):

* ``deployment`` — threading/deployment policy. Only ``"serial"`` is
  honored: ``turing_pool.c`` exists but no shell routes work through it
  yet (P3).
* ``destination`` — only ``"native"``; destination-language and shell
  options (``source_language``, ``shell_language``) remain per-call
  parameters until routed through here.
* ``constant_arguments`` — the precise list of entry arguments that MAY be
  baked to constants. Nothing bakes arguments yet, so a non-empty list
  refuses rather than pretending.
* ``symbolic_arguments`` — the precise list that MUST remain symbolic
  (the fluid's runtime-extent loops are the canonical members). Recorded
  and vacuously honored today -- no pass bakes arguments -- and every
  future specializer must treat this list as a veto.

Horizon, named so it has an address when it arrives: syscall routing --
letting a native artifact's OS interactions be described and redirected
through the contract rather than linked ambiently.

Resolution order: an explicit ``set_active_contract`` wins; else the
``TURING_WORK_CONTRACT`` environment variable names a preset; else
``develop``. The two legacy variables ``TURING_POW_INEXACT`` and
``TURING_FMA_CONTRACT`` remain honored as single-field overrides on top of
the resolved preset, so every measurement recorded against them still means
what it meant.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any

_HONORED_DEPLOYMENT = ("serial", "auto")
_HONORED_COMPILER = ("zig-cc",)
_HONORED_DESTINATION = ("native",)
_HONORED_GLSL_GEMM = ("glslblas_gemm", "source_algorithm")


@dataclasses.dataclass(frozen=True)
class LoopOptimizationContract:
    """Priorities and tunable parameters for loop identities."""

    # Unrolling is a low-priority representation identity.  Semantic
    # preservation closures (carried recurrence, effects, publication) veto it.
    unroll_limit: int = 8
    # Number of adjacent unit-stride outputs held as one recurrence vector.
    register_block_width: int = 4

    def __post_init__(self) -> None:
        if int(self.unroll_limit) < 1:
            raise ValueError("loop unroll_limit must be positive")
        if int(self.register_block_width) < 1:
            raise ValueError("register_block_width must be positive")


@dataclasses.dataclass(frozen=True)
class ShaderOptimizationContract:
    """Backend-identity choices for shader compilation.

    ``glslblas_gemm`` is the performance default. ``source_algorithm`` is the
    deliberately humble lowering of the same canonical BLAS role and exists
    for proof, profiling, and driver-comparison work.
    """

    blas_gemm: str = "glslblas_gemm"

    def __post_init__(self) -> None:
        if self.blas_gemm not in _HONORED_GLSL_GEMM:
            raise ValueError(
                f"shader blas_gemm={self.blas_gemm!r} is not honored; "
                f"honored: {_HONORED_GLSL_GEMM}"
            )


@dataclasses.dataclass(frozen=True)
class WorkContract:
    """One complete answer to "how faithful, and to whom?"."""

    name: str
    # Emission keeps a same-block register for a slot's known content.
    register_reuse: bool
    # ir_identities may fire the bit-changing reductions (sqrt family).
    inexact_identities: bool
    # Multiply-add contraction: `contract` flags + host target named.
    contract_multiply_add: bool
    # --- wired policy axes (see module docstring) ---
    compiler: str = "zig-cc"
    compiler_flags: tuple[str, ...] = ()
    resolver_epsilon: float = 1.0e-12
    # ExtractionContract instance or path; None = historical no-gate default.
    extraction: Any = None
    # Repository-SSA deployment policy. ``auto`` consumes the module's
    # proved Deploy/Join frames and selects real backend executors; ``serial``
    # preserves the recorded linear schedule.
    deployment: str = "serial"
    destination: str = "native"
    constant_arguments: tuple[str, ...] = ()
    symbolic_arguments: tuple[str, ...] = ()
    # Compile-complementary loop policy.  This is a subcontract so identity
    # parameters have one authority without bloating every work preset.
    loops: LoopOptimizationContract = dataclasses.field(
        default_factory=LoopOptimizationContract,
    )
    # Shader backend identity policy. The optimized identity is the default
    # for every preset; a caller must explicitly request the source algorithm.
    shaders: ShaderOptimizationContract = dataclasses.field(
        default_factory=ShaderOptimizationContract,
    )

    def __post_init__(self) -> None:
        # Refuse, never fall back (fusion_levels doctrine): a contract
        # naming behavior no layer honors must fail at construction, not
        # quietly compile something else.
        if self.deployment not in _HONORED_DEPLOYMENT:
            raise ValueError(
                f"deployment={self.deployment!r} is not honored; "
                f"honored: {_HONORED_DEPLOYMENT}"
            )
        if self.compiler not in _HONORED_COMPILER:
            raise ValueError(
                f"compiler={self.compiler!r} is not honored yet; "
                f"honored: {_HONORED_COMPILER}"
            )
        if self.destination not in _HONORED_DESTINATION:
            raise ValueError(
                f"destination={self.destination!r} is not honored yet; "
                f"honored: {_HONORED_DESTINATION}"
            )
        if self.constant_arguments:
            raise ValueError(
                "constant_arguments is declared but nothing bakes arguments "
                "yet; an accepted list would be a silent lie"
            )
        if not float(self.resolver_epsilon) > 0.0:
            raise ValueError("resolver_epsilon must be positive")

    def describe(self) -> str:
        held = [
            f"register_reuse={'on' if self.register_reuse else 'off'}",
            f"identities={'inexact' if self.inexact_identities else 'exact-only'}",
            f"fma={'contract' if self.contract_multiply_add else 'none'}",
            f"unroll<={self.loops.unroll_limit}",
            f"register-block={self.loops.register_block_width}",
            f"glsl-gemm={self.shaders.blas_gemm}",
            f"deployment={self.deployment}",
        ]
        return f"{self.name}: " + ", ".join(held)


PRESETS: dict[str, WorkContract] = {
    "prove": WorkContract(
        "prove", register_reuse=False, inexact_identities=False,
        contract_multiply_add=False,
    ),
    "develop": WorkContract(
        "develop", register_reuse=True, inexact_identities=False,
        contract_multiply_add=False,
    ),
    "deploy": WorkContract(
        "deploy", register_reuse=True, inexact_identities=True,
        contract_multiply_add=False,
        # -O3 raises the optimizer's effort without changing semantics.
        # Deliberately NO -march=native here: deploy's documented meaning
        # is "inexact set, STABLE ACROSS HOSTS", and native tuning is the
        # opposite of host-stable.
        compiler_flags=("-O3",),
        deployment="auto",
    ),
    "fast": WorkContract(
        "fast", register_reuse=True, inexact_identities=True,
        contract_multiply_add=True,
        # -ffast-math is the toolchain spelling of what this preset
        # already licenses (inexact identities + FMA contraction);
        # -march=native arrives via the contraction switch itself in
        # compile_artifact, so it is not restated here.
        compiler_flags=("-O3", "-ffast-math"),
        deployment="auto",
    ),
}

_active: WorkContract | None = None


def set_active_contract(contract: WorkContract | str | None) -> None:
    """Pin the contract for this process; ``None`` returns to resolution."""

    global _active
    if isinstance(contract, str):
        contract = _named(contract)
    _active = contract


def _named(name: str) -> WorkContract:
    preset = PRESETS.get(str(name).strip().lower())
    if preset is None:
        # Refuse, never fall back: a caller who asked for a contract and
        # silently got another is the failure shape this module exists to
        # prevent (same doctrine as fusion_levels).
        raise ValueError(
            f"unknown work contract {name!r}; presets: {sorted(PRESETS)}"
        )
    return preset


def _flag(variable: str) -> bool | None:
    raw = os.environ.get(variable)
    if raw is None or raw == "":
        return None
    return raw not in ("0",)


def active_contract() -> WorkContract:
    """The contract in force: pinned, else named by environment, else develop.

    Legacy single-field overrides (``TURING_POW_INEXACT``,
    ``TURING_FMA_CONTRACT``) apply on top, so a measurement script that sets
    only one of them gets exactly the historical meaning.
    """

    contract = _active
    if contract is None:
        named = os.environ.get("TURING_WORK_CONTRACT")
        contract = _named(named) if named else PRESETS["develop"]

    inexact = _flag("TURING_POW_INEXACT")
    fma = _flag("TURING_FMA_CONTRACT")
    if inexact is None and fma is None:
        return contract
    return dataclasses.replace(
        contract,
        name=f"{contract.name}+overrides",
        inexact_identities=(
            contract.inexact_identities if inexact is None else inexact
        ),
        contract_multiply_add=(
            contract.contract_multiply_add if fma is None else fma
        ),
    )

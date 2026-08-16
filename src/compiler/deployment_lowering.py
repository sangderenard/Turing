"""Per-backend deployment lowering contract: strategies, legality, fallback.

The deployment layer's law is that ``Deploy``/``Join`` are *permissions*,
not commands.  Serial lowering -- run the lanes in the recorded order,
ignore the markers -- is total and correct for every backend, because lane
independence (verified by ``deployment_ssa_binding``) means any execution
order inside the frame yields the same values, and a REDUCE join's operator
is associative by ``DeploymentJoin``'s own construction rules.  Parallel
strategies are per-backend refinements of that baseline:

- ``serial``   : the recorded linear schedule (always available)
- ``pool``     : hand lanes/chunks to persistent, already-running workers
                 (browser tile workers; ``deployment_host_pool``;
                 the native ``turing_pool.c`` runtime)
- ``dispatch`` : the deploy IS the GPU grid dimension; a BARRIER join is
                 the end-of-dispatch boundary (webgpu/glsl compute)
- ``simd``     : lane-width vectorization of a REDUCE join
                 (``llvm_simd_deployment``, the first consumer)

Profiles are declared here per backend and kept deliberately honest: a
strategy appears only when a real executor exists in the repo today, with
its location named -- the same discipline as ``BackendOperatorInventory``
pinning each vocabulary to the table it derives from.  Selection never
raises for a missing capability; it degrades to ``serial`` and says why,
mirroring the shortfall pattern every SSA backend already follows.
"""

from __future__ import annotations

from dataclasses import dataclass

from .deployment_frame import DeploymentJoinMode

SERIAL = "serial"
POOL = "pool"
DISPATCH = "dispatch"
SIMD = "simd"

DEPLOYMENT_STRATEGIES = (SERIAL, POOL, DISPATCH, SIMD)


@dataclass(frozen=True)
class DeploymentLoweringProfile:
    """What one backend can actually do with a deployment frame.

    ``parallel_join_modes`` lists the join modes the backend can honor
    *concurrently*; every backend honors every mode serially.  ``executor``
    names the real code that implements the non-serial strategies, so this
    table cannot drift into aspiration.
    """

    backend: str
    strategies: tuple[str, ...]
    parallel_join_modes: tuple[str, ...] = ()
    executor: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        unknown = set(self.strategies) - set(DEPLOYMENT_STRATEGIES)
        if unknown:
            raise ValueError(
                f"unknown deployment strategies {sorted(unknown)!r}; "
                f"one of {DEPLOYMENT_STRATEGIES}"
            )
        if SERIAL not in self.strategies:
            raise ValueError(
                "every profile must include the serial strategy; it is the "
                "total fallback that makes deployment a permission"
            )
        modes = {mode.value for mode in DeploymentJoinMode}
        bad = set(self.parallel_join_modes) - modes
        if bad:
            raise ValueError(
                f"unknown join modes {sorted(bad)!r}; one of {sorted(modes)}"
            )


_PROFILES: dict[str, DeploymentLoweringProfile] = {}


def register_deployment_profile(profile: DeploymentLoweringProfile) -> None:
    _PROFILES[profile.backend] = profile


def deployment_profile(backend: str) -> DeploymentLoweringProfile:
    """Profile for ``backend``; unknown backends get an honest serial-only."""

    found = _PROFILES.get(str(backend))
    if found is not None:
        return found
    return DeploymentLoweringProfile(
        backend=str(backend),
        strategies=(SERIAL,),
        note="no declared deployment profile; serial fallback only",
    )


def deployment_profiles() -> tuple[DeploymentLoweringProfile, ...]:
    return tuple(_PROFILES.values())


# Seeded from executors that exist in the repository today.  A strategy is
# listed if and only if the named executor is real code, not a plan.
register_deployment_profile(DeploymentLoweringProfile(
    backend="python",
    strategies=(SERIAL, POOL),
    parallel_join_modes=("barrier", "indexed", "reduce"),
    executor="src/compiler/deployment_host_pool.py:HostDeploymentPool",
    note="persistent daemon threads; REDUCE folds strictly in lane order "
         "unless allow_reassociation grants tree folding",
))
register_deployment_profile(DeploymentLoweringProfile(
    backend="llvm",
    strategies=(SERIAL, POOL, SIMD),
    parallel_join_modes=("barrier", "indexed", "reduce"),
    executor=(
        "src/compiler/deployment_host_pool.py:HostDeploymentPool "
        "(artifact entry releases the GIL across ctypes calls); "
        "src/common/tensors/accelerator_backends/llvm_simd_deployment.py "
        "(REDUCE, scale 1)"
    ),
))
register_deployment_profile(DeploymentLoweringProfile(
    backend="c",
    strategies=(SERIAL, POOL),
    parallel_join_modes=("barrier",),
    executor=(
        "src/common/tensors/accelerator_backends/c_backend/turing_pool.c "
        "(persistent workers, atomic chunk claiming, barrier join)"
    ),
))
register_deployment_profile(DeploymentLoweringProfile(
    backend="fortran",
    strategies=(SERIAL,),
    note="runs under the C shell; inherits the native pool once "
         "profiled_c_shell's launch seam routes through turing_pool",
))
register_deployment_profile(DeploymentLoweringProfile(
    backend="wasm",
    strategies=(SERIAL, POOL),
    parallel_join_modes=("barrier",),
    executor="src/compiler/wasm_html_shell.py:ensureTileWorkers/executeDeploy",
    note="copy-in/copy-out tile workers; lockstep batches today, chunk "
         "claiming is the planned refinement",
))
register_deployment_profile(DeploymentLoweringProfile(
    backend="webgpu",
    strategies=(SERIAL, DISPATCH),
    parallel_join_modes=("barrier",),
    executor="src/compiler/ssa_webgpu_backend.py (deploy is the grid "
             "dimension; barrier join is the end of the dispatch)",
))
register_deployment_profile(DeploymentLoweringProfile(
    backend="glsl",
    strategies=(SERIAL, DISPATCH),
    parallel_join_modes=("barrier",),
    executor="src/common/tensors/accelerator_backends/glsl_backend.py",
))


# Execution classes (deployment_classification) -> preferred strategy order.
# Selection walks the preference and takes the first strategy the profile
# declares with the frame's join mode honored; otherwise serial, with the
# reason recorded.
_CLASS_PREFERENCES: dict[str, tuple[str, ...]] = {
    # graphics-output constrains where results LAND, not how they are
    # computed: a CPU backend still pools the numeric work and the surface
    # reads the joined outputs.
    "graphics-output": (DISPATCH, POOL, SERIAL),
    "shader-compute": (DISPATCH, POOL, SERIAL),
    "thread-workers": (POOL, SERIAL),
    "host-linear": (SERIAL,),
}


@dataclass(frozen=True)
class DeploymentStrategyChoice:
    """One region's lowering decision for one backend, with its evidence.

    ``workers`` is the measured best pool size when a calibration verdict
    contributed to the decision; ``None`` means "no measurement -- use the
    executor's own default sizing".
    """

    backend: str
    strategy: str
    join_mode: str
    execution_class: str
    reasons: tuple[str, ...]
    workers: int | None = None
    # True when a measured verdict, not a capability gap, demoted a legal
    # pool to serial -- the typed signal downstream gating keys on.
    calibration_demoted: bool = False

    @property
    def parallel(self) -> bool:
        return self.strategy != SERIAL


def select_deployment_strategy(
    *,
    backend: str,
    execution_class: str,
    join_mode: str = "barrier",
    calibration=None,
) -> DeploymentStrategyChoice:
    """Choose how ``backend`` should lower a frame of the given class.

    Never raises on a capability gap: the serial baseline is always legal,
    and the reason trail says exactly which refinement was unavailable and
    why -- so a report reads like a shortfall list, not a stack trace.

    ``calibration`` is an optional ``CalibrationVerdict``
    (deployment_calibration): measurement refines the decision *within*
    legality -- a measured-slower pool degrades to serial with the measured
    ratio recorded, and a measured winner carries its best worker count.
    Calibration never overrides a legality veto in the other direction.
    """

    profile = deployment_profile(backend)
    preferences = _CLASS_PREFERENCES.get(
        str(execution_class), (SERIAL,)
    )
    reasons: list[str] = []
    demoted_by_measurement = False
    if str(execution_class) not in _CLASS_PREFERENCES:
        reasons.append(
            f"unknown execution class {execution_class!r}; treating as "
            "host-linear"
        )
    for strategy in preferences:
        if strategy == SERIAL:
            reasons.append("serial baseline selected")
            break
        if strategy not in profile.strategies:
            reasons.append(
                f"{strategy} unavailable: backend {profile.backend!r} does "
                "not declare it"
                + (f" ({profile.note})" if profile.note else "")
            )
            continue
        if str(join_mode) not in profile.parallel_join_modes:
            reasons.append(
                f"{strategy} declared but join mode {join_mode!r} is not "
                f"in its parallel set {profile.parallel_join_modes!r}"
            )
            continue
        if (
            strategy == POOL
            and calibration is not None
            and calibration.best_strategy != "pool"
        ):
            demoted_by_measurement = True
            reasons.append(
                f"pool legal but demoted by calibration: measured "
                f"{calibration.speedup:.2f}x against serial on "
                f"{calibration.machine}"
            )
            continue
        reasons.append(
            f"{strategy} selected via {profile.executor or 'declared profile'}"
        )
        workers = None
        if strategy == POOL and calibration is not None:
            workers = int(calibration.best_workers)
            reasons.append(
                f"calibration measured {calibration.speedup:.2f}x at "
                f"{workers} worker(s)"
            )
        return DeploymentStrategyChoice(
            backend=profile.backend,
            strategy=strategy,
            join_mode=str(join_mode),
            execution_class=str(execution_class),
            reasons=tuple(reasons),
            workers=workers,
        )
    return DeploymentStrategyChoice(
        backend=profile.backend,
        strategy=SERIAL,
        join_mode=str(join_mode),
        execution_class=str(execution_class),
        reasons=tuple(reasons),
        calibration_demoted=demoted_by_measurement,
    )


@dataclass(frozen=True)
class SerialLegalizationReport:
    function: str
    removed_markers: int


def legalize_deployments_serial(function) -> SerialLegalizationReport:
    """Total serial legalization: drop frame markers from the stream.

    For a backend that wants plain straight-line code, this removes every
    ``Deploy``/``Join`` marker instruction (bound or not) while leaving the
    lane instructions -- which are already in a correct serial order --
    untouched.  Correctness follows from lane independence and join
    associativity; see the module docstring.  ``deployment_memberships``
    attributes are preserved so region provenance survives the
    legalization.
    """

    removed = 0
    for block in function.blocks.values():
        kept = []
        for instruction in block.instrs:
            attributes = instruction.attributes or {}
            if attributes.get("deployment_frame") and instruction.op in (
                "Deploy", "Join"
            ):
                removed += 1
                continue
            kept.append(instruction)
        block.instrs = kept
    return SerialLegalizationReport(
        function=function.name, removed_markers=removed,
    )


__all__ = [
    "DEPLOYMENT_STRATEGIES",
    "DISPATCH",
    "POOL",
    "SERIAL",
    "SIMD",
    "DeploymentLoweringProfile",
    "DeploymentStrategyChoice",
    "SerialLegalizationReport",
    "deployment_profile",
    "deployment_profiles",
    "legalize_deployments_serial",
    "register_deployment_profile",
    "select_deployment_strategy",
]

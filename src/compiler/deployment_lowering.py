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
class ComputeDispatchLimits:
    """Backend-supplied limits used to make one compute launch decision."""

    max_group_count: tuple[int, int, int]
    max_group_size: tuple[int, int, int]
    max_invocations: int

    def __post_init__(self) -> None:
        if len(self.max_group_count) != 3 or len(self.max_group_size) != 3:
            raise ValueError("compute limits must describe exactly x/y/z")
        if any(int(value) < 1 for value in self.max_group_count):
            raise ValueError("compute group-count limits must be positive")
        if any(int(value) < 1 for value in self.max_group_size):
            raise ValueError("compute group-size limits must be positive")
        if int(self.max_invocations) < 1:
            raise ValueError("compute invocation limit must be positive")


@dataclass(frozen=True)
class ComputeDispatchPlan:
    """A deterministic one-dimensional compute mapping into an x/y/z grid."""

    count: int
    workgroup_size: tuple[int, int, int]
    groups: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("compute dispatch count cannot be negative")
        if len(self.workgroup_size) != 3 or len(self.groups) != 3:
            raise ValueError("compute geometry must describe exactly x/y/z")
        if any(value < 1 for value in self.workgroup_size):
            raise ValueError("compute workgroup dimensions must be positive")
        if self.count == 0:
            if self.groups != (0, 0, 0):
                raise ValueError("zero-work compute geometry must skip dispatch")
        elif any(value < 1 for value in self.groups):
            raise ValueError("nonempty compute grid dimensions must be positive")
        if self.count > self.capacity:
            raise ValueError("compute geometry does not cover its logical work")

    @property
    def capacity(self) -> int:
        return (
            self.workgroup_size[0] * self.workgroup_size[1]
            * self.workgroup_size[2] * self.groups[0]
            * self.groups[1] * self.groups[2]
        )

    @property
    def skipped(self) -> bool:
        return self.count == 0

    def as_record(self) -> dict[str, object]:
        return {
            "count": self.count,
            "workgroup_size": list(self.workgroup_size),
            "groups": list(self.groups),
            "capacity": self.capacity,
        }


def plan_compute_dispatch(
    count: int,
    *,
    limits: ComputeDispatchLimits,
    preferred_local_size: int = 256,
    minimum_local_size: int = 32,
) -> ComputeDispatchPlan:
    """Map flat work to a legal compute grid shared by GLSL and WebGPU.

    The mathematical work remains a flat, stable identity space.  Only its
    deployment is folded through x, y, and z, so emitters can use the same
    linear-index formula and obtain identical coverage on either shader API.
    """

    count = int(count)
    preferred_local_size = int(preferred_local_size)
    minimum_local_size = int(minimum_local_size)
    if count < 0:
        raise ValueError("launch count cannot be negative")
    if preferred_local_size < 1:
        raise ValueError("preferred local size must be positive")
    if minimum_local_size < 1:
        raise ValueError("minimum local size must be positive")
    local_cap = min(
        preferred_local_size,
        int(limits.max_group_size[0]),
        int(limits.max_invocations),
    )
    local = 1 << (local_cap.bit_length() - 1)
    if count:
        small_target = 1 << (count - 1).bit_length()
        local = min(local, max(min(minimum_local_size, local), small_target))
    if count == 0:
        return ComputeDispatchPlan(0, (local, 1, 1), (0, 0, 0))

    needed = (count + local - 1) // local
    group_x = min(needed, int(limits.max_group_count[0]))
    remaining = (needed + group_x - 1) // group_x
    group_y = min(remaining, int(limits.max_group_count[1]))
    group_z = (remaining + group_y - 1) // group_y
    if group_z > int(limits.max_group_count[2]):
        capacity = (
            int(limits.max_group_count[0])
            * int(limits.max_group_count[1])
            * int(limits.max_group_count[2])
            * local
        )
        raise ValueError(
            f"launch count {count} exceeds one-dispatch capacity {capacity}"
        )
    return ComputeDispatchPlan(
        count,
        (local, 1, 1),
        (int(group_x), int(group_y), int(group_z)),
    )


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
    note="desktop OpenGL compute; retained as the historical target name",
))
register_deployment_profile(DeploymentLoweringProfile(
    backend="native_glsl",
    strategies=(SERIAL, DISPATCH),
    parallel_join_modes=("barrier",),
    executor=(
        "src/common/tensors/accelerator_backends/glsl_backend.py; "
        "src/compiler/glsl_blas_deployment.py native SDL/OpenGL shell"
    ),
    note=(
        "explicit desktop-native GLSL dispatcher option; shares GLSL "
        "operator coverage but is distinct from browser shader targets"
    ),
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

    ``chunk`` is the strategic tiling geometry for the pool executors --
    how many elements one worker claims per grab (turing_pool.c's atomic
    chunk claiming, the wasm tile workers' batches). ``None`` means "no
    evidence -- the executor's own default chunking", which is exactly
    today's behavior. A chunk is CHOSEN, at build time, from the same
    evidence chain as everything else here: the measured verdict's best
    worker count, the task's work extent, and the cores stated for the
    deploy target -- never probed at runtime, because no mechanism exists
    to supply a runtime choice and none is being invented.
    """

    backend: str
    strategy: str
    join_mode: str
    execution_class: str
    reasons: tuple[str, ...]
    workers: int | None = None
    chunk: int | None = None
    compute: ComputeDispatchPlan | None = None
    # True when a measured verdict, not a capability gap, demoted a legal
    # pool to serial -- the typed signal downstream gating keys on.
    calibration_demoted: bool = False

    @property
    def parallel(self) -> bool:
        return self.strategy != SERIAL

    def as_record(self) -> dict[str, object]:
        """JSON-shaped decision evidence for manifests and backend APIs."""

        return {
            "strategy": self.strategy,
            "join_mode": self.join_mode,
            "execution_class": self.execution_class,
            "workers": self.workers,
            "chunk": self.chunk,
            "compute": (
                None if self.compute is None else self.compute.as_record()
            ),
            "calibration_demoted": self.calibration_demoted,
            "reasons": list(self.reasons),
        }


# One worker claim should not be the whole lane range: pools balance load
# by letting fast workers claim more chunks, which needs several claims
# per worker to exist. Four is turing_pool.c's own working ratio.
_CHUNK_CLAIMS_PER_WORKER = 4


def _strategic_chunk(
    work: int | None, workers: int | None, reasons: list[str],
) -> int | None:
    """Chunk size from evidence, or ``None`` (executor default) without it."""

    if work is None or not workers:
        return None
    work = int(work)
    if work <= 0:
        return None
    chunk = max(1, work // (int(workers) * _CHUNK_CLAIMS_PER_WORKER))
    reasons.append(
        f"chunk {chunk} chosen: {work} work over {workers} worker(s) at "
        f"{_CHUNK_CLAIMS_PER_WORKER} claims each for load balance"
    )
    return chunk


def select_deployment_strategy(
    *,
    backend: str,
    execution_class: str,
    join_mode: str = "barrier",
    calibration=None,
    work: int | None = None,
    cores: int | None = None,
    nesting_depth: int = 0,
    compute_limits: ComputeDispatchLimits | None = None,
    preferred_local_size: int = 256,
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

    The strategic-tiling evidence, every piece optional and inert when
    absent (absence of evidence changes nothing -- the deployment stage's
    own law):

    * ``work`` -- the task at hand: the frame's work extent (elements /
      steps its lanes cover). With a worker count it yields the CHUNK the
      pool executors claim by -- the tiling geometry, chosen here at build
      time and recorded on the choice.
    * ``cores`` -- cores stated for the deploy target. Used as the worker
      count only when no calibration verdict measured one (measurement
      outranks a core count).
    * ``nesting_depth`` -- how many enclosing parallel deployments contain
      this frame. Nested recognition is TEMPERED, not forbidden: the
      worker budget divides by ``1 + depth`` so an inner pool cannot
      multiply against the pools above it; a budget tempered to one worker
      demotes to serial with the tempering recorded (running a one-worker
      pool inside another pool's lane is pure overhead).
    """

    profile = deployment_profile(backend)
    preferences = _CLASS_PREFERENCES.get(
        str(execution_class), (SERIAL,)
    )
    reasons: list[str] = []
    demoted_by_measurement = False
    nesting_depth = max(0, int(nesting_depth))
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
        workers = None
        if strategy == POOL:
            if calibration is not None:
                workers = int(calibration.best_workers)
                reasons.append(
                    f"calibration measured {calibration.speedup:.2f}x at "
                    f"{workers} worker(s)"
                )
            elif cores:
                stated_cores = max(1, int(cores))
                if profile.backend in {"python", "llvm", "c"}:
                    # These CPU pools enlist the caller in frame draining.
                    # ``workers`` is parked background threads, not total
                    # active execution slots. Browser workers do not share
                    # this execution model.
                    workers = max(0, stated_cores - 1)
                    reasons.append(
                        f"no calibration verdict; {workers} background "
                        f"worker(s) from {stated_cores} stated core(s), with "
                        "the caller as one execution slot"
                    )
                else:
                    workers = stated_cores
                    reasons.append(
                        f"no calibration verdict; {workers} worker(s) from "
                        f"{stated_cores} stated core(s)"
                    )
            if workers is not None and workers < 1:
                reasons.append(
                    "no background worker remains after caller participation; "
                    "serial for this frame"
                )
                continue
            if workers is not None and nesting_depth:
                tempered = max(1, workers // (1 + nesting_depth))
                reasons.append(
                    f"nested inside {nesting_depth} parallel "
                    f"deployment(s): worker budget tempered "
                    f"{workers} -> {tempered}"
                )
                workers = tempered
                if workers <= 1:
                    reasons.append(
                        "tempered budget is one worker; a one-worker pool "
                        "inside another pool's lane is pure overhead -- "
                        "serial for this frame"
                    )
                    continue
        compute = None
        if strategy == DISPATCH and work is not None:
            if compute_limits is None:
                reasons.append(
                    "dispatch geometry deferred: backend limits were not "
                    "supplied"
                )
            else:
                compute = plan_compute_dispatch(
                    work,
                    limits=compute_limits,
                    preferred_local_size=preferred_local_size,
                )
                reasons.append(
                    "compute geometry chosen: workgroup "
                    f"{compute.workgroup_size}, grid {compute.groups}, "
                    f"covering {compute.count} item(s)"
                )
        reasons.append(
            f"{strategy} selected via {profile.executor or 'declared profile'}"
        )
        chunk = (
            _strategic_chunk(work, workers, reasons)
            if strategy == POOL else None
        )
        return DeploymentStrategyChoice(
            backend=profile.backend,
            strategy=strategy,
            join_mode=str(join_mode),
            execution_class=str(execution_class),
            reasons=tuple(reasons),
            workers=workers,
            chunk=chunk,
            compute=compute,
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
    "ComputeDispatchLimits",
    "ComputeDispatchPlan",
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
    "plan_compute_dispatch",
    "register_deployment_profile",
    "select_deployment_strategy",
]

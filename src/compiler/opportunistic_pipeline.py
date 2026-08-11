"""An opportunistic, staged compiler harness.

The compiler is a chain of stages (source -> process graph -> ... -> dual IR ->
SSA -> target source -> executable). This harness drives that chain WITHOUT
caring how any individual stage is implemented: each stage-span is provided by a
*provider*, and the harness picks, for the next thing it needs, the best
provider available -- preferring a compiled (DLL) span over Python, and, within
a tier, the largest fused span. Python is the fallback wherever nothing better
exists yet. As we compile stages into native whole programs (DLLs) that export
their entry, we register them as providers and the harness starts using them
automatically -- "the best version of itself it can get".

Two axes of native acceleration, both first-class here:

* **Vertical stage-span providers** -- a compiled slice of the pipeline
  (e.g. a DLL that goes dual_ir -> ssa). Registered with ``register_span``.
* **Foundational library providers** -- a compiled DLL of the substrate every
  stage links against (``ProcessGraph`` ops, ``AbstractTensor`` ops). These do
  not span pipeline kinds; they make the *existing* Python stages faster from
  underneath. Registered with ``register_foundation`` and consulted by the
  planner as a global tier hint. Compiling ProcessGraph / AbstractTensor first
  is often the higher-leverage move, since it accelerates all eight stages at
  once with no inter-stage ABI seam -- the harness records both so the choice
  can be measured, not guessed.

Nothing here fuses or bakes; it only *routes* between provider implementations
and checkpoints the artifact at every seam so a rerun resumes.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


# --- the kind schema ----------------------------------------------------------
#: A pipeline's schema is just an ordered tuple of wire "kinds": a provider
#: consumes one and produces a later one, and the distance between them is its
#: span. The core engine is schema-agnostic -- pass any project's kind sequence
#: to ``Pipeline(kinds=...)`` and it is a general staged-build rig. The compiler
#: is one instantiation of it (``COMPILER_KINDS`` below).
#:
#: This is the compiler's full segment map -- most seams have no Python provider
#: wired yet; they are the iteration targets (register a provider consuming/
#: producing that kind to light the segment up).
COMPILER_KINDS: tuple[str, ...] = (
    "source",           # utf-8 Python text (+ entrypoint/params in the context)
    "process_graph",    # structural AST ProcessGraph (build_from_ast)
    "annotated_graph",  # + topology reduction, map_ir, class nav, dep regions
    "deployment_plan",  # dispatch subgraphs / whole-graph coverage
    "dual_ir",          # region FusedPrograms + ControlProgram + map_ir (shell)
    "folded_dual_ir",   # + central IR folds + consistency repair
    "ssa_module",       # whole-program SSA (lower_precompile_and_control_to_ssa)
    "target_source",    # emitted backend source (Fortran / WASM / C / GLSL)
    "executable",       # built native artifact / bundle
)
#: Back-compat / default schema alias.
KIND_ORDER = COMPILER_KINDS

#: Provider tiers, lowest is preferred (fastest / most compiled).
TIER_NATIVE = 0   # a compiled DLL span
TIER_PYTHON = 1   # the Python fallback


class PipelineError(RuntimeError):
    pass


# --- artifacts flowing between stages -----------------------------------------
@dataclass
class Artifact:
    """One value on the wire between stages, tagged with its kind. The kind is a
    bare string validated against the owning pipeline's schema when it is built,
    not globally -- so an artifact is not tied to any one project's kinds."""

    kind: str
    value: Any
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildContext:
    """Everything a provider might need beyond the input artifact: the program's
    entrypoint and parameter contract, where to build, and the checkpoint sink.
    Providers read what they need and ignore the rest."""

    entrypoint: str = ""
    feeds: Mapping[str, Any] = field(default_factory=dict)
    mutable_parameters: tuple[str, ...] = ()
    directory: Optional[Path] = None
    name: Optional[str] = None
    checkpoint: Any = False           # threaded into the AOT checkpoint system
    progress: Optional[Callable[[str], None]] = None

    def report(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)


# --- providers ----------------------------------------------------------------
@dataclass
class Provider:
    """One implementation of a stage-span: it consumes one kind and produces a
    later one. ``run`` does the work. ``tier`` orders preference (native first)
    and ``cache`` says whether the harness may checkpoint the produced artifact
    (heavy, unpicklable outputs opt out and rely on the stage's own
    checkpointing instead)."""

    name: str
    consumes: str
    produces: str
    run: Callable[[Artifact, BuildContext], Artifact]
    tier: int = TIER_PYTHON
    cache: bool = False


@dataclass
class Foundation:
    """A compiled substrate library (e.g. ProcessGraph ops, AbstractTensor ops)
    that accelerates the Python stages from underneath rather than spanning
    pipeline kinds. Recorded so the planner can note that the Python tier is, in
    practice, native-accelerated -- and so we can measure whether compiling the
    substrate first beats compiling vertical spans."""

    name: str
    library_path: Path
    accelerates: tuple[str, ...] = ()   # e.g. ("ProcessGraph", "AbstractTensor")
    loaded: bool = False


# --- the checkpoint store -----------------------------------------------------
def _repo_cache_root() -> Path:
    try:
        from ..common.tensors.accelerator_backends.shell_archive import (
            shell_archive_root,
        )

        return shell_archive_root().parent
    except Exception:
        return Path(".turing-cache")


class ArtifactCheckpointStore:
    """Content-addressed checkpoints of stage outputs. Retains the pipeline's
    intermediate artifacts so a rerun resumes at the furthest completed seam --
    the harness-level complement to the AOT capture checkpoint (which the source
    providers still thread through via ``BuildContext.checkpoint``)."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else (
            _repo_cache_root() / "pipeline-artifacts"
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, produces: str, provider: str, input_digest: str) -> str:
        raw = f"{produces}\x00{provider}\x00{input_digest}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:32]

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.pkl"

    def load(self, key: str) -> Optional[Artifact]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            from joblib.externals import cloudpickle

            with path.open("rb") as stream:
                return cloudpickle.load(stream)
        except Exception:
            return None

    def save(self, key: str, artifact: Artifact) -> Optional[Path]:
        try:
            from joblib.externals import cloudpickle

            path = self._path(key)
            with path.open("wb") as stream:
                cloudpickle.dump(artifact, stream, protocol=5)
            return path
        except Exception:
            return None


def _digest_input(artifact: Artifact, context: BuildContext) -> str:
    """A stable digest of what a provider will consume: the input identity plus
    the parts of the context that change the result. Source text digests by
    content; a heavier artifact digests by an explicit ``meta['digest']`` if it
    set one, else falls back to its own id (uncacheable across processes)."""

    hasher = hashlib.sha256()
    hasher.update(artifact.kind.encode("utf-8"))
    if artifact.kind == "source" and isinstance(artifact.value, str):
        hasher.update(artifact.value.encode("utf-8"))
    elif "digest" in artifact.meta:
        hasher.update(str(artifact.meta["digest"]).encode("utf-8"))
    else:
        hasher.update(repr(id(artifact.value)).encode("utf-8"))
    for part in (
        context.entrypoint,
        ",".join(sorted(context.mutable_parameters)),
        ",".join(sorted(context.feeds)),
    ):
        hasher.update(b"\x00")
        hasher.update(part.encode("utf-8"))
    return hasher.hexdigest()[:32]


# --- the pipeline -------------------------------------------------------------
@dataclass
class PlanStep:
    provider: Provider


class Pipeline:
    """Registry of providers plus the planner and runner. Build a target kind
    from a source artifact; the planner chooses providers preferring native
    over Python and, within a tier, the largest fused span."""

    def __init__(
        self,
        kinds: Sequence[str] = COMPILER_KINDS,
        checkpoints: Optional[ArtifactCheckpointStore] = None,
    ):
        self.kinds: tuple[str, ...] = tuple(kinds)
        self._index = {kind: i for i, kind in enumerate(self.kinds)}
        if len(self._index) != len(self.kinds):
            raise PipelineError("pipeline kinds must be unique")
        self._providers: list[Provider] = []
        self._foundations: list[Foundation] = []
        self.checkpoints = checkpoints or ArtifactCheckpointStore()

    def _idx(self, kind: str) -> int:
        try:
            return self._index[kind]
        except KeyError:
            raise PipelineError(
                f"kind {kind!r} is not in this pipeline's schema {self.kinds}"
            )

    def span(self, provider: Provider) -> int:
        return self._idx(provider.produces) - self._idx(provider.consumes)

    # -- registration --
    def register(self, provider: Provider) -> Provider:
        start, end = self._idx(provider.consumes), self._idx(provider.produces)
        if end <= start:
            raise PipelineError(
                f"provider {provider.name!r} must produce a later kind than it "
                f"consumes ({provider.consumes} -> {provider.produces})"
            )
        self._providers.append(provider)
        return provider

    def register_span(
        self,
        name: str,
        consumes: str,
        produces: str,
        run: Callable[[Artifact, BuildContext], Artifact],
        *,
        tier: int = TIER_PYTHON,
        cache: bool = False,
    ) -> Provider:
        return self.register(
            Provider(name, consumes, produces, run, tier=tier, cache=cache)
        )

    def register_dll_span(
        self,
        name: str,
        consumes: str,
        produces: str,
        *,
        library_path: str | Path,
        export: str,
        encode: Callable[[Any], bytes],
        decode: Callable[[bytes], Any],
        cache: bool = False,
    ) -> Provider:
        """Register a compiled stage-span exported from a DLL. ``encode``/
        ``decode`` are the ABI codec for this seam (the produced/consumed wire
        format); when a seam has no codec yet, invoking the provider raises so
        the missing codec is explicit rather than silent."""

        run = _make_dll_runner(name, library_path, export, encode, decode, produces)
        return self.register(
            Provider(name, consumes, produces, run, tier=TIER_NATIVE, cache=cache)
        )

    def register_foundation(
        self,
        name: str,
        library_path: str | Path,
        accelerates: Sequence[str] = (),
    ) -> Foundation:
        foundation = Foundation(name, Path(library_path), tuple(accelerates))
        self._foundations.append(foundation)
        return foundation

    @property
    def foundations(self) -> tuple[Foundation, ...]:
        return tuple(self._foundations)

    # -- planning --
    def plan(self, source_kind: str, target_kind: str) -> list[Provider]:
        """Choose the best provider path source_kind -> target_kind. Optimal by
        (fewest Python hops, then fewest hops overall, then largest spans), i.e.
        prefer the most-compiled coverage and, within that, the biggest fused
        spans. A DAG DP over KIND_ORDER, so it never dead-ends on a locally
        greedy pick."""

        start, target = self._idx(source_kind), self._idx(target_kind)
        if target < start:
            raise PipelineError(
                f"cannot go backwards: {source_kind} -> {target_kind}"
            )
        # cost tuple: (sum_tier, hops, -sum_span); lower is better.
        best: dict[int, tuple] = {start: (0, 0, 0)}
        prev: dict[int, tuple[int, Provider]] = {}
        for index in range(start, target + 1):
            if index not in best:
                continue
            base = best[index]
            for provider in self._providers:
                p_start, p_end = self._idx(provider.consumes), self._idx(provider.produces)
                if p_start != index or p_end > target:
                    continue
                candidate = (
                    base[0] + provider.tier,
                    base[1] + 1,
                    base[2] - (p_end - p_start),
                )
                if p_end not in best or candidate < best[p_end]:
                    best[p_end] = candidate
                    prev[p_end] = (index, provider)
        if target not in best:
            reachable = ", ".join(sorted(self.kinds[i] for i in best))
            raise PipelineError(
                f"no provider path from {source_kind} to {target_kind}; "
                f"reachable so far: {reachable}. Register a provider that "
                f"produces {target_kind} (or an intermediate kind)."
            )
        chain: list[Provider] = []
        cursor = target
        while cursor != start:
            from_index, provider = prev[cursor]
            chain.append(provider)
            cursor = from_index
        chain.reverse()
        return chain

    # -- running --
    def build(
        self,
        target_kind: str,
        source: Artifact,
        context: BuildContext,
        *,
        use_checkpoints: bool = True,
    ) -> Artifact:
        chain = self.plan(source.kind, target_kind)
        context.report(
            "plan: "
            + " -> ".join(
                [source.kind]
                + [f"[{p.name}]{p.produces}" for p in chain]
            )
        )
        artifact = source
        for provider in chain:
            input_digest = _digest_input(artifact, context)
            key = self.checkpoints.key(provider.produces, provider.name, input_digest)
            if use_checkpoints and provider.cache:
                cached = self.checkpoints.load(key)
                if cached is not None:
                    context.report(
                        f"resume: {provider.produces} from checkpoint "
                        f"({provider.name})"
                    )
                    artifact = cached
                    continue
            context.report(f"run: {provider.name} ({provider.consumes} -> {provider.produces})")
            started = time.time()
            artifact = provider.run(artifact, context)
            if artifact.kind != provider.produces:
                raise PipelineError(
                    f"provider {provider.name!r} promised {provider.produces} "
                    f"but produced {artifact.kind}"
                )
            artifact.meta.setdefault("elapsed_s", round(time.time() - started, 3))
            if use_checkpoints and provider.cache:
                saved = self.checkpoints.save(key, artifact)
                if saved is not None:
                    context.report(f"checkpoint: {provider.produces} -> {saved.name}")
        return artifact


def _make_dll_runner(
    name: str,
    library_path: str | Path,
    export: str,
    encode: Callable[[Any], bytes],
    decode: Callable[[bytes], Any],
    produces: str,
) -> Callable[[Artifact, BuildContext], Artifact]:
    def _run(artifact: Artifact, context: BuildContext) -> Artifact:
        if encode is None or decode is None:
            raise PipelineError(
                f"DLL provider {name!r} has no ABI codec for its seam yet; "
                "supply encode/decode for this kind before using it"
            )
        import ctypes

        payload = encode(artifact.value)
        library = ctypes.CDLL(str(library_path))
        entry = getattr(library, export)
        entry.restype = ctypes.c_void_p
        # Minimal bytes-in/bytes-out ABI: (ptr, len) -> (ptr); length protocol
        # to be finalized with the first real DLL. Kept explicit so the seam is
        # visible rather than assumed.
        raise PipelineError(
            f"DLL provider {name!r} loaded {library_path} but the bytes ABI "
            "for this seam is not finalized; wire it with the first compiled "
            "stage"
        )

    return _run


# --- the Python providers that work today -------------------------------------
def _run_source_to_dual_ir(artifact: Artifact, context: BuildContext) -> Artifact:
    """source -> dual_ir via the whole-program no-bake AOT capture. Parameters
    stay symbolic (mutable_parameters), and the AOT capture checkpoint is
    threaded through, so the pipeline retains that checkpoint layer too."""

    from ..common.tensors.accelerator_backends.aot_compile import compile_ast_aot
    from ..common.tensors.accelerator_backends.dual_ir_shell import (
        compose_dual_ir_shell,
    )

    compilation = compile_ast_aot(
        artifact.value,
        context.entrypoint,
        dict(context.feeds),
        precompile_only=True,
        bake_mode="whole_program",
        mutable_parameters=tuple(context.mutable_parameters),
        checkpoint=context.checkpoint,
        progress=context.progress,
    )
    shell = compose_dual_ir_shell(compilation)
    return Artifact(
        "dual_ir",
        {"compilation": compilation, "shell": shell},
        meta={"digest": getattr(shell, "name", context.entrypoint)},
    )


def _run_ir_folds(artifact: Artifact, context: BuildContext) -> Artifact:
    """dual_ir -> folded_dual_ir: apply the central, backend-neutral IR folds
    (byte/string idioms, split, interning) to every region program and run the
    region-consistency scan. These are general compilation concerns that today
    only run inside site_bundle; lifting them here makes every target (Fortran
    included) get lowerable, integrity-checked region programs."""

    from .ir_byte_idioms import fold_byte_string_idioms
    from .ir_string_interning import fold_string_split, intern_string_constants
    from .string_table import StringTable

    data = dict(artifact.value)
    compilation = data["compilation"]
    table = StringTable()

    region_programs = dict(getattr(compilation, "region_programs", {}) or {})
    folded: dict[int, Any] = {}
    dangling = 0
    for index, captured in region_programs.items():
        program = getattr(captured, "program", captured)
        program = intern_string_constants(
            fold_string_split(fold_byte_string_idioms(program)), table
        )
        produced = set(program.feeds) | {s.result_id for s in program.steps}
        for step in program.steps:
            if any(v not in produced for v in step.input_ids):
                dangling += 1
                break
        folded[index] = replace(captured, program=program) if hasattr(
            captured, "program"
        ) else program
    context.report(
        f"folds: {len(folded)} region program(s); "
        f"{dangling} still with dangling operands"
    )
    data["folded_region_programs"] = folded
    data["string_table"] = table
    return Artifact("folded_dual_ir", data, meta=dict(artifact.meta))


def _run_source_to_executable(artifact: Artifact, context: BuildContext) -> Artifact:
    """source -> executable via the Fortran C-shell. The current biggest working
    Python span; the harness prefers it for an executable target until finer or
    native providers cover the same reach."""

    from .fortran_c_shell import compile_ast_fortran_c_shell

    if context.directory is None:
        raise PipelineError("building an executable needs context.directory")
    executable = compile_ast_fortran_c_shell(
        artifact.value,
        context.entrypoint,
        dict(context.feeds),
        context.directory,
        name=context.name or context.entrypoint,
        mutable_parameters=tuple(context.mutable_parameters),
        checkpoint=context.checkpoint,
        progress=context.progress,
    )
    return Artifact("executable", executable)


def compiler_pipeline() -> Pipeline:
    """The compiler as one instantiation of the general rig: a pipeline over
    ``COMPILER_KINDS`` with the Python providers that work today. Native DLL
    spans (``register_dll_span``) and substrate foundations
    (``register_foundation``) are added as they are compiled; the planner will
    prefer them automatically.

    A different large project scaffolds its own pipeline the same way:
    ``Pipeline(kinds=(...its stages...))`` plus its own providers -- the engine
    (planning, checkpointing, native/DLL preference) is unchanged."""

    pipeline = Pipeline(COMPILER_KINDS)
    pipeline.register_span(
        "python:aot-capture", "source", "dual_ir",
        _run_source_to_dual_ir, tier=TIER_PYTHON, cache=True,
    )
    pipeline.register_span(
        "python:ir-folds", "dual_ir", "folded_dual_ir",
        _run_ir_folds, tier=TIER_PYTHON, cache=False,
    )
    pipeline.register_span(
        "python:fortran-shell", "source", "executable",
        _run_source_to_executable, tier=TIER_PYTHON, cache=False,
    )
    return pipeline


#: Back-compat alias -- the compiler pipeline was the first (and default) rig.
def default_pipeline() -> Pipeline:
    return compiler_pipeline()


__all__ = [
    "COMPILER_KINDS",
    "KIND_ORDER",
    "TIER_NATIVE",
    "TIER_PYTHON",
    "Artifact",
    "BuildContext",
    "Provider",
    "Foundation",
    "Pipeline",
    "ArtifactCheckpointStore",
    "PipelineError",
    "compiler_pipeline",
    "default_pipeline",
]

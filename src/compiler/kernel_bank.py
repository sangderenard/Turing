"""A bank of compiled kernel variants, and a coordinator that routes calls.

The problem this solves (2026-08-20): ``src/common/tensors/blas.py`` proved
that plainly-authored kernels compile and run correctly, and
``docs/BLAS_VS_NUMPY_PROFILE.md`` measured that WHICH artifact should serve
a call depends on the call's details (size regime, contract, whether a
specialized build exists). That decision must not live in user code. This
module automates the whole arc:

* **compile variants** of a kernel across a parameter matrix
  (work contract x optional size specialization), on demand or ahead of
  time;
* **contain the filesystem impact**: every artifact lives under one bank
  root, one directory per variant, each with a ``manifest.json`` that makes
  the directory self-describing (schema ``turing.kernel-bank.v1``);
* **verify on admission**: an artifact enters the bank only after its
  output matches the kernel's own Python reference on seeded probe inputs.
  A variant that fails verification is recorded as refused, with the
  disagreement, and is never routed to. This is the guard that makes
  size-specialization safe to even attempt while the known literal-bound
  dead-store defect (``docs/FUNCTION_TO_DEPLOYMENT_HANDOFF.md`` section
  4.2) is open: a wrong specialized build is caught at admission, not in
  a user's numbers.
* **select the variant matched to a call**: ``KernelBank.select(name,
  sizes=..., contract=...)`` answers "which admitted artifact serves this
  task" -- exact-size specialized if one is admitted, else parametric,
  compiling on miss when allowed -- and returns it ready to run.
* **coordinate launches**: ``LaunchCoordinator.launch(name, **arguments)``
  is the per-call path that takes this out of user hands entirely --
  derive the sizes from the call, select (optionally triggering a
  verified specialized build for next time), execute, fall back to the
  Python reference if nothing compiled serves, log the decision to
  ``routing_log.jsonl``.

Boundary, stated precisely: this coordinates LAUNCHES, one call at a
time -- it does not BATCH. There is no queueing, no grouping of calls, no
deferred scheduling anywhere here; each launch resolves and executes
immediately. Cross-call scheduling and deployment placement remain the
compiler's own deployment machinery (the ``deployment_classification`` /
Deploy-Join / ``turing_pool`` arc).

Contract tie-ins, deliberate:

* Variants are keyed and compiled UNDER a named work contract
  (``src/compiler/work_contract.py``); the contract name is part of the
  variant identity.
* Size specialization is this tree's first argument-baking specializer,
  which makes it the promised first tester of the contract's
  ``symbolic_arguments`` veto list ("every future specializer must treat
  this list as a veto" -- HANDOFF_WORK_CONTRACT_AND_EMISSION_REDUCTION).
  A request to bake a parameter named in the active contract's veto list
  refuses loudly. The contract's ``constant_arguments`` axis itself still
  refuses non-empty lists; when that axis is wired, this specializer
  should migrate onto it (see docs/KERNEL_BANK_DESIGN.md).

Coordination with the progressive-region-replacement effort (not yet
started, being built by another agent): the bank IS the artifact store
that effort needs. A "region" registers exactly like a BLAS kernel does --
a ``KernelSpec`` with authored source, an entrypoint name, a Python
reference, and an input generator -- and inherits compilation, contained
storage, verified admission, and routing for free. The manifest schema and
the admission rule are the interface; see docs/KERNEL_BANK_DESIGN.md.

Binding rules (each one paid for, see tools/compile_blas_probe.py and
docs/FUNCTION_TO_DEPLOYMENT_HANDOFF.md section 2.1): SSA value ids are
derived fresh from the loaded module's ``parameter_names``/``named_outputs``
metadata at materialization time, NEVER stored numbers (ids are unstable
across lowerings); a positional zip against ``fn.args`` is measurably wrong.
"""
from __future__ import annotations

import ast
import hashlib
import json
import pickle
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

MANIFEST_SCHEMA = "turing.kernel-bank.v2"  # v2: + profile, data_layout


class BankRefusal(RuntimeError):
    """A variant request the bank refuses to serve, with the reason."""


@dataclass(frozen=True)
class KernelSpec:
    """One compilable kernel: authored source, oracle, and input shape."""

    name: str
    source: str
    function_name: str
    reference: Callable[..., Any]
    parameter_order: tuple[str, ...]
    #: The parameters that carry problem size (loop bounds / extents).
    size_parameters: tuple[str, ...]
    #: (sizes, rng) -> {parameter: value} covering parameter_order.
    example_inputs: Callable[[Mapping[str, int], np.random.Generator], dict]
    #: HOW A DEPLOYMENT MATRIX KNOWS PER-ITEM DATA SIZE. Each array
    #: parameter's element count as a product of size-parameter names --
    #: ``{"A": ("m", "k"), "x": ("n",), "alpha": ()}`` -- declared by the
    #: kernel author because the flat-buffer convention erases it: a
    #: compiled variant's ABI sees only 1-D spans, and a custom loop's
    #: stride arithmetic (``a[i * k + p]``) is opaque to any partitioner.
    #: From this declaration, ``item_data(axis, sizes)`` derives what one
    #: index of a loop axis consumes of every parameter -- which is the
    #: fact a runtime deployment matrix needs to hand each lane/chunk its
    #: byte ranges. ``None`` means undeclared: partitioning machinery must
    #: refuse rather than guess.
    extents: Mapping[str, tuple[str, ...]] | None = None

    def item_data(
        self, axis: str, sizes: Mapping[str, int],
    ) -> dict[str, dict[str, int]]:
        """Elements each parameter contributes per item of ``axis``.

        Returns ``{"split": {parameter: elements_per_item},
        "shared": {parameter: total_elements}}``: a parameter whose extent
        includes ``axis`` is SPLIT across items (one ``axis`` index owns
        the product of its remaining dimensions); one whose extent does
        not include ``axis`` is SHARED whole by every item. Size
        parameters and other scalars are omitted -- they are broadcast by
        value, not partitioned. Raises ``BankRefusal`` when extents were
        never declared, because a partitioner guessing byte ranges is how
        buffers get torn.
        """

        if self.extents is None:
            raise BankRefusal(
                f"{self.name}: no extents declared; per-item data size "
                "cannot be derived and must not be guessed"
            )
        if axis not in self.size_parameters:
            raise BankRefusal(
                f"{self.name}: {axis!r} is not a size parameter "
                f"{self.size_parameters!r}"
            )
        split: dict[str, int] = {}
        shared: dict[str, int] = {}
        for parameter, dims in self.extents.items():
            if not dims:
                continue  # scalar: broadcast by value
            count = 1
            for dim in dims:
                count *= int(sizes[dim]) if isinstance(dim, str) else int(dim)
            if axis in dims:
                per_item = count // int(sizes[axis])
                split[parameter] = per_item
            else:
                shared[parameter] = count
        return {"split": split, "shared": shared}


@dataclass
class CompiledVariant:
    """A live, admitted variant: everything needed to run it."""

    spec: KernelSpec
    key: str
    directory: Path
    contract: str | None
    specialized: dict[str, int]
    module: Any
    outputs: Any
    native: Any
    id_by_name: dict[str, int]
    output_names: set[str]
    ret_ids: tuple[int, ...]
    _executions: dict = field(default_factory=dict)

    def run(self, arguments: Mapping[str, Any]):
        from src.compiler.ssa_llvm_backend import prepare_artifact_execution

        live = {
            parameter: value for parameter, value in arguments.items()
            if parameter not in self.specialized
        }
        signature = tuple(
            (parameter, tuple(np.shape(live[parameter])))
            for parameter in sorted(live)
        )
        execution = self._executions.get(signature)
        # The creation path COPIES every feed. prepare_artifact_execution
        # binds arrays by reference, so without the copy a caller's array
        # (or worse, a VIEW into a larger array -- what a tiling composer
        # passes) becomes the cached execution's own buffer: the next
        # same-signature call then writes its inputs THROUGH that view into
        # the caller's memory, silently corrupting whatever lived there.
        # The reuse path below already copies by construction.
        feeds = {
            self.id_by_name[p]: np.array(live[p], dtype=float, copy=True)
            for p in live
        }
        if execution is None:
            execution = prepare_artifact_execution(self.native, feeds)
            self._executions[signature] = execution
        else:
            for parameter, value in live.items():
                buffer = execution.buffers[self.id_by_name[parameter]]
                array = np.asarray(value)
                if array.ndim:
                    np.asarray(buffer).reshape(array.shape)[...] = array
                else:
                    buffer[...] = array
        execution.run()
        inout = next(
            (p for p in self.output_names if p in self.id_by_name), None,
        )
        if inout is not None:
            return np.array(execution.buffers[self.id_by_name[inout]])
        if not self.ret_ids:
            raise BankRefusal(
                f"{self.spec.name}: no named output and no Ret record"
            )
        value = execution.buffers[self.ret_ids[0]]
        return float(np.asarray(value).reshape(-1)[0])


def _specialize_source(spec: KernelSpec, sizes: Mapping[str, int]) -> str:
    """Bake size parameters as constants: drop from the signature, assign
    in a prologue. The admission check decides whether the result is
    actually correct (section 4.2's dead-store defect makes that a live
    question); the bank never trusts a specialized build unverified."""

    tree = ast.parse(spec.source)
    target = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == spec.function_name
    )
    baked = {name: int(value) for name, value in sizes.items()}
    target.args.args = [
        argument for argument in target.args.args
        if argument.arg not in baked
    ]
    prologue = [
        ast.parse(f"{name} = {value}").body[0]
        for name, value in baked.items()
    ]
    target.body = prologue + target.body
    return ast.unparse(tree) + "\n"


class KernelBank:
    """Compile, verify, store, and serve kernel variants under one root."""

    def __init__(self, root: str | Path, specs: Mapping[str, KernelSpec]):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.specs = dict(specs)
        self._live: dict[str, CompiledVariant] = {}
        self._fingerprint = self._compiler_fingerprint()

    # -- identity ---------------------------------------------------------
    @staticmethod
    def _compiler_fingerprint() -> str:
        """Newest mtime over the compiler package -- the same authority
        ``symbolic_fluid_native_runtime._cache_is_stale`` uses. An artifact
        built by an older compiler is stale, not trusted."""

        compiler_dir = Path(__file__).resolve().parent
        newest = 0.0
        for source in compiler_dir.rglob("*.py"):
            if "__pycache__" in source.parts:
                continue
            try:
                newest = max(newest, source.stat().st_mtime)
            except OSError:
                continue
        return f"{newest:.0f}"

    def variant_key(self, name: str, *, contract: str | None,
                    specialized: Mapping[str, int] | None) -> str:
        spec = self.specs[name]
        payload = json.dumps({
            "kernel": name,
            "source": hashlib.sha256(
                spec.source.encode("utf-8")
            ).hexdigest(),
            "contract": contract or "develop",
            "specialized": dict(sorted((specialized or {}).items())),
            "compiler": self._fingerprint,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def variant_directory(self, name: str, key: str) -> Path:
        return self.root / name / key

    # -- admission --------------------------------------------------------
    def get(self, name: str, *, contract: str | None = None,
            specialized: Mapping[str, int] | None = None,
            compile_missing: bool = True) -> CompiledVariant:
        """Return an admitted variant, materializing or compiling as needed.

        Raises :class:`BankRefusal` if the variant was previously refused
        (recorded on disk), fails verification now, or is absent with
        ``compile_missing=False``.
        """

        spec = self.specs[name]
        specialized = dict(specialized or {})
        if specialized:
            from .work_contract import active_contract

            vetoed = set(specialized) & set(
                map(str, active_contract().symbolic_arguments)
            )
            if vetoed:
                raise BankRefusal(
                    f"{name}: parameters {sorted(vetoed)} are on the active "
                    "contract's symbolic_arguments veto list; a specializer "
                    "must treat that list as a veto"
                )
            unknown = set(specialized) - set(spec.size_parameters)
            if unknown:
                raise BankRefusal(
                    f"{name}: cannot bake non-size parameters "
                    f"{sorted(unknown)}"
                )
            # A baked size at or below the loop unroll limit makes the
            # evaporator unroll that loop, and the tiny-trip evaporation
            # defect (pinned strict-xfail in test_compiled_linalg.py:
            # the outer loop's iteration is dropped, its induction
            # variable leaks as a free formal, and the NATIVE BUILD
            # ACCESS-VIOLATES) then kills the admission probe's process
            # outright -- a crash the verify gate cannot catch. Refuse
            # before compiling. The limit is read from the loop
            # composer's own declaration, not restated here.
            from .loop_composer import LoopBackendCapabilities

            unroll_limit = int(
                LoopBackendCapabilities.__dataclass_fields__[
                    "unroll_limit"
                ].default
            )
            tiny = {
                parameter: value
                for parameter, value in specialized.items()
                if int(value) <= unroll_limit
            }
            if tiny:
                raise BankRefusal(
                    f"{name}: baked size(s) {tiny} are at or below the "
                    f"loop unroll limit ({unroll_limit}); the tiny-trip "
                    "evaporation defect makes such builds crash natively "
                    "(see tests/test_compiled_linalg.py's pin) -- refused "
                    "before compilation"
                )
        key = self.variant_key(
            name, contract=contract, specialized=specialized,
        )
        if key in self._live:
            return self._live[key]
        directory = self.variant_directory(name, key)
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("verification", {}).get("admitted") is False:
                raise BankRefusal(
                    f"{name}[{key}] was refused at admission: "
                    f"{manifest['verification'].get('reason', 'unrecorded')}"
                )
            variant = self._materialize(
                spec, key, directory, contract, specialized, manifest,
            )
            self._live[key] = variant
            return variant
        if not compile_missing:
            raise BankRefusal(f"{name}[{key}] is not in the bank")
        variant = self._compile_and_admit(
            spec, key, directory, contract, specialized,
        )
        self._live[key] = variant
        return variant

    def _lower_and_emit(self, spec: KernelSpec, key: str,
                        contract: str | None,
                        specialized: Mapping[str, int],
                        directory: Path):
        from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
        )
        from src.compiler.work_contract import set_active_contract

        source = (
            _specialize_source(spec, specialized) if specialized
            else spec.source
        )
        tag = f"kb_{spec.name}_{key}"
        set_active_contract(contract)
        try:
            module_pickle = directory / "control_module.pkl"
            module = outputs = None
            if module_pickle.is_file():
                try:
                    with module_pickle.open("rb") as stream:
                        module, outputs = pickle.load(stream)
                except Exception:
                    module = outputs = None
            if module is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    module, outputs, _exports = lower_ast_source_to_ssa(
                        source, spec.function_name, name=tag,
                    )
                try:
                    with module_pickle.open("wb") as stream:
                        pickle.dump((module, outputs), stream, protocol=5)
                except Exception:
                    module_pickle.unlink(missing_ok=True)
            entrypoint = f"{tag}__{spec.function_name}"
            artifact = emit_ssa_function_to_llvm(module, entrypoint)
            if not artifact.complete:
                raise BankRefusal(
                    f"{spec.name}: {len(artifact.shortfalls)} emission "
                    "shortfall(s): " + "; ".join(
                        s.reason[:120] for s in artifact.shortfalls[:3]
                    )
                )
            native = compile_artifact(artifact, directory=directory)
        finally:
            set_active_contract(None)
        function = module.functions[entrypoint]
        table = dict(
            function.metadata.get("parameter_names")
            or function.metadata.get("value_names") or ()
        )
        id_by_name = {str(k): int(v) for k, v in table.items()}
        output_names = {
            str(p) for p, _v in (function.metadata.get("named_outputs") or ())
        }
        ret_ids = tuple(
            int(value.id) for value in (outputs.get(entrypoint) or ())
        )
        return module, outputs, native, id_by_name, output_names, ret_ids

    def _materialize(self, spec, key, directory, contract, specialized,
                     manifest) -> CompiledVariant:
        (module, outputs, native, id_by_name, output_names,
         ret_ids) = self._lower_and_emit(
            spec, key, contract, specialized, directory,
        )
        return CompiledVariant(
            spec, key, directory, contract, dict(specialized),
            module, outputs, native, id_by_name, output_names, ret_ids,
        )

    def _compile_and_admit(self, spec, key, directory, contract,
                           specialized) -> CompiledVariant:
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "kernel": spec.name,
            "key": key,
            "contract": contract or "develop",
            "specialized": dict(specialized),
            "compiler_fingerprint": self._fingerprint,
            "source_sha256": hashlib.sha256(
                spec.source.encode("utf-8")
            ).hexdigest(),
            "built_unix": time.time(),
        }
        try:
            variant = self._materialize(
                spec, key, directory, contract, specialized, manifest,
            )
            verification = self._verify(variant)
        except BankRefusal as refusal:
            manifest["verification"] = {
                "admitted": False, "reason": str(refusal)[:400],
            }
            (directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8",
            )
            raise
        manifest["verification"] = verification
        manifest["binding"] = {
            "parameter_ids_at_build": variant.id_by_name,
            "note": "informational only; ids are re-derived at load "
                    "(unstable across lowerings)",
        }
        if spec.extents is not None:
            manifest["data_layout"] = {
                "extents": {
                    parameter: list(dims)
                    for parameter, dims in spec.extents.items()
                },
                "note": "element counts as products of size parameters; "
                        "the declaration item_data() partitions by",
            }
        if verification["admitted"]:
            manifest["profile"] = self._profile(variant, verification)
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )
        if not verification["admitted"]:
            raise BankRefusal(
                f"{spec.name}[{key}] failed verification: "
                f"{verification['reason']}"
            )
        return variant

    def _profile(self, variant: CompiledVariant, verification: dict) -> dict:
        """Auto-collected performance profile, taken at build time.

        Every admitted variant gets a LAUNCH average and a COMPUTE average,
        because the deployment strategy layer is starved without them: the
        single admission timing is a cold call, which conflates dispatch
        overhead with arithmetic and understates steady state badly (noted
        the day the bank was built). The split here is measured, not
        modeled: warm repeats on the already-prepared execution give the
        compute average; cold repeats with the execution cache cleared give
        the full launch+compute cost; launch is their difference, floored
        at zero. Timings are medians (robust to a scheduler hiccup at
        build time), sample counts recorded, sizes stated -- so a
        performance chart row is auditable evidence, deployment-
        classification style, not a mystery number.
        """

        spec = variant.spec
        sizes = {
            str(parameter): int(value)
            for parameter, value in (
                verification.get("probe_sizes") or {}
            ).items()
        }
        rng = np.random.default_rng(
            int(hashlib.sha256(variant.key.encode()).hexdigest()[:8], 16)
        )
        sample = spec.example_inputs(sizes, rng)

        warm: list[float] = []
        for _ in range(5):
            started = time.perf_counter()
            variant.run(sample)
            warm.append(time.perf_counter() - started)
        cold: list[float] = []
        for _ in range(3):
            variant._executions.clear()
            started = time.perf_counter()
            variant.run(sample)
            cold.append(time.perf_counter() - started)
        compute_avg = float(np.median(warm))
        cold_avg = float(np.median(cold))
        return {
            "sizes": sizes,
            "compute_avg_seconds": compute_avg,
            "cold_avg_seconds": cold_avg,
            "launch_avg_seconds": max(0.0, cold_avg - compute_avg),
            "warm_samples": len(warm),
            "cold_samples": len(cold),
        }

    def performance_chart(self, name: str) -> list[dict]:
        """Every profiled variant of ``name``: the chart the deployment
        strategy calls read. One row per manifest carrying a profile,
        newest build last; refused variants have no profile and no row."""

        rows = []
        for manifest in self.inventory():
            if manifest.get("kernel") != name:
                continue
            profile = manifest.get("profile")
            if not profile:
                continue
            rows.append({
                "key": manifest.get("key"),
                "contract": manifest.get("contract"),
                "specialized": manifest.get("specialized") or {},
                "sizes": profile.get("sizes") or {},
                "launch_avg_seconds": profile.get("launch_avg_seconds"),
                "compute_avg_seconds": profile.get("compute_avg_seconds"),
                "cold_avg_seconds": profile.get("cold_avg_seconds"),
                "built_unix": manifest.get("built_unix"),
            })
        rows.sort(key=lambda row: row.get("built_unix") or 0)
        return rows

    def _verify(self, variant: CompiledVariant) -> dict:
        """The admission check: compiled output must match the Python
        reference on seeded probe inputs, or the variant is refused."""

        spec = variant.spec
        sizes = {p: 6 + 2 * index for index, p in
                 enumerate(spec.size_parameters)}
        sizes.update(variant.specialized)
        rng = np.random.default_rng(
            int(hashlib.sha256(variant.key.encode()).hexdigest()[:8], 16)
        )
        sample = spec.example_inputs(sizes, rng)
        reference_args = [
            (np.array(sample[p], copy=True)
             if isinstance(sample[p], np.ndarray) else sample[p])
            for p in spec.parameter_order
        ]
        expected = spec.reference(*reference_args)
        started = time.perf_counter()
        produced = variant.run(sample)
        elapsed = time.perf_counter() - started
        try:
            worst = float(np.max(np.abs(
                np.asarray(produced, dtype=np.float64)
                - np.asarray(expected, dtype=np.float64)
            )))
        except Exception as error:
            return {
                "admitted": False,
                "reason": f"result shape mismatch: {error}",
                "probe_sizes": sizes,
            }
        admitted = bool(worst <= 1e-9)
        return {
            "admitted": admitted,
            "reason": None if admitted else f"worst |err| {worst:.3e}",
            "worst_abs_error": worst,
            "probe_sizes": sizes,
            "probe_call_seconds": elapsed,
        }

    # -- selection --------------------------------------------------------
    def select(self, name: str, *, sizes: Mapping[str, int] | None = None,
               contract: str | None = None,
               allow_specialized: bool = True,
               compile_missing: bool = True) -> tuple[CompiledVariant, str]:
        """The variant matched to a call's details, and which kind it is.

        Returns ``(variant, "specialized" | "parametric")``. An exact-size
        specialized artifact wins when one is ADMITTED (a previously
        refused specialization is skipped silently -- its refusal is on
        disk for anyone auditing); otherwise the parametric artifact
        serves every size, compiled on miss when allowed. Raises
        :class:`BankRefusal` when nothing can serve the call. No policy
        beyond that lives here -- whether to call this at all is the
        launch coordination layer's decision.
        """

        sizes = dict(sizes or {})
        if allow_specialized and sizes:
            try:
                return self.get(
                    name, contract=contract, specialized=sizes,
                    compile_missing=False,
                ), "specialized"
            except BankRefusal:
                pass
        return self.get(
            name, contract=contract, specialized=None,
            compile_missing=compile_missing,
        ), "parametric"

    # -- inventory --------------------------------------------------------
    def inventory(self) -> list[dict]:
        rows = []
        for manifest_path in sorted(self.root.glob("*/*/manifest.json")):
            try:
                rows.append(json.loads(
                    manifest_path.read_text(encoding="utf-8")
                ))
            except (OSError, ValueError):
                continue
        return rows


class LaunchCoordinator:
    """Per-call launch coordination: one call in, one routed execution out.

    Given a call's details (kernel name, argument values, the sizes they
    imply), pick the variant that serves it -- exact-size specialized if
    admitted, else parametric (compiled on miss when allowed) -- run it,
    and fall back to the kernel's own Python reference when nothing
    compiled can serve. Every decision is appended to
    ``routing_log.jsonl`` in the bank root so the routing history is
    auditable (and available to the progressive-region-replacement effort
    as evidence of what actually runs where).

    This is strictly PER-CALL coordination. There is deliberately no
    batching here -- no queueing, no grouping of calls, no deferred
    scheduling; each ``launch`` resolves and executes immediately.
    """

    def __init__(self, bank: KernelBank, *, contract: str | None = None,
                 prefer_specialized: bool = True,
                 compile_missing: bool = True,
                 specialize_missing: bool = False,
                 fallback_to_reference: bool = True):
        self.bank = bank
        self.contract = contract
        self.prefer_specialized = prefer_specialized
        self.compile_missing = compile_missing
        #: When True, a launch with concrete sizes and no admitted
        #: specialized build triggers one (verified at admission) so the
        #: NEXT launch at these sizes routes to it.
        self.specialize_missing = specialize_missing
        self.fallback_to_reference = fallback_to_reference
        self.log_path = bank.root / "routing_log.jsonl"

    def _log(self, record: dict) -> None:
        record["unix"] = time.time()
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    def launch(self, name: str, **arguments):
        spec = self.bank.specs[name]
        sizes = {
            p: int(arguments[p]) for p in spec.size_parameters
            if p in arguments
        }
        attempts: list[str] = []
        if self.specialize_missing and sizes:
            try:
                self.bank.get(
                    name, contract=self.contract, specialized=sizes,
                    compile_missing=True,
                )
            except BankRefusal as refusal:
                attempts.append(f"specialize: {refusal}")
        try:
            variant, kind = self.bank.select(
                name, sizes=sizes, contract=self.contract,
                allow_specialized=self.prefer_specialized,
                compile_missing=self.compile_missing,
            )
            result = variant.run(arguments)
            self._log({"kernel": name, "route": kind,
                       "key": variant.key, "sizes": sizes})
            return result
        except BankRefusal as refusal:
            attempts.append(f"select: {refusal}")
        if self.fallback_to_reference:
            self._log({"kernel": name, "route": "reference",
                       "sizes": sizes, "attempts": attempts})
            ordered = [arguments[p] for p in spec.parameter_order]
            return spec.reference(*ordered)
        raise BankRefusal(
            f"{name}: no route available; attempts: {attempts}"
        )


# ---------------------------------------------------------------------------
# BLAS registration
# ---------------------------------------------------------------------------

def _blas_example_inputs(arity: tuple[str, ...]):
    def generate(sizes: Mapping[str, int], rng: np.random.Generator) -> dict:
        m = int(sizes.get("m", sizes.get("n", 4)))
        n = int(sizes["n"])
        k = int(sizes.get("k", 4))
        values: dict[str, Any] = {}
        for parameter in arity:
            if parameter in ("m", "n", "k"):
                values[parameter] = int(sizes[parameter])
            elif parameter == "alpha":
                values[parameter] = 1.5
            elif parameter == "beta":
                values[parameter] = 0.5
            elif parameter == "A":
                rows = m if "m" in arity else n
                cols = k if "k" in arity else n
                values[parameter] = rng.uniform(-2, 2, size=rows * cols)
            elif parameter == "B":
                values[parameter] = rng.uniform(-2, 2, size=k * n)
            elif parameter == "C":
                values[parameter] = rng.uniform(-2, 2, size=m * n)
            elif parameter == "y":
                length = m if "m" in arity else n
                values[parameter] = rng.uniform(-2, 2, size=length)
            elif parameter == "x":
                values[parameter] = rng.uniform(-2, 2, size=n)
            else:
                raise ValueError(f"no sample rule for {parameter!r}")
        return values
    return generate


def derive_size_parameters_from_source(
    source: str, function_name: str,
) -> tuple[str, ...]:
    """The parameters that bound loops, read from the authored source.

    A size parameter IS a parameter some ``range`` stops at -- that is the
    whole definition, and the source states it; a side table restating it
    would drift. Order follows first lexical appearance as a bound, which
    is the nesting order for these kernels.
    """

    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    parameters = {argument.arg for argument in function.args.args}
    ordered: list[str] = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            arguments = node.iter.args
            stop = arguments[0] if len(arguments) == 1 else arguments[1]
            if (
                isinstance(stop, ast.Name)
                and stop.id in parameters
                and stop.id not in ordered
            ):
                ordered.append(stop.id)
    return tuple(ordered)


def derive_extents_from_source(
    source: str, function_name: str,
) -> dict[str, tuple[str, ...]]:
    """Each parameter's extent product, read from the AUTHORED SOURCE.

    The kernel's own loop bounds and index arithmetic are the single
    authority on data layout -- a hand-maintained mirror table would drift
    the moment a kernel changes. The flat row-major convention the whole
    bank compiles under makes derivation exact:

    * ``x[i]`` with ``for i in range(n)`` -> ``x`` spans ``(n,)``;
    * ``A[i * k + p]`` -> the multiplier is the row stride and the row
      variable's own loop bound is the leading dimension: ``(bound(i), k)``.

    A parameter never subscripted is a scalar, ``()``. An index shape this
    reader does not recognize raises -- a partitioner working from a
    guessed extent is how buffers get torn, so unknown stays unknown
    loudly.
    """

    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    parameters = [argument.arg for argument in function.args.args]

    bounds: dict[str, str] = {}
    for node in ast.walk(function):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            arguments = node.iter.args
            stop = arguments[0] if len(arguments) == 1 else arguments[1]
            if isinstance(stop, ast.Name):
                bounds[node.target.id] = stop.id

    def index_dims(expression: ast.AST) -> tuple[str, ...] | None:
        if isinstance(expression, ast.Name):
            bound = bounds.get(expression.id)
            return (bound,) if bound else None
        if isinstance(expression, ast.BinOp) and isinstance(
            expression.op, ast.Add
        ):
            for product, offset in (
                (expression.left, expression.right),
                (expression.right, expression.left),
            ):
                if not (
                    isinstance(product, ast.BinOp)
                    and isinstance(product.op, ast.Mult)
                    and isinstance(product.left, ast.Name)
                    and isinstance(product.right, ast.Name)
                ):
                    continue
                names = (product.left.id, product.right.id)
                for variable, stride in (names, names[::-1]):
                    row_bound = bounds.get(variable)
                    inner = index_dims(offset)
                    if row_bound and inner is not None and stride not in bounds:
                        return (row_bound, stride)
        return None

    extents: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(function):
        if not (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in parameters
        ):
            continue
        dims = index_dims(node.slice)
        if dims is None:
            raise BankRefusal(
                f"{function_name}: cannot derive an extent from the index "
                f"expression {ast.unparse(node)!r}; declare extents "
                "explicitly rather than letting a partitioner guess"
            )
        previous = extents.get(node.value.id)
        if previous is not None and previous != dims:
            raise BankRefusal(
                f"{function_name}: parameter {node.value.id!r} is indexed "
                f"with disagreeing extents {previous!r} and {dims!r}"
            )
        extents[node.value.id] = dims
    for parameter in parameters:
        extents.setdefault(parameter, ())
    return extents


def blas_kernel_specs() -> dict[str, KernelSpec]:
    from src.common.tensors.blas import KERNELS

    specs = {}
    for _level, name, source, reference, arity in KERNELS:
        specs[name] = KernelSpec(
            name=name,
            source=source,
            function_name=name,
            reference=reference,
            parameter_order=tuple(arity),
            size_parameters=derive_size_parameters_from_source(source, name),
            example_inputs=_blas_example_inputs(tuple(arity)),
            extents=derive_extents_from_source(source, name),
        )
    return specs


def open_blas_bank(root: str | Path = "build/kernel_bank") -> KernelBank:
    return KernelBank(root, blas_kernel_specs())

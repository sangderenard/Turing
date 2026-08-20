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

MANIFEST_SCHEMA = "turing.kernel-bank.v1"


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
        feeds = {self.id_by_name[p]: live[p] for p in live}
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
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )
        if not verification["admitted"]:
            raise BankRefusal(
                f"{spec.name}[{key}] failed verification: "
                f"{verification['reason']}"
            )
        return variant

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


_BLAS_SIZE_PARAMETERS = {
    "scal": ("n",), "axpy": ("n",), "dot": ("n",),
    "gemv": ("m", "n"), "gemm": ("m", "n", "k"),
}


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
            size_parameters=_BLAS_SIZE_PARAMETERS[name],
            example_inputs=_blas_example_inputs(tuple(arity)),
        )
    return specs


def open_blas_bank(root: str | Path = "build/kernel_bank") -> KernelBank:
    return KernelBank(root, blas_kernel_specs())

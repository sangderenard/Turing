"""Receipt-gated installation of compiler products in isolated workers.

Compiler bootstrap products are ordinary project-compilation products.  They
are never trusted merely because a DLL exists: activation reloads the product,
re-runs the semantic verifier named by its persisted receipt, and then uses the
same authored-source-preserving installer as standard objects.  Child compiler
workers inherit the product list through the environment, so a bounded crawl
uses the same proven deployments without surrendering source recursively.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Callable


COMPILER_BOOTSTRAP_PRODUCTS_ENV = "TURING_COMPILER_BOOTSTRAP_PRODUCTS"
COMPILER_BOOTSTRAP_REGISTRY_ENV = "TURING_COMPILER_BOOTSTRAP_REGISTRY"
COMPILER_BOOTSTRAP_REGISTRY_SCHEMA = "turing.compiler-bootstrap-registry.v1"
PROVENANCE_CORE_MODULE = "src.compiler.identity_concordance"
PROVENANCE_SOURCE_ONLY_ENV = "TURING_PROVENANCE_SOURCE_ONLY"
_PROVENANCE_CORE_STATE = {"backend": "python", "reason": "no installed native core"}


def provenance_core_state() -> dict[str, Any]:
    """The whole module selected at process startup; never per-access routing."""
    return dict(_PROVENANCE_CORE_STATE)


def _verified_provenance_core(record):
    """Check a whole-module link without importing compiler implementation."""
    source = Path(__file__).with_name("identity_concordance.py")
    library = Path(record["native_module"]).resolve()
    receipt_path = Path(record["verification_receipt"]).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    repository = Path(__file__).resolve().parents[2]
    ledger = {item["path"]: item["sha256"]
              for item in (receipt.get("compiler_toolchain") or {}).get("files", ())}
    required = {"src/compiler/identity_concordance.py", "src/compiler/id_space.py",
                "src/compiler/monotonic_ids.py", "src/compiler/concordance_declarations.py"}
    if (receipt.get("schema") != "turing.native-provenance-core-verification.v1"
            or receipt.get("status") != "verified"
            or receipt.get("scope") != "whole-module"
            or receipt.get("module") != PROVENANCE_CORE_MODULE
            or receipt.get("python_cache_tag") != sys.implementation.cache_tag
            or receipt.get("source_sha256") != _file_sha256(source)
            or receipt.get("library_sha256") != _file_sha256(library)
            or record.get("verification_sha256") != _file_sha256(receipt_path)
            or receipt.get("fallback_probe_count") != 0
            or type(receipt.get("native_probe_count")) is not int
            or receipt["native_probe_count"] < 1
            or not required.issubset(ledger)
            or any(_file_sha256(repository / path) != digest for path, digest in ledger.items())):
        raise ValueError("native provenance core does not have a matching whole-module verification")
    return library, receipt


def activate_native_provenance_core(registry_path=None) -> bool:
    """Select the complete native module before any book imports exist.

    The native artifact must implement Python's module initialization ABI for
    the canonical module, including its complete operator/query/formatting
    surface. A raw bind(C) library or a set of accessor wrappers cannot enter
    this route. Existing processes never mix Python and native book state.
    """
    if os.environ.get(PROVENANCE_SOURCE_ONLY_ENV) == "1":
        _PROVENANCE_CORE_STATE.update(backend="python", reason="isolated provenance bootstrap")
        return False
    if PROVENANCE_CORE_MODULE in sys.modules:
        return _PROVENANCE_CORE_STATE["backend"] == "native"
    path = compiler_bootstrap_registry_path(registry_path)
    if not path.is_file():
        return False
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
        if registry.get("schema") != COMPILER_BOOTSTRAP_REGISTRY_SCHEMA:
            raise ValueError("unsupported compiler bootstrap registry schema")
        record = registry.get("provenance_core")
        if not record:
            return False
        library, receipt = _verified_provenance_core(record)
        from importlib.machinery import ExtensionFileLoader
        from importlib.util import module_from_spec, spec_from_file_location
        loader = ExtensionFileLoader(PROVENANCE_CORE_MODULE, str(library))
        spec = spec_from_file_location(PROVENANCE_CORE_MODULE, library, loader=loader)
        module = module_from_spec(spec)
        sys.modules[PROVENANCE_CORE_MODULE] = module
        try:
            loader.exec_module(module)
        except BaseException:
            sys.modules.pop(PROVENANCE_CORE_MODULE, None)
            raise
        setattr(sys.modules[__package__], "identity_concordance", module)
        _PROVENANCE_CORE_STATE.update(backend="native", reason="verified whole-module link",
                                      library=str(library), verification=receipt)
        return True
    except Exception as error:
        sys.modules.pop(PROVENANCE_CORE_MODULE, None)
        _PROVENANCE_CORE_STATE.update(backend="python", reason=str(error))
        return False


def publish_native_provenance_core(native_module, verification_receipt, *, registry_path=None):
    """Atomically install one verified coarse module for fresh compiler workers."""
    library = Path(native_module).resolve()
    verification = Path(verification_receipt).resolve()
    record = {"native_module": str(library), "verification_receipt": str(verification),
              "verification_sha256": _file_sha256(verification)}
    _verified_provenance_core(record)
    destination = compiler_bootstrap_registry_path(registry_path)
    registry = (json.loads(destination.read_text(encoding="utf-8")) if destination.is_file()
                else {"schema": COMPILER_BOOTSTRAP_REGISTRY_SCHEMA, "products": []})
    if registry.get("schema") != COMPILER_BOOTSTRAP_REGISTRY_SCHEMA:
        raise ValueError("unsupported compiler bootstrap registry schema")
    registry["provenance_core"] = record
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        _probe_native_provenance_core(temporary, library)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _probe_native_provenance_core(registry_path, library):
    """Re-prove import, complete surface and book semantics in a fresh process."""
    import subprocess
    import tempfile
    environment = os.environ.copy()
    environment.pop(PROVENANCE_SOURCE_ONLY_ENV, None)
    environment.pop(COMPILER_BOOTSTRAP_PRODUCTS_ENV, None)
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                     dir=Path(registry_path).parent, encoding="utf-8") as candidate:
        json.dump({"schema": COMPILER_BOOTSTRAP_REGISTRY_SCHEMA, "products": [],
                   "provenance_core": registry["provenance_core"]}, candidate)
    environment[COMPILER_BOOTSTRAP_REGISTRY_ENV] = candidate.name
    repository = Path(__file__).resolve().parents[2]
    probe = r'''
import pathlib, sys
from src.compiler.compiler_bootstrap_runtime import provenance_core_state, _provenance_core_surface_findings
state = provenance_core_state()
assert state["backend"] == "native", state
assert pathlib.Path(state["library"]).resolve() == pathlib.Path(sys.argv[1]).resolve()
import src.compiler.identity_concordance as core
source = pathlib.Path("src/compiler/identity_concordance.py")
findings = _provenance_core_surface_findings(core, source.read_text())
assert not findings, findings
import pytest
raise SystemExit(pytest.main(["-q", "tests/test_copy_on_read_scopes.py",
                             "tests/test_provenance_compartment_operations.py",
                             "tests/test_provenance_storage_contracts.py",
                             "tests/test_pythonic_control_torture.py::test_pythonic_control_native_effects[c-guarded_splice]"]))
'''
    try:
        completed = subprocess.run([sys.executable, "-c", probe, str(library)],
                                   cwd=repository, env=environment,
                                   capture_output=True, text=True)
    finally:
        Path(candidate.name).unlink(missing_ok=True)
    if completed.returncode:
        raise ValueError("native provenance module failed fresh-process verification:\n"
                         + completed.stdout + completed.stderr)


def _provenance_core_surface_findings(module, source):
    """Check authored exports without importing a Python copy of the core.

    Constants and class descriptors are part of the module surface too. This
    is an import-shape check; the operation suites separately prove semantics.
    """
    findings = []
    def names(target):
        if isinstance(target, ast.Name):
            return (target.id,)
        if isinstance(target, (ast.Tuple, ast.List)):
            return tuple(name for child in target.elts for name in names(child))
        return ()
    def declarations(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                yield node.name, node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    for name in names(target):
                        yield name, node
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                for name in names(node.target):
                    yield name, node
    for name, node in declarations(ast.parse(source).body):
        if not hasattr(module, name):
            findings.append({"kind": "missing-module-symbol", "symbol": name})
        elif isinstance(node, ast.ClassDef):
            owner = getattr(module, name)
            for member, _ in declarations(node.body):
                if (not hasattr(owner, member)
                        and member not in getattr(owner, "__dataclass_fields__", {})):
                    findings.append({"kind": "missing-class-member", "symbol": f"{name}.{member}"})
    return findings


class ProvenanceCoreRestartRequired(RuntimeError):
    """A stage is persisted; restart before the next compile or activation."""

    def __init__(
        self, message: str, *, requires_process_restart: bool = True,
    ) -> None:
        super().__init__(message)
        self.requires_process_restart = bool(requires_process_restart)


def ensure_native_provenance_core(
    directory, *, jobs=None, schedule_options=None,
    stop_after_failure=True, progress=None,
):
    """Blocking first stage; no outer compiler workers may start before it."""
    import subprocess
    if provenance_core_state()["backend"] == "native":
        return provenance_core_state()
    registry_path = compiler_bootstrap_registry_path()
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        record = registry.get("provenance_core")
        if record:
            try:
                library, _ = _verified_provenance_core(record)
            except (ValueError, OSError, KeyError):
                pass  # An obsolete product is rebuilt from the current source.
            else:
                _probe_native_provenance_core(registry_path, library)
                raise ProvenanceCoreRestartRequired("restart to select installed native provenance core")
    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment[PROVENANCE_SOURCE_ONLY_ENV] = "1"
    environment.pop(COMPILER_BOOTSTRAP_PRODUCTS_ENV, None)
    isolated_registry = destination / "source-compiler-registry.json"
    isolated_registry.write_text(json.dumps({"schema": COMPILER_BOOTSTRAP_REGISTRY_SCHEMA,
                                             "products": []}) + "\n", encoding="utf-8")
    environment[COMPILER_BOOTSTRAP_REGISTRY_ENV] = str(isolated_registry)
    if progress is not None:
        progress("compiler-bootstrap: growing provenance core through SSA enclosures; outer workers blocked")
    command = [sys.executable, "-m", "tools.bootstrap_compiler_concordance",
               "--grow-stage", "--jobs", str(12 if jobs is None else jobs),
               "--schedule-json", json.dumps(dict(schedule_options or {}))]
    if not stop_after_failure:
        command.append("--continue-after-failure")
    command.extend(("--output", str(destination)))
    subprocess.run(command,
                   cwd=Path(__file__).resolve().parents[2], env=environment, check=True)
    receipt = json.loads((destination / "compartment.json").read_text(encoding="utf-8"))
    if receipt.get("stage") == "native-growth-stage-persisted":
        # The compilation stage itself ran in the isolated child above, so
        # this coordinator has no live IdentityBook or worker state to shed.
        # Continue from the persisted receipts in this process.  A genuine
        # process restart remains mandatory only after a native core is
        # installed or selected.
        raise ProvenanceCoreRestartRequired(
            "provenance interior stage persisted; continue with enclosing units",
            requires_process_restart=False,
        )
    if receipt.get("stage") != "native-module-verified":
        raise RuntimeError("provenance-first bootstrap blocked: whole native module is not verified; "
                           f"stage={receipt.get('stage')!r}; receipt={destination / 'compartment.json'}")
    publish_native_provenance_core(receipt["native_module"], receipt["verification_receipt"])
    raise ProvenanceCoreRestartRequired("native provenance installed; restart before compiling outer units")


@dataclass(frozen=True)
class CompilerBootstrapActivation:
    product: str
    qualified_name: str
    adapter: str
    status: str
    native_probe_count: int
    fallback_probe_count: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "qualified_name": self.qualified_name,
            "adapter": self.adapter,
            "status": self.status,
            "native_probe_count": int(self.native_probe_count),
            "fallback_probe_count": int(self.fallback_probe_count),
        }


def _compute_dispatch_is_native_safe(
    count,
    *,
    limits,
    preferred_local_size=256,
    minimum_local_size=32,
) -> bool:
    """Mirror only the native library's terminating validation boundary."""

    try:
        count = int(count)
        preferred = int(preferred_local_size)
        minimum = int(minimum_local_size)
        if count < 0 or preferred < 1 or minimum < 1:
            return False
        local_cap = min(
            preferred,
            int(limits.max_group_size[0]),
            int(limits.max_invocations),
        )
        if local_cap < 1:
            return False
        local = 1 << (local_cap.bit_length() - 1)
        if count:
            small_target = 1 << (count - 1).bit_length()
            local = min(local, max(min(minimum, local), small_target))
        if count == 0:
            return True
        needed = (count + local - 1) // local
        group_x = min(needed, int(limits.max_group_count[0]))
        if group_x < 1:
            return False
        remaining = (needed + group_x - 1) // group_x
        group_y = min(remaining, int(limits.max_group_count[1]))
        if group_y < 1:
            return False
        group_z = (remaining + group_y - 1) // group_y
        return group_z <= int(limits.max_group_count[2])
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


def _activate_compute_dispatch(product, qualified_name: str):
    from . import deployment_lowering as owner

    generous = owner.ComputeDispatchLimits(
        (65535, 65535, 65535), (1024, 1024, 64), 1024,
    )
    compact = owner.ComputeDispatchLimits((4, 3, 2), (64, 64, 64), 64)
    probes = (
        {"count": 0, "limits": generous},
        {"count": 1, "limits": generous},
        {"count": 1024, "limits": generous},
        {
            "count": 1000,
            "limits": compact,
            "preferred_local_size": 64,
            "minimum_local_size": 8,
        },
        {"count": -1, "limits": generous},
    )
    deployed = product.verify_native_record_return_callable(
        qualified_name,
        owner.plan_compute_dispatch,
        probes,
        native_precondition=_compute_dispatch_is_native_safe,
        activation_adapter="compute-dispatch-record-v1",
    )
    return owner, deployed


def _activate_qualified_scalar(product, qualified_name: str):
    """Re-prove one ABI-selected scalar leaf and return its authored owner."""

    from .project_compilation_product import _resolve_product_callable

    link = dict(product.links[str(qualified_name)])
    source_module = str(link.get("source_module") or "")
    if not source_module:
        raise ValueError("scalar bootstrap link has no source module")
    owner, authored = _resolve_product_callable(
        source_module, str(qualified_name),
    )
    library = product.root / str(link.get("native_library") or "")
    receipt_path = library.parent / "native-verification.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    probes = []
    for record in receipt.get("probes") or ():
        arguments = ast.literal_eval(str(record.get("arguments") or "()"))
        keywords = ast.literal_eval(str(record.get("keywords") or "{}"))
        probes.append(dict(keywords) if keywords else tuple(arguments))
    deployed = product.verify_native_scalar_callable(
        str(qualified_name),
        authored,
        probes,
        activation_adapter="qualified-scalar-call-v1",
    )
    return owner, deployed


def _activate_descriptor_call(product, qualified_name: str):
    """Rebuild a descriptor-selected verifier from authored contracts."""

    from .project_compilation_product import (
        _resolve_product_callable,
        verify_project_unit_automatically, verify_project_product_cases,
    )

    link = dict(product.links[str(qualified_name)])
    source_module = str(link.get("source_module") or "")
    if not source_module:
        raise ValueError("bootstrap link has no source module")
    owner, _authored = _resolve_product_callable(
        source_module, str(qualified_name),
    )
    library = product.root / str(link.get("native_library") or "")
    receipt = json.loads((library.parent / "native-verification.json").read_text(encoding="utf-8"))
    if receipt.get("verification_oracle") == "supplied-results":
        cases_path = library.parent / "native-cases.json"
        if (not cases_path.is_file() or _file_sha256(cases_path) != receipt.get("golden_cases_sha256")):
            raise ValueError("verified golden cases are missing or changed")
        for field, path in (("source_sha256", Path(str(product.manifest.get("source") or ""))),
                            ("api_sha256", product.root / str(link.get("native_api") or "")),
                            ("library_sha256", library)):
            digest = (hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                      if field == "source_sha256" and path.is_file() else _file_sha256(path) if path.is_file() else None)
            if digest != receipt.get(field):
                raise ValueError(f"golden replay {field} changed after verification")
        replay = json.loads(cases_path.read_text(encoding="utf-8"))
        if replay.get("qualified_name") != str(qualified_name):
            raise ValueError("golden cases name another authored unit")
        specification = ast.literal_eval(replay["options"])
        specification["probes"] = [ast.literal_eval(probe["keywords"]) if ast.literal_eval(probe["keywords"])
                                   else ast.literal_eval(probe["arguments"]) for probe in replay["probes"]]
        specification["expected_probe_results"] = [ast.literal_eval(probe["expected_result"]) for probe in replay["probes"]]
        callables = {}
        results = verify_project_product_cases(product, {str(qualified_name): specification}, _verified_callables=callables)
        if not results[0].get("verified"):
            raise ValueError(f"golden native replay failed: {results[0].get('reason')}")
        deployed = callables[str(qualified_name)]
    else:
        deployed = verify_project_unit_automatically(product, str(qualified_name))
    return owner, deployed


_ACTIVATION_ADAPTERS: dict[str, Callable[..., tuple[Any, Callable[..., Any]]]] = {
    "compute-dispatch-record-v1": _activate_compute_dispatch,
    "qualified-scalar-call-v1": _activate_qualified_scalar,
    "descriptor-call-v1": _activate_descriptor_call,
}
_ACTIVE_DEPLOYMENTS: dict[tuple[str, str], Callable[..., Any]] = {}
_REGISTRY_ACTIVATION_LOCK = threading.RLock()
_ACTIVATED_REGISTRY_SHA256: str | None = None
_INITIALIZING_REGISTRY_SHA256: str | None = None
_REGISTRY_FAILURES: list[dict[str, str]] = []


def compiler_bootstrap_registry_path(value: str | Path | None = None) -> Path:
    """Return the local durable registry used by normal compiler entrypoints."""

    selected = (
        os.environ.get(COMPILER_BOOTSTRAP_REGISTRY_ENV, "")
        if value is None else str(value)
    )
    if selected:
        return Path(selected).resolve()
    return (
        Path(__file__).resolve().parents[2]
        / "build" / "compiler-bootstrap-registry.json"
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_compiler_bootstrap_products(
    paths,
    *,
    registry_path: str | Path | None = None,
) -> Path:
    """Atomically pin installable verified products for future compilers."""

    from .project_compilation_product import open_project_compilation_product

    records = []
    for raw_path in dict.fromkeys(Path(path).resolve() for path in paths):
        product = open_project_compilation_product(raw_path)
        manifest_path = product.root / "manifest.json"
        source_path = Path(str(product.manifest.get("source") or ""))
        source_sha256 = str(product.manifest.get("source_sha256") or "")
        if (
            not source_path.is_file()
            or not source_sha256
            or _file_sha256(source_path) != source_sha256
        ):
            continue
        installable = []
        for qualified_name, link_value in sorted(product.links.items()):
            link = dict(link_value)
            if link.get("kind") == "source-region-integral":
                continue
            library = product.root / str(link.get("native_library") or "")
            api = product.root / str(link.get("native_api") or "")
            receipt_path = library.parent / "native-verification.json"
            if (
                not library.is_file()
                or not api.is_file()
                or not receipt_path.is_file()
            ):
                continue
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            adapter = str(receipt.get("activation_adapter") or "")
            if (
                receipt.get("status") != "verified"
                or adapter not in _ACTIVATION_ADAPTERS
                or str(receipt.get("api_sha256") or "") != _file_sha256(api)
                or str(receipt.get("library_sha256") or "")
                != _file_sha256(library)
            ):
                continue
            installable.append({
                "qualified_name": str(qualified_name),
                "activation_adapter": adapter,
                "verification_receipt": receipt_path.relative_to(
                    product.root
                ).as_posix(),
            })
        if not installable:
            continue
        records.append({
            "product": product.root.as_posix(),
            "manifest_sha256": _file_sha256(manifest_path),
            "source_sha256": source_sha256,
            "installable": installable,
        })
    destination = compiler_bootstrap_registry_path(registry_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": COMPILER_BOOTSTRAP_REGISTRY_SCHEMA,
        "products": records,
    }
    from .compiler_source_tree import deployment_source_trees
    payload["source_trees"] = deployment_source_trees(record["product"] for record in records)
    if destination.is_file():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous.get("schema") != COMPILER_BOOTSTRAP_REGISTRY_SCHEMA:
            raise ValueError("unsupported compiler bootstrap registry schema")
        if previous.get("provenance_core"):
            payload["provenance_core"] = previous["provenance_core"]
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8", newline="\n",
    )
    os.replace(temporary, destination)
    return destination


def registered_compiler_bootstrap_product_paths(
    registry_path: str | Path | None = None,
) -> tuple[Path, ...]:
    """Load only products whose pinned manifests still match byte-for-byte."""

    path = compiler_bootstrap_registry_path(registry_path)
    if not path.is_file():
        return ()
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema") != COMPILER_BOOTSTRAP_REGISTRY_SCHEMA:
        raise ValueError("unsupported compiler bootstrap registry schema")
    selected = []
    for record in registry.get("products") or ():
        product = Path(str(record.get("product") or "")).resolve()
        manifest = product / "manifest.json"
        if (
            not manifest.is_file()
            or _file_sha256(manifest)
            != str(record.get("manifest_sha256") or "")
        ):
            raise ValueError(
                f"registered compiler product changed or vanished: {product}"
            )
        selected.append(product)
    return tuple(dict.fromkeys(selected))


def compiler_bootstrap_product_paths(value: str | None = None) -> tuple[Path, ...]:
    raw = os.environ.get(COMPILER_BOOTSTRAP_PRODUCTS_ENV, "") if value is None else value
    return tuple(dict.fromkeys(
        Path(item).resolve()
        for item in str(raw).split(os.pathsep)
        if item.strip()
    ))


def set_compiler_bootstrap_products(paths) -> tuple[Path, ...]:
    resolved = tuple(dict.fromkeys(Path(path).resolve() for path in paths))
    os.environ[COMPILER_BOOTSTRAP_PRODUCTS_ENV] = os.pathsep.join(
        str(path) for path in resolved
    )
    return resolved


def activate_compiler_bootstrap_products(paths=None, *, selected_names=None) -> tuple[CompilerBootstrapActivation, ...]:
    """Re-prove and install every receipt-declared compiler deployment."""

    from .project_compilation_product import open_project_compilation_product

    selected = (
        compiler_bootstrap_product_paths()
        if paths is None else tuple(Path(path).resolve() for path in paths)
    )
    activations = []
    for product_path in selected:
        product = open_project_compilation_product(product_path)
        for qualified_name, link_value in sorted(product.links.items()):
            if selected_names is not None and (str(product.root), str(qualified_name)) not in selected_names:
                continue
            link = dict(link_value)
            if link.get("source_module") == PROVENANCE_CORE_MODULE:
                continue  # Bootstrap elects this stateful module only as a whole.
            if link.get("kind") == "source-region-integral":
                continue
            library = product.root / str(link.get("native_library") or "")
            receipt_path = library.parent / "native-verification.json"
            if not library.is_file() or not receipt_path.is_file():
                continue
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            adapter_name = str(receipt.get("activation_adapter") or "")
            if not adapter_name:
                continue
            deployment_key = (
                product.root.as_posix(), str(qualified_name),
            )
            existing = _ACTIVE_DEPLOYMENTS.get(deployment_key)
            if existing is not None:
                verification = dict(getattr(
                    existing, "__turing_native_verification__", {},
                ))
                activations.append(CompilerBootstrapActivation(
                    product=product.root.as_posix(),
                    qualified_name=str(qualified_name),
                    adapter=adapter_name,
                    status=str(verification.get("status") or ""),
                    native_probe_count=int(
                        verification.get("native_probe_count") or 0
                    ),
                    fallback_probe_count=int(
                        verification.get("fallback_probe_count") or 0
                    ),
                ))
                continue
            try:
                adapter = _ACTIVATION_ADAPTERS[adapter_name]
            except KeyError as error:
                raise ValueError(
                    f"unknown compiler bootstrap activation adapter {adapter_name!r}"
                ) from error
            owner, deployed = adapter(product, str(qualified_name))
            installed = product.install_callable(
                str(qualified_name),
                owner,
                deployed,
                targeted_source_fallback=True,
            )
            _ACTIVE_DEPLOYMENTS[deployment_key] = (
                installed.__turing_deployed_callable__
            )
            verification = dict(
                installed.__turing_deployed_callable__.__turing_native_verification__
            )
            activations.append(CompilerBootstrapActivation(
                product=product.root.as_posix(),
                qualified_name=str(qualified_name),
                adapter=adapter_name,
                status=str(verification.get("status") or ""),
                native_probe_count=int(verification.get("native_probe_count") or 0),
                fallback_probe_count=int(verification.get("fallback_probe_count") or 0),
            ))
    return tuple(activations)


def activate_registered_compiler_bootstraps(
    registry_path: str | Path | None = None,
) -> tuple[CompilerBootstrapActivation, ...]:
    """Initialize once, allowing target imports to re-enter source compilation."""
    global _INITIALIZING_REGISTRY_SHA256
    path = compiler_bootstrap_registry_path(registry_path)
    digest = _file_sha256(path) if path.is_file() else "absent"
    with _REGISTRY_ACTIVATION_LOCK:
        if _INITIALIZING_REGISTRY_SHA256 == digest:
            return ()
        previous = _INITIALIZING_REGISTRY_SHA256
        _INITIALIZING_REGISTRY_SHA256 = digest
        try:
            return _activate_registered_compiler_bootstraps(registry_path)
        finally:
            _INITIALIZING_REGISTRY_SHA256 = previous


def _activate_registered_compiler_bootstraps(
    registry_path: str | Path | None = None,
) -> tuple[CompilerBootstrapActivation, ...]:
    """Activate the current pinned registry once per registry revision.

    A stale or source-incompatible product never prevents compilation. Its
    refusal is retained in runtime state and the authored Python callable
    remains authoritative.
    """

    global _ACTIVATED_REGISTRY_SHA256

    path = compiler_bootstrap_registry_path(registry_path)
    registry_digest = _file_sha256(path) if path.is_file() else "absent"
    with _REGISTRY_ACTIVATION_LOCK:
        if registry_digest == _ACTIVATED_REGISTRY_SHA256:
            return ()
        failures = []
        activations = []
        registry = {}
        try:
            registry = (
                {"schema": COMPILER_BOOTSTRAP_REGISTRY_SCHEMA, "products": []}
                if not path.is_file()
                else json.loads(path.read_text(encoding="utf-8"))
            )
            if registry.get("schema") != COMPILER_BOOTSTRAP_REGISTRY_SCHEMA:
                raise ValueError("unsupported compiler bootstrap registry schema")
            selected_records = tuple(registry.get("products") or ())
        except Exception as error:
            failures.append({
                "product": "",
                "error_type": type(error).__name__,
                "error": str(error),
            })
            selected_records = ()
        tree_products = {cut["product"] for tree in registry.get("source_trees", ())
                         if tree.get("module") != PROVENANCE_CORE_MODULE
                         for node in tree.get("nodes", {}).values()
                         for cut in node.get("available", ())}
        initialized_paths = []
        for record in selected_records:
            product_path = Path(str(record.get("product") or "")).resolve()
            try:
                manifest = product_path / "manifest.json"
                if (
                    not manifest.is_file()
                    or _file_sha256(manifest)
                    != str(record.get("manifest_sha256") or "")
                ):
                    raise ValueError(
                        "registered compiler product changed or vanished: "
                        f"{product_path}"
                    )
                if str(product_path) in tree_products:
                    initialized_paths.append(product_path)
                else:
                    activations.extend(activate_compiler_bootstrap_products((product_path,)))
            except Exception as error:
                failures.append({
                    "product": product_path.as_posix(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
        if initialized_paths:
            try:
                from .compiler_source_tree import initialize_source_tree
                state = initialize_source_tree(initialized_paths)
                activations.extend(CompilerBootstrapActivation(**record) for record in state["activations"])
            except Exception as error:
                failures.append({"product": ",".join(map(str, initialized_paths)),
                                 "error_type": type(error).__name__, "error": str(error)})
        _REGISTRY_FAILURES[:] = failures
        _ACTIVATED_REGISTRY_SHA256 = registry_digest
        return tuple(activations)


def compiler_bootstrap_registry_state() -> dict[str, Any]:
    """Expose registry selection and non-fatal activation refusals."""

    path = compiler_bootstrap_registry_path()
    return {
        "schema": "turing.compiler-bootstrap-registry-state.v1",
        "registry": path.as_posix(),
        "registry_sha256": (
            _file_sha256(path) if path.is_file() else None
        ),
        "activated_registry_sha256": _ACTIVATED_REGISTRY_SHA256,
        "failures": list(_REGISTRY_FAILURES),
    }


def compiler_bootstrap_runtime_state() -> tuple[dict[str, Any], ...]:
    """Return live native/fallback routing evidence for installed products."""

    records = []
    for (product, qualified_name), deployed in sorted(_ACTIVE_DEPLOYMENTS.items()):
        verification = dict(getattr(
            deployed, "__turing_native_verification__", {},
        ))
        routes = dict(getattr(deployed, "__turing_native_route_counts__", {}))
        records.append({
            "product": product,
            "qualified_name": qualified_name,
            "activation_adapter": str(
                verification.get("activation_adapter") or ""
            ),
            "verification_status": str(verification.get("status") or ""),
            "verification_native_probe_count": int(
                verification.get("native_probe_count") or 0
            ),
            "verification_fallback_probe_count": int(
                verification.get("fallback_probe_count") or 0
            ),
            "runtime_native_calls": int(routes.get("native") or 0),
            "runtime_fallback_calls": int(routes.get("fallback") or 0),
            "post_activation_native_calls": max(
                0,
                int(routes.get("native") or 0)
                - int(verification.get("native_probe_count") or 0),
            ),
            "post_activation_fallback_calls": max(
                0,
                int(routes.get("fallback") or 0)
                - int(verification.get("fallback_probe_count") or 0),
            ),
        })
    return tuple(records)


__all__ = [
    "PROVENANCE_CORE_MODULE",
    "ProvenanceCoreRestartRequired",
    "activate_native_provenance_core",
    "ensure_native_provenance_core",
    "provenance_core_state",
    "publish_native_provenance_core",
    "COMPILER_BOOTSTRAP_REGISTRY_ENV",
    "COMPILER_BOOTSTRAP_REGISTRY_SCHEMA",
    "COMPILER_BOOTSTRAP_PRODUCTS_ENV",
    "CompilerBootstrapActivation",
    "activate_compiler_bootstrap_products",
    "activate_registered_compiler_bootstraps",
    "compiler_bootstrap_registry_path",
    "compiler_bootstrap_registry_state",
    "compiler_bootstrap_runtime_state",
    "compiler_bootstrap_product_paths",
    "publish_compiler_bootstrap_products",
    "registered_compiler_bootstrap_product_paths",
    "set_compiler_bootstrap_products",
]

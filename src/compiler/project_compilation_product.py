"""Isolated authored-call compilation for large Python projects.

Each selected source-level function or method is compiled in a fresh process.
The worker owns and then releases its entire ProcessGraph/planner heap.  The
parent retains only small receipts and publishes one deterministic catalogue
manifest, so a failed or exceptionally large call cannot destabilize later
catalogue entries.
"""

from __future__ import annotations

import ast
import copyreg
import dataclasses
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import symtable
import sys
import threading
import time
import types
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_PRODUCT_SCHEMA = "turing.project-compilation-product.v1"
UNIT_ARTIFACT_SCHEMA = "turing.project-compilation-unit.v1"
LINK_TABLE_SCHEMA = "turing.project-compilation-links.v1"
SOURCE_REGION_INTEGRAL_SCHEMA = "turing.source-region-integral.v1"
DEFAULT_WORKER_RESERVATION_BYTES = 4 * 1024 ** 3
DEFAULT_WORKER_LIMIT_BYTES = 4 * 1024 ** 3
DEFAULT_UNIT_TIMEOUT_SECONDS = 5 * 60
DEFAULT_PROJECT_EXTRACTION_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "extraction_contracts"
    / "program_extraction.yaml"
)


class NativeInstallationRequiredError(RuntimeError):
    """A bootstrap unit claimed completion without a live native replacement."""

    def __init__(self, failures: Iterable[Mapping[str, Any]]):
        self.failures = tuple(dict(failure) for failure in failures)
        names = ", ".join(
            str(failure.get("qualified_name") or "<unknown>")
            for failure in self.failures
        )
        super().__init__(
            "compiler bootstrap completed Python units without verified native "
            f"installation: {names}"
        )


def compiler_toolchain_fingerprint() -> dict[str, Any]:
    """Hash the authored compiler implementation that gives a frozen plan meaning.

    Source and graph hashes prove that the planned *program* did not change.
    Meta-compilation also needs the complementary fact: the reducer, planner,
    SSA lowerer, and backend identities interpreting that graph are unchanged.
    The per-file ledger makes a stale result explainable without becoming a
    cache or assigning identity to runtime objects.
    """

    root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "extraction_contracts/program_extraction.yaml",
        "src/common/tensors/fused_ir.py",
        "src/common/tensors/source_realization.py",
        "src/common/tensors/topological_reducer.py",
        "src/common/tensors/accelerator_backends/aot_compile.py",
        "src/common/tensors/accelerator_backends/c_backend_llvm_ssa.py",
        "src/common/tensors/accelerator_backends/llvm_repository_ssa.py",
        "src/compiler/compilation_units.py",
        "src/compiler/control_source.py",
        "src/compiler/deployment_frame.py",
        "src/compiler/extraction_contract.py",
        "src/compiler/fortran_c_shell.py",
        "src/compiler/hierarchical_control.py",
        "src/compiler/hierarchical_plan.py",
        "src/compiler/ir_identities.py",
        "src/compiler/ir_indexing.py",
        "src/compiler/ir_sequence_tables.py",
        "src/compiler/loop_composer.py",
        "src/compiler/loop_ir.py",
        "src/compiler/precompile_ssa_validator.py",
        "src/compiler/precompile_to_ssa.py",
        "src/compiler/process_graph_fusion.py",
        "src/compiler/shell_reference_tables.py",
        "src/compiler/ssa_features.py",
        "src/compiler/ssa_fortran_backend.py",
        "src/compiler/string_table.py",
        "src/compiler/topology_catalogue.py",
        "src/transmogrifier/function_table.py",
        "src/transmogrifier/graph/graph_express2.py",
        "src/transmogrifier/graph/python_special_cases.py",
        "src/transmogrifier/ssa.py",
        "src/transmogrifier/ssa_registry.py",
        "src/transmogrifier/tensor_ssa_reference.py",
    )
    paths = {
        root / relative_path for relative_path in relative_paths
    }
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(
            (path for path in paths if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    ]
    payload = json.dumps(
        files, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": "turing.compiler-toolchain-fingerprint.v1",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "files": files,
    }


def changed_compiler_toolchain_files(
    pinned: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return changed files in the current frozen-graph interpretation set.

    A historical ledger may contain orchestration files no longer considered
    semantic. Extra historical entries therefore do not stale a graph; every
    file in the *current* interpretation set must still be present and equal.
    """

    current = compiler_toolchain_fingerprint()
    pinned_files = {
        str(item.get("path")): str(item.get("sha256"))
        for item in pinned.get("files") or ()
        if isinstance(item, Mapping) and item.get("path")
    }
    current_files = {
        str(item.get("path")): str(item.get("sha256"))
        for item in current.get("files") or ()
        if isinstance(item, Mapping) and item.get("path")
    }
    if not pinned_files:
        return ("<toolchain-ledger>",)
    return tuple(sorted(
        path for path, digest in current_files.items()
        if pinned_files.get(path) != digest
    ))


def _same_authored_parameter_surface(
    authored: Sequence[str], native: Sequence[str],
) -> bool:
    """Require exact unique source coverage without imposing ABI order."""

    authored_names = tuple(map(str, authored))
    native_names = tuple(map(str, native))
    return (
        len(authored_names) == len(set(authored_names))
        and len(native_names) == len(set(native_names))
        and len(authored_names) == len(native_names)
        and set(authored_names) == set(native_names)
    )


@dataclass(frozen=True)
class AuthoredCall:
    qualified_name: str
    line: int
    kind: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "qualified_name": self.qualified_name,
            "line": int(self.line),
            "kind": self.kind,
        }


@dataclass(frozen=True)
class ProjectCompilationProduct:
    """Loaded link table over independently published repository-SSA units."""

    root: Path
    manifest: Mapping[str, Any]
    links: Mapping[str, Mapping[str, Any]]

    def load_repository_ssa(self, qualified_name: str) -> tuple[Any, Any, Any]:
        try:
            link = self.links[str(qualified_name)]
        except KeyError as error:
            raise KeyError(
                f"project product has no compiled call {qualified_name!r}"
            ) from error
        with (self.root / str(link["artifact"])).open("rb") as stream:
            return pickle.load(stream)

    def install_callable(
        self,
        qualified_name: str,
        owner: Any,
        deployed_callable: Callable[..., Any],
        *,
        targeted_source_fallback: bool = False,
    ) -> Callable[..., Any]:
        """Install a verified native unit without surrendering authored source.

        Installation is deliberately stricter than loading.  The callable must
        carry the receipt produced by one of this product's native verifiers,
        the on-disk receipt and native artifacts must still have the recorded
        hashes, and the callable currently owned by ``owner`` must come from the
        exact authored source revision used to build the product.
        """

        name = str(qualified_name)
        if name not in self.links:
            raise KeyError(
                f"project product has no compiled call {name!r}"
            )
        link = dict(self.links[name])
        verification = getattr(
            deployed_callable, "__turing_native_verification__", None,
        )
        if not isinstance(verification, Mapping):
            raise ValueError(
                f"refusing to install unverified native callable {name!r}"
            )
        if (
            verification.get("status") != "verified"
            or str(verification.get("qualified_name") or "") != name
            or str(verification.get("source_sha256") or "")
            != str(self.manifest.get("source_sha256") or "")
        ):
            raise ValueError(
                f"native verification receipt does not match {name!r}"
            )
        probe_count = int(verification.get("probe_count") or 0)
        if probe_count <= 0:
            raise ValueError(
                f"native verification for {name!r} contains no behavior probes"
            )
        if verification.get("abi_kind") in {
            "scalar", "sequence", "scalar-record",
        } and (
            int(verification.get("native_probe_count") or 0) != probe_count
            or int(verification.get("fallback_probe_count") or 0) != 0
        ):
            raise ValueError(
                f"native verification for {name!r} did not prove every probe "
                "through the native route"
            )
        if verification.get("abi_kind") == "record-return" and (
            int(verification.get("native_probe_count") or 0) <= 0
            or int(verification.get("native_probe_count") or 0)
            + int(verification.get("fallback_probe_count") or 0)
            != probe_count
        ):
            raise ValueError(
                f"native verification for {name!r} did not account for "
                "every record-return probe or prove a native route"
            )
        api_path = self.root / str(link.get("native_api") or "")
        library_path = self.root / str(link.get("native_library") or "")
        if not api_path.is_file() or not library_path.is_file():
            raise ValueError(f"native artifacts for {name!r} are incomplete")
        current_hashes = {
            "api_sha256": hashlib.sha256(api_path.read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
        }
        for field, current in current_hashes.items():
            if str(verification.get(field) or "") != current:
                raise ValueError(
                    f"native artifact {field.removesuffix('_sha256')!r} for "
                    f"{name!r} changed after verification"
                )
        receipt_path = library_path.parent / "native-verification.json"
        if not receipt_path.is_file():
            raise ValueError(f"native verification receipt for {name!r} is missing")
        persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
        if persisted != dict(verification):
            raise ValueError(
                f"native verification receipt for {name!r} does not match "
                "the loaded callable"
            )

        attribute = name.rsplit(".", 1)[-1]
        import inspect

        descriptor = inspect.getattr_static(owner, attribute)
        property_descriptor = (
            descriptor if isinstance(descriptor, property) else None
        )
        authored = (
            property_descriptor.fget
            if property_descriptor is not None else getattr(owner, attribute)
        )
        if authored is None:
            raise ValueError(f"property {name!r} has no authored getter")
        authored = getattr(
            authored, "__turing_authored_source_callable__", authored,
        )

        authored_path_text = inspect.getsourcefile(authored)
        if not authored_path_text:
            raise ValueError(
                f"cannot establish authored source revision for {name!r}"
            )
        authored_path = Path(authored_path_text).resolve()
        expected_source_hash = str(self.manifest.get("source_sha256") or "")
        if not expected_source_hash or not authored_path.is_file():
            raise ValueError(
                f"authored source for {name!r} is unavailable for revision proof"
            )
        current_source_hash = hashlib.sha256(
            authored_path.read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
        if current_source_hash != expected_source_hash:
            raise ValueError(
                f"authored source for {name!r} changed after compilation"
            )
        from ..common.tensors.source_realization import (
            deployed_with_authored_fallback,
            install_authored_deployment,
        )

        if property_descriptor is not None:
            installed = deployed_with_authored_fallback(
                authored, deployed_callable,
                identity=name,
                targeted=bool(targeted_source_fallback),
            )
            setattr(owner, attribute, property(
                installed,
                property_descriptor.fset,
                property_descriptor.fdel,
                property_descriptor.__doc__,
            ))
            return installed

        return install_authored_deployment(
            owner, attribute, deployed_callable,
            identity=name,
            targeted=bool(targeted_source_fallback),
        )

    def verify_native_scalar_callable(
        self,
        qualified_name: str,
        authored_callable: Callable[..., Any],
        probes: Sequence[Sequence[Any] | Mapping[str, Any]],
        *,
        activation_adapter: str | None = None,
        ignored_source_parameters: Iterable[str] = (),
        probe_factory: str | None = None,
        native_result_codec: str | None = None,
        expected_probe_results: Sequence[Any] | None = None,
    ) -> Callable[..., Any]:
        """Load a scalar native unit only after ABI and behavior agree.

        This intentionally supports the narrow scalar surface the emitted API
        can describe completely today.  Arrays, records, workspaces, invented
        inputs, missing source parameters, and ambiguous returns are rejected
        rather than guessed.  Successful probes are persisted beside the unit
        and tied to the source, API, and library hashes.
        """

        import ctypes
        import inspect
        from .compiled_program_api import load_api

        if not probes:
            raise ValueError("native verification requires at least one probe")
        if (
            expected_probe_results is not None
            and len(expected_probe_results) != len(probes)
        ):
            raise ValueError(
                "expected native probe results must align one-to-one with probes"
            )

        name = str(qualified_name)
        try:
            link = dict(self.links[name])
        except KeyError as error:
            raise KeyError(f"project product has no compiled call {name!r}") from error
        required = ("native_api", "native_library", "native_entrypoint")
        missing = tuple(item for item in required if not link.get(item))
        if missing:
            raise ValueError(f"native unit {name!r} is missing {missing!r}")
        api_path = self.root / str(link["native_api"])
        library_path = self.root / str(link["native_library"])
        descriptor = load_api(api_path)
        entry_name = str(link["native_entrypoint"])
        entry = next((
            item for item in descriptor.get("entry_points", ())
            if str(item.get("name")) == entry_name
        ), None)
        if entry is None:
            raise ValueError(f"native API does not declare {entry_name!r}")
        parameters = tuple(entry.get("parameters") or ())
        if any(parameter.get("shape") or parameter.get("extents")
               or parameter.get("extent") for parameter in parameters):
            raise ValueError("scalar verifier refuses array-shaped native ABI")
        inputs = tuple(
            parameter for parameter in parameters
            if parameter.get("role") in {"input", "inout"}
        )
        outputs = tuple(
            parameter for parameter in parameters
            if parameter.get("role") in {"output", "inout"}
        )
        if len(outputs) != 1 or outputs[0].get("role") != "output":
            raise ValueError("scalar verifier requires exactly one distinct output")
        authored_descriptor = getattr(
            authored_callable, "__turing_authored_source_callable__",
            authored_callable,
        )
        authored = (
            authored_descriptor.fget
            if isinstance(authored_descriptor, property)
            else authored_descriptor
        )
        if authored is None:
            raise ValueError("native verifier requires a property getter")
        signature = inspect.signature(authored)
        source_parameters = tuple(
            parameter.name for parameter in signature.parameters.values()
            if parameter.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        )
        native_source_parameters = tuple(
            str(parameter.get("source_name") or "") for parameter in inputs
        )
        if (
            len(native_source_parameters) != len(set(native_source_parameters))
            or not set(native_source_parameters) <= set(source_parameters)
        ):
            raise ValueError(
                "native ABI does not correlate uniquely to authored parameters: "
                f"authored={source_parameters!r}, native={native_source_parameters!r}"
            )
        omitted_source_parameters = tuple(
            parameter for parameter in source_parameters
            if parameter not in native_source_parameters
        )
        ignored = tuple(dict.fromkeys(map(str, ignored_source_parameters)))
        if set(ignored) != set(omitted_source_parameters):
            raise ValueError(
                "native ABI omitted authored parameters without an exact "
                "unused-source proof: "
                f"omitted={omitted_source_parameters!r}, proven={ignored!r}"
            )
        ctypes_types = {
            "c_int32": ctypes.c_int32,
            "c_int64": ctypes.c_int64,
            "c_float": ctypes.c_float,
            "c_double": ctypes.c_double,
            "c_bool": ctypes.c_bool,
            "c_uint8": ctypes.c_uint8,
        }
        try:
            argument_types = [
                ctypes_types[str(parameter["ctypes"])] for parameter in inputs
            ]
            output_type = ctypes_types[str(outputs[0]["ctypes"])]
        except KeyError as error:
            raise ValueError(f"unsupported scalar native type {error.args[0]!r}") from error
        runtime_handles = []
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            for dependency in (
                descriptor.get("metadata", {}).get("runtime_dependencies", ())
            ):
                dependency_path = Path(str(dependency.get("path") or ""))
                if dependency_path.is_file():
                    parent = str(dependency_path.parent.resolve())
                    if parent not in {str(handle) for handle in runtime_handles}:
                        runtime_handles.append(os.add_dll_directory(parent))
        library = ctypes.CDLL(str(library_path.resolve()))
        native = getattr(library, str(entry.get("symbol") or entry_name))
        native.argtypes = [*argument_types, ctypes.POINTER(output_type)]
        native.restype = None

        route_counts = {"native": 0, "fallback": 0}
        result_codecs = {
            None: lambda value: value,
            "unsigned-c_int32-v1": lambda value: int(value) & 0xFFFFFFFF,
            "unsigned-c_int64-v1": lambda value: int(value) & 0xFFFFFFFFFFFFFFFF,
        }
        try:
            decode_native_result = result_codecs[native_result_codec]
        except KeyError as error:
            raise ValueError(
                f"unknown native scalar result codec {native_result_codec!r}"
            ) from error

        def deployed(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            native_inputs = []
            for source_name, native_type in zip(
                native_source_parameters, argument_types, strict=True,
            ):
                value = bound.arguments[source_name]
                converted = native_type(value)
                if isinstance(value, int) and not isinstance(value, bool):
                    if int(converted.value) != int(value):
                        route_counts["fallback"] += 1
                        return authored(*args, **kwargs)
                native_inputs.append(converted.value)
            result = output_type()
            native(*native_inputs, ctypes.byref(result))
            route_counts["native"] += 1
            return decode_native_result(result.value)

        probe_records = []
        for probe_index, probe in enumerate(probes):
            if isinstance(probe, Mapping):
                arguments, keywords = (), dict(probe)
            else:
                arguments, keywords = tuple(probe), {}
            expected = (
                authored(*arguments, **keywords)
                if expected_probe_results is None else
                expected_probe_results[probe_index]
            )
            actual = deployed(*arguments, **keywords)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"native equivalence failed for {name}{arguments!r}: "
                    f"authored={expected!r}, native={actual!r}"
                )
            probe_records.append({
                "arguments": repr(arguments),
                "keywords": repr(keywords),
                "result": repr(actual),
            })
        verification = {
            "schema": "turing.native-callable-verification.v1",
            "qualified_name": name,
            "abi_kind": "scalar",
            "source_sha256": str(self.manifest.get("source_sha256") or ""),
            "api_sha256": hashlib.sha256(api_path.read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
            "entrypoint": entry_name,
            "authored_parameters": list(source_parameters),
            "native_source_parameters": list(native_source_parameters),
            "ignored_source_parameters": list(omitted_source_parameters),
            "native_result_codec": native_result_codec,
            "probe_count": len(probe_records),
            "native_probe_count": int(route_counts["native"]),
            "fallback_probe_count": int(route_counts["fallback"]),
            "probes": probe_records,
            "status": "verified",
            **({
                "activation_adapter": str(activation_adapter),
            } if activation_adapter is not None else {}),
            **({"probe_factory": str(probe_factory)} if probe_factory else {}),
        }
        receipt_path = library_path.parent / "native-verification.json"
        _atomic_json(receipt_path, verification)
        deployed.__turing_native_verification__ = verification
        deployed.__turing_native_library__ = library
        deployed.__turing_native_runtime_handles__ = tuple(runtime_handles)
        return deployed

    def verify_native_sequence_callable(
        self,
        qualified_name: str,
        authored_callable: Callable[..., Any],
        probes: Sequence[Sequence[Any] | Mapping[str, Any]],
        *,
        capacity: int = 64,
        native_precondition: Callable[..., bool] | None = None,
    ) -> Callable[..., Any]:
        """Load a one-column sequence unit after exact ABI/behavior proof.

        The wrapper is driven entirely by the emitted sequence descriptor,
        return surface, and runtime binding.  Anonymous SSA ids and generated
        argument order are never interpreted by convention.  Unsupported or
        incomplete aggregate ABIs are rejected before the library is called.
        """

        import copy
        import ctypes
        import inspect
        import itertools
        from collections.abc import Iterator
        from .compiled_program_api import load_api

        if not probes:
            raise ValueError("native verification requires at least one probe")
        if int(capacity) <= 0:
            raise ValueError("sequence verification capacity must be positive")
        name = str(qualified_name)
        try:
            link = dict(self.links[name])
        except KeyError as error:
            raise KeyError(f"project product has no compiled call {name!r}") from error
        required = ("native_api", "native_library", "native_entrypoint")
        missing = tuple(item for item in required if not link.get(item))
        if missing:
            raise ValueError(f"native unit {name!r} is missing {missing!r}")
        api_path = self.root / str(link["native_api"])
        library_path = self.root / str(link["native_library"])
        descriptor = load_api(api_path)
        entry_name = str(link["native_entrypoint"])
        entry = next((
            item for item in descriptor.get("entry_points", ())
            if str(item.get("name")) == entry_name
        ), None)
        if entry is None:
            raise ValueError(f"native API does not declare {entry_name!r}")
        metadata = dict(descriptor.get("metadata") or {})
        sequence_descriptors = tuple(
            (metadata.get("sequence_tables") or {}).get(entry_name, ())
        )
        return_surfaces = tuple(
            (metadata.get("sequence_output_surfaces") or {}).get(entry_name, ())
        )
        runtime_bindings = tuple(
            (metadata.get("sequence_runtime_bindings") or {}).get(entry_name, ())
        )
        if len(return_surfaces) != 1:
            raise ValueError(
                "sequence verifier requires exactly one returned sequence"
            )
        descriptors_by_id = {
            int(item["sequence_id"]): dict(item)
            for item in sequence_descriptors
        }
        bindings_by_id = {
            int(item["sequence_id"]): dict(item)
            for item in runtime_bindings
        }
        surface = dict(return_surfaces[0])
        sequence_id = int(surface.get("sequence_id", -1))
        sequence = descriptors_by_id.get(sequence_id)
        binding = bindings_by_id.get(sequence_id)
        if sequence is None or binding is None:
            raise ValueError(
                "returned sequence lacks descriptor or runtime binding"
            )
        if set(descriptors_by_id) != set(bindings_by_id):
            raise ValueError(
                "sequence runtime bindings do not cover every descriptor: "
                f"descriptors={sorted(descriptors_by_id)!r}, "
                f"bindings={sorted(bindings_by_id)!r}"
            )
        if len(sequence.get("column_value_ids") or ()) != 1:
            raise ValueError(
                "returned bytes/list materialization requires one resident column"
            )
        materialization = str(surface.get("materialization_identity") or "")
        factories: dict[str, Callable[[Iterable[Any]], Any]] = {
            "builtins.bytes": bytes,
            "builtins.bytearray": bytearray,
            "builtins.list": list,
            "builtins.tuple": tuple,
        }
        if materialization not in factories:
            raise ValueError(
                f"unsupported sequence materialization {materialization!r}"
            )

        parameters = tuple(entry.get("parameters") or ())
        parameter_by_name = {
            str(parameter.get("name")): parameter for parameter in parameters
        }
        storage_names: set[str] = set()
        extent_policies: dict[str, str] = {}
        for runtime_binding in bindings_by_id.values():
            storage_names.update(map(
                str, runtime_binding.get("column_parameters") or (),
            ))
            storage_names.add(str(runtime_binding["length_parameter"]))
            storage_names.add(str(runtime_binding["capacity_parameter"]))
            if runtime_binding.get("status_parameter"):
                storage_names.add(str(runtime_binding["status_parameter"]))
            for key, value in dict(
                runtime_binding.get("extent_parameters") or {}
            ).items():
                key, value = str(key), str(value)
                previous = extent_policies.setdefault(key, value)
                if previous != value:
                    raise ValueError(
                        f"conflicting extent policy for {key!r}"
                    )
        authored_descriptor = getattr(
            authored_callable, "__turing_authored_source_callable__",
            authored_callable,
        )
        authored = (
            authored_descriptor.fget
            if isinstance(authored_descriptor, property)
            else authored_descriptor
        )
        if authored is None:
            raise ValueError("native verifier requires a property getter")
        signature = inspect.signature(authored)
        authored_parameters = tuple(
            parameter.name for parameter in signature.parameters.values()
            if parameter.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        )
        def source_root(source_name: str) -> str:
            return str(source_name).split(".", 1)[0]

        source_inputs = tuple(
            parameter for parameter in parameters
            if source_root(str(parameter.get("source_name") or ""))
            in authored_parameters
        )
        workspace_names = {
            str(parameter["name"])
            for parameter in parameters
            if str(parameter.get("role") or "") == "workspace"
        }
        described_names = {
            *(str(parameter["name"]) for parameter in source_inputs),
            *storage_names,
            *extent_policies,
            *workspace_names,
        }
        abi_names = {str(parameter.get("name")) for parameter in parameters}
        if described_names != abi_names:
            raise ValueError(
                "sequence runtime binding does not cover the exact native ABI: "
                f"missing={sorted(abi_names - described_names)!r}, "
                f"extra={sorted(described_names - abi_names)!r}"
            )
        native_source_parameters = tuple(dict.fromkeys(
            source_root(str(parameter["source_name"]))
            for parameter in source_inputs
        ))
        if not _same_authored_parameter_surface(
            authored_parameters, native_source_parameters,
        ):
            raise ValueError(
                "native ABI does not exactly match authored parameters: "
                f"authored={authored_parameters!r}, "
                f"native={native_source_parameters!r}"
            )
        ctypes_types = {
            "c_int32": ctypes.c_int32,
            "c_int64": ctypes.c_int64,
            "c_float": ctypes.c_float,
            "c_double": ctypes.c_double,
            "c_bool": ctypes.c_bool,
            "c_uint8": ctypes.c_uint8,
        }
        try:
            native_types = {
                parameter_name: ctypes_types[str(parameter["ctypes"])]
                for parameter_name, parameter in parameter_by_name.items()
            }
        except KeyError as error:
            raise ValueError(
                f"unsupported sequence native type {error.args[0]!r}"
            ) from error
        runtime_handles = []
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            parents: set[str] = set()
            for dependency in metadata.get("runtime_dependencies", ()):
                dependency_path = Path(str(dependency.get("path") or ""))
                if dependency_path.is_file():
                    parent = str(dependency_path.parent.resolve())
                    if parent not in parents:
                        parents.add(parent)
                        runtime_handles.append(os.add_dll_directory(parent))
        library = ctypes.CDLL(str(library_path.resolve()))
        packed_contract = dict(
            (metadata.get("packed_entrypoints") or {}).get(entry_name) or {}
        )
        if packed_contract:
            if (
                packed_contract.get("schema")
                != "turing.packed-pointer-array.v1"
                or int(packed_contract.get("parameter_count", -1))
                != len(parameters)
                or not packed_contract.get("symbol")
            ):
                raise ValueError("invalid packed native entrypoint contract")
            native = getattr(library, str(packed_contract["symbol"]))
            native.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
            ]
            native.restype = ctypes.c_int
        else:
            native = getattr(library, str(entry.get("symbol") or entry_name))
            native.argtypes = [
                (
                    ctypes.POINTER(native_types[str(parameter["name"])])
                    if str(parameter.get("passing")) == "reference"
                    else native_types[str(parameter["name"])]
                )
                for parameter in parameters
            ]
            native.restype = None

        sequence_runtime: dict[int, dict[str, Any]] = {}
        parameter_sequence_ids: dict[str, int] = {}
        for resident_id, runtime_binding in bindings_by_id.items():
            column_names = tuple(map(
                str, runtime_binding.get("column_parameters") or (),
            ))
            length_name = str(runtime_binding["length_parameter"])
            capacity_name = str(runtime_binding["capacity_parameter"])
            status_name = runtime_binding.get("status_parameter")
            source_names = tuple(map(
                str, descriptors_by_id[resident_id].get("source_names") or (),
            ))
            source_name = next((
                candidate for candidate in source_names
                if source_root(candidate) in authored_parameters
            ), None)
            if source_name is None:
                source_name = next((
                    str(parameter_by_name[column].get("source_name") or "")
                    for column in column_names
                    if source_root(str(
                        parameter_by_name[column].get("source_name") or ""
                    )) in authored_parameters
                ), None)
            record = {
                "column_names": column_names,
                "length_name": length_name,
                "capacity_name": capacity_name,
                "status_name": (
                    None if status_name is None else str(status_name)
                ),
                "source_name": source_name,
                "source_transform": descriptors_by_id[resident_id].get(
                    "source_transform"
                ),
                "exhausted_values": {
                    int(value)
                    for status, value in dict(
                        runtime_binding.get("status_values") or {}
                    ).items()
                    if "capacity" in str(status).casefold()
                    or "exhaust" in str(status).casefold()
                },
            }
            sequence_runtime[resident_id] = record
            for parameter_name in filter(None, (
                *record["column_names"], length_name, capacity_name,
                record["status_name"],
            )):
                parameter_sequence_ids[str(parameter_name)] = resident_id
        validation_contracts = tuple(
            (metadata.get("validation_contracts") or {}).get(entry_name, ())
        )
        source_value_ids: dict[int, str] = {}
        for parameter in source_inputs:
            parameter_name = str(parameter["name"])
            if not parameter_name.startswith("t"):
                raise ValueError(
                    "validation contract requires explicit SSA input identity"
                )
            try:
                source_value_ids[int(parameter_name[1:])] = str(
                    parameter["source_name"]
                )
            except ValueError as error:
                raise ValueError(
                    "validation contract has a non-SSA native parameter"
                ) from error
        validation_operators = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: a / b,
            "neg": lambda a: -a,
            "lt": lambda a, b: a < b,
            "le": lambda a, b: a <= b,
            "gt": lambda a, b: a > b,
            "ge": lambda a, b: a >= b,
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
            "and": lambda a, b: bool(a) and bool(b),
            "or": lambda a, b: bool(a) or bool(b),
            "not": lambda a: not bool(a),
            "bitand": lambda a, b: int(a) & int(b),
            "bitor": lambda a, b: int(a) | int(b),
            "bitxor": lambda a, b: int(a) ^ int(b),
            "shl": lambda a, b: int(a) << int(b),
            "shr": lambda a, b: int(a) >> int(b),
            "int": lambda a: int(a),
            "float": lambda a: float(a),
            "bool": lambda a: bool(a),
        }

        def expression_requirements(expression: Mapping[str, Any]) -> None:
            operation = str(expression.get("op") or "")
            if operation == "value":
                value_id = int(expression.get("value_id", -1))
                if value_id not in source_value_ids:
                    raise ValueError(
                        "validation predicate depends on non-source SSA value "
                        f"{value_id}"
                    )
                return
            if operation == "const":
                return
            if operation not in validation_operators:
                raise ValueError(
                    f"unsupported validation predicate operation {operation!r}"
                )
            for operand in expression.get("operands") or ():
                expression_requirements(dict(operand))

        for contract in validation_contracts:
            expression = contract.get("predicate_expression")
            if not isinstance(expression, Mapping):
                if native_precondition is None:
                    raise ValueError(
                        "sequence verifier requires a native precondition "
                        "while a validation contract lacks a structured "
                        "predicate"
                    )
                continue
            expression_requirements(expression)

        def evaluate_expression(
            expression: Mapping[str, Any], values: Mapping[int, Any],
        ) -> Any:
            operation = str(expression.get("op") or "")
            if operation == "value":
                return values[int(expression["value_id"])]
            if operation == "const":
                return expression.get("literal")
            operands = [
                evaluate_expression(dict(operand), values)
                for operand in expression.get("operands") or ()
            ]
            return validation_operators[operation](*operands)

        route_counts: dict[str, Any] = {
            "invoked": 0, "native": 0, "fallback": 0, "last_fallback": None,
        }

        def deployed(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()

            def fallback(reason: str | None = None) -> Any:
                route_counts["fallback"] += 1
                route_counts["last_fallback"] = reason
                return authored(*bound.args, **bound.kwargs)

            if native_precondition is not None and not bool(
                native_precondition(*bound.args, **bound.kwargs)
            ):
                return fallback("native precondition")

            def resolve_source(source_name: str) -> Any:
                parts = str(source_name).split(".")
                value = bound.arguments[parts[0]]
                for attribute in parts[1:]:
                    value = getattr(value, attribute)
                return value

            validation_values = {
                value_id: resolve_source(source_name)
                for value_id, source_name in source_value_ids.items()
            }
            for contract in validation_contracts:
                expression = contract.get("predicate_expression")
                if not isinstance(expression, Mapping):
                    continue
                predicate = evaluate_expression(dict(expression), validation_values)
                if bool(predicate) is not bool(contract.get("expect_true", True)):
                    return fallback()
            sequence_source_parameter_names = {
                str(column_name)
                for record in sequence_runtime.values()
                if record.get("source_name") is not None
                for column_name in record.get("column_names") or ()
            }
            source_values: dict[str, Any] = {}
            materialized_sources: dict[str, list[Any]] = {}
            for parameter in source_inputs:
                parameter_name = str(parameter["name"])
                if parameter_name in sequence_source_parameter_names:
                    continue
                source_name = str(parameter["source_name"])
                try:
                    source_value = resolve_source(source_name)
                except (AttributeError, KeyError):
                    return fallback()
                source_transform = str(
                    parameter.get("source_transform") or ""
                )
                if source_transform == "utf8_length":
                    if not isinstance(source_value, str):
                        return fallback()
                    source_value = len(source_value.encode("utf-8"))
                elif source_transform == "sequence_length":
                    try:
                        source_value = len(source_value)
                    except TypeError:
                        return fallback()
                elif source_transform == "materialized_length":
                    if source_name not in materialized_sources:
                        try:
                            materialized_sources[source_name] = list(
                                source_value
                            )
                        except TypeError:
                            return fallback()
                        bound.arguments[source_name] = materialized_sources[
                            source_name
                        ]
                    source_value = len(materialized_sources[source_name])
                elif source_transform:
                    return fallback()
                if str(parameter.get("passing")) == "reference":
                    if isinstance(source_value, str):
                        source_value = source_value.encode("utf-8")
                    try:
                        source_items = list(source_value)
                    except TypeError:
                        return fallback(
                            f"source span {source_name} is not iterable"
                        )
                    array_type = native_types[parameter_name]
                    storage = (array_type * max(1, len(source_items)))()
                    try:
                        for index, item in enumerate(source_items):
                            storage[index] = array_type(item).value
                    except (TypeError, ValueError, OverflowError):
                        return fallback(
                            f"source span {source_name} cannot be encoded as "
                            f"{parameter.get('ctypes')}"
                        )
                    source_values[parameter_name] = storage
                    continue
                try:
                    converted = native_types[parameter_name](source_value)
                except (TypeError, ValueError, OverflowError):
                    return fallback()
                if isinstance(source_value, int) and not isinstance(source_value, bool):
                    if int(converted.value) != int(source_value):
                        return fallback()
                source_values[parameter_name] = converted.value
            residents: dict[int, dict[str, Any]] = {}

            def native_sequence_cell(
                cell: Any, column_type: type[ctypes._SimpleCData],
            ) -> Any:
                """Encode one authored sequence cell by repository semantics.

                Runtime strings and compile-time string literals meet in the
                repository's universal signed FNV-1a token namespace.  This is
                not a verifier convention: ``string_token`` is the exact
                lowering used by keyed string tables and SSA string constants
                in every backend.  Record-local vocabularies remain separate
                physical contracts and are not guessed here.
                """

                if isinstance(cell, str):
                    from .string_table import string_token

                    cell = string_token(cell)
                return column_type(cell).value

            for resident_id, runtime in sequence_runtime.items():
                column_names = tuple(runtime.get("column_names") or ())
                source_name = runtime.get("source_name")
                try:
                    source_sequence = (
                        None if source_name is None
                        else resolve_source(str(source_name))
                    )
                except (AttributeError, KeyError):
                    return fallback()
                if runtime.get("source_transform") == "utf8":
                    if not isinstance(source_sequence, str):
                        return fallback()
                    source_sequence = source_sequence.encode("utf-8")
                elif runtime.get("source_transform") in {
                    "row_count", "join_bytes",
                }:
                    source_key = str(source_name)
                    if source_key not in materialized_sources:
                        try:
                            materialized_sources[source_key] = list(
                                source_sequence
                            )
                        except TypeError:
                            return fallback()
                        # A generator has now been consumed.  Preserve authored
                        # fallback semantics by replacing its bound argument
                        # with the one exact materialization shared by every
                        # ABI view.
                        bound.arguments[source_key] = materialized_sources[
                            source_key
                        ]
                    materialized = materialized_sources[source_key]
                    if runtime.get("source_transform") == "row_count":
                        source_sequence = [0] * len(materialized)
                    else:
                        try:
                            source_sequence = b"".join(materialized)
                        except (TypeError, ValueError):
                            return fallback()
                elif runtime.get("source_transform") not in {None, ""}:
                    return fallback()
                if source_sequence is not None:
                    try:
                        source_items = list(source_sequence)
                    except TypeError:
                        return fallback()
                    if len(source_items) > int(capacity):
                        return fallback()
                else:
                    source_items = []
                columns = []
                for column_index, column_name in enumerate(column_names):
                    column_type = native_types[str(column_name)]
                    column = (column_type * int(capacity))()
                    try:
                        for item_index, item in enumerate(source_items):
                            cell = (
                                item[column_index]
                                if len(column_names) > 1 else item
                            )
                            column[item_index] = native_sequence_cell(
                                cell, column_type,
                            )
                    except (IndexError, TypeError, ValueError, OverflowError):
                        return fallback(
                            "source sequence "
                            f"{source_name or resident_id} cannot populate "
                            f"physical column {column_index}"
                        )
                    columns.append(column)
                length_name = str(runtime["length_name"])
                status_name = runtime.get("status_name")
                residents[resident_id] = {
                    "columns": tuple(columns),
                    "length": native_types[length_name](len(source_items)),
                    "status": (
                        native_types[str(status_name)](0)
                        if status_name is not None else None
                    ),
                }
            call_arguments = []
            workspaces: dict[str, Any] = {}
            for parameter in parameters:
                parameter_name = str(parameter["name"])
                native_type = native_types[parameter_name]
                if parameter_name in extent_policies:
                    policy = extent_policies[parameter_name]
                    if policy == "unit":
                        value = 1
                    elif policy == "capacity":
                        value = int(capacity)
                    elif policy.startswith("source_length:"):
                        try:
                            value = len(resolve_source(policy.split(":", 1)[1]))
                        except (AttributeError, KeyError, TypeError):
                            return fallback()
                    else:
                        return fallback()
                    call_arguments.append(native_type(value).value)
                elif parameter_name in source_values:
                    call_arguments.append(source_values[parameter_name])
                elif parameter_name in parameter_sequence_ids:
                    resident_id = parameter_sequence_ids[parameter_name]
                    runtime = sequence_runtime[resident_id]
                    resident = residents[resident_id]
                    if parameter_name in runtime.get("column_names", ()):
                        column_index = tuple(runtime["column_names"]).index(
                            parameter_name
                        )
                        call_arguments.append(resident["columns"][column_index])
                    elif parameter_name == runtime["length_name"]:
                        call_arguments.append(ctypes.byref(resident["length"]))
                    elif parameter_name == runtime["capacity_name"]:
                        call_arguments.append(native_type(int(capacity)).value)
                    elif parameter_name == runtime.get("status_name"):
                        call_arguments.append(ctypes.byref(resident["status"]))
                    else:  # pragma: no cover - exact map built above
                        raise AssertionError(parameter_name)
                elif parameter_name in workspace_names:
                    if str(parameter.get("passing")) == "value":
                        workspace = native_type(int(capacity))
                        workspaces[parameter_name] = workspace
                        call_arguments.append(workspace.value)
                    elif tuple(parameter.get("shape") or ()) == (1,):
                        workspace = native_type(0)
                        workspaces[parameter_name] = workspace
                        call_arguments.append(ctypes.byref(workspace))
                    else:
                        workspace = (native_type * int(capacity))()
                        workspaces[parameter_name] = workspace
                        call_arguments.append(workspace)
                else:  # pragma: no cover - exact coverage checked above
                    raise AssertionError(parameter_name)
            if packed_contract:
                packed_keepalive = []
                packed_addresses = []
                for parameter, argument in zip(
                    parameters, call_arguments, strict=True,
                ):
                    parameter_name = str(parameter["name"])
                    if str(parameter.get("passing")) == "reference":
                        address = ctypes.cast(
                            argument, ctypes.c_void_p
                        ).value
                    else:
                        cell = native_types[parameter_name](argument)
                        packed_keepalive.append(cell)
                        address = ctypes.addressof(cell)
                    if address is None:
                        return fallback(
                            f"packed ABI parameter {parameter_name} is null"
                        )
                    packed_addresses.append(address)
                packed_arguments = (ctypes.c_void_p * len(parameters))(
                    *packed_addresses
                )
                packed_keepalive.append(packed_arguments)
                if int(native(packed_arguments, len(parameters))) != 1:
                    return fallback("packed ABI rejected its argument frame")
            else:
                native(*call_arguments)
            route_counts["invoked"] += 1
            route_counts["last_sequence_lengths"] = {
                int(resident_id): int(resident["length"].value)
                for resident_id, resident in residents.items()
                if int(resident["length"].value) != 0
            }
            for resident_id, runtime in sequence_runtime.items():
                status = residents[resident_id]["status"]
                if (
                    status is not None
                    and int(status.value) in runtime["exhausted_values"]
                ):
                    return fallback(
                        "resident sequence "
                        f"{resident_id} reported capacity status "
                        f"{int(status.value)}"
                    )
            returned = residents[sequence_id]
            columns = returned["columns"]
            if len(columns) != 1:
                return fallback("returned sequence does not have one column")
            column = columns[0]
            logical_length = int(returned["length"].value)
            if logical_length < 0 or logical_length > int(capacity):
                return fallback(
                    f"returned sequence length {logical_length} is outside "
                    f"capacity {int(capacity)}"
                )
            values = [column[index] for index in range(logical_length)]
            if materialization in {"builtins.bytes", "builtins.bytearray"}:
                integral = [int(value) for value in values]
                if any(
                    float(value) != float(integer) or not 0 <= integer <= 255
                    for value, integer in zip(values, integral, strict=True)
                ):
                    return fallback(
                        "returned byte sequence contains a non-byte value"
                    )
                values = integral
            result = factories[materialization](values)
            route_counts["native"] += 1
            return result

        def independent_probe_values(
            arguments: tuple[Any, ...], keywords: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any], tuple[Any, ...], dict[str, Any]]:
            """Give authored and native routes equivalent one-shot inputs."""

            try:
                return (
                    copy.deepcopy(arguments), copy.deepcopy(keywords),
                    copy.deepcopy(arguments), copy.deepcopy(keywords),
                )
            except (TypeError, ValueError):
                pass

            def fork(value: Any) -> tuple[Any, Any]:
                if isinstance(value, Iterator):
                    return tuple(itertools.tee(value, 2))
                if isinstance(value, tuple):
                    pairs = tuple(fork(item) for item in value)
                    return (
                        tuple(pair[0] for pair in pairs),
                        tuple(pair[1] for pair in pairs),
                    )
                if isinstance(value, list):
                    pairs = tuple(fork(item) for item in value)
                    return (
                        [pair[0] for pair in pairs],
                        [pair[1] for pair in pairs],
                    )
                if isinstance(value, dict):
                    pairs = {key: fork(item) for key, item in value.items()}
                    return (
                        {key: pair[0] for key, pair in pairs.items()},
                        {key: pair[1] for key, pair in pairs.items()},
                    )
                return value, value

            expected_arguments, actual_arguments = fork(arguments)
            expected_keywords, actual_keywords = fork(keywords)
            return (
                tuple(expected_arguments), dict(expected_keywords),
                tuple(actual_arguments), dict(actual_keywords),
            )

        probe_records = []
        for probe in probes:
            if isinstance(probe, Mapping):
                arguments, keywords = (), dict(probe)
            else:
                arguments, keywords = tuple(probe), {}
            (
                expected_arguments, expected_keywords,
                actual_arguments, actual_keywords,
            ) = independent_probe_values(arguments, keywords)
            try:
                expected = authored(*expected_arguments, **expected_keywords)
            except Exception as expected_error:
                try:
                    deployed(*actual_arguments, **actual_keywords)
                except type(expected_error) as actual_error:
                    if str(actual_error) != str(expected_error):
                        raise ValueError(
                            f"native fallback exception mismatch for {name}"
                            f"{arguments!r}: authored={expected_error!r}, "
                            f"deployed={actual_error!r}"
                        ) from actual_error
                    probe_records.append({
                        "arguments": repr(arguments),
                        "keywords": repr(keywords),
                        "exception": repr(actual_error),
                    })
                    continue
                raise ValueError(
                    f"native deployment did not preserve exception for "
                    f"{name}{arguments!r}: {expected_error!r}"
                ) from expected_error
            native_before = route_counts["native"]
            actual = deployed(*actual_arguments, **actual_keywords)
            if route_counts["native"] == native_before:
                raise ValueError(
                    f"native equivalence probe for {name}{arguments!r} "
                    "used authored fallback instead of accepting the native "
                    f"result after {route_counts['invoked']} invocation(s): "
                    f"{route_counts.get('last_fallback') or 'precondition'}"
                )
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"native equivalence failed for {name}{arguments!r}: "
                    f"authored={expected!r}, native={actual!r}, "
                    "nonzero_sequence_lengths="
                    f"{route_counts.get('last_sequence_lengths', {})!r}"
                )
            probe_records.append({
                "arguments": repr(arguments),
                "keywords": repr(keywords),
                "result": repr(actual),
            })
        verification = {
            "schema": "turing.native-callable-verification.v1",
            "qualified_name": name,
            "abi_kind": "sequence",
            "sequence_id": sequence_id,
            "materialization_identity": materialization,
            "capacity": int(capacity),
            "source_sha256": str(self.manifest.get("source_sha256") or ""),
            "api_sha256": hashlib.sha256(api_path.read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
            "entrypoint": entry_name,
            "authored_parameters": list(authored_parameters),
            "probe_count": len(probe_records),
            "native_probe_count": int(route_counts["native"]),
            "fallback_probe_count": int(route_counts["fallback"]),
            "probes": probe_records,
            "status": "verified",
        }
        receipt_path = library_path.parent / "native-verification.json"
        _atomic_json(receipt_path, verification)
        deployed.__turing_native_verification__ = verification
        deployed.__turing_native_library__ = library
        deployed.__turing_native_runtime_handles__ = tuple(runtime_handles)
        return deployed

    def verify_native_record_return_callable(
        self,
        qualified_name: str,
        authored_callable: Callable[..., Any],
        probes: Sequence[Sequence[Any] | Mapping[str, Any]],
        *,
        native_precondition: Callable[..., bool] | None = None,
        activation_adapter: str | None = None,
    ) -> Callable[..., Any]:
        """Verify a free function returning one descriptor-defined record.

        Input records are flattened solely through emitted ``source_name``
        correlations, and the returned object is reconstructed solely through
        the final record descriptor whose leaf IDs exactly equal the ABI
        outputs.  A native precondition is mandatory while emitted validation
        contracts can terminate the process; rejected calls use the retained
        authored function before the DLL is entered.
        """

        import copy
        import ctypes
        import inspect
        import re
        import typing
        from .compiled_program_api import load_api

        if not probes:
            raise ValueError("native verification requires at least one probe")
        name = str(qualified_name)
        try:
            link = dict(self.links[name])
        except KeyError as error:
            raise KeyError(f"project product has no compiled call {name!r}") from error
        required = ("native_api", "native_library", "native_entrypoint")
        missing = tuple(item for item in required if not link.get(item))
        if missing:
            raise ValueError(f"native unit {name!r} is missing {missing!r}")
        api_path = self.root / str(link["native_api"])
        library_path = self.root / str(link["native_library"])
        descriptor = load_api(api_path)
        entry_name = str(link["native_entrypoint"])
        entry = next((
            dict(item) for item in descriptor.get("entry_points", ())
            if str(item.get("name")) == entry_name
        ), None)
        if entry is None:
            raise ValueError(f"native API does not declare {entry_name!r}")
        metadata = dict(descriptor.get("metadata") or {})
        validation_contracts = tuple(
            (metadata.get("validation_contracts") or {}).get(entry_name, ())
        )
        if validation_contracts and native_precondition is None:
            raise ValueError(
                "record-return verifier requires a native precondition while "
                "the emitted ABI contains terminating validation contracts"
            )

        parameters = tuple(map(dict, entry.get("parameters") or ()))
        inputs = tuple(
            parameter for parameter in parameters
            if str(parameter.get("role")) in {"input", "inout"}
        )
        outputs = tuple(
            parameter for parameter in parameters
            if str(parameter.get("role")) == "output"
        )
        output_ids = tuple(
            int(str(parameter["name"])[1:])
            for parameter in outputs
            if re.fullmatch(r"t[0-9]+", str(parameter.get("name") or ""))
        )
        if len(output_ids) != len(outputs) or not outputs:
            raise ValueError(
                "record-return verifier requires explicit scalar SSA outputs"
            )

        def flattened_field_ids(record: Mapping[str, Any]) -> tuple[int, ...]:
            return tuple(
                int(value_id)
                for field in record.get("fields") or ()
                for value_id in field.get("value_ids") or ()
            )

        records = tuple(map(dict, (
            metadata.get("record_tables") or {}
        ).get(entry_name, ())))
        return_records = tuple(
            record for record in records
            if flattened_field_ids(record) == output_ids
        )
        if len(return_records) != 1:
            raise ValueError(
                "record-return ABI must have exactly one final record whose "
                f"leaf IDs equal outputs {output_ids!r}"
            )
        return_record = return_records[0]
        fields = tuple(map(dict, return_record.get("fields") or ()))
        if not fields or any(
            str(field.get("storage")) not in {"scalar", "span"}
            or field.get("dtype") is None
            or not field.get("value_ids")
            for field in fields
        ):
            raise ValueError(
                "record-return verifier supports typed scalar/span fields only"
            )

        authored_descriptor = getattr(
            authored_callable, "__turing_authored_source_callable__",
            authored_callable,
        )
        authored = (
            authored_descriptor.fget
            if isinstance(authored_descriptor, property)
            else authored_descriptor
        )
        if authored is None:
            raise ValueError("record-return verifier requires an authored callable")
        signature = inspect.signature(authored)
        authored_parameters = tuple(signature.parameters)

        def source_root(source_name: str) -> str:
            return str(source_name).split(".", 1)[0]

        if any(not parameter.get("source_name") for parameter in inputs):
            raise ValueError("record-return native input lacks authored correlation")
        native_roots = tuple(dict.fromkeys(
            source_root(str(parameter["source_name"])) for parameter in inputs
        ))
        if set(native_roots) != set(authored_parameters):
            raise ValueError(
                "native ABI does not exactly match authored parameters: "
                f"authored={authored_parameters!r}, native={native_roots!r}"
            )

        try:
            return_type = typing.get_type_hints(authored).get("return")
        except (NameError, TypeError):
            return_type = signature.return_annotation
        if not isinstance(return_type, type) or (
            return_type.__name__ != str(return_record.get("identity") or "")
        ):
            raise ValueError(
                "authored return annotation does not match emitted record "
                f"{return_record.get('identity')!r}"
            )

        ctypes_types = {
            "c_int32": ctypes.c_int32,
            "c_int64": ctypes.c_int64,
            "c_float": ctypes.c_float,
            "c_double": ctypes.c_double,
            "c_bool": ctypes.c_bool,
            "c_uint8": ctypes.c_uint8,
        }
        try:
            native_types = {
                str(parameter["name"]): ctypes_types[str(parameter["ctypes"])]
                for parameter in parameters
            }
        except KeyError as error:
            raise ValueError(
                f"unsupported record-return native type {error.args[0]!r}"
            ) from error

        fixed_lengths = {
            int(tuple(parameter.get("shape") or ())[0])
            for parameter in inputs
            if len(tuple(parameter.get("shape") or ())) == 1
        }
        dynamic_extent_sources: dict[str, int] = {}
        for parameter in inputs:
            shape = tuple(map(int, parameter.get("shape") or ()))
            extent = parameter.get("extent")
            if extent is not None and len(shape) == 1:
                prior = dynamic_extent_sources.setdefault(str(extent), shape[0])
                if prior != shape[0]:
                    raise ValueError(f"conflicting extent {extent!r}")

        def extent_value(parameter_name: str) -> int:
            fixed = re.fullmatch(r"extent_([1-9][0-9]*)", parameter_name)
            if fixed is not None:
                return int(fixed.group(1))
            if parameter_name in dynamic_extent_sources:
                return dynamic_extent_sources[parameter_name]
            if len(fixed_lengths) == 1:
                return next(iter(fixed_lengths))
            raise ValueError(
                f"cannot derive dynamic extent {parameter_name!r} exactly"
            )

        runtime_handles = []
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            parents: set[str] = set()
            for dependency in metadata.get("runtime_dependencies", ()):
                dependency_path = Path(str(dependency.get("path") or ""))
                if dependency_path.is_file():
                    parent = str(dependency_path.parent.resolve())
                    if parent not in parents:
                        parents.add(parent)
                        runtime_handles.append(os.add_dll_directory(parent))
        library = ctypes.CDLL(str(library_path.resolve()))
        native = getattr(library, str(entry.get("symbol") or entry_name))
        native.argtypes = [
            (
                native_types[str(parameter["name"])]
                if str(parameter.get("passing")) == "value"
                else ctypes.POINTER(native_types[str(parameter["name"])])
            )
            for parameter in parameters
        ]
        native.restype = None
        route_counts = {"native": 0, "fallback": 0, "last_route": None}

        def resolve_source(bound: inspect.BoundArguments, source_name: str) -> Any:
            parts = str(source_name).split(".")
            value = bound.arguments[parts[0]]
            for part in parts[1:]:
                value = getattr(value, part)
            return value

        def deployed(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if native_precondition is not None and not bool(
                native_precondition(*bound.args, **bound.kwargs)
            ):
                route_counts["fallback"] += 1
                route_counts["last_route"] = "fallback"
                return authored(*args, **kwargs)
            keepalive: list[Any] = []
            source_storage: dict[tuple[str, str], Any] = {}
            output_storage: dict[str, Any] = {
                str(parameter["name"]): native_types[str(parameter["name"])]()
                for parameter in outputs
            }
            native_arguments = []
            for parameter in parameters:
                parameter_name = str(parameter["name"])
                role = str(parameter.get("role") or "")
                native_type = native_types[parameter_name]
                if role == "extent":
                    native_arguments.append(extent_value(parameter_name))
                    continue
                if role == "output":
                    value = output_storage[parameter_name]
                    keepalive.append(value)
                    native_arguments.append(ctypes.byref(value))
                    continue
                if role not in {"input", "inout"}:
                    raise ValueError(
                        f"unsupported record-return ABI role {role!r}"
                    )
                source_name = str(parameter["source_name"])
                source = resolve_source(bound, source_name)
                shape = tuple(map(int, parameter.get("shape") or ()))
                key = (source_name, str(parameter["ctypes"]))
                if shape:
                    if len(shape) != 1 or len(source) != shape[0]:
                        route_counts["fallback"] += 1
                        route_counts["last_route"] = "fallback"
                        return authored(*args, **kwargs)
                    resident = source_storage.get(key)
                    if resident is None:
                        converted = [native_type(value) for value in source]
                        if any(
                            isinstance(value, int) and not isinstance(value, bool)
                            and int(cell.value) != int(value)
                            for value, cell in zip(source, converted, strict=True)
                        ):
                            route_counts["fallback"] += 1
                            route_counts["last_route"] = "fallback"
                            return authored(*args, **kwargs)
                        resident = (native_type * shape[0])(*(
                            cell.value for cell in converted
                        ))
                        source_storage[key] = resident
                        keepalive.append(resident)
                    native_arguments.append(resident)
                else:
                    resident = native_type(source)
                    if (
                        isinstance(source, int) and not isinstance(source, bool)
                        and int(resident.value) != int(source)
                    ):
                        route_counts["fallback"] += 1
                        route_counts["last_route"] = "fallback"
                        return authored(*args, **kwargs)
                    keepalive.append(resident)
                    native_arguments.append(
                        resident.value
                        if str(parameter.get("passing")) == "value"
                        else ctypes.byref(resident)
                    )
            native(*native_arguments)
            route_counts["native"] += 1
            route_counts["last_route"] = "native"
            leaves = {
                int(parameter_name[1:]): output_storage[parameter_name].value
                for parameter_name in output_storage
            }
            result_fields = {}
            for field in fields:
                field_values = tuple(
                    leaves[int(value_id)]
                    for value_id in field.get("value_ids") or ()
                )
                result_fields[str(field["name"])] = (
                    field_values[0]
                    if str(field.get("storage")) == "scalar"
                    else tuple(field_values)
                )
            return return_type(**result_fields)

        probe_records = []
        for probe in probes:
            if isinstance(probe, Mapping):
                arguments, keywords = (), dict(probe)
            else:
                arguments, keywords = tuple(probe), {}
            expected_arguments = copy.deepcopy(arguments)
            expected_keywords = copy.deepcopy(keywords)
            try:
                expected = authored(*expected_arguments, **expected_keywords)
            except Exception as expected_error:
                before = route_counts["fallback"]
                try:
                    deployed(*copy.deepcopy(arguments), **copy.deepcopy(keywords))
                except type(expected_error) as actual_error:
                    if str(actual_error) != str(expected_error):
                        raise ValueError(
                            f"record-return fallback exception mismatch for "
                            f"{name}: authored={expected_error!r}, "
                            f"deployed={actual_error!r}"
                        ) from actual_error
                    if route_counts["fallback"] == before:
                        raise ValueError(
                            "invalid record-return probe entered native code"
                        )
                    probe_records.append({
                        "arguments": repr(arguments),
                        "keywords": repr(keywords),
                        "route": "fallback",
                        "exception": repr(actual_error),
                    })
                    continue
                raise ValueError(
                    f"record-return deployment lost {expected_error!r}"
                ) from expected_error
            actual = deployed(
                *copy.deepcopy(arguments), **copy.deepcopy(keywords)
            )
            route = str(route_counts["last_route"])
            if route != "native":
                raise ValueError(
                    f"valid equivalence probe for {name} used fallback"
                )
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"native equivalence failed for {name}{arguments!r}: "
                    f"authored={expected!r}, native={actual!r}"
                )
            probe_records.append({
                "arguments": repr(arguments),
                "keywords": repr(keywords),
                "route": route,
                "result": repr(actual),
            })

        verification = {
            "schema": "turing.native-callable-verification.v1",
            "qualified_name": name,
            "abi_kind": "record-return",
            "record_identity": str(return_record.get("identity") or ""),
            "source_sha256": str(self.manifest.get("source_sha256") or ""),
            "api_sha256": hashlib.sha256(api_path.read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
            "entrypoint": entry_name,
            "authored_parameters": list(authored_parameters),
            "probe_count": len(probe_records),
            "native_probe_count": int(route_counts["native"]),
            "fallback_probe_count": int(route_counts["fallback"]),
            "probes": probe_records,
            "status": "verified",
            **({
                "activation_adapter": str(activation_adapter),
            } if activation_adapter is not None else {}),
        }
        _atomic_json(library_path.parent / "native-verification.json", verification)
        deployed.__turing_native_verification__ = verification
        deployed.__turing_native_library__ = library
        deployed.__turing_native_runtime_handles__ = tuple(runtime_handles)
        deployed.__turing_native_precondition__ = native_precondition
        # The mutable counter is diagnostic state, not compiler identity. It
        # lets an isolated worker prove that ordinary compiler execution used
        # the installed native route after activation; semantic receipts and
        # deterministic SSA ids remain independent of it.
        deployed.__turing_native_route_counts__ = route_counts
        return deployed

    def verify_native_scalar_record_callable(
        self,
        qualified_name: str,
        authored_callable: Callable[..., Any],
        probes: Sequence[Sequence[Any] | Mapping[str, Any]],
    ) -> Callable[..., Any]:
        """Verify a method whose receiver is a typed scalar-record layout.

        The emitted record table, not a Python class-name convention, defines
        the receiver's physical field order and typed column offsets. Nested
        records, spans, references, and ambiguous outputs are refused.
        Mutable fields are copied back only when the emitted descriptor marks
        them writable.
        """

        import copy
        import ctypes
        import inspect
        import re
        from .compiled_program_api import load_api

        if not probes:
            raise ValueError("native verification requires at least one probe")
        name = str(qualified_name)
        try:
            link = dict(self.links[name])
        except KeyError as error:
            raise KeyError(f"project product has no compiled call {name!r}") from error
        required = ("native_api", "native_library", "native_entrypoint")
        missing = tuple(item for item in required if not link.get(item))
        if missing:
            raise ValueError(f"native unit {name!r} is missing {missing!r}")
        api_path = self.root / str(link["native_api"])
        library_path = self.root / str(link["native_library"])
        descriptor = load_api(api_path)
        entry_name = str(link["native_entrypoint"])
        entry = next((
            item for item in descriptor.get("entry_points", ())
            if str(item.get("name")) == entry_name
        ), None)
        if entry is None:
            raise ValueError(f"native API does not declare {entry_name!r}")
        records = tuple(
            (descriptor.get("metadata", {}).get("record_tables") or {}).get(
                entry_name, ()
            )
        )
        if len(records) != 1:
            raise ValueError(
                "scalar-record verifier requires exactly one receiver record"
            )
        record = dict(records[0])
        fields = tuple(map(dict, record.get("fields") or ()))
        if not fields or any(
            str(field.get("storage")) != "scalar"
            or len(field.get("value_ids") or ()) != 1
            or field.get("dtype") is None
            for field in fields
        ):
            raise ValueError(
                "scalar-record verifier refuses non-scalar or untyped fields"
            )
        parameters = tuple(map(dict, entry.get("parameters") or ()))
        receiver_ids = tuple(sorted({
            int(field["value_ids"][0]) for field in fields
        }))
        receivers = {
            value_id: next((
                parameter for parameter in parameters
                if str(parameter.get("name")) == f"t{value_id}"
            ), None)
            for value_id in receiver_ids
        }
        if any(receiver is None or not receiver.get("shape")
               for receiver in receivers.values()):
            raise ValueError("native API does not expose every receiver column")
        outputs = tuple(
            parameter for parameter in parameters
            if parameter.get("role") == "output"
        )
        if len(outputs) != 1 or outputs[0].get("shape"):
            raise ValueError(
                "scalar-record verifier requires one scalar return output"
            )
        ctypes_types = {
            "c_int32": ctypes.c_int32,
            "c_int64": ctypes.c_int64,
            "c_float": ctypes.c_float,
            "c_double": ctypes.c_double,
            "c_bool": ctypes.c_bool,
            "c_uint8": ctypes.c_uint8,
        }
        try:
            receiver_types = {
                value_id: ctypes_types[str(receiver["ctypes"])]
                for value_id, receiver in receivers.items()
            }
            output_type = ctypes_types[str(outputs[0]["ctypes"])]
        except KeyError as error:
            raise ValueError(
                f"unsupported scalar native type {error.args[0]!r}"
            ) from error
        column_field_counts = {
            value_id: max(
                int(field.get("offset", index))
                for index, field in enumerate(fields)
                if int(field["value_ids"][0]) == value_id
            ) + 1
            for value_id in receiver_ids
        }
        for value_id, receiver in receivers.items():
            if tuple(map(int, receiver.get("shape") or ())) != (
                column_field_counts[value_id],
            ):
                raise ValueError(
                    "receiver column shape does not match its record field offsets"
                )
        authored_descriptor = getattr(
            authored_callable, "__turing_authored_source_callable__",
            authored_callable,
        )
        authored = (
            authored_descriptor.fget
            if isinstance(authored_descriptor, property)
            else authored_descriptor
        )
        if authored is None:
            raise ValueError("scalar-record verifier requires a property getter")
        signature = inspect.signature(authored)
        source_parameters = tuple(signature.parameters)
        if not source_parameters or source_parameters[0] not in {"self", "cls"}:
            raise ValueError("scalar-record verifier requires a method receiver")
        direct_inputs = tuple(
            parameter for parameter in parameters
            if parameter.get("role") in {"input", "inout"}
            and str(parameter.get("name")) not in {
                f"t{value_id}" for value_id in receiver_ids
            }
        )
        direct_names = tuple(
            str(parameter.get("source_name") or "")
            for parameter in direct_inputs
        )
        if direct_names != source_parameters[1:]:
            raise ValueError(
                "native ABI does not exactly match authored non-receiver "
                f"parameters: authored={source_parameters[1:]!r}, "
                f"native={direct_names!r}"
            )
        runtime_handles = []
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            for dependency in descriptor.get("metadata", {}).get(
                "runtime_dependencies", ()
            ):
                dependency_path = Path(str(dependency.get("path") or ""))
                if dependency_path.is_file():
                    runtime_handles.append(os.add_dll_directory(
                        str(dependency_path.parent.resolve())
                    ))
        library = ctypes.CDLL(str(library_path.resolve()))
        native = getattr(library, str(entry.get("symbol") or entry_name))

        def parameter_ctype(parameter: Mapping[str, Any]):
            try:
                scalar = ctypes_types[str(parameter["ctypes"])]
            except KeyError as error:
                raise ValueError(
                    f"unsupported scalar native type {error.args[0]!r}"
                ) from error
            return (
                scalar
                if str(parameter.get("passing")) == "value"
                else ctypes.POINTER(scalar)
            )

        native.argtypes = [parameter_ctype(parameter) for parameter in parameters]
        native.restype = None

        def deployed(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            receiver_object = bound.arguments[source_parameters[0]]
            arenas = {
                value_id: (receiver_types[value_id] * column_field_counts[value_id])()
                for value_id in receiver_ids
            }
            for index, field in enumerate(fields):
                value_id = int(field["value_ids"][0])
                offset = int(field.get("offset", index))
                arenas[value_id][offset] = receiver_types[value_id](
                    getattr(receiver_object, str(field["name"]))
                ).value
            result = output_type()
            keepalive: list[Any] = [*arenas.values(), result]
            native_arguments = []
            for parameter in parameters:
                parameter_name = str(parameter.get("name"))
                role = str(parameter.get("role"))
                if role == "extent":
                    fixed = re.fullmatch(r"extent_([1-9][0-9]*)", parameter_name)
                    if fixed is None or int(fixed.group(1)) not in set(
                        column_field_counts.values()
                    ):
                        raise ValueError(
                            "receiver extent is not a fixed record-column count"
                        )
                    native_arguments.append(int(fixed.group(1)))
                elif parameter_name.startswith("t") and parameter_name[1:].isdigit() \
                        and int(parameter_name[1:]) in arenas:
                    native_arguments.append(arenas[int(parameter_name[1:])])
                elif parameter is outputs[0]:
                    native_arguments.append(ctypes.byref(result))
                elif role in {"input", "inout"}:
                    scalar_type = ctypes_types[str(parameter["ctypes"])]
                    scalar = scalar_type(bound.arguments[
                        str(parameter.get("source_name"))
                    ])
                    keepalive.append(scalar)
                    native_arguments.append(
                        scalar.value
                        if str(parameter.get("passing")) == "value"
                        else ctypes.byref(scalar)
                    )
                else:
                    raise ValueError(
                        f"unsupported native record parameter {parameter_name!r}"
                    )
            native(*native_arguments)
            for index, field in enumerate(fields):
                if field.get("writable"):
                    value_id = int(field["value_ids"][0])
                    setattr(
                        receiver_object,
                        str(field["name"]),
                        arenas[value_id][int(field.get("offset", index))],
                    )
            return result.value

        probe_records = []
        for probe in probes:
            if isinstance(probe, Mapping):
                arguments, keywords = (), dict(probe)
            else:
                arguments, keywords = tuple(probe), {}
            expected_arguments = copy.deepcopy(arguments)
            expected_keywords = copy.deepcopy(keywords)
            native_arguments = copy.deepcopy(arguments)
            native_keywords = copy.deepcopy(keywords)
            expected = authored(*expected_arguments, **expected_keywords)
            actual = deployed(*native_arguments, **native_keywords)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"native equivalence failed for {name}{arguments!r}: "
                    f"authored={expected!r}, native={actual!r}"
                )
            probe_records.append({
                "arguments": repr(arguments),
                "keywords": repr(keywords),
                "result": repr(actual),
            })
        verification = {
            "schema": "turing.native-callable-verification.v1",
            "qualified_name": name,
            "abi_kind": "scalar-record",
            "source_sha256": str(self.manifest.get("source_sha256") or ""),
            "api_sha256": hashlib.sha256(api_path.read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
            "entrypoint": entry_name,
            "record_identity": str(record.get("identity") or ""),
            "probe_count": len(probe_records),
            "native_probe_count": len(probe_records),
            "fallback_probe_count": 0,
            "probes": probe_records,
            "status": "verified",
        }
        _atomic_json(library_path.parent / "native-verification.json", verification)
        deployed.__turing_native_verification__ = verification
        deployed.__turing_native_library__ = library
        deployed.__turing_native_runtime_handles__ = tuple(runtime_handles)
        return deployed


def open_project_compilation_product(
    directory: str | Path,
) -> ProjectCompilationProduct:
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    link_table = json.loads((root / "links.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != PROJECT_PRODUCT_SCHEMA:
        raise ValueError(f"unsupported project product schema {manifest.get('schema')!r}")
    if link_table.get("schema") != LINK_TABLE_SCHEMA:
        raise ValueError(f"unsupported project link schema {link_table.get('schema')!r}")
    return ProjectCompilationProduct(
        root=root,
        manifest=manifest,
        links={
            str(item["qualified_name"]): item
            for item in link_table.get("links", ())
        },
    )


def _resolve_product_callable(
    source_module: str,
    qualified_name: str,
) -> tuple[Any, Callable[..., Any]]:
    """Resolve an authored module/class callable without executing a project."""

    if ".<locals>." in str(qualified_name):
        raise ValueError("lexically nested callables have no importable owner")
    module = importlib.import_module(str(source_module))
    parts = str(qualified_name).split(".")
    owner: Any = module
    for component in parts[:-1]:
        owner = getattr(owner, component)
    callable_value = getattr(owner, parts[-1])
    if not callable(callable_value):
        raise TypeError(f"authored target {qualified_name!r} is not callable")
    return owner, callable_value


def verify_project_unit_automatically(
    product: ProjectCompilationProduct,
    qualified_name: str,
) -> Callable[..., Any]:
    """Select a verifier from emitted ABI metadata and authored contracts."""

    import inspect
    from .compiled_program_api import load_api

    sample_values = {
        "c_bool": (False, True, True),
        "c_int32": (-3, 1, 5),
        "c_int64": (-3, 1, 5),
        "c_uint8": (0, 1, 5),
        "c_float": (-3.25, 1.25, 4.5),
        "c_double": (-3.25, 1.25, 4.5),
    }
    name = str(qualified_name)
    link = dict(product.links[name])
    source_module = str(link.get("source_module") or "")
    if not source_module:
        raise ValueError("unit receipt has no importable source module")
    _owner, authored = _resolve_product_callable(source_module, name)
    api_path = product.root / str(link.get("native_api") or "")
    descriptor = load_api(api_path)
    entry_name = str(link.get("native_entrypoint") or "")
    entry = next((
        item for item in descriptor.get("entry_points", ())
        if str(item.get("name")) == entry_name
    ), None)
    if entry is None:
        raise ValueError("native entrypoint is absent from its API")
    parameters = tuple(entry.get("parameters") or ())
    inputs = tuple(
        parameter for parameter in parameters
        if parameter.get("role") in {"input", "inout"}
    )
    outputs = tuple(
        parameter for parameter in parameters
        if parameter.get("role") in {"output", "inout"}
    )
    if (
        len(outputs) != 1
        or outputs[0].get("role") != "output"
        or any(
            parameter.get("shape") or parameter.get("extents")
            or parameter.get("extent")
            for parameter in parameters
        )
    ):
        raise ValueError(
            "emitted descriptor has no automatically provable scalar ABI; "
            "retain source or select its record/sequence verifier"
        )
    domains = {}
    for parameter in inputs:
        source_name = str(parameter.get("source_name") or "")
        if not source_name or "." in source_name:
            raise ValueError("native input lacks a direct source parameter")
        try:
            domains[source_name] = sample_values[str(parameter["ctypes"])]
        except KeyError as error:
            raise ValueError(
                f"no scalar probe domain for {error.args[0]!r}"
            ) from error
    if len(domains) != len(inputs):
        raise ValueError("native scalar inputs repeat a source parameter")
    authored_descriptor = getattr(
        authored, "__turing_authored_source_callable__", authored,
    )
    authored_function = (
        authored_descriptor.fget
        if isinstance(authored_descriptor, property) else authored_descriptor
    )
    signature = inspect.signature(authored_function)
    if any(
        parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        for parameter in signature.parameters.values()
    ):
        raise ValueError(
            "automatic scalar verification requires a finite authored "
            "parameter surface"
        )
    source_parameters = tuple(
        parameter.name for parameter in signature.parameters.values()
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    )
    source_path = Path(str(product.manifest.get("source") or ""))
    if not source_path.is_file():
        raise ValueError("project product has no current authored source")
    parameter_contract = authored_parameter_contract(
        source_path.read_text(encoding="utf-8"), name,
    )
    used_parameters = set(map(
        str, parameter_contract.get("used_parameters") or (),
    ))
    ignored = tuple(
        parameter for parameter in source_parameters
        if parameter not in domains
    )
    illegally_ignored = tuple(
        parameter for parameter in ignored if parameter in used_parameters
    )
    if illegally_ignored:
        raise ValueError(
            "native ABI omitted source-read parameters "
            f"{illegally_ignored!r}"
        )
    positional_only = tuple(
        parameter.name for parameter in signature.parameters.values()
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY
    )
    keyword_only = tuple(
        parameter.name for parameter in signature.parameters.values()
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY
    )
    if positional_only and keyword_only:
        raise ValueError(
            "automatic scalar probes cannot mix positional-only and "
            "keyword-only parameters"
        )
    probes = []
    for probe_index in range(3):
        values = {
            parameter: (
                domains[parameter][probe_index]
                if parameter in domains else None
            )
            for parameter in source_parameters
        }
        probes.append(
            tuple(values[parameter] for parameter in source_parameters)
            if positional_only else values
        )
    output_ctype = str(outputs[0].get("ctypes") or "")
    signed_widths = {"c_int32": 32, "c_int64": 64}
    width = signed_widths.get(output_ctype)
    authored_probe_results = [
        authored(*probe) if isinstance(probe, tuple) else authored(**probe)
        for probe in probes
    ]
    result_codec = None
    if (
        width is not None
        and authored_probe_results
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            and 0 <= value < (1 << width)
            for value in authored_probe_results
        )
        and any(value >= (1 << (width - 1)) for value in authored_probe_results)
    ):
        result_codec = f"unsigned-{output_ctype}-v1"
    return product.verify_native_scalar_callable(
        name,
        authored,
        tuple(probes),
        activation_adapter="descriptor-call-v1",
        ignored_source_parameters=ignored,
        probe_factory="authored-contract-scalar-v1",
        native_result_codec=result_codec,
        expected_probe_results=authored_probe_results,
    )


def verify_project_units_automatically(
    directory: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Verify emitted units through descriptor-selected, source-proven ABIs."""

    product = open_project_compilation_product(directory)
    results = []
    for qualified_name, link_value in sorted(product.links.items()):
        link = dict(link_value)
        if link.get("kind") == "source-region-integral":
            continue
        record = {
            "qualified_name": str(qualified_name),
            "status": "unsupported",
        }
        try:
            deployed = verify_project_unit_automatically(
                product, str(qualified_name),
            )
            verification = dict(deployed.__turing_native_verification__)
            record.update({
                "status": "verified",
                "probe_count": int(verification["probe_count"]),
                "native_probe_count": int(
                    verification["native_probe_count"]
                ),
                "fallback_probe_count": int(
                    verification["fallback_probe_count"]
                ),
            })
        except Exception as error:
            record.update({
                "reason": f"{type(error).__name__}: {error}",
            })
        results.append(record)
    return tuple(results)


def verify_project_scalar_units_automatically(
    directory: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Compatibility name for descriptor-selected automatic verification."""

    return verify_project_units_automatically(directory)


def _defined_names(statement: ast.stmt) -> tuple[str, ...]:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return (statement.name,)
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return tuple(
            alias.asname or alias.name.split(".", 1)[0]
            for alias in statement.names
        )
    targets: list[ast.AST] = []
    if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = (
            list(statement.targets)
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
    return tuple(
        node.id
        for target in targets
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )


def _loaded_external_names(statement: ast.stmt) -> set[str]:
    loaded = {
        node.id
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    local = {
        node.id
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    local.update(
        argument.arg
        for node in ast.walk(statement)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            *((node.args.vararg,) if node.args.vararg is not None else ()),
            *((node.args.kwarg,) if node.args.kwarg is not None else ()),
        )
    )
    return loaded - local


@dataclass(frozen=True)
class _AuthoredDefinition:
    qualified_name: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    ancestors: tuple[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, ...]
    kind: str


def _authored_definitions(source: str) -> dict[str, _AuthoredDefinition]:
    """Index authored callables, including their lexical nested functions."""

    module = ast.parse(source)
    indexed: dict[str, _AuthoredDefinition] = {}

    def add_function(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        qualified_name: str,
        ancestors: tuple[
            ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, ...
        ],
        kind: str,
    ) -> None:
        indexed[qualified_name] = _AuthoredDefinition(
            qualified_name, node, ancestors, kind,
        )
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add_function(
                    statement,
                    f"{qualified_name}.<locals>.{statement.name}",
                    (*ancestors, node),
                    "nested-function",
                )

    for statement in module.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_function(statement, statement.name, (), "function")
        elif isinstance(statement, ast.ClassDef):
            for member in statement.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add_function(
                        member, f"{statement.name}.{member.name}",
                        (statement,), "method",
                    )
    return indexed


def _partition_nested_authored_source(
    source: str,
    definition: _AuthoredDefinition,
) -> tuple[str, tuple[str, ...]]:
    """Keep an exact nested definition and only its lexical source shells."""

    module = ast.parse(source)
    module_definitions: dict[str, ast.stmt] = {}
    selected: dict[int, ast.stmt] = {}
    for statement in module.body:
        if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
            selected[id(statement)] = statement
        for name in _defined_names(statement):
            module_definitions[name] = statement

    # A nested function remains in its exact original lexical location. Only
    # headers of its enclosing class/functions are retained; the nested body
    # itself supplies the suite, so unrelated outer statements never enter the
    # worker. This is a compile partition, not a rewritten source authority.
    structural_nodes = (*definition.ancestors, definition.node)
    selected[id(definition.node)] = definition.node
    pending_names = sorted(_loaded_external_names(definition.node), reverse=True)
    while pending_names:
        name = pending_names.pop()
        dependency = module_definitions.get(name)
        if dependency is None or dependency in structural_nodes:
            continue
        if id(dependency) in selected:
            continue
        selected[id(dependency)] = dependency
        pending_names.extend(sorted(_loaded_external_names(dependency), reverse=True))

    retained_lines: set[int] = set()
    selected_names: set[str] = {definition.node.name}
    for ancestor in definition.ancestors:
        starts = [int(ancestor.lineno)]
        starts.extend(int(item.lineno) for item in ancestor.decorator_list)
        first_body = min(
            (int(item.lineno) for item in ancestor.body),
            default=int(ancestor.end_lineno) + 1,
        )
        retained_lines.update(range(min(starts), first_body))
        selected_names.add(ancestor.name)
        if isinstance(ancestor, ast.ClassDef):
            for member in ancestor.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                retained_lines.update(range(
                    int(member.lineno), int(member.end_lineno) + 1,
                ))
                selected_names.update(_defined_names(member))
    for statement in selected.values():
        starts = [int(statement.lineno)]
        starts.extend(
            int(item.lineno)
            for item in getattr(statement, "decorator_list", ())
        )
        retained_lines.update(range(
            min(starts), int(getattr(statement, "end_lineno", statement.lineno)) + 1,
        ))
        selected_names.update(_defined_names(statement))
    lines = source.splitlines(keepends=True)
    partitioned = "".join(
        line if number in retained_lines else ("\n" if line.endswith("\n") else "")
        for number, line in enumerate(lines, 1)
    )
    ast.parse(partitioned)
    available = {call.qualified_name for call in discover_authored_calls(partitioned)}
    if definition.qualified_name not in available:
        raise RuntimeError(
            f"source partition lost nested call {definition.qualified_name!r}; "
            f"available={tuple(sorted(available))!r}"
        )
    return partitioned, tuple(sorted(selected_names))


def partition_authored_source(
    source: str,
    qualified_name: str,
    *,
    linked_dependencies: Iterable[str] = (),
) -> tuple[str, tuple[str, ...]]:
    """Retain the exact lexical closure of one top-level call.

    Omitted source lines become blank lines instead of being removed, keeping
    every selected AST node's original line number and source spelling.  The
    closure follows module-level definitions/imports referenced by the target;
    a selected method retains its complete class as the indivisible authored
    definition containing that method.
    """

    indexed = _authored_definitions(source)
    indexed_definition = indexed.get(str(qualified_name))
    if indexed_definition is None:
        raise ValueError(f"unknown authored project call {qualified_name!r}")
    if indexed_definition.kind == "nested-function":
        return _partition_nested_authored_source(source, indexed_definition)

    module = ast.parse(source)
    owner_name = str(qualified_name).split(".", 1)[0]
    definitions: dict[str, ast.stmt] = {}
    future_imports: list[ast.stmt] = []
    for statement in module.body:
        if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
            future_imports.append(statement)
        for name in _defined_names(statement):
            definitions[name] = statement
    if owner_name not in definitions:
        raise ValueError(f"unknown authored project call {qualified_name!r}")
    # Linked dependency definitions remain in the lexical closure as the
    # authored call-frame/type contract. The planner replaces their execution
    # with the attached repository-SSA module; retaining the definition here
    # avoids guessing an ABI from a binary artifact or inventing a stub body.
    linked_dependencies = tuple(map(str, linked_dependencies))

    owner_statement = definitions[owner_name]
    method_name = (
        str(qualified_name).split(".", 1)[1]
        if "." in str(qualified_name) else None
    )
    class_owner = (
        owner_statement
        if method_name is not None and isinstance(owner_statement, ast.ClassDef)
        else None
    )
    if class_owner is not None:
        target_statement = next((
            member
            for member in class_owner.body
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            and member.name == method_name
        ), None)
        if target_statement is None:
            raise ValueError(f"unknown authored project call {qualified_name!r}")
    else:
        target_statement = owner_statement

    selected: dict[int, ast.stmt] = {id(item): item for item in future_imports}
    class_support: list[ast.stmt] = []
    if class_owner is not None:
        class_support = [
            member for member in class_owner.body
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        selected.update((id(item), item) for item in class_support)
        linked_method_names = {
            name.split(".", 1)[1]
            for name in linked_dependencies
            if name.startswith(class_owner.name + ".")
        }
        selected.update(
            (id(member), member)
            for member in class_owner.body
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            and member.name in linked_method_names
        )
    pending = [target_statement]
    while pending:
        statement = pending.pop()
        if id(statement) in selected:
            continue
        selected[id(statement)] = statement
        for name in sorted(_loaded_external_names(statement), reverse=True):
            dependency = definitions.get(name)
            if dependency is not None and id(dependency) not in selected:
                pending.append(dependency)

    lines = source.splitlines(keepends=True)
    retained_lines: set[int] = set()
    selected_names: set[str] = set()
    if class_owner is not None:
        starts = [int(class_owner.lineno)]
        starts.extend(
            int(decorator.lineno)
            for decorator in class_owner.decorator_list
        )
        class_start = min(starts)
        first_body_line = min(
            (int(member.lineno) for member in class_owner.body),
            default=int(class_owner.end_lineno) + 1,
        )
        retained_lines.update(range(class_start, first_body_line))
        selected_names.add(class_owner.name)
    for statement in selected.values():
        starts = [int(statement.lineno)]
        starts.extend(
            int(decorator.lineno)
            for decorator in getattr(statement, "decorator_list", ())
        )
        start = min(starts)
        end = int(getattr(statement, "end_lineno", statement.lineno))
        retained_lines.update(range(start, end + 1))
        selected_names.update(_defined_names(statement))
    partitioned = "".join(
        line if line_number in retained_lines else ("\n" if line.endswith("\n") else "")
        for line_number, line in enumerate(lines, 1)
    )
    # Validate both the partition's syntax and that the requested callable is
    # still discoverable before a costly worker is launched.
    available = {call.qualified_name for call in discover_authored_calls(partitioned)}
    if qualified_name not in available:
        raise RuntimeError(
            f"source partition lost authored call {qualified_name!r}; "
            f"available={tuple(sorted(available))!r}"
        )
    return partitioned, tuple(sorted(selected_names))


def discover_authored_calls(source: str) -> tuple[AuthoredCall, ...]:
    """Return every independently addressable implemented function.

    ``pass`` and ``...`` remain valid concrete no-op implementations. On a
    ``typing.Protocol``, ``abstractmethod``, or ``overload`` surface, however,
    that body is only a declaration. Treating one as compilable used to
    produce an empty native subroutine and falsely label it complete.
    """

    module = ast.parse(source)

    def terminal_name(expression: ast.expr) -> str:
        if isinstance(expression, ast.Name):
            return expression.id
        if isinstance(expression, ast.Attribute):
            return expression.attr
        return ""

    def declaration_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body.pop(0)
        return (
            len(body) == 1
            and (
                isinstance(body[0], ast.Pass)
                or (
                    isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and body[0].value.value is Ellipsis
                )
            )
        )

    protocol_names = {"Protocol"}
    declaration_decorators = {"overload", "abstractmethod"}
    for statement in module.body:
        if not isinstance(statement, ast.ImportFrom):
            continue
        if statement.module in {"typing", "typing_extensions"}:
            protocol_names.update(
                alias.asname or alias.name
                for alias in statement.names if alias.name == "Protocol"
            )
            declaration_decorators.update(
                alias.asname or alias.name
                for alias in statement.names if alias.name == "overload"
            )
        elif statement.module == "abc":
            declaration_decorators.update(
                alias.asname or alias.name
                for alias in statement.names if alias.name == "abstractmethod"
            )
    protocol_classes = {
        statement.name
        for statement in module.body
        if isinstance(statement, ast.ClassDef)
        and any(
            terminal_name(base) in protocol_names for base in statement.bases
        )
    }
    declarations = set()
    for statement in module.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = {
                terminal_name(item) for item in statement.decorator_list
            }
            if declaration_body(statement) and decorators & declaration_decorators:
                declarations.add(statement.name)
        elif isinstance(statement, ast.ClassDef):
            for member in statement.body:
                if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                decorators = {
                    terminal_name(item) for item in member.decorator_list
                }
                if declaration_body(member) and (
                    statement.name in protocol_classes
                    or bool(decorators & declaration_decorators)
                ):
                    declarations.add(f"{statement.name}.{member.name}")

    calls = [
        AuthoredCall(name, int(definition.node.lineno), definition.kind)
        for name, definition in _authored_definitions(source).items()
        if name not in declarations
    ]
    return tuple(sorted(calls, key=lambda item: (item.qualified_name, item.line)))


def authored_return_contract(source: str, qualified_name: str) -> dict[str, Any]:
    """Describe explicit returns owned by one callable, excluding children."""

    definition = _authored_definitions(source).get(str(qualified_name))
    if definition is None:
        raise ValueError(f"unknown authored project call {qualified_name!r}")
    returns: list[ast.Return] = []

    class ReturnVisitor(ast.NodeVisitor):
        def visit_Return(self, node: ast.Return) -> None:
            returns.append(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is definition.node:
                for statement in node.body:
                    self.visit(statement)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

    ReturnVisitor().visit(definition.node)
    valued = tuple(node for node in returns if node.value is not None)
    return {
        "explicit_return_count": len(returns),
        "valued_return_count": len(valued),
        "requires_value_publication": bool(valued),
        "annotation": (
            None
            if definition.node.returns is None
            else ast.unparse(definition.node.returns)
        ),
    }


def authored_parameter_contract(
    source: str, qualified_name: str,
) -> dict[str, Any]:
    """Describe parameters whose values the authored callable actually reads.

    An unused formal need not enter a native ABI.  A used formal must remain
    source-bound directly, through a sequence view, or through a declared
    record/value ABI; otherwise a compiled unit has silently lost part of its
    program even if its return and control counts look complete.
    """

    definition = _authored_definitions(source).get(str(qualified_name))
    if definition is None:
        raise ValueError(f"unknown authored project call {qualified_name!r}")
    arguments = definition.node.args
    ordered = tuple(
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    ) + (() if arguments.vararg is None else (arguments.vararg.arg,)) + (
        () if arguments.kwarg is None else (arguments.kwarg.arg,)
    )
    reads: set[str] = set()

    class ParameterReadVisitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load) and node.id in ordered:
                reads.add(node.id)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is definition.node:
                for statement in node.body:
                    self.visit(statement)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

    ParameterReadVisitor().visit(definition.node)
    return {
        "parameters": list(ordered),
        "used_parameters": [name for name in ordered if name in reads],
    }


def authored_closure_contract(
    source: str, qualified_name: str,
) -> dict[str, Any]:
    """Describe exact lexical values a nested callable must receive.

    Python's symbol table already distinguishes a true closure cell from a
    module global. Reusing it prevents a nested body that reads outer locals
    from being mistaken for a closed zero-argument program.
    """

    definition = _authored_definitions(source).get(str(qualified_name))
    if definition is None:
        raise ValueError(f"unknown authored project call {qualified_name!r}")
    if definition.kind != "nested-function":
        return {"captures": []}
    root = symtable.symtable(source, "<authored-project>", "exec")
    pending = list(root.get_children())
    selected = None
    while pending:
        table = pending.pop()
        if (
            table.get_type() == "function"
            and table.get_name() == definition.node.name
            and int(table.get_lineno()) == int(definition.node.lineno)
        ):
            selected = table
            break
        pending.extend(table.get_children())
    if selected is None:
        raise RuntimeError(
            f"Python symbol table lost nested callable {qualified_name!r}"
        )
    return {
        "captures": sorted(
            symbol.get_name()
            for symbol in selected.get_symbols()
            if symbol.is_free()
        ),
    }


def authored_control_contract(source: str, qualified_name: str) -> dict[str, int]:
    """Count control owned by one callable without borrowing child bodies."""

    definition = _authored_definitions(source).get(str(qualified_name))
    if definition is None:
        raise ValueError(f"unknown authored project call {qualified_name!r}")
    counts = {
        "if_count": 0,
        "raise_guard_count": 0,
        "loop_count": 0,
        "break_count": 0,
        "continue_count": 0,
        "loop_early_return_count": 0,
    }

    class ControlVisitor(ast.NodeVisitor):
        loop_depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is definition.node:
                for statement in node.body:
                    self.visit(statement)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_If(self, node: ast.If) -> None:
            counts["if_count"] += 1
            if (
                node.body
                and all(isinstance(item, ast.Raise) for item in node.body)
                and not node.orelse
            ):
                counts["raise_guard_count"] += 1
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            counts["loop_count"] += 1
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        visit_AsyncFor = visit_For

        def visit_While(self, node: ast.While) -> None:
            counts["loop_count"] += 1
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_Break(self, node: ast.Break) -> None:
            counts["break_count"] += 1

        def visit_Continue(self, node: ast.Continue) -> None:
            counts["continue_count"] += 1

        def visit_Return(self, node: ast.Return) -> None:
            if self.loop_depth:
                counts["loop_early_return_count"] += 1
            self.generic_visit(node)

    ControlVisitor().visit(definition.node)
    return counts


def _authored_dependency_graph(source: str) -> dict[str, set[str]]:
    """Resolve direct module-function and statically typed method calls."""

    module = ast.parse(source)
    functions = {
        statement.name: statement
        for statement in module.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    classes = {
        statement.name: statement
        for statement in module.body
        if isinstance(statement, ast.ClassDef)
    }
    indexed = _authored_definitions(source)
    methods = {
        name: definition.node
        for name, definition in indexed.items()
        if definition.kind == "method"
    }
    definitions = {
        name: definition.node for name, definition in indexed.items()
    }
    graph: dict[str, set[str]] = {name: set() for name in definitions}

    direct_nested: dict[str, dict[str, str]] = {}
    for name in definitions:
        if ".<locals>." not in name:
            parent = name
        else:
            parent = name.rsplit(".<locals>.", 1)[0]
        if name != parent:
            direct_nested.setdefault(parent, {})[name.rsplit(".", 1)[-1]] = name

    def body_nodes(definition: ast.AST) -> Iterable[ast.AST]:
        """Walk one callable without attributing nested bodies to its parent."""

        pending = list(reversed(getattr(definition, "body", ())))
        while pending:
            node = pending.pop()
            if isinstance(node, (
                ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
            )):
                continue
            yield node
            for child in reversed(list(ast.iter_child_nodes(node))):
                if isinstance(child, (
                    ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
                )):
                    continue
                pending.append(child)

    for qualified_name, definition in definitions.items():
        owner_class = (
            qualified_name.split(".", 1)[0]
            if "." in qualified_name else None
        )
        receiver_classes = {
            "self": owner_class,
            "cls": owner_class,
        } if owner_class is not None else {}
        # A parameter annotation is authored static receiver information, not
        # a runtime guess. Use it to crawl method dependencies of compiler
        # phases such as ``build_module(body: CodeBuilder)``. Previously only
        # constructor assignments were tracked, so an explicitly typed method
        # call disappeared from the unit graph.
        arguments = definition.args
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            annotation = argument.annotation
            if isinstance(annotation, ast.Name):
                annotated_class = annotation.id
            elif isinstance(annotation, ast.Constant) and isinstance(
                annotation.value, str
            ):
                annotated_class = annotation.value.rsplit(".", 1)[-1]
            elif isinstance(annotation, ast.Attribute):
                annotated_class = annotation.attr
            else:
                annotated_class = None
            if annotated_class in classes:
                receiver_classes[argument.arg] = annotated_class
        for node in body_nodes(definition):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in classes
            ):
                continue
            targets = (
                node.targets if isinstance(node, ast.Assign)
                else (node.target,)
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    receiver_classes[target.id] = value.func.id

        for node in body_nodes(definition):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
                lexical_parent = (
                    qualified_name.rsplit(".<locals>.", 1)[0]
                    if ".<locals>." in qualified_name else qualified_name
                )
                nested = (
                    direct_nested.get(qualified_name, {}).get(name)
                    or direct_nested.get(lexical_parent, {}).get(name)
                )
                if nested is not None:
                    graph[qualified_name].add(nested)
                elif name in functions:
                    graph[qualified_name].add(name)
                elif name in classes:
                    constructor = f"{name}.__init__"
                    if constructor in methods:
                        graph[qualified_name].add(constructor)
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            receiver = node.func.value
            if not isinstance(receiver, ast.Name):
                continue
            receiver_class = receiver_classes.get(receiver.id)
            candidate = (
                None if receiver_class is None
                else f"{receiver_class}.{node.func.attr}"
            )
            if candidate in methods:
                graph[qualified_name].add(candidate)
    return graph


def dependency_ordered_authored_calls(
    source: str,
    qualified_names: Iterable[str],
) -> tuple[str, ...]:
    """Give source calls a deterministic dependency-first launch order."""

    dependency_graph = _authored_dependency_graph(source)
    requested = tuple(dict.fromkeys(map(str, qualified_names)))
    selected_set = set(requested)
    pending = list(requested)
    while pending:
        name = pending.pop()
        for dependency in sorted(dependency_graph.get(name, ())):
            if dependency in selected_set:
                continue
            selected_set.add(dependency)
            pending.append(dependency)
    selected = tuple(sorted(selected_set))
    dependencies = {
        name: dependency_graph.get(name, set()) & selected_set
        for name in selected
    }
    ordered: list[str] = []
    visited: set[str] = set()
    active: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in active:
            return
        active.add(name)
        for dependency in sorted(dependencies[name]):
            visit(dependency)
        active.remove(name)
        visited.add(name)
        ordered.append(name)

    for name in sorted(selected):
        visit(name)
    return tuple(ordered)


def authored_call_dependencies(
    source: str,
    qualified_names: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Return direct same-source function dependencies for selected calls."""

    dependency_graph = _authored_dependency_graph(source)
    selected = tuple(dict.fromkeys(map(str, qualified_names)))
    selected_set = set(selected)
    return {
        name: tuple(sorted(dependency_graph.get(name, set()) & selected_set))
        for name in selected
    }


def compilation_creep_frontier(
    records: Iterable[Mapping[str, Any]],
    dependencies: Mapping[str, Iterable[str]],
    contained_integrals: Mapping[str, Iterable[str]] | None = None,
) -> list[dict[str, Any]]:
    """Describe the deterministic next work after a bounded catalogue pass."""

    frontier = []
    for source_record in records:
        record = dict(source_record)
        if record.get("status") == "complete":
            continue
        qualified_name = str(record["qualified_name"])
        dependency_names = tuple(sorted(map(
            str, dependencies.get(qualified_name, ()),
        )))
        authored_subunits = tuple(sorted(map(
            str, (contained_integrals or {}).get(qualified_name, ()),
        )))
        process_graph_subunits = tuple(sorted(map(
            str, record.get("process_graph_subunits") or (),
        )))
        status = str(record.get("status"))
        if status == "blocked":
            action = "retry-after-authored-dependencies"
        elif record.get("error_type") == "ResourceLimitExceeded":
            action = (
                "compile-deeper-authored-integrals"
                if authored_subunits else
                "compile-resolved-process-graph-units"
                if process_graph_subunits else
                "subdivide-minimum-authored-integral"
            )
        elif status == "partial":
            action = (
                "lower-compiler-graph-table-abi"
                if record.get("unresolved_program_abi_references")
                else "closure-convert-lexical-captures"
                if record.get("missing_lexical_captures")
                else "materialize-required-source-values"
                if record.get("unresolved_required_source_values")
                else str(record.get("control_frontier_action"))
                if record.get("control_frontier_action")
                else
                "repair-root-return-lowering"
                if record.get("root_return_publication_complete") is False
                else "repair-linked-call-frame"
                if int(record.get("unresolved_call_count") or 0) > 0
                else "materialize-extraction-boundaries"
            )
        else:
            error = str(record.get("error") or "")
            action = next((
                candidate_action
                for fragment, candidate_action in (
                    (
                        "static Python reference cannot be assigned through "
                        "a runtime tensor index",
                        "model-static-value-table-boundary",
                    ),
                    (
                        "iterable-access=closure_aggregate",
                        "model-resident-closure-aggregate",
                    ),
                    (
                        "generator consumer has no safe resident query lowering",
                        "lower-generator-sequence-query",
                    ),
                    (
                        "resolved-schema-conflict",
                        "repair-shared-sequence-schema",
                    ),
                    (
                        "cyclic loop-control containment",
                        "repair-control-region-containment",
                    ),
                    (
                        "nested control regions are absent from parent body",
                        "repair-control-region-containment",
                    ),
                )
                if fragment in error
            ), (
                "repair-stale-planned-node-reference"
                if record.get("error_type") == "KeyError"
                else "resolve-compiler-semantic-boundary"
            ))
        frontier.append({
            "qualified_name": qualified_name,
            "status": status,
            "action": action,
            "authored_subunits": list(authored_subunits),
            "resolved_process_graph_subunits": list(process_graph_subunits),
            "dependencies": list(dependency_names),
            "minimum_authored_integral": not (
                authored_subunits or process_graph_subunits
            ),
            **({"failure_stage": record["failure_stage"]}
               if record.get("failure_stage") else {}),
            **({"error_type": record["error_type"]}
               if record.get("error_type") else {}),
        })
    return frontier


def encoded_call_name(qualified_name: str) -> str:
    """Encode a qualified source identity reversibly as a safe filename."""

    encoded = []
    for character in str(qualified_name):
        if character.isascii() and (character.isalnum() or character in "-_."):
            encoded.append(character)
        else:
            encoded.append(f"_u{ord(character):06x}_")
    return "".join(encoded)


def authored_definition_sha256(source: str, qualified_name: str) -> str:
    """Hash one exact authored definition, independently of sibling edits."""

    try:
        definition = _authored_definitions(source)[str(qualified_name)]
    except KeyError as error:
        raise ValueError(
            f"source has no authored definition {qualified_name!r}"
        ) from error
    node = definition.node
    start_line = min((
        int(decorator.lineno)
        for decorator in node.decorator_list
    ), default=int(node.lineno))
    end_line = int(node.end_lineno or node.lineno)
    lines = source.splitlines(keepends=True)
    exact = "".join(lines[start_line - 1:end_line])
    return hashlib.sha256(exact.encode("utf-8")).hexdigest()


def native_unit_name(qualified_name: str) -> str:
    """Return a deterministic Fortran-safe module identity (<=63 chars)."""

    source_name = str(qualified_name)
    encoded = encoded_call_name(source_name).replace(".", "__").replace("-", "_")
    candidate = "project_unit__" + encoded
    if len(candidate) <= 63:
        return candidate
    digest = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:16]
    return candidate[:44] + "__" + digest


def _require_native_root_semantics(module: Any, entrypoint: str) -> Any:
    """Return a root only when every required source value was lowered.

    A target compiler accepting an incomplete SSA graph proves only that the
    remainder is syntactically compilable.  It is not evidence that the
    authored unit has been compiled, so publication must stop here and retain
    the source implementation.
    """

    root_function = module.functions.get(str(entrypoint))
    if root_function is None:
        raise RuntimeError(
            f"repository SSA unit has no root function {entrypoint!r}"
        )
    unresolved_required = tuple(root_function.metadata.get(
        "unresolved_required_source_values", ()
    ))
    if unresolved_required:
        unresolved_ids = tuple(int(row[0]) for row in unresolved_required)
        raise RuntimeError(
            "native emission refused: root SSA still requires unlowered "
            f"source values {unresolved_ids}"
        )
    unexplained_arguments = _unexplained_root_argument_ids(root_function)
    if unexplained_arguments:
        raise RuntimeError(
            "native emission refused: root SSA exposes compiler-intermediate "
            f"values as public inputs {unexplained_arguments}"
        )
    return root_function


def _unexplained_root_argument_ids(function: Any) -> tuple[int, ...]:
    """Return formals that have no authored parameter/storage correlation."""

    authored_ids = {
        int(value_id)
        for _name, value_id in function.metadata.get("parameter_names", ())
    }
    authored_projection_ids = {
        int(value_id)
        for value_id, source_name, transform
        in function.metadata.get("scalar_source_transforms", ())
        if str(source_name) and str(transform)
    }
    unexplained = []
    for argument in function.args:
        value_id = int(argument.id)
        accounting = dict(argument.accounting or {})
        if value_id in authored_ids or value_id in authored_projection_ids:
            continue
        if any(accounting.get(key) not in {None, ""} for key in (
            "program_abi_parameter",
            "source_name",
            "source_parameter",
            "sequence_source_name",
            "compiler_frame_storage",
            "linked_call_frame_storage",
            "returned_record_storage",
        )):
            continue
        unexplained.append(value_id)
    return tuple(unexplained)


def _unexplained_root_argument_details(
    function: Any,
) -> tuple[dict[str, Any], ...]:
    """Describe exactly where every unaffiliated root formal is consumed.

    A bare numeric frontier is deterministic but not actionable: an isolated
    worker has already released the ProcessGraph heap by the time a developer
    reads ``unit.json``. Persist the surviving SSA use sites and the small set
    of structural attributes that identify their owner. This remains purely
    diagnostic--it neither blesses the argument nor changes native-publication
    eligibility.
    """

    unexplained = set(_unexplained_root_argument_ids(function))
    arguments = {
        int(argument.id): argument
        for argument in function.args
        if int(argument.id) in unexplained
    }
    uses: dict[int, list[dict[str, Any]]] = {
        value_id: [] for value_id in unexplained
    }
    structural_attributes = (
        "binding", "callee", "region_index", "source_value_id",
        "plan_callsite_id", "semantic_result_id", "tensor_operation",
    )
    for block_name, block in function.blocks.items():
        for instruction_index, instruction in enumerate(block.instrs):
            for argument_index, argument in enumerate(instruction.args):
                value_id = int(argument.id)
                if value_id not in unexplained:
                    continue
                roles = tuple(getattr(instruction, "arg_roles", ()) or ())
                role = (
                    str(roles[argument_index])
                    if argument_index < len(roles) else f"arg:{argument_index}"
                )
                attributes = dict(instruction.attributes or {})
                uses[value_id].append({
                    "block": str(block_name),
                    "instruction_index": int(instruction_index),
                    "operation": str(instruction.op),
                    "role": role,
                    **{
                        key: attributes[key]
                        for key in structural_attributes
                        if key in attributes and isinstance(
                            attributes.get(key),
                            (type(None), bool, int, float, str),
                        )
                    },
                })
    return tuple({
        "value_id": int(value_id),
        "dtype": (
            None if arguments[value_id].dtype is None
            else str(arguments[value_id].dtype)
        ),
        "shape": list(arguments[value_id].shape or ()),
        "accounting": {
            str(key): value
            for key, value in dict(
                arguments[value_id].accounting or {}
            ).items()
            if isinstance(value, (type(None), bool, int, float, str))
            or (
                isinstance(value, (tuple, list))
                and all(isinstance(item, (type(None), bool, int, float, str))
                        for item in value)
            )
        },
        "uses": uses[value_id],
    } for value_id in sorted(unexplained))


def source_region_integral_accounting(
    module: Any,
    outputs: Mapping[str, Sequence[Any]],
) -> tuple[dict[str, Any], ...]:
    """Audit independently compilable planned regions in repository SSA.

    Presence in the hierarchy plan is not completeness.  A region is
    publishable only when its operands are defined by its formal ABI, its
    output values exist, every boundary value has a concrete native type, and
    pointer bases have a physical rank/storage contract.  Calls and retained
    Python references remain with the authored enclosing unit until their own
    exact compartment ABI exists.
    """

    records = []
    for function_name, function in module.functions.items():
        provenance = dict(
            function.metadata.get("source_region_integral") or {}
        )
        if not provenance:
            continue
        shortfalls: list[dict[str, Any]] = []
        if provenance.get("schema") != SOURCE_REGION_INTEGRAL_SCHEMA:
            shortfalls.append({"kind": "missing-structural-provenance"})

        arguments = {int(value.id): value for value in function.args}
        available = set(arguments)
        produced: dict[int, Any] = {}
        undefined_uses: list[tuple[str, int]] = []
        address_base_ids: set[int] = set()
        retained_calls: list[str] = []
        retained_references: list[int] = []
        for block in function.blocks.values():
            for instruction in block.instrs:
                for argument in instruction.args:
                    value_id = int(argument.id)
                    if value_id not in available:
                        undefined_uses.append((str(instruction.op), value_id))
                if (
                    str(instruction.op).casefold() == "getelementptr"
                    and instruction.args
                ):
                    address_base_ids.add(int(instruction.args[0].id))
                if str(instruction.op).casefold() == "call":
                    retained_calls.append(str(
                        instruction.attributes.get("callee") or "<dynamic>"
                    ))
                if str(instruction.op).casefold() == "staticref":
                    retained_references.append(int(
                        instruction.attributes.get("reference_handle", -1)
                    ))
                if instruction.res is not None:
                    result_id = int(instruction.res.id)
                    produced[result_id] = instruction.res
                    available.add(result_id)

        published_outputs = tuple(outputs.get(function_name, ()))
        missing_outputs = tuple(
            int(value.id) for value in published_outputs
            if int(value.id) not in available
        )
        if undefined_uses:
            shortfalls.append({
                "kind": "undefined-operands",
                "occurrences": [
                    {"operation": operation, "value_id": value_id}
                    for operation, value_id in undefined_uses
                ],
            })
        if missing_outputs:
            shortfalls.append({
                "kind": "missing-output-values",
                "value_ids": list(missing_outputs),
            })
        if not published_outputs:
            shortfalls.append({"kind": "no-published-outputs"})

        boundary_values = tuple(function.args) + published_outputs
        unresolved_types = tuple(
            int(value.id) for value in boundary_values
            if str(value.dtype or "").casefold() in {
                "", "none", "unknown", "opaque", "opaque_ref",
            }
        )
        if unresolved_types:
            shortfalls.append({
                "kind": "unresolved-boundary-types",
                "value_ids": list(dict.fromkeys(unresolved_types)),
            })

        sequence_value_ids: set[int] = set()
        sequence_table = getattr(module, "sequence_tables", {}).get(
            function_name
        )
        if sequence_table is not None:
            for descriptor in getattr(sequence_table, "sequences", {}).values():
                sequence_value_ids.update(map(
                    int, descriptor.column_value_ids
                ))
        unresolved_address_bases = []
        for value_id in sorted(address_base_ids):
            value = arguments.get(value_id) or produced.get(value_id)
            accounting = dict(getattr(value, "accounting", None) or {})
            if (
                value is None
                or (
                    not tuple(value.shape or ())
                    and value_id not in sequence_value_ids
                    and str(accounting.get("program_abi_storage") or "")
                    not in {"span", "sequence", "tensor"}
                    and int(accounting.get("program_abi_rank") or 0) <= 0
                    and int(accounting.get("ssa_call_rank") or 0) <= 0
                )
            ):
                unresolved_address_bases.append(value_id)
        if unresolved_address_bases:
            shortfalls.append({
                "kind": "unresolved-address-base-contracts",
                "value_ids": unresolved_address_bases,
            })
        if retained_calls:
            shortfalls.append({
                "kind": "retained-region-calls",
                "callees": retained_calls,
            })
        if retained_references:
            shortfalls.append({
                "kind": "retained-python-references",
                "handles": retained_references,
            })
        unresolved_required = tuple(
            function.metadata.get("unresolved_required_source_values", ())
        )
        if unresolved_required:
            shortfalls.append({
                "kind": "unresolved-required-source-values",
                "values": [list(row) for row in unresolved_required],
            })

        token_chain = tuple(map(
            str, provenance.get("identity_token_chain") or ()
        ))
        if not token_chain:
            shortfalls.append({"kind": "missing-identity-token-chain"})
        records.append({
            "schema": SOURCE_REGION_INTEGRAL_SCHEMA,
            "ssa_function": str(function_name),
            "owner": str(provenance.get("owner") or ""),
            "plan_name": str(provenance.get("plan_name") or ""),
            "region_index": int(provenance.get("region_index", -1)),
            "closure_id": int(provenance.get("closure_id", -1)),
            "identity_token_chain": list(token_chain),
            "inputs": [
                {
                    "value_id": int(value.id),
                    "shape": list(value.shape or ()),
                    "dtype": str(value.dtype or "unknown"),
                }
                for value in function.args
            ],
            "outputs": [
                {
                    "value_id": int(value.id),
                    "shape": list(value.shape or ()),
                    "dtype": str(value.dtype or "unknown"),
                }
                for value in published_outputs
            ],
            "shortfalls": shortfalls,
            "complete": not shortfalls,
        })
    return tuple(sorted(
        records,
        key=lambda record: (
            tuple(record["identity_token_chain"]), record["ssa_function"]
        ),
    ))


def _source_region_module(module: Any, function_name: str) -> Any:
    """Detach one audited source region and its function-scoped ABI tables."""

    from ..transmogrifier.ssa import IRModule

    name = str(function_name)
    return IRModule(
        {name: module.functions[name]},
        recursion_table={
            name: module.recursion_table[name]
            for _ in (0,) if name in module.recursion_table
        },
        deployment_table={
            name: module.deployment_table[name]
            for _ in (0,) if name in module.deployment_table
        },
        tensor_tables={
            name: module.tensor_tables[name]
            for _ in (0,) if name in module.tensor_tables
        },
        sequence_tables={
            name: module.sequence_tables[name]
            for _ in (0,) if name in module.sequence_tables
        },
        record_tables={
            name: module.record_tables[name]
            for _ in (0,) if name in module.record_tables
        },
        reference_tables={
            name: module.reference_tables[name]
            for _ in (0,) if name in module.reference_tables
        },
        call_table={
            name: module.call_table[name]
            for _ in (0,) if name in module.call_table
        },
        metadata={
            "source_region_integral": dict(
                module.functions[name].metadata["source_region_integral"]
            ),
        },
    )


def verify_structural_resident_table_integral(
    module: Any,
    outputs: Mapping[str, Sequence[Any]],
    function_name: str,
    *,
    repository_ssa_sha256: str,
) -> dict[str, Any]:
    """Prove a resident-table mutation integral by executing repository SSA.

    Structural integrals have no numerical return value: their observable ABI
    is the caller-owned table storage they mutate.  Consequently a normal
    output comparison would certify nothing.  This verifier first checks the
    table/helper ABI and then exercises the three exhaustive outcomes of one
    unique-key table store: insert, update, and fixed-capacity rejection.
    """

    import numpy as np

    from .ssa_reference_evaluator import SSAReferenceEvaluator

    name = str(function_name)
    if name not in module.functions:
        raise ValueError(f"structural integral function {name!r} is absent")
    function = module.functions[name]
    contract = dict(function.metadata.get("structural_integral_contract") or {})
    if contract.get("schema") != "turing.structural-resident-table-integral.v1":
        raise ValueError("structural integral has no resident-table contract")
    sequences = tuple(map(dict, contract.get("sequences") or ()))
    stores = tuple(map(dict, contract.get("stores") or ()))
    if len(sequences) != 1 or len(stores) != 1:
        raise ValueError(
            "resident-table verification currently requires exactly one "
            "sequence and one store per deterministic integral"
        )
    if tuple(outputs.get(name, ())):
        raise ValueError("resident-table mutation unexpectedly publishes outputs")

    sequence = sequences[0]
    store = stores[0]
    sequence_id = int(sequence["sequence_id"])
    if int(store["sequence_value_id"]) != sequence_id:
        raise ValueError("store and resident sequence identities disagree")
    root_table = getattr(module, "sequence_tables", {}).get(name)
    descriptor = (
        None if root_table is None
        else root_table.sequences.get(sequence_id)
    )
    if descriptor is None:
        raise ValueError("root function does not publish its resident sequence")
    expected_dtypes = tuple(map(str, sequence.get("column_dtypes") or ()))
    descriptor_contract = {
        "sequence_id": int(descriptor.sequence_id),
        "column_value_ids": list(map(int, descriptor.column_value_ids)),
        "length_address_id": int(descriptor.length_address_id),
        "capacity_value_id": int(descriptor.capacity_value_id),
        "status_address_id": int(descriptor.status_address_id),
        "live_flags_value_id": int(descriptor.live_flags_value_id),
        "column_dtypes": list(map(str, descriptor.column_dtypes)),
        "key_columns": list(map(int, descriptor.key_columns)),
        "capacity_policy": str(descriptor.capacity_policy.value),
        "writable": bool(descriptor.writable),
    }
    if descriptor_contract["sequence_id"] != sequence_id:
        raise ValueError("resident sequence descriptor identity changed")
    if descriptor_contract["column_dtypes"] != list(expected_dtypes):
        raise ValueError("resident sequence column dtypes changed")
    if len(descriptor.column_value_ids) != int(sequence["column_count"]):
        raise ValueError("resident sequence column count changed")
    if tuple(descriptor.key_columns) != (0,):
        raise ValueError("resident table is not a unique first-column mapping")
    if str(descriptor.capacity_policy.value) != "fixed":
        raise ValueError("resident table does not have fixed-capacity semantics")
    if not bool(sequence.get("writable")) or not bool(descriptor.writable):
        raise ValueError("resident table is not writable")

    calls = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if str(instruction.op).casefold() == "call"
        and instruction.attributes.get("ssa_sequence_operation") == "table_store"
        and int(instruction.attributes.get("sequence_id", -1)) == sequence_id
    ]
    if len(calls) != 1:
        raise ValueError("resident-table root must contain one exact store call")
    call = calls[0]
    helper_name = str(call.attributes.get("callee") or "")
    helper = module.functions.get(helper_name)
    helper_table = getattr(module, "sequence_tables", {}).get(helper_name)
    helper_descriptor = (
        None if helper_table is None
        else helper_table.sequences.get(sequence_id)
    )
    if helper is None or helper_descriptor is None:
        raise ValueError("resident-table store helper is not self-contained")
    if len(call.args) != len(helper.args) or any(
        (str(actual.dtype), tuple(actual.shape or ()))
        != (str(formal.dtype), tuple(formal.shape or ()))
        for actual, formal in zip(call.args, helper.args, strict=True)
    ):
        raise ValueError("resident-table helper call ABI is not positional-exact")
    # Caller and helper share the physical table slots.  The final key/value
    # formals are deliberately local to the helper's SSA namespace and are
    # correlated positionally, not by accidentally equal integer ids.
    if tuple(int(value.id) for value in call.args[:6]) != tuple(
        int(value.id) for value in helper.args[:6]
    ):
        raise ValueError("resident-table helper changed its physical table ABI")
    helper_descriptor_contract = {
        **descriptor_contract,
        "column_value_ids": list(map(int, helper_descriptor.column_value_ids)),
        "length_address_id": int(helper_descriptor.length_address_id),
        "capacity_value_id": int(helper_descriptor.capacity_value_id),
        "status_address_id": int(helper_descriptor.status_address_id),
        "live_flags_value_id": int(helper_descriptor.live_flags_value_id),
        "column_dtypes": list(map(str, helper_descriptor.column_dtypes)),
        "key_columns": list(map(int, helper_descriptor.key_columns)),
        "capacity_policy": str(helper_descriptor.capacity_policy.value),
        "writable": bool(helper_descriptor.writable),
    }
    if helper_descriptor_contract != descriptor_contract:
        raise ValueError("root and helper resident-sequence ABIs disagree")
    if helper.metadata.get("ssa_sequence_operation") != "table_store":
        raise ValueError("resident helper does not declare table-store semantics")

    key_id = int(store["key_value_id"])
    value_id = int(store["stored_value_id"])
    arguments = {int(value.id): value for value in function.args}
    if key_id not in arguments or value_id not in arguments:
        raise ValueError("resident store key/value is absent from the root ABI")
    record_identity = sequence.get("value_record")
    value_accounting = dict(arguments[value_id].accounting or {})
    if record_identity is not None and (
        value_accounting.get("structural_record_identity") != record_identity
        or value_accounting.get("structural_record_handle") is not True
    ):
        raise ValueError("stored record handle lost its structural identity")

    numpy_dtypes = {
        "bool": np.bool_, "int": np.int64, "int32": np.int32,
        "int64": np.int64, "float32": np.float32, "float64": np.float64,
    }
    try:
        column_types = tuple(numpy_dtypes[item.casefold()] for item in expected_dtypes)
    except KeyError as error:
        raise ValueError(
            f"no deterministic resident-table probes for dtype {error.args[0]!r}"
        ) from error
    if len(column_types) != 2:
        raise ValueError("mapping verifier requires key and value columns")

    capacity = 3
    columns = [np.zeros(capacity, dtype=dtype) for dtype in column_types]
    length = np.zeros(1, dtype=np.int64)
    status = np.zeros(1, dtype=np.int64)
    live = np.zeros(capacity, dtype=np.bool_)
    feeds = {
        int(value_id): column
        for value_id, column in zip(
            descriptor.column_value_ids, columns, strict=True
        )
    }
    feeds.update({
        int(descriptor.length_address_id): length,
        int(descriptor.capacity_value_id): capacity,
        int(descriptor.status_address_id): status,
        int(descriptor.live_flags_value_id): live,
        key_id: column_types[0](7).item(),
        value_id: column_types[1](42).item(),
    })
    if set(feeds) != set(arguments):
        raise ValueError(
            "resident-table root ABI has unexplained arguments: "
            f"{sorted(set(arguments) - set(feeds))}"
        )
    evaluator = SSAReferenceEvaluator(module)
    probes = []

    evaluator.run(name, feeds)
    if not (
        int(length[0]) == 1 and int(status[0]) == 1
        and columns[0].tolist() == [7, 0, 0]
        and columns[1].tolist() == [42, 0, 0]
        and live.tolist() == [True, False, False]
    ):
        raise ValueError("resident-table insert semantics disagree")
    probes.append({"outcome": "inserted", "status": 1, "length": 1})

    feeds[value_id] = column_types[1](99).item()
    evaluator.run(name, feeds)
    if not (
        int(length[0]) == 1 and int(status[0]) == 3
        and columns[0].tolist() == [7, 0, 0]
        and columns[1].tolist() == [99, 0, 0]
        and live.tolist() == [True, False, False]
    ):
        raise ValueError("resident-table update semantics disagree")
    probes.append({"outcome": "updated", "status": 3, "length": 1})

    columns[0][:] = [1, 2, 3]
    columns[1][:] = [11, 22, 33]
    length[0] = capacity
    status[0] = 0
    live[:] = True
    before = tuple(column.copy() for column in columns)
    evaluator.run(name, feeds)
    if not (
        int(length[0]) == capacity and int(status[0]) == 2
        and all(np.array_equal(left, right) for left, right in zip(columns, before))
        and live.tolist() == [True, True, True]
    ):
        raise ValueError("resident-table capacity rejection semantics disagree")
    probes.append({
        "outcome": "capacity-exhausted", "status": 2,
        "length": capacity, "storage_unchanged": True,
    })

    return {
        "schema": "turing.structural-integral-verification.v1",
        "status": "verified",
        "ssa_function": name,
        "identity_token_chain": list(map(
            str, function.metadata.get("subdivision_identity_token_chain") or (),
        )),
        "repository_ssa_sha256": str(repository_ssa_sha256),
        "probe_count": len(probes),
        "abi": {
            "helper": helper_name,
            "sequence": descriptor_contract,
            "stored_record_identity": record_identity,
        },
        "probes": probes,
    }


def _verify_native_scalar_source_region(
    module: Any,
    outputs: Mapping[str, Sequence[Any]],
    region_record: Mapping[str, Any],
    api_path: Path,
    library_path: Path,
) -> dict[str, Any]:
    """Prove a scalar source-region DLL against repository-SSA execution."""

    import ctypes
    import math
    import re

    import numpy as np

    from .compiled_program_api import load_api
    from .ssa_reference_evaluator import SSAReferenceEvaluator

    name = str(region_record["ssa_function"])
    input_records = tuple(region_record.get("inputs", ()))
    output_records = tuple(region_record.get("outputs", ()))
    if any(len(tuple(item.get("shape") or ())) > 1 for item in input_records):
        raise ValueError("automatic source-region verification is at most rank-1")
    if any(item.get("shape") for item in output_records):
        raise ValueError("automatic source-region verification has scalar outputs")
    descriptor = load_api(api_path)
    entry = next((
        item for item in descriptor.get("entry_points", ())
        if str(item.get("name")) == name
    ), None)
    if entry is None:
        raise ValueError(f"native API does not declare source region {name!r}")
    parameters = tuple(entry.get("parameters") or ())
    native_inputs = tuple(
        item for item in parameters if item.get("role") == "input"
    )
    native_outputs = tuple(
        item for item in parameters if item.get("role") == "output"
    )
    expected_input_names = tuple(
        f"t{int(item['value_id'])}" for item in input_records
    )
    expected_output_names = tuple(
        f"t{int(item['value_id'])}" for item in output_records
    )
    if tuple(str(item.get("name")) for item in native_inputs) != expected_input_names:
        raise ValueError("native source-region input ABI changed during emission")
    if tuple(str(item.get("name")) for item in native_outputs) != expected_output_names:
        raise ValueError("native source-region output ABI changed during emission")
    ctypes_types = {
        "c_int32": ctypes.c_int32,
        "c_int64": ctypes.c_int64,
        "c_float": ctypes.c_float,
        "c_double": ctypes.c_double,
        "c_bool": ctypes.c_bool,
        "c_uint8": ctypes.c_uint8,
    }
    try:
        input_types = tuple(
            ctypes_types[str(item["ctypes"])] for item in native_inputs
        )
        output_types = tuple(
            ctypes_types[str(item["ctypes"])] for item in native_outputs
        )
    except KeyError as error:
        raise ValueError(
            f"unsupported source-region scalar type {error.args[0]!r}"
        ) from error

    runtime_handles = []
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        seen_directories = set()
        for dependency in (
            descriptor.get("metadata", {}).get("runtime_dependencies", ())
        ):
            dependency_path = Path(str(dependency.get("path") or ""))
            if dependency_path.is_file():
                directory = str(dependency_path.parent.resolve())
                if directory not in seen_directories:
                    seen_directories.add(directory)
                    runtime_handles.append(os.add_dll_directory(directory))
    library = ctypes.CDLL(str(library_path.resolve()))
    native = getattr(library, str(entry.get("symbol") or name))
    native.argtypes = [
        (
            ctypes_types[str(parameter["ctypes"])]
            if parameter.get("role") == "extent"
            or (
                parameter.get("role") == "input"
                and not parameter.get("shape")
                and not parameter.get("extents")
                and str(parameter.get("passing")) == "value"
            )
            else ctypes.POINTER(ctypes_types[str(parameter["ctypes"])])
        )
        for parameter in parameters
    ]
    native.restype = None

    samples_by_kind = {
        "bool": (False, True, True),
        "int": (-3, 1, 5),
        "int32": (-3, 1, 5),
        "int64": (-3, 1, 5),
        "float": (-3.25, 1.25, 4.5),
        "float32": (-3.25, 1.25, 4.5),
        "float64": (-3.25, 1.25, 4.5),
    }
    function = module.functions[name]
    input_ids = {int(record["value_id"]) for record in input_records}
    probe_domains: dict[int, dict[str, Any]] = {}
    producers = {
        int(instruction.res.id): instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    }

    def source_input_dependencies(value_id: int) -> set[int]:
        """Return integral inputs that can influence one SSA value.

        Probe restrictions belong to the compiled region's dataflow, not to
        incidental value numbers.  Walking producers also reaches inputs used
        through GetElementPtr/Load projections, which is important for fixed
        tuple fields participating in a divisor.
        """

        pending = [int(value_id)]
        visited: set[int] = set()
        dependencies: set[int] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in input_ids:
                dependencies.add(current)
                continue
            producer = producers.get(current)
            if producer is not None:
                pending.extend(int(argument.id) for argument in producer.args)
        return dependencies

    for block in function.blocks.values():
        for instruction in block.instrs:
            operation = str(instruction.op).casefold()
            if len(instruction.args) < 2:
                continue
            if operation in {"shl", "shr"}:
                count_id = int(instruction.args[1].id)
                if count_id not in input_ids:
                    continue
                record = next(
                    item for item in input_records
                    if int(item["value_id"]) == count_id
                )
                dtype = str(record.get("dtype") or "").casefold()
                bit_width = 64 if dtype in {"int64", "i64", "long"} else 32
                probe_domains[count_id] = {
                    "kind": "shift-count",
                    "minimum": 0,
                    "maximum_exclusive": bit_width,
                    "samples": (0, 1, min(5, bit_width - 1)),
                }
            elif operation in {
                "div", "floordiv", "truediv", "mod", "rem", "remainder",
            }:
                for dependency_id in source_input_dependencies(
                    int(instruction.args[1].id)
                ):
                    existing = probe_domains.get(dependency_id)
                    if existing is not None and existing["kind"] == "shift-count":
                        continue
                    probe_domains[dependency_id] = {
                        "kind": "positive-divisor-dependency",
                        "minimum_exclusive": 0,
                        "samples": (1, 2, 5),
                    }
    evaluator = SSAReferenceEvaluator(module)
    output_values = tuple(outputs.get(name, ()))
    probes = []
    for probe_index in range(3):
        values_by_name: dict[str, Any] = {}
        extent_values: dict[str, int] = {}
        feeds = {}
        keepalive = []
        for record, parameter, native_type in zip(
            input_records, native_inputs, input_types, strict=True
        ):
            dtype = str(record.get("dtype") or "").casefold()
            if dtype not in samples_by_kind:
                raise ValueError(
                    f"no deterministic scalar probes for dtype {dtype!r}"
                )
            shape = tuple(map(int, parameter.get("shape") or ()))
            dynamic_extents = tuple(map(str, parameter.get("extents") or ()))
            if shape or dynamic_extents:
                if shape:
                    element_count = 1
                    for dimension in shape:
                        element_count *= int(dimension)
                elif len(tuple(record.get("shape") or ())) == 1:
                    element_count = int(tuple(record["shape"])[0])
                else:
                    fixed = tuple(
                        int(match.group(1))
                        for extent in dynamic_extents
                        for match in (re.fullmatch(r"extent_([1-9][0-9]*)", extent),)
                        if match is not None
                    )
                    element_count = fixed[0] if len(fixed) == 1 else 8
                element_count = max(int(element_count), 1)
                domain = probe_domains.get(int(record["value_id"]))
                samples = (
                    domain["samples"]
                    if domain is not None else samples_by_kind[dtype]
                )
                payload = [
                    native_type(samples[(probe_index + index) % len(samples)]).value
                    for index in range(element_count)
                ]
                array = (native_type * element_count)(*payload)
                keepalive.append(array)
                values_by_name[str(parameter["name"])] = array
                feeds[int(record["value_id"])] = np.asarray(payload)
                for extent in dynamic_extents:
                    extent_values[extent] = element_count
            else:
                domain = probe_domains.get(int(record["value_id"]))
                value = (
                    domain["samples"][probe_index]
                    if domain is not None
                    else samples_by_kind[dtype][probe_index]
                )
                converted = native_type(value).value
                values_by_name[str(parameter["name"])] = converted
                feeds[int(record["value_id"])] = converted
        expected_state = evaluator.run(name, feeds).values
        result_cells = [kind() for kind in output_types]
        output_cells = {
            str(parameter["name"]): cell
            for parameter, cell in zip(
                native_outputs, result_cells, strict=True
            )
        }
        native_arguments = []
        for parameter in parameters:
            parameter_name = str(parameter["name"])
            if parameter.get("role") == "extent":
                if parameter_name not in extent_values:
                    fixed = re.fullmatch(
                        r"extent_([1-9][0-9]*)", parameter_name
                    )
                    if fixed is None:
                        raise ValueError(
                            f"unresolved source-region extent {parameter_name!r}"
                        )
                    extent_values[parameter_name] = int(fixed.group(1))
                native_arguments.append(extent_values[parameter_name])
            elif parameter.get("role") == "input":
                native_arguments.append(values_by_name[parameter_name])
            elif parameter.get("role") == "output":
                native_arguments.append(ctypes.byref(output_cells[parameter_name]))
            else:
                raise ValueError(
                    f"unsupported source-region parameter role "
                    f"{parameter.get('role')!r}"
                )
        native(*native_arguments)
        expected = tuple(
            expected_state[int(value.id)] for value in output_values
        )
        actual = tuple(cell.value for cell in result_cells)
        disagreements = []
        for record, wanted, observed in zip(
            output_records, expected, actual, strict=True
        ):
            dtype = str(record.get("dtype") or "").casefold()
            if dtype.startswith("float"):
                wanted = float(wanted)
                agrees = math.isclose(
                    float(observed), float(wanted), rel_tol=1e-12, abs_tol=1e-12
                )
            elif dtype == "bool":
                wanted = bool(wanted)
                agrees = type(observed) is bool and observed == wanted
            else:
                wanted = int(wanted)
                agrees = type(observed) is type(wanted) and observed == wanted
            if not agrees:
                disagreements.append({
                    "value_id": int(record["value_id"]),
                    "expected": repr(wanted),
                    "actual": repr(observed),
                })
        if disagreements:
            raise ValueError(
                f"native source-region equivalence failed for {name!r}: "
                f"{disagreements!r}"
            )
        probes.append({
            "inputs": {
                str(record["value_id"]): repr(value)
                for record, parameter in zip(
                    input_records, native_inputs, strict=True
                )
                for value in (feeds[int(record["value_id"])],)
            },
            "outputs": {
                str(record["value_id"]): repr(value)
                for record, value in zip(output_records, actual, strict=True)
            },
        })
    for handle in runtime_handles:
        handle.close()
    return {
        "schema": "turing.native-source-region-verification.v1",
        "status": "verified",
        "ssa_function": name,
        "identity_token_chain": list(region_record["identity_token_chain"]),
        "source_sha256": str(region_record["source_sha256"]),
        "authored_source_sha256": str(
            region_record["authored_source_sha256"]
        ),
        "repository_ssa_sha256": str(region_record["artifact_sha256"]),
        "api_sha256": hashlib.sha256(api_path.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
        "probe_count": len(probes),
        "native_probe_count": len(probes),
        "fallback_probe_count": 0,
        "input_domains": {
            str(value_id): {
                key: value for key, value in domain.items()
                if key != "samples"
            }
            for value_id, domain in sorted(probe_domains.items())
        },
        "probes": probes,
    }


def detach_repository_ssa_frontend(module: Any) -> Any:
    """Remove live frontend heaps from a finished repository-SSA artifact.

    Function references remain in their original deterministic address slots;
    only representations that have completed their job are released.  In
    particular, a resolved ProcessGraph can retain imported Python modules in
    ``python_bindings``. Such modules are neither repository SSA nor pickleable
    deployment state. The authored source hash, extraction receipt, parameter
    contracts, recursion fact, call records, and lowered function bodies remain
    available for linking and source realization.
    """

    table = getattr(module, "function_table", None)
    if table is None:
        return module
    for entry in table:
        entry.graph = None
        entry.python_callable = None
        entry.implementations.clear()
        entry.metadata = {
            str(key): value
            for key, value in dict(entry.metadata or {}).items()
            if isinstance(value, (type(None), bool, int, float, str, bytes))
            or (
                isinstance(value, (tuple, list))
                and all(isinstance(item, (type(None), bool, int, float, str))
                        for item in value)
            )
        }
    module.metadata["frontend_representations_detached"] = True
    return module


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    # Windows refuses replacement while an observer has the destination open.
    # Progress readers are deliberately lock-free, so tolerate that short
    # sharing window without turning telemetry into a compiler failure.
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(value, encoding="utf-8", newline="")
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01)


def _file_sha256(path: Path) -> str:
    """Hash a potentially large worker artifact without retaining it in RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump_resolved_process_graph(graph: Any, stream: Any) -> None:
    """Serialize a reduced graph while preserving importable module bindings.

    ProcessGraph keeps modules such as ``re`` in its compile-time binding
    namespace.  Modules are process state to stock pickle, but their import
    names are deterministic translation necessities.  A private dispatch
    table encodes only that type as ``import_module(name)``; it neither mutates
    the live graph nor changes global pickle behavior.
    """

    pickler = pickle.Pickler(stream, protocol=5)
    pickler.dispatch_table = copyreg.dispatch_table.copy()
    pickler.dispatch_table[types.ModuleType] = lambda module: (
        importlib.import_module,
        (str(module.__name__),),
    )
    pickler.dump(graph)


def _authored_module_record_contracts(
    source_file: Path,
) -> tuple[str | None, dict[tuple[str, str], str]]:
    """Return immutable imported-dataclass contracts for AST ingestion.

    Meta-compilation partitions source text but must not sever explicit type
    and import contracts in doing so.  Resolve the module name only through a
    real package path already present on ``sys.path``; importing that authored
    module is the same Python source realization used to discover the target.
    Only dataclass field/factory facts cross into the ProcessGraph: imported
    implementation objects must not enlarge source pursuit.
    """

    source_file = source_file.resolve()
    candidates: list[tuple[int, str]] = []
    for entry in sys.path:
        try:
            root = Path(entry or os.curdir).resolve()
            relative = source_file.relative_to(root).with_suffix("")
        except (OSError, ValueError):
            continue
        parts = list(relative.parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        if not parts:
            continue
        package_parts = parts if source_file.name == "__init__.py" else parts[:-1]
        if any(
            not (root.joinpath(*parts[:index], "__init__.py")).is_file()
            for index in range(1, len(package_parts) + 1)
        ):
            continue
        candidates.append((len(root.parts), ".".join(parts)))
    if not candidates:
        return None, {}
    module_name = max(candidates)[1]
    module = sys.modules.get(module_name)
    if module is None:
        module = importlib.import_module(module_name)
    contracts: dict[tuple[str, str], str] = {}
    for binding_name, bound_value in vars(module).items():
        if not isinstance(bound_value, type) or not dataclasses.is_dataclass(
            bound_value
        ):
            continue
        for declared_field in dataclasses.fields(bound_value):
            factory = declared_field.default_factory
            kind = next((
                candidate.__name__
                for candidate in (list, set, dict, tuple)
                if factory is candidate
            ), None)
            if kind is not None:
                contracts[(str(binding_name), str(declared_field.name))] = kind
    return module_name, contracts


def compile_project_call(
    source_path: str | Path,
    qualified_name: str,
    directory: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
    extraction_contract: str | Path | None = DEFAULT_PROJECT_EXTRACTION_CONTRACT,
    linked_units: Mapping[str, tuple[str | Path, str]] | None = None,
    linked_regions: Mapping[
        tuple[str, ...], tuple[str | Path, str | Path]
    ] | None = None,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Worker operation: compile and atomically publish one authored call."""

    from ..common.tensors.source_realization import authored_source_realization
    from .fortran_c_shell import lower_ast_source_to_ssa

    source_file = Path(source_path).resolve()
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = source_file.read_text(encoding="utf-8")
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    authored_source_sha256 = authored_definition_sha256(
        source, qualified_name
    )
    linked_units = dict(linked_units or {})
    linked_regions = dict(linked_regions or {})
    compile_source, retained_symbols = partition_authored_source(
        source, qualified_name,
        linked_dependencies=linked_units,
    )
    compile_source_sha256 = hashlib.sha256(
        compile_source.encode("utf-8")
    ).hexdigest()
    compile_source_path = root / "compile-source.py"
    _atomic_text(compile_source_path, compile_source)
    source_module_name, authored_record_contracts = (
        _authored_module_record_contracts(source_file)
    )
    artifact_name = encoded_call_name(qualified_name)
    trace_path = root / "compile-progress.json"
    process_graph_plan_path = root / "process-graph-units.json"
    resolved_process_graph_path = root / "resolved-process-graph.pkl"
    trace_events: list[dict[str, Any]] = []
    trace_started = time.perf_counter()
    last_trace_write = 0.0

    def report(message: str) -> None:
        nonlocal last_trace_write
        elapsed = time.perf_counter() - trace_started
        event = {
            "elapsed_seconds": elapsed,
            "message": str(message),
            "resident_bytes": resident_bytes(os.getpid()),
        }
        graph_detail = str(message).startswith("[graph-build #")
        if not graph_detail or elapsed - last_trace_write >= 1.0:
            trace_events.append(event)
            del trace_events[:-128]
            try:
                _atomic_json(trace_path, {
                    "schema": "turing.project-compilation-unit-progress.v1",
                    "qualified_name": str(qualified_name),
                    "process_id": os.getpid(),
                    "current": event,
                    "events": trace_events,
                })
            except OSError:
                # Compilation is authoritative; tracing is observational.
                pass
            last_trace_write = elapsed
        if progress is not None and (
            not graph_detail
            or str(message).split("#", 1)[-1].split("]", 1)[0].endswith("000")
        ):
            progress(message)

    def publish_process_graph_plan(plan: Mapping[str, Any]) -> None:
        _atomic_json(process_graph_plan_path, {
            **dict(plan),
            "compiler_toolchain": compiler_toolchain_fingerprint(),
        })
        report(
            "unit: published resolved ProcessGraph compilation plan with "
            f"{len(tuple(plan.get('units') or ()))} unit(s)"
        )

    def publish_resolved_process_graph(graph: Any) -> None:
        temporary = resolved_process_graph_path.with_name(
            resolved_process_graph_path.name + ".tmp"
        )
        with temporary.open("wb") as stream:
            _dump_resolved_process_graph(graph, stream)
        os.replace(temporary, resolved_process_graph_path)
        report(
            "unit: published exact reduced ProcessGraph worker input "
            f"({resolved_process_graph_path.stat().st_size} bytes)"
        )

    started = time.perf_counter()
    report("unit: source partition complete")
    linked_repository_ssa = {}
    for dependency_name, (artifact, root_symbol) in sorted(
        linked_units.items()
    ):
        with Path(artifact).resolve().open("rb") as stream:
            dependency_module, dependency_outputs, _dependency_exports = (
                pickle.load(stream)
            )
        linked_repository_ssa[str(dependency_name)] = (
            dependency_module, str(root_symbol), dependency_outputs,
        )
    if linked_repository_ssa:
        report(
            "unit: linked repository dependencies "
            + ", ".join(sorted(linked_repository_ssa))
        )
    linked_source_region_ssa = {}
    for token_chain, (artifact, verification_path) in sorted(
        linked_regions.items(), key=lambda item: tuple(item[0])
    ):
        artifact_path = Path(artifact).resolve()
        verification_file = Path(verification_path).resolve()
        verification = json.loads(
            verification_file.read_text(encoding="utf-8")
        )
        if verification.get("status") != "verified":
            continue
        if tuple(map(
            str, verification.get("identity_token_chain") or ()
        )) != tuple(map(str, token_chain)):
            continue
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != str(
            verification.get("repository_ssa_sha256") or ""
        ):
            continue
        with artifact_path.open("rb") as stream:
            linked_module, linked_outputs, _linked_exports = pickle.load(stream)
        linked_source_region_ssa[tuple(map(str, token_chain))] = (
            linked_module, linked_outputs, verification,
        )
    if linked_source_region_ssa:
        report(
            "unit: available verified source regions "
            + str(len(linked_source_region_ssa))
        )
    with authored_source_realization(targets=(
        call.qualified_name for call in discover_authored_calls(compile_source)
    )):
        module, outputs, exports = lower_ast_source_to_ssa(
            compile_source,
            qualified_name,
            name=artifact_name,
            runtime_closure_only=True,
            progress=report,
            extraction_contract=extraction_contract,
            external_class_field_aggregate_kinds=authored_record_contracts,
            linked_repository_ssa=linked_repository_ssa,
            linked_source_region_ssa=linked_source_region_ssa,
            compilation_unit_plan_sink=publish_process_graph_plan,
            resolved_process_graph_sink=publish_resolved_process_graph,
            stop_after_compilation_unit_plan=bool(plan_only),
        )
    if plan_only:
        receipt = {
            "schema": "turing.project-compilation-unit-plan.v1",
            "qualified_name": str(qualified_name),
            "source": source_file.as_posix(),
            "source_module": source_module_name,
            "source_sha256": source_sha256,
            "authored_source_sha256": authored_source_sha256,
            "compile_source_sha256": compile_source_sha256,
            "compile_source": compile_source_path.name,
            "process_graph_unit_plan": process_graph_plan_path.name,
            "resolved_process_graph": resolved_process_graph_path.name,
            "resolved_process_graph_sha256": _file_sha256(
                resolved_process_graph_path
            ),
            "compiler_toolchain": compiler_toolchain_fingerprint(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(root / "plan-unit.json", receipt)
        return receipt
    return_contract = authored_return_contract(source, qualified_name)
    control_contract = authored_control_contract(source, qualified_name)
    parameter_contract = authored_parameter_contract(source, qualified_name)
    closure_contract = authored_closure_contract(source, qualified_name)
    export_names = tuple(map(str, exports))
    root_symbol = export_names[0] if export_names else None
    root_function = (
        None if root_symbol is None else module.functions.get(root_symbol)
    )
    root_return_value_ids = tuple(
        int(argument.id)
        for block in (() if root_function is None else root_function.blocks.values())
        for instruction in block.instrs
        if str(instruction.op).casefold() in {"ret", "return"}
        for argument in instruction.args
    )
    published_root_outputs = tuple(
        () if root_symbol is None else dict(outputs or {}).get(root_symbol, ())
    )
    return_publication_complete = not return_contract[
        "requires_value_publication"
    ] or bool(root_return_value_ids and published_root_outputs)
    root_metadata = {} if root_function is None else root_function.metadata
    unresolved_required_source_values = tuple(
        tuple(row)
        for row in root_metadata.get(
            "unresolved_required_source_values", ()
        )
    )
    unexplained_root_argument_ids = (
        () if root_function is None
        else _unexplained_root_argument_ids(root_function)
    )
    root_semantic_complete = bool(
        root_function is not None
        and not unresolved_required_source_values
        and not unexplained_root_argument_ids
    )
    represented_parameters = {
        str(source_name)
        for source_name, _value_id in root_metadata.get(
            "parameter_names", ()
        )
    }
    represented_parameters.update(
        str(source_name)
        for _sequence_id, source_names in root_metadata.get(
            "sequence_value_names", ()
        )
        for source_name in source_names
    )
    represented_parameters.update(map(
        str, dict(root_metadata.get("parameter_record_abi") or {})
    ))
    represented_parameters.update(map(
        str, dict(root_metadata.get("parameter_value_abi") or {})
    ))
    represented_parameters.update(
        str(accounting["program_abi_parameter"])
        for value in (() if root_function is None else root_function.args)
        for accounting in (dict(value.accounting or {}),)
        if accounting.get("program_abi_parameter")
    )
    missing_used_parameters = tuple(
        name for name in parameter_contract["used_parameters"]
        if name not in represented_parameters
    )
    missing_lexical_captures = tuple(
        name for name in closure_contract["captures"]
        if name not in represented_parameters
    )
    parameter_publication_complete = not (
        missing_used_parameters or missing_lexical_captures
    )
    unresolved_program_abi_references = tuple(sorted({
        (
            str(accounting.get("program_abi_parameter") or ""),
            str(accounting.get("program_abi_field") or ""),
        )
        for value in (() if root_function is None else root_function.args)
        for accounting in (dict(value.accounting or {}),)
        if accounting.get("program_abi_storage") == "reference"
    }))
    root_instructions = tuple(
        instruction
        for block in (() if root_function is None else root_function.blocks.values())
        for instruction in block.instrs
    )
    validation_count = sum(
        1 for instruction in root_instructions
        if instruction.attributes.get("callee") == "turing_validation_error"
    )
    lowered_loop_ids = {
        int(instruction.attributes["source_loop_node_id"])
        for instruction in root_instructions
        if instruction.attributes.get("source_loop_node_id") is not None
    }
    lowered_break_count = sum(
        1 for instruction in root_instructions
        if instruction.attributes.get("source_control") == "break"
    )
    lowered_continue_count = sum(
        1 for instruction in root_instructions
        if instruction.attributes.get("source_control") == "continue"
    )
    lowered_loop_early_return_count = sum(
        1 for instruction in root_instructions
        if instruction.attributes.get("source_control") == "loop-return"
    )
    source_conditional_count = int(
        ({} if root_function is None else root_function.metadata).get(
            "source_conditional_count", 0
        )
    )
    lowered_conditional_count = int(
        ({} if root_function is None else root_function.metadata).get(
            "lowered_conditional_count", 0
        )
    )
    required_ordinary_conditionals = max(
        0,
        int(control_contract["if_count"])
        - int(control_contract["raise_guard_count"]),
    )
    control_shortfalls = []
    if validation_count < control_contract["raise_guard_count"]:
        control_shortfalls.append("validation-guards")
    if len(lowered_loop_ids) < control_contract["loop_count"]:
        control_shortfalls.append("loops")
    if (
        source_conditional_count < required_ordinary_conditionals
        or lowered_conditional_count < required_ordinary_conditionals
    ):
        control_shortfalls.append("conditionals")
    if lowered_break_count < control_contract["break_count"]:
        control_shortfalls.append("break")
    if lowered_continue_count < control_contract["continue_count"]:
        control_shortfalls.append("continue")
    if (
        lowered_loop_early_return_count
        < control_contract["loop_early_return_count"]
    ):
        control_shortfalls.append("loop-early-return")
    control_complete = not control_shortfalls
    linked_call_counts = {
        root_symbol: 0
        for _dependency, (_module, root_symbol, _outputs)
        in linked_repository_ssa.items()
    }
    for function in module.functions.values():
        for block in function.blocks.values():
            for instruction in block.instrs:
                callee = instruction.attributes.get("callee")
                if callee in linked_call_counts:
                    linked_call_counts[callee] += 1
    missing_linked_calls = tuple(sorted(
        dependency
        for dependency, (_module, root_symbol, _outputs)
        in linked_repository_ssa.items()
        if linked_call_counts[root_symbol] == 0
    ))
    module.metadata["linked_repository_call_accounting"] = {
        "callee_counts": linked_call_counts,
        "missing_dependencies": missing_linked_calls,
        "complete": not missing_linked_calls,
    }
    source_region_integrals = [
        dict(record)
        for record in source_region_integral_accounting(module, outputs or {})
    ]
    for region_record in source_region_integrals:
        region_record["authored_qualified_name"] = str(qualified_name)
        region_record["source_sha256"] = source_sha256
        region_record["authored_source_sha256"] = authored_source_sha256
        region_record["compile_source_sha256"] = compile_source_sha256
        if not region_record["complete"]:
            continue
        region_name = str(region_record["ssa_function"])
        # The parent unit directory already supplies the authored namespace.
        # Use the region's deterministic scoped coordinates on disk and keep
        # the full reversible identity token chain in the receipt. Repeating
        # the qualified owner in every nested pathname can exceed Windows'
        # loader/toolchain path limits without adding any identity information.
        region_root = (
            root / "source-regions"
            / f"closure_{int(region_record['closure_id'])}"
            / f"region_{int(region_record['region_index'])}"
        )
        region_root.mkdir(parents=True, exist_ok=True)
        region_artifact = region_root / "repository-ssa.pkl"
        temporary_region_artifact = region_artifact.with_name(
            region_artifact.name + ".tmp"
        )
        region_module = _source_region_module(module, region_name)
        region_outputs = {
            region_name: tuple((outputs or {}).get(region_name, ()))
        }
        with temporary_region_artifact.open("wb") as stream:
            pickle.dump(
                (region_module, region_outputs, (region_name,)),
                stream,
                protocol=5,
            )
        os.replace(temporary_region_artifact, region_artifact)
        region_record["artifact"] = region_artifact.relative_to(root).as_posix()
        region_record["artifact_sha256"] = hashlib.sha256(
            region_artifact.read_bytes()
        ).hexdigest()
    detach_repository_ssa_frontend(module)
    artifact_path = root / "repository-ssa.pkl"
    temporary_artifact = artifact_path.with_name(artifact_path.name + ".tmp")
    with temporary_artifact.open("wb") as stream:
        pickle.dump((module, outputs, exports), stream, protocol=5)
    os.replace(temporary_artifact, artifact_path)
    receipt = {
        "schema": UNIT_ARTIFACT_SCHEMA,
        "qualified_name": str(qualified_name),
        "source": source_file.as_posix(),
        "source_module": source_module_name,
        "source_sha256": source_sha256,
        "authored_source_sha256": authored_source_sha256,
        "compile_source_sha256": compile_source_sha256,
        "compile_source": compile_source_path.name,
        "process_graph_unit_plan": process_graph_plan_path.name,
        "resolved_process_graph": resolved_process_graph_path.name,
        "resolved_process_graph_sha256": _file_sha256(
            resolved_process_graph_path
        ),
        "retained_source_symbols": list(retained_symbols),
        "linked_repository_dependencies": [
            {
                "qualified_name": str(name),
                "artifact": Path(artifact).resolve().as_posix(),
                "root": str(root_symbol),
            }
            for name, (artifact, root_symbol) in sorted(linked_units.items())
        ],
        "artifact": artifact_path.name,
        "functions": sorted(module.functions),
        "exports": list(exports),
        "elapsed_seconds": time.perf_counter() - started,
        "compilation_unit_plan": dict(
            module.metadata.get("compilation_unit_plan") or {}
        ),
        "extraction_contract": dict(
            module.metadata.get("extraction_contract") or {}
        ),
        "extraction_boundary_accounting": dict(
            module.metadata.get("extraction_boundary_accounting") or {}
        ),
        "repository_ssa_complete": bool(
            (module.metadata.get("extraction_boundary_accounting") or {}).get(
                "repository_ssa_complete", True
            )
        ) and not missing_linked_calls,
        "linked_repository_call_accounting": dict(
            module.metadata.get("linked_repository_call_accounting") or {}
        ),
        "linked_source_region_integrals": list(
            module.metadata.get("linked_source_region_integrals") or ()
        ),
        "source_region_integrals": source_region_integrals,
        "authored_return_contract": return_contract,
        "authored_control_contract": control_contract,
        "authored_parameter_contract": parameter_contract,
        "authored_closure_contract": closure_contract,
        "root_parameter_accounting": {
            "represented_parameters": sorted(represented_parameters),
            "missing_used_parameters": list(missing_used_parameters),
            "missing_lexical_captures": list(missing_lexical_captures),
            "unresolved_reference_fields": [
                {"parameter": parameter, "field": field}
                for parameter, field in unresolved_program_abi_references
            ],
            "complete": parameter_publication_complete,
        },
        "root_semantic_accounting": {
            "unresolved_required_source_values": [
                list(row) for row in unresolved_required_source_values
            ],
            "unexplained_root_argument_ids": list(
                unexplained_root_argument_ids
            ),
            "unexplained_root_argument_details": (
                [] if root_function is None else list(
                    _unexplained_root_argument_details(root_function)
                )
            ),
            "complete": root_semantic_complete,
        },
        "unresolved_program_abi_references": [
            {"parameter": parameter, "field": field}
            for parameter, field in unresolved_program_abi_references
        ],
        "root_return_accounting": {
            "root_symbol": root_symbol,
            "ret_value_ids": list(root_return_value_ids),
            "published_output_value_ids": [
                int(value.id) for value in published_root_outputs
            ],
            "complete": return_publication_complete,
        },
        "root_control_accounting": {
            "validation_count": validation_count,
            "loop_count": len(lowered_loop_ids),
            "source_conditional_count": source_conditional_count,
            "lowered_conditional_count": lowered_conditional_count,
            "break_count": lowered_break_count,
            "continue_count": lowered_continue_count,
            "loop_early_return_count": lowered_loop_early_return_count,
            "shortfalls": control_shortfalls,
            "complete": control_complete,
        },
    }
    receipt["repository_ssa_complete"] = bool(
        receipt["repository_ssa_complete"]
        and return_publication_complete
        and control_complete
        and parameter_publication_complete
        and root_semantic_complete
    )
    _atomic_json(root / "unit.json", receipt)
    return receipt


def compile_resolved_process_graph_unit(
    graph_path: str | Path,
    plan_path: str | Path,
    unit_index: int,
    directory: str | Path,
    *,
    linked_units: Mapping[int, str | Path] | None = None,
    allow_function_shell_cut: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Compile one planned unit without repeating source ingestion/reduction.

    This is deliberately only a compilation product.  Installation remains
    gated on the ordinary semantic and ABI verifiers; a successfully lowered
    unit is never silently promoted to a native compiler replacement.
    """

    from .fortran_c_shell import lower_resolved_process_graph_unit_to_ssa

    resolved_path = Path(graph_path).resolve()
    unit_plan_path = Path(plan_path).resolve()
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = json.loads(unit_plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "turing.compilation-unit-plan.v1":
        raise ValueError("unsupported ProcessGraph compilation-unit plan")
    pinned_toolchain = plan.get("compiler_toolchain")
    if (
        isinstance(pinned_toolchain, Mapping)
        and changed_compiler_toolchain_files(pinned_toolchain)
    ):
        raise ValueError(
            "compiler toolchain changed after ProcessGraph planning; "
            "regenerate the frozen plan before compiling it"
        )
    units = tuple(plan.get("units") or ())
    selected_index = int(unit_index)
    if selected_index < 0 or selected_index >= len(units):
        raise IndexError(
            f"compilation-unit index {selected_index} outside 0..{len(units)-1}"
        )
    selected = dict(units[selected_index])
    references = tuple(map(int, selected.get("function_references") or ()))
    qualified_names = tuple(map(str, selected.get("qualified_names") or ()))
    if not references or not qualified_names:
        raise ValueError("planned ProcessGraph unit has no authored functions")

    started = time.perf_counter()
    if progress is not None:
        progress(
            "resolved-unit: loading post-reduction ProcessGraph "
            f"for unit {selected_index}"
        )
    with resolved_path.open("rb") as stream:
        graph = pickle.load(stream)
    linked_repository_ssa: dict[int, tuple[Any, str, Mapping[str, Any]]] = {}
    linked_unit_receipts = []
    for dependency_index, dependency_root_value in sorted(
        dict(linked_units or {}).items()
    ):
        dependency_root = Path(dependency_root_value).resolve()
        dependency_receipt = json.loads(
            (dependency_root / "unit.json").read_text(encoding="utf-8")
        )
        if dependency_receipt.get("status") != "verified":
            raise ValueError(
                f"resolved unit {int(dependency_index)} is not verified and "
                "cannot be linked"
            )
        dependency_unit = dict(dependency_receipt.get("unit") or {})
        dependency_references = tuple(map(
            int, dependency_unit.get("function_references") or (),
        ))
        dependency_exports = tuple(map(
            str, dependency_receipt.get("exports") or (),
        ))
        artifact = dependency_root / str(dependency_receipt["artifact"])
        with artifact.open("rb") as stream:
            dependency_module, dependency_outputs, stored_exports = pickle.load(
                stream
            )
        if dependency_exports != tuple(map(str, stored_exports)):
            raise ValueError(
                f"resolved unit {int(dependency_index)} export receipt drifted"
            )
        if len(dependency_references) != len(dependency_exports):
            raise ValueError(
                f"resolved unit {int(dependency_index)} has ambiguous "
                "function/export correlation"
            )
        for reference, root_symbol in zip(
            dependency_references, dependency_exports, strict=True,
        ):
            linked_repository_ssa[int(reference)] = (
                dependency_module, str(root_symbol), dependency_outputs,
            )
        linked_unit_receipts.append({
            "unit_index": int(dependency_index),
            "root": dependency_root.as_posix(),
            "artifact_sha256": str(
                dependency_receipt.get("artifact_sha256") or ""
            ),
        })
    retained_dependency_units: set[int] = set()
    pending_dependencies = list(map(
        int, selected.get("dependency_units") or (),
    ))
    while pending_dependencies:
        dependency_index = int(pending_dependencies.pop())
        if dependency_index in retained_dependency_units:
            continue
        retained_dependency_units.add(dependency_index)
        pending_dependencies.extend(map(
            int, units[dependency_index].get("dependency_units") or (),
        ))
    authored_dependency_references = tuple(sorted({
        int(reference)
        for dependency_index in retained_dependency_units
        for reference in units[dependency_index].get("function_references") or ()
        if int(reference) not in linked_repository_ssa
    }))
    artifact_name = encoded_call_name("__".join(qualified_names))
    module, outputs, exports = lower_resolved_process_graph_unit_to_ssa(
        graph,
        references,
        linked_repository_ssa=linked_repository_ssa,
        authored_dependency_references=(
            () if allow_function_shell_cut
            else authored_dependency_references
        ),
        name=artifact_name,
        allow_function_shell_cut=allow_function_shell_cut,
        progress=progress,
    )
    unit_accounting = []
    source_fallback_exports = []
    selected_symbol_references = {
        str(graph.function_table.entry(reference).qualified_name).replace(
            ".", "__"
        ): int(reference)
        for reference in references
    }
    selected_symbol_references.update({
        str(graph.function_table.entry(reference).name): int(reference)
        for reference in references
    })
    for export_name in exports:
        function = module.functions[str(export_name)]
        recorded_reference = function.metadata.get(
            "source_function_reference"
        )
        reference = (
            int(recorded_reference)
            if recorded_reference is not None else (
                references[0] if len(references) == 1 else None
            )
        )
        symbol_tail = str(export_name).removeprefix(
            f"{artifact_name}__"
        )
        symbol_matches = {
            candidate_reference
            for spelling, candidate_reference in selected_symbol_references.items()
            if symbol_tail == spelling
            or symbol_tail.startswith(f"{spelling}__specialized_")
            or str(export_name).endswith(f"__{spelling}")
            or f"__{spelling}__specialized_" in str(export_name)
        }
        if len(symbol_matches) == 1:
            reference = symbol_matches.pop()
        if reference in authored_dependency_references:
            source_fallback_exports.append({
                "function_reference": int(reference),
                "qualified_name": str(
                    graph.function_table.entry(reference).qualified_name
                ),
                "ssa_function": str(export_name),
            })
            continue
        correlated = reference in references
        qualified_name = (
            str(graph.function_table.entry(reference).qualified_name)
            if correlated else "<uncorrelated-export>"
        )
        source_graph = (
            graph.function_table.entry(int(reference)).graph.G
            if correlated else graph.G
        )
        source_conditionals = sum(
            str(data.get("type") or "") in {"If", "IfExp"}
            for _node_id, data in source_graph.nodes(data=True)
        )
        published_outputs = tuple(outputs.get(str(export_name), ()))
        sequence_table = module.sequence_tables.get(str(export_name))
        record_table = module.record_tables.get(str(export_name))
        conceptual_aggregate_ids = {
            *(
                map(int, getattr(sequence_table, "sequences", {}).keys())
                if sequence_table is not None else ()
            ),
            *(
                map(int, getattr(record_table, "records", {}).keys())
                if record_table is not None else ()
            ),
            *(
                int(record_id)
                for record_id, _layout in function.metadata.get(
                    "record_return_layouts", ()
                )
            ),
        }
        unresolved_required = tuple(
            function.metadata.get("unresolved_required_source_values", ())
        )
        structural_shortfalls = tuple(
            function.metadata.get("structural_output_shortfalls", ())
        )
        unresolved_boundary_ids = tuple(sorted({
            int(value.id)
            for value in (*function.args, *published_outputs)
            if str(value.dtype or "").casefold() in {
                "", "none", "unknown", "opaque", "opaque_ref",
            }
            and int(value.id) not in conceptual_aggregate_ids
        }))
        unexplained_arguments = _unexplained_root_argument_ids(function)
        unresolved_calls = tuple(
            {
                "callsite_id": int(record.callsite_id),
                "callee": str(record.callee_symbol or record.callee_name),
            }
            for record in module.call_table.get(str(export_name), ())
            if str(record.resolution) == "unresolved"
        )
        lowered_conditionals = int(
            function.metadata.get("lowered_conditional_count", 0)
        )
        shortfalls = []
        if not correlated:
            shortfalls.append({
                "kind": "missing-function-reference-correlation",
                "ssa_function": str(export_name),
            })
        if unresolved_required:
            shortfalls.append({
                "kind": "unresolved-required-source-values",
                "values": [list(row) for row in unresolved_required],
            })
        if structural_shortfalls:
            shortfalls.append({
                "kind": "structural-output-shortfalls",
                "values": [list(row) for row in structural_shortfalls],
            })
        if unexplained_arguments:
            shortfalls.append({
                "kind": "unexplained-public-arguments",
                "value_ids": list(unexplained_arguments),
            })
        if unresolved_calls:
            shortfalls.append({
                "kind": "unresolved-authored-calls",
                "calls": list(unresolved_calls),
            })
        if unresolved_boundary_ids:
            shortfalls.append({
                "kind": "unresolved-boundary-types",
                "value_ids": list(unresolved_boundary_ids),
            })
        if source_conditionals != lowered_conditionals:
            shortfalls.append({
                "kind": "conditional-accounting-mismatch",
                "source": source_conditionals,
                "lowered": lowered_conditionals,
            })
        unit_accounting.append({
            "qualified_name": str(qualified_name),
            "function_reference": int(reference) if correlated else None,
            "ssa_function": str(export_name),
            "source_conditionals": source_conditionals,
            "lowered_conditionals": lowered_conditionals,
            "published_output_ids": [
                int(value.id) for value in published_outputs
            ],
            "shortfalls": shortfalls,
            "complete": not shortfalls,
        })
    accounted_references = {
        int(accounting["function_reference"])
        for accounting in unit_accounting
        if accounting.get("function_reference") is not None
    }
    for missing_reference in sorted(set(references) - accounted_references):
        unit_accounting.append({
            "qualified_name": str(
                graph.function_table.entry(missing_reference).qualified_name
            ),
            "function_reference": int(missing_reference),
            "ssa_function": None,
            "source_conditionals": None,
            "lowered_conditionals": None,
            "published_output_ids": [],
            "shortfalls": [{
                "kind": "missing-selected-unit-export",
                "function_reference": int(missing_reference),
            }],
            "complete": False,
        })
    if authored_dependency_references:
        for accounting in unit_accounting:
            accounting["shortfalls"].append({
                "kind": "authored-dependency-fallbacks",
                "function_references": list(authored_dependency_references),
            })
            accounting["complete"] = False
    compiled_complete = all(
        record["complete"] for record in unit_accounting
    )
    detach_repository_ssa_frontend(module)
    artifact_path = root / "repository-ssa.pkl"
    temporary = artifact_path.with_name(artifact_path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump((module, outputs, exports), stream, protocol=5)
    os.replace(temporary, artifact_path)
    receipt = {
        "schema": "turing.resolved-process-graph-unit.v1",
        "status": (
            "compiled-unverified" if compiled_complete else "partial"
        ),
        "unit_index": selected_index,
        "unit": selected,
        "qualified_names": list(qualified_names),
        "resolved_process_graph": resolved_path.as_posix(),
        "resolved_process_graph_sha256": _file_sha256(resolved_path),
        "process_graph_unit_plan": unit_plan_path.as_posix(),
        "process_graph_unit_plan_sha256": _file_sha256(unit_plan_path),
        "compiler_toolchain": (
            pinned_toolchain if isinstance(pinned_toolchain, Mapping) else None
        ),
        "artifact": artifact_path.name,
        "artifact_sha256": _file_sha256(artifact_path),
        "functions": sorted(module.functions),
        "exports": list(exports),
        "linked_verified_units": linked_unit_receipts,
        "authored_dependency_fallbacks": list({
            (
                int(record["function_reference"]),
                str(record["qualified_name"]),
                str(record["ssa_function"]),
            ): record
            for record in source_fallback_exports
        }.values()),
        "repository_ssa_accounting": unit_accounting,
        "repository_ssa_complete": compiled_complete,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(root / "unit.json", receipt)
    return receipt


def compile_process_graph_subdivision_integral(
    subdivision_plan_path: str | Path,
    integral_index: int,
    directory: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Compile one deterministic child region from a blocked control owner."""

    plan_path = Path(subdivision_plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "turing.process-graph-subdivision-plan.v1":
        raise ValueError("unsupported ProcessGraph subdivision plan schema")
    integrals = tuple(plan.get("integrals") or ())
    index = int(integral_index)
    if not 0 <= index < len(integrals):
        raise IndexError(
            f"subdivision integral {index} outside 0..{len(integrals) - 1}"
        )
    integral = dict(integrals[index])
    resolved_path = Path(str(plan["resolved_process_graph"])).resolve()
    unit_plan_path = Path(str(plan["process_graph_unit_plan"])).resolve()
    for path, field in (
        (resolved_path, "resolved_process_graph_sha256"),
        (unit_plan_path, "process_graph_unit_plan_sha256"),
    ):
        actual = _file_sha256(path)
        if actual != str(plan.get(field) or ""):
            raise ValueError(f"subdivision source {path} changed after planning")
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def compile_function_shell(
        selected_integral: Mapping[str, Any],
        *,
        subdivision_kind: str = "function-shell",
        routing_reason: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Keep stateful structure on the canonical source/control path.

        A planner region is not necessarily a numerical kernel.  In
        particular, record-field and resident-table mutations are already
        lowered by the complete authored function pipeline; converting such
        a region to ``FusedProgram`` first erases the containing storage ABI.
        The safe integral for that case is therefore the exact authored
        function shell, with same-SCC callees retained as source.
        """

        references = tuple(map(
            int, selected_integral.get("function_references") or (),
        ))
        names = tuple(map(
            str, selected_integral.get("qualified_names") or (),
        ))
        if len(references) != 1 or len(names) != 1:
            raise ValueError(
                "function-shell integral requires one exact authored "
                "function/name correlation"
            )
        source_unit_plan = json.loads(
            unit_plan_path.read_text(encoding="utf-8")
        )
        source_units = list(map(dict, source_unit_plan.get("units") or ()))
        parent_index = int(selected_integral.get("parent_unit_index", -1))
        if not 0 <= parent_index < len(source_units):
            raise ValueError(
                "function-shell integral has no valid parent unit in its "
                "pinned compilation plan"
            )
        parent_unit = dict(source_units[parent_index])
        source_units[parent_index] = {
            **parent_unit,
            "qualified_names": list(names),
            "function_references": list(references),
            # Same-SCC callees deliberately remain authored source, while the
            # parent's external unit closure stays visible to repair ordering
            # and can be linked once those units verify.
            "dynamic_call_nodes": [],
            "external_references": [],
            "recursive": False,
            "source_nodes": None,
        }
        shell_plan_path = root / "function-shell-unit-plan.json"
        _atomic_json(shell_plan_path, {
            **source_unit_plan,
            "units": source_units,
        })
        if progress is not None:
            progress(
                "function-shell: lowering complete selected authored "
                f"function {names[0]} with source callees"
            )
        receipt = compile_resolved_process_graph_unit(
            resolved_path,
            shell_plan_path,
            parent_index,
            root,
            allow_function_shell_cut=True,
            progress=progress,
        )
        receipt.update({
            "schema": "turing.process-graph-subdivision-product.v1",
            "subdivision_plan": plan_path.as_posix(),
            "subdivision_plan_sha256": _file_sha256(plan_path),
            "integral_index": index,
            "integral": dict(selected_integral),
            "qualified_names": list(names),
            "subdivision_kind": str(subdivision_kind),
            **({
                "subdivision_routing": dict(routing_reason),
            } if routing_reason is not None else {}),
        })
        _atomic_json(root / "unit.json", receipt)
        return receipt

    if str(integral.get("kind") or "") == "function-shell":
        return compile_function_shell(integral)
    if progress is not None:
        progress(f"subdivision: loading {tuple(integral['identity_token_chain'])}")
    with resolved_path.open("rb") as stream:
        graph = pickle.load(stream)
    from .fortran_c_shell import (
        extract_resolved_process_graph_subdivision_programs,
    )
    from .precompile_to_ssa import lower_fused_integral_to_repository_ssa

    programs = extract_resolved_process_graph_subdivision_programs(
        graph, integral, progress=progress,
    )
    stateful_operations = tuple(sorted({
        str(step.op_name)
        for captured in programs.values()
        for step in getattr(getattr(captured, "program", captured), "steps", ())
        if str(getattr(step, "op_name", "")).casefold() in {
            "indexedstore", "delitem", "setattr",
        }
    }))
    structurally_typed = all(
        bool((getattr(program, "extras", None) or {}).get(
            "structural_resident_table_contract"
        ))
        for captured in programs.values()
        for program in (getattr(captured, "program", captured),)
        if any(
            str(getattr(step, "op_name", "")).casefold() in {
                "indexedstore", "delitem", "setattr",
            }
            for step in getattr(program, "steps", ())
        )
    )
    if stateful_operations and not structurally_typed:
        if progress is not None:
            progress(
                "subdivision: stateful region has no complete resident "
                "table/record contract and cannot enter numeric lowering; "
                "operations="
                + ",".join(stateful_operations)
            )
    region_receipts = []
    single_region = len(programs) == 1
    for region_index, captured in sorted(programs.items()):
        if progress is not None:
            progress(
                "repository SSA: lowering subdivision region "
                f"{int(region_index)} of {len(programs)}"
            )
        identity_payload = json.dumps(
            [*integral["identity_token_chain"], f"region:{int(region_index)}"],
            separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        function_name = (
            "subdivision_region_"
            + hashlib.sha256(identity_payload).hexdigest()[:20]
        )
        program = getattr(captured, "program", captured)
        module, outputs, exports, shortfalls = (
            lower_fused_integral_to_repository_ssa(
                program, function_name=function_name,
            )
        )
        function = module.functions[function_name]
        function.metadata["source_qualified_name"] = str(
            (integral.get("qualified_names") or ("?",))[0]
        )
        function.metadata["subdivision_identity_token_chain"] = tuple(map(
            str, integral["identity_token_chain"],
        ))
        boundary_values = tuple((
            *function.args,
            *outputs.get(function_name, ()),
        ))
        unresolved_boundary_types = tuple(sorted({
            int(value.id)
            for value in boundary_values
            if str(value.dtype or "unknown").casefold()
            in {"", "unknown", "none", "ssa.aggregate"}
        }))
        boundary_identity_tokens = {
            int(value_id): tuple(map(str, token_chain))
            for value_id, token_chain in function.metadata.get(
                "ssa_identity_tokens", ()
            )
        }
        accounting_shortfalls = [
            {
                "kind": str(item.domain),
                "name": str(item.name),
                "location": str(item.location),
                "reason": str(item.reason),
            }
            for item in shortfalls
        ]
        accounting_shortfalls.extend(map(
            dict,
            (getattr(program, "extras", None) or {}).get(
                "structural_boundary_shortfalls", ()
            ),
        ))
        if unresolved_boundary_types:
            accounting_shortfalls.append({
                "kind": "unresolved-boundary-types",
                "value_ids": list(unresolved_boundary_types),
                "value_identities": [
                    {
                        "value_id": value_id,
                        "identity_token_chain": list(
                            boundary_identity_tokens.get(value_id, ())
                        ),
                    }
                    for value_id in unresolved_boundary_types
                ],
            })
        artifact_name = (
            "repository-ssa.pkl" if single_region
            else f"region-{int(region_index)}-repository-ssa.pkl"
        )
        artifact_path = root / artifact_name
        temporary = artifact_path.with_name(artifact_path.name + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump((module, outputs, exports), stream, protocol=5)
        os.replace(temporary, artifact_path)
        artifact_sha256 = _file_sha256(artifact_path)
        verification = None
        verification_error = None
        if function.metadata.get("structural_integral_kind") == (
            "resident-table-mutation"
        ) and not accounting_shortfalls:
            try:
                verification = verify_structural_resident_table_integral(
                    module,
                    outputs,
                    function_name,
                    repository_ssa_sha256=artifact_sha256,
                )
            except Exception as error:
                verification_error = {
                    "kind": "structural-verification-failed",
                    "reason": f"{type(error).__name__}: {error}",
                }
                accounting_shortfalls.append(verification_error)
            else:
                verification_name = (
                    "repository-verification.json" if single_region
                    else f"region-{int(region_index)}-verification.json"
                )
                _atomic_json(root / verification_name, verification)
        region_receipts.append({
            "region_index": int(region_index),
            "ssa_function": function_name,
            "artifact": artifact_name,
            "artifact_sha256": artifact_sha256,
            "complete": not bool(accounting_shortfalls),
            "shortfalls": accounting_shortfalls,
            "verification_status": (
                str(verification["status"])
                if verification is not None else "compiled-unverified"
            ),
            **({
                "verification": verification_name,
                "verification_sha256": _file_sha256(root / verification_name),
                "probe_count": int(verification["probe_count"]),
            } if verification is not None else {}),
        })
    complete = bool(region_receipts) and all(
        item["complete"] for item in region_receipts
    )
    verified = complete and all(
        item.get("verification_status") == "verified"
        for item in region_receipts
    )
    receipt = {
        "schema": "turing.process-graph-subdivision-product.v1",
        "status": (
            "source-only" if not region_receipts else
            "verified" if verified else
            "compiled-unverified" if complete else "partial"
        ),
        "subdivision_plan": plan_path.as_posix(),
        "subdivision_plan_sha256": _file_sha256(plan_path),
        "integral_index": index,
        "integral": integral,
        "qualified_names": list(integral.get("qualified_names") or ()),
        "regions": region_receipts,
        "repository_ssa_complete": complete,
        **({
            "artifact": region_receipts[0]["artifact"],
            "exports": [region_receipts[0]["ssa_function"]],
            "repository_ssa_accounting": [{
                "qualified_name": str(
                    (integral.get("qualified_names") or ("?",))[0]
                ),
                "complete": bool(region_receipts[0]["complete"]),
                "shortfalls": list(region_receipts[0]["shortfalls"]),
            }],
        } if single_region else ({
            "repository_ssa_accounting": [{
                "qualified_name": str(
                    (integral.get("qualified_names") or ("?",))[0]
                ),
                "complete": False,
                "shortfalls": [{
                    "kind": "no-numeric-regions",
                    "action": "retain-authored-source",
                }],
            }],
        } if not region_receipts else {})),
    }
    _atomic_json(root / "unit.json", receipt)
    return receipt


def _windows_memory_bytes(process_id: int) -> tuple[int, int] | None:
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    handle = kernel32.OpenProcess(0x1000 | 0x0010, False, int(process_id))
    if not handle:
        return None
    try:
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb,
        ):
            return None
        return int(counters.WorkingSetSize), int(counters.PrivateUsage)
    finally:
        kernel32.CloseHandle(handle)


def _posix_memory_bytes(process_id: int) -> tuple[int, int] | None:
    status = Path(f"/proc/{int(process_id)}/status")
    if not status.is_file():
        return None
    fields = {}
    for line in status.read_text(encoding="ascii", errors="replace").splitlines():
        if line.startswith(("VmRSS:", "VmData:", "VmStk:")):
            fields[line.split(":", 1)[0]] = int(line.split()[1]) * 1024
    if "VmRSS" not in fields:
        return None
    private = fields.get("VmData", 0) + fields.get("VmStk", 0)
    return fields["VmRSS"], max(fields["VmRSS"], private)


def process_memory_bytes(process_id: int) -> tuple[int, int] | None:
    """Return resident and committed/private bytes for one compiler worker."""

    return (
        _windows_memory_bytes(process_id)
        if os.name == "nt" else _posix_memory_bytes(process_id)
    )


def resident_bytes(process_id: int) -> int | None:
    measured = process_memory_bytes(process_id)
    return None if measured is None else measured[0]


def publish_process_graph_subdivision_plan(
    directory: str | Path,
    records: Sequence[Mapping[str, Any]],
    resolved_process_graph: str | Path,
    process_graph_unit_plan: str | Path,
) -> dict[str, Any] | None:
    """Turn structured worker refusals into deterministic child integrals."""

    root = Path(directory).resolve()
    resolved_path = Path(resolved_process_graph).resolve()
    unit_plan_path = Path(process_graph_unit_plan).resolve()
    unit_plan = json.loads(unit_plan_path.read_text(encoding="utf-8"))
    planned_units = tuple(map(dict, unit_plan.get("units") or ()))
    reference_to_unit = {
        int(reference): int(index)
        for index, planned_unit in enumerate(planned_units)
        for reference in planned_unit.get("function_references") or ()
    }
    integrals_by_identity: dict[tuple[str, ...], dict[str, Any]] = {}
    for source_record in records:
        record = dict(source_record)
        unit_index = int(record.get("unit_index", -1))
        unit = dict(record.get("unit") or {})
        qualified_names = tuple(map(
            str, unit.get("qualified_names") or record.get("qualified_names") or (),
        ))
        function_references = tuple(map(
            int, unit.get("function_references") or (),
        ))
        resource_phase = str((record.get("stage") or {}).get("phase") or "")
        if (
            str(record.get("error_type") or "") == "ResourceLimitExceeded"
            and resource_phase == "deployment-instantiation"
        ):
            if len(function_references) != len(qualified_names):
                raise ValueError(
                    "resource-bound unit lacks exact function/name "
                    f"correlation: references={function_references!r} "
                    f"names={qualified_names!r}"
                )
            resource = str(record.get("resource") or "resource-limit")
            for reference, qualified_name in zip(
                function_references, qualified_names, strict=True,
            ):
                token_chain = (
                    "process-graph-subdivision",
                    qualified_name,
                    "function-shell",
                )
                integral = {
                    "schema": "turing.process-graph-subdivision-integral.v1",
                    "identity_token_chain": list(token_chain),
                    "parent_unit_index": unit_index,
                    "qualified_names": [qualified_name],
                    "function_references": [reference],
                    "kind": "function-shell",
                    "region_indices": [],
                    "blockers": [
                        f"resource:{resource}",
                        f"phase:{resource_phase}",
                    ],
                }
                existing = integrals_by_identity.get(token_chain)
                if existing is not None and existing != integral:
                    raise ValueError(
                        "conflicting subdivision receipts for deterministic "
                        f"identity {token_chain!r}"
                    )
                integrals_by_identity[token_chain] = integral
            continue
        if record.get("frontier_kind") != "compilation-subdivision-required":
            continue
        for source_boundary in record.get("subdivision_boundaries") or ():
            boundary = dict(source_boundary)
            owner_reference = boundary.get("function_reference")
            owner_reference = (
                None if owner_reference is None else int(owner_reference)
            )
            owner_index = (
                unit_index if owner_reference is None
                else reference_to_unit.get(owner_reference, unit_index)
            )
            owner_unit = (
                dict(planned_units[owner_index])
                if 0 <= owner_index < len(planned_units) else unit
            )
            owner_names = tuple(map(str, (
                (boundary["qualified_name"],)
                if boundary.get("qualified_name") else
                owner_unit.get("qualified_names") or qualified_names
            )))
            owner_references = (
                (owner_reference,) if owner_reference is not None else
                tuple(map(
                    int, owner_unit.get("function_references")
                    or unit.get("function_references") or (),
                ))
            )
            loop_node_id = int(boundary["loop_node_id"])
            region_indices = tuple(sorted(map(
                int, boundary.get("region_indices") or (),
            )))
            token_chain = (
                "process-graph-subdivision",
                *owner_names,
                str(boundary.get("kind") or "source-boundary"),
                f"loop:{loop_node_id}",
                *(f"region:{index}" for index in region_indices),
            )
            integral = {
                "schema": "turing.process-graph-subdivision-integral.v1",
                "identity_token_chain": list(token_chain),
                "parent_unit_index": owner_index,
                "qualified_names": list(owner_names),
                "function_references": list(owner_references),
                "kind": str(boundary.get("kind") or "source-boundary"),
                "loop_node_id": loop_node_id,
                "region_indices": list(region_indices),
                "blockers": list(map(str, boundary.get("blockers") or ())),
            }
            existing = integrals_by_identity.get(token_chain)
            if existing is not None and existing != integral:
                raise ValueError(
                    "conflicting subdivision receipts for deterministic "
                    f"identity {token_chain!r}"
                )
            integrals_by_identity[token_chain] = integral
    integrals = [
        integrals_by_identity[identity]
        for identity in sorted(integrals_by_identity)
    ]
    if not integrals:
        return None
    plan = {
        "schema": "turing.process-graph-subdivision-plan.v1",
        **({
            "compiler_toolchain": unit_plan["compiler_toolchain"],
        } if unit_plan.get("compiler_toolchain") is not None else {}),
        "resolved_process_graph": resolved_path.as_posix(),
        "resolved_process_graph_sha256": _file_sha256(resolved_path),
        "process_graph_unit_plan": unit_plan_path.as_posix(),
        "process_graph_unit_plan_sha256": _file_sha256(unit_plan_path),
        "integrals": integrals,
    }
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "subdivision-integrals.json", plan)
    return plan


def ready_process_graph_unit_indices(
    pending: Sequence[int],
    units: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any] | None],
) -> tuple[int, ...]:
    """Return pending units whose direct dependencies have terminated.

    A dependency does not have to compile successfully before its caller can
    be attempted: partial/failed dependencies deliberately remain authored
    source fallbacks.  It does have to reach a terminal receipt, however, or
    a parallel catalogue pass can race past a dependency that is about to be
    verified and needlessly retain source in the caller.  Waiting only for
    terminal state preserves parallel leaf compilation and makes verified
    native creep deterministic rather than dependent on worker timing.
    """

    ready = []
    for raw_index in pending:
        index = int(raw_index)
        dependencies = tuple(map(
            int, units[index].get("dependency_units") or (),
        ))
        if all(records[dependency] is not None for dependency in dependencies):
            ready.append(index)
    return tuple(ready)


def compile_resolved_process_graph_plan(
    graph_path: str | Path,
    plan_path: str | Path,
    directory: str | Path,
    *,
    python_executable: str | Path = sys.executable,
    jobs: int | None = None,
    max_total_resident_bytes: int | None = None,
    worker_resident_reservation_bytes: int = DEFAULT_WORKER_RESERVATION_BYTES,
    max_worker_memory_bytes: int | None = DEFAULT_WORKER_LIMIT_BYTES,
    unit_timeout_seconds: float | None = DEFAULT_UNIT_TIMEOUT_SECONDS,
    worker_environment: Mapping[str, str] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Crawl a resolved unit plan in isolated, resource-bounded workers.

    Units are launched in the plan's deterministic dependency-first order.
    Verified predecessors are linked when available; incomplete predecessors
    do not prevent exploratory lowering, but can never be linked or mistaken
    for an installable dependency.
    """

    resolved_path = Path(graph_path).resolve()
    unit_plan_path = Path(plan_path).resolve()
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = json.loads(unit_plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "turing.compilation-unit-plan.v1":
        raise ValueError("unsupported ProcessGraph compilation-unit plan")
    pinned_toolchain = plan.get("compiler_toolchain")
    if (
        isinstance(pinned_toolchain, Mapping)
        and changed_compiler_toolchain_files(pinned_toolchain)
    ):
        raise ValueError(
            "compiler toolchain changed after ProcessGraph planning; "
            "regenerate the frozen plan before crawling its units"
        )
    units = tuple(dict(unit) for unit in plan.get("units") or ())
    worker_limit = int(jobs or min(4, os.cpu_count() or 1))
    if worker_limit < 1:
        raise ValueError("resolved plan worker count must be positive")
    reservation = int(worker_resident_reservation_bytes)
    if reservation < 1:
        raise ValueError("resolved plan worker reservation must be positive")
    total_limit = (
        None if max_total_resident_bytes is None
        else int(max_total_resident_bytes)
    )
    worker_limit_bytes = (
        None if max_worker_memory_bytes is None
        else int(max_worker_memory_bytes)
    )
    timeout = (
        None if unit_timeout_seconds is None
        else float(unit_timeout_seconds)
    )
    repository_root = Path(__file__).resolve().parents[2]
    records: list[dict[str, Any] | None] = [None] * len(units)
    existing_progress_path = root / "progress.json"
    if existing_progress_path.is_file():
        existing_progress = json.loads(
            existing_progress_path.read_text(encoding="utf-8")
        )
        if (
            Path(str(existing_progress.get("resolved_process_graph"))).resolve()
            != resolved_path
            or Path(str(existing_progress.get("process_graph_unit_plan"))).resolve()
            != unit_plan_path
        ):
            raise ValueError(
                "existing resolved-plan product belongs to different pinned inputs"
            )
        for index, expected_unit in enumerate(units):
            destination = root / "units" / f"unit_{index:03d}"
            record_path = next((
                candidate for candidate in (
                    destination / "unit.json", destination / "failure.json",
                ) if candidate.is_file()
            ), None)
            if record_path is None:
                continue
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if (
                int(record.get("unit_index", -1)) != index
                or dict(record.get("unit") or {}) != dict(expected_unit)
            ):
                raise ValueError(
                    f"existing terminal receipt for unit {index} does not "
                    "match the pinned unit plan"
                )
            records[index] = record
    pending = [
        index for index, record in enumerate(records) if record is None
    ]
    running: dict[int, dict[str, Any]] = {}

    def report(stage: str, **details: Any) -> None:
        if progress is not None:
            progress({"stage": stage, **details})

    def unit_root(index: int) -> Path:
        return root / "units" / f"unit_{int(index):03d}"

    def write_progress() -> None:
        _atomic_json(root / "progress.json", {
            "schema": "turing.resolved-process-graph-plan-progress.v1",
            "resolved_process_graph": resolved_path.as_posix(),
            "process_graph_unit_plan": unit_plan_path.as_posix(),
            "worker_jobs": worker_limit,
            "max_total_resident_bytes": total_limit,
            "worker_resident_reservation_bytes": reservation,
            "max_worker_memory_bytes": worker_limit_bytes,
            "unit_timeout_seconds": timeout,
            "pending": pending,
            "running": [{
                "unit_index": index,
                "process_id": int(state["process"].pid),
                "resident_bytes": int(state["resident"]),
                "private_bytes": int(state["private"]),
                "elapsed_seconds": time.perf_counter() - state["started"],
            } for index, state in sorted(running.items())],
            "completed": [record for record in records if record is not None],
        })

    def launch(index: int) -> None:
        destination = unit_root(index)
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(python_executable), "-m", "tools.compile_project_catalogue",
            "--resolved-process-graph", str(resolved_path),
            "--process-graph-plan", str(unit_plan_path),
            "--planned-unit", str(index),
            "--output", str(destination),
        ]
        linked = []
        for dependency in map(int, units[index].get("dependency_units") or ()):
            record = records[dependency]
            if record is None or record.get("status") != "verified":
                continue
            dependency_root = unit_root(dependency)
            command.extend((
                "--linked-planned-unit",
                json.dumps({
                    "unit_index": dependency,
                    "root": dependency_root.as_posix(),
                }, sort_keys=True),
            ))
            linked.append(dependency)
        worker_log = (destination / "worker.log").open(
            "w", encoding="utf-8", newline="\n",
        )
        process = subprocess.Popen(
            command,
            cwd=repository_root,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            env=(None if worker_environment is None else dict(worker_environment)),
        )
        running[index] = {
            "process": process,
            "started": time.perf_counter(),
            "resident": 0,
            "private": 0,
            "linked": linked,
            "worker_log": worker_log,
        }
        report(
            "resolved_unit_start", unit_index=index,
            qualified_names=units[index].get("qualified_names", ()),
            linked_verified_units=linked,
        )

    def stop_for_resource(index: int, kind: str, detail: str) -> None:
        state = running[index]
        process = state["process"]
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        state["worker_log"].close()
        destination = unit_root(index)
        durable_stage = None
        progress_path = destination / "compile-progress.json"
        if progress_path.is_file():
            try:
                durable_stage = json.loads(
                    progress_path.read_text(encoding="utf-8")
                ).get("current")
            except (OSError, TypeError, ValueError):
                durable_stage = None
        failure = {
            "schema": "turing.resolved-process-graph-unit-failure.v1",
            "status": "failed",
            "unit_index": index,
            "unit": units[index],
            "error_type": "ResourceLimitExceeded",
            "resource": kind,
            "error": detail,
            "elapsed_seconds": time.perf_counter() - state["started"],
            "peak_resident_bytes": int(state["resident"]),
            "peak_private_bytes": int(state["private"]),
            **({"stage": durable_stage} if durable_stage else {}),
        }
        _atomic_json(destination / "failure.json", failure)
        records[index] = failure
        del running[index]
        report(
            "resolved_unit_resource_failure",
            unit_index=index,
            status="failed",
            resource=kind,
            error=detail,
            resource_stage=durable_stage,
            elapsed_seconds=failure["elapsed_seconds"],
            peak_resident_bytes=failure["peak_resident_bytes"],
            peak_private_bytes=failure["peak_private_bytes"],
        )

    write_progress()
    while pending or running:
        measured_total = sum(
            int(state["resident"]) for state in running.values()
        )
        while pending and len(running) < worker_limit and (
            not running
            or total_limit is None
            or measured_total + reservation <= total_limit
        ):
            ready = ready_process_graph_unit_indices(pending, units, records)
            if not ready:
                if running:
                    break
                blocked = {
                    int(index): tuple(map(
                        int, units[int(index)].get("dependency_units") or (),
                    ))
                    for index in pending
                }
                raise ValueError(
                    "ProcessGraph unit plan has no dependency-ready unit; "
                    f"blocked={blocked!r}"
                )
            index = int(ready[0])
            pending.remove(index)
            launch(index)
            measured_total += reservation
        time.sleep(0.1)
        for index, state in tuple(running.items()):
            process = state["process"]
            measured = process_memory_bytes(process.pid)
            if measured is not None:
                resident, private = measured
                state["resident"] = max(int(state["resident"]), int(resident))
                state["private"] = max(int(state["private"]), int(private))
            elapsed = time.perf_counter() - state["started"]
            if (
                worker_limit_bytes is not None
                and int(state["private"]) > worker_limit_bytes
            ):
                stop_for_resource(
                    index, "private-memory",
                    f"private memory {int(state['private'])} exceeded "
                    f"{worker_limit_bytes} bytes",
                )
                continue
            if timeout is not None and elapsed > timeout:
                stop_for_resource(
                    index, "elapsed-time",
                    f"elapsed time {elapsed:.3f}s exceeded {timeout:.3f}s",
                )
                continue
            return_code = process.poll()
            if return_code is None:
                continue
            state["worker_log"].close()
            destination = unit_root(index)
            receipt_path = destination / "unit.json"
            failure_path = destination / "failure.json"
            if return_code == 0 and receipt_path.is_file():
                record = json.loads(receipt_path.read_text(encoding="utf-8"))
            elif failure_path.is_file():
                record = json.loads(failure_path.read_text(encoding="utf-8"))
            else:
                record = {
                    "schema": "turing.resolved-process-graph-unit-failure.v1",
                    "status": "failed",
                    "unit_index": index,
                    "unit": units[index],
                    "error_type": "WorkerExit",
                    "error": f"worker exited with code {return_code}",
                }
                _atomic_json(failure_path, record)
            record["peak_resident_bytes"] = int(state["resident"])
            record["peak_private_bytes"] = int(state["private"])
            record["linked_verified_units"] = list(state["linked"])
            records[index] = record
            del running[index]
            report(
                "resolved_unit_finish", unit_index=index,
                status=record.get("status"),
                elapsed_seconds=record.get("elapsed_seconds"),
            )
        write_progress()

    completed_records = [dict(record) for record in records if record is not None]
    subdivision_plan = publish_process_graph_subdivision_plan(
        root, completed_records, resolved_path, unit_plan_path,
    )
    subdivision_integrals = (
        [] if subdivision_plan is None
        else list(subdivision_plan["integrals"])
    )
    subdivision_plan_path = root / "subdivision-integrals.json"
    manifest = {
        "schema": "turing.resolved-process-graph-plan-product.v1",
        "resolved_process_graph": resolved_path.as_posix(),
        "resolved_process_graph_sha256": _file_sha256(resolved_path),
        "process_graph_unit_plan": unit_plan_path.as_posix(),
        "process_graph_unit_plan_sha256": _file_sha256(unit_plan_path),
        "worker_jobs": worker_limit,
        "units": completed_records,
        "counts": {
            status: sum(record.get("status") == status for record in completed_records)
            for status in ("verified", "compiled-unverified", "partial", "failed")
        },
        "subdivision_integral_count": len(subdivision_integrals),
        **({
            "subdivision_integrals": subdivision_plan_path.name,
            "subdivision_integrals_sha256": _file_sha256(subdivision_plan_path),
        } if subdivision_integrals else {}),
    }
    _atomic_json(root / "manifest.json", manifest)
    return manifest


def compile_process_graph_subdivision_plan(
    plan_path: str | Path,
    directory: str | Path,
    *,
    python_executable: str | Path = sys.executable,
    jobs: int | None = None,
    max_total_resident_bytes: int | None = None,
    worker_resident_reservation_bytes: int = DEFAULT_WORKER_RESERVATION_BYTES,
    max_worker_memory_bytes: int | None = DEFAULT_WORKER_LIMIT_BYTES,
    unit_timeout_seconds: float | None = DEFAULT_UNIT_TIMEOUT_SECONDS,
    worker_environment: Mapping[str, str] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compile every deterministic child integral in bounded workers."""

    source_plan_path = Path(plan_path).resolve()
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    if source_plan.get("schema") != "turing.process-graph-subdivision-plan.v1":
        raise ValueError("unsupported ProcessGraph subdivision plan schema")
    integrals = tuple(map(dict, source_plan.get("integrals") or ()))
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    worker_limit = int(jobs or min(4, os.cpu_count() or 1))
    reservation = int(worker_resident_reservation_bytes)
    if worker_limit < 1 or reservation < 1:
        raise ValueError("subdivision worker count/reservation must be positive")
    total_limit = (
        None if max_total_resident_bytes is None
        else int(max_total_resident_bytes)
    )
    worker_limit_bytes = (
        None if max_worker_memory_bytes is None
        else int(max_worker_memory_bytes)
    )
    timeout = (
        None if unit_timeout_seconds is None
        else float(unit_timeout_seconds)
    )
    repository_root = Path(__file__).resolve().parents[2]
    pending = list(range(len(integrals)))
    running: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any] | None] = [None] * len(integrals)

    def report(stage: str, **details: Any) -> None:
        if progress is not None:
            progress({"stage": stage, **details})

    def integral_root(index: int) -> Path:
        return root / "integrals" / f"integral_{int(index):03d}"

    def write_progress() -> None:
        _atomic_json(root / "progress.json", {
            "schema": "turing.process-graph-subdivision-progress.v1",
            "subdivision_plan": source_plan_path.as_posix(),
            "subdivision_plan_sha256": _file_sha256(source_plan_path),
            "worker_jobs": worker_limit,
            "max_total_resident_bytes": total_limit,
            "worker_resident_reservation_bytes": reservation,
            "max_worker_memory_bytes": worker_limit_bytes,
            "unit_timeout_seconds": timeout,
            "pending": pending,
            "running": [{
                "integral_index": index,
                "process_id": int(state["process"].pid),
                "resident_bytes": int(state["resident"]),
                "private_bytes": int(state["private"]),
                "elapsed_seconds": time.perf_counter() - state["started"],
            } for index, state in sorted(running.items())],
            "completed": [record for record in records if record is not None],
        })

    def launch(index: int) -> None:
        destination = integral_root(index)
        destination.mkdir(parents=True, exist_ok=True)
        worker_log = (destination / "worker.log").open(
            "w", encoding="utf-8", newline="\n",
        )
        process = subprocess.Popen([
            str(python_executable), "-m", "tools.compile_project_catalogue",
            "--subdivision-plan", str(source_plan_path),
            "--subdivision-integral", str(index),
            "--output", str(destination),
        ], cwd=repository_root, stdout=worker_log, stderr=subprocess.STDOUT,
            env=(None if worker_environment is None else dict(worker_environment)))
        running[index] = {
            "process": process,
            "started": time.perf_counter(),
            "resident": 0,
            "private": 0,
            "worker_log": worker_log,
        }
        report(
            "subdivision_integral_start", integral_index=index,
            identity_token_chain=integrals[index].get(
                "identity_token_chain", (),
            ),
        )

    def stop_for_resource(index: int, kind: str, detail: str) -> None:
        state = running[index]
        process = state["process"]
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        state["worker_log"].close()
        destination = integral_root(index)
        durable_stage = None
        progress_path = destination / "compile-progress.json"
        if progress_path.is_file():
            try:
                durable_stage = json.loads(
                    progress_path.read_text(encoding="utf-8")
                ).get("current")
            except (OSError, TypeError, ValueError):
                pass
        failure = {
            "schema": "turing.process-graph-subdivision-failure.v1",
            "status": "failed",
            "integral_index": index,
            "integral": integrals[index],
            "qualified_names": list(
                integrals[index].get("qualified_names") or ()
            ),
            "error_type": "ResourceLimitExceeded",
            "resource": kind,
            "error": detail,
            "elapsed_seconds": time.perf_counter() - state["started"],
            "peak_resident_bytes": int(state["resident"]),
            "peak_private_bytes": int(state["private"]),
            **({"stage": durable_stage} if durable_stage else {}),
        }
        _atomic_json(destination / "failure.json", failure)
        records[index] = failure
        del running[index]
        report("subdivision_integral_resource_failure", **failure)

    write_progress()
    while pending or running:
        measured_total = sum(
            int(state["resident"]) for state in running.values()
        )
        while pending and len(running) < worker_limit and (
            not running or total_limit is None
            or measured_total + reservation <= total_limit
        ):
            index = pending.pop(0)
            launch(index)
            measured_total += reservation
        time.sleep(0.1)
        for index, state in tuple(running.items()):
            process = state["process"]
            measured = process_memory_bytes(process.pid)
            if measured is not None:
                resident, private = measured
                state["resident"] = max(int(state["resident"]), int(resident))
                state["private"] = max(int(state["private"]), int(private))
            elapsed = time.perf_counter() - state["started"]
            if (
                worker_limit_bytes is not None
                and int(state["private"]) > worker_limit_bytes
            ):
                stop_for_resource(
                    index, "private-memory",
                    f"private memory {int(state['private'])} exceeded "
                    f"{worker_limit_bytes} bytes",
                )
                continue
            if timeout is not None and elapsed > timeout:
                stop_for_resource(
                    index, "elapsed-time",
                    f"elapsed time {elapsed:.3f}s exceeded {timeout:.3f}s",
                )
                continue
            return_code = process.poll()
            if return_code is None:
                continue
            state["worker_log"].close()
            destination = integral_root(index)
            receipt_path = destination / "unit.json"
            failure_path = destination / "failure.json"
            if return_code == 0 and receipt_path.is_file():
                record = json.loads(receipt_path.read_text(encoding="utf-8"))
            elif failure_path.is_file():
                record = json.loads(failure_path.read_text(encoding="utf-8"))
            else:
                record = {
                    "schema": "turing.process-graph-subdivision-failure.v1",
                    "status": "failed",
                    "integral_index": index,
                    "integral": integrals[index],
                    "error_type": "WorkerExit",
                    "error": f"worker exited with code {return_code}",
                }
                _atomic_json(failure_path, record)
            record["peak_resident_bytes"] = int(state["resident"])
            record["peak_private_bytes"] = int(state["private"])
            records[index] = record
            del running[index]
            report(
                "subdivision_integral_finish", integral_index=index,
                status=record.get("status"),
            )
        write_progress()

    completed = [dict(record) for record in records if record is not None]
    nested_subdivision = (
        None if not completed else publish_process_graph_subdivision_plan(
            root,
            completed,
            source_plan["resolved_process_graph"],
            source_plan["process_graph_unit_plan"],
        )
    )
    nested_integrals = (
        [] if nested_subdivision is None
        else list(nested_subdivision.get("integrals") or ())
    )
    statuses = (
        "compiled-unverified", "partial", "source-only", "failed",
    )
    manifest = {
        "schema": "turing.process-graph-subdivision-product-plan.v1",
        "subdivision_plan": source_plan_path.as_posix(),
        "subdivision_plan_sha256": _file_sha256(source_plan_path),
        "worker_jobs": worker_limit,
        "integrals": completed,
        "counts": {
            status: sum(record.get("status") == status for record in completed)
            for status in statuses
        },
        "subdivision_integral_count": len(nested_integrals),
        **({
            "subdivision_integrals": "subdivision-integrals.json",
            "subdivision_integrals_sha256": _file_sha256(
                root / "subdivision-integrals.json"
            ),
        } if nested_integrals else {}),
    }
    _atomic_json(root / "manifest.json", manifest)
    return manifest


def compile_process_graph_creep(
    graph_path: str | Path,
    plan_path: str | Path,
    directory: str | Path,
    *,
    python_executable: str | Path = sys.executable,
    jobs: int | None = None,
    max_total_resident_bytes: int | None = None,
    worker_resident_reservation_bytes: int = DEFAULT_WORKER_RESERVATION_BYTES,
    max_worker_memory_bytes: int | None = DEFAULT_WORKER_LIMIT_BYTES,
    unit_timeout_seconds: float | None = DEFAULT_UNIT_TIMEOUT_SECONDS,
    max_subdivision_depth: int = 32,
    bootstrap_products: Iterable[str | Path] = (),
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Autonomously crawl bounded units and every strictly deeper cut.

    The ordinary resolved/subdivision crawlers deliberately seal one level at
    a time.  This driver is the missing compiler-owned feedback loop: terminal
    worker receipts become the next subdivision plan without requiring a
    person to select an index or invoke another command.  Plan identities are
    remembered across the complete run, so a resource-bound minimum integral
    cannot enqueue itself forever; that fixed point is published as an exact
    terminal frontier.

    Verified artifacts are inventoried but never promoted merely because they
    compiled.  Installation remains receipt-gated by the compiler bootstrap
    runtime, and workers inherit whatever verified products that runtime has
    activated through its environment.
    """

    resolved_path = Path(graph_path).resolve()
    unit_plan_path = Path(plan_path).resolve()
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    depth_limit = int(max_subdivision_depth)
    if depth_limit < 0:
        raise ValueError("maximum subdivision depth must be non-negative")
    selected_bootstrap_products = tuple(dict.fromkeys(
        Path(path).resolve() for path in bootstrap_products
    ))
    worker_environment = os.environ.copy()
    if selected_bootstrap_products:
        from .compiler_bootstrap_runtime import COMPILER_BOOTSTRAP_PRODUCTS_ENV

        worker_environment[COMPILER_BOOTSTRAP_PRODUCTS_ENV] = os.pathsep.join(
            str(path) for path in selected_bootstrap_products
        )

    def report(stage: str, **details: Any) -> None:
        if progress is not None:
            progress({"stage": stage, **details})

    def plan_identity(path: Path, kind: str) -> tuple[str, tuple[tuple[str, ...], ...]]:
        source = json.loads(path.read_text(encoding="utf-8"))
        if kind == "resolved":
            identities = tuple(
                tuple(map(str, unit.get("qualified_names") or ()))
                for unit in source.get("units") or ()
            )
        else:
            identities = tuple(sorted(
                tuple(map(str, integral.get("identity_token_chain") or ()))
                for integral in source.get("integrals") or ()
            ))
        digest = hashlib.sha256(json.dumps(
            {"kind": kind, "identities": identities},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        return digest, identities

    queue: list[dict[str, Any]] = [{
        "kind": "resolved",
        "plan": unit_plan_path,
        "depth": 0,
        "parent_round": None,
    }]
    seen: dict[str, dict[str, Any]] = {}
    rounds: list[dict[str, Any]] = []
    fixed_points: list[dict[str, Any]] = []
    verified_products: list[dict[str, Any]] = []

    def write_progress() -> None:
        _atomic_json(root / "creep-progress.json", {
            "schema": "turing.compiler-creep-progress.v1",
            "resolved_process_graph": resolved_path.as_posix(),
            "process_graph_unit_plan": unit_plan_path.as_posix(),
            "max_subdivision_depth": depth_limit,
            "pending": [{
                **item,
                "plan": Path(item["plan"]).as_posix(),
            } for item in queue],
            "rounds": rounds,
            "fixed_points": fixed_points,
            "verified_products": verified_products,
        })

    write_progress()
    while queue:
        item = queue.pop(0)
        kind = str(item["kind"])
        source_plan = Path(item["plan"]).resolve()
        depth = int(item["depth"])
        identity, identities = plan_identity(source_plan, kind)
        previous = seen.get(identity)
        if previous is not None:
            fixed_points.append({
                "kind": "repeated-subdivision-plan",
                "plan_kind": kind,
                "plan_identity": identity,
                "depth": depth,
                "first_depth": int(previous["depth"]),
                "identity_token_chains": [list(chain) for chain in identities],
                "action": (
                    "add-a-smaller-complete-integral-boundary-or-lowering-abi"
                ),
            })
            write_progress()
            continue
        if depth > depth_limit:
            fixed_points.append({
                "kind": "maximum-subdivision-depth",
                "plan_kind": kind,
                "plan_identity": identity,
                "depth": depth,
                "identity_token_chains": [list(chain) for chain in identities],
                "action": "raise-depth-only-after-proving-strictly-smaller-cuts",
            })
            write_progress()
            continue
        seen[identity] = {"depth": depth, "plan": source_plan.as_posix()}
        round_index = len(rounds)
        round_root = root / (
            f"round_{round_index:03d}_resolved"
            if kind == "resolved" else
            f"round_{round_index:03d}_subdivision_{identity[:12]}"
        )
        report(
            "creep_round_start", round=round_index, kind=kind,
            depth=depth, plan=source_plan.as_posix(),
        )
        common = {
            "python_executable": python_executable,
            "jobs": jobs,
            "max_total_resident_bytes": max_total_resident_bytes,
            "worker_resident_reservation_bytes": (
                worker_resident_reservation_bytes
            ),
            "max_worker_memory_bytes": max_worker_memory_bytes,
            "unit_timeout_seconds": unit_timeout_seconds,
            "worker_environment": worker_environment,
            "progress": lambda event, round_index=round_index: report(
                "creep_worker", round=round_index, event=dict(event),
            ),
        }
        if kind == "resolved":
            product = compile_resolved_process_graph_plan(
                resolved_path, source_plan, round_root, **common,
            )
            product_records = tuple(product.get("units") or ())
            child_name = product.get("subdivision_integrals")
        else:
            product = compile_process_graph_subdivision_plan(
                source_plan, round_root, **common,
            )
            product_records = tuple(product.get("integrals") or ())
            child_name = product.get("subdivision_integrals")
        for record_index, record_value in enumerate(product_records):
            record = dict(record_value)
            if record.get("status") != "verified":
                continue
            compartment_root = (
                round_root / "units" / f"unit_{record_index:03d}"
                if kind == "resolved" else
                round_root / "integrals" / f"integral_{record_index:03d}"
            )
            verified_products.append({
                "round": round_index,
                "kind": kind,
                "root": compartment_root.as_posix(),
                "qualified_names": list(
                    record.get("qualified_names")
                    or (record.get("unit") or {}).get("qualified_names")
                    or ()
                ),
                "identity_token_chain": list(
                    (record.get("integral") or {}).get(
                        "identity_token_chain", ()
                    )
                ),
                "status": "verified",
            })
        round_record = {
            "round": round_index,
            "kind": kind,
            "depth": depth,
            "plan": source_plan.as_posix(),
            "plan_identity": identity,
            "product": round_root.as_posix(),
            "counts": dict(product.get("counts") or {}),
            "child_integral_count": int(
                product.get("subdivision_integral_count") or 0
            ),
        }
        rounds.append(round_record)
        if child_name:
            child_plan = round_root / str(child_name)
            queue.append({
                "kind": "subdivision",
                "plan": child_plan,
                "depth": depth + 1,
                "parent_round": round_index,
            })
        report("creep_round_finish", **round_record)
        write_progress()

    counts: dict[str, int] = {}
    for round_record in rounds:
        for status, count in dict(round_record.get("counts") or {}).items():
            counts[str(status)] = counts.get(str(status), 0) + int(count)
    manifest = {
        "schema": "turing.compiler-creep-product.v1",
        "resolved_process_graph": resolved_path.as_posix(),
        "resolved_process_graph_sha256": _file_sha256(resolved_path),
        "process_graph_unit_plan": unit_plan_path.as_posix(),
        "process_graph_unit_plan_sha256": _file_sha256(unit_plan_path),
        "max_subdivision_depth": depth_limit,
        "bootstrap_products": [
            path.as_posix() for path in selected_bootstrap_products
        ],
        "rounds": rounds,
        "counts": counts,
        "verified_products": verified_products,
        "fixed_points": fixed_points,
        "status": "frontier" if fixed_points or counts.get("failed", 0) else "sealed",
    }
    _atomic_json(root / "manifest.json", manifest)
    return manifest


def dependency_ordered_records(
    root: Path,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Order completed worker records by their resolved unit dependencies."""

    selected = {str(record["qualified_name"]) for record in records}
    dependencies: dict[str, set[str]] = {name: set() for name in selected}
    for record in records:
        if record["status"] != "complete":
            continue
        receipt_path = root / str(record["path"]) / "unit.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        units = list(
            (receipt.get("compilation_unit_plan") or {}).get("units") or ()
        )
        unit_by_name = {
            str(name): index
            for index, unit in enumerate(units)
            for name in unit.get("qualified_names", ())
        }
        root_name = str(record["qualified_name"])
        root_index = unit_by_name.get(root_name)
        if root_index is None:
            continue
        pending_units = list(units[root_index].get("dependency_units", ()))
        visited_units: set[int] = set()
        while pending_units:
            unit_index = int(pending_units.pop())
            if unit_index in visited_units or not 0 <= unit_index < len(units):
                continue
            visited_units.add(unit_index)
            dependency_unit = units[unit_index]
            dependencies[root_name].update(
                str(name)
                for name in dependency_unit.get("qualified_names", ())
                if str(name) in selected
            )
            pending_units.extend(dependency_unit.get("dependency_units", ()))

    ordered_names: list[str] = []
    visited_names: set[str] = set()
    active_names: set[str] = set()

    def order(name: str) -> None:
        if name in visited_names:
            return
        if name in active_names:
            return
        active_names.add(name)
        for dependency in sorted(dependencies.get(name, ())):
            order(dependency)
        active_names.remove(name)
        visited_names.add(name)
        ordered_names.append(name)

    for name in sorted(selected):
        order(name)
    by_name = {str(record["qualified_name"]): dict(record) for record in records}
    return [by_name[name] for name in ordered_names]


def compile_project_product(
    source_path: str | Path,
    directory: str | Path,
    *,
    entries: Iterable[str] | None = None,
    expand_entry_dependencies: bool = True,
    python_executable: str | Path = sys.executable,
    jobs: int | None = None,
    max_total_resident_bytes: int | None = None,
    worker_resident_reservation_bytes: int = DEFAULT_WORKER_RESERVATION_BYTES,
    max_worker_memory_bytes: int | None = DEFAULT_WORKER_LIMIT_BYTES,
    unit_timeout_seconds: float | None = DEFAULT_UNIT_TIMEOUT_SECONDS,
    extraction_contract: str | Path | None = DEFAULT_PROJECT_EXTRACTION_CONTRACT,
    emit_native: bool = False,
    seed_product: str | Path | None = None,
    bootstrap_products: Iterable[str | Path] = (),
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compile authored calls in isolated workers across several cores.

    ``jobs`` bounds concurrently resident compiler workers.  The optional
    aggregate resident-memory limit is an admission ceiling. Each running
    worker reserves ``worker_resident_reservation_bytes`` before launch;
    observed RSS is also recorded, but is not mistaken for a finished heap's
    eventual size. A per-worker committed-memory limit prevents page-thrashing
    compiles from monopolizing the machine; a time limit bounds code integrals
    whose planning does not converge. Both limits publish durable failures so
    a later creep pass can divide those exact authored calls more deeply.
    """

    source_file = Path(source_path).resolve()
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = source_file.read_text(encoding="utf-8")
    inherited_bootstrap_products = tuple(dict.fromkeys(
        Path(path).resolve() for path in bootstrap_products
    ))
    seed_root = None if seed_product is None else Path(seed_product).resolve()
    if seed_root is not None and seed_root.is_file():
        if seed_root.name != "manifest.json":
            raise ValueError(
                "seed product file must be its manifest.json or a product directory"
            )
        seed_root = seed_root.parent
    seed_regions_by_parent: dict[str, list[dict[str, Any]]] = {}
    if seed_root is not None:
        if not (seed_root / "links.json").is_file():
            raise ValueError(
                f"seed project has no link table: {seed_root / 'links.json'}"
            )
        seed_links = json.loads(
            (seed_root / "links.json").read_text(encoding="utf-8")
        )
        if seed_links.get("schema") != LINK_TABLE_SCHEMA:
            raise ValueError("seed project has an unsupported link schema")
        for link in seed_links.get("links", ()):
            if (
                link.get("kind") != "source-region-integral"
                or not link.get("native_verification")
            ):
                continue
            parent = str(link.get("authored_qualified_name") or "")
            if parent:
                seed_regions_by_parent.setdefault(parent, []).append(dict(link))
    authored_source_path = root / "authored-source.py"
    _atomic_text(authored_source_path, source)
    discovered = discover_authored_calls(source)
    by_name = {call.qualified_name: call for call in discovered}
    requested_names = (
        tuple(call.qualified_name for call in discovered)
        if entries is None
        else tuple(dict.fromkeys(map(str, entries)))
    )
    selected_names = (
        dependency_ordered_authored_calls(source, requested_names)
        if expand_entry_dependencies else requested_names
    )
    unknown = tuple(name for name in selected_names if name not in by_name)
    if unknown:
        raise ValueError(f"unknown authored project calls: {unknown!r}")
    source_dependencies = authored_call_dependencies(source, selected_names)
    contained_integrals = {name: [] for name in selected_names}
    for candidate in selected_names:
        if ".<locals>." not in candidate:
            continue
        parent = candidate.rsplit(".<locals>.", 1)[0]
        if parent in contained_integrals:
            contained_integrals[parent].append(candidate)
    index_by_name = {
        name: index for index, name in enumerate(selected_names)
    }
    worker_limit = int(jobs or min(4, os.cpu_count() or 1))
    if worker_limit < 1:
        raise ValueError("project compilation jobs must be positive")
    memory_ceiling = (
        None
        if max_total_resident_bytes is None
        else int(max_total_resident_bytes)
    )
    if memory_ceiling is not None and memory_ceiling < 1:
        raise ValueError("project compilation memory ceiling must be positive")
    worker_reservation = int(worker_resident_reservation_bytes)
    if worker_reservation < 1:
        raise ValueError("project compilation worker reservation must be positive")
    worker_memory_limit = (
        None if max_worker_memory_bytes is None else int(max_worker_memory_bytes)
    )
    if worker_memory_limit is not None and worker_memory_limit < 1:
        raise ValueError("project compilation worker memory limit must be positive")
    timeout_seconds = (
        None if unit_timeout_seconds is None else float(unit_timeout_seconds)
    )
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("project compilation unit timeout must be positive")
    extraction_contract_path = (
        None
        if extraction_contract is None
        else Path(extraction_contract).resolve()
    )
    if (
        extraction_contract_path is not None
        and not extraction_contract_path.is_file()
    ):
        raise FileNotFoundError(extraction_contract_path)

    def report(stage: str, **details: Any) -> None:
        if progress is not None:
            progress({"stage": stage, **details})

    records: list[dict[str, Any] | None] = [None] * len(selected_names)
    pending = list(enumerate(selected_names))
    running: dict[int, dict[str, Any]] = {}
    last_progress_write = 0.0

    def write_live_progress() -> None:
        def running_record(state: Mapping[str, Any]) -> dict[str, Any]:
            stage_path = Path(state["root"]) / "compile-progress.json"
            stage = None
            if stage_path.is_file():
                try:
                    stage = json.loads(
                        stage_path.read_text(encoding="utf-8")
                    ).get("current")
                except (OSError, ValueError, TypeError):
                    # A worker publishes by atomic replace, but a filesystem
                    # observer can still race deletion or external tooling.
                    # Stage telemetry is diagnostic and must never obstruct
                    # the compilation it describes.
                    stage = None
            return {
                "qualified_name": state["qualified_name"],
                "process_id": int(state["process"].pid),
                "resident_bytes": int(state["resident"]),
                "peak_resident_bytes": int(state["peak_resident"]),
                "private_bytes": int(state["private"]),
                "peak_private_bytes": int(state["peak_private"]),
                **({"stage": stage} if stage is not None else {}),
            }

        _atomic_json(root / "progress.json", {
            "schema": "turing.project-compilation-progress.v1",
            "source": source_file.as_posix(),
            "worker_jobs": worker_limit,
            "max_total_resident_bytes": memory_ceiling,
            "worker_resident_reservation_bytes": worker_reservation,
            "max_worker_memory_bytes": worker_memory_limit,
            "unit_timeout_seconds": timeout_seconds,
            "extraction_contract": (
                None
                if extraction_contract_path is None
                else extraction_contract_path.as_posix()
            ),
            "pending": [name for _index, name in pending],
            "running": [
                running_record(state)
                for _index, state in sorted(running.items())
            ],
            "completed": [record for record in records if record is not None],
        })

    def launch(index: int, qualified_name: str) -> None:
        unit_root = root / "units" / encoded_call_name(qualified_name)
        unit_root.mkdir(parents=True, exist_ok=True)
        report(
            "unit_start", qualified_name=qualified_name,
            index=index, count=len(selected_names),
        )
        command = [
            str(python_executable), "-m", "tools.compile_project_catalogue",
            "--worker", "--source", str(source_file),
            "--entry", qualified_name, "--output", str(unit_root),
        ]
        if extraction_contract_path is not None:
            command.extend((
                "--extraction-contract", str(extraction_contract_path),
            ))
        for dependency_name in source_dependencies.get(qualified_name, ()):
            dependency_record = records[index_by_name[dependency_name]]
            if (
                dependency_record is None
                or dependency_record.get("status") != "complete"
            ):
                # A mutually recursive SCC has no dependency-first member.
                # Its first worker retains the cycle's authored bodies; later
                # members can link whatever that SCC successfully publishes.
                continue
            dependency_root = root / str(dependency_record["path"])
            dependency_receipt = json.loads(
                (dependency_root / "unit.json").read_text(encoding="utf-8")
            )
            exports = tuple(map(str, dependency_receipt.get("exports", ())))
            if len(exports) != 1:
                raise RuntimeError(
                    f"linked unit {dependency_name!r} must publish exactly one "
                    f"root export, got {exports!r}"
                )
            command.extend((
                "--linked-unit",
                json.dumps({
                    "qualified_name": dependency_name,
                    "artifact": str(
                        dependency_root / dependency_receipt["artifact"]
                    ),
                    "root": exports[0],
                }, sort_keys=True),
            ))
        current_authored_hash = authored_definition_sha256(
            source, qualified_name
        )
        for region in sorted(
            seed_regions_by_parent.get(qualified_name, ()),
            key=lambda item: tuple(item.get("identity_token_chain") or ()),
        ):
            if str(region.get("authored_source_sha256") or "") != current_authored_hash:
                continue
            command.extend((
                "--linked-region",
                json.dumps({
                    "identity_token_chain": list(
                        region["identity_token_chain"]
                    ),
                    "artifact": str(seed_root / str(region["artifact"])),
                    "verification": str(
                        seed_root / str(region["native_verification"])
                    ),
                }, sort_keys=True),
            ))
        worker_environment = os.environ.copy()
        if inherited_bootstrap_products:
            from .compiler_bootstrap_runtime import (
                COMPILER_BOOTSTRAP_PRODUCTS_ENV,
            )

            worker_environment[COMPILER_BOOTSTRAP_PRODUCTS_ENV] = os.pathsep.join(
                str(path) for path in inherited_bootstrap_products
            )
        process = subprocess.Popen(command, env=worker_environment)
        running[index] = {
            "qualified_name": qualified_name,
            "root": unit_root,
            "process": process,
            "started": time.perf_counter(),
            "peak_resident": 0,
            "resident": 0,
            "peak_private": 0,
            "private": 0,
            "resource_failure": None,
        }
        write_live_progress()

    def finish(index: int, state: Mapping[str, Any]) -> None:
        qualified_name = str(state["qualified_name"])
        unit_root = Path(state["root"])
        process = state["process"]
        elapsed = time.perf_counter() - float(state["started"])
        unit_receipt_path = unit_root / "unit.json"
        process_graph_plan_path = unit_root / "process-graph-units.json"
        process_graph_plan = (
            json.loads(process_graph_plan_path.read_text(encoding="utf-8"))
            if process_graph_plan_path.is_file() else None
        )
        process_graph_subunits = sorted({
            str(name)
            for unit in (
                () if process_graph_plan is None
                else process_graph_plan.get("units") or ()
            )
            for name in unit.get("qualified_names") or ()
            if str(name) != qualified_name
        })
        if process.returncode == 0 and unit_receipt_path.is_file():
            unit_receipt = json.loads(unit_receipt_path.read_text(encoding="utf-8"))
            repository_complete = bool(
                unit_receipt.get("repository_ssa_complete", True)
            )
            record = {
                "qualified_name": qualified_name,
                "status": "complete" if repository_complete else "partial",
                "path": unit_root.relative_to(root).as_posix(),
                "peak_resident_bytes": int(state["peak_resident"]),
                "peak_private_bytes": int(state["peak_private"]),
                "elapsed_seconds": elapsed,
                "exports": unit_receipt.get("exports", []),
                "repository_ssa_complete": repository_complete,
                "source_region_integrals": list(
                    unit_receipt.get("source_region_integrals", ())
                ),
                **({
                    "process_graph_unit_plan": (
                        process_graph_plan_path.relative_to(root).as_posix()
                    ),
                    "process_graph_subunits": process_graph_subunits,
                } if process_graph_plan is not None else {}),
                **({
                    "unmaterialized_extraction_boundaries": len(
                        (unit_receipt.get("extraction_boundary_accounting") or {}).get(
                            "unmaterialized", ()
                        )
                    ),
                    "unresolved_call_count": len(
                        (unit_receipt.get("extraction_boundary_accounting") or {}).get(
                            "unresolved_call_records", ()
                        )
                    ),
                    "root_return_publication_complete": bool(
                        (unit_receipt.get("root_return_accounting") or {}).get(
                            "complete", True
                        )
                    ),
                    "missing_used_parameters": list(
                        (
                            unit_receipt.get("root_parameter_accounting")
                            or {}
                        ).get("missing_used_parameters", ())
                    ),
                    "missing_lexical_captures": list(
                        (
                            unit_receipt.get("root_parameter_accounting")
                            or {}
                        ).get("missing_lexical_captures", ())
                    ),
                    "unresolved_required_source_values": list(
                        (
                            unit_receipt.get("root_semantic_accounting")
                            or {}
                        ).get("unresolved_required_source_values", ())
                    ),
                    "unresolved_program_abi_references": list(
                        unit_receipt.get(
                            "unresolved_program_abi_references", ()
                        )
                    ),
                    "control_shortfalls": list(
                        (unit_receipt.get("root_control_accounting") or {}).get(
                            "shortfalls", ()
                        )
                    ),
                    "control_frontier_action": next((
                        f"repair-{shortfall}-control-lowering"
                        for shortfall in (
                            unit_receipt.get("root_control_accounting") or {}
                        ).get("shortfalls", ())
                    ), None),
                } if not repository_complete else {}),
            }
        else:
            failure_path = unit_root / "failure.json"
            failure = (
                json.loads(failure_path.read_text(encoding="utf-8"))
                if failure_path.is_file() else {}
            )
            record = {
                "qualified_name": qualified_name,
                "status": "failed",
                "path": unit_root.relative_to(root).as_posix(),
                "peak_resident_bytes": int(state["peak_resident"]),
                "peak_private_bytes": int(state["peak_private"]),
                "elapsed_seconds": elapsed,
                "returncode": int(process.returncode),
                **({
                    "error_type": failure.get("error_type"),
                    "error": failure.get("error"),
                    "failure_stage": failure.get("stage"),
                    "failure": failure_path.relative_to(root).as_posix(),
                    **({
                        "frontier_kind": failure.get("frontier_kind"),
                        "subdivision_boundaries": list(
                            failure.get("subdivision_boundaries") or ()
                        ),
                    } if failure.get("frontier_kind") else {}),
                } if failure else {}),
                **({
                    "process_graph_unit_plan": (
                        process_graph_plan_path.relative_to(root).as_posix()
                    ),
                    "process_graph_subunits": process_graph_subunits,
                } if process_graph_plan is not None else {}),
            }
        records[index] = record
        report("unit_complete", **record, index=index, count=len(selected_names))

    while pending or running:
        measured_total = sum(int(state["resident"]) for state in running.values())
        reserved_total = len(running) * worker_reservation
        admitted = False
        while pending and len(running) < worker_limit and (
            memory_ceiling is None
            or not running
            or (
                reserved_total + worker_reservation <= memory_ceiling
                and measured_total < memory_ceiling
            )
        ):
            ready_position = None
            for position, (candidate_index, candidate_name) in enumerate(pending):
                dependency_records = [
                    records[index_by_name[dependency]]
                    for dependency in source_dependencies.get(candidate_name, ())
                ]
                # A terminal dependency is sufficient to launch the caller.
                # Complete dependencies are linked in ``launch``; partial or
                # failed dependencies remain as their exact authored bodies in
                # the caller's source partition.  Blocking the caller here
                # discarded the next semantic frontier and prevented the
                # catalogue from creeping past one unsupported leaf.
                if all(
                    record is not None
                    for record in dependency_records
                ):
                    ready_position = position
                    break
            if ready_position is None and not running and pending:
                # The remaining dependency graph is cyclic. Compile one exact
                # authored SCC member with its cycle bodies retained instead
                # of deadlocking the scheduler or inventing an invalid cut.
                ready_position = 0
            if ready_position is None:
                break
            index, qualified_name = pending.pop(ready_position)
            launch(index, qualified_name)
            admitted = True
            # With a memory ceiling, sample the newly launched process before
            # admitting another. Without one, fill every available core now.
            if memory_ceiling is not None:
                break
        completed = []
        for index, state in running.items():
            process = state["process"]
            current = process_memory_bytes(process.pid)
            if current is not None:
                resident, private = current
                state["resident"] = int(resident)
                state["peak_resident"] = max(
                    int(state["peak_resident"]), int(resident),
                )
                state["private"] = int(private)
                state["peak_private"] = max(
                    int(state["peak_private"]), int(private),
                )
            elapsed = time.perf_counter() - float(state["started"])
            limit_reason = None
            if (
                worker_memory_limit is not None
                and int(state["private"]) > worker_memory_limit
            ):
                limit_reason = (
                    "committed/private memory "
                    f"{int(state['private'])} exceeded {worker_memory_limit} bytes"
                )
            elif timeout_seconds is not None and elapsed > timeout_seconds:
                limit_reason = (
                    f"elapsed time {elapsed:.3f}s exceeded "
                    f"{timeout_seconds:.3f}s"
                )
            if limit_reason is not None and process.poll() is None:
                stage_path = Path(state["root"]) / "compile-progress.json"
                stage = None
                if stage_path.is_file():
                    try:
                        stage = json.loads(
                            stage_path.read_text(encoding="utf-8")
                        ).get("current")
                    except (OSError, ValueError, TypeError):
                        stage = None
                failure = {
                    "schema": "turing.project-compilation-failure.v1",
                    "qualified_name": state["qualified_name"],
                    "error_type": "ResourceLimitExceeded",
                    "error": limit_reason,
                    "stage": stage,
                    "resident_bytes": int(state["resident"]),
                    "private_bytes": int(state["private"]),
                    "elapsed_seconds": elapsed,
                }
                _atomic_json(Path(state["root"]) / "failure.json", failure)
                state["resource_failure"] = failure
                process.terminate()
            if process.poll() is not None:
                completed.append(index)
        for index in completed:
            state = running.pop(index)
            finish(index, state)
        now = time.monotonic()
        if completed or admitted or now - last_progress_write >= 5.0:
            write_live_progress()
            last_progress_write = now
        if running and not completed:
            time.sleep(0.1)
        elif pending and not running and not admitted:
            # The first worker is always admitted even under a ceiling; this
            # branch is defensive against a future admission-policy change.
            time.sleep(0.1)

    completed_records = dependency_ordered_records(
        root, [record for record in records if record is not None],
    )

    if emit_native:
        from .fortran_c_shell import compile_fortran_module_c_shell
        from .ssa_fortran_backend import emit_module

        for record in completed_records:
            if record.get("status") != "complete":
                continue
            unit_root = root / str(record["path"])
            unit_receipt = json.loads(
                (unit_root / "unit.json").read_text(encoding="utf-8")
            )
            artifact_path = unit_root / str(unit_receipt["artifact"])
            try:
                with artifact_path.open("rb") as stream:
                    module, outputs, exports = pickle.load(stream)
                export_names = tuple(map(str, exports))
                if not export_names:
                    raise RuntimeError("repository SSA unit has no root export")
                source_suffix = str(record["qualified_name"]).replace(".", "__")
                entrypoint = next((
                    name for name in export_names
                    if name.endswith("__" + source_suffix)
                ), export_names[0])
                _require_native_root_semantics(module, entrypoint)
                # Files may retain reversible dotted source spelling, but a
                # Fortran module identifier may not. Keep the source identity
                # in the receipt and use a deterministic target-safe symbol.
                native_name = native_unit_name(str(record["qualified_name"]))
                emitted = emit_module(
                    module,
                    name=native_name,
                    outputs=outputs,
                    extra_roots=export_names,
                    progress=lambda message, qualified_name=record["qualified_name"]: report(
                        "native_emit",
                        qualified_name=qualified_name,
                        message=message,
                    ),
                )
                if not emitted.complete:
                    raise RuntimeError(
                        "native Fortran emission has semantic shortfalls: "
                        + "; ".join(
                            shortfall.format()
                            for shortfall in emitted.shortfalls
                        )
                    )
                executable = compile_fortran_module_c_shell(
                    emitted,
                    {},
                    unit_root / "native",
                    entrypoint=entrypoint,
                    name=native_name,
                    library=True,
                )
                record["native_status"] = "complete"
                record["native_library"] = executable.executable_path.relative_to(
                    root
                ).as_posix()
                record["native_api"] = executable.api_path.relative_to(
                    root
                ).as_posix()
                record["native_entrypoint"] = entrypoint
            except Exception as error:
                failure_path = unit_root / "native-failure.json"
                _atomic_json(failure_path, {
                    "schema": "turing.project-native-emission-failure.v1",
                    "qualified_name": record["qualified_name"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                record["native_status"] = "failed"
                record["native_failure"] = failure_path.relative_to(
                    root
                ).as_posix()

        # A partial authored method can still contain complete planner-owned
        # source integrals. Emit those compartments independently; never make
        # their success upgrade the enclosing method's status.
        for record in completed_records:
            unit_root = root / str(record["path"])
            unit_receipt_path = unit_root / "unit.json"
            if not unit_receipt_path.is_file():
                continue
            unit_receipt = json.loads(
                unit_receipt_path.read_text(encoding="utf-8")
            )
            region_records = [
                dict(item)
                for item in unit_receipt.get("source_region_integrals", ())
            ]
            changed = False
            for region_record in region_records:
                if not region_record.get("complete"):
                    continue
                region_name = str(region_record["ssa_function"])
                artifact_path = unit_root / str(region_record["artifact"])
                native_root = artifact_path.parent / "native"
                try:
                    with artifact_path.open("rb") as stream:
                        region_module, region_outputs, region_exports = (
                            pickle.load(stream)
                        )
                    _require_native_root_semantics(region_module, region_name)
                    native_name = native_unit_name(
                        "::".join(region_record["identity_token_chain"])
                    )
                    emitted = emit_module(
                        region_module,
                        name=native_name,
                        outputs=region_outputs,
                        extra_roots=tuple(map(str, region_exports)),
                        progress=lambda message, qualified_name=record["qualified_name"], region_name=region_name: report(
                            "native_region_emit",
                            qualified_name=qualified_name,
                            region=region_name,
                            message=message,
                        ),
                    )
                    executable = compile_fortran_module_c_shell(
                        emitted,
                        {},
                        native_root,
                        entrypoint=region_name,
                        name=native_name,
                        library=True,
                    )
                    verification = _verify_native_scalar_source_region(
                        region_module,
                        region_outputs,
                        region_record,
                        executable.api_path,
                        executable.executable_path,
                    )
                    verification_path = (
                        executable.executable_path.parent
                        / "native-verification.json"
                    )
                    _atomic_json(verification_path, verification)
                    region_record["native_status"] = "complete"
                    region_record["native_verification_status"] = "verified"
                    region_record["native_verification"] = (
                        verification_path.relative_to(unit_root).as_posix()
                    )
                    region_record["native_library"] = (
                        executable.executable_path.relative_to(unit_root).as_posix()
                    )
                    region_record["native_api"] = (
                        executable.api_path.relative_to(unit_root).as_posix()
                    )
                    region_record["native_entrypoint"] = region_name
                except Exception as error:
                    failure_path = native_root / "native-failure.json"
                    failure_path.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_json(failure_path, {
                        "schema": "turing.project-native-region-emission-failure.v1",
                        "qualified_name": record["qualified_name"],
                        "ssa_function": region_name,
                        "identity_token_chain": region_record[
                            "identity_token_chain"
                        ],
                        "error_type": type(error).__name__,
                        "error": str(error),
                    })
                    region_record["native_status"] = "failed"
                    region_record["native_failure"] = (
                        failure_path.relative_to(unit_root).as_posix()
                    )
                changed = True
            if changed:
                unit_receipt["source_region_integrals"] = region_records
                _atomic_json(unit_receipt_path, unit_receipt)
                record["source_region_integrals"] = region_records

    manifest = {
        "schema": PROJECT_PRODUCT_SCHEMA,
        "source": source_file.as_posix(),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "authored_source": authored_source_path.name,
        "calls": [by_name[name].to_mapping() for name in selected_names],
        "worker_jobs": worker_limit,
        "max_total_resident_bytes": memory_ceiling,
        "worker_resident_reservation_bytes": worker_reservation,
        "max_worker_memory_bytes": worker_memory_limit,
        "unit_timeout_seconds": timeout_seconds,
        "extraction_contract": (
            None
            if extraction_contract_path is None
            else extraction_contract_path.as_posix()
        ),
        "native_emission_requested": bool(emit_native),
        "seed_product": None if seed_root is None else seed_root.as_posix(),
        "bootstrap_products": [
            path.as_posix() for path in inherited_bootstrap_products
        ],
        "units": completed_records,
        "creep_frontier": compilation_creep_frontier(
            completed_records, source_dependencies, contained_integrals,
        ),
    }
    _atomic_json(root / "manifest.json", manifest)
    links = []
    for record in completed_records:
        if record["status"] != "complete":
            continue
        unit_root = root / str(record["path"])
        receipt = json.loads((unit_root / "unit.json").read_text(encoding="utf-8"))
        links.append({
            "qualified_name": str(record["qualified_name"]),
            "source_module": str(receipt.get("source_module") or ""),
            "authored_source_sha256": str(
                receipt.get("authored_source_sha256") or ""
            ),
            "artifact": (
                Path(str(record["path"])) / str(receipt["artifact"])
            ).as_posix(),
            "exports": list(receipt.get("exports", ())),
            **({
                "native_library": record["native_library"],
                "native_api": record["native_api"],
                "native_entrypoint": record["native_entrypoint"],
            } if record.get("native_status") == "complete" else {}),
        })
    for record in completed_records:
        unit_path = Path(str(record["path"]))
        for region in record.get("source_region_integrals", ()):
            if not region.get("complete") or not region.get("artifact"):
                continue
            links.append({
                "qualified_name": str(region["ssa_function"]),
                "kind": "source-region-integral",
                "authored_qualified_name": str(record["qualified_name"]),
                "authored_source_sha256": str(
                    region.get("authored_source_sha256") or ""
                ),
                "identity_token_chain": list(
                    region["identity_token_chain"]
                ),
                "artifact": (
                    unit_path / str(region["artifact"])
                ).as_posix(),
                "exports": [str(region["ssa_function"])],
                **({
                    "native_library": (
                        unit_path / str(region["native_library"])
                    ).as_posix(),
                    "native_api": (
                        unit_path / str(region["native_api"])
                    ).as_posix(),
                    "native_entrypoint": str(
                        region["native_entrypoint"]
                    ),
                    "native_verification": (
                        unit_path / str(region["native_verification"])
                    ).as_posix(),
                } if region.get("native_status") == "complete" else {}),
            })
    _atomic_json(root / "links.json", {
        "schema": LINK_TABLE_SCHEMA,
        "order": "dependencies-first",
        "links": links,
    })
    if emit_native:
        automatic_verification = list(verify_project_units_automatically(root))
        manifest["automatic_native_verification"] = automatic_verification
        verified_by_name = {
            str(item["qualified_name"]): dict(item)
            for item in automatic_verification
        }
        for record in manifest["units"]:
            verification = verified_by_name.get(str(record["qualified_name"]))
            if verification is not None:
                record["native_verification_status"] = str(
                    verification["status"]
                )
                if verification.get("reason"):
                    record["native_verification_reason"] = str(
                        verification["reason"]
                    )
        _atomic_json(root / "manifest.json", manifest)
    return manifest


def compile_project_bootstrap_creep(
    source_path: str | Path,
    directory: str | Path,
    *,
    entries: Iterable[str] | None = None,
    expand_entry_dependencies: bool = True,
    python_executable: str | Path = sys.executable,
    jobs: int | None = None,
    max_total_resident_bytes: int | None = None,
    worker_resident_reservation_bytes: int = DEFAULT_WORKER_RESERVATION_BYTES,
    max_worker_memory_bytes: int | None = DEFAULT_WORKER_LIMIT_BYTES,
    unit_timeout_seconds: float | None = DEFAULT_UNIT_TIMEOUT_SECONDS,
    extraction_contract: str | Path | None = DEFAULT_PROJECT_EXTRACTION_CONTRACT,
    bootstrap_products: Iterable[str | Path] = (),
    seed_product: str | Path | None = None,
    crawl_timed_out_units: bool = False,
    max_rounds: int = 16,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compile, prove, install for later workers, and repeat to a fixed point.

    This is the compiler-owned project bootstrap loop.  It discovers the
    authored catalogue itself, runs every pass in the existing bounded worker
    scheduler, emits native products, proves eligible deployments, and gives
    only those receipt-backed products to the next pass.  Partial products are
    also used as source-region seeds, so a successful inner integral can make
    its enclosing authored call tractable without a person selecting either.
    """

    source_file = Path(source_path).resolve()
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    round_limit = int(max_rounds)
    if round_limit < 1:
        raise ValueError("project bootstrap creep needs at least one round")
    active_products = list(dict.fromkeys(
        Path(path).resolve() for path in bootstrap_products
    ))
    rounds: list[dict[str, Any]] = []
    installed_names: set[str] = set()
    installed_regions: set[tuple[str, ...]] = set()
    prior_seed = (
        None if seed_product is None else Path(seed_product).resolve()
    )
    fixed_point: dict[str, Any] | None = None

    def report(stage: str, **details: Any) -> None:
        if progress is not None:
            progress({"stage": stage, **details})

    def write_progress(status: str) -> None:
        _atomic_json(root / "creep-progress.json", {
            "schema": "turing.project-bootstrap-creep-progress.v1",
            "source": source_file.as_posix(),
            "status": status,
            "max_rounds": round_limit,
            "active_products": [path.as_posix() for path in active_products],
            "installed_qualified_names": sorted(installed_names),
            "installed_source_regions": [
                list(chain) for chain in sorted(installed_regions)
            ],
            "rounds": rounds,
            **({"fixed_point": fixed_point} if fixed_point is not None else {}),
        })

    write_progress("running")
    for round_index in range(round_limit):
        round_root = root / f"round_{round_index:03d}"
        round_root.mkdir(parents=True, exist_ok=True)
        report(
            "bootstrap_creep_round_start", round=round_index,
            active_product_count=len(active_products),
        )
        product = compile_project_product(
            source_file,
            round_root,
            entries=entries,
            expand_entry_dependencies=expand_entry_dependencies,
            python_executable=python_executable,
            jobs=jobs,
            max_total_resident_bytes=max_total_resident_bytes,
            worker_resident_reservation_bytes=(
                worker_resident_reservation_bytes
            ),
            max_worker_memory_bytes=max_worker_memory_bytes,
            unit_timeout_seconds=unit_timeout_seconds,
            extraction_contract=extraction_contract,
            emit_native=True,
            seed_product=prior_seed,
            bootstrap_products=active_products,
            progress=lambda event, round_index=round_index: report(
                "bootstrap_creep_worker", round=round_index,
                event=dict(event),
            ),
        )
        completed_names = {
            str(unit["qualified_name"])
            for unit in product.get("units") or ()
            if unit.get("status") == "complete"
        }
        verification_by_name = {
            str(record["qualified_name"]): dict(record)
            for record in product.get("automatic_native_verification") or ()
        }
        native_completion_failures = []
        for unit in product.get("units") or ():
            if unit.get("status") != "failed":
                continue
            error_type = str(unit.get("error_type") or "")
            error = str(unit.get("error") or "")
            # These are durable continuation frontiers, not terminal compiler
            # failures.  Let the subdivision/deep-retry pass below consume
            # their published ProcessGraph products before enforcing native
            # installation.  Treating them as installation failures here
            # made the creep loop raise before it could reach the code that
            # exists specifically to compile their child integrals.
            if (
                (
                    error_type == "CompilationSubdivisionRequired"
                    and str(unit.get("frontier_kind") or "")
                    == "compilation-subdivision-required"
                )
                or (
                    error_type == "ResourceLimitExceeded"
                    and "elapsed time" in error
                )
            ):
                continue
            reason = ": ".join(part for part in (error_type, error) if part)
            native_completion_failures.append({
                "qualified_name": str(
                    unit.get("qualified_name") or "<unknown>"
                ),
                "stage": "compiler-unit",
                "reason": reason or "compiler unit reported failed",
            })
        for qualified_name in sorted(completed_names):
            verification = verification_by_name.get(qualified_name)
            if verification is None or verification.get("status") != "verified":
                native_completion_failures.append({
                    "qualified_name": qualified_name,
                    "stage": "native-verification",
                    "reason": (
                        "no automatic native verification record"
                        if verification is None else
                        str(verification.get("reason") or verification.get("status"))
                    ),
                })
        activations = ()
        if not native_completion_failures and completed_names:
            from .compiler_bootstrap_runtime import (
                activate_compiler_bootstrap_products,
            )

            try:
                activations = activate_compiler_bootstrap_products((round_root,))
            except Exception as error:
                native_completion_failures.append({
                    "qualified_name": ",".join(sorted(completed_names)),
                    "stage": "native-installation",
                    "reason": f"{type(error).__name__}: {error}",
                })
            else:
                activation_by_name = {
                    activation.qualified_name: activation
                    for activation in activations
                }
                for qualified_name in sorted(completed_names):
                    activation = activation_by_name.get(qualified_name)
                    if (
                        activation is None
                        or activation.status != "verified"
                        or activation.native_probe_count < 1
                        or activation.fallback_probe_count != 0
                    ):
                        native_completion_failures.append({
                            "qualified_name": qualified_name,
                            "stage": "native-installation",
                            "reason": (
                                "no receipt-backed native activation"
                                if activation is None else
                                "activation did not prove an exclusively native probe path"
                            ),
                        })
        completion_requirement = {
            "schema": "turing.bootstrap-native-completion-requirement.v1",
            "status": "failed" if native_completion_failures else "satisfied",
            "completed_qualified_names": sorted(completed_names),
            "activated_qualified_names": sorted(
                activation.qualified_name for activation in activations
            ),
            "failures": native_completion_failures,
        }
        product["native_completion_requirement"] = completion_requirement
        _atomic_json(round_root / "manifest.json", product)
        if native_completion_failures:
            failure = {
                **completion_requirement,
                "source": source_file.as_posix(),
                "round": round_index,
            }
            _atomic_json(round_root / "native-installation-failure.json", failure)
            rounds.append({
                "round": round_index,
                "product": round_root.as_posix(),
                "hard_failure": failure,
            })
            write_progress("hard-failed")
            raise NativeInstallationRequiredError(native_completion_failures)
        verified_names = {
            str(record["qualified_name"])
            for record in product.get("automatic_native_verification") or ()
            if record.get("status") == "verified"
        }
        verified_regions = {
            tuple(map(str, region.get("identity_token_chain") or ()))
            for unit in product.get("units") or ()
            for region in unit.get("source_region_integrals") or ()
            if region.get("native_verification_status") == "verified"
        }
        verified_regions.discard(())
        new_names = verified_names - installed_names
        new_regions = verified_regions - installed_regions
        if new_names:
            active_products.append(round_root.resolve())
        installed_names.update(verified_names)
        installed_regions.update(verified_regions)
        prior_seed = round_root.resolve()
        unit_counts: dict[str, int] = {}
        for unit in product.get("units") or ():
            status = str(unit.get("status") or "unknown")
            unit_counts[status] = unit_counts.get(status, 0) + 1
        subdivision_creeps = []
        for unit in product.get("units") or ():
            if unit.get("status") == "complete":
                continue
            plan_name = unit.get("process_graph_unit_plan")
            unit_name = unit.get("path")
            plan_path = (
                None if not plan_name else round_root / str(plan_name)
            )
            strict_child_plan = False
            if plan_path is not None and plan_path.is_file():
                try:
                    strict_child_plan = len(
                        json.loads(plan_path.read_text(encoding="utf-8")).get(
                            "units", ()
                        )
                    ) > 1
                except (OSError, TypeError, ValueError):
                    strict_child_plan = False
            if (
                not crawl_timed_out_units
                and unit.get("error_type") == "ResourceLimitExceeded"
                and "elapsed time" in str(unit.get("error") or "")
                and not strict_child_plan
            ):
                subdivision_creeps.append({
                    "qualified_name": str(
                        unit.get("qualified_name") or "unit"
                    ),
                    "status": "deferred-timeout-retry",
                    "verified_product_count": 0,
                    "fixed_point_count": 0,
                })
                continue
            if not plan_name or not unit_name:
                continue
            graph_path = (
                round_root / str(unit_name) / "resolved-process-graph.pkl"
            )
            if plan_path is None or not plan_path.is_file() or not graph_path.is_file():
                continue
            qualified_name = str(unit.get("qualified_name") or "unit")
            subdivision_root = (
                round_root / "process-graph-creeps"
                / encoded_call_name(qualified_name)
            )
            subdivision = compile_process_graph_creep(
                graph_path,
                plan_path,
                subdivision_root,
                python_executable=python_executable,
                jobs=jobs,
                max_total_resident_bytes=max_total_resident_bytes,
                worker_resident_reservation_bytes=(
                    worker_resident_reservation_bytes
                ),
                max_worker_memory_bytes=max_worker_memory_bytes,
                unit_timeout_seconds=unit_timeout_seconds,
                bootstrap_products=active_products,
                progress=lambda event, round_index=round_index, qualified_name=qualified_name: report(
                    "bootstrap_creep_subdivision", round=round_index,
                    qualified_name=qualified_name, event=dict(event),
                ),
            )
            subdivision_creeps.append({
                "qualified_name": qualified_name,
                "product": subdivision_root.as_posix(),
                "status": str(subdivision.get("status") or ""),
                "verified_product_count": len(
                    subdivision.get("verified_products") or ()
                ),
                "fixed_point_count": len(
                    subdivision.get("fixed_points") or ()
                ),
            })
        verification_frontier = [
            dict(record)
            for record in product.get("automatic_native_verification") or ()
            if record.get("status") != "verified"
        ]
        round_record = {
            "round": round_index,
            "product": round_root.as_posix(),
            "unit_counts": unit_counts,
            "new_verified_qualified_names": sorted(new_names),
            "new_verified_source_regions": [
                list(chain) for chain in sorted(new_regions)
            ],
            "creep_frontier": list(product.get("creep_frontier") or ()),
            "native_verification_frontier": verification_frontier,
            "process_graph_creeps": subdivision_creeps,
        }
        rounds.append(round_record)
        report("bootstrap_creep_round_finish", **round_record)
        if not new_names and not new_regions:
            fixed_point = {
                "kind": "no-new-proven-deployments",
                "round": round_index,
                "action": (
                    "lower-or-verify-the-persisted-creep-frontier; source-"
                    "fallback-remains-authoritative"
                ),
            }
            break
        write_progress("running")
    else:
        fixed_point = {
            "kind": "maximum-bootstrap-rounds",
            "round": round_limit - 1,
            "action": "raise-only-after-inspecting-the-persisted-frontier",
        }

    manifest = {
        "schema": "turing.project-bootstrap-creep-product.v1",
        "source": source_file.as_posix(),
        "source_sha256": _file_sha256(source_file),
        "max_rounds": round_limit,
        "rounds": rounds,
        "active_products": [path.as_posix() for path in active_products],
        "installed_qualified_names": sorted(installed_names),
        "installed_source_regions": [
            list(chain) for chain in sorted(installed_regions)
        ],
        "fixed_point": fixed_point,
        "status": (
            "sealed"
            if rounds
            and not rounds[-1]["creep_frontier"]
            and not rounds[-1]["native_verification_frontier"]
            else "frontier"
        ),
    }
    if active_products:
        from .compiler_bootstrap_runtime import (
            publish_compiler_bootstrap_products,
        )

        registry_path = publish_compiler_bootstrap_products(active_products)
        manifest["compiler_bootstrap_registry"] = registry_path.as_posix()
    _atomic_json(root / "manifest.json", manifest)
    write_progress(str(manifest["status"]))
    return manifest


__all__ = [
    "AuthoredCall",
    "DEFAULT_PROJECT_EXTRACTION_CONTRACT",
    "LINK_TABLE_SCHEMA",
    "NativeInstallationRequiredError",
    "PROJECT_PRODUCT_SCHEMA",
    "ProjectCompilationProduct",
    "SOURCE_REGION_INTEGRAL_SCHEMA",
    "UNIT_ARTIFACT_SCHEMA",
    "compile_project_call",
    "compile_project_bootstrap_creep",
    "compile_project_product",
    "compile_process_graph_creep",
    "compile_process_graph_subdivision_integral",
    "compile_process_graph_subdivision_plan",
    "compilation_creep_frontier",
    "authored_call_dependencies",
    "authored_control_contract",
    "authored_closure_contract",
    "authored_definition_sha256",
    "authored_return_contract",
    "dependency_ordered_records",
    "dependency_ordered_authored_calls",
    "detach_repository_ssa_frontend",
    "discover_authored_calls",
    "encoded_call_name",
    "native_unit_name",
    "process_memory_bytes",
    "partition_authored_source",
    "publish_process_graph_subdivision_plan",
    "ready_process_graph_unit_indices",
    "open_project_compilation_product",
    "resident_bytes",
    "source_region_integral_accounting",
    "verify_structural_resident_table_integral",
    "verify_project_unit_automatically",
    "verify_project_units_automatically",
    "verify_project_scalar_units_automatically",
]

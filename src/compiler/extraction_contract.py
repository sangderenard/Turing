"""Exhaustive, declarative policy for program-source extraction boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatchcase
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import sysconfig
import threading
from typing import Any, Iterable, Mapping

import yaml


class ExtractionAction(str, Enum):
    INGEST_PYTHON = "ingest_python"
    INTRINSIC = "intrinsic"
    PYTHON_HOST_CALL = "python_host_call"
    USE_NATIVE = "use_native"
    DECOMPILE_MACHINE = "decompile_machine"
    REJECT = "reject"


EXTRACTION_CLASSES = (
    "authored_python",
    "repository_python",
    "third_party_python",
    "stdlib_python",
    "builtin",
    "native_extension",
    "dynamic_library",
    "unknown",
)


@dataclass(frozen=True, slots=True)
class ExecutionContract:
    """Whole-program execution semantics surrounding extraction choices."""

    host_runtime: str
    dependency_search: str
    native_lowering: str
    dispatch_unit: str
    unlowered_behavior: str
    require_full_native: bool
    backward_source: str
    python_callbacks: str
    numeric_semantics: str
    scalar_promotion: str

    @classmethod
    def from_mapping(cls, raw: Any) -> "ExecutionContract":
        if not isinstance(raw, Mapping):
            raise ExtractionContractError("execution must be a mapping")
        required = {
            "host_runtime", "dependency_search", "native_lowering",
            "dispatch_unit", "unlowered_behavior", "require_full_native",
            "backward_source", "python_callbacks",
            "numeric_semantics", "scalar_promotion",
        }
        missing = sorted(required - set(raw))
        extra = sorted(set(raw) - required)
        if missing or extra:
            raise ExtractionContractError(
                "execution must define exactly the execution-mode fields; "
                f"missing={missing}, extra={extra}"
            )
        choices = {
            "host_runtime": {"python", "native"},
            "dependency_search": {"reachable", "none"},
            "native_lowering": {"opportunistic", "required"},
            "dispatch_unit": {"isolated_numeric_subgraph", "whole_program"},
            "unlowered_behavior": {"execute_in_python", "reject"},
            "backward_source": {"process_graph", "authored_python"},
            "python_callbacks": {"contract_only", "reject"},
            "numeric_semantics": {"abstract_tensor", "python_scalar"},
            "scalar_promotion": {"all_numeric", "tensor_context_only", "none"},
        }
        values = {name: str(raw[name]) for name in choices}
        for name, allowed in choices.items():
            if values[name] not in allowed:
                raise ExtractionContractError(
                    f"execution.{name} must be one of {sorted(allowed)}"
                )
        require_full_native = raw["require_full_native"]
        if not isinstance(require_full_native, bool):
            raise ExtractionContractError(
                "execution.require_full_native must be boolean"
            )
        if (
            values["native_lowering"] == "opportunistic"
            and require_full_native
        ):
            raise ExtractionContractError(
                "opportunistic native lowering cannot require full native coverage"
            )
        if (
            values["unlowered_behavior"] == "execute_in_python"
            and values["host_runtime"] != "python"
        ):
            raise ExtractionContractError(
                "execute_in_python fallback requires host_runtime: python"
            )
        return cls(
            **values,
            require_full_native=require_full_native,
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "host_runtime": self.host_runtime,
            "dependency_search": self.dependency_search,
            "native_lowering": self.native_lowering,
            "dispatch_unit": self.dispatch_unit,
            "unlowered_behavior": self.unlowered_behavior,
            "require_full_native": self.require_full_native,
            "backward_source": self.backward_source,
            "python_callbacks": self.python_callbacks,
            "numeric_semantics": self.numeric_semantics,
            "scalar_promotion": self.scalar_promotion,
        }


@dataclass(frozen=True, slots=True)
class ExtractionSubject:
    module: str
    qualname: str
    kind: str
    occurrence: str
    origin: str
    classification: str
    source_available: bool

    @property
    def identity(self) -> str:
        return f"{self.module}.{self.qualname}".strip(".")


@dataclass(frozen=True, slots=True)
class ExtractionDecision:
    subject: ExtractionSubject
    action: ExtractionAction
    rule_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ingest_parent(self) -> bool:
        return self.action in {
            ExtractionAction.INGEST_PYTHON,
            ExtractionAction.DECOMPILE_MACHINE,
        } or (
            self.action is ExtractionAction.INTRINSIC
            and self.subject.source_available
            and bool(self.parameters.get("ingest_fallback_source", False))
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "identity": self.subject.identity,
            "module": self.subject.module,
            "qualname": self.subject.qualname,
            "kind": self.subject.kind,
            "occurrence": self.subject.occurrence,
            "origin": self.subject.origin,
            "classification": self.subject.classification,
            "source_available": self.subject.source_available,
            "action": self.action.value,
            "rule_id": self.rule_id,
            "parameters": dict(self.parameters),
        }


class ExtractionContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProgramABIField:
    """Physical field promised by the source-program boundary."""

    storage: str
    dtype: str | None = None
    rank: int = 0
    mutable: bool = False
    default: Any = None
    has_default: bool = False
    #: Concrete extents, when the boundary can state them.
    #:
    #: ``rank`` says how many axes a span has; ``shape`` says how long they
    #: are. The distinction is load-bearing rather than cosmetic: the
    #: scalar-versus-tensor decision (``hierarchical_plan`` -- an operation is
    #: spelled with its SCALAR opcode when the result and every operand have
    #: an EMPTY shape) reads extents, not rank. A parameter that never states
    #: its extents is therefore indistinguishable from a rank-0 scalar and its
    #: arithmetic is compiled as scalar arithmetic, which is the wrong program
    #: for a tensor and reports no shortfall while being so.
    shape: tuple[int, ...] | None = None

    @classmethod
    def from_mapping(cls, location: str, raw: Any) -> "ProgramABIField":
        if not isinstance(raw, Mapping):
            raise ExtractionContractError(f"{location} must be a mapping")
        allowed = {"storage", "dtype", "rank", "mutable", "default", "shape"}
        extra = sorted(set(raw) - allowed)
        if extra:
            raise ExtractionContractError(
                f"{location} has unknown fields {extra}"
            )
        storage = str(raw.get("storage") or "")
        if storage not in {"scalar", "span", "record", "reference", "keyed"}:
            raise ExtractionContractError(
                f"{location}.storage must be scalar, span, record, reference, "
                "or keyed"
            )
        dtype = raw.get("dtype")
        if storage in {"scalar", "span", "keyed"} and not dtype:
            raise ExtractionContractError(
                f"{location}.dtype is required for {storage} storage"
            )
        rank = raw.get("rank", 0)
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise ExtractionContractError(
                f"{location}.rank must be a non-negative integer"
            )
        if storage == "scalar" and rank != 0:
            raise ExtractionContractError(
                f"{location}.rank must be zero for scalar storage"
            )
        if storage == "span" and rank == 0:
            raise ExtractionContractError(
                f"{location}.rank must be positive for span storage"
            )
        # A keyed field is a mapping whose keys are words. It lowers to the
        # repository's universal string tokens: parallel key/value vectors plus
        # a length, so a constant key and a name hashed at run time land on the
        # same slot without a shared intern counter. Its rank is the vector's,
        # which is always one -- the mapping is flat.
        if storage == "keyed" and rank not in {0, 1}:
            raise ExtractionContractError(
                f"{location}.rank must be zero or one for keyed storage"
            )
        mutable = raw.get("mutable", False)
        if not isinstance(mutable, bool):
            raise ExtractionContractError(
                f"{location}.mutable must be boolean"
            )
        raw_shape = raw.get("shape")
        shape: tuple[int, ...] | None = None
        if raw_shape is not None:
            if not isinstance(raw_shape, (list, tuple)):
                raise ExtractionContractError(
                    f"{location}.shape must be a list of extents"
                )
            extents = []
            for position, extent in enumerate(raw_shape):
                if not isinstance(extent, int) or isinstance(extent, bool):
                    raise ExtractionContractError(
                        f"{location}.shape[{position}] must be an integer"
                    )
                if extent <= 0:
                    raise ExtractionContractError(
                        f"{location}.shape[{position}] must be positive"
                    )
                extents.append(int(extent))
            shape = tuple(extents)
            # rank and shape are two statements about the same axes; letting
            # them disagree would leave the boundary describing two different
            # values, so the disagreement is the diagnostic.
            if len(shape) != rank:
                raise ExtractionContractError(
                    f"{location}.shape has {len(shape)} extents but rank is "
                    f"{rank}"
                )
        return cls(
            storage,
            None if dtype is None else str(dtype),
            rank,
            mutable,
            raw.get("default"),
            "default" in raw,
            shape,
        )

    def receipt(self) -> dict[str, Any]:
        result = {
            "storage": self.storage,
            "dtype": self.dtype,
            "rank": self.rank,
            "mutable": self.mutable,
        }
        if self.shape is not None:
            result["shape"] = list(self.shape)
        if self.has_default:
            result["default"] = self.default
        return result


@dataclass(frozen=True, slots=True)
class ProgramABIRecord:
    identity: str
    fields: Mapping[str, ProgramABIField]

    def receipt(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "fields": {
                name: field.receipt() for name, field in self.fields.items()
            },
        }


@dataclass(frozen=True, slots=True)
class ProgramABIBinding:
    function: str
    parameter: str
    record: str

    def receipt(self) -> dict[str, str]:
        return {
            "function": self.function,
            "parameter": self.parameter,
            "record": self.record,
        }


@dataclass(frozen=True, slots=True)
class ProgramABIValueBinding:
    """Physical ABI and source-language type for one function parameter."""

    function: str
    parameter: str
    field: ProgramABIField
    python_type: str

    def receipt(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "parameter": self.parameter,
            **self.field.receipt(),
            "python_type": self.python_type,
        }


@dataclass(frozen=True, slots=True)
class ProgramABIContract:
    """Typed host/native boundary independent of Python object identity."""

    records: Mapping[str, ProgramABIRecord] = field(default_factory=dict)
    bindings: tuple[ProgramABIBinding, ...] = ()
    values: tuple[ProgramABIValueBinding, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Any) -> "ProgramABIContract":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ExtractionContractError("program_abi must be a mapping")
        extra = sorted(set(raw) - {"records", "bindings", "values"})
        if extra:
            raise ExtractionContractError(
                f"program_abi has unknown fields {extra}"
            )
        raw_records = raw.get("records", {})
        if not isinstance(raw_records, Mapping):
            raise ExtractionContractError("program_abi.records must be a mapping")
        records: dict[str, ProgramABIRecord] = {}
        for name, record_raw in raw_records.items():
            location = f"program_abi.records.{name}"
            if not isinstance(record_raw, Mapping):
                raise ExtractionContractError(f"{location} must be a mapping")
            extra_record = sorted(set(record_raw) - {"identity", "fields"})
            if extra_record:
                raise ExtractionContractError(
                    f"{location} has unknown fields {extra_record}"
                )
            raw_fields = record_raw.get("fields", {})
            if not isinstance(raw_fields, Mapping) or not raw_fields:
                raise ExtractionContractError(
                    f"{location}.fields must be a non-empty mapping"
                )
            fields = {
                str(field_name): ProgramABIField.from_mapping(
                    f"{location}.fields.{field_name}", field_raw,
                )
                for field_name, field_raw in raw_fields.items()
            }
            identity = str(record_raw.get("identity") or name)
            records[str(name)] = ProgramABIRecord(identity, fields)
        raw_bindings = raw.get("bindings", ())
        if not isinstance(raw_bindings, list):
            raise ExtractionContractError("program_abi.bindings must be a list")
        bindings = []
        for index, binding_raw in enumerate(raw_bindings):
            location = f"program_abi.bindings[{index}]"
            if not isinstance(binding_raw, Mapping):
                raise ExtractionContractError(f"{location} must be a mapping")
            if set(binding_raw) != {"function", "parameter", "record"}:
                raise ExtractionContractError(
                    f"{location} must define exactly function, parameter, record"
                )
            binding = ProgramABIBinding(
                str(binding_raw["function"]),
                str(binding_raw["parameter"]),
                str(binding_raw["record"]),
            )
            if binding.record not in records:
                raise ExtractionContractError(
                    f"{location}.record names unknown record {binding.record!r}"
                )
            bindings.append(binding)
        raw_values = raw.get("values", ())
        if not isinstance(raw_values, list):
            raise ExtractionContractError("program_abi.values must be a list")
        values = []
        for index, value_raw in enumerate(raw_values):
            location = f"program_abi.values[{index}]"
            if not isinstance(value_raw, Mapping):
                raise ExtractionContractError(f"{location} must be a mapping")
            required = {"function", "parameter", "storage", "python_type"}
            if not required.issubset(value_raw):
                raise ExtractionContractError(
                    f"{location} must define function, parameter, storage, "
                    "and python_type"
                )
            field_raw = {
                key: value
                for key, value in value_raw.items()
                if key not in {"function", "parameter", "python_type"}
            }
            field = ProgramABIField.from_mapping(location, field_raw)
            python_type = str(value_raw.get("python_type") or "")
            if not python_type:
                raise ExtractionContractError(
                    f"{location}.python_type must be a qualified identity"
                )
            values.append(ProgramABIValueBinding(
                str(value_raw["function"]),
                str(value_raw["parameter"]),
                field,
                python_type,
            ))
        return cls(records, tuple(bindings), tuple(values))

    def records_for_function(
        self,
        function_name: str,
        *,
        method_owner: str | None = None,
        parameters: Iterable[str] = (),
    ) -> dict[str, ProgramABIRecord]:
        result = {}
        for binding in self.bindings:
            if fnmatchcase(str(function_name), binding.function):
                result[binding.parameter] = self.records[binding.record]
        # An instance method's receiver already has an exact source-level
        # class identity.  Let that identity select the matching record ABI
        # instead of requiring every contract to repeat ``parameter: self``
        # for every method.  This is schema correlation only: ambiguity is
        # left unresolved and no Python class is constructed or inspected.
        parameter_names = set(map(str, parameters))
        if (
            method_owner is not None
            and "self" in parameter_names
            and "self" not in result
        ):
            owner = str(method_owner)
            candidates = tuple(
                record
                for name, record in self.records.items()
                if (
                    str(name) == owner
                    or str(record.identity) == owner
                    or str(record.identity).rsplit(".", 1)[-1] == owner
                )
            )
            if len(candidates) == 1:
                result["self"] = candidates[0]
        return result

    def values_for_function(
        self, function_name: str,
    ) -> dict[str, ProgramABIValueBinding]:
        result = {}
        for binding in self.values:
            if fnmatchcase(str(function_name), binding.function):
                result[binding.parameter] = binding
        return result

    def receipt(self) -> dict[str, Any]:
        return {
            "records": {
                name: record.receipt() for name, record in self.records.items()
            },
            "bindings": [binding.receipt() for binding in self.bindings],
            "values": [binding.receipt() for binding in self.values],
        }


class ExtractionContract:
    """Resolve every callable to one explicit extraction disposition."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).resolve()
        raw_text = self.path.read_text(encoding="utf-8")
        raw = yaml.safe_load(raw_text)
        if not isinstance(raw, Mapping):
            raise ExtractionContractError("extraction contract must be a mapping")
        if int(raw.get("version", 0)) != 1:
            raise ExtractionContractError("extraction contract version must be 1")
        if raw.get("mode") != "exhaustive":
            raise ExtractionContractError("extraction contract mode must be exhaustive")
        defaults = raw.get("defaults")
        if not isinstance(defaults, Mapping):
            raise ExtractionContractError("defaults must be a mapping")
        missing = sorted(set(EXTRACTION_CLASSES) - set(defaults))
        extra = sorted(set(defaults) - set(EXTRACTION_CLASSES))
        if missing or extra:
            raise ExtractionContractError(
                f"defaults must cover exactly all extraction classes; missing={missing}, extra={extra}"
            )
        self.defaults = {
            name: self._normalize_choice(f"default:{name}", choice)
            for name, choice in defaults.items()
        }
        # Preserve the extraction-choice validation order: an unsafe or
        # incomplete disposition table remains the first reported defect.
        # The execution model surrounds those choices; it does not obscure
        # them.
        self.execution = ExecutionContract.from_mapping(raw.get("execution"))
        self.program_abi = ProgramABIContract.from_mapping(raw.get("program_abi"))
        rules = raw.get("rules", ())
        if not isinstance(rules, list):
            raise ExtractionContractError("rules must be a list")
        self.rules = []
        seen = set()
        for ordinal, rule in enumerate(rules):
            if not isinstance(rule, Mapping):
                raise ExtractionContractError(f"rule {ordinal} must be a mapping")
            rule_id = str(rule.get("id") or "")
            if not rule_id or rule_id in seen:
                raise ExtractionContractError(f"rule {ordinal} has missing/duplicate id")
            seen.add(rule_id)
            match = rule.get("match", {})
            if not isinstance(match, Mapping):
                raise ExtractionContractError(f"rule {rule_id} match must be a mapping")
            choice = self._normalize_choice(rule_id, rule)
            self.rules.append((rule_id, dict(match), choice))
        roots = raw.get("roots", {})
        self.authored_roots = self._roots(roots.get("authored", ()))
        self.repository_roots = self._roots(roots.get("repository", ()))
        self.stdlib_root = Path(sysconfig.get_paths()["stdlib"]).resolve()
        self.limits = dict(raw.get("limits") or {})
        self.fingerprint = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        self._lock = threading.Lock()
        self._decisions: dict[
            tuple[str, str, str, str], ExtractionDecision
        ] = {}
        self._source_origins: set[Path] = set()
        self._source_bytes = 0

    def _roots(self, values: Any) -> tuple[Path, ...]:
        if not isinstance(values, list):
            raise ExtractionContractError("root groups must be lists")
        result = []
        for value in values:
            path = Path(str(value))
            if not path.is_absolute():
                path = self.path.parent / path
            result.append(path.resolve())
        return tuple(result)

    @staticmethod
    def _normalize_choice(rule_id: str, raw: Any) -> tuple[ExtractionAction, dict[str, Any]]:
        if isinstance(raw, str):
            action_text, parameters = raw, {}
        elif isinstance(raw, Mapping):
            action_text = raw.get("action")
            parameters = dict(raw.get("parameters") or {})
        else:
            raise ExtractionContractError(f"choice {rule_id} must be text or mapping")
        try:
            action = ExtractionAction(str(action_text))
        except ValueError as exc:
            raise ExtractionContractError(
                f"choice {rule_id} has unknown action {action_text!r}"
            ) from exc
        if action is ExtractionAction.DECOMPILE_MACHINE and not parameters.get("explicit_opt_in"):
            raise ExtractionContractError(
                f"choice {rule_id} enables decompilation without explicit_opt_in"
            )
        fallback_source = parameters.get("ingest_fallback_source", False)
        if not isinstance(fallback_source, bool):
            raise ExtractionContractError(
                f"choice {rule_id} parameter ingest_fallback_source must be boolean"
            )
        required = {
            ExtractionAction.INGEST_PYTHON: {"follow_reachable_calls"},
            ExtractionAction.INTRINSIC: {"lowering_namespace"},
            ExtractionAction.PYTHON_HOST_CALL: {"execution", "callbacks"},
            ExtractionAction.USE_NATIVE: {"loader", "symbol_resolution", "callbacks"},
            ExtractionAction.DECOMPILE_MACHINE: {
                "explicit_opt_in", "max_functions", "max_total_bytes",
                "max_dependency_depth",
            },
            ExtractionAction.REJECT: {"reason"},
        }[action]
        missing = sorted(required - set(parameters))
        if missing:
            raise ExtractionContractError(
                f"choice {rule_id} action {action.value} lacks parameters {missing}"
            )
        return action, parameters

    @staticmethod
    def _within(path: Path, roots: tuple[Path, ...]) -> bool:
        return any(path == root or root in path.parents for root in roots)

    def subject(self, value: Any) -> ExtractionSubject:
        target = value.__func__ if inspect.ismethod(value) else value
        occurrence = (
            "definition"
            if (
                inspect.isclass(target)
                or inspect.isfunction(target)
                or inspect.ismethod(value)
                or inspect.isbuiltin(target)
                or inspect.ismodule(target)
            )
            else "instance"
        )
        definition_target = type(target) if occurrence == "instance" else target
        module = str(getattr(definition_target, "__module__", ""))
        qualname = str(getattr(
            definition_target,
            "__qualname__",
            getattr(definition_target, "__name__", ""),
        ))
        kind = (
            "class" if inspect.isclass(target)
            else "builtin" if inspect.isbuiltin(target)
            else "method" if inspect.ismethod(value)
            else "function" if inspect.isfunction(target)
            else type(target).__name__
        )
        try:
            origin = str(
                inspect.getsourcefile(definition_target)
                or inspect.getfile(definition_target)
                or ""
            )
        except (OSError, TypeError):
            defining_module = (
                inspect.getmodule(definition_target) or sys.modules.get(module)
            )
            origin = str(getattr(defining_module, "__file__", "") or "")
        suffix = Path(origin).suffix.casefold()
        try:
            inspect.getsource(definition_target)
            source_available = True
        except (OSError, TypeError):
            source_available = False
        resolved = Path(origin).resolve() if origin else None
        if module == "builtins":
            classification = "builtin"
        elif suffix in {".dll"}:
            classification = "dynamic_library"
        elif suffix in {".pyd", ".so", ".dylib"}:
            classification = "native_extension"
        elif inspect.isbuiltin(target):
            classification = "native_extension"
        elif resolved is not None and self._within(resolved, self.authored_roots):
            classification = "authored_python"
        elif resolved is not None and self._within(resolved, self.repository_roots):
            classification = "repository_python"
        elif resolved is not None and (
            "site-packages" in {part.casefold() for part in resolved.parts}
        ):
            classification = "third_party_python"
        elif resolved is not None and (resolved == self.stdlib_root or self.stdlib_root in resolved.parents):
            classification = "stdlib_python"
        else:
            classification = "unknown"
        return ExtractionSubject(
            module,
            qualname,
            kind,
            occurrence,
            origin,
            classification,
            source_available,
        )

    @staticmethod
    def _matches(subject: ExtractionSubject, match: Mapping[str, Any]) -> bool:
        values = {
            "module": subject.module,
            "qualname": subject.qualname,
            "identity": subject.identity,
            "kind": subject.kind,
            "occurrence": subject.occurrence,
            "origin": subject.origin.replace("\\", "/"),
            "classification": subject.classification,
        }
        for key, expected in match.items():
            if key == "source_available":
                if bool(expected) != subject.source_available:
                    return False
                continue
            if key not in values:
                raise ExtractionContractError(f"unknown match field {key!r}")
            patterns = expected if isinstance(expected, list) else [expected]
            if not any(fnmatchcase(values[key], str(pattern)) for pattern in patterns):
                return False
        return True

    def decide(self, value: Any) -> ExtractionDecision:
        subject = self.subject(value)
        key = (
            subject.module,
            subject.qualname,
            subject.occurrence,
            subject.origin,
        )
        with self._lock:
            cached = self._decisions.get(key)
        if cached is not None:
            return cached
        rule_id = f"default:{subject.classification}"
        action, parameters = self.defaults[subject.classification]
        for candidate_id, match, choice in self.rules:
            if self._matches(subject, match):
                rule_id = candidate_id
                action, parameters = choice
                break
        decision = ExtractionDecision(subject, action, rule_id, parameters)
        with self._lock:
            if action is ExtractionAction.INGEST_PYTHON and subject.origin:
                source_path = Path(subject.origin).resolve()
                if source_path not in self._source_origins:
                    limits = dict(self.limits.get("python_source") or {})
                    max_files = int(limits.get("max_files", 0))
                    max_bytes = int(limits.get("max_total_bytes", 0))
                    try:
                        source_bytes = source_path.stat().st_size
                    except OSError:
                        source_bytes = 0
                    if max_files and len(self._source_origins) + 1 > max_files:
                        raise ExtractionContractError(
                            f"python source file ceiling exceeded at {source_path}"
                        )
                    if max_bytes and self._source_bytes + source_bytes > max_bytes:
                        raise ExtractionContractError(
                            f"python source byte ceiling exceeded at {source_path}"
                        )
                    self._source_origins.add(source_path)
                    self._source_bytes += source_bytes
            self._decisions[key] = decision
        return decision

    def __call__(self, value: Any) -> bool:
        return self.decide(value).ingest_parent

    def receipts(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            decisions = tuple(self._decisions.values())
        return tuple(
            decision.receipt()
            for decision in sorted(decisions, key=lambda item: item.subject.identity)
        )

    def receipt_json(self) -> str:
        return json.dumps(self.receipts(), indent=2, sort_keys=True)


__all__ = [
    "EXTRACTION_CLASSES",
    "ExtractionAction",
    "ExtractionContract",
    "ExtractionContractError",
    "ExtractionDecision",
    "ExtractionSubject",
    "ExecutionContract",
    "ProgramABIBinding",
    "ProgramABIContract",
    "ProgramABIField",
    "ProgramABIRecord",
]

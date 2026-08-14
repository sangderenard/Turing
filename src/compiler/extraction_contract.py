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
from typing import Any, Mapping

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
class ExtractionSubject:
    module: str
    qualname: str
    kind: str
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
        }

    def receipt(self) -> dict[str, Any]:
        return {
            "identity": self.subject.identity,
            "module": self.subject.module,
            "qualname": self.subject.qualname,
            "kind": self.subject.kind,
            "origin": self.subject.origin,
            "classification": self.subject.classification,
            "source_available": self.subject.source_available,
            "action": self.action.value,
            "rule_id": self.rule_id,
            "parameters": dict(self.parameters),
        }


class ExtractionContractError(ValueError):
    pass


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
        self._decisions: dict[tuple[str, str, str], ExtractionDecision] = {}
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
        module = str(getattr(target, "__module__", ""))
        qualname = str(getattr(target, "__qualname__", getattr(target, "__name__", "")))
        kind = (
            "class" if inspect.isclass(target)
            else "builtin" if inspect.isbuiltin(target)
            else "method" if inspect.ismethod(value)
            else "function" if inspect.isfunction(target)
            else type(target).__name__
        )
        try:
            origin = str(inspect.getsourcefile(target) or inspect.getfile(target) or "")
        except (OSError, TypeError):
            defining_module = inspect.getmodule(target) or sys.modules.get(module)
            origin = str(getattr(defining_module, "__file__", "") or "")
        suffix = Path(origin).suffix.casefold()
        try:
            inspect.getsource(target)
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
            module, qualname, kind, origin, classification, source_available
        )

    @staticmethod
    def _matches(subject: ExtractionSubject, match: Mapping[str, Any]) -> bool:
        values = {
            "module": subject.module,
            "qualname": subject.qualname,
            "identity": subject.identity,
            "kind": subject.kind,
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
        key = (subject.module, subject.qualname, subject.origin)
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
]

"""Reusable exact-teacher/learned-duty boundary for local graph appendages.

An appendage is replaceable only at an explicitly declared graph cut: the
candidate receives the same ordered state and inputs, predicts a declared
subset of outputs, and is periodically compared with the exact compiled
teacher.  Parameters remain external artifacts; this contract never freezes
them into the equation graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class GraphAppendageReplacementContract:
    identity: str
    exact_entrypoint: str
    input_names: tuple[str, ...]
    state_names: tuple[str, ...]
    output_names: tuple[str, ...]
    learned_output_names: tuple[str, ...]
    novelty_channels: tuple[str, ...]
    learned_state_names: tuple[str, ...] = ()
    permits_live_duty: bool = False
    parameter_transport: str = "json-dtype-shape-content-hash"
    candidate_device: str = "gpu"

    def __post_init__(self) -> None:
        if not self.identity or not self.exact_entrypoint:
            raise ValueError("appendage identity and exact entrypoint are required")
        for label, names in (
            ("inputs", self.input_names), ("state", self.state_names),
            ("outputs", self.output_names),
            ("learned outputs", self.learned_output_names),
        ):
            if not names or len(names) != len(set(names)):
                raise ValueError(f"appendage {label} must be nonempty and unique")
        if not set(self.learned_output_names) <= set(self.output_names):
            raise ValueError("learned outputs must be an exact-output subset")
        if not set(self.learned_state_names) <= set(self.state_names):
            raise ValueError("learned state must be an exact-state subset")
        if self.permits_live_duty and set(self.learned_state_names) != set(self.state_names):
            raise ValueError(
                "live duty for a stateful appendage requires its complete next-state boundary"
            )
        if self.candidate_device.casefold() != "gpu":
            raise ValueError("live learned appendage candidates are GPU operators")

    @property
    def digest(self) -> str:
        payload = json.dumps(self.as_manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_manifest(self) -> Mapping[str, object]:
        return {
            "schema": "turing.graph-appendage-replacement.v1",
            "identity": self.identity,
            "exact_entrypoint": self.exact_entrypoint,
            "input_names": list(self.input_names),
            "state_names": list(self.state_names),
            "output_names": list(self.output_names),
            "learned_output_names": list(self.learned_output_names),
            "learned_state_names": list(self.learned_state_names),
            "novelty_channels": list(self.novelty_channels),
            "parameter_transport": self.parameter_transport,
            "candidate_device": self.candidate_device,
            "permits_live_duty": self.permits_live_duty,
            "invariants": {
                "exact_teacher_first": True,
                "periodic_exact_trials": True,
                "candidate_must_predict_complete_state_for_live_duty": True,
                "replacement_boundary_is_graph_cut_not_host_wrapper": True,
            },
        }


@dataclass(frozen=True, slots=True)
class AppendageDutyTelemetry:
    """Small shader/HUD ABI; it never participates in the physics result."""

    learned_active: bool
    exact_trial_active: bool
    exact_duty_alpha: float
    normalized_trial_loss: float

    def sampled_attachment_pulse(self, sample_index: int) -> float:
        if sample_index < 0:
            raise ValueError("sample index cannot be negative")
        alpha = min(1.0, max(0.0, float(self.exact_duty_alpha)))
        loss = 0.0 if not math.isfinite(self.normalized_trial_loss) else min(
            1.0, max(0.0, float(self.normalized_trial_loss))
        )
        phase = (sample_index * 0.6180339887498949) % 1.0
        event = self.exact_trial_active or phase < max(alpha, 0.035)
        return (0.35 + 0.65 * loss) if event else 0.0


def validate_parameter_artifact(
    artifact: Mapping[str, object], *, required_arrays: Sequence[str],
) -> str:
    """Validate external candidate parameters and return their content digest."""

    arrays = artifact.get("arrays")
    if not isinstance(arrays, Mapping):
        raise ValueError("appendage parameter artifact requires an arrays mapping")
    for name in required_arrays:
        row = arrays.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"missing parameter array {name}")
        if row.get("dtype") not in ("float32", "float64"):
            raise ValueError(f"parameter array {name} has unsupported dtype")
        shape = row.get("shape")
        if not isinstance(shape, list) or not shape or any(
            not isinstance(value, int) or value <= 0 for value in shape
        ):
            raise ValueError(f"parameter array {name} has invalid shape")
        digest = row.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"parameter array {name} lacks a content hash")
    encoded = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AppendageDutyTelemetry", "GraphAppendageReplacementContract",
    "validate_parameter_artifact",
]

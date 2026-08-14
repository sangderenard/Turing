"""State cascade coordinating growth flags with boundary-driven restarts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable

from ..transmogrifier.graph.boundary_namespace import BoundaryNamespace


class GrowthCascadeState(str, Enum):
    OBSERVING = "observing"
    FLAGGED = "flagged"
    WAITING_BOUNDARY = "waiting-boundary"
    RESTARTING = "restarting"
    COMPLETE = "complete"
    EXHAUSTED = "exhausted"
    STOPPED = "stopped"


@dataclass
class BoundaryRestartCascade:
    root: Path
    language: str = "python"
    max_restarts: int = 4
    wait_seconds: float = 300.0
    poll_seconds: float = 0.25
    stop_event: threading.Event | None = None
    status_sink: Callable[[str], None] = print
    state: GrowthCascadeState = GrowthCascadeState.OBSERVING
    attempt: int = 0
    last_flag: Path | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if self.max_restarts < 0:
            raise ValueError("max_restarts cannot be negative")
        if self.wait_seconds < 0.0 or self.poll_seconds <= 0.0:
            raise ValueError("growth restart timing is invalid")
        self.stop_event = self.stop_event or threading.Event()

    def fingerprint(self) -> str:
        return BoundaryNamespace(self.root, self.language).fingerprint()

    def observing(self) -> str:
        self.state = GrowthCascadeState.OBSERVING
        fingerprint = self.fingerprint()
        self.status_sink(
            f"[translation-cascade] OBSERVING attempt={self.attempt + 1} "
            f"boundary={fingerprint[:12]}"
        )
        return fingerprint

    def flag(self, error: Any, fingerprint: str) -> Path:
        self.state = GrowthCascadeState.FLAGGED
        flag_directory = self.root / ".growth_flags"
        flag_directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = flag_directory / f"{stamp}-attempt-{self.attempt + 1}.flag.json"
        suffix = 1
        while path.exists():
            path = flag_directory / (
                f"{stamp}-attempt-{self.attempt + 1}-{suffix}.flag.json"
            )
            suffix += 1
        payload = {
            "version": 1,
            "state": self.state.value,
            "attempt": self.attempt + 1,
            "owner": str(getattr(error, "owner", "unknown")),
            "boundary_hint": str(
                getattr(error, "boundary_hint", self.language)
            ),
            "node_count": int(getattr(error, "node_count", 0)),
            "depth": getattr(error, "depth", None),
            "height": getattr(error, "height", None),
            "stages": dict(getattr(error, "stages", {})),
            "message": str(error),
            "boundary_fingerprint": fingerprint,
            "suggested_directory": str(
                self.root / str(getattr(error, "boundary_hint", self.language))
            ),
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.last_flag = path
        self.status_sink(
            f"[translation-cascade] FLAGGED receipt={path} "
            f"hint={payload['boundary_hint']}"
        )
        return path

    def wait_for_change(self, fingerprint: str) -> bool:
        if self.attempt >= self.max_restarts:
            self.state = GrowthCascadeState.EXHAUSTED
            self.status_sink(
                "[translation-cascade] EXHAUSTED restart budget; "
                "the failed attempt remains visible"
            )
            return False
        self.state = GrowthCascadeState.WAITING_BOUNDARY
        self.status_sink(
            "[translation-cascade] WAITING_BOUNDARY install/edit a precise "
            f"*.node.json below {self.root}"
        )
        deadline = time.monotonic() + self.wait_seconds
        while time.monotonic() <= deadline:
            if self.stop_event.is_set():
                self.state = GrowthCascadeState.STOPPED
                return False
            current = self.fingerprint()
            if current != fingerprint:
                self.attempt += 1
                self.state = GrowthCascadeState.RESTARTING
                self._update_flag("restarting", current)
                self.status_sink(
                    f"[translation-cascade] RESTARTING attempt={self.attempt + 1} "
                    f"boundary={current[:12]}"
                )
                return True
            self.stop_event.wait(self.poll_seconds)
        self.state = GrowthCascadeState.EXHAUSTED
        self._update_flag("wait-expired", fingerprint)
        self.status_sink(
            "[translation-cascade] EXHAUSTED boundary wait expired; "
            "rerun after installing the rule"
        )
        return False

    def complete(self) -> None:
        self.state = GrowthCascadeState.COMPLETE
        self._update_flag("complete", self.fingerprint())
        self.status_sink(
            f"[translation-cascade] COMPLETE attempts={self.attempt + 1}"
        )

    def _update_flag(self, state: str, fingerprint: str) -> None:
        if self.last_flag is None or not self.last_flag.exists():
            return
        try:
            payload = json.loads(self.last_flag.read_text(encoding="utf-8"))
            payload["state"] = state
            payload["current_boundary_fingerprint"] = fingerprint
            self.last_flag.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            # A user may be inspecting/editing the receipt. State transitions
            # remain authoritative in memory; receipt refresh is best effort.
            pass


__all__ = ["BoundaryRestartCascade", "GrowthCascadeState"]

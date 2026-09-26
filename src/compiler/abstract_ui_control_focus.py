"""Backend-neutral routing between captured devices and interaction contexts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ControlFocusPolicy:
    identity: str
    actor: str | None
    initial: str = "game"
    routes: tuple[str, ...] = ("game", "projected-pointer", "dialogue")
    switch_action: str = "secondary-action"
    dialogue_priority: int = 100
    return_rule: str = "resume-previous"

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": "abstract-ui-control-focus-v0",
            "identity": self.identity,
            "actor": self.actor,
            "initial": self.initial,
            "routes": list(self.routes),
            "switch_action": self.switch_action,
            "dialogue": {
                "priority": self.dialogue_priority,
                "return_rule": self.return_rule,
                "response_required": True,
            },
            "projected_pointer": {
                "source": "captured-pointer-motion",
                "destination": "document-coordinate-space",
                "clamp": "viewport-bounds",
                "activation": self.switch_action,
                "native_realization": "relative-while-locked-or-absolute-after-release",
            },
        }


__all__ = ["ControlFocusPolicy"]

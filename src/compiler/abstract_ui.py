"""Small common object and packet convention for AbstractUI projections."""

from __future__ import annotations

from dataclasses import dataclass, replace
import html
import json
from pathlib import Path
from typing import Any, Mapping

ABSTRACT_UI_OBJECT_VERSION = "abstract-ui-object-v0"


ABSTRACT_UI_JAVASCRIPT_SYSTEM_ROOT = r"""// abstract-ui:system-root
const abstractUISystemTimer = (() => {
  let actionEdges = null;
  let pendingActions = [];
  let sequence = 0;
  const schedule = globalThis.requestAnimationFrame
    ? callback => globalThis.requestAnimationFrame(callback)
    : callback => globalThis.setTimeout(() => callback(Date.now()), 16);
  function connect(destination) { actionEdges = destination; }
  function issue(action) { pendingActions.push(action); }
  function frame(now) {
    sequence += 1;
    if (actionEdges) {
      actionEdges.time = now;
      actionEdges.update(pendingActions.splice(0));
    }
    schedule(frame);
  }
  schedule(frame);
  return {identity: "system-root/timer", connect, issue, get sequence() { return sequence; }};
})();"""


def javascript_with_system_root(
    behavior: str,
    *,
    timer_identity: str = "system-root/timer",
) -> str:
    """Place the standard timer at the root of an AbstractUI JS emission."""

    source = str(behavior)
    if source.startswith("// abstract-ui:system-root"):
        return source
    root = ABSTRACT_UI_JAVASCRIPT_SYSTEM_ROOT.replace(
        '"system-root/timer"', json.dumps(str(timer_identity)), 1,
    )
    return f"{root}\n\n{source}"


@dataclass(frozen=True, slots=True)
class AbstractUIPacket:
    """One language packet carried by an AbstractUI object."""

    identity: str
    language: str
    source: str
    media_type: str
    role: str = "projection"
    dependencies: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class AbstractUIInteraction:
    """The entire abstract event contract carried by an interactive node."""

    type: str
    destination: str

    def to_data(self) -> dict[str, str]:
        return {"type": self.type, "destination": self.destination}


@dataclass(frozen=True, slots=True)
class AbstractUI:
    """Bare-bones UI graph value plus ordered backend-language packets.

    HTML describes semantic structure, CSS describes its presentation graph,
    JavaScript describes behavior, and JSON transports the neutral data graph.
    Backends may add other packet languages without changing this object.
    """

    identity: str
    model: Mapping[str, Any]
    packets: tuple[AbstractUIPacket, ...] = ()
    title: str = "AbstractUI"
    schema_version: str = ABSTRACT_UI_OBJECT_VERSION

    def packet(self, language: str, *, role: str | None = None) -> AbstractUIPacket:
        matches = tuple(
            packet for packet in self.packets
            if packet.language == language and (role is None or packet.role == role)
        )
        if not matches:
            raise KeyError(f"AbstractUI has no {language!r} packet")
        if len(matches) != 1:
            raise ValueError(
                f"AbstractUI packet selection is ambiguous for {language!r}"
            )
        return matches[0]

    def packets_for(self, language: str) -> tuple[AbstractUIPacket, ...]:
        return tuple(packet for packet in self.packets if packet.language == language)

    def with_packet(self, packet: AbstractUIPacket) -> "AbstractUI":
        if any(existing.identity == packet.identity for existing in self.packets):
            raise ValueError(f"AbstractUI packet already exists: {packet.identity}")
        return replace(self, packets=(*self.packets, packet))

    def interaction(
        self, interaction_type: str, destination: str,
    ) -> AbstractUIInteraction:
        """Describe an event without binding it to a host implementation."""

        if not interaction_type or not destination:
            raise ValueError("interaction type and destination must be non-empty")
        return AbstractUIInteraction(str(interaction_type), str(destination))

    @property
    def css(self) -> str:
        return self.packet("css").source

    @property
    def script(self) -> str:
        return self.packet("javascript").source

    @property
    def javascript(self) -> str:
        return self.script

    def document(self) -> str:
        """Assemble the conventional packets into a portable HTML document."""

        body = self.packet("html", role="structure").source
        css = self.packet("css", role="presentation").source
        script = self.packet("javascript", role="behavior").source
        model = self.packet("json", role="model").source
        if not script.startswith("// abstract-ui:system-root"):
            raise ValueError(
                "AbstractUI javascript behavior must include the system-root prelude"
            )
        if "</script" in script.lower():
            raise ValueError("javascript packet may not terminate its host script")
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(self.title, quote=True)}</title>
<style>{css}</style>
</head>
<body>
{body}
<script type="application/json" id="abstract-ui-model">{model}</script>
<script>
{script}
</script>
</body>
</html>
"""

    @property
    def html(self) -> str:
        """The assembled page; the bare structure remains an HTML packet."""

        return self.document()

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.document(), encoding="utf-8")
        return target


__all__ = [
    "ABSTRACT_UI_OBJECT_VERSION",
    "ABSTRACT_UI_JAVASCRIPT_SYSTEM_ROOT",
    "AbstractUI",
    "AbstractUIInteraction",
    "AbstractUIPacket",
    "javascript_with_system_root",
]

"""Account-scoped browser transport for Spectral's original camera routing grid."""
from __future__ import annotations

import sys
from threading import RLock

_lock = RLock()
_panels = {}


def command(account_id: int, request: dict, spectral_root, *, radio_snapshot=None) -> dict:
    if not isinstance(account_id, int) or account_id <= 0:
        raise ValueError("the camera routing grid requires a signed-in account")
    action = request.get("action")
    with _lock:
        if action == "open":
            root = str(spectral_root)
            if root not in sys.path:
                sys.path.insert(0, root)
            from camera_software.trigger_grid_bridge import CameraTriggerGrid
            receivers = request.get("receivers")
            buses = request.get("buses", [])
            if (not isinstance(receivers, list) or not isinstance(buses, list)
                    or any(not isinstance(k, str) or len(k) > 80 for k in receivers + buses)):
                raise ValueError("camera routing needs bounded named light buses and receivers")
            panel = CameraTriggerGrid(receivers, buses, saved=request.get("graph"),
                                      generators=request.get("generators"))
            _panels[account_id] = panel
            return panel.frame()
        if action == "event":
            panel = _panels.get(account_id)
            if panel is None:
                raise ValueError("open the camera routing grid before sending pointer events")
            return panel.event(request.get("name"), request.get("x", 0), request.get("y", 0),
                               button=request.get("button", 1), wheel_y=request.get("wheel_y", 0),
                               shift=request.get("shift", False))
        if action == "radio":
            panel = _panels.get(account_id)
            if panel is None:
                raise ValueError("open the camera routing grid before connecting radio")
            if radio_snapshot is None:
                raise ValueError("radio measurement source is unavailable")
            return panel.radio(radio_snapshot())
        if action == "pulse":
            panel = _panels.get(account_id)
            if panel is None:
                raise ValueError("open the camera routing grid before firing")
            return panel.pulse()
        if action == "configure":
            panel = _panels.get(account_id)
            if panel is None:
                raise ValueError("open the camera routing grid before configuring generators")
            from camera_software.trigger_generators import validate_generators
            panel.generators = validate_generators(request.get("generators"))
            return {"generators": list(panel.generators)}
        raise ValueError("unknown camera routing grid action")

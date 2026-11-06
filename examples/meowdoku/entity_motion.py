"""Player motion for entity games: one law, run by the server and the page.

A player ship is integrated, never placed.  Per axis its velocity relaxes
toward a commanded velocity with damping time ``tau`` (implicit Euler, so a
step of any length is stable -- the server steps at the dt system's pace, a
browser at its frame rate), and a held pointer commands the velocity that
would close the gap to the pointer in ``reach`` seconds; keys command
``axis * speed``.  Walls stop the ship and zero the velocity into them.

The function uses only arithmetic and ``abs`` -- no branch, no library -- so
the SAME text runs on AbstractTensor spans inside a dt-system law and on
plain numbers, and is what the source compiler lowers to the page's
predictor: a direct WebAssembly core (``build_predictor``), the route and
ABI of ``examples/mobile_reference_app`` (its ``turing-wasm-shell.js`` loads
it).  The float annotations are for that lowering; the server passes spans.  The page runs it for its own ship on every
frame and reconciles against the server's state (see ``entities.py``).
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("x", "v", "axis", "target", "lo", "hi", "speed", "tau", "reach", "dt")
OUTPUTS = ("x_next", "v_next")


def damped_axis(x: float, v: float, axis: float, target: float, lo: float, hi: float,
                speed: float, tau: float, reach: float, dt: float):
    """One axis of a damped ship over ``dt``.  ``target < 0``: no pointer."""
    steering = (target >= 0.0) * 1.0
    gap = (target - x) / reach
    pointer = (abs(gap + speed) - abs(gap - speed)) * 0.5
    keys = (abs(axis + 1.0) - abs(axis - 1.0)) * 0.5 * speed
    command = steering * pointer + (1.0 - steering) * keys
    moved = v + (command - v) * (dt / (tau + dt))
    free = x + moved * dt
    x_next = (abs(free - lo) - abs(free - hi) + lo + hi) * 0.5
    v_next = (x_next - x) / dt
    return x_next, v_next


def ground_travel(u: float, v: float, gx: float, gy: float, turn: float, speed: float, dt: float):
    """The land under a spherical game's ship, as one law for server and page:
    the ground frame's heading (unit ``gx, gy`` on the tileable map) turns by
    ``turn`` radians -- an exact rotation by the rational (Cayley) form of the
    half-angle, so a unit heading stays unit with no transcendental -- and the
    map point (``u, v``) travels ``speed * dt`` along it.  Arithmetic only:
    the server's NumPy and the page's WebAssembly compute the same IEEE values."""
    h = turn * 0.5
    d = 1.0 + h * h
    c = (1.0 - h * h) / d
    s = (2.0 * h) / d
    gx_next = gx * c - gy * s
    gy_next = gx * s + gy * c
    u_next = u + gx_next * speed * dt
    v_next = v + gy_next * speed * dt
    return u_next, v_next, gx_next, gy_next


def follow_view(view: float, x: float, follow: float, width: float):
    """A scrolling view that follows a runner: never back, and forward by as
    much as the runner is past ``follow`` of the ``width`` (``max`` written
    with ``abs``, as ``damped_axis`` is)."""
    gap = x - view - follow * width
    view_next = view + (gap + abs(gap)) * 0.5
    return (view_next,)


#: The engine laws the page runs, each lowered to its own WebAssembly core:
#: (function, input names, output names, file stem).
PREDICTORS = (
    (damped_axis, NAMES, OUTPUTS, "motion"),
    (ground_travel, ("u", "v", "gx", "gy", "turn", "speed", "dt"), ("u_next", "v_next", "gx_next", "gy_next"), "travel"),
    (follow_view, ("view", "x", "follow", "width"), ("view_next",), "follow"),
)


def build_predictor(out_dir: Path) -> dict:
    """Lower every law in PREDICTORS to a WebAssembly core in ``out_dir``
    (``<stem>.wasm`` + ``<stem>.abi.json``), each rebuilt only when its
    source changes; returns the contracts by stem."""
    return {stem: _build_one(out_dir, fn, names, outputs, stem) for fn, names, outputs, stem in PREDICTORS}


def _build_one(out_dir: Path, fn, names, outputs_names, stem: str) -> dict:
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_wasm_backend import emit_ssa_module_to_wasm_core

    out_dir = Path(out_dir)
    source = inspect.getsource(fn)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    abi_path, wasm_path = out_dir / f"{stem}.abi.json", out_dir / f"{stem}.wasm"
    if abi_path.exists() and wasm_path.exists():
        contract = json.loads(abi_path.read_text(encoding="utf-8"))
        if contract.get("source_digest") == digest:
            return contract
    module, outputs, exports = lower_ast_source_to_ssa(
        source, fn.__name__, name="entity_motion",
        extraction_contract=ExtractionContract(ROOT / "extraction_contracts/program_extraction.yaml"),
        progress_bars=False)
    entry = exports[0]
    wasm = emit_ssa_module_to_wasm_core(module, entry, entry_name=entry)
    if not wasm.complete:
        raise RuntimeError(f"{fn.__name__} WebAssembly emission shortfalls: {wasm.shortfalls}")
    argument_ids = tuple(int(value.id) for value in module.functions[entry].args)
    output_ids = tuple(int(value.id) for value in outputs[entry])
    input_names = dict(zip(argument_ids, names))
    output_names = dict(zip(output_ids, outputs_names))
    descriptors, offset = [], 8
    for kind, value_id in wasm.parameters:
        descriptor = {"kind": kind, "value_id": value_id,
                      "role": "output" if value_id in output_names else "input",
                      "name": output_names.get(value_id, input_names.get(value_id, f"value_{value_id}"))}
        if kind == "buffer":
            descriptor["offset"] = offset
            descriptor["dtype"] = dict(wasm.output_dtypes).get(value_id, "float64")
            offset += 8
        descriptors.append(descriptor)
    contract = {"schema": "turing.mobile-wasm-abi.v1", "entrypoint": entry, "inputs": list(names),
                "outputs": list(outputs_names), "parameters": descriptors, "source_digest": digest}
    out_dir.mkdir(parents=True, exist_ok=True)
    wasm_path.write_bytes(wasm.binary)
    abi_path.write_text(json.dumps(contract, indent=1), encoding="utf-8")
    return contract

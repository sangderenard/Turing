"""Host-facing product of a direct Wasm core artifact: contract, files, page.

``emit_ssa_module_to_wasm_core`` produces a module whose exported entry takes
i32 byte offsets for buffers and values for scalars, and whose linear memory
the host lays out (``WasmCoreExecution`` does so in node).  A page needs the
same facts as data: which parameter is which, its kind, dtype, element count
and byte offset, and which host functions the module imports.  This module
publishes exactly that, in the shape ``examples/mobile_reference_app`` already
reads (``program.abi.json``, schema ``turing.wasm-core-abi.v1``), and can
inline everything into ONE html file, which is what a game or a demo ships.

The offsets here are the ones ``WasmCoreExecution`` computes, written once:
the reserved static prefix, then each buffer in parameter order, each rounded
up to eight bytes.  A page and the node harness therefore read the same cell
for the same value.
"""

from __future__ import annotations

import base64
import json
from math import prod
from pathlib import Path
from typing import Any, Mapping, Sequence

_SHELL_JS = (
    Path(__file__).resolve().parents[2]
    / "examples/mobile_reference_app/web/turing-wasm-shell.js"
)
_ITEMSIZE = {"float64": 8, "int64": 8, "int32": 4, "float32": 4}


def core_contract(
    artifact: Any, module: Any, entry: str, *,
    input_names: Mapping[int, str] | None = None,
    output_names: Mapping[int, str] | None = None,
    controls: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The invocation contract of ``artifact`` as plain data.

    Names come from the root's ``parameter_names`` and ``named_outputs``
    metadata unless given by value id.  Element counts come from the
    artifact's own tables (``buffer_shapes`` / ``output_elements``); a buffer
    whose shape is unknown is one cell, as the node harness allocates it.

    ``controls`` maps a parameter name, or a ProgramABI field name a root
    formal carries in its accounting (``program_abi_field``), to the semantic
    action or shell resource a shell fills it from; the parameter's
    descriptor then carries ``"control"`` and a page binds it generically
    (``TuringWasmProgram.controlFeeds``).  A control naming nothing in the
    contract is refused by name.
    """

    function = module.functions[entry]
    controls = {str(k): str(v) for k, v in dict(controls or {}).items()}
    field_of: dict[int, str] = {}
    for formal in function.args:
        accounting = dict(getattr(formal, "accounting", None) or {})
        if accounting.get("program_abi_field") is not None:
            field_of[int(formal.id)] = str(accounting["program_abi_field"])
    names: dict[int, str] = {
        int(value_id): str(name)
        for name, value_id in tuple(function.metadata.get("parameter_names") or ())
    }
    for name, value_id in tuple(function.metadata.get("named_outputs") or ()):
        names.setdefault(int(value_id), str(name))
    names.update({int(k): str(v) for k, v in dict(input_names or {}).items()})
    names.update({int(k): str(v) for k, v in dict(output_names or {}).items()})
    output_ids = set(int(v) for v in artifact.output_order)
    dtypes = dict(artifact.output_dtypes)
    shapes = dict(zip(artifact.buffer_order, artifact.buffer_shapes))
    elements = dict(artifact.output_elements)
    offset = (max(8, int(artifact.reserved_bytes)) + 7) // 8 * 8
    parameters = []
    for kind, value_id in artifact.parameters:
        value_id = int(value_id)
        descriptor: dict[str, Any] = {
            "kind": kind,
            "value_id": value_id,
            "role": "output" if value_id in output_ids else "input",
            "name": names.get(value_id, f"value_{value_id}"),
        }
        control = controls.get(descriptor["name"], controls.get(field_of.get(value_id, "")))
        if control is not None and descriptor["role"] == "input":
            descriptor["control"] = control
        if kind == "buffer":
            dtype = dtypes.get(value_id, "float64")
            shape = tuple(shapes.get(value_id, ()) or ())
            count = elements.get(value_id)
            if count is None:
                count = prod(shape) if shape and all(
                    type(axis) is int and axis >= 0 for axis in shape) else 1
            descriptor.update({
                "dtype": dtype, "count": int(count), "shape": list(shape),
                "offset": offset,
            })
            offset += (int(count) * _ITEMSIZE.get(dtype, 8) + 7) // 8 * 8
        parameters.append(descriptor)
    bound = {p["control"] for p in parameters if "control" in p}
    unbound = sorted(set(controls.values()) - bound)
    if unbound:
        raise ValueError(
            f"controls {unbound} name no input parameter of {artifact.name}: "
            f"{sorted(controls)} against {[p['name'] for p in parameters]}")
    return {
        "schema": "turing.wasm-core-abi.v1",
        "entrypoint": artifact.name,
        "memory_bytes": offset,
        "reserved_bytes": int(artifact.reserved_bytes),
        "imports": [list(item) for item in artifact.imports],
        "inputs": [p["name"] for p in parameters if p["role"] == "input"],
        "outputs": [p["name"] for p in parameters if p["role"] == "output"],
        "controls": {p["name"]: p["control"] for p in parameters if "control" in p},
        "parameters": parameters,
    }


def shell_source() -> str:
    """The page-shell binding (``turing-wasm-shell.js``), verbatim."""

    return _SHELL_JS.read_text(encoding="utf-8")


def write_core_web_product(
    directory: str | Path, artifact: Any, contract: Mapping[str, Any], *,
    identity: str = "turing-wasm-core",
) -> Path:
    """Write ``program.wasm``, ``program.abi.json``, the shell binding and an
    ``application.json`` receipt into ``directory``; returns the directory."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "program.wasm").write_bytes(artifact.binary)
    (output / "program.abi.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    (output / "turing-wasm-shell.js").write_text(shell_source(), encoding="utf-8")
    (output / "application.json").write_text(json.dumps({
        "schema": "turing.application-product.v1",
        "identity": identity,
        "program": {"api": "program.abi.json", "entrypoint": contract["entrypoint"]},
        "targets": {"web": {"execution": {
            "kind": "repository-ssa-wasm", "artifact": "program.wasm"}}},
    }, indent=2), encoding="utf-8")
    return output


def emit_core_single_file_page(
    artifact: Any, contract: Mapping[str, Any], *, title: str,
    body_html: str = "", script: str = "",
) -> str:
    """One html file: the module (base64), its contract, the shell binding,
    and either the caller's own ``body_html``/``script`` (a game) or a plain
    panel that runs the entry on typed feeds and prints the outputs.

    ``script`` runs as a module after ``program`` (a ``TuringWasmProgram``)
    and ``contract`` are in scope.
    """

    shell = shell_source().replace("export const ", "const ").replace("export class ", "class ")
    binary = base64.b64encode(artifact.binary).decode("ascii")
    if not body_html:
        rows = "".join(
            f'<label>{p["name"]} <input data-feed="{p["name"]}" value="0"></label>'
            for p in contract["parameters"] if p["role"] == "input"
        )
        body_html = (
            f"<h1>{title}</h1><form id=\"feeds\">{rows}"
            "<button type=\"submit\">Run</button></form><pre id=\"out\"></pre>"
        )
        script = script or """
document.querySelector('#feeds').addEventListener('submit', event => {
  event.preventDefault();
  const feeds = {};
  for (const input of document.querySelectorAll('[data-feed]')) {
    const text = input.value.trim();
    feeds[input.dataset.feed] = text.includes(',') ? text.split(',').map(Number) : Number(text);
  }
  const result = program.invoke(feeds);
  document.querySelector('#out').textContent = JSON.stringify(result, (k, v) =>
    typeof v === 'bigint' ? v.toString() : v, 2);
});
"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>body{{font:15px system-ui;margin:0;padding:16px;background:#101827;color:#e8ecf4}}
label{{display:block;margin:6px 0}}input{{font:inherit}}button{{font:inherit;margin-top:8px}}
pre{{background:#182438;padding:12px;overflow:auto}}</style>
</head><body>
{body_html}
<script id="turing-contract" type="application/json">{json.dumps(contract)}</script>
<script id="turing-module" type="application/wasm-base64">{binary}</script>
<script type="module">
{shell}
const contract = JSON.parse(document.querySelector('#turing-contract').textContent);
const raw = atob(document.querySelector('#turing-module').textContent.trim());
const bytes = new Uint8Array(raw.length);
for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
const program = await TuringWasmProgram.fromBytes(bytes, contract);
{script}
</script>
</body></html>
"""


__all__ = [
    "core_contract", "emit_core_single_file_page", "shell_source",
    "write_core_web_product",
]

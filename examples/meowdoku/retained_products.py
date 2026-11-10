"""Read-only validation of an explicitly retained fountain release.

This does not certify the current Python source or execute/compile a shader.
An operator can inventory an already accepted release, then pin its manifests,
WGSL and advertised CPU fallbacks across a backend-only restart::

    python -m examples.meowdoku.retained_products --web-dir build/meowdoku/web --write-receipt build/fountain-release.json
    python examples/meowdoku/server.py --retain-fountain-products build/fountain-release.json

Keep the receipt outside the served directory. Never refresh it to conceal a
validation failure; restore the accepted artifacts or validate a new release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from .startup import FOUNTAIN_FAMILIES

SCHEMA = "turing.retained-fountain-products.v1"


def _require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive(value) -> bool:
    return type(value) is int and value > 0


def _names(value, label: str) -> list[str]:
    _require(isinstance(value, list) and all(isinstance(v, str) and v for v in value), f"{label}: expected names")
    _require(len(value) == len(set(value)), f"{label}: duplicate names")
    return value


def _json(data: bytes, label: str) -> dict:
    try:
        result = json.loads(data)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{label}: invalid JSON ({error})") from error
    _require(isinstance(result, dict), f"{label}: expected a JSON object")
    return result


def inspect_products(web_dir: Path, expected: dict[str, str] | None = None) -> dict:
    """Inspect the retained files, optionally checking an independently saved hash inventory.

    Only the manifests' own ABI is used. Source/build modules are not imported,
    digests are never recomputed or changed, and no product is written.
    """
    root = Path(web_dir).resolve()
    files, families = {}, {}

    def read(name: str) -> bytes:
        _require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", name) is not None
                 and name not in (".", ".."), f"unsafe retained filename: {name!r}")
        path = root / name
        _require(path.resolve().parent == root, f"{name}: file escapes web directory")
        try:
            data = path.read_bytes()
        except OSError as error:
            raise ValueError(f"{name}: missing or unreadable retained product ({error})") from error
        _require(bool(data), f"{name}: empty retained product")
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None:
            _require(name in expected, f"{name}: absent from retained receipt")
            _require(expected[name] == digest, f"{name}: SHA-256 differs from retained receipt; restore the accepted product")
        files[name] = digest
        return data

    for prefix, family in FOUNTAIN_FAMILIES:
        name = f"{prefix}.manifest.json"
        manifest = _json(read(name), name)
        schema = "turing.fountain3d-products.v1" if prefix.startswith("fountain3d") else "turing.fountain-products.v3"
        _require(manifest.get("schema") == schema, f"{name}: unsupported manifest schema")
        model = manifest.get("cost_model")
        _require(isinstance(model, dict) and "min_speedup" in model, f"{name}: missing cost_model.min_speedup")
        _require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in model.values()),
                 f"{name}: invalid cost_model")
        law, sizes = manifest.get("law"), manifest.get("sizes")
        _require(isinstance(law, dict) and isinstance(sizes, dict) and sizes, f"{name}: missing law or sizes")
        arguments = _names(law.get("arguments"), f"{name} arguments")
        outputs = _names(law.get("outputs"), f"{name} outputs")
        state = _names(law.get("state"), f"{name} state")
        uniforms = _names(law.get("uniforms"), f"{name} uniforms")
        diagnostics = _names(law.get("diagnostics"), f"{name} diagnostics")
        _require(arguments and outputs and state and set(state + uniforms) <= set(arguments), f"{name}: invalid law names")
        _require(set(diagnostics) <= set(outputs), f"{name}: missing diagnostic outputs")
        _require(type(law.get("dt")) in (int, float) and math.isfinite(law["dt"]) and law["dt"] > 0,
                 f"{name}: invalid law dt")
        cpu_sizes = []
        for tag, record in sizes.items():
            label = f"{name} size {tag}"
            _require(isinstance(record, dict) and record.get("tag") == tag, f"{label}: size tag mismatch")
            _require(_positive(record.get("n")) and _positive(record.get("pad")) and record["n"] <= record["pad"],
                     f"{label}: invalid particle counts")
            _require(record.get("route") in ("wasm", "webgpu-compute") and isinstance(record.get("verdict"), dict),
                     f"{label}: missing deployment verdict")
            speedup = record["verdict"].get("speedup")
            _require(type(speedup) in (int, float) and math.isfinite(speedup) and speedup >= 0,
                     f"{label}: invalid deployment speedup")
            _require(isinstance(record.get("source_digest"), str) and record["source_digest"], f"{label}: missing original source digest")
            abi = record.get("wgsl_abi")
            _require(isinstance(abi, dict) and abi.get("count") == record["pad"], f"{label}: WGSL count mismatch")
            _require(abi.get("arguments") == arguments and abi.get("outputs") == outputs and abi.get("uniforms") == uniforms,
                     f"{label}: WGSL/law argument, output or uniform mismatch")
            for key in ("slots", "bindings", "slots_per_binding", "arena_elements", "dispatches"):
                _require(_positive(abi.get(key)), f"{label}: invalid WGSL {key}")
            for key in ("groups", "workgroup_size"):
                _require(isinstance(abi.get(key), list) and len(abi[key]) == 3 and all(_positive(v) for v in abi[key]),
                         f"{label}: invalid WGSL {key}")
            entries = _names(abi.get("entries"), f"{label} entries")
            _require(entries and len(entries) == abi["dispatches"], f"{label}: dispatch/entry mismatch")
            for key, names, limit in (("feed_slots", arguments, abi["slots"]), ("output_slots", outputs, abi["slots"]),
                                      ("uniform_index", uniforms, len(uniforms))):
                slots = abi.get(key)
                _require(isinstance(slots, dict) and set(slots) <= set(names)
                         and all(type(v) is int and 0 <= v < limit for v in slots.values()), f"{label}: invalid {key}")
            _require(set(state) <= set(abi["feed_slots"]) and set(outputs) == set(abi["output_slots"]),
                     f"{label}: missing state/output slots")
            _require(set(abi["uniform_index"].values()) == set(range(len(abi["uniform_index"]))),
                     f"{label}: non-dense uniform indices exceed the runtime head buffer")
            _require((abi["slots"] + abi["slots_per_binding"] - 1) // abi["slots_per_binding"] == abi["bindings"]
                     and abi["arena_elements"] >= abi["slots"] * record["pad"], f"{label}: undersized arena")
            _require(isinstance(record.get("memory"), dict) and record["memory"].get("arena_bytes") == 4 * abi["arena_elements"],
                     f"{label}: arena memory mismatch")
            wgsl = record.get("wgsl")
            _require(wgsl == f"{prefix}_{tag}.wgsl", f"{label}: unexpected WGSL filename")
            try:
                shader = read(wgsl).decode("utf-8")
            except UnicodeError as error:
                raise ValueError(f"{wgsl}: invalid UTF-8") from error
            functions = set(re.findall(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", shader))
            _require("@compute" in shader and set(entries) <= functions,
                     f"{wgsl}: missing advertised compute entry")
            has_wasm, has_abi = "wasm" in record, "wasm_abi" in record
            _require(has_wasm == has_abi, f"{label}: incomplete CPU fallback reference")
            if has_wasm:
                wasm, abi_name = record["wasm"], record["wasm_abi"]
                _require(wasm == f"{prefix}_{tag}.wasm" and abi_name == f"{prefix}_{tag}.abi.json",
                         f"{label}: unexpected CPU filename")
                _require(read(wasm).startswith(b"\0asm\x01\0\0\0"), f"{wasm}: invalid WebAssembly header")
                cpu = _json(read(abi_name), abi_name)
                _require(cpu.get("schema") == "turing.graph-buffer-abi.v1" and cpu.get("wasm") == wasm
                         and cpu.get("count") == record["pad"] and cpu.get("argument_names") == arguments,
                         f"{abi_name}: CPU/law ABI mismatch")
                order = cpu.get("order")
                _require(isinstance(order, list) and order and all(type(v) is int for v in order)
                         and len(set(order)) == len(order), f"{abi_name}: invalid buffer order")
                _require(all(isinstance(cpu.get(k), list) and len(cpu[k]) == len(order) for k in ("shapes", "dtypes")),
                         f"{abi_name}: incomplete buffer descriptors")
                inputs, result = cpu.get("inputs"), cpu.get("outputs")
                _require(isinstance(inputs, dict) and set(inputs) == set(arguments)
                         and all(type(v) is int and v in order for v in inputs.values()),
                         f"{abi_name}: invalid input references")
                _require(isinstance(result, list) and all(isinstance(v, dict) for v in result)
                         and [v.get("name") for v in result] == outputs and all(v.get("id") in order for v in result),
                         f"{abi_name}: invalid output references")
                cpu_sizes.append(tag)
        families[family] = {"sizes": list(sizes), "cpu_sizes": cpu_sizes}
    if expected is not None:
        _require(set(files) == set(expected), "retained receipt contains files no longer advertised by the fountain manifests")
    return {"schema": SCHEMA, "files": files, "families": families}


def validate_receipt(web_dir: Path, receipt_path: Path) -> dict:
    """Verify pinned bytes and their manifest contracts; never fall back to a build."""
    try:
        receipt = _json(Path(receipt_path).read_bytes(), str(receipt_path))
    except OSError as error:
        raise ValueError(f"retained receipt {receipt_path}: missing or unreadable ({error})") from error
    _require(receipt.get("schema") == SCHEMA, f"{receipt_path}: unsupported retained receipt schema")
    hashes = receipt.get("files")
    _require(isinstance(hashes, dict) and hashes and all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes.values()),
             f"{receipt_path}: invalid SHA-256 inventory")
    actual = inspect_products(web_dir, hashes)
    _require(actual["families"] == receipt.get("families"), f"{receipt_path}: family inventory mismatch")
    return actual


def prepare_retained(web_dir: Path, receipt_path: Path, ctx) -> None:
    receipt = validate_receipt(web_dir, receipt_path)
    for family, record in receipt["families"].items():
        ctx.ready(family, f"retained {len(record['sizes'])} sizes; SHA-256/ABI checked; source freshness not checked")
        cpu = record["cpu_sizes"]
        if cpu:
            ctx.ready(f"{family}_cpu", f"retained CPU sizes only: {', '.join(cpu)}; no builds")
        else:
            ctx.skipped(f"{family}_cpu", "no CPU fallbacks advertised in the retained manifest; no builds")
    print(f"[meowdoku] retained {len(receipt['files'])} fountain files from {receipt_path}; no fountain builds; source freshness not checked", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--web-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-receipt", type=Path, help="inventory an already accepted release; fails if receipt exists")
    mode.add_argument("--check-receipt", type=Path, help="verify an existing receipt without changing any file")
    args = parser.parse_args(argv)
    try:
        if args.write_receipt:
            _require(not args.write_receipt.resolve().is_relative_to(args.web_dir.resolve()), "keep the receipt outside the served directory")
            receipt = inspect_products(args.web_dir)
            with args.write_receipt.open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2)
                stream.write("\n")
        else:
            receipt = validate_receipt(args.web_dir, args.check_receipt)
    except (ValueError, OSError) as error:
        print(f"retained fountain validation failed: {error}", flush=True)
        return 1
    print(f"{len(receipt['families'])} families, {len(receipt['files'])} files: artifact integrity/ABI checked, not source freshness or GPU execution", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

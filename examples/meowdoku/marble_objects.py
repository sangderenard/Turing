"""Immutable, content-addressed marble/die designs shared by Studio and simulator.

The authored recipe, material float tables and physical fields are the source.
The browser regenerates the closed triangles and verifies geometry_sha256 on
load. Derived mass/inertia are retained as a receipt, not trusted by the store.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path

VERSION = 1
MAX_BYTES = 256 * 1024
MAX_ENTRIES = 128
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_TABLES = ("pbr", "phong", "enamel", "texstack")


class ObjectError(ValueError):
    pass


def _finite_tree(value, depth=0):
    if depth > 8:
        raise ObjectError("object is too deeply nested")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ObjectError("nonfinite number")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 80:
                raise ObjectError("invalid field name")
            _finite_tree(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item, depth + 1)
    elif not (value is None or isinstance(value, (str, bool))):
        raise ObjectError("unsupported JSON value")


def validate_object(value: object) -> dict:
    if not isinstance(value, dict) or value.get("v") != VERSION:
        raise ObjectError("expected version 1 marble object")
    if set(value) != {"v", "name", "generator", "design", "materials", "physical", "derived", "geometry_sha256"}:
        raise ObjectError("unexpected or missing object fields")
    if not isinstance(value["name"], str) or not 1 <= len(value["name"].strip()) <= 80:
        raise ObjectError("name must be 1..80 characters")
    if value["generator"] != "meowdoku-marble-object-v1":
        raise ObjectError("unsupported mesh generator")
    if (not isinstance(value["design"], dict) or value["design"].get("base-shape") not in
            ("marble", "d4", "d6", "d8", "d10", "d12", "d20")):
        raise ObjectError("invalid design")
    if not isinstance(value["physical"], dict) or not isinstance(value["derived"], dict):
        raise ObjectError("physical and derived properties are required")
    if not _DIGEST.fullmatch(str(value["geometry_sha256"])):
        raise ObjectError("invalid geometry digest")
    materials = value["materials"]
    if not isinstance(materials, dict) or set(materials) != {"index", "extra", "tables"}:
        raise ObjectError("invalid material library")
    if not isinstance(materials["index"], dict) or not isinstance(materials["extra"], dict):
        raise ObjectError("invalid material ids")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 or v > 1024
           for v in [*materials["index"].values(), *materials["extra"].values()]):
        raise ObjectError("material id out of range")
    if not isinstance(materials["tables"], dict) or set(materials["tables"]) != set(_TABLES):
        raise ObjectError("missing material float table")
    for name in _TABLES:
        table = materials["tables"][name]
        if not isinstance(table, dict) or set(table) != {"stride", "data"}:
            raise ObjectError(f"invalid {name} table")
        stride, data = table["stride"], table["data"]
        if not isinstance(stride, int) or not 1 <= stride <= 128 or not isinstance(data, list) or len(data) > 20000 or len(data) % stride:
            raise ObjectError(f"invalid {name} record layout")
        if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in data):
            raise ObjectError(f"invalid {name} float")
    maximum_id = max([*materials["index"].values(), *materials["extra"].values()], default=-1)
    if any(len(materials["tables"][name]["data"]) // materials["tables"][name]["stride"] <= maximum_id
           for name in _TABLES):
        raise ObjectError("material id exceeds saved float tables")
    _finite_tree(value)
    if not isinstance(value["derived"].get("massKg"), (int, float)) or value["derived"]["massKg"] <= 0:
        raise ObjectError("positive mass receipt required")
    return value


def source_digest(value: dict) -> str:
    source = {key: value[key] for key in ("v", "generator", "design", "materials", "physical", "geometry_sha256")}
    canonical = json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class Store:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.lock = threading.Lock()

    def entries(self) -> list[dict]:
        if not self.directory.exists():
            return []
        out = []
        for path in self.directory.glob("*.json"):
            if not _DIGEST.fullmatch(path.stem):
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                out.append({"digest": path.stem, "name": value["name"],
                            "shape": value["design"]["base-shape"],
                            "kind": value["design"]["kind"],
                            "massKg": value["derived"]["massKg"],
                            "mechanicsExact": bool(value["derived"].get("massExact")),
                            "bytes": path.stat().st_size})
            except (ValueError, KeyError, OSError, TypeError):
                continue
        return sorted(out, key=lambda entry: entry["name"].casefold())

    def get(self, digest: str) -> dict:
        if not _DIGEST.fullmatch(digest):
            raise KeyError(digest)
        path = self.directory / f"{digest}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        validate_object(value)
        if source_digest(value) != digest:
            raise ObjectError("stored object digest mismatch")
        return {**value, "digest": digest}

    def put(self, value: dict) -> dict:
        validate_object(value)
        digest = source_digest(value)
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(raw) > MAX_BYTES:
            raise ObjectError(f"saved object exceeds {MAX_BYTES} bytes")
        with self.lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{digest}.json"
            if path.exists():
                return {"digest": digest, "duplicate": True}
            if len(self.entries()) >= MAX_ENTRIES:
                raise ObjectError("marble object library is full")
            temp = self.directory / f".{digest}.{os.getpid()}.tmp"
            try:
                temp.write_bytes(raw)
                os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)
        return {"digest": digest, "duplicate": False}

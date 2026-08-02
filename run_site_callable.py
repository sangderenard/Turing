"""Invoke one trusted module-level callable for the local inspection site."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
from pathlib import Path
import sys
import uuid


def _encode(value):
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
            "bytes": len(value),
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_encode(item) for item in value]}
    if isinstance(value, dict):
        return {
            "kind": "mapping",
            "items": {str(key): _encode(item) for key, item in value.items()},
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "scalar", "value": value}
    raise TypeError(f"unsupported site result type: {type(value).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--callable", required=True)
    parser.add_argument("--arguments-json", default="{}")
    arguments = parser.parse_args()

    kwargs = json.loads(arguments.arguments_json)
    if not isinstance(kwargs, dict):
        raise TypeError("arguments JSON must be an object")
    module_name = f"_turing_site_{uuid.uuid4().hex}"
    specification = importlib.util.spec_from_file_location(module_name, arguments.source)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {arguments.source}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    function = getattr(module, arguments.callable)
    if not callable(function):
        raise TypeError(f"{arguments.callable!r} is not callable")
    print(json.dumps({"ok": True, "result": _encode(function(**kwargs))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

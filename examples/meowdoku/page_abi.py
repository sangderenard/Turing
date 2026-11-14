"""Bounded display and requirement descriptor for an inventoried page version.

This is descriptive metadata. It never grants a capability or forwards a
control; the API catalog and signed-in principal remain the authority.
"""
from __future__ import annotations

import json
import re

from meowdoku.authz import CAPS

SCHEMA = "meowdoku.page.v1"
SURFACES = frozenset(("dom", "canvas", "webgl", "webgpu", "mixed"))
NETWORK = frozenset(("none", "optional", "required", "unknown"))
STANDALONE = frozenset(("unknown", "requires-site"))
IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9._-]{0,63}\Z")
MAX_BYTES = 8192


class PageABIError(ValueError):
    pass


def _names(value, field):
    if not isinstance(value, list) or len(value) > 16 or any(not isinstance(name, str) or
            not IDENTIFIER.fullmatch(name) for name in value) or len(value) != len(set(value)):
        raise PageABIError(f"invalid {field}")
    return list(value)


def validate(value, inventory):
    """Return a normalized, JSON-only v1 descriptor, or fail before registration."""
    if not isinstance(value, dict) or set(value) != {"schema", "presentation", "execution", "access", "controls"}:
        raise PageABIError("invalid page contract")
    if value["schema"] != SCHEMA:
        raise PageABIError("unknown page contract schema")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PageABIError("invalid page contract JSON") from error
    if len(encoded) > MAX_BYTES:
        raise PageABIError("page contract is too large")
    presentation, execution = value["presentation"], value["execution"]
    access, controls = value["access"], value["controls"]
    if not isinstance(presentation, dict) or set(presentation) != {"title", "summary", "poster", "surface"}:
        raise PageABIError("invalid page presentation")
    if (not isinstance(presentation["title"], str) or not 1 <= len(presentation["title"].strip()) <= 120 or
            not isinstance(presentation["summary"], str) or len(presentation["summary"]) > 500 or
            not isinstance(presentation["surface"], str) or presentation["surface"] not in SURFACES):
        raise PageABIError("invalid page presentation")
    poster = presentation["poster"]
    if poster is not None and (not isinstance(poster, str) or poster not in inventory or
            not poster.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) or inventory[poster][0] > 2 * 1024 * 1024):
        raise PageABIError("poster is not an inventoried image")
    if not isinstance(execution, dict) or set(execution) != {"standalone", "network", "compute", "site_endpoints"}:
        raise PageABIError("invalid page execution")
    # A verified offline claim needs a trusted build receipt; v1 cannot yet
    # produce that receipt, so it cannot claim one.
    if (not isinstance(execution["standalone"], str) or execution["standalone"] not in STANDALONE or
            not isinstance(execution["network"], str) or execution["network"] not in NETWORK):
        raise PageABIError("invalid page execution")
    compute = _names(execution["compute"], "compute requirements")
    endpoints = _names(execution["site_endpoints"], "site endpoints")
    if not isinstance(access, dict) or set(access) != {"required_capabilities"}:
        raise PageABIError("invalid page access")
    capabilities = _names(access["required_capabilities"], "required capabilities")
    if not set(capabilities).issubset(CAPS):
        raise PageABIError("unknown required capability")
    if not isinstance(controls, dict) or set(controls) != {"local", "server"}:
        raise PageABIError("invalid page controls")
    if controls["local"] not in (None, "site-nav.registerControlSuite"):
        raise PageABIError("unknown local control adapter")
    server = controls["server"]
    if server is not None and (not isinstance(server, dict) or set(server) != {"map", "contract"} or
            any(not isinstance(server[key], str) or not IDENTIFIER.fullmatch(server[key]) for key in ("map", "contract"))):
        raise PageABIError("invalid Living Data control contract")
    return {
        "schema": SCHEMA,
        "presentation": {"title": presentation["title"].strip(), "summary": presentation["summary"],
                         "poster": poster, "surface": presentation["surface"]},
        "execution": {"standalone": execution["standalone"], "network": execution["network"],
                      "compute": compute, "site_endpoints": endpoints},
        "access": {"required_capabilities": capabilities},
        "controls": {"local": controls["local"], "server": server},
    }


def from_manifest(manifest, inventory):
    value = manifest.get("page_contract")
    return validate(value, inventory) if value is not None else None

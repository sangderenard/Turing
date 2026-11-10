"""Copy the canonical registry append method into a reviewed generated host.

Only the source-owned method and the existing utility content key change.
Read its Python literal statically: importing src.compiler would load a native
DLL while another task may be rebuilding that library. No compiler is run.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/compiler/javascript_runtime_utilities.py"
BEGIN = "  // studio-runtime-append-begin"
END = "  // studio-runtime-append-end"


def canonical_append():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    rows = [node for node in tree.body if isinstance(node, ast.Assign) and
            any(isinstance(target, ast.Name) and target.id == "WORLD_REGISTRY_SOURCE"
                for target in node.targets)]
    if len(rows) != 1:
        raise ValueError("Canonical world registry literal is not unique")
    value = rows[0].value
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "strip" and not value.args and not value.keywords:
        value = value.func.value
    literal = ast.literal_eval(value)
    if not isinstance(literal, str):
        raise ValueError("Canonical world registry must remain authored source")
    source = literal.strip()
    start, end = source.find(BEGIN), source.find(END)
    if start < 0 or end < start or source.count(BEGIN) != 1 or source.count(END) != 1:
        raise ValueError("Canonical append method boundary is unavailable")
    end += len(END)
    if source[end:end + 1] == "\n":
        end += 1
    method = source[start:end]
    old_source = source[:start] + source[end:]
    match = re.search(r"^class TuringWorldRegistry \{[\s\S]*?^}", old_source, re.M)
    if not match or "appendObjects(" not in method or "</script" in method.lower():
        raise ValueError("Canonical append source is not the expected class method")
    return method, match.group(), hashlib.sha256(source.encode("utf-8")).hexdigest()


def patch_studio_registry_append(source: bytes):
    method, expected_old_class, canonical_sha = canonical_append()
    document = source.decode("utf-8")
    pattern = r"^// turing-runtime-utility: turing\.world\.registry \[javascript-utility:sha256:([0-9a-f]{64})\]\r?\n"
    headers = list(re.finditer(pattern, document, re.M))
    if len(headers) != 1:
        raise ValueError("Original registry utility header is not unique")
    header = headers[0]
    next_header = document.find("// turing-runtime-utility:", header.end())
    if next_header < 0:
        raise ValueError("Original registry utility boundary is unavailable")
    segment = document[header.end():next_header]
    normalized = segment.replace("\r\n", "\n").strip()
    if hashlib.sha256(normalized.encode("utf-8")).hexdigest() != header[1]:
        raise ValueError("Original registry utility content key disagrees")
    if BEGIN in document:
        # A current matched build already carries the canonical method. Accept
        # only the entire exact canonical utility, including its content key;
        # an installed marker alone is not proof and cannot skip validation.
        if document.count(BEGIN) != 1 or document.count(END) != 1 or header[1] != canonical_sha or \
                method not in normalized:
            raise ValueError("Installed registry append is not the exact canonical utility")
        return source, {"schema":"studio-canonical-registry-append-v1",
            "source_sha256":hashlib.sha256(source).hexdigest(),
            "output_sha256":hashlib.sha256(source).hexdigest(),
            "canonical_utility_sha256":canonical_sha,
            "method_sha256":hashlib.sha256(method.encode("utf-8")).hexdigest(),
            "previous_inline_utility_sha256":header[1],"inline_utility_sha256":header[1],
            "already_canonical":True,"original_recovered_exactly":True,"compiler_imported":False,
            "worker_model_controls_unchanged":True,"published":False}
    classes = list(re.finditer(r"^class TuringWorldRegistry \{[\s\S]*?^}", segment, re.M))
    if len(classes) != 1 or classes[0].group().replace("\r\n", "\n") != expected_old_class:
        raise ValueError("Generated registry differs outside the canonical append method")
    anchor = "  object(identity) { return this.objects.get(identity) || null; }"
    if segment.count(anchor) != 1:
        raise ValueError("Original registry append seam is not unique")
    nl = "\r\n" if "\r\n" in segment else "\n"
    insertion = method.replace("\n", nl)
    replacement = segment.replace(anchor, insertion + anchor, 1)
    key = hashlib.sha256(replacement.replace("\r\n", "\n").strip().encode("utf-8")).hexdigest()
    new_header = header.group().replace(header[1], key)
    result = document[:header.start()] + new_header + replacement + document[next_header:]
    restored = result[:header.start()] + header.group() + replacement.replace(insertion + anchor, anchor, 1) + document[next_header:]
    if restored != document:
        raise AssertionError("Registry extension changed unrelated original bytes")
    output = result.encode("utf-8")
    return output, {"schema": "studio-canonical-registry-append-v1",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "canonical_utility_sha256": canonical_sha,
        "method_sha256": hashlib.sha256(method.encode("utf-8")).hexdigest(),
        "previous_inline_utility_sha256": header[1], "inline_utility_sha256": key,
        "original_recovered_exactly": True, "compiler_imported": False,
        "worker_model_controls_unchanged": True, "published": False}

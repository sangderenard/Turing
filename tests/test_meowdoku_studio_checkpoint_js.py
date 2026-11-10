"""Checkpoint contract uses the store's real receipt and existing scene ABI."""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from meowdoku import marble_objects as M


def subject():
    coordinates = [0,0,0, 0,0,1, 1,0,0, 0,1,0, 0,1,0, 0,1,0]
    geometry_hash = hashlib.sha256(struct.pack("<18fI", *coordinates, 0)).hexdigest()
    material = [0] * 16
    material[0], material[7] = .125, 1
    return {"v":1,"name":"Retained subject","generator":"fixture","design":{"kind":"triangle-fixture"},
            "physical":{"radius-mm":10.0},"geometry_sha256":geometry_hash,
            "materials":{"tables":{"pbr":{"stride":16,"data":material}}}}


def test_source_receipt_keeps_existing_authoritative_digest_bytes():
    value = subject()
    source = {k:value[k] for k in ("v","generator","design","materials","physical","geometry_sha256")}
    previous = json.dumps(source,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    assert M.source_receipt(value) == previous
    assert M.source_digest(value) == hashlib.sha256(previous).hexdigest()
    assert b'"radius-mm":10.0' in previous  # JS would reformat this as 10
    value["name"] = "A renamed object"
    assert M.source_receipt(value) == previous
    value["materials"]["tables"]["pbr"]["data"][0] = .2
    assert M.source_receipt(value) != previous


def test_checkpoint_node_contract_with_real_python_identity_receipt():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the browser checkpoint contract")
    env = {**os.environ,"STUDIO_CHECKPOINT_RECEIPT":M.source_receipt(subject()).decode()}
    result = subprocess.run([node,str(ROOT / "tests/meowdoku_studio_checkpoint_check.mjs")],
                            env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr

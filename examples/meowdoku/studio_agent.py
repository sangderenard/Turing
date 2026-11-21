"""Bounded Node scene-product worker and existing ray-job commit seam.

The existing HTTP/Nodus owner supplies authorization and account scope.
By default this module starts no listener, browser, renderer or job store.
An explicitly configured phong_host additionally admits render-captured PNG
products through the same authorized dispatch, never through ray-job commit.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 32 * 1024 * 1024


class StudioAgentError(ValueError):
    pass


def product_bytes(product: dict, *, content_type: str | None = None) -> bytes:
    if not isinstance(product, dict) or set(product) != {"content_type", "encoding", "sha256", "blob"}:
        raise StudioAgentError("Invalid Studio product envelope")
    if product["content_type"] not in ("application/json", "application/octet-stream", "image/png") or product["encoding"] != "base64":
        raise StudioAgentError("Unsupported Studio blob format")
    if content_type is not None and product["content_type"] != content_type:
        raise StudioAgentError("Wrong Studio product type")
    blob = product["blob"]
    if not isinstance(blob, str) or len(blob) > ((MAX_BYTES + 2) // 3) * 4:
        raise StudioAgentError("Studio blob exceeds byte budget")
    try:
        raw = base64.b64decode(blob, validate=True)
    except (ValueError, TypeError) as error:
        raise StudioAgentError("Invalid Studio blob encoding") from error
    if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != product["sha256"]:
        raise StudioAgentError("Studio blob digest or size differs")
    return raw


class StudioAgentProducts:
    def __init__(self, assets: Path, *, defaults: Path | None = None, node: str | None = None, timeout: float = 60, phong_host=None):
        self.assets = Path(assets).resolve(strict=True)
        self.defaults = Path(defaults).resolve(strict=True) if defaults is not None else None
        self.node = node
        if not 0 < timeout <= 120:
            raise StudioAgentError("Invalid Studio worker timeout")
        self.timeout = timeout
        self.phong_host = phong_host

    def build(self, request: dict) -> dict:
        """Product preparation inside an authorized Nodus dispatch.

        No ray job is submitted here. render-captured uses the optional bounded
        software-WebGL host; scene/ray-payload remain the pure Node builder.
        """
        if isinstance(request, dict) and request.get("operation") == "render-captured":
            if (set(request) != {"schema", "operation", "camera"} or
                    request.get("schema") != "turing.studio-agent-request.v1"):
                raise StudioAgentError("render-captured accepts only schema, operation and camera; no settings override")
            if self.phong_host is None:
                raise StudioAgentError("This host has no trusted captured Studio Phong scene")
            product = self.phong_host.render(request["camera"])
            product_bytes(product, content_type="image/png")
            return product
        raw = json.dumps(request, allow_nan=False, separators=(",", ":")).encode()
        if len(raw) > MAX_BYTES:
            raise StudioAgentError("Studio request exceeds byte budget")
        node = shutil.which(self.node or "node")
        if not node:
            raise StudioAgentError("Node is required for the existing Studio scene builders")
        command = [node, str(ROOT / "tools/studio_agent_product.mjs"), "--assets", str(self.assets)]
        if self.defaults:
            command += ["--defaults", str(self.defaults)]
        try:
            result = subprocess.run(command, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=ROOT, timeout=self.timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired as error:
            raise StudioAgentError("Studio scene preparation timed out; no ray job was created") from error
        if result.returncode:
            raise StudioAgentError(result.stderr.decode("utf8", errors="replace")[:2048].strip() or "Studio scene preparation failed")
        if len(result.stdout) > MAX_BYTES * 4 // 3 + 1024:
            raise StudioAgentError("Studio result exceeds byte budget")
        try:
            product = json.loads(result.stdout)
        except (ValueError, UnicodeError) as error:
            raise StudioAgentError("Studio worker returned invalid JSON") from error
        product_bytes(product)
        return product


def commit_studio_ray(product, *, authorize, photo_jobs, saved_objects, owners):
    """Turn the exact prepared MRP1 blob into an existing owned PhotoJobs job.

    Call ONLY from the authenticated owner/producer's commit callback, not
    ServicePeer.dispatch. The callback must check current server authority;
    a payload address/engine/capability string is never authorization.
    Errors propagate: the producer retains an uncertain outcome for inspection
    rather than automatically retrying an action that may already have run.
    """
    if not callable(authorize):
        raise StudioAgentError("Trusted ray-job authorization is required")
    if authorize() is not True:
        raise PermissionError("Server ray-job authorization was not granted")
    if not isinstance(owners, (tuple, list)) or not owners or not all(isinstance(owner, str) and owner for owner in owners):
        raise StudioAgentError("Existing job owner scope is required")
    raw = product_bytes(product, content_type="application/octet-stream")
    if raw[:4] != b"MRP1":
        raise StudioAgentError("Expected the existing MRP1 ray payload")
    # PhotoJobs remains the authority for saved geometry, physical parameters,
    # single-GPU admission, ownership, process lifecycle and actual status.
    return photo_jobs.start(raw, saved_objects, owners=tuple(owners))

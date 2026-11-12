"""Bounded internal execution of the existing standalone Studio WebGL2 host.

Trusted startup supplies a captured full-scene asset and Chrome executable.
Requests supply only camera matrices/dimensions, never paths, URLs or GLSL.
This is captured-scene Phong, not settings-to-scene rendering, field/DOF replay,
an HTTP account endpoint, or a replacement for PhotoJobs. Software WebGL keeps
this optional execution lane off PhotoJobs' single native GPU.
"""
from __future__ import annotations

import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import tempfile
import threading

from .studio_agent import ROOT, MAX_BYTES, StudioAgentError

WEB = ROOT / "examples/meowdoku/web"
# Exact transitive module set of standalone_studio.js, no repository server.
MODULES = ("standalone_studio.js", "marble_studio_scene_asset.js",
           "marble_studio_subject_renderer.js", "marble_glass_preview.js",
           "marble_bubbles.js", "marble_mesh.js")


def validate_camera(camera):
    if not isinstance(camera, dict) or set(camera) != {"view", "projection", "width", "height"}:
        raise StudioAgentError("Phong camera requires view, projection, width and height")
    for key in ("view", "projection"):
        value = camera[key]
        try:
            valid = isinstance(value, list) and len(value) == 16 and all(
                type(x) in (float, int) and math.isfinite(x) for x in value)
        except OverflowError:
            valid = False
        if not valid:
            raise StudioAgentError(f"Phong {key} requires 16 finite numbers")
    if [camera["view"][i] for i in (3, 7, 11, 15)] != [0, 0, 0, 1]:
        raise StudioAgentError("Phong view must be affine")
    for key in ("width", "height"):
        if type(camera[key]) is not int or not 16 <= camera[key] <= 2048:
            raise StudioAgentError(f"Phong {key} must be 16..2048")
    return json.loads(json.dumps(camera, allow_nan=False))


class StudioPhongHost:
    """One render at a time, temporary loopback capability, disposable profile.

    The enclosing authenticated service owns admission/authorization. This class
    returns bytes only, saves no user artifact, and never submits a ray job.
    The existing JS hydrator verifies scene identity and every GPU input before
    drawing; unsupported captured resources fail rather than being replaced.
    """
    def __init__(self, scene_asset: Path, *, chrome: Path, timeout=60):
        if type(timeout) not in (int, float) or not 0 < timeout <= 120:
            raise StudioAgentError("Phong timeout must be 0..120 seconds")
        self.chrome = Path(chrome).resolve(strict=True)
        self.timeout = timeout
        path = Path(scene_asset).resolve(strict=True)
        if path.stat().st_size > MAX_BYTES:
            raise StudioAgentError("Captured Studio scene exceeds 32 MiB")
        self.scene = path.read_bytes()
        asset = json.loads(self.scene)
        if asset.get("schema") != "turing.studio-scene-asset.v1":
            raise StudioAgentError("A captured full Studio scene asset is required")
        self.identity = asset.get("identity")
        self.shader_sha256 = {stage: hashlib.sha256(asset["render"]["shaders"][stage].encode()).hexdigest()
                              for stage in ("vertex", "fragment")}
        self.sources = {name: (WEB / name).read_bytes() for name in MODULES}
        self.lock = threading.Lock()

    def describe(self):
        return {"operation": "render-captured", "scene_identity": self.identity,
                "scene_sha256": hashlib.sha256(self.scene).hexdigest(),
                "shader_sha256": dict(self.shader_sha256), "runtime": "Chrome WebGL2 SwiftShader",
                "limits": ["trusted captured full scene only; no settings overrides",
                           "no active lightfield, shadows, thermal or DOF replay",
                           "still PNG only; no simulation or animation", "16..2048 pixels per dimension"],
                "module_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in self.sources.items()}}

    def render(self, camera):
        camera = validate_camera(camera)
        if not self.lock.acquire(blocking=False):
            raise StudioAgentError("Captured Studio Phong host is busy")
        try:
            return self._render(camera)
        finally:
            self.lock.release()

    def _render(self, camera):
        token = secrets.token_hex(32)
        prefix = "/" + token + "/"
        done = threading.Event()
        receipts = []
        page = b'''<!doctype html><meta charset="utf-8"><canvas></canvas><script type="module" src="run.js"></script>'''
        script = b'''import {createStandaloneStudioScene} from './standalone_studio.js';
let runner;
try {
 const [sceneAsset,camera]=await Promise.all(['scene.json','camera.json'].map(p=>fetch(p).then(r=>r.json())));
 runner=await createStandaloneStudioScene({canvas:document.querySelector('canvas'),sceneAsset});
 const frame=runner.render(camera);
 await fetch('receipt',{method:'POST',body:JSON.stringify({ok:true,frame})});
} catch(error) {
 await fetch('receipt',{method:'POST',body:JSON.stringify({ok:false,error:String(error).slice(0,2048)})});
} finally {runner?.dispose();}
'''
        files = {**self.sources, "index.html": page, "run.js": script,
                 "scene.json": self.scene, "camera.json": json.dumps(camera).encode()}

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(5)

            def log_message(self, *args):
                pass

            def allowed(self):
                return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

            def do_GET(self):
                name = self.path[len(prefix):] if self.path.startswith(prefix) else ""
                if not self.allowed() or name not in files:
                    self.send_error(404)
                    return
                raw = files[name]
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript" if name.endswith(".js") else
                                 "application/json" if name.endswith(".json") else "text/html")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; connect-src 'self'")
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                origin = f"http://127.0.0.1:{self.server.server_port}"
                if (not self.allowed() or self.path != prefix + "receipt" or
                        self.headers.get("Origin") != origin or done.is_set()):
                    self.send_error(404)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= MAX_BYTES:
                        raise ValueError("receipt size")
                    raw = self.rfile.read(size)
                    if len(raw) != size:
                        raise ValueError("incomplete receipt")
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise ValueError("receipt object")
                except (ValueError, OSError):
                    self.send_error(400)
                    return
                receipts.append(value)
                self.send_response(204)
                self.end_headers()
                done.set()

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        httpd.daemon_threads = True
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="studio-phong-", ignore_cleanup_errors=True) as profile:
                command = [str(self.chrome), "--headless=new", "--no-first-run", "--disable-extensions",
                           "--disable-background-networking", "--disable-sync", "--no-default-browser-check",
                           "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                           f"--user-data-dir={profile}", f"http://127.0.0.1:{httpd.server_port}{prefix}index.html"]
                process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                           start_new_session=os.name != "nt")
                try:
                    if not done.wait(self.timeout):
                        raise StudioAgentError("Captured Studio Phong render timed out")
                finally:
                    # Headless Chrome has child processes. Terminate only this
                    # disposable invocation, never a user's browser/profile.
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    else:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        if os.name == "nt":
                            process.kill()
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)
        result = receipts[0]
        if result.get("ok") is not True:
            raise StudioAgentError(str(result.get("error", "Captured Studio render failed"))[:2048])
        frame = result.get("frame", {})
        if frame.get("scene_identity") != self.identity or frame.get("camera") != camera:
            raise StudioAgentError("Captured Studio result identity or camera differs")
        data_url = frame.get("data_url", "")
        if not isinstance(data_url, str) or not data_url.startswith("data:image/png;base64,"):
            raise StudioAgentError("Captured Studio did not return a PNG")
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
        if (len(raw) > MAX_BYTES or raw[:8] != b"\x89PNG\r\n\x1a\n" or len(raw) < 24 or
                int.from_bytes(raw[16:20], "big") != camera["width"] or
                int.from_bytes(raw[20:24], "big") != camera["height"]):
            raise StudioAgentError("Captured Studio PNG dimensions or size differ")
        return {"content_type": "image/png", "encoding": "base64", "sha256": hashlib.sha256(raw).hexdigest(),
                "blob": base64.b64encode(raw).decode()}

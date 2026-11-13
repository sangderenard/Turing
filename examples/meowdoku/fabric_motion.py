"""Owned Studio sessions backed by the persistent native Nodus spring row host."""

from __future__ import annotations

import math
import os
from pathlib import Path
import secrets
import struct
import subprocess
import threading
import time

from .fabric_udp import FabricUdpReceiver


NETWORKS, NODES, FIELDS = 8, 256, 38
COUNT = NETWORKS * NODES * FIELDS


class FabricMotionError(ValueError):
    pass


class FabricMotion:
    def __init__(self, host: Path, plugin: Path, core_dir: Path, state_dir: Path,
                 port: int = 48768):
        self.host, self.plugin, self.core_dir = map(Path, (host, plugin, core_dir))
        for path in (self.host, self.plugin):
            if not path.is_file():
                raise FabricMotionError(f"native fabric component missing: {path}")
        self.state_dir = Path(state_dir) / "fabric-motion"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.port = port
        self.receiver = FabricUdpReceiver(port=port).start()
        self.lock = threading.Lock()
        self.sessions = {}

    def _validate(self, data, side):
        if not isinstance(side, int) or not 2 <= side <= 16:
            raise FabricMotionError("grid side must be 2–16")
        if not isinstance(data, list) or len(data) != COUNT:
            raise FabricMotionError("expected one 8×256×38 row configuration")
        if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in data):
            raise FabricMotionError("row configuration must be finite")
        if any(data[node*FIELDS+6] != (1 if node < side*side else 0)
               for node in range(NETWORKS*NODES)):
            raise FabricMotionError("only the selected sheet may occupy row nodes")
        for node in range(side*side):
            base = node*FIELDS
            if (data[base+7] not in (0, 1, 2) or not 0 <= data[base+12] <= 1000 or
                    not 0 <= data[base+13] <= 10 or
                    any(abs(data[base+axis]) > 64 for axis in range(3))):
                raise FabricMotionError("fabric node exceeds the Studio row limits")
            for edge in range(4):
                if data[base+33+edge] not in (0, 1):
                    raise FabricMotionError("invalid edge mask")
                if data[base+33+edge] and (data[base+17+edge] != int(data[base+17+edge]) or
                                           not 0 <= data[base+17+edge] < side*side):
                    raise FabricMotionError("invalid neighbor")
        return data

    def _owner_session(self, owner, token):
        session = self.sessions.get(token)
        if session is None or session["owner"] != owner:
            raise FabricMotionError("fabric session not found")
        return session

    def start(self, owner, data, side, *, backend="cpu", substeps=1, gpu_target_ms=0,
              floor_y=None, table_contact=None):
        data = self._validate(data, side)
        if backend not in ("cpu", "vulkan"):
            raise FabricMotionError("fabric backend must be cpu or vulkan")
        if type(substeps) is not int or not 1 <= substeps <= 128:
            raise FabricMotionError("fabric substeps must be an integer from 1 to 128")
        if not isinstance(gpu_target_ms, (float, int)) or not math.isfinite(gpu_target_ms) or not 0 <= gpu_target_ms <= 8:
            raise FabricMotionError("GPU target must be 0–8 ms")
        if floor_y is not None and (not isinstance(floor_y, (float, int)) or
                                    not math.isfinite(floor_y) or not -64 <= floor_y <= 64):
            raise FabricMotionError("floor contact must be a finite Studio height")
        if table_contact is not None and (not isinstance(table_contact, list) or
            len(table_contact) != 6 or table_contact[0] != 2 or
            not all(isinstance(v, (float, int)) and math.isfinite(v) for v in table_contact) or
            not (table_contact[2] < table_contact[3] and table_contact[4] < table_contact[5])):
            raise FabricMotionError("invalid stage table contact")
        with self.lock:
            for token, session in list(self.sessions.items()):
                if session["process"].poll() is not None or time.monotonic()-session["last_seen"]>600:
                    if session["process"].poll() is None:
                        session["process"].stdin.close()
                        try:
                            session["process"].wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            session["process"].terminate()
                            session["process"].wait(timeout=1)
                    self.sessions.pop(token)
                    session["path"].unlink(missing_ok=True)
            if len(self.sessions) >= 4:
                raise FabricMotionError("fabric service is at its four-session limit")
            token = secrets.token_urlsafe(24)
            identity = secrets.randbelow(2**53-1) + 1
            data[37] = identity
            path = self.state_dir / (token + ".bin")
            path.write_bytes(struct.pack("<" + "d"*COUNT, *data))
            env = os.environ.copy()
            env["NODUS_FABRIC_BACKEND"] = backend
            env["PATH"] = str(self.core_dir) + os.pathsep + env.get("PATH", "")
            try:
                process = subprocess.Popen([str(self.host), str(self.plugin), str(path),
                                            "127.0.0.1", str(self.port)],
                                           stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, env=env,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except OSError:
                path.unlink(missing_ok=True)
                raise FabricMotionError("native fabric host could not start") from None
            try:
                commands = f"substeps {substeps}\ntarget {gpu_target_ms}\n"
                if floor_y is not None:
                    commands += f"floor {floor_y}\n"
                if table_contact is not None:
                    commands += "table " + " ".join(map(str, table_contact[1:])) + "\n"
                process.stdin.write(commands.encode())
                process.stdin.flush()
            except OSError:
                process.terminate()
                process.wait(timeout=1)
                path.unlink(missing_ok=True)
                raise FabricMotionError("native fabric host rejected settings") from None
            self.sessions[token] = {"owner": owner, "identity": identity, "process": process,
                                    "path": path, "side": side, "frozen": False,
                                    "backend": backend, "substeps": substeps,
                                    "gpu_target_ms": gpu_target_ms,
                                    "last_seen": time.monotonic()}
            return {"token": token, "identity": identity, "side": side, "running": True,
                    "backend": backend, "substeps": substeps, "gpuTargetMs": gpu_target_ms}

    def state(self, owner, token):
        with self.lock:
            session = self._owner_session(owner, token)
            session["last_seen"] = time.monotonic()
            process = session["process"]
            frame = self.receiver.latest(session["identity"])
            if frame is not None:
                positions = frame["positions"]
                if len(positions) != session["side"]**2 or any(node not in positions for node in range(session["side"]**2)):
                    frame = None
                else:
                    frame["positions"] = [positions[node] for node in range(session["side"]**2)]
            return {"running": process.poll() is None, "frozen": session["frozen"],
                    "backend": session["backend"], "substeps": session["substeps"],
                    "gpuTargetMs": session["gpu_target_ms"],
                    "identity": session["identity"], "side": session["side"], "frame": frame}

    def command(self, owner, token, action):
        if action not in ("pause", "resume", "stop"):
            raise FabricMotionError("unknown fabric action")
        with self.lock:
            session = self._owner_session(owner, token)
            session["last_seen"] = time.monotonic()
            process = session["process"]
            if process.poll() is not None:
                raise FabricMotionError("native fabric host exited")
            try:
                process.stdin.write(("quit" if action == "stop" else action).encode() + b"\n")
                process.stdin.flush()
            except OSError:
                raise FabricMotionError("native fabric host is unavailable") from None
            if action == "stop":
                self.sessions.pop(token)
                process.stdin.close()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=1)
                session["path"].unlink(missing_ok=True)
            else:
                session["frozen"] = action == "pause"
            return {"running": action != "stop", "frozen": session["frozen"]}

    def close(self):
        with self.lock:
            for session in self.sessions.values():
                process = session["process"]
                process.stdin.close()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=1)
                session["path"].unlink(missing_ok=True)
            self.sessions.clear()
        self.receiver.close()

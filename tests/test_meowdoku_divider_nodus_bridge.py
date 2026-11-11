"""examples/meowdoku/divider_nodus_bridge.py against a FAKE Nodus contract endpoint speaking the real 72-byte frames.
No Nodus build, no compiler: the fake plays the compile host.  Ephemeral loopback ports, temp dirs only."""
from __future__ import annotations

import json
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import divider_nodus_bridge as br  # noqa: E402
from meowdoku import source_divider as sd  # noqa: E402
from meowdoku.state_paths import StatePaths  # noqa: E402
from meowdoku.compile_artifacts import Artifacts
from meowdoku.workspace_accounting import Store

TOOLS = br.DEFAULT_NODUS_TOOLS
FRAME = struct.Struct("<II8Q")                    # the layout in contract_service_peer.py
MAGIC = 0x31434E4E
SUBMIT, ADMITTED, ROUTE, DECISION, HELLO = 6, 7, 8, 9, 5

ONE = "def scale(x: AbstractTensor, y: AbstractTensor) -> AbstractTensor:\n    return x * y\n"
SCALAR = "def scale(x: float, y: float) -> float:\n    return x * y\n"
TWO = ONE + "\ndef shift(x: AbstractTensor) -> AbstractTensor:\n    return x + 1.0\n"
BAD = "def tensorish(x: list) -> AbstractTensor:\n    return x\n"
CHAIN = ("def a(x: float) -> float:\n    return b(x)\n\ndef b(x: float) -> float:\n    return a(x)\n")


class FakeHost:
    """Plays the compile host on a loopback port: Hello, Submit -> Admitted + Route(status), then reads the Decision."""

    def __init__(self, jobs_dir, status=1, *, silent=False, write_result=True):
        self.jobs_dir, self.status, self.silent, self.write_result = Path(jobs_dir), status, silent, write_result
        self.connections = 0
        self.submits, self.request_existed, self.decisions, self.hellos = [], [], [], []
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(4)
        self.port = self.server.getsockname()[1]
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _read(self, conn):
        data = b""
        while len(data) < FRAME.size:
            chunk = conn.recv(FRAME.size - len(data))
            if not chunk:
                return None
            data += chunk
        magic, kind, *values = FRAME.unpack(data)
        assert magic == MAGIC
        return kind, values

    def _serve(self):
        self.server.settimeout(0.1)
        while not self.stop.is_set():
            try:
                conn, _ = self.server.accept()
            except (socket.timeout, OSError):
                continue
            self.connections += 1
            conn.settimeout(5)
            try:
                frame = self._read(conn)
                if frame is None:
                    continue
                self.hellos.append(frame)
                frame = self._read(conn)
                if frame is None or frame[0] != SUBMIT:
                    continue
                values = frame[1]
                self.submits.append(tuple(values[:4]))
                cid = values[0]
                self.request_existed.append((self.jobs_dir / f"{cid}.request.json").is_file())
                if self.silent:
                    self.stop.wait(3)
                    continue
                conn.sendall(FRAME.pack(MAGIC, ADMITTED, cid, *([0] * 7)))
                if self.write_result:
                    product = self.jobs_dir / str(cid)
                    product.mkdir(parents=True, exist_ok=True)
                    (product / "compile-host.result.json").write_text(
                        json.dumps({"status": self.status, "error": "fake host says no" if self.status >= 16 else "",
                                    "wall_seconds": 1.5}), encoding="utf-8")
                conn.sendall(FRAME.pack(MAGIC, ROUTE, cid, values[1], values[2], values[3], 1, 1, 0, self.status))
                decision = self._read(conn)
                if decision is not None:
                    self.decisions.append(decision)
            except OSError:
                pass
            finally:
                conn.close()

    def close(self):
        self.stop.set()
        self.server.close()
        self.thread.join(2)


def _setup(tmp_path, text=ONE, backends=("c",), name="m.py"):
    state = StatePaths(tmp_path / "state")
    db = sd.connect(state.accounts_db)
    db.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY)")
    db.execute("INSERT OR IGNORE INTO users VALUES(1)")
    job = sd.submit(db, state, sd.intake_text(state, name, text), owner_key="account:1", backends=backends)
    return state, db, job


def _route(tmp_path, state, host=None, port=1):
    jobs = state.content_dir / "workspaces"
    jobs.mkdir(parents=True, exist_ok=True)
    store = Store(state.accounts_db, jobs, capacity_bytes=100_000_000, user_quota_bytes=50_000_000, free_floor_bytes=0)
    artifacts = Artifacts(state, store, reservation_bytes=1_000_000, review_after_s=3600)
    return br.NodusCompileRoute("127.0.0.1", host.port if host else port, 7, jobs, TOOLS,
                                source_root=state.jobs_dir, artifacts=artifacts)


def _attempt(db, job, piece_id=0, backend="c"):
    return dict(db.execute("SELECT * FROM divider_attempts WHERE job_id=? AND piece_id=? AND backend=?",
                           (job, piece_id, backend)).fetchone())


def test_request_exists_before_submit_and_schema(tmp_path):
    state, db, job = _setup(tmp_path)
    route = _route(tmp_path, state)
    host = FakeHost(route.jobs_dir, status=1)
    route.port = host.port
    try:
        result = br.run_one(db, route, lease_owner="w", timeout_s=5, poll_s=0.02)
    finally:
        host.close()
    assert host.request_existed == [True]
    assert host.hellos[0][1][0] == 2 and host.hellos[0][1][1] == 7          # engine role, service id
    assert host.submits == [(result["contract_id"], 7, 1, 0)]
    assert host.decisions and host.decisions[0][1][7] == 1                  # committed
    request = json.loads((route.jobs_dir / f"{result['contract_id']}.request.json").read_text(encoding="utf-8"))
    assert request["row_entry"] == "scale" and request["backend"] == "c" and request["source_text"] == ONE
    assert [c["name"] for c in request["columns"]] == ["x", "y", "result"]
    assert [c["role"] for c in request["columns"]] == ["input", "input", "output"]
    assert all(c["rank"] == 1 and c["shape"] is None for c in request["columns"])
    assert request["columns"][2]["shape_from"] == "x"
    import yaml
    contract = yaml.safe_load(Path(request["extraction_contract"]).read_text(encoding="utf-8"))
    assert [v["parameter"] for v in contract["program_abi"]["values"]] == ["x", "y"]
    assert all(v["rank"] == 1 and v["storage"] == "span" for v in contract["program_abi"]["values"])
    assert request["output_dir"] == str((route.jobs_dir / str(result["contract_id"])).resolve())
    assert Path(request["extraction_contract"]).is_absolute()
    row = _attempt(db, job)
    assert row["state"] == "complete" and "nodus-row-tool: Adopted" in row["stage"] and row["seconds"] == 1.5


def test_each_status_maps_to_attempt_outcome(tmp_path):
    expected = {1: "complete", 2: "complete", 3: "complete", 16: "refused", 17: "error", 18: "error"}
    for code, state_name in expected.items():
        sub = tmp_path / f"s{code}"
        sub.mkdir()
        state, db, job = _setup(sub)
        route = _route(sub, state)
        host = FakeHost(route.jobs_dir, status=code)
        route.port = host.port
        try:
            result = br.run_one(db, route, lease_owner="w", timeout_s=5, poll_s=0.02)
        finally:
            host.close()
        row = _attempt(db, job)
        assert row["state"] == state_name, (code, row)
        assert result["recorded"] == state_name and result["status"] == br.STATUS_NAMES[code]
        if code == 16:
            assert row["error_type"] == "RequestMissing"
        if code in (17, 18):
            assert row["error"] == "fake host says no"
        assert sd.summary(db, job)["lowered_pieces"] == (1 if state_name == "complete" else 0)


def test_request_missing_is_a_refusal_not_compiled(tmp_path):
    state, db, job = _setup(tmp_path)
    route = _route(tmp_path, state)
    host = FakeHost(route.jobs_dir, status=16, write_result=False)
    route.port = host.port
    try:
        br.run_one(db, route, lease_owner="w", timeout_s=5, poll_s=0.02)
    finally:
        host.close()
    row = _attempt(db, job)
    assert row["state"] == "refused" and row["error_type"] == "RequestMissing"
    assert sd.summary(db, job)["pieces"][0]["best"] is None


def test_cancelled_is_not_a_verdict(tmp_path):
    state, db, job = _setup(tmp_path)
    route = _route(tmp_path, state)
    host = FakeHost(route.jobs_dir, status=19)
    route.port = host.port
    try:
        result = br.run_one(db, route, lease_owner="w", timeout_s=5, poll_s=0.02)
    finally:
        host.close()
    assert result["recorded"] is None and result["state"] == "unfinished" and result["status"] == "Cancelled"
    assert _attempt(db, job)["state"] == "leased"


def test_timeout_leaves_piece_leased_and_unfinished(tmp_path):
    state, db, job = _setup(tmp_path)
    route = _route(tmp_path, state)
    host = FakeHost(route.jobs_dir, silent=True)
    route.port = host.port
    try:
        started = time.monotonic()
        result = br.run_one(db, route, lease_owner="w", timeout_s=0.5, poll_s=0.05)
        assert time.monotonic() - started < 3
    finally:
        host.close()
    assert result["status"] == "Timeout" and result["recorded"] is None
    row = _attempt(db, job)
    assert row["state"] == "leased" and row["lease_owner"] == "w"
    again = br.run_one(db, route, lease_owner="w", timeout_s=0.5)               # same owner, lease alive -> nothing taken
    assert again["state"] == "busy"


def test_inexpressible_piece_refused_without_contacting_endpoint(tmp_path):
    state, db, job = _setup(tmp_path, BAD)
    route = _route(tmp_path, state)
    host = FakeHost(route.jobs_dir)
    route.port = host.port
    try:
        result = br.run_one(db, route, lease_owner="w", timeout_s=2, poll_s=0.02)
    finally:
        host.close()
    assert host.connections == 0
    assert result["recorded"] == "refused" and result["detail"]["error_type"] == "RequestNotExpressible"
    assert "'x'" in result["detail"]["error"] and "AbstractTensor" in result["detail"]["error"]
    assert not list(route.jobs_dir.glob("*.request.json"))


def test_multi_name_piece_and_unserved_backend_refused(tmp_path):
    state, db, job = _setup(tmp_path, CHAIN)
    route = _route(tmp_path, state)
    host = FakeHost(route.jobs_dir)
    route.port = host.port
    try:
        result = br.run_one(db, route, lease_owner="w", timeout_s=2, poll_s=0.02)
        assert host.connections == 0 and "exactly one function" in result["detail"]["error"]
        sub = tmp_path / "w"
        sub.mkdir()
        state2, db2, job2 = _setup(sub, ONE, backends=("wasm",))
        result2 = br.run_one(db2, _route(sub, state2, host), lease_owner="w", timeout_s=2, poll_s=0.02)
    finally:
        host.close()
    assert host.connections == 0 and result2["detail"]["error_type"] == "BackendNotServed"


def test_contract_ids_stable_and_distinct(tmp_path):
    jobs = tmp_path / "ids"
    jobs.mkdir()
    ident = {"job_id": "aaaaaaaaaaaa", "piece_id": 0, "backend": "c", "generation": 1}
    first = br.allocate_contract_id(ident, jobs)
    assert br.allocate_contract_id(ident, jobs) == first and 0 < first < 2 ** 63
    seen = {first}
    for variant in ({**ident, "piece_id": 1}, {**ident, "job_id": "bbbbbbbbbbbb"}, {**ident, "generation": 2}, {**ident, "backend": "llvm"}):
        cid = br.allocate_contract_id(variant, jobs)
        assert cid not in seen
        seen.add(cid)
    # a forced collision on the hashed candidate is skipped by incrementing, and the owner keeps its own id
    other = {"job_id": "cccccccccccc", "piece_id": 9, "backend": "c", "generation": 1}
    (tmp_path / "scratch").mkdir(exist_ok=True)
    cand = br.allocate_contract_id(other, tmp_path / "scratch")
    (jobs / f"{cand}.divider.json").write_text(json.dumps(ident), encoding="utf-8")       # someone else owns that id here
    moved = br.allocate_contract_id(other, jobs)
    assert moved != cand and br.allocate_contract_id(other, jobs) == moved


def test_one_piece_per_run_one_even_with_several_queued(tmp_path):
    state, db, job = _setup(tmp_path, TWO)
    route = _route(tmp_path, state)
    host = FakeHost(route.jobs_dir, status=1)
    route.port = host.port
    try:
        first = br.run_one(db, route, lease_owner="w", timeout_s=5, poll_s=0.02)
        counts = {r[0]: r[1] for r in db.execute("SELECT state, COUNT(*) FROM divider_attempts GROUP BY state")}
        assert counts == {"complete": 1, "queued": 1} and len(host.submits) == 1
        second = br.run_pending(db, state, route, max_pieces=5, lease_owner="w", timeout_s=5, poll_s=0.02)
    finally:
        host.close()
    assert first["piece_id"] != second[0]["piece_id"] and len(second) == 1 and len(host.submits) == 2
    assert br.run_one(db, route, lease_owner="w", timeout_s=1) is None


def test_connect_failure_leaves_lease(tmp_path):
    state, db, job = _setup(tmp_path)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    result = br.run_one(db, _route(tmp_path, state, port=port), lease_owner="w", timeout_s=1, poll_s=0.02)
    assert result["status"] == "ConnectFailed" and result["recorded"] is None
    assert _attempt(db, job)["state"] == "leased"


def test_configured_backend_never_relabels_an_existing_attempt(tmp_path):
    state, db, job = _setup(tmp_path, backends=('llvm',))
    route = _route(tmp_path, state)
    route.backend = 'c'
    result = br.run_one(db, route, lease_owner='w', timeout_s=1)
    assert result['recorded'] == 'refused'
    assert result['detail']['error_type'] == 'BackendNotServed'
    assert _attempt(db, job, backend='llvm')['state'] == 'refused'
    assert not list(route.jobs_dir.glob('*.request.json'))


def test_cli_dispatch_refused_without_contacting_host(tmp_path):
    state, db, job = _setup(tmp_path)
    jobs = tmp_path / "compile"
    jobs.mkdir()
    host = FakeHost(jobs)
    try:
        proc = subprocess.run([sys.executable, str(ROOT / "examples/meowdoku/divider_nodus_bridge.py"),
                               "run-once", "--db", str(state.accounts_db), "--jobs-dir", str(jobs),
                               "--port", str(host.port), "--service-id", "7"], capture_output=True, text=True, timeout=30)
    finally:
        host.close()
    assert proc.returncode == 2 and "site-only" in proc.stderr
    assert host.connections == 0 and _attempt(db, job)["state"] == "queued"


def test_scalar_pieces_refused_and_no_rank_zero_request_is_ever_produced(tmp_path):
    state, db, job = _setup(tmp_path, SCALAR)
    route = _route(tmp_path, state)
    host = FakeHost(route.jobs_dir)
    route.port = host.port
    try:
        result = br.run_one(db, route, lease_owner="w", timeout_s=2, poll_s=0.02)
        assert host.connections == 0 and result["recorded"] == "refused"
        assert "rank-0" in result["detail"]["error"] and "turing_row_tool_compiler.py" in result["detail"]["error"]
        sub = tmp_path / "b"
        sub.mkdir()
        state2, db2, job2 = _setup(sub, TWO)
        route2 = _route(sub, state2, host)
        for _ in range(2):
            br.run_one(db2, route2, lease_owner="w2", timeout_s=2, poll_s=0.02)
    finally:
        host.close()
    requests = list(route2.jobs_dir.glob("*.request.json"))
    assert len(requests) == 2
    for path in requests:
        for column in json.loads(path.read_text(encoding="utf-8"))["columns"]:
            assert column["rank"] != 0 and column["shape"] != []

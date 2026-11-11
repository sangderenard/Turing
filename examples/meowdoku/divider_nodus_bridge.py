"""Divider -> Nodus compile host bridge: drain source_divider's lease queue ONE piece at a time through a Nodus contract port.

source_divider has a lease queue (``claim_next``) and no worker that talks to Nodus.  Nodus has a compile HOST
(``include/turing_compile_host_tool.h``) that takes compile contracts strictly one at a time.  This module is the producer between
them.  It adds no queue, no compiler and no wire protocol:

  claim_next (source_divider)  ->  request_for_piece: ``<jobs_dir>/<id>.request.json`` (the schema of
  nodus/tools/turing_row_tool_compiler.py --request) written atomically BEFORE the contract is submitted
  ->  submit_and_wait: ONE contract (id, service_id, 1, 0) over the loopback contract port using nodus' own
  ``contract_service_peer.EnginePeer`` (the producer side of ServicePeer; the host side is the compile host itself)
  ->  the Result payload is a CompileStatus  ->  ``source_divider.complete`` / ``refuse`` records the attempt.

File convention (owned by the Nodus host, see the header): request ``<jobs_dir>/<id>.request.json``, product directory
``<jobs_dir>/<id>/``, outcome ``<jobs_dir>/<id>/compile-host.result.json``.  The bridge additionally writes
``<jobs_dir>/<id>.divider.json`` (who the id belongs to).

Contract id: ``sha256(job_id NUL piece_id NUL backend NUL generation)`` -> first 8 bytes big-endian, masked to 63 bits, never 0.
``generation`` is the lease generation claim_next returned, i.e. the attempt number of that (piece, backend) row.  The same
(piece, attempt) therefore always yields the same id.  Uniqueness is not left to the hash: ``<id>.divider.json`` names its owner, and
an id already owned by a different (job, piece, backend, generation), or whose request/product name exists with no sidecar, is
skipped by incrementing.  Within one jobs_dir ids never collide across jobs.

RECORDING.  Attempt rows are keyed (job, piece, backend) and ``complete``/``refuse`` only update the row that was leased, so the
attempt is recorded on the claimed row (its own backend, e.g. ``c`` or ``llvm``) and the stage text carries ``nodus-row-tool``.
A row with the literal backend ``nodus-row-tool`` cannot be created without editing source_divider (BACKENDS / submit).

Single in flight: run_one never claims while this lease owner still holds an unexpired lease.  A timeout, a connection failure,
a Cancelled status or an unknown status is NOT a verdict: the lease is left in place (it expires through
``source_divider.reclaim_expired`` and the piece is requeued with the next generation, hence a new contract id) and the outcome is
reported.  Nothing here compiles, imports or runs the piece.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import select
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "examples")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from meowdoku import source_divider as sd                # noqa: E402
from meowdoku.state_paths import StatePaths              # noqa: E402

BACKEND_NAME = "nodus-row-tool"
DRIVER_BACKENDS = ("c", "llvm", "fortran")              # turing_row_tool_compiler._BACKENDS
DEFAULT_NODUS_TOOLS = ROOT.parent / "nodus" / "tools"
DEFAULT_TIMEOUT_S = 600.0
STATUS_NAMES = {1: "Adopted", 2: "CompiledNotAdopted", 3: "ExistingProduct", 16: "RequestMissing", 17: "CompileFailed",
                18: "AdoptFailed", 19: "Cancelled"}
_MASK63 = (1 << 63) - 1
TENSOR_TYPE = "src.common.tensors.abstraction.AbstractTensor"


@dataclass
class NodusCompileRoute:
    """Where compile requests go: the contract port of a canvas hosting a TuringCompileHost, and its jobs directory.

    ``service_id`` is the engine participant id (Hello identity and the contract's engine field).  ``source_root`` is where
    ``<job_id>/source/<file>`` of the divider lives (default ``jobs_dir``: the divider's state jobs dir and the compile host's
    jobs dir may be the same directory; prefer a dedicated compile jobs dir and pass the divider's as ``source_root``)."""
    host: str
    port: int
    service_id: int
    jobs_dir: Path
    nodus_tools: Path = None
    source_root: Path | None = None
    backend: str | None = None                  # driver backend; default = the claimed attempt's backend (must be c|llvm|fortran)
    extraction_contract: Path | None = None     # default source_divider.EXTRACTION_CONTRACT
    turing_root: Path | None = None             # default this repository

    def __post_init__(self):
        self.jobs_dir = Path(self.jobs_dir)
        self.nodus_tools = Path(self.nodus_tools) if self.nodus_tools is not None else DEFAULT_NODUS_TOOLS
        self.source_root = Path(self.source_root) if self.source_root is not None else self.jobs_dir


@dataclass
class CompileOutcome:
    status_code: int | None
    status_name: str
    accepted: bool                      # the host's Route (the Result) arrived and was committed
    result_path: Path | None            # compile-host.result.json when present
    detail: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------------------- nodus peer module
_PEER_MODULES: dict = {}


def load_peer_module(nodus_tools):
    """``contract_service_peer`` loaded from ``nodus_tools`` by path (sys.path untouched); one load per directory."""
    path = (Path(nodus_tools) / "contract_service_peer.py").resolve()
    module = _PEER_MODULES.get(path)
    if module is None:
        if not path.is_file():
            raise FileNotFoundError(f"contract_service_peer.py not found in {nodus_tools}")
        spec = importlib.util.spec_from_file_location(f"_nodus_contract_service_peer_{len(_PEER_MODULES)}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _PEER_MODULES[path] = module
    return module


# ------------------------------------------------------------------------------------------------------------------- request
def _identity(piece_row: dict) -> dict:
    return {"job_id": str(piece_row["job_id"]), "piece_id": int(piece_row["piece_id"]), "backend": str(piece_row["backend"]),
            "generation": int(piece_row["generation"])}


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def allocate_contract_id(identity: dict, jobs_dir: Path) -> int:
    """Stable 63-bit id for (job, piece, backend, generation); see the module docstring for the derivation and the collision rule."""
    jobs_dir = Path(jobs_dir)
    key = "\0".join(str(identity[k]) for k in ("job_id", "piece_id", "backend", "generation"))
    candidate = (int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") & _MASK63) or 1
    while True:
        sidecar = jobs_dir / f"{candidate}.divider.json"
        if sidecar.is_file():
            try:
                owner = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                owner = None
            if owner == identity:
                return candidate
        elif not (jobs_dir / f"{candidate}.request.json").exists() and not (jobs_dir / str(candidate)).exists():
            _write_atomic(sidecar, json.dumps(identity))
            return candidate
        candidate = (candidate + 1 & _MASK63) or 1


def _annotation_name(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):         # string annotation
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return None
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None


def _express(piece_row: dict, source_root: Path):
    """-> (row_entry, source_text, columns, abi_values) or raises ValueError naming the failed requirement.

    NO SCALAR CORES (the owner's rule): every column is rank 1, ``shape: null`` (the batch length arrives with the Nodus handle)
    and the output takes its length from the first input via ``shape_from``, exactly as nodus/tests/
    turing_nodus_runtime_shape_crossing_test.py:70-80 declares them.  The driver has no per-row application of a scalar function
    (tools/turing_row_tool_compiler.py lowers the source as written, :303-316 / :823); the fixture's function is elementwise over
    AbstractTensor operands, typed by a per-request extraction contract ``program_abi`` (same test, :30-43).  So only a function
    whose formals and return are annotated ``AbstractTensor`` is expressible; scalar annotations are refused."""
    names = list(piece_row.get("names") or [])
    if len(names) != 1:
        raise ValueError(f"row_entry takes exactly one function; the piece has {len(names)} ({names})")
    entry = names[0]
    if not entry.isidentifier():
        raise ValueError(f"row_entry must be an identifier, {entry!r} is a nested or qualified name")
    path = Path(source_root) / str(piece_row["job_id"]) / "source" / str(piece_row["file"])
    try:
        text = path.read_bytes().decode("utf-8-sig")
        tree = ast.parse(text)
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise ValueError(f"source {path} is not readable as Python: {type(error).__name__}") from error
    found = sd._function_nodes(tree).get(entry)
    if found is None:
        raise ValueError(f"{entry!r} is not a function in {piece_row['file']}")
    node, in_class = found
    a = node.args
    if isinstance(node, ast.AsyncFunctionDef) or in_class:
        raise ValueError("row_entry must be a plain module-level function (not async, not a method)")
    if a.vararg or a.kwarg or a.kwonlyargs:
        raise ValueError("the column ABI is positional: *args, **kwargs and keyword-only formals cannot be expressed")
    formals = list(a.posonlyargs) + list(a.args)
    if not formals:
        raise ValueError("a batched row needs at least one AbstractTensor input to supply the batch length")
    for arg in formals:
        kind = _annotation_name(arg.annotation) if arg.annotation is not None else None
        if kind != "AbstractTensor":
            raise ValueError(f"formal {arg.arg!r} is annotated {kind!r}: a scalar function cannot be requested because it would be a "
                             "rank-0 (scalar core) request, which is never produced, and the driver has no per-row application of a "
                             "scalar function (tools/turing_row_tool_compiler.py:303-316 lowers the source as written; the batched "
                             "fixture tests/turing_nodus_runtime_shape_crossing_test.py:26-43 uses AbstractTensor operands under a "
                             "program_abi contract); annotate every formal AbstractTensor")
    if (_annotation_name(node.returns) if node.returns is not None else None) != "AbstractTensor":
        raise ValueError("the return annotation must be AbstractTensor (one output column of the batch length)")
    columns = [{"name": f.arg, "role": "input", "dtype": "f64", "rank": 1, "shape": None} for f in formals]
    taken = {c["name"] for c in columns}
    out_name = "result"
    while out_name in taken:
        out_name += "_"
    columns.append({"name": out_name, "role": "output", "dtype": "f64", "rank": 1, "shape": None, "shape_from": columns[0]["name"]})
    abi = [{"function": entry, "parameter": c["name"], "storage": "span", "dtype": "float64", "rank": 1,
            "python_type": TENSOR_TYPE} for c in columns if c["role"] == "input"]
    return entry, text, columns, abi


def _write_contract(base: Path, path: Path, abi_values: list) -> None:
    """Per-request extraction contract: the explicit base contract with ``program_abi`` replaced, as the batched fixture does."""
    import yaml
    value = yaml.safe_load(Path(base).read_text(encoding="utf-8"))
    value["program_abi"] = {"records": {}, "bindings": [], "values": abi_values}
    _write_atomic(path, yaml.safe_dump(value))


def request_for_piece(db, piece_row: dict, *, jobs_dir, source_root=None, driver_backend=None, extraction_contract=None,
                      turing_root=None):
    """Write the driver request for one CLAIMED piece -> (contract_id, request_path), or record a refusal and return None.

    ``piece_row`` is the claim_next dict (job_id, piece_id, backend, generation, worker_id, names, file).  The request has the keys
    the driver reads: row_entry, columns (ordered input.../output), backend, output_dir (= ``<jobs_dir>/<id>``), extraction_contract
    (explicit path), turing_root, product_name, source_text, allow_uncompiled=false.  Refused (never submitted) when:
      - the claimed backend (or ``driver_backend``) is not c|llvm|fortran;
      - the piece has other than one function name, or the name is nested/qualified, async, or a method;
      - a formal is not annotated AbstractTensor (scalar annotations would be rank-0 scalar cores: never produced), or is
        *args/**kwargs/keyword-only; the return annotation is not AbstractTensor; there is no formal;
      - the source file is unreadable or not parseable."""
    jobs_dir = Path(jobs_dir)
    backend = driver_backend or piece_row["backend"]
    reason = None
    if backend not in DRIVER_BACKENDS:
        reason, error_type = f"the Nodus driver compiles {DRIVER_BACKENDS}; the attempt backend is {backend!r}", "BackendNotServed"
    else:
        try:
            entry, text, columns, abi = _express(piece_row, Path(source_root) if source_root else jobs_dir)
        except ValueError as error:
            reason, error_type = str(error), "RequestNotExpressible"
    if reason is not None:
        sd.refuse(db, piece_row["job_id"], piece_row["piece_id"], piece_row["backend"], piece_row["worker_id"],
                  piece_row["generation"], state="refused", stage=f"{BACKEND_NAME}: request", error_type=error_type, error=reason)
        return None
    contract_id = allocate_contract_id(_identity(piece_row), jobs_dir)
    request = {"row_entry": entry, "columns": columns, "backend": backend, "output_dir": str(jobs_dir.resolve() / str(contract_id)),
               "extraction_contract": str(Path(extraction_contract or sd.EXTRACTION_CONTRACT).resolve()),
               "turing_root": str(Path(turing_root or ROOT).resolve()), "product_name": f"div_{contract_id}",
               "source_text": text, "allow_uncompiled": False}
    contract_path = jobs_dir / f"{contract_id}.extraction-contract.yaml"
    _write_contract(Path(extraction_contract or sd.EXTRACTION_CONTRACT), contract_path, abi)
    request["extraction_contract"] = str(contract_path.resolve())
    path = jobs_dir / f"{contract_id}.request.json"
    _write_atomic(path, json.dumps(request, indent=1))
    return contract_id, path


# ------------------------------------------------------------------------------------------------------------------- submit
class _Producer:
    """EnginePeer producer: owns exactly one contract; commits only its own Route."""

    def __init__(self, contract):
        self.contract, self.admitted_seen, self.result = contract, False, None

    def request(self):
        return None

    def admitted(self, identifier):
        if identifier == self.contract[0]:
            self.admitted_seen = True

    def route(self, result):
        if result[0] != self.contract[0]:
            return 0                                   # not ours: hold, never commit a foreign result
        self.result = tuple(result)
        return 1


def _default_peer_factory(route, producer, module):
    if route.host not in ("127.0.0.1", "localhost"):
        raise ValueError("the contract port is loopback only (connect_loopback); host must be 127.0.0.1")
    return module.EnginePeer(module.connect_loopback(route.port, module.ROLE_ENGINE, route.service_id), producer)


def _wait_readable(peer, seconds):
    conn = getattr(peer, "connection", None)
    try:
        if conn is not None:
            select.select([conn], [], [], seconds)
            return
    except (OSError, ValueError):
        pass
    time.sleep(seconds)


def submit_and_wait(route: NodusCompileRoute, contract_id: int, *, timeout_s: float, poll_s: float = 0.2,
                    peer_factory=None) -> CompileOutcome:
    """Submit ONE contract (id, service_id, revision 1, payload 0) and wait for its Route (the Result).  Sleeps in select, never spins.

    ``peer_factory(route, producer) -> peer`` (default: nodus EnginePeer over connect_loopback) is the test seam.  A timeout,
    a failed connection or a closed peer is reported as such (status_code None), never as a compile failure."""
    result_dir = Path(route.jobs_dir) / str(contract_id)
    result_path = result_dir / "compile-host.result.json"
    contract = (int(contract_id), int(route.service_id), 1, 0)
    producer = _Producer(contract)
    started = time.monotonic()
    try:
        module = load_peer_module(route.nodus_tools)
        peer = (peer_factory or (lambda r, p: _default_peer_factory(r, p, module)))(route, producer)
    except (OSError, ValueError, FileNotFoundError) as error:
        return CompileOutcome(None, "ConnectFailed", False, None, {"error": f"{type(error).__name__}: {error}"})
    try:
        if not peer.submit(contract):
            return CompileOutcome(None, "SubmitRefused", False, None, {"error": "peer did not queue the submit"})
        deadline = started + float(timeout_s)
        while producer.result is None:
            peer.poll()
            if getattr(peer, "error", None) is not None or getattr(peer, "closed", False):
                return CompileOutcome(None, "PeerClosed", False, None,
                                      {"error": repr(getattr(peer, "error", None)), "admitted": producer.admitted_seen})
            if producer.result is not None:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return CompileOutcome(None, "Timeout", False, None, {"admitted": producer.admitted_seen,
                                                                        "waited_s": round(time.monotonic() - started, 3)})
            _wait_readable(peer, min(poll_s, remaining))
        for _ in range(20):                                    # flush the Decision(commit) frame
            peer.poll()
            if not getattr(peer, "outgoing", b""):
                break
            time.sleep(0.01)
    finally:
        peer.close()
    code = int(producer.result[7])
    detail = {"admitted": producer.admitted_seen, "generation": producer.result[4], "waited_s": round(time.monotonic() - started, 3)}
    if result_path.is_file():
        try:
            detail["host_result"] = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            detail["host_result_error"] = f"{type(error).__name__}: {error}"
    return CompileOutcome(code, STATUS_NAMES.get(code, f"Unknown({code})"), True, result_path if result_path.is_file() else None, detail)


# ---------------------------------------------------------------------------------------------------------------- recording
def _record(db, claim, outcome: CompileOutcome, seconds):
    """Map a delivered status onto source_divider.complete / refuse.  Returns the recorded state, or None (left leased)."""
    key = (claim["job_id"], claim["piece_id"], claim["backend"], claim["worker_id"], claim["generation"])
    host = outcome.detail.get("host_result") or {}
    wall = host.get("wall_seconds", seconds)
    note = f"{BACKEND_NAME}: {outcome.status_name}"
    if outcome.status_code in (1, 2, 3):
        # Adopted / CompiledNotAdopted / ExistingProduct: a retained product exists under <jobs_dir>/<id>/
        return "complete" if sd.complete(db, *key, stage=note, seconds=wall) else "lost-lease"
    if outcome.status_code == 16:
        state, error_type = "refused", "RequestMissing"
    elif outcome.status_code == 17:
        state, error_type = "error", "CompileFailed"
    elif outcome.status_code == 18:
        state, error_type = "error", "AdoptFailed"
    else:
        return None                                            # 19 Cancelled or an unknown code is not a verdict
    error = host.get("error") or outcome.status_name
    ok = sd.refuse(db, *key, state=state, stage=note, error_type=error_type, error=str(error)[:300], seconds=wall)
    return state if ok else "lost-lease"


def _attempt_row(db, claim):
    return db.execute("SELECT state, stage, error_type, error FROM divider_attempts WHERE job_id=? AND piece_id=? AND backend=?",
                      (claim["job_id"], claim["piece_id"], claim["backend"])).fetchone()


def run_one(db, route: NodusCompileRoute, *, lease_owner: str, timeout_s: float = DEFAULT_TIMEOUT_S, poll_s: float = 0.2,
            peer_factory=None) -> dict | None:
    """Claim exactly one attempt, request, submit, wait, record.  None when nothing is queued.

    Returns {"state": "busy"} without claiming when ``lease_owner`` still holds an unexpired lease (single in flight).  Otherwise a
    dict: job_id, piece_id, backend, generation, contract_id, status, recorded (the attempt state written, or None when the lease
    was deliberately left in place), detail."""
    sd.reclaim_expired(db)
    live = db.execute("SELECT job_id, piece_id, backend FROM divider_attempts WHERE state='leased' AND lease_owner=? AND "
                      "lease_deadline > ?", (lease_owner, time.time())).fetchone()
    if live is not None:
        return {"state": "busy", "job_id": live["job_id"], "piece_id": live["piece_id"], "backend": live["backend"]}
    claim = sd.claim_next(db, lease_owner, float(timeout_s) + 30.0)
    if claim is None:
        return None
    base = {"job_id": claim["job_id"], "piece_id": claim["piece_id"], "backend": claim["backend"], "generation": claim["generation"]}
    built = request_for_piece(db, claim, jobs_dir=route.jobs_dir, source_root=route.source_root, driver_backend=route.backend,
                              extraction_contract=route.extraction_contract, turing_root=route.turing_root)
    if built is None:
        row = _attempt_row(db, claim)
        return {**base, "state": "refused", "contract_id": None, "status": None, "recorded": row["state"],
                "detail": {"error_type": row["error_type"], "error": row["error"]}}
    contract_id, request_path = built
    started = time.monotonic()
    outcome = submit_and_wait(route, contract_id, timeout_s=timeout_s, poll_s=poll_s, peer_factory=peer_factory)
    recorded = _record(db, claim, outcome, time.monotonic() - started) if outcome.status_code is not None else None
    return {**base, "state": recorded or "unfinished", "contract_id": contract_id, "request": str(request_path),
            "status": outcome.status_name, "status_code": outcome.status_code, "recorded": recorded,
            "result_path": str(outcome.result_path) if outcome.result_path else None, "detail": outcome.detail}


def run_pending(db, paths, route: NodusCompileRoute, *, max_pieces: int = 1, lease_owner: str = "divider-nodus-bridge",
                timeout_s: float = DEFAULT_TIMEOUT_S, **kwargs) -> list:
    """Up to ``max_pieces`` run_one calls, strictly in sequence; stops at an empty queue or at the first piece left unfinished.

    ``paths`` is the divider's StatePaths (or its root); it supplies ``source_root`` when the route has none of its own."""
    state = StatePaths.coerce(paths)
    if route.source_root == route.jobs_dir and state.jobs_dir != route.jobs_dir:
        route = NodusCompileRoute(route.host, route.port, route.service_id, route.jobs_dir, route.nodus_tools,
                                  source_root=state.jobs_dir, backend=route.backend, extraction_contract=route.extraction_contract,
                                  turing_root=route.turing_root)
    results = []
    for _ in range(int(max_pieces)):
        result = run_one(db, route, lease_owner=lease_owner, timeout_s=timeout_s, **kwargs)
        if result is None:
            break
        results.append(result)
        if result.get("recorded") is None and result.get("state") in ("busy", "unfinished"):
            break
    return results


def make_hook(paths, route: NodusCompileRoute, *, max_pieces: int = 1, **kwargs):
    """A callable(job_id) for ``sessions.divider_hook`` (server.py calls ``hook(job_id)`` after an accepted upload).  It opens its own
    accounts.db connection per call; the queue is global, so job_id is not used.  It BLOCKS the calling request thread for up to the
    timeout: install it only where that is acceptable."""
    state = StatePaths.coerce(paths)

    def hook(job_id=None):
        db = sd.connect(state.accounts_db)
        try:
            return run_pending(db, state, route, max_pieces=max_pieces, **kwargs)
        finally:
            db.close()
    return hook


# ------------------------------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run-once", help="claim one queued piece, compile it through the Nodus host, record the attempt")
    p.add_argument("--db", required=True, help="accounts db (source_divider tables)")
    p.add_argument("--jobs-dir", required=True, help="the compile host's jobs dir (<id>.request.json lives here)")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--service-id", type=int, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    p.add_argument("--nodus-tools", default=None)
    p.add_argument("--source-root", default=None, help="the divider's jobs dir holding <job_id>/source (default: --jobs-dir)")
    p.add_argument("--backend", choices=DRIVER_BACKENDS, default=None)
    p.add_argument("--lease-owner", default="divider-nodus-bridge")
    args = parser.parse_args(argv)
    route = NodusCompileRoute(args.host, args.port, args.service_id, Path(args.jobs_dir),
                              Path(args.nodus_tools) if args.nodus_tools else DEFAULT_NODUS_TOOLS,
                              source_root=Path(args.source_root) if args.source_root else None, backend=args.backend)
    db = sd.connect(args.db)
    try:
        result = run_one(db, route, lease_owner=args.lease_owner, timeout_s=args.timeout)
    finally:
        db.close()
    print(json.dumps(result if result is not None else {"state": "idle"}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded Nodus compute preparation/commit, retaining the existing photo API.

Trusted host operations select builders and commits; clients supply only JSON arguments.
Studio scene products and ray-photo admission use the existing StudioAgentProducts,
studio_product_nodus contract transport and PhotoJobs ownership checks. No new renderer
or parallel scheduler is introduced. Authority is rechecked before building and committing.
One contract runs at a time; an uncertain commit is never retried automatically.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import json
from pathlib import Path
import re
import secrets
import select
import threading
import time

from meowdoku import studio_agent, studio_product_nodus as adapter

SCHEMA = "turing.studio-agent-request.v1"
RID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_QUEUED = 4
MAX_RECEIPTS = 64
REQUEST_TIMEOUT_S = 240.0
STALE_PAYLOAD_S = 1800.0
TERMINAL = ("committed", "refused", "failed", "uncertain")


class PictureError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ComputeOperation:
    """Trusted host declaration, never a client-supplied executable or destination.

    prepare(arguments) validates and forms the exact request sent over Nodus.
    build(request) prepares its product. commit(entry, product) owns side effects
    and returns the receipt. The queue rechecks authority before both stages.
    """
    capability: str
    prepare: object
    build: object
    commit: object

    def __post_init__(self):
        if not self.capability or not all(callable(f) for f in (self.prepare, self.build, self.commit)):
            raise ValueError("compute operation needs a capability and trusted callbacks")


class _Entry:
    def __init__(self, account_id, owners, authorized, request, request_id, digest):
        self.id = secrets.token_hex(16)
        self.account_id, self.owners, self.authorized = account_id, tuple(owners), authorized
        self.request, self.request_id, self.digest = request, request_id, digest
        self.state, self.detail, self.job = "queued", None, None
        self.operation, self.compute, self.result = "studio.ray-photo", False, None
        self.created = time.time()
        self.finished = None

    def end(self, state, detail=None, job=None):
        self.state, self.detail, self.job, self.finished = state, detail, job, time.time()

    def receipt(self):
        out = {"schema": "speaktome.picture.v1", "picture_id": self.id, "request_id": self.request_id, "state": self.state,
               "created": self.created, "finished": self.finished}
        if self.compute:
            out.update(schema="speaktome.compute.v1", compute_id=self.id, operation=self.operation)
            out.pop("picture_id", None)
            if self.result is not None:
                out["result"] = self.result
        if self.detail:
            out["detail"] = self.detail
        if self.job is not None:
            out["job"] = self.job
            if isinstance(self.job, dict) and self.job.get("digest"):
                out["status_url"] = "/api/marble_photo?digest=" + self.job["digest"]
        return out


class NodusComputeRequests:
    """Existing bounded Nodus queue, now dispatching declared compute operations.

    PhotoJobs remains the photo-specific GPU admission/lifecycle owner.
    """

    def __init__(self, *, port, prefix, products, store_dir, photo_jobs, saved_objects_for, claim,
                 host_id=7101, engine_id=7102, nodus_tools=adapter.DEFAULT_NODUS_TOOLS, connect=None, log=print, operations=None, registry=None):
        if not isinstance(port, int) or not 0 < port < 65536 or not isinstance(prefix, str) or not prefix:
            raise ValueError("a Studio Nodus port and prefix are required")
        self.port, self.prefix, self.products = port, prefix, products
        self.registry = registry
        declared = {
            "studio.ray-photo": ComputeOperation("photo_submit", lambda args: self._studio_request(args, "ray-payload"), products.build, self._photo_commit),
            "studio.scene": ComputeOperation("photo_submit", lambda args: self._studio_request(args, "scene"), products.build, self._product_commit),
        }
        for name, operation in (operations or {}).items():
            if name in declared or not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", name) or not isinstance(operation, ComputeOperation):
                raise ValueError("invalid or duplicate trusted compute operation")
            declared[name] = operation
        self.operations = MappingProxyType(declared)
        self.photo_jobs, self.saved_objects_for, self.claim = photo_jobs, saved_objects_for, claim
        self.host_id, self.engine_id, self.log = int(host_id), int(engine_id), log
        self.native, payload = adapter.nodus_modules(nodus_tools)
        store_dir = Path(store_dir)
        self.product_directory = store_dir / "products"
        self.requests = payload.PayloadStore(store_dir / "requests")
        self.results = payload.PayloadStore(store_dir / "results", max_bytes=adapter.SCENE_PRODUCT_MAX_BYTES)
        self._connect = connect or (lambda role, identity: self.native.connect_loopback(self.port, role, identity))
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.entries: dict = {}         # picture id -> _Entry (the newest MAX_RECEIPTS)
        self.by_request: dict = {}      # (account id, request id) -> picture id
        self.queue: list = []
        self.current = None             # the entry in flight; read by the host peer's dispatch thread
        self.dispatcher = None
        self.receipts: list = []        # loans whose result reached the FIFO
        self.closed = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="studio-product")
        self.thread = threading.Thread(target=self._run, name="studio-pictures", daemon=True)
        self.thread.start()

    # ---- the surface the REST handler uses ----------------------------------------------------------------------
    def submit(self, account_id, owners, authorized, body) -> dict:
        """Queue one picture for an authenticated account. ``authorized()`` is rechecked before any side effect."""
        if not isinstance(body, dict) or set(body) - {"request_id", "settings", "photo"}:
            raise PictureError("expected request_id and optional settings, photo")
        request = self._studio_request({key: body[key] for key in ("settings", "photo") if key in body}, "ray-payload")
        return self._enqueue(account_id, owners, authorized, body.get("request_id"), request, "studio.ray-photo", False)

    def _studio_request(self, arguments, operation):
        if not isinstance(arguments, dict) or set(arguments) - {"settings", "photo"}:
            raise PictureError("Studio arguments accept settings and photo only")
        if operation == "scene" and "photo" in arguments:
            raise PictureError("scene preparation does not accept photo options")
        if any(not isinstance(value, dict) for value in arguments.values()):
            raise PictureError("settings and photo must be objects")
        if "settings" not in arguments and self.products.defaults is None:
            raise PictureError("settings are required: this host has no baked defaults")
        return {"schema": SCHEMA, "operation": operation, **arguments}

    def capability(self, operation):
        declaration = self.operations.get(operation) if isinstance(operation, str) else None
        if declaration is None:
            raise PictureError("compute operation is not configured on this host", 400)
        return declaration.capability

    def submit_compute(self, account_id, owners, authorized, body):
        if not isinstance(body, dict) or set(body) != {"request_id", "operation", "arguments"}:
            raise PictureError("expected request_id, operation and arguments")
        operation = body["operation"]
        self.capability(operation)
        request = self.operations[operation].prepare(body["arguments"])
        return self._enqueue(account_id, owners, authorized, body["request_id"], request, operation, True)

    def _enqueue(self, account_id, owners, authorized, request_id, request, operation, compute):
        if not isinstance(request_id, str) or not RID.fullmatch(request_id):
            raise PictureError("request_id must contain 1-64 letters, digits, underscores or hyphens")
        try:
            digest = hashlib.sha256(json.dumps({"operation": operation, "request": request}, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()
        except (ValueError, TypeError, OverflowError) as error:
            raise PictureError("the request must be finite JSON data") from error
        # Freeze the admitted request so later caller mutation cannot alter its digest.
        request = json.loads(json.dumps(request, allow_nan=False))
        if type(account_id) is not int or account_id <= 0:
            raise PictureError("an authenticated account is required", 401)
        if not callable(authorized) or not owners:
            raise PictureError("an authenticated owner is required", 401)
        with self.lock:
            known = self.entries.get(self.by_request.get((account_id, request_id)))
            if known is not None:
                if known.digest != digest or known.compute != compute:
                    raise PictureError("request_id already used with different content", 409)
                return dict(known.receipt(), replayed=True)
            if len(self.queue) + (self.current is not None) >= MAX_QUEUED:
                raise PictureError("picture queue is full; try again when one finishes", 429)
            entry = _Entry(account_id, owners, authorized, request, request_id, digest)
            entry.operation, entry.compute = operation, compute
            self.entries[entry.id] = entry
            self.by_request[(account_id, request_id)] = entry.id
            self.queue.append(entry)
            for stale in list(self.entries)[:-MAX_RECEIPTS]:
                old = self.entries[stale]
                if old.state in TERMINAL:
                    self.by_request.pop((old.account_id, old.request_id), None)
                    (self.product_directory / str(old.account_id) / (old.id + ".json")).unlink(missing_ok=True)
                    del self.entries[stale]
            receipt = entry.receipt()
        self.wake.set()
        return receipt

    def status(self, account_id, picture_id) -> dict:
        with self.lock:
            entry = self.entries.get(picture_id)
            if entry is None or entry.account_id != account_id:
                raise PictureError("picture not found", 404)
            return entry.receipt()

    def product(self, account_id, compute_id):
        receipt = self.status(account_id, compute_id)
        if receipt.get("state") != "committed" or not receipt.get("result", {}).get("product_url"):
            raise PictureError("compute product is not available", 404)
        path = self.product_directory / str(account_id) / (compute_id + ".json")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise PictureError("compute product has expired", 404) from error

    def _product_commit(self, entry, product):
        # Reuse Studio's exact bounded blob envelope; no ray job is implied.
        studio_agent.product_bytes(product)
        directory = self.product_directory / str(entry.account_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (entry.id + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(product, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
        return {"product_url": "/api/compute/requests/" + entry.id + "/product",
                "sha256": product["sha256"], "content_type": product["content_type"]}

    def _photo_commit(self, entry, product):
        job = studio_agent.commit_studio_ray(product, authorize=entry.authorized, photo_jobs=self.photo_jobs,
                                            saved_objects=self.saved_objects_for(entry.account_id), owners=entry.owners)
        self.claim(entry.account_id, job["digest"])
        return dict(job, status_url="/api/marble_photo?digest=" + job["digest"])

    def close(self):
        self.closed = True
        self.wake.set()
        self.thread.join(timeout=5)
        self.executor.shutdown(wait=False, cancel_futures=True)

    # ---- host side: pure preparation --------------------------------------------------------------------------
    def _authorize(self, loan, request):
        entry = self.current
        if entry is None or loan.contract != entry.contract[0] or entry.authorized() is not True:
            raise PermissionError("Studio product request is not authorized")
        return {"picture": entry.id, "operation": entry.operation}

    def _build(self, request, context):
        """Never raises for a failed preparation: the failure travels back as the product and ends the contract."""
        try:
            return self.operations[context["operation"]].build(request)
        except studio_agent.StudioAgentError as error:
            return {"error": str(error)[:2048]}

    def _dispatch(self, loan):
        return self.dispatcher.dispatch(loan)

    def _on_receipt(self, loan, accepted):
        self.dispatcher.on_receipt(loan, accepted)
        self.receipts.append((loan, accepted))

    # ---- engine side: the one place a job is born ---------------------------------------------------------------
    def _commit(self, entry, product):
        if isinstance(product, dict) and set(product) == {"error"}:
            entry.end("failed", f"scene preparation failed: {product['error']}; no photo job was created")
            return 1
        if entry.authorized() is not True:
            entry.end("refused", "authority was withdrawn before commit")
            return 1
        try:
            result = self.operations[entry.operation].commit(entry, product)
        except PermissionError:
            entry.end("refused", "authority was withdrawn before commit")
            return 1
        if entry.compute:
            entry.result = result
            entry.end("committed")
        else:
            entry.end("committed", job=result)
        return 1

    # ---- the thread ---------------------------------------------------------------------------------------------
    def _prune(self):
        cutoff = time.time() - STALE_PAYLOAD_S
        # These are disposable compute results, not durable photo artifacts.
        # Old-process receipts are gone after restart; retain no orphan forever.
        for path in self.product_directory.glob("*/*.json"):
            try:
                if path.stat().st_mtime < cutoff and path.stem not in self.entries:
                    path.unlink()
            except OSError:
                pass
        for store in (self.requests, self.results):
            for path in store.directory.glob("*.json"):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                except OSError:
                    pass

    def _host(self, host):
        """A live host peer, reconnecting when the last one closed or stopped on an error."""
        if host is not None and (host.closed or host.error is not None):
            self.log(f"[studio] host peer ended ({host.error!r}); reconnecting")
            host.close()
            host = None
        if host is None:
            connection = self._connect(self.native.ROLE_HOST, self.host_id)
            host = self.native.ServicePeer(connection, self.executor, self._dispatch, self._on_receipt, announce=False)
        return host

    def _run(self):
        host, retry_at = None, 0.0
        while not self.closed:
            try:
                if time.monotonic() >= retry_at:
                    host = self._host(host)
            except OSError as error:
                host, retry_at = None, time.monotonic() + 5.0
                self.log(f"[studio] Nodus port {self.port} is not reachable: {error}")
            with self.lock:
                entry = self.queue.pop(0) if self.queue and host is not None else None
                self.current = entry
            if entry is None:
                if host is not None:
                    host.poll()
                self.wake.wait(0.25)
                self.wake.clear()
                continue
            try:
                if self.registry is not None:
                    self.registry.begin("nodus_compute", entry.id, owner="account:" + str(entry.account_id), label=entry.operation)
                self._one(entry, host)
            except Exception as error:   # the thread must outlive any single request
                if entry.state not in TERMINAL:
                    entry.end("uncertain", f"{type(error).__name__}: {error}")
                self.log(f"[studio] picture {entry.id} ended with {type(error).__name__}: {error}")
            finally:
                if self.registry is not None:
                    self.registry.end("nodus_compute", entry.id)
                with self.lock:
                    self.current = None

    def _one(self, entry, host):
        self._prune()
        if entry.authorized() is not True:
            entry.end("refused", "authority was withdrawn before preparation; no photo job was created")
            return
        contract_id = secrets.randbits(62) | 1
        entry.contract = adapter.retain_request(self.requests, prefix=self.prefix, contract_id=contract_id,
                                                engine_id=self.engine_id, revision=1, request=entry.request)
        self.receipts = []
        self.dispatcher = adapter.StudioProductDispatcher(prefix=self.prefix, requests=self.requests, results=self.results,
                                                          authorize=self._authorize, build_scene=self._build)
        producer = adapter.StudioProductProducer(entry.contract, prefix=self.prefix, results=self.results,
                                                 commit=lambda product: self._commit(entry, product))
        entry.state = "preparing"
        engine = self.native.EnginePeer(self._connect(self.native.ROLE_ENGINE, self.engine_id), producer)
        deadline = time.monotonic() + REQUEST_TIMEOUT_S
        try:
            if not engine.submit(entry.contract):
                entry.end("failed", "the Nodus engine peer did not queue the request; no photo job was created")
                return
            while entry.state not in TERMINAL and not producer.uncertain:
                host.poll()
                engine.poll()
                if host.error is not None or host.closed:
                    entry.end("failed", f"the Studio host peer stopped ({host.error!r}); no photo job was created")
                    break
                if engine.error is not None or engine.closed:
                    entry.end("failed", f"the Nodus engine peer stopped ({engine.error!r}); no photo job was created")
                    break
                if time.monotonic() >= deadline or self.closed:
                    entry.end("failed", "no result arrived in time; no photo job was created")
                    break
                select.select([host.connection, engine.connection], [], [], 0.05)
            if producer.uncertain and entry.state not in TERMINAL:
                entry.end("uncertain", f"the commit raised {type(producer.error).__name__}: {producer.error}; "
                                       "a photo job may or may not exist and nothing was retried")
            settle = time.monotonic() + 3.0     # let the Decision leave and the FIFO receipt arrive
            while time.monotonic() < settle and not (self.receipts and not engine.outgoing):
                host.poll()
                engine.poll()
                if host.error is not None or host.closed or engine.error is not None or engine.closed:
                    break
                time.sleep(0.01)
        finally:
            engine.close()
            for loan, _accepted in self.receipts:
                try:
                    self.dispatcher.release_result(loan)
                except (KeyError, OSError):
                    pass
            try:
                self.requests.release(entry.contract[3])
            except OSError:
                pass


# Existing hosts and picture callers retain their API and the same queue.
StudioPictures = NodusComputeRequests

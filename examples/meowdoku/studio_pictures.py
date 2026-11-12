"""Studio pictures: settings -> the existing Nodus contract port -> an owned photo job.

This is the host integration docs/STUDIO_AGENT_PRODUCT.md left open. It adds no renderer, page or job queue:

* the request is a ``turing.studio-agent-request.v1`` with operation ``ray-payload``;
* ``studio_product_nodus`` carries it: this process is the HOST peer of a dedicated prefix (its dispatch is the pure
  ``StudioAgentProducts.build``) and, for each request, the ENGINE peer whose producer commit births the job;
* the commit is ``studio_agent.commit_studio_ray`` on the existing ``PhotoJobs``, with the account's own saved
  objects and owner keys, after authority is rechecked; then the existing account claim.

Identity and authority never come from the request: the caller passes the authenticated account, its owner keys and
a recheck callback. One request is in flight at a time (there is one GPU and one engine identity). A commit that
raised is recorded as uncertain and is never retried. The client work-sharing bridge options of the HTTP photo path
are not carried: a payload that asks for client intersections is refused by ``PhotoJobs.start`` itself.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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


class _Entry:
    def __init__(self, account_id, owners, authorized, request, request_id, digest):
        self.id = secrets.token_hex(16)
        self.account_id, self.owners, self.authorized = account_id, tuple(owners), authorized
        self.request, self.request_id, self.digest = request, request_id, digest
        self.state, self.detail, self.job = "queued", None, None
        self.created = time.time()
        self.finished = None

    def end(self, state, detail=None, job=None):
        self.state, self.detail, self.job, self.finished = state, detail, job, time.time()

    def receipt(self):
        out = {"schema": "speaktome.picture.v1", "picture_id": self.id, "request_id": self.request_id, "state": self.state,
               "created": self.created, "finished": self.finished}
        if self.detail:
            out["detail"] = self.detail
        if self.job is not None:
            out["job"] = self.job
            if isinstance(self.job, dict) and self.job.get("digest"):
                out["status_url"] = "/api/marble_photo?digest=" + self.job["digest"]
        return out


class StudioPictures:
    """One background thread owns both peers of the Studio prefix; ``submit`` and ``status`` are the whole surface."""

    def __init__(self, *, port, prefix, products, store_dir, photo_jobs, saved_objects_for, claim,
                 host_id=7101, engine_id=7102, nodus_tools=adapter.DEFAULT_NODUS_TOOLS, connect=None, log=print):
        if not isinstance(port, int) or not 0 < port < 65536 or not isinstance(prefix, str) or not prefix:
            raise ValueError("a Studio Nodus port and prefix are required")
        self.port, self.prefix, self.products = port, prefix, products
        self.photo_jobs, self.saved_objects_for, self.claim = photo_jobs, saved_objects_for, claim
        self.host_id, self.engine_id, self.log = int(host_id), int(engine_id), log
        self.native, payload = adapter.nodus_modules(nodus_tools)
        store_dir = Path(store_dir)
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
        request_id = body.get("request_id")
        if not isinstance(request_id, str) or not RID.fullmatch(request_id):
            raise PictureError("request_id must contain 1-64 letters, digits, underscores or hyphens")
        for key in ("settings", "photo"):
            if key in body and not isinstance(body[key], dict):
                raise PictureError(f"{key} must be an object")
        if "settings" not in body and self.products.defaults is None:
            raise PictureError("settings are required: this host has no baked default settings")
        request = {"schema": SCHEMA, "operation": "ray-payload",
                   **{key: body[key] for key in ("settings", "photo") if key in body}}
        try:
            digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()
        except (ValueError, TypeError, OverflowError) as error:
            raise PictureError("the request must be finite JSON data") from error
        if not callable(authorized) or not owners:
            raise PictureError("an authenticated owner is required", 401)
        with self.lock:
            known = self.entries.get(self.by_request.get((account_id, request_id)))
            if known is not None:
                if known.digest != digest:
                    raise PictureError("request_id already used with different content", 409)
                return dict(known.receipt(), replayed=True)
            if len(self.queue) + (self.current is not None) >= MAX_QUEUED:
                raise PictureError("picture queue is full; try again when one finishes", 429)
            entry = _Entry(account_id, owners, authorized, request, request_id, digest)
            self.entries[entry.id] = entry
            self.by_request[(account_id, request_id)] = entry.id
            self.queue.append(entry)
            for stale in list(self.entries)[:-MAX_RECEIPTS]:
                old = self.entries[stale]
                if old.state in TERMINAL:
                    self.by_request.pop((old.account_id, old.request_id), None)
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
        return {"picture": entry.id}

    def _build(self, request, context):
        """Never raises for a failed preparation: the failure travels back as the product and ends the contract."""
        try:
            return self.products.build(request)
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
        try:
            job = studio_agent.commit_studio_ray(product, authorize=entry.authorized, photo_jobs=self.photo_jobs,
                                                 saved_objects=self.saved_objects_for(entry.account_id), owners=entry.owners)
        except PermissionError:
            entry.end("refused", "authority was withdrawn before the job was created; no photo job was created")
            return 1
        self.claim(entry.account_id, job["digest"])
        entry.end("committed", job=job)
        return 1

    # ---- the thread ---------------------------------------------------------------------------------------------
    def _prune(self):
        cutoff = time.time() - STALE_PAYLOAD_S
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
                self._one(entry, host)
            except Exception as error:   # the thread must outlive any single request
                if entry.state not in TERMINAL:
                    entry.end("uncertain", f"{type(error).__name__}: {error}")
                self.log(f"[studio] picture {entry.id} ended with {type(error).__name__}: {error}")
            finally:
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

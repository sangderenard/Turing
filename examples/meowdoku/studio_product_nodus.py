"""Small Studio blob adapter for the existing Nodus contract peers.

The owner selects an already-authorized prefix/connection. ``engine`` is the
producer identity, not a renderer address or permission. This module creates no
socket, listener, queue, scheduler, account or credential. Use Nodus EnginePeer
and ServicePeer unchanged; the caller owns their polling and executor.

The host's build_scene callback prepares a product. It may return a packed
scene, a bounded explicitly configured captured-scene PNG, or a JSON-compatible
blob descriptor, never create a ray job or durable artifact. Job birth
belongs to the producer's authorized commit callback, after native validation.
No credentials belong in either payload. FIFO receipt is not producer commit:
the owner explicitly releases request/result payloads after reconciliation.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_MODULES = {}
DEFAULT_NODUS_TOOLS = Path(__file__).resolve().parents[3] / "nodus" / "tools"
# A measured full Studio scene is 28,460,700 raw JSON bytes, about 38 MB when
# base64-wrapped. The existing store's 32 MiB default cannot retain that product.
# Explicitly configure result PayloadStore(max_bytes=SCENE_PRODUCT_MAX_BYTES).
# This also accommodates the scene builder's 32 MiB raw limit plus base64/JSON.
SCENE_PRODUCT_MAX_BYTES = 48 * 1024 * 1024


def nodus_modules(nodus_tools=DEFAULT_NODUS_TOOLS):
    """Return the actual (contract_service_peer, contract_payload_store) modules."""
    directory = Path(nodus_tools).resolve()
    if directory not in _MODULES:
        loaded = []
        for stem in ("contract_service_peer", "contract_payload_store"):
            path = directory / (stem + ".py")
            if not path.is_file():
                raise FileNotFoundError(path)
            name = f"_studio_nodus_{len(_MODULES)}_{stem}"
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded.append(module)
        _MODULES[directory] = tuple(loaded)
    return _MODULES[directory]


def _values(values, count):
    values = tuple(values)
    if len(values) != count or any(type(v) is not int or not 0 <= v < 2**64 for v in values):
        raise ValueError("Expected native uint64 contract values")
    if not values[0] or not values[1]:
        raise ValueError("Contract and producer identities must be nonzero")
    return values


def _request_identity(prefix, contract):
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("An owner-selected Nodus prefix is required")
    return dict(prefix=prefix, contract=contract[0], engine=contract[1], revision=contract[2])


def _result_identity(prefix, values):
    values = _values(values, 7)
    return {**_request_identity(prefix, values), "request": values[3],
            "generation": values[4], "host": values[5], "deadline": values[6]}


def retain_request(requests, *, prefix, contract_id, engine_id, revision, request):
    """Publish one bounded JSON blob before submitting its native contract.

    The caller owns the store and releases contract[3] only after reconciliation.
    Binary products use the application's existing JSON-compatible descriptor.
    """
    values = _values((contract_id, engine_id, revision, 0), 4)
    handle, _ = requests.publish(_request_identity(prefix, values), request)
    return (*values[:3], handle)


class StudioProductDispatcher:
    """ServicePeer callbacks: authorize, prepare a blob, retain its result.

    authorize(loan, request) is mandatory trusted host code. It returns a trusted
    context or raises PermissionError; None/False also deny. A caller-provided
    capabilities field is ordinary data, never permission. Authorization runs
    before build_scene(request, context), including for server/ray requests.
    """

    def __init__(self, *, prefix, requests, results, authorize, build_scene):
        _request_identity(prefix, (1, 1, 0))
        if not callable(authorize) or not callable(build_scene):
            raise TypeError("Trusted authorization and pure scene builder callbacks are required")
        self.prefix, self.requests, self.results = prefix, requests, results
        self.authorize, self.build_scene = authorize, build_scene
        self._retained = {}

    def dispatch(self, loan):
        identity = _result_identity(self.prefix, loan.values())
        key = tuple(loan.values())
        if key in self._retained:
            return self._retained[key][0]
        request = self.requests.read(loan.payload, _request_identity(self.prefix, key))
        context = self.authorize(loan, request)
        if context is None or context is False:
            raise PermissionError("Studio product request is not authorized")
        product = self.build_scene(request, context)
        handle, digest = self.results.publish(identity, product)
        self._retained[key] = [handle, digest, None]
        return handle

    def on_receipt(self, loan, accepted):
        """Record FIFO admission only. Do not release either payload here."""
        self._retained[tuple(loan.values())][2] = bool(accepted)

    def release_result(self, loan):
        """Owner only, after producer commit/rejection/cancel is reconciled."""
        key = tuple(loan.values())
        self.results.release(self._retained[key][0])
        del self._retained[key]


class StudioProductProducer:
    """EnginePeer producer callbacks for exactly one retained contract.

    commit(product) must authorize any side effect and return 1 (committed),
    0 (hold without a side effect), or -1 (native reject/requeue). The owner must
    reconcile/close a rejected one-shot registration; rejection is not cancel.
    Held results may be asked
    again by Nodus. A commit exception/invalid disposition is uncertain and
    stays held without retry; the owner must reconcile it outside this adapter.
    A terminal decision is cached, so repeated ROUTEs cannot create another job.
    """

    def __init__(self, contract, *, prefix, results, commit):
        self.contract = _values(contract, 4)
        _request_identity(prefix, self.contract)
        if not callable(commit):
            raise TypeError("An authorized producer commit callback is required")
        self.prefix, self.results, self.commit = prefix, results, commit
        self.admitted_seen = False
        self.result_handle = None
        self.uncertain = False
        self.error = None
        self._result = None
        self._decision = None

    def request(self):
        return None

    def admitted(self, identifier):
        if identifier == self.contract[0]:
            self.admitted_seen = True

    def route(self, result):
        result = _values(result, 8)
        if result[:4] != self.contract or (self._result is not None and result != self._result):
            return -1
        if self._decision is not None:
            return self._decision
        if self.uncertain:
            return 0
        product = self.results.read(result[7], _result_identity(self.prefix, result[:7]))
        self._result, self.result_handle = result, result[7]
        try:
            decision = self.commit(product)
            if type(decision) is not int or decision not in (-1, 0, 1):
                raise ValueError("Producer commit must return -1, 0 or 1")
        except Exception as error:
            self.error, self.uncertain = error, True
            return 0
        if decision:
            self._decision = decision
        return decision

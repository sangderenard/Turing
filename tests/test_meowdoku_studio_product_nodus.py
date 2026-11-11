"""Actual Nodus Python peers and PayloadStore; a test-only native wire endpoint.

socketpair/temp stores only. No browser, live service, C++ negotiator or renderer
is started; the test injects the existing protocol's native-side frames.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import importlib.util
import os
import socket
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku import studio_product_nodus as adapter


@pytest.fixture
def stores(tmp_path):
    peer, payload = adapter.nodus_modules()
    return peer, payload.PayloadStore(tmp_path / "requests", max_bytes=4096), payload.PayloadStore(tmp_path / "results", max_bytes=4096)


def receive(sock, peers, native, count=1):
    data = bytearray()
    deadline = time.monotonic() + 3
    while len(data) < count * native.FRAME.size and time.monotonic() < deadline:
        for peer in peers:
            peer.poll()
            assert peer.error is None
        try:
            data.extend(sock.recv(count * native.FRAME.size - len(data)))
        except BlockingIOError:
            pass
        if len(data) < count * native.FRAME.size:
            time.sleep(.001)
    assert len(data) == count * native.FRAME.size
    return [native.decode_frame(data[i:i + native.FRAME.size]) for i in range(0, len(data), native.FRAME.size)]


def prepare(stores, *, request=None, authorize=None, build=None):
    native, requests, results = stores
    request = {"operation": "scene", "settings": {}} if request is None else request
    contract = adapter.retain_request(requests, prefix="studio.product", contract_id=11, engine_id=2, revision=9, request=request)
    dispatcher = adapter.StudioProductDispatcher(prefix="studio.product", requests=requests, results=results,
        authorize=authorize or (lambda loan, request: {"trusted": True}),
        build_scene=build or (lambda request, context: {"blob": "scene product"}))
    loan = native.Loan(*contract, 3, 4, 100)
    return native, contract, loan, dispatcher


def test_real_peers_exchange_retained_blob_and_commit_once(stores):
    work, commits = [], []
    native, contract, loan, dispatcher = prepare(stores,
        build=lambda request, context: work.append((request, context)) or {"blob": "MRP1-base64-owned-by-caller"})
    producer = adapter.StudioProductProducer(contract, prefix="studio.product", results=stores[2],
        commit=lambda blob: commits.append(blob) or 1)
    ea, eb = socket.socketpair()
    ha, hb = socket.socketpair()
    eb.setblocking(False)
    hb.setblocking(False)
    with ThreadPoolExecutor(max_workers=1) as executor:
        engine = native.EnginePeer(ea, producer)
        service = native.ServicePeer(ha, executor, dispatcher.dispatch, dispatcher.on_receipt, announce=False)
        try:
            assert engine.submit(contract)
            assert engine.submit(contract)  # Actual EnginePeer sends once until admitted.
            assert receive(eb, [engine], native) == [(native.SUBMIT, (*contract, 0, 0, 0, 0))]
            eb.sendall(native.encode_admitted(contract[0]))
            engine.poll()
            assert producer.admitted_seen
            frame = native.encode(native.OFFER, loan)
            hb.sendall(frame[:13])
            service.poll()
            assert not work
            hb.sendall(frame[13:])
            frames = receive(hb, [service], native, 2)
            assert [kind for kind, _ in frames] == [native.ACCEPT, native.RESULT]
            handle = frames[1][1][-1]
            assert service.loan == loan
            assert commits == []  # Host preparation cannot birth a ray job.
            assert len(work) == 1
            assert stores[2]._path(handle).exists()
            hb.sendall(native.encode(native.RECEIPT, loan, 1))
            service.poll()
            assert service.loan is None
            assert stores[2]._path(handle).exists()  # FIFO acceptance is not commit.
            route = native.encode_route((*loan.values(), handle))
            eb.sendall(route)
            decision = receive(eb, [engine], native)[0]
            assert decision[0] == native.DECISION and decision[1][-1] == native.DECISION_COMMIT
            assert len(commits) == 1
            eb.sendall(route)
            assert receive(eb, [engine], native)[0] == decision
            assert len(commits) == 1
            assert stores[2]._path(handle).exists()
            dispatcher.release_result(loan)  # Explicit owner reconciliation.
            stores[1].release(contract[3])
            assert not stores[2]._path(handle).exists()
            assert not stores[1]._path(contract[3]).exists()
        finally:
            engine.close()
            service.close()
            eb.close()
            hb.close()


@pytest.mark.parametrize("operation", ["scene", "ray"])
def test_trusted_authorization_denies_before_work_despite_payload_caps(stores, operation):
    work = []
    native, contract, loan, dispatcher = prepare(stores,
        request={"operation": operation, "capabilities": ["photo_submit"], "role": "admin"},
        authorize=lambda loan, request: None,
        build=lambda *args: work.append(args))
    a, b = socket.socketpair()
    with ThreadPoolExecutor(max_workers=1) as executor:
        service = native.ServicePeer(a, executor, dispatcher.dispatch, dispatcher.on_receipt, announce=False)
        try:
            b.sendall(native.encode(native.OFFER, loan))
            deadline = time.monotonic() + 3
            while service.error is None and time.monotonic() < deadline:
                service.poll()
                time.sleep(.001)
            assert isinstance(service.error, PermissionError)
            for _ in range(4):
                service.poll()
            assert work == []
            assert list(stores[2].directory.glob("*.json")) == []
            assert stores[1]._path(contract[3]).exists()
        finally:
            service.close()
            b.close()


def test_prefix_and_native_identity_mismatches_never_build_or_commit(stores):
    work, commits = [], []
    native, contract, loan, dispatcher = prepare(stores, build=lambda *args: work.append(args) or {"blob": "ok"})
    bad = native.Loan(loan.contract, loan.engine + 1, loan.revision, loan.payload, loan.generation, loan.host, loan.deadline)
    with pytest.raises(ValueError, match="identity differs"):
        dispatcher.dispatch(bad)
    dispatcher.prefix = "different.prefix"
    with pytest.raises(ValueError, match="identity differs"):
        dispatcher.dispatch(loan)
    assert work == []
    dispatcher.prefix = "studio.product"
    handle = dispatcher.dispatch(loan)
    producer = adapter.StudioProductProducer(contract, prefix="studio.product", results=stores[2], commit=lambda value: commits.append(value) or 1)
    assert producer.route((*bad.values(), handle)) == -1
    changed_generation = (*loan.values()[:4], loan.generation + 1, loan.host, loan.deadline, handle)
    with pytest.raises(ValueError, match="identity differs"):
        producer.route(changed_generation)
    assert not commits


@pytest.mark.parametrize("failure", ["raise", "invalid"])
def test_uncertain_commit_is_held_without_side_effect_replay(stores, failure):
    native, contract, loan, dispatcher = prepare(stores)
    handle = dispatcher.dispatch(loan)
    calls = []
    def commit(product):
        calls.append(product)
        if failure == "raise":
            raise RuntimeError("Job birth outcome is uncertain")
        return "created"
    producer = adapter.StudioProductProducer(contract, prefix="studio.product", results=stores[2], commit=commit)
    a, b = socket.socketpair()
    b.setblocking(False)
    engine = native.EnginePeer(a, producer)
    try:
        route = native.encode_route((*loan.values(), handle))
        for _ in range(3):
            b.sendall(route)
            assert receive(b, [engine], native)[0][1][-1] == native.DECISION_HOLD
        assert len(calls) == 1
        assert producer.uncertain and producer.error is not None
        assert stores[2]._path(handle).exists()
    finally:
        engine.close()
        b.close()


def test_hold_then_reject_is_terminal_and_different_result_is_refused(stores):
    native, contract, loan, dispatcher = prepare(stores)
    handle = dispatcher.dispatch(loan)
    decisions = iter([0, -1])
    calls = []
    producer = adapter.StudioProductProducer(contract, prefix="studio.product", results=stores[2],
        commit=lambda product: calls.append(product) or next(decisions))
    result = (*loan.values(), handle)
    assert producer.route(result) == 0
    assert producer.route(result) == -1
    assert producer.route(result) == -1
    assert producer.route((*loan.values(), handle + 1)) == -1
    assert len(calls) == 2


def test_payload_bounds_and_nonfinite_values_are_existing_store_rules(stores):
    for value in ({"settings": "x" * 5000}, {"setting": float("nan")}):
        with pytest.raises(ValueError):
            adapter.retain_request(stores[1], prefix="studio.product", contract_id=1, engine_id=2, revision=0, request=value)
    with pytest.raises(ValueError):
        adapter.retain_request(stores[1], prefix="studio.product", contract_id=True, engine_id=2, revision=0, request={})
    with pytest.raises(TypeError):
        adapter.StudioProductDispatcher(prefix="studio.product", requests=stores[1], results=stores[2], authorize=None, build_scene=lambda *args: {})


def test_retained_result_reuses_blob_and_never_persists_trusted_context(stores):
    calls = []
    native, contract, loan, dispatcher = prepare(stores,
        authorize=lambda *args: {"trusted_context_only": "not a wire capability"},
        build=lambda request, context: calls.append(context) or {"scene": request})
    handle = dispatcher.dispatch(loan)
    assert dispatcher.dispatch(loan) == handle
    assert len(calls) == 1
    assert "trusted_context_only" not in stores[2]._path(handle).read_text()
    dispatcher.on_receipt(loan, False)
    assert stores[2]._path(handle).exists()


def _native_fifo_probe(directory):
    """Run only in a bounded subprocess: the existing DLL owns the actual FIFO."""
    native, payload = adapter.nodus_modules()
    nodus = adapter.DEFAULT_NODUS_TOOLS.parent
    dll = Path(os.environ.get("NODUS_STUDIO_NATIVE_DLL", nodus / "build" / "Release" / "nodus_contract_service.dll"))
    if not dll.is_file():
        print(f"NATIVE_DEPENDENCY_UNAVAILABLE: {dll}", flush=True)
        return 77
    keepers = []
    if hasattr(os, "add_dll_directory"):
        keepers.append(os.add_dll_directory(str(dll.parent)))
        torch = importlib.util.find_spec("torch")
        if torch and torch.origin:
            libraries = Path(torch.origin).parent / "lib"
            if libraries.is_dir():
                keepers.append(os.add_dll_directory(str(libraries)))
    sys.path.insert(0, str(adapter.DEFAULT_NODUS_TOOLS))
    from nodus_mail_host import NativeMailHost
    os.chdir(directory)  # Native scratch state never touches a live workspace.
    stores = native, payload.PayloadStore(directory / "requests"), payload.PayloadStore(directory / "results")
    builds, commits, receipts, pending, decisions = [], [], [], {}, {}
    native, contract, loan, dispatcher = prepare(stores,
        build=lambda request, context: builds.append(request) or {"blob": "native-fifo-scene"})
    producer = adapter.StudioProductProducer(contract, prefix="studio.product", results=stores[2],
        commit=lambda value: commits.append(value) or 1)
    def route(result):
        work, current = result.loan.work, result.loan
        values = (work.id, work.engine, work.revision, work.payload,
                  current.generation, current.host, current.deadline, result.payload)
        pending.setdefault(current.generation, values)
        return decisions.get(current.generation, 0)
    def receipt(current, accepted):
        dispatcher.on_receipt(current, accepted)
        receipts.append(current)
        assert accepted
    a, b = socket.socketpair()
    host = None
    try:
        try:
            host = NativeMailHost(dll, a, contract[1], lambda: None if producer.admitted_seen else contract,
                                  route, producer.admitted, capacity=8, ticks=1000000)
        except OSError as error:
            print(f"NATIVE_DEPENDENCY_UNAVAILABLE: {type(error).__name__}: {error}", flush=True)
            return 77
        with ThreadPoolExecutor(max_workers=1) as executor:
            service = native.ServicePeer(b, executor, dispatcher.dispatch, receipt)
            try:
                deadline = time.monotonic() + 5
                retired = False
                while time.monotonic() < deadline:
                    host.step()
                    service.poll()
                    assert not host.errors, host.errors
                    assert service.error is None, service.error
                    # Owner validation/commit occurs outside native callbacks.
                    for generation, result in pending.items():
                        if generation not in decisions:
                            decisions[generation] = producer.route(result)
                    if commits and receipts and host.retire(contract[0]):
                        retired = True
                        break
                    time.sleep(.001)
                assert retired and producer.admitted_seen, {
                    "admitted": producer.admitted_seen, "builds": len(builds), "commits": len(commits),
                    "receipts": len(receipts), "pending": pending, "decisions": decisions,
                    "peer_closed": service.closed, "peer_loan": service.loan, "peer_result_sent": service.result_sent}
                assert len(builds) == len(commits) == len(receipts) == 1
                assert commits == [{"blob": "native-fifo-scene"}]
                assert stores[2]._path(producer.result_handle).exists()
                dispatcher.release_result(receipts[0])
                stores[1].release(contract[3])
                print("NATIVE_FIFO_PASS: one admitted request, one pure build, one FIFO receipt, one owner commit, retired", flush=True)
            finally:
                service.close()
    finally:
        if host is not None:
            host.close()
        a.close()
        b.close()
        for keeper in keepers:
            keeper.close()
    return 0


def test_actual_native_fifo_blob_exchange_when_installed(tmp_path):
    if not os.environ.get("NODUS_STUDIO_NATIVE_DLL"):
        pytest.skip("Set NODUS_STUDIO_NATIVE_DLL to the explicit installed native host artifact to test")
    result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), str(tmp_path)],
                            capture_output=True, text=True, timeout=20)
    if result.returncode == 77:
        pytest.skip(result.stdout.strip())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NATIVE_FIFO_PASS" in result.stdout
    print(result.stdout.strip())


if __name__ == "__main__":
    raise SystemExit(_native_fifo_probe(Path(sys.argv[1])))

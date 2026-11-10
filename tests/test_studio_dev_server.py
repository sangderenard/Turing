"""Bounded localhost/SQL tests. Never touch production state or listeners."""
import importlib.util
from pathlib import Path
import socket
import sqlite3
import threading
import urllib.request
import urllib.error
import pytest

spec = importlib.util.spec_from_file_location('studio_dev', Path(__file__).resolve().parents[1] / 'tools/serve_studio_game_layer.py')
dev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dev)


def test_identity_port_and_fenced_release(tmp_path):
    db = tmp_path / 'instances.db'
    first, identity = dev.acquire(db, 'one', tmp_path, tmp_path, [0])
    second = None
    try:
        assert first.server_address[0] == '127.0.0.1'
        with pytest.raises(RuntimeError, match='already leased'):
            dev.acquire(db, 'one', tmp_path, tmp_path, [0])
        second, other = dev.acquire(db, 'two', tmp_path, tmp_path, [first.server_port, 0])
        assert second.server_port != first.server_port
        dev.renew(db, identity)
        dev.release(db, 'wrong-generation')
        assert len(dev.status(db)) == 2
        dev.release(db, identity)
        with pytest.raises(RuntimeError, match='lost'):
            dev.renew(db, identity)
        assert dev.status(db)[0]['identity'] == other
    finally:
        first.server_close()
        if second:
            second.server_close()


def test_expired_name_does_not_replace_listener(tmp_path):
    db = tmp_path / 'instances.db'
    server, identity = dev.acquire(db, 'one', tmp_path, tmp_path, [0])
    port = server.server_port
    with sqlite3.connect(db) as conn:
        conn.execute('UPDATE dev_server_leases SET expires=0')
    try:
        with pytest.raises(RuntimeError, match='still has a listener'):
            dev.acquire(db, 'one', tmp_path, tmp_path, [0])
    finally:
        server.server_close()
    replacement, new_identity = dev.acquire(db, 'one', tmp_path, tmp_path, [port])
    try:
        assert new_identity != identity
        dev.release(db, identity)
        assert dev.status(db)[0]['identity'] == new_identity
    finally:
        replacement.server_close()
        dev.release(db, new_identity)


def test_collision_does_not_stop_foreign_listener(tmp_path):
    with socket.socket() as foreign:
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            foreign.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        foreign.bind(('127.0.0.1', 0))
        foreign.listen()
        with pytest.raises(RuntimeError, match='No available'):
            dev.acquire(tmp_path / 'db', 'one', tmp_path, tmp_path, [foreign.getsockname()[1]])
        assert foreign.fileno() >= 0


def test_real_http_overlay_readonly_and_containment(tmp_path):
    stage, assets = tmp_path / 'stage', tmp_path / 'assets'
    stage.mkdir()
    assets.mkdir()
    (stage / 'marble_studio_layer.html').write_text('stage')
    (assets / 'fallback.js').write_text('asset')
    (tmp_path / 'private.txt').write_text('must not serve')
    server, identity = dev.acquire(tmp_path / 'db', 'http', stage, assets, [0])
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        for path, expected in [('/', b'stage'), ('/fallback.js', b'asset')]:
            with urllib.request.urlopen(base + path) as response:
                assert response.read() == expected
                assert response.headers['Cache-Control'] == 'no-store'
        for path in ['/api/test', '/%61pi/test', '/../private.txt', '/..%5cprivate.txt', '/C:/Windows/win.ini']:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base + path)
            assert error.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(base + '/', data=b'no'))
        assert error.value.code == 405
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        dev.release(tmp_path / 'db', identity)


def test_concurrent_claim_one_winner(tmp_path):
    db = tmp_path / 'db'
    # Establish schema before simultaneous acquisition; both still take the same SQL lock.
    with sqlite3.connect(db) as conn:
        conn.execute(dev.DDL)
    barrier = threading.Barrier(2)
    winners, errors = [], []
    def attempt():
        barrier.wait()
        try:
            winners.append(dev.acquire(db, 'same', tmp_path, tmp_path, [0]))
        except RuntimeError as error:
            errors.append(str(error))
    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        assert len(winners) == len(errors) == 1
        assert 'already leased' in errors[0]
    finally:
        for server, identity in winners:
            server.server_close()
            dev.release(db, identity)


def test_status_does_not_create_database(tmp_path):
    db = tmp_path / 'absent.db'
    assert dev.status(db) == []
    assert not db.exists()

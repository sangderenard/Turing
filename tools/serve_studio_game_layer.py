"""SQL-leased, read-only localhost QA for Studio/game staged and live assets.

Run again after a shutdown to reclaim an expired lease; never kills listeners.
Mutable lease state is separate from the production accounts/state database.
"""
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit
import argparse
import json
import os
import signal
import socket
import sqlite3
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'build/dot-studio-sync-20261110'
LIVE = ROOT / 'build/meowdoku/web'
DEFAULT_DB = Path.home() / '.turing-dev/sql/instances.sqlite'
LEASE_SECONDS = 15
DDL = '''CREATE TABLE IF NOT EXISTS dev_server_leases (
 name TEXT PRIMARY KEY, identity TEXT NOT NULL UNIQUE, port INTEGER NOT NULL UNIQUE,
 host TEXT NOT NULL CHECK(host = '127.0.0.1'), pid INTEGER NOT NULL,
 stage TEXT NOT NULL, assets TEXT NOT NULL, started REAL NOT NULL,
 heartbeat REAL NOT NULL, expires REAL NOT NULL)'''


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, '.js': 'text/javascript', '.mjs': 'text/javascript'}

    def translate_path(self, path):
        relative = unquote(urlsplit(path).path).lstrip('/') or 'marble_studio_layer.html'
        stage = Path(getattr(self.server, 'stage', STAGE)).resolve()
        assets = Path(getattr(self.server, 'assets', LIVE)).resolve()
        # Resolve before containment checks, including Windows drive/backslash paths
        # and links. A dev web root must never expose files outside those roots.
        for root in (stage, assets):
            candidate = (root / relative).resolve()
            if candidate.is_relative_to(root) and candidate.is_file():
                return str(candidate)
        return str(stage / '__dev_missing_file__')

    def do_HEAD(self):
        path = unquote(urlsplit(self.path).path).replace('\\', '/')
        if path == '/api' or path.startswith('/api/'):
            self.send_error(404, 'Read-only QA: production APIs are not proxied')
            return
        for name in ('If-Modified-Since', 'If-None-Match'):
            if name in self.headers:
                del self.headers[name]
        super().do_HEAD()

    def do_GET(self):
        path = unquote(urlsplit(self.path).path).replace('\\', '/')
        if path == '/api' or path.startswith('/api/'):
            self.send_error(404, 'Read-only QA: production APIs are not proxied')
            return
        for name in ('If-Modified-Since', 'If-None-Match'):
            if name in self.headers:
                del self.headers[name]
        super().do_GET()

    def do_POST(self):
        self.send_error(405, 'Read-only QA: no production mutations')

    def list_directory(self, path):
        self.send_error(404, 'Directory listings are disabled')

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


class DevServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can allow two listeners to bind one port. Exclusive
    # ownership must come from the held OS socket, not a probe then later bind.
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def status(db_path):
    path = Path(db_path).resolve()
    if not path.exists():
        return []
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute('SELECT * FROM dev_server_leases ORDER BY name')]


def acquire(db_path, name, stage, assets, ports):
    """SQL transaction + bound socket are one acquisition. No port-probe race."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    server = None
    try:
        db.execute(DDL)
        db.execute('BEGIN IMMEDIATE')
        now = time.time()
        row = db.execute('SELECT port, expires FROM dev_server_leases WHERE name=?', (name,)).fetchone()
        if row:
            if row[1] > now:
                raise RuntimeError(f'{name} already leased on 127.0.0.1:{row[0]}')
            # A delayed heartbeat does not authorize a second instance while the
            # original still listens. Refuse even if that listener is unrelated.
            with socket.socket() as probe:
                probe.settimeout(0.25)
                if probe.connect_ex(('127.0.0.1', row[0])) == 0:
                    raise RuntimeError(f'{name} lease expired but port {row[0]} still has a listener')
        for port in ports:
            if db.execute('SELECT 1 FROM dev_server_leases WHERE port=? AND expires>?', (port, now)).fetchone():
                continue
            try:
                server = DevServer(('127.0.0.1', port), Handler)
                break
            except OSError:
                continue
        if server is None:
            raise RuntimeError('No available localhost port in the requested range')
        port = server.server_port
        identity = uuid.uuid4().hex
        db.execute('DELETE FROM dev_server_leases WHERE (name=? OR port=?) AND expires<=?', (name, port, now))
        db.execute('INSERT INTO dev_server_leases VALUES (?,?,?,?,?,?,?,?,?,?)',
                   (name, identity, port, '127.0.0.1', os.getpid(), str(Path(stage).resolve()),
                    str(Path(assets).resolve()), now, now, now + LEASE_SECONDS))
        db.commit()
        server.stage, server.assets = Path(stage), Path(assets)
        return server, identity
    except BaseException:
        db.rollback()
        if server:
            server.server_close()
        raise
    finally:
        db.close()


def renew(db_path, identity):
    with sqlite3.connect(db_path, timeout=2) as db:
        now = time.time()
        changed = db.execute('UPDATE dev_server_leases SET heartbeat=?, expires=? WHERE identity=?',
                             (now, now + LEASE_SECONDS, identity)).rowcount
        if changed != 1:
            raise RuntimeError('Dev instance lost its SQL lease')


def release(db_path, identity):
    with sqlite3.connect(db_path, timeout=2) as db:
        db.execute('DELETE FROM dev_server_leases WHERE identity=?', (identity,))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', default='studio-game')
    ap.add_argument('--stage', type=Path, default=STAGE)
    ap.add_argument('--assets', type=Path, default=LIVE)
    ap.add_argument('--db', type=Path, default=DEFAULT_DB)
    ap.add_argument('--port', type=int, help='Request exactly this port; otherwise lease 8900..8999')
    ap.add_argument('--status', action='store_true')
    args = ap.parse_args(argv)
    if args.status:
        print(json.dumps(status(args.db), indent=2))
        return 0
    if not args.name.strip() or len(args.name) > 128:
        ap.error('--name must contain 1..128 characters')
    if args.port is not None and not 1024 <= args.port <= 65535:
        ap.error('--port must be 1024..65535')
    if not args.stage.is_dir() or not args.assets.is_dir():
        ap.error('--stage and --assets must be existing asset directories')
    server, identity = acquire(args.db, args.name, args.stage, args.assets,
                               [args.port] if args.port else range(8900, 9000))
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    for sig in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), stop)
    server.timeout = 0.5
    print(json.dumps({'name': args.name, 'identity': identity, 'pid': os.getpid(),
                      'url': f'http://127.0.0.1:{server.server_port}/marble_studio_layer.html',
                      'db': str(args.db)}), flush=True)
    next_renew = time.monotonic() + LEASE_SECONDS / 3
    try:
        while not stopping:
            server.handle_request()
            if time.monotonic() >= next_renew:
                renew(args.db, identity)
                next_renew = time.monotonic() + LEASE_SECONDS / 3
    finally:
        server.server_close()
        release(args.db, identity)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


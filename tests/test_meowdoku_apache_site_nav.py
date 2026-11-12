"""Exercise the real Apache HTML filter against an isolated loopback fixture.

No production config, service, TLS, backend or session is touched. Run with:
MEOWDOKU_TEST_HTTPD=C:/Apache24/bin/httpd.exe python -m pytest <this file> -q
"""
from __future__ import annotations

import gzip
import http.client
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = Path(__file__).resolve().parents[1]
FRAGMENT = ROOT / 'examples/meowdoku/deploy/apache/meowdoku-proxy.conf'
NAV = b"<script type='module' src='/site-nav.js' data-site-nav='allow'></script>"
HTML = b'<!doctype html><html><body><h1>Fixture</h1></body></html>'


@pytest.fixture(scope='module')
def apache_front(tmp_path_factory):
    exe = os.environ.get('MEOWDOKU_TEST_HTTPD')
    if not exe:
        pytest.skip('Set MEOWDOKU_TEST_HTTPD to exercise an isolated Apache instance')
    exe = Path(exe).resolve()
    home = exe.parent.parent
    temp = tmp_path_factory.mktemp('apache_site_nav')
    source = FRAGMENT.read_text(encoding='utf-8')
    block = source.split('    # BEGIN shared site navigation filter\n', 1)[1].split('    # END shared site navigation filter', 1)[0]
    observed = []

    class Backend(BaseHTTPRequestHandler):
        def do_GET(self):
            observed.append((self.path, self.headers.get('Cookie'), self.headers.get('Authorization')))
            payload = HTML
            ctype = 'text/html; charset=utf-8'
            extra = {}
            status = 200
            if self.path == '/existing.html':
                payload = HTML.replace(b'</body>', NAV + b'</body>')
            elif self.path == '/upper.html':
                payload = HTML.replace(b'</body>', b'</BODY>')
            elif self.path == '/binary.wasm':
                ctype = 'application/wasm'
                payload = b'\x00asm\x01\x00</body>\xff'
            elif self.path == '/api/json':
                ctype = 'application/json'
                payload = b'{"example":"</body>"}'
            elif self.path == '/compressed.html':
                extra['Content-Encoding'] = 'gzip'
                payload = gzip.compress(HTML, mtime=0)
            elif self.path == '/download.html':
                extra['Content-Disposition'] = 'attachment; filename=page.html'
            elif self.path == '/denied.html':
                status = 403
            elif self.path == '/partial.html':
                status = 206
                extra['Content-Range'] = f'bytes 0-{len(HTML)-1}/{len(HTML)}'
            self.send_response(status)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Set-Cookie', 'fixture_session=retained; HttpOnly')
            for name, value in extra.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    backend = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    modules = ['authz_core', 'authz_host', 'log_config', 'mime', 'proxy', 'proxy_http', 'filter', 'substitute']
    conf = '\n'.join([
        f'ServerRoot "{home.as_posix()}"',
        f'Listen 127.0.0.1:{port}',
        'ServerName localhost',
        f'PidFile "{temp.as_posix()}/httpd.pid"',
        f'ErrorLog "{temp.as_posix()}/error.log"',
        *[f'LoadModule {name}_module modules/mod_{name}.so' for name in modules],
        f'TypesConfig "{home.as_posix()}/conf/mime.types"',
        f'ProxyPass / http://127.0.0.1:{backend.server_port}/',
        block,
    ])
    config = temp / 'httpd.conf'
    config.write_text(conf, encoding='utf-8')
    subprocess.run([str(exe), '-t', '-f', str(config)], check=True, capture_output=True, text=True)
    process = subprocess.Popen([str(exe), '-X', '-f', str(config)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def request(path):
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
        try:
            conn.request('GET', path, headers={'Cookie': 'fixture=abc', 'Authorization': 'Bearer fixture-only'})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()
    try:
        for _ in range(60):
            if process.poll() is not None:
                raise AssertionError(process.communicate()[1].decode(errors='replace') + (temp / 'error.log').read_text())
            try:
                request('/ready')
                break
            except OSError:
                time.sleep(.1)
        else:
            raise AssertionError('Isolated Apache did not become ready')
        yield request, observed
    finally:
        # Only this fixture-owned process is terminated, never the live service.
        process.terminate()
        process.communicate(timeout=10)
        backend.shutdown()
        backend.server_close()
        thread.join(timeout=3)


def test_html_injects_existing_module(apache_front):
    request, observed = apache_front
    for path in ('/', '/page.html', '/site/manual/', '/upper.html'):
        status, headers, body = request(path)
        assert status == 200
        assert body.count(NAV) == 1
        assert b'<h1>Fixture</h1>' in body
        assert headers['Set-Cookie'] == 'fixture_session=retained; HttpOnly'
        assert observed[-1] == (path, 'fixture=abc', 'Bearer fixture-only')


def test_existing_tag_keeps_identical_module_identity(apache_front):
    _, _, body = apache_front[0]('/existing.html')
    # Browsers evaluate a module URL once per document, even with multiple tags.
    # Do not use a version query or wrapper URL that creates a second identity.
    assert body.count(NAV) == 2
    assert b'/site-nav.js?' not in body


@pytest.mark.parametrize('path, expected', [
    ('/api/html', HTML), ('/api', HTML), ('/ws', HTML), ('/ws/spectate', HTML),
    ('/binary.wasm', b'\x00asm\x01\x00</body>\xff'),
    ('/api/json', b'{"example":"</body>"}'),
    ('/download.html', HTML), ('/compressed.html', gzip.compress(HTML, mtime=0)),
    ('/denied.html', HTML), ('/partial.html', HTML),
])
def test_non_document_responses_are_unchanged(apache_front, path, expected):
    status, _, body = apache_front[0](path)
    assert status == {'/denied.html': 403, '/partial.html': 206}.get(path, 200)
    assert body == expected

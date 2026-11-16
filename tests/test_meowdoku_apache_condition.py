"""The site condition block of the Apache fragment, run in an isolated Apache with a loopback fixture backend.

No production config, service, TLS, backend or session is touched. Run with:
MEOWDOKU_TEST_HTTPD=C:/Apache24/bin/httpd.exe python -m pytest <this file> -q
"""
import http.client
import os
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FRAGMENT = ROOT / 'examples/meowdoku/deploy/apache/meowdoku-proxy.conf'
CARDS = ROOT / 'examples/meowdoku/deploy/apache/condition'


def _free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.fixture(scope='module')
def apache_front(tmp_path_factory):
    exe = os.environ.get('MEOWDOKU_TEST_HTTPD')
    if not exe:
        pytest.skip('Set MEOWDOKU_TEST_HTTPD to exercise an isolated Apache instance')
    exe = Path(exe).resolve()
    home = exe.parent.parent
    temp = tmp_path_factory.mktemp('apache_condition')
    condition = temp / 'condition'
    condition.mkdir()
    for card in ('yellow', 'red'):
        shutil.copyfile(CARDS / f'{card}.html', condition / f'{card}.html')
    source = FRAGMENT.read_text(encoding='utf-8')
    block = source.split('    # BEGIN site condition\n', 1)[1].split('    # END site condition', 1)[0]
    observed = []

    class Backend(BaseHTTPRequestHandler):
        def do_GET(self):
            observed.append(self.path)
            payload = b'{"ok": true}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    backend = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    port, dead = _free_port(), _free_port()
    modules = ['authz_core', 'authz_host', 'log_config', 'mime', 'alias', 'setenvif', 'headers', 'rewrite', 'filter',
               'proxy', 'proxy_http']
    conf = '\n'.join([
        f'ServerRoot "{home.as_posix()}"',
        f'Listen 127.0.0.1:{port}',
        'ServerName localhost',
        f'PidFile "{temp.as_posix()}/httpd.pid"',
        f'ErrorLog "{temp.as_posix()}/error.log"',
        *[f'LoadModule {name}_module modules/mod_{name}.so' for name in modules],
        f'TypesConfig "{home.as_posix()}/conf/mime.types"',
        f'Define MEOWDOKU_CONDITION "{condition.as_posix()}"',
        block,
        f'RewriteRule ^/dead$ http://127.0.0.1:{dead}/ [P,L]',
        f'RewriteRule ^/(.*)$ http://127.0.0.1:{backend.server_port}/$1 [P,L]',
    ])
    config = temp / 'httpd.conf'
    config.write_text(conf, encoding='utf-8')
    subprocess.run([str(exe), '-t', '-f', str(config)], check=True, capture_output=True, text=True)
    process = subprocess.Popen([str(exe), '-X', '-f', str(config)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def request(path):
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            conn.request('GET', path)
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read().decode('utf-8')
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
        yield request, observed, condition
    finally:
        # Only this fixture-owned process is terminated, never the live service.
        process.terminate()
        process.communicate(timeout=10)
        backend.shutdown()
        backend.server_close()
        thread.join(timeout=3)


def test_a_posted_card_turns_every_request_away_without_a_restart(apache_front):
    request, observed, condition = apache_front
    assert request('/api/state')[0] == 200
    (condition / 'yellow.on').write_text('test\n', encoding='utf-8')
    try:
        observed.clear()
        for path in ('/', '/api/state', '/mcp', '/ws'):
            status, headers, body = request(path)
            assert status == 503 and 'We are under mild attack' in body, path
            assert headers.get('Retry-After') == '300', path
        assert observed == [], 'nothing reaches the backend while a card is posted'
        for path in ('/api/condition', '/api/condition/controls.js'):
            assert request(path)[0] == 200, path
        assert observed == ['/api/condition', '/api/condition/controls.js'], 'only the switch passes a card'
        assert request('/api/condition/other')[0] == 503
        (condition / 'red.on').write_text('test\n', encoding='utf-8')
        status, headers, body = request('/api/state')
        assert status == 503 and 'genuine threat' in body, 'the red card is read first'
    finally:
        for name in ('yellow.on', 'red.on'):
            (condition / name).unlink(missing_ok=True)
    assert request('/api/state')[0] == 200, 'taking the card down opens the site again'


def test_an_unreachable_backend_shows_the_red_card_and_flags_are_not_served(apache_front):
    request, _, condition = apache_front
    status, headers, body = request('/dead')
    assert status == 503 and 'genuine threat' in body
    assert 'Retry-After' not in headers, 'Retry-After belongs to a posted card'
    (condition / 'yellow.on').write_text('test\n', encoding='utf-8')
    try:
        assert request('/_condition/yellow.on')[0] == 403
    finally:
        (condition / 'yellow.on').unlink()
    status, headers, body = request('/_condition/yellow.html')
    assert status == 200 and headers.get('Cache-Control') == 'no-store' and 'site-nav' not in body

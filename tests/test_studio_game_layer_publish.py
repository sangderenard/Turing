from pathlib import Path
import io
import json
import shutil
import ssl
import stat
from types import SimpleNamespace
from urllib.parse import urlsplit
import pytest
from tools import publish_studio_game_layer as publisher


def fixture(tmp_path):
    stage, live = tmp_path / 'stage', tmp_path / 'live'
    stage.mkdir(); live.mkdir()
    main = b'const original = true;'
    html = b'<script src="marble_studio.js"></script>'
    (live / 'marble_studio.js').write_bytes(main)
    (live / 'marble_studio.html').write_bytes(html)
    original = tmp_path / 'original-game.html'
    original.write_bytes(b'original worker and game')
    files = {'marble_studio_layer.html': html.replace(b'marble_studio.js', b'./marble_studio_layer_main.js'),
             'marble_studio_layer_main.js': b'original plus opt-in',
             'mechanical_creature_studio_layer.html': b'complete original game plus bridge',
             'studio-game-layer/module.js': b'export const optional=true;'}
    for name, data in files.items():
        p = stage / name; p.parent.mkdir(exist_ok=True); p.write_bytes(data)
    receipt = {'files': {n: publisher.digest(b) for n, b in files.items()},
               'studio': {'source_sha256': publisher.digest(main)},
               'original_game_path': str(original),
               'original_game_sha256': publisher.digest(original.read_bytes())}
    (stage / 'stage-receipt.json').write_text(json.dumps(receipt))
    return stage, live, receipt


def read_json(path):
    return json.loads(path.read_text())


def tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def fake_https(monkeypatch, live, *, on_request=None):
    class Response:
        def __init__(self, url):
            self.url = url
        def geturl(self):
            return self.url
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            name = urlsplit(self.url).path.lstrip('/')
            if on_request is not None:
                answer = on_request(name)
                if answer is not None:
                    return answer
            return (live / name).read_bytes()
    monkeypatch.setattr(publisher, 'urlopen', lambda url, **kwargs: Response(url))
    calls = []
    def context(**kwargs):
        calls.append(kwargs)
        return object()
    monkeypatch.setattr(publisher.ssl, 'create_default_context', context)
    return calls


def previous(tmp_path):
    stage, live, receipt = fixture(tmp_path)
    for name in receipt['files']:
        target = live / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes((stage / name).read_bytes())
    prior = {'published': True, 'files': receipt['files'],
             'https_verified': list(receipt['files']),
             'original_game_sha256': receipt['original_game_sha256']}
    (stage / 'publication.json').write_text(json.dumps(prior))
    new = tmp_path / 'upgrade'
    shutil.copytree(stage, new, ignore=shutil.ignore_patterns('publication.json'))
    name = 'studio-game-layer/module.js'
    body = b'export const optional="upgraded";'
    (new / name).write_bytes(body)
    receipt['files'][name] = publisher.digest(body)
    (new / 'stage-receipt.json').write_text(json.dumps(receipt))
    return new, live, stage / 'publication.json'


def test_additive_guard_rejects_changed_source_and_existing_destination(tmp_path):
    stage, live, _ = fixture(tmp_path)
    assert len(publisher.checked_files(stage, live)[1]) == 4
    (live / 'marble_studio_layer_main.js').write_bytes(b'other work')
    with pytest.raises(ValueError, match='Existing destination differs'):
        publisher.checked_files(stage, live)
    (live / 'marble_studio.js').write_bytes(b'newer published source')
    with pytest.raises(ValueError, match='Published Studio changed'):
        publisher.checked_files(stage, live)


def test_missing_server_never_writes_live(tmp_path, monkeypatch):
    stage, live, _ = fixture(tmp_path)
    before = tree(live)
    monkeypatch.setattr(publisher, 'STAGE', stage)
    monkeypatch.setattr(publisher, 'LIVE', live)
    monkeypatch.setattr(publisher.ssl, 'create_default_context', lambda **kwargs: object())
    def unavailable(*args, **kwargs):
        raise ConnectionRefusedError('server stopped for DLL access')
    monkeypatch.setattr(publisher, 'urlopen', unavailable)
    with pytest.raises(ConnectionRefusedError):
        publisher.publish()
    assert before == tree(live)
    assert not (stage / 'publication.json').exists()


def test_staged_hash_and_html_are_bound(tmp_path):
    stage, live, _ = fixture(tmp_path)
    (stage / 'studio-game-layer/module.js').write_bytes(b'changed after QA')
    with pytest.raises(ValueError, match='Staged bytes changed'):
        publisher.checked_files(stage, live)
    # Repair the stage to isolate the independent primary-HTML source guard.
    (stage / 'studio-game-layer/module.js').write_bytes(b'export const optional=true;')
    (live / 'marble_studio.html').write_bytes(b'changed published HTML')
    with pytest.raises(ValueError, match='Published Studio HTML changed'):
        publisher.checked_files(stage, live)


def test_original_game_source_is_checked(tmp_path):
    stage, live, receipt = fixture(tmp_path)
    Path(receipt['original_game_path']).write_bytes(b'concurrent original change')
    with pytest.raises(ValueError, match='Original game source changed'):
        publisher.checked_files(stage, live)


def test_successful_additive_publish_installs_entry_last_and_preserves_sources(tmp_path, monkeypatch):
    stage, live, receipt = fixture(tmp_path)
    fake_https(monkeypatch, live)
    before = tree(live)
    actual_order = []
    install = publisher._install
    def observe(temporary, target, expected):
        assert target.read_bytes() if target.exists() else expected is None
        assert temporary.read_bytes() == (stage / target.relative_to(live)).read_bytes()
        actual_order.append(target.relative_to(live).as_posix())
        install(temporary, target, expected)
    monkeypatch.setattr(publisher, '_install', observe)
    result = publisher.publish(stage=stage, live=live)
    assert result['published'] and result['replaced'] == []
    assert actual_order[-1] == 'marble_studio_layer.html'
    assert set(actual_order) == set(receipt['files'])
    for name, body in before.items():
        assert (live / name).read_bytes() == body
    assert set(result['https_verified']) == set(receipt['files'])


def test_upgrade_backs_up_exact_prior_bytes_and_preserves_unrelated_files(tmp_path, monkeypatch):
    stage, live, prior = previous(tmp_path)
    (live / 'unrelated.txt').write_bytes(b'keep this work')
    before, frozen = tree(live), tree(prior.parent)
    contexts = fake_https(monkeypatch, live)
    name = 'studio-game-layer/module.js'
    install = publisher._install
    def inspect_backup(temporary, target, expected):
        assert (stage / 'publication-backup' / name).read_bytes() == before[name]
        assert read_json(stage / 'publication-in-progress.json')['before'][name] == publisher.digest(before[name])
        install(temporary, target, expected)
    monkeypatch.setattr(publisher, '_install', inspect_backup)
    result = publisher.publish(stage=stage, live=live, prior_publication=prior, ca_file=None)
    assert result['published'] and result['replaced'] == [name]
    assert result['installed'] == [name]
    assert result['multi_file_atomic'] is False
    assert contexts == [{'cafile': None}]
    assert tree(prior.parent) == frozen
    assert (live / name).read_bytes() == (stage / name).read_bytes()
    for unchanged in set(before) - {name}:
        assert (live / unchanged).read_bytes() == before[unchanged]


@pytest.mark.parametrize('tamper', ['failed', 'hash', 'verification', 'prior-stage', 'installed'])
def test_upgrade_refuses_tampered_prior_or_installed_bytes(tmp_path, monkeypatch, tamper):
    stage, live, prior = previous(tmp_path)
    data = read_json(prior)
    if tamper == 'failed':
        data['published'] = False
    elif tamper == 'hash':
        data['files']['studio-game-layer/module.js'] = publisher.digest(b'forged')
    elif tamper == 'verification':
        data['https_verified'].pop()
    elif tamper == 'prior-stage':
        (prior.parent / 'studio-game-layer/module.js').write_bytes(b'tampered frozen stage')
    else:
        (live / 'studio-game-layer/module.js').write_bytes(b'other live work')
    prior.write_text(json.dumps(data))
    before = tree(live)
    fake_https(monkeypatch, live)
    with pytest.raises(ValueError):
        publisher.publish(stage=stage, live=live, prior_publication=prior)
    assert tree(live) == before
    assert not (stage / 'publication-in-progress.json').exists()


def test_upgrade_without_prior_cannot_overwrite(tmp_path, monkeypatch):
    stage, live, _ = previous(tmp_path)
    before = tree(live)
    fake_https(monkeypatch, live)
    with pytest.raises(ValueError, match='Existing destination differs'):
        publisher.publish(stage=stage, live=live)
    assert tree(live) == before


def test_failed_https_verification_rolls_back_replacements_and_additions(tmp_path, monkeypatch):
    stage, live, prior = previous(tmp_path)
    new = 'studio-game-layer/new.js'
    receipt = read_json(stage / 'stage-receipt.json')
    (stage / new).write_bytes(b'new file')
    receipt['files'][new] = publisher.digest(b'new file')
    (stage / 'stage-receipt.json').write_text(json.dumps(receipt))
    before = tree(live)
    fake_https(monkeypatch, live, on_request=lambda name: b'bad response' if name != 'marble_studio.js' else None)
    with pytest.raises(ValueError, match='Served hash differs'):
        publisher.publish(stage=stage, live=live, prior_publication=prior)
    assert tree(live) == before
    failure = read_json(stage / 'publication.json')
    assert failure['published'] is False and failure['rollback']['complete'] is True
    assert failure['rollback']['restored'] == ['studio-game-layer/module.js']
    assert failure['rollback']['removed'] == [new]


def test_rollback_leaves_concurrent_changes_and_reports_conflict(tmp_path, monkeypatch):
    stage, live, prior = previous(tmp_path)
    name = 'studio-game-layer/module.js'
    def changed_during_verification(request):
        if request != 'marble_studio.js':
            (live / name).write_bytes(b'new work by another writer')
            return b'bad response'
    fake_https(monkeypatch, live, on_request=changed_during_verification)
    with pytest.raises(ValueError, match='Served hash differs'):
        publisher.publish(stage=stage, live=live, prior_publication=prior)
    assert (live / name).read_bytes() == b'new work by another writer'
    failure = read_json(stage / 'publication.json')
    assert failure['rollback']['complete'] is False
    assert failure['rollback']['conflicts'][0]['file'] == name
    assert (stage / 'publication-backup' / name).read_bytes() == (prior.parent / name).read_bytes()


def test_additive_concurrent_creation_cannot_be_overwritten(tmp_path, monkeypatch):
    stage, live, _ = fixture(tmp_path)
    fake_https(monkeypatch, live)
    link = publisher.os.link
    def racing_link(source, target):
        target.write_bytes(b'concurrent creation')
        return link(source, target)
    monkeypatch.setattr(publisher.os, 'link', racing_link)
    with pytest.raises(FileExistsError):
        publisher.publish(stage=stage, live=live)
    assert (live / 'marble_studio_layer_main.js').read_bytes() == b'concurrent creation'
    assert read_json(stage / 'publication.json')['installed'] == []


@pytest.mark.parametrize('change', ['stage', 'stage-receipt', 'source', 'destination', 'prior', 'backup'])
def test_rechecks_all_inputs_immediately_before_mutation(tmp_path, monkeypatch, change):
    stage, live, prior = previous(tmp_path)
    before = tree(live)
    fake_https(monkeypatch, live)
    temporary = publisher._temporary
    name = 'studio-game-layer/module.js'
    def change_after_preparation(target, body):
        result = temporary(target, body)
        if target == live / name:
            if change == 'stage':
                (stage / 'marble_studio_layer_main.js').write_bytes(b'changed staged dependency')
            elif change == 'stage-receipt':
                (stage / 'stage-receipt.json').write_text((stage / 'stage-receipt.json').read_text() + ' ')
            elif change == 'source':
                (live / 'marble_studio.js').write_bytes(b'new source')
            elif change == 'destination':
                (live / name).write_bytes(b'new destination')
            elif change == 'backup':
                (stage / 'publication-backup' / name).write_bytes(b'changed backup')
            else:
                prior.write_text(prior.read_text() + ' ')
        return result
    monkeypatch.setattr(publisher, '_temporary', change_after_preparation)
    with pytest.raises(ValueError):
        publisher.publish(stage=stage, live=live, prior_publication=prior)
    expected = dict(before)
    if change == 'source':
        expected['marble_studio.js'] = b'new source'
    elif change == 'destination':
        expected[name] = b'new destination'
    assert tree(live) == expected
    assert read_json(stage / 'publication.json')['installed'] == []


def test_unrelated_concurrent_change_is_preserved_and_triggers_rollback(tmp_path, monkeypatch):
    stage, live, prior = previous(tmp_path)
    (live / 'unrelated.txt').write_bytes(b'old')
    def changed(request):
        if request != 'marble_studio.js':
            (live / 'unrelated.txt').write_bytes(b'new other work')
    fake_https(monkeypatch, live, on_request=changed)
    with pytest.raises(ValueError, match='Pre-existing files changed'):
        publisher.publish(stage=stage, live=live, prior_publication=prior)
    assert (live / 'unrelated.txt').read_bytes() == b'new other work'
    assert (live / 'studio-game-layer/module.js').read_bytes() == (prior.parent / 'studio-game-layer/module.js').read_bytes()


@pytest.mark.parametrize('name', ['marble_studio.js', 'marble_studio.html', '../outside.js',
                                   'studio-game-layer/../marble_studio.js',
                                   'studio-game-layer/module.js:stream',
                                   'studio-game-layer/CON.js', 'studio-game-layer/module.js.'])
def test_unsafe_or_primary_destinations_are_never_publishable(tmp_path, name):
    stage, live, receipt = fixture(tmp_path)
    receipt['files'][name] = publisher.digest(b'bad')
    (stage / 'stage-receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        publisher.checked_files(stage, live)


def test_custom_ca_is_forwarded_and_http_is_rejected(tmp_path, monkeypatch):
    stage, live, _ = fixture(tmp_path)
    contexts = fake_https(monkeypatch, live)
    with pytest.raises(ValueError, match='Trusted HTTPS'):
        publisher.publish(stage=stage, live=live, base_url='http://example.test/')
    publisher.publish(stage=stage, live=live, ca_file=tmp_path / 'custom-ca.pem')
    assert contexts == [{'cafile': str(tmp_path / 'custom-ca.pem')}]


def test_temp_cleanup_failure_does_not_lose_installed_tracking(tmp_path, monkeypatch):
    stage, live, _ = fixture(tmp_path)
    before = tree(live)
    fake_https(monkeypatch, live)
    unlink = Path.unlink
    failed = False
    def fail_once(path, *args, **kwargs):
        nonlocal failed
        if path.parent == live and path.name.endswith('.studio-layer.tmp') and not failed:
            failed = True
            raise PermissionError('injected temporary cleanup failure')
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', fail_once)
    with pytest.raises(PermissionError, match='temporary cleanup'):
        publisher.publish(stage=stage, live=live)
    failure = read_json(stage / 'publication.json')
    assert failure['installed'] == ['marble_studio_layer_main.js']
    assert failure['rollback']['removed'] == failure['installed']
    # The private temporary survives the injected OS denial. It is not served
    # as any manifest path and must not be mistaken for an installed layer.
    leftovers = [name for name in tree(live) if name.endswith('.studio-layer.tmp')]
    assert len(leftovers) == 1
    assert {name: body for name, body in tree(live).items() if name not in leftovers} == before


def test_mutation_failure_after_first_upgrade_rolls_back_prior_changes(tmp_path, monkeypatch):
    stage, live, prior = previous(tmp_path)
    receipt = read_json(stage / 'stage-receipt.json')
    name = 'marble_studio_layer_main.js'
    (stage / name).write_bytes(b'new bootstrap')
    receipt['files'][name] = publisher.digest(b'new bootstrap')
    (stage / 'stage-receipt.json').write_text(json.dumps(receipt))
    before = tree(live)
    fake_https(monkeypatch, live)
    install = publisher._install
    count = 0
    def fail_second(temporary, target, expected):
        nonlocal count
        count += 1
        if count == 2:
            raise PermissionError('injected mutation failure')
        install(temporary, target, expected)
    monkeypatch.setattr(publisher, '_install', fail_second)
    with pytest.raises(PermissionError, match='mutation failure'):
        publisher.publish(stage=stage, live=live, prior_publication=prior)
    assert tree(live) == before
    failure = read_json(stage / 'publication.json')
    assert failure['rollback']['complete'] is True
    assert failure['rollback']['restored'] == [name]


def test_failed_receipt_write_still_reports_rollback(tmp_path, monkeypatch, capsys):
    stage, live, prior = previous(tmp_path)
    before = tree(live)
    fake_https(monkeypatch, live)
    record = publisher._record
    def failed_receipt(path, result):
        if path.name == 'publication.json':
            raise PermissionError('injected receipt failure')
        return record(path, result)
    monkeypatch.setattr(publisher, '_record', failed_receipt)
    with pytest.raises(PermissionError, match='receipt failure'):
        publisher.publish(stage=stage, live=live, prior_publication=prior)
    assert tree(live) == before
    report = json.loads(capsys.readouterr().out.strip())
    assert report['rollback']['complete'] is True
    assert report['receipt_error'] == 'injected receipt failure'


@pytest.mark.parametrize('location', ['root', 'directory', 'file'])
def test_windows_reparse_attribute_refuses_paths_on_python311(tmp_path, monkeypatch, location):
    root = tmp_path / 'live'
    target = root / 'studio-game-layer' / 'module.js'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'local fixture only')
    marked = {'root': root, 'directory': target.parent, 'file': target}[location]
    lstat = Path.lstat
    def reparse_lstat(path, *args, **kwargs):
        if path == marked:
            return SimpleNamespace(st_mode=stat.S_IFDIR,
                                   st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        if marked in path.parents:
            raise AssertionError('Must reject the ancestor before inspecting its child')
        return lstat(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'lstat', reparse_lstat)
    with pytest.raises(ValueError, match='Redirected publication path'):
        publisher._path(root, 'studio-game-layer/module.js')


def test_snapshot_rejects_unrelated_reparse_directory_before_traversal(tmp_path, monkeypatch):
    stage, live, _ = fixture(tmp_path)
    redirect = live / 'unrelated-directory'
    redirect.mkdir()
    (redirect / 'never-read.txt').write_bytes(b'local fixture, no actual redirect')
    fake_https(monkeypatch, live)
    lstat, iterdir, read_bytes = Path.lstat, Path.iterdir, Path.read_bytes
    def reparse_lstat(path, *args, **kwargs):
        if path == redirect:
            return SimpleNamespace(st_mode=stat.S_IFDIR,
                                   st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        return lstat(path, *args, **kwargs)
    def checked_iterdir(path):
        assert path != redirect, 'Must reject reparse directory before descending'
        return iterdir(path)
    def checked_read(path):
        assert redirect not in path.parents, 'Must not read through reparse directory'
        return read_bytes(path)
    monkeypatch.setattr(Path, 'lstat', reparse_lstat)
    monkeypatch.setattr(Path, 'iterdir', checked_iterdir)
    monkeypatch.setattr(Path, 'read_bytes', checked_read)
    with pytest.raises(ValueError, match='Redirected publication path'):
        publisher.publish(stage=stage, live=live)
    assert not (stage / 'publication-in-progress.json').exists()
    assert not (live / 'marble_studio_layer_main.js').exists()


def routed_https(monkeypatch, *, body=b'verified bytes', status=200, certificate_error=False, reply=None):
    """Exercise real HTTPConnection routing/parsing without any network access."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    sockets, connections, wraps = [], [], []
    class Socket:
        def __init__(self):
            self.request = b''
            self.closed = False
        def setsockopt(self, *args):
            pass
        def sendall(self, data):
            self.request += data
        def makefile(self, *args):
            response_status = status
            extra = ''
            if reply is None:
                data = body(self.request) if callable(body) else body
            else:
                response_status, response_headers, data = reply(self.request)
                extra = ''.join(f'{key}: {value}\r\n' for key, value in response_headers)
            headers = (f'HTTP/1.1 {response_status} Test\r\nContent-Length: {len(data)}\r\n' + extra +
                       'Location: http://insecure.invalid/\r\n\r\n').encode()
            return io.BytesIO(headers + data)
        def close(self):
            self.closed = True
    def create_connection(address, timeout, source_address):
        connections.append((address, timeout, source_address))
        sock = Socket()
        sockets.append(sock)
        return sock
    def wrap_socket(sock, *, server_hostname):
        wraps.append((server_hostname, context.check_hostname, context.verify_mode))
        if certificate_error:
            raise ssl.SSLCertVerificationError('certificate rejected')
        return sock
    monkeypatch.setattr(publisher.socket, 'create_connection', create_connection)
    monkeypatch.setattr(context, 'wrap_socket', wrap_socket)
    return context, connections, wraps, sockets


@pytest.mark.parametrize('origin,override,expected', [
    ('https://studio.example/', 443, 443),
    ('https://studio.example:8766/', 443, 443),
    ('https://studio.example:8766/', None, 8766),
])
def test_connection_override_preserves_origin_sni_host_and_verification(tmp_path, monkeypatch,
                                                                      origin, override, expected):
    context, connections, wraps, sockets = routed_https(monkeypatch)
    result = publisher._served(origin + 'asset.js?version=1', context, 7,
                               connect_host='127.0.0.1', connect_port=override)
    assert result == b'verified bytes'
    assert connections == [(('127.0.0.1', expected), 7, None)]
    assert wraps == [('studio.example', True, ssl.CERT_REQUIRED)]
    assert b'GET /asset.js?version=1 HTTP/1.1\r\n' in sockets[0].request
    assert ('Host: ' + urlsplit(origin).netloc + '\r\n').encode() in sockets[0].request
    assert b'Host: 127.0.0.1' not in sockets[0].request
    assert sockets[0].closed


@pytest.mark.parametrize('status', [301, 302, 307, 404])
def test_connection_override_refuses_redirects_and_non_200_responses(monkeypatch, status):
    context, connections, _, sockets = routed_https(monkeypatch, status=status)
    with pytest.raises(ValueError, match='requires status 200'):
        publisher._served('https://studio.example/asset.js', context, 5,
                          connect_host='127.0.0.1', connect_port=443)
    assert len(connections) == 1
    assert sockets[0].closed


@pytest.mark.parametrize('mode', [ssl.CERT_REQUIRED, ssl.CERT_OPTIONAL, ssl.CERT_NONE])
def test_connection_override_rejects_disabled_hostname_or_certificate_checks(monkeypatch, mode):
    context, connections, _, _ = routed_https(monkeypatch)
    context.check_hostname = False
    context.verify_mode = mode
    with pytest.raises(ValueError, match='requires certificate and hostname verification'):
        publisher._served('https://studio.example/asset.js', context, 5, connect_host='127.0.0.1')
    assert connections == []


def test_connection_override_propagates_certificate_failure(monkeypatch):
    context, _, wraps, sockets = routed_https(monkeypatch, certificate_error=True)
    with pytest.raises(ssl.SSLCertVerificationError, match='certificate rejected'):
        publisher._served('https://studio.example/asset.js', context, 5, connect_host='127.0.0.1')
    assert wraps == [('studio.example', True, ssl.CERT_REQUIRED)]
    assert sockets[0].request == b'' and sockets[0].closed


@pytest.mark.parametrize('url,host,port', [
    ('http://studio.example/', '127.0.0.1', 443),
    ('https://user:secret@studio.example/', '127.0.0.1', 443),
    ('https://studio.example/', None, 443),
    ('https://studio.example/', 'http://127.0.0.1', 443),
    ('https://studio.example/', '127.0.0.1', 0),
    ('https://studio.example/', '127.0.0.1', 65536),
])
def test_connection_override_refuses_insecure_or_invalid_configuration(monkeypatch, url, host, port):
    context, connections, _, _ = routed_https(monkeypatch)
    with pytest.raises(ValueError):
        publisher._served(url, context, 5, connect_host=host, connect_port=port)
    assert connections == []


def test_publish_uses_and_records_connection_override_for_every_verification(tmp_path, monkeypatch):
    stage, live, receipt = fixture(tmp_path)
    def body(request):
        path = request.split(b' ', 2)[1].decode().lstrip('/')
        return (live / path).read_bytes()
    context, connections, wraps, _ = routed_https(monkeypatch, body=body)
    ca_calls = []
    def default_context(**kwargs):
        ca_calls.append(kwargs)
        return context
    monkeypatch.setattr(publisher.ssl, 'create_default_context', default_context)
    result = publisher.publish(stage=stage, live=live, base_url='https://studio.example/',
                               ca_file=None, connect_host='127.0.0.1', connect_port=443)
    assert result['published']
    assert ca_calls == [{'cafile': None}]
    assert result['verification_base_url'] == 'https://studio.example/'
    assert result['verification_connection'] == {'connect_host': '127.0.0.1', 'connect_port': 443}
    assert len(connections) == len(receipt['files']) + 1
    assert all(c[0] == ('127.0.0.1', 443) for c in connections)
    assert all(w == ('studio.example', True, ssl.CERT_REQUIRED) for w in wraps)


def development_service(tmp_path, monkeypatch, live, *, fault=None):
    from http.cookies import SimpleCookie
    requests = []
    secret = tmp_path / 'operator.secret'
    secret.write_text('fixture-password')
    def reply(request):
        head, _, raw_body = request.partition(b'\r\n\r\n')
        lines = head.decode().split('\r\n')
        method, path, _ = lines[0].split(' ')
        headers = dict(line.split(': ', 1) for line in lines[1:])
        cookies = SimpleCookie(headers.get('Cookie', ''))
        signed_in = cookies.get('meowdoku_account') is not None
        route = cookies['meowdoku_route'].value if 'meowdoku_route' in cookies else 'a'
        requests.append((method, path, signed_in, route))
        outgoing = []
        status = 200
        if path == '/api/account/login':
            assert method == 'POST'
            assert json.loads(raw_body) == {'username': 'codex', 'password': 'fixture-password'}
            if fault == 'login':
                status, payload = 401, {'error': 'fixture-password fixture-session'}
            else:
                outgoing = [('Set-Cookie', 'meowdoku_route=a; Path=/; Secure')]
                if fault != 'no-session':
                    outgoing.append(('Set-Cookie', 'meowdoku_account=fixture-session; Path=/; Secure; HttpOnly'))
                payload = {'account': {'username': 'codex'}}
        elif path == '/api/whoami':
            payload = {'signed_in': signed_in,
                       'username': 'different-user' if fault == 'identity' else 'codex',
                       'capabilities': [] if fault == 'capability' else ['dev_ui'],
                       'instance': 'meowdoku-dev' if route == 'b' and fault != 'wrong-instance' else 'meowdoku-a'}
        elif path == '/api/route':
            assert method == 'GET', 'Publication must not change stored site preference'
            payload = {'dev_available': True, 'dev_route': 'b',
                       'route': 'a' if fault == 'wrong-route' else route,
                       'selected': route, 'preference': 'stable'}
        else:
            assert signed_in and route == 'b', 'Every file must use its authenticated Development route'
            if fault == 'expire' and path != '/marble_studio.js':
                outgoing.append(('Set-Cookie', 'meowdoku_account=; Max-Age=0; Path=/; Secure'))
            return 200, outgoing, (live / path.lstrip('/')).read_bytes()
        return status, outgoing, json.dumps(payload).encode()
    context, _, wraps, _ = routed_https(monkeypatch, reply=reply)
    monkeypatch.setattr(publisher.ssl, 'create_default_context', lambda **kwargs: context)
    return {'site': 'dev', 'user': 'codex', 'password_file': str(secret), 'ca_file': None,
            'base_url': 'https://studio.example/', 'connect_host': '127.0.0.1', 'connect_port': 443}, requests, wraps


def test_development_publication_verifies_every_file_with_authenticated_route(tmp_path, monkeypatch, capsys):
    stage, live, receipt = fixture(tmp_path)
    options, requests, wraps = development_service(tmp_path, monkeypatch, live)
    result = publisher.publish(stage=stage, live=live, **options)
    assert result['published'] and result['verification_site'] == 'dev'
    assert result['verification_instance'] == 'meowdoku-dev'
    files = [(path.lstrip('/'), signed, route) for method, path, signed, route in requests
             if not path.startswith('/api/')]
    assert [name for name, _, _ in files] == ['marble_studio.js'] + result['https_verified']
    assert len(files) == len(receipt['files']) + 1
    assert all(signed and route == 'b' for _, signed, route in files)
    assert all(w == ('studio.example', True, ssl.CERT_REQUIRED) for w in wraps)
    assert sum(method == 'POST' for method, *_ in requests) == 1
    output = capsys.readouterr().out + json.dumps(result)
    assert 'fixture-password' not in output and 'fixture-session' not in output
    assert all(b'fixture-password' not in body and b'fixture-session' not in body for body in tree(stage).values())


@pytest.mark.parametrize('fault', ['login', 'no-session', 'identity', 'capability', 'wrong-instance', 'wrong-route'])
def test_development_rejects_missing_auth_or_wrong_target_before_writes(tmp_path, monkeypatch, fault, capsys):
    stage, live, _ = fixture(tmp_path)
    before = tree(live)
    options, _, _ = development_service(tmp_path, monkeypatch, live, fault=fault)
    with pytest.raises(ValueError) as raised:
        publisher.publish(stage=stage, live=live, **options)
    assert tree(live) == before
    assert not (stage / 'publication-in-progress.json').exists()
    output = str(raised.value) + capsys.readouterr().out
    assert 'fixture-password' not in output and 'fixture-session' not in output


def test_development_session_expiry_rolls_back_and_omits_secrets(tmp_path, monkeypatch, capsys):
    stage, live, prior = previous(tmp_path)
    before = tree(live)
    options, _, _ = development_service(tmp_path, monkeypatch, live, fault='expire')
    with pytest.raises(ValueError, match='Authenticated operator'):
        publisher.publish(stage=stage, live=live, prior_publication=prior, **options)
    assert tree(live) == before
    report = read_json(stage / 'publication.json')
    assert report['rollback']['complete'] is True
    output = capsys.readouterr().out + json.dumps(report)
    assert 'fixture-password' not in output and 'fixture-session' not in output


def test_development_root_cannot_use_anonymous_stable_verification(tmp_path, monkeypatch):
    stage, live, _ = fixture(tmp_path)
    monkeypatch.setattr(publisher, 'DEVELOPMENT_LIVE', live)
    with pytest.raises(ValueError, match='requires authenticated --site dev'):
        publisher.publish(stage=stage, live=live)


def test_development_requires_explicit_operator_before_authentication(tmp_path):
    stage, live, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='requires an operator user'):
        publisher.publish(stage=stage, live=live, site='dev')

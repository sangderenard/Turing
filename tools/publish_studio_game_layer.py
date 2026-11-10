"""Publish the reviewed private layer, optionally upgrading a prior publication.

Default operation is additive. Upgrades require a successful publication.json
and its unchanged adjacent stage. Primary Studio/original game are never targets.
Backups and a write-ahead journal stay in the new stage; no server is started.
Each file is installed atomically, with the entry last. This is NOT a multi-file
transaction: already-open pages may observe mixed dependency generations. Hash
checks immediately precede replacement/rollback, but the filesystem offers no
compare-and-swap against non-cooperating writers. Quiesce those writers first.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import http.client
import http.cookiejar
import json
import os
import re
import socket
import ssl
import stat
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from examples.meowdoku.deploy.agent_messenger import Client as _OperatorClient, read_secret

STAGE = ROOT / 'build/dot-studio-sync-20261110'
LIVE = ROOT / 'build/meowdoku/web'
DEVELOPMENT_LIVE = ROOT.parent / 'srv/state-dev/web'
BASE_URL = 'https://127.0.0.1:8766/'
CA_FILE = ROOT / 'build/meowdoku/tls/ca.crt'
ORIGINAL_GAME = ROOT / 'build/dot-newer-tire-dynamics-20261110/index.html'
ENTRIES = {'marble_studio_layer.html', 'marble_studio_layer_main.js',
           'mechanical_creature_studio_layer.html'}
_DEFAULT_CA = object()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _manifest(names):
    if not isinstance(names, dict) or not ENTRIES.issubset(names):
        raise ValueError('The complete additive layer is required')
    seen = set()
    for name, sha in names.items():
        parts = PurePosixPath(name).parts
        if (not re.fullmatch(r'[A-Za-z0-9_./-]+', name)
                or '/'.join(parts) != name or '..' in parts or '.' in parts
                or PurePosixPath(name).is_absolute()
                or any(p.endswith('.') or p.split('.')[0].upper() in
                       {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)),
                        *(f'LPT{i}' for i in range(10))} for p in parts)
                or name.lower() in seen):
            raise ValueError('Unsafe additive path: ' + name)
        if name not in ENTRIES and not name.startswith('studio-game-layer/'):
            raise ValueError('Only the private layer dependency folder is publishable')
        if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha):
            raise ValueError('Invalid file hash: ' + name)
        seen.add(name.lower())


def _path(root, name):
    """Refuse symlinks/junctions, including directory redirects on Windows."""
    path = root / name
    # Check ancestors first so even lstat of a child cannot traverse a known
    # redirect. Path.is_junction is absent on the deployed Python 3.11 runtime;
    # lstat exposes the Windows reparse attribute there (including junctions).
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError('Redirected publication path: ' + str(part))
    return path


def _snapshot(live):
    """Check every directory before descending, not after an rglob traversal."""
    result = {}
    pending = [live]
    while pending:
        directory = pending.pop()
        _path(directory, '')
        for entry in directory.iterdir():
            path = _path(directory, entry.name)
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                result[path.relative_to(live).as_posix()] = digest(path.read_bytes())
    return result


def _hash(path):
    return digest(path.read_bytes()) if path.exists() else None


def _stage(stage):
    raw = _path(stage, 'stage-receipt.json').read_bytes()
    receipt = json.loads(raw)
    _manifest(receipt['files'])
    files = {}
    for name, sha in receipt['files'].items():
        body = _path(stage, name).read_bytes()
        if digest(body) != sha:
            raise ValueError('Staged bytes changed: ' + name)
        files[name] = body
    return receipt, files, raw


def _prior(path):
    if path is None:
        return None, None
    path = Path(path).absolute()
    raw = _path(path.parent, path.name).read_bytes()
    prior = json.loads(raw)
    stage, _, stage_raw = _stage(path.parent)
    if (prior.get('published') is not True or prior.get('files') != stage['files']
            or not isinstance(prior.get('https_verified'), list)
            or sorted(prior['https_verified']) != sorted(stage['files'])
            or prior.get('original_game_sha256') != stage.get('original_game_sha256')
            or ('stage_receipt_sha256' in prior
                and prior['stage_receipt_sha256'] != digest(stage_raw))):
        raise ValueError('Prior publication is not an intact successful stage receipt')
    return prior, raw


def _sources(stage, live, receipt, files):
    if digest(_path(live, 'marble_studio.js').read_bytes()) != receipt['studio']['source_sha256']:
        raise ValueError('Published Studio changed; rebase and verify the candidate first')
    main_html = _path(live, 'marble_studio.html').read_bytes()
    old = b'src="marble_studio.js"'
    if main_html.count(old) != 1:
        old = b'src="./marble_studio.js"'
    if main_html.count(old) != 1 or main_html.replace(
            old, b'src="./marble_studio_layer_main.js"') != files['marble_studio_layer.html']:
        raise ValueError('Published Studio HTML changed; rebase and verify first')
    original = Path(receipt.get('original_game_path', ORIGINAL_GAME))
    if not original.is_absolute():
        raise ValueError('Original game source must be an absolute path')
    expected = receipt.get('original_game_sha256')
    if not expected or _hash(_path(original.parent, original.name)) != expected:
        raise ValueError('Original game source changed; rebase and verify first')


def checked_files(stage, live, prior_publication=None):
    stage, live = Path(stage).absolute(), Path(live).absolute()
    receipt, files, _ = _stage(stage)
    _sources(stage, live, receipt, files)
    prior, _ = _prior(prior_publication)
    for name, body in files.items():
        current = _hash(_path(live, name))
        if current is not None and current != digest(body):
            if prior is None:
                raise ValueError('Existing destination differs; no overwrite authorized by this additive publisher: ' + name)
            if current != prior['files'].get(name):
                raise ValueError('Installed destination differs from prior publication: ' + name)
    return receipt, files


def _durable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as output:
        output.write(body)
        output.flush()
        os.fsync(output.fileno())
    _sync_directory(path.parent)


def _sync_directory(path):
    # Windows fsync flushes file contents; directory fsync is unavailable there.
    if os.name != 'nt':
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _temporary(target, body):
    fd, name = tempfile.mkstemp(prefix='.' + target.name + '.', suffix='.studio-layer.tmp', dir=target.parent)
    path = Path(name)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _assert_hash(target, expected):
    if _hash(target) != expected:
        raise ValueError('Concurrent destination change; refusing mutation: ' + str(target))


def _install(temporary, target, expected):
    _assert_hash(target, expected)
    if expected is None:
        # Exclusive hard-link publication is atomic and cannot overwrite a file
        # created after the absence check. Temporary is on the same filesystem.
        os.link(temporary, target)
    else:
        os.replace(temporary, target)


def _rollback(live, installed, files, backups):
    result = {'restored': [], 'removed': [], 'conflicts': [], 'errors': []}
    for name in reversed(installed):
        temporary = None
        try:
            target = _path(live, name)
            _assert_hash(target, digest(files[name]))
            if name in backups:
                backup, expected = backups[name]
                body = _path(backup.parent, backup.name).read_bytes()
                if digest(body) != expected:
                    raise ValueError('Backup hash changed: ' + name)
                temporary = _temporary(target, body)
                _assert_hash(target, digest(files[name]))
                os.replace(temporary, target)
                result['restored'].append(name)
            else:
                _assert_hash(target, digest(files[name]))
                target.unlink()
                result['removed'].append(name)
            _sync_directory(target.parent)
        except ValueError as error:
            result['conflicts'].append({'file': name, 'error': str(error)})
        except OSError as error:
            result['errors'].append({'file': name, 'error': str(error)})
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    result['complete'] = not result['conflicts'] and not result['errors']
    return result


def _record(path, result):
    temporary = _temporary(path, json.dumps(result, indent=2).encode('utf-8'))
    try:
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _connection_override(connect_host, connect_port):
    if connect_host is None:
        if connect_port is not None:
            raise ValueError('connect_port requires connect_host')
    elif (not isinstance(connect_host, str) or not connect_host
          or any(c.isspace() or c in '/?#@' for c in connect_host)):
        raise ValueError('connect_host must be a bare hostname or address')
    if connect_port is not None and (type(connect_port) is not int or not 1 <= connect_port <= 65535):
        raise ValueError('connect_port must be an integer from 1 to 65535')


class _ConnectHTTPSConnection(http.client.HTTPSConnection):
    """Override only TCP routing; stdlib HTTPS retains origin SNI/cert checks."""

    def __init__(self, host, *, port, context, timeout, connect_host, connect_port):
        if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
            raise ValueError('Connection override requires certificate and hostname verification')
        super().__init__(host, port=port, context=context, timeout=timeout)
        self._connect_host = connect_host
        self._connect_port = connect_port
        # HTTPConnection's socket factory is deliberately isolated from host:
        # inherited HTTPSConnection.connect still wraps with self.host as SNI.
        self._create_connection = self._connect_address

    def _connect_address(self, address, *args, **kwargs):
        port = self._connect_port if self._connect_port is not None else address[1]
        return socket.create_connection((self._connect_host, port), *args, **kwargs)


class _NoVerificationRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Authenticated verification refuses redirects')


class _OriginHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, origin, context, connect_host, connect_port):
        super().__init__(context=context)
        self.origin = origin
        self.context = context
        self.connect_host = connect_host or origin.hostname
        self.connect_port = connect_port

    def https_open(self, request):
        target = urlsplit(request.full_url)
        if (target.scheme, target.netloc) != ('https', self.origin.netloc):
            raise ValueError('Authenticated verification cannot leave its HTTPS origin')
        def connection(host, **kwargs):
            return _ConnectHTTPSConnection(
                self.origin.hostname, port=self.origin.port or 443,
                context=self.context, timeout=kwargs['timeout'],
                connect_host=self.connect_host, connect_port=self.connect_port)
        return self.do_open(connection, request)


class _DevelopmentVerificationClient(_OperatorClient):
    """Use the operator's established login, keeping session/route cookies in memory."""

    def __init__(self, base, user, password_file, state_dir, context, transport):
        origin = urlsplit(base)
        if (origin.scheme != 'https' or not origin.hostname or origin.username is not None
                or origin.password is not None or origin.path not in ('', '/')
                or origin.query or origin.fragment):
            raise ValueError('Development verification requires an HTTPS origin without a path')
        if not user:
            raise ValueError('Development verification requires an operator user')
        self.base, self.user = base.rstrip('/'), user
        self.cookies = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(self.cookies),
            _OriginHTTPSHandler(origin, context, **transport), _NoVerificationRedirect())
        # Reuse the established helper and POST /api/account/login. The secret
        # is neither retained on this object nor put in receipts or exceptions.
        self._login(read_secret(user, password_file, state_dir))
        who = self.call('/api/whoami')
        self._assert_account(who)
        route = self.call('/api/route')
        if route.get('dev_available') is not True or route.get('dev_route') != 'b':
            raise ValueError('Development route b is not available')
        # This is a request-local routing choice, not POST /api/route, which
        # would change the operator account's persistent site preference.
        for cookie in list(self.cookies):
            if cookie.name == 'meowdoku_route':
                self.cookies.clear(cookie.domain, cookie.path, cookie.name)
        self.cookies.set_cookie(http.cookiejar.Cookie(
            0, 'meowdoku_route', 'b', None, False, origin.hostname, False, False,
            '/', True, True, None, True, None, None, {}, False))
        self.assert_target()

    def call(self, path, body=None, timeout=15):
        try:
            return super().call(path, body, timeout)
        except Exception:
            # Never echo a response body, request body, or cookie-bearing
            # exception, including an error produced by the login endpoint.
            raise ValueError('Authenticated verification request failed: ' + path.split('?')[0]) from None

    def _assert_account(self, who):
        if (who.get('signed_in') is not True or who.get('username') != self.user
                or 'dev_ui' not in who.get('capabilities', [])):
            raise ValueError('Authenticated operator with dev_ui is required')

    def assert_target(self):
        who = self.call('/api/whoami')
        self._assert_account(who)
        if who.get('instance') != 'meowdoku-dev':
            raise ValueError('Verification reached the wrong instance; expected meowdoku-dev')
        route = self.call('/api/route')
        if (route.get('route') != 'b' or route.get('selected') != 'b'
                or route.get('dev_route') != 'b' or route.get('dev_available') is not True):
            raise ValueError('Verification did not confirm Development route b')

    def read_file(self, name, timeout):
        self.assert_target()
        try:
            request = urllib.request.Request(self.base + '/' + name,
                                             headers={'Accept-Encoding': 'identity'})
            with self.opener.open(request, timeout=timeout) as response:
                if response.status != 200:
                    raise ValueError('Unexpected status')
                body = response.read()
        except Exception:
            raise ValueError('Authenticated file verification failed: ' + name) from None
        self.assert_target()
        return body


def _served(url, context, timeout, *, connect_host=None, connect_port=None):
    origin = urlsplit(url)
    if (origin.scheme != 'https' or not origin.hostname or origin.username is not None
            or origin.password is not None or origin.fragment):
        raise ValueError('Trusted HTTPS origin without credentials or fragment is required')
    _connection_override(connect_host, connect_port)
    if connect_host is not None:
        connection = _ConnectHTTPSConnection(
            origin.hostname, port=origin.port or 443, context=context, timeout=timeout,
            connect_host=connect_host, connect_port=connect_port)
        try:
            path = origin.path or '/'
            if origin.query:
                path += '?' + origin.query
            connection.request('GET', path, headers={'Host': origin.netloc,
                                                     'Accept-Encoding': 'identity'})
            response = connection.getresponse()
            # Never follow a redirect through the pinned TCP destination.
            if response.status != 200:
                raise ValueError('HTTPS verification requires status 200, got ' + str(response.status))
            return response.read()
        finally:
            connection.close()
    with urlopen(url, context=context, timeout=timeout) as response:
        if urlsplit(response.geturl()).scheme != 'https':
            raise ValueError('HTTPS verification redirected to an insecure URL')
        return response.read()


def publish(*, stage=None, live=None, base_url=None, prior_publication=None, ca_file=_DEFAULT_CA,
            connect_host=None, connect_port=None, site='stable', user=None,
            password_file=None, state_dir=None):
    stage, live = Path(stage or STAGE).absolute(), Path(live or LIVE).absolute()
    base_url = (base_url or BASE_URL).rstrip('/') + '/'
    if urlsplit(base_url).scheme != 'https':
        raise ValueError('Trusted HTTPS is required')
    _connection_override(connect_host, connect_port)
    if site not in ('stable', 'dev'):
        raise ValueError('Publication site must be stable or dev')
    if live == DEVELOPMENT_LIVE.absolute() and site != 'dev':
        raise ValueError('Development web root requires authenticated --site dev verification')
    if site == 'dev' and (not user or live == LIVE.absolute()):
        raise ValueError('Development publication requires an operator user and a Development web root')
    if site == 'stable' and any(value is not None for value in (user, password_file, state_dir)):
        raise ValueError('Operator credentials are only accepted with --site dev')
    if stage == live or live in stage.parents or stage in live.parents:
        raise ValueError('Stage and live directories must be separate')
    if prior_publication is not None and Path(prior_publication).absolute().parent == stage:
        raise ValueError('Upgrade requires a new stage; preserve the prior publication')
    if (stage / 'publication.json').exists() or (stage / 'publication-in-progress.json').exists():
        raise ValueError('Publication already attempted in this stage; use a fresh verified stage')
    receipt, files = checked_files(stage, live, prior_publication)
    _, _, stage_raw = _stage(stage)
    prior, prior_raw = _prior(prior_publication)
    ca_file = CA_FILE if ca_file is _DEFAULT_CA else ca_file
    context = ssl.create_default_context(cafile=str(ca_file) if ca_file is not None else None)
    transport = {'connect_host': connect_host, 'connect_port': connect_port}
    authenticated = (_DevelopmentVerificationClient(base_url, user, password_file, state_dir, context, transport)
                     if site == 'dev' else None)
    def served(name, timeout):
        if authenticated is not None:
            return authenticated.read_file(name, timeout)
        return _served(base_url + name, context, timeout, **transport)
    if digest(served('marble_studio.js', 5)) != receipt['studio']['source_sha256']:
        raise ValueError('Serving a different Studio build; no live writes made')
    before = _snapshot(live)
    expected = {name: before.get(name) for name in files}
    order = sorted(files, key=lambda name: (name == 'marble_studio_layer.html', name))
    changes = [name for name in order if expected[name] != digest(files[name])]
    backups = {}
    installed = []

    def guard():
        if authenticated is not None:
            authenticated.assert_target()
        current, bodies, raw = _stage(stage)
        if raw != stage_raw or bodies != files:
            raise ValueError('Stage changed after publication preflight')
        _sources(stage, live, current, bodies)
        if _prior(prior_publication)[1] != prior_raw:
            raise ValueError('Prior publication changed after preflight')
        for name, (backup, sha) in backups.items():
            if _hash(_path(backup.parent, backup.name)) != sha:
                raise ValueError('Backup hash changed: ' + name)
        for name in files:
            _assert_hash(_path(live, name), digest(files[name]) if name in installed else expected[name])

    guard()
    for name in changes:
        if expected[name] is not None:
            guard()
            body = _path(live, name).read_bytes()
            if digest(body) != expected[name] or prior is None or expected[name] != prior['files'].get(name):
                raise ValueError('Installed destination differs from prior publication: ' + name)
            backup = _path(stage, 'publication-backup/' + name)
            _durable(backup, body)
            backups[name] = (backup, expected[name])
    journal = {'schema': 'studio-game-layer-publication-v2', 'published': False,
               'stage_receipt_sha256': digest(stage_raw), 'files': receipt['files'],
               'verification_base_url': base_url, 'verification_connection': transport,
               'verification_site': site,
               'verification_instance': 'meowdoku-dev' if authenticated is not None else None,
               'before': expected, 'planned': changes, 'installed': [],
               'backups': {n: {'path': p.relative_to(stage).as_posix(), 'sha256': sha}
                           for n, (p, sha) in backups.items()}}
    _durable(stage / 'publication-in-progress.json', json.dumps(journal, indent=2).encode('utf-8'))
    try:
        for name in changes:
            guard()
            target = _path(live, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = _temporary(target, files[name])
            try:
                guard()
                _install(temporary, target, expected[name])
                installed.append(name)
                _sync_directory(target.parent)
                journal['installed'] = list(installed)
                _record(stage / 'publication-in-progress.json', journal)
            finally:
                temporary.unlink(missing_ok=True)
        guard()
        verified = []
        for name in order:
            if digest(served(name, 15)) != receipt['files'][name]:
                raise ValueError('Served hash differs: ' + name)
            verified.append(name)
        guard()
        unrelated = {name: sha for name, sha in before.items() if name not in installed}
        changed = [name for name, sha in unrelated.items() if _hash(_path(live, name)) != sha]
        if changed:
            raise ValueError('Pre-existing files changed during publication: ' + ', '.join(changed))
        result = {**journal, 'published': True, 'installed': installed, 'https_verified': verified,
                  'unchanged_preexisting_files': len(unrelated),
                  'replaced': list(backups), 'server_restarted': False,
                  'prior_publication_sha256': digest(prior_raw) if prior_raw is not None else None,
                  'original_game_sha256': receipt['original_game_sha256'],
                  'multi_file_atomic': False}
        _record(stage / 'publication.json', result)
    except BaseException as error:
        rollback = _rollback(live, installed, files, backups)
        result = {**journal, 'published': False, 'installed': installed, 'error': str(error),
                  'rollback': rollback, 'server_restarted': False, 'automatic_retry': False}
        try:
            _record(stage / 'publication.json', result)
        except OSError as receipt_error:
            result['receipt_error'] = str(receipt_error)
            error.add_note('Failure receipt could not be saved: ' + str(receipt_error))
        print(json.dumps(result), flush=True)
        raise
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--live', type=Path, default=LIVE)
    parser.add_argument('--base-url', default=BASE_URL)
    parser.add_argument('--prior-publication', type=Path)
    parser.add_argument('--site', choices=('stable', 'dev'), default='stable')
    parser.add_argument('--user', help='Existing operator account for Development verification')
    parser.add_argument('--password-file', help='Existing secret file; never copied to receipts')
    parser.add_argument('--state-dir', help='Existing state root for the operator secret helper')
    parser.add_argument('--connect-host', help='TCP destination only; HTTPS origin/SNI/Host remain the base URL')
    parser.add_argument('--connect-port', type=int, help='TCP destination port (defaults to the base URL port)')
    trust = parser.add_mutually_exclusive_group()
    trust.add_argument('--ca-file', type=Path, default=CA_FILE)
    trust.add_argument('--system-trust', action='store_true')
    args = parser.parse_args()
    publish(stage=args.stage, live=args.live, base_url=args.base_url,
            prior_publication=args.prior_publication,
            ca_file=None if args.system_trust else args.ca_file,
            connect_host=args.connect_host, connect_port=args.connect_port,
            site=args.site, user=args.user, password_file=args.password_file, state_dir=args.state_dir)

"""Publish the explicitly reviewed additive layer; never start a server.

The primary Studio and original game files are not destinations. The entry page
is installed last, after all dependencies. A running trusted HTTPS service and
unchanged source/stage hashes are required before the first live write.
"""
from pathlib import Path
import hashlib
import json
import os
import ssl
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'build/dot-studio-sync-20261110'
LIVE = ROOT / 'build/meowdoku/web'
BASE_URL = 'https://127.0.0.1:8766/'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_files(stage, live):
    receipt = json.loads((stage / 'stage-receipt.json').read_text())
    names = receipt['files']
    allowed_entries = {'marble_studio_layer.html', 'marble_studio_layer_main.js',
                       'mechanical_creature_studio_layer.html'}
    if not allowed_entries.issubset(names):
        raise ValueError('The complete additive layer is required')
    if digest((live / 'marble_studio.js').read_bytes()) != receipt['studio']['source_sha256']:
        raise ValueError('Published Studio changed; rebase and verify the candidate first')
    main_html = (live / 'marble_studio.html').read_bytes()
    candidate_html = (stage / 'marble_studio_layer.html').read_bytes()
    expected = main_html.replace(b'src="marble_studio.js"', b'src="./marble_studio_layer_main.js"')
    if expected == main_html:
        expected = main_html.replace(b'src="./marble_studio.js"', b'src="./marble_studio_layer_main.js"')
    if expected != candidate_html:
        raise ValueError('Published Studio HTML changed; rebase and verify first')
    files = {}
    for name, sha in names.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:
            raise ValueError('Unsafe additive path')
        if name not in allowed_entries and not name.startswith('studio-game-layer/'):
            raise ValueError('Only the private layer dependency folder is publishable')
        body = (stage / path).read_bytes()
        if digest(body) != sha:
            raise ValueError('Staged bytes changed: ' + name)
        destination = live / path
        if destination.exists() and destination.read_bytes() != body:
            raise ValueError('Existing destination differs; no overwrite authorized by this additive publisher: ' + name)
        files[name] = body
    return receipt, files


def publish():
    receipt, files = checked_files(STAGE, LIVE)
    context = ssl.create_default_context(cafile=str(ROOT / 'build/meowdoku/tls/ca.crt'))
    # If the DLL worker stopped the server this raises before any live writes.
    with urlopen(BASE_URL + 'marble_studio.js', context=context, timeout=5) as response:
        served = response.read()
    if digest(served) != receipt['studio']['source_sha256']:
        raise ValueError('Serving a different Studio build; no live writes made')
    before = {p.relative_to(LIVE).as_posix(): digest(p.read_bytes())
              for p in LIVE.rglob('*') if p.is_file()}
    # Recheck the small source guard after the snapshot, before committing.
    checked_files(STAGE, LIVE)
    order = sorted(files, key=lambda name: (name == 'marble_studio_layer.html', name))
    installed = []
    try:
        for name in order:
            target = LIVE / name
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.studio-layer.tmp')
            if temporary.exists():
                raise ValueError('Unexpected temporary file: ' + str(temporary))
            temporary.write_bytes(files[name])
            os.replace(temporary, target)
            installed.append(name)
        verified = []
        for name in order:
            with urlopen(BASE_URL + name, context=context, timeout=15) as response:
                if digest(response.read()) != receipt['files'][name]:
                    raise ValueError('Served hash differs: ' + name)
            verified.append(name)
        changed = [name for name, sha in before.items() if digest((LIVE / name).read_bytes()) != sha]
        if changed:
            raise ValueError('Pre-existing files changed during publication: ' + ', '.join(changed))
        result = {'published': True, 'installed': installed, 'https_verified': verified,
                  'unchanged_preexisting_files': len(before), 'files': receipt['files'],
                  'server_restarted': False, 'original_game_sha256': receipt['original_game_sha256']}
    except Exception as error:
        result = {'published': False, 'installed': installed, 'error': str(error),
                  'server_restarted': False, 'automatic_retry': False}
        (STAGE / 'publication.json').write_text(json.dumps(result, indent=2))
        raise
    (STAGE / 'publication.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    publish()

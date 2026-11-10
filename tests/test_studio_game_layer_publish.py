from pathlib import Path
import json
import pytest
from tools import publish_studio_game_layer as publisher


def fixture(tmp_path):
    stage, live = tmp_path / 'stage', tmp_path / 'live'
    stage.mkdir(); live.mkdir()
    main = b'const original = true;'
    html = b'<script src="marble_studio.js"></script>'
    (live / 'marble_studio.js').write_bytes(main)
    (live / 'marble_studio.html').write_bytes(html)
    files = {'marble_studio_layer.html': html.replace(b'marble_studio.js', b'./marble_studio_layer_main.js'),
             'marble_studio_layer_main.js': b'original plus opt-in',
             'mechanical_creature_studio_layer.html': b'complete original game plus bridge',
             'studio-game-layer/module.js': b'export const optional=true;'}
    for name, data in files.items():
        p = stage / name; p.parent.mkdir(exist_ok=True); p.write_bytes(data)
    receipt = {'files': {n: publisher.digest(b) for n, b in files.items()},
               'studio': {'source_sha256': publisher.digest(main)}}
    (stage / 'stage-receipt.json').write_text(json.dumps(receipt))
    return stage, live, receipt


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
    before = {p.name: p.read_bytes() for p in live.iterdir()}
    monkeypatch.setattr(publisher, 'STAGE', stage)
    monkeypatch.setattr(publisher, 'LIVE', live)
    monkeypatch.setattr(publisher.ssl, 'create_default_context', lambda **kwargs: object())
    def unavailable(*args, **kwargs):
        raise ConnectionRefusedError('server stopped for DLL access')
    monkeypatch.setattr(publisher, 'urlopen', unavailable)
    with pytest.raises(ConnectionRefusedError):
        publisher.publish()
    assert before == {p.name: p.read_bytes() for p in live.iterdir()}
    assert not (stage / 'publication.json').exists()


def test_staged_hash_and_html_are_bound(tmp_path):
    stage, live, _ = fixture(tmp_path)
    (stage / 'studio-game-layer/module.js').write_bytes(b'changed after QA')
    with pytest.raises(ValueError, match='Staged bytes changed'):
        publisher.checked_files(stage, live)
    (live / 'marble_studio.html').write_bytes(b'changed published HTML')
    with pytest.raises(ValueError, match='Published Studio HTML changed'):
        publisher.checked_files(stage, live)

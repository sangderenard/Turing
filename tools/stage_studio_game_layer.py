"""Stage only the optional Studio/original-game integration, never publish it."""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.add_studio_game_layer import patch_studio_game_layer
from tools.add_mechanical_creature_appearance import patch_playable_page, behavior_bounds, NEWER_SHA256
from tools.add_mechanical_creature_studio_building import patch_studio_building
from tools.add_mechanical_creature_studio_bridge import patch_studio_bridge


def stage(destination: Path):
    live = ROOT / 'build/meowdoku/web'
    source = ROOT / 'examples/meowdoku/web'
    game = ROOT / 'build/dot-newer-tire-dynamics-20261110/index.html'
    original = game.read_bytes()
    if hashlib.sha256(original).hexdigest() != NEWER_SHA256:
        raise ValueError('Stage must be explicitly rebased on the reviewed newer game candidate')
    definition = json.loads((ROOT / 'build/dot-studio-game-transfer-20261110/studio-building-definition-v2.json').read_text())
    data, appearance = patch_playable_page(original)
    data, building = patch_studio_building(data, definition)
    data, bridge = patch_studio_bridge(data, expected_source_sha256=hashlib.sha256(data).hexdigest())
    html = data.decode('utf-8')
    if html.count('</body>') != 1:
        raise ValueError('Unique game module attachment seam is required')
    html = html.replace('</body>', '<script type="module" src="./studio-game-layer/mechanical_creature_studio_layer_host.js"></script>\n</body>')
    files = {'mechanical_creature_studio_layer.html': html.encode('utf-8')}
    main, main_receipt = patch_studio_game_layer((live / 'marble_studio.js').read_bytes(), './studio-game-layer/')
    files['marble_studio_layer_main.js'] = main
    studio_html = (live / 'marble_studio.html').read_bytes()
    old = b'src="marble_studio.js"'
    if studio_html.count(old) != 1:
        old = b'src="./marble_studio.js"'
    if studio_html.count(old) != 1:
        raise ValueError('Unique Studio bootstrap script is required')
    files['marble_studio_layer.html'] = studio_html.replace(old, b'src="./marble_studio_layer_main.js"')
    pending = ['marble_studio_game_layer.js', 'marble_studio_world_packet.js', 'mechanical_creature_studio_layer_host.js']
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        body = (source / name).read_bytes()
        files['studio-game-layer/' + name] = body
        for dependency in re.findall(r"(?:from\s*|import\s*)['\"]\./([^'\"]+)['\"]", body.decode('utf-8')):
            if '/' in dependency or '..' in dependency:
                raise ValueError('Unexpected dependency outside bounded module folder')
            pending.append(dependency)
    destination.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    start, end = behavior_bounds(html)
    (destination / 'candidate-game-behavior.js').write_text(html[start:end], encoding='utf-8')
    receipt = {'schema':'studio-game-layer-stage-v1','published':False,'original_game_sha256':NEWER_SHA256,
               'original_game_untouched':hashlib.sha256(game.read_bytes()).hexdigest()==NEWER_SHA256,
               'appearance':appearance,'building':building,'bridge':bridge,'studio':main_receipt,
               'files':{name:hashlib.sha256(body).hexdigest() for name,body in files.items()}}
    (destination / 'stage-receipt.json').write_text(json.dumps(receipt,indent=2), encoding='utf-8')
    print(json.dumps({'staged':str(destination),'files':len(files),'newer_base':NEWER_SHA256,'published':False}))


if __name__ == '__main__':
    stage(ROOT / 'build/dot-studio-sync-20261110')

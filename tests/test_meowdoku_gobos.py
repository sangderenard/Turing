"""The maximum Advanced gobo scene passes the actual MRP1 decoder unchanged."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples'))
from meowdoku import marble_photo as photos


def test_advanced_gobos_survive_worker_decode_and_scene_limits():
    result = subprocess.run(
        ['node', 'tests/meowdoku_marble_gobo_scene_check.mjs', '--json'],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=15)
    payload = json.loads(result.stdout)
    proof = np.zeros(1, dtype=[('positions', '<f4', (9,)),
                              ('normals', '<f4', (9,)), ('id', '<u4')])
    proof['positions'] = [0, 0, 0, 1, 0, 0, 0, 1, 0]
    proof['normals'] = [0, 0, 1] * 3
    saved = {'digest': 'a' * 64,
             'geometry_sha256': hashlib.sha256(proof.tobytes()).hexdigest(),
             'physical': {'radius-mm': 10},
             'materials': {'tables': {'pbr': {'stride': 16, 'data': [0] * 16}}}}
    *_, config, _digest = photos.decode_upload(
        base64.b64decode(payload['wire']), SimpleNamespace(get=lambda _: saved))
    assert config['fixtures'] == payload['scene']
    assert payload['counts']['gobos'] == 1536
    assert len(config['fixtures']['material_ids']) == sum(payload['counts'].values())
    assert len(config['fixtures']['material_ids']) <= 5000
    assert len(config['fixtures']['pbr']) // 16 <= 64
    assert config['scale_m'] == .01

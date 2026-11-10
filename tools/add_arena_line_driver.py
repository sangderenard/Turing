"""Guarded reversible adapter for original game controls and route paint only."""
from pathlib import Path
import hashlib
ROOT = Path(__file__).resolve().parents[1]
ANCHOR = 'function updateActiveVehicle(dt,throttle,steering,brake) {'
GUARD = 'function updateViewportControls(dt) {'
CONTROL = '    const throttle=mobileClamp(Number(controls.throttle||0)*(running?1.18:1));'
PAINT = 'function buildNavigationRouteMesh() {\n  const vertices = [], color = colorVector(model.appearance.colors.active);'


def patch_arena_line_driver(source: bytes, *, expected_source_sha256: str):
    digest = hashlib.sha256(source).hexdigest()
    if digest != expected_source_sha256:
        raise ValueError('Arena driver source hash mismatch')
    document = source.decode('utf-8')
    if 'const arenaLineRuntime=' in document:
        raise ValueError('Arena driver already installed')
    helper = (ROOT / 'examples/meowdoku/web/arena_line_driver_host.js').read_text(encoding='utf-8-sig')
    if '</script' in helper.lower():
        raise ValueError('Unsafe script terminator')
    newline = '\r\n' if '\r\n' in document else '\n'
    paint = PAINT.replace('\n', newline)
    changes = [(ANCHOR, helper+newline+ANCHOR),
               (GUARD, GUARD+newline+'  arenaLineGuardFocus();'),
               (CONTROL, '    Object.assign(controls,arenaLineControls(controls));'+newline+CONTROL),
               (paint, paint+newline+'  for(const value of arenaLinePaintVertices())vertices.push(value);')]
    for before, after in changes:
        if document.count(before) != 1:
            raise ValueError('Original arena driver seam is not unique: '+before[:70])
    output = document
    for before, after in changes:
        output = output.replace(before, after, 1)
    restored = output
    for before, after in reversed(changes):
        restored = restored.replace(after, before, 1)
    if restored != document:
        raise AssertionError('Arena adapter touched unrelated game bytes')
    data = output.encode('utf-8')
    return data, {'schema':'arena-line-driver-patch-v1','source_sha256':digest,
                  'output_sha256':hashlib.sha256(data).hexdigest(),'source_recovered_exactly':True,
                  'physics':'original-vehicle-control-worker','clock':'original-game-frame',
                  'pose_writes':False,'starter_policy':'uncalibrated','published':False}

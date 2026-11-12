"""Add an optional explicit destination to the original player respawn, reversibly."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'examples/meowdoku/web/mechanical_creature_studio_entry.js'
SIGNATURE = 'function respawnViewportActor(){'
DESTINATION = 'spawn=actor?.entity?.pose?.position||actor?.record?.pose?.position;'
ANCHOR = 'function runEntityCycle(now) {'
QUALIFICATION = 'const vehicleQualification = {'
QUALIFIED_LAUNCH = '  registerPlayerPhysicsBody();requestAnimationFrame(runEntityCycle);'
ENGINE_STATE = '          stateLoopRuntime.engineSleepReason=message.reason||null;'
QUALIFICATION_FINISH = '    if(failures)failures.textContent=[...q.failures].join(" · ");}\n}'


def patch_studio_game_entry(source: bytes, *, expected_source_sha256: str) -> tuple[bytes, dict]:
    digest = hashlib.sha256(source).hexdigest()
    if digest != expected_source_sha256.lower():
        raise ValueError('Unverified original game or derived candidate for Studio entry')
    document = source.decode('utf-8')
    if 'mechanicalCreatureStudioEntryReceipt' in document:
        raise ValueError('Studio player entry is already installed')
    if 'const mechanicalCreatureStudioBridgeState' not in document:
        raise ValueError('Studio player entry requires the original runtime bridge')
    helper = HELPER.read_bytes().decode('utf-8')
    if '</script' in helper.lower():
        raise ValueError('Unsafe helper script terminator')
    changes = [(SIGNATURE, 'function respawnViewportActor(studioSpawn=null){'),
               (DESTINATION, 'spawn=studioSpawn||mechanicalCreatureStudioRespawnPosition()||actor?.entity?.pose?.position||actor?.record?.pose?.position;'),
               (ANCHOR, helper + '\n' + ANCHOR)]
    has_qualification = QUALIFICATION in document
    if has_qualification:
        nl = '\r\n' if '\r\n' in document else '\n'
        finish = QUALIFICATION_FINISH.replace('\n', nl)
        changes.extend([(QUALIFIED_LAUNCH, QUALIFIED_LAUNCH + nl + '  notifyMechanicalCreatureStudioQualification();'),
            (finish, finish[:-1] + '  notifyMechanicalCreatureStudioQualification();' + nl + '}'),
            (ENGINE_STATE, ENGINE_STATE + nl + '          wakeMechanicalCreatureStudioQualification(message);')])
    for before, _ in changes:
        if document.count(before) != 1:
            raise ValueError('Original player entry seam is not unique')
    output = document
    for before, after in changes:
        output = output.replace(before, after, 1)
    restored = output
    for before, after in reversed(changes):
        restored = restored.replace(after, before, 1)
    if restored != document:
        raise AssertionError('Player entry changed unrelated original game bytes')
    encoded = output.encode('utf-8')
    return encoded, {'schema':'mechanical-creature-studio-entry-patch-v1',
        'source_sha256':digest,'output_sha256':hashlib.sha256(encoded).hexdigest(),
        'source_recovered_exactly':True,'entry':'outside-studio-entrance',
        'player':'original-native-respawn','vehicle':'original-parked-body',
        'new_controller':False,'new_clock':False,'published':False}

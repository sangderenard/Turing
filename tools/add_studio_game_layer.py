"""Add the opt-in original-game layer to a current Studio module, reversibly.

Only three guarded seams change. The caller stages/publishes explicitly; this
tool never exports dirty Studio sources or changes existing live assets.
"""
from __future__ import annotations
import hashlib


def patch_studio_game_layer(source: bytes, prefix: str = './') -> tuple[bytes, dict]:
    document = source.decode('utf-8')
    if 'mountStudioGameLayer' in document:
        raise ValueError('Studio game layer is already installed')
    if not prefix.startswith('./') or any(c in prefix for c in ('"', "'", '\n', '<')):
        raise ValueError('A safe relative module prefix is required')
    nl = '\r\n' if '\r\n' in document else '\n'
    imports = (f'import {{mountStudioGameLayer}} from "{prefix}marble_studio_game_layer.js";{nl}'
               f'import {{captureStudioProgramState,studioWorldPacket}} from "{prefix}marble_studio_world_packet.js";{nl}'
               f'let studioGameLayer=null;{nl}')
    frame_anchor = '    if(dofActive)try {'
    frame_hook = ('    if(!simOn)studioGameLayer?.frame({revision:studioSceneRevision,gl,view,phase:"transparent",'
                  'projection:perspective(w/h,viewFovDegrees,viewFar,viewNear)});' + nl)
    opaque_anchor = '      gl.bindVertexArray(glassOpaque.vao);gl.drawArrays(gl.TRIANGLES,0,glassOpaque.count);'
    opaque_hook = (nl + '      if(!simOn)studioGameLayer?.draw({gl,view,phase:"opaque",'
                   'projection:perspective(w/h,viewFovDegrees,viewFar,viewNear)});')
    boot_anchor = '    frame();'
    boot_hook = ('    studioGameLayer=mountStudioGameLayer({doc:document,canvas,capture:()=>{' + nl
                 + '      const capture=captureCurrentStudioScene(),render=captureStudioProgramState(gl,P,tables);' + nl
                 + "      return studioWorldPacket(capture,render,{classIdentity:'python:camera_software.exposure_timing.SceneCameraCoordinator'});" + nl
                 + '    }});' + nl)
    if document.count(frame_anchor) != 1 or document.count(boot_anchor) != 1 or document.count(opaque_anchor) != 1:
        raise ValueError('The current Studio display/boot seam is not unique')
    result = imports + document.replace(frame_anchor, frame_hook + frame_anchor, 1).replace(boot_anchor, boot_hook + boot_anchor, 1).replace(opaque_anchor, opaque_anchor + opaque_hook, 1)
    recovered = result.removeprefix(imports).replace(frame_hook + frame_anchor, frame_anchor, 1).replace(boot_hook + boot_anchor, boot_anchor, 1).replace(opaque_anchor + opaque_hook, opaque_anchor, 1)
    if recovered != document:
        raise AssertionError('Studio source could not be recovered exactly')
    output = result.encode('utf-8')
    return output, {'schema': 'studio-original-game-layer-patch-v1',
                    'source_sha256': hashlib.sha256(source).hexdigest(),
                    'output_sha256': hashlib.sha256(output).hexdigest(),
                    'original_recovered_exactly': True, 'enabled_by_default': False,
                    'photo_camera_changed': False, 'physics_replaced': False}

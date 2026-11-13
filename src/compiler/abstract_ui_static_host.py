"""Stage static inspection of an exact compiled Living Data map, without physics.

The original model, shader and mesh functions remain byte-identical. Startup
calls for gameplay are omitted; requested cameras use the existing renderer.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / 'examples/meowdoku/web/living_data_static_host.js'


def patch_static_host(source: bytes, *, expected_source_sha256: str):
    digest = hashlib.sha256(source).hexdigest()
    if digest != expected_source_sha256:
        raise ValueError('Static host requires an exact source hash')
    document = source.decode('utf8')
    if 'livingDataStaticHost' in document:
        raise ValueError('Static host already installed')
    nl = '\r\n' if '\r\n' in document else '\n'
    helper = HELPER.read_text(encoding='utf-8-sig').replace('__STATIC_SOURCE_SHA256__', digest)
    if '</script' in helper.lower():
        raise ValueError('Unsafe script terminator')
    navigation = '''navigationRuntime.ready = initializeNavigationRuntime().catch(error => {
  navigationRuntime.error = error.message; console.error(error); throw error;
});'''
    startup = '''renderWorld();
renderEntityMezzanine();
renderFilesystemDistrict();
renderJavaScriptDistrict();
inspectNode(model, null);
initializeVehicleFirstExperience();
vehicleRuntime.electrical.ignitionOn=false;vehicleRuntime.electrical.starterEngaged=false;
initializeWorldPhysicsWasm();
armEngineSoundOnFirstGesture();'''
    replacements = [
      (navigation, 'navigationRuntime.ready = Promise.resolve(); // Static map: no navigation worker.'),
      (startup, 'renderWorld();\ninspectNode(model, null);\n' + helper),
      ('function updateShaderViewer() {', 'function updateShaderViewer(staticCamera=null) {'),
      ('const width = Math.max(1, Math.round(canvas.clientWidth * ratio));',
       'const width = staticCamera?.width ?? Math.max(1, Math.round(canvas.clientWidth * ratio));'),
      ('const height = Math.max(1, Math.round(canvas.clientHeight * ratio));',
       'const height = staticCamera?.height ?? Math.max(1, Math.round(canvas.clientHeight * ratio));'),
      ('  shaderViewer.cameraPosition = [...cameraPosition];\n  shaderViewer.cameraFacing = [...cameraFacing];',
       '  if(staticCamera){cameraPosition=[...staticCamera.position];cameraFacing=[...staticCamera.facing];}\n  shaderViewer.cameraPosition = [...cameraPosition];\n  shaderViewer.cameraFacing = [...cameraFacing];'),
      ('    gl.viewport(0, 0, width, height);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);',
       '    staticCamera?.beforeDraw?.();\n    gl.viewport(0, 0, width, height);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);'),
      ('    drawShaderCrosshair(gl, width, height);',
       '    if(!staticCamera)drawShaderCrosshair(gl, width, height);\n    staticCamera?.afterDraw?.();'),
      ('async function initializeWorldPhysicsWasm() {',
       'async function initializeWorldPhysicsWasm() {\n  throw Error("Physics is unavailable in the static map host");'),
      ('function runEntityCycle(now) {',
       'function runEntityCycle(now) {\n  throw Error("Game updates are unavailable in the static map host");'),
      ('  const valuesByMember = {pbr, phong, enamel, texstack, thermal_k:[0]};',
       '  const valuesByMember = {pbr, phong, enamel, texstack, thermal_k:[0]};\n'
       '  shaderViewer.staticMaterialState = {colors:colors.map(color=>Array.from(new Float32Array(color))),\n'
       '    materials:Object.fromEntries(Object.entries(valuesByMember).map(([key,value])=>[key,Array.from(new Float32Array(value))]))};'),
      ('  const portalTubes = buildPortalTubeMesh();',
       '  const portalTubes = buildPortalTubeMesh();\n'
       '  let staticOverlayVertex=shaderViewer.baseMesh.length/9;\n'
       '  shaderViewer.staticMeshOverlaySpans=[['
       '"navigation-routes",routes],["portal-tubes",portalTubes],["portal-splats",portalSplats]]\n'
       '    .map(([kind,mesh])=>{const span={kind,firstVertex:staticOverlayVertex,vertexCount:mesh.length/9};\n'
       '      staticOverlayVertex+=span.vertexCount;return span;}).filter(span=>span.vertexCount);'),
    ]
    replacements = [(a.replace('\n',nl),b.replace('\n',nl)) for a,b in replacements]
    output = document
    for before, after in replacements:
        if output.count(before) != 1:
            raise ValueError(f'Static host seam is not unique: {before[:85]}')
        output = output.replace(before, after, 1)
    restored=output
    for before,after in reversed(replacements):
        restored=restored.replace(after,before,1)
    if restored!=document:
        raise AssertionError('Static adapter changed unrelated source bytes')
    result=output.encode('utf8')
    return result, {'schema':'living-data-static-host-patch.v1','source_sha256':digest,
      'output_sha256':hashlib.sha256(result).hexdigest(),'source_recovered_exactly':True,
      'physics_started':False,'renderer':'original-pluck-phong','published':False}

// Static inspection adapter. Existing registry, mesh builder and Phong renderer
// remain authoritative; no worker, controller, game clock or physics is started.
// Static analytic inspection never waits behind the game's qualification screen.
document.body.classList.remove('qualification-pending');
document.getElementById('vehicle-qualification')?.remove();
const livingDataStaticReady = new Promise((resolve, reject) => {
  let attempts = 0;
  function check() {
    if (shaderViewer.gl && shaderViewer.program) {
      const choice = VIEWPORT_SHADER_CHOICES.find(c => c.identity === shaderViewer.shaderChoice);
      if (!choice || !choice.identity.endsWith('/pluck-phong')) {
        reject(Error('Existing Pluck Phong shader is required')); return;
      }
      resolve(true); return;
    }
    if (++attempts > 600) { reject(Error('Static Phong renderer did not initialize')); return; }
    requestAnimationFrame(check);
  }
  requestAnimationFrame(check);
});
const livingDataStaticSource = '__STATIC_SOURCE_SHA256__';
function livingDataStaticReceipt() {
  return {source_sha256: livingDataStaticSource,
    renderer_identity: shaderViewer.shaderChoice, mode:'static-map',
    physics_started: Boolean(stateLoopRuntime.worker),
    simulation_advanced:false};
}
function livingDataStaticNode(node) {
  const fields=['identity','kind','name','label','parent','position','dependencies',
    'member_kind','type_name','transform','form','capabilities','semantic_parts',
    'metaphor','parameters','implied_code','interaction','module','source_kind'];
  return Object.fromEntries(fields.filter(key=>node[key]!==undefined)
    .map(key=>[key,structuredClone(node[key])]));
}
let livingDataStaticPinned = null;
let livingDataStaticMeshPinned = null;
let livingDataStaticInvalidated = false;
function livingDataStaticCanonical() {
  const regions=model.regions.map(region=>({...livingDataStaticNode(region),
    buildings:region.buildings.map(building=>({...livingDataStaticNode(building),
      rooms:building.rooms.map(livingDataStaticNode)}))}));
  return {map:{schema:model.schema,...livingDataStaticNode(model),regions,
    tracks:structuredClone(model.tracks),world:{root:model.world.root,
      objects:[...turingWorld.objects.values()].map(object=>structuredClone(object))}},
    geometry:{extent:structuredClone(model.document_geometry.extent),
      origin:structuredClone(model.document_geometry.origin),
      boxes:shaderViewer.geometry.map(box=>structuredClone(box))},
    renderer_identity:shaderViewer.shaderChoice};
}
function livingDataStaticCheckPinned() {
  if(livingDataStaticInvalidated) throw Error('Static map changed after attachment; reload a new host');
  const canonical=livingDataStaticCanonical(), bytes=JSON.stringify(canonical);
  if(livingDataStaticPinned===null) livingDataStaticPinned=bytes;
  else if(livingDataStaticPinned!==bytes) {
    livingDataStaticInvalidated=true;
    throw Error('Static map changed after attachment; reload a new host');
  }
  return canonical;
}
globalThis.livingDataStaticHost = Object.freeze({
  ready: livingDataStaticReady,
  snapshot() {
    if(!shaderViewer.gl||!shaderViewer.program) throw Error('Static renderer is not ready');
    if(stateLoopRuntime.worker) throw Error('Static host refuses a physics worker');
    const canonical=livingDataStaticCheckPinned();
    return {...livingDataStaticReceipt(),...canonical};
  },
  meshSnapshot() {
    // Copy the actual uploaded triangle stream, never extrude the boxes again.
    // Static hosts do not silently omit separate dynamic/solver draw buffers.
    const snapshot=this.snapshot();
    if(!shaderViewer.mesh?.length||!shaderViewer.staticMaterialState)
      throw Error('Static realized mesh and material resources are not ready');
    if(shaderViewer.vehicleMesh?.length||shaderViewer.authoredWorldSurface?.mesh?.length)
      throw Error('Static mesh export refuses separate dynamic draw buffers');
    const result={schema:'living-data-realized-mesh.v1',...livingDataStaticReceipt(),
      world_identity:model.identity,coordinate_space:'world',up_axis:'Y',stride:9,
      vertices:Array.from(shaderViewer.mesh),
      material_colors:structuredClone(shaderViewer.staticMaterialState.colors),
      materials:structuredClone(shaderViewer.staticMaterialState.materials),
      identity_spans:structuredClone(shaderViewer.identitySpans),
      semantic_part_spans:structuredClone(shaderViewer.semanticPartSpans),
      overlays:structuredClone(shaderViewer.staticMeshOverlaySpans||[]),
      world:snapshot.map.world,geometry:snapshot.geometry};
    const bytes=JSON.stringify(result);
    if(livingDataStaticMeshPinned===null)livingDataStaticMeshPinned=bytes;
    else if(livingDataStaticMeshPinned!==bytes){livingDataStaticInvalidated=true;
      throw Error('Static realized mesh changed after attachment; reload a new host');}
    return result;
  },
  async render(request={}) {
    await livingDataStaticReady;
    if(stateLoopRuntime.worker) throw Error('Static host refuses a physics worker');
    const vector=value=>Array.isArray(value)&&value.length===3&&
      value.every(x=>typeof x==='number'&&Number.isFinite(x)&&Math.abs(x)<=1e6);
    const position=request.position, facing=request.facing;
    const width=request.width??512,height=request.height??384;
    if(!vector(position)||!vector(facing)||Math.hypot(...facing)<1e-9||
      !Number.isInteger(width)||!Number.isInteger(height)||width<16||height<16||width>1024||height>1024)
      throw TypeError('Finite camera vectors and 16..1024 pixel dimensions required');
    const camera={position:[...position],facing:normalized3(facing),width,height};
    livingDataStaticCheckPinned();
    // Only renderer-local camera variables change. Never set an actor pose.
    let focus=null;
    if(request.optics){
      const {renderLivingDataFocus}=await import('./living_data_focus.js');
      focus=await renderLivingDataFocus({gl:shaderViewer.gl,canvas:shaderViewer.canvas,
        mesh:this.meshSnapshot(),camera:{...camera,optics:request.optics},check:livingDataStaticCheckPinned,
        draw:(beforeDraw,afterDraw)=>updateShaderViewer({...camera,beforeDraw,afterDraw})});
    }else updateShaderViewer(camera);
    livingDataStaticCheckPinned();
    const gl=shaderViewer.gl, pixels=new Uint8Array(width*height*4);
    gl.finish();gl.readPixels(0,0,width,height,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
    if(gl.getError()!==gl.NO_ERROR) throw Error('Native Phong framebuffer readback failed');
    const output=document.createElement('canvas');output.width=width;output.height=height;
    const context=output.getContext('2d'),frame=context.createImageData(width,height);
    for(let y=0;y<height;y++)frame.data.set(pixels.subarray((height-1-y)*width*4,(height-y)*width*4),y*width*4);
    context.putImageData(frame,0,0);
    livingDataStaticCheckPinned();
    return {...livingDataStaticReceipt(),camera:{...camera,...(focus?{optics:focus}:{})},width,height,
      data_url:output.toDataURL('image/png')};
  }
});

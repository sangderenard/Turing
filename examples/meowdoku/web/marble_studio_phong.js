// Hidden WebGL2 staging with the incumbent Studio mesh upload functions.
// No draw, active-scene replacement, camera change or physics step occurs.
import {phongInterior} from "./marble_bubbles.js";
import {isOpaquePreviewMaterial,transparentTriangleOrder} from "./marble_glass_preview.js";
import {sameStudioTransitionState} from "./marble_studio_transition.js";

export function studioPhongFaces(capture,dynamic) {
  const state=capture?.state,subject=state?.subject,faces=dynamic?.subject?.faces,layout=subject?.phong;
  if(state?.channel!=="studio"||dynamic?.schema!=="turing.studio-dynamic-geometry.v1"||
      dynamic.revision!==capture.revision||!Number.isSafeInteger(dynamic.step)||dynamic.step<1||
      !Array.isArray(faces)||!Number.isSafeInteger(layout?.interiorCount)||layout.interiorCount<0||
      !Number.isSafeInteger(layout?.glassCount)||layout.glassCount<0||faces.length!==layout.interiorCount+layout.glassCount)
    throw new Error("Phong staging needs the same paused native geometry and authored subject partition");
  if(dynamic.subject.id!==subject.id||!sameStudioTransitionState(dynamic.staticRay,state.ray)||
      !sameStudioTransitionState(dynamic.camera,state.camera)||
      faces.some((face,i)=>face[3]!==subject.faces[i]?.[3]))
    throw new Error("Phong staging cannot substitute materials, static geometry, camera, or subject identity");
  const pbr=state.ray.pbr,opaque=face=>isOpaquePreviewMaterial(pbr[face[3]*16+5],pbr[face[3]*16+7]);
  const insideOpaque=[],outsideOpaque=[],transparent=[];
  for(const face of phongInterior({interior:faces.slice(0,layout.interiorCount),bubbles:layout.bubbles}))
    (opaque(face)?insideOpaque:transparent).push(face);
  for(const face of faces.slice(layout.interiorCount))(opaque(face)?outsideOpaque:transparent).push(face);
  // Subject and furniture glass need one world-space sort, as in Studio's
  // existing transparentSet. The static opaque furniture remains untouched.
  for(const part of state.parts)if(part.id.startsWith("table/")||part.id.startsWith("gobo/"))
    for(const face of part.faces)if(!opaque(face))transparent.push(face);
  return {insideOpaque,outsideOpaque,transparent};
}

function waitFence(gl,fence,{timeoutMs=10000}={}) {
  const started=performance.now();
  return new Promise((resolve,reject)=>{
    const poll=()=>{
      if(gl.isContextLost())return reject(new Error("Phong context was lost while staging"));
      const status=gl.clientWaitSync(fence,0,0);
      if(status===gl.ALREADY_SIGNALED||status===gl.CONDITION_SATISFIED)return resolve();
      if(status===gl.WAIT_FAILED)return reject(new Error("Phong staging GPU fence failed"));
      if(performance.now()-started>timeoutMs)return reject(new Error("Phong staging GPU fence timed out"));
      setTimeout(poll,4);
    };poll();
  });
}

export async function stageStudioPhong({gl,mesh,transparentMesh,dispose,capture,dynamic,signal}) {
  if(!gl||gl.isContextLost())throw new Error("A live Studio WebGL2 context is required");
  for(const fn of [mesh,transparentMesh,dispose])if(typeof fn!=="function")throw new TypeError("Existing Studio mesh functions are required");
  if(signal?.aborted)throw new Error("Phong staging was canceled");
  const plan=studioPhongFaces(capture,dynamic),allocated=[];let fence=null,released=false;
  const bindings=()=>({vao:gl.getParameter(gl.VERTEX_ARRAY_BINDING),array:gl.getParameter(gl.ARRAY_BUFFER_BINDING)});
  const restore=state=>{gl.bindVertexArray(state.vao);gl.bindBuffer(gl.ARRAY_BUFFER,state.array);};
  const release=()=>{if(released)return;released=true;for(const resource of allocated)dispose(resource);};
  const prior=bindings();
  let bundle;
  try {
    try {
      const upload=(faces,transparent)=>{const result=(transparent?transparentMesh:mesh)(faces);allocated.push(result);
        if(!result?.vao||result.buffers?.some(buffer=>!buffer)||result.count!==faces.length*3)
          throw new Error("Phong mesh allocation was incomplete");return result;};
      bundle={insideOpaque:upload(plan.insideOpaque,false),outsideOpaque:upload(plan.outsideOpaque,false),
        transparent:upload(plan.transparent,true)};
      const view=capture.state.camera.view;
      if(!view||view.length!==16||!Array.from(view).every(Number.isFinite))throw new Error("Phong staging needs its exact displayed view matrix");
      gl.bindVertexArray(bundle.transparent.vao);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,bundle.transparent.indexBuffer);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,transparentTriangleOrder(bundle.transparent.centres,view),gl.DYNAMIC_DRAW);
      bundle.transparent.sortAxis=[view[2],view[6],view[10]];
      fence=gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE,0);
      if(!fence)throw new Error("Phong staging could not create its completion fence");
      gl.flush();
    } finally {restore(prior);}
    // Aborting does not delete in-flight buffers. Finish this bounded upload
    // fence, then discard the unshown candidate instead of revealing it.
    await waitFence(gl,fence);
    const beforeRead=bindings();
    try {
      // Read every uploaded attribute back, using the existing mesh layout.
      // This is a one-time staging check, not a per-frame GPU readback loop.
      for(const [key,resource]of Object.entries(bundle)) {
        const faces=plan[key],position=[],normal=[],material=[],group=[];
        for(const [a,b,c,id,ns,gid=0]of faces)for(const [i,p]of [a,b,c].entries()) {
          position.push(...p);normal.push(...ns[i]);material.push(id);group.push(gid);
        }
        const expected=[new Float32Array(position),new Float32Array(normal),new Int32Array(material),new Int32Array(group)];
        for(let i=0;i<4;i++) {
          gl.bindBuffer(gl.ARRAY_BUFFER,resource.buffers[i]);
          if(gl.getBufferParameter(gl.ARRAY_BUFFER,gl.BUFFER_SIZE)!==expected[i].byteLength)
            throw new Error("Phong staging buffer length differs from its exact authored faces");
          const read=new expected[i].constructor(expected[i].length);
          if(read.length)gl.getBufferSubData(gl.ARRAY_BUFFER,0,read);
          if(!read.every((v,k)=>Object.is(v,expected[i][k])))throw new Error("Phong staging GPU attributes differ from the source snapshot");
        }
        if(key==="transparent") {
          gl.bindVertexArray(resource.vao);
          const expected=transparentTriangleOrder(resource.centres,capture.state.camera.view),read=new Uint32Array(expected.length);
          if(gl.getBufferParameter(gl.ELEMENT_ARRAY_BUFFER,gl.BUFFER_SIZE)!==expected.byteLength)
            throw new Error("Phong staging transparency indices are incomplete");
          if(read.length)gl.getBufferSubData(gl.ELEMENT_ARRAY_BUFFER,0,read);
          if(!read.every((v,k)=>v===expected[k]))throw new Error("Phong staging transparency order differs from the displayed camera");
        }
      }
    } finally {restore(beforeRead);}
    if(gl.isContextLost())throw new Error("Phong context was lost before staging completed");
    if(signal?.aborted)throw new Error("Phong staging was canceled");
    return {get ready(){return !released&&!gl.isContextLost();},revision:capture.revision,step:dynamic.step,bundle,release,
      geometry:dynamic,coordinateSpace:"studio-world",visible:false,usesCurrentMaterialTables:true};
  } catch(error) {release();throw error;}
  finally {if(fence)gl.deleteSync(fence);}
}

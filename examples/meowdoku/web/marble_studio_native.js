// Connect the existing compiled-scene receipt to the existing hidden runtime.
// No scheduler, renderer, active-scene mutation, fetch endpoint or UI lives here.
import {sameStudioTransitionState} from "./marble_studio_transition.js";
import {studioCapturedPhysicsFrame,STUDIO_NATIVE_STATE} from "./marble_studio_physics.js";
import {studioDisplayedStart,studioDynamicGeometry} from "./marble_studio_dynamic.js";
import {validateStudioRigidBinding} from "./marble_studio_checkpoint.js";

const encoder=new TextEncoder(),digestPattern=/^[0-9a-f]{64}$/;
const same=(a,b)=>sameStudioTransitionState(a,b);
const abort=signal=>{if(signal?.aborted)throw new Error("Hidden Studio preparation was canceled");};
const sha=async bytes=>[...new Uint8Array(await crypto.subtle.digest("SHA-256",bytes))]
  .map(x=>x.toString(16).padStart(2,"0")).join("");
const filename=value=>typeof value==="string"&&/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(value)&&!value.includes("..");

// The server's raw bytes carry its canonical number spelling. Hash those
// exact bytes, then compare parsed captures structurally; never recreate a
// Python JSON digest using JavaScript's different float serialization.
export async function admitStudioCompiledCandidate(capture,compiled,{signal}={}) {
  const saved=structuredClone(capture),bundle=structuredClone(compiled),receipt=bundle?.receipt;
  abort(signal);
  if(receipt?.schema!=="turing.studio-compiled-candidate.v1"||receipt.stage!=="compiled-not-executed"||
      receipt.sourceRevision!==saved.revision||!digestPattern.test(receipt.captureDigest)||!digestPattern.test(receipt.meshDigest)||
      receipt.nativeStepRequired!==true||receipt.rendererStagingRequired!==true||receipt.revisionCheckRequired!==true)
    throw new Error("A complete unexecuted Studio product receipt is required");
  const parsed={};let total=0;
  for(const key of ["capture","rigidScene","wholeMesh","manifest"]) {
    const name=receipt[key],record=receipt.files?.[name],text=bundle.files?.[name];
    if(!filename(name)||typeof text!=="string"||!digestPattern.test(record?.sha256)||
        !Number.isSafeInteger(record.bytes)||record.bytes<1||record.bytes>64*1024*1024)
      throw new Error(`Missing bounded ${key} artifact receipt`);
    const bytes=encoder.encode(text);total+=bytes.length;
    if(total>96*1024*1024||bytes.length!==record.bytes||await sha(bytes)!==record.sha256)
      throw new Error(`${key} artifact bytes differ from the compiled scene`);
    abort(signal);parsed[key]=JSON.parse(text);
  }
  if(receipt.files[receipt.capture].sha256!==receipt.captureDigest||!same(saved,parsed.capture))
    throw new Error("Compiled product is not for this exact displayed Studio capture");
  const frame=studioCapturedPhysicsFrame(saved),manifest=parsed.manifest,rigidScene=parsed.rigidScene,mesh=parsed.wholeMesh;
  const source=manifest.source_capture,rec=manifest.sizes?.studio;
  if(source?.capture_digest!==receipt.captureDigest||source.mesh_digest!==receipt.meshDigest||source.revision!==saved.revision||
      rigidScene.sourceCaptureDigest!==receipt.captureDigest||rigidScene.sourceMeshDigest!==receipt.meshDigest||rigidScene.sourceRevision!==saved.revision||
      mesh.schema!=="turing.studio-whole-mesh-bake.v1"||mesh.digest!==receipt.meshDigest)
    throw new Error("Capture, whole mesh and native product identities differ");
  if(!same(rigidScene.frame,{unitM:frame.unitM,originM:frame.originM})||!same(rigidScene.room,frame.room))
    throw new Error("Compiled native frame differs from the displayed Studio room");
  if(rec?.n!==1||!Number.isSafeInteger(rec.pad)||rec.pad<1||rec.pad>16||rec.display_stride!==1||
      !same(manifest.law?.state,STUDIO_NATIVE_STATE)||!Number.isFinite(manifest.law.dt)||manifest.law.dt<=0||
      !filename(rec.wgsl)||!digestPattern.test(rec.wgsl_sha256)||receipt.files[rec.wgsl]?.sha256!==rec.wgsl_sha256||
      !Number.isSafeInteger(receipt.files[rec.wgsl]?.bytes)||receipt.files[rec.wgsl].bytes<1)
    throw new Error("Compiled product does not contain this exact one-marble native ABI");
  // The current isolated builder emits WGSL only. A future WASM lane needs
  // equivalent binary/ABI receipt checking before it can join this boundary.
  if(rec.wasm||rec.wasm_abi)throw new Error("Hidden Studio WASM byte admission is not implemented");
  const uniforms=rec.uniforms;
  for(let i=0;i<4;i++)if(uniforms?.[`r${i}`]!==Math.fround(frame.radiusM)||uniforms?.[`swell${i}`]!==0)
    throw new Error("Native species radius differs from the displayed marble");
  if(uniforms.emit_rate!==0)throw new Error("Hidden preparation may not emit replacement particles");
  validateStudioRigidBinding({handoff:{uniforms},manifest,rigidScene,rayOriginM:saved.state.frame.rayOriginM});
  const url=new URL(bundle.manifestUrl);
  if(!["http:","https:"].includes(url.protocol)||url.username||url.password||url.search||url.hash||
      url.pathname.split("/").at(-1)!==receipt.manifest)
    throw new Error("Native product needs its explicit immutable manifest URL");
  const seed=studioDisplayedStart(saved,frame,rec.pad);
  const spec={manifest:url.href,size:"studio",epoch:saved.revision,studio_count:1,
    relativeProductAssets:true,uniforms:structuredClone(uniforms),start:seed.start};
  // ComputeRuntime invokes this before capability/device work and again after
  // its await. It verifies what it fetched, not only the caller's metadata.
  const validateManifest=(loaded,request)=>{
    abort(signal);
    if(!same(loaded,manifest)||!same(request,spec))throw new Error("Fetched native product or full-state request changed after scene admission");
    return true;
  };
  return {capture:saved,frame,manifest,rigidScene,receipt,spec,validateManifest};
}

export async function prepareStudioNativeCandidate({capture,compiled,createRuntime,signal,maxNativeSteps=96}) {
  if(typeof createRuntime!=="function")throw new TypeError("A fresh ComputeRuntime factory is required");
  const admitted=await admitStudioCompiledCandidate(capture,compiled,{signal});abort(signal);
  const runtime=createRuntime();
  // Never retire a visible runtime supplied by mistake. Ownership starts only
  // after this fresh, empty instance check; the active scene is not an input.
  if(!runtime||runtime.spec||runtime.product||runtime.ready||runtime.preparing||runtime.retiring||
      typeof runtime.preparePausedFirstFrame!=="function"||typeof runtime.disposeAfterIdle!=="function")
    throw new Error("The native candidate factory must return a fresh detached ComputeRuntime");
  try {
    const checkpoint=await runtime.preparePausedFirstFrame(admitted.spec,{signal,maxNativeSteps,
      validateManifest:admitted.validateManifest});
    abort(signal);
    if(runtime.running!==false||runtime.handoffPaused!==true||runtime.shown!==checkpoint.frame||runtime.steps!==checkpoint.step)
      throw new Error("Native candidate is not paused at its actual displayed first frame");
    validateStudioRigidBinding({handoff:checkpoint,manifest:admitted.manifest,rigidScene:admitted.rigidScene,
      rayOriginM:admitted.capture.state.frame.rayOriginM});
    const dynamic=studioDynamicGeometry(admitted.capture,admitted.frame,checkpoint,admitted.manifest.sizes.studio.pad);
    return {runtime,checkpoint,dynamic,frame:admitted.frame,manifest:admitted.manifest,
      rigidScene:admitted.rigidScene,receipt:admitted.receipt};
  } catch(error) {
    try {await runtime.disposeAfterIdle();}
    catch(cleanup){throw new AggregateError([error,cleanup],"Hidden native candidate failed and cleanup also failed");}
    throw error;
  }
}

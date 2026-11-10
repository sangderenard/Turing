// Hidden candidate preparation and one final synchronous ownership transfer.
// This module never swaps the active scene, moves a camera, starts rays, or
// resumes physics. takeReady() must be consumed at one render-frame boundary.
// Real renderer adapters must stage BOTH consumers before returning receipts.

export class StudioSceneChangedError extends Error {
  constructor(message="The Studio scene changed while physics was preparing") {
    super(message);this.name="StudioSceneChangedError";
  }
}

// Exact structural comparison, including signed zero and typed-array bytes.
// An authored revision catches change-and-revert; this catches mutations that
// missed the revision event. No tolerance, probabilistic hash or camera reset.
export function sameStudioTransitionState(a,b) {
  if(Object.is(a,b))return true;
  if(!a||!b||typeof a!=="object"||typeof b!=="object"||a.constructor!==b.constructor)return false;
  if(a instanceof ArrayBuffer||ArrayBuffer.isView(a)) {
    const bytes=v=>v instanceof ArrayBuffer?new Uint8Array(v):new Uint8Array(v.buffer,v.byteOffset,v.byteLength);
    const x=bytes(a),y=bytes(b);
    return x.length===y.length&&x.every((v,i)=>v===y[i]);
  }
  if(!Array.isArray(a)&&Object.getPrototypeOf(a)!==Object.prototype)
    throw new TypeError("Studio capture must contain only plain data and typed arrays");
  if(Array.isArray(a)&&a.length!==b.length)return false;
  const ka=Object.keys(a).sort(),kb=Object.keys(b).sort();
  return ka.length===kb.length&&ka.every((key,i)=>key===kb[i]&&sameStudioTransitionState(a[key],b[key]));
}

function detachedCapture(value) {
  if(!value||!Number.isSafeInteger(value.revision)||value.revision<0||!value.state||typeof value.state!=="object")
    throw new TypeError("A synchronous Studio scene revision and detached state are required");
  const result=structuredClone({revision:value.revision,state:value.state});
  // Compare distinct copies so unsupported nested containers cannot bypass the
  // structural check via identity. JSON conversion would lose float32 bits.
  sameStudioTransitionState(result,structuredClone(result));
  return result;
}

function validatePaused(candidate) {
  const runtime=candidate?.runtime,checkpoint=candidate?.checkpoint;
  if(!runtime?.ready||runtime.handoffPaused!==true||runtime.running!==false||
      !Number.isSafeInteger(checkpoint?.frame)||checkpoint.frame<1||
      !Number.isSafeInteger(checkpoint?.step)||checkpoint.step<1||
      !Number.isFinite(checkpoint.seconds)||checkpoint.seconds<=0||!checkpoint.start||runtime.shown!==checkpoint.frame||runtime.steps!==checkpoint.step)
    throw new Error("Physics candidate must be warmed, exactly read back, and paused at its first displayed frame");
}
function validatePresentation(task) {
  for(const kind of ["phong","ray"]) {
    const receipt=task.presentation?.[kind];
    if(receipt?.ready!==true||receipt.revision!==task.capture.revision||receipt.step!==task.candidate.checkpoint.step)
      throw new Error(`${kind} must acknowledge the same staged scene revision and paused native step`);
  }
}

export function createStudioPhysicsTransition({captureScene,prepareCandidate,stagePresentation,releaseCandidate,onStatus=()=>{}}) {
  for(const [name,fn]of Object.entries({captureScene,prepareCandidate,stagePresentation,releaseCandidate,onStatus}))
    if(typeof fn!=="function")throw new TypeError(`${name} must be a function`);
  let generation=0,current=null,status={phase:"idle",revision:null,generation:0,error:null};
  const work=new Set(),cleanupErrors=[];
  const track=promise=>{work.add(promise);promise.then(()=>work.delete(promise),()=>work.delete(promise));return promise;};
  const publish=(phase,task,error=null)=>{
    if(task&&current!==task)return;
    status={phase,revision:task?.capture.revision??null,generation,error:error?String(error.message||error):null};
    // A UI status observer does not own native resources or commit semantics.
    try {onStatus({...status});} catch { /* A failed label cannot leak a candidate. */ }
  };
  const stillCurrent=task=>{
    if(current!==task||task.controller.signal.aborted)throw new StudioSceneChangedError("Physics preparation was canceled or superseded");
    const now=detachedCapture(captureScene());
    if(!sameStudioTransitionState(task.capture,now))throw new StudioSceneChangedError();
  };
  const retire=(task,reason)=>{
    if(!task||task.taken||task.released||!task.candidate)return Promise.resolve();
    task.released=true;
    return track(Promise.resolve().then(()=>releaseCandidate({candidate:task.candidate,presentation:task.presentation,reason}))
      .catch(error=>{cleanupErrors.push(error);throw error;}));
  };
  function prepare() {
    const capture=detachedCapture(captureScene());
    if(current&&!current.controller.signal.aborted&&sameStudioTransitionState(current.capture,capture)&&
        (status.phase==="preparing"||status.phase==="staging"||status.phase==="ready-paused"))return current.promise;
    const previous=current;
    previous?.controller.abort();
    const task={capture,controller:new AbortController(),candidate:null,presentation:null,taken:false,released:false};
    current=task;generation++;
    // Assign the promise before notifying observers: an observer may cancel
    // or request the same scene synchronously without escaping coalescing.
    task.promise=track(Promise.resolve().then(async()=>{
      try {
        // Only an unshown candidate is retired here. The previously visible
        // runtime was transferred out and is never owned by this coordinator.
        if(previous?.promise)await previous.promise.catch(()=>{});
        await retire(previous,"superseded");stillCurrent(task);
        task.candidate=await prepareCandidate({capture:structuredClone(capture),signal:task.controller.signal});
        stillCurrent(task);validatePaused(task.candidate);publish("staging",task);stillCurrent(task);
        task.presentation=await stagePresentation({capture:structuredClone(capture),candidate:task.candidate,signal:task.controller.signal});
        stillCurrent(task);validatePaused(task.candidate);
        validatePresentation(task);
        publish("ready-paused",task);
        return {revision:capture.revision,step:task.candidate.checkpoint.step};
      } catch(error) {
        try {await retire(task,"rejected");} catch(cleanup) {error=new AggregateError([error,cleanup],"Physics preparation failed and candidate cleanup also failed");}
        publish(error instanceof StudioSceneChangedError?"stale":"failed",task,error);
        throw error;
      }
    }));
    publish("preparing",task);
    return task.promise;
  }
  function takeReady() {
    const task=current;
    if(!task||status.phase!=="ready-paused")return null;
    try {stillCurrent(task);validatePaused(task.candidate);validatePresentation(task);}
    catch(error) {
      task.controller.abort();publish("stale",task,error);
      retire(task,"stale-before-reveal").catch(()=>{});throw error;
    }
    // No await between final scene check and transfer. The consumer atomically
    // installs this bundle for Phong AND rays; it must retain the old bundle
    // until its own in-flight readers release it. No implicit Continue call.
    task.taken=true;current=null;publish("transferred",null);
    return Object.freeze({revision:task.capture.revision,state:structuredClone(task.capture.state),
      candidate:task.candidate,presentation:task.presentation});
  }
  function cancel(reason="canceled") {
    const task=current;current=null;generation++;task?.controller.abort();publish("idle",null);
    // The native readback or renderer staging may still be using the candidate.
    // Signal cancellation now, but release resources only after that work lands.
    return track(Promise.resolve(task?.promise).catch(()=>{}).then(()=>retire(task,reason)));
  }
  return {prepare,takeReady,cancel,
    state:()=>({...status,cleanupFailures:cleanupErrors.length}),
    async settled(){while(work.size)await Promise.allSettled([...work]);if(cleanupErrors.length)throw new AggregateError(cleanupErrors,"Failed to release hidden physics candidates");}};
}

// Valid sensor samples cover the continuously running Phong view. Untouched
// pixels and obsolete revisions remain transparent, never counterfeit rays.
import {packMarblePhoto, photoBounceCount, photoBounceMode, showBounceDiagnostic} from "./marble_photo.js";
import {photoGeometryPolicy, showGeometryDiagnostic} from "./marble_geometry_policy.js";
import {createLiveParticipation} from "./marble_live_participation.js";
import {downloadLivePng} from "./marble_downloads.js";
import {createRodBundleWarning} from "./marble_optical_diagnostic.js";

export const MAX_LIVE_HISTORY_BYTES=2*1024*1024*1024, LIVE_HISTORY_WARNING_BYTES=512*1024*1024;
export function liveDimensions(edge, aspect, samples, checkpoint=false) {
  if (![128,256,512,768,1024,2048,4096].includes(edge) || !Number.isFinite(aspect) || aspect < .1 || aspect > 10 ||
      !Number.isSafeInteger(samples) || samples < 1 || typeof checkpoint!=="boolean") throw new Error("Invalid live resolution or history.");
  const [width,height] = aspect >= 1 ? [edge,Math.max(16,Math.round(edge/aspect))] : [Math.max(16,Math.round(edge*aspect)),edge];
  const bytes=width*height*3*(4*samples+(checkpoint?32:8));
  if(bytes>MAX_LIVE_HISTORY_BYTES)throw new Error("Ray history exceeds 2 GiB. Lower history length or resolution.");
  return {width,height,bytes};
}
export function liveHistoryLimit(edge,aspect,checkpoint=false,blockLimit=MAX_LIVE_HISTORY_BYTES){
  const {width,height}=liveDimensions(edge,aspect,1,checkpoint),entries=width*height*3,stateBytes=checkpoint?32:8;
  if(entries*stateBytes>blockLimit)throw new Error("This GPU cannot fit the history state at this resolution.");
  return Math.max(1,Math.min(Math.floor((MAX_LIVE_HISTORY_BYTES-entries*stateBytes)/(entries*4)),Math.floor(blockLimit/(entries*4))));
}

export function createLiveRayView({doc=document, canvas, getGeometry, getScene, getPose, getFilmResponse, ensureSaved,
  getOpticalMode=()=>"legacy", participationFactory=createLiveParticipation, fieldView=null}) {
  const $=id=>doc.getElementById(id), toggle=$("live-rays"), length=$("live-history"), resolution=$("live-resolution"), output=$("live-status");
  const participation=participationFactory(doc);
  const rodWarning=createRodBundleWarning(doc,'live-optical-warning');
  const screen=$("live-ray-view"), context=screen.getContext("2d");
  const count=$("live-history-count"),memory=$("live-history-memory"),checkpoint=$("live-checkpoint");
  const adjustmentSpecs={exposure:[-6,6,0],whitebalance:[-3,3,0],saturation:[0,2,1],
    vibrance:[-1,1,0],black_point:[0,.5,0],firefly_threshold:[2,64,8],firefly_strength:[0,1,1]};
  const adjustmentNode=key=>$(`live-${key.replaceAll("_","-")}`);
  const tint=$("live-tint"),normalization=$("live-normalization"),white=$("live-normalization-white");
  const percentile=$("live-normalization-percentile");
  const batch=$("live-ray-batch"),download=$("live-download");
  let lastImage=null,startIntent=0;
  let deviceBlock=MAX_LIVE_HISTORY_BYTES,lastWhite=1,displayNormalization=normalization.value;
  const state={enabled:false,generation:0,digest:null,objectDigest:null,timer:null,busy:false,
    revision:0,poseKey:"",imageKey:"",heartbeat:0,restart:false,notBefore:0,sceneVersion:0,disposed:false,stopping:Promise.resolve(),
    clientIntersections:false,geometryPolicy:null,geometryError:null,controlError:null,displayError:null,cameraError:null,pendingSave:false,pendingMessage:""};
  let session=crypto.randomUUID().replaceAll("-","");
  const text=value=>{output.textContent=value;};
  const present=on=>{screen.style.opacity=on?"1":"0";screen.setAttribute("aria-hidden",on?"false":"true");};
  const blank=()=>{present(false);context.clearRect(0,0,screen.width,screen.height);state.imageKey="";};
  const show=on=>{screen.hidden=!on;canvas.style.opacity="";if(!on)present(false);canvas.closest(".stage").classList.toggle("live-ray-mode",on);};
  const aspect=()=>Math.max(.1,Math.min(10,canvas.clientWidth/Math.max(1,canvas.clientHeight)));
  const syncBudget=()=>{
    const max=liveHistoryLimit(+resolution.value,aspect(),checkpoint.checked,deviceBlock);
    length.max=count.max=String(max);
    const previous=+length.value;
    length.value=String(Math.max(1,Math.min(max,Number.isSafeInteger(previous)?previous:16)));
    count.value=length.value;$("live-history-out").textContent=length.value;
    const {width,height,bytes}=liveDimensions(+resolution.value,aspect(),+length.value,checkpoint.checked);
    const warning=bytes>LIVE_HISTORY_WARNING_BYTES;
    memory.textContent=`${(bytes/1048576).toFixed(2)} MiB (${bytes.toLocaleString("en-US")} bytes) / 2 GiB · ${width} x ${height} · max ${max.toLocaleString("en-US")} rays/pixel${warning?" · high GPU memory use":""}`;
    memory.style.color=warning?"#ff0000":"";memory.style.fontWeight=warning?"900":"";
    if(previous>max)text(`History reduced to ${max} rays per pixel to fit this resolution and GPU limit. Accumulation resets.`);
  };
  const settings=()=>{
    const display={display_look:normalization.value==="specular"?"reference":($("live-display-look")?.value||"reference"),tint:+tint.value/100,normalization:normalization.value,normalization_white:+white.value,
      normalization_percentile:percentile.value.trim()===""?NaN:Number(percentile.value),
      tone_curve:normalization.value==="specular"?"linear":"exponential"};
    if(!Number.isFinite(display.tint)||Math.abs(display.tint)>3||!["auto99","manual","percentile","specular"].includes(display.normalization)||
      !Number.isFinite(display.normalization_white)||display.normalization_white<1e-6||display.normalization_white>1e12||
      !Number.isFinite(display.normalization_percentile)||display.normalization_percentile<1||display.normalization_percentile>100)
      throw new Error("Invalid live tint, normalization white, or percentile (1 to 100).");
    if(!["reference","color_priority"].includes(display.display_look))throw new Error("Choose Reference or Color priority display.");
    for(const [key,[low,high,defaultValue]] of Object.entries(adjustmentSpecs)){
      const node=adjustmentNode(key),raw=node?.value;
      const value=node? (raw.trim()===""?NaN:Number(raw)/100) : defaultValue;
      if(!Number.isFinite(value)||value<low||value>high)throw new Error(`Invalid live ${key.replaceAll("_"," ")}. Previous display and rays retained.`);
      display[key]=value;
    }
    display.firefly_filter=!!$("live-firefly-filter")?.checked;
    const rayBatch=Number(batch.value);
    if(!Number.isSafeInteger(rayBatch)||rayBatch<128||rayBatch>65536)throw new Error("Live ray batch must be an integer from 128 to 65,536.");
    return {history_samples:+length.value,aspect:aspect(),session_id:session,checkpoint:checkpoint.checked,ray_batch_size:rayBatch,display};
  };
  const validate=()=>{
    if(!state.objectDigest)throw new Error("Save or load the design before starting live rays.");
    if($("preview-engine").value!=="visual" || $("preview-motion").value!=="off")
      throw new Error("Live rays follow orbit and zoom on a static design. Turn preview motion off and use Visual preview first.");
    if (+resolution.value>512 && !((state.restart||!state.digest)?participation.requested():state.clientIntersections)) throw new Error("Live sizes above 512 require cooperative device intersections.");
    liveDimensions(+resolution.value,settings().aspect,+length.value,checkpoint.checked);
    photoBounceCount(doc);
  };
  const request=async(url,options={})=>{
    const response=await fetch(url,{cache:"no-store",...options,signal:AbortSignal.timeout(15000)});
    let value;
    try {value=await response.json();}
    catch {throw Object.assign(new Error(`Live rays HTTP ${response.status}: the server returned an unreadable response. Check that the server is running, then retry.`),{status:response.status});}
    if(!response.ok)throw Object.assign(new Error(value.error||`Live rays HTTP ${response.status}`),{status:response.status});
    return value;
  };
  const post=(url,body)=>request(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const cancel=async digest=>{if(digest)await post("/api/marble_photo/live",{digest,stop:true});};
  // One asynchronous pump owns start, control, polling and image delivery.
  // A later toggle/scene revision invalidates every earlier await boundary.
  const current=g=>state.enabled&&state.generation===g&&!state.disposed;
  const schedule=()=>{if(state.enabled&&!state.disposed)state.timer=setTimeout(pump,250);};
  async function pump() {
    if(state.busy||!state.enabled||state.disposed)return;
    if(Date.now()<state.notBefore){schedule();return;}
    state.busy=true;
    const generation=state.generation;
    try {
      await state.stopping;
      if(!current(generation))return;
      if(state.pendingSave&&!state.digest){text(state.pendingMessage);return;}
      state.controlError=Number.isSafeInteger(+batch.value)&&+batch.value>=128&&+batch.value<=65536?null:"Enter a ray batch from 128 to 65,536. The previous worker is retained.";
      state.displayError=null;
      if(!state.controlError){try{settings();}catch(error){state.displayError=error.message;}}
      let requestedPose=null;state.cameraError=null;
      try{requestedPose=getPose();}catch(error){
        if(!state.digest)throw error;
        state.cameraError=error.message;blank();
      }
      if(!state.pendingSave&&(!state.controlError||!state.digest)&&(!state.displayError||!state.digest))validate();
      let geometryPolicy=state.geometryPolicy;
      // Recheck after the debounce, before any owned worker is canceled. An
      // incomplete budget edit must leave the previous live scene running.
      if(!state.pendingSave&&!state.controlError&&!state.displayError&&!state.cameraError&&(state.restart||!state.digest)){
        try{geometryPolicy=photoGeometryPolicy(doc);state.geometryError=null;}
        catch(error){state.restart=false;state.geometryError=error.message;text(`Live scene unchanged: ${error.message}`);return;}
      }
      if(!state.pendingSave&&!state.controlError&&!state.displayError&&!state.cameraError&&(state.restart||!state.digest)) {
        const previousDigest=state.digest;
        const pose=requestedPose,live=settings(),clientIntersections=participation.requested();
        const fieldTicket=fieldView?.capture({saved:true});
        const body=packMarblePhoto(getGeometry(),state.objectDigest,{...getScene(),camera:{...pose,resolution:+resolution.value},
          resolution:+resolution.value,seconds:3600,transport:"fixed:3",orbitFrames:1,live,clientIntersections,receiveLightField:!!fieldTicket,...geometryPolicy,
          filmResponse:getFilmResponse(),maxBounces:photoBounceCount(doc),bounceMode:photoBounceMode(doc),gamma:+$("photo-gamma").value/100});
        text("Preparing the real ray-traced view…");
        showGeometryDiagnostic(doc,null);
        // The server validates the entire replacement scene before replacing
        // this session's worker. Do not pre-cancel and defeat that transaction.
        let job;
        try{job=await request("/api/marble_photo",{method:"POST",headers:{"Content-Type":"application/octet-stream"},body});}
        catch(error){
          if(previousDigest&&current(generation)&&error.status===400){
            state.restart=false;state.geometryError=error.message;blank();
            text(`Live scene unchanged: ${error.message}`);return;
          }
          throw error;
        }
        if(!current(generation)){
          const reusedActive=job.digest===previousDigest&&state.digest===previousDigest&&state.enabled&&!state.disposed;
          if(!reusedActive){if(state.digest===previousDigest)state.digest=null;await cancel(job.digest);}
          return;
        }
        state.clientIntersections=clientIntersections;participation.connect(job,clientIntersections);
        state.digest=job.digest;state.restart=false;state.geometryPolicy=geometryPolicy;state.geometryError=null;
        if(job.digest!==previousDigest){state.revision=0;state.poseKey=JSON.stringify({camera:pose,live});}
        state.heartbeat=0;
        fieldView?.connect(job.digest,fieldTicket,state.revision);
        blank();show(true);
      }
      // A rejected resize may leave UI history capacity sized for a different
      // sensor. Renew and display the accepted scene without sending those
      // unaccepted settings into its old buffers. A valid scene edit retries.
      const accepted=(state.geometryError||state.pendingSave||state.controlError||state.displayError||state.cameraError)?JSON.parse(state.poseKey):null;
      const digest=state.digest,pose=accepted?.camera??requestedPose,live=accepted?.live??settings(),poseKey=JSON.stringify({camera:pose,live});
      if(poseKey!==state.poseKey || Date.now()-state.heartbeat>3000){
        const changed=poseKey!==state.poseKey;
        if(changed){
          state.revision++;
          const previous=JSON.parse(state.poseKey);
          const resets=JSON.stringify(previous.camera)!==JSON.stringify(pose)||["history_samples","aspect","checkpoint","ray_batch_size"].some(key=>previous.live[key]!==live[key]);
          if(resets){blank();text("Refreshing rays for the current camera and history settings.");}
          else text("Updating live color and cleanup without resetting rays.");
        }
        await post("/api/marble_photo/live",{digest,revision:state.revision,camera:pose,live});
        if(!current(generation))return;
        state.poseKey=poseKey;state.heartbeat=Date.now();
        fieldView?.revise(digest,state.revision);
      }
      const status=await request(`/api/marble_photo?digest=${digest}`);
      if(!current(generation))return;
      const opticalImageKey=`${digest}/${status.image??''}/${status.live_revision??''}`;
      if(status.live_revision===state.revision)rodWarning.observe(status,opticalImageKey);
      showBounceDiagnostic(doc,status);
      showGeometryDiagnostic(doc,status);
      if(status.state==="error")throw new Error(status.error||"The ray worker stopped.");
      if(status.state==="canceled"||status.state==="complete")throw new Error(status.reason||"The live ray session ended.");
      if(state.cameraError){blank();text(`Live camera unchanged: ${state.cameraError} Previous worker retained; complete the edit to resume.`);return;}
      if(state.pendingSave||state.geometryError||state.controlError){blank();text(state.pendingSave?state.pendingMessage:state.controlError??`Live scene unchanged: ${state.geometryError}. Phong preview shown.`);return;}
      if(state.displayError){text(state.displayError);return;}
      if(status.state!=="live"||status.live_revision!==state.revision)return;
      void fieldView?.observe(status,digest);
      // This viewer displays status.image, whose recipe is display_transform.
      // Retouched-image metadata belongs to a different image, not this frame.
      const displayWhite=status.display_transform?.effective_white??status.display_white;
      if(Number.isFinite(displayWhite)&&displayWhite>0)lastWhite=displayWhite;
      const key=`${digest}/${status.image}/${state.revision}`;
      if(status.image&&key!==state.imageKey){
        const response=await fetch(`/api/marble_photo/image?digest=${digest}&name=${encodeURIComponent(status.image)}`,{cache:"no-store",signal:AbortSignal.timeout(10000)});
        if(response.status===404)return; // superseded between status and image; next poll catches up
        if(!response.ok)throw new Error(`Ray image HTTP ${response.status}`);
        const blob=await response.blob(),bitmap=await createImageBitmap(blob);
        try {
          if(!current(generation)||state.pendingSave||state.geometryError||state.controlError)return;
          let latestPose;
          try{latestPose=getPose();}catch(error){state.cameraError=error.message;blank();text(`Live camera unchanged: ${error.message} Previous worker retained.`);return;}
          if(JSON.stringify({camera:latestPose,live:settings()})!==state.poseKey)return;
          screen.width=bitmap.width;screen.height=bitmap.height;context.drawImage(bitmap,0,0);
          state.imageKey=key;lastImage={blob,digest};download.disabled=false;present(true);
          rodWarning.displayed(status,opticalImageKey);
        } finally {bitmap.close();}
      }
      if(Number.isFinite(status.history_block_limit)&&status.history_block_limit>0&&status.history_block_limit!==deviceBlock){deviceBlock=status.history_block_limit;syncBudget();}
      text(state.geometryError?`Live scene unchanged: ${state.geometryError}`:`Live rays · ${status.width} x ${status.height} · ${status.history_filled}/${status.history_samples} rays per pixel${Number.isFinite(status.ray_batch_size)?` | ${status.ray_batch_size.toLocaleString("en-US")} forward rays/sweep`:""}${Number.isFinite(status.coverage_fraction)?` | ${(100*status.coverage_fraction).toFixed(1)}% sampled`:""} · ${(status.history_bytes/1048576).toFixed(1)} MiB history${status.checkpoint_enabled?` · ${Number(status.total_samples||0).toLocaleString("en-US")} total samples/pixel (${Number(status.checkpoint_samples||0).toLocaleString("en-US")} checkpointed)`:""}`);
    } catch(error) {
      if(current(generation))await stop(`Live rays stopped: ${error.message||error}`);
    } finally {state.busy=false;schedule();}
  }
  async function stop(message="Live rays off. Phong camera restored.") {
    startIntent++;
    participation.stop();
    state.enabled=false;state.generation++;toggle.checked=false;
    if(state.timer)clearTimeout(state.timer);state.timer=null;
    const digest=state.digest;fieldView?.stop(digest);state.digest=null;state.restart=false;
    show(false);blank();text(message);
    const generation=state.generation;
    const stopping=state.stopping.then(()=>cancel(digest));
    state.stopping=stopping.catch(()=>{});
    try{await stopping;}catch(error){if(!state.enabled&&state.generation===generation)text(`${message} Worker cleanup was not confirmed; its client lease expires within 30 seconds. ${error.message||error}`);}
  }
  async function start() {
    const intent=++startIntent;
    try{if(ensureSaved)await ensureSaved();}
    catch(error){if(intent===startIntent){toggle.checked=false;text(`Live rays not started: ${error.message||error}`);}return;}
    if(intent!==startIntent||!toggle.checked||state.disposed)return;
    try{validate();photoGeometryPolicy(doc);state.geometryError=null;}
    catch(error){toggle.checked=false;text(error.message);return;}
    // A later toggle has a new immutable server identity. Even an uncertain
    // old stop response can only name its old digest, never this new session.
    session=crypto.randomUUID().replaceAll("-","");
    state.enabled=true;state.generation++;state.restart=true;state.notBefore=0;
    // Phong keeps drawing below the transparent overlay during preparation.
    blank();show(true);text("Preparing the real ray-traced view…");
    if(!state.busy)await pump();
  }
  const change=()=>{if(toggle.checked)void start().catch(error=>stop(`Live rays not started: ${error.message||error}`));else void stop();};
  toggle.addEventListener("change",change);
  const restart=()=>{
    fieldView?.invalidate();
    if(!state.enabled)return;
    blank();
    if(state.pendingSave){state.restart=true;return;}
    try{photoGeometryPolicy(doc);state.geometryError=null;}
    catch(error){state.restart=false;state.geometryError=error.message;text(`Live scene unchanged: ${error.message}`);return;}
    state.generation++;state.restart=true;state.notBefore=Date.now()+350;blank();
    if(!state.busy){if(state.timer)clearTimeout(state.timer);schedule();}
  };
  resolution.addEventListener("change",()=>{
    try{syncBudget();state.geometryError=null;restart();}
    catch(error){state.geometryError=error.message;state.restart=false;text(`Live resolution unchanged: ${error.message}`);}
  });
  length.addEventListener("input",()=>{syncBudget();if(state.enabled)blank();});
  count.addEventListener("change",()=>{length.value=count.value;syncBudget();if(state.enabled)blank();});
  checkpoint.addEventListener("change",()=>{syncBudget();if(state.enabled)blank();});
  batch.addEventListener("change",()=>{if(state.enabled)blank();});
  download.disabled=true;
  download.addEventListener("click",async()=>{
    if(!lastImage)return;
    const image=lastImage;
    try{await downloadLivePng(image.blob,{digest:image.digest});}
    catch(error){text(`Ray image download failed: ${error.message||error}`);}
  });
  const syncDisplay=()=>{
    $("live-tint-out").textContent=(+tint.value/100).toFixed(2);white.disabled=normalization.value!=="manual";
    if($("live-display-look"))$("live-display-look").disabled=normalization.value==="specular";
    percentile.disabled=!["percentile","specular"].includes(normalization.value);
    $("live-normalization-percentile-out").textContent=`${percentile.value}%`;
    for(const key of Object.keys(adjustmentSpecs)){
      const node=adjustmentNode(key),out=$(`live-${key.replaceAll("_","-")}-out`);
      if(out&&node)out.textContent=(Number(node.value)/100).toFixed(2)+(key==="firefly_threshold"?"x":"");
      if(node&&key.startsWith("firefly_"))node.disabled=!$("live-firefly-filter")?.checked;
    }
  };
  tint.addEventListener("input",syncDisplay);percentile.addEventListener("input",syncDisplay);
  for(const key of Object.keys(adjustmentSpecs))adjustmentNode(key)?.addEventListener("input",syncDisplay);
  $("live-firefly-filter")?.addEventListener("change",syncDisplay);
  normalization.addEventListener("change",()=>{
    if(normalization.value==="manual")white.value=String(lastWhite);
    if(normalization.value==="specular"&&displayNormalization!=="specular")percentile.value="95";
    displayNormalization=normalization.value;
    syncDisplay();
  });
  syncBudget();syncDisplay();
  // History changes go through the tiny control update, not a scene upload.
  const onPhotoInput=event=>{
    if(event.target.closest("#photo-film-response")||["photo-bounces","photo-bounces-slider","photo-bounce-mode","photo-gamma","photo-triangle-budget","photo-sphere-coverage"].includes(event.target.id))restart();
  };
  $("photo-controls").addEventListener("input",onPhotoInput);
  $("photo-controls").addEventListener("change",onPhotoInput);
  const resize=new ResizeObserver(()=>{syncBudget();restart();});resize.observe(canvas);
  const onHide=()=>{if(doc.hidden&&state.enabled){if(state.clientIntersections)text("Device work paused while hidden; cooperative Live retains its pending rays and history.");else void stop("Live rays paused because this tab is hidden.");}};
  doc.addEventListener("visibilitychange",onHide);
  const onPageHide=()=>{
    if(state.digest&&!state.clientIntersections)navigator.sendBeacon("/api/marble_photo/live",new Blob([JSON.stringify({digest:state.digest,stop:true})],{type:"application/json"}));
  };
  doc.defaultView.addEventListener("pagehide",onPageHide);
  return {
    saved(digest){const changed=digest!==state.objectDigest||state.pendingSave;state.objectDigest=digest;state.pendingSave=false;if(changed)restart();},
    pendingSave(message="Saving object changes. Phong preview shown while rays update."){fieldView?.invalidate();state.pendingSave=true;state.pendingMessage=message;blank();if(state.enabled)text(message);},
    cameraChanged(){if(state.enabled)blank();},
    stale(){state.objectDigest=null;if(state.enabled)void stop("Design changed. Save it again before starting live rays.");},
    sceneChanged:restart,
    get enabled(){return state.enabled;},
    stop,
    dispose(){state.disposed=true;void stop();resize.disconnect();doc.removeEventListener("visibilitychange",onHide);
      doc.defaultView.removeEventListener("pagehide",onPageHide);toggle.removeEventListener("change",change);
      $("photo-controls").removeEventListener("input",onPhotoInput);$("photo-controls").removeEventListener("change",onPhotoInput);},
  };
}

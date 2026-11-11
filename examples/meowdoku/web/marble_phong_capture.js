// Capture only the incumbent Studio's completed canvas. No shader, renderer,
// scene reconstruction, ray job, microphone or screen-capture permission.
export const PHONG_CAPTURE_LIMITS = Object.freeze({seconds:120,bytes:128*1024*1024,pixels:16777216});
export function phongRecordingMime(Recorder=globalThis.MediaRecorder) {
  if(typeof Recorder!=='function'||typeof Recorder.isTypeSupported!=='function')return null;
  return ['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm','video/mp4'].find(type=>Recorder.isTypeSupported(type))??null;
}
export function phongCaptureFilename(type,date=new Date()) {
  const ext=type==='image/png'?'png':type.startsWith('video/mp4')?'mp4':type.startsWith('video/webm')?'webm':null;
  if(!ext)throw Error('Unsupported Phong capture format.');
  return `marble-phong-${date.toISOString().replace(/[-:.]/g,'')}.${ext}`;
}

export function createPhongCapture({canvas,makeCanvas=()=>document.createElement('canvas'),
  Recorder=globalThis.MediaRecorder,now=()=>performance.now(),setTimer=setTimeout,clearTimer=clearTimeout,
  onState=()=>{},onArtifact=()=>{},limits=PHONG_CAPTURE_LIMITS}={}) {
  if(!canvas)throw Error('The displayed Studio canvas is required.');
  let photo=null,video=null,disposed=false,ready=false;
  const mime=phongRecordingMime(Recorder);
  function emit(text){if(!disposed)onState({text,photoPending:!!photo,videoState:video?.state??'idle',ready,videoSupported:!!mime});}
  function dimensions(){
    const {width,height}=canvas;
    if(!Number.isSafeInteger(width)||!Number.isSafeInteger(height)||width<1||height<1||width*height>limits.pixels)
      throw Error('Capture needs a canvas between 1 and 16 million pixels. Reduce the window size.');
    return {width,height};
  }
  function surface(){const size=dimensions(),target=makeCanvas();target.width=size.width;target.height=size.height;
    const context=target.getContext('2d',{alpha:false});if(!context)throw Error('Canvas capture is unavailable.');
    context.drawImage(canvas,0,0);return {target,context,...size};}
  function release(v){clearTimer(v.timer);clearTimer(v.watchdog);for(const track of v.stream?.getTracks()??[])track.stop();
    if(v.target){v.target.width=1;v.target.height=1;}}
  function finish(v,error=null){
    if(video!==v)return;
    video=null;release(v);
    if(error||v.discard||disposed){const reason=error??new Error('Recording canceled.');v.reject(reason);emit(reason.message);return;}
    const blob=new Blob(v.chunks,{type:v.recorder?.mimeType||mime});
    if(!blob.size){const reason=new Error('The browser produced no video frames.');v.reject(reason);emit(reason.message);return;}
    const artifact={blob,filename:phongCaptureFilename(blob.type),width:v.width,height:v.height,fps:v.fps,
      seconds:Math.max(0,(now()-v.started)/1000),reason:v.reason,kind:'video'};
    v.resolve(artifact);onArtifact(artifact);emit(`${v.reason||'Recording finished.'} ${v.width} × ${v.height} video ready to download.`);
  }
  function stop(reason='Recording finished.'){
    const v=video;if(!v)return Promise.resolve(null);
    if(v.state==='stopping')return v.done;
    v.reason=reason;
    if(v.state==='pending'){finish(v,new Error('Recording canceled before a completed frame.'));return v.done;}
    v.state='stopping';clearTimer(v.timer);emit('Finishing Phong video…');
    v.watchdog=setTimer(()=>finish(v,new Error('The browser did not finish recording. Please try again.')),10000);
    try{v.recorder.stop();}catch(error){finish(v,error);}return v.done;
  }
  function fail(error){
    ready=false;const message=error?.message||String(error);
    if(photo){const request=photo;photo=null;clearTimer(request.timer);request.reject(new Error(message));}
    if(video){if(video.state==='pending')finish(video,new Error(message));else stop(`Recording stopped: ${message}`);}
    emit(message);
  }
  function takePhoto(){
    if(disposed)return Promise.reject(new Error('Capture is closed.'));
    if(photo)return Promise.reject(new Error('A Phong photo is already being prepared.'));
    return new Promise((resolve,reject)=>{const request={resolve,reject};photo=request;
      request.timer=setTimer(()=>{if(photo===request){photo=null;reject(new Error(request.encoding?'PNG encoding timed out.':'No completed Studio frame arrived.'));emit('Photo timed out.');}},10000);
      emit('Waiting for a completed Phong frame…');});
  }
  function start({fps=30}={}){
    if(disposed)throw Error('Capture is closed.');if(video)throw Error('A Phong recording is already active.');
    if(!mime)throw Error('Video recording is unavailable in this browser. PNG photos remain available.');
    if(![24,30,60].includes(fps))throw Error('Choose 24, 30 or 60 frames per second.');
    dimensions();
    const v={state:'pending',fps,chunks:[],bytes:0,lastFrame:-Infinity,frames:0};
    v.done=new Promise((resolve,reject)=>{v.resolve=resolve;v.reject=reject;});
    // The UI receives errors through onState; a caller may also await done.
    v.done.catch(()=>{});video=v;
    v.timer=setTimer(()=>{if(video===v)finish(v,new Error('No completed Studio frame arrived.'));},10000);
    emit('Waiting to record the next completed Phong frame…');return v.done;
  }
  function frame(){
    if(disposed)return;const first=!ready;ready=true;
    if(photo){
      const request=photo;
      // Keep photo pending while the encoder owns the snapshot, preventing a
      // repeated click from allocating unbounded full-resolution canvases.
      if(!request.encoding)try{
        request.encoding=true;const shot=surface();
        shot.target.toBlob(blob=>{
          shot.target.width=1;shot.target.height=1;
          if(photo!==request)return;clearTimer(request.timer);photo=null;
          if(!blob?.size){request.reject(new Error('The browser could not encode the Phong PNG.'));emit('PNG encoding failed.');return;}
          const artifact={blob,filename:phongCaptureFilename('image/png'),width:shot.width,height:shot.height,kind:'photo'};
          request.resolve(artifact);if(!disposed)onArtifact(artifact);emit(`${shot.width} × ${shot.height} Phong PNG ready to download.`);
        },'image/png');
      }catch(error){clearTimer(request.timer);photo=null;request.reject(error);emit(error.message);}
    }
    const v=video;
    if(v&&v.state!=='stopping')try{
      if(v.state==='pending'){
        clearTimer(v.timer);Object.assign(v,surface());
        if(typeof v.target.captureStream!=='function')throw Error('Canvas video capture is unavailable in this browser.');
        v.stream=v.target.captureStream(0);v.track=v.stream.getVideoTracks()[0];
        if(!v.track)throw Error('The browser did not create a canvas video track.');
        // Older implementations without manual requestFrame use their own
        // canvas clock; copies still occur only after completed Studio frames.
        if(typeof v.track.requestFrame!=='function'){
          for(const track of v.stream.getTracks())track.stop();v.stream=v.target.captureStream(v.fps);v.track=v.stream.getVideoTracks()[0];
        }
        v.recorder=new Recorder(v.stream,{mimeType:mime,videoBitsPerSecond:8000000});
        v.recorder.ondataavailable=event=>{if(video!==v||!event.data?.size)return;
          if(v.bytes+event.data.size>limits.bytes){v.error=new Error('Recording exceeded its 128 MiB memory limit.');v.discard=true;stop(v.error.message);return;}
          v.chunks.push(event.data);v.bytes+=event.data.size;};
        v.recorder.onerror=event=>{v.error=event.error??new Error('Video encoder failed.');stop(v.error.message);};
        v.recorder.onstop=()=>finish(v,v.error);
        v.recorder.start(1000);v.state='recording';v.started=now();
        v.timer=setTimer(()=>stop('Recording stopped at the two-minute limit.'),limits.seconds*1000);
        emit(`Recording Phong at up to ${v.fps} fps, ${v.width} × ${v.height}. No audio. Stop to save.`);
      }
      if(canvas.width!==v.width||canvas.height!==v.height){stop('Recording stopped because the preview size changed.');return;}
      const time=now();if(time-v.lastFrame>=1000/v.fps){v.context.drawImage(canvas,0,0);v.track?.requestFrame?.();v.lastFrame=time;v.frames++;}
    }catch(error){finish(v,error);}
    else if(first&&!photo&&!v)emit('Ready. Capture the Phong canvas without starting a ray job.');
  }
  function dispose(){if(disposed)return;disposed=true;
    if(photo){clearTimer(photo.timer);photo.reject(new Error('Capture closed.'));photo=null;}
    if(video){const v=video;v.discard=true;try{if(v.recorder?.state!=='inactive')v.recorder?.stop();}finally{finish(v,new Error('Capture closed.'));}}
  }
  return {frame,fail,takePhoto,start,stop,dispose,get state(){return {ready,photoPending:!!photo,videoState:video?.state??'idle',videoSupported:!!mime};}};
}

export function mountPhongCapture({doc=document,canvas,gl,urls=URL,...options}) {
  const $=id=>doc.getElementById(id),photo=$('phong-capture-photo'),start=$('phong-capture-start'),stop=$('phong-capture-stop'),fps=$('phong-capture-fps');
  const links=new Map();
  const capture=createPhongCapture({canvas,...options,onState(state){
    $('phong-capture-status').textContent=state.text;
    photo.disabled=!gl||!state.ready||state.photoPending;
    start.disabled=!gl||!state.ready||!state.videoSupported||state.videoState!=='idle';
    stop.disabled=!['pending','recording'].includes(state.videoState);fps.disabled=state.videoState!=='idle';
  },onArtifact(artifact){
    const link=$(artifact.kind==='photo'?'phong-capture-png':'phong-capture-video'),old=links.get(link);
    const url=urls.createObjectURL(artifact.blob);links.set(link,url);link.href=url;link.download=artifact.filename;
    link.textContent=`Download ${artifact.kind==='photo'?'PNG':artifact.blob.type.startsWith('video/mp4')?'MP4':'WebM'} (${(artifact.blob.size/1048576).toFixed(2)} MiB)`;
    link.hidden=false;if(old)urls.revokeObjectURL(old);
  }});
  const report=error=>{$('phong-capture-status').textContent=error.message||String(error);};
  photo.addEventListener('click',()=>capture.takePhoto().catch(report));
  start.addEventListener('click',()=>{try{capture.start({fps:Number(fps.value)}).catch(report);}catch(error){report(error);}});
  stop.addEventListener('click',()=>capture.stop().catch(report));
  canvas.addEventListener('webglcontextlost',()=>capture.fail(new Error('WebGL context lost.')));
  doc.addEventListener('visibilitychange',()=>{if(doc.hidden)capture.stop('Recording stopped because this page was hidden.').catch(report);});
  globalThis.addEventListener?.('pagehide',event=>{
    if(event.persisted){capture.stop('Recording stopped because this page was left.').catch(report);return;}
    capture.dispose();for(const url of links.values())urls.revokeObjectURL(url);links.clear();
  });
  photo.disabled=start.disabled=true;stop.disabled=true;
  if(!capture.state.videoSupported)$('phong-capture-status').textContent='This browser supports PNG photos but not video recording.';
  return capture;
}

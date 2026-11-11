// Receive the existing forward volume, never sensor pixels or invented lights.
import {inverseAffine} from './marble_studio_subject_renderer.js';

const DIGEST=/^[0-9a-f]{64}$/, NAME=/^light-field-[0-9a-f]{32}\.bin$/;
export const MAX_FIELD_BYTES=256*256*256*12;
export function validateLightField(value,digest,revision) {
  const vector=x=>Array.isArray(x)&&x.length===3&&x.every(Number.isFinite);
  if(!value||value.schema!=='studio.forward-field.v1'||!DIGEST.test(digest)||value.digest!==digest||
      value.live_revision!==revision||!Number.isSafeInteger(value.sequence)||value.sequence<0||
      !NAME.test(value.name)||value.format!=='rgb-f32le'||value.layout!=='zyx-interleaved-rgb'||
      value.color_space!=='linear-srgb'||!vector(value.dimensions)||
      !value.dimensions.every(n=>Number.isSafeInteger(n)&&n>=1&&n<=256)||
      !vector(value.bounds_min_m)||!vector(value.bounds_max_m)||
      value.bounds_max_m.some((n,i)=>n<=value.bounds_min_m[i])||
      value.byte_length!==value.dimensions.reduce((a,b)=>a*b,12)||value.byte_length>MAX_FIELD_BYTES)
    throw Error('Invalid or stale forward-field metadata.');
  return structuredClone(value);
}

export function fieldFromView(view,unitM,origin,boundsMin,boundsMax) {
  if(!Number.isFinite(unitM)||unitM<=0||![origin,boundsMin,boundsMax].every(v=>
      v?.length===3&&Array.from(v).every(Number.isFinite))||boundsMax.some((v,i)=>v<=boundsMin[i]))
    throw Error('Invalid forward-field coordinate mapping.');
  // The shared affine inverse already handles every Studio model/view frame.
  // vPosV includes each draw's object transform; inverse(view) recovers world.
  const inverse=inverseAffine(view),result=new Float32Array(inverse);
  for(let row=0;row<3;row++) {
    const span=boundsMax[row]-boundsMin[row],scale=unitM/span;
    for(let col=0;col<4;col++)result[col*4+row]=inverse[col*4+row]*scale;
    result[12+row]+=(origin[row]-boundsMin[row])/span;
  }
  if(!Array.from(result).every(Number.isFinite))throw Error('Forward-field coordinates exceed float32 range.');
  return result;
}

export function decodeLightField(buffer,metadata) {
  if(!(buffer instanceof ArrayBuffer)||buffer.byteLength!==metadata.byte_length)
    throw Error('Incomplete forward-field payload.');
  const view=new DataView(buffer),values=new Float32Array(buffer.byteLength/4);
  for(let i=0;i<values.length;i++) {
    const v=view.getFloat32(i*4,true);if(!Number.isFinite(v))throw Error('Nonfinite forward-field sample.');
    values[i]=v; // Signed out-of-gamut linear RGB is intentional.
  }
  return values;
}

export function createStudioLightField({gl,doc=document,getSceneKey,fetcher=fetch,onChange=()=>{},fallbackTexture=null}) {
  const toggle=doc.getElementById('phong-light-field'),output=doc.getElementById('phong-light-field-status');
  let epoch=0,source=null,metadata=null,texture=null,pending=null,disposed=false;
  const say=text=>{if(output)output.textContent=text;};
  const drop=()=>{
    if(texture){
      const active=gl.getParameter(gl.ACTIVE_TEXTURE);gl.activeTexture(gl.TEXTURE8);
      if(gl.getParameter(gl.TEXTURE_BINDING_3D)===texture)gl.bindTexture(gl.TEXTURE_3D,fallbackTexture);
      gl.deleteTexture(texture);gl.activeTexture(active);
    }
    texture=null;metadata=null;
  };
  const requested=()=>!disposed&&toggle?.checked===true;
  const invalidate=(reason='Scene changed. Waiting for matching ray data.')=>{
    epoch++;source=null;pending?.controller.abort();pending=null;drop();if(requested())say(reason);
  };
  const key=context=>getSceneKey(context);
  const current=s=>requested()&&source===s&&s.epoch===epoch&&key(s.context)===s.key;
  const capture=(context={})=>{
    if(!requested())return null;
    try{return {key:key(context),context:{...context},epoch};}
    catch(error){say(`Field not requested: ${error.message}`);return null;}
  };
  const connect=(digest,ticket,revision=0)=>{
    if(!ticket||!requested()||ticket.epoch!==epoch||!DIGEST.test(digest))return;
    try{if(key(ticket.context)!==ticket.key)return;
      pending?.controller.abort();pending=null;drop();source={...ticket,digest,revision};
      say('Opted in. Waiting for the ray tracer’s forward volume.');
    }catch(error){invalidate(error.message);}
  };
  const observe=async(status,digest)=>{
    const s=source;if(!s||s.digest!==digest)return;
    try {
      if(!current(s)){drop();return;}
      if(['error','canceled'].includes(status.state)){invalidate('Ray job ended without a current field.');return;}
      if(!status.light_field)return;
      const next=validateLightField(status.light_field,digest,s.revision);
      if(metadata&&next.sequence<=metadata.sequence||pending)return;
      const controller=new AbortController(),work={controller};pending=work;
      const timer=setTimeout(()=>controller.abort(),15000);
      try {
        const response=await fetcher(`/api/marble_photo/field?digest=${digest}&name=${encodeURIComponent(next.name)}`,
          {cache:'no-store',signal:controller.signal});
        if(response.status===404||response.status===409)return; // status superseded; next poll retries
        if(!response.ok)throw Error(`Forward-field HTTP ${response.status}`);
        const length=response.headers?.get?.('Content-Length');
        if(length!==null&&length!==undefined&&Number(length)!==next.byte_length)throw Error('Forward-field length changed.');
        const bytes=await response.arrayBuffer();if(!current(s))return;
        const values=decodeLightField(bytes,next);
        if(next.sha256){
          if(!DIGEST.test(next.sha256))throw Error('Invalid forward-field content hash.');
          const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');
          if(hash!==next.sha256)throw Error('Forward-field content hash mismatch.');
        }
        if(!current(s))return;
        const previousActive=gl.getParameter(gl.ACTIVE_TEXTURE);gl.activeTexture(gl.TEXTURE8);
        const previous=gl.getParameter(gl.TEXTURE_BINDING_3D),oldTexture=texture,unpack=gl.getParameter(gl.UNPACK_ALIGNMENT);
        let nextTexture=null;
        try {
          if(next.dimensions.some(n=>n>gl.getParameter(gl.MAX_3D_TEXTURE_SIZE)))throw Error('Forward volume exceeds this GPU’s texture limit.');
          nextTexture=gl.createTexture();if(!nextTexture)throw Error('Forward volume allocation failed.');
          gl.bindTexture(gl.TEXTURE_3D,nextTexture);gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);
          for(const parameter of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_3D,parameter,gl.NEAREST);
          for(const parameter of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T,gl.TEXTURE_WRAP_R])gl.texParameteri(gl.TEXTURE_3D,parameter,gl.CLAMP_TO_EDGE);
          gl.texImage3D(gl.TEXTURE_3D,0,gl.RGB32F,...next.dimensions,0,gl.RGB,gl.FLOAT,values);
          if(gl.getError()!==gl.NO_ERROR)throw Error('Forward volume upload failed.');
          drop();texture=nextTexture;metadata=next;nextTexture=null;
        }finally{
          if(nextTexture)gl.deleteTexture(nextTexture);
          gl.pixelStorei(gl.UNPACK_ALIGNMENT,unpack);gl.bindTexture(gl.TEXTURE_3D,previous===oldTexture&&oldTexture!==null?texture:previous);gl.activeTexture(previousActive);
        }
        say(`Ray field received: ${next.dimensions.join(' × ')} voxels · ${Number(next.ray_jobs??0).toLocaleString()} forward rays. Existing Phong volume glow; not directional lighting.`);
      }finally{clearTimeout(timer);if(pending===work)pending=null;}
    }catch(error){
      if(source===s&&s.epoch===epoch&&error.name!=='AbortError'){drop();say(`Field unavailable: ${error.message}`);}
    }
  };
  const bind=(program,view,unitM,origin)=>{
    const uniform=name=>gl.getUniformLocation(program,name);
    gl.uniform1f(uniform('uFieldGain'),0);gl.uniform1i(uniform('uFieldUseTransform'),0);
    if(!texture||!source)return false;
    try {
      if(!current(source)){drop();return false;}
      if(uniform('uFieldFromView')===null||uniform('uFieldUseTransform')===null)throw Error('This shader export does not support field coordinates yet.');
      const previous=gl.getParameter(gl.ACTIVE_TEXTURE);gl.activeTexture(gl.TEXTURE8);gl.bindTexture(gl.TEXTURE_3D,texture);gl.activeTexture(previous);
      gl.uniformMatrix4fv(uniform('uFieldFromView'),false,fieldFromView(view,unitM,origin,metadata.bounds_min_m,metadata.bounds_max_m));
      gl.uniform1i(uniform('uFieldVolume'),8);gl.uniform1i(uniform('uFieldUseTransform'),1);gl.uniform1f(uniform('uFieldGain'),1);
      return true;
    }catch(error){drop();say(`Field unavailable: ${error.message}`);return false;}
  };
  const change=()=>{invalidate();say(requested()?'Opted in for the next RGB still or Live job.':'Ray field off. Ordinary Phong lighting is unchanged.');onChange();};
  toggle?.addEventListener('change',change);
  return {requested,capture,connect,observe,bind,invalidate,
    revise(digest,revision){if(source?.digest!==digest||source.revision===revision)return;
      const context=source.context;invalidate('Camera updated. Waiting for matching field data.');connect(digest,capture(context),revision);},
    stop(digest){if(source?.digest===digest)invalidate('Ray field stopped.');},
    get metadata(){return metadata&&structuredClone(metadata);},
    dispose(){disposed=true;invalidate();toggle?.removeEventListener('change',change);}};
}

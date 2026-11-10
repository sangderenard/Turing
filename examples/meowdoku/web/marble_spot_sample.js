// Measurements from a completed, profiled HDR. No PNG readback, ray launch,
// sensor inversion, physical exposure change or automatic retouch happens here.
import {parseNpz} from './npz.js';

export const SPOT_LUMA = Object.freeze([.2126390058715103,.7151686787677559,.0721923153607337]);
export const MAX_SPOT_PIXELS = 4194304;
export const MAX_SPOT_HDR_BYTES = 64 * 1024 * 1024;
const canceled = signal => {if (signal?.aborted) throw new Error('Spot measurement canceled.');};
const finite = (x, low, high, name) => {
  if (typeof x !== 'number' || !Number.isFinite(x) || x < low || x > high) throw new RangeError(`Invalid ${name}.`);
  return x;
};

// Sigma is a fraction of the shorter image dimension. Zero means one existing
// accumulated pixel, never a single ray. Gaussian support is truncated at 3σ
// in each axis, clipped to the actual image, and renormalized at its edges.
export function validateSpotSpread(value = 0) {return finite(value, 0, .5, 'Gaussian spread (0–0.5)');}

function inspectArray(array) {
  const shape = array?.shape;
  if (!Array.isArray(shape) || shape.length !== 3 || shape[2] !== 3 ||
      !shape.slice(0,2).every(n => Number.isSafeInteger(n) && n > 0) ||
      shape[0] * shape[1] > MAX_SPOT_PIXELS || !['<f4','<f8'].includes(array.dtype) ||
      array.data?.length !== shape[0] * shape[1] * 3)
    throw new Error('Expected bounded H×W×3 floating profiled linear sRGB.');
  return {height:shape[0],width:shape[1],data:array.data};
}

export async function readSpotHdr(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength > MAX_SPOT_HDR_BYTES)
    throw new Error('Completed HDR exceeds the spot-sampling byte limit.');
  const arrays = await parseNpz(buffer, ['linear_srgb_exposure']);
  const array = arrays.get('linear_srgb_exposure');
  if (!array) throw new Error('This HDR has no profiled linear_srgb_exposure. Raw sensor channels are not interchangeable.');
  inspectArray(array);
  return array;
}

export function sampleSpotHdr(array, {u, v, spread = 0, signal} = {}) {
  finite(u,0,1,'reticle horizontal position');finite(v,0,1,'reticle vertical position');
  validateSpotSpread(spread);canceled(signal);
  const {width,height,data} = inspectArray(array);
  const cx = Math.min(width-1, Math.floor(u*width)), cy = Math.min(height-1, Math.floor(v*height));
  const sigma = spread*Math.min(width,height), radius = sigma ? Math.ceil(3*sigma) : 0;
  const bounds = [Math.max(0,cx-radius),Math.max(0,cy-radius),Math.min(width-1,cx+radius),Math.min(height-1,cy+radius)];
  const sum=[0,0,0];let weights=0,pixels=0,negativePixels=0;
  for(let y=bounds[1];y<=bounds[3];y++) {
    canceled(signal);
    for(let x=bounds[0];x<=bounds[2];x++) {
      const dx=x-cx,dy=y-cy;
      if(sigma && (Math.abs(dx)>3*sigma || Math.abs(dy)>3*sigma))continue;
      const weight=sigma?Math.exp(-.5*(dx*dx+dy*dy)/(sigma*sigma)):1;
      const offset=(y*width+x)*3, rgb=[data[offset],data[offset+1],data[offset+2]];
      if(!rgb.every(Number.isFinite))throw new Error('HDR sample contains nonfinite values; no adjustment proposed.');
      rgb.forEach((value,i)=>{sum[i]+=weight*value;});
      if(rgb.some(value=>value<0))negativePixels++;
      weights+=weight;pixels++;
    }
  }
  const rgb=sum.map(value=>value/weights),luminance=rgb.reduce((value,x,i)=>value+x*SPOT_LUMA[i],0);
  if(![...rgb,luminance].every(Number.isFinite))throw new Error('HDR sample is outside finite measurement range.');
  return {rgb,luminance,pixelCount:pixels,negativePixels,center:[cx,cy],bounds,width,height,
    spread,sigmaPixels:sigma,weightSum:weights,source:'profiled-linear-srgb-completed-hdr',
    sampling:sigma?'Gaussian-weighted accumulated pixels':'One accumulated pixel',
    limitations:'Existing accumulated sensor samples, not individual rays. No clipping or display tone mapping was applied.'};
}

// Returns a proposal, never writes controls or sends a retouch. Exposure means
// display EV in the existing retouch pipeline, not shutter time/sensor energy.
// whiteReference is the actual recorded/current display white, not a guessed
// percentile. targetFraction is a declared scene-linear metering convention.
export function proposeSpotAdjustment(sample, {kind,params={},whiteReference,targetFraction=.18} = {}) {
  if(!Array.isArray(sample?.rgb)||sample.rgb.length!==3||!sample.rgb.every(Number.isFinite))throw new Error('A valid HDR sample is required.');
  const rgb=sample.rgb,Y=rgb.reduce((s,x,i)=>s+x*SPOT_LUMA[i],0);
  if(kind==='white') {
    if(rgb.some(x=>x<=0)||Y<=0)throw new Error('Neutral-point correction requires three positive sampled channels.');
    const gains=rgb.map(x=>Y/x);
    if(gains.some(x=>!Number.isFinite(x)||x>3))throw new Error('Neutral-point correction exceeds the existing 0–3 RGB gain range.');
    return {kind,params:{red:gains[0],green:gains[1],blue:gains[2],whitebalance:0,tint:0},
      explanation:'Treat the selected point as neutral. Replace RGB/white-balance/tint gains, preserve its linear luminance, and leave exposure unchanged. Not a calibrated illuminant measurement.'};
  }
  if(kind!=='exposure')throw new Error('Choose exposure or white-point sampling.');
  finite(whiteReference,1e-6,1e12,'recorded display white');finite(targetFraction,.001,1,'linear target fraction');
  const wb=2**finite(params.whitebalance??0,-3,3,'white balance'),tn=2**finite(params.tint??0,-3,3,'tint');
  const gains=[finite(params.red??1,0,3,'red gain')*wb,finite(params.green??1,0,3,'green gain')/tn,finite(params.blue??1,0,3,'blue gain')/wb];
  const corrected=rgb.reduce((s,x,i)=>s+x*gains[i]*SPOT_LUMA[i],0);
  if(corrected<=0)throw new Error('Exposure metering requires positive corrected linear luminance.');
  const exposure=Math.log2(targetFraction*whiteReference/corrected);
  finite(exposure,-6,6,'proposed exposure (outside existing ±6 EV travel)');
  return {kind,params:{exposure},targetFraction,whiteReference,
    explanation:`Set display exposure to ${exposure.toFixed(2)} EV for ${(targetFraction*100).toFixed(1)}% of the recorded white before tone mapping. Shutter time and raw HDR are unchanged.`};
}

// getSource is owned by the photo controller. It must return null unless the
// displayed final frame is complete/profiled, otherwise {id,url,revision,
// complete:true,profiled:true}. id binds job+HDR filename; revision also binds
// display/frame selection. No URL is inferred from a canvas or screenshot.
export function createCompletedHdrSampler({getSource,fetchImpl=(...args)=>fetch(...args)} = {}) {
  if(typeof getSource!=='function')throw new TypeError('Completed HDR source reader required.');
  let cached=null,generation=0;
  const source=()=>{
    const s=getSource();
    if(!s||s.complete!==true||s.profiled!==true||typeof s.id!=='string'||!s.id||typeof s.url!=='string'||!s.url||
       !(typeof s.revision==='string'&&s.revision.length>0||typeof s.revision==='number'&&Number.isFinite(s.revision)))
      throw new Error('Select the completed final photo with a recorded linear-sRGB profile.');
    return {...s};
  };
  const key=s=>JSON.stringify([s.id,s.url,s.revision]);
  return {
    clear(){generation++;cached=null;},
    async sample({u,v,spread=0,signal}={}) {
      const before=source(),identity=key(before),token=generation;
      const check=()=>{canceled(signal);if(token!==generation||key(source())!==identity)throw new Error('Photo changed during spot measurement; choose the point again.');};
      check();let array=cached?.identity===identity?cached.array:null;
      if(!array){
        const response=await fetchImpl(before.url,{cache:'no-store',signal});check();
        if(!response.ok)throw new Error(`Completed HDR HTTP ${response.status}.`);
        const length=Number(response.headers?.get('Content-Length'));
        if(length>MAX_SPOT_HDR_BYTES)throw new Error('Completed HDR exceeds the spot-sampling byte limit.');
        const buffer=await response.arrayBuffer();check();array=await readSpotHdr(buffer);check();
        cached={identity,array};
      }
      if((before.width!==undefined&&before.width!==array.shape[1])||
         (before.height!==undefined&&before.height!==array.shape[0]))
        throw new Error('Completed HDR dimensions differ from the displayed image.');
      const result=sampleSpotHdr(array,{u,v,spread,signal});check();
      return {...result,sourceId:before.id,sourceRevision:before.revision};
    },
  };
}

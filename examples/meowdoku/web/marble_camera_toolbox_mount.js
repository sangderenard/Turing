import {createCameraToolbox} from './marble_camera_toolbox.js';
import {createCompletedHdrSampler,proposeSpotAdjustment} from './marble_spot_sample.js';
import {createRangefinderPanel} from './marble_rangefinder_panel.js';
import {rangefinderKey} from './marble_rangefinder.js';

// Matches the existing photo-panel CSS: borderless/padding-free canvas,
// centered object-fit:contain inside an aspect-ratio:1 box. Parent figure
// padding/border are outside getBoundingClientRect and are not image pixels.
export function containedCanvasRect(canvas) {
  const rect=canvas.getBoundingClientRect(),width=canvas.width,height=canvas.height;
  if(!(width>0&&height>0&&rect.width>0&&rect.height>0))throw new Error('The photo image is not decoded or visible.');
  const scale=Math.min(rect.width/width,rect.height/height),w=width*scale,h=height*scale;
  return {left:rect.left+(rect.width-w)/2,top:rect.top+(rect.height-h)/2,width:w,height:h};
}

// One opt-in initializer for the Studio bootstrap. Focus stays disabled until
// the actual full mixed-geometry reader exists. Photo metering reads saved HDR,
// presents an exact proposal, and writes only on the separate Apply click.
export function mountStudioCameraToolbox({doc=document,photo,canvas=doc.getElementById('view'),
    photoCanvas=doc.getElementById('photo-image'),mount=doc.getElementById('photo-controls'),getScene,createRangeBackend}={}) {
  if(!photo?.completedHdr||!photo?.subscribeCompletedHdr||!photo?.applySpotAdjustment||!photoCanvas)
    throw new Error('Camera tools need the actual completed-photo controller and image surface.');
  const sampler=createCompletedHdrSampler({getSource:()=>photo.completedHdr()});
  const make=(tag,text)=>{const el=doc.createElement(tag);el.textContent=text;return el;};
  const proposalText=make('p','No adjustment proposed. Choose a point in a completed HDR photo.');proposalText.id='camera-tool-proposal';
  proposalText.style.cssText='margin:0;font-size:12px';
  const apply=make('button','Apply proposed photo adjustment');apply.id='camera-tool-apply';apply.type='button';apply.disabled=true;
  let pendingProposal=null,lastKey=null;
  const rangePanel=typeof getScene==='function'?createRangefinderPanel({doc,photo,getScene,createBackend:createRangeBackend}):null;
  const surfaces=[{element:photoCanvas,kinds:['exposure','white'],label:'the completed photo',
    bounds:()=>containedCanvasRect(photoCanvas),revision:()=>photo.completedHdr()?.revision??null}];
  if(canvas)surfaces.push({element:canvas,kinds:['focus','range'],label:'Studio',revision:()=>{
    if(!getScene)return null;try{return rangefinderKey(getScene());}catch{return null;}
  }});
  const toolbox=createCameraToolbox({doc,mount,surfaces,
    range:rangePanel?.read,
    sample:async request=>{
      const source=photo.completedHdr();
      if(!source)throw new Error('Select a completed final profiled photo.');
      const measurement=await sampler.sample(request);
      const proposal=proposeSpotAdjustment(measurement,{kind:request.kind,params:source.params,whiteReference:source.whiteReference});
      return {...measurement,proposal,message:`${measurement.sampling}: linear RGB ${measurement.rgb.map(x=>x.toPrecision(5)).join(', ')}; luminance ${measurement.luminance.toPrecision(5)}. Review the proposal before applying.`};
    },
    onInvalidate:()=>{rangePanel?.invalidate();pendingProposal=null;apply.disabled=true;proposalText.textContent='No pending adjustment. Sample a completed photo to propose one.';},
    onResult:(result,request)=>{
      if(request.kind==='range'){rangePanel.accept(result);return;}
      pendingProposal={proposal:result.proposal,sourceId:result.sourceId,sourceRevision:result.sourceRevision};
      proposalText.textContent=result.proposal.explanation;apply.disabled=false;
    }});
  toolbox.root.append(proposalText,apply);
  if(rangePanel)toolbox.root.append(rangePanel.root);
  const refresh=source=>{
    const key=source?`${source.id}/${source.revision}`:null;if(key===lastKey)return;lastKey=key;
    pendingProposal=null;apply.disabled=true;proposalText.textContent='No adjustment proposed. Choose a point in a completed HDR photo.';
    const white=source?.whiteReference,canExpose=!!source&&Number.isFinite(white)&&white>=1e-6&&white<=1e12;
    toolbox.setCapabilities({
      range:{available:!!rangePanel,description:'All current Studio mesh intersections along one straight ray; no refraction.'},
      focus:{available:false,reason:'Full-scene refractive spot focus is not available. The camera-target distance is not a surface measurement.'},
      exposure:{available:canExpose,reason:source?'This photo has no valid recorded display-white reference.':'Select the completed final profiled photo.'},
      white:{available:!!source,reason:'Select the completed final profiled photo.'}});
  };
  // Force the initial disabled reasons even when no exposure has completed.
  lastKey=Symbol('uninitialized');const unsubscribe=photo.subscribeCompletedHdr(refresh);
  const applyClick=()=>{
    if(!pendingProposal)return;
    const selected=pendingProposal;
    try{photo.applySpotAdjustment(selected.proposal,selected);pendingProposal=null;apply.disabled=true;
      proposalText.textContent='Proposed display adjustment sent through the existing photo retouch controls. Raw HDR is unchanged.';
    }catch(error){pendingProposal=null;apply.disabled=true;proposalText.textContent=error.message;}
  };
  apply.addEventListener('click',applyClick);
  return {toolbox,dispose(){unsubscribe();sampler.clear();rangePanel?.dispose();apply.removeEventListener('click',applyClick);toolbox.dispose();}};
}

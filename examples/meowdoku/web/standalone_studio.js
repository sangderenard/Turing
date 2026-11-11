// Canvas-only Studio full-scene and explicit subject-only runners. No UI controls, event listeners, clock,
// game engine or render loop. The caller explicitly supplies each camera state.
import {createStudioSubjectRenderer,validateStudioSubjectPacket} from './marble_studio_subject_renderer.js';
import {hydrateStudioSceneAsset,createStudioSceneRenderer} from './marble_studio_scene_asset.js';
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const matrix=(value,label)=>{
 if(!Array.isArray(value)||value.length!==16||!value.every(Number.isFinite))throw Error(label+' requires 16 finite matrix values');
 return [...value];
};
export function validateStudioCameraState(state){
 if(!state||Object.keys(state).some(k=>!['view','projection','width','height'].includes(k)))throw Error('Only camera matrices and frame dimensions are supported');
 const view=matrix(state.view,'view'),projection=matrix(state.projection,'projection');
 if(view[3]!==0||view[7]!==0||view[11]!==0||view[15]!==1)throw Error('view must be affine');
 for(const key of ['width','height'])if(!Number.isInteger(state[key])||state[key]<16||state[key]>2048)throw Error(key+' must be 16..2048');
 return {view,projection,width:state.width,height:state.height};
}
export function createStandaloneStudio({canvas,packet,samplers,model=I,lightTransform=I}){
 // No inferred samplers/materials: require the same explicit captured resource contract.
 packet=structuredClone(packet);samplers=structuredClone(samplers);
 validateStudioSubjectPacket(packet,samplers);
 const sceneModel=matrix(model,'model'),sceneLightTransform=matrix(lightTransform,'lightTransform');
 const gl=canvas.getContext('webgl2',{antialias:false,preserveDrawingBuffer:true,alpha:false});
 if(!gl)throw Error('Studio requires WebGL2');
 const renderer=createStudioSubjectRenderer({gl,packet,samplers});
 return standaloneHost(canvas,gl,renderer,camera=>renderer.draw({view:camera.view,projection:camera.projection,model:sceneModel,lightTransform:sceneLightTransform}));
}
// Full Studio scene, including captured room, fixtures and subject.
export async function createStandaloneStudioScene({canvas,sceneAsset}){
 const asset=await hydrateStudioSceneAsset(structuredClone(sceneAsset));
 const gl=canvas.getContext('webgl2',{antialias:false,preserveDrawingBuffer:true,alpha:false});
 if(!gl)throw Error('Studio requires WebGL2');
 const renderer=createStudioSceneRenderer({gl,asset});
 return standaloneHost(canvas,gl,renderer,camera=>renderer.draw({view:camera.view,projection:camera.projection}));
}
function standaloneHost(canvas,gl,renderer,draw){
 let closed=false;
 return Object.freeze({
  render(state){
   if(closed||gl.isContextLost())throw Error('Standalone Studio is closed or context lost');
   const camera=validateStudioCameraState(state);
   canvas.width=camera.width;canvas.height=camera.height;
   gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,camera.width,camera.height);
   gl.disable(gl.SCISSOR_TEST);gl.colorMask(true,true,true,true);gl.depthMask(true);
   gl.clearColor(0,0,0,1);gl.clearDepth(1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
   const counts=draw(camera);
   gl.finish();if(gl.isContextLost()||gl.getError()!==gl.NO_ERROR)throw Error('Studio framebuffer failed');
   return {schema:'turing.standalone-studio-frame.v1',mode:renderer.mode,subject_id:renderer.subjectId??null,scene_identity:renderer.identity??null,
    revision:renderer.revision,camera,counts,data_url:canvas.toDataURL('image/png'),simulation:false};
  },
  dispose(){if(!closed){renderer.dispose();closed=true;}},
 });
}

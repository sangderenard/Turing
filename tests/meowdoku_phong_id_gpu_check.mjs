import {createPhongPrimaryIdPass} from '../examples/meowdoku/web/marble_phong_id_pass.js';
import {capturePhongTriangleSource,bindPhongVisibilityViews} from '../examples/meowdoku/web/marble_phong_visibility.js';
import {packStudioScene} from '../examples/meowdoku/web/marble_studio_scene.js';
const out=document.querySelector('#result'),canvas=document.querySelector('#view');
const assert=(test,why)=>{if(!test)throw Error(why);};
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const perspective=[1,0,0,0,0,1,0,0,0,0,-10.1/9.9,-1,0,0,-2/9.9,0];
try {
 const gl=canvas.getContext('webgl2',{antialias:false,preserveDrawingBuffer:true});
 assert(gl,'WebGL2 required');
 const [vert,frag,assets]=await Promise.all([
  fetch('../build/meowdoku/web/shaders/base_material.webgl.vert.glsl').then(r=>r.text()),
  fetch('../build/meowdoku/web/shaders/base_material.webgl.shadow.frag.glsl').then(r=>r.text()),
  fetch('../build/meowdoku/web/assets.json').then(r=>r.json())]);
 const program=gl.createProgram();
 for(const [type,source]of[[gl.VERTEX_SHADER,vert],[gl.FRAGMENT_SHADER,frag.replace('#version 300 es',
  '#version 300 es\n#define PLUCK_TRANSMISSION 1\n#define PLUCK_PREMULTIPLIED_GLASS 1')]]){
  const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);
  assert(gl.getShaderParameter(shader,gl.COMPILE_STATUS),gl.getShaderInfoLog(shader));gl.attachShader(program,shader);
 }
 gl.linkProgram(program);assert(gl.getProgramParameter(program,gl.LINK_STATUS),gl.getProgramInfoLog(program));gl.useProgram(program);
 const loc=(p,n)=>gl.getUniformLocation(p,n);
 for(const [index,name]of ['pbr','phong','enamel','texstack'].entries()){
  const table=assets.materials[name],width=256,height=Math.ceil(table.data.length/256),data=new Float32Array(width*height);data.set(table.data);
  gl.activeTexture(gl.TEXTURE0+index);gl.bindTexture(gl.TEXTURE_2D,gl.createTexture());
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.R32F,width,height,0,gl.RED,gl.FLOAT,data);gl.uniform1i(loc(program,'turing_ssbo_'+name),index);
 }
 for(const [unit,name,pixel]of[[4,'uEmitUv',[0,255,128,255]],[5,'uColorUv',[255,255,255,255]],
  [6,'uDepthUv',[0,0,0,255]],[7,'uRemitUv',[0,0,0,255]],[9,'uShadowMaps',[255,255,255,255]]]){
  gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D_ARRAY,gl.createTexture());
  gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
  gl.texImage3D(gl.TEXTURE_2D_ARRAY,0,gl.RGBA8,1,1,2,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([...pixel,...pixel]));gl.uniform1i(loc(program,name),unit);
 }
 gl.activeTexture(gl.TEXTURE0+8);gl.bindTexture(gl.TEXTURE_3D,gl.createTexture());
 gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
 gl.texImage3D(gl.TEXTURE_3D,0,gl.RGBA8,1,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array(4));gl.uniform1i(loc(program,'uFieldVolume'),8);
 gl.activeTexture(gl.TEXTURE0+10);gl.bindTexture(gl.TEXTURE_2D,gl.createTexture());
 gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
 gl.texImage2D(gl.TEXTURE_2D,0,gl.R32F,256,1,0,gl.RED,gl.FLOAT,new Float32Array(256));gl.uniform1i(loc(program,'turing_ssbo_thermal_k'),10);
 gl.uniformMatrix4fv(loc(program,'uMV'),false,I);gl.uniformMatrix4fv(loc(program,'uMVP'),false,perspective);
 gl.uniformMatrix3fv(loc(program,'uCatCcmMatrix'),false,[1,0,0,0,1,0,0,0,1]);
 gl.uniform1f(loc(program,'uExposure'),1);gl.uniform1f(loc(program,'uAmbientFloorGain'),1);gl.uniform1i(loc(program,'uRenderPass'),0);
 gl.uniform1i(loc(program,'uNumLights'),2);gl.uniform3fv(loc(program,'uLightPos[0]'),[1,2,3,4,5,6]);
 gl.uniform3fv(loc(program,'uLightColor[0]'),[1,1,1,.5,.6,.7]);gl.uniform1fv(loc(program,'uLightIntensity[0]'),[.5,.25]);
 gl.vertexAttribI4i(5,0,0,0,0);
 const opaque=assets.material_index.ceramic_porcelain,glass=assets.material_index.marble_clear_glass;
 const face=(a,b,c,id=opaque)=>[a,b,c,id,[[0,0,1],[0,0,1],[0,0,1]],7];
 const quad=(s,z)=>[face([-s,-s,z],[s,-s,z],[s,s,z]),face([-s,-s,z],[s,s,z],[-s,s,z])];
 const faces=[...quad(.6,-1),...quad(1.2,-2),face([-.1,-.1,-.5],[.1,-.1,-.5],[0,.1,-.5],glass),
  face([.65,.65,-1],[.65,.9,-1],[.9,.65,-1])]; // Reverse winding: original discard.
 const capture={revision:1,state:packStudioScene({channel:'studio',unitM:1,rayOriginM:[0,0,0],parts:[],
  pbr:{stride:assets.materials.pbr.floats_per_record,data:assets.materials.pbr.data},
  subject:{id:'actual-shader-fixture',faces,model:I},camera:{view:I,projection:perspective},room:null})};
 const source=await capturePhongTriangleSource(capture);
 const [view]=await bindPhongVisibilityViews([{index:0,direction:[0,0,1],condition_key:'fixed-test-materials',
  camera:{eye_m:[0,0,0],target_m:[0,0,-1],fov_degrees:90,projection:perspective,view:I}}]);
 gl.viewport(3,4,51,52);gl.disable(gl.DEPTH_TEST);gl.depthMask(false);gl.enable(gl.BLEND);
 const oldActive=gl.getParameter(gl.ACTIVE_TEXTURE),pass=createPhongPrimaryIdPass({gl,program});
 const sample=await pass.sample({source,view,width:96,height:96,
  configure:p=>{
   assert(JSON.stringify(Array.from(gl.getUniform(p,loc(p,'uLightPos[1]'))))==='[4,5,6]','All original uniform-array elements must survive');
   assert(gl.getUniform(p,loc(p,'uLightIntensity[1]'))===.25,'Scalar uniform-array element preserved');
   gl.uniformMatrix4fv(loc(p,'uMV'),false,I);gl.uniformMatrix4fv(loc(p,'uMVP'),false,perspective);}});
 assert(JSON.stringify(sample.triangleIds)===JSON.stringify([0,1]),'Only front opaque triangles should survive, got '+sample.triangleIds);
 assert(sample.retainedTransparentSourceIndices?.includes(4),'Glass must remain in the unpruned transparent source');
 assert(gl.getParameter(gl.CURRENT_PROGRAM)===program,'program restored');
 assert(gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING)===null,'framebuffer restored');
 assert(!gl.isEnabled(gl.DEPTH_TEST)&&!gl.getParameter(gl.DEPTH_WRITEMASK)&&gl.isEnabled(gl.BLEND),'depth/blend restored');
 assert(gl.getParameter(gl.ACTIVE_TEXTURE)===oldActive,'active texture restored');
 assert(JSON.stringify(Array.from(gl.getParameter(gl.VIEWPORT)))==='[3,4,51,52]','viewport restored');
 assert(gl.getError()===gl.NO_ERROR,'No GPU errors');
 pass.dispose();
 out.textContent=JSON.stringify({pass:true,renderer:gl.getParameter(gl.RENDERER),triangleIds:sample.triangleIds,
  hiddenOpaque:[2,3],glassExcludedFromOpaque:4,backfaceDiscarded:5,producer:sample.producer,
  sample,fullScenePruning:false},null,2);
}catch(error){out.textContent=JSON.stringify({pass:false,error:String(error),stack:error.stack},null,2);}

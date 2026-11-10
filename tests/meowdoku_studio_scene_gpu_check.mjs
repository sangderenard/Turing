import {studioFlatSamplerPacket} from '../examples/meowdoku/web/marble_studio_subject_renderer.js';
import {hydrateStudioSceneAsset,createStudioSceneRenderer,studioSceneDrawPlan} from '../examples/meowdoku/web/marble_studio_scene_asset.js';
import {triangleCentres,transparentTriangleOrder} from '../examples/meowdoku/web/marble_glass_preview.js';
const out=document.querySelector('#result'),canvas=document.querySelector('#view'),assert=(x,m)=>{if(!x)throw Error(m);};
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const multiply=(a,b)=>{const m=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)m[c*4+r]+=a[k*4+r]*b[c*4+k];return m;};
try {
 const assetURL=new URLSearchParams(window.location.search).get('asset')||'../build/dot-studio-game-transfer-20261110/complete-studio.scene-asset.json';
 const response=await fetch(assetURL);assert(response.ok,'Asset HTTP '+response.status);
 const asset=await hydrateStudioSceneAsset(await response.json()),packet=asset.packet;
 const gl=canvas.getContext('webgl2',{antialias:false,preserveDrawingBuffer:true,alpha:false});assert(gl,'WebGL2 unavailable');
 const sampler=studioFlatSamplerPacket(),placement=[1,0,0,0,0,1,0,0,0,0,1,0,32,1.544,-18.00788329297247,1],
 entrance={partId:'room/front',widthM:2,heightM:2.4},plan=studioSceneDrawPlan(packet,{placement,entrance,samplers:sampler});
 // Baseline uses the real current Studio upload/sort/draw functions verbatim,
 // scoped to this independent test context, not a reimplementation of materials.
 const source=await fetch('../examples/meowdoku/web/marble_studio.js').then(r=>r.text());
 const extract=name=>{const start=source.indexOf(`function ${name}(`);assert(start>=0,'Missing Studio '+name);
  const open=source.indexOf('{',start);let depth=1,i=open+1;for(;depth;i++){if(source[i]==='{')depth++;if(source[i]==='}')depth--;}
  return source.slice(start,i);};
 const canonical=Function('gl','triangleCentres','transparentTriangleOrder',
  [extract('mesh'),extract('transparentMesh'),extract('drawTransparent')].join('\n')+'\nreturn {mesh,transparentMesh,drawTransparent};')(gl,triangleCentres,transparentTriangleOrder);
 const program=gl.createProgram();
 for(const [name,type]of [['vertex',gl.VERTEX_SHADER],['fragment',gl.FRAGMENT_SHADER]]) {
  const s=gl.createShader(type);gl.shaderSource(s,packet.render.shaders[name]);gl.compileShader(s);
  assert(gl.getShaderParameter(s,gl.COMPILE_STATUS),gl.getShaderInfoLog(s));gl.attachShader(program,s);
 }
 gl.linkProgram(program);assert(gl.getProgramParameter(program,gl.LINK_STATUS),gl.getProgramInfoLog(program));gl.useProgram(program);
 const location=n=>gl.getUniformLocation(program,n);
 for(const u of packet.render.uniforms) {
  const loc=location(u.name),v=u.value;
  switch(u.type) {
   case gl.FLOAT:gl.uniform1f(loc,v);break;
   case gl.FLOAT_VEC2:gl.uniform2fv(loc,v);break;
   case gl.FLOAT_VEC3:gl.uniform3fv(loc,v);break;
   case gl.FLOAT_VEC4:gl.uniform4fv(loc,v);break;
   case gl.FLOAT_MAT3:gl.uniformMatrix3fv(loc,false,v);break;
   case gl.FLOAT_MAT4:gl.uniformMatrix4fv(loc,false,v);break;
   case gl.BOOL:case gl.INT:case gl.SAMPLER_2D:case gl.SAMPLER_2D_ARRAY:case gl.SAMPLER_3D:gl.uniform1i(loc,Number(v));break;
   default:throw Error('Fixture unsupported actual uniform '+u.name+' type '+u.type);
  }
 }
 const textures=[];
 const tex=row=>{const target=gl[row.target],texture=gl.createTexture();textures.push({target,texture,unit:row.unit});
  gl.activeTexture(gl.TEXTURE0+row.unit);gl.bindTexture(target,texture);
  gl.texParameteri(target,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(target,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
  if(row.data===null)gl.texStorage3D(target,1,gl[row.internalFormat],row.width,row.height,row.depth);
  else {const data=row.type==='FLOAT'?new Float32Array(row.data):new Uint8Array(row.data);
   if(row.target==='TEXTURE_2D')gl.texImage2D(target,0,gl[row.internalFormat],row.width,row.height,0,gl[row.format],gl[row.type],data);
   else gl.texImage3D(target,0,gl[row.internalFormat],row.width,row.height,row.depth,0,gl[row.format],gl[row.type],data);}
 };
 for(const [unit,name]of ['pbr','phong','enamel','texstack'].entries()) {
  const values=packet.render.materials[name].data,width=256,height=Math.ceil(values.length/256),data=new Float32Array(width*height);data.set(values);
  tex({unit,target:'TEXTURE_2D',internalFormat:'R32F',format:'RED',type:'FLOAT',width,height,data});
 }
 for(const row of sampler.textures)tex(row);
 const groups=plan.groups.filter(g=>g.faces.length).map(g=>({...g,mesh:g.phase==='transparent'?canonical.transparentMesh(g.faces):canonical.mesh(g.faces)}));
 const renderer=createStudioSceneRenderer({gl,asset,placement,entrance});
 // Camera inside the transferred room, facing the subject and source fixtures.
 const view=[1,0,0,0,0,1,0,0,0,0,1,0,-32,-1,-(-12),1],f=1/Math.tan(55*Math.PI/360),
   projection=[f,0,0,0,0,f,0,0,0,0,-100.02/99.98,-1,0,0,-4/99.98,0];
 const inverse=m=>{const a=m[0],b=m[4],c=m[8],d=m[1],e=m[5],f=m[9],g=m[2],h=m[6],i=m[10],det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g),
   o=[(e*i-f*h)/det,(f*g-d*i)/det,(d*h-e*g)/det,0,(c*h-b*i)/det,(a*i-c*g)/det,(b*g-a*h)/det,0,(b*f-c*e)/det,(c*d-a*f)/det,(a*e-b*d)/det,0,0,0,0,1];
   for(let r=0;r<3;r++)o[12+r]=-(o[r]*m[12]+o[4+r]*m[13]+o[8+r]*m[14]);return o;};
 const lightView=multiply(multiply(view,plan.model),inverse(packet.render.referenceCamera.view)),u=new Map(packet.render.uniforms.map(x=>[x.name,x.value]));
 const bindBaseline=()=>{gl.useProgram(program);for(const t of textures){gl.activeTexture(gl.TEXTURE0+t.unit);gl.bindSampler(t.unit,null);gl.bindTexture(t.target,t.texture);}
   for(let i=0;i<u.get('uNumLights');i++){
     for(const [stem,w]of [['uLightPos',1],['uLightSpotDir',0]]){const v=u.get(`${stem}[${i}]`),q=[0,1,2].map(r=>lightView[r]*v[0]+lightView[4+r]*v[1]+lightView[8+r]*v[2]+lightView[12+r]*w),len=Math.hypot(...q);
       gl.uniform3fv(location(`${stem}[${i}]`),w||!len?q:q.map(x=>x/len));}
     gl.uniform1f(location(`uLightIntensity[${i}]`),u.get(`uLightIntensity[${i}]`)*plan.model[0]**2);
   }
   gl.vertexAttrib2f(3,0,0);gl.vertexAttribI4i(5,1,0,0,0);gl.vertexAttrib3f(6,0,0,0);
 };
 const clean=()=>{gl.viewport(0,0,canvas.width,canvas.height);gl.disable(gl.SCISSOR_TEST);gl.disable(gl.STENCIL_TEST);gl.disable(gl.BLEND);gl.disable(gl.CULL_FACE);gl.disable(gl.POLYGON_OFFSET_FILL);gl.disable(gl.RASTERIZER_DISCARD);
   gl.disable(gl.SAMPLE_ALPHA_TO_COVERAGE);gl.disable(gl.SAMPLE_COVERAGE);gl.enable(gl.DITHER);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LESS);gl.depthRange(0,1);gl.frontFace(gl.CCW);gl.colorMask(true,true,true,true);gl.depthMask(true);
   gl.clearColor(.035,.06,.09,1);gl.clearDepth(1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);};
 const pixels=()=>{const a=new Uint8Array(canvas.width*canvas.height*4);gl.readPixels(0,0,canvas.width,canvas.height,gl.RGBA,gl.UNSIGNED_BYTE,a);return a;};
 clean();bindBaseline();
 for(const group of groups){const mv=multiply(view,group.model);gl.uniformMatrix4fv(location('uMV'),false,mv);gl.uniformMatrix4fv(location('uMVP'),false,multiply(projection,mv));gl.uniform1i(location('uEnableEmissionDirect'),Number(group.emissionDirect));
   if(group.phase==='transparent'){gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.blendEquation(gl.FUNC_ADD);gl.depthMask(false);canonical.drawTransparent(group.mesh,mv);}
   else{gl.disable(gl.BLEND);gl.depthMask(true);gl.bindVertexArray(group.mesh.vao);gl.drawArrays(gl.TRIANGLES,0,group.mesh.count);}}
 const baseline=pixels();assert(gl.getError()===gl.NO_ERROR,'Baseline GL error');
 clean();gl.enable(gl.CULL_FACE);gl.enable(gl.SCISSOR_TEST);gl.scissor(3,4,9,8);gl.disable(gl.DEPTH_TEST);gl.depthMask(false);
 gl.blendFunc(gl.SRC_ALPHA,gl.ONE);gl.activeTexture(gl.TEXTURE0+5);
 const state=()=>JSON.stringify(['CULL_FACE','SCISSOR_TEST','DEPTH_TEST'].map(n=>gl.isEnabled(gl[n])).concat([gl.getParameter(gl.DEPTH_WRITEMASK),gl.getParameter(gl.ACTIVE_TEXTURE),gl.getParameter(gl.BLEND_SRC_RGB),gl.getParameter(gl.BLEND_DST_RGB)]));
 const before=state();renderer.draw({view,projection});assert(before===state(),'Original game state not restored');const candidate=pixels();
 let changed=0,maxDelta=0,lit=0;for(let i=0;i<candidate.length;i++){const d=Math.abs(candidate[i]-baseline[i]);if(d)changed++;maxDelta=Math.max(maxDelta,d);}
 for(let i=0;i<baseline.length;i+=4)if(baseline[i]!==9||baseline[i+1]!==15||baseline[i+2]!==23)lit++;
 assert(lit>1000,'Whole Studio scene not visible');assert(changed===0,`Scene pixel mismatch ${changed}, max ${maxDelta}`);
 clean();renderer.draw({view,projection,phase:'opaque'});renderer.draw({view,projection,phase:'transparent'});assert(pixels().every((x,i)=>x===candidate[i]),'Scene split phases differ');
 clean();renderer.draw({view,projection});assert(pixels().every((x,i)=>x===candidate[i]),'Scene repeat differs');assert(gl.getError()===gl.NO_ERROR,'Scene renderer GL error');
 // A fixed game camera must re-sort moving subject glass with static furniture.
 // Compare range streaming against fresh native planning at each exact pose.
 let poseFrames=0;
 for(const shift of [-1.5,2.25,0]) {
   const pose=Array.from(packet.scene.subject.model);pose[12]+=shift;pose[14]-=shift;
   const other={...asset,packet:{...packet,scene:{...packet.scene,subject:{...packet.scene.subject,model:pose}}}};
   const reference=createStudioSceneRenderer({gl,asset:other,placement,entrance});
   clean();reference.draw({view,projection});const expected=pixels();
   const oldBuffer=gl.getParameter(gl.ARRAY_BUFFER_BINDING);renderer.setSubjectPose(pose);
   assert(gl.getParameter(gl.ARRAY_BUFFER_BINDING)===oldBuffer,'Pose update leaked buffer binding');
   clean();renderer.draw({view,projection,phase:'opaque'});renderer.draw({view,projection,phase:'transparent'});
   assert(pixels().every((x,i)=>x===expected[i]),'Streamed pose differs from full native scene planning');
   assert(gl.getError()===gl.NO_ERROR,'Pose streaming GL error');reference.dispose();poseFrames++;
 }
 renderer.dispose();assert(!renderer.ready,'Disposed scene still ready');
 out.textContent=JSON.stringify({pass:true,poseFrames,counts:renderer.counts,canonicalStudioPixelDifferences:changed,maxDelta,visiblePixels:lit,fixtureDirectEmission:true,sharedGlassSort:true,splitPhasesMatch:true,restoredState:true,sourceDoorway:renderer.entrance.sourceBounds},null,2);
}catch(error){out.textContent=JSON.stringify({pass:false,error:String(error),stack:error.stack},null,2);}

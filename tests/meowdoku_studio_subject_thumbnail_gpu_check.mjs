import {renderStudioSubjectThumbnail} from '../examples/meowdoku/web/marble_studio_subject_thumbnail.js';
import {studioFlatSamplerPacket} from '../examples/meowdoku/web/marble_studio_subject_renderer.js';
const out=document.querySelector('#result'),assert=(x,m)=>{if(!x)throw Error(m);};
try {
 const path=new URLSearchParams(window.location.search).get('asset')||'../build/dot-studio-game-transfer-20261110/studio-marble-appearance.json',
  response=await fetch(path);assert(response.ok,'Asset HTTP '+response.status);const packet=await response.json();
 const gl=document.querySelector('#view').getContext('webgl2',{antialias:false});assert(gl,'WebGL2 required');
 const target=()=>{const f=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,f);
  for(let i=0;i<2;i++){const b=gl.createRenderbuffer();gl.bindRenderbuffer(gl.RENDERBUFFER,b);gl.renderbufferStorage(gl.RENDERBUFFER,gl.RGBA8,32,32);
   gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0+i,gl.RENDERBUFFER,b);}
  assert(gl.checkFramebufferStatus(gl.FRAMEBUFFER)===gl.FRAMEBUFFER_COMPLETE,'Incumbent fixture target incomplete');return f;};
 const draw=target(),read=target();gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,draw);gl.drawBuffers([gl.NONE,gl.COLOR_ATTACHMENT1]);
 gl.bindFramebuffer(gl.READ_FRAMEBUFFER,read);gl.readBuffer(gl.COLOR_ATTACHMENT1);
 const pack=gl.createBuffer();gl.bindBuffer(gl.PIXEL_PACK_BUFFER,pack);gl.bufferData(gl.PIXEL_PACK_BUFFER,2048,gl.STREAM_READ);
 gl.pixelStorei(gl.PACK_ALIGNMENT,1);gl.pixelStorei(gl.PACK_ROW_LENGTH,7);gl.pixelStorei(gl.PACK_SKIP_PIXELS,2);gl.pixelStorei(gl.PACK_SKIP_ROWS,3);
 gl.viewport(3,5,17,19);gl.clearColor(.2,.3,.4,.5);gl.clearDepth(.3);gl.colorMask(false,true,false,true);gl.depthMask(false);
 gl.enable(gl.SCISSOR_TEST);gl.scissor(1,2,4,5);gl.enable(gl.RASTERIZER_DISCARD);gl.activeTexture(gl.TEXTURE0+13);
 const snapshot=()=>{
  const names=['DRAW_FRAMEBUFFER_BINDING','READ_FRAMEBUFFER_BINDING','RENDERBUFFER_BINDING','VIEWPORT','READ_BUFFER','DRAW_BUFFER0','DRAW_BUFFER1',
   'COLOR_CLEAR_VALUE','DEPTH_CLEAR_VALUE','COLOR_WRITEMASK','DEPTH_WRITEMASK','PIXEL_PACK_BUFFER_BINDING',
   'PACK_ALIGNMENT','PACK_ROW_LENGTH','PACK_SKIP_PIXELS','PACK_SKIP_ROWS','ACTIVE_TEXTURE','CURRENT_PROGRAM','VERTEX_ARRAY_BINDING','ARRAY_BUFFER_BINDING','SCISSOR_BOX'];
  return [...names.map(n=>{const v=gl.getParameter(gl[n]);return ArrayBuffer.isView(v)?Array.from(v):v;}),gl.isEnabled(gl.SCISSOR_TEST),gl.isEnabled(gl.RASTERIZER_DISCARD)];
 };
 const same=(a,b)=>a.length===b.length&&a.every((v,i)=>Array.isArray(v)?JSON.stringify(v)===JSON.stringify(b[i]):v===b[i]);
 const before=snapshot(),result=renderStudioSubjectThumbnail({gl,packet,samplers:studioFlatSamplerPacket()});
 assert(same(before,snapshot()),'Same-context thumbnail changed incumbent GL state');assert(result.dataUrl.startsWith('data:image/png;base64,'),'PNG encoding failed');
 const image=document.querySelector('#thumbnail');image.src=result.dataUrl;await image.decode();assert(image.naturalWidth===128&&image.naturalHeight===128,'Wrong image size');
 const readCanvas=document.createElement('canvas');readCanvas.width=128;readCanvas.height=128;const ctx=readCanvas.getContext('2d');ctx.drawImage(image,0,0);
 const pixels=ctx.getImageData(0,0,128,128).data;let changed=0,edgeChanged=0;
 for(let y=0;y<128;y++)for(let x=0;x<128;x++) {
  const i=(y*128+x)*4,diff=pixels[i]!==pixels[0]||pixels[i+1]!==pixels[1]||pixels[i+2]!==pixels[2];
  if(diff){changed++;if(x===0||y===0||x===127||y===127)edgeChanged++;}assert(pixels[i+3]===255,'Incomplete opaque readback');
 }
 assert(changed>1000,'Thumbnail subject was not visible');assert(edgeChanged===0,'Thumbnail clipped its subject');
 const again=renderStudioSubjectThumbnail({gl,packet,samplers:studioFlatSamplerPacket()});
 assert(again.dataUrl===result.dataUrl,'Repeated thumbnail pixels differ');assert(same(before,snapshot()),'Repeated thumbnail altered GL state');
 assert(gl.getError()===gl.NO_ERROR,'Thumbnail GL error');
 out.textContent=JSON.stringify({pass:true,width:128,height:128,visiblePixels:changed,edgeChanged,
  sameGameContext:true,incumbentFramebufferReadDrawSelectorsRestored:true,clearViewportPackStateRestored:true,
  repeatMatches:true,counts:result.counts,rayTraceBake:false},null,2);
}catch(error){out.textContent=JSON.stringify({pass:false,error:String(error),stack:error.stack},null,2);}

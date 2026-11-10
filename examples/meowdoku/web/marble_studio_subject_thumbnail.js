// A thumbnail of the actual subject renderer, using the incumbent game's GL.
// No second WebGL context, alternate shader, world insertion or texture bake.
import {createStudioSubjectRenderer,validateStudioLightTransform} from './marble_studio_subject_renderer.js';
import {chaseViewMatrix} from './marble_chase_camera.js';

const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const finite=(x,n)=>x&&x.length===n&&Array.from(x).every(Number.isFinite);
const point=(m,p,w=1)=>[0,1,2].map(i=>m[i]*p[0]+m[i+4]*p[1]+m[i+8]*p[2]+m[i+12]*w);

export function studioSubjectThumbnailFrame({packet,model=packet?.scene?.subject?.model,lightTransform=I,width=128,height=128}={}) {
  if(!Number.isSafeInteger(width)||!Number.isSafeInteger(height)||width<16||height<16||width>512||height>512)
    throw Error('Thumbnail dimensions must be integers between 16 and 512.');
  if(!finite(model,16)||model[3]!==0||model[7]!==0||model[11]!==0||model[15]!==1)throw Error('Thumbnail model must be finite and affine.');
  validateStudioLightTransform(lightTransform);
  const faces=packet?.scene?.subject?.faces,reference=packet?.render?.referenceCamera,
    fov=reference?.fovDegrees,view=reference?.view;
  if(!Array.isArray(faces)||!faces.length||faces.length>200000||!finite(view,16)||!Number.isFinite(fov)||fov<=1||fov>=179)
    throw Error('Exact subject geometry and captured camera are required.');
  const lower=[Infinity,Infinity,Infinity],upper=[-Infinity,-Infinity,-Infinity];
  const visit=fn=>{for(const face of faces)for(let i=0;i<3;i++) {
    if(!finite(face[i],3))throw Error('Nonfinite subject vertex.');
    const p=point(model,face[i]);if(!p.every(Number.isFinite))throw Error('Nonfinite transformed subject vertex.');fn(p);
  }};
  visit(p=>{for(let i=0;i<3;i++){lower[i]=Math.min(lower[i],p[i]);upper[i]=Math.max(upper[i],p[i]);}});
  const center=lower.map((v,i)=>(v+upper[i])/2);let radius=0;
  visit(p=>{radius=Math.max(radius,Math.hypot(...p.map((v,i)=>v-center[i])));});
  if(!(radius>0)||!Number.isFinite(radius))throw Error('Thumbnail subject must have finite nonzero bounds.');
  // Reference camera's backward axis, rotated with the imported lighting rig.
  // Existing camera helper keeps the same Y-up convention as Studio/game.
  const backward=point(lightTransform,[view[2],view[6],view[10]],0),length=Math.hypot(...backward);
  if(!(length>0))throw Error('Captured camera direction is degenerate.');
  const direction=backward.map(v=>v/length),half=fov*Math.PI/360,aspect=width/height,
    limitingHalf=Math.min(half,Math.atan(Math.tan(half)*aspect)),distance=radius/Math.sin(limitingHalf)*1.08,
    eye=center.map((v,i)=>v+direction[i]*distance),near=Math.max(radius*.01,distance-radius*1.25),far=distance+radius*1.25,
    f=1/Math.tan(half),projection=new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]);
  return {view:chaseViewMatrix({eye,target:center}),projection,model:Array.from(model),lightTransform:Array.from(lightTransform),
    width,height,bounds:{lower,upper,center,radius},eye,near,far,fovDegrees:fov};
}

export function renderStudioSubjectThumbnail({gl,packet,samplers,model=packet?.scene?.subject?.model,lightTransform=I,width=128,height=128}={}) {
  const frame=studioSubjectThumbnailFrame({packet,model,lightTransform,width,height});
  if(!gl||gl.isContextLost()||typeof gl.createVertexArray!=='function')throw Error('Use the existing game WebGL2 context.');
  if(Math.max(width,height)>gl.getParameter(gl.MAX_RENDERBUFFER_SIZE))throw Error('Thumbnail exceeds the GPU renderbuffer limit.');
  const names=['DRAW_FRAMEBUFFER_BINDING','READ_FRAMEBUFFER_BINDING','RENDERBUFFER_BINDING','VIEWPORT',
    'COLOR_CLEAR_VALUE','DEPTH_CLEAR_VALUE','COLOR_WRITEMASK','DEPTH_WRITEMASK','PIXEL_PACK_BUFFER_BINDING'],
    saved=Object.fromEntries(names.map(n=>[n,gl.getParameter(gl[n])])),
    pack=Object.fromEntries(['PACK_ALIGNMENT','PACK_ROW_LENGTH','PACK_SKIP_PIXELS','PACK_SKIP_ROWS'].map(n=>[n,gl.getParameter(gl[n])])),
    scissor=gl.isEnabled(gl.SCISSOR_TEST),discard=gl.isEnabled(gl.RASTERIZER_DISCARD),
    buffers=[],pixels=new Uint8Array(width*height*4);
  let framebuffer=null,renderer=null,counts;
  try {
    // Its constructor and draw preserve shader, texture, VAO, sampler, unpack,
    // attribute and pass state. This helper owns only the detached image target.
    renderer=createStudioSubjectRenderer({gl,packet,samplers});
    framebuffer=gl.createFramebuffer();if(!framebuffer)throw Error('Cannot allocate thumbnail framebuffer.');
    gl.bindFramebuffer(gl.FRAMEBUFFER,framebuffer);
    for(const [format,attachment]of [[gl.RGBA8,gl.COLOR_ATTACHMENT0],[gl.DEPTH_COMPONENT24,gl.DEPTH_ATTACHMENT]]) {
      const buffer=gl.createRenderbuffer();if(!buffer)throw Error('Cannot allocate thumbnail renderbuffer.');buffers.push(buffer);
      gl.bindRenderbuffer(gl.RENDERBUFFER,buffer);gl.renderbufferStorage(gl.RENDERBUFFER,format,width,height);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER,attachment,gl.RENDERBUFFER,buffer);
    }
    // Read/draw buffer selectors are framebuffer-owned. Never change selectors
    // on either incumbent framebuffer; rebinding below restores them verbatim.
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);gl.readBuffer(gl.COLOR_ATTACHMENT0);
    if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('Thumbnail framebuffer is incomplete.');
    gl.viewport(0,0,width,height);gl.disable(gl.SCISSOR_TEST);gl.disable(gl.RASTERIZER_DISCARD);
    gl.colorMask(true,true,true,true);gl.depthMask(true);gl.clearColor(.035,.06,.09,1);gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    counts=renderer.draw(frame);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER,null);
    for(const name of Object.keys(pack))gl.pixelStorei(gl[name],name==='PACK_ALIGNMENT'?4:0);
    gl.readPixels(0,0,width,height,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
    if(gl.isContextLost())throw Error('Context was lost during thumbnail readback.');
  }finally {
    if(renderer)renderer.dispose();
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER,saved.READ_FRAMEBUFFER_BINDING);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,saved.DRAW_FRAMEBUFFER_BINDING);
    gl.bindRenderbuffer(gl.RENDERBUFFER,saved.RENDERBUFFER_BINDING);
    gl.viewport(...saved.VIEWPORT);gl.clearColor(...saved.COLOR_CLEAR_VALUE);gl.clearDepth(saved.DEPTH_CLEAR_VALUE);
    gl.colorMask(...saved.COLOR_WRITEMASK);gl.depthMask(saved.DEPTH_WRITEMASK);
    gl[scissor?'enable':'disable'](gl.SCISSOR_TEST);gl[discard?'enable':'disable'](gl.RASTERIZER_DISCARD);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER,saved.PIXEL_PACK_BUFFER_BINDING);
    for(const [name,value]of Object.entries(pack))gl.pixelStorei(gl[name],value);
    for(const buffer of buffers)gl.deleteRenderbuffer(buffer);if(framebuffer)gl.deleteFramebuffer(framebuffer);
  }
  // Only a 2D canvas encodes the already-rendered image. GL's bottom-first rows
  // become normal top-first image rows; opaque background needs no alpha guess.
  const image=document.createElement('canvas');image.width=width;image.height=height;
  const context=image.getContext('2d');if(!context)throw Error('Cannot encode the rendered thumbnail.');
  const data=context.createImageData(width,height);
  for(let y=0;y<height;y++)data.data.set(pixels.subarray((height-1-y)*width*4,(height-y)*width*4),y*width*4);
  context.putImageData(data,0,0);
  return {dataUrl:image.toDataURL('image/png'),width,height,frame,counts,
    renderer:'exact-Studio-Phong-on-existing-game-context',lightfieldBaked:false};
}

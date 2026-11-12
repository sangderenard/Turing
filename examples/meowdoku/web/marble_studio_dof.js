// Screen-space Phong preview only. This is the analytical camera's geometric
// circle of confusion, not compound-lens transport or refractive autofocus.
const positive=(x,name)=>{x=Number(x);if(!Number.isFinite(x)||x<=0)throw new Error(`${name} must be positive`);return x;};
export function dofParameters(camera,{unitM,near,far,height,maxRadius=64}) {
  const focus=positive(camera?.focus_distance_m,"Focus distance"),aperture=Number(camera?.aperture_radius_m);
  const fov=positive(camera?.fov_degrees,"Field of view");
  if(fov>=179||!Number.isFinite(aperture)||aperture<0)throw new Error("Invalid DOF camera");
  unitM=positive(unitM,"Scene metres per unit");near=positive(near,"Near plane");far=positive(far,"Far plane");
  height=positive(height,"Image height");maxRadius=positive(maxRadius,"Preview radius limit");
  if(far<=near||maxRadius>128)throw new Error("Invalid DOF bounds");
  return {focus,aperture,unitM,near,far,height,maxRadius,pixelScale:height/(2*Math.tan(fov*Math.PI/360))};
}
export function dofAxialDepth(depth,near,far) {
  if(!Number.isFinite(depth)||depth<0||depth>1||!(near>0&&far>near))throw new Error("Invalid projected depth");
  return near*far/(far-depth*(far-near));
}
export function dofCircleRadius(depthM,p) {
  depthM=positive(depthM,"Axial depth");
  return p.aperture*p.pixelScale*(1/p.focus-1/depthM);
}
// Invert a rigid world-to-camera view, including the current chase/player pose.
// Translation is expressed in metres. The optical axis is camera -Z.
export function dofCameraFromView(view,{unitM,targetDistance,fovDegrees,apertureRadiusM=.002}) {
  if(!view||view.length!==16||!Array.from(view).every(Number.isFinite))throw new Error("Invalid camera view");
  unitM=positive(unitM,"Scene metres per unit");targetDistance=positive(targetDistance,"Camera target distance");
  const eye=[0,1,2].map(i=>-(view[i*4]*view[12]+view[i*4+1]*view[13]+view[i*4+2]*view[14])*unitM);
  return {eye_m:eye,target_m:eye.map((v,i)=>v-view[i*4+2]*targetDistance*unitM),
    fov_degrees:fovDegrees,focus_distance_m:targetDistance*unitM,aperture_radius_m:apertureRadiusM};
}

export function dofFocusAt(uv,p,camera,aspect=1) {
  const shift=camera.tilt_shift??[0,0],tilt=camera.lens_tilt??[0,0],t=Math.tan(camera.fov_degrees*Math.PI/360);
  return p.focus/Math.max(.005,1-(2*uv[0]-1+shift[0])*t*aspect*Math.tan(tilt[1])-(2*uv[1]-1+shift[1])*t*Math.tan(tilt[0]));
}
const VERTEX=`#version 300 es
void main(){vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));gl_Position=vec4(p*2.0-1.0,0.0,1.0);}`;
const FRAGMENT=`#version 300 es
precision highp float;
uniform sampler2D uColor;
uniform highp sampler2D uDepth;
uniform vec2 uSize;
uniform vec2 uClip;
uniform vec4 uPlane; // tan(tiltY), tan(tiltX), tan(fov/2)*aspect, tan(fov/2)
uniform vec2 uShift;
uniform vec4 uOptics; // physical unit, focus metres, aperture*pixelScale, radius cap
out vec4 color;
float axial(vec2 uv){float d=texture(uDepth,uv).r;return uClip.x*uClip.y/(uClip.y-d*(uClip.y-uClip.x))*uOptics.x;}
float coc(float z,vec2 uv){float d=max(.005,1.0-dot((2.0*uv-1.0+uShift)*uPlane.zw,uPlane.xy));return clamp(uOptics.z*(d/uOptics.y-1.0/z),-uOptics.w,uOptics.w);}
vec3 linearColor(vec3 s){return mix(s/12.92,pow((s+0.055)/1.055,vec3(2.4)),step(vec3(0.04045),s));}
vec3 displayColor(vec3 s){return mix(s*12.92,1.055*pow(max(s,vec3(0)),vec3(1.0/2.4))-0.055,step(vec3(0.0031308),s));}
void main(){
  vec2 uv=gl_FragCoord.xy/uSize;
  vec4 center=texture(uColor,uv);
  float z=axial(uv),c=coc(z,uv),r=abs(c);
  if(r<0.5){color=center;return;}
  vec3 sum=linearColor(center.rgb);float total=1.0;
  // Stable per-pixel rotation breaks ring/banding artifacts without temporal
  // swimming. Forty-eight equal-area disk samples, not an image-wide blur.
  float rotation=6.28318530718*fract(52.9829189*fract(dot(floor(gl_FragCoord.xy),vec2(0.06711056,0.00583715))));
  for(int i=0;i<48;i++){
    float fraction=(float(i)+0.5)/48.0,dist=r*sqrt(fraction),angle=rotation+float(i)*2.39996322973;
    vec2 q=uv+vec2(cos(angle),sin(angle))*dist/uSize;
    if(any(lessThan(q,vec2(0)))||any(greaterThan(q,vec2(1))))continue;
    float qz=axial(q),qc=coc(qz,q);
    // Reject background samples through a nearer foreground silhouette.
    // A nearer sample contributes only if its own blur disk reaches this pixel.
    // This avoids color leaking over in-focus foreground edges; it cannot
    // reconstruct surfaces hidden behind them or transparent glass depth.
    float tolerance=max(z*0.002,0.0000001);
    if(qz>z+tolerance && c<0.0)continue;
    if(qz<z-tolerance && abs(qc)+0.5<dist)continue;
    float weight=1.0;
    sum+=linearColor(texture(uColor,q).rgb)*weight;total+=weight;
  }
  color=vec4(displayColor(sum/total),center.a);
}`;

function program(gl) {
  const shaders=[];let p=null;
  try {
    for(const [type,source] of [[gl.VERTEX_SHADER,VERTEX],[gl.FRAGMENT_SHADER,FRAGMENT]]){
      const shader=gl.createShader(type);if(!shader)throw new Error("Cannot allocate DOF shader");shaders.push(shader);
      gl.shaderSource(shader,source);gl.compileShader(shader);
      if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(shader)||"DOF shader compilation failed");
    }
    p=gl.createProgram();if(!p)throw new Error("Cannot allocate DOF program");
    for(const shader of shaders)gl.attachShader(p,shader);gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p)||"DOF link failed");
    return p;
  } catch(error){if(p)gl.deleteProgram(p);throw error;}
  finally{for(const shader of shaders)gl.deleteShader(shader);}
}

export function createStudioDof(gl,{samples=4,maxRadius=64}={}) {
  let p=null,vao=null,target=null,pending=null,disposed=false;
  const caps=[gl.DEPTH_TEST,gl.BLEND,gl.CULL_FACE,gl.SCISSOR_TEST,gl.STENCIL_TEST,gl.RASTERIZER_DISCARD,gl.SAMPLE_ALPHA_TO_COVERAGE,gl.SAMPLE_COVERAGE];
  const units=[gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS)-2,gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS)-1];
  function textureState(){const active=gl.getParameter(gl.ACTIVE_TEXTURE),textures=units.map(unit=>{
    gl.activeTexture(gl.TEXTURE0+unit);return {texture:gl.getParameter(gl.TEXTURE_BINDING_2D),sampler:gl.getParameter(gl.SAMPLER_BINDING)};});
    gl.activeTexture(active);return {active,textures};}
  function restoreTextures(s){units.forEach((unit,i)=>{gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,s.textures[i].texture);gl.bindSampler(unit,s.textures[i].sampler);});gl.activeTexture(s.active);}
  function release(t){if(!t)return;for(const x of [t.color,t.depth])if(x)gl.deleteTexture(x);for(const x of [t.msColor,t.msDepth])if(x)gl.deleteRenderbuffer(x);for(const x of [t.fbo,t.msFbo])if(x)gl.deleteFramebuffer(x);}
  function allocate(w,h){
    const read=gl.getParameter(gl.READ_FRAMEBUFFER_BINDING),draw=gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING),rb=gl.getParameter(gl.RENDERBUFFER_BINDING),tex=textureState();
    const t={w,h,samples:Math.max(0,Math.min(Math.floor(samples),gl.getParameter(gl.MAX_SAMPLES)))};
    try {
      t.fbo=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,t.fbo);
      for(const [key,format,attachment] of [["color",gl.RGBA8,gl.COLOR_ATTACHMENT0],["depth",gl.DEPTH_COMPONENT24,gl.DEPTH_ATTACHMENT]]){
        t[key]=gl.createTexture();if(!t[key])throw new Error("Cannot allocate DOF textures");
        gl.activeTexture(gl.TEXTURE0+units[0]);gl.bindTexture(gl.TEXTURE_2D,t[key]);gl.texStorage2D(gl.TEXTURE_2D,1,format,w,h);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
        gl.framebufferTexture2D(gl.FRAMEBUFFER,attachment,gl.TEXTURE_2D,t[key],0);
      }
      if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw new Error("DOF color/depth target is incomplete");
      if(t.samples>1){
        t.msFbo=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,t.msFbo);
        for(const [key,format,attachment] of [["msColor",gl.RGBA8,gl.COLOR_ATTACHMENT0],["msDepth",gl.DEPTH_COMPONENT24,gl.DEPTH_ATTACHMENT]]){
          t[key]=gl.createRenderbuffer();gl.bindRenderbuffer(gl.RENDERBUFFER,t[key]);gl.renderbufferStorageMultisample(gl.RENDERBUFFER,t.samples,format,w,h);
          gl.framebufferRenderbuffer(gl.FRAMEBUFFER,attachment,gl.RENDERBUFFER,t[key]);
        }
        if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw new Error("DOF multisample target is incomplete");
      }
      return t;
    }catch(error){release(t);throw error;}
    finally{gl.bindFramebuffer(gl.READ_FRAMEBUFFER,read);gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,draw);gl.bindRenderbuffer(gl.RENDERBUFFER,rb);restoreTextures(tex);}
  }
  return {
    begin(width,height,{singleSample=false}={}){
      if(disposed||pending||gl.isContextLost())throw new Error("DOF renderer is not available");
      if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||Math.max(width,height)>gl.getParameter(gl.MAX_TEXTURE_SIZE))throw new Error("Invalid DOF image size");
      if(!p){p=program(gl);vao=gl.createVertexArray();}
      if(!target||target.w!==width||target.h!==height){const next=allocate(width,height);release(target);target=next;}
      // A secondary linear renderer resolves color and copies opaque depth to
      // this target. WebGL forbids single-sample -> multisample depth blits.
      pending={read:gl.getParameter(gl.READ_FRAMEBUFFER_BINDING),draw:gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING),
        multisample:!!target.msFbo&&!singleSample};
      gl.bindFramebuffer(gl.FRAMEBUFFER,pending.multisample?target.msFbo:target.fbo);
    },
    finish(camera,{unitM,near,far}){
      if(!pending)throw new Error("DOF frame has not begun");
      const original=pending;pending=null;
      const saved={program:gl.getParameter(gl.CURRENT_PROGRAM),vao:gl.getParameter(gl.VERTEX_ARRAY_BINDING),viewport:gl.getParameter(gl.VIEWPORT),
        depthMask:gl.getParameter(gl.DEPTH_WRITEMASK),colorMask:gl.getParameter(gl.COLOR_WRITEMASK),caps:caps.map(x=>gl.isEnabled(x)),textures:textureState()};
      try {
        const params=dofParameters(camera,{unitM,near,far,height:target.h,maxRadius});
        if(original.multisample){gl.bindFramebuffer(gl.READ_FRAMEBUFFER,target.msFbo);gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,target.fbo);
          gl.disable(gl.SCISSOR_TEST);gl.blitFramebuffer(0,0,target.w,target.h,0,0,target.w,target.h,gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT,gl.NEAREST);}
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,original.draw);gl.viewport(0,0,target.w,target.h);
        caps.forEach(x=>gl.disable(x));gl.depthMask(false);gl.colorMask(true,true,true,true);
        gl.useProgram(p);gl.bindVertexArray(vao);
        for(const [i,key,uniform] of [[0,"color","uColor"],[1,"depth","uDepth"]]){gl.activeTexture(gl.TEXTURE0+units[i]);gl.bindTexture(gl.TEXTURE_2D,target[key]);gl.bindSampler(units[i],null);gl.uniform1i(gl.getUniformLocation(p,uniform),units[i]);}
        gl.uniform2f(gl.getUniformLocation(p,"uSize"),target.w,target.h);gl.uniform2f(gl.getUniformLocation(p,"uClip"),near,far);
        gl.uniform4f(gl.getUniformLocation(p,"uOptics"),unitM,params.focus,params.aperture*params.pixelScale,maxRadius);
        const tilt=camera.lens_tilt??[0,0],shift=camera.tilt_shift??[0,0],t=Math.tan(camera.fov_degrees*Math.PI/360);
        gl.uniform4f(gl.getUniformLocation(p,"uPlane"),Math.tan(tilt[1]),Math.tan(tilt[0]),t*target.w/target.h,t);
        gl.uniform2f(gl.getUniformLocation(p,"uShift"),...shift);
        gl.drawArrays(gl.TRIANGLES,0,3);
        return {samples:original.multisample?target.samples:0,diskSamples:48,maxRadius,depthMode:"opaque",colorMode:"display-linearized",...params};
      }finally{
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER,original.read);gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,original.draw);
        gl.useProgram(saved.program);gl.bindVertexArray(saved.vao);gl.viewport(...saved.viewport);gl.depthMask(saved.depthMask);gl.colorMask(...saved.colorMask);
        caps.forEach((x,i)=>saved.caps[i]?gl.enable(x):gl.disable(x));restoreTextures(saved.textures);
      }
    },
    cancel(){if(pending){gl.bindFramebuffer(gl.READ_FRAMEBUFFER,pending.read);gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,pending.draw);pending=null;}},
    dispose(){this.cancel();release(target);target=null;if(p)gl.deleteProgram(p);if(vao)gl.deleteVertexArray(vao);p=null;vao=null;disposed=true;},
  };
}

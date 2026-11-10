// Browser geometry query backend. Intersection arithmetic is compiler-built
// from Spectral's original sphere kernel, never reimplemented in JavaScript.
export const MAX_CLIENT_RAYS=4096, MAX_CLIENT_SPHERES=65;
export function validateSphereScene(scene) {
  if(!scene || typeof scene.revision!=="string" || !scene.revision || scene.revision.length>256 ||
     !Array.isArray(scene.spheres) || scene.spheres.length<1 || scene.spheres.length>MAX_CLIENT_SPHERES)
    throw new Error("A revision and 1–65 exact spheres are required.");
  const ids=new Set();
  for(const s of scene.spheres) {
    if(!Array.isArray(s.center)||s.center.length!==3||!s.center.every(Number.isFinite)||
       !Number.isFinite(s.radius)||s.radius<=0||!Number.isSafeInteger(s.owner)||s.owner<0||s.owner>16777215||ids.has(s.owner))
      throw new Error("Invalid or duplicate sphere owner, center or radius.");
    ids.add(s.owner);
    if(![...s.center,s.radius].map(Math.fround).every(Number.isFinite)||Math.fround(s.radius)<=0)
      throw new Error("Sphere is outside float32 representation.");
  }
  return scene;
}
export function validateRayBatch(rays) {
  if(!(rays instanceof Float32Array)||rays.length%8||rays.length<8||rays.length>8*MAX_CLIENT_RAYS)
    throw new Error("Ray batch must contain 1–4096 eight-float records.");
  for(let i=0;i<rays.length;i+=8) {
    for(let j=0;j<8;j++)if(!Number.isFinite(rays[i+j]))throw new Error("Nonfinite ray.");
    const n=Math.hypot(rays[i+4],rays[i+5],rays[i+6]);
    if(n<1e-20||n>1e20||rays[i+3]<0||rays[i+7]<=rays[i+3])throw new Error("Invalid ray direction or distance interval.");
  }
  return rays.length/8;
}
export async function verifiedText(url,hash) {
  const response=await fetch(url,{cache:"no-store"});if(!response.ok)throw new Error(`Kernel fetch ${response.status}`);
  const bytes=await response.arrayBuffer();
  const digest=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",bytes)),v=>v.toString(16).padStart(2,"0")).join("");
  if(digest!==hash)throw new Error("Kernel content differs from manifest.");
  return new TextDecoder().decode(bytes);
}
export class ClientSphereQuery {
  static async create(base=new URL("./",import.meta.url),canvas=null) {
    const response=await fetch(new URL("client_ray_sphere.manifest.json",base),{cache:"no-store"});
    if(!response.ok)throw new Error(`Query manifest ${response.status}`);
    const m=await response.json();
    if(m.schema!=="turing.client-sphere-query.v1"||m.maximum_batch!==4096||m.maximum_spheres!==65)throw new Error("Unsupported query ABI.");
    const texts=await Promise.all(["vert","frag"].map(k=>verifiedText(new URL(m.artifacts[k].file,base),m.artifacts[k].sha256)));
    return new ClientSphereQuery(canvas||new OffscreenCanvas(1,1),texts,m);
  }
  constructor(canvas,texts,manifest) {
    this.manifest=manifest;this.scene=null;this.closed=false;this.busy=false;this.textures=[];
    const gl=this.gl=canvas.getContext("webgl2",{antialias:false,alpha:false,preserveDrawingBuffer:false});
    if(!gl||!gl.getExtension("EXT_color_buffer_float"))throw new Error("This device needs WebGL2 float render targets for client geometry queries.");
    this.program=gl.createProgram();
    const shaders=[];
    try {
      for(let i=0;i<2;i++) {const s=gl.createShader(i?gl.FRAGMENT_SHADER:gl.VERTEX_SHADER);shaders.push(s);gl.shaderSource(s,texts[i]);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));gl.attachShader(this.program,s);}
      gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(this.program));
    } catch(e) {gl.deleteProgram(this.program);throw e;} finally {for(const s of shaders)gl.deleteShader(s);}
    this.vao=gl.createVertexArray();this.fb=gl.createFramebuffer();
    for(let i=0;i<5;i++){const t=gl.createTexture();this.textures.push(t);gl.bindTexture(gl.TEXTURE_2D,t);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);}
  }
  setScene(scene) {
    if(this.closed||this.busy)throw new Error("Query backend is closed or busy.");validateSphereScene(scene);
    const count=scene.spheres.length,data=new Float32Array(count*8);
    scene.spheres.forEach((s,i)=>{data.set([...s.center,s.radius],i*4);data[count*4+i*4]=s.owner;});
    const gl=this.gl;gl.bindTexture(gl.TEXTURE_2D,this.textures[2]);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,count,2,0,gl.RGBA,gl.FLOAT,data);
    this.scene={revision:scene.revision,count};return {revision:scene.revision,spheres:count,geometryBytes:data.byteLength};
  }
  async query(rays,revision) {
    if(this.closed||this.busy||!this.scene||revision!==this.scene.revision)throw new Error("Stale scene, unavailable backend or overlapping query.");
    const count=validateRayBatch(rays),width=Math.min(256,count),height=Math.ceil(count/width),size=width*height*4;
    const origins=new Float32Array(size),directions=new Float32Array(size);
    for(let i=0;i<count;i++){origins.set(rays.subarray(i*8,i*8+4),i*4);directions.set(rays.subarray(i*8+4,i*8+8),i*4);}
    this.busy=true;const gl=this.gl,start=performance.now();let fence=null;
    try {
      if(gl.isContextLost())throw new Error("Client query graphics context was lost.");
      gl.useProgram(this.program);gl.bindVertexArray(this.vao);gl.disable(gl.BLEND);gl.disable(gl.DEPTH_TEST);gl.disable(gl.SCISSOR_TEST);
      for(let i=0;i<3;i++){gl.activeTexture(gl.TEXTURE0+i);gl.bindTexture(gl.TEXTURE_2D,this.textures[i]);if(i<2)gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,width,height,0,gl.RGBA,gl.FLOAT,i?directions:origins);gl.uniform1i(gl.getUniformLocation(this.program,["uOrigins","uDirections",(this.geometrySampler||"uSpheres")][i]),i);}
      gl.bindFramebuffer(gl.FRAMEBUFFER,this.fb);
      for(let i=3;i<5;i++){gl.bindTexture(gl.TEXTURE_2D,this.textures[i]);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,width,height,0,gl.RGBA,gl.FLOAT,null);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0+i-3,gl.TEXTURE_2D,this.textures[i],0);}
      // Restore sphere sampler after output allocation used the active unit.
      gl.activeTexture(gl.TEXTURE2);gl.bindTexture(gl.TEXTURE_2D,this.textures[2]);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0,gl.COLOR_ATTACHMENT1]);
      if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw new Error("Client query framebuffer is incomplete.");
      for(const [name,v] of [[(this.geometryCount||"uSphereCount"),this.scene.count],["uRayCount",count],["uWidth",width]])gl.uniform1i(gl.getUniformLocation(this.program,name),v);
      gl.viewport(0,0,width,height);gl.drawArrays(gl.TRIANGLES,0,3);fence=gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE,0);gl.flush();
      while(true){if(this.closed||gl.isContextLost())throw new Error("Client query canceled or context lost.");const wait=gl.clientWaitSync(fence,0,0);if(wait===gl.WAIT_FAILED)throw new Error("Client query wait failed.");if(wait!==gl.TIMEOUT_EXPIRED)break;if(performance.now()-start>5000)throw new Error("Client query exceeded five seconds.");await new Promise(r=>setTimeout(r,2));}
      const hit=new Float32Array(size),normal=new Float32Array(size);gl.readBuffer(gl.COLOR_ATTACHMENT0);gl.readPixels(0,0,width,height,gl.RGBA,gl.FLOAT,hit);gl.readBuffer(gl.COLOR_ATTACHMENT1);gl.readPixels(0,0,width,height,gl.RGBA,gl.FLOAT,normal);
      if(gl.getError()!==gl.NO_ERROR)throw new Error("Client query GPU error; result rejected.");
      return {revision,count,hits:hit.slice(0,count*4),normals:normal.slice(0,count*4),elapsedMs:performance.now()-start,estimatedGpuBytes:(size*4*4+this.scene.count*(this.geometryStrideBytes||32)),kernel:this.manifest.source_sha256};
    } finally {if(fence)gl.deleteSync(fence);this.busy=false;}
  }
  close(){if(this.closed)return;this.closed=true;const gl=this.gl;for(const t of this.textures)gl.deleteTexture(t);gl.deleteFramebuffer(this.fb);gl.deleteVertexArray(this.vao);gl.deleteProgram(this.program);gl.getExtension("WEBGL_lose_context")?.loseContext();}
}

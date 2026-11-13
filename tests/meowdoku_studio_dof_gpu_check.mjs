import {createStudioDof} from '../examples/meowdoku/web/marble_studio_dof.js';
const canvas=document.getElementById('view'),out=document.getElementById('result'),gl=canvas.getContext('webgl2',{antialias:false,preserveDrawingBuffer:true});
const vertex=`#version 300 es
void main(){vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));gl_Position=vec4(p*2.0-1.0,0.0,1.0);}`;
const fragment=`#version 300 es
precision highp float;uniform int uEdges;out vec4 color;
void main(){float z=gl_FragCoord.x<128.0?0.05:(gl_FragCoord.x<256.0?0.1:0.2);float d=(1.0-0.01/z)/(1.0-0.01);gl_FragDepth=d;
float v=mod(floor(gl_FragCoord.x/3.0)+floor(gl_FragCoord.y/3.0),2.0);color=vec4(vec3(v),1);if(uEdges==1){z=gl_FragCoord.x<192.0?0.05:0.2;gl_FragDepth=(1.0-0.01/z)/0.99;color=gl_FragCoord.x<192.0?vec4(1,0,0,1):vec4(0,0,1,1);}}`;
function prog(){let p=gl.createProgram();for(const [type,source]of[[gl.VERTEX_SHADER,vertex],[gl.FRAGMENT_SHADER,fragment]]){let s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));gl.attachShader(p,s);gl.deleteShader(s);}gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));return p;}
const p=prog(),vao=gl.createVertexArray(),dof=createStudioDof(gl);
function assert(x,message){if(!x)throw Error(message);}
function pixels(aperture,focus=.1,edges=0){
 dof.begin(canvas.width,canvas.height);gl.viewport(0,0,canvas.width,canvas.height);gl.disable(gl.SCISSOR_TEST);gl.clearColor(0,0,0,1);gl.depthMask(true);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.useProgram(p);gl.bindVertexArray(vao);gl.uniform1i(gl.getUniformLocation(p,'uEdges'),edges);gl.drawArrays(gl.TRIANGLES,0,3);
 const receipt=dof.finish({fov_degrees:50,focus_distance_m:focus,aperture_radius_m:aperture},{unitM:1,near:.01,far:1});
 assert(gl.getParameter(gl.CURRENT_PROGRAM)===p,'program restore');assert(gl.getParameter(gl.VERTEX_ARRAY_BINDING)===vao,'VAO restore');assert(gl.isEnabled(gl.DEPTH_TEST),'depth state restore');assert(gl.getParameter(gl.DEPTH_WRITEMASK),'depth mask restore');assert(gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING)===null,'output framebuffer restore');
 const bytes=new Uint8Array(canvas.width*canvas.height*4);gl.readPixels(0,0,canvas.width,canvas.height,gl.RGBA,gl.UNSIGNED_BYTE,bytes);assert(gl.getError()===gl.NO_ERROR,'WebGL error');return {bytes,receipt};
}
function variance(bytes,lo,hi){let a=[];for(let y=30;y<162;y++)for(let x=lo;x<hi;x++)a.push(bytes[(y*384+x)*4]);const mean=a.reduce((a,b)=>a+b)/a.length;return a.reduce((s,x)=>s+(x-mean)**2,0)/a.length;}
function diff(a,b,lo,hi){let n=0,max=0;for(let y=30;y<162;y++)for(let x=lo;x<hi;x++){const d=Math.abs(a[(y*384+x)*4]-b[(y*384+x)*4]);max=Math.max(max,d);n+=d;}return {mean:n/((hi-lo)*132),max};}
document.getElementById('run').onclick=()=>{try{
 const sharp=pixels(0),wide=pixels(.006),small=pixels(.0001),near=pixels(.006,.05),regions=[[30,98],[158,226],[286,354]];
 const statistics=regions.map(([a,b])=>({sharp:variance(sharp.bytes,a,b),wide:variance(wide.bytes,a,b),small:variance(small.bytes,a,b),focusNear:variance(near.bytes,a,b),unchanged:diff(sharp.bytes,wide.bytes,a,b)}));
 assert(statistics[0].wide<statistics[0].sharp*.15,'near plane must actually blur');assert(statistics[2].wide<statistics[2].sharp*.15,'far plane must actually blur');assert(statistics[1].unchanged.max===0,'focus plane must remain pixel-exact');assert(statistics[0].small>statistics[0].wide*4,'smaller pupil must reduce blur');assert(statistics[0].focusNear>statistics[0].sharp*.9,'moving focus must sharpen near plane');assert(statistics[1].focusNear<statistics[1].sharp*.15,'moving focus must blur old focus plane');
 const edge=pixels(.006,.1,1);let blueLeak=0;for(let y=40;y<152;y++)for(let x=176;x<192;x++)blueLeak=Math.max(blueLeak,edge.bytes[(y*384+x)*4+2]);assert(blueLeak===0,'no far blue bleed across foreground edge');
 dof.begin(384,192);dof.cancel();assert(gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING)===null,'cancel restores target');
 canvas.width=256;canvas.height=128;pixels(.002);canvas.width=384;canvas.height=192;pixels(.006);
 const ext=gl.getExtension('WEBGL_debug_renderer_info');out.textContent=JSON.stringify({passed:true,renderer:ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER),statistics,blueLeak,receipt:wide.receipt,resize:true,cancel:true,glErrors:0},null,2);
 }catch(error){out.textContent=error.stack;}};

export {pixels};

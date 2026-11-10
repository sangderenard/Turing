// Independent browser QA instrumentation, not loaded by the product page.
const locations=new WeakMap(),values={},draws=[],errors=[];
const p=WebGL2RenderingContext.prototype;
const locate=p.getUniformLocation;
p.getUniformLocation=function(program,name){const result=locate.call(this,program,name);if(result)locations.set(result,name);return result;};
for(const method of ['uniform1i','uniform1f','uniform2fv','uniform3fv']){
 const original=p[method];p[method]=function(location,value){const name=locations.get(location);if(name)values[name]=ArrayBuffer.isView(value)||Array.isArray(value)?Array.from(value):value;return original.apply(this,arguments);};
}
let last=0, pending=false;
for(const drawName of ['drawArrays','drawElements']) {
const draw=p[drawName];
p[drawName]=function(){const result=draw.apply(this,arguments);
 if(!pending&&performance.now()-last>750){ pending=true; queueMicrotask(()=>{ pending=false;
  last=performance.now();const w=this.drawingBufferWidth,h=this.drawingBufferHeight;
  const pixels=new Uint8Array(w*h*4),read=this.getParameter(this.READ_FRAMEBUFFER_BINDING),pack=this.getParameter(this.PIXEL_PACK_BUFFER_BINDING);
  this.bindFramebuffer(this.READ_FRAMEBUFFER,null);this.bindBuffer(this.PIXEL_PACK_BUFFER,null);
  this.readPixels(0,0,w,h,this.RGBA,this.UNSIGNED_BYTE,pixels);
  this.bindFramebuffer(this.READ_FRAMEBUFFER,read);this.bindBuffer(this.PIXEL_PACK_BUFFER,pack);
  let sum=0,max=0,nonzero=0,hash=2166136261;
  for(let i=0;i<pixels.length;i++)if(i%4!==3){const v=pixels[i];sum+=v;max=Math.max(max,v);if(v)nonzero++;hash=Math.imul(hash^v,16777619)>>>0;}
  const out=document.getElementById('phong-probe')??document.body.appendChild(Object.assign(document.createElement('pre'),{id:'phong-probe'}));
  out.style.cssText='position:fixed;left:0;bottom:0;z-index:10000;max-width:65vw;max-height:20vh;overflow:auto;background:#111;color:white;font:11px monospace';
  out.textContent=JSON.stringify({timestamp:Date.now(),w,h,sum,max,nonzero,hash,error:this.getError(),errors,lights:values.uNumLights,sources:Object.entries(values).filter(([k])=>/^uLight(Color|Intensity|GroupId|SpotCone)\[(10|11|30|60|90)\]$/.test(k)),calibration:values.uLightCalibration,exposure:values.uExposure});
 }); }
 return result;
};
}
for(const name of ['drawArrays','drawElements','uniform1i','uniform1f','uniform2fv','uniform3fv','uniformMatrix4fv','texImage2D','bindTexture','readPixels']){
 const original=p[name];p[name]=function(){const result=original.apply(this,arguments);const error=this.getError();if(error&&errors.length<12)errors.push({name,error,uniform:locations.get(arguments[0])});return result;};
}




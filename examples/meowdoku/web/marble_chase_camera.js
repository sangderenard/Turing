// Camera-only filtering of completed, displayed particle samples. No physics
// prediction, orientation integration or position writes belong here. Y is up;
// yaw uses the same screen-right/forward convention as cameraNudgeGravity.
const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
const angle=v=>Math.atan2(Math.sin(v),Math.cos(v));
const blend=(a,b,dt,tau)=>a+(b-a)*(-Math.expm1(-dt/tau));
const finite=(v,name)=>{if(!Number.isFinite(v))throw new TypeError(`${name} must be finite`);return v;};
const point=v=>{if(!Array.isArray(v)||v.length!==3||!v.every(Number.isFinite))throw new TypeError("Camera sample needs a finite XYZ position");return [...v];};

export class MarbleChaseCamera {
  constructor({yaw=-.5,pitch=.55,zoom=12}={}) {
    this.yaw=finite(yaw,"yaw");this.pitch=clamp(finite(pitch,"pitch"),.16,1.25);
    this.zoom=clamp(finite(zoom,"zoom"),6,80);
    this.dragging=false;this.manualUntil=-Infinity;this.clear();
  }

  clear() {this.sample=null;this.focus=null;this.velocity=[0,0,0];this.lead=[0,0,0];this.lastNow=null;this.pose=null;}
  orbit(dx,dy,now) {
    this.yaw=angle(this.yaw+finite(dx,"orbit yaw"));
    this.pitch=clamp(this.pitch+finite(dy,"orbit pitch"),.16,1.25);
    this.manualUntil=finite(now,"orbit time")+2;
  }
  drag(active,now) {this.dragging=!!active;this.manualUntil=finite(now,"drag time")+2;}
  zoomBy(delta) {this.zoom=clamp(this.zoom*Math.exp(clamp(finite(delta,"zoom delta"),-2,2)),6,80);}
  recenter(now) {
    const [x,,z]=this.velocity;
    if(Math.hypot(x,z)>0)this.yaw=Math.atan2(x,-z);
    this.pitch=.55;this.zoom=12;this.manualUntil=finite(now,"recenter time")+2;
    this.focus=null;
  }

  update({position,radius,sampleSeconds,nowSeconds,epoch=0,steering=false,autoHeading=true}) {
    const p=point(position),r=finite(radius,"radius"),t=finite(sampleSeconds,"sample time"),now=finite(nowSeconds,"camera time");
    if(r<=0)throw new RangeError("Camera radius must be positive");
    const elapsed=this.lastNow===null?0:now-this.lastNow;
    const dt=clamp(elapsed,0,.1),old=this.sample;
    // Seeking, rebirth, a different run and a long tab interruption are not
    // physical velocities. Reset the filter rather than fly across the scene.
    const reset=!old||old.epoch!==epoch||t<old.time||elapsed<0||elapsed>.5||
      Math.hypot(...p.map((v,i)=>v-old.position[i]))>Math.max(r,old?.radius||r)*this.zoom*8;
    if(reset) {this.focus=[...p];this.velocity=[0,0,0];this.lead=[0,0,0];}
    if(!reset&&t>old.time) {
      const span=t-old.time;
      this.velocity=span>.5?[0,0,0]:p.map((v,i)=>(v-old.position[i])/span);
    }
    if(reset||t!==old.time)this.sample={position:p,time:t,radius:r,epoch,arrived:now};
    // A paused/blocked recording must settle, not continue extrapolating.
    const fresh=now-this.sample.arrived<.15;
    const velocity=fresh?this.velocity:[0,0,0],speed=Math.hypot(velocity[0],velocity[2]);
    if(autoHeading&&!steering&&!this.dragging&&now>=this.manualUntil&&speed>r*.75) {
      const target=Math.atan2(velocity[0],-velocity[2]),delta=angle(target-this.yaw);
      this.yaw=angle(this.yaw+clamp(delta*(-Math.expm1(-dt/.8)),-1.4*dt,1.4*dt));
    }
    // Lead is an aim offset, not an extrapolated body pose. Cap it by body
    // size so a fast contact or snapshot gap cannot throw the marble offscreen.
    const gain=speed>0?Math.min(.16,3*r/speed):0;
    const targetLead=[velocity[0]*gain,0,velocity[2]*gain];
    if(!this.focus)this.focus=[...p];
    this.focus=this.focus.map((v,i)=>blend(v,p[i],dt,.09));
    const lag=Math.hypot(...this.focus.map((v,i)=>v-p[i]));
    if(lag>3*r)this.focus=this.focus.map((v,i)=>p[i]+(v-p[i])*3*r/lag);
    this.lead=this.lead.map((v,i)=>blend(v,targetLead[i],dt,.18));
    const target=this.focus.map((v,i)=>v+this.lead[i]+(i===1?r*.45:0));
    const distance=r*this.zoom,c=Math.cos(this.pitch),s=Math.sin(this.pitch);
    const eye=[target[0]-Math.sin(this.yaw)*c*distance,target[1]+s*distance,target[2]+Math.cos(this.yaw)*c*distance];
    this.lastNow=now;
    this.pose={eye,target,yaw:this.yaw,pitch:this.pitch,distance,radius:r};
    return this.pose;
  }
}

// A Y-up world-to-view matrix. Keep camera transforms in the same convention
// as the Studio, with no roll inherited from a spinning marble or tilting case.
export function chaseViewMatrix({eye,target}) {
  const e=point(eye),t=point(target),z=e.map((v,i)=>v-t[i]),length=Math.hypot(...z);
  if(length<=0)throw new RangeError("Camera eye and target must differ");
  z.forEach((v,i)=>z[i]=v/length);
  const x=[z[2],0,-z[0]],side=Math.hypot(...x);
  if(side<1e-8)throw new RangeError("Camera direction cannot be vertical");
  x.forEach((v,i)=>x[i]=v/side);
  const y=[z[1]*x[2],z[2]*x[0]-z[0]*x[2],-z[1]*x[0]];
  const dot=v=>-v.reduce((sum,n,i)=>sum+n*e[i],0);
  return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,dot(x),dot(y),dot(z),1]);
}

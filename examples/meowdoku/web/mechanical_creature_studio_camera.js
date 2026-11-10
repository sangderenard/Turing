// Matrix form of the existing game's _PLUCK_LIVING_MAP_VERTEX_ADAPTER.
// It changes no camera, control, movement, world state or projection policy.
const vector=(p,name)=>{if(!p||p.length!==3||!Array.from(p).every(Number.isFinite))throw Error(name+' needs 3 finite values');return Array.from(p);};
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=(p,name)=>{const n=Math.hypot(...p);if(!(n>0))throw Error(name+' is degenerate');return p.map(x=>x/n);};

export function mechanicalCreatureStudioCamera({cameraPosition,cameraFacing,width,height}) {
  const eye=vector(cameraPosition,'Existing game eye'),forward=unit(vector(cameraFacing,'Existing game facing'),'Existing game facing');
  if(!Number.isFinite(width)||!Number.isFinite(height)||width<=0||height<=0)throw Error('Existing game viewport is invalid');
  const right=unit(cross(forward,[0,1,0]),'Existing game camera right'),up=unit(cross(right,forward),'Existing game camera up');
  // Game shader uses +Z forward. Studio uses the ordinary -Z view frame.
  // Flipping both view Z and the projection convention gives the same clip
  // coordinates and depth; the authored Studio shader remains unchanged.
  const view=new Float32Array([right[0],up[0],-forward[0],0,right[1],up[1],-forward[1],0,
    right[2],up[2],-forward[2],0,-dot(right,eye),-dot(up,eye),dot(forward,eye),1]);
  const aspect=width/Math.max(1,height),tangentHalfFov=.70,near=.04,far=128;
  const projection=new Float32Array([1/(tangentHalfFov*aspect),0,0,0,0,1/tangentHalfFov,0,0,
    0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]);
  return {view,projection,eye,right,up,forward,near,far,tangentHalfFov};
}

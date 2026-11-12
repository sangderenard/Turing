// Existing Spectral analytical camera convention: normalized principal-point
// shift and tilted focus plane. Not a compound lens prescription.
export function cameraMovements(camera, {shift=[0,0],tiltDegrees=[0,0]}={}) {
  for(const values of [shift,tiltDegrees])if(values.length!==2||!values.every(Number.isFinite))throw Error('Invalid camera movement');
  if(shift.some(v=>Math.abs(v)>1)||tiltDegrees.some(v=>Math.abs(v)>30))throw Error('Camera movement outside travel');
  return {...camera,tilt_shift:[...shift],lens_tilt:tiltDegrees.map(v=>v*Math.PI/180)};
}
export function shiftedProjection(matrix, camera) {
  const result=new Float32Array(matrix),shift=camera.tilt_shift??[0,0];
  result[8]+=shift[0];result[9]+=shift[1];return result;
}
export function studioCameraPose({yaw,pitch,distance,target,freeEye=null,height=null,floor=0}) {
  const direction=[-Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch),Math.cos(yaw)*Math.cos(pitch)];
  const eye=freeEye?[...freeEye]:target.map((v,i)=>v+distance*direction[i]);
  if(height!==null)eye[1]=floor+height;
  return {eye,target:freeEye?eye.map((v,i)=>v-distance*direction[i]):[...target]};
}

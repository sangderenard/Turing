// Pose ownership only. Camera construction remains photoCameraFromStudio;
// lens settings remain owned by the existing photo lens controls.
import {photoCameraFromStudio} from './marble_photo.js';
const point=(v,name)=>{if(!Array.isArray(v)||v.length!==3||!v.every(Number.isFinite))throw Error(`Invalid ${name}.`);return [...v];};
const clone=value=>structuredClone(value);

export function studioDisplayedSubjectCenter(transform,rollPositionM=null) {
  const center=point(transform?.subjectCenter,'subject centre');
  if(!(Number.isFinite(transform.unitM)&&transform.unitM>0))throw Error('Invalid Studio unit.');
  if(rollPositionM){const roll=point(rollPositionM,'rolling position');for(const i of [0,2])center[i]+=roll[i]/transform.unitM;}
  return center;
}

export function linkedPhotoOrbit({yaw,pitch,distance,photoTransform}) {
  const photoTarget=point(photoTransform?.subjectCenter,'photographed subject centre');
  // Reuse angle/zoom controls, but retain Exposure's own physical scale/FOV.
  // Studio-only profile edits must not mutate the protected Exposure camera.
  // The photographed subject gets its own centre, not Studio's table height.
  return photoCameraFromStudio(yaw,pitch,distance,photoTransform.unitM,photoTransform.fovDegrees,photoTarget);
}

function validScene(scene) {
  const c=scene?.camera;
  const eye=point(c?.eye_m,'photographic eye'),target=point(c?.target_m,'photographic target');
  if(!(Math.hypot(...eye.map((x,i)=>x-target[i]))>0)||!(c.fov_degrees>0&&c.fov_degrees<179))throw Error('Invalid photographic camera pose.');
  return clone(scene);
}

export class PhotographicCameraBinding {
  constructor(){this.followUser=true;this.held=null;}
  hold(scene){const next=validScene(scene);this.held=next;this.followUser=false;return this.resolve(next);}
  follow(){this.followUser=true;}
  resolve(linked){return this.followUser?validScene(linked):clone(this.held);}
}

export function mountPhotographicCameraBinding({doc=document,binding,captureLinked,framePhoto,onChange=()=>{}}) {
  const make=(tag,text)=>{const e=doc.createElement(tag);if(text!==undefined)e.textContent=text;return e;};
  const root=make('section');root.id='photo-camera-binding';root.className='wide';root.style.cssText='display:grid;gap:5px';
  const label=make('label'),follow=make('input');label.style.cssText='display:flex;align-items:center;gap:6px';follow.style.cssText='width:auto;flex:none';follow.type='checkbox';follow.id='photo-camera-follow-user';follow.checked=true;
  label.append(follow,make('span',' Studio photo pose follows user orbit'));
  const copy=make('button','Copy user orbit and hold photographic pose');copy.type='button';copy.id='photo-camera-copy-user';
  const frame=make('button','Frame marble with photographic camera');frame.type='button';frame.id='photo-camera-frame-subject';
  const status=make('output');status.id='photo-camera-binding-status';status.setAttribute('role','status');
  const refresh=()=>{follow.checked=binding.followUser;status.textContent=binding.followUser?
    'When Photo camera is Studio, angle and zoom controls follow the user. Exposure retains its own scale and subject centre.':
    'Photographic pose is held in metres. User orbit and walking do not move it. Lens controls still apply. Live overlay follows the user view.';};
  const run=operation=>{try{operation();refresh();onChange();}catch(error){follow.checked=binding.followUser;status.textContent=error.message;}};
  const toggle=()=>run(()=>{if(follow.checked)binding.follow();else binding.hold(captureLinked());});
  const copyClick=()=>run(()=>binding.hold(captureLinked()));
  const frameClick=()=>run(()=>binding.hold(framePhoto()));
  follow.addEventListener('change',toggle);copy.addEventListener('click',copyClick);frame.addEventListener('click',frameClick);
  root.append(label,copy,frame,status);doc.getElementById('photo-controls').append(root);refresh();
  return {root,refresh,dispose(){follow.removeEventListener('change',toggle);copy.removeEventListener('click',copyClick);frame.removeEventListener('click',frameClick);root.remove();}};
}

// Physical dimensions, independent of pixel resolution and tile allocation.
export const CAMERA_BACK_SCHEMA = "studio-camera-back-v1";
export const CAMERA_BACK_PRESETS = Object.freeze([
  {key:"micro_four_thirds",label:"Micro Four Thirds",width_m:.0173,height_m:.013},
  {key:"aps_c",label:"APS-C",width_m:.0236,height_m:.0157},
  {key:"full_frame",label:"35 mm full frame",width_m:.036,height_m:.024},
  {key:"medium_44x33",label:"Medium format",width_m:.044,height_m:.033},
  {key:"film_6x6",label:"6 × 6 film (active area)",width_m:.056,height_m:.056},
  {key:"large_4x5",label:"4 × 5 film (active area)",width_m:.120,height_m:.095},
]);
const number = (v, low, high, name) => {
  if(typeof v!=="number" || !Number.isFinite(v) || v<low || v>high)throw new RangeError(`${name} must be between ${low} and ${high}.`);
  return v;
};
export function cameraBack(key, widthMm, heightMm) {
  if(key==null || key==="reference")return null;
  const preset=CAMERA_BACK_PRESETS.find(p=>p.key===key);
  if(!preset && key!=="custom")throw new RangeError("Unknown camera back.");
  return {schema:CAMERA_BACK_SCHEMA,key,width_m:preset?.width_m??number(widthMm/1000,.001,.3,"Back width (m)"),
    height_m:preset?.height_m??number(heightMm/1000,.001,.3,"Back height (m)"),fit:"center_crop"};
}
export function sensorGate(back, aspect=1) {
  number(aspect,.01,100,"Output aspect");
  if(!back)return {width_m:.024*aspect,height_m:.024,output_aspect:aspect,fit:"reference_height"};
  number(back.width_m,.001,.3,"Back width (m)");number(back.height_m,.001,.3,"Back height (m)");
  const height=Math.min(back.height_m,back.width_m/aspect);
  return {width_m:height*aspect,height_m:height,output_aspect:aspect,fit:"center_crop"};
}
export function cameraWithBack(camera, back, aspect=1) {
  const result={...camera};
  let reference=camera.back_reference_fov_degrees??camera.fov_degrees;
  delete result.camera_back;delete result.sensor_gate;delete result.back_reference_fov_degrees;
  if(!back){
    if(camera.back_reference_fov_degrees!=null)result.fov_degrees=camera.lens_selection ?
      2*Math.atan(.024/(2*camera.lens_selection.focal_length_m))*180/Math.PI : reference;
    return result;
  }
  const f=camera.lens_selection?.focal_length_m ?? .024/(2*Math.tan(number(reference,.001,100,"Reference field of view")*Math.PI/360));
  reference=2*Math.atan(.024/(2*f))*180/Math.PI;
  const gate=sensorGate(back,aspect),fov=2*Math.atan(gate.height_m/(2*f))*180/Math.PI;
  number(fov,.001,100,"Back field of view; increase focal length or reduce back if too wide");
  return {...result,camera_back:{...back},sensor_gate:gate,back_reference_fov_degrees:reference,fov_degrees:fov};
}

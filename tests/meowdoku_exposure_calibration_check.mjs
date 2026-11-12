import assert from 'node:assert/strict';
import {studioExposureTargetArtwork,validateStudioArtwork} from '../examples/meowdoku/web/marble_studio_grid.js';
import {studioExposureTargetGobo,studioGoboFaces} from '../examples/meowdoku/web/marble_studio_gobos.js';
import {fitMonotoneExposureBezier,fitExposureControls,logarithmicMonoPixels} from '../examples/meowdoku/web/marble_calibration.js';
import {calibrationImageMap,measureCalibrationMap,calibrationSparseRuns} from '../examples/meowdoku/web/marble_calibration_map.js';

const {artwork,reference}=studioExposureTargetArtwork();
const names=[...new Set(['white_plaster','black_matte_paint',...artwork.rectangles.map(r=>r.material)])];
assert.equal(artwork.rectangles.length,33*33);
assert.deepEqual(validateStudioArtwork(artwork,names),artwork);
assert.deepEqual(reference.neutralReflectance,{black:.025,gray:.18,white:.90});
assert.equal(reference.qr.payload,'STUDIO-EXPOSURE-v1');
assert.equal(reference.qr.channel,'nir850');
assert.deepEqual(reference.qr.bounds,[-.5,-.5,.5,.5]);
assert(artwork.rectangles.some(r=>r.material.endsWith('__nir850')));
for(const material of ['black_matte_paint','reference_gray18_matte','white_plaster',
  'reference_red_matte','reference_green_matte','reference_blue_matte','reference_yellow_matte'])
  assert(artwork.rectangles.some(r=>r.material===material));
const gobo=studioExposureTargetGobo({id:7,unitMm:304.8,center:[0,0,-.2]});
assert.equal(gobo.width,64/304.8);
const faces=studioGoboFaces({version:1,gobos:[gobo]},Object.fromEntries(names.map((n,i)=>[n,i])));
assert(faces.length>2048);

const fitted=fitMonotoneExposureBezier([{x:500,y:.7},{x:0,y:.02},{x:100,y:.2},{x:250,y:.42}]);
assert.equal(fitted.bias,.02);
assert.equal(fitted.segments.length,3);
for(const segment of fitted.segments)assert(segment.p0[1]<=segment.p1[1]&&segment.p1[1]<=segment.p2[1]&&segment.p2[1]<=segment.p3[1]);
assert.throws(()=>fitMonotoneExposureBezier([{x:0,y:0},{x:1,y:2},{x:2,y:1}]),/nonmonotone/);
const log=logarithmicMonoPixels({shape:[1,3],dtype:'<f4',data:new Float32Array([0,4,16])},4);
assert.deepEqual([...log.rgba.filter((_,i)=>i%4===0)],[0,128,255]);
// The spectral QR spans the full visible chart, aligning every paint region.
const corners=[{x:16,y:16},{x:112,y:16},{x:112,y:112},{x:16,y:112}];
const map=calibrationImageMap(corners,128,128);
const data=new Float32Array(128*128*3);
for(let p=0;p<map.pixels.length;p++)for(let c=0;c<3;c++)data[p*3+c]=map.pixels[p]+1;
const regions=measureCalibrationMap(map,{shape:[128,128,3],data});
for(const name of ['blackCenter','gray18','whiteRing','red','green','blue','yellow'])
  assert(regions[name].usable,`${name} needs measurable pixels`);
assert.equal(regions.gray18.mean[0],2);
assert.equal(regions.red.mean[0],5);
const sparse=calibrationSparseRuns(map);
assert.deepEqual(sparse.regions.gray18.reduce((n,[,length])=>n+length,0),regions.gray18.pixels);
for(const runs of Object.values(sparse.regions))for(const [start,length] of runs)
  assert.equal(Math.floor(start/128),Math.floor((start+length-1)/128));
const records=[1500,0,300,600,900,1200,1800].map(strength=>{
  const reading=v=>({gray18:{mean:[v,v*1.1,v*1.2],pixels:10,usable:true}});
  const value=strength/1500;
  return {mockLux:value*500,exposureSeconds:1,
    phong:{regions:reading(20+value*100)},
    raytrace:{linear:{sensorRegions:reading(.01+value*.4)}},
    peanut:{linear:{monoRegions:{gray18:{mean:[.005+value*.2],pixels:10,usable:true}}}}};
});
const controls=fitExposureControls(records);
assert.equal(controls.film.red.segments.length,6);
assert.equal(controls.phong.green.bias,22/255);
assert.equal(controls.monochromeSensor.bias,.005);
console.log('Exposure target geometry, monotone Bézier fit and logarithmic monochrome display passed');

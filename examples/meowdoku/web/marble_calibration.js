// Automatic colour calibration: upload a YAML recipe describing a sweep of
// single-light test cases; this module drives the ENTIRE sweep itself --
// zeroing every fixture, lighting exactly one at its declared colour,
// capturing the live Phong preview, exposing the identical rig through the
// Spectral ray tracer, and assembling a report + downloadable dataset --
// with no further manual slider changes once "Run calibration" is clicked.
import { packMarblePhoto } from "./marble_photo.js";
import { parseNpz } from "./npz.js";
import {STUDIO_LIGHTS} from './marble_light_strengths.js';
import {calibrationImageMap,measureCalibrationMap,detectCalibrationQr,calibrationSparseRuns} from './marble_calibration_map.js';

// ─────────────────────────────────────────────────────────────────────────
// A narrow YAML subset parser. Only what the calibration recipe below
// actually uses: nested mappings, lists of mappings, bare/quoted string
// scalars, numbers, inline [a, b, c] arrays, and a `#`-starts-a-comment
// rule matching real YAML (a `#` only starts a comment when preceded by
// whitespace or at the start of the token, never mid-token -- so hex
// colours like #ff0000 parse as literal strings, not truncated comments).
// This is not a general YAML implementation; a document outside this
// subset will throw rather than silently misparse.
function stripComment(line) {
  let inQuote = null;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuote) { if (ch === inQuote) inQuote = null; continue; }
    if (ch === '"' || ch === "'") { inQuote = ch; continue; }
    if (ch === "#" && (i === 0 || /\s/.test(line[i - 1]))) return line.slice(0, i);
  }
  return line;
}
function parseScalar(raw) {
  const text = raw.trim();
  if (text === "" || text === "~" || text === "null") return null;
  if (text === "true") return true;
  if (text === "false") return false;
  if ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith("'") && text.endsWith("'")))
    return text.slice(1, -1);
  if (text.startsWith("[") && text.endsWith("]"))
    return text.slice(1, -1).split(",").map((part) => parseScalar(part));
  if (/^-?\d+(\.\d+)?$/.test(text)) return Number(text);
  return text;
}
function indentOf(line) {
  const match = /^ */.exec(line);
  if (/\t/.test(line.slice(0, match[0].length + 1))) throw new Error("calibration YAML: tabs are not supported, use spaces");
  return match[0].length;
}
export function parseYaml(source) {
  const lines = String(source).split(/\r\n|\n/)
    .map(stripComment)
    .map((line) => line.replace(/\s+$/, ""))
    .filter((line) => line.trim() !== "");
  let cursor = 0;
  function parseBlock(indent) {
    if (cursor >= lines.length) return null;
    const first = lines[cursor];
    if (indentOf(first) < indent) return null;
    if (indentOf(first) > indent) throw new Error(`calibration YAML: unexpected indent at "${first}"`);
    const isList = /^-(\s|$)/.test(first.slice(indent));
    if (isList) {
      const items = [];
      while (cursor < lines.length && indentOf(lines[cursor]) === indent && /^-(\s|$)/.test(lines[cursor].slice(indent))) {
        const rest = lines[cursor].slice(indent + 1).replace(/^\s+/, "");
        if (rest === "") {
          cursor++;
          if (cursor < lines.length && indentOf(lines[cursor]) > indent) items.push(parseBlock(indentOf(lines[cursor])));
          else items.push(null);
        } else if (/^[^:]+:(\s|$)/.test(rest)) {
          // An inline "- key: value" starts a mapping at this item's own
          // effective indent (indent + 1 for the dash + however many
          // spaces separate the dash from the key).
          const afterDash = lines[cursor].slice(indent + 1);
          const leadingSpaces = afterDash.length - afterDash.replace(/^\s+/, "").length;
          const itemIndent = indent + 1 + leadingSpaces;
          lines[cursor] = " ".repeat(itemIndent) + rest;
          items.push(parseBlock(itemIndent));
        } else {
          cursor++;
          items.push(parseScalar(rest));
        }
      }
      return items;
    }
    const obj = {};
    while (cursor < lines.length && indentOf(lines[cursor]) === indent) {
      const line = lines[cursor];
      const body = line.slice(indent);
      const colon = body.indexOf(":");
      if (colon < 0) throw new Error(`calibration YAML: expected "key: value" at "${line}"`);
      const key = body.slice(0, colon).trim();
      const valueText = body.slice(colon + 1);
      cursor++;
      if (valueText.trim() === "") {
        // A "key:" with nothing on its own line either introduces a
        // nested block (mapping or list) at whatever deeper indent the
        // next line actually uses, or is a legitimately empty value.
        obj[key] = (cursor < lines.length && indentOf(lines[cursor]) > indent)
          ? parseBlock(indentOf(lines[cursor])) : null;
      } else {
        obj[key] = parseScalar(valueText);
      }
    }
    return obj;
  }
  // Nested blocks need a concrete starting indent; scan the first line.
  if (!lines.length) throw new Error("calibration YAML: empty document");
  const result = parseBlock(indentOf(lines[0]));
  if (cursor < lines.length) throw new Error(`calibration YAML: unconsumed content at "${lines[cursor]}"`);
  return result;
}

// ─────────────────────────────────────────────────────────────────────────
// Studio fixture roles this module knows how to isolate. Every case zeroes
// ALL ten before applying its own role -- background and kicker panels
// have caught out a manual sweep before; this makes that impossible.
const ROLE_IDS = {
  key_left: ["key-filter-left", "light-strength-left"],
  key_right: ["key-filter-right", "light-strength-right"],
  background: ["background-filter", "light-bg-strength"],
  kicker_left: ["kicker-left-filter", "light-kicker-left-strength"],
  kicker_right: ["kicker-right-filter", "light-kicker-right-strength"],
  fill_upper: ["fill-upper-filter", "light-fill-upper-strength"],
  fill_lower: ["fill-lower-filter", "light-fill-lower-strength"],
  table_left: ["table-left-filter", "light-table-left-strength"],
  table_right: ["table-right-filter", "light-table-right-strength"],
  underlight: ["under-filter", "light-under-strength"],
};
function setControl(doc, id, value) {
  const el = doc.getElementById(id);
  if (!el) throw new Error(`calibration: missing control #${id}`);
  el.value = String(value);
  if (el.tagName === "SELECT" && el.value !== String(value))
    throw new Error(`calibration: #${id} has no option "${value}" (check the recipe against this control's real choices)`);
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
}
function applyRoomAndSubject(doc, calib) {
  const subject = calib.subject || {};
  for (const [id, value] of [["shell", subject.shell], ["eye", subject.eye], ["accent", subject.accent]])
    if (value) setControl(doc, id, value);
  const room = calib.room || {};
  for (const prefix of ["wall", "floor", "ceiling"]) {
    const material = room[`${prefix}_material`];
    if (material) { setControl(doc, `${prefix}-pattern`, "solid"); setControl(doc, `${prefix}-material-a`, material); }
  }
  if (calib.key_shape) setControl(doc, "key-shape", calib.key_shape);
}
function applyCase(doc, calib, kase) {
  const role = kase.role || "key_left";
  if (!(role in ROLE_IDS)) throw new Error(`calibration: unknown light role "${role}"`);
  // Calibration must measure its declared source, not a transient solo mask.
  doc.getElementById("light-solo-clear")?.dispatchEvent(new Event("clear-all-solos"));
  for (const [roleName, [filterId, strengthId]] of Object.entries(ROLE_IDS)) {
    // A single-light recipe must also isolate the exposure route if a user
    // previously split its levels. Calibration explicitly owns both lanes.
    const link = doc.getElementById(`${roleName.replaceAll('_','-')}-levels-linked`);
    if (link && !link.checked) {
      link.checked = true;
      link.dispatchEvent(new Event("change", {bubbles:true}));
    }
    setControl(doc, strengthId, 0);
    if (roleName === role) {
      if (kase.filter) setControl(doc, filterId, kase.filter);
      setControl(doc, strengthId, kase.strength ?? 0);
    }
  }
}
const settle = (ms = 400) => new Promise((resolve) => setTimeout(resolve, ms));

// ─────────────────────────────────────────────────────────────────────────
// Pixel capture + background/foreground segmentation. Two renderers at two
// different resolutions/aspect ratios can't be compared pixel-for-pixel,
// so instead of trusting a fixed sample rectangle, every image is
// segmented against its OWN four-corner background colour -- robust to
// framing differences between the live Phong canvas and the ray-traced
// output.
function segmentForeground(width, height, data) {
  const corners = [[1, 1], [width - 2, 1], [1, height - 2], [width - 2, height - 2]];
  let br = 0, bg = 0, bb = 0, bn = 0;
  for (const [cx, cy] of corners) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    const x = cx + dx, y = cy + dy;
    if (x < 0 || y < 0 || x >= width || y >= height) continue;
    const i = (y * width + x) * 4;
    br += data[i]; bg += data[i + 1]; bb += data[i + 2]; bn++;
  }
  br /= bn; bg /= bn; bb /= bn;
  let sr = 0, sg = 0, sb = 0, sn = 0;
  const threshold = 18;
  for (let i = 0; i < width * height * 4; i += 4) {
    const dr = data[i] - br, dg = data[i + 1] - bg, db = data[i + 2] - bb;
    if (Math.sqrt(dr * dr + dg * dg + db * db) > threshold) { sr += data[i]; sg += data[i + 1]; sb += data[i + 2]; sn++; }
  }
  if (!sn) return { mean: [br, bg, bb], background: [br, bg, bb], coverage: 0 };
  return { mean: [sr / sn, sg / sn, sb / sn], background: [br, bg, bb], coverage: sn / (width * height) };
}
function readGlPixels(gl, canvas) {
  const width = canvas.width, height = canvas.height;
  const raw = new Uint8Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, raw);
  // WebGL's readback is bottom-up; flip into normal top-down row order.
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) data.set(raw.subarray((height - 1 - y) * width * 4, (height - y) * width * 4), y * width * 4);
  return { width, height, data };
}
function frameMean(data) {
  const sum=[0,0,0];
  for(let i=0;i<data.length;i+=4)for(let c=0;c<3;c++)sum[c]+=data[i+c];
  return sum.map(value=>value/(data.length/4)/255);
}

export function fitMonotoneExposureBezier(points) {
  const sorted=[...points].sort((a,b)=>a.x-b.x);
  if(sorted.length<3||sorted.some(p=>!Number.isFinite(p.x)||!Number.isFinite(p.y)||p.x<0||p.y<0))
    throw Error('Exposure fit requires at least three finite nonnegative readings');
  if(sorted.some((p,i)=>i>0&&(p.x<=sorted[i-1].x||p.y<sorted[i-1].y)))
    throw Error('Exposure sweep is duplicated or nonmonotone; keep raw readings and repeat the calibration');
  const secants=sorted.slice(1).map((p,i)=>(p.y-sorted[i].y)/(p.x-sorted[i].x));
  const slopes=sorted.map((_,i)=>i===0?secants[0]:i===sorted.length-1?secants.at(-1):
    secants[i-1]*secants[i]<=0?0:2*secants[i-1]*secants[i]/(secants[i-1]+secants[i]));
  const segments=secants.map((secant,i)=>{
    const a=sorted[i],b=sorted[i+1],dx=b.x-a.x;
    let left=Math.min(slopes[i],3*secant),right=Math.min(slopes[i+1],3*secant);
    if(left+right>3*secant){const factor=3*secant/(left+right);left*=factor;right*=factor;}
    return {p0:[a.x,a.y],p1:[a.x+dx/3,a.y+left*dx/3],
      p2:[b.x-dx/3,b.y-right*dx/3],p3:[b.x,b.y]};
  });
  return {kind:'monotone-piecewise-cubic-bezier',points:sorted,segments,
    bias:sorted[0].x===0?sorted[0].y:null,
    activationAtMiddle:secants[Math.floor(secants.length/2)],
    domain:[sorted[0].x,sorted.at(-1).x],extrapolation:'disabled'};
}

export function fitExposureControls(records) {
  const usable=records.filter(r=>!r.error&&Number.isFinite(r.mockLux));
  const channelFit=(read,dose)=>Array.from({length:3},(_,channel)=>
    fitMonotoneExposureBezier(usable.map(r=>({x:dose(r),y:read(r,channel)}))));
  const gray=.18;
  const phong=channelFit((r,c)=>r.phong.regions.gray18.mean?.[c]/255,r=>r.mockLux*gray);
  const film=channelFit((r,c)=>r.raytrace.linear.sensorRegions.gray18.mean?.[c],
    r=>r.mockLux*gray*r.exposureSeconds);
  const peanut=fitMonotoneExposureBezier(usable.map(r=>({
    x:r.mockLux*gray*r.exposureSeconds,y:r.peanut.linear.monoRegions.gray18.mean?.[0]})));
  return {version:2,reference:{lux:500,reflectance:gray,
      meaning:'nominal mock lux times authored gray-card reflectance; not measured radiometry'},
    phong:{red:phong[0],green:phong[1],blue:phong[2]},
    film:{red:film[0],green:film[1],blue:film[2]},monochromeSensor:peanut,
    map:'QR-registered gray18 patch; other neutral rings and color swatches retained as checks',
    activation:'local slope of each measured curve',bias:'dark reading at zero mock lux',
    policy:'candidate response calibration; no control changes until held-out readings are verified'};
}

const EXPOSURE_RECIPE={name:'500 mock lux Studio reference',dual_camera:true,
  transport:'fixed:8',
  time_model:'static-case',
  description:'Fixed reflective card, RGB scene camera and monochrome thin-lens peanut sensor in disjoint forward lanes of the same captured state slice.',
  resolution:512,exposure_seconds:1,max_bounces:4,key_shape:'bare',
  mock_lux_at_strength:{strength:1500,lux:500},
  room:{wall_material:'white_plaster',floor_material:'white_plaster',ceiling_material:'white_plaster'},
  // Register the QR at the nominal lit reference before the dark case.
  cases:[1500,0,300,600,900,1200,1800].map(strength=>({name:`key ${strength}`,role:'key_left',
    filter:'#ffffff',strength}))};
function pixelsToPngDataUrl(width, height, data) {
  const scratch = document.createElement("canvas");
  scratch.width = width; scratch.height = height;
  scratch.getContext("2d").putImageData(new ImageData(data, width, height), 0, 0);
  return scratch.toDataURL("image/png");
}
function mapOverlayPng(width,height,data,map) {
  const colors=[[0,0,0],[128,128,128],[255,255,255],[64,64,64],
    [255,64,64],[64,255,64],[64,64,255],[255,255,64]];
  const marked=new Uint8ClampedArray(data);
  for(let p=0;p<map.pixels.length;p++){
    const region=map.pixels[p];if(region<0)continue;
    for(let c=0;c<3;c++)marked[4*p+c]=Math.round(.4*marked[4*p+c]+.6*colors[region][c]);
  }
  const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
  const context=canvas.getContext('2d');context.putImageData(new ImageData(marked,width,height),0,0);
  context.strokeStyle='#ff00ff';context.lineWidth=2;context.beginPath();
  map.qrCorners.forEach(({x,y},i)=>i?context.lineTo(x,y):context.moveTo(x,y));
  context.closePath();context.stroke();
  return canvas.toDataURL('image/png');
}
async function mapCapturedPixels(width,height,data,registered=null) {
  const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
  canvas.getContext('2d').putImageData(new ImageData(data,width,height),0,0);
  const map=registered??calibrationImageMap(await detectCalibrationQr(canvas),width,height);
  if(map.width!==width||map.height!==height)throw Error('The registered calibration view changed size');
  return {map,regions:measureCalibrationMap(map,{shape:[height,width,4],data}),
    overlay:mapOverlayPng(width,height,data,map)};
}
async function sampleImageUrl(url,registration=null,markerUrl=null) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`calibration: image fetch HTTP ${response.status}`);
  const blob = await response.blob();
  const bitmap = await createImageBitmap(blob);
  const scratch = document.createElement("canvas");
  scratch.width = bitmap.width; scratch.height = bitmap.height;
  const ctx = scratch.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  const image = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  const seg = segmentForeground(bitmap.width, bitmap.height, image.data);
  const map=registration===true?calibrationImageMap(await detectCalibrationQr(bitmap,markerUrl??url),bitmap.width,bitmap.height):registration;
  if(map&&(map.width!==bitmap.width||map.height!==bitmap.height))throw Error('The registered sensor view changed size');
  return { ...seg, png: scratch.toDataURL("image/png"),map,
    regions:map?measureCalibrationMap(map,{shape:[bitmap.height,bitmap.width,4],data:image.data}):null,
    overlay:map?mapOverlayPng(bitmap.width,bitmap.height,image.data,map):null };
}

// ─────────────────────────────────────────────────────────────────────────
async function pollPhotoJob(digest, { intervalMs = 300, timeoutMs = 120000,
  onStatus=()=>{}, canceled=()=>false } = {}) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if(canceled())return {state:'canceled',digest};
    const response = await fetch(`/api/marble_photo?digest=${digest}`, { cache: "no-store" });
    const value = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(value.error || `calibration: photo status HTTP ${response.status}`);
    await onStatus(value);
    if (value.state === "complete" || value.state === "error" || value.state === "canceled") return value;
    await settle(intervalMs);
  }
  throw new Error("calibration: exposure timed out");
}

// The downloadable dataset retains the actual HDR archive, not inferred
// linear values recovered from a tone-mapped PNG. Bound aggregate archive
// retention because base64/JSON is larger than the original compressed files.
export const MAX_CALIBRATION_HDR_BYTES = 64 * 1024 * 1024;

export function summarizeLinearArray(array) {
  const {shape, data, dtype} = array || {};
  if (!Array.isArray(shape) || shape.length !== 3 || shape[2] !== 3 ||
      !shape.every(Number.isSafeInteger) || shape.some(n => n < 1) ||
      !["<f4", "<f8"].includes(dtype) || data?.length !== shape[0] * shape[1] * 3)
    throw new Error("calibration: expected finite floating HxWx3 HDR samples");
  const sum=[0,0,0], correction=[0,0,0], minimum=[Infinity,Infinity,Infinity], maximum=[-Infinity,-Infinity,-Infinity], negative=[0,0,0];
  for(let i=0;i<data.length;i++){
    const c=i%3,v=data[i];
    if(!Number.isFinite(v))throw new Error("calibration: HDR contains nonfinite samples");
    const delta=v-correction[c],next=sum[c]+delta;
    correction[c]=(next-sum[c])-delta;sum[c]=next;
    minimum[c]=Math.min(minimum[c],v);maximum[c]=Math.max(maximum[c],v);
    if(v<0)negative[c]++;
  }
  if(!sum.every(Number.isFinite))throw new Error("calibration: HDR statistics overflow");
  const pixelCount=shape[0]*shape[1];
  return {shape:[...shape],dtype,pixelCount,region:"entire sensor frame; no brightness-dependent segmentation",
    normalization:"none",sum,mean:sum.map(v=>v/pixelCount),minimum,maximum,negativeSamples:negative};
}

export function summarizeLinearMono(array) {
  const {shape,data,dtype}=array||{};
  if(!Array.isArray(shape)||shape.length!==2||!shape.every(Number.isSafeInteger)||
      shape.some(n=>n<1)||!["<f4","<f8"].includes(dtype)||data?.length!==shape[0]*shape[1])
    throw new Error('calibration: expected finite floating HxW monochrome HDR');
  let sum=0,correction=0,minimum=Infinity,maximum=-Infinity;
  for(const value of data){
    if(!Number.isFinite(value))throw new Error('calibration: monochrome HDR contains nonfinite samples');
    const delta=value-correction,next=sum+delta;correction=(next-sum)-delta;sum=next;
    minimum=Math.min(minimum,value);maximum=Math.max(maximum,value);
  }
  return {shape:[...shape],dtype,pixelCount:data.length,normalization:'none',sum,mean:sum/data.length,minimum,maximum};
}

export function logarithmicMonoPixels(array,stops=12) {
  const summary=summarizeLinearMono(array),white=Math.max(summary.maximum,Number.MIN_VALUE);
  if(!Number.isFinite(stops)||stops<1||stops>32)throw Error('Invalid logarithmic display range');
  const floor=white/2**stops,rgba=new Uint8ClampedArray(array.data.length*4);
  for(let i=0;i<array.data.length;i++){
    const value=Math.max(0,array.data[i]);
    const tone=Math.max(0,Math.min(1,Math.log2(Math.max(value,floor)/floor)/stops));
    rgba.set([Math.round(tone*255),Math.round(tone*255),Math.round(tone*255),255],i*4);
  }
  return {width:array.shape[1],height:array.shape[0],rgba,white,stops,
    meaning:'logarithmic presentation of unchanged linear monochrome HDR'};
}

function monoLogPng(array) {
  const display=logarithmicMonoPixels(array),canvas=document.createElement('canvas');
  canvas.width=display.width;canvas.height=display.height;
  canvas.getContext('2d').putImageData(new ImageData(display.rgba,display.width,display.height),0,0);
  return {png:canvas.toDataURL('image/png'),white:display.white,stops:display.stops};
}

export async function captureLinearCalibration(status, budget={remaining:MAX_CALIBRATION_HDR_BYTES}, lane=0, map=null) {
  const digest=status?.digest,filename=status?.hdr;
  if(!/^[a-f0-9]{64}$/.test(digest||"") || !/^exposure-\d{6}\.npz$/.test(filename||""))
    throw new Error("calibration: completed exposure has no valid raw HDR archive");
  if(!Number.isSafeInteger(budget.remaining)||budget.remaining<=0||budget.remaining>MAX_CALIBRATION_HDR_BYTES)
    throw new Error("calibration: raw HDR dataset limit reached (64 MiB)");
  const url=`/api/marble_photo/hdr?digest=${digest}&name=${encodeURIComponent(filename)}&lane=${lane}`;
  const response=await fetch(url,{cache:"no-store"});
  if(!response.ok)throw new Error(`calibration: HDR fetch HTTP ${response.status}`);
  const declared=Number(response.headers?.get("Content-Length"));
  if(Number.isFinite(declared)&&declared>budget.remaining)throw new Error("calibration: raw HDR dataset limit reached (64 MiB)");
  const buffer=await response.arrayBuffer();
  if(buffer.byteLength>budget.remaining)throw new Error("calibration: raw HDR dataset limit reached (64 MiB)");
  const arrays=await parseNpz(buffer,["linear_rgb_exposure","linear_srgb_exposure","linear_mono_exposure"]);
  const sensor=arrays.has("linear_rgb_exposure")?summarizeLinearArray(arrays.get("linear_rgb_exposure")):null;
  const profiled=arrays.has("linear_srgb_exposure")?summarizeLinearArray(arrays.get("linear_srgb_exposure")):null;
  const mono=arrays.has("linear_mono_exposure")?summarizeLinearMono(arrays.get("linear_mono_exposure")):null;
  const sensorRegions=map&&sensor?measureCalibrationMap(map,arrays.get('linear_rgb_exposure')):null;
  const monoRegions=map&&mono?measureCalibrationMap(map,arrays.get('linear_mono_exposure')):null;
  if(!sensor&&!mono)throw Error('calibration: HDR has no known linear sensor channel');
  const logarithmic=mono?monoLogPng(arrays.get("linear_mono_exposure")):null;
  const bytes=new Uint8Array(buffer),parts=[];
  for(let i=0;i<bytes.length;i+=32768)parts.push(String.fromCharCode(...bytes.subarray(i,i+32768)));
  const archiveBase64=btoa(parts.join(""));
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",buffer)),v=>v.toString(16).padStart(2,"0")).join("");
  budget.remaining-=buffer.byteLength;
  return {version:1,status:"captured",sensor,profiled,mono,sensorRegions,monoRegions,logarithmic,
    archive:{filename,sha256:hash,byteLength:buffer.byteLength,encoding:"base64",data:archiveBase64},
    provenance:{digest,transport:status.transport??null,filmResponse:status.film_response??null,
      colorProfile:status.color_profile??null,displayTransform:status.display_transform??null,
      displayWhite:status.display_white??null,cameraSettings:status.camera_settings??null,
      geometryStats:status.geometry_stats??null,elapsedSeconds:status.elapsed_seconds??null,
      frameElapsedSeconds:status.frame_elapsed_seconds??null,forwardRayJobs:status.forward_ray_jobs??null,
      sensorSites:status.sensor_sites??null,epoch:status.epoch??null},
    limitation:"Raw accumulated linear sensor values, not radiometrically calibrated units. QR-mapped regions are sparse chart readings; the logarithmic PNG is display-only."};
}

async function exposeCase(geometry, baseDigest, calib, kase, cameraOptions, transport=calib.transport??'fixed:3', peanutOptions=null, watch=null) {
  const seconds=kase.exposure_seconds ?? calib.exposure_seconds ?? 0.1;
  const options={
    resolution: calib.resolution ?? 128,
    seconds,
    maxBounces: calib.max_bounces ?? 2,
    transport,
    gamma: 1.0,
    orbitFrames: 1,
    ...cameraOptions,
    ...(kase.exposure_iso?{exposureIso:kase.exposure_iso}:{}),
  };
  if(calib.dual_camera)options.exposureSubsteps=Math.ceil(seconds*10);
  if(peanutOptions){
    const fixture=options.fixtures;
    if(!fixture?.pbr||!fixture.emitter_profiles)
      throw Error('Calibration needs the captured Studio fixture light state');
    options.fixtures={...fixture,emissive_material_ids:[...new Set([
      ...Array.from({length:STUDIO_LIGHTS.length},(_,i)=>i),
      ...(fixture.emitter_profiles.records??[]).map(record=>record.material),
      ...(fixture.shop_material_ids??[])])]};
    const {resolution:meterResolution,...meterPose}=peanutOptions.camera;
    if(meterResolution!==options.resolution)
      throw Error('Both disjoint sensors need the same capture resolution');
    options.sensorLanes=[{}, {...meterPose,sensor_transport:transport,sensor_output:'mono'}];
    const frames=watch?.frameSeconds?Math.ceil(seconds/watch.frameSeconds):1;
    options.exposureStateFrames=Array.from({length:frames},(_,i)=>({
      t0:i*seconds/frames,t1:(i+1)*seconds/frames,pbr:fixture.pbr,
      emitter_profiles:fixture.emitter_profiles}));
    if(watch?.regions)options.calibrationRegions=watch.regions;
  }
  const body = packMarblePhoto(geometry, baseDigest, options);
  const deadline=Date.now()+30000;
  while(true){
    const response = await fetch("/api/marble_photo", { method: "POST",
      headers: { "Content-Type": "application/octet-stream" }, body });
    const value = await response.json().catch(() => ({}));
    if(response.ok){
      watch?.onStart?.(value.digest);
      return pollPhotoJob(value.digest,watch?{timeoutMs:12*60*60*1000,
        onStatus:watch.onStatus,canceled:watch.canceled}:{});
    }
    if(response.status!==409||!String(value.error||'').includes('GPU is busy')||Date.now()>=deadline)
      throw new Error(value.error || `calibration: expose HTTP ${response.status}`);
    await settle(250);
  }
}

// ─────────────────────────────────────────────────────────────────────────
function gainReport(records) {
  // Legacy visual ratios use segmented, encoded 8-bit PNG means. Automatic
  // normalization, tone mapping, gamut mapping and sRGB transfer have already
  // happened, so these numbers cannot be used as linear calibration gains.
  const channels = ["r", "g", "b"];
  const report = { phong: {}, raytrace: {}, kind:"display-code-ratios",
    warning:"Visual PNG ratios only, not sensor gains or a fitted color profile. Use retained raw HDR for linear analysis." };
  for (const renderer of ["phong", "raytrace"]) {
    for (const [ci, channel] of channels.entries()) {
      const relevant = records.filter((r) => !r.error && r.intendedRgb[ci] > 200);
      if (!relevant.length) { report[renderer][channel] = null; continue; }
      const ratios = relevant.map((r) => r[renderer].mean[ci] / 255);
      report[renderer][channel] = ratios.reduce((a, b) => a + b, 0) / ratios.length;
    }
  }
  return report;
}
function hexToRgb(hex) {
  const match = /^#?([0-9a-fA-F]{6})$/.exec(String(hex).trim());
  if (!match) throw new Error(`calibration: invalid colour "${hex}"`);
  const n = parseInt(match[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function renderGallery(doc, records, report) {
  const gallery = doc.getElementById("calibration-gallery");
  gallery.hidden = false;
  gallery.replaceChildren();
  const swatch = (rgb) => `<span style="display:inline-block;width:12px;height:12px;margin-right:4px;vertical-align:middle;border:1px solid #556;background:rgb(${rgb.map((v) => Math.round(v)).join(",")})"></span>`;
  const rows = records.map((r) => {
    if (r.error) return `<tr><td>${r.name}</td><td colspan="3">error: ${r.error}</td></tr>`;
    return `<tr><td>${r.name}</td>
      <td>${swatch(r.intendedRgb)}${r.intendedRgb.map((v) => Math.round(v)).join(",")}</td>
      <td>${swatch(r.phong.mean)}${r.phong.mean.map((v) => Math.round(v)).join(",")}</td>
      <td>${swatch(r.raytrace.mean)}${r.raytrace.mean.map((v) => Math.round(v)).join(",")}</td></tr>`;
  }).join("");
  const gain = (value) => (value == null ? "n/a" : value.toFixed(3));
  const table = document.createElement("table");
  table.style.cssText = "width:100%;border-collapse:collapse;font-size:11px;margin-top:6px";
  table.innerHTML = `<thead><tr><th>Case</th><th>Intended</th><th>Phong</th><th>Ray-traced</th></tr></thead><tbody>${rows}</tbody>
    <tfoot><tr><td>PNG ratio (visual only)</td><td>-</td>
      <td>R ${gain(report.phong.r)} · G ${gain(report.phong.g)} · B ${gain(report.phong.b)}</td>
      <td>R ${gain(report.raytrace.r)} · G ${gain(report.raytrace.g)} · B ${gain(report.raytrace.b)}</td></tr></tfoot>`;
  gallery.append(table);
  const explanation=document.createElement("p");
  explanation.textContent="PNG ratios are display-only. The dataset retains raw HDR and QR-mapped sparse chart regions per sensor lane. Candidate curves use the gray patch; color swatches and held-out readings still need validation.";
  gallery.append(explanation);
  const thumbs = document.createElement("div");
  thumbs.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;margin-top:6px";
  for (const r of records) {
    if (r.error) continue;
    for (const [label, src] of [["phong", r.phong.png], ["Phong map",r.phong.imageMap?.overlay],
      ["ray", r.raytrace.png],["ray map",r.raytrace.imageMap?.overlay],
      ["peanut log",r.peanut?.linear?.logarithmic?.png]]) {
      if(!src)continue;
      const img = document.createElement("img");
      img.src = src; img.title = `${r.name} · ${label}`;
      img.style.cssText = "width:56px;height:56px;object-fit:cover;border:1px solid #536b82;border-radius:4px";
      thumbs.append(img);
    }
  }
  gallery.append(thumbs);
  if(report.exposure){
    const fitted=document.createElement('pre');
    fitted.textContent=report.exposure.error?`Exposure fit unavailable: ${report.exposure.error}`:
      `500 mock lux reference · QR-mapped gray patch · Phong R bias ${report.exposure.phong.red.bias?.toPrecision(4)} / activation ${report.exposure.phong.red.activationAtMiddle?.toPrecision(4)} · RGB film and monochrome sensor have separate piecewise Bézier curves in the dataset. Controls await held-out validation.`;
    gallery.append(fitted);
  }
}

// ─────────────────────────────────────────────────────────────────────────
export async function runCalibrationSet(doc, ctx, calib, onProgress) {
  const { saveObject, getGeometry, getCameraOptions } = ctx;
  if(calib.dual_camera)ctx.assertStaticCalibrationScene?.();
  if(calib.dual_camera&&!ctx.getPeanutCameraOptions)throw Error('Calibration needs the Studio peanut camera pose');
  if(calib.dual_camera&&!ctx.getGobos?.().gobos.some(g=>g.artwork?.rectangles?.[0]?.id.startsWith('exposure-')))
    throw Error('Add the exposure reference card to the backdrop before running the dual-camera sweep');
  // Compare both renderers under A, retaining the user's B overrides intact.
  const profileOwner=doc.getElementById('lighting-profiles');
  profileOwner?.dispatchEvent(new Event('room-calibration-start'));
  const shopSwitches=calib.dual_camera?[...doc.querySelectorAll('input[id^="shop-light-input-"][id$="-enabled"]')]
    .map(input=>({input,checked:input.checked})):[];
  try {
  for(const {input} of shopSwitches)if(input.checked){input.checked=false;input.dispatchEvent(new Event('change',{bubbles:true}));}
  applyRoomAndSubject(doc, calib);
  await settle(300);
  const baseDigest = await saveObject();
  if (!baseDigest) throw new Error("calibration: could not save the calibration subject design");
  const cases = calib.cases || [];
  if (!cases.length) throw new Error("calibration: recipe has no cases");
  const records = [], rawBudget={remaining:MAX_CALIBRATION_HDR_BYTES};
  const registered={phong:null,ray:null,peanut:null};
  let registeredCamera=null;
  for (const kase of cases) {
    if(calib.dual_camera)ctx.assertStaticCalibrationScene?.();
    onProgress?.(`Setting up "${kase.name}"…`);
    applyCase(doc, calib, kase);
    await settle(400);
    const geometry = getGeometry();
    if (!geometry) throw new Error("calibration: no marble geometry is ready");
    const phongShot = readGlPixels(ctx.gl, ctx.canvas);
    const phongSeg = segmentForeground(phongShot.width, phongShot.height, phongShot.data);
    const phongFrameMean=frameMean(phongShot.data);
    const phongPng = pixelsToPngDataUrl(phongShot.width, phongShot.height, phongShot.data);
    let phongMap=null;
    if(calib.dual_camera){
      phongMap=await mapCapturedPixels(phongShot.width,phongShot.height,phongShot.data,registered.phong);
      registered.phong=phongMap.map;
    }
    const cameraOptions=calib.dual_camera?(ctx.getCalibrationCameraOptions?.()??getCameraOptions()):getCameraOptions(),
      peanutOptions=calib.dual_camera?ctx.getPeanutCameraOptions(cameraOptions):null;
    if(calib.dual_camera){
      const key=JSON.stringify([cameraOptions.camera,peanutOptions.camera]);
      if(registeredCamera!==null&&registeredCamera!==key)
        throw Error('Calibration camera pose changed after QR registration');
      registeredCamera=key;
    }
    const sceneRevision=ctx.getSceneRevision?.();
    onProgress?.(`Exposing "${kase.name}" on the ray tracer…`);
    try {
      const status = await exposeCase(geometry, baseDigest, calib, kase, cameraOptions,
        calib.dual_camera?'fixed:3':calib.transport??'fixed:3', peanutOptions);
      if (status.state !== "complete") {
        records.push({ name: kase.name, intendedRgb: hexToRgb(kase.filter), error: status.error || status.state });
        continue;
      }
      const imageUrl = `/api/marble_photo/image?digest=${status.digest || ""}&name=${encodeURIComponent(status.image)}`;
      const raytraceSample = await sampleImageUrl(imageUrl,calib.dual_camera?(registered.ray??true):null);
      if(calib.dual_camera)registered.ray=raytraceSample.map;
      let linear;
      try {linear=await captureLinearCalibration(status,rawBudget,0,raytraceSample.map);}
      catch(error){linear={version:1,status:"unavailable",error:error.message||String(error)};}
      let peanut=null;
      if(calib.dual_camera){
        if(ctx.getSceneRevision?.()!==sceneRevision)throw Error('Studio scene changed during the shared sensor exposure');
        const meter=status.sensor_lanes?.find(lane=>lane.index===1);
        if(meter?.transport!=='fixed:1'||!meter.hdr||!Number.isSafeInteger(meter.sensor_sites))
          throw Error('The shared slice has no completed monochrome peanut sensor');
        const meterStatus={...status,hdr:meter.hdr.split('/').at(-1),
          transport:'fixed:1',sensor_sites:meter.sensor_sites};
        const meterImage=await sampleImageUrl(`/api/marble_photo/image?digest=${status.digest}&name=${encodeURIComponent(meter.image.split('/').at(-1))}&lane=1`,registered.peanut??true);
        registered.peanut=meterImage.map;
        peanut={digest:status.digest,camera:meter.camera,
          linear:await captureLinearCalibration(meterStatus,rawBudget,1,meterImage.map),
          imageMap:{qrCorners:meterImage.map.qrCorners,regions:meterImage.regions,
            overlay:meterImage.overlay}};
      }
      records.push({
        name: kase.name, intendedRgb: hexToRgb(kase.filter), intendedStrength: kase.strength ?? 0,
        exposureSeconds:kase.exposure_seconds??calib.exposure_seconds??0.1,
        exposureIso:kase.exposure_iso??100,
        mockLux:calib.mock_lux_at_strength?((kase.strength??0)/calib.mock_lux_at_strength.strength)*calib.mock_lux_at_strength.lux:null,
        phong: { mean: phongSeg.mean, frameMean:phongFrameMean, coverage: phongSeg.coverage, png: phongPng,
          regions:phongMap?.regions??null,imageMap:phongMap?{qrCorners:phongMap.map.qrCorners,
            overlay:phongMap.overlay}:null },
        raytrace: { mean: raytraceSample.mean, coverage: raytraceSample.coverage, png: raytraceSample.png,
          displayWhite: status.display_white, filmResponse: status.film_response, linear,
          regions:raytraceSample.regions,imageMap:raytraceSample.map?{qrCorners:raytraceSample.map.qrCorners,
            overlay:raytraceSample.overlay}:null },
        peanut,
      });
    } catch (error) {
      records.push({ name: kase.name, intendedRgb: hexToRgb(kase.filter), error: error.message || String(error) });
    }
    onProgress?.(`"${kase.name}" done.`);
  }
  return records;
  } finally {
    for(const {input,checked} of shopSwitches)if(input.checked!==checked){input.checked=checked;input.dispatchEvent(new Event('change',{bubbles:true}));}
    profileOwner?.dispatchEvent(new Event('room-calibration-end'));
  }
}

function watchCurveSvg(doc, curves) {
  const svg=doc.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.setAttribute('viewBox','0 0 480 160');svg.setAttribute('role','img');
  svg.setAttribute('aria-label','Accumulating linear sensor response curves');
  svg.style.cssText='width:100%;max-width:520px;background:#111b2a;border:1px solid #4c6682';
  const all=curves.flatMap(curve=>curve.points);
  const xmax=Math.max(1,...all.map(p=>p.x)),ymax=Math.max(1e-9,...all.map(p=>p.y));
  const xy=p=>`${20+440*p[0]/xmax} ${140-120*p[1]/ymax}`;
  for(const {points,color,fit} of curves){
    if(points.length<2)continue;
    const path=doc.createElementNS('http://www.w3.org/2000/svg','path');
    path.setAttribute('d',fit?`M${xy(fit.segments[0].p0)} `+
      fit.segments.map(segment=>`C${xy(segment.p1)},${xy(segment.p2)},${xy(segment.p3)}`).join(' '):
      points.map((p,i)=>`${i?'L':'M'}${xy([p.x,p.y])}`).join(' '));
    path.setAttribute('fill','none');path.setAttribute('stroke',color);
    path.setAttribute('stroke-width','2');svg.append(path);
  }
  return svg;
}

export function mountCalibrationWatch(doc,container,{fixedLane=null,laneCount=2}={}) {
  container.hidden=false;container.replaceChildren();
  const row=doc.createElement('div');row.style.cssText='display:flex;gap:10px;flex-wrap:wrap;align-items:start';
  const imageWrap=doc.createElement('div');imageWrap.style.cssText='position:relative;max-width:512px';
  const image=doc.createElement('img');image.alt='Accumulating calibrated Studio scene';
  image.style.cssText='display:block;width:100%;max-width:512px';
  const detail=doc.createElement('div');detail.style.cssText='position:absolute;bottom:8px;left:8px;right:8px;color:white;background:#101b2cc9;padding:4px 8px;border-radius:4px;font-size:12px';
  detail.textContent='The little calibration crew is checking the chart.';
  imageWrap.append(image,detail);
  const panel=doc.createElement('div');panel.style.cssText='min-width:240px;flex:1';
  const selector=doc.createElement('select');
  for(let lane=0;lane<laneCount;lane++){
    const option=doc.createElement('option');option.value=String(lane);
    option.textContent=lane===0?'RGB scene camera':lane===1?'Monochrome peanut sensor':`Sensor lane ${lane}`;
    selector.append(option);
  }
  if(fixedLane!==null){selector.value=String(fixedLane);selector.hidden=true;}
  const falseColorLabel=doc.createElement('label');
  const falseColor=doc.createElement('input');falseColor.type='checkbox';falseColor.checked=true;falseColor.disabled=true;
  falseColorLabel.append(falseColor,' Show spectral false color');
  const lookStatus=doc.createElement('small');lookStatus.style.cssText='display:block;min-height:1.3em';
  const info=doc.createElement('p'),plot=doc.createElement('div');
  panel.append(selector,falseColorLabel,lookStatus,info,plot);row.append(imageWrap,panel);container.append(row);
  const jokes=['A marble-sized technician checks the focus chart.',
    'The peanut camera is pretending to be very serious.',
    'A tiny clipboard records absolutely no invented photons.',
    'The grey ring insists it is eighteen percent.',
    'The lab mascot waits for the next sensor slice.'];
  let joke=0;
  const timer=setInterval(()=>{detail.textContent=jokes[joke++%jokes.length];},23000);
  const points=Array.from({length:32},()=>[]);let current=null,lastEpoch=-1;
  function draw(){
    if(!current)return;
    const lane=fixedLane??Number(selector.value),record=current.sensor_lanes?.find(item=>item.index===lane);
    falseColorLabel.hidden=lane!==0;
    falseColor.disabled=lane!==0||!record?.image;
    if(record?.image){
      const alternate=lane===0&&!falseColor.checked&&current.retouch_params?.spectral_look==='visible_reference'
        ?current.retouched_image:null;
      image.src=`/api/marble_photo/image?digest=${current.digest}&name=${encodeURIComponent((alternate??record.image).split('/').at(-1))}${lane?`&lane=${lane}`:''}&epoch=${current.epoch}`;
    }
    const curves=[];
    for(const [index,color] of (lane?[ [0,'#eeeeee'] ]:[[0,'#ff6b6b'],[1,'#74dc80'],[2,'#7ba8ff']])){
      const samples=points[lane].map(p=>({x:p.x,y:p.y[index]}));
      try{
        const fit=samples.length>=3?fitMonotoneExposureBezier(samples):null;
        curves.push({points:samples,color,fit});
      }catch{curves.push({points:samples,color,fit:null});}
    }
    plot.replaceChildren(watchCurveSvg(doc,curves));
    info.textContent=`Studio tick ${current.epoch??0} · ${current.exposure_integrated_seconds??0} simulated seconds · ${points[lane].length} sparse linear readings. Horizontal scale: nominal grey-card mock lux × seconds. The lines refit as accumulated sensor data arrives.`;
  }
  selector.addEventListener('change',draw);
  falseColor.addEventListener('change',async()=>{
    draw();
    if(falseColor.checked){lookStatus.textContent='Spectral false color; raw sensor readings are unchanged.';return;}
    if(!current?.digest)return;
    lookStatus.textContent='Preparing visible-band preview…';
    try{
      const response=await fetch('/api/marble_photo/retouch',{method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify({digest:current.digest,spectral_look:'visible_reference'})});
      const value=await response.json().catch(()=>({}));
      if(!response.ok)throw Error(value.error||`retouch HTTP ${response.status}`);
      const refreshed=await fetch(`/api/marble_photo?digest=${current.digest}`,{cache:'no-store'});
      if(refreshed.ok)current=await refreshed.json();
      lookStatus.textContent='Visible-band preview requested; raw sensor readings are unchanged.';
      draw();
    }catch(error){lookStatus.textContent=`Visible preview: ${error.message||error}`;}
  });
  return {update(value){
    current=value;if(!value.sensor_lanes||value.epoch===lastEpoch)return;
    lastEpoch=value.epoch;
    const history=value.calibration_history??[value];
    for(const item of history){
      const elapsed=item.exposure_integrated_seconds;
      if(!Number.isFinite(elapsed))continue;
      for(const lane of item.sensor_lanes??[]){
        const mean=lane.calibration_regions?.regions?.gray18?.mean;
        const dose=elapsed*500*.18;
        if(!mean||!points[lane.index]||points[lane.index].at(-1)?.x>=dose)continue;
        points[lane.index].push({x:dose,y:mean});
      }
    }
    draw();
  },stop(){clearInterval(timer);detail.textContent='Calibration watcher stopped.';}};
}

export function initCalibration(doc, ctx) {
  const upload = doc.getElementById("calibration-yaml-upload");
  const runButton = doc.getElementById("calibration-run");
  const status = doc.getElementById("calibration-status");
  const downloadButton = doc.getElementById("calibration-download");
  const builtIn=doc.getElementById('calibration-exposure-preset');
  const watchButton=doc.getElementById('calibration-watch');
  const watchDuration=doc.getElementById('calibration-watch-duration');
  const stopButton=doc.getElementById('calibration-watch-stop');
  const watchView=doc.getElementById('calibration-watch-view');
  let calib = null, lastDataset = null;
  let watcher=null,watchDigest=null,stopRequested=false;
  builtIn?.addEventListener('click',()=>{
    calib=structuredClone(EXPOSURE_RECIPE);
    if(!ctx.getGobos?.().gobos.some(g=>g.artwork?.rectangles?.[0]?.id.startsWith('exposure-')))
      doc.getElementById('studio-gobo-add-exposure')?.click();
    if(!ctx.getGobos?.().gobos.some(g=>g.artwork?.rectangles?.[0]?.id.startsWith('exposure-'))){
      status.textContent='Could not add the reflective exposure card. Check the gobo controls.';return;
    }
    runButton.disabled=false;watchButton.disabled=false;downloadButton.hidden=true;
    status.textContent='500 mock lux sequence loaded with a reflective card. Click Watch accumulating calibration to start the two sensors, or Run calibration for the seven-level sweep. No YAML upload is needed.';
  });
  upload.addEventListener("change", async () => {
    const file = upload.files?.[0];
    runButton.disabled = true;
    watchButton.disabled=true;
    downloadButton.hidden = true;
    if (!file) { calib = null; status.textContent='Choose Set up 500 mock lux exposure calibration to enable the watcher without a YAML file.'; return; }
    try {
      calib = parseYaml(await file.text());
      if (!Array.isArray(calib.cases) || !calib.cases.length) throw new Error("recipe has no \"cases\" list");
      status.textContent = `Loaded "${calib.name || file.name}" · ${calib.cases.length} case(s). Click Run calibration.`;
      runButton.disabled = false;
      watchButton.disabled = !calib.dual_camera;
      status.textContent += calib.dual_camera
        ? ' Watch accumulating calibration is also available.'
        : ' This recipe is sweep-only; use Set up 500 mock lux exposure calibration for the two-sensor watcher.';
    } catch (error) {
      calib = null;
      status.textContent = `Could not read calibration recipe: ${error.message || error}`;
    }
  });
  stopButton.addEventListener('click',async()=>{
    stopRequested=true;stopButton.disabled=true;
    if(watchDigest)await fetch('/api/marble_photo/cancel',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({digest:watchDigest})});
    watcher?.stop();status.textContent='Calibration watcher stopped.';
  });
  watchButton.addEventListener('click',async()=>{
    if(!calib?.dual_camera){status.textContent='Choose Set up 500 mock lux exposure calibration first, or upload a dual-camera recipe, to enable the watcher.';return;}
    watchButton.disabled=true;runButton.disabled=true;upload.disabled=true;stopButton.disabled=false;
    stopRequested=false;watchDigest=null;
    status.textContent='Preparing calibration scene and saving the subject before sensor registration.';
    const shopSwitches=[...doc.querySelectorAll('input[id^="shop-light-input-"][id$="-enabled"]')]
      .map(input=>({input,checked:input.checked}));
    const profileOwner=doc.getElementById('lighting-profiles');
    profileOwner?.dispatchEvent(new Event('room-calibration-start'));
    try{
      ctx.assertStaticCalibrationScene?.();
      if(!ctx.getGobos?.().gobos.some(g=>g.enabled&&g.artwork?.rectangles?.[0]?.id.startsWith('exposure-')))
        throw Error('Add and enable the exposure reference card first');
      for(const {input} of shopSwitches)if(input.checked){input.checked=false;input.dispatchEvent(new Event('change',{bubbles:true}));}
      applyRoomAndSubject(doc,calib);
      const reference=calib.cases.find(item=>item.strength===1500)??calib.cases[0];
      applyCase(doc,calib,reference);await settle(400);
      const baseDigest=await ctx.saveObject(),geometry=ctx.getGeometry();
      if(!baseDigest||!geometry)throw Error('Calibration subject is unavailable');
      const cameraOptions=ctx.getCalibrationCameraOptions?.()??ctx.getCameraOptions(),
        peanutOptions=ctx.getPeanutCameraOptions(cameraOptions);
      const sceneRevision=ctx.getSceneRevision?.();
      status.textContent='Registering the two ray sensors against the QR card…';
      const registered=await exposeCase(geometry,baseDigest,calib,reference,cameraOptions,'fixed:8',peanutOptions,
        {onStart:digest=>{watchDigest=digest;},canceled:()=>stopRequested});
      if(stopRequested)return;
      if(registered.state!=='complete')throw Error(registered.error||registered.state);
      if(!registered.marker)throw Error('The main camera spectral QR layer is missing');
      if(ctx.getSceneRevision?.()!==sceneRevision)throw Error('Studio scene changed during QR registration');
      const rgb=await sampleImageUrl(`/api/marble_photo/image?digest=${registered.digest}&name=${encodeURIComponent(registered.image)}`,true,
        `/api/marble_photo/image?digest=${registered.digest}&name=${encodeURIComponent(registered.marker)}`);
      const meter=registered.sensor_lanes?.find(item=>item.index===1);
      if(!meter?.image||!meter.marker)throw Error('Monochrome sensor registration layers are missing');
      const mono=await sampleImageUrl(`/api/marble_photo/image?digest=${registered.digest}&name=${encodeURIComponent(meter.image.split('/').at(-1))}&lane=1`,true,
        `/api/marble_photo/image?digest=${registered.digest}&name=${encodeURIComponent(meter.marker?.split('/').at(-1))}&lane=1`);
      const regions=[rgb.map,mono.map].map(calibrationSparseRuns);
      if(regions.some(spec=>spec.regions.gray18.reduce((n,[,length])=>n+length,0)<3))
        throw Error('The QR-registered grey patch is too small for sparse readings');
      watchView.hidden=false;watchView.replaceChildren();
      const intro=doc.createElement('p');
      intro.textContent='Each sensor has its own static published watcher page. Open either page at any time; recent sparse readings are retained in job status.';
      watchView.append(intro);
      watcher={update:value=>{
        if(value.calibration_history?.length)
          status.textContent=`Accumulating static Studio scene: tick ${value.epoch}, ${value.exposure_integrated_seconds} simulated seconds, ${value.calibration_history.length} sparse readings retained.`;
      },stop:()=>{}};
      status.textContent='Accumulating the unchanged 500 mock lux scene…';
      const duration=Number(watchDuration?.value??12);
      if(!Number.isSafeInteger(duration)||duration<12||duration>3600)throw Error('Watcher duration must be between 12 and 3600 Studio seconds');
      const longCase={...reference,exposure_seconds:duration};
      const result=await exposeCase(geometry,baseDigest,calib,longCase,cameraOptions,'fixed:8',peanutOptions,
        {regions,frameSeconds:1,onStart:digest=>{
          watchDigest=digest;
          for(const [lane,label] of [[0,'Watch RGB scene camera'],[1,'Watch monochrome peanut']]){
            const link=doc.createElement('a');
            link.href=`/marble_calibration_watch.html?digest=${digest}&lane=${lane}`;
            link.textContent=label;link.target='_blank';link.rel='noopener';
            link.style.cssText='display:inline-block;margin:0 12px 8px 0';watchView.append(link);
          }
        },
          canceled:()=>stopRequested,onStatus:value=>watcher?.update(value)});
      if(!stopRequested)status.textContent=result.state==='complete'?`Calibration watcher completed ${duration} seconds of Studio time. Start again for another accumulation.`:
        `Calibration watcher ended: ${result.error||result.state}`;
    }catch(error){if(!stopRequested)status.textContent=`Calibration watcher: ${error.message||error}`;}
    finally{
      watcher?.stop();watchDigest=null;stopButton.disabled=true;
      for(const {input,checked} of shopSwitches)if(input.checked!==checked){input.checked=checked;input.dispatchEvent(new Event('change',{bubbles:true}));}
      profileOwner?.dispatchEvent(new Event('room-calibration-end'));
      watchButton.disabled=false;runButton.disabled=false;upload.disabled=false;
    }
  });
  runButton.addEventListener("click", async () => {
    if (!calib) return;
    runButton.disabled = true; upload.disabled = true; downloadButton.hidden = true;
    try {
      const records = await runCalibrationSet(doc, ctx, calib, (message) => { status.textContent = message; });
      const report = gainReport(records.filter((r) => !r.error));
      if(calib.dual_camera){
        try{report.exposure=fitExposureControls(records);}
        catch(error){report.exposure={error:error.message||String(error)};}
      }
      renderGallery(doc, records, report);
      lastDataset = {schema:"marble-calibration-raw-v1", recipe:calib, records, report,
        rawArchiveLimitBytes:MAX_CALIBRATION_HDR_BYTES,
        profileFitted:!!report.exposure&&!report.exposure.error, captured_at:new Date().toISOString()};
      downloadButton.hidden = false;
      const failed = records.filter((r) => r.error).length;
      const rawCaptured=records.filter(r=>r.raytrace?.linear?.status==="captured").length;
      status.textContent = `Calibration complete: ${records.length - failed}/${records.length} image case(s); ${rawCaptured}/${records.length} main HDR archives retained. ${report.exposure&&!report.exposure.error?'Candidate Phong and monochrome response curves fitted.':'No response curve fitted.'}`;
    } catch (error) {
      status.textContent = `Calibration failed: ${error.message || error}`;
    } finally {
      runButton.disabled = false; upload.disabled = false;
    }
  });
  downloadButton.addEventListener("click", () => {
    if (!lastDataset) return;
    const blob = new Blob([JSON.stringify(lastDataset, null, 1)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = doc.createElement("a");
    a.href = url; a.download = `marble-calibration-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });
}

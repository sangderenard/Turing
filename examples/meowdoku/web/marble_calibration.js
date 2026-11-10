// Automatic colour calibration: upload a YAML recipe describing a sweep of
// single-light test cases; this module drives the ENTIRE sweep itself --
// zeroing every fixture, lighting exactly one at its declared colour,
// capturing the live Phong preview, exposing the identical rig through the
// Spectral ray tracer, and assembling a report + downloadable dataset --
// with no further manual slider changes once "Run calibration" is clicked.
import { packMarblePhoto } from "./marble_photo.js";
import { parseNpz } from "./npz.js";

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
function pixelsToPngDataUrl(width, height, data) {
  const scratch = document.createElement("canvas");
  scratch.width = width; scratch.height = height;
  scratch.getContext("2d").putImageData(new ImageData(data, width, height), 0, 0);
  return scratch.toDataURL("image/png");
}
async function sampleImageUrl(url) {
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
  return { ...seg, png: scratch.toDataURL("image/png") };
}

// ─────────────────────────────────────────────────────────────────────────
async function pollPhotoJob(digest, { intervalMs = 300, timeoutMs = 120000 } = {}) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const response = await fetch(`/api/marble_photo?digest=${digest}`, { cache: "no-store" });
    const value = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(value.error || `calibration: photo status HTTP ${response.status}`);
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

export async function captureLinearCalibration(status, budget={remaining:MAX_CALIBRATION_HDR_BYTES}) {
  const digest=status?.digest,filename=status?.hdr;
  if(!/^[a-f0-9]{64}$/.test(digest||"") || !/^exposure-\d{6}\.npz$/.test(filename||""))
    throw new Error("calibration: completed exposure has no valid raw HDR archive");
  if(!Number.isSafeInteger(budget.remaining)||budget.remaining<=0||budget.remaining>MAX_CALIBRATION_HDR_BYTES)
    throw new Error("calibration: raw HDR dataset limit reached (64 MiB)");
  const url=`/api/marble_photo/hdr?digest=${digest}&name=${encodeURIComponent(filename)}`;
  const response=await fetch(url,{cache:"no-store"});
  if(!response.ok)throw new Error(`calibration: HDR fetch HTTP ${response.status}`);
  const declared=Number(response.headers?.get("Content-Length"));
  if(Number.isFinite(declared)&&declared>budget.remaining)throw new Error("calibration: raw HDR dataset limit reached (64 MiB)");
  const buffer=await response.arrayBuffer();
  if(buffer.byteLength>budget.remaining)throw new Error("calibration: raw HDR dataset limit reached (64 MiB)");
  const arrays=await parseNpz(buffer,["linear_rgb_exposure","linear_srgb_exposure"]);
  const sensor=summarizeLinearArray(arrays.get("linear_rgb_exposure"));
  const profiled=arrays.has("linear_srgb_exposure")?summarizeLinearArray(arrays.get("linear_srgb_exposure")):null;
  const bytes=new Uint8Array(buffer),parts=[];
  for(let i=0;i<bytes.length;i+=32768)parts.push(String.fromCharCode(...bytes.subarray(i,i+32768)));
  const archiveBase64=btoa(parts.join(""));
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",buffer)),v=>v.toString(16).padStart(2,"0")).join("");
  budget.remaining-=buffer.byteLength;
  return {version:1,status:"captured",sensor,profiled,
    archive:{filename,sha256:hash,byteLength:buffer.byteLength,encoding:"base64",data:archiveBase64},
    provenance:{digest,transport:status.transport??null,filmResponse:status.film_response??null,
      colorProfile:status.color_profile??null,displayTransform:status.display_transform??null,
      displayWhite:status.display_white??null,cameraSettings:status.camera_settings??null,
      geometryStats:status.geometry_stats??null,elapsedSeconds:status.elapsed_seconds??null,
      frameElapsedSeconds:status.frame_elapsed_seconds??null,forwardRayJobs:status.forward_ray_jobs??null,
      sensorSites:status.sensor_sites??null,epoch:status.epoch??null},
    limitation:"Raw accumulated linear sensor values, not radiometrically calibrated units. Whole-frame statistics include scene transport and geometry; they are not a sensor-profile fit. PNG statistics are display-only."};
}

async function exposeCase(geometry, baseDigest, calib, getCameraOptions) {
  const body = packMarblePhoto(geometry, baseDigest, {
    resolution: calib.resolution ?? 128,
    seconds: calib.exposure_seconds ?? 0.1,
    maxBounces: calib.max_bounces ?? 2,
    transport: "fixed:3",
    gamma: 1.0,
    orbitFrames: 1,
    ...getCameraOptions(),
  });
  const response = await fetch("/api/marble_photo", { method: "POST",
    headers: { "Content-Type": "application/octet-stream" }, body });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.error || `calibration: expose HTTP ${response.status}`);
  return pollPhotoJob(value.digest);
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
  explanation.textContent="PNG ratios are display-only, not calibration gains. Download the dataset for exact raw HDR archives, unnormalized full-frame sensor statistics and recorded color/exposure provenance. A measured target and held-out tests are still required before fitting a profile.";
  gallery.append(explanation);
  const thumbs = document.createElement("div");
  thumbs.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;margin-top:6px";
  for (const r of records) {
    if (r.error) continue;
    for (const [label, src] of [["phong", r.phong.png], ["ray", r.raytrace.png]]) {
      const img = document.createElement("img");
      img.src = src; img.title = `${r.name} · ${label}`;
      img.style.cssText = "width:56px;height:56px;object-fit:cover;border:1px solid #536b82;border-radius:4px";
      thumbs.append(img);
    }
  }
  gallery.append(thumbs);
}

// ─────────────────────────────────────────────────────────────────────────
export async function runCalibrationSet(doc, ctx, calib, onProgress) {
  const { saveObject, getGeometry, getCameraOptions } = ctx;
  // Compare both renderers under A, retaining the user's B overrides intact.
  const profileOwner=doc.getElementById('lighting-profiles');
  profileOwner?.dispatchEvent(new Event('room-calibration-start'));
  try {
  applyRoomAndSubject(doc, calib);
  await settle(300);
  const baseDigest = await saveObject();
  if (!baseDigest) throw new Error("calibration: could not save the calibration subject design");
  const cases = calib.cases || [];
  if (!cases.length) throw new Error("calibration: recipe has no cases");
  const records = [], rawBudget={remaining:MAX_CALIBRATION_HDR_BYTES};
  for (const kase of cases) {
    onProgress?.(`Setting up "${kase.name}"…`);
    applyCase(doc, calib, kase);
    await settle(400);
    const geometry = getGeometry();
    if (!geometry) throw new Error("calibration: no marble geometry is ready");
    const phongShot = readGlPixels(ctx.gl, ctx.canvas);
    const phongSeg = segmentForeground(phongShot.width, phongShot.height, phongShot.data);
    const phongPng = pixelsToPngDataUrl(phongShot.width, phongShot.height, phongShot.data);
    onProgress?.(`Exposing "${kase.name}" on the ray tracer…`);
    try {
      const status = await exposeCase(geometry, baseDigest, calib, getCameraOptions);
      if (status.state !== "complete") {
        records.push({ name: kase.name, intendedRgb: hexToRgb(kase.filter), error: status.error || status.state });
        continue;
      }
      const imageUrl = `/api/marble_photo/image?digest=${status.digest || ""}&name=${encodeURIComponent(status.image)}`;
      const raytraceSample = await sampleImageUrl(imageUrl);
      let linear;
      try {linear=await captureLinearCalibration(status,rawBudget);}
      catch(error){linear={version:1,status:"unavailable",error:error.message||String(error)};}
      records.push({
        name: kase.name, intendedRgb: hexToRgb(kase.filter), intendedStrength: kase.strength ?? 0,
        phong: { mean: phongSeg.mean, coverage: phongSeg.coverage, png: phongPng },
        raytrace: { mean: raytraceSample.mean, coverage: raytraceSample.coverage, png: raytraceSample.png,
          displayWhite: status.display_white, filmResponse: status.film_response, linear },
      });
    } catch (error) {
      records.push({ name: kase.name, intendedRgb: hexToRgb(kase.filter), error: error.message || String(error) });
    }
    onProgress?.(`"${kase.name}" done.`);
  }
  return records;
  } finally {profileOwner?.dispatchEvent(new Event('room-calibration-end'));}
}

export function initCalibration(doc, ctx) {
  const upload = doc.getElementById("calibration-yaml-upload");
  const runButton = doc.getElementById("calibration-run");
  const status = doc.getElementById("calibration-status");
  const downloadButton = doc.getElementById("calibration-download");
  let calib = null, lastDataset = null;
  upload.addEventListener("change", async () => {
    const file = upload.files?.[0];
    runButton.disabled = true;
    downloadButton.hidden = true;
    if (!file) { calib = null; return; }
    try {
      calib = parseYaml(await file.text());
      if (!Array.isArray(calib.cases) || !calib.cases.length) throw new Error("recipe has no \"cases\" list");
      status.textContent = `Loaded "${calib.name || file.name}" · ${calib.cases.length} case(s). Click Run calibration.`;
      runButton.disabled = false;
    } catch (error) {
      calib = null;
      status.textContent = `Could not read calibration recipe: ${error.message || error}`;
    }
  });
  runButton.addEventListener("click", async () => {
    if (!calib) return;
    runButton.disabled = true; upload.disabled = true; downloadButton.hidden = true;
    try {
      const records = await runCalibrationSet(doc, ctx, calib, (message) => { status.textContent = message; });
      const report = gainReport(records.filter((r) => !r.error));
      renderGallery(doc, records, report);
      lastDataset = {schema:"marble-calibration-raw-v1", recipe:calib, records, report,
        rawArchiveLimitBytes:MAX_CALIBRATION_HDR_BYTES,
        profileFitted:false, captured_at:new Date().toISOString()};
      downloadButton.hidden = false;
      const failed = records.filter((r) => r.error).length;
      const rawCaptured=records.filter(r=>r.raytrace?.linear?.status==="captured").length;
      status.textContent = `Calibration complete: ${records.length - failed}/${records.length} image case(s); ${rawCaptured}/${records.length} raw HDR archives retained. No color profile fitted.`;
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

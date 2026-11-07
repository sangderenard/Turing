// [QRWALL] The page check: render the tip jar's QR wall in a real browser (its own headless Chrome, tests/meowdoku_cdp.mjs), capture the scene canvas
// head-on and at oblique angles, and have a QR decoder read each capture.  The decoder is tools/meowdoku_qr_decode.py (OpenCV's QRCodeDetector;
// run it first: `python tools/meowdoku_qr_decode.py --port 8961 --out <dir> [--lib <pip --target dir with cv2>]`); the decoded text must equal the
// jar's URI (the check prints MATCH / NO MATCH, never the address).  The page is a Meowdoku server on --port serving a SCRATCH copy of the web
// export (never the live site's):
//   node tests/meowdoku_qrwall_page_check.mjs --port 8960 --decoder 8961 --out <dir>
import { launch } from "./meowdoku_cdp.mjs";
import { mkdirSync } from "node:fs";

const arg = (name, dflt) => { const i = process.argv.indexOf("--" + name); return i > 0 ? process.argv[i + 1] : dflt; };
const port = arg("port", "8960"), decoder = arg("decoder", "8961"), out = arg("out", "."), width = Number(arg("width", "1100")), height = Number(arg("height", "800"));
mkdirSync(out, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const b = await launch({ port: Number(arg("chrome-port", "9261")), width, height });
let failed = 0;
try {
  await b.goto(`http://127.0.0.1:${port}/`);
  await b.eval("(()=>{const s=document.getElementById('game'); s.value='tipjar'; s.dispatchEvent(new Event('change'));})()");
  for (let i = 0; i < 80; i++) { if (await b.eval("!!(window.meowdokuDebug && meowdokuDebug.qrwall && meowdokuDebug.qrwall.state.mesh)")) break; await sleep(500); }
  if (!(await b.eval("!!meowdokuDebug.qrwall.state.mesh"))) throw new Error("the QR wall was never built: " + b.logs.slice(-5).join(" | "));
  await sleep(1500);
  // the views: [name, pitch (rad, 0 = head-on; the angle off the wall's normal), yaw (rad), head-on goal first?]
  const views = [["head_on", 0, 0, true], ["oblique25", -25 * Math.PI / 180, 0, false], ["oblique40", -40 * Math.PI / 180, 0, false],
                 ["oblique_diag35", -35 * Math.PI / 180, 0.5, false], ["default_view_framed", -0.55, 0.4, false], ["default_view_as_opened", null, null, false]];
  for (const [name, pitch, yaw, headOn] of views) {
    if (headOn) await b.eval("meowdokuDebug.qrwall.headOn()");
    else if (pitch === null) await b.eval("meowdokuDebug.resetCameraView(); 0");       // the view the page opens in, nothing adjusted
    else await b.eval(`(() => {                                              // an ORBIT to (pitch, yaw), then the pan and zoom a user would make to keep the wall in view
      const q = meowdokuDebug.qrwall, c = meowdokuDebug.camView, w = q.state.wall, canvas = document.getElementById("gl"), r = canvas.getBoundingClientRect();
      const panel = document.getElementById("tipjar").getBoundingClientRect(), right = Math.max(0, r.right - panel.left);
      const centreX = -right / r.width, spanX = 2 - 2 * right / r.width, h = w.size / 2;
      const corners = [[-h, -h], [h, -h], [h, h], [-h, h]].map(([a, b]) => [w.cx + a, w.y, w.cz + b]);
      const set = (lz, px, pz) => { for (const o of [c.cur, c.goal]) Object.assign(o, { pitch: ${pitch}, yaw: ${yaw}, lz, px, pz }); };
      const meas = (lz, px, pz) => { set(lz, px, pz); const n = corners.map((p) => q.project(p)), xs = n.map((v) => v[0]), ys = n.map((v) => v[1]);
        return { cx: (Math.min(...xs) + Math.max(...xs)) / 2, cy: (Math.min(...ys) + Math.max(...ys)) / 2, sx: Math.max(...xs) - Math.min(...xs), sy: Math.max(...ys) - Math.min(...ys) }; };
      let lz = c.cur.lz, px = c.cur.px, pz = c.cur.pz;
      for (let it = 0; it < 40; it++) {
        const m = meas(lz, px, pz), e = 0.02, a = meas(lz, px + e, pz), b2 = meas(lz, px, pz + e);
        lz += Math.log(Math.max(m.sx / (spanX * 0.92), m.sy / (2 * 0.92)));
        const J = [[(a.cx - m.cx) / e, (b2.cx - m.cx) / e], [(a.cy - m.cy) / e, (b2.cy - m.cy) / e]], det = J[0][0] * J[1][1] - J[0][1] * J[1][0], ex = centreX - m.cx, ey = -m.cy;
        if (Math.abs(det) > 1e-9) { px += (J[1][1] * ex - J[0][1] * ey) / det; pz += (J[0][0] * ey - J[1][0] * ex) / det; }
      }
      set(lz, px, pz); return [lz, px, pz];
    })()`);
    await sleep(2500);                                                       // the camera eases to its goal
    const posted = await b.eval(`(async () => {
      meowdokuDebug.drawFrame(performance.now());
      const canvas = document.getElementById("gl"), png = canvas.toDataURL("image/png"), t = meow.game.tipjar.uri;
      const r = await (await fetch("http://127.0.0.1:${decoder}/", { method: "POST", headers: { "Content-Type": "text/plain" }, body: JSON.stringify({ name: ${JSON.stringify(name)}, expected: t, png, extra: "canvas " + canvas.width + "x" + canvas.height }) })).json();
      return r.verdict; })()`);
    await b.shot(`${out}/${name}_page.png`);
    console.log(`${name}: ${posted}`);
    if (posted !== "MATCH") failed++;
  }
  // the cost: GPU-synchronised (a one-pixel readback) frames with the wall and with the toggle off
  const cost = `(() => { const gl = document.getElementById("gl").getContext("webgl2"), px = new Uint8Array(4); const t0 = performance.now();
    for (let i = 0; i < 60; i++) { meowdokuDebug.drawFrame(performance.now()); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); } return (performance.now() - t0) / 60; })()`;
  await b.eval("meowdokuDebug.qrwall.headOn(); 0"); await sleep(2000);
  const withWall = await b.eval(cost);
  const spec = await b.eval(`(() => { const el = document.querySelector('[data-ctl="qrwall"]'); return !!el; })()`);
  await b.eval("localStorage.setItem('meowdoku.qrwall', '0'); meowdokuDebug.qrwall.state.on = false; 0");
  await sleep(500);
  const without = await b.eval(cost);
  const gone = await b.eval("!meowdokuDebug.qrwall.state.mesh");
  console.log(`ms/frame (GPU-synced) with the wall ${withWall.toFixed(2)}, without ${without.toFixed(2)}; toggle control present: ${spec}; wall gone when off: ${gone}`);
  if (!spec || !gone) failed++;
} finally { b.close(); }
process.exit(failed ? 1 : 0);

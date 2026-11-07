// A tiny Chrome DevTools driver for the Meowdoku page checks (no npm packages: Node's own WebSocket and fetch).  It launches its OWN Chrome
// (a scratch profile and a debugging port), so a shared browser pane is never touched.
//   const b = await launch({ port: 9261, width: 1100, height: 800 }); await b.goto(url); const v = await b.eval("1+1"); await b.shot(path); b.close();
import { spawn } from "node:child_process";
import { writeFileSync, mkdtempSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROMES = ["C:/Program Files/Google/Chrome/Application/chrome.exe", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"];

export async function launch({ port = 9261, width = 1100, height = 800, headless = true } = {}) {
  const exe = process.env.CHROME || CHROMES.find(existsSync);
  if (!exe) throw new Error("no Chrome or Edge found (set CHROME)");
  const profile = mkdtempSync(join(tmpdir(), "meow-cdp-"));
  const args = [`--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, `--window-size=${width},${height}`, "--no-first-run", "--no-default-browser-check",
    "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--use-angle=d3d11", "about:blank"];
  if (headless) args.unshift("--headless=new");
  const proc = spawn(exe, args, { stdio: "ignore", windowsHide: true });
  let targets = null;
  for (let i = 0; i < 60 && !targets; i++) {
    await new Promise((r) => setTimeout(r, 250));
    try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); } catch { /* not up yet */ }
  }
  const page = targets && targets.find((t) => t.type === "page");
  if (!page) { proc.kill(); throw new Error("Chrome did not start"); }
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 0; const pending = new Map(); const logs = [];
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { const { res, rej } = pending.get(m.id); pending.delete(m.id); m.error ? rej(new Error(JSON.stringify(m.error))) : res(m.result); }
    else if (m.method === "Runtime.consoleAPICalled") logs.push(m.params.args.map((a) => a.value ?? a.description).join(" "));
    else if (m.method === "Runtime.exceptionThrown") logs.push("EXCEPTION " + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
  };
  const send = (method, params = {}) => new Promise((res, rej) => { const i = ++id; pending.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params })); });
  await send("Runtime.enable"); await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
  return {
    logs, send,
    async goto(url) { await send("Page.navigate", { url }); await new Promise((r) => setTimeout(r, 1500)); },
    async eval(expr) {
      const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true, timeout: 120000 });
      if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
      return r.result.value;
    },
    async shot(path) { const r = await send("Page.captureScreenshot", { format: "png" }); writeFileSync(path, Buffer.from(r.data, "base64")); },
    close() { try { ws.close(); } catch { /* closed */ } proc.kill(); },
  };
}

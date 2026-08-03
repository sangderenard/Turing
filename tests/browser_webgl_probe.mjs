import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {spawn} from "node:child_process";

const [chrome, pageURL, screenshotPath, minimumRevisionText] = process.argv.slice(2);
const minimumRevision = Math.max(1, Number(minimumRevisionText || 1));
if (!chrome || !pageURL) {
  throw new Error("usage: node browser_webgl_probe.mjs <chrome> <page-url>");
}

const profile = await fs.mkdtemp(path.join(os.tmpdir(), "turing-webgl-probe-"));
const browser = spawn(chrome, [
  "--headless=new",
  "--no-first-run",
  "--no-default-browser-check",
  "--disable-background-networking",
  "--enable-unsafe-swiftshader",
  "--use-angle=swiftshader",
  "--remote-allow-origins=*",
  "--remote-debugging-port=0",
  `--user-data-dir=${profile}`,
  pageURL,
], {stdio: ["ignore", "ignore", "pipe"]});

const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
async function waitFor(callback, timeout = 30000) {
  const deadline = Date.now() + timeout;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      const result = await callback();
      if (result) return result;
    } catch (error) {
      lastError = error;
    }
    await delay(50);
  }
  throw lastError || new Error("browser probe timed out");
}

let socket = null;
try {
  const portFile = path.join(profile, "DevToolsActivePort");
  const port = await waitFor(async () => {
    const body = await fs.readFile(portFile, "utf8");
    return Number(body.split(/\r?\n/, 1)[0]);
  });
  const target = await waitFor(async () => {
    const response = await fetch(`http://127.0.0.1:${port}/json`);
    const targets = await response.json();
    return targets.find(item => item.type === "page" && item.url === pageURL);
  });
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, {once: true});
    socket.addEventListener("error", reject, {once: true});
  });
  let nextID = 1;
  function request(method, params) {
    const id = nextID++;
    return new Promise((resolve, reject) => {
      const receive = event => {
        const message = JSON.parse(event.data);
        if (message.id !== id) return;
        socket.removeEventListener("message", receive);
        if (message.error) reject(new Error(message.error.message));
        else resolve(message.result);
      };
      socket.addEventListener("message", receive);
      socket.send(JSON.stringify({id, method, params}));
    });
  }
  await waitFor(async () => {
    const state = await request("Runtime.evaluate", {
      expression: "({ready: document.readyState, href: location.href})",
      returnByValue: true,
    });
    const value = state.result && state.result.value;
    return value && value.ready === "complete" && value.href === pageURL;
  });
  const expression = `(async () => {
    const deadline = performance.now() + 30000;
    while (!window.TuringShaderLiaison && performance.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    const liaison = window.TuringShaderLiaison;
    if (!liaison) throw new Error("shader liaison was not installed");
    await liaison.ready;
    while ((!liaison.wasm || liaison.wasm.outputFrame().revision < ${minimumRevision}) &&
           performance.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    await new Promise(resolve => requestAnimationFrame(() =>
      requestAnimationFrame(resolve)));
    liaison.gl.finish();
    const x = Math.floor(liaison.canvas.width / 2);
    const y = Math.floor(liaison.canvas.height / 2);
    const pixel = new Uint8Array(4);
    liaison.gl.readPixels(x, y, 1, 1, liaison.gl.RGBA,
                          liaison.gl.UNSIGNED_BYTE, pixel);
    return {
      error: liaison.canvas.dataset.error || null,
      revision: liaison.wasm ? liaison.wasm.outputFrame().revision : -1,
      running: liaison.wasm ? liaison.wasm.running : false,
      status: document.getElementById("status")?.textContent || "",
      outputCount: liaison.wasm ? liaison.wasm.outputFrame().outputs.length : 0,
      width: liaison.canvas.width,
      height: liaison.canvas.height,
      center: Array.from(pixel),
      glError: liaison.gl.getError(),
    };
  })()`;
  const evaluated = await request("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (evaluated.exceptionDetails) {
    throw new Error(evaluated.exceptionDetails.text || "page evaluation failed");
  }
  if (screenshotPath) {
    const screenshot = await request("Page.captureScreenshot", {
      format: "png",
      fromSurface: true,
    });
    await fs.writeFile(screenshotPath, Buffer.from(screenshot.data, "base64"));
  }
  process.stdout.write(JSON.stringify(evaluated.result.value));
} finally {
  if (socket) socket.close();
  browser.kill();
  if (browser.exitCode === null) {
    await Promise.race([
      new Promise(resolve => browser.once("exit", resolve)),
      delay(5000),
    ]);
  }
  await fs.rm(profile, {recursive: true, force: true, maxRetries: 5, retryDelay: 100});
}

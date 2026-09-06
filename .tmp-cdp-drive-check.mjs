const targets = await (await fetch("http://127.0.0.1:9223/json")).json();
const target = targets.find(item => item.type === "page" && item.url.startsWith("http://127.0.0.1:8787/"));
if (!target) throw new Error("localhost browser target not found");
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, {once: true});
  socket.addEventListener("error", reject, {once: true});
});
let nextId = 1;
const pending = new Map();
socket.addEventListener("message", event => {
  const message = JSON.parse(event.data);
  if (!message.id) return;
  const request = pending.get(message.id);
  if (!request) return;
  pending.delete(message.id);
  if (message.error) request.reject(new Error(JSON.stringify(message.error)));
  else request.resolve(message.result);
});
function call(method, params = {}) {
  const id = nextId++;
  socket.send(JSON.stringify({id, method, params}));
  return new Promise((resolve, reject) => pending.set(id, {resolve, reject}));
}
const wait = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
async function evaluate(expression) {
  const result = await call("Runtime.evaluate", {expression, awaitPromise: true, returnByValue: true});
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
  return result.result.value;
}
await call("Runtime.enable");
await call("Page.enable");
await call("Page.navigate", {url: `http://127.0.0.1:8787/?drive-check=${Date.now()}`});
await wait(12000);
const read = `(() => ({
  computeMode: vehicleRuntime.computeMode,
  ready: stateLoopRuntime.ready,
  forcingWasm: stateLoopRuntime.forcingWasm,
  lastWorkerCrash: stateLoopRuntime.lastWorkerCrash,
  error: vehicleRuntime.error,
  rpm: vehicleRuntime.powertrain?.engineRPM || 0,
  engineTorque: vehicleRuntime.powertrain?.engineTorque || 0,
  drivelineTorque: vehicleRuntime.powertrain?.drivelineTorque || 0,
  wheelOmegas: vehicleRuntime.state?.wheelOmegas || null,
  position: vehicleRuntime.state?.position || null,
  shockValues: Object.fromEntries([...document.querySelectorAll("[data-shock-parameter]")]
    .map(input=>[input.dataset.shockParameter,Number(input.value)])),
  active: vehicleRuntime.active?.identity || null
}))()`;
const before = await evaluate(read);
const crashSource = await evaluate(`fetch(stateLoopRuntime.workerUrl).then(response=>response.text()).then(text=>
  text.split("\\n").slice(1025,1037).map((line,index)=>String(index+1026).padStart(4," ")+": "+line).join("\\n"))`);
await call("Input.dispatchKeyEvent", {type:"keyDown",code:"KeyW",key:"w",windowsVirtualKeyCode:87,nativeVirtualKeyCode:87});
await wait(3000);
const during = await evaluate(read);
await call("Input.dispatchKeyEvent", {type:"keyUp",code:"KeyW",key:"w",windowsVirtualKeyCode:87,nativeVirtualKeyCode:87});
await wait(500);
const after = await evaluate(read);
await evaluate(`stateLoopRuntime.worker.postMessage({type:"support",identity:vehicleRuntime.active.identity,y:-20})`);
await wait(1500);
const recovered = await evaluate(read);
await call("Input.dispatchKeyEvent", {type:"keyDown",code:"KeyW",key:"w",windowsVirtualKeyCode:87,nativeVirtualKeyCode:87});
await wait(2000);
await call("Input.dispatchKeyEvent", {type:"keyUp",code:"KeyW",key:"w",windowsVirtualKeyCode:87,nativeVirtualKeyCode:87});
await wait(1000);
const recoveredDrive = await evaluate(read);
console.log(JSON.stringify({crashSource, before, during, after, recovered, recoveredDrive}, null, 2));
socket.close();

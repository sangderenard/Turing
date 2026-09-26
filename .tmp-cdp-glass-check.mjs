import {writeFile} from "node:fs/promises";
const targets=await (await fetch("http://127.0.0.1:9223/json")).json();
const target=targets.find(item=>item.type==="page"&&item.url.startsWith("http://127.0.0.1:8787/"));
if(!target)throw new Error("localhost browser target not found");
const socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{
  socket.addEventListener("open",resolve,{once:true});socket.addEventListener("error",reject,{once:true});});
let nextId=1;const pending=new Map();socket.addEventListener("message",event=>{const message=JSON.parse(event.data);
  if(!message.id||!pending.has(message.id))return;const {resolve,reject}=pending.get(message.id);pending.delete(message.id);
  message.error?reject(new Error(JSON.stringify(message.error))):resolve(message.result);});
const call=(method,params={})=>{const id=nextId++;socket.send(JSON.stringify({id,method,params}));
  return new Promise((resolve,reject)=>pending.set(id,{resolve,reject}));};
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
await call("Runtime.enable");await call("Page.enable");
await call("Page.navigate",{url:`http://127.0.0.1:8787/?glass-check=${Date.now()}`});await wait(12000);
const inspected=await call("Runtime.evaluate",{expression:`(() => ({
  mode:vehicleRuntime.computeMode,ready:stateLoopRuntime.ready,error:vehicleRuntime.error,
  shell:vehicleRuntime.bodyShell,shellOptions:model.vehicle_slot.vehicles[0].body_shells.map(item=>item.identity),
  shellRoles:vehicleRuntime.bodyShellBoxes.map(item=>item.palette_role),
  shellColor:model.appearance.colors["body-shell-glass"],
  renderPass:shaderViewer.locations.uRenderPass!==null,
  linkage:(()=>{const graph=vehicleRuntime.active.physics.mechanical_graph,edge=id=>graph.edges.find(item=>item.identity===id);
    return {halfshaft:edge("drivetrain.front_left_halfshaft"),outerCv:edge("suspension.front_left.outer_halfshaft_joint"),
      steeringArm:edge("suspension.front_left.steering_arm"),joint:graph.nodes.find(item=>item.identity==="suspension.front_left.halfshaft_joint")};})(),
  shockValues:Object.fromEntries([...document.querySelectorAll("[data-shock-parameter]")].map(input=>[input.dataset.shockParameter,Number(input.value)]))
}))()`,returnByValue:true});
const shot=await call("Page.captureScreenshot",{format:"png",captureBeyondViewport:false});
await writeFile(".tmp-glass-check.png",Buffer.from(shot.data,"base64"));
console.log(JSON.stringify(inspected.result.value,null,2));socket.close();

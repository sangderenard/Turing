import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {bindMarbleDriveInput, marbleDriveAxes} from "../examples/meowdoku/web/marble_drive_input.js";
import {cameraNudgeGravity} from "../examples/meowdoku/web/marble_sim_preview.js";

class Target {
  listeners = new Map(); attrs = new Map(); hidden = false;
  addEventListener(type, fn) { if(!this.listeners.has(type))this.listeners.set(type,new Set());this.listeners.get(type).add(fn); }
  removeEventListener(type, fn) { this.listeners.get(type)?.delete(fn); }
  getAttribute(name) { return this.attrs.get(name) ?? null; }
  setAttribute(name, value) { this.attrs.set(name,value); }
  removeAttribute(name) { this.attrs.delete(name); }
  focus(options) { this.focusOptions=options; }
  emit(type, fields={}) { const event={prevented:false,preventDefault(){this.prevented=true;},...fields};
    for(const fn of this.listeners.get(type)||[])fn(event);return event; }
}
const canvas=new Target(), host=new Target(), doc=new Target(), updates=[];
let enabled=true;
const drive=bindMarbleDriveInput(canvas,{window:host,document:doc,enabled:()=>enabled,onChange:axes=>updates.push(axes)});
assert.equal(canvas.getAttribute("tabindex"),"0");
canvas.emit("pointerdown");assert.deepEqual(canvas.focusOptions,{preventScroll:true});
assert.deepEqual(marbleDriveAxes(["KeyW","ArrowUp","KeyS"]),[0,0]);
assert.deepEqual(marbleDriveAxes(["KeyA","KeyW"]),[-1,1]);
assert.equal(canvas.emit("keydown",{code:"KeyW"}).prevented,true);
assert.deepEqual(drive.axes(),[0,1]);
canvas.emit("keydown",{code:"KeyW",repeat:true});assert.equal(updates.length,1);
canvas.emit("keydown",{code:"KeyD"});assert.deepEqual(drive.axes(),[1,1]);
const gravity=cameraNudgeGravity(...drive.axes(),0);
assert.ok(Math.abs(Math.hypot(...gravity)-1)<1e-12);
assert.ok(Math.abs(Math.hypot(gravity[0],gravity[1])-.72)<1e-12);
host.emit("keyup",{code:"KeyW"});assert.deepEqual(drive.axes(),[1,0]);
canvas.emit("blur");assert.deepEqual(drive.axes(),[0,0]);
// Form edits cannot reach the canvas listener. A window keydown is ignored.
host.emit("keydown",{code:"ArrowUp"});assert.deepEqual(drive.axes(),[0,0]);
for(const fields of [{ctrlKey:true},{metaKey:true},{altKey:true},{isComposing:true}]){
  assert.equal(canvas.emit("keydown",{code:"KeyW",...fields}).prevented,false);
}
canvas.emit("keydown",{code:"KeyA"});host.emit("blur");assert.deepEqual(drive.axes(),[0,0]);
canvas.emit("keydown",{code:"KeyS"});doc.hidden=true;doc.emit("visibilitychange");assert.deepEqual(drive.axes(),[0,0]);
doc.hidden=false;canvas.emit("keydown",{code:"KeyW"});enabled=false;assert.deepEqual(drive.axes(),[0,0]);
assert.equal(canvas.emit("keydown",{code:"ArrowUp"}).prevented,false);
enabled=true;canvas.emit("keydown",{code:"KeyW"});drive.clear();assert.deepEqual(drive.axes(),[0,0]);
canvas.emit("keydown",{code:"KeyW"});drive.dispose();assert.equal(canvas.getAttribute("tabindex"),null);
canvas.emit("keydown",{code:"KeyD"});assert.deepEqual(drive.axes(),[0,0]);
canvas.setAttribute("tabindex","2");const second=bindMarbleDriveInput(canvas,{window:host,document:doc});
second.dispose();assert.equal(canvas.getAttribute("tabindex"),"2");
// Execute the real Studio adapter with the real gravity mapper. It must not
// replace positions/velocities or introduce a separate body integration loop.
const source=readFileSync(new URL("../examples/meowdoku/web/marble_studio.js",import.meta.url),"utf8");
const adapter=source.match(/function driveGravity\(\) \{[\s\S]*?\n\}/)?.[0];
assert.ok(adapter);
const intent={direction:"neutral",keyboard:[1,1]}, output={},calls=[];
const run=new Function("drive","sim","driveViewYaw","$","cameraNudgeGravity","DRIVE",`${adapter};driveGravity();`);
const directions={neutral:[0,0],left:[-1,0],right:[1,0],up:[0,1],down:[0,-1]};
const invoke=yaw=>run(intent,{runtime:{ready:true,setGravity:(...v)=>calls.push(v)}},()=>yaw,()=>output,cameraNudgeGravity,directions);
invoke(0);assert.equal(intent.active,true);assert.match(output.textContent,/up\/right/);
assert.deepEqual(calls.at(-1),cameraNudgeGravity(1,1,0));
invoke(Math.PI/2);assert.deepEqual(calls.at(-1),cameraNudgeGravity(1,1,Math.PI/2));
intent.direction="left";invoke(0);assert.deepEqual(calls.at(-1),cameraNudgeGravity(-1,0,0));
intent.direction="neutral";intent.keyboard=[0,0];invoke(0);
assert.equal(intent.active,false);assert.deepEqual(calls.at(-1),[0,0,-1]);
assert.match(source,/if\(drive.active&&drive.yaw!==driveViewYaw\(\)\)driveGravity\(\)/);
assert.match(source,/else \{drive.direction="neutral";drive.input\?\.clear\(\);driveGravity\(\)/);
console.log("PASS marble drive input: focused keys, aliases, diagonals, native gravity, release and interruption cleanup");

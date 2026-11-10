// The shared Studio/native coordinate boundary. This module does not activate
// a product or build a second scene. Pass exact authored indexed parts, not
// visually similar SDFs or proxy colliders. Whole-scene wiring is gated on its
// topology/manifold certificate; room/table/photo drawing stays unchanged.
import {studioScaleTransform} from "./marble_studio_scale.js";
import {encodeStateStart, decodeStateStart} from "./compute_state.js";

export const STUDIO_SCENE_SCHEMA = "turing.studio-rigid-scene.v1";
export const STUDIO_NATIVE_STATE = ["x","y","z","vx","vy","vz","wx","wy","wz","age","rs"];

function vector(value, length, name) {
  if (!Array.isArray(value) || value.length !== length || !value.every(Number.isFinite))
    throw new TypeError(`${name} must have ${length} finite coordinates`);
  return [...value];
}

export function studioPhysicsFrame({scale, radiusMm, tableEnabled, room}) {
  const transform = studioScaleTransform(scale, radiusMm, tableEnabled);
  return nativeFrame(transform,room);
}

// A displayed capture already contains the exact Studio metre reference and
// current world pose. Never re-evaluate a saved Exposure profile to seed it.
export function studioCapturedPhysicsFrame(capture) {
  const state=capture?.state,unitM=state?.frame?.unitM,radius=state?.subject?.radius;
  if(state?.schema!=="turing.studio-visible-scene.v1"||state.channel!=="studio"||!state.room||
      !Number.isFinite(unitM)||unitM<=0||!Number.isFinite(radius)||radius<=0)
    throw new Error("A displayed Studio room and subject scale are required");
  if(state.authoring?.object?.design?.["base-shape"]!=="marble")
    throw new Error("The native spherical particle law requires an authored marble");
  const center=vector(state.subject.position,3,"Displayed subject position");
  return nativeFrame({unitM,radiusM:radius*unitM,subjectRadius:radius,subjectCenter:center},state.room);
}

function nativeFrame(transform,room) {
  const low = [room.leftX,room.floorY,room.backZ], high = [room.rightX,room.ceilingY,room.frontZ];
  vector(low,3,"Room lower bound"); vector(high,3,"Room upper bound");
  if (low.some((v,i)=>v >= high[i])) throw new RangeError("Invalid Studio room bounds");
  // Translation avoids the existing law's parked (0,0,0) sentinel and fixed
  // Y-grid origin. It is independent of marble size in a fixed Studio rig.
  // Axes are identical Y-up right-handed frames, so quaternion/spin need no
  // reflection, quarter-turn or radius-dependent adaptation.
  const originM = [0,(1-room.floorY)*transform.unitM,0];
  const toNative = point => vector(point,3,"Studio point").map((v,i)=>v*transform.unitM+originM[i]);
  const toStudio = point => vector(point,3,"Native point").map((v,i)=>(v-originM[i])/transform.unitM);
  return {...transform, originM, toNative, toStudio,
    room: {low:toNative(low),high:toNative(high)}};
}

export function studioRigidPacket(frame, parts) {
  if (!Array.isArray(parts)) throw new TypeError("Authored Studio parts are required");
  return {schema:STUDIO_SCENE_SCHEMA, frame:{unitM:frame.unitM,originM:[...frame.originM]},
    room:structuredClone(frame.room), parts:parts.map(part=>{
      if (part.convex !== true || typeof part.closed !== "boolean")
        throw new Error(`Unsupported Studio rigid topology: ${part.id}`);
      // This copies every source identity and triangle unchanged. The native
      // admission gate validates closedness, convexity and compound ownership.
      return {...structuredClone(part), vertices:part.vertices.map(frame.toNative)};
    })};
}

export function studioNativeStart(frame, {position=frame.subjectCenter,
  quaternion=[0,0,0,1], velocityM=[0,0,0], angularVelocity=[0,0,0], age=0}={}, pad=2) {
  const p=frame.toNative(position), v=vector(velocityM,3,"Velocity in metres per second");
  const w=vector(angularVelocity,3,"Angular velocity in radians per second");
  const q=vector(quaternion,4,"Orientation");
  if (Math.abs(Math.hypot(...q)-1)>1e-6) throw new RangeError("Orientation must be a unit quaternion");
  if (!Number.isFinite(age) || age < 0) throw new RangeError("Invalid native age");
  if (p.some((value,i)=>value-frame.radiusM < frame.room.low[i]-1e-12 ||
      value+frame.radiusM > frame.room.high[i]+1e-12))
    throw new RangeError("The marble does not fit inside this Studio room at its displayed pose");
  const values=[...p,...v,...w,age,frame.radiusM];
  const columns=Object.fromEntries(STUDIO_NATIVE_STATE.map((name,i)=>[name,[values[i]]]));
  return encodeStateStart(columns,pad,STUDIO_NATIVE_STATE,[q]);
}

// Apply only a successful exact runtime.pauseAtDisplayedState() result. The
// static preview/photo/live consumers share this returned pose instead of
// resetting the marble or deriving orientation from position. Camera state is
// intentionally absent: entering/leaving physics never reframes it here.
export function studioPoseFromHandoff(frame, handoff, pad, id=0) {
  const columns=decodeStateStart(handoff.start,pad,STUDIO_NATIVE_STATE);
  if (!Number.isInteger(id) || id < 0 || id >= pad) throw new RangeError("Invalid particle ID");
  const quaternion=vector(handoff.start.orientations_xyzw?.[id],4,"Recorded orientation");
  if (Math.abs(Math.hypot(...quaternion)-1)>1e-6) throw new RangeError("Recorded orientation is not unit length");
  const point=[columns.x[id],columns.y[id],columns.z[id]];
  if (point.every(v=>v===0)) throw new Error("A parked particle has no displayed Studio pose");
  return {position:frame.toStudio(point), quaternion,
    velocityM:[columns.vx[id],columns.vy[id],columns.vz[id]],
    angularVelocity:[columns.wx[id],columns.wy[id],columns.wz[id]],
    radiusM:columns.rs[id],age:columns.age[id],frame:handoff.frame,step:handoff.step};
}

// Preserve the original one-second bounded replay allowance when the exact
// same native law uses smaller physical steps for a small marble. This does
// not widen other games' default or permit an unbounded reconstruction.
export function studioHandoffOptions(manifest, maxReplaySeconds=1) {
  const dt=manifest?.law?.dt;
  if (!Number.isFinite(dt) || dt<=0 || !Number.isFinite(maxReplaySeconds) ||
      maxReplaySeconds<0 || maxReplaySeconds>1)
    throw new RangeError("Studio handoff needs a valid timestep and at most one second of replay");
  const steps=Math.max(0,Math.ceil(maxReplaySeconds/dt-1e-9));
  if (!Number.isSafeInteger(steps) || steps>5760)
    throw new RangeError("Studio handoff exceeds the built native-step budget");
  return {maxReplaySteps:steps};
}

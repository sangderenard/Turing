// Existing Living Data Map walking controller, extracted without changing its
// function bodies from src/compiler/abstract_ui_div_map.py. The parity tests
// reject divergence from that source. This is NOT a new movement/physics law.
//
// The caller owns the real authored world, colliders, support sampler, physics
// runtime, actor state, action handlers and clock. All dependencies are required;
// there is deliberately no collision-free, camera-only or fake-body fallback.
// Coordinates are Y-up and viewportControls.position is the player's EYE.
// updateViewportControls(dt) and stepCompiledWorldPhysics(dt) are the incumbent
// separate phases; call them from the existing owner, not a new animation loop.
// This preserves the original position-driven walking and does not establish
// reciprocal native Studio player/marble contact.
//
// Use the actual serialized model.viewer.control_policy, including its authored
// scene overrides. POLICY_DEFAULTS below are the class defaults, not a replacement
// for the Living Map model's move_speed=1.65 and scaled embodiment. Context state
// records remain identity-stable; update their fields in place. In particular,
// latestControlInput is a holder whose position field follows the current input.
// attach() binds key/look/focus listeners only, returns an idempotent detach,
// and never starts a clock. The owner retains click capture, pointer tool actions,
// render-camera binding, world navigation, contact solving and snapshot delivery.

const freezeData = value => {
  if (value && typeof value === "object") {
    Object.values(value).forEach(freezeData);
    Object.freeze(value);
  }
  return value;
};

export const LIVING_MAP_VIEWPORT_BINDINGS = freezeData([
  {
    "action": "move-forward",
    "inputs": [
      "keyboard:KeyW",
      "gamepad:left-y-negative"
    ]
  },
  {
    "action": "move-backward",
    "inputs": [
      "keyboard:KeyS",
      "gamepad:left-y-positive"
    ]
  },
  {
    "action": "strafe-left",
    "inputs": [
      "keyboard:KeyA",
      "gamepad:left-x-negative"
    ]
  },
  {
    "action": "strafe-right",
    "inputs": [
      "keyboard:KeyD",
      "gamepad:left-x-positive"
    ]
  },
  {
    "action": "run",
    "inputs": [
      "keyboard:ShiftLeft",
      "keyboard:ShiftRight"
    ]
  },
  {
    "action": "jump",
    "inputs": [
      "keyboard:Space"
    ]
  },
  {
    "action": "respawn",
    "inputs": [
      "keyboard:KeyR",
      "gamepad:button-3"
    ]
  },
  {
    "action": "front-differential-brake-toggle",
    "inputs": [
      "keyboard:KeyZ",
      "gamepad:button-4"
    ]
  },
  {
    "action": "rear-differential-brake-toggle",
    "inputs": [
      "keyboard:KeyX",
      "gamepad:button-4"
    ]
  },
  {
    "action": "vehicle-headlights-toggle",
    "inputs": [
      "keyboard:KeyL"
    ]
  },
  {
    "action": "vehicle-horn",
    "inputs": [
      "keyboard:KeyH"
    ]
  },
  {
    "action": "vehicle-ignition-toggle",
    "inputs": [
      "keyboard:KeyK"
    ]
  },
  {
    "action": "vehicle-starter",
    "inputs": [
      "keyboard:KeyI"
    ]
  },
  {
    "action": "look",
    "inputs": [
      "pointer:relative-motion",
      "gamepad:right-stick"
    ]
  },
  {
    "action": "primary-action",
    "inputs": [
      "pointer:button-0",
      "gamepad:button-0"
    ]
  },
  {
    "action": "secondary-action",
    "inputs": [
      "pointer:button-2",
      "gamepad:button-1"
    ]
  }
]);

export const LIVING_MAP_VIEWPORT_POLICY_DEFAULTS = freezeData({
  "activation": "highlight",
  "release": [
    "escape",
    "focus-loss"
  ],
  "captures": [
    "keyboard",
    "pointer",
    "gamepad"
  ],
  "pointer_mode": "relative-when-available",
  "gamepad_selection": "first-connected",
  "move_speed": 3.2,
  "run_multiplier": 2.0,
  "jump_speed": 3.6,
  "look_sensitivity": 0.0025
});

export const LIVING_MAP_WALK_REQUIRED_STATE = Object.freeze([
  "model", "viewportControls", "shaderViewer", "entityState", "physicsRuntime",
  "portalRuntime", "mobileControlState", "vehicleRuntime", "controlFocus",
  "stateLoopRuntime", "latestControlInput", "primaryActionState", "secondaryActionState",
  "navigator", "document", "performance",
]);

export const LIVING_MAP_WALK_REQUIRED_CALLBACKS = Object.freeze([
  "setShaderOnlyMode", "closeToolDialogue", "selectHotbarSlot", "cycleActiveToolMode",
  "clearActiveVehicle", "setActiveVehicle", "closeSceneContextMenu",
  "beginViewportPrimary", "endViewportPrimary", "beginViewportSecondary", "endViewportSecondary",
  "respawnViewportActor", "controlVehicleTransmission", "controlVehicleAuxiliary",
  "controlVehicleFuelIgnition", "cancelEntityNavigation", "updateActiveVehicle",
  "requestRepresentationTransition", "sampleContactSurface", "reservePhysicsSnapshotSlot",
  "runCompiledPhysicsState",
]);

export const LIVING_MAP_WALK_SOURCE_FUNCTIONS = Object.freeze([
  "normalized3",
  "mobileClamp",
  "viewportBinding",
  "setViewportControlHighlight",
  "seedViewportControlPose",
  "requestViewportControls",
  "viewportKeyboardInput",
  "viewportPointerLook",
  "vehicleControlValue",
  "viewportInputValue",
  "registerPlayerPhysicsBody",
  "stepCompiledWorldPhysics",
  "requestViewportJump",
  "playerColliderVerticalOverlap",
  "resolvePlayerHorizontalMotion",
  "resolvePlayerVerticalSupport",
  "updateViewportControls"
]);

// Exact incumbent state initializer, with a separate instance for each owner.
export function createLivingMapControlState(model) {
const viewportControls = {
  policy: model.viewer.control_policy,
  highlighted: false,
  pointerLocked: false,
  keys: new Set(),
  observedKeys: new Set(),
  pointerButtons: 0,
  lastPointerMotion: -Infinity,
  position: null,
  yaw: -Math.PI / 2,
  pitch: -0.18,
  gamepadIdentity: null,
  gamepadPrimaryDown: false,
  gamepadSecondaryDown: false,
  jumpDown: false,
  respawnDown: false,
  frontDifferentialBrakeDown:false,
  rearDifferentialBrakeDown:false,
  headlightsDown:false,
  hornDown:false,
  ignitionDown:false,starterDown:false,
  horizontalVelocity: [0, 0],
  colliderSides: new Map(),
  representationTransition: null
};
  return viewportControls;
}

export function createLivingMapWalkController(context) {
  if (!context || typeof context !== "object") throw new TypeError("Living Map walk context is required");
  for (const name of LIVING_MAP_WALK_REQUIRED_STATE) {
    if (context[name] === null || typeof context[name] !== "object") {
      throw new TypeError(`Living Map walk requires state: ${name}`);
    }
  }
  for (const name of LIVING_MAP_WALK_REQUIRED_CALLBACKS) {
    if (typeof context[name] !== "function") {
      throw new TypeError(`Living Map walk requires existing callback: ${name}`);
    }
  }
  const {model, viewportControls, shaderViewer, entityState, physicsRuntime,
    portalRuntime, mobileControlState, vehicleRuntime, controlFocus, stateLoopRuntime,
    latestControlInput, primaryActionState, secondaryActionState, navigator, document, performance,
    setShaderOnlyMode, closeToolDialogue, selectHotbarSlot, cycleActiveToolMode,
    clearActiveVehicle, setActiveVehicle, closeSceneContextMenu,
    beginViewportPrimary, endViewportPrimary, beginViewportSecondary, endViewportSecondary,
    respawnViewportActor, controlVehicleTransmission, controlVehicleAuxiliary,
    controlVehicleFuelIgnition, cancelEntityNavigation, updateActiveVehicle,
    requestRepresentationTransition, sampleContactSurface, reservePhysicsSnapshotSlot,
    runCompiledPhysicsState} = context;
  if (!model.viewer?.camera || !model.document_geometry || !viewportControls.policy ||
      !Array.isArray(shaderViewer.colliders) || !Array.isArray(shaderViewer.geometry) ||
      !(entityState instanceof Map) || !(portalRuntime.transits instanceof Map) ||
      !(stateLoopRuntime.bodies instanceof Map) || !(physicsRuntime.parameters instanceof Map)) {
    throw new TypeError("Living Map walk requires the authored camera/policy/world, collider arrays and physics state maps");
  }

function normalized3(vector) {
  const length = Math.hypot(vector[0], vector[1], vector[2]) || 1;
  return vector.map(component => component / length);
}

function mobileClamp(value, minimum = -1, maximum = 1) {
  return Math.max(minimum, Math.min(maximum, Number(value) || 0));
}

function viewportBinding(action, input) {
  return Boolean(viewportControls.policy?.bindings.some(
    binding => binding.action === action && binding.inputs.includes(input)
  ));
}

function setViewportControlHighlight(highlighted) {
  viewportControls.highlighted = Boolean(highlighted);
  if (viewportControls.highlighted) {
    seedViewportControlPose();
    shaderViewer.active = true;
  } else {
    const bounds = shaderViewer.mapElement?.getBoundingClientRect();
    const position = latestControlInput?.position;
    shaderViewer.active = Boolean(bounds && position && position[0] >= bounds.left &&
      position[0] <= bounds.right && position[1] >= bounds.top && position[1] <= bounds.bottom);
  }
  shaderViewer.element?.classList.toggle("controls-captured", viewportControls.highlighted);
  if (!viewportControls.highlighted) {
    viewportControls.keys.clear();
    viewportControls.gamepadIdentity = null;
    if (document.pointerLockElement === shaderViewer.canvas) document.exitPointerLock?.();
  }
}

function seedViewportControlPose() {
  if (viewportControls.position) return;
  const eyeHeight = model.viewer.camera.eye_height;
  const courtyard = shaderViewer.geometry.find(box => box.kind === "courtyard");
  const firstRoom = shaderViewer.geometry.find(box => box.kind === "room");
  const fallbackPosition = [
    courtyard?.center[0] ?? firstRoom?.center[0] ?? 0,
    eyeHeight,
    courtyard ? courtyard.center[1] + courtyard.half_extent[1] * 0.72
      : (firstRoom?.center[1] ?? 0) + 1.2
  ];
  const fallbackTarget = [firstRoom?.center[0] ?? fallbackPosition[0],
    eyeHeight, firstRoom?.center[1] ?? fallbackPosition[2] - 1];
  const actor = entityState.get(viewportControls.policy?.actor);
  viewportControls.position = actor?.worldPosition ? [...actor.worldPosition] :
    shaderViewer.inhabitedCameraPosition ? [...shaderViewer.inhabitedCameraPosition] : fallbackPosition;
  if (actor) actor.worldPosition = [...viewportControls.position];
  const facing = shaderViewer.inhabitedCameraFacing || normalized3([
    fallbackTarget[0] - viewportControls.position[0],
    fallbackTarget[1] - viewportControls.position[1],
    fallbackTarget[2] - viewportControls.position[2]
  ]);
  viewportControls.yaw = Math.atan2(facing[2], facing[0]);
  viewportControls.pitch = Math.asin(Math.max(-1, Math.min(1, facing[1])));
}

function requestViewportControls() {
  setViewportControlHighlight(true);
  if (!controlFocus.dialogue) controlFocus.mode = "game";
  shaderViewer.element?.focus();
  if (viewportControls.policy?.captures.includes("pointer") &&
      viewportControls.policy.pointer_mode === "relative-when-available") {
    const request = shaderViewer.canvas?.requestPointerLock?.();
    if (request?.catch) request.catch(() => {});
  }
}

function viewportKeyboardInput(event) {
  if (event.type === "keydown" && event.code === "Escape" && shaderViewer.shaderOnly) {
    event.preventDefault();
    setShaderOnlyMode(false);
    return;
  }
  if (!viewportControls.policy?.captures.includes("keyboard")) return;
  if (controlFocus.dialogue && event.type === "keydown" && event.code === "Escape") {
    event.preventDefault(); closeToolDialogue(); return;
  }
  const hotbarSlot = /^Digit([0-9])$/.exec(event.code);
  if (hotbarSlot && viewportControls.highlighted && controlFocus.mode !== "dialogue") {
    event.preventDefault();
    if (event.type === "keydown" && !event.repeat) {
      selectHotbarSlot(hotbarSlot[1] === "0" ? 10 : Number(hotbarSlot[1]));
    }
    return;
  }
  if(event.code==="KeyM"&&viewportControls.highlighted&&controlFocus.mode!=="dialogue"){
    event.preventDefault();if(event.type==="keydown"&&!event.repeat)cycleActiveToolMode();return;
  }
  if(event.code==="KeyV"&&viewportControls.highlighted&&controlFocus.mode!=="dialogue"){
    event.preventDefault();if(event.type==="keydown"&&!event.repeat){
      if(vehicleRuntime.active)clearActiveVehicle();
      else if(vehicleRuntime.parkedState)setActiveVehicle(model.vehicle_slot.vehicles[0].identity,{placeAtActor:false});
    }return;
  }
  const mapped = viewportControls.policy.bindings.some(binding =>
    binding.inputs.includes(`keyboard:${event.code}`));
  if (!mapped && event.code !== "Escape") return;
  if (mapped) {
    if (event.type === "keydown") viewportControls.observedKeys.add(event.code);
    else viewportControls.observedKeys.delete(event.code);
  }
  if (!viewportControls.highlighted) return;
  event.preventDefault();
  if (event.code === "Escape") {
    closeSceneContextMenu();
    setViewportControlHighlight(false);
    shaderViewer.element?.blur();
    return;
  }
  if (event.type === "keydown") viewportControls.keys.add(event.code);
  else viewportControls.keys.delete(event.code);
}

function viewportPointerLook(event) {
  if (!viewportControls.highlighted || !viewportBinding("look", "pointer:relative-motion")) return;
  if (viewportControls.policy.pointer_mode === "relative-when-available" &&
      document.pointerLockElement !== shaderViewer.canvas) return;
  viewportControls.yaw += event.movementX * viewportControls.policy.look_sensitivity;
  viewportControls.pitch = Math.max(-1.35, Math.min(1.35,
    viewportControls.pitch - event.movementY * viewportControls.policy.look_sensitivity));
}

function vehicleControlValue(control, gamepad) {
  const tokens = String(control).trim().split(/\s+/);
  const terms = tokens.length === 1 && !/^[+-]/.test(tokens[0])
    ? [[1, tokens[0]]] : tokens.map(token => [token[0] === "-" ? -1 : 1, token.slice(1)]);
  const mobile = {"move-forward": Math.max(0, -mobileControlState.move[1]),
    "move-backward": Math.max(0, mobileControlState.move[1]),
    "strafe-right": Math.max(0, mobileControlState.move[0]),
    "strafe-left": Math.max(0, -mobileControlState.move[0])};
  return terms.reduce((sum, [sign, action]) =>
    sum + sign * Math.max(viewportInputValue(action, gamepad), mobile[action] || 0), 0);
}

function viewportInputValue(action, gamepad) {
  let value = 0;
  viewportControls.policy.bindings
    .filter(binding => binding.action === action)
    .flatMap(binding => binding.inputs)
    .forEach(input => {
      if (input.startsWith("keyboard:") && viewportControls.keys.has(input.slice(9))) value = 1;
      if (!gamepad) return;
      const leftX = Number(gamepad.axes[0] || 0), leftY = Number(gamepad.axes[1] || 0);
      if (input === "gamepad:left-y-negative") value = Math.max(value, Math.max(0, -leftY));
      if (input === "gamepad:left-y-positive") value = Math.max(value, Math.max(0, leftY));
      if (input === "gamepad:left-x-negative") value = Math.max(value, Math.max(0, -leftX));
      if (input === "gamepad:left-x-positive") value = Math.max(value, Math.max(0, leftX));
      const button=/^gamepad:button-(\d+)$/.exec(input);
      if(button)value=Math.max(value,Number(gamepad.buttons[Number(button[1])]?.value||0));
    });
  return value < 0.12 ? 0 : value;
}

function registerPlayerPhysicsBody() {
  if (!stateLoopRuntime.ready || stateLoopRuntime.actorRegistered || !viewportControls.position) return;
  const identity = viewportControls.policy?.actor;
  if (!identity) return;
  const snapshot = reservePhysicsSnapshotSlot(identity);
  snapshot.position[0] = viewportControls.position[0];
  snapshot.position[1] = viewportControls.position[1];
  snapshot.position[2] = viewportControls.position[2];
  snapshot.lastSubmittedPosition = [viewportControls.position[0], viewportControls.position[2]];
  stateLoopRuntime.worker.postMessage({type: "upsert", body: {identity,
    slot: snapshot.slot, generation: snapshot.generation,
    position: [...viewportControls.position], velocity: [0, physicsRuntime.verticalVelocity, 0],
    radius: physicsRuntime.parameters.get("radius") || .001, overrides: {}}});
  stateLoopRuntime.actorRegistered = true;
}

function stepCompiledWorldPhysics(dt) {
  if(vehicleRuntime.active)return;
  if (portalRuntime.transits.has(viewportControls.policy?.actor)) return;
  if (stateLoopRuntime.worker) {
    registerPlayerPhysicsBody();
    const identity = viewportControls.policy?.actor;
    if (stateLoopRuntime.ready && identity && viewportControls.position) {
      const body = stateLoopRuntime.bodies.get(identity);
      if (body) {
        const submitted = body.lastSubmittedPosition;
        const proposed = [viewportControls.position[0], viewportControls.position[2]];
        if (!submitted || Math.hypot(proposed[0]-submitted[0], proposed[1]-submitted[1]) > 1e-8) {
          body.controlGeneration += 1;
          body.lastSubmittedPosition = proposed;
          stateLoopRuntime.worker.postMessage({type: "control", identity,
            position: proposed, generation: body.controlGeneration});
        }
        const previousY=viewportControls.position[1];
        const support=resolvePlayerVerticalSupport(previousY,body.position[1],body.velocity[1]);
        viewportControls.position = [proposed[0],support.y,proposed[1]];
        physicsRuntime.verticalVelocity = support.velocity;
        physicsRuntime.grounded=Boolean(support.identity);
        if(support.identity&&Math.abs(body.position[1]-support.y)>1e-6){
          body.position[1]=support.y;body.velocity[1]=0;
          stateLoopRuntime.worker.postMessage({type:"support",identity,y:support.y});
        }
        physicsRuntime.last = {position_x_next: proposed[0], position_y_next: support.y,
          position_z_next: proposed[1], velocity_y_next: support.velocity,
          contactIdentity: support.identity||body.contactIdentity};
        const actor = entityState.get(identity);
        if (actor) { actor.worldPosition = [...viewportControls.position]; actor.velocity[1] = support.velocity; }
      }
    }
    return;
  }
  const instance = physicsRuntime.instance;
  const plugin = physicsRuntime.plugin;
  if (!instance || !plugin?.abi || !viewportControls.position || dt <= 0) return;
  const actor = entityState.get(viewportControls.policy?.actor);
  const radius = physicsRuntime.parameters.get("radius") || 0.001;
  const result = runCompiledPhysicsState(viewportControls.position,
    [0, physicsRuntime.verticalVelocity, 0], radius, dt);
  if (!result) {
    physicsRuntime.error = "compiled physics produced a non-finite state";
    return;
  }
  const previousY=viewportControls.position[1];
  const support=resolvePlayerVerticalSupport(previousY,result.position_y_next,result.velocity_y_next);
  viewportControls.position = [viewportControls.position[0],support.y,viewportControls.position[2]];
  physicsRuntime.verticalVelocity = support.velocity;
  physicsRuntime.grounded=Boolean(support.identity);
  physicsRuntime.last = {...result,position_y_next:support.y,velocity_y_next:support.velocity,
    contactIdentity:support.identity||result.contactIdentity};
  if (actor) {
    actor.worldPosition = [...viewportControls.position];
    actor.velocity[1] = result.velocity_y_next;
  }
}

function requestViewportJump() {
  seedViewportControlPose();
  const identity = viewportControls.policy?.actor;
  const eyeHeight = Number(model.viewer.camera.eye_height);
  const jumpSpeed = Number(viewportControls.policy?.jump_speed || 3.6);
  const body = identity ? stateLoopRuntime.bodies.get(identity) : null;
  const height = body?.position[1] ?? viewportControls.position?.[1] ?? eyeHeight;
  const verticalVelocity = body?.velocity[1] ?? physicsRuntime.verticalVelocity;
  const grounded = (physicsRuntime.grounded || height <= eyeHeight + 0.035) && verticalVelocity <= 0.15;
  if (!identity || !grounded) return false;
  cancelEntityNavigation(identity);
  physicsRuntime.grounded=false;
  physicsRuntime.supportSuppressedUntil=performance.now()+180;
  if (stateLoopRuntime.ready && body) {
    body.velocity[1] += jumpSpeed;
    stateLoopRuntime.worker.postMessage({type: "player-jump", identity,
      deltaVelocity: jumpSpeed});
  } else {
    physicsRuntime.verticalVelocity += jumpSpeed;
  }
  const actor = entityState.get(identity);
  if (actor) actor.velocity[1] = jumpSpeed;
  return true;
}

function playerColliderVerticalOverlap(position, collider) {
  const eyeHeight=Number(model.viewer.camera.eye_height),radius=Number(
    model.viewer.camera.collision_radius||physicsRuntime.parameters.get("radius")||.001);
  const foot=position[1]-eyeHeight,head=position[1]+radius;
  return head>collider.minimum[1]+1e-5&&foot<collider.maximum[1]-1e-5;
}

function resolvePlayerHorizontalMotion(previous, target) {
  const radius=Number(model.viewer.camera.collision_radius||.001),sides=viewportControls.colliderSides;
  let x=target[0],z=target[2],blockedX=false,blockedZ=false;
  const colliders=shaderViewer.colliders.filter(collider=>collider.role!=="projectile-body"&&
    !collider.surface&&
    playerColliderVerticalOverlap(previous,collider));
  colliders.forEach(collider=>{
    const mnX=collider.minimum[0]-radius,mxX=collider.maximum[0]+radius;
    const mnZ=collider.minimum[2]-radius,mxZ=collider.maximum[2]+radius;
    let remembered=sides.get(collider.identity);
    if(previous[0]<=mnX)remembered="west";else if(previous[0]>=mxX)remembered="east";
    else if(previous[2]<=mnZ)remembered="south";else if(previous[2]>=mxZ)remembered="north";
    if(remembered)sides.set(collider.identity,remembered);
    if(z>mnZ&&z<mxZ){
      if(previous[0]<=mnX&&x>mnX){x=mnX;blockedX=true;sides.set(collider.identity,"west");}
      else if(previous[0]>=mxX&&x<mxX){x=mxX;blockedX=true;sides.set(collider.identity,"east");}
      else if(previous[0]>mnX&&previous[0]<mxX&&remembered==="west"&&x>previous[0]){x=mnX;blockedX=true;}
      else if(previous[0]>mnX&&previous[0]<mxX&&remembered==="east"&&x<previous[0]){x=mxX;blockedX=true;}
    }
    if(x>mnX&&x<mxX){
      if(previous[2]<=mnZ&&z>mnZ){z=mnZ;blockedZ=true;sides.set(collider.identity,"south");}
      else if(previous[2]>=mxZ&&z<mxZ){z=mxZ;blockedZ=true;sides.set(collider.identity,"north");}
      else if(previous[2]>mnZ&&previous[2]<mxZ&&remembered==="south"&&z>previous[2]){z=mnZ;blockedZ=true;}
      else if(previous[2]>mnZ&&previous[2]<mxZ&&remembered==="north"&&z<previous[2]){z=mxZ;blockedZ=true;}
    }
  });
  return {position:[x,target[1],z],blockedX,blockedZ};
}

function resolvePlayerVerticalSupport(previousY,nextY,verticalVelocity) {
  const eyeHeight=Number(model.viewer.camera.eye_height),radius=Number(
    model.viewer.camera.collision_radius||.001),x=viewportControls.position[0],z=viewportControls.position[2];
  const previousFoot=previousY-eyeHeight,nextFoot=nextY-eyeHeight;
  if(performance.now()<physicsRuntime.supportSuppressedUntil&&verticalVelocity>0)
    return {y:nextY,velocity:verticalVelocity,identity:null};
  const surfaceContact=sampleContactSurface(x,z,previousFoot,nextFoot,verticalVelocity,.055);
  if(surfaceContact){
    const tangentConstraintVelocity=viewportControls.horizontalVelocity[0]*surfaceContact.gradient[0]+
      viewportControls.horizontalVelocity[1]*surfaceContact.gradient[1];
    return {y:surfaceContact.height+eyeHeight,
      velocity:Math.max(verticalVelocity,tangentConstraintVelocity),
      identity:surfaceContact.identity,gradient:surfaceContact.gradient,normal:surfaceContact.normal};
  }
  let support=null;
  if(verticalVelocity<=.15)shaderViewer.colliders.forEach(collider=>{
    if(collider.role==="projectile-body")return;
    const top=collider.maximum[1];
    if(x<collider.minimum[0]-radius||x>collider.maximum[0]+radius||
       z<collider.minimum[2]-radius||z>collider.maximum[2]+radius)return;
    const crossed=previousFoot>=top-.035&&nextFoot<=top+.012;
    if(crossed&&(!support||top>support.top))support={top,identity:collider.identity};
  });
  if(support)return {y:support.top+eyeHeight,velocity:0,identity:support.identity};
  const onWorld=nextY<=eyeHeight+.035;
  return {y:nextY,velocity:verticalVelocity,identity:onWorld?"world-floor":null};
}

function updateViewportControls(dt) {
  if (!viewportControls.highlighted || !viewportControls.policy) return;
  seedViewportControlPose();
  if (portalRuntime.transits.has(viewportControls.policy.actor)) {
    viewportControls.horizontalVelocity[0] = 0;
    viewportControls.horizontalVelocity[1] = 0;
    return;
  }
  const pads = viewportControls.policy.captures.includes("gamepad") && navigator.getGamepads
    ? [...navigator.getGamepads()].filter(Boolean) : [];
  const gamepad = viewportControls.policy.gamepad_selection === "first-connected" ? pads[0] : null;
  viewportControls.gamepadIdentity = gamepad?.id || null;
  if (Math.abs(mobileControlState.look[0]) + Math.abs(mobileControlState.look[1]) > 0.02) {
    viewportControls.yaw += mobileControlState.look[0] * dt * mobileControlState.touchLookSpeed;
    viewportControls.pitch = Math.max(-1.35, Math.min(1.35,
      viewportControls.pitch - mobileControlState.look[1] * dt * mobileControlState.touchLookSpeed));
  }
  if (gamepad && viewportBinding("look", "gamepad:right-stick")) {
    const lookX = Math.abs(gamepad.axes[2] || 0) < 0.12 ? 0 : gamepad.axes[2];
    const lookY = Math.abs(gamepad.axes[3] || 0) < 0.12 ? 0 : gamepad.axes[3];
    viewportControls.yaw += lookX * dt * 2.4;
    viewportControls.pitch = Math.max(-1.35, Math.min(1.35,
      viewportControls.pitch - lookY * dt * 2.4));
  }
  const gamepadPrimary = Boolean(gamepad?.buttons[0]?.pressed &&
    viewportBinding("primary-action", "gamepad:button-0"));
  if (gamepadPrimary && !viewportControls.gamepadPrimaryDown) {
    beginViewportPrimary(null,"gamepad");
  }
  if (!gamepadPrimary && viewportControls.gamepadPrimaryDown) endViewportPrimary("gamepad");
  viewportControls.gamepadPrimaryDown = gamepadPrimary;
  const gamepadSecondary = Boolean(gamepad?.buttons[1]?.pressed &&
    viewportBinding("secondary-action", "gamepad:button-1"));
  if (gamepadSecondary && !viewportControls.gamepadSecondaryDown) {
    beginViewportSecondary(null,"gamepad");
  }
  if (!gamepadSecondary && viewportControls.gamepadSecondaryDown) endViewportSecondary("gamepad");
  viewportControls.gamepadSecondaryDown = gamepadSecondary;
  const jumping = viewportInputValue("jump", gamepad) > 0;
  if (jumping && !viewportControls.jumpDown && !vehicleRuntime.active) requestViewportJump();
  viewportControls.jumpDown = jumping;
  const respawning=viewportInputValue("respawn",gamepad)>0;
  if(respawning&&!viewportControls.respawnDown)respawnViewportActor();
  viewportControls.respawnDown=respawning;
  const frontDifferentialBrake=viewportInputValue("front-differential-brake-toggle",gamepad)>0,
    rearDifferentialBrake=viewportInputValue("rear-differential-brake-toggle",gamepad)>0;
  if(vehicleRuntime.active&&frontDifferentialBrake&&!viewportControls.frontDifferentialBrakeDown)
    controlVehicleTransmission({frontDifferentialBrake:!vehicleRuntime.transmission.frontDifferentialBrake});
  if(vehicleRuntime.active&&rearDifferentialBrake&&!viewportControls.rearDifferentialBrakeDown)
    controlVehicleTransmission({rearDifferentialBrake:!vehicleRuntime.transmission.rearDifferentialBrake});
  viewportControls.frontDifferentialBrakeDown=frontDifferentialBrake;
  viewportControls.rearDifferentialBrakeDown=rearDifferentialBrake;
  const headlights=viewportInputValue("vehicle-headlights-toggle",gamepad)>0,
    horn=viewportInputValue("vehicle-horn",gamepad)>0,
    ignition=viewportInputValue("vehicle-ignition-toggle",gamepad)>0,
    starter=viewportInputValue("vehicle-starter",gamepad)>0;
  if(vehicleRuntime.active&&headlights&&!viewportControls.headlightsDown)
    controlVehicleAuxiliary({headlightsOn:!vehicleRuntime.electrical.headlightsOn});
  if(vehicleRuntime.active&&horn!==viewportControls.hornDown)controlVehicleAuxiliary({hornOn:horn});
  if(vehicleRuntime.active&&ignition&&!viewportControls.ignitionDown)
    controlVehicleAuxiliary({ignitionOn:!vehicleRuntime.electrical.ignitionOn});
  if(vehicleRuntime.active&&starter!==viewportControls.starterDown)controlVehicleAuxiliary({starterEngaged:starter});
  viewportControls.headlightsDown=headlights;viewportControls.hornDown=horn;
  viewportControls.ignitionDown=ignition;viewportControls.starterDown=starter;
  const forward = mobileClamp(viewportInputValue("move-forward", gamepad) -
    viewportInputValue("move-backward", gamepad) - mobileControlState.move[1]);
  const strafe = mobileClamp(viewportInputValue("strafe-right", gamepad) -
    viewportInputValue("strafe-left", gamepad) + mobileControlState.move[0]);
  if (Math.abs(forward) + Math.abs(strafe) > 0.05) {
    cancelEntityNavigation(viewportControls.policy.actor);
  }
  const facingX = Math.cos(viewportControls.yaw), facingZ = Math.sin(viewportControls.yaw);
  const running = viewportInputValue("run", gamepad) > 0;
  if(vehicleRuntime.active){
    // The vehicle's shell controls come from its declared control record:
    // each slot that names a hookup is sampled by that hookup, nothing is
    // composed by name here.  ``run`` still scales throttle on the page
    // (the record's ``control_mapping`` declares ``run: boost``) until that
    // composition moves into the law.
    const controls={};
    for(const row of vehicleRuntime.active.control_record||[]){
      if(row.control)controls[row.name]=mobileClamp(vehicleControlValue(row.control,gamepad));
    }
    const throttle=mobileClamp(Number(controls.throttle||0)*(running?1.18:1));
    updateActiveVehicle(dt,throttle,Number(controls.steering||0),Number(controls.brake||0));return;
  }
  const desiredSpeed = viewportControls.policy.move_speed *
    (running ? viewportControls.policy.run_multiplier : 1);
  const previousPosition = [...viewportControls.position];
  const desired=[(facingX*forward-facingZ*strafe)*desiredSpeed,
    (facingZ*forward+facingX*strafe)*desiredSpeed];
  const velocity=viewportControls.horizontalVelocity,inputMagnitude=Math.hypot(forward,strafe);
  if(inputMagnitude>.05){
    const acceleration=18,delta=[desired[0]-velocity[0],desired[1]-velocity[1]];
    const deltaLength=Math.hypot(...delta),scale=deltaLength>acceleration*dt?
      acceleration*dt/deltaLength:1;
    velocity[0]+=delta[0]*scale;velocity[1]+=delta[1]*scale;
  }else{
    const friction=Math.exp(-18*dt);velocity[0]*=friction;velocity[1]*=friction;
    if(Math.hypot(...velocity)<.008){velocity[0]=0;velocity[1]=0;}
  }
  const target=[previousPosition[0]+velocity[0]*dt,previousPosition[1],
    previousPosition[2]+velocity[1]*dt];
  const resolved=resolvePlayerHorizontalMotion(previousPosition,target);
  viewportControls.position=resolved.position;
  if(resolved.blockedX)velocity[0]=0;if(resolved.blockedZ)velocity[1]=0;
  const envelope = shaderViewer.geometry.find(box => box.identity ===
    model.document_geometry.representation_boundary?.inside);
  if (envelope && (
      Math.abs(viewportControls.position[0] - envelope.center[0]) > envelope.half_extent[0] ||
      Math.abs(viewportControls.position[2] - envelope.center[1]) > envelope.half_extent[1])) {
    requestRepresentationTransition(viewportControls.position);
    viewportControls.position = previousPosition;
  }
  const actor = entityState.get(viewportControls.policy.actor);
  if (actor) {
    actor.worldPosition = [...viewportControls.position];
    actor.velocity = dt > 0 ? [
      (viewportControls.position[0] - previousPosition[0]) / dt,
      physicsRuntime.verticalVelocity,
      (viewportControls.position[2] - previousPosition[2]) / dt,
    ] : [0, 0, 0];
    actor.facing = [facingX, facingZ, 0];
  }
}

  // Event ownership only. Input handlers above are the unchanged original
  // implementation. Capturing controls remains an explicit caller/user action.
  // Releasing a view never writes body position, velocity, force or gravity.
  let attached = null;
  function attach({keyboard = document, pointer = document,
    element = shaderViewer.element, host = globalThis.window} = {}) {
    if (attached) return attached;
    for (const [name, target] of Object.entries({keyboard, pointer, element, host, document})) {
      if (typeof target?.addEventListener !== "function" || typeof target?.removeEventListener !== "function") {
        throw new TypeError(`Living Map walk listener target is required: ${name}`);
      }
    }
    const remove = [];
    const listen = (target, type, callback) => {
      target.addEventListener(type, callback);
      remove.push(() => target.removeEventListener(type, callback));
    };
    const release = (reason = "focus-loss") => {
      // The owning page already releases held tool actions on window blur.
      // Keep that ownership here as well when listeners are hidden or removed.
      if (primaryActionState.down) endViewportPrimary(reason);
      if (secondaryActionState.down) endViewportSecondary(reason);
      setViewportControlHighlight(false);
      viewportControls.observedKeys.clear();
      viewportControls.pointerButtons = 0;
    };
    listen(keyboard, "keydown", viewportKeyboardInput);
    listen(keyboard, "keyup", viewportKeyboardInput);
    listen(pointer, "mousemove", viewportPointerLook);
    listen(document, "pointerlockchange", () => {
      viewportControls.pointerLocked = document.pointerLockElement === shaderViewer.canvas;
    });
    listen(element, "blur", () => {
      if (document.pointerLockElement !== shaderViewer.canvas && controlFocus.mode === "game" &&
          !controlFocus.dialogue) release();
    });
    listen(host, "blur", () => release("window-blur"));
    listen(document, "visibilitychange", () => { if (document.hidden) release("visibility-loss"); });
    let active = true;
    attached = () => {
      if (!active) return;
      active = false;
      remove.splice(0).forEach(callback => callback());
      release("detach");
      attached = null;
    };
    return attached;
  }

  return Object.freeze({attach, normalized3, mobileClamp, viewportBinding, setViewportControlHighlight, seedViewportControlPose, requestViewportControls, viewportKeyboardInput, viewportPointerLook, vehicleControlValue, viewportInputValue, registerPlayerPhysicsBody, stepCompiledWorldPhysics, requestViewportJump, playerColliderVerticalOverlap, resolvePlayerHorizontalMotion, resolvePlayerVerticalSupport, updateViewportControls});
}

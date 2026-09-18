// abstract-ui:system-root
const abstractUISystemTimer = (() => {
  let actionEdges = null;
  let pendingActions = [];
  let sequence = 0;
  const schedule = globalThis.requestAnimationFrame
    ? callback => globalThis.requestAnimationFrame(callback)
    : callback => globalThis.setTimeout(() => callback(Date.now()), 16);
  function connect(destination) { actionEdges = destination; }
  function issue(action) { pendingActions.push(action); }
  function frame(now) {
    sequence += 1;
    if (actionEdges) {
      actionEdges.time = now;
      actionEdges.update(pendingActions.splice(0));
    }
    schedule(frame);
  }
  schedule(frame);
  return {identity: "introspection-world:python:src.compiler.mechanical_creature.MechanicalCreature/timer", connect, issue, get sequence() { return sequence; }};
})();

// turing-runtime-utility: turing.wasm.registry [javascript-utility:sha256:e9935fd68b60237134a24dcda72022e009cf01d02ab9694845a5fa53921b3b5b]
class TuringWasmModuleRegistry {
  constructor(modules = []) {
    this.descriptors = new Map();
    this.instances = new Map();
    modules.forEach(descriptor => this.register(descriptor));
  }
  register(descriptor) {
    if (!descriptor?.content_key || !descriptor?.binary_base64) {
      throw new TypeError("WASM modules require content_key and binary_base64");
    }
    const previous = this.descriptors.get(descriptor.content_key);
    if (previous && previous.binary_base64 !== descriptor.binary_base64) {
      throw new Error(`WASM content identity collision: ${descriptor.content_key}`);
    }
    this.descriptors.set(descriptor.content_key, descriptor);
    return descriptor.content_key;
  }
  async instantiate(contentKey, imports = {}) {
    if (this.instances.has(contentKey)) return this.instances.get(contentKey);
    const descriptor = this.descriptors.get(contentKey);
    if (!descriptor) throw new Error(`unknown WASM module ${contentKey}`);
    const pending = (async () => {
      const bytes = Uint8Array.from(atob(descriptor.binary_base64), value => value.charCodeAt(0));
      const result = await WebAssembly.instantiate(bytes, imports);
      return result.instance;
    })();
    this.instances.set(contentKey, pending);
    try { return await pending; }
    catch (error) { this.instances.delete(contentKey); throw error; }
  }
}
function turingCreateWasmRegistry(modules = []) {
  return new TuringWasmModuleRegistry(modules);
}

// turing-runtime-utility: turing.world.registry [javascript-utility:sha256:42623b16214975c22e2b1edadbe15c4261d6d10374ce98c7351cd6069c5fbbdc]
class TuringWorldRegistry {
  constructor(world) {
    if (!world?.identity || !Array.isArray(world.objects)) {
      throw new TypeError("world registry requires identity and objects");
    }
    this.world = world;
    this.objects = new Map();
    this.children = new Map();
    this.semanticParts = new Map();
    this.objectRuntimeIds = new Map();
    this.partRuntimeIds = new Map();
    this.runtimeObjects = [null];
    this.runtimeParts = [null];
    world.objects.forEach(object => {
      if (!object.identity || this.objects.has(object.identity)) {
        throw new Error(`duplicate or empty world identity ${object.identity}`);
      }
      this.objects.set(object.identity, object);
      if (!this.children.has(object.parent)) this.children.set(object.parent, []);
      this.children.get(object.parent).push(object.identity);
      (object.semantic_parts || []).forEach(part => {
        if (this.semanticParts.has(part.identity)) {
          throw new Error(`duplicate semantic part identity ${part.identity}`);
        }
        this.semanticParts.set(part.identity, {object: object.identity, ...part});
      });
    });
    const specialization = world.identity_specialization || {
      authority: "authored-string-identity", missing_runtime_id: 0,
      objects: [...this.objects.keys()].map((identity, index) =>
        ({identity, runtime_id: index + 1})),
      semantic_parts: [...this.semanticParts.keys()].map((identity, index) =>
        ({identity, runtime_id: index + 1})),
    };
    if (specialization.authority !== "authored-string-identity" ||
        specialization.missing_runtime_id !== 0) {
      throw new Error("world identity specialization has an incompatible authority");
    }
    (specialization?.objects || []).forEach(entry => {
      const object = this.objects.get(entry.identity);
      if (!object || entry.runtime_id <= 0 || this.runtimeObjects[entry.runtime_id]) {
        throw new Error(`invalid runtime object identity ${entry.runtime_id}`);
      }
      this.objectRuntimeIds.set(entry.identity, entry.runtime_id);
      this.runtimeObjects[entry.runtime_id] = object;
    });
    (specialization?.semantic_parts || []).forEach(entry => {
      const part = this.semanticParts.get(entry.identity);
      if (!part || entry.runtime_id <= 0 || this.runtimeParts[entry.runtime_id]) {
        throw new Error(`invalid runtime semantic-part identity ${entry.runtime_id}`);
      }
      this.partRuntimeIds.set(entry.identity, entry.runtime_id);
      this.runtimeParts[entry.runtime_id] = part;
    });
  }
  object(identity) { return this.objects.get(identity) || null; }
  part(identity) { return this.semanticParts.get(identity) || null; }
  objectRuntimeId(identity) { return this.objectRuntimeIds.get(identity) || 0; }
  partRuntimeId(identity) { return this.partRuntimeIds.get(identity) || 0; }
  objectFromRuntimeId(runtimeId) { return this.runtimeObjects[runtimeId] || null; }
  partFromRuntimeId(runtimeId) { return this.runtimeParts[runtimeId] || null; }
  containedBy(identity) {
    return [...(this.children.get(identity) || [])].map(child => this.objects.get(child));
  }
  ancestry(identity) {
    const result = [];
    let object = this.object(identity);
    const visited = new Set();
    while (object && !visited.has(object.identity)) {
      visited.add(object.identity); result.push(object);
      object = this.object(object.parent);
    }
    return result;
  }
}
function turingCreateWorldRegistry(world) { return new TuringWorldRegistry(world); }

// turing-runtime-utility: turing.revision.channel [javascript-utility:sha256:1fcc8019c6fc1cb35c046f9a2606cfc8e85ef2b56775684ad9f9905bd0e2e8be]
class TuringRevisionChannel {
  constructor(identity, revision = 0) {
    this.identity = String(identity);
    this.revision = Number(revision);
    this.listeners = new Set();
  }
  subscribe(listener) {
    if (typeof listener !== "function") throw new TypeError("revision listener must be callable");
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
  publish(record) {
    const next = Number(record?.revision);
    if (!Number.isInteger(next) || next <= this.revision) {
      throw new RangeError(`revision must increase beyond ${this.revision}`);
    }
    this.revision = next;
    const event = Object.freeze({...record, channel: this.identity, revision: next});
    this.listeners.forEach(listener => listener(event));
    return event;
  }
}
function turingCreateRevisionChannel(identity, revision = 0) {
  return new TuringRevisionChannel(identity, revision);
}

const SELF_SCRIPT = document.currentScript;
const SELF_SOURCE = SELF_SCRIPT.textContent.replace(/^\n/, "").trimEnd();
const model = JSON.parse(document.getElementById("abstract-ui-model").textContent);
const turingWasmModules = turingCreateWasmRegistry(model.world.wasm_modules);
const turingWorld = turingCreateWorldRegistry(model.world);
const turingWorldRevisions = turingCreateRevisionChannel(
  `${model.world.identity}/revisions`, model.scene_mesh.revision || 0
);
const index = new Map();
const mapRoot = document.getElementById("map-root");
const inspector = document.getElementById("inspector-content");
const status = document.getElementById("map-status");

const VIEWPORT_VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aColor;
uniform vec2 uResolution;
uniform vec3 uCameraPosition;
uniform vec3 uCameraFacing;
out vec3 vNormal;
out vec3 vColor;
out vec3 vWorldPosition;
out float vDepth;
void main(){
  vec3 forward=normalize(uCameraFacing);
  vec3 right=normalize(cross(forward,vec3(0,1,0)));
  vec3 cameraUp=normalize(cross(right,forward));
  vec3 relative=aPosition-uCameraPosition;
  vec3 view=vec3(dot(relative,right),dot(relative,cameraUp),dot(relative,forward));
  float aspect=uResolution.x/max(1.0,uResolution.y);
  float tangentHalfFov=0.70;
  float nearPlane=0.04,farPlane=128.0;
  float clipZ=((farPlane+nearPlane)/(farPlane-nearPlane))*view.z-
    (2.0*farPlane*nearPlane)/(farPlane-nearPlane);
  gl_Position=vec4(view.x/(tangentHalfFov*aspect),view.y/tangentHalfFov,clipZ,view.z);
  vNormal=aNormal;vColor=aColor;vWorldPosition=aPosition;vDepth=view.z;
}`;

const VIEWPORT_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vColor;
in vec3 vWorldPosition;
in float vDepth;
out vec4 fragmentColor;
uniform vec3 uSkyColor;
uniform vec3 uLightColor;
uniform vec3 uKeyLightDirection;
uniform float uAmbientLight;
uniform float uWorldTileSize;
uniform float uWorldTileMajorEvery;
uniform float uWorldTileStrength;
uniform vec3 uHeadlightLeft;
uniform vec3 uHeadlightRight;
uniform vec3 uHeadlightForward;
uniform float uHeadlightActive;
uniform vec3 uTailLightLeft;
uniform vec3 uTailLightRight;
uniform vec3 uVehicleRearward;
uniform float uTailLightActive;
uniform float uBrakeLightActive;
void main(){
  if(vDepth<=0.04)discard;
  vec3 normal=normalize(gl_FrontFacing?vNormal:-vNormal);
  vec3 keyLight=normalize(uKeyLightDirection);
  vec3 fillLight=normalize(vec3(0.72,0.35,-0.28));
  float key=max(dot(normal,keyLight),0.0);
  vec3 leftRay=vWorldPosition-uHeadlightLeft,rightRay=vWorldPosition-uHeadlightRight;
  float leftDistance=max(.05,length(leftRay)),rightDistance=max(.05,length(rightRay));
  vec3 leftDirection=leftRay/leftDistance,rightDirection=rightRay/rightDistance;
  float leftCone=smoothstep(.90,.965,dot(leftDirection,normalize(uHeadlightForward)));
  float rightCone=smoothstep(.90,.965,dot(rightDirection,normalize(uHeadlightForward)));
  float headlight=uHeadlightActive*(
    leftCone*max(dot(normal,-leftDirection),0.0)/(1.0+.10*leftDistance*leftDistance)+
    rightCone*max(dot(normal,-rightDirection),0.0)/(1.0+.10*rightDistance*rightDistance));
  vec3 tailLeftRay=vWorldPosition-uTailLightLeft,tailRightRay=vWorldPosition-uTailLightRight;
  float tailLeftDistance=max(.05,length(tailLeftRay)),tailRightDistance=max(.05,length(tailRightRay));
  vec3 tailLeftDirection=tailLeftRay/tailLeftDistance,tailRightDirection=tailRightRay/tailRightDistance;
  vec3 rearward=normalize(uVehicleRearward);
  float tailLeftCone=smoothstep(.68,.92,dot(tailLeftDirection,rearward));
  float tailRightCone=smoothstep(.68,.92,dot(tailRightDirection,rearward));
  float tailSurface=(
    tailLeftCone*max(dot(normal,-tailLeftDirection),0.0)/(1.0+.55*tailLeftDistance*tailLeftDistance)+
    tailRightCone*max(dot(normal,-tailRightDirection),0.0)/(1.0+.55*tailRightDistance*tailRightDistance));
  float tailLight=(.42*uTailLightActive+1.35*uBrakeLightActive)*tailSurface;
  float illumination=uAmbientLight+0.72*key+
    0.28*max(dot(normal,fillLight),0.0);
  float blueCourse=smoothstep(.10,.34,vColor.b-max(vColor.r,vColor.g)*.72);
  float fog=max(exp(-vDepth*0.018),blueCourse*.48);
  float tileSize=max(0.05,uWorldTileSize);
  vec2 tileCoordinate=vWorldPosition.xz/tileSize;
  vec2 tileEdge=min(fract(tileCoordinate),1.0-fract(tileCoordinate));
  float grid=1.0-smoothstep(0.018,0.065,min(tileEdge.x,tileEdge.y));
  vec2 majorCoordinate=tileCoordinate/max(1.0,uWorldTileMajorEvery);
  vec2 majorEdge=min(fract(majorCoordinate),1.0-fract(majorCoordinate));
  float major=1.0-smoothstep(0.010,0.040,min(majorEdge.x,majorEdge.y));
  float upward=smoothstep(0.55,0.92,normal.y);
  float motionGrid=upward*max(grid*0.38,major)*uWorldTileStrength;
  vec3 tiledColor=mix(vColor,vColor+vec3(0.22,0.19,0.10),motionGrid);
  vec3 lit=tiledColor*(illumination+3.2*headlight+.55*tailLight)+uLightColor*key*0.10+
    vec3(1.0,.86,.62)*headlight*.36+vec3(1.0,.055,.018)*tailLight+vColor*blueCourse*.22;
  lit=clamp((lit*(2.51*lit+0.03))/(lit*(2.43*lit+0.59)+0.14),0.0,1.0);
  fragmentColor=vec4(mix(uSkyColor,lit,fog),1.0);
}`;

const VIEWPORT_SKY_VERTEX_SHADER = `#version 300 es
precision highp float;
void main(){
  vec2 position=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
  gl_Position=vec4(position*2.0-1.0,1.0,1.0);
}`;

const VIEWPORT_SKY_FRAGMENT_SHADER = `#version 300 es
precision highp float;
uniform vec2 uResolution;
uniform vec3 uCameraFacing;
uniform vec3 uSunDirection;
uniform vec3 uMoonDirection;
uniform vec3 uDayZenith;
uniform vec3 uNightZenith;
uniform vec3 uHorizonColor;
uniform vec3 uSunColor;
uniform vec3 uMoonColor;
out vec4 fragmentColor;
void main(){
  vec2 uv=(gl_FragCoord.xy/uResolution)*2.0-1.0;
  vec3 forward=normalize(uCameraFacing);
  vec3 right=normalize(cross(forward,vec3(0,1,0)));
  vec3 cameraUp=normalize(cross(right,forward));
  float aspect=uResolution.x/max(1.0,uResolution.y);
  vec3 ray=normalize(forward+right*uv.x*0.70*aspect+cameraUp*uv.y*0.70);
  float sunHeight=smoothstep(-0.18,0.22,uSunDirection.y);
  vec3 zenith=mix(uNightZenith,uDayZenith,sunHeight);
  float dome=pow(clamp(ray.y*0.5+0.5,0.0,1.0),0.62);
  vec3 color=mix(uHorizonColor,zenith,dome);
  float below=1.0-smoothstep(-0.42,-0.12,ray.y);
  color=mix(color,uNightZenith*0.32,below);
  float sunDisc=smoothstep(0.99972,0.99994,dot(ray,normalize(uSunDirection)));
  float sunGlow=pow(max(dot(ray,normalize(uSunDirection)),0.0),192.0);
  float moonDisc=smoothstep(0.99978,0.99995,dot(ray,normalize(uMoonDirection)));
  color+=uSunColor*(sunDisc*1.8+sunGlow*0.32)*sunHeight;
  color+=uMoonColor*(moonDisc*1.25)*smoothstep(-0.1,0.15,uMoonDirection.y);
  fragmentColor=vec4(color,1.0);
}`;

const VIEWPORT_SHADER_CHOICES = [{
  identity: "living-map-default",
  label: "Living map",
  vertex_source: VIEWPORT_VERTEX_SHADER,
  fragment_source: VIEWPORT_FRAGMENT_SHADER,
  adapter: "native-living-map",
  resource_bindings: []
}, ...(model.viewer.shader_choices || [])];
const VIEWPORT_DEFAULT_SHADER = model.viewer.default_shader || "living-map-default";

const VIEWPORT_CROSSHAIR_VERTEX_SHADER = `#version 300 es
precision highp float;
void main(){
  vec2 position=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
  gl_Position=vec4(position*2.0-1.0,0.0,1.0);
}`;

const VIEWPORT_CROSSHAIR_FRAGMENT_SHADER = `#version 300 es
precision highp float;
uniform vec2 uResolution;
uniform vec3 uIdleColor;
uniform vec3 uTargetColor;
uniform float uHasTarget;
out vec4 fragmentColor;
void main(){
  vec2 delta=abs(gl_FragCoord.xy-uResolution*0.5);
  float vertical=(1.0-step(0.8,delta.x))*(1.0-step(11.0,delta.y));
  float horizontal=(1.0-step(0.8,delta.y))*(1.0-step(11.0,delta.x));
  float centerGap=1.0-step(2.6,max(delta.x,delta.y));
  float alpha=max(vertical,horizontal)*(1.0-centerGap);
  if(alpha<0.01)discard;
  fragmentColor=vec4(mix(uIdleColor,uTargetColor,uHasTarget),0.92*alpha);
}`;

const VEHICLE_HUD_VERTEX_SHADER=`#version 300 es
precision highp float;
layout(location=0) in vec2 aUnit;
uniform vec2 uResolution;
uniform vec4 uRect;
uniform float uAngle;
out vec2 vUnit;
void main(){
  vec2 local=(aUnit-.5)*uRect.zw;
  float c=cos(uAngle),s=sin(uAngle);
  vec2 pixel=uRect.xy+uRect.zw*.5+mat2(c,s,-s,c)*local;
  vec2 clip=vec2(pixel.x/uResolution.x*2.0-1.0,1.0-pixel.y/uResolution.y*2.0);
  gl_Position=vec4(clip,0.0,1.0);vUnit=aUnit;
}`;

const VEHICLE_HUD_FRAGMENT_SHADER=`#version 300 es
precision highp float;
in vec2 vUnit;
uniform vec4 uColor;
uniform float uEllipse;
out vec4 fragmentColor;
void main(){
  if(uEllipse>.5&&dot((vUnit-.5)*2.0,(vUnit-.5)*2.0)>1.0)discard;
  fragmentColor=uColor;
}`;

const VEHICLE_WHEEL_VERTEX_SHADER=`#version 300 es
precision highp float;
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
uniform mat4 uWheelModel;
uniform vec2 uResolution;
uniform vec3 uCameraPosition;
uniform vec3 uCameraFacing;
out vec3 vLocalPosition;
out vec3 vNormal;
out float vDepth;
void main(){
  vec3 worldPosition=(uWheelModel*vec4(aPosition,1.0)).xyz;
  vec3 forward=normalize(uCameraFacing),right=normalize(cross(forward,vec3(0,1,0)));
  vec3 cameraUp=normalize(cross(right,forward)),relative=worldPosition-uCameraPosition;
  vec3 view=vec3(dot(relative,right),dot(relative,cameraUp),dot(relative,forward));
  float aspect=uResolution.x/max(1.0,uResolution.y),nearPlane=.04,farPlane=128.0;
  float clipZ=((farPlane+nearPlane)/(farPlane-nearPlane))*view.z-
    (2.0*farPlane*nearPlane)/(farPlane-nearPlane);
  gl_Position=vec4(view.x/(.70*aspect),view.y/.70,clipZ,view.z);
  vLocalPosition=aPosition;vNormal=normalize(mat3(uWheelModel)*aNormal);vDepth=view.z;
}`;
const VEHICLE_WHEEL_FRAGMENT_SHADER=`#version 300 es
precision highp float;
in vec3 vLocalPosition;
in vec3 vNormal;
in float vDepth;
uniform float uTreadPhase;
uniform vec3 uRubberColor;
uniform vec3 uTreadColor;
uniform vec3 uRimColor;
uniform vec3 uRotorColor;
uniform vec3 uLightDirection;
out vec4 fragmentColor;
void main(){
  if(vDepth<=.04)discard;
  float angle=atan(vLocalPosition.y,vLocalPosition.x)+uTreadPhase;
  float tread=.5+.5*cos(angle*16.0);
  float cap=smoothstep(.55,.9,abs(vNormal.z)),radial=length(vLocalPosition.xy);
  float treadBlock=smoothstep(.28,.78,tread)*(1.0-cap);
  vec3 textureColor=mix(uRubberColor,uTreadColor,treadBlock*.72);
  float tireSide=cap*smoothstep(.70,.82,radial);
  float rimDisc=cap*(1.0-smoothstep(.66,.74,radial));
  float spokeWave=.5+.5*cos(angle*6.0);
  float spokes=rimDisc*smoothstep(.68,.9,spokeWave)*smoothstep(.18,.28,radial);
  float hub=rimDisc*(1.0-smoothstep(.16,.25,radial));
  float rotor=rimDisc*smoothstep(.27,.34,radial)*(1.0-smoothstep(.48,.57,radial));
  textureColor=mix(textureColor,uRubberColor,tireSide);
  textureColor=mix(textureColor,uRotorColor,rotor*.82);
  textureColor=mix(textureColor,uRimColor,max(spokes,hub));
  float light=.3+.7*max(dot(normalize(vNormal),normalize(uLightDirection)),0.0);
  fragmentColor=vec4(textureColor*light,1.0);
}`;

const shaderViewer = {
  element: null, canvas: null, readout: null, gl: null, program: null,
  crosshairProgram: null, crosshairLocations: {}, skyProgram: null, skyLocations: {},
  vehicleHudProgram:null,vehicleHudLocations:{},vehicleHudVao:null,
  wheelProgram:null,wheelLocations:{},wheelVao:null,wheelVertexCount:0,
  vao:null,buffer:null,vertexCount:0,vehicleVao:null,vehicleBuffer:null,vehicleVertexCount:0,
  vehicleMesh:null,active:false,mapElement:null,
  context2d: null, geometry: [], mesh: null, baseMesh: null, sceneWasm: null, softwareWasm: null,
  softwareWasmPending: false,
  identitySpans: [], semanticPartSpans: [], colliders: [],
  formBaselines: new Map(), revision: 0,
  cameraPosition: null, cameraFacing: null, crosshairIdentity: null,
  focusTooltip: null, contextMenu: null, placementOverlay: null, celestialStatus: null,
  softwareTriangleCount: 0, softwareOnscreenCount: 0,
  inhabitedCameraPosition: null, inhabitedCameraFacing: null,
  backend: "pending", locations: {}, shaderChoice: VIEWPORT_DEFAULT_SHADER,
  shaderSelect: null, shaderResources: new Map(), shaderPrograms: new Map(), celestial: null,
  shaderOnly: false, shaderOnlyToggle: null, telemetry: null,
  shadow: null, shadowMatrix: null, cameraDepth: null
};

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
  starterDown:false,
  horizontalVelocity: [0, 0],
  colliderSides: new Map(),
  representationTransition: null
};

const vehicleRuntime = {
  instance:null,plugin:null,pending:false,error:null,active:null,state:null,box:null,cabinBox:null,frameBoxes:[],
  rollCageBoxes:[],mechanicalLinkBoxes:[],suspensionLinkBoxes:[],powertrainBoxes:[],bodyShellBoxes:[],mechanicalNodePositions:new Map(),
  accumulator:0,lastSpringForces:[0,0,0,0],contactAreas:[0,0,0,0],
  frictionUtilizations:[0,0,0,0],contactModes:[0,0,0,0],compressions:[0,0,0,0],
  tractionScales:[1,1,1,1],brakeScales:[1,1,1,1],damperScales:[1,1,1,1],
  radialProbePenetrations:Array.from({length:4},()=>Array(15).fill(0)),radialProbeActiveCounts:[0,0,0,0],
  powertrain:{engineTorque:0,clutchTorque:0,transmissionOutputTorque:0,drivelineTorque:0,
    frontDifferentialTorque:0,rearDifferentialTorque:0,engineAccelerationTorque:0,
    engineAngularAcceleration:0,reactionTorque:[0,0,0],mountTorque:[0,0,0]},
  transmission:{mode:"automatic",gear:2,displayGear:2,torqueReserve:0,reason:"initial-second",
    lowRange:false,transferRange:"high",frontDiffLock:false,rearDiffLock:false,centerDiffLock:false,frontDriveShare:.42,
    frontDiffMode:"open",rearDiffMode:"open",centerDiffMode:"open",
    frontDifferentialBrake:false,rearDifferentialBrake:false,
    smoothLaunch:false,tractionControlEnabled:true,absEnabled:true,tiltEnabled:true,tractionControlAuthority:1,absAuthority:1},
  dyno:null,dynoRequest:0,
  wheelBoxes:[],wheelAngles:[0,0,0,0],wheelPhaseTime:null,camera:null,
  parkedState:null,inventoryItem:null,inventorySlot:null,worldMarker:null,presentationAccumulator:0,
  cameraChassisYaw:null,
  computeMode:"resident-webgpu-pending",gpu:null,contactMonitor:null,transmissionControls:null,transferHud:null,
  brakeLocks:{front_left:false,front_right:false,rear_left:false,rear_right:false},
  powerUnitPreset:null,
  engineEquationMode:"linear-playable",
  transmissionPreset:null,
  chassisProfile:null,
  wheelPart:null,
  clutchPreset:null,
  wheelAlignment:null,
  bodyShell:null,
  turretSystem:{fireTakeover:true,ammoCount:60,target:null,targetDistance:0,interlockReason:"no-target",turrets:[],
    outriggers:{commanded:false,extension:0,anchors:{}}},
  chassisLeveling:null,
  steeringSystem:null,
  shockParameters:{},
  electrical:{headlightsOn:false,hornOn:false,ignitionOn:true,starterEngaged:false,stateOfCharge:1,
    fuelMassKg:0,totalMassKg:0,fuelIdentity:"pump-gasoline-93",ignitionProfileIdentity:"gasoline-distributor",
    requestedIgnitionProfileIdentity:"gasoline-distributor",ignitionTimingOffsetCycles:0,combustionSharpness:1,
    timingErrorDegrees:0,combustionStress:0,computerOnline:true,ecuOnline:true,lightingCircuitOnline:true,
    tailLightsOn:false,brakeLightsOn:false,tirePressurePa:155000,tirePressureTargetPa:155000,
    pneumaticCompressorOn:false,pneumaticCompressorPowerW:0,hydraulicPumpOn:false,hydraulicPumpPowerW:0},
  driverAssistance:{drivingMode:"road",governorRpm:6500,cruiseEnabled:false,
    cruiseTargetSpeedMps:0,cruiseIntegral:0,cruiseThrottle:0,cruiseBrake:0,tiltEnabled:true,
    tiltAuthority:0,tiltRisk:0,tiltGovernorRpm:6500,rearDifferentialBrakeCommand:0},
  damage:null,
  frameFault:null,disabledPresentationStages:new Set(),shaderHudHitRegions:[]
};

function reportRuntimeFault(stage,error){
  const message=String(error?.stack||error?.message||error),signature=`${stage}:${message}`;
  if(vehicleRuntime.frameFault?.signature!==signature){
    console.error(`AbstractUI runtime stage failed: ${stage}`,error);
    vehicleRuntime.frameFault={stage,message,signature,time:performance.now()};
  }
  if(shaderViewer.readout)shaderViewer.readout.textContent=`runtime fault · ${stage} · ${String(error?.message||error)}`;
}

const mobileControlState = {
  element: null, motionButton: null, statusElement: null,
  move: [0, 0], look: [0, 0], tilt: [0, 0],
  touchLookSpeed: 0.72,
  orientation: null, acceleration: null, baseline: null,
  listenersInstalled: false, motionEnabled: false
};

const musicRoomRuntime = {
  context: null, buffer: null, source: null, gain: null, instance: null,
  input: null, outputs: [], run: null, startedAt: 0, playing: false,
  bands: [0, 0, 0], level: 0, button: null, loadButton: null,
  trackLabel: "generated loop", impactAt: 0, sequence: 0
};

const engineSoundRuntime={context:null,node:null,gain:null,profile:null,pending:null,lastTelemetryAt:0,error:null};
const vehicleHornRuntime={oscillators:[],gain:null};

async function setVehicleHorn(active){
  if(!active){for(const oscillator of vehicleHornRuntime.oscillators)try{oscillator.stop();}catch(_error){}
    vehicleHornRuntime.oscillators=[];vehicleHornRuntime.gain=null;return;}
  if(vehicleHornRuntime.oscillators.length)return;const engine=await ensureEngineSound(),context=engine.context;
  if(!context)return;const now=context.currentTime,gain=context.createGain();gain.gain.setValueAtTime(.0001,now);
  gain.gain.exponentialRampToValueAtTime(.075,now+.025);gain.connect(context.destination);
  vehicleHornRuntime.gain=gain;vehicleHornRuntime.oscillators=[285,359].map((frequency,index)=>{
    const oscillator=context.createOscillator();oscillator.type=index?"square":"sawtooth";
    oscillator.frequency.value=frequency;oscillator.connect(gain);oscillator.start(now);return oscillator;});
}

function engineSoundWorkletSource(){return `
class TuringEnginePCMProcessor extends AudioWorkletProcessor{
  constructor(options){super();this.blockSize=options.processorOptions.blockSize;this.inputNames=options.processorOptions.inputs;
    this.bank=new Map();this.phase=0;this.profile=options.processorOptions.initialProfile;this.metrics={rpm:0,load:0,power:0,
      throttle:0,transient:0,damage:0,stall:1,ignition_timing_cycles:0,combustion_character:1};this.filterCoefficients=Float32Array.from(
      options.processorOptions.rolloffCoefficients||[1]);this.filterHistory=new Float32Array(this.filterCoefficients.length);
    this.filterCursor=0;
    for(const descriptor of options.processorOptions.kernels){const module=new WebAssembly.Module(descriptor.binary),
        instance=new WebAssembly.Instance(module,{}),memory=instance.exports.memory,base=Math.max(64,Math.ceil(descriptor.reservedBytes/4)*4),
        inputOffsets=this.inputNames.map((_,index)=>base+index*this.blockSize*4),outputOffset=base+this.inputNames.length*this.blockSize*4,
        required=outputOffset+this.blockSize*4;if(required>memory.buffer.byteLength)memory.grow(Math.ceil((required-memory.buffer.byteLength)/65536));
      const views=inputOffsets.map(offset=>new Float32Array(memory.buffer,offset,this.blockSize)),output=new Float32Array(
        memory.buffer,outputOffset,this.blockSize);for(let index=0;index<this.blockSize;index++)views[0][index]=index;
      this.bank.set(descriptor.identity,{instance,views,output,run:()=>instance.exports[descriptor.entrypoint](this.blockSize,...inputOffsets,outputOffset)});}
    this.port.onmessage=event=>{const message=event.data||{};if(message.type==='telemetry'){
      for(const name of Object.keys(this.metrics))if(Number.isFinite(message[name]))this.metrics[name]=Number(message[name]);
      if(this.bank.has(message.profile))this.profile=message.profile;}};
  }
  process(_inputs,outputs){const kernel=this.bank.get(this.profile)||this.bank.values().next().value,channel=outputs[0]?.[0];
    if(!kernel||!channel)return true;const names=this.inputNames,values={sample_rate:sampleRate,phase_start:this.phase,...this.metrics};
    for(let index=1;index<names.length;index++)kernel.views[index].fill(Number(values[names[index]]||0));kernel.run();
    const coefficients=this.filterCoefficients,history=this.filterHistory,n=coefficients.length;
    for(let sample=0;sample<channel.length;sample++){history[this.filterCursor]=kernel.output[sample];let filtered=0;
      for(let tap=0;tap<n;tap++)filtered+=coefficients[tap]*history[(this.filterCursor-tap+n)%n];
      channel[sample]=filtered;this.filterCursor=(this.filterCursor+1)%n;}
    this.phase=(this.phase+this.metrics.rpm*this.blockSize/(120*sampleRate))%1;return true;}
}
registerProcessor('turing-engine-pcm',TuringEnginePCMProcessor);`}

function engineSoundProfileForPreset(identity){return model.engine_audio?.preset_profiles?.[identity]||"inline-four";}

async function ensureEngineSound(){
  const runtime=engineSoundRuntime,descriptor=model.engine_audio;if(!descriptor||runtime.node)return runtime;
  if(runtime.pending)return runtime.pending;runtime.pending=(async()=>{const Context=window.AudioContext||window.webkitAudioContext,
      context=runtime.context||(runtime.context=new Context({latencyHint:"interactive"}));if(context.state==="suspended")await context.resume();
    const url=URL.createObjectURL(new Blob([engineSoundWorkletSource()],{type:"text/javascript"}));
    try{await context.audioWorklet.addModule(url);}finally{URL.revokeObjectURL(url);}
    const kernels=descriptor.kernels.map(kernel=>({identity:kernel.identity,entrypoint:kernel.entrypoint,
      reservedBytes:kernel.reserved_bytes,binary:base64Bytes(kernel.binary_base64)})),profile=engineSoundProfileForPreset(
        vehicleRuntime.powerUnitPreset),node=new AudioWorkletNode(context,"turing-engine-pcm",{numberOfInputs:0,numberOfOutputs:1,
        outputChannelCount:[1],processorOptions:{blockSize:descriptor.block_size,inputs:descriptor.inputs,kernels,initialProfile:profile,
          rolloffCoefficients:descriptor.mid_high_rolloff?.coefficients||[1]}}),
      highpass=context.createBiquadFilter(),compressor=context.createDynamicsCompressor(),gain=context.createGain();
    highpass.type="highpass";highpass.frequency.value=24;
    compressor.threshold.value=-12;compressor.knee.value=8;
    compressor.ratio.value=3.2;compressor.attack.value=.004;compressor.release.value=.09;gain.gain.value=.68;
    node.connect(highpass).connect(compressor).connect(gain).connect(context.destination);runtime.node=node;runtime.gain=gain;
    runtime.profile=profile;runtime.pending=null;return runtime;})().catch(error=>{runtime.pending=null;runtime.error=String(error?.message||error);
      reportRuntimeFault("engine-pcm-audio",error);return runtime;});return runtime.pending;
}

function armEngineSoundOnFirstGesture(){const enter=()=>{cleanup();ensureEngineSound().catch(()=>{});},cleanup=()=>{
    window.removeEventListener("pointerdown",enter,true);window.removeEventListener("keydown",enter,true);};
  window.addEventListener("pointerdown",enter,true);window.addEventListener("keydown",enter,true);
}

function updateEngineSoundTelemetry(throttle){const runtime=engineSoundRuntime,vehicle=vehicleRuntime.active||
    (vehicleRuntime.parkedState?model.vehicle_slot?.vehicles?.[0]:null);
  if(!runtime.node||!vehicle)return;const now=performance.now();if(now-runtime.lastTelemetryAt<30)return;runtime.lastTelemetryAt=now;
  const powertrain=vehicleRuntime.powertrain,preset=vehicle.power_unit_presets?.find(item=>item.identity===vehicleRuntime.powerUnitPreset),
    rpm=Math.max(0,Number(powertrain.engineRPM||0)),omega=Math.max(0,Number(powertrain.engineAngularSpeed||rpm*Math.PI/30)),
    torque=Math.abs(Number(powertrain.engineTorque||0)),maximumTorque=Math.max(1,Number(preset?.configuration?.clutch_maximum_torque_nm||400)),
    idle=Math.max(1,Number(preset?.configuration?.idle_rpm||850)),redline=Math.max(idle+1,Number(preset?.configuration?.redline_rpm||6500)),
    failed=Object.values(vehicleRuntime.damage?.members||{}).filter(item=>item.failed).length,
    memberCount=Math.max(1,Object.keys(vehicleRuntime.damage?.members||{}).length),combustionDamage=Number(
      vehicleRuntime.electrical.combustionDamage||0),profile=engineSoundProfileForPreset(
      vehicleRuntime.powerUnitPreset),message={type:"telemetry",profile,rpm,load:Math.min(1.5,torque/maximumTorque),
      power:Math.min(1.5,torque*omega/(maximumTorque*redline*Math.PI/30)),throttle:Math.min(1,Math.abs(Number(throttle||0))),
      transient:Math.min(1.5,Math.abs(Number(powertrain.engineAccelerationTorque||0))/maximumTorque),
      damage:Math.max(failed/memberCount,combustionDamage),
      stall:Math.max(0,Math.min(1,1-rpm/idle)),ignition_timing_cycles:Number(
        vehicleRuntime.electrical.ignitionTimingOffsetCycles||0),combustion_character:Number(
        vehicleRuntime.electrical.combustionSharpness||1)};runtime.node.port.postMessage(message);runtime.profile=profile;
}

function base64Bytes(encoded) {
  const raw=atob(encoded),bytes=new Uint8Array(raw.length);
  for(let i=0;i<raw.length;i+=1)bytes[i]=raw.charCodeAt(i);
  return bytes;
}

async function ensureMusicRoomAudio() {
  const state=musicRoomRuntime,descriptor=model.music_room;
  if(!state.context)state.context=new (window.AudioContext||window.webkitAudioContext)();
  if(state.context.state==="suspended")await state.context.resume();
  if(!state.buffer)state.buffer=await state.context.decodeAudioData(
    base64Bytes(descriptor.track.wav_base64).buffer.slice(0));
  if(!state.instance){
    state.instance=await turingWasmModules.instantiate(descriptor.module);
    const memory=state.instance.exports.memory,base=Math.ceil(descriptor.reserved_bytes/4)*4;
    const inputOffset=base,outputOffsets=Array.from({length:descriptor.output_bins},
      (_,i)=>inputOffset+descriptor.fft_size*4+i*4);
    const required=outputOffsets.at(-1)+4;
    if(required>memory.buffer.byteLength)memory.grow(Math.ceil((required-memory.buffer.byteLength)/65536));
    state.input=new Float32Array(memory.buffer,inputOffset,descriptor.fft_size);
    state.outputs=outputOffsets.map(offset=>new Float32Array(memory.buffer,offset,1));
    state.run=()=>state.instance.exports[descriptor.entrypoint](1,inputOffset,...outputOffsets);
  }
  return state;
}

async function toggleMusicRoom() {
  const state=await ensureMusicRoomAudio();
  if(state.playing){state.source?.stop();state.source=null;state.playing=false;
    if(state.button){state.button.textContent=`Play ${state.trackLabel}`;state.button.classList.remove("playing");}
    return;
  }
  const source=state.context.createBufferSource(),gain=state.context.createGain();
  source.buffer=state.buffer;source.loop=true;gain.gain.value=.62;
  source.connect(gain).connect(state.context.destination);source.start();
  state.source=source;state.gain=gain;state.startedAt=state.context.currentTime;state.playing=true;
  if(state.button){state.button.textContent=`Pause ${state.trackLabel} · FFT live`;state.button.classList.add("playing");}
}

async function loadMusicRoomFile(file) {
  if(!file)return false;
  const state=musicRoomRuntime;
  if(state.source){try{state.source.stop();}catch(_error){}state.source=null;}
  state.playing=false;
  const context=state.context||(state.context=new (window.AudioContext||window.webkitAudioContext)());
  if(context.state==="suspended")await context.resume();
  state.buffer=await context.decodeAudioData((await file.arrayBuffer()).slice(0));
  state.trackLabel=file.name.replace(/\.[^.]+$/u,"")||"loaded track";
  state.startedAt=0;state.bands=[0,0,0];state.level=0;
  if(state.button){state.button.textContent=`Play ${state.trackLabel}`;
    state.button.classList.remove("playing");state.button.style.background="";}
  if(state.loadButton)state.loadButton.textContent=`Loaded · ${file.name}`;
  return true;
}

function updateMusicRoomAnalysis() {
  const state=musicRoomRuntime;if(!state.playing||!state.input||!state.buffer)return;
  const channel=state.buffer.getChannelData(0),cursor=Math.floor(
    (state.context.currentTime-state.startedAt)*state.buffer.sampleRate)%channel.length;
  for(let i=0;i<state.input.length;i+=1)state.input[i]=channel[(cursor+i)%channel.length];
  state.run();
  const magnitudes=state.outputs.map(view=>Math.sqrt(Math.max(0,view[0]))/32);
  const ranges=[[1,3],[3,9],[9,24]],target=ranges.map(([a,b])=>{
    let sum=0;for(let i=a;i<b;i+=1)sum+=magnitudes[i];return Math.min(1,sum/(b-a)*2.8);
  });
  state.bands=state.bands.map((value,i)=>value*.78+target[i]*.22);
  state.level=Math.min(1,(state.bands[0]+state.bands[1]+state.bands[2])*.48);
  if(state.button)state.button.style.background=`linear-gradient(90deg,rgba(30,210,255,${.15+state.bands[0]*.55}),rgba(255,45,180,${.15+state.bands[1]*.55}),rgba(255,220,55,${.12+state.bands[2]*.5}))`;
}

function spectrumColorHex(offset=0) {
  const b=musicRoomRuntime.bands,phase=(musicRoomRuntime.sequence++*.17+offset)%1;
  const rgb=[.18+b[0]*.8,.16+b[1]*.84,.2+b[2]*.78];
  if(!musicRoomRuntime.playing){rgb[phase<.5?0:2]+=.45;rgb[1]+=.22;}
  return `#${rgb.map(value=>Math.round(Math.min(1,value)*255).toString(16).padStart(2,"0")).join("")}`;
}

function playToyImpact(speed,position) {
  const state=musicRoomRuntime,now=performance.now();
  if(!state.context||now-state.impactAt<28||speed<.45)return;
  state.impactAt=now;const context=state.context,t=context.currentTime;
  const oscillator=context.createOscillator(),gain=context.createGain();
  oscillator.type="triangle";oscillator.frequency.setValueAtTime(
    115+Math.min(900,speed*72)+state.bands[1]*240,t);
  oscillator.frequency.exponentialRampToValueAtTime(58,t+.13);
  gain.gain.setValueAtTime(Math.min(.16,.018+speed*.012),t);
  gain.gain.exponentialRampToValueAtTime(.0001,t+.14);
  oscillator.connect(gain).connect(context.destination);oscillator.start(t);oscillator.stop(t+.15);
}

const physicsRuntime = {
  instance: null, plugin: null, pending: false, error: null,
  verticalVelocity: 0, last: null, supportSuppressedUntil: 0,
  parameters: new Map((model.physics_program?.parameters || []).map(parameter =>
    [parameter.name, Number(parameter.default)]))
};

const stateLoopRuntime = {
  worker: null, workerUrl: null, ready: false, mode: "initializing",
  engineStage: "full-dynamics", engineSleeping: false, engineSleepReason: null,
  latestSequence: 0, appliedSequence: 0, bodies: new Map(), actorRegistered: false,
  fallbackReason: null, lastWorkerCrash: null, forcingWasm: false,
  snapshotStride: 162, snapshotCapacity: 128,
  freeSlots: [], slotRecords: [], slotGenerations: new Uint32Array(128)
};

function initializePhysicsSnapshotSlots(capacity) {
  stateLoopRuntime.snapshotCapacity = capacity;
  stateLoopRuntime.freeSlots = Array.from({length: capacity}, (_, index) => capacity - index - 1);
  stateLoopRuntime.slotRecords = new Array(capacity).fill(null);
  stateLoopRuntime.slotGenerations = new Uint32Array(capacity);
}

function reservePhysicsSnapshotSlot(identity) {
  const existing = stateLoopRuntime.bodies.get(identity);
  if (existing) return existing;
  const slot = stateLoopRuntime.freeSlots.pop();
  if (slot === undefined) throw new Error("physics snapshot capacity exhausted");
  const generation = ++stateLoopRuntime.slotGenerations[slot];
  const record = {identity, slot, generation, position: [0,0,0], velocity: [0,0,0],
    contactIdentity: null, contactRuntimePartId: 0, controlGeneration: 0,
    snapshotControlGeneration: 0, lastSubmittedPosition: null};
  stateLoopRuntime.slotRecords[slot] = record;
  stateLoopRuntime.bodies.set(identity, record);
  return record;
}

function releasePhysicsSnapshotSlot(identity) {
  const record = stateLoopRuntime.bodies.get(identity);
  if (!record) return;
  stateLoopRuntime.bodies.delete(identity);
  stateLoopRuntime.slotRecords[record.slot] = null;
  stateLoopRuntime.freeSlots.push(record.slot);
}

initializePhysicsSnapshotSlots(stateLoopRuntime.snapshotCapacity);

const navigationRuntime = {
  kernels: new Map(),
  assignments: new Map(Object.entries(model.navigation?.assignments || {})),
  routes: new Map(),
  waypointQueues: new Map(), presenceHooks: new Set(), waypointSequence: 0,
  ready: null,
  error: null,
  lastQuaternion: new Map(), statusElement: null,
  worker: null, workerUrl: null, workerReady: false,
  requestSequence: 0, pending: new Map(), generations: new Map()
};

const sourceScopeRuntime = {
  closures: [], byIdentity: new Map(), visited: new Set(), current: null,
  district: null, grid: null, statusElement: null
};

const documentWorldSync = {
  dirty: true, revision: 0, frames: [], resizeObserver: null, structuralSignature: null
};

function navigationBytes(encoded) {
  const raw = atob(encoded);
  const bytes = new Uint8Array(raw.length);
  for (let index = 0; index < raw.length; index += 1) bytes[index] = raw.charCodeAt(index);
  return bytes;
}

async function installNavigationKernel(descriptor) {
  if (descriptor?.schema !== "abstract-ui-navigation-assembly-v0" ||
      descriptor?.format !== "webassembly") {
    throw new Error("navigation kernel must implement abstract-ui-navigation-assembly-v0");
  }
  const instantiated = await WebAssembly.instantiate(navigationBytes(descriptor.binary_base64), {});
  const instance = instantiated.instance;
  if (!(instance.exports.memory instanceof WebAssembly.Memory) ||
      typeof instance.exports[descriptor.entrypoint] !== "function") {
    throw new Error(`navigation kernel ${descriptor.identity} does not implement its declared ABI`);
  }
  navigationRuntime.kernels.set(descriptor.identity, {descriptor, instance});
  if (navigationRuntime.workerReady) {
    await requestNavigationWorker("install", {descriptor});
  }
  return descriptor.identity;
}

function assignNavigationKernel(entityIdentity, kernelIdentity) {
  if (!entityState.has(entityIdentity)) throw new Error(`unknown navigation entity ${entityIdentity}`);
  if (!navigationRuntime.kernels.has(kernelIdentity)) throw new Error(`unknown navigation kernel ${kernelIdentity}`);
  cancelEntityNavigation(entityIdentity);
  navigationRuntime.assignments.set(entityIdentity, kernelIdentity);
  return kernelIdentity;
}

function cancelEntityNavigation(entityIdentity) {
  navigationRuntime.generations.set(entityIdentity,
    (navigationRuntime.generations.get(entityIdentity) || 0) + 1);
  clearNavigationRouteOverlay(entityIdentity);
  const removed = navigationRuntime.routes.delete(entityIdentity);
  const queue = navigationRuntime.waypointQueues.get(entityIdentity);
  if (queue) {
    queue.waypoints.length = 0;
    queue.pauseUntil = 0;
    queue.presenceReady = true;
    queue.presenceToken += 1;
    queue.current = null;
  }
  if (navigationRuntime.statusElement?.dataset.entity === entityIdentity) {
    navigationRuntime.statusElement.classList.remove("active", "planning", "presence");
    navigationRuntime.statusElement.dataset.queueDepth = "0";
    navigationRuntime.statusElement.dataset.queuedWaypoints = "0";
  }
  return removed || Boolean(queue?.planning || queue?.waypoints.length || queue?.pauseUntil);
}

function navigationGridSpec() {
  const grid = model.navigation.grid;
  const domainKinds = new Set(grid.domain_geometry_kinds || ["courtyard", "building", "room"]);
  const domain = shaderViewer.geometry.filter(box => domainKinds.has(box.kind));
  const axisProfile = axis => {
    const landmarks = [];
    domain.forEach(box => {
      const center = box.center[axis], half = box.half_extent[axis];
      landmarks.push(center-half, center, center+half);
      (box.openings || []).forEach(opening => {
        const runsAlongAxis = (axis === 0 && ["north","south"].includes(opening.side)) ||
          (axis === 1 && ["east","west"].includes(opening.side));
        if (runsAlongAxis) {
          const absolute = center + Number(opening.offset || 0);
          landmarks.push(absolute-Number(opening.width)*0.5,
            absolute, absolute+Number(opening.width)*0.5);
        }
      });
    });
    if (!landmarks.length) return [{world: 0, traversal: 0}, {world: 1, traversal: 1}];
    const unique = [...new Set(landmarks.map(value => Number(value.toFixed(9))))].sort((a,b) => a-b);
    let traversal = 0;
    return unique.map((world, index) => {
      if (index) {
        const distance = world - unique[index-1];
        traversal += distance <= 2 ? distance : 2 + Math.log1p(distance-2);
      }
      return {world, traversal};
    });
  };
  const xProfile = axisProfile(0), zProfile = axisProfile(1);
  const origin = [0, 0];
  const extent = [xProfile.at(-1).traversal, zProfile.at(-1).traversal];
  return {width: grid.width, height: grid.height, origin, extent, xProfile, zProfile, domainKinds,
    cellX: extent[0] / grid.width, cellZ: extent[1] / grid.height};
}

function navigationAxisTransform(value, profile, source, target) {
  if (value <= profile[0][source]) {
    const left = profile[0], right = profile[1];
    return left[target] + (value-left[source]) *
      (right[target]-left[target]) / Math.max(1e-9, right[source]-left[source]);
  }
  if (value >= profile.at(-1)[source]) {
    const left = profile.at(-2), right = profile.at(-1);
    return right[target] + (value-right[source]) *
      (right[target]-left[target]) / Math.max(1e-9, right[source]-left[source]);
  }
  let low = 0, high = profile.length-2;
  while (low < high) {
    const middle = Math.floor((low+high)/2);
    if (value > profile[middle+1][source]) low = middle+1;
    else high = middle;
  }
  const index = low;
  const left = profile[index], right = profile[index+1];
  const t = (value-left[source]) / Math.max(1e-9, right[source]-left[source]);
  return left[target] + (right[target]-left[target]) * t;
}

function navigationWorldToTraversal(position, spec) {
  return [navigationAxisTransform(position[0], spec.xProfile, "world", "traversal"),
    navigationAxisTransform(position[1], spec.zProfile, "world", "traversal")];
}

function navigationTraversalToWorld(position, spec) {
  return [navigationAxisTransform(position[0], spec.xProfile, "traversal", "world"),
    navigationAxisTransform(position[1], spec.zProfile, "traversal", "world")];
}

function elementOffsetWithin(element, root) {
  const frame = element.getBoundingClientRect(), rootFrame = root.getBoundingClientRect();
  const scaleX = rootFrame.width / Math.max(1, Number(root.offsetWidth || rootFrame.width));
  const scaleY = rootFrame.height / Math.max(1, Number(root.offsetHeight || rootFrame.height));
  return {left: (frame.left-rootFrame.left)/scaleX + Number(root.scrollLeft || 0),
    top: (frame.top-rootFrame.top)/scaleY + Number(root.scrollTop || 0),
    width: frame.width/scaleX, height: frame.height/scaleY};
}

function viewportToElementPoint(position, element) {
  const frame = element.getBoundingClientRect();
  const scaleX = frame.width / Math.max(1, Number(element.offsetWidth || frame.width));
  const scaleY = frame.height / Math.max(1, Number(element.offsetHeight || frame.height));
  return [(position[0]-frame.left)/scaleX + Number(element.scrollLeft || 0),
    (position[1]-frame.top)/scaleY + Number(element.scrollTop || 0)];
}

function resyncDocumentWorldMap() {
  const root = shaderViewer.mapElement;
  if (!root) return documentWorldSync;
  const domainKinds = new Set(model.navigation?.grid.domain_geometry_kinds ||
    ["courtyard", "building", "room"]);
  const frames = [];
  shaderViewer.geometry.forEach(box => {
    if (!domainKinds.has(box.kind)) return;
    const element = root.querySelector(`[data-node-id="${CSS.escape(box.identity)}"]`);
    if (!element) return;
    const documentFrame = elementOffsetWithin(element, root);
    if (!documentFrame.width || !documentFrame.height) return;
    const worldFrame = {
      left: box.center[0]-box.half_extent[0], right: box.center[0]+box.half_extent[0],
      top: box.center[1]-box.half_extent[1], bottom: box.center[1]+box.half_extent[1]
    };
    const color = box.appearance?.face_color || model.appearance.colors[box.palette_role];
    if (color) element.style.setProperty("--world-object-color", color);
    element.dataset.worldFrame = [worldFrame.left, worldFrame.top,
      worldFrame.right, worldFrame.bottom].map(value => value.toFixed(6)).join(",");
    element.dataset.documentFrame = [documentFrame.left, documentFrame.top,
      documentFrame.width, documentFrame.height].map(value => value.toFixed(3)).join(",");
    element.dataset.coordinateSyncRevision = String(documentWorldSync.revision+1);
    frames.push({identity: box.identity, kind: box.kind, box, element,
      document: documentFrame, world: worldFrame,
      worldArea: Math.max(1e-9, (worldFrame.right-worldFrame.left)*(worldFrame.bottom-worldFrame.top)),
      documentArea: documentFrame.width*documentFrame.height});
  });
  const axisProfile = (frame, axis) => {
    const worldStart = axis === 0 ? frame.world.left : frame.world.top;
    const worldEnd = axis === 0 ? frame.world.right : frame.world.bottom;
    const documentStart = axis === 0 ? frame.document.left : frame.document.top;
    const documentEnd = documentStart + (axis === 0 ? frame.document.width : frame.document.height);
    const landmarks = [{world: worldStart, document: documentStart},
      {world: worldEnd, document: documentEnd}];
    frames.filter(child => child.box.parent_identity === frame.identity).forEach(child => {
      landmarks.push(axis === 0
        ? {world: child.world.left, document: child.document.left}
        : {world: child.world.top, document: child.document.top});
      landmarks.push(axis === 0
        ? {world: child.world.right, document: child.document.left+child.document.width}
        : {world: child.world.bottom, document: child.document.top+child.document.height});
    });
    const grouped = new Map();
    landmarks.forEach(point => {
      const key = point.world.toFixed(9), values = grouped.get(key) || [];
      values.push(point.document); grouped.set(key, values);
    });
    return [...grouped.entries()].map(([world, values]) => ({world: Number(world),
      document: values.reduce((sum,value) => sum+value,0)/values.length}))
      .sort((left,right) => left.world-right.world);
  };
  frames.forEach(frame => {
    frame.xProfile = axisProfile(frame, 0);
    frame.yProfile = axisProfile(frame, 1);
  });
  documentWorldSync.frames = frames;
  documentWorldSync.revision += 1;
  documentWorldSync.dirty = false;
  root.dataset.coordinateSyncRevision = String(documentWorldSync.revision);
  root.dataset.coordinateSync = "rendered-border-frames";
  return documentWorldSync;
}

function markDocumentWorldSyncForGeometry() {
  const domainKinds = new Set(model.navigation?.grid.domain_geometry_kinds ||
    ["courtyard", "building", "room"]);
  const signature = JSON.stringify(shaderViewer.geometry.filter(box => domainKinds.has(box.kind))
    .map(box => [box.identity, box.center, box.half_extent, box.palette_role,
      box.appearance?.face_color || null]));
  if (signature !== documentWorldSync.structuralSignature) {
    documentWorldSync.structuralSignature = signature;
    documentWorldSync.dirty = true;
  }
}

function synchronizedDocumentWorldMap() {
  return documentWorldSync.dirty ? resyncDocumentWorldMap() : documentWorldSync;
}

function worldToDocumentPoint(position) {
  const sync = synchronizedDocumentWorldMap();
  const containing = sync.frames.filter(frame =>
    position[0] >= frame.world.left && position[0] <= frame.world.right &&
    position[1] >= frame.world.top && position[1] <= frame.world.bottom)
    .sort((left,right) => left.worldArea-right.worldArea)[0];
  if (containing) {
    return [navigationAxisTransform(position[0], containing.xProfile, "world", "document"),
      navigationAxisTransform(position[1], containing.yProfile, "world", "document")];
  }
  const root = shaderViewer.mapElement, spec = navigationGridSpec();
  const traversal = navigationWorldToTraversal(position, spec);
  return [(traversal[0]-spec.origin[0])/spec.extent[0]*Number(root?.offsetWidth || 1),
    (traversal[1]-spec.origin[1])/spec.extent[1]*Number(root?.offsetHeight || 1)];
}

function documentToWorldPoint(position) {
  const sync = synchronizedDocumentWorldMap();
  const containing = sync.frames.filter(frame =>
    position[0] >= frame.document.left &&
    position[0] <= frame.document.left+frame.document.width &&
    position[1] >= frame.document.top &&
    position[1] <= frame.document.top+frame.document.height)
    .sort((left,right) => left.documentArea-right.documentArea)[0];
  if (containing) {
    return [navigationAxisTransform(position[0], containing.xProfile, "document", "world"),
      navigationAxisTransform(position[1], containing.yProfile, "document", "world")];
  }
  const root = shaderViewer.mapElement, spec = navigationGridSpec();
  return navigationTraversalToWorld([
    spec.origin[0]+position[0]/Math.max(1, Number(root?.offsetWidth || 1))*spec.extent[0],
    spec.origin[1]+position[1]/Math.max(1, Number(root?.offsetHeight || 1))*spec.extent[1]
  ], spec);
}

function openingContainsCoordinate(box, side, coordinate, clearance) {
  return (box.openings || []).some(opening => opening.side === side &&
    Math.abs(coordinate - (Number(opening.offset) || 0)) <=
      Number(opening.width) * 0.5 - clearance * 0.35);
}

function navigationWorldPointBlocked(x, z, clearance = model.navigation.grid.clearance) {
  const domainKinds = new Set(model.navigation.grid.domain_geometry_kinds || ["courtyard","building","room"]);
  return shaderViewer.geometry.some(box => {
    if (!domainKinds.has(box.kind)) return false;
    const [cx, cz] = box.center, [hx, hz] = box.half_extent;
    const thickness = Number(box.wall_thickness || 0.04) * 0.5 + clearance;
    const yaw = boxYawDegrees(box)*Math.PI/180;
    const dx=x-cx,dz=z-cz,cosine=Math.cos(yaw),sine=Math.sin(yaw);
    const localX=dx*cosine+dz*sine,localZ=-dx*sine+dz*cosine;
    const withinX = Math.abs(localX) <= hx + thickness;
    const withinZ = Math.abs(localZ) <= hz + thickness;
    if (!withinX || !withinZ) return false;
    const south = Math.abs(localZ + hz) <= thickness &&
      !openingContainsCoordinate(box, "south", localX, clearance);
    const north = Math.abs(localZ - hz) <= thickness &&
      !openingContainsCoordinate(box, "north", localX, clearance);
    const west = Math.abs(localX + hx) <= thickness &&
      !openingContainsCoordinate(box, "west", localZ, clearance);
    const east = Math.abs(localX - hx) <= thickness &&
      !openingContainsCoordinate(box, "east", localZ, clearance);
    return south || north || west || east;
  });
}

function navigationPointBlocked(x, z, spec, clearance = model.navigation.grid.clearance) {
  const world = navigationTraversalToWorld([x,z], spec);
  return navigationWorldPointBlocked(world[0], world[1], clearance);
}

function navigationCellToWorld(cell, spec) {
  return [spec.origin[0] + ((cell % spec.width) + 0.5) * spec.cellX,
    spec.origin[1] + (Math.floor(cell / spec.width) + 0.5) * spec.cellZ];
}

function navigationWorldToCell(position, spec) {
  const traversal = navigationWorldToTraversal(position, spec);
  const x = Math.max(0, Math.min(spec.width - 1,
    Math.floor((traversal[0] - spec.origin[0]) / spec.cellX)));
  const z = Math.max(0, Math.min(spec.height - 1,
    Math.floor((traversal[1] - spec.origin[1]) / spec.cellZ)));
  return z * spec.width + x;
}

function buildNavigationGrid(spec) {
  const blocked = new Int32Array(spec.width * spec.height);
  for (let cell = 0; cell < blocked.length; cell += 1) {
    const [x, z] = navigationCellToWorld(cell, spec);
    blocked[cell] = navigationPointBlocked(x, z, spec) ? 1 : 0;
  }
  return blocked;
}

function nearestOpenNavigationCell(cell, blocked, spec) {
  if (!blocked[cell]) return cell;
  const startX = cell % spec.width, startZ = Math.floor(cell / spec.width);
  for (let radius = 1; radius < Math.max(spec.width, spec.height); radius += 1) {
    for (let dz = -radius; dz <= radius; dz += 1) for (let dx = -radius; dx <= radius; dx += 1) {
      if (Math.max(Math.abs(dx), Math.abs(dz)) !== radius) continue;
      const x = startX + dx, z = startZ + dz;
      if (x >= 0 && z >= 0 && x < spec.width && z < spec.height && !blocked[z * spec.width + x]) {
        return z * spec.width + x;
      }
    }
  }
  return -1;
}

function navigationLineClear(left, right, blocked, spec) {
  let x0 = left % spec.width, z0 = Math.floor(left / spec.width);
  const x1 = right % spec.width, z1 = Math.floor(right / spec.width);
  const dx = Math.abs(x1 - x0), dz = Math.abs(z1 - z0);
  const sx = x0 < x1 ? 1 : -1, sz = z0 < z1 ? 1 : -1;
  let error = dx - dz;
  while (true) {
    if (blocked[z0 * spec.width + x0]) return false;
    if (x0 === x1 && z0 === z1) break;
    const doubled = error * 2;
    if (doubled > -dz) { error -= dz; x0 += sx; }
    if (doubled < dx) { error += dx; z0 += sz; }
  }
  return navigationSegmentClear(navigationCellToWorld(left, spec),
    navigationCellToWorld(right, spec), spec);
}

function navigationSegmentClear(left, right, spec) {
  const leftWorld = navigationTraversalToWorld(left, spec);
  const rightWorld = navigationTraversalToWorld(right, spec);
  const worldDistance = Math.hypot(rightWorld[0]-leftWorld[0], rightWorld[1]-leftWorld[1]);
  const traversalDistance = Math.hypot(right[0]-left[0], right[1]-left[1]);
  const worldInterval = Math.max(0.025, model.navigation.grid.clearance*0.28);
  const traversalInterval = Math.max(0.012, Math.min(spec.cellX,spec.cellZ)*0.28);
  const steps = Math.max(1, Math.ceil(worldDistance/worldInterval),
    Math.ceil(traversalDistance/traversalInterval));
  for (let step = 0; step <= steps; step += 1) {
    const t = step/steps;
    const point = [left[0]+(right[0]-left[0])*t, left[1]+(right[1]-left[1])*t];
    if (navigationPointBlocked(point[0], point[1], spec)) return false;
  }
  return true;
}

function simplifyNavigationCells(cells, blocked, spec) {
  if (cells.length <= 2) return cells;
  const simplified = [cells[0]];
  let anchor = 0;
  while (anchor < cells.length - 1) {
    let visible = cells.length - 1;
    while (visible > anchor + 1 && !navigationLineClear(cells[anchor], cells[visible], blocked, spec)) visible -= 1;
    simplified.push(cells[visible]); anchor = visible;
  }
  return simplified;
}

function catmullRomPoint(p0, p1, p2, p3, t) {
  const t2 = t * t, t3 = t2 * t;
  return [0, 1].map(axis => 0.5 * ((2 * p1[axis]) + (-p0[axis] + p2[axis]) * t +
    (2*p0[axis] - 5*p1[axis] + 4*p2[axis] - p3[axis]) * t2 +
    (-p0[axis] + 3*p1[axis] - 3*p2[axis] + p3[axis]) * t3));
}

function quaternionForDirection(dx, dz) {
  const yaw = Math.atan2(dz, dx);
  return [0, -Math.sin(yaw * 0.5), 0, Math.cos(yaw * 0.5)];
}

function quaternionSlerp(left, right, t) {
  let target = [...right];
  let dot = left.reduce((sum, value, index) => sum + value * target[index], 0);
  if (dot < 0) { target = target.map(value => -value); dot = -dot; }
  if (dot > 0.9995) {
    const mixed = left.map((value, index) => value + (target[index] - value) * t);
    const length = Math.hypot(...mixed) || 1; return mixed.map(value => value / length);
  }
  const angle = Math.acos(Math.max(-1, Math.min(1, dot)));
  const scale = Math.sin(angle) || 1;
  return left.map((value, index) =>
    (value * Math.sin((1-t)*angle) + target[index] * Math.sin(t*angle)) / scale);
}

function navigationSpline(points, spec) {
  if (points.length < 2) return [];
  const positions = [];
  for (let segment = 0; segment < points.length - 1; segment += 1) {
    const p0 = points[Math.max(0, segment - 1)], p1 = points[segment];
    const p2 = points[segment + 1], p3 = points[Math.min(points.length - 1, segment + 2)];
    const curve = Array.from({length: 9}, (_, index) => catmullRomPoint(p0, p1, p2, p3, index / 8));
    const safe = curve.every(point => !navigationPointBlocked(point[0], point[1], spec)) &&
      curve.slice(1).every((point,index) => navigationSegmentClear(curve[index], point, spec));
    const samples = safe ? curve : Array.from({length: 9}, (_, index) => {
      const t = index / 8; return [p1[0] + (p2[0]-p1[0])*t, p1[1] + (p2[1]-p1[1])*t];
    });
    positions.push(...samples.slice(segment ? 1 : 0));
  }
  let distance = 0;
  return positions.map((position, index) => {
    const next = positions[Math.min(positions.length - 1, index + 1)];
    const previous = positions[Math.max(0, index - 1)];
    if (index) distance += Math.hypot(position[0]-previous[0], position[1]-previous[1]);
    return {position, distance, quaternion: quaternionForDirection(
      next[0] - previous[0], next[1] - previous[1])};
  });
}

function navigationPolyline(points) {
  let distance = 0;
  return points.map((position, index) => {
    const previous = points[Math.max(0, index-1)];
    const next = points[Math.min(points.length-1, index+1)];
    if (index) distance += Math.hypot(position[0]-previous[0], position[1]-previous[1]);
    return {position: [...position], distance, quaternion: quaternionForDirection(
      next[0]-previous[0], next[1]-previous[1])};
  });
}

function clearNavigationRouteOverlay(entityIdentity) {
  const route = navigationRuntime.routes.get(entityIdentity);
  route?.overlay?.group?.remove();
  if (route) route.overlay = null;
}

function navigationRouteDocumentPoints(route) {
  const project = traversal => worldToDocumentPoint(
    navigationTraversalToWorld(traversal, route.spec));
  const subdivide = (left, right, leftDocument, rightDocument, depth) => {
    const middle = [(left[0]+right[0])*0.5, (left[1]+right[1])*0.5];
    const middleDocument = project(middle);
    const linearMiddle = [(leftDocument[0]+rightDocument[0])*0.5,
      (leftDocument[1]+rightDocument[1])*0.5];
    const deviation = Math.hypot(middleDocument[0]-linearMiddle[0],
      middleDocument[1]-linearMiddle[1]);
    const projectedLength = Math.hypot(rightDocument[0]-leftDocument[0],
      rightDocument[1]-leftDocument[1]);
    if (depth < 9 && (deviation > 0.35 || projectedLength > 10)) {
      return [...subdivide(left, middle, leftDocument, middleDocument, depth+1),
        ...subdivide(middle, right, middleDocument, rightDocument, depth+1)];
    }
    return [rightDocument];
  };
  if (!route.samples.length) return [];
  const points = [project(route.samples[0].position)];
  route.samples.slice(1).forEach((sample, index) => {
    const left = route.samples[index].position, right = sample.position;
    points.push(...subdivide(left, right, points.at(-1), project(right), 0));
  });
  return points;
}

function installNavigationRouteOverlay(entityIdentity, route) {
  const root = shaderViewer.mapElement;
  if (!root) return;
  let layer = root.querySelector(".navigation-route-layer");
  if (!layer) {
    layer = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    layer.classList.add("navigation-route-layer");
    layer.setAttribute("aria-hidden", "true");
    root.append(layer);
  }
  const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
  group.dataset.navigationEntity = entityIdentity;
  group.dataset.projection = "adaptive-nonlinear-context-transform";
  const base = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  const progress = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  base.classList.add("navigation-route-base");
  progress.classList.add("navigation-route-progress");
  group.append(base, progress); layer.append(group);
  route.overlay = {group, base, progress, syncRevision: -1, length: 0};
  updateNavigationRouteOverlay(route, 0);
}

function updateNavigationRouteOverlay(route, ratio) {
  const overlay = route.overlay;
  if (!overlay) return;
  const sync = synchronizedDocumentWorldMap();
  if (overlay.syncRevision !== sync.revision) {
    const points = navigationRouteDocumentPoints(route)
      .map(point => `${point[0].toFixed(3)},${point[1].toFixed(3)}`).join(" ");
    overlay.base.setAttribute("points", points);
    overlay.progress.setAttribute("points", points);
    overlay.length = Number(overlay.base.getTotalLength?.() || 0);
    overlay.syncRevision = sync.revision;
  }
  const progress = Math.max(0, Math.min(1, ratio));
  const length = Math.max(1, overlay.length);
  overlay.progress.setAttribute("stroke-dasharray", `${length} ${length}`);
  overlay.progress.setAttribute("stroke-dashoffset", String(length*(1-progress)));
  overlay.progress.style.stroke = `hsl(${35+175*progress} 92% 62%)`;
  overlay.group.dataset.progress = progress.toFixed(4);
}

function runNavigationAssemblyKernel(kernel, blocked, spec, start, goal) {
  const {instance, descriptor} = kernel;
  const heapBase = Number(instance.exports.__heap_base?.value || 196608);
  const blockedPointer = (heapBase + 15) & ~15;
  const pathPointer = blockedPointer + blocked.byteLength;
  const required = pathPointer + blocked.byteLength;
  const memory = instance.exports.memory;
  if (required > memory.buffer.byteLength) memory.grow(Math.ceil((required-memory.buffer.byteLength)/65536));
  new Int32Array(memory.buffer, blockedPointer, blocked.length).set(blocked);
  const length = instance.exports[descriptor.entrypoint](
    blockedPointer, spec.width, spec.height, start, goal, pathPointer, blocked.length);
  if (length <= 0) return [];
  return Array.from(new Int32Array(memory.buffer, pathPointer, length));
}

function planNavigationRoute(startPosition, destination, spec, kernel) {
  const blocked = buildNavigationGrid(spec);
  let start = nearestOpenNavigationCell(navigationWorldToCell(startPosition, spec), blocked, spec);
  let goal = nearestOpenNavigationCell(navigationWorldToCell(destination, spec), blocked, spec);
  if (start < 0 || goal < 0) throw new Error("no open navigation cell near route endpoint");
  let cells = [];
  for (let attempt = 0; attempt < blocked.length; attempt += 1) {
    cells = runNavigationAssemblyKernel(kernel, blocked, spec, start, goal);
    if (!cells.length) break;
    const unsafe = cells.slice(1).findIndex((cell, index) => !navigationSegmentClear(
      navigationCellToWorld(cells[index], spec), navigationCellToWorld(cell, spec), spec));
    if (unsafe < 0) break;
    const rightIndex = unsafe+1;
    const rejectIndex = rightIndex === cells.length-1 ? rightIndex-1 : rightIndex;
    const rejected = cells[rejectIndex];
    if (rejected === start || rejected === goal) { cells = []; break; }
    blocked[rejected] = 1;
    cells = [];
  }
  if (!cells.length) throw new Error("navigation kernel found no route");
  const waypoints = simplifyNavigationCells(cells, blocked, spec)
    .map(cell => navigationCellToWorld(cell, spec));
  const exactStart = navigationWorldToTraversal(startPosition, spec);
  if (waypoints.length < 2 || navigationSegmentClear(exactStart, waypoints[1], spec)) {
    waypoints[0] = exactStart;
  }
  waypoints[waypoints.length - 1] = navigationCellToWorld(goal, spec);
  let samples = waypoints.length === 1
    ? [{position: [...waypoints[0]], distance: 0, quaternion: quaternionForDirection(1, 0)}]
    : navigationSpline(waypoints, spec);
  let collisionCertified = samples.slice(1).every((sample, index) =>
    navigationSegmentClear(samples[index].position, sample.position, spec));
  if (!collisionCertified) {
    samples = navigationPolyline(waypoints);
    collisionCertified = samples.slice(1).every((sample, index) =>
      navigationSegmentClear(samples[index].position, sample.position, spec));
  }
  if (!collisionCertified) throw new Error("spline failed continuous wall-clearance certification");
  return {cells: cells.length, waypoints: waypoints.length, samples,
    destination: waypoints.at(-1), collisionCertified};
}

function navigationPlannerWorkerSource() {
  const definitions = [navigationBytes, boxYawDegrees, navigationAxisTransform,
    navigationWorldToTraversal, navigationTraversalToWorld,
    openingContainsCoordinate, navigationWorldPointBlocked, navigationPointBlocked,
    navigationCellToWorld, navigationWorldToCell, buildNavigationGrid,
    nearestOpenNavigationCell, navigationLineClear, navigationSegmentClear,
    simplifyNavigationCells, catmullRomPoint, quaternionForDirection,
    navigationSpline, navigationPolyline, runNavigationAssemblyKernel,
    planNavigationRoute].map(fn => fn.toString()).join("\n");
  return `${definitions}
let model={navigation:{grid:{clearance:.12}}},shaderViewer={geometry:[]};
const kernels=new Map();
async function install(descriptor){
  const made=await WebAssembly.instantiate(navigationBytes(descriptor.binary_base64),{});
  kernels.set(descriptor.identity,{descriptor,instance:made.instance});
}
onmessage=async event=>{const message=event.data||{};
  try{
    if(message.type==="init"){
      for(const descriptor of message.kernels||[])await install(descriptor);
      postMessage({type:"ready"});return;
    }
    if(message.type==="install"){
      await install(message.descriptor);
      postMessage({type:"result",requestId:message.requestId,result:message.descriptor.identity});return;
    }
    if(message.type==="plan"){
      model={navigation:{grid:message.grid}};shaderViewer={geometry:message.geometry};
      const kernel=kernels.get(message.kernelIdentity);
      if(!kernel)throw new Error("navigation worker kernel is not installed: "+message.kernelIdentity);
      const result=planNavigationRoute(message.startPosition,message.destination,message.spec,kernel);
      postMessage({type:"result",requestId:message.requestId,result});return;
    }
  }catch(error){postMessage({type:"error",requestId:message.requestId,error:error.message||String(error)});}
};`;
}

function requestNavigationWorker(type, payload) {
  if (!navigationRuntime.workerReady || !navigationRuntime.worker) {
    return Promise.reject(new Error("navigation planning worker is not ready"));
  }
  const requestId = ++navigationRuntime.requestSequence;
  return new Promise((resolve, reject) => {
    navigationRuntime.pending.set(requestId, {resolve, reject});
    navigationRuntime.worker.postMessage({type, requestId, ...payload});
  });
}

async function initializeNavigationPlannerWorker() {
  navigationRuntime.workerUrl = URL.createObjectURL(new Blob(
    [navigationPlannerWorkerSource()], {type: "text/javascript"}));
  const worker = new Worker(navigationRuntime.workerUrl);
  navigationRuntime.worker = worker;
  await new Promise((resolve, reject) => {
    worker.onmessage = event => {
      const message = event.data || {};
      if (message.type === "ready") {
        navigationRuntime.workerReady = true; resolve(); return;
      }
      const pending = navigationRuntime.pending.get(message.requestId);
      if (!pending) return;
      navigationRuntime.pending.delete(message.requestId);
      if (message.type === "error") pending.reject(new Error(message.error));
      else pending.resolve(message.result);
    };
    worker.onerror = event => reject(new Error(event.message || "navigation planning worker failed"));
    worker.postMessage({type: "init", kernels: model.navigation?.kernels || []});
  });
}

async function locateEntity(entityIdentity, destination) {
  await navigationRuntime.ready;
  const state = entityState.get(entityIdentity);
  if (!state) throw new Error(`unknown navigation entity ${entityIdentity}`);
  const kernelIdentity = navigationRuntime.assignments.get(entityIdentity) || model.navigation.default_kernel;
  const kernel = navigationRuntime.kernels.get(kernelIdentity);
  if (!kernel) throw new Error(`navigation kernel is not installed: ${kernelIdentity}`);
  const spec = navigationGridSpec();
  const origin = model.document_geometry.origin || [0, 0];
  const startPosition = state.worldPosition ? [state.worldPosition[0], state.worldPosition[2]] :
    shaderViewer.cameraPosition ? [shaderViewer.cameraPosition[0], shaderViewer.cameraPosition[2]] :
    [origin[0] + model.document_geometry.extent[0]*0.5, origin[1] + model.document_geometry.extent[1]*0.5];
  const generation = (navigationRuntime.generations.get(entityIdentity) || 0) + 1;
  navigationRuntime.generations.set(entityIdentity, generation);
  const workerSpec = {...spec, domainKinds: [...spec.domainKinds]};
  const domainKinds = new Set(model.navigation.grid.domain_geometry_kinds ||
    ["courtyard", "building", "room"]);
  const geometry = shaderViewer.geometry.filter(box => domainKinds.has(box.kind));
  if (navigationRuntime.statusElement) {
    navigationRuntime.statusElement.classList.add("planning");
    navigationRuntime.statusElement.dataset.entity = entityIdentity;
    navigationRuntime.statusElement.dataset.plannerThread = "dedicated-worker";
    navigationRuntime.statusElement.textContent = "planning route in navigation worker…";
  }
  const plan = await requestNavigationWorker("plan", {startPosition, destination,
    spec: workerSpec, kernelIdentity, grid: model.navigation.grid, geometry});
  if (navigationRuntime.generations.get(entityIdentity) !== generation) {
    throw new Error("navigation planning was interrupted");
  }
  const {samples, collisionCertified} = plan;
  clearNavigationRouteOverlay(entityIdentity);
  const activeRoute = {samples, distance: 0,
    total: samples.at(-1)?.distance || 0, speed: model.navigation.traversal.speed,
    kernel: kernelIdentity, destination: plan.destination, spec, overlay: null};
  navigationRuntime.routes.set(entityIdentity, activeRoute);
  installNavigationRouteOverlay(entityIdentity, activeRoute);
  const certifiedStart = navigationTraversalToWorld(samples[0].position, spec);
  state.worldPosition = [certifiedStart[0], model.viewer.camera.eye_height, certifiedStart[1]];
  if (navigationRuntime.statusElement) {
    navigationRuntime.statusElement.classList.remove("planning");
    navigationRuntime.statusElement.classList.add("active");
    navigationRuntime.statusElement.dataset.entity = entityIdentity;
    navigationRuntime.statusElement.dataset.kernel = kernelIdentity;
    navigationRuntime.statusElement.dataset.cells = String(plan.cells);
    navigationRuntime.statusElement.dataset.waypoints = String(plan.waypoints);
    navigationRuntime.statusElement.dataset.traversalChart = "linear-to-nonlinear-piecewise";
    navigationRuntime.statusElement.dataset.collisionCertified = String(collisionCertified);
    navigationRuntime.statusElement.dataset.propulsionSpeed = String(model.navigation.traversal.speed);
    navigationRuntime.statusElement.dataset.plannerThread = "dedicated-worker";
    navigationRuntime.statusElement.textContent =
      `auto-locate · ${kernel.descriptor.label} · ${plan.cells} cells · ${plan.waypoints} waypoints`;
  }
  return {entity: entityIdentity, kernel: kernelIdentity, cells: plan.cells,
    waypoints: plan.waypoints, samples: samples.length, plannerThread: "dedicated-worker"};
}

function navigationWaypointQueue(entityIdentity) {
  if (!navigationRuntime.waypointQueues.has(entityIdentity)) {
    navigationRuntime.waypointQueues.set(entityIdentity, {
      waypoints: [], planning: false, current: null, pauseUntil: 0,
      presenceReady: true, presenceToken: 0, arrivals: 0
    });
  }
  return navigationRuntime.waypointQueues.get(entityIdentity);
}

function navigationQueueDepth(entityIdentity) {
  const queue = navigationRuntime.waypointQueues.get(entityIdentity);
  return (queue?.waypoints.length || 0) + (navigationRuntime.routes.has(entityIdentity) ? 1 : 0) +
    (queue?.planning ? 1 : 0) + (queue?.pauseUntil ? 1 : 0);
}

function updateNavigationQueueStatus(entityIdentity) {
  if (!navigationRuntime.statusElement) return;
  navigationRuntime.statusElement.dataset.queueDepth = String(navigationQueueDepth(entityIdentity));
  navigationRuntime.statusElement.dataset.queuedWaypoints = String(
    navigationRuntime.waypointQueues.get(entityIdentity)?.waypoints.length || 0);
}

function reportNavigationFailure(entityIdentity, error) {
  if (error.message === "navigation planning was interrupted") return;
  navigationRuntime.error = error.message;
  if (navigationRuntime.statusElement) {
    navigationRuntime.statusElement.classList.remove("active", "planning", "presence");
    navigationRuntime.statusElement.dataset.collisionCertified = "false";
    navigationRuntime.statusElement.textContent = `auto-locate failed · ${error.message}`;
  }
  shaderViewer.readout.textContent = `auto-locate failed: ${error.message}`;
}

async function advanceNavigationWaypointQueue(entityIdentity) {
  const queue = navigationWaypointQueue(entityIdentity);
  if (queue.planning || queue.pauseUntil || navigationRuntime.routes.has(entityIdentity) ||
      !queue.waypoints.length) return;
  const waypoint = queue.waypoints.shift();
  queue.planning = true;
  queue.current = waypoint;
  updateNavigationQueueStatus(entityIdentity);
  try {
    let route = null, lastError = null;
    for (let candidateIndex = 0; candidateIndex < waypoint.candidates.length; candidateIndex += 1) {
      const candidate = waypoint.candidates[candidateIndex];
      try {
        route = await locateEntity(entityIdentity, candidate);
        waypoint.destination = [...candidate];
        waypoint.candidateIndex = candidateIndex;
        break;
      } catch (error) {
        if (error.message === "navigation planning was interrupted") throw error;
        lastError = error;
      }
    }
    if (!route) throw new Error(
      `no navigable endpoint candidate (${waypoint.candidates.length} tried) · ${lastError?.message || "unknown planner failure"}`);
    shaderViewer.readout.textContent = `${shaderViewer.backend} · auto-locate · ` +
      `${route.cells} cells · ${route.waypoints} waypoints · ${queue.waypoints.length} queued` +
      `${waypoint.candidateIndex ? ` · endpoint fallback ${waypoint.candidateIndex+1}/${waypoint.candidates.length}` : ""}`;
  } catch (error) {
    reportNavigationFailure(entityIdentity, error);
  } finally {
    queue.planning = false;
    updateNavigationQueueStatus(entityIdentity);
    if (!navigationRuntime.routes.has(entityIdentity) && !queue.pauseUntil && queue.waypoints.length) {
      void advanceNavigationWaypointQueue(entityIdentity);
    }
  }
}

function enqueueEntityWaypoint(entityIdentity, destination, candidates = [destination]) {
  if (!entityState.has(entityIdentity)) throw new Error(`unknown navigation entity ${entityIdentity}`);
  const queue = navigationWaypointQueue(entityIdentity);
  const waypoint = {
    identity: `navigation-waypoint:${++navigationRuntime.waypointSequence}`,
    destination: [Number(destination[0]), Number(destination[1])],
    candidates: candidates.map(point => [Number(point[0]), Number(point[1])]),
    candidateIndex: 0,
    queuedAt: performance.now()
  };
  queue.waypoints.push(waypoint);
  updateNavigationQueueStatus(entityIdentity);
  void advanceNavigationWaypointQueue(entityIdentity);
  return {identity: waypoint.identity, entity: entityIdentity,
    destination: [...waypoint.destination], queueDepth: navigationQueueDepth(entityIdentity)};
}

function navigationOpeningPoint(box, opening, inset) {
  const [cx, cz] = box.center, [hx, hz] = box.half_extent;
  const offset = Number(opening.offset || 0);
  let local;
  if (opening.side === "south") local=[offset,-hz+inset];
  else if (opening.side === "north") local=[offset,hz-inset];
  else if (opening.side === "west") local=[-hx+inset,offset];
  else local=[hx-inset,offset];
  const yaw=boxYawDegrees(box)*Math.PI/180,cosine=Math.cos(yaw),sine=Math.sin(yaw);
  return [cx+local[0]*cosine-local[1]*sine,cz+local[0]*sine+local[1]*cosine];
}

function navigationEndpointCandidates(clickedGeometry, clickedWorldPoint) {
  const candidates = [[...clickedWorldPoint]];
  if (!clickedGeometry) return candidates;
  candidates.push([...clickedGeometry.center]);
  const standOff = Number(model.navigation.grid.clearance) +
    Number(clickedGeometry.wall_thickness || 0.04) * 0.5 + 0.025;
  (clickedGeometry.openings || []).forEach(opening => {
    candidates.push(navigationOpeningPoint(clickedGeometry, opening, standOff));
    candidates.push(navigationOpeningPoint(clickedGeometry, opening, -standOff));
  });
  const unique = [];
  candidates.forEach(point => {
    if (!unique.some(other => Math.hypot(point[0]-other[0], point[1]-other[1]) < 1e-6)) unique.push(point);
  });
  return unique;
}

function beginNavigationPresence(entityIdentity, route) {
  const queue = navigationWaypointQueue(entityIdentity);
  const pauseMilliseconds = Number(model.navigation.waypoints?.presence_pause_seconds || 0.85) * 1000;
  const token = ++queue.presenceToken;
  const holds = [];
  const detail = {
    entity: entityIdentity, waypoint: queue.current?.identity || null,
    destination: [...route.destination], arrivedAt: performance.now(),
    queueRemaining: queue.waypoints.length,
    hold(promise) { holds.push(Promise.resolve(promise)); }
  };
  queue.arrivals += 1;
  queue.pauseUntil = performance.now() + pauseMilliseconds;
  queue.presenceReady = false;
  window.dispatchEvent(new CustomEvent(
    model.navigation.waypoints?.presence_event || "abstract-ui:navigation-presence", {detail}));
  navigationRuntime.presenceHooks.forEach(hook => {
    try { detail.hold(hook(detail)); }
    catch (error) { console.error("navigation presence hook failed", error); }
  });
  Promise.allSettled(holds).then(() => {
    if (queue.presenceToken === token) queue.presenceReady = true;
  });
  if (!holds.length) queue.presenceReady = true;
  if (navigationRuntime.statusElement) {
    navigationRuntime.statusElement.classList.remove("active", "planning");
    navigationRuntime.statusElement.classList.add("presence");
    navigationRuntime.statusElement.dataset.entity = entityIdentity;
    navigationRuntime.statusElement.dataset.presenceWaypoint = detail.waypoint || "";
    navigationRuntime.statusElement.textContent =
      `presence pause · ${queue.waypoints.length} waypoint${queue.waypoints.length === 1 ? "" : "s"} queued`;
  }
  updateNavigationQueueStatus(entityIdentity);
}

function updateNavigationPresencePauses(now) {
  navigationRuntime.waypointQueues.forEach((queue, entityIdentity) => {
    if (!queue.pauseUntil || now < queue.pauseUntil || !queue.presenceReady) return;
    queue.pauseUntil = 0;
    queue.current = null;
    if (navigationRuntime.statusElement?.dataset.entity === entityIdentity) {
      navigationRuntime.statusElement.classList.remove("presence");
      navigationRuntime.statusElement.textContent = queue.waypoints.length
        ? `presence complete · continuing to ${queue.waypoints.length} queued waypoint${queue.waypoints.length === 1 ? "" : "s"}`
        : "auto-locate · arrived · presence hooks complete";
    }
    updateNavigationQueueStatus(entityIdentity);
    void advanceNavigationWaypointQueue(entityIdentity);
  });
}

function updateEntityNavigation(dt) {
  navigationRuntime.routes.forEach((route, entityIdentity) => {
    const state = entityState.get(entityIdentity);
    if (!state || !route.samples.length) {
      clearNavigationRouteOverlay(entityIdentity);
      navigationRuntime.routes.delete(entityIdentity); return;
    }
    route.distance = Math.min(route.total, route.distance + route.speed * dt);
    let index = 0;
    while (index < route.samples.length - 2 && route.samples[index + 1].distance < route.distance) index += 1;
    const left = route.samples[index], right = route.samples[Math.min(route.samples.length - 1, index + 1)];
    const span = Math.max(1e-9, right.distance - left.distance);
    const t = Math.max(0, Math.min(1, (route.distance - left.distance) / span));
    const traversalPosition = [left.position[0] + (right.position[0]-left.position[0])*t,
      left.position[1] + (right.position[1]-left.position[1])*t];
    const position = navigationTraversalToWorld(traversalPosition, route.spec);
    const quaternion = quaternionSlerp(left.quaternion, right.quaternion, t);
    const facing = [1 - 2*quaternion[1]*quaternion[1], -2*quaternion[1]*quaternion[3]];
    const previous = state.worldPosition || [position[0], model.viewer.camera.eye_height, position[1]];
    state.worldPosition = [position[0], previous[1], position[1]];
    state.velocity = dt > 0 ? [(state.worldPosition[0]-previous[0])/dt, 0,
      (state.worldPosition[2]-previous[2])/dt] : [0,0,0];
    state.facing = [facing[0], facing[1], 0];
    navigationRuntime.lastQuaternion.set(entityIdentity, quaternion);
    if (navigationRuntime.statusElement &&
        navigationRuntime.statusElement.dataset.entity === entityIdentity) {
      navigationRuntime.statusElement.dataset.quaternion = quaternion.map(value => value.toFixed(6)).join(",");
      navigationRuntime.statusElement.dataset.progress = route.total > 0 ?
        (route.distance / route.total).toFixed(4) : "1.0000";
    }
    updateNavigationRouteOverlay(route, route.total > 0 ? route.distance/route.total : 1);
    if (entityIdentity === viewportControls.policy?.actor) {
      viewportControls.position = [...state.worldPosition];
      viewportControls.yaw = Math.atan2(facing[1], facing[0]);
    }
    if (route.distance >= route.total) {
      clearNavigationRouteOverlay(entityIdentity);
      navigationRuntime.routes.delete(entityIdentity);
      state.velocity = [0,0,0];
      beginNavigationPresence(entityIdentity, route);
    }
  });
  updateNavigationPresencePauses(performance.now());
}

async function autoLocateFromMapClick(event) {
  if (!model.navigation || (event.button != null && event.button !== 0)) return;
  try {
    if (!shaderViewer.mapElement) return;
    const clickedIdentity = event.target.closest("[data-node-id]")?.dataset.nodeId;
    const clickedGeometry = shaderViewer.geometry.find(box => box.identity === clickedIdentity);
    const localClick = viewportToElementPoint(
      [Number(event.clientX), Number(event.clientY)], shaderViewer.mapElement);
    const clickedWorldPoint = documentToWorldPoint(localClick);
    const candidates = navigationEndpointCandidates(clickedGeometry, clickedWorldPoint);
    const destination = candidates[0];
    const actor = viewportControls.policy?.actor;
    if (!actor) throw new Error("viewport control policy has no navigation actor");
    const waypoint = enqueueEntityWaypoint(actor, destination, candidates);
    shaderViewer.readout.textContent = `${shaderViewer.backend} · waypoint queued · depth ${waypoint.queueDepth}`;
  } catch (error) {
    reportNavigationFailure(viewportControls.policy?.actor, error);
  }
}

async function initializeNavigationRuntime() {
  await Promise.all((model.navigation?.kernels || []).map(installNavigationKernel));
  await initializeNavigationPlannerWorker();
  if (navigationRuntime.statusElement) navigationRuntime.statusElement.textContent =
    `${model.navigation.kernels.length} assembly kernel ready in navigation worker · ` +
    "click the top-down map to auto-locate";
  globalThis.abstractUINavigation = Object.freeze({
    installAssemblyKernel: installNavigationKernel,
    assignKernel: assignNavigationKernel,
    locate: locateEntity,
    enqueueWaypoint: enqueueEntityWaypoint,
    cancel: cancelEntityNavigation,
    onPresence(callback) {
      if (typeof callback !== "function") throw new TypeError("presence hook must be a function");
      navigationRuntime.presenceHooks.add(callback);
      return () => navigationRuntime.presenceHooks.delete(callback);
    },
    queuedWaypoints(entityIdentity) {
      return (navigationRuntime.waypointQueues.get(entityIdentity)?.waypoints || [])
        .map(waypoint => ({...waypoint, destination: [...waypoint.destination]}));
    },
    assignments: () => Object.fromEntries(navigationRuntime.assignments),
    activeRoutes: () => [...navigationRuntime.routes.keys()]
  });
}

navigationRuntime.ready = initializeNavigationRuntime().catch(error => {
  navigationRuntime.error = error.message; console.error(error); throw error;
});

const controlFocus = {
  policy: model.control_focus,
  mode: model.control_focus?.initial || "game",
  previousMode: null,
  dialogue: null,
  projectedPosition: [0, 0]
};

const hotbarState = {
  model: model.hotbar,
  inventory: model.inventory,
  activeSlot: model.hotbar?.active_slot || null
};
const toolModeState = {
  byTool: new Map((model.tools||[]).filter(tool=>tool.modes?.length).map(tool=>[
    tool.identity,tool.default_mode||tool.modes[0].name])),
  button: null
};
const secondaryActionState = {
  down: false, source: null, startedAt: 0, position: null
};
const primaryActionState = {
  down: false, source: null, targetIdentity: null
};
const depthMapRuntime={brushRadius:.72,step:.16,minimumHeight:.02,maximumHeight:1.8};
const projectileAttractorRuntime = {
  active: false, strength: 0, effectiveRadius: 0, members: new Set(),
  absorbed: 0, forceEpsilon: .035, softening: .18, baseStrength: .012,
  growthPerSecond: .42, targetStrength: 1.15, absorptionRadius: .52
};

const toolDialogueState = {element: null, tool: null, target: null};
const placementState = {
  policy: model.placement, payload: null, selectedRecipe: null,
  offsets: {x: 0, y: 0, z: 0, yaw: 0}, sequence: 0, statusElement: null,
  focusedIdentity: null, hoverIdentity: null, focusBaseline: null, focusElement: null
};

const portalRuntime = {
  splats: [],
  graph: {identity: model.placement?.portal_contract?.backing_graph ||
      `${model.placement?.identity || model.identity}/port-graphs/default`,
    sigma: 4.0, traversalSpeed: 3.2, nodes: [], edges: []},
  transits: new Map(), lastPositions: new Map(), cooldowns: new Map(),
  cooldownMilliseconds: 240
};

const livingEditPersistence = {
  dirty: new Set(),
  cookieName: `abstractui_edits_${[...model.identity].reduce(
    (hash, character) => ((hash * 31 + character.charCodeAt(0)) >>> 0), 2166136261).toString(16)}`,
  storageKey: `abstractui:edits:${model.identity}`,
  backend: "none"
};

function colorVector(value) {
  const match = /^#([0-9a-f]{6})$/i.exec(value || "");
  if (!match) return [0.5, 0.5, 0.5];
  const number = Number.parseInt(match[1], 16);
  return [((number >> 16) & 255) / 255, ((number >> 8) & 255) / 255, (number & 255) / 255];
}

function celestialState() {
  const date=new Date();
  const solarHours=date.getHours()+date.getMinutes()/60+date.getSeconds()/3600;
  const angle=(solarHours/24)*Math.PI*2-Math.PI*.5;
  const sunDirection=normalized3([Math.cos(angle)*.82,Math.sin(angle),-Math.cos(angle)*.35]);
  const moonDirection=sunDirection.map(value=>-value);
  const sunElevation=Math.max(0,sunDirection[1]),moonElevation=Math.max(0,moonDirection[1]);
  const key=sunElevation>=moonElevation?"sun":"moon";
  return {solarHours,sunDirection,moonDirection,sunElevation,moonElevation,key,
    sunColor:colorVector(model.appearance.colors.sun),moonColor:colorVector(model.appearance.colors.moon),
    dayZenith:colorVector(model.appearance.colors["sky-day"]),
    nightZenith:colorVector(model.appearance.colors["sky-night"]),
    horizon:colorVector(model.appearance.colors["sky-horizon"])};
}

function directionInCameraSpace(direction,cameraFacing) {
  const forward=normalized3(cameraFacing),right=normalized3(cross3(forward,[0,1,0]));
  const up=normalized3(cross3(right,forward));
  return [direction[0]*right[0]+direction[1]*right[1]+direction[2]*right[2],
    direction[0]*up[0]+direction[1]*up[1]+direction[2]*up[2],
    direction[0]*forward[0]+direction[1]*forward[1]+direction[2]*forward[2]];
}

function positionInCameraSpace(position,cameraPosition,cameraFacing) {
  const relative=position.map((value,index)=>value-cameraPosition[index]);
  return directionInCameraSpace(relative,cameraFacing);
}

function publishCelestialStatus(state){
  const readout=shaderViewer.celestialStatus;if(!readout)return;
  readout.dataset.key=state.key;
  const direction=state.key==="sun"?state.sunDirection:state.moonDirection;
  readout.textContent=`${state.key} key · local solar ${state.solarHours.toFixed(2)}h · direction ${direction.map(value=>value.toFixed(2)).join(", ")}`;
}

function updateCelestialLighting(gl, program, cameraFacing, state) {
  const uniform=name=>gl.getUniformLocation(program,name);
  const bands=musicRoomRuntime.bands,beat=musicRoomRuntime.level;
  const keyDirection=state.key==="sun"?state.sunDirection:state.moonDirection;
  const natural=state.key==="sun"?state.sunColor:state.moonColor;
  const keyColor=natural.map((value,index)=>Math.min(1,value*.82+[bands[0],bands[1],bands[2]][index]*.55));
  let location=uniform("uKeyLightDirection");
  if(location!==null)gl.uniform3fv(location,keyDirection);
  location=uniform("uAmbientLight");
  if(location!==null)gl.uniform1f(location,(state.key==="sun"?.42:.24)+beat*.14);
  location=uniform("uLightColor");if(location!==null)gl.uniform3fv(location,keyColor);
  location=uniform("uSkyColor");if(location!==null)gl.uniform3fv(location,
    state.key==="sun"?state.dayZenith:state.nightZenith);
  const sunView=directionInCameraSpace(state.sunDirection,cameraFacing).map(value=>value*64);
  const moonView=directionInCameraSpace(state.moonDirection,cameraFacing).map(value=>value*64);
  const room=model.music_room?.room,center=room?.center||[0,0];
  const camera=shaderViewer.cameraPosition||[0,1.6,0];
  const musicLights=[[center[0]-1,1.0,center[1]],[center[0]+1,1.65,center[1]],
    [center[0],2.25,center[1]-1]].map(point=>positionInCameraSpace(point,camera,cameraFacing));
  const celestialPositions=[...sunView,...moonView];
  location=uniform("uNumLights");if(location!==null)gl.uniform1i(location,5);
  location=uniform("uLightPos[0]");if(location!==null)gl.uniform3fv(location,[...celestialPositions,...musicLights.flat()]);
  location=uniform("uLightColor[0]");if(location!==null)gl.uniform3fv(location,
    [...state.sunColor,...state.moonColor,0.08,0.82,1.0,1.0,0.08,0.58,1.0,0.78,0.08]);
  location=uniform("uLightIntensity[0]");if(location!==null)gl.uniform1fv(location,
    [6800*state.sunElevation,1050*state.moonElevation,8+72*bands[0],8+72*bands[1],8+72*bands[2]]);
  location=uniform("uLightGroupId[0]");if(location!==null)gl.uniform1iv(location,new Int32Array([-2,-3,-4,-5,-6]));
  location=uniform("uExposure");if(location!==null)gl.uniform1f(location,.72/(1+beat*.55));
  location=uniform("uShadowMatrices[0]");if(location!==null&&shaderViewer.shadowMatrix)
    gl.uniformMatrix4fv(location,false,shaderViewer.shadowMatrix);
  location=uniform("uShadowLightCount");if(location!==null)gl.uniform1i(location,shaderViewer.shadow?.layers||0);
  location=uniform("uShadowEnabled");if(location!==null)gl.uniform1i(location,shaderViewer.shadow?1:0);
  if(shaderViewer.shadow){gl.activeTexture(gl.TEXTURE12);gl.bindTexture(gl.TEXTURE_2D_ARRAY,shaderViewer.shadow.texture);}
  publishCelestialStatus(state);
}

function drawSkyHalfDome(gl,width,height,cameraFacing,state){
  if(!shaderViewer.skyProgram)return;
  gl.useProgram(shaderViewer.skyProgram);gl.disable(gl.DEPTH_TEST);gl.disable(gl.BLEND);
  const locations=shaderViewer.skyLocations;
  gl.uniform2f(locations.uResolution,width,height);gl.uniform3fv(locations.uCameraFacing,cameraFacing);
  gl.uniform3fv(locations.uSunDirection,state.sunDirection);gl.uniform3fv(locations.uMoonDirection,state.moonDirection);
  gl.uniform3fv(locations.uDayZenith,state.dayZenith);gl.uniform3fv(locations.uNightZenith,state.nightZenith);
  gl.uniform3fv(locations.uHorizonColor,state.horizon);gl.uniform3fv(locations.uSunColor,state.sunColor);
  gl.uniform3fv(locations.uMoonColor,state.moonColor);gl.drawArrays(gl.TRIANGLES,0,3);
  gl.enable(gl.DEPTH_TEST);gl.clear(gl.DEPTH_BUFFER_BIT);
}

function compileShader(gl, kind, source) {
  const shader = gl.createShader(kind);
  gl.shaderSource(shader, source); gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || "shader compilation failed");
  }
  return shader;
}

function linkViewerProgram(gl, choice) {
  const program = gl.createProgram();
  gl.attachShader(program, compileShader(gl, gl.VERTEX_SHADER, choice.vertex_source));
  gl.attachShader(program, compileShader(gl, gl.FRAGMENT_SHADER, choice.fragment_source));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) || "shader link failed");
  }
  return program;
}

const SHADOW_VERTEX_SHADER=`#version 300 es
precision highp float;
layout(location=0) in vec3 aPosition;
uniform mat4 uLightViewProjection;
void main(){gl_Position=uLightViewProjection*vec4(aPosition,1.0);}`;
const SHADOW_FRAGMENT_SHADER=`#version 300 es
precision highp float;
void main(){}`;
const CAMERA_DEPTH_VERTEX_SHADER=`#version 300 es
precision highp float;
layout(location=0) in vec3 aPosition;
uniform vec2 uResolution;
uniform vec3 uCameraPosition;
uniform vec3 uCameraFacing;
void main(){
  vec3 forward=normalize(uCameraFacing),right=normalize(cross(forward,vec3(0,1,0)));
  vec3 cameraUp=normalize(cross(right,forward)),relative=aPosition-uCameraPosition;
  vec3 view=vec3(dot(relative,right),dot(relative,cameraUp),dot(relative,forward));
  float aspect=uResolution.x/max(1.0,uResolution.y),tangentHalfFov=.70,nearPlane=.04,farPlane=128.0;
  float clipZ=((farPlane+nearPlane)/(farPlane-nearPlane))*view.z-
    (2.0*farPlane*nearPlane)/(farPlane-nearPlane);
  gl_Position=vec4(view.x/(tangentHalfFov*aspect),view.y/tangentHalfFov,clipZ,view.z);
}`;

function multiplyMat4(a,b){
  const out=new Float32Array(16);
  for(let column=0;column<4;column+=1)for(let row=0;row<4;row+=1){
    let sum=0;for(let lane=0;lane<4;lane+=1)sum+=a[lane*4+row]*b[column*4+lane];
    out[column*4+row]=sum;
  }
  return out;
}

function lookAtMat4(eye,target,up){
  const z=normalized3(eye.map((value,index)=>value-target[index]));
  const x=normalized3(cross3(up,z)),y=cross3(z,x);
  return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,
    -x.reduce((s,v,i)=>s+v*eye[i],0),-y.reduce((s,v,i)=>s+v*eye[i],0),
    -z.reduce((s,v,i)=>s+v*eye[i],0),1]);
}

function orthoMat4(left,right,bottom,top,near,far){
  return new Float32Array([2/(right-left),0,0,0,0,2/(top-bottom),0,0,0,0,-2/(far-near),0,
    -(right+left)/(right-left),-(top+bottom)/(top-bottom),-(far+near)/(far-near),1]);
}

function initializeShadowPass(gl){
  const program=gl.createProgram();
  gl.attachShader(program,compileShader(gl,gl.VERTEX_SHADER,SHADOW_VERTEX_SHADER));
  gl.attachShader(program,compileShader(gl,gl.FRAGMENT_SHADER,SHADOW_FRAGMENT_SHADER));
  gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(program)||"shadow shader link failed");
  const size=1024,layers=5,texture=gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D_ARRAY,texture);
  gl.texImage3D(gl.TEXTURE_2D_ARRAY,0,gl.DEPTH_COMPONENT24,size,size,layers,0,
    gl.DEPTH_COMPONENT,gl.UNSIGNED_INT,null);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  const framebuffer=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,framebuffer);
  gl.framebufferTextureLayer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,texture,0,0);
  gl.drawBuffers([gl.NONE]);gl.readBuffer(gl.NONE);
  if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)
    throw new Error("shadow framebuffer is incomplete");
  gl.bindFramebuffer(gl.FRAMEBUFFER,null);
  shaderViewer.shadow={program,texture,framebuffer,size,layers,
    matrices:new Float32Array(layers*16),signatures:new Array(layers).fill(null),
    geometryRevision:-1,matrixLocation:gl.getUniformLocation(program,"uLightViewProjection")};
}

function initializeCameraDepthPass(gl){
  const program=gl.createProgram();
  gl.attachShader(program,compileShader(gl,gl.VERTEX_SHADER,CAMERA_DEPTH_VERTEX_SHADER));
  gl.attachShader(program,compileShader(gl,gl.FRAGMENT_SHADER,SHADOW_FRAGMENT_SHADER));
  gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(program)||"camera depth shader link failed");
  const texture=gl.createTexture(),framebuffer=gl.createFramebuffer();
  shaderViewer.cameraDepth={program,texture,framebuffer,width:0,height:0,format:"DEPTH_COMPONENT24",
    locations:{uResolution:gl.getUniformLocation(program,"uResolution"),
      uCameraPosition:gl.getUniformLocation(program,"uCameraPosition"),
      uCameraFacing:gl.getUniformLocation(program,"uCameraFacing")}};
}

function renderCameraDepthPass(gl,width,height,cameraPosition,cameraFacing){
  const depth=shaderViewer.cameraDepth;if(!depth||!shaderViewer.vao)return;
  if(depth.width!==width||depth.height!==height){
    depth.width=width;depth.height=height;gl.bindTexture(gl.TEXTURE_2D,depth.texture);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.DEPTH_COMPONENT24,width,height,0,gl.DEPTH_COMPONENT,gl.UNSIGNED_INT,null);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER,depth.framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.TEXTURE_2D,depth.texture,0);
    gl.drawBuffers([gl.NONE]);gl.readBuffer(gl.NONE);
    if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)
      throw new Error("camera depth framebuffer is incomplete");
  }else gl.bindFramebuffer(gl.FRAMEBUFFER,depth.framebuffer);
  gl.viewport(0,0,width,height);gl.clear(gl.DEPTH_BUFFER_BIT);gl.useProgram(depth.program);
  gl.uniform2f(depth.locations.uResolution,width,height);
  gl.uniform3fv(depth.locations.uCameraPosition,cameraPosition);
  gl.uniform3fv(depth.locations.uCameraFacing,cameraFacing);
  drawSceneMeshes(gl);
  gl.bindFramebuffer(gl.FRAMEBUFFER,null);
}

function shadowLightDescriptors(celestial){
  const envelope=shaderViewer.geometry.find(box=>box.kind==="world-envelope")||shaderViewer.geometry[0];
  const worldCenter=[envelope?.center?.[0]||0,2,envelope?.center?.[1]||0];
  const worldSpan=Math.max(12,Number(envelope?.half_extent?.[0]||12),Number(envelope?.half_extent?.[1]||12));
  const room=model.music_room?.room,center=room?.center||[0,0];
  const roomTarget=[center[0],Math.min(1.2,Number(room?.height||2.4)*.5),center[1]];
  const pointLights=[[center[0]-1,1.0,center[1]],[center[0]+1,1.65,center[1]],
    [center[0],2.25,center[1]-1]];
  return [celestial.sunDirection,celestial.moonDirection].map(direction=>({
    eye:worldCenter.map((value,index)=>value+direction[index]*worldSpan*1.8),
    target:worldCenter,span:worldSpan,near:.1,far:worldSpan*4.5
  })).concat(pointLights.map(eye=>({eye,target:roomTarget,span:Math.max(3,Number(room?.height||2.4)*2),
    near:.03,far:Math.max(8,Number(room?.height||2.4)*5)})));
}

function renderShadowPass(gl,celestial){
  const shadow=shaderViewer.shadow;if(!shadow||!shaderViewer.vao)return null;
  const geometryChanged=shadow.geometryRevision!==shaderViewer.revision;
  shadowLightDescriptors(celestial).slice(0,shadow.layers).forEach((light,layer)=>{
    const signature=[...light.eye,...light.target,light.span,light.near,light.far].join("|");
    const view=lookAtMat4(light.eye,light.target,
      Math.abs(normalized3(light.eye.map((v,i)=>v-light.target[i]))[1])>.96?[0,0,1]:[0,1,0]);
    const matrix=multiplyMat4(orthoMat4(-light.span,light.span,-light.span,light.span,
      light.near,light.far),view);
    shadow.matrices.set(matrix,layer*16);
    if(!geometryChanged&&shadow.signatures[layer]===signature)return;
    gl.bindFramebuffer(gl.FRAMEBUFFER,shadow.framebuffer);
    gl.framebufferTextureLayer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,shadow.texture,0,layer);
    gl.viewport(0,0,shadow.size,shadow.size);gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.useProgram(shadow.program);gl.uniformMatrix4fv(shadow.matrixLocation,false,matrix);
    gl.bindVertexArray(shaderViewer.vao);gl.drawArrays(gl.TRIANGLES,0,shaderViewer.vertexCount);
    shadow.signatures[layer]=signature;
  });
  shadow.geometryRevision=shaderViewer.revision;gl.bindFramebuffer(gl.FRAMEBUFFER,null);
  shaderViewer.shadowMatrix=shadow.matrices;return shadow.matrices;
}

function createScalarTexture(gl, unit, values) {
  const texture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, values.length, 1, 0,
    gl.RED, gl.FLOAT, new Float32Array(values));
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return texture;
}

function createIdentityTexture(gl, unit, target, rgba) {
  const texture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(target, texture);
  gl.texImage3D(target, 0, gl.RGBA8, 1, 1, 1, 0, gl.RGBA,
    gl.UNSIGNED_BYTE, new Uint8Array(rgba));
  gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(target, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  return texture;
}

function livingMapMaterialColors() {
  const unique = [];
  const priority = ["portal-in", "portal-out", "body-shell-glass"]
    .map(role => model.appearance.colors[role]).filter(Boolean);
  [...priority, ...Object.entries(model.appearance.colors)
    .filter(([role]) => !["portal-in", "portal-out"].includes(role))
    .map(([, value]) => value)].forEach(value => {
    const color = colorVector(value);
    if (!unique.some(candidate => candidate.every((item, index) =>
      Math.abs(item - color[index]) < 1e-6))) unique.push(color);
  });
  return unique.slice(0, 16);
}

function configurePluckPhongResources(gl, program, choice) {
  const colors = livingMapMaterialColors();
  const pbr = [], phong = [], enamel = [], texstack = [];
  colors.forEach((color, colorIndex) => {
    const records = model.material_catalog?.records || [];
    let material = colorIndex < 2 ? null : records.reduce((closest, candidate) => {
      const albedo = candidate.pbr.slice(0, 3);
      const distance = albedo.reduce((sum, value, index) =>
        sum + (value - color[index]) ** 2, 0);
      return closest === null || distance < closest.distance ? {candidate, distance} : closest;
    }, null)?.candidate;
    if(colorIndex===2){const plastic=records.find(record=>record.identity==="acrylic_pmma");
      material=plastic||material;}
    if (material) {
      const materialPbr=[...material.pbr],materialPhong=[...material.phong];
      if(colorIndex===2){materialPbr.splice(0,8,.18,.52,.64,.16,0,.72,1.586,.30);
        materialPhong.splice(0,8,.055,.62,96,.018,.035,.12,.15,0);}
      pbr.push(...materialPbr); phong.push(...materialPhong);
      enamel.push(...material.enamel); texstack.push(...material.texture_stack);
    } else {
      pbr.push(...color, 0.78, 0.0, 0.0, 1.5, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
      phong.push(0.06, 0.12, 18.0, 0.0,
        color[0] * 0.35, color[1] * 0.35, color[2] * 0.35, 0.0);
      enamel.push(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
      texstack.push(-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
  });
  const valuesByMember = {pbr, phong, enamel, texstack};
  const textures = [];
  (choice.resource_bindings || []).forEach(binding => {
    const unit = Number(binding.recommended_texture_unit);
    textures.push(createScalarTexture(gl, unit, valuesByMember[binding.member_name]));
    const location = gl.getUniformLocation(program, binding.sampler);
    if (location !== null) gl.uniform1i(location, unit);
  });
  const identityFeeds = [
    ["uEmitUv", 0, gl.TEXTURE_2D_ARRAY, [0, 255, 128, 255]],
    ["uColorUv", 1, gl.TEXTURE_2D_ARRAY, [0, 0, 0, 0]],
    ["uDepthUv", 2, gl.TEXTURE_2D_ARRAY, [0, 0, 0, 0]],
    ["uRemitUv", 3, gl.TEXTURE_2D_ARRAY, [0, 0, 0, 0]],
    ["uFieldVolume", 4, gl.TEXTURE_3D, [0, 0, 0, 0]],
  ];
  identityFeeds.forEach(([name, unit, target, rgba]) => {
    textures.push(createIdentityTexture(gl, unit, target, rgba));
    const location = gl.getUniformLocation(program, name);
    if (location !== null) gl.uniform1i(location, unit);
  });
  const uniform = name => gl.getUniformLocation(program, name);
  let location = uniform("uMaterialCount");
  if (location !== null) gl.uniform1i(location, colors.length);
  location = uniform("uMaterialColors[0]");
  if (location !== null) gl.uniform3fv(location, colors.flat());
  location = uniform("uNumLights"); if (location !== null) gl.uniform1i(location, 2);
  location = uniform("uLightPos[0]"); if (location !== null) gl.uniform3fv(location, [0,48,0, 0,-48,0]);
  location = uniform("uLightColor[0]"); if (location !== null) gl.uniform3fv(location, [1,.92,.72, .62,.76,1]);
  location = uniform("uLightIntensity[0]"); if (location !== null) gl.uniform1fv(location, [6800,1050]);
  location = uniform("uLightGroupId[0]"); if (location !== null) gl.uniform1iv(location, new Int32Array([-2,-3]));
  location = uniform("uLightCalibration"); if (location !== null) gl.uniform1f(location, 1.0);
  location = uniform("uCatCcmMatrix");
  if (location !== null) gl.uniformMatrix3fv(location, false,
    [1,0,0, 0,1,0, 0,0,1]);
  location = uniform("uEnableSpecular"); if (location !== null) gl.uniform1i(location, 1);
  location = uniform("uEnableEmissionDirect"); if (location !== null) gl.uniform1i(location, 0);
  location = uniform("uRenderPass"); if (location !== null) gl.uniform1i(location, 0);
  location = uniform("uFieldGain"); if (location !== null) gl.uniform1f(location, 0.0);
  location = uniform("uExposure"); if (location !== null) gl.uniform1f(location, 0.72);
  location = uniform("uShadowMaps"); if (location !== null) gl.uniform1i(location, 12);
  location = uniform("uShadowTexelSize"); if (location !== null) gl.uniform2f(location, 1/1024, 1/1024);
  location = uniform("uShadowBias"); if (location !== null) gl.uniform1f(location, 0.0022);
  shaderViewer.shaderResources.set(choice.identity, textures);
}

function activateViewportShader(choiceIdentity) {
  const gl = shaderViewer.gl;
  const choice = VIEWPORT_SHADER_CHOICES.find(item => item.identity === choiceIdentity);
  if (!gl || !choice) throw new Error(`unknown viewport shader ${choiceIdentity}`);
  let program = shaderViewer.shaderPrograms.get(choice.identity);
  if (!program) {
    program = linkViewerProgram(gl, choice);
    shaderViewer.shaderPrograms.set(choice.identity, program);
  }
  gl.useProgram(program);
  shaderViewer.program = program;
  shaderViewer.shaderChoice = choice.identity;
  shaderViewer.locations = {};
  ["uResolution", "uCameraPosition", "uCameraFacing", "uSkyColor", "uLightColor",
   "uKeyLightDirection", "uAmbientLight", "uWorldTileSize", "uWorldTileMajorEvery",
   "uWorldTileStrength", "uHeadlightLeft", "uHeadlightRight", "uHeadlightForward",
   "uHeadlightActive", "uTailLightLeft", "uTailLightRight", "uVehicleRearward",
   "uTailLightActive", "uBrakeLightActive", "uRenderPass"]
    .forEach(name => shaderViewer.locations[name] = gl.getUniformLocation(program, name));
  if (choice.adapter === "living-map-camera+palette-material-textures" &&
      !shaderViewer.shaderResources.has(choice.identity)) {
    configurePluckPhongResources(gl, program, choice);
  }
  const sky = colorVector(model.appearance.colors.sky);
  let location = shaderViewer.locations.uSkyColor;
  if (location !== null) gl.uniform3fv(location, sky);
  location = shaderViewer.locations.uLightColor;
  if (location !== null && choice.identity === "living-map-default") {
    gl.uniform3fv(location, colorVector(model.appearance.colors.sun));
  }
  const cues=model.vehicle_slot?.vehicles?.[0]?.configuration?.presentation||{};
  location=shaderViewer.locations.uWorldTileSize;if(location!==null)
    gl.uniform1f(location,Number(cues.world_tile_size||.5));
  location=shaderViewer.locations.uWorldTileMajorEvery;if(location!==null)
    gl.uniform1f(location,Number(cues.world_tile_major_every||4));
  location=shaderViewer.locations.uWorldTileStrength;if(location!==null)
    gl.uniform1f(location,Number(cues.world_tile_strength||.2));
  shaderViewer.backend = `WebGL2 · ${choice.label}`;
  if (shaderViewer.shaderSelect) shaderViewer.shaderSelect.value = choice.identity;
}

function buildExtrudedBoxMesh(geometry, colors) {
  const vertices = [];
  const spans = [];
  const semanticPartSpans = [];
  const colliders = [];
  const worldObjectIndex = new Map((model.world?.object_order || [])
    .map((identity, index) => [identity, index]));
  const faces = [
    [[4,5,6,4,6,7],[0,0,1]], [[1,0,3,1,3,2],[0,0,-1]],
    [[5,1,2,5,2,6],[1,0,0]], [[0,4,7,0,7,3],[-1,0,0]],
    [[7,6,2,7,2,3],[0,1,0]], [[0,1,5,0,5,4],[0,-1,0]]
  ];
  function prism(centerX, centerZ, halfX, halfZ, bottom, top, color,
                 objectIdentity, partIdentity, role, openingIdentity=null,
                 collidable=false) {
    const firstVertex = vertices.length / 9;
    const corners = [
      [centerX-halfX,bottom,centerZ-halfZ], [centerX+halfX,bottom,centerZ-halfZ],
      [centerX+halfX,top,centerZ-halfZ], [centerX-halfX,top,centerZ-halfZ],
      [centerX-halfX,bottom,centerZ+halfZ], [centerX+halfX,bottom,centerZ+halfZ],
      [centerX+halfX,top,centerZ+halfZ], [centerX-halfX,top,centerZ+halfZ]
    ];
    faces.forEach(([indices, normal]) => indices.forEach(index => {
      vertices.push(...corners[index], ...normal, ...color);
    }));
    semanticPartSpans.push({
      identity: partIdentity, objectIdentity, role, openingIdentity,
      runtimePartId: turingWorld.partRuntimeId(partIdentity),
      runtimeObjectId: turingWorld.objectRuntimeId(objectIdentity),
      firstVertex, vertexCount: vertices.length / 9 - firstVertex,
      primitive: "box", revision: shaderViewer.revision
    });
    if (collidable) colliders.push({
      identity: partIdentity, objectIdentity, role, openingIdentity,
      runtimePartId: turingWorld.partRuntimeId(partIdentity),
      minimum: [centerX-halfX, bottom, centerZ-halfZ],
      maximum: [centerX+halfX, top, centerZ+halfZ],
    });
  }
  function heightFieldPrism(box, centerX, centerZ, halfX, halfZ, bottom, color, boxIndex) {
    const firstVertex=vertices.length/9,partIdentity=`${box.identity}/surface:gradient`;
    const surface=box.surface,origin=surface.origin,gradient=surface.gradient;
    const height=(x,z)=>origin[1]+gradient[0]*(x-origin[0])+gradient[1]*(z-origin[2]);
    const corners=[
      [centerX-halfX,bottom,centerZ-halfZ],[centerX+halfX,bottom,centerZ-halfZ],
      [centerX+halfX,height(centerX+halfX,centerZ-halfZ),centerZ-halfZ],
      [centerX-halfX,height(centerX-halfX,centerZ-halfZ),centerZ-halfZ],
      [centerX-halfX,bottom,centerZ+halfZ],[centerX+halfX,bottom,centerZ+halfZ],
      [centerX+halfX,height(centerX+halfX,centerZ+halfZ),centerZ+halfZ],
      [centerX-halfX,height(centerX-halfX,centerZ+halfZ),centerZ+halfZ]
    ];
    const triangles=[4,5,6,4,6,7, 1,0,3,1,3,2, 5,1,2,5,2,6,
      0,4,7,0,7,3, 7,6,2,7,2,3, 0,1,5,0,5,4];
    for(let index=0;index<triangles.length;index+=3){
      const a=corners[triangles[index]],b=corners[triangles[index+1]],c=corners[triangles[index+2]];
      const ab=b.map((value,axis)=>value-a[axis]),ac=c.map((value,axis)=>value-a[axis]);
      const cross=[ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]];
      const length=Math.max(1e-9,Math.hypot(...cross)),normal=cross.map(value=>value/length);
      [a,b,c].forEach(point=>vertices.push(...point,...normal,...color));
    }
    const runtimePartId=turingWorld.partRuntimeId(partIdentity);
    semanticPartSpans.push({identity:partIdentity,objectIdentity:box.identity,role:"contact-surface",
      openingIdentity:null,runtimePartId,runtimeObjectId:turingWorld.objectRuntimeId(box.identity),
      firstVertex,vertexCount:vertices.length/9-firstVertex,primitive:"height-field-prism",revision:shaderViewer.revision});
    colliders.push({identity:partIdentity,objectIdentity:box.identity,role:"contact-surface",
      runtimePartId,minimum:[centerX-halfX,bottom,centerZ-halfZ],
      maximum:[centerX+halfX,box.height+bottom,centerZ+halfZ],
      surface:{...surface,origin:[...surface.origin],gradient:[...surface.gradient],
        bounds:{...surface.bounds}}});
    spans.push({identity:box.identity,kind:box.kind,boxIndex,firstVertex,
      vertexCount:vertices.length/9-firstVertex,worldObjectIndex:worldObjectIndex.get(box.identity),
      runtimeObjectId:turingWorld.objectRuntimeId(box.identity),
      semanticParts:semanticPartSpans.slice(-1),revision:shaderViewer.revision});
  }
  function sampledHeightFieldPrism(box,bottom,color,boxIndex){
    const firstVertex=vertices.length/9,partIdentity=`${box.identity}/surface:sampled-terrain`,surface=box.surface,
      [columns,rows]=surface.resolution,[cellX,cellZ]=surface.cell_size,[originX,,originZ]=surface.origin,
      point=(column,row)=>[originX+column*cellX,Number(surface.heights[row*columns+column]),originZ+row*cellZ],
      emitTriangle=(a,b,c,triangleColor)=>{const ab=b.map((value,axis)=>value-a[axis]),ac=c.map((value,axis)=>value-a[axis]),
        cross=[ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]],
        length=Math.max(1e-9,Math.hypot(...cross)),normal=cross.map(value=>value/length);
        [a,b,c].forEach(p=>vertices.push(...p,...normal,...triangleColor));};
    for(let row=0;row<rows-1;row+=1)for(let column=0;column<columns-1;column+=1){
      const p00=point(column,row),p10=point(column+1,row),p01=point(column,row+1),p11=point(column+1,row+1);
      const meanHeight=(p00[1]+p10[1]+p01[1]+p11[1])*.25,
        level=Math.max(0,Math.min(1,(meanHeight-.02)/Math.max(.01,Number(box.height)-.02))),
        checker=(column+row)%2,cellColor=color.map((value,index)=>Math.max(0,Math.min(1,
          value*(checker ? .9 : .68)+level*([.18,.13,.05][index]))));
      emitTriangle(p00,p11,p10,cellColor);emitTriangle(p00,p01,p11,cellColor);
    }
    const runtimePartId=turingWorld.partRuntimeId(partIdentity),domain=surface.domain;
    semanticPartSpans.push({identity:partIdentity,objectIdentity:box.identity,role:"contact-surface",
      openingIdentity:null,runtimePartId,runtimeObjectId:turingWorld.objectRuntimeId(box.identity),
      firstVertex,vertexCount:vertices.length/9-firstVertex,primitive:"sampled-height-field-prism",revision:shaderViewer.revision});
    colliders.push({identity:partIdentity,objectIdentity:box.identity,role:"contact-surface",runtimePartId,
      minimum:[domain.minimum_x,bottom,domain.minimum_z],maximum:[domain.maximum_x,box.height,domain.maximum_z],
      surface:{...surface,origin:[...surface.origin],cell_size:[...surface.cell_size],
        resolution:[...surface.resolution],heights:[...surface.heights],domain:{...surface.domain}}});
    spans.push({identity:box.identity,kind:box.kind,boxIndex,firstVertex,
      vertexCount:vertices.length/9-firstVertex,worldObjectIndex:worldObjectIndex.get(box.identity),
      runtimeObjectId:turingWorld.objectRuntimeId(box.identity),semanticParts:semanticPartSpans.slice(-1),
      revision:shaderViewer.revision});
  }
  function sphere(centerX, centerY, centerZ, radius, color, objectIdentity,
                  partIdentity, boxIndex) {
    const firstVertex = vertices.length / 9;
    const point = (latitude, longitude) => {
      const theta = Math.PI * latitude / 8;
      const phi = Math.PI * 2 * longitude / 12;
      const normal = [Math.sin(theta) * Math.cos(phi), Math.cos(theta),
        Math.sin(theta) * Math.sin(phi)];
      return {position: [centerX + radius * normal[0], centerY + radius * normal[1],
        centerZ + radius * normal[2]], normal};
    };
    const emit = value => vertices.push(...value.position, ...value.normal, ...color);
    for (let latitude = 0; latitude < 8; latitude += 1) {
      for (let longitude = 0; longitude < 12; longitude += 1) {
        const a = point(latitude, longitude), b = point(latitude + 1, longitude);
        const c = point(latitude + 1, longitude + 1), d = point(latitude, longitude + 1);
        [a, c, b, a, d, c].forEach(emit);
      }
    }
    semanticPartSpans.push({
      identity: partIdentity, objectIdentity, role: "projectile-body",
      openingIdentity: null, runtimePartId: turingWorld.partRuntimeId(partIdentity),
      runtimeObjectId: turingWorld.objectRuntimeId(objectIdentity), firstVertex,
      vertexCount: vertices.length / 9 - firstVertex, primitive: "sphere",
      revision: shaderViewer.revision
    });
    colliders.push({identity: partIdentity, objectIdentity, role: "projectile-body",
      openingIdentity: null, runtimePartId: turingWorld.partRuntimeId(partIdentity),
      minimum: [centerX-radius, centerY-radius, centerZ-radius],
      maximum: [centerX+radius, centerY+radius, centerZ+radius]});
    spans.push({identity: objectIdentity, kind: "physics-ball", boxIndex,
      firstVertex, vertexCount: vertices.length / 9 - firstVertex,
      worldObjectIndex: worldObjectIndex.get(objectIdentity),
      runtimeObjectId: turingWorld.objectRuntimeId(objectIdentity),
      semanticParts: semanticPartSpans.slice(-1), revision: shaderViewer.revision});
  }
  function vehicleWheel(box,boxIndex){
    const firstVertex=vertices.length/9,state=box.wheel_state,segments=24;
    const nominalRadius=Number(state.radius),pressureRatio=Math.max(.28,Math.min(1.65,Number(
      state.tirePressurePa||155000)/Math.max(1,Number(state.referenceTirePressurePa||155000)))),
      carcassScale=.94+.06*Math.sqrt(pressureRatio),flatten=Math.max(0,.11*(1-pressureRatio)),
      radius=nominalRadius*carcassScale,halfWidth=Number(state.width)*.5*(1+Math.max(-.08,Math.min(.22,(1-pressureRatio)*.24))),
      deformedRadius=(angle,value)=>value*(1-flatten*Math.pow(Math.max(0,-Math.sin(angle)),4));
    const spin=Number(state.spin||0),steer=Number(state.steer||0);
    const rotation=state.chassisRotation||[0,0,0],roll=rotation[0],yaw=rotation[1],pitch=rotation[2];
    const cr=Math.cos(roll),sr=Math.sin(roll),cp=Math.cos(pitch),sp=Math.sin(pitch),
      cy=Math.cos(yaw),sy=Math.sin(yaw),cs=Math.cos(spin),ss=Math.sin(spin),
      ct=Math.cos(steer),st=Math.sin(steer),origin=state.chassisPosition,center=state.localCenter;
    const bodyTransform=point=>{
      const rolled=[point[0],point[1]*cr-point[2]*sr,point[1]*sr+point[2]*cr];
      const pitched=[rolled[0]*cp-rolled[1]*sp,rolled[0]*sp+rolled[1]*cp,rolled[2]];
      return[pitched[0]*cy-pitched[2]*sy,pitched[1],pitched[0]*sy+pitched[2]*cy];
    };
    const transform=(point,translate=true)=>{
      const spun=[point[0]*cs-point[1]*ss,point[0]*ss+point[1]*cs,point[2]];
      const steered=[spun[0]*ct+spun[2]*st,spun[1],-spun[0]*st+spun[2]*ct],turned=bodyTransform(steered);
      if(!translate)return turned;
      const chassisCenter=bodyTransform(center);
      return[origin[0]+chassisCenter[0]+turned[0],origin[1]+chassisCenter[1]+turned[1],
        origin[2]+chassisCenter[2]+turned[2]];
    },transformNoSpin=(point,translate=true)=>{
      const steered=[point[0]*ct+point[2]*st,point[1],-point[0]*st+point[2]*ct],turned=bodyTransform(steered);
      if(!translate)return turned;const chassisCenter=bodyTransform(center);
      return[origin[0]+chassisCenter[0]+turned[0],origin[1]+chassisCenter[1]+turned[1],
        origin[2]+chassisCenter[2]+turned[2]];
    };
    const rubber=colorVector(state.tireColor||box.appearance?.face_color||colors.line),
      tread=colorVector(state.treadColor||box.appearance?.tread_color||colors.line),
      rim=colorVector(state.rimColor||colors["rollbar-silver"]||colors.active),
      rotor=colorVector(colors["suspension-yellow"]||colors.active),
      silver=colorVector(colors["rollbar-silver"]||colors.line),
      black=colorVector(colors["drivetrain-black"]||colors.line);
    const emit=(point,normal,color)=>vertices.push(...transform(point),...transform(normal,false),...color);
    const agricultural=state.carcassProfile==="thin-agricultural-steel-disc",
      profile=agricultural?[[-halfWidth,radius*.90],[-halfWidth*.78,radius*.985],[0,radius],
        [halfWidth*.78,radius*.985],[halfWidth,radius*.90]]:[[-halfWidth,radius*.82],[-halfWidth*.72,radius*.96],[0,radius],
        [halfWidth*.72,radius*.96],[halfWidth,radius*.82]],rimRadius=Number(state.rimRadius||radius*.55),
      hubRadius=rimRadius*.24;
    for(let ring=0;ring<profile.length-1;ring+=1)for(let segment=0;segment<segments;segment+=1){
      const a=segment*Math.PI*2/segments,b=(segment+1)*Math.PI*2/segments,
        [za,ra]=profile[ring],[zb,rb]=profile[ring+1],shoulder=(rb-ra)/(zb-za||1),
        dra=deformedRadius(a,ra),drb=deformedRadius(b,ra),drc=deformedRadius(b,rb),drd=deformedRadius(a,rb),
        pa=[Math.cos(a)*dra,Math.sin(a)*dra,za],pb=[Math.cos(b)*drb,Math.sin(b)*drb,za],
        pc=[Math.cos(b)*drc,Math.sin(b)*drc,zb],pd=[Math.cos(a)*drd,Math.sin(a)*drd,zb],
        na=[Math.cos(a),Math.sin(a),-shoulder],nb=[Math.cos(b),Math.sin(b),-shoulder],
        treadColor=ring===2&&segment%2?rim:tread;
      [[pa,na],[pb,nb],[pc,nb],[pa,na],[pc,nb],[pd,na]].forEach(([p,n])=>emit(p,n,treadColor));
    }
    for(const side of [-1,1])for(let segment=0;segment<segments;segment+=1){
      const a=segment*Math.PI*2/segments,b=(segment+1)*Math.PI*2/segments,z=side*halfWidth,
        sideRadiusA=deformedRadius(a,radius*.82),sideRadiusB=deformedRadius(b,radius*.82),
        outerA=[Math.cos(a)*sideRadiusA,Math.sin(a)*sideRadiusA,z],outerB=[Math.cos(b)*sideRadiusB,Math.sin(b)*sideRadiusB,z],
        innerA=[Math.cos(a)*rimRadius,Math.sin(a)*rimRadius,z],innerB=[Math.cos(b)*rimRadius,Math.sin(b)*rimRadius,z],n=[0,0,side];
      [[outerA,n],[outerB,n],[innerB,n],[outerA,n],[innerB,n],[innerA,n]].forEach(([p,q])=>emit(p,q,rubber));
    }
    const tireVertexCount=vertices.length/9-firstVertex;
    for(const side of [-1,1])for(let segment=0;segment<segments;segment+=1){
      const a=segment*Math.PI*2/segments,b=(segment+1)*Math.PI*2/segments,n=[0,0,side],
        rotorOuter=rimRadius*.72,rotorInner=rimRadius*.32,rz=side*(halfWidth+.001),
        roA=[Math.cos(a)*rotorOuter,Math.sin(a)*rotorOuter,rz],roB=[Math.cos(b)*rotorOuter,Math.sin(b)*rotorOuter,rz],
        riA=[Math.cos(a)*rotorInner,Math.sin(a)*rotorInner,rz],riB=[Math.cos(b)*rotorInner,Math.sin(b)*rotorInner,rz];
      [[roA,n],[roB,n],[riB,n],[roA,n],[riB,n],[riA,n]].forEach(([p,q])=>emit(p,q,rotor));
    }
    if(agricultural)for(const side of [-1,1])for(let segment=0;segment<segments;segment+=1){
      const a=segment*Math.PI*2/segments,b=(segment+1)*Math.PI*2/segments,z=side*(halfWidth+.003),n=[0,0,side],
        inner=hubRadius*.72,outer=rimRadius*.985,
        ia=[Math.cos(a)*inner,Math.sin(a)*inner,z],ib=[Math.cos(b)*inner,Math.sin(b)*inner,z],
        oa=[Math.cos(a)*outer,Math.sin(a)*outer,z],ob=[Math.cos(b)*outer,Math.sin(b)*outer,z];
      [[ia,n],[ib,n],[ob,n],[ia,n],[ob,n],[oa,n]].forEach(([p,q])=>emit(p,q,rim));
    }
    if(!agricultural)for(const side of [-1,1])for(let spoke=0;spoke<6;spoke+=1){
      const angle=spoke*Math.PI/3,width=.10,z=side*(halfWidth+.003),n=[0,0,side],
        a=angle-width,b=angle+width,inner=hubRadius,outer=rimRadius*.94,
        points=[[Math.cos(a)*inner,Math.sin(a)*inner,z],[Math.cos(b)*inner,Math.sin(b)*inner,z],
          [Math.cos(b)*outer,Math.sin(b)*outer,z],[Math.cos(a)*outer,Math.sin(a)*outer,z]];
      [[points[0],n],[points[1],n],[points[2],n],[points[0],n],[points[2],n],[points[3],n]]
        .forEach(([p,q])=>emit(p,q,rim));
    }
    for(const side of [-1,1])for(let segment=0;segment<segments;segment+=1){
      const a=segment*Math.PI*2/segments,b=(segment+1)*Math.PI*2/segments,z=side*(halfWidth+.004),n=[0,0,side],
        pa=[Math.cos(a)*hubRadius,Math.sin(a)*hubRadius,z],pb=[Math.cos(b)*hubRadius,Math.sin(b)*hubRadius,z];
      [[pa,n],[pb,n],[[0,0,z],n]].forEach(([p,q])=>emit(p,q,rim));
    }
    // The rotor above uses the hub/wheel spin transform. The caliper, bearing
    // outer race, upright and steering arm below use the knuckle transform and
    // therefore steer but do not rotate with the hub.
    const inboard=String(state.name||"").includes("left")?1:-1,
      fixedEmit=(point,normal,shade)=>vertices.push(...transformNoSpin(point),...transformNoSpin(normal,false),...shade),
      fixedCuboid=(c,h,shade)=>{const points=[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
        .map(sign=>sign.map((value,index)=>c[index]+value*h[index]));
        faces.forEach(([indices,normal])=>indices.forEach(index=>fixedEmit(points[index],normal,shade)));},
      fixedTube=(start,end,tubeRadius,shade,sides=10)=>{const tubeAxis=normalized3(end.map((value,index)=>value-start[index])),
        helper=Math.abs(tubeAxis[1])<.85?[0,1,0]:[1,0,0],tu=normalized3([tubeAxis[1]*helper[2]-tubeAxis[2]*helper[1],
          tubeAxis[2]*helper[0]-tubeAxis[0]*helper[2],tubeAxis[0]*helper[1]-tubeAxis[1]*helper[0]]),
        tv=normalized3([tubeAxis[1]*tu[2]-tubeAxis[2]*tu[1],tubeAxis[2]*tu[0]-tubeAxis[0]*tu[2],tubeAxis[0]*tu[1]-tubeAxis[1]*tu[0]]);
        for(let segment=0;segment<sides;segment+=1){const aa=segment*Math.PI*2/sides,ab=(segment+1)*Math.PI*2/sides,
          ra=tu.map((value,index)=>value*Math.cos(aa)+tv[index]*Math.sin(aa)),rb=tu.map((value,index)=>value*Math.cos(ab)+tv[index]*Math.sin(ab)),
          p0=start.map((value,index)=>value+ra[index]*tubeRadius),p1=start.map((value,index)=>value+rb[index]*tubeRadius),
          p2=end.map((value,index)=>value+rb[index]*tubeRadius),p3=end.map((value,index)=>value+ra[index]*tubeRadius);
          [[p0,ra],[p1,rb],[p2,rb],[p0,ra],[p2,rb],[p3,ra]].forEach(([p,n])=>fixedEmit(p,n,shade));}},
      bearingZ=inboard*(halfWidth+.010),knuckleZ=inboard*(halfWidth+.045),cvEndZ=inboard*(halfWidth+.095);
    fixedTube([0,0,bearingZ-inboard*.012],[0,0,bearingZ+inboard*.012],rimRadius*.30,silver,16);
    fixedTube([0,-rimRadius*.48,knuckleZ],[0,rimRadius*.48,knuckleZ],rimRadius*.105,rotor,10);
    fixedTube([0,0,knuckleZ],[0,0,cvEndZ],rimRadius*.24,black,14);
    fixedTube([0,0,cvEndZ],[0,0,cvEndZ+inboard*.030],rimRadius*.16,silver,12);
    fixedTube([0,rimRadius*.18,knuckleZ],[rimRadius*.52,rimRadius*.28,knuckleZ],rimRadius*.075,rotor,9);
    fixedCuboid([rimRadius*.16,rimRadius*.58,knuckleZ-inboard*.010],
      [rimRadius*.20,rimRadius*.16,.022],rotor);
    const tireIdentity=`${box.identity}/tire:pneumatic-carcass`,wheelIdentity=`${box.identity}/wheel:rim-hub-brake`;
    semanticPartSpans.push({identity:tireIdentity,objectIdentity:box.identity,role:"pneumatic-tire",
      openingIdentity:null,runtimePartId:turingWorld.partRuntimeId(tireIdentity),runtimeObjectId:0,
      firstVertex,vertexCount:tireVertexCount,primitive:"balloon-tire-sidewall-and-tread",revision:shaderViewer.revision});
    semanticPartSpans.push({identity:wheelIdentity,objectIdentity:box.identity,role:"wheel-rim-hub-brake",
      openingIdentity:null,runtimePartId:turingWorld.partRuntimeId(wheelIdentity),runtimeObjectId:0,
      firstVertex:firstVertex+tireVertexCount,vertexCount:vertices.length/9-firstVertex-tireVertexCount,
      primitive:"heavy-six-spoke-wheel-hub-and-brake",revision:shaderViewer.revision});
    const wheelEndIdentity=`${box.identity}/wheel-end:cv-bearing-knuckle-rotor-caliper`;
    semanticPartSpans.push({identity:wheelEndIdentity,objectIdentity:box.identity,role:"wheel-end-load-path",
      openingIdentity:null,runtimePartId:turingWorld.partRuntimeId(wheelEndIdentity),runtimeObjectId:0,
      firstVertex:firstVertex+tireVertexCount,vertexCount:vertices.length/9-firstVertex-tireVertexCount,
      primitive:"cv-bearing-knuckle-steering-arm-rotor-caliper",revision:shaderViewer.revision});
    spans.push({identity:box.identity,kind:"vehicle-wheel",boxIndex,firstVertex,
      vertexCount:vertices.length/9-firstVertex,worldObjectIndex:undefined,runtimeObjectId:0,
      semanticParts:semanticPartSpans.slice(-3),revision:shaderViewer.revision});
  }
  function vehicleLink(box,boxIndex){
    const firstVertex=vertices.length/9,state=box.link_state,segments=8,
      rotation=state.chassisRotation||[0,0,0],roll=rotation[0],yaw=rotation[1],pitch=rotation[2],
      cr=Math.cos(roll),sr=Math.sin(roll),cp=Math.cos(pitch),sp=Math.sin(pitch),
      cy=Math.cos(yaw),sy=Math.sin(yaw),origin=state.chassisPosition||[0,0,0];
    const bodyTransform=point=>{
      const rolled=[point[0],point[1]*cr-point[2]*sr,point[1]*sr+point[2]*cr],
        pitched=[rolled[0]*cp-rolled[1]*sp,rolled[0]*sp+rolled[1]*cp,rolled[2]];
      return[pitched[0]*cy-pitched[2]*sy,pitched[1],pitched[0]*sy+pitched[2]*cy];
    };
    const world=point=>{const transformed=bodyTransform(point);return origin.map((value,index)=>value+transformed[index]);},
      a=world(state.localA),b=world(state.localB),axis=normalized3(b.map((value,index)=>value-a[index])),
      helper=Math.abs(axis[1])<.85?[0,1,0]:[1,0,0],
      u=normalized3([axis[1]*helper[2]-axis[2]*helper[1],axis[2]*helper[0]-axis[0]*helper[2],
        axis[0]*helper[1]-axis[1]*helper[0]]),
      v=normalized3([axis[1]*u[2]-axis[2]*u[1],axis[2]*u[0]-axis[0]*u[2],
        axis[0]*u[1]-axis[1]*u[0]]),radius=Number(state.radius||.012),
      color=colorVector(box.appearance?.face_color||colors.line),
      emit=(point,normal,shade=color)=>vertices.push(...point,...normal,...shade),
      tube=(start,end,tubeRadius,shade=color,sideCount=segments)=>{
        const tubeAxis=normalized3(end.map((value,index)=>value-start[index])),
          tubeHelper=Math.abs(tubeAxis[1])<.85?[0,1,0]:[1,0,0],
          tubeU=normalized3([tubeAxis[1]*tubeHelper[2]-tubeAxis[2]*tubeHelper[1],
            tubeAxis[2]*tubeHelper[0]-tubeAxis[0]*tubeHelper[2],tubeAxis[0]*tubeHelper[1]-tubeAxis[1]*tubeHelper[0]]),
          tubeV=normalized3([tubeAxis[1]*tubeU[2]-tubeAxis[2]*tubeU[1],
            tubeAxis[2]*tubeU[0]-tubeAxis[0]*tubeU[2],tubeAxis[0]*tubeU[1]-tubeAxis[1]*tubeU[0]]);
        for(let segment=0;segment<sideCount;segment+=1){
          const angleA=segment*Math.PI*2/sideCount,angleB=(segment+1)*Math.PI*2/sideCount,
            radialA=tubeU.map((value,index)=>value*Math.cos(angleA)+tubeV[index]*Math.sin(angleA)),
            radialB=tubeU.map((value,index)=>value*Math.cos(angleB)+tubeV[index]*Math.sin(angleB)),
            a0=start.map((value,index)=>value+radialA[index]*tubeRadius),
            a1=start.map((value,index)=>value+radialB[index]*tubeRadius),
            b0=end.map((value,index)=>value+radialA[index]*tubeRadius),
            b1=end.map((value,index)=>value+radialB[index]*tubeRadius);
          [[a0,radialA],[b0,radialA],[b1,radialB],[a0,radialA],[b1,radialB],[a1,radialB]]
            .forEach(([point,normal])=>emit(point,normal,shade));
        }
      };
    if(box.mechanical_edge?.constraint==="spring-damper"){
      const edge=box.mechanical_edge,visual=edge.visualization||{},silver=colorVector(colors["rollbar-silver"]||colors.line),
        length=Math.max(.001,Math.hypot(...b.map((value,index)=>value-a[index]))),
        turns=Number(visual.active_turns||8),wire=Math.max(.0015,Number(visual.wire_radius_m||radius*.24)),
        springRadius=Math.max(radius*1.45,wire*2.8),steps=Math.max(48,Math.round(turns*10));
      tube(a,b,Number(visual.damper_shaft_radius_m||radius*.35),silver,8);
      let previous=null;
      for(let step=0;step<=steps;step+=1){
        const t=.08+.84*step/steps,angle=turns*Math.PI*2*step/steps,
          point=a.map((value,index)=>value+axis[index]*length*t+springRadius*(u[index]*Math.cos(angle)+v[index]*Math.sin(angle)));
        if(previous)tube(previous,point,wire,color,6);previous=point;
      }
      const preload=Math.max(0,Number(edge.static_preload_compression_m||0)),collarT=Math.min(.22,.07+preload/Math.max(.001,length)*.35),
        collarCenter=a.map((value,index)=>value+axis[index]*length*collarT),collarHalf=Math.max(.006,wire*1.4);
      tube(collarCenter.map((value,index)=>value-axis[index]*collarHalf),
        collarCenter.map((value,index)=>value+axis[index]*collarHalf),springRadius*1.12,silver,10);
    }else if(box.mechanical_edge?.routing==="relaxed-multi-segment-harness"){
      const route=(state.routeLocalPoints?.length>=2?state.routeLocalPoints:[state.localA,state.localB]).map(world);
      for(let index=1;index<route.length;index+=1)tube(route[index-1],route[index],radius,color,6);
    }else tube(a,b,radius,color,segments);
    const partIdentity=`${box.identity}/surface:member`;
    semanticPartSpans.push({identity:partIdentity,objectIdentity:box.identity,role:box.suspension_role||"mechanical-link",
      openingIdentity:null,runtimePartId:turingWorld.partRuntimeId(partIdentity),runtimeObjectId:0,
      firstVertex,vertexCount:vertices.length/9-firstVertex,primitive:"state-driven-link",revision:shaderViewer.revision});
    spans.push({identity:box.identity,kind:box.kind,boxIndex,firstVertex,
      vertexCount:vertices.length/9-firstVertex,worldObjectIndex:undefined,runtimeObjectId:0,
      semanticParts:semanticPartSpans.slice(-1),revision:shaderViewer.revision});
  }
  function vehiclePowertrainPart(box,boxIndex){
    const firstVertex=vertices.length/9,mesh=box.part_mesh||{},center=[Number(box.center[0]),Number(box.center_y),Number(box.center[1])],
      half=[Number(box.half_extent[0]),Number(box.height)*.5,Number(box.half_extent[1])],
      primary=colorVector(box.appearance?.face_color||colors[box.palette_role]||colors.line),
      silver=colorVector(colors["rollbar-silver"]||colors.line),black=colorVector(colors["drivetrain-black"]||colors.line),
      accent=colorVector(colors["engine-accent"]||colors.active),yellow=colorVector(colors["actuator-yellow"]||colors.active),
      emit=(point,normal,shade=primary)=>vertices.push(...point,...normal,...shade),
      cuboid=(c,h,shade=primary)=>{const p=[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
        .map(sign=>sign.map((value,index)=>c[index]+value*h[index]));
        faces.forEach(([indices,normal])=>indices.forEach(index=>emit(p[index],normal,shade)));},
      tube=(start,end,radius,shade=primary,sides=10)=>{const axis=normalized3(end.map((value,index)=>value-start[index])),
        helper=Math.abs(axis[1])<.85?[0,1,0]:[1,0,0],u=normalized3([axis[1]*helper[2]-axis[2]*helper[1],
          axis[2]*helper[0]-axis[0]*helper[2],axis[0]*helper[1]-axis[1]*helper[0]]),
        v=normalized3([axis[1]*u[2]-axis[2]*u[1],axis[2]*u[0]-axis[0]*u[2],axis[0]*u[1]-axis[1]*u[0]]);
        for(let side=0;side<sides;side+=1){const aa=side*Math.PI*2/sides,ab=(side+1)*Math.PI*2/sides,
          ra=u.map((value,index)=>value*Math.cos(aa)+v[index]*Math.sin(aa)),rb=u.map((value,index)=>value*Math.cos(ab)+v[index]*Math.sin(ab)),
          p0=start.map((value,index)=>value+ra[index]*radius),p1=start.map((value,index)=>value+rb[index]*radius),
          p2=end.map((value,index)=>value+rb[index]*radius),p3=end.map((value,index)=>value+ra[index]*radius);
          [[p0,ra],[p1,rb],[p2,rb],[p0,ra],[p2,rb],[p3,ra]].forEach(([point,normal])=>emit(point,normal,shade));}},
      ellipsoid=(c,r,shade=primary,rings=8,sides=14)=>{const point=(ring,side)=>{const theta=Math.PI*ring/rings,
        phi=Math.PI*2*side/sides,normal=normalized3([Math.sin(theta)*Math.cos(phi)/Math.max(1e-6,r[0]),
          Math.cos(theta)/Math.max(1e-6,r[1]),Math.sin(theta)*Math.sin(phi)/Math.max(1e-6,r[2])]);
        return{position:[c[0]+r[0]*Math.sin(theta)*Math.cos(phi),c[1]+r[1]*Math.cos(theta),
          c[2]+r[2]*Math.sin(theta)*Math.sin(phi)],normal};};
        for(let ring=0;ring<rings;ring+=1)for(let side=0;side<sides;side+=1){const a=point(ring,side),b=point(ring+1,side),
          c0=point(ring+1,side+1),d=point(ring,side+1);[a,b,c0,a,c0,d].forEach(item=>emit(item.position,item.normal,shade));}},
      ring=(c,rx,ry,wire,shade=black)=>{let previous=null;for(let step=0;step<=32;step+=1){const angle=step*Math.PI*2/32,
        point=[c[0]+rx*Math.cos(angle),c[1]+ry*Math.sin(angle),c[2]];if(previous)tube(previous,point,wire,shade,6);previous=point;}},
      annularDiscX=(c,outer,inner,halfThickness,shade=silver,sides=24)=>{
        const point=(x,radius,angle)=>[c[0]+x,c[1]+radius*Math.cos(angle),c[2]+radius*Math.sin(angle)];
        for(let side=0;side<sides;side+=1){const aa=side*Math.PI*2/sides,ab=(side+1)*Math.PI*2/sides,
          op0=point(halfThickness,outer,aa),op1=point(halfThickness,outer,ab),ip0=point(halfThickness,inner,aa),
          ip1=point(halfThickness,inner,ab),om0=point(-halfThickness,outer,aa),om1=point(-halfThickness,outer,ab),
          im0=point(-halfThickness,inner,aa),im1=point(-halfThickness,inner,ab);
          [op0,op1,ip0,op1,ip1,ip0].forEach(p=>emit(p,[1,0,0],shade));
          [om0,im0,om1,om1,im0,im1].forEach(p=>emit(p,[-1,0,0],shade));
        }};
    if(mesh.shape==="fuel-tank"){
      ellipsoid(center,[half[0],half[1],half[2]],primary,10,18);
      for(const offset of [-half[2]*.46,half[2]*.46])ring([center[0],center[1],center[2]+offset],half[0]*1.04,half[1]*1.06,.006,black);
      tube([center[0]+half[0]*.45,center[1]+half[1]*.72,center[2]+half[2]*.30],
        [center[0]+half[0]*.45,center[1]+half[1]*1.55,center[2]+half[2]*.30],.018,primary,12);
      tube([center[0]+half[0]*.45,center[1]+half[1]*1.52,center[2]+half[2]*.30],
        [center[0]+half[0]*.45,center[1]+half[1]*1.68,center[2]+half[2]*.30],.024,silver,12);
      tube([center[0]+half[0]*.80,center[1]-half[1]*.35,center[2]],
        [center[0]+half[0]*1.30,center[1]-half[1]*.35,center[2]],.009,accent,8);
    }else if(mesh.shape==="battery-pack"){
      cuboid(center,half,primary);
      for(const offset of [-.58,-.20,.20,.58])cuboid([center[0]+half[0]*offset,center[1]+half[1]*1.04,center[2]],
        [half[0]*.035,half[1]*.10,half[2]*.92],silver);
      for(const side of [-1,1]){const terminal=[center[0]+side*half[0]*.62,center[1]+half[1]*1.22,center[2]-half[2]*.50];
        tube(terminal,[terminal[0],terminal[1]+half[1]*.28,terminal[2]],.012,side<0?black:accent,10);}
      cuboid([center[0]+half[0]*1.34,center[1],center[2]],[half[0]*.22,half[1]*.76,half[2]*.78],black);
    }else if(mesh.shape==="engine-cylinder"){
      const axis=normalized3(mesh.axis||[0,1,0]),length=Math.max(.025,half[1]*1.75),start=center.map((value,index)=>value-axis[index]*length*.5),
        end=center.map((value,index)=>value+axis[index]*length*.5);
      tube(start,end,Math.max(half[0],half[2]),primary,12);
      tube(end,end.map((value,index)=>value+axis[index]*length*.18),Math.max(half[0],half[2])*1.12,accent,12);
    }else if(mesh.shape==="round-shaft"){
      const axis=normalized3(mesh.axis||[1,0,0]),span=Math.max(half[0],half[2],half[1]),
        start=center.map((value,index)=>value-axis[index]*span),end=center.map((value,index)=>value+axis[index]*span);
      tube(start,end,Math.max(.009,Math.min(half[0],half[1],half[2])),primary,12);
    }else if(mesh.shape==="throttle-body"){
      tube([center[0],center[1],center[2]-half[2]],[center[0],center[1],center[2]+half[2]],Math.max(half[0],half[1]),primary,14);
      tube([center[0]+half[0],center[1]+half[1],center[2]],[center[0]+half[0]*1.45,center[1]+half[1]*1.25,center[2]],.006,yellow,8);
    }else if(mesh.shape==="differential-brake"){
      tube([center[0]-half[0],center[1],center[2]],[center[0]+half[0],center[1],center[2]],half[2],silver,18);
      annularDiscX(center,half[2]*.96,half[2]*.30,half[0]*.96,silver,28);
      tube([center[0]-half[0]*1.08,center[1],center[2]],[center[0]+half[0]*1.08,center[1],center[2]],half[2]*.42,black,12);
      cuboid([center[0],center[1]+half[2]*.56,center[2]+half[2]*.58],
        [half[0]*1.35,half[2]*.34,half[2]*.28],primary);
    }else if(mesh.shape==="ignition-coil"){
      cuboid(center,half,black);tube([center[0],center[1]+half[1],center[2]],
        [center[0],center[1]+half[1]*1.72,center[2]],Math.max(.005,half[0]*.32),primary,10);
    }else if(mesh.shape==="distributor"){
      tube([center[0],center[1]-half[1],center[2]],[center[0],center[1]+half[1],center[2]],
        Math.max(half[0],half[2]),black,14);
      tube([center[0],center[1]+half[1]*.72,center[2]],[center[0],center[1]+half[1]*1.08,center[2]],
        Math.max(half[0],half[2])*1.18,accent,14);
    }else cuboid(center,half,primary);
    const partIdentity=`${box.identity}/surface:procedural-part`;
    semanticPartSpans.push({identity:partIdentity,objectIdentity:box.identity,role:mesh.shape||"powertrain-part",
      openingIdentity:null,runtimePartId:turingWorld.partRuntimeId(partIdentity),runtimeObjectId:0,firstVertex,
      vertexCount:vertices.length/9-firstVertex,primitive:`procedural-${mesh.shape||"part"}`,revision:shaderViewer.revision});
    rotateBoxRealization(box,firstVertex,colliders.length);
    spans.push({identity:box.identity,kind:box.kind,boxIndex,firstVertex,vertexCount:vertices.length/9-firstVertex,
      worldObjectIndex:undefined,runtimeObjectId:0,semanticParts:semanticPartSpans.slice(-1),revision:shaderViewer.revision});
  }
  function rotateBoxRealization(box, firstVertex, firstCollider) {
    const rotation=box.placement?.rotation||[0,0,0],roll=Number(rotation[0]||0)*Math.PI/180,
      yaw=boxYawDegrees(box)*Math.PI/180,pitch=Number(rotation[2]||0)*Math.PI/180;
    if(Math.abs(yaw)+Math.abs(roll)+Math.abs(pitch)<1e-9)return;
    const cr=Math.cos(roll),sr=Math.sin(roll),cp=Math.cos(pitch),sp=Math.sin(pitch),
      cyaw=Math.cos(yaw),syaw=Math.sin(yaw),cx=box.center[0],
      centerY=Number(box.center_y??(Number(box.placement?.elevation||0)+Number(box.height)*.5)),cz=box.center[1];
    const rotate=([x,y,z])=>{const r=[x,y*cr-z*sr,y*sr+z*cr],p=[r[0]*cp-r[1]*sp,r[0]*sp+r[1]*cp,r[2]];
      return[p[0]*cyaw-p[2]*syaw,p[1],p[0]*syaw+p[2]*cyaw];};
    for(let vertex=firstVertex;vertex<vertices.length/9;vertex+=1){
      const offset=vertex*9,point=rotate([vertices[offset]-cx,vertices[offset+1]-centerY,vertices[offset+2]-cz]),
        normal=rotate([vertices[offset+3],vertices[offset+4],vertices[offset+5]]);
      vertices[offset]=cx+point[0];vertices[offset+1]=centerY+point[1];vertices[offset+2]=cz+point[2];
      vertices[offset+3]=normal[0];vertices[offset+4]=normal[1];vertices[offset+5]=normal[2];
    }
    colliders.slice(firstCollider).forEach(collider=>{
      const corners=[[collider.minimum[0],collider.minimum[2]],[collider.maximum[0],collider.minimum[2]],
        [collider.maximum[0],collider.maximum[2]],[collider.minimum[0],collider.maximum[2]]]
        .map(([x,z])=>{const point=rotate([x-cx,0,z-cz]);return[cx+point[0],cz+point[2]];});
      collider.minimum[0]=Math.min(...corners.map(point=>point[0]));
      collider.maximum[0]=Math.max(...corners.map(point=>point[0]));
      collider.minimum[2]=Math.min(...corners.map(point=>point[1]));
      collider.maximum[2]=Math.max(...corners.map(point=>point[1]));
      collider.rotation=[Number(rotation[0]||0),boxYawDegrees(box),Number(rotation[2]||0)];
      if(collider.surface){
        const origin=collider.surface.origin,dx=origin[0]-cx,dz=origin[2]-cz;
        collider.surface.origin=[cx+dx*cyaw-dz*syaw,origin[1],cz+dx*syaw+dz*cyaw];
        const gradient=collider.surface.gradient;
        collider.surface.gradient=[gradient[0]*cyaw-gradient[1]*syaw,
          gradient[0]*syaw+gradient[1]*cyaw];
      }
    });
  }
  function wallSegments(box, side, halfLength, fixedCenter, horizontalAxis, color,
                        thickness, baseY, floorTop, boxTop, collidable) {
    const openings = (box.openings || []).filter(opening => opening.side === side);
    let intervals = [[-halfLength, halfLength]];
    openings.forEach(opening => {
      const openingStart = Math.max(-halfLength, opening.offset - opening.width * 0.5);
      const openingEnd = Math.min(halfLength, opening.offset + opening.width * 0.5);
      intervals = intervals.flatMap(([start, end]) => {
        if (openingEnd <= start || openingStart >= end) return [[start, end]];
        return [[start, openingStart], [openingEnd, end]].filter(pair => pair[1] - pair[0] > 1e-4);
      });
      const openingBottom = Math.max(floorTop, Math.min(boxTop,
        baseY+Number(opening.bottom ?? floorTop-baseY)));
      const openingTop = Math.min(boxTop, openingBottom + Number(opening.height));
      if (openingEnd > openingStart && openingBottom > floorTop + 1e-4) {
        const center = (openingStart + openingEnd) * 0.5;
        const half = (openingEnd - openingStart) * 0.5;
        if (horizontalAxis === "x") prism(box.center[0] + center, fixedCenter,
          half, thickness * 0.5, floorTop, openingBottom, color, box.identity,
          `${box.identity}/surface:${side}`, "opening-sill", opening.identity,
          collidable);
        else prism(fixedCenter, box.center[1] + center,
          thickness * 0.5, half, floorTop, openingBottom, color, box.identity,
          `${box.identity}/surface:${side}`, "opening-sill", opening.identity,
          collidable);
      }
      if (openingEnd > openingStart && openingTop < boxTop - 1e-4) {
        const center = (openingStart + openingEnd) * 0.5;
        const half = (openingEnd - openingStart) * 0.5;
        if (horizontalAxis === "x") prism(box.center[0] + center, fixedCenter,
          half, thickness * 0.5, openingTop, boxTop, color, box.identity,
          `${box.identity}/surface:${side}`, "opening-lintel", opening.identity,
          collidable);
        else prism(fixedCenter, box.center[1] + center,
          thickness * 0.5, half, openingTop, boxTop, color, box.identity,
          `${box.identity}/surface:${side}`, "opening-lintel", opening.identity,
          collidable);
      }
      if (opening.kind === "portal" && openingEnd > openingStart && openingTop > openingBottom) {
        const center = (openingStart + openingEnd) * 0.5;
        const half = (openingEnd - openingStart) * 0.5;
        const portalColor = colorVector(model.appearance.colors[
          opening.port_role === "out" ? "portal-out" : "portal-in"]);
        const skin = Math.min(0.008, thickness * 0.22);
        const role = opening.port_role === "out" ? "portal-out-surface" : "portal-in-surface";
        if (horizontalAxis === "x") prism(box.center[0] + center, fixedCenter,
          half, skin, openingBottom, openingTop, portalColor, box.identity,
          `${opening.identity}/surface:${opening.port_role || "in"}`, role, opening.identity, false);
        else prism(fixedCenter, box.center[1] + center,
          skin, half, openingBottom, openingTop, portalColor, box.identity,
          `${opening.identity}/surface:${opening.port_role || "in"}`, role, opening.identity, false);
      }
    });
    intervals.forEach(([start, end]) => {
      const center = (start + end) * 0.5, half = (end - start) * 0.5;
      if (horizontalAxis === "x") prism(box.center[0] + center, fixedCenter,
        half, thickness * 0.5, floorTop, boxTop, color, box.identity,
        `${box.identity}/surface:${side}`, "boundary-wall", null, collidable);
      else prism(fixedCenter, box.center[1] + center,
        thickness * 0.5, half, floorTop, boxTop, color, box.identity,
        `${box.identity}/surface:${side}`, "boundary-wall", null, collidable);
    });
  }
  geometry.forEach((box, boxIndex) => {
    if (box.placement?.custody === "inventory") return;
    const firstVertex = vertices.length / 9;
    const firstPart = semanticPartSpans.length;
    const firstCollider = colliders.length;
    const [centerX, centerZ] = box.center;
    const [halfX, halfZ] = box.half_extent;
    const baseY=Number(box.placement?.elevation || 0),boxTop=baseY+Number(box.height);
    const floorHeight = Math.min(box.height, Number(box.floor_height ?? 0.035));
    const thickness = Math.max(0.01, Math.min(halfX, halfZ,
      Number(box.appearance?.wall_thickness ?? box.wall_thickness ?? 0.04)));
    const floorColor = colorVector(box.appearance?.face_color || colors[box.palette_role] || colors.room);
    const wallColor = colorVector(box.appearance?.wall_color ||
      colors[box.wall_palette_role] || colors.line);
    if(box.geometry_mode==="vehicle-link"){
      vehicleLink(box,boxIndex);return;
    }
    if(box.geometry_mode==="vehicle-wheel"){
      vehicleWheel(box,boxIndex);return;
    }
    if(box.geometry_mode==="vehicle-powertrain-part"){
      vehiclePowertrainPart(box,boxIndex);return;
    }
    if (box.geometry_mode === "sphere") {
      sphere(centerX, Number(box.center_y ?? box.radius), centerZ,
        Number(box.radius), floorColor, box.identity,
        `${box.identity}/surface:body`, boxIndex);
      return;
    }
    if (box.geometry_mode === "height-field-prism") {
      heightFieldPrism(box,centerX,centerZ,halfX,halfZ,baseY,floorColor,boxIndex);
      rotateBoxRealization(box,firstVertex,firstCollider);
      return;
    }
    if(box.geometry_mode==="sampled-height-field-prism"){
      sampledHeightFieldPrism(box,baseY,floorColor,boxIndex);return;
    }
    if (box.geometry_mode === "solid") {
      prism(centerX, centerZ, halfX, halfZ, baseY, boxTop, floorColor,
        box.identity, `${box.identity}/surface:body`, "body", null,
        box.physics?.welded !== true && box.placement?.custody !== "preview");
      rotateBoxRealization(box,firstVertex,firstCollider);
      spans.push({identity: box.identity, kind: box.kind, boxIndex,
        firstVertex, vertexCount: vertices.length / 9 - firstVertex,
        worldObjectIndex: worldObjectIndex.get(box.identity),
        runtimeObjectId: turingWorld.objectRuntimeId(box.identity),
        semanticParts: semanticPartSpans.slice(firstPart),
        revision: shaderViewer.revision});
      return;
    }
    prism(centerX, centerZ, halfX, halfZ, baseY, baseY+floorHeight, floorColor,
      box.identity, `${box.identity}/surface:floor`, "floor");
    wallSegments(box, "south", halfX, centerZ-halfZ+thickness*0.5, "x",
      wallColor, thickness, baseY, baseY+floorHeight, boxTop, box.kind !== "world-envelope" &&
        box.placement?.custody !== "preview");
    wallSegments(box, "north", halfX, centerZ+halfZ-thickness*0.5, "x",
      wallColor, thickness, baseY, baseY+floorHeight, boxTop, box.kind !== "world-envelope" &&
        box.placement?.custody !== "preview");
    wallSegments(box, "west", halfZ, centerX-halfX+thickness*0.5, "z",
      wallColor, thickness, baseY, baseY+floorHeight, boxTop, box.kind !== "world-envelope" &&
        box.placement?.custody !== "preview");
    wallSegments(box, "east", halfZ, centerX+halfX-thickness*0.5, "z",
      wallColor, thickness, baseY, baseY+floorHeight, boxTop, box.kind !== "world-envelope" &&
        box.placement?.custody !== "preview");
    if (box.kind !== "world-envelope" &&
        box.height >= model.document_geometry.boundary_semantics.ceiling.absolute_maximum - 1e-6) {
      prism(centerX, centerZ, halfX, halfZ, boxTop-floorHeight, boxTop,
        wallColor, box.identity, `${box.identity}/surface:ceiling`, "ceiling");
    }
    rotateBoxRealization(box,firstVertex,firstCollider);
    spans.push({identity: box.identity, kind: box.kind, boxIndex,
      firstVertex, vertexCount: vertices.length / 9 - firstVertex,
      worldObjectIndex: worldObjectIndex.get(box.identity),
      runtimeObjectId: turingWorld.objectRuntimeId(box.identity),
      semanticParts: semanticPartSpans.slice(firstPart),
      revision: shaderViewer.revision});
  });
  return {mesh: new Float32Array(vertices), spans, semanticPartSpans, colliders};
}

function buildNavigationRouteMesh() {
  const vertices = [], color = colorVector(model.appearance.colors.active);
  const height = 0.058, halfWidth = 0.022;
  const emit = point => vertices.push(...point, 0, 1, 0, ...color);
  navigationRuntime.routes.forEach(route => {
    const points = route.samples.map(sample =>
      navigationTraversalToWorld(sample.position, route.spec));
    for (let index = 1; index < points.length; index += 1) {
      const left = points[index-1], right = points[index];
      const dx = right[0]-left[0], dz = right[1]-left[1];
      const length = Math.hypot(dx, dz);
      if (length < 1e-8) continue;
      const px = -dz/length*halfWidth, pz = dx/length*halfWidth;
      const a = [left[0]+px,height,left[1]+pz], b = [left[0]-px,height,left[1]-pz];
      const c = [right[0]-px,height,right[1]-pz], d = [right[0]+px,height,right[1]+pz];
      [a,b,c,a,c,d].forEach(emit);
    }
  });
  return new Float32Array(vertices);
}

function buildPortalSplatMesh() {
  const vertices = [];
  portalRuntime.splats.forEach(splat => {
    const color = colorVector(model.appearance.colors[
      splat.port_role === "out" ? "portal-out" : "portal-in"]);
    const center = splat.center.map((value, axis) => value + splat.normal[axis] * 0.008);
    const emit = point => vertices.push(...point, ...splat.normal, ...color);
    const rim = step => {
      const angle = Math.PI * 2 * step / 28;
      const horizontal = Math.cos(angle) * splat.radius;
      const vertical = Math.sin(angle) * splat.radius;
      return center.map((value, axis) => value + splat.tangent[axis] * horizontal +
        splat.bitangent[axis] * vertical);
    };
    for (let step = 0; step < 28; step += 1) {
      emit(center); emit(rim(step)); emit(rim(step + 1));
    }
  });
  return new Float32Array(vertices);
}

function portalCurvePoint(edge, t) {
  const inverse = 1 - t;
  return [0, 1, 2].map(axis => inverse ** 3 * edge.control_points[0][axis] +
    3 * inverse ** 2 * t * edge.control_points[1][axis] +
    3 * inverse * t ** 2 * edge.control_points[2][axis] +
    t ** 3 * edge.control_points[3][axis]);
}

function portalCurveTangent(edge, t) {
  const inverse = 1 - t, points = edge.control_points;
  return normalized3([0, 1, 2].map(axis =>
    3 * inverse ** 2 * (points[1][axis] - points[0][axis]) +
    6 * inverse * t * (points[2][axis] - points[1][axis]) +
    3 * t ** 2 * (points[3][axis] - points[2][axis])));
}

function quaternionNormalize(quaternion) {
  const length = Math.hypot(...quaternion) || 1;
  return quaternion.map(value => value / length);
}

function quaternionMultiply(left, right) {
  const [lx, ly, lz, lw] = left, [rx, ry, rz, rw] = right;
  return [lw * rx + lx * rw + ly * rz - lz * ry,
    lw * ry - lx * rz + ly * rw + lz * rx,
    lw * rz + lx * ry - ly * rx + lz * rw,
    lw * rw - lx * rx - ly * ry - lz * rz];
}

function quaternionBetween(from, to) {
  const dot = Math.max(-1, Math.min(1, from.reduce(
    (sum, value, axis) => sum + value * to[axis], 0)));
  if (dot < -0.9999) {
    let axis = normalized3(cross3(from, [1, 0, 0]));
    if (Math.hypot(...axis) < 0.1) axis = normalized3(cross3(from, [0, 1, 0]));
    return [axis[0], axis[1], axis[2], 0];
  }
  const axis = cross3(from, to);
  return quaternionNormalize([axis[0], axis[1], axis[2], 1 + dot]);
}

function quaternionRotate(quaternion, vector) {
  const vectorQuaternion = [vector[0], vector[1], vector[2], 0];
  const conjugate = [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]];
  return quaternionMultiply(quaternionMultiply(quaternion, vectorQuaternion), conjugate).slice(0, 3);
}

function portalCurveFrame(edge, source, t) {
  const clampedT = Math.max(0, Math.min(1, t));
  let previousT = 0, previousTangent = portalCurveTangent(edge, 0);
  let frameU = source?.tangent ? [...source.tangent] :
    normalized3(cross3(previousTangent, [0, 1, 0]));
  const axialComponent = frameU.reduce((sum, value, axis) =>
    sum + value * previousTangent[axis], 0);
  frameU = normalized3(frameU.map((value, axis) =>
    value - previousTangent[axis] * axialComponent));
  if (Math.hypot(...frameU) < 0.1) frameU = [1, 0, 0];
  let orientation = [0, 0, 0, 1];
  while (previousT < clampedT - 1e-8) {
    const nextT = Math.min(clampedT, previousT + 1 / 24);
    const tangent = portalCurveTangent(edge, nextT);
    const relaxation = quaternionBetween(previousTangent, tangent);
    orientation = quaternionNormalize(quaternionMultiply(relaxation, orientation));
    frameU = normalized3(quaternionRotate(relaxation, frameU));
    previousT = nextT;
    previousTangent = tangent;
  }
  return {tangent: previousTangent, frameU,
    frameV: normalized3(cross3(frameU, previousTangent)), orientation};
}

function portalTubeRadius(t, source, target) {
  const sourceThroat = Math.max(0.01, Number(source?.tube_throat_radius || 0.045));
  const targetThroat = Math.max(0.01, Number(target?.tube_throat_radius || 0.045));
  const throat = sourceThroat * (1 - t) + targetThroat * t, flareLength = 0.22;
  const smoothstep = value => value * value * (3 - 2 * value);
  const entry = smoothstep(Math.max(0, Math.min(1, (flareLength - t) / flareLength)));
  const exit = smoothstep(Math.max(0, Math.min(1, (t - 1 + flareLength) / flareLength)));
  return throat + (Math.max(throat, Number(source?.radius || throat)) - throat) * entry +
    (Math.max(throat, Number(target?.radius || throat)) - throat) * exit;
}

function portalApertureClass(splat) {
  return splat?.aperture_class || "person";
}

function activePortalPlacementProfile() {
  const profiles = model.placement?.portal_contract?.modes || {};
  const mode = activeToolMode()?.name || "standard";
  return {name: mode, ...(profiles[mode] || profiles.standard || {
    aperture_class: "person", aperture_scale: 1, tube_scale: 1, handle_scale: 1})};
}

function rebuildPortalGraph() {
  const graph = portalRuntime.graph;
  graph.nodes = portalRuntime.splats.map(splat => ({identity: splat.identity,
    port_role: splat.port_role, center: [...splat.center]}));
  const inputs = portalRuntime.splats.filter(splat => splat.port_role === "in");
  const outputs = portalRuntime.splats.filter(splat => splat.port_role === "out");
  graph.edges = [];
  inputs.forEach(source => {
    const candidates = outputs.filter(target =>
      portalApertureClass(target) === portalApertureClass(source)).map(target => {
      const distance = Math.hypot(...target.center.map((value, axis) => value - source.center[axis]));
      return {target, distance, raw: Math.exp(-(distance ** 2) / (2 * graph.sigma ** 2))};
    });
    const total = candidates.reduce((sum, candidate) => sum + candidate.raw, 0) || 1;
    candidates.forEach(candidate => {
      const handleScale = Math.max(1, Number(source.handle_scale || source.aperture_scale || 1),
        Number(candidate.target.handle_scale || candidate.target.aperture_scale || 1));
      const lift = Math.max(0.75, Math.min(3.2, candidate.distance * 0.28)) * handleScale;
      const sourceControl = source.center.map((value, axis) => value +
        source.normal[axis] * -lift);
      const targetControl = candidate.target.center.map((value, axis) => value +
        candidate.target.normal[axis] * -lift);
      const edge = {identity: `${graph.identity}/edges/${source.identity.split("/").at(-1)}-to-${candidate.target.identity.split("/").at(-1)}`,
        source: source.identity, target: candidate.target.identity,
        weight: candidate.raw / total, distance: candidate.distance,
        aperture_class: portalApertureClass(source), aperture_scale: handleScale,
        path_model: "relaxed-quaternion-cubic",
        control_points: [[...source.center], sourceControl, targetControl, [...candidate.target.center]]};
      let length = 0, previous = edge.control_points[0];
      for (let step = 1; step <= 20; step += 1) {
        const point = portalCurvePoint(edge, step / 20);
        length += Math.hypot(...point.map((value, axis) => value - previous[axis]));
        previous = point;
      }
      edge.length = length;
      graph.edges.push(edge);
    });
  });
  return graph;
}

function buildPortalTubeMesh() {
  const vertices = [], sides = 12, segments = 24;
  const emit = (point, normal, color) => vertices.push(...point, ...normal, ...color);
  portalRuntime.graph.edges.forEach(edge => {
    const source = portalRuntime.splats.find(splat => splat.identity === edge.source);
    const target = portalRuntime.splats.find(splat => splat.identity === edge.target);
    const inColor = colorVector(model.appearance.colors["portal-in"]);
    const outColor = colorVector(model.appearance.colors["portal-out"]);
    const rings = [];
    for (let step = 0; step <= segments; step += 1) {
      const t = step / segments, center = portalCurvePoint(edge, t);
      const frame = portalCurveFrame(edge, source, t);
      const radius = portalTubeRadius(t, source, target);
      const color = inColor.map((value, axis) => value * (1 - t) + outColor[axis] * t);
      rings.push(Array.from({length: sides}, (_, side) => {
        const angle = Math.PI * 2 * side / sides;
        const normal = normalized3(frame.frameU.map((value, axis) => value * Math.cos(angle) +
          frame.frameV[axis] * Math.sin(angle)));
        return {point: center.map((value, axis) => value + normal[axis] * radius), normal, color};
      }));
    }
    for (let step = 0; step < segments; step += 1) for (let side = 0; side < sides; side += 1) {
      const next = (side + 1) % sides;
      const a = rings[step][side], b = rings[step][next];
      const c = rings[step + 1][next], d = rings[step + 1][side];
      [a, b, c, a, c, d].forEach(vertex => emit(vertex.point, vertex.normal, vertex.color));
    }
  });
  return new Float32Array(vertices);
}

function cloneGeometryBox(box) {
  return {...box, center: [...box.center], half_extent: [...box.half_extent],
    appearance: box.appearance ? {...box.appearance} : undefined,
    placement: box.placement ? {...box.placement} : undefined,
    openings: (box.openings || []).map(opening => ({...opening}))};
}

function livingEditPayload() {
  const objects = {};
  livingEditPersistence.dirty.forEach(identity => {
    const box = shaderViewer.geometry.find(item => item.identity === identity);
    if (box) objects[identity] = {
      height: box.height,
      appearance: {...(box.appearance || {})},
      openings: (box.openings || []).map(opening => ({...opening})),
      center: [...box.center], placement: box.placement ? {...box.placement} : null,
    };
  });
  return JSON.stringify({version: model.persistence?.version || 1,
    revision: shaderViewer.revision, objects,
    portal_splats: portalRuntime.splats,
    portal_graph: {identity: portalRuntime.graph.identity, sigma: portalRuntime.graph.sigma,
      traversalSpeed: portalRuntime.graph.traversalSpeed},
    placement_recipes: Object.fromEntries((model.placement?.recipes || []).map(recipe =>
      [recipe.identity, recipe.stock])),
    tool_modes: Object.fromEntries(toolModeState.byTool),
    vehicle_hydraulics: model.vehicle_slot?.vehicles?.[0]?.chassis_leveling||null,
    vehicle_tire_pressure_target_pa:Number(vehicleRuntime.electrical?.tirePressureTargetPa||0),
    physics: Object.fromEntries(physicsRuntime.parameters)});
}

function saveLivingEdits(identity) {
  if (identity) livingEditPersistence.dirty.add(identity);
  const payload = livingEditPayload();
  let cookieStored = false;
  try {
    const encoded = encodeURIComponent(payload);
    document.cookie = `${livingEditPersistence.cookieName}=${encoded}; ` +
      "max-age=31536000; SameSite=Lax; path=/";
    cookieStored = document.cookie.split("; ").some(entry =>
      entry === `${livingEditPersistence.cookieName}=${encoded}`);
  } catch (_) {}
  try {
    localStorage.setItem(livingEditPersistence.storageKey, payload);
    livingEditPersistence.backend = cookieStored ? "cookie+local-storage" : "local-storage";
  } catch (_) {
    livingEditPersistence.backend = cookieStored ? "cookie" : "none";
  }
}

function returnLivingMapToDefaults() {
  try {
    document.cookie = `${livingEditPersistence.cookieName}=; max-age=0; ` +
      "expires=Thu, 01 Jan 1970 00:00:00 GMT; SameSite=Lax; path=/";
  } catch (_) {}
  try { localStorage.removeItem(livingEditPersistence.storageKey); } catch (_) {}
  livingEditPersistence.dirty.clear();
  livingEditPersistence.backend = "none";
  const control = document.querySelector("[data-return-defaults]");
  if (control) { control.disabled = true; control.textContent = "Restoring defaults…"; }
  location.reload();
}

function migrateLegacyPortalOpenings() {
  const legacy = [];
  shaderViewer.geometry.forEach(host => {
    const retained = [];
    (host.openings || []).forEach(opening => {
      if (opening.kind !== "portal") { retained.push(opening); index.set(opening.identity, opening); return; }
      legacy.push({host, opening});
      const transform = portalOpeningTransform(host, opening);
      const role = opening.port_role || (legacy.length % 2 ? "in" : "out");
      portalRuntime.splats.push({identity: opening.identity, port_role: role,
        port_set_identity: portalRuntime.graph.identity, backing: "probabilistic-tube-graph",
        backing_graph: portalRuntime.graph.identity, distribution: "normalized-spatial-gaussian",
        intermediary_manifold: "directed-tube-edge", center: [...transform.position],
        normal: [transform.normal[0], 0, transform.normal[1]],
        tangent: [transform.tangent[0], 0, transform.tangent[1]], bitangent: [0, 1, 0],
        radius: Math.max(0.18, Math.min(transform.width, transform.height) * 0.5),
        object_identity: host.identity, part_identity: opening.identity,
        triangle_memberships: [], division: {operation: "legacy-opening-to-radial-chart",
          domain: "triangle-barycentric-subdomains"}});
    });
    if (retained.length !== (host.openings || []).length) {
      host.openings = retained;
      livingEditPersistence.dirty.add(host.identity);
      const worldHost = model.world.objects.find(candidate => candidate.identity === host.identity);
      if (worldHost?.form) worldHost.form.openings = (worldHost.form.openings || [])
        .filter(opening => opening.kind !== "portal");
    }
  });
  placementState.sequence = Math.max(placementState.sequence, portalRuntime.splats.length);
  rebuildPortalGraph();
  return legacy.length;
}

function restoreLivingEdits() {
  let payload = null;
  try {
    const entry = document.cookie.split("; ").find(item =>
      item.startsWith(`${livingEditPersistence.cookieName}=`));
    if (entry) {
      payload = decodeURIComponent(entry.slice(entry.indexOf("=") + 1));
      livingEditPersistence.backend = "cookie";
    }
  } catch (_) {}
  if (!payload) {
    try {
      payload = localStorage.getItem(livingEditPersistence.storageKey);
      if (payload) livingEditPersistence.backend = "local-storage";
    } catch (_) {}
  }
  if (!payload) return;
  try {
    const saved = JSON.parse(payload);
    if (saved.version !== (model.persistence?.version || 1)) return;
    Object.entries(saved.objects || {}).forEach(([identity, edit]) => {
      const box = shaderViewer.geometry.find(item => item.identity === identity);
      if (!box) return;
      if (Number.isFinite(Number(edit.height))) box.height = Math.max(0.04, Number(edit.height));
      box.appearance = {...(box.appearance || {}), ...(edit.appearance || {})};
      if (Array.isArray(edit.openings)) box.openings = edit.openings.map(opening => ({...opening}));
      if (Array.isArray(edit.center) && edit.center.length === 2) box.center = edit.center.map(Number);
      if (edit.placement) box.placement = {...(box.placement || {}), ...edit.placement};
      livingEditPersistence.dirty.add(identity);
    });
    if (Array.isArray(saved.portal_splats)) portalRuntime.splats = saved.portal_splats
      .filter(splat => ["in", "out"].includes(splat.port_role))
      .map(splat => ({...splat, center: [...splat.center], normal: [...splat.normal],
        tangent: [...splat.tangent], bitangent: [...splat.bitangent]}));
    if (saved.portal_graph) {
      if (saved.portal_graph.identity) portalRuntime.graph.identity = saved.portal_graph.identity;
      if (Number(saved.portal_graph.sigma) > 0) portalRuntime.graph.sigma = Number(saved.portal_graph.sigma);
      if (Number(saved.portal_graph.traversalSpeed) > 0) portalRuntime.graph.traversalSpeed =
        Number(saved.portal_graph.traversalSpeed);
    }
    const migratedPortals = migrateLegacyPortalOpenings();
    Object.entries(saved.tool_modes||{}).forEach(([identity,mode])=>{
      const tool=model.tools?.find(item=>item.identity===identity);
      if(tool?.modes?.some(item=>item.name===mode))toolModeState.byTool.set(identity,mode);
    });
    Object.entries(saved.placement_recipes || {}).forEach(([identity, rawStock]) => {
      const recipe = model.placement?.recipes.find(item => item.identity === identity);
      const stock = Number(rawStock);
      if (!recipe || !Number.isFinite(stock)) return;
      recipe.stock = Math.max(0, Math.min(recipe.maximum_stack, stock));
      const inventoryItem = hotbarState.inventory.items.find(item =>
        item.properties?.recipe === identity);
      if (inventoryItem) inventoryItem.quantity = recipe.stock;
    });
    Object.entries(saved.physics || {}).forEach(([name, rawValue]) => {
      const descriptor = model.physics_program?.parameters.find(item => item.name === name);
      const value = Number(rawValue);
      if (!descriptor || !Number.isFinite(value)) return;
      physicsRuntime.parameters.set(name, value);
      if(name==="linear_drag"&&model.projectiles?.archetype?.physics)
        model.projectiles.archetype.physics.linear_drag=value;
      descriptor.value = value;
      const input = shaderViewer.element?.querySelector(
        `[data-physics-parameter="${CSS.escape(name)}"]`);
      if (input) input.value = String(value);
    });
    const savedVehicle = model.vehicle_slot?.vehicles?.[0];
    if (savedVehicle && saved.vehicle_hydraulics) {
      savedVehicle.chassis_leveling = {
        ...savedVehicle.chassis_leveling,
        ...saved.vehicle_hydraulics,
        manual_corner_targets_m: {
          ...savedVehicle.chassis_leveling.manual_corner_targets_m,
          ...(saved.vehicle_hydraulics.manual_corner_targets_m || {}),
        },
        programmed_poses: {
          ...(savedVehicle.chassis_leveling.programmed_poses || {}),
          ...(saved.vehicle_hydraulics.programmed_poses || {}),
        },
      };
    }
    if (savedVehicle && Number(saved.vehicle_tire_pressure_target_pa) > 0) {
      savedVehicle.tire_pressure_target_pa = Number(saved.vehicle_tire_pressure_target_pa);
    }
    shaderViewer.revision = Number(saved.revision || 0);
    model.scene_mesh.revision = shaderViewer.revision;
    refreshInventoryCounts();
    rebuildPortalGraph();
    if (migratedPortals) setPlacementStatus(
      `restored ${migratedPortals} legacy portal port${migratedPortals === 1 ? "" : "s"} as graph splats`);
  } catch (error) {
    console.warn("AbstractUI saved edits were ignored", error);
  }
}

function installSceneMesh(mesh,{publish=true}={}) {
  shaderViewer.mesh = mesh;
  shaderViewer.vertexCount = mesh.length / 9;
  if (shaderViewer.gl && shaderViewer.buffer) {
    shaderViewer.gl.bindBuffer(shaderViewer.gl.ARRAY_BUFFER, shaderViewer.buffer);
    shaderViewer.gl.bufferData(shaderViewer.gl.ARRAY_BUFFER, mesh, shaderViewer.gl.DYNAMIC_DRAW);
  }
  if (shaderViewer.softwareWasm && shaderViewer.softwareWasm.count !== shaderViewer.vertexCount) {
    shaderViewer.softwareWasm = null;
  }
  if (!shaderViewer.softwareWasm && shaderViewer.context2d && !shaderViewer.softwareWasmPending) {
    shaderViewer.softwareWasmPending = true;
    initializeSoftwareMeshWasm().catch(error => console.error(error)).finally(() => {
      shaderViewer.softwareWasmPending = false;
    });
  } else if (shaderViewer.softwareWasm) {
    for (let index = 0; index < shaderViewer.vertexCount; index += 1) {
      shaderViewer.softwareWasm.arrays.vertex_x[index] = mesh[index * 9];
      shaderViewer.softwareWasm.arrays.vertex_y[index] = mesh[index * 9 + 1];
      shaderViewer.softwareWasm.arrays.vertex_z[index] = mesh[index * 9 + 2];
    }
  }
  if(publish){markDocumentWorldSyncForGeometry();publishSceneMeshToDocument();}
}

function vehiclePresentationGeometry(){
  const useHostWheels=!shaderViewer.wheelProgram;
  return [...vehicleRuntime.frameBoxes,...vehicleRuntime.rollCageBoxes,
    ...vehicleRuntime.mechanicalLinkBoxes,...vehicleRuntime.powertrainBoxes,
    ...vehicleRuntime.bodyShellBoxes,
    ...(useHostWheels?vehicleRuntime.wheelBoxes:[])].filter(Boolean);
}

function installVehiclePresentationMesh(mesh){
  shaderViewer.vehicleMesh=mesh;shaderViewer.vehicleVertexCount=mesh.length/9;
  const gl=shaderViewer.gl;if(!gl)return;
  if(!shaderViewer.vehicleVao){
    shaderViewer.vehicleVao=gl.createVertexArray();shaderViewer.vehicleBuffer=gl.createBuffer();
    gl.bindVertexArray(shaderViewer.vehicleVao);gl.bindBuffer(gl.ARRAY_BUFFER,shaderViewer.vehicleBuffer);
    const stride=9*Float32Array.BYTES_PER_ELEMENT;
    gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,stride,0);
    gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,3,gl.FLOAT,false,stride,3*4);
    gl.enableVertexAttribArray(2);gl.vertexAttribPointer(2,3,gl.FLOAT,false,stride,6*4);
  }else gl.bindBuffer(gl.ARRAY_BUFFER,shaderViewer.vehicleBuffer);
  gl.bufferData(gl.ARRAY_BUFFER,mesh,gl.DYNAMIC_DRAW);
}

function drawSceneMeshes(gl){
  gl.bindVertexArray(shaderViewer.vao);gl.drawArrays(gl.TRIANGLES,0,shaderViewer.vertexCount);
  if(shaderViewer.vehicleVao&&shaderViewer.vehicleVertexCount){
    gl.bindVertexArray(shaderViewer.vehicleVao);gl.drawArrays(gl.TRIANGLES,0,shaderViewer.vehicleVertexCount);
  }
}

function rebuildPortableSceneMesh({dynamicOnly=false}={}) {
  const dynamicGeometry=vehiclePresentationGeometry();
  if(dynamicOnly){
    installVehiclePresentationMesh(buildExtrudedBoxMesh(dynamicGeometry,model.appearance.colors).mesh);return;
  }
  const dynamicSet=new Set(dynamicGeometry),staticGeometry=shaderViewer.geometry.filter(box=>!dynamicSet.has(box)),
    realization=buildExtrudedBoxMesh(staticGeometry,model.appearance.colors),
    dynamicRealization=buildExtrudedBoxMesh(dynamicGeometry,model.appearance.colors);
  shaderViewer.baseMesh = realization.mesh;
  const routes = buildNavigationRouteMesh();
  if(!dynamicOnly)rebuildPortalGraph();
  const portalSplats = buildPortalSplatMesh();
  const portalTubes = buildPortalTubeMesh();
  if (routes.length || portalSplats.length || portalTubes.length) {
    const combined = new Float32Array(realization.mesh.length + routes.length +
      portalTubes.length + portalSplats.length);
    combined.set(realization.mesh);
    combined.set(routes, realization.mesh.length);
    combined.set(portalTubes, realization.mesh.length + routes.length);
    combined.set(portalSplats, realization.mesh.length + routes.length + portalTubes.length);
    realization.mesh = combined;
  }
  const originalIndex=box=>shaderViewer.geometry.indexOf(box),staticVertices=realization.mesh.length/9;
  shaderViewer.identitySpans=[...realization.spans.map((span,index)=>({...span,boxIndex:originalIndex(staticGeometry[index])})),
    ...dynamicRealization.spans.map((span,index)=>({...span,boxIndex:originalIndex(dynamicGeometry[index]),
      firstVertex:span.firstVertex+staticVertices}))];
  shaderViewer.semanticPartSpans=[...realization.semanticPartSpans,...dynamicRealization.semanticPartSpans];
  shaderViewer.colliders=realization.colliders;
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"colliders",
    colliders:realization.colliders.filter(collider=>collider.role!=="projectile-body")});
  installSceneMesh(realization.mesh,{publish:true});installVehiclePresentationMesh(dynamicRealization.mesh);
}

async function initializeSceneMeshWasm() {
  const descriptor = model.scene_mesh;
  const instance = await turingWasmModules.instantiate(descriptor.module);
  const count = shaderViewer.geometry.length * descriptor.topology.vertices_per_instance;
  const arrayParameters = descriptor.parameters.filter(parameter => parameter.role !== "extent");
  let cursor = Math.ceil(descriptor.reserved_bytes / 8) * 8;
  const requiredBytes = cursor + arrayParameters.length * count * Float64Array.BYTES_PER_ELEMENT;
  if (requiredBytes > instance.exports.memory.buffer.byteLength) {
    instance.exports.memory.grow(Math.ceil((requiredBytes - instance.exports.memory.buffer.byteLength) / 65536));
  }
  const offsets = {}, arrays = {};
  arrayParameters.forEach(parameter => {
    offsets[parameter.name] = cursor;
    arrays[parameter.name] = new Float64Array(instance.exports.memory.buffer, cursor, count);
    cursor += count * Float64Array.BYTES_PER_ELEMENT;
  });
  shaderViewer.sceneWasm = {
    descriptor, count, offsets, arrays, run: instance.exports[descriptor.entrypoint]
  };
}

function rebuildSceneMesh() {
  const wasm = shaderViewer.sceneWasm;
  if (!wasm) throw new Error("scene mesh WASM is not initialized");
  const topology = wasm.descriptor.topology;
  const normals = [], roles = [];
  shaderViewer.identitySpans = [];
  let vertex = 0;
  shaderViewer.geometry.forEach((box, boxIndex) => {
    const firstVertex = vertex;
    topology.faces.forEach(face => face.corners.forEach(cornerIndex => {
      const unit = topology.corners[cornerIndex];
      wasm.arrays.center_x[vertex] = box.center[0];
      wasm.arrays.center_z[vertex] = box.center[1];
      wasm.arrays.half_x[vertex] = box.half_extent[0];
      wasm.arrays.half_z[vertex] = box.half_extent[1];
      wasm.arrays.height[vertex] = box.height;
      wasm.arrays.unit_x[vertex] = unit[0];
      wasm.arrays.unit_y[vertex] = unit[1];
      wasm.arrays.unit_z[vertex] = unit[2];
      normals.push(face.normal); roles.push(box.palette_role); vertex += 1;
    }));
    shaderViewer.identitySpans.push({
      identity: box.identity, kind: box.kind, boxIndex,
      firstVertex, vertexCount: vertex - firstVertex,
      worldObjectIndex: model.world.object_order.indexOf(box.identity),
      runtimeObjectId: turingWorld.objectRuntimeId(box.identity),
      semanticParts: [], revision: shaderViewer.revision
    });
  });
  const args = wasm.descriptor.parameters.map(parameter =>
    parameter.role === "extent" ? wasm.count : wasm.offsets[parameter.name]);
  wasm.run(...args);
  const mesh = new Float32Array(wasm.count * 9);
  for (let index = 0; index < wasm.count; index += 1) {
    const offset = index * 9;
    const color = colorVector(model.appearance.colors[roles[index]] || model.appearance.colors.room);
    mesh.set([wasm.arrays.vertex_x[index], wasm.arrays.vertex_y[index], wasm.arrays.vertex_z[index],
      ...normals[index], ...color], offset);
  }
  if (mesh.length !== shaderViewer.geometry.length * topology.vertices_per_instance * 9 ||
      !mesh.every(Number.isFinite)) {
    throw new Error("compiled scene mesh produced an invalid vertex buffer");
  }
  const positionMagnitude = mesh.reduce((maximum, value, index) =>
    index % 9 < 3 ? Math.max(maximum, Math.abs(value)) : maximum, 0);
  if (positionMagnitude < 0.01) throw new Error("compiled scene mesh collapsed to the origin");
  installSceneMesh(mesh);
}

function publishSceneMeshToDocument() {
  const nodes = [...document.querySelectorAll("[data-node-id]")];
  shaderViewer.identitySpans.forEach(span => {
    const box = shaderViewer.geometry[span.boxIndex];
    const worldObject = turingWorld.objectFromRuntimeId(span.runtimeObjectId) ||
      model.world?.objects?.[span.worldObjectIndex];
    if (worldObject && worldObject.identity === span.identity) {
      worldObject.form = {...worldObject.form,
        half_extent: [...box.half_extent], height: box.height,
        wall_thickness: box.appearance?.wall_thickness ?? box.wall_thickness,
        radius: box.appearance?.radius ?? box.radius,
        openings: (box.openings || []).map(opening => ({...opening}))};
      worldObject.persistence.revision = shaderViewer.revision;
      worldObject.mesh = {
        revision: shaderViewer.revision,
        firstVertex: span.firstVertex,
        vertexCount: span.vertexCount,
        semanticParts: span.semanticParts.map(part => ({...part})),
      };
    }
    const element = nodes.find(node => node.dataset.nodeId === span.identity);
    if (!element) return;
    element.dataset.meshIdentity = span.identity;
    element.dataset.meshRevision = String(shaderViewer.revision);
    element.dataset.meshKind = span.kind;
    element.dataset.worldObjectIndex = String(span.worldObjectIndex);
    element.dataset.runtimeObjectId = String(span.runtimeObjectId);
    element.dataset.semanticPartCount = String(span.semanticParts.length);
    element.dataset.wallSource = model.scene_mesh.boundary_contract.source;
    element.dataset.wallHeight = String(box.height);
    element.dataset.interior = "hollow";
    element.dataset.floor = "mandatory-slab";
    element.dataset.openingCount = String((box.openings || []).length);
    element.dataset.ceiling = box.height >=
      model.document_geometry.boundary_semantics.ceiling.absolute_maximum ? "capped" : "open";
    element.style.setProperty("--mesh-height", String(box.height));
    element.style.setProperty("--mesh-width", String(box.half_extent[0]));
    element.style.setProperty("--mesh-depth", String(box.half_extent[1]));
    element.style.setProperty("--wall-height", String(box.height));
    element.style.setProperty("--wall-thickness", String(
      box.appearance?.wall_thickness ?? box.wall_thickness ?? 0.04));
    element.style.setProperty("--object-radius", `${box.appearance?.radius ?? box.radius ?? 8}px`);
    element.style.setProperty("--object-face", box.appearance?.face_color || "transparent");
    element.style.setProperty("--object-wall", box.appearance?.wall_color ||
      model.appearance.colors[box.wall_palette_role] || model.appearance.colors.line);
    element.style.borderColor = "var(--object-wall)";
    element.style.borderWidth = "calc(1px + var(--wall-thickness) * 20px)";
    element.style.borderRadius = "var(--object-radius)";
    if (box.appearance?.face_color) element.style.background = "var(--object-face)";
  });
}

async function initializeSoftwareMeshWasm() {
  const descriptor = model.software_mesh;
  const instance = await turingWasmModules.instantiate(descriptor.module);
  const count = shaderViewer.mesh.length / 9;
  const arrayParameters = descriptor.parameters.filter(parameter => parameter.role !== "extent");
  let cursor = Math.ceil(descriptor.reserved_bytes / 8) * 8;
  const requiredBytes = cursor + arrayParameters.length * count * Float64Array.BYTES_PER_ELEMENT;
  const currentBytes = instance.exports.memory.buffer.byteLength;
  if (requiredBytes > currentBytes) {
    instance.exports.memory.grow(Math.ceil((requiredBytes - currentBytes) / 65536));
  }
  const offsets = {}, arrays = {};
  arrayParameters.forEach(parameter => {
    offsets[parameter.name] = cursor;
    arrays[parameter.name] = new Float64Array(instance.exports.memory.buffer, cursor, count);
    cursor += count * Float64Array.BYTES_PER_ELEMENT;
  });
  for (let index = 0; index < count; index += 1) {
    arrays.vertex_x[index] = shaderViewer.mesh[index * 9];
    arrays.vertex_y[index] = shaderViewer.mesh[index * 9 + 1];
    arrays.vertex_z[index] = shaderViewer.mesh[index * 9 + 2];
  }
  shaderViewer.softwareWasm = {
    descriptor, count, offsets, arrays,
    run: instance.exports[descriptor.entrypoint]
  };
}

function initializeVehicleWheelRenderer(gl){
  const program=gl.createProgram();
  gl.attachShader(program,compileShader(gl,gl.VERTEX_SHADER,VEHICLE_WHEEL_VERTEX_SHADER));
  gl.attachShader(program,compileShader(gl,gl.FRAGMENT_SHADER,VEHICLE_WHEEL_FRAGMENT_SHADER));
  gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(program)||"vehicle wheel shader link failed");
  const vertices=[],segments=48,crossSections=16,major=.75,tube=.25,
    emit=(position,normal)=>vertices.push(...position,...normal),
    tirePoint=(theta,phi)=>{const radial=major+tube*Math.cos(phi);return[
      [Math.cos(theta)*radial,Math.sin(theta)*radial,.5*Math.sin(phi)],
      [Math.cos(theta)*Math.cos(phi),Math.sin(theta)*Math.cos(phi),Math.sin(phi)]];};
  for(let segment=0;segment<segments;segment+=1)for(let cross=0;cross<crossSections;cross+=1){
    const a=segment*Math.PI*2/segments,b=(segment+1)*Math.PI*2/segments,
      c=cross*Math.PI*2/crossSections,d=(cross+1)*Math.PI*2/crossSections,
      pa=tirePoint(a,c),pb=tirePoint(b,c),pc=tirePoint(b,d),pd=tirePoint(a,d);
    [pa,pb,pc,pa,pc,pd].forEach(([p,n])=>emit(p,n));
  }
  for(const side of [-1,1])for(let segment=0;segment<segments;segment+=1){
    const a=segment*Math.PI*2/segments,b=(segment+1)*Math.PI*2/segments,z=side*.501,
      pa=[Math.cos(a)*.51,Math.sin(a)*.51,z],pb=[Math.cos(b)*.51,Math.sin(b)*.51,z],
      normal=[0,0,side];[[pa,normal],[pb,normal],[[0,0,z],normal]].forEach(([p,n])=>emit(p,n));
  }
  const vao=gl.createVertexArray(),buffer=gl.createBuffer();gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(vertices),gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,24,0);
  gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,3,gl.FLOAT,false,24,12);
  shaderViewer.wheelProgram=program;shaderViewer.wheelVao=vao;shaderViewer.wheelVertexCount=vertices.length/6;
  ["uWheelModel","uResolution","uCameraPosition","uCameraFacing","uTreadPhase","uRubberColor",
   "uTreadColor","uRimColor","uRotorColor","uLightDirection"].forEach(name=>
     shaderViewer.wheelLocations[name]=gl.getUniformLocation(program,name));
}

function rotateVehiclePresentationVector(point,state,steer=0){
  const ct=Math.cos(steer),st=Math.sin(steer),steered=[point[0]*ct+point[2]*st,point[1],-point[0]*st+point[2]*ct];
  const cr=Math.cos(state.roll),sr=Math.sin(state.roll),rolled=[steered[0],steered[1]*cr-steered[2]*sr,
    steered[1]*sr+steered[2]*cr],cp=Math.cos(state.pitch),sp=Math.sin(state.pitch),
    pitched=[rolled[0]*cp-rolled[1]*sp,rolled[0]*sp+rolled[1]*cp,rolled[2]],
    cy=Math.cos(state.yaw),sy=Math.sin(state.yaw);
  return[pitched[0]*cy-pitched[2]*sy,pitched[1],pitched[0]*sy+pitched[2]*cy];
}

function drawVehicleWheels(gl,width,height,cameraPosition,cameraFacing,celestial){
  if(vehicleRuntime.disabledPresentationStages.has("wheel-shader"))return;
  const state=vehicleRuntime.state||vehicleRuntime.parkedState,vehicle=vehicleRuntime.active||model.vehicle_slot?.vehicles?.[0];
  if(!state||!shaderViewer.wheelProgram||!vehicleRuntime.wheelBoxes.length)return;
  const config=vehicle.configuration,presentation=config.presentation,
    part=vehicle.wheel_parts?.find(item=>item.identity===vehicleRuntime.wheelPart)||vehicle.wheel_parts?.[0],
    rubber=colorVector(part?.tire_color||"#202624"),tread=colorVector(part?.tread_color||"#687672"),
    rim=colorVector(part?.rim_color||model.appearance.colors["rollbar-silver"]),
    rotor=colorVector(model.appearance.colors["suspension-yellow"]),locations=shaderViewer.wheelLocations;
  try{
  gl.useProgram(shaderViewer.wheelProgram);gl.uniform2f(locations.uResolution,width,height);
  gl.uniform3fv(locations.uCameraPosition,cameraPosition);gl.uniform3fv(locations.uCameraFacing,cameraFacing);
  gl.uniform3fv(locations.uRubberColor,rubber);gl.uniform3fv(locations.uTreadColor,tread);
  gl.uniform3fv(locations.uRimColor,rim);gl.uniform3fv(locations.uRotorColor,rotor);
  gl.uniform3fv(locations.uLightDirection,celestial.key==="sun"?celestial.sunDirection:celestial.moonDirection);
  gl.bindVertexArray(shaderViewer.wheelVao);
  vehicleRuntime.wheelBoxes.forEach((wheel,index)=>{
    const stateData=wheel.wheel_state,radius=Number(stateData.radius),tireWidth=Number(stateData.width),
      center=rotateVehiclePresentationVector(stateData.localCenter,state,0),
      x=rotateVehiclePresentationVector([radius,0,0],state,stateData.steer),
      y=rotateVehiclePresentationVector([0,radius,0],state,stateData.steer),
      z=rotateVehiclePresentationVector([0,0,tireWidth],state,stateData.steer),
      matrix=new Float32Array([x[0],x[1],x[2],0,y[0],y[1],y[2],0,z[0],z[1],z[2],0,
        state.position[0]+center[0],state.position[1]+center[1],state.position[2]+center[2],1]);
    gl.uniformMatrix4fv(locations.uWheelModel,false,matrix);
    gl.uniform1f(locations.uTreadPhase,vehicleRuntime.wheelAngles[index]||0);
    gl.drawArrays(gl.TRIANGLES,0,shaderViewer.wheelVertexCount);
  });
  }catch(error){vehicleRuntime.disabledPresentationStages.add("wheel-shader");reportRuntimeFault("wheel-shader",error);}
}

async function initializeShaderViewer() {
  const baseGeometry = model.document_geometry.boxes;
  // Vehicle-first startup realizes the Springtail before this deferred shader
  // initialization runs.  Preserve those dynamic pieces when installing the
  // authored world rather than silently replacing the complete scene.
  const mountedVehicleGeometry=vehicleRuntime.active?[
    ...vehicleRuntime.frameBoxes,...vehicleRuntime.rollCageBoxes,
    ...vehicleRuntime.mechanicalLinkBoxes,...vehicleRuntime.powertrainBoxes,...vehicleRuntime.bodyShellBoxes
  ]:[];
  const geometry = [...baseGeometry,...mountedVehicleGeometry];
  const colors = model.appearance.colors;
  shaderViewer.geometry = geometry;
  geometry.forEach(box => shaderViewer.formBaselines.set(box.identity, cloneGeometryBox(box)));
  restoreLivingEdits();
  // Keep the last-known-visible host extrusion as the live rendering path.
  // The Python→WASM constructor remains a published, executable contract but
  // must not replace presentation until it has an independent visual gate.
  rebuildPortableSceneMesh();
  shaderViewer.sceneWasm = null;
  const gl = shaderViewer.canvas.getContext("webgl2", {antialias: true});
  if (!gl) {
    shaderViewer.context2d = shaderViewer.canvas.getContext("2d");
    try {
      await initializeSoftwareMeshWasm();
      shaderViewer.backend = "Canvas2D + Python→WASM";
      shaderViewer.readout.textContent = `Canvas2D + Python→WASM · ${geometry.length} extrusions · click for WASD + mouse · gamepad auto`;
    } catch (error) {
      shaderViewer.backend = "Canvas2D";
      shaderViewer.readout.textContent = `Canvas2D · ${geometry.length} extrusions · WASM error: ${error.message}`;
      console.error(error);
    }
    return;
  }
  try {
    shaderViewer.gl = gl;
    activateViewportShader(VIEWPORT_DEFAULT_SHADER);
    const program = shaderViewer.program;
    if (["uResolution", "uCameraPosition", "uCameraFacing"]
        .some(name => shaderViewer.locations[name] === null)) {
      throw new Error("shader program omitted a required viewport uniform");
    }
    const skyProgram=gl.createProgram();
    gl.attachShader(skyProgram,compileShader(gl,gl.VERTEX_SHADER,VIEWPORT_SKY_VERTEX_SHADER));
    gl.attachShader(skyProgram,compileShader(gl,gl.FRAGMENT_SHADER,VIEWPORT_SKY_FRAGMENT_SHADER));
    gl.linkProgram(skyProgram);
    if(!gl.getProgramParameter(skyProgram,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(skyProgram));
    shaderViewer.skyProgram=skyProgram;
    ["uResolution","uCameraFacing","uSunDirection","uMoonDirection","uDayZenith",
     "uNightZenith","uHorizonColor","uSunColor","uMoonColor"].forEach(name=>
      shaderViewer.skyLocations[name]=gl.getUniformLocation(skyProgram,name));
    if(Object.values(shaderViewer.skyLocations).some(location=>location===null)){
      throw new Error("half-dome sky shader omitted a required uniform");
    }
    const crosshairProgram = gl.createProgram();
    gl.attachShader(crosshairProgram, compileShader(gl, gl.VERTEX_SHADER, VIEWPORT_CROSSHAIR_VERTEX_SHADER));
    gl.attachShader(crosshairProgram, compileShader(gl, gl.FRAGMENT_SHADER, VIEWPORT_CROSSHAIR_FRAGMENT_SHADER));
    gl.linkProgram(crosshairProgram);
    if (!gl.getProgramParameter(crosshairProgram, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(crosshairProgram));
    }
    shaderViewer.crosshairProgram = crosshairProgram;
    ["uResolution","uIdleColor","uTargetColor","uHasTarget"].forEach(name =>
      shaderViewer.crosshairLocations[name] = gl.getUniformLocation(crosshairProgram, name));
    if (Object.values(shaderViewer.crosshairLocations).some(location => location === null)) {
      throw new Error("crosshair shader omitted a required uniform");
    }
    const hudProgram=gl.createProgram();
    gl.attachShader(hudProgram,compileShader(gl,gl.VERTEX_SHADER,VEHICLE_HUD_VERTEX_SHADER));
    gl.attachShader(hudProgram,compileShader(gl,gl.FRAGMENT_SHADER,VEHICLE_HUD_FRAGMENT_SHADER));
    gl.linkProgram(hudProgram);
    if(!gl.getProgramParameter(hudProgram,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(hudProgram));
    shaderViewer.vehicleHudProgram=hudProgram;
    ["uResolution","uRect","uAngle","uColor","uEllipse"].forEach(name=>
      shaderViewer.vehicleHudLocations[name]=gl.getUniformLocation(hudProgram,name));
    if(Object.values(shaderViewer.vehicleHudLocations).some(location=>location===null))
      throw new Error("vehicle HUD shader omitted a required uniform");
    const hudVao=gl.createVertexArray(),hudBuffer=gl.createBuffer();
    gl.bindVertexArray(hudVao);gl.bindBuffer(gl.ARRAY_BUFFER,hudBuffer);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([0,0,1,0,0,1,0,1,1,0,1,1]),gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
    shaderViewer.vehicleHudVao=hudVao;
    // Tires have a purpose-built round/tread shader.  Initialize it as part of
    // the same WebGL presentation graph; leaving this constructor disconnected
    // made a mounted vehicle look absent even though its state existed.
    initializeVehicleWheelRenderer(gl);
    if(vehicleRuntime.active&&vehicleRuntime.state){
      updateVehiclePresentation(vehicleRuntime.active,vehicleRuntime.state,0,0);
      updateVehicleBodyPresentation(vehicleRuntime.active,vehicleRuntime.state);
    }
    const mesh = shaderViewer.mesh;
    const vao = gl.createVertexArray(), buffer = gl.createBuffer();
    gl.bindVertexArray(vao); gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, mesh, gl.STATIC_DRAW);
    const stride = 9 * Float32Array.BYTES_PER_ELEMENT;
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 3 * 4);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, stride, 6 * 4);
    shaderViewer.vao=vao;shaderViewer.buffer=buffer;shaderViewer.vertexCount=mesh.length/9;
    installVehiclePresentationMesh(buildExtrudedBoxMesh(
      vehiclePresentationGeometry(),model.appearance.colors).mesh);
    initializeShadowPass(gl);
    initializeCameraDepthPass(gl);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);
    const sky = colorVector(colors.sky); gl.clearColor(sky[0], sky[1], sky[2], 1);
    const uploadError = gl.getError();
    if (uploadError !== gl.NO_ERROR) throw new Error(`viewport uniform upload failed (${uploadError})`);
    shaderViewer.readout.textContent = `${shaderViewer.backend} · ${geometry.length} extrusions · click for WASD + mouse · gamepad auto`;
  } catch (error) {
    shaderViewer.gl = null; shaderViewer.program = null;
    const replacement = document.createElement("canvas");
    replacement.className = shaderViewer.canvas.className;
    replacement.setAttribute("aria-label", shaderViewer.canvas.getAttribute("aria-label"));
    shaderViewer.canvas.replaceWith(replacement);
    shaderViewer.canvas = replacement;
    shaderViewer.context2d = replacement.getContext("2d");
    try {
      await initializeSoftwareMeshWasm();
      shaderViewer.backend = "Canvas2D + Python→WASM";
      shaderViewer.readout.textContent = `Canvas2D + Python→WASM · ${geometry.length} extrusions · WebGL2 error: ${error.message}`;
    } catch (wasmError) {
      shaderViewer.backend = "Canvas2D";
      shaderViewer.readout.textContent = `Canvas2D · rendering initialization failed: ${wasmError.message}`;
      console.error(wasmError);
    }
    console.error(error);
  }
  // Cover both initialization orders: worker-first and renderer-first. The
  // resident tire gather needs immutable terrain/wall buffers before its
  // first dispatch; a remount must never be required to complete this bind.
  if(stateLoopRuntime.ready&&stateLoopRuntime.worker&&shaderViewer.colliders.length)
    stateLoopRuntime.worker.postMessage({type:"colliders",colliders:shaderViewer.colliders});
}

function normalized3(vector) {
  const length = Math.hypot(vector[0], vector[1], vector[2]) || 1;
  return vector.map(component => component / length);
}

function cross3(left, right) {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0]
  ];
}

function rayBoxDistance(origin, direction, box) {
  if(box.geometry_mode==="sphere"){
    const radius=Number(box.radius||box.half_extent?.[0]||0);
    const center=[Number(box.center?.[0]||0),Number(box.center_y??radius),
      Number(box.center?.[1]||0)];
    const offset=origin.map((value,index)=>value-center[index]);
    const along=offset.reduce((sum,value,index)=>sum+value*direction[index],0);
    const discriminant=along*along-(offset.reduce((sum,value)=>sum+value*value,0)-radius*radius);
    if(discriminant<0)return null;
    const root=Math.sqrt(discriminant),near=-along-root,far=-along+root;
    return near>=.02?near:(far>=.02?far:null);
  }
  const yaw=boxYawDegrees(box)*Math.PI/180,cosine=Math.cos(yaw),sine=Math.sin(yaw);
  const dx=origin[0]-box.center[0],dz=origin[2]-box.center[1];
  const localOrigin=[dx*cosine+dz*sine,origin[1],-dx*sine+dz*cosine];
  const localDirection=[direction[0]*cosine+direction[2]*sine,direction[1],
    -direction[0]*sine+direction[2]*cosine];
  const baseY=Number(box.placement?.elevation || 0);
  const minimum = [-box.half_extent[0], baseY, -box.half_extent[1]];
  const maximum = [box.half_extent[0], baseY+box.height, box.half_extent[1]];
  let enter = -Infinity, leave = Infinity;
  for (let axis = 0; axis < 3; axis += 1) {
    if (Math.abs(localDirection[axis]) < 1e-9) {
      if (localOrigin[axis] < minimum[axis] || localOrigin[axis] > maximum[axis]) return null;
      continue;
    }
    const near = (minimum[axis] - localOrigin[axis]) / localDirection[axis];
    const far = (maximum[axis] - localOrigin[axis]) / localDirection[axis];
    enter = Math.max(enter, Math.min(near, far));
    leave = Math.min(leave, Math.max(near, far));
    if (leave < enter) return null;
  }
  if (leave < 0) return null;
  return enter >= 0 ? enter : leave;
}

function placementBoundaryHost(box) {
  return Boolean(box && box.kind !== "world-envelope" &&
    box.geometry_mode !== "solid" && box.geometry_mode !== "sphere" &&
    Number(box.height) > Number(box.floor_height || 0) + 1e-4 &&
    Array.isArray(box.half_extent));
}

function raySceneTriangle(origin = shaderViewer.cameraPosition,
                          direction = normalized3(shaderViewer.cameraFacing || [0, 0, -1])) {
  const mesh = shaderViewer.baseMesh;
  if (!mesh || !origin) return null;
  let nearest = null;
  for (let offset = 0; offset + 26 < mesh.length; offset += 27) {
    const vertex = corner => [mesh[offset + corner * 9], mesh[offset + corner * 9 + 1],
      mesh[offset + corner * 9 + 2]];
    const a = vertex(0), b = vertex(1), c = vertex(2);
    const edgeAB = b.map((value, axis) => value - a[axis]);
    const edgeAC = c.map((value, axis) => value - a[axis]);
    const p = cross3(direction, edgeAC);
    const determinant = edgeAB.reduce((sum, value, axis) => sum + value * p[axis], 0);
    if (Math.abs(determinant) < 1e-8) continue;
    const inverse = 1 / determinant;
    const fromA = origin.map((value, axis) => value - a[axis]);
    const baryB = fromA.reduce((sum, value, axis) => sum + value * p[axis], 0) * inverse;
    if (baryB < 0 || baryB > 1) continue;
    const q = cross3(fromA, edgeAB);
    const baryC = direction.reduce((sum, value, axis) => sum + value * q[axis], 0) * inverse;
    if (baryC < 0 || baryB + baryC > 1) continue;
    const distance = edgeAC.reduce((sum, value, axis) => sum + value * q[axis], 0) * inverse;
    if (distance <= 0.02 || (nearest && distance >= nearest.distance)) continue;
    let normal = normalized3(cross3(edgeAB, edgeAC));
    if (normal.reduce((sum, value, axis) => sum + value * direction[axis], 0) > 0) {
      normal = normal.map(value => -value);
    }
    const triangleIndex = offset / 27;
    const firstVertex = triangleIndex * 3;
    const span = shaderViewer.identitySpans.find(candidate => firstVertex >= candidate.firstVertex &&
      firstVertex < candidate.firstVertex + candidate.vertexCount);
    const semanticPart = shaderViewer.semanticPartSpans.find(candidate =>
      firstVertex >= candidate.firstVertex && firstVertex < candidate.firstVertex + candidate.vertexCount);
    nearest = {distance, triangleIndex, firstVertex, vertices: [a, b, c], normal,
      barycentric: [1 - baryB - baryC, baryB, baryC],
      position: origin.map((value, axis) => value + direction[axis] * distance),
      objectIdentity: span?.identity || semanticPart?.objectIdentity || null,
      partIdentity: semanticPart?.identity || null};
  }
  return nearest;
}

function portalTriangleMembership(hit, radius) {
  const mesh = shaderViewer.baseMesh;
  if (!mesh) return [];
  const memberships = [];
  for (let offset = 0; offset + 26 < mesh.length; offset += 27) {
    const triangleIndex = offset / 27, firstVertex = triangleIndex * 3;
    const semanticPart = shaderViewer.semanticPartSpans.find(candidate =>
      firstVertex >= candidate.firstVertex && firstVertex < candidate.firstVertex + candidate.vertexCount);
    if (hit.partIdentity && semanticPart?.identity !== hit.partIdentity) continue;
    const points = [0, 1, 2].map(corner => [mesh[offset + corner * 9],
      mesh[offset + corner * 9 + 1], mesh[offset + corner * 9 + 2]]);
    const centroid = [0, 1, 2].map(axis => points.reduce((sum, point) => sum + point[axis], 0) / 3);
    const delta = centroid.map((value, axis) => value - hit.position[axis]);
    const distance = Math.hypot(...delta);
    const edgeAB = points[1].map((value, axis) => value - points[0][axis]);
    const edgeAC = points[2].map((value, axis) => value - points[0][axis]);
    const normal = normalized3(cross3(edgeAB, edgeAC));
    if (distance <= radius * 1.5 && Math.abs(normal.reduce(
        (sum, value, axis) => sum + value * hit.normal[axis], 0)) >= 0.92) {
      memberships.push({triangle_index: triangleIndex,
        domain: triangleIndex === hit.triangleIndex ? "barycentric-center" : "radial-intersection",
        barycentric_center: triangleIndex === hit.triangleIndex ? [...hit.barycentric] : null});
    }
  }
  return memberships;
}

function pickCrosshairIdentity() {
  if (!shaderViewer.cameraPosition || !shaderViewer.cameraFacing) return null;
  const direction = normalized3(shaderViewer.cameraFacing);
  let nearest = null;
  const selectedPlacementRecipe = model.placement?.recipes.find(candidate =>
    candidate.identity === placementState.selectedRecipe);
  if (selectedPlacementRecipe?.opening_kind === "portal") {
    const triangleHit = raySceneTriangle();
    const box = shaderViewer.geometry.find(candidate => candidate.identity === triangleHit?.objectIdentity);
    if (triangleHit && box) nearest = {distance: triangleHit.distance, box, triangleHit,
      span: shaderViewer.identitySpans.find(span => span.identity === box.identity)};
  }
  shaderViewer.geometry.forEach((box, boxIndex) => {
    if (selectedPlacementRecipe?.opening_kind === "portal") return;
    if (box.placement?.custody === "inventory" || box.placement?.custody === "preview") return;
    if (selectedPlacementRecipe?.placement_kind === "subtractive" &&
        !placementBoundaryHost(box)) return;
    const distance = rayBoxDistance(shaderViewer.cameraPosition, direction, box);
    if (distance === null || (nearest && distance >= nearest.distance)) return;
    nearest = {distance, box, span: shaderViewer.identitySpans.find(span => span.boxIndex === boxIndex)};
  });
  shaderViewer.crosshairIdentity = nearest?.box.identity || null;
  const tooltip = shaderViewer.focusTooltip;
  if (tooltip) {
    tooltip.hidden = !nearest;
    if (nearest) {
      const subject = index.get(nearest.box.identity);
      const name = subject?.name || nearest.box.identity;
      tooltip.dataset.focusIdentity = nearest.box.identity;
      tooltip.textContent = `${nearest.box.kind} · ${name}\n${nearest.box.identity}\n` +
        `floor + hollow interior · wall ${nearest.box.height.toFixed(2)} high · ` +
        `${(nearest.box.openings || []).length} opening(s) · revision ${shaderViewer.revision}`;
      const canvas = shaderViewer.canvas;
      const left = canvas.offsetLeft + canvas.clientWidth * 0.5 + 15;
      const top = canvas.offsetTop + canvas.clientHeight * 0.5 + 15;
      tooltip.style.left = `${Math.max(6, Math.min(left,
        shaderViewer.element.clientWidth - tooltip.offsetWidth - 6))}px`;
      tooltip.style.top = `${top}px`;
    }
  }
  placementState.hoverIdentity = nearest?.box.identity || null;
  updatePlacementFocusVisuals(nearest?.box || null);
  return nearest;
}

function applyFormInstruction(identity, instructionIdentity) {
  const box = shaderViewer.geometry.find(item => item.identity === identity);
  const instruction = model.scene_mesh.context_menu.items
    .flatMap(item => item.instructions || [])
    .find(item => item.identity === instructionIdentity);
  if (!box || !instruction) return;
  if (instruction.operation === "restore") {
    const baseline = shaderViewer.formBaselines.get(identity);
    if (baseline) Object.assign(box, cloneGeometryBox(baseline));
  } else if (instruction.operation === "scale") {
    if (instruction.parameter === "height") box.height = Math.max(0.08, box.height * instruction.operand);
    if (instruction.parameter === "half_extent.x") {
      box.half_extent[0] = Math.max(0.04, box.half_extent[0] * instruction.operand);
    }
    if (instruction.parameter === "half_extent.z") {
      box.half_extent[1] = Math.max(0.04, box.half_extent[1] * instruction.operand);
    }
  }
  shaderViewer.revision += 1;
  model.scene_mesh.revision = shaderViewer.revision;
  turingWorldRevisions.publish({
    revision: shaderViewer.revision, identity, operation: "apply-form"
  });
  rebuildPortableSceneMesh();
  saveLivingEdits(identity);
  const actor = viewportControls.policy?.actor || model.identity;
  const edge = actionEdges.register(actor, "apply-form", identity);
  abstractUISystemTimer.issue({
    actor, type: "apply-form", destination: identity, edge,
    instruction: instructionIdentity, revision: shaderViewer.revision,
    issued_at: performance.now()
  });
  status.textContent = `${instruction.label} → ${identity} · mesh/document revision ${shaderViewer.revision}`;
}

function closeSceneContextMenu() {
  shaderViewer.contextMenu?.remove();
  shaderViewer.contextMenu = null;
}

function openCrosshairContextMenu(clientX, clientY) {
  const target = pickCrosshairIdentity();
  closeSceneContextMenu();
  if (!target) return;
  if (document.pointerLockElement === shaderViewer.canvas) document.exitPointerLock?.();
  const menu = div("context-menu");
  menu.dataset.contextIdentity = target.box.identity;
  menu.setAttribute("role", "menu");
  menu.append(div("context-title", `${target.box.kind} · ${target.box.identity}`));
  const formContract = model.scene_mesh.context_menu.items.find(item => item.identity === "context:form");
  const form = div("context-item", formContract.label);
  form.tabIndex = 0; form.setAttribute("role", "menuitem");
  const submenu = div("form-submenu");
  formContract.instructions.forEach(instruction => {
    const item = div("context-item", instruction.label);
    item.tabIndex = 0; item.setAttribute("role", "menuitem");
    item.dataset.formInstruction = instruction.identity;
    item.addEventListener("click", event => {
      event.stopPropagation();
      applyFormInstruction(target.box.identity, instruction.identity);
      closeSceneContextMenu();
    });
    item.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); item.click(); }
    });
    submenu.append(item);
  });
  form.append(submenu);
  form.addEventListener("click", event => {
    if (event.target === form) form.classList.toggle("expanded");
  });
  form.addEventListener("keydown", event => {
    if (event.target === form && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault(); form.classList.toggle("expanded");
    }
  });
  menu.append(form);
  document.body.append(menu);
  const bounds = menu.getBoundingClientRect();
  menu.style.left = `${Math.max(6, Math.min(innerWidth - bounds.width - 6, clientX))}px`;
  menu.style.top = `${Math.max(6, Math.min(innerHeight - bounds.height - 6, clientY))}px`;
  shaderViewer.contextMenu = menu;
  form.focus();
}

function openCrosshairContextMenuAtViewportCenter() {
  const bounds = shaderViewer.canvas?.getBoundingClientRect();
  if (bounds) openCrosshairContextMenu(bounds.left + bounds.width / 2, bounds.top + bounds.height / 2);
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

function setControlFocus(mode) {
  if (!controlFocus.policy?.routes.includes(mode) || controlFocus.dialogue) return controlFocus.mode;
  if (mode === controlFocus.mode) return mode;
  const previous = controlFocus.mode;
  controlFocus.mode = mode;
  shaderViewer.element?.classList.toggle("projected-pointer-focus", mode === "projected-pointer");
  if (mode === "projected-pointer") {
    const bounds = shaderViewer.canvas?.getBoundingClientRect();
    if (bounds) controlFocus.projectedPosition = [bounds.width * 0.5, bounds.height * 0.5];
    if (document.pointerLockElement === shaderViewer.canvas) document.exitPointerLock?.();
  } else if (mode === "game" && viewportControls.highlighted) {
    const request = shaderViewer.canvas?.requestPointerLock?.();
    if (request?.catch) request.catch(() => {});
  }
  const actor = viewportControls.policy?.actor || model.identity;
  const edge = actionEdges.register(actor, "switch-focus", controlFocus.policy.identity);
  abstractUISystemTimer.issue({actor, type: "switch-focus", destination: controlFocus.policy.identity,
    edge, from: previous, to: mode, issued_at: performance.now()});
  return mode;
}

function toggleControlFocus() {
  return setControlFocus(controlFocus.mode === "game" ? "projected-pointer" : "game");
}

function claimDialogueFocus(dialogueIdentity) {
  if (controlFocus.dialogue) return false;
  controlFocus.previousMode = controlFocus.mode;
  controlFocus.dialogue = dialogueIdentity;
  controlFocus.mode = "dialogue";
  if (document.pointerLockElement === shaderViewer.canvas) document.exitPointerLock?.();
  return true;
}

function releaseDialogueFocus(dialogueIdentity) {
  if (controlFocus.dialogue !== dialogueIdentity) return false;
  controlFocus.dialogue = null;
  controlFocus.mode = controlFocus.previousMode || "game";
  controlFocus.previousMode = null;
  shaderViewer.element?.classList.toggle("projected-pointer-focus",
    controlFocus.mode === "projected-pointer");
  if (controlFocus.mode === "game" && viewportControls.highlighted) {
    const request = shaderViewer.canvas?.requestPointerLock?.();
    if (request?.catch) request.catch(() => {});
  }
  return true;
}

function selectHotbarSlot(number) {
  if(primaryActionState.down)endViewportPrimary("tool-switch");
  if(secondaryActionState.down)endViewportSecondary("tool-switch");
  const slot = hotbarState.model?.slots.find(candidate => candidate.number === number);
  if (!slot) return;
  hotbarState.activeSlot = number;
  hotbarState.model.active_slot = number;
  const item = hotbarState.inventory.items.find(candidate => candidate.identity === slot.item);
  if(item?.properties?.vehicle){
    const placeAtActor=model.vehicle_slot?.initial_state?.placement==="at-player-spawn";
    setActiveVehicle(item.properties.vehicle,{placeAtActor,inventoryItem:item});
  }
  const routedTool = item?.properties?.tool || (item?.is_tool ? item.entity : null);
  hotbarState.inventory.active_tool = routedTool
    ? {item: item.identity, entity: routedTool} : null;
  if (item?.is_tool && model.tools?.find(tool => tool.identity === item.entity)?.name ===
      "Placement tool") placementState.selectedRecipe = null;
  if (item?.properties?.recipe) selectPlacementRecipe(item.properties.recipe);
  shaderViewer.element?.querySelectorAll("[data-hotbar-slot]").forEach(element =>
    element.classList.toggle("active", Number(element.dataset.hotbarSlot) === hotbarState.activeSlot));
  refreshToolModeControl();
  const actor = viewportControls.policy?.actor || model.identity;
  const destination = item?.identity || hotbarState.inventory.identity;
  const edge = actionEdges.register(actor, "select-hotbar-slot", destination);
  abstractUISystemTimer.issue({actor, type: "select-hotbar-slot", destination, edge,
    slot: number, item: item?.identity || null, issued_at: performance.now()});
}

function activeToolObject() {
  const entity = hotbarState.inventory?.active_tool?.entity;
  return model.tools?.find(tool => tool.identity === entity) || null;
}

function activeToolMode(tool=activeToolObject()) {
  if(!tool?.modes?.length)return null;
  const name=toolModeState.byTool.get(tool.identity)||tool.default_mode||tool.modes[0].name;
  return tool.modes.find(mode=>mode.name===name)||tool.modes[0];
}

function refreshToolModeControl() {
  const button=toolModeState.button,tool=activeToolObject(),mode=activeToolMode(tool);
  if(!button)return;
  button.hidden=!mode;button.disabled=!mode;
  button.textContent=mode?`mode · ${mode.name}`:"mode · —";
  button.title=mode?.description||"The active tool has no alternate modes";
  button.dataset.toolMode=mode?.name||"";
}

function cycleActiveToolMode() {
  const tool=activeToolObject();if(!tool?.modes?.length)return false;
  if(primaryActionState.down)endViewportPrimary("mode-switch");
  if(secondaryActionState.down)endViewportSecondary("mode-switch");
  const current=activeToolMode(tool),index=tool.modes.findIndex(mode=>mode.name===current.name);
  const next=tool.modes[(index+1)%tool.modes.length];toolModeState.byTool.set(tool.identity,next.name);
  refreshToolModeControl();saveLivingEdits(null);
  setPlacementStatus(`${tool.name} mode · ${next.name} · ${next.description}`);
  const actor=viewportControls.policy?.actor||model.identity;
  const edge=actionEdges.register(actor,"set-tool-mode",next.identity);
  abstractUISystemTimer.issue({actor,tool:tool.identity,type:"set-tool-mode",destination:next.identity,
    mode:next.name,edge,issued_at:performance.now()});
  return true;
}

function applyAestheticValue(targetIdentity, name, rawValue) {
  const box = shaderViewer.geometry.find(item => item.identity === targetIdentity);
  if (!box) return;
  const property = toolDialogueState.tool?.dialogue.properties.find(item => item.name === name);
  if (!property) return;
  const value = property.input_kind === "range" ? Number(rawValue) : rawValue;
  if (name === "height") box.height = Math.max(0.04, Number(value));
  else {
    box.appearance = {...(box.appearance || {}), [name]: value};
  }
  shaderViewer.revision += 1;
  model.scene_mesh.revision = shaderViewer.revision;
  turingWorldRevisions.publish({
    revision: shaderViewer.revision, identity: targetIdentity,
    operation: "apply-aesthetic"
  });
  rebuildPortableSceneMesh();
  saveLivingEdits(targetIdentity);
}

function applyAestheticPreset(presetIdentity) {
  const preset = toolDialogueState.tool?.dialogue.presets.find(item => item.identity === presetIdentity);
  if (!preset || !toolDialogueState.target) return;
  const box = shaderViewer.geometry.find(item => item.identity === toolDialogueState.target);
  if (!box) return;
  Object.entries(preset.values).forEach(([name, value]) => {
    const property = toolDialogueState.tool.dialogue.properties.find(item => item.name === name);
    if (!property) return;
    if (name === "height") box.height = Math.max(0.04, Number(value));
    else box.appearance = {...(box.appearance || {}), [name]: value};
  });
  shaderViewer.revision += 1;
  model.scene_mesh.revision = shaderViewer.revision;
  turingWorldRevisions.publish({
    revision: shaderViewer.revision, identity: toolDialogueState.target,
    operation: "apply-aesthetic-preset"
  });
  rebuildPortableSceneMesh();
  saveLivingEdits(toolDialogueState.target);
  toolDialogueState.element?.querySelectorAll("[data-aesthetic-property]").forEach(input => {
    input.value = input.dataset.aestheticProperty === "height"
      ? String(box.height) : String(box.appearance?.[input.dataset.aestheticProperty] ?? input.value);
  });
}

function closeToolDialogue() {
  const identity = toolDialogueState.tool?.dialogue.identity;
  toolDialogueState.element?.remove();
  toolDialogueState.element = null;
  toolDialogueState.tool = null;
  toolDialogueState.target = null;
  if (identity) releaseDialogueFocus(identity);
}

function openToolDialogue(tool, targetIdentity) {
  const target = index.get(targetIdentity);
  const box = shaderViewer.geometry.find(item => item.identity === targetIdentity);
  if (!tool.dialogue || !target || !box) return;
  closeToolDialogue();
  if (!claimDialogueFocus(tool.dialogue.identity)) return;
  const root = div("tool-dialogue");
  root.dataset.toolDialogue = tool.dialogue.identity;
  root.setAttribute("role", "dialog");
  root.setAttribute("aria-modal", "false");
  root.setAttribute("aria-label", tool.dialogue.title);
  const head = div("tool-dialogue-head");
  const heading = div("");
  heading.append(div("kind", tool.name), div("node-name", tool.dialogue.title),
    div("tool-dialogue-target", `${target.kind} · ${target.name}\n${targetIdentity}`));
  head.append(heading);
  root.append(head);
  const presets = div("tool-presets");
  tool.dialogue.presets.forEach(preset => {
    const item = div("tool-preset", preset.name);
    item.dataset.aestheticPreset = preset.identity;
    item.tabIndex = 0; item.setAttribute("role", "button");
    presets.append(item);
  });
  root.append(presets);
  tool.dialogue.properties.forEach(property => {
    const row = div("tool-property");
    const label = document.createElement("label");
    label.textContent = property.label;
    const input = document.createElement("input");
    input.type = property.input_kind;
    input.dataset.aestheticProperty = property.name;
    input.dataset.aestheticTarget = targetIdentity;
    const presentationDefault = property.name === "face_color"
      ? model.appearance.colors[box.palette_role]
      : (property.name === "wall_color"
        ? (model.appearance.colors[box.wall_palette_role] || model.appearance.colors.line)
        : (property.name === "wall_thickness" ? box.wall_thickness
          : (property.name === "radius" ? box.radius : property.default)));
    input.value = property.name === "height" ? String(box.height) :
      String(box.appearance?.[property.name] ?? presentationDefault);
    if (property.minimum !== undefined) input.min = String(property.minimum);
    if (property.maximum !== undefined) input.max = String(property.maximum);
    if (property.step !== undefined) input.step = String(property.step);
    label.htmlFor = input.id = `${tool.dialogue.identity}/${property.name}`;
    row.append(label, input); root.append(row);
  });
  const done = div("dialogue-done", "Done");
  done.dataset.dialogueResponse = "done";
  done.tabIndex = 0; done.setAttribute("role", "button");
  root.append(done);
  document.body.append(root);
  toolDialogueState.element = root;
  toolDialogueState.tool = tool;
  toolDialogueState.target = targetIdentity;
  root.querySelector("input")?.focus();
}

function setPlacementStatus(message) {
  if (placementState.statusElement) placementState.statusElement.textContent = message;
}

function placementToolActive() {
  return activeToolObject()?.name === "Placement tool";
}

function placementFocusBox() {
  return placementState.payload?.box || geometryBox(placementState.focusedIdentity) ||
    (placementToolActive() ? geometryBox(placementState.hoverIdentity) : null);
}

function syncPlacementGimbalInputs() {
  shaderViewer.element?.querySelectorAll("[data-placement-axis]").forEach(input => {
    input.value = String(placementState.offsets[input.dataset.placementAxis] || 0);
  });
}

function selectPlacementFocus(identity) {
  const box = geometryBox(identity);
  if (!box || box.kind === "world-envelope") return false;
  if (placementState.focusedIdentity && placementState.focusedIdentity !== identity &&
      placementState.focusBaseline) cancelPlacementFocus();
  placementState.focusedIdentity = identity;
  placementState.focusBaseline = cloneGeometryBox(box);
  placementState.offsets = {x:0,y:0,z:0,yaw:0};
  syncPlacementGimbalInputs();
  setPlacementStatus(`gimbal focus · ${box.label || identity} · adjust x/z/yaw, then apply`);
  updatePlacementFocusVisuals(box);
  return true;
}

function commitPlacementFocus() {
  const box = geometryBox(placementState.focusedIdentity);
  if (!box || !placementState.focusBaseline) return false;
  box.placement = {...(box.placement || {}), custody:box.placement?.custody || "placed",
    rotation:[0,boxYawDegrees(box),0]};
  const worldObject=model.world.objects.find(candidate=>candidate.identity===box.identity);
  if(worldObject){
    worldObject.transform.position=[box.center[0],Number(box.placement.elevation || 0),box.center[1]];
    worldObject.transform.rotation=[...box.placement.rotation];
  }
  shaderViewer.revision += 1; model.scene_mesh.revision = shaderViewer.revision;
  saveLivingEdits(box.identity);
  placementState.focusBaseline = cloneGeometryBox(box);
  placementState.offsets = {x:0,y:0,z:0,yaw:0};
  syncPlacementGimbalInputs(); rebuildPortableSceneMesh();
  setPlacementStatus(`applied mesh transform · ${box.label || box.identity} · bbox synchronized`);
  return true;
}

function cancelPlacementFocus() {
  const box = geometryBox(placementState.focusedIdentity);
  if (box && placementState.focusBaseline) Object.assign(box, cloneGeometryBox(placementState.focusBaseline));
  placementState.focusedIdentity = null; placementState.focusBaseline = null;
  placementState.offsets = {x:0,y:0,z:0,yaw:0};
  syncPlacementGimbalInputs();
  if (box) rebuildPortableSceneMesh();
  updatePlacementFocusVisuals(null);
}

function projectPlacementBoxBounds(box) {
  const canvas = shaderViewer.canvas;
  const camera = shaderViewer.cameraPosition, facing = normalized3(shaderViewer.cameraFacing || [0,0,-1]);
  if (!canvas || !camera) return null;
  const right = normalized3(cross3(facing,[0,1,0]));
  const up = normalized3(cross3(right,facing));
  const aspect = canvas.clientWidth/Math.max(1,canvas.clientHeight), tangent = .70;
  const yaw = boxYawDegrees(box)*Math.PI/180, cosine=Math.cos(yaw), sine=Math.sin(yaw);
  const projected=[];
  const baseY=Number(box.placement?.elevation || 0);
  [-1,1].forEach(sx => [-1,1].forEach(sz => [baseY,baseY+box.height].forEach(y => {
    const lx=sx*box.half_extent[0], lz=sz*box.half_extent[1];
    const world=[box.center[0]+lx*cosine-lz*sine,y,box.center[1]+lx*sine+lz*cosine];
    const relative=[world[0]-camera[0],world[1]-camera[1],world[2]-camera[2]];
    const view=[relative[0]*right[0]+relative[1]*right[1]+relative[2]*right[2],
      relative[0]*up[0]+relative[1]*up[1]+relative[2]*up[2],
      relative[0]*facing[0]+relative[1]*facing[1]+relative[2]*facing[2]];
    if(view[2]>.04) projected.push([canvas.offsetLeft+canvas.clientWidth*(.5+.5*view[0]/(tangent*aspect*view[2])),
      canvas.offsetTop+canvas.clientHeight*(.5-.5*view[1]/(tangent*view[2]))]);
  })));
  if(!projected.length)return null;
  const xs=projected.map(point=>point[0]),ys=projected.map(point=>point[1]);
  return {left:Math.max(canvas.offsetLeft,Math.min(...xs)),top:Math.max(canvas.offsetTop,Math.min(...ys)),
    right:Math.min(canvas.offsetLeft+canvas.clientWidth,Math.max(...xs)),
    bottom:Math.min(canvas.offsetTop+canvas.clientHeight,Math.max(...ys))};
}

function updatePlacementFocusVisuals(crosshairBox=null) {
  const active=placementToolActive();
  const focused=geometryBox(placementState.focusedIdentity);
  const box=placementState.payload?.box || focused || (active ? crosshairBox : null);
  document.querySelectorAll("[data-placement-focused],[data-placement-hover]").forEach(element => {
    element.removeAttribute("data-placement-focused"); element.removeAttribute("data-placement-hover");
  });
  if (box) {
    const element=shaderViewer.mapElement?.querySelector(`[data-node-id="${CSS.escape(box.identity)}"]`);
    if(element)element.dataset[focused || placementState.payload ? "placementFocused" : "placementHover"]="true";
  }
  const focusCard=placementState.focusElement;
  if(focusCard){
    focusCard.dataset.hasFocus=String(Boolean(box));
    const label=focusCard.querySelector(".placement-focus-label");
    if(label)label.textContent=box ? `${box.label || box.identity} · bbox ${(box.half_extent[0]*2).toFixed(2)} × ${(box.half_extent[1]*2).toFixed(2)} × ${Number(box.height).toFixed(2)} · yaw ${boxYawDegrees(box).toFixed(1)}°` : "No mesh focused · aim or click a map object";
  }
  const overlay=shaderViewer.placementOverlay;
  if(!overlay)return;
  const bounds=box ? projectPlacementBoxBounds(box) : null;
  overlay.hidden=!bounds;
  if(bounds){
    overlay.style.left=`${bounds.left}px`;overlay.style.top=`${bounds.top}px`;
    overlay.style.width=`${Math.max(2,bounds.right-bounds.left)}px`;
    overlay.style.height=`${Math.max(2,bounds.bottom-bounds.top)}px`;
    overlay.querySelector(".placement-bbox-label").textContent=box.label || box.identity;
  }
}

function refreshInventoryCounts() {
  shaderViewer.element?.querySelectorAll("[data-hotbar-slot]").forEach(element => {
    const slot = hotbarState.model.slots.find(candidate =>
      candidate.number === Number(element.dataset.hotbarSlot));
    const item = hotbarState.inventory.items.find(candidate => candidate.identity === slot?.item);
    element.classList.toggle("occupied", Boolean(item));
    const name = element.querySelector(".hotbar-item");
    if (name) name.textContent = item?.name || "—";
    const count = element.querySelector(".hotbar-count");
    if (count) {
      count.dataset.inventoryCount = item?.identity || "";
      count.textContent = item && item.maximum_stack > 1 ? String(item.quantity) : "";
    }
  });
  shaderViewer.element?.querySelectorAll("[data-inventory-count]").forEach(element => {
    const item = hotbarState.inventory.items.find(candidate =>
      candidate.identity === element.dataset.inventoryCount);
    element.textContent = item && item.maximum_stack > 1 ? String(item.quantity) : "";
  });
  shaderViewer.element?.querySelectorAll("[data-placement-recipe]").forEach(element => {
    const recipe = model.placement.recipes.find(candidate =>
      candidate.identity === element.dataset.placementRecipe);
    element.textContent = `${recipe.name} ×${recipe.stock}`;
    element.classList.toggle("active", placementState.selectedRecipe === recipe.identity);
  });
}

function selectPlacementRecipe(identity) {
  const recipe = model.placement?.recipes.find(candidate => candidate.identity === identity);
  if (!recipe || recipe.stock <= 0) return false;
  placementState.selectedRecipe = identity;
  if (placementState.payload?.box) {
    placementState.payload.box.placement.custody = "inventory";
    rebuildPortableSceneMesh();
  }
  placementState.payload = null;
  refreshInventoryCounts();
  setPlacementStatus(recipe.opening_kind === "portal"
    ? `${recipe.name}: left action places IN · right action places OUT`
    : `${recipe.name}: aim at a boundary and use primary action`);
  return true;
}

function placementPreviewCenter(box) {
  const facing = normalized3(shaderViewer.cameraFacing || [0, 0, -1]);
  const camera = shaderViewer.cameraPosition || [box.center[0], 0, box.center[1]];
  let center = [camera[0] + facing[0] * 1.8 + placementState.offsets.x,
    camera[2] + facing[2] * 1.8 + placementState.offsets.z];
  const mode = placementState.snapMode || "object-face";
  if (mode === "grid") {
    const step = Number(model.placement.grid_step);
    center = center.map(value => Math.round(value / step) * step);
  }
  const candidates = shaderViewer.geometry.filter(candidate => candidate !== box &&
    candidate.placement?.custody !== "inventory" && candidate.kind !== "world-envelope");
  if ((mode === "object-center" || mode === "object-face") && candidates.length) {
    const target = candidates.map(candidate => ({candidate, distance: Math.hypot(
      candidate.center[0] - center[0], candidate.center[1] - center[1])}))
      .sort((left, right) => left.distance - right.distance)[0];
    if (target.distance <= Number(model.placement.snap_distance) +
        Math.max(...target.candidate.half_extent)) {
      if (mode === "object-center") center = [...target.candidate.center];
      else {
        const dx = center[0] - target.candidate.center[0];
        const dz = center[1] - target.candidate.center[1];
        if (Math.abs(dx) >= Math.abs(dz)) center[0] = target.candidate.center[0] +
          Math.sign(dx || 1) * (target.candidate.half_extent[0] + box.half_extent[0]);
        else center[1] = target.candidate.center[1] + Math.sign(dz || 1) *
          (target.candidate.half_extent[1] + box.half_extent[1]);
      }
    }
  }
  return center;
}

function takeFocusedObjectToInventory(targetIdentity) {
  const box = shaderViewer.geometry.find(candidate => candidate.identity === targetIdentity);
  if (!box || box.kind === "world-envelope") return false;
  const semanticOwner = box.parent_identity || model.identity;
  box.placement = {...(box.placement || {}), custody: "inventory",
    semantic_owner: semanticOwner, source_container: box.spatial_container || semanticOwner,
    original_center: [...box.center]};
  let item = hotbarState.inventory.items.find(candidate => candidate.entity === box.identity);
  if (!item) {
    const slot = hotbarState.model.slots.find(candidate => !candidate.item)?.number || null;
    item = {identity: `${hotbarState.inventory.identity}/held:${box.identity}`,
      entity: box.identity, name: box.label || box.identity, is_tool: false, slot,
      quantity: 1, maximum_stack: 1, stack_key: null,
      properties: {operation: "place-preserved-object",
        tool: model.placement.identity.replace("/placement", "/tools/placement")}};
    hotbarState.inventory.items.push(item);
    if (slot) hotbarState.model.slots.find(candidate => candidate.number === slot).item = item.identity;
  } else item.quantity = 1;
  box.placement.custody = "preview";
  box.center = placementPreviewCenter(box);
  placementState.payload = {identity: box.identity, box, item,
    semantic_owner: semanticOwner, source_container: box.placement.source_container};
  placementState.selectedRecipe = null;
  rebuildPortableSceneMesh(); refreshInventoryCounts();
  setPlacementStatus(`previewing ${item.name} · owner remains ${semanticOwner}`);
  return true;
}

function commitPlacementPayload() {
  const payload = placementState.payload;
  if (!payload?.box) return false;
  payload.box.placement.custody = "placed";
  payload.box.placement.rotation = [0, placementState.offsets.yaw, 0];
  payload.box.placement.elevation = placementState.offsets.y;
  payload.item.quantity = 0;
  const worldObject = model.world.objects.find(candidate => candidate.identity === payload.identity);
  if (worldObject) {
    worldObject.transform.position = [payload.box.center[0], placementState.offsets.y,
      payload.box.center[1]];
    worldObject.transform.placed_in = payload.source_container;
  }
  placementState.payload = null;
  shaderViewer.revision += 1; model.scene_mesh.revision = shaderViewer.revision;
  turingWorldRevisions.publish({revision: shaderViewer.revision,
    identity: payload.identity, operation: "place-preserved-identity"});
  rebuildPortableSceneMesh(); refreshInventoryCounts(); saveLivingEdits(payload.identity);
  setPlacementStatus(`placed ${payload.identity} · semantic ownership preserved`);
  return true;
}

function placeSubtractiveRecipe(targetIdentity, requestedPortalRole = null) {
  const recipe = model.placement?.recipes.find(candidate =>
    candidate.identity === placementState.selectedRecipe);
  if (recipe?.opening_kind === "portal") return placePortalSplat(requestedPortalRole);
  const host = shaderViewer.geometry.find(candidate => candidate.identity === targetIdentity);
  if (!recipe) { setPlacementStatus("no placement recipe is armed"); return false; }
  if (recipe.stock <= 0) {
    setPlacementStatus(`${recipe.name} stock is empty · use Return to defaults to replenish it`);
    return false;
  }
  if (!host) {
    setPlacementStatus(`${recipe.name}: aim the crosshair at a hollow boundary wall`);
    return false;
  }
  if (!placementBoundaryHost(host)) {
    setPlacementStatus(`${recipe.name} cannot cut ${host.label || host.identity} · aim at a room, building, or courtyard wall`);
    return false;
  }
  const camera = shaderViewer.cameraPosition || [host.center[0], 0, host.center[1] - 1];
  const dx = camera[0] - host.center[0], dz = camera[2] - host.center[1];
  const side = Math.abs(dx) > Math.abs(dz) ? (dx > 0 ? "east" : "west") :
    (dz > 0 ? "north" : "south");
  const halfLength = side === "north" || side === "south"
    ? host.half_extent[0] : host.half_extent[1];
  const axisOffset = side === "north" || side === "south"
    ? placementState.offsets.x : placementState.offsets.z;
  const offset = Math.max(-halfLength + recipe.width * 0.5,
    Math.min(halfLength - recipe.width * 0.5, axisOffset));
  const identity = `${host.identity}/opening:${recipe.opening_kind}:placed-${++placementState.sequence}`;
  const opening = {
    identity, kind: recipe.opening_kind, side, offset, width: recipe.width,
    bottom: recipe.opening_kind === "window" ? 0.56 + placementState.offsets.y : 0,
    height: recipe.height, semantic_owner: host.identity,
    placement: {custody: "placed", placement_kind: "subtractive",
      recipe: recipe.identity, yaw: placementState.offsets.yaw},
  };
  host.openings = [...(host.openings || []), opening];
  index.set(identity, opening);
  const worldHost = model.world.objects.find(candidate => candidate.identity === host.identity);
  if (worldHost) {
    worldHost.form.openings = [...(worldHost.form.openings || []), {...opening}];
    worldHost.semantic_parts = [...(worldHost.semantic_parts || []), {
      identity, role: "opening", opening_kind: recipe.opening_kind, side,
      material_role: "void",
    }];
  }
  recipe.stock -= 1;
  const inventoryItem = hotbarState.inventory.items.find(candidate =>
    candidate.properties?.recipe === recipe.identity);
  if (inventoryItem) inventoryItem.quantity = recipe.stock;
  shaderViewer.revision += 1; model.scene_mesh.revision = shaderViewer.revision;
  turingWorldRevisions.publish({revision: shaderViewer.revision, identity,
    operation: "place-subtractive-opening", host: host.identity});
  rebuildPortableSceneMesh(); refreshInventoryCounts(); saveLivingEdits(host.identity);
  setPlacementStatus(`placed ${recipe.opening_kind} in ${host.label || host.identity} · ${recipe.stock} left`);
  return true;
}

function placePortalSplat(portRole) {
  const recipe = model.placement?.recipes.find(candidate =>
    candidate.identity === placementState.selectedRecipe);
  if (recipe?.opening_kind !== "portal" || !["in", "out"].includes(portRole)) {
    setPlacementStatus("arm portal stock, then use left action for IN or right action for OUT");
    return false;
  }
  if (recipe.stock <= 0) {
    setPlacementStatus(`${recipe.name} stock is empty · use Return to defaults to replenish it`);
    return false;
  }
  const hit = raySceneTriangle();
  if (!hit) {
    setPlacementStatus(`${portRole.toUpperCase()}: aim the crosshair at a rendered mesh triangle`);
    return false;
  }
  let tangent = normalized3(cross3(hit.normal, [0, 1, 0]));
  if (Math.hypot(...tangent) < 0.1) {
    const facing = normalized3(shaderViewer.cameraFacing || [0, 0, -1]);
    tangent = normalized3(cross3(facing, hit.normal));
  }
  if (Math.hypot(...tangent) < 0.1) tangent = [1, 0, 0];
  const bitangent = normalized3(cross3(hit.normal, tangent));
  const profile = activePortalPlacementProfile();
  const apertureScale = Math.max(1, Number(profile.aperture_scale || 1));
  const tubeScale = Math.max(1, Number(profile.tube_scale || apertureScale));
  const radius = Math.max(0.16, Number(recipe.width || 0.9) * 0.5) * apertureScale;
  const sequence = ++placementState.sequence;
  const identity = `${portalRuntime.graph.identity}/nodes/${portRole}-${sequence}`;
  const splat = {identity, port_role: portRole,
    port_set_identity: portalRuntime.graph.identity,
    backing: "probabilistic-tube-graph", backing_graph: portalRuntime.graph.identity,
    distribution: "normalized-spatial-gaussian",
    intermediary_manifold: "directed-tube-edge",
    tool_mode: profile.name, aperture_class: profile.aperture_class || "person",
    aperture_scale: apertureScale, tube_throat_radius: 0.045 * tubeScale,
    handle_scale: Math.max(1, Number(profile.handle_scale || apertureScale)),
    center: [...hit.position], normal: [...hit.normal], tangent, bitangent, radius,
    object_identity: hit.objectIdentity, part_identity: hit.partIdentity,
    triangle_memberships: portalTriangleMembership(hit, radius),
    division: {operation: "radial-chart-clipping",
      domain: "triangle-barycentric-subdomains", center_triangle: hit.triangleIndex,
      center_barycentric: [...hit.barycentric]},
    placement: {custody: "placed", placement_kind: "manifold", recipe: recipe.identity}};
  portalRuntime.splats.push(splat);
  index.set(identity, splat);
  recipe.stock -= 1;
  const inventoryItem = hotbarState.inventory.items.find(candidate =>
    candidate.properties?.recipe === recipe.identity);
  if (inventoryItem) inventoryItem.quantity = recipe.stock;
  rebuildPortalGraph();
  shaderViewer.revision += 1; model.scene_mesh.revision = shaderViewer.revision;
  turingWorldRevisions.publish({revision: shaderViewer.revision, identity,
    operation: "deploy-portal-graph-node", object_identity: hit.objectIdentity,
    part_identity: hit.partIdentity, port_role: portRole,
    backing_graph: portalRuntime.graph.identity,
    triangle_memberships: splat.triangle_memberships.map(member => member.triangle_index)});
  rebuildPortableSceneMesh(); refreshInventoryCounts(); saveLivingEdits();
  const inputCount = portalRuntime.splats.filter(candidate => candidate.port_role === "in").length;
  const outputCount = portalRuntime.splats.length - inputCount;
  setPlacementStatus(`${portRole.toUpperCase()} splat deployed · ${inputCount} IN / ` +
    `${outputCount} OUT · ${profile.name} ${profile.aperture_class} aperture · ` +
    `${portalRuntime.graph.edges.length} relaxed quaternion tube${portalRuntime.graph.edges.length === 1 ? "" : "s"}`);
  return true;
}

function placementPrimary(target) {
  if (placementState.selectedRecipe) {
    const recipe = model.placement?.recipes.find(candidate =>
      candidate.identity === placementState.selectedRecipe);
    return recipe?.opening_kind === "portal"
      ? placePortalSplat("in") : placeSubtractiveRecipe(target);
  }
  if (placementState.payload) return commitPlacementPayload();
  return target ? takeFocusedObjectToInventory(target) : false;
}

function placementSecondary(target) {
  const recipe = model.placement?.recipes.find(candidate =>
    candidate.identity === placementState.selectedRecipe);
  if (recipe?.opening_kind === "portal") return placePortalSplat("out");
  cancelPlacement();
  return true;
}

function cancelPlacement() {
  if (placementState.payload?.box) {
    placementState.payload.box.placement.custody = "inventory";
    placementState.payload = null; rebuildPortableSceneMesh();
  }
  if (placementState.focusedIdentity) cancelPlacementFocus();
  placementState.selectedRecipe = null; refreshInventoryCounts();
  setPlacementStatus("placement idle · choose stock or pick a focused object");
}

function setPlacementAxis(axis, rawValue) {
  if (!(axis in placementState.offsets)) return;
  placementState.offsets[axis] = Number(rawValue) || 0;
  if (placementState.payload?.box) {
    placementState.payload.box.center = placementPreviewCenter(placementState.payload.box);
    rebuildPortableSceneMesh();
  } else if (placementState.focusedIdentity && placementState.focusBaseline) {
    const box=geometryBox(placementState.focusedIdentity), baseline=placementState.focusBaseline;
    box.center=[baseline.center[0]+placementState.offsets.x,
      baseline.center[1]+placementState.offsets.z];
    box.placement={...(baseline.placement || {}),rotation:[0,
      boxYawDegrees(baseline)+placementState.offsets.yaw,0],
      elevation:Number(baseline.placement?.elevation || 0)+placementState.offsets.y};
    rebuildPortableSceneMesh();
    setPlacementStatus(`transform preview · ${box.label || box.identity} · x ${placementState.offsets.x.toFixed(2)} · z ${placementState.offsets.z.toFixed(2)} · yaw ${boxYawDegrees(box).toFixed(1)}°`);
  }
}

function applyDepthMapTool(action){
  const tool=activeToolObject(),mode=activeToolMode(tool),hit=raySceneTriangle();
  if(tool?.name!=="Depth-map tool"||!mode)return false;
  const terrain=shaderViewer.geometry.find(box=>box.identity===hit?.objectIdentity&&
    box.surface?.kind==="sampled-height-field");
  if(!terrain){setPlacementStatus("depth map · aim at sampled terrain");return true;}
  if(mode.name==="middle"&&action==="secondary"){
    depthMapRuntime.brushRadius=Math.min(3.2,depthMapRuntime.brushRadius*1.22);
    const presentation=vehicleRuntime.active?.configuration?.presentation||
      model.vehicle_slot?.vehicles?.[0]?.configuration?.presentation;
    if(presentation){presentation.world_tile_size=Math.min(2.4,Number(presentation.world_tile_size||.35)*1.18);
      model.motion_cues.world_tiling.tile_size=presentation.world_tile_size;}
    const tileLocation=shaderViewer.locations?.uWorldTileSize;
    if(shaderViewer.gl&&shaderViewer.program&&tileLocation!==null&&tileLocation!==undefined){
      shaderViewer.gl.useProgram(shaderViewer.program);shaderViewer.gl.uniform1f(tileLocation,Number(presentation?.world_tile_size||.35));}
    setPlacementStatus(`depth map · texture ${Number(presentation?.world_tile_size||0).toFixed(2)} · brush ${depthMapRuntime.brushRadius.toFixed(2)}`);
    return true;
  }
  const surface=terrain.surface,[columns,rows]=surface.resolution,[cellX,cellZ]=surface.cell_size,
    radius=depthMapRuntime.brushRadius,middle=Number(surface.middle_height??.12);
  let changed=0;
  for(let row=0;row<rows;row+=1)for(let column=0;column<columns;column+=1){
    const index=row*columns+column,x=surface.origin[0]+column*cellX,z=surface.origin[2]+row*cellZ,
      distance=Math.hypot(x-hit.position[0],z-hit.position[2]);if(distance>radius)continue;
    const falloff=(1-distance/radius)**2,current=Number(surface.heights[index]);let next=current;
    if(mode.name==="sculpt")next=current+(action==="secondary"?1:-1)*depthMapRuntime.step*falloff;
    else next=current+(middle-current)*.48*falloff;
    surface.heights[index]=Math.max(depthMapRuntime.minimumHeight,Math.min(depthMapRuntime.maximumHeight,next));changed+=1;
  }
  if(!changed)return true;
  terrain.floor_height=Math.min(...surface.heights);terrain.height=Math.max(...surface.heights);
  shaderViewer.revision+=1;model.scene_mesh.revision=shaderViewer.revision;
  turingWorldRevisions.publish({revision:shaderViewer.revision,identity:terrain.identity,
    operation:mode.name==="sculpt"?(action==="secondary"?"raise-depth":"lower-depth"):"relax-depth-to-middle"});
  rebuildPortableSceneMesh();saveLivingEdits(terrain.identity);
  setPlacementStatus(`depth map · ${mode.name} · ${changed} samples · brush ${radius.toFixed(2)}`);return true;
}

function routeActiveToolHook(action, position = null) {
  const tool = activeToolObject();
  const hook = tool?.hooks.find(candidate => candidate.action === action);
  if (!tool || !hook) return false;
  const target = pickCrosshairIdentity()?.box.identity || null;
  const actor = viewportControls.policy?.actor || model.identity;
  const destination = hook.destination === "focused-identity" ? (target || model.identity) : hook.destination;
  const edge = actionEdges.register(tool.identity, hook.operation, destination);
  abstractUISystemTimer.issue({actor, tool: tool.identity, type: hook.operation,
    destination, edge, issued_at: performance.now()});
  if (hook.operation === "open-dialogue" && target) openToolDialogue(tool, target);
  if (hook.operation === "placement-primary") placementPrimary(target);
  if (hook.operation === "placement-secondary") placementSecondary(target);
  if (hook.operation === "fire-projectile") firePhysicsBall();
  if (hook.operation === "collect-projectile-pickup") collectNearbyProjectile(true);
  if (hook.operation === "depth-map-primary") applyDepthMapTool("primary");
  if (hook.operation === "depth-map-secondary") applyDepthMapTool("secondary");
  if (hook.operation === "toggle-focus-context") {
    const mode = toggleControlFocus();
    if (mode === "projected-pointer") {
      if (position) openCrosshairContextMenu(position[0], position[1]);
      else openCrosshairContextMenuAtViewportCenter();
    } else closeSceneContextMenu();
  }
  return true;
}

function turretBallisticDirection(origin,target,speed){
  const dx=target[0]-origin[0],dz=target[2]-origin[2],horizontal=Math.hypot(dx,dz),dy=target[1]-origin[1],
    gravity=9.81,v=Math.max(1,Number(speed||1)),root=v**4-gravity*(gravity*horizontal**2+2*dy*v**2);
  if(horizontal<1e-5)return{direction:[0,Math.sign(dy)||1,0],reachable:true};
  if(root<0)return{direction:normalized3([dx,dy,dz]),reachable:false};
  const angle=Math.atan((v*v-Math.sqrt(root))/(gravity*horizontal)),cosine=Math.cos(angle);
  return{direction:[dx/horizontal*cosine,Math.sin(angle),dz/horizontal*cosine],reachable:true};
}

function turretFriendlyRayEntry(origin,direction,targetDistance,turretName){
  const own=[vehicleRuntime.box,...vehicleRuntime.bodyShellBoxes].filter(box=>box&&
    box.turret_name!==turretName&&box.turret_part!=="post"&&box.turret_part!=="gimbal"&&box.turret_part!=="weapon");
  for(const box of own){const distance=rayBoxDistance(origin,direction,box);
    if(distance!==null&&distance>.14&&distance<targetDistance-.06)return{identity:box.identity,distance};}
  const actor=viewportControls.policy?.actor;
  for(const runtime of entityState.values()){
    if(!runtime?.position||runtime.entity?.traits?.projectile)continue;
    if(runtime.entity?.principal!==actor&&runtime.entity?.identity!==actor&&!runtime.entity?.traits?.friendly)continue;
    const center=runtime.position,offset=origin.map((value,index)=>value-center[index]),along=offset.reduce(
      (sum,value,index)=>sum+value*direction[index],0),radius=Number(runtime.entity?.geometry?.parameters?.radius||.28),
      discriminant=along*along-(offset.reduce((sum,value)=>sum+value*value,0)-radius*radius);
    if(discriminant>=0){const distance=-along-Math.sqrt(discriminant);if(distance>.14&&distance<targetDistance-.06)
      return{identity:runtime.entity.identity,distance};}
  }
  return null;
}

function updateVehicleTurretTargeting(vehicle,state){
  const runtime=vehicleRuntime.turretSystem,assembly=vehicle.body_shells?.find(item=>item.identity==="five-turret-carrier");
  if(vehicleRuntime.bodyShell!==assembly?.identity){runtime.target=null;runtime.turrets=[];runtime.interlockReason="carrier-not-installed";return;}
  const viewOrigin=shaderViewer.cameraPosition||viewportControls.position,viewDirection=normalized3(shaderViewer.cameraFacing||[1,0,0]),
    hit=raySceneTriangle(viewOrigin,viewDirection),target=hit?.position||null,speed=Number(assembly.ammunition?.muzzle_speed_m_s||72);
  runtime.target=target;runtime.targetDistance=Number(hit?.distance||0);runtime.interlockReason=target?null:"no-surface-at-focus";
  runtime.turrets=(assembly.turrets||[]).map(turret=>{
    const offset=rotateVehiclePresentationVector(turret.local_position,state,0),pivot=state.position.map((value,index)=>value+offset[index]),
      solution=target?turretBallisticDirection(pivot,target,speed):{direction:rotateVehiclePresentationVector([1,0,0],state,0),reachable:false},
      direction=normalized3(solution.direction),yaw=Math.atan2(direction[2],direction[0]),pitch=Math.atan2(direction[1],Math.hypot(direction[0],direction[2])),
      relativeYaw=Math.atan2(Math.sin(yaw-state.yaw),Math.cos(yaw-state.yaw)),weapon=assembly.weapon||{},
      inGimbal=Math.abs(relativeYaw)<=Number(weapon.yaw_limit_degrees||175)*Math.PI/180&&
        pitch>=Number(weapon.pitch_min_degrees||-28)*Math.PI/180&&pitch<=Number(weapon.pitch_max_degrees||72)*Math.PI/180,
      muzzle=pivot.map((value,index)=>value+direction[index]*.34),friendly=target?
        turretFriendlyRayEntry(muzzle,direction,Math.hypot(...target.map((value,index)=>value-muzzle[index])),turret.identity):null,
      inhibited=!target||!solution.reachable||!inGimbal||Boolean(friendly),reason=!target?"no-target":!solution.reachable?
        "ballistic-range":!inGimbal?"gimbal-limit":friendly?`friendly:${friendly.identity}`:null;
    return{name:turret.identity,pivot,muzzle,direction,yaw,pitch,relativeYaw,inhibited,reason};
  });
  const blocked=runtime.turrets.filter(turret=>turret.inhibited);runtime.interlockReason=blocked.length?
    blocked.map(turret=>`${turret.name}:${turret.reason}`).join(" · "):null;
}

function fireVehicleTurrets(){
  const vehicle=vehicleRuntime.active,runtime=vehicleRuntime.turretSystem,
    assembly=vehicle?.body_shells?.find(item=>item.identity==="five-turret-carrier");
  if(!vehicle||vehicleRuntime.bodyShell!==assembly?.identity)return false;
  if(!vehicleRuntime.electrical.computerOnline){setPlacementStatus("turret computer offline · firing disengaged");return true;}
  updateVehicleTurretTargeting(vehicle,vehicleRuntime.state);
  const friendlyBlock=runtime.turrets.find(turret=>String(turret.reason||"").startsWith("friendly:"));
  if(friendlyBlock){setPlacementStatus(`friendly-fire interlock · whole volley disengaged · ${friendlyBlock.reason}`);return true;}
  const ready=runtime.turrets.filter(turret=>!turret.inhibited).slice(0,Math.max(0,runtime.ammoCount));
  if(!ready.length){setPlacementStatus(`turret interlock · ${runtime.interlockReason||"no ammunition"}`);return true;}
  const ammunition=assembly.ammunition||{},shots=[];
  for(const turret of ready){if(firePhysicsBall(1,{ammunitionAuthority:"vehicle-turret",origin:turret.muzzle,
      direction:turret.direction,speed:Number(ammunition.muzzle_speed_m_s||72),massKg:Number(ammunition.round_mass_kg||0),
      owner:vehicle.identity})){
    shots.push({turretIdentity:turret.name,localPosition:[...(assembly.turrets.find(item=>item.identity===turret.name)?.local_position||[0,0,0])],
      direction:[...turret.direction],recoilImpulseNs:Number(ammunition.recoil_impulse_n_s||49)});}}
  runtime.ammoCount=Math.max(0,runtime.ammoCount-shots.length);
  if(stateLoopRuntime.ready&&shots.length)stateLoopRuntime.worker.postMessage({type:"vehicle-body-wrenches",identity:vehicle.identity,
    assemblyIdentity:assembly.identity,shots,ammoCount:runtime.ammoCount,roundMassKg:Number(ammunition.round_mass_kg||0)});
  setPlacementStatus(`turret volley · ${shots.length} fired · ${runtime.ammoCount} rounds · ${runtime.interlockReason||"clear"}`);
  updateVehicleTransmissionControls();return true;
}

function beginViewportPrimary(position=null,source="pointer") {
  issueViewportAction("primary-action","press");
  if(vehicleRuntime.active&&vehicleRuntime.bodyShell==="five-turret-carrier"&&vehicleRuntime.turretSystem.fireTakeover){
    if(primaryActionState.down)return false;primaryActionState.down=true;primaryActionState.source=source;
    return fireVehicleTurrets();}
  const tool=activeToolObject(),mode=activeToolMode(tool);
  if(tool?.name!=="Physics-ball gun"||mode?.name!=="attractor"){
    return routeActiveToolHook("primary-action",position);
  }
  if(primaryActionState.down)return false;
  primaryActionState.down=true;primaryActionState.source=source;
  const target=pickCrosshairIdentity()?.box.identity||null;
  if(target&&projectilePickupState.has(target)){
    collectProjectilePickup(projectilePickupState.get(target));
    primaryActionState.down=false;primaryActionState.source=null;return true;
  }
  primaryActionState.targetIdentity=projectileState.has(target)?target:null;
  if(primaryActionState.targetIdentity){
    const state=projectileState.get(primaryActionState.targetIdentity);
    if(state.sleeping)wakeProjectilePhysics(state,"crosshair-attractor");
    setPlacementStatus(`crosshair pull · ${state.box.label} · hold primary to reel into ammunition`);
  }else setPlacementStatus("crosshair pull · aim directly at a physics ball");
  return Boolean(primaryActionState.targetIdentity);
}

function endViewportPrimary(source="pointer") {
  if(!primaryActionState.down)return false;
  primaryActionState.down=false;primaryActionState.source=null;primaryActionState.targetIdentity=null;
  issueViewportAction("primary-action","release");return true;
}

function updateHeldToolPrimary(dt) {
  if(!primaryActionState.down||!primaryActionState.targetIdentity||dt<=0)return;
  const tool=activeToolObject(),mode=activeToolMode(tool);
  if(tool?.name!=="Physics-ball gun"||mode?.name!=="attractor")return;
  const state=projectileState.get(primaryActionState.targetIdentity);
  const center=shaderViewer.cameraPosition||viewportControls.position;
  if(!state||!center){endViewportPrimary("target-lost");return;}
  const delta=center.map((value,index)=>value-state.position[index]);
  const distance=Math.hypot(...delta);if(distance<=projectileAttractorRuntime.absorptionRadius){
    const label=state.box.label;endViewportPrimary("absorbed");
    if(absorbProjectileIntoAmmo(state,"crosshair-attractor-absorption"))
      setPlacementStatus(`absorbed ${label} from crosshairs into ammunition`);
    return;
  }
  const field=projectileAttractorRuntime;
  const mass=Math.max(.001,Number(model.projectiles.archetype.physics.mass||.12));
  const force=field.targetStrength/(distance*distance+field.softening*field.softening);
  if(state.sleeping)wakeProjectilePhysics(state,"crosshair-attractor");
  const acceleration=Math.min(28,force/mass);
  state.velocity=state.velocity.map((value,index)=>value+delta[index]/distance*acceleration*dt);
  synchronizePortalBody(state.identity,state.position,state.velocity);
  setPlacementStatus(`crosshair pull · ${state.box.label} · ${distance.toFixed(2)} away · release primary to let go`);
}

function beginViewportSecondary(position=null,source="pointer") {
  if(secondaryActionState.down)return false;
  secondaryActionState.down=true;secondaryActionState.source=source;
  secondaryActionState.startedAt=performance.now();secondaryActionState.position=position;
  issueViewportAction("secondary-action","press");
  const tool=activeToolObject(),mode=activeToolMode(tool);
  if(tool?.name==="Physics-ball gun"&&mode?.name==="normal"){
    setPlacementStatus("charging projectile exit velocity · release secondary to fire");return true;
  }
  if(tool?.name==="Physics-ball gun"&&mode?.name==="attractor"){
    projectileAttractorRuntime.active=true;projectileAttractorRuntime.strength=0;
    projectileAttractorRuntime.effectiveRadius=0;projectileAttractorRuntime.absorbed=0;
    projectileAttractorRuntime.members.clear();
    setPlacementStatus("attractor charging · force radius emerges at epsilon cutoff");return true;
  }
  return routeActiveToolHook("secondary-action",position);
}

function endViewportSecondary(source="pointer") {
  if(!secondaryActionState.down)return false;
  const held=Math.max(0,(performance.now()-secondaryActionState.startedAt)/1000);
  secondaryActionState.down=false;secondaryActionState.source=null;
  issueViewportAction("secondary-action","release",held);
  const tool=activeToolObject(),mode=activeToolMode(tool);
  if(tool?.name==="Physics-ball gun"&&mode?.name==="normal"){
    const charge=Math.min(1,held/2),exitScale=.45+charge*1.75;
    firePhysicsBall(exitScale);
    setPlacementStatus(`charged launch · ${(exitScale*100).toFixed(0)}% exit velocity · ${held.toFixed(2)}s`);
  }else if(tool?.name==="Physics-ball gun"&&mode?.name==="attractor"){
    const members=projectileAttractorRuntime.members.size;
    const absorbed=projectileAttractorRuntime.absorbed;
    projectileAttractorRuntime.active=false;projectileAttractorRuntime.members.clear();
    setPlacementStatus(`attractor released · ${members} influenced · ${absorbed} absorbed into ammunition`);
  }
  return true;
}

function updateHeldToolSecondary(dt) {
  if(!secondaryActionState.down||dt<=0)return;
  const tool=activeToolObject(),mode=activeToolMode(tool);
  const held=Math.max(0,(performance.now()-secondaryActionState.startedAt)/1000);
  if(tool?.name==="Physics-ball gun"&&mode?.name==="normal"){
    const exitScale=.45+Math.min(1,held/2)*1.75;
    setPlacementStatus(`charging exit velocity · ${(exitScale*100).toFixed(0)}% · release to fire`);
    return;
  }
  if(tool?.name!=="Physics-ball gun"||mode?.name!=="attractor")return;
  const field=projectileAttractorRuntime;
  field.strength=Math.min(2.4,field.baseStrength+field.growthPerSecond*held);
  field.effectiveRadius=Math.sqrt(Math.max(0,field.strength/field.forceEpsilon-
    field.softening*field.softening));
  field.members.clear();
  const center=shaderViewer.cameraPosition||viewportControls.position;if(!center)return;
  const mass=Math.max(.001,Number(model.projectiles.archetype.physics.mass||.12));
  projectileState.forEach(state=>{
    const delta=center.map((value,index)=>value-state.position[index]);
    const distance=Math.hypot(...delta);if(distance<1e-4)return;
    const force=field.strength/(distance*distance+field.softening*field.softening);
    if(force<field.forceEpsilon)return;
    field.members.add(state.identity);
    if(distance<=field.absorptionRadius&&absorbProjectileIntoAmmo(
        state,"attractor-field-absorption")){field.absorbed+=1;return;}
    if(state.sleeping)wakeProjectilePhysics(state,"attractor-field-epsilon");
    const acceleration=Math.min(22,force/mass);
    state.velocity=state.velocity.map((value,index)=>value+delta[index]/distance*acceleration*dt);
    synchronizePortalBody(state.identity,state.position,state.velocity);
  });
  setPlacementStatus(`attractor · strength ${field.strength.toFixed(2)} · ε ${field.forceEpsilon.toFixed(3)} · derived radius ${field.effectiveRadius.toFixed(2)} · ${field.members.size} members · ${field.absorbed} absorbed`);
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

function mobileClamp(value, minimum = -1, maximum = 1) {
  return Math.max(minimum, Math.min(maximum, Number(value) || 0));
}

function refreshMobileTilt() {
  const state = mobileControlState;
  if (!state.motionEnabled || !state.baseline) {
    state.tilt = [0, 0]; return;
  }
  const betaDelta = state.orientation ? state.baseline.beta - state.orientation.beta : 0;
  const gammaDelta = state.orientation ? state.orientation.gamma - state.baseline.gamma : 0;
  const accelerationX = state.acceleration && state.baseline.acceleration
    ? (state.acceleration.x - state.baseline.acceleration.x) / 18 : 0;
  const accelerationY = state.acceleration && state.baseline.acceleration
    ? (state.baseline.acceleration.y - state.acceleration.y) / 18 : 0;
  state.tilt = [mobileClamp(betaDelta / 24 + accelerationY * 0.18),
    mobileClamp(gammaDelta / 24 + accelerationX * 0.18)];
}

function calibrateMobileMotion() {
  const state = mobileControlState;
  state.baseline = state.orientation || state.acceleration ? {
    beta: state.orientation?.beta || 0, gamma: state.orientation?.gamma || 0,
    acceleration: state.acceleration ? {...state.acceleration} : null} : null;
  refreshMobileTilt();
  if (state.statusElement) state.statusElement.textContent = state.baseline
    ? "motion centered · sensor telemetry saved for later use"
    : "waiting for the first orientation sample…";
}

function installMobileMotionListeners() {
  if (mobileControlState.listenersInstalled) return;
  mobileControlState.listenersInstalled = true;
  window.addEventListener("deviceorientation", event => {
    if (!Number.isFinite(event.beta) || !Number.isFinite(event.gamma)) return;
    mobileControlState.orientation = {beta: Number(event.beta), gamma: Number(event.gamma)};
    if (!mobileControlState.baseline) calibrateMobileMotion();
    else refreshMobileTilt();
  }, {passive: true});
  window.addEventListener("devicemotion", event => {
    const acceleration = event.accelerationIncludingGravity || event.acceleration;
    if (!acceleration) return;
    mobileControlState.acceleration = {x: Number(acceleration.x || 0),
      y: Number(acceleration.y || 0), z: Number(acceleration.z || 0)};
    if (!mobileControlState.baseline) calibrateMobileMotion();
    else refreshMobileTilt();
  }, {passive: true});
}

async function requestMobileMotionControls() {
  const state = mobileControlState;
  if (state.motionEnabled) { state.baseline = null; calibrateMobileMotion(); return true; }
  if (!window.isSecureContext) {
    if (state.statusElement) state.statusElement.textContent =
      "motion sensors require HTTPS (or localhost); touch controls still work";
    return false;
  }
  try {
    const permissionRequests = [];
    const orientationAPI = window.DeviceOrientationEvent;
    const motionAPI = window.DeviceMotionEvent;
    if (typeof orientationAPI?.requestPermission === "function") {
      permissionRequests.push(orientationAPI.requestPermission());
    }
    if (typeof motionAPI?.requestPermission === "function") {
      permissionRequests.push(motionAPI.requestPermission());
    }
    const permissions = await Promise.all(permissionRequests);
    if (permissions.some(permission => permission !== "granted")) {
      throw new Error("sensor permission was not granted");
    }
    if (!orientationAPI && !motionAPI) throw new Error("this browser exposes no motion sensors");
    state.motionEnabled = true;
    installMobileMotionListeners();
    state.motionButton?.classList.add("enabled");
    if (state.motionButton) state.motionButton.textContent = "Recalibrate tilt";
    if (state.statusElement) state.statusElement.textContent =
      "motion enabled · telemetry only; movement remains on the touch stick";
    setViewportControlHighlight(true);
    return true;
  } catch (error) {
    if (state.statusElement) state.statusElement.textContent =
      `motion unavailable · ${error.message} · touch controls still work`;
    return false;
  }
}

function bindMobileStick(element, channel) {
  const knob = element.querySelector(".mobile-stick-knob");
  let activePointer = null;
  const updatePoint = (clientX,clientY) => {
    const bounds = element.getBoundingClientRect();
    const radius = Math.max(1, bounds.width * 0.5 - 22);
    let x = mobileClamp((clientX - (bounds.left + bounds.width * 0.5)) / radius);
    let y = mobileClamp((clientY - (bounds.top + bounds.height * 0.5)) / radius);
    const length = Math.hypot(x, y);
    if (length > 1) { x /= length; y /= length; }
    mobileControlState[channel] = [x, y];
    knob.style.transform = `translate(${x * radius}px,${y * radius}px)`;
  };
  const release = event => {
    if (event.pointerId !== activePointer) return;
    activePointer = null; mobileControlState[channel] = [0, 0];
    knob.style.transform = "translate(0,0)";
  };
  if("PointerEvent" in window){
    element.addEventListener("pointerdown", event => {
      event.preventDefault();event.stopPropagation();activePointer=event.pointerId;
      try{element.setPointerCapture?.(event.pointerId);}catch(_){}
      setViewportControlHighlight(true);updatePoint(event.clientX,event.clientY);
    });
    element.addEventListener("pointermove", event => {
      if(event.pointerId===activePointer)updatePoint(event.clientX,event.clientY);
    });
    element.addEventListener("pointerup",release);element.addEventListener("pointercancel",release);
    element.addEventListener("lostpointercapture",release);
  }else{
    const touch=(event,ending=false)=>{const touches=[...event.changedTouches],point=touches.find(item=>
      activePointer===null||item.identifier===activePointer);if(!point)return;
      event.preventDefault();event.stopPropagation();
      if(activePointer===null){activePointer=point.identifier;setViewportControlHighlight(true);}
      if(ending){activePointer=null;mobileControlState[channel]=[0,0];knob.style.transform="translate(0,0)";}
      else updatePoint(point.clientX,point.clientY);};
    element.addEventListener("touchstart",event=>touch(event),{passive:false});
    element.addEventListener("touchmove",event=>touch(event),{passive:false});
    element.addEventListener("touchend",event=>touch(event,true),{passive:false});
    element.addEventListener("touchcancel",event=>touch(event,true),{passive:false});
  }
}

function renderMobileControls() {
  const controls = div("mobile-controls");
  controls.setAttribute("aria-label", "Mobile movement controls");
  const motion = div("mobile-motion");
  const motionButton = document.createElement("button");
  motionButton.type = "button"; motionButton.className = "mobile-motion-button";
  motionButton.textContent = "Enable motion sensors";
  const motionStatus = div("mobile-motion-status", "touch sticks are ready · sensors are optional telemetry");
  motionButton.addEventListener("pointerdown", event => event.stopPropagation());
  motionButton.addEventListener("click", event => {
    event.preventDefault(); event.stopPropagation(); void requestMobileMotionControls();
  });
  motion.append(motionButton, motionStatus);
  const stick = (channel, label) => {
    const pad = div(`mobile-stick ${channel}`); pad.dataset.label = label;
    pad.append(div("mobile-stick-knob")); bindMobileStick(pad, channel); return pad;
  };
  const actions = div("mobile-actions");
  [["fire / use", "primary-action", ""], ["alt / OUT", "secondary-action", ""],
   ["jump", "jump", "jump"]].forEach(([label, action, className]) => {
    const button = document.createElement("button"); button.type = "button";
    button.className = `mobile-action ${className}`; button.textContent = label;
    button.addEventListener("pointerdown", event => {
      event.preventDefault(); event.stopPropagation(); setViewportControlHighlight(true);
      if (action === "jump") requestViewportJump();
      else if(action==="secondary-action")beginViewportSecondary(null,"mobile");
      else if(action==="primary-action")beginViewportPrimary(null,"mobile");
      else { issueViewportAction(action); routeActiveToolHook(action); }
    });
    if(action==="secondary-action"||action==="primary-action"){
      const release=event=>{event.preventDefault();event.stopPropagation();
        if(action==="secondary-action")endViewportSecondary("mobile");
        else endViewportPrimary("mobile");};
      button.addEventListener("pointerup",release);button.addEventListener("pointercancel",release);
    }
    actions.append(button);
  });
  controls.append(motion, stick("move", "move"), stick("look", "look"), actions);
  mobileControlState.element = controls; mobileControlState.motionButton = motionButton;
  mobileControlState.statusElement = motionStatus;
  return controls;
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

function setShaderOnlyMode(active) {
  shaderViewer.shaderOnly = Boolean(active);
  const button = shaderViewer.shaderOnlyToggle;
  if (button) {
    (shaderViewer.shaderOnly ? shaderViewer.element : shaderViewer.telemetry)?.append(button);
  }
  document.body.classList.toggle("shader-only", shaderViewer.shaderOnly);
  if(vehicleRuntime.contactMonitor){
    if(shaderViewer.shaderOnly)vehicleRuntime.contactMonitor.classList.remove("expanded");
    vehicleRuntime.contactMonitor.hidden=shaderViewer.shaderOnly||!vehicleRuntime.active;
  }
  if (button) {
    button.setAttribute("aria-pressed", String(shaderViewer.shaderOnly));
    button.textContent = shaderViewer.shaderOnly ? "Exit shader only" : "Shader only";
    button.title = shaderViewer.shaderOnly
      ? "Restore the living data map (Escape)"
      : "Hide the page and fill it with the first-person shader";
  }
  if (!shaderViewer.shaderOnly && document.pointerLockElement === shaderViewer.canvas) {
    document.exitPointerLock?.();
  }
  requestAnimationFrame(() => updateShaderViewer());
}

function armBrowserFullscreenOnFirstGesture() {
  const policy=model.vehicle_slot?.initial_state;if(policy?.browser_fullscreen!=="request-on-first-user-gesture")return;
  const cleanup=()=>{window.removeEventListener("pointerdown",enter,true);window.removeEventListener("keydown",enter,true);};
  const enter=event=>{if(event.type==="keydown"&&event.code==="Escape")return;if(document.fullscreenElement){cleanup();return;}
    cleanup();const request=document.documentElement.requestFullscreen?.({navigationUI:"hide"});
    if(request?.catch)request.catch(error=>reportRuntimeFault("initial-browser-fullscreen",error));};
  window.addEventListener("pointerdown",enter,true);window.addEventListener("keydown",enter,true);
}

function initializeVehicleFirstExperience() {
  const policy=model.vehicle_slot?.initial_state,identity=policy?.mounted_vehicle,
    vehicle=model.vehicle_slot?.vehicles?.find(candidate=>candidate.identity===identity);if(!vehicle)return false;
  const inventoryItem=hotbarState.inventory.items.find(item=>item.properties?.vehicle===identity);
  const mounted=setActiveVehicle(identity,{placeAtActor:policy.placement==="at-player-spawn",inventoryItem});
  if(!mounted)return false;
  if(policy.presentation==="full-viewport-driving")setShaderOnlyMode(true);
  setViewportControlHighlight(true);shaderViewer.active=true;
  setPlacementStatus(`${vehicle.name} initial drive · V dismount · R respawns · fullscreen is user-controlled`);return true;
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

function issueViewportAction(type,phase="trigger",duration=0) {
  const destination = viewportControls.policy?.actor;
  if (!destination) return;
  const edge = actionEdges.register(model.viewer.identity, type, destination);
  abstractUISystemTimer.issue({
    actor: destination, type, phase, duration, destination, edge, issued_at: performance.now()
  });
}

async function initializeWorldPhysicsWasm() {
  if (physicsRuntime.instance || stateLoopRuntime.worker || physicsRuntime.pending || !model.physics_program) return;
  const plugin = model.world.plugins.find(item => item.identity === model.physics_program.plugin);
  if (!plugin) throw new Error("symbolic world physics plugin is missing");
  physicsRuntime.pending = true;
  try {
    const deployment = model.loop_deployment;
    const workerPlan = deployment?.workers?.find(item => item.target === "javascript-worker");
    const module = model.world.wasm_modules.find(item => item.content_key === plugin.module);
    const vehicle=model.vehicle_slot?.vehicles?.[0];
    if (workerPlan && module && typeof Worker === "function" && typeof Blob === "function") {
      const moduleBytes=descriptor=>{const raw=atob(descriptor.binary_base64),bytes=new Uint8Array(raw.length);
        for(let index=0;index<raw.length;index+=1)bytes[index]=raw.charCodeAt(index);return bytes;},
        bytes=moduleBytes(module),fallbackDescriptor=capability=>{const fallbackPlugin=
          model.world.plugins.find(item=>item.capability===capability),fallbackModule=fallbackPlugin&&
          model.world.wasm_modules.find(item=>item.content_key===fallbackPlugin.module);return fallbackPlugin&&fallbackModule?{
            wasm:moduleBytes(fallbackModule),abi:fallbackPlugin.abi,entrypoint:fallbackPlugin.entrypoint}:null;},
        vehicleWasm=fallbackDescriptor("vehicle-physics"),contactWasm=fallbackDescriptor("vehicle-contact-fallback");
      stateLoopRuntime.workerUrl = URL.createObjectURL(new Blob([workerPlan.source],
        {type: "text/javascript"}));
      const worker = new Worker(stateLoopRuntime.workerUrl);
      stateLoopRuntime.worker = worker;
      worker.onmessage = event => {
        const message = event.data || {};
        if (message.type === "ready") {
          if (message.snapshotCapacity !== stateLoopRuntime.snapshotCapacity ||
              message.stride !== stateLoopRuntime.snapshotStride) {
            throw new Error("physics worker snapshot ABI mismatch");
          }
          stateLoopRuntime.ready = true; stateLoopRuntime.mode = "dedicated-worker";
          vehicleRuntime.computeMode=message.vehicleCompute||"resident-webgpu-fault";
          worker.postMessage({type: "colliders", colliders: shaderViewer.colliders});
          registerPlayerPhysicsBody();
          registerActiveVehiclePhysicsBody();
        } else if(message.type==="vehicle-gpu-error"){
          vehicleRuntime.error=message.error;vehicleRuntime.computeMode="resident-webgpu-fault";
          worker.postMessage({type:"vehicle-disable-gpu",reason:message.error||"resident WebGPU runtime fault"});
        } else if(message.type==="vehicle-gpu-recovered"){
          vehicleRuntime.error=null;vehicleRuntime.computeMode="resident-webgpu-graph";
        } else if(message.type==="vehicle-wasm-fallback"){
          vehicleRuntime.error=null;vehicleRuntime.computeMode="resident-wasm-fallback";
        } else if(message.type==="vehicle-dyno-result"){
          vehicleRuntime.dyno=message.result||null;updateVehicleTransmissionControls();
        } else if(message.type==="vehicle-damage"){
          vehicleRuntime.damage=message.damage||null;const vehicle=vehicleRuntime.active;
          if(vehicle&&vehicleRuntime.damage){for(const edge of vehicle.physics.mechanical_graph.edges){const state=
              vehicleRuntime.damage.members?.[edge.identity];if(state){edge.runtime_rest_length=state.restLength;
                edge.runtime_failed=Boolean(state.failed);}}
            const shellPrefix=vehicleRuntime.bodyShell==="five-turret-carrier"?"armor.mount.":"body_shell.mount.",
              shellMounts=vehicle.physics.mechanical_graph.edges.filter(edge=>edge.identity.startsWith(shellPrefix));
            if(shellMounts.length&&shellMounts.every(edge=>vehicleRuntime.damage.members?.[edge.identity]?.failed))
              vehicleRuntime.bodyShellBoxes=[];
            if(vehicleRuntime.state){updateVehiclePresentation(vehicle,vehicleRuntime.state,0,0);
              rebuildPortableSceneMesh({dynamicOnly:true});}updateVehicleTransmissionControls();}
        } else if(message.type==="vehicle-energy"){
          if(message.identity===(vehicleRuntime.active?.identity||vehicleRuntime.parkedState?.identity)){
            vehicleRuntime.electrical={...vehicleRuntime.electrical,...(message.energy||{})};
            vehicleRuntime.driverAssistance={...vehicleRuntime.driverAssistance,...(message.driverAssistance||{})};
            updateVehicleTransmissionControls();}
        } else if(message.type==="vehicle-alignment"){
          if(message.identity===vehicleRuntime.active?.identity){vehicleRuntime.wheelAlignment=message.alignment||vehicleRuntime.wheelAlignment;
            updateVehicleTransmissionControls();}
        } else if (message.type === "engine-state") {
          stateLoopRuntime.engineStage=message.stage||(
            message.sleeping?"asleep":"full-dynamics");
          stateLoopRuntime.engineSleeping=Boolean(message.sleeping);
          stateLoopRuntime.engineSleepReason=message.reason||null;
        } else if (message.type === "snapshot-buffer") {
          if (message.sequence > stateLoopRuntime.latestSequence) {
            stateLoopRuntime.latestSequence = message.sequence;
            const values = new Float64Array(message.buffer), stride = stateLoopRuntime.snapshotStride;
            stateLoopRuntime.slotRecords.forEach((body, slot) => {
              if (!body) return;
              const offset = slot * stride;
              if (values[offset] !== 1 || values[offset + 1] !== body.generation) return;
              body.position[0] = values[offset + 2]; body.position[1] = values[offset + 3];
              body.position[2] = values[offset + 4]; body.velocity[0] = values[offset + 5];
              body.velocity[1] = values[offset + 6]; body.velocity[2] = values[offset + 7];
              body.roll = values[offset + 8]; body.pitch = values[offset + 9]; body.yaw = values[offset + 10];
              body.rollVelocity = values[offset + 11]; body.pitchVelocity = values[offset + 12];
              body.yawVelocity = values[offset + 13]; body.contactRuntimePartId = values[offset + 14];
              body.springForces = Array.from(values.slice(offset + 15, offset + 19));
              body.contactAreas = Array.from(values.slice(offset + 19, offset + 23));
              body.frictionUtilizations = Array.from(values.slice(offset + 23, offset + 27));
              body.contactModes = Array.from(values.slice(offset + 27, offset + 31));
              body.compressions = Array.from(values.slice(offset + 31, offset + 35));
              body.wheelOmegas = Array.from(values.slice(offset + 35, offset + 39));
              body.tractionScales = Array.from(values.slice(offset + 39, offset + 43));
              body.brakeScales = Array.from(values.slice(offset + 43, offset + 47));
              body.powertrain={engineTorque:values[offset+47],clutchTorque:values[offset+48],
                transmissionOutputTorque:values[offset+49],drivelineTorque:values[offset+50],
                frontDifferentialTorque:values[offset+51],rearDifferentialTorque:values[offset+52],
                engineAccelerationTorque:values[offset+53],engineAngularAcceleration:values[offset+54],
                reactionTorque:Array.from(values.slice(offset+55,offset+58)),
                mountTorque:Array.from(values.slice(offset+58,offset+61)),
                engineAngularSpeed:values[offset+69],engineRPM:values[offset+70]};
              Object.assign(body.powertrain,{tractionControlDissipationTorque:values[offset+143],
                serviceBrakeReactionTorque:values[offset+144],rollingResistanceReactionTorque:values[offset+145],
                tireContactReactionTorque:values[offset+146],drivetrainChassisReactionTorque:values[offset+147]});
              body.transmission={gear:Math.max(1,values[offset+61]),displayGear:values[offset+62],
                mode:values[offset+63]===1?"automatic":"manual"};
              body.snapshotControlGeneration = values[offset + 64];
              body.damperScales = Array.from(values.slice(offset + 65, offset + 69));
              body.radialProbePenetrations=Array.from({length:4},(_,wheel)=>
                Array.from(values.slice(offset+71+wheel*15,offset+86+wheel*15)));
              body.radialProbeActiveCounts=Array.from(values.slice(offset+131,offset+135)).map(Math.round);
              Object.assign(body.transmission,{lowRange:values[offset+135]===1,frontDiffLock:values[offset+136]===1,
                rearDiffLock:values[offset+137]===1,centerDiffLock:values[offset+138]===1,
                frontDriveShare:values[offset+139],tractionControlEnabled:values[offset+140]===1,
                absEnabled:values[offset+141]===1,smoothLaunch:values[offset+142]===1,
                tractionControlAuthority:values[offset+148],absAuthority:values[offset+149],
                frontDiffMode:values[offset+150]>=.7?"locked":values[offset+150]>.01?"limited-slip":"open",
                rearDiffMode:values[offset+151]>=.7?"locked":values[offset+151]>.01?"limited-slip":"open",
                centerDiffMode:values[offset+152]>=.7?"locked":values[offset+152]>.01?"limited-slip":"open"});
              body.frontKnuckleSteerAngle=values[offset+153];body.rearKnuckleSteerAngle=values[offset+154];
              body.wheelSteerAngles={front_left:values[offset+155],front_right:values[offset+156],
                rear_left:values[offset+157],rear_right:values[offset+158]};
              body.steeringWrench={columnAngle:values[offset+159],frontRackTravel:values[offset+160],
                rearRackTravel:values[offset+161]};
              const presentedState=vehicleRuntime.state?.identity===body.identity?vehicleRuntime.state:
                vehicleRuntime.parkedState?.identity===body.identity?vehicleRuntime.parkedState:null;
              if(presentedState){presentedState.position=[...body.position];presentedState.velocity=[...body.velocity];
                presentedState.roll=body.roll;presentedState.pitch=body.pitch;presentedState.yaw=body.yaw;
                presentedState.rollVelocity=body.rollVelocity;presentedState.pitchVelocity=body.pitchVelocity;
                presentedState.yawVelocity=body.yawVelocity;presentedState.frontKnuckleSteerAngle=body.frontKnuckleSteerAngle;
                presentedState.rearKnuckleSteerAngle=body.rearKnuckleSteerAngle;
                presentedState.wheelSteerAngles={...body.wheelSteerAngles};
                presentedState.steeringWrench={...body.steeringWrench};
                ["front_left","front_right","rear_left","rear_right"].forEach((name,index)=>{
                  presentedState.wheelOmegas[name]=body.wheelOmegas?.[index]||0;
                  presentedState.compressions[name]=body.compressions?.[index]||0;});}
              if (presentedState) {
                vehicleRuntime.lastSpringForces = [...body.springForces];
                vehicleRuntime.contactAreas = [...body.contactAreas];
                vehicleRuntime.frictionUtilizations = [...body.frictionUtilizations];
                vehicleRuntime.contactModes = [...body.contactModes];
                vehicleRuntime.compressions = [...body.compressions];
                vehicleRuntime.tractionScales = [...body.tractionScales];
                vehicleRuntime.brakeScales = [...body.brakeScales];
                vehicleRuntime.damperScales = [...body.damperScales];
                vehicleRuntime.radialProbePenetrations=body.radialProbePenetrations.map(samples=>[...samples]);
                vehicleRuntime.radialProbeActiveCounts=[...body.radialProbeActiveCounts];
                vehicleRuntime.powertrain={...body.powertrain,reactionTorque:[...body.powertrain.reactionTorque],
                  mountTorque:[...body.powertrain.mountTorque]};
                presentedState.frontKnuckleSteerAngle=body.frontKnuckleSteerAngle;
                presentedState.rearKnuckleSteerAngle=body.rearKnuckleSteerAngle;
                const transmissionChanged=Object.keys(body.transmission).some(key=>
                  vehicleRuntime.transmission[key]!==body.transmission[key]);
                vehicleRuntime.transmission={...vehicleRuntime.transmission,...body.transmission};
                if(transmissionChanged)updateVehicleTransmissionControls();
              }
              body.contactIdentity = shaderViewer.colliders.find(collider =>
                collider.runtimePartId === body.contactRuntimePartId)?.identity || null;
            });
          }
          worker.postMessage({type: "recycle", sequence: message.sequence,
            buffer: message.buffer}, [message.buffer]);
        }
      };
      worker.onerror = event => {
        const reason = `${event.message||"physics worker failed"} @ ${event.filename||"worker"}:${event.lineno||0}:${event.colno||0}`;
        stateLoopRuntime.lastWorkerCrash=reason;
        physicsRuntime.error = reason;vehicleRuntime.error=reason;
        vehicleRuntime.computeMode="resident-webgpu-fault";
        if(stateLoopRuntime.forcingWasm){worker.terminate();stateLoopRuntime.worker=null;stateLoopRuntime.ready=false;return;}
        stateLoopRuntime.forcingWasm=true;
        worker.terminate();model.vehicle_slot.programs[0]=null;stateLoopRuntime.worker=null;stateLoopRuntime.ready=false;
        stateLoopRuntime.actorRegistered=false;stateLoopRuntime.bodies.clear();
        stateLoopRuntime.latestSequence=0;stateLoopRuntime.appliedSequence=0;
        initializePhysicsSnapshotSlots(stateLoopRuntime.snapshotCapacity);
        queueMicrotask(()=>initializeWorldPhysicsWasm());
      };
      worker.postMessage({type: "init", wasm: bytes, abi: plugin.abi,
        entrypoint: plugin.entrypoint, parameters: Object.fromEntries(physicsRuntime.parameters),
        snapshotCapacity: stateLoopRuntime.snapshotCapacity,
        vehicleWebgpu:model.vehicle_slot?.programs?.[0],
        vehicleWasm,contactWasm,
        worldBottom:model.contact_surfaces?.world_bottom});
      physicsRuntime.plugin = plugin; physicsRuntime.error = null;
      return;
    }
    throw new Error("resident vehicle graph requires the dedicated physics worker");
  } catch (error) {
    physicsRuntime.error = String(error?.message || error);
    console.error("symbolic world physics unavailable", error);
  } finally {
    physicsRuntime.pending = false;
  }
}

function buildVehicleBodyShell(vehicle,state){
  if(vehicleRuntime.bodyShell==="bare-frame")return [];
  const shell=vehicle.body_shells?.find(item=>item.identity===vehicleRuntime.bodyShell),
    chassis=vehicle.configuration.chassis,halfLength=Number(chassis.half_length),halfWidth=Number(chassis.half_width);
  if(shell?.identity==="five-turret-carrier"){
    const steel=model.appearance.colors["drivetrain-black"]||"#30383c",accent=model.appearance.colors["engine-accent"]||"#db9f3e",
      armorHeight=.39,armorY=.29,armorThickness=.018,
      armorSpecs=[
        ["armor-left",[0,armorY,-halfWidth*1.04],[halfLength*.91,armorThickness],armorHeight],
        ["armor-right",[0,armorY,halfWidth*1.04],[halfLength*.91,armorThickness],armorHeight],
        ["armor-front",[halfLength*.91,armorY,0],[armorThickness,halfWidth*1.04],armorHeight],
        ["armor-rear",[-halfLength*.91,armorY,0],[armorThickness,halfWidth*1.04],armorHeight],
        ["armor-left-upper",[-.02,.58,-halfWidth*.79],[halfLength*.62,armorThickness],.25],
        ["armor-right-upper",[-.02,.58,halfWidth*.79],[halfLength*.62,armorThickness],.25]],
      parts=armorSpecs.map(([name,localCenter,halfExtent,height])=>({identity:`${vehicle.identity}/turret-carrier:${name}`,
        kind:"vehicle-armor-segment",label:name,parent_identity:vehicle.identity,center:[...state.position],center_y:state.position[1],
        half_extent:halfExtent,height,floor_height:height,wall_thickness:armorThickness,palette_role:"drivetrain-black",
        wall_palette_role:"drivetrain-black",geometry_mode:"solid",openings:[],local_center:localCenter,
        appearance:{face_color:steel,material_profile:"quenched-steel-plate"},presentation_layer:"mounted-breakable-armor",
        placement:{custody:"placed",elevation:state.position[1],rotation:[0,0,0]},
        physics:{enabled:true,collider:"segmented-armor-contact",mass:Number(shell.armor?.mass_kg||0)/armorSpecs.length,
          mount_authority:"body-assembly-wrench-interface"}}));
    for(const turret of shell.turrets||[]){
      const name=turret.identity,base=[...turret.local_position],postHeight=Number(turret.post_height_m||.4),
        postCenter=[base[0],base[1]-postHeight*.5,base[2]],gimbalCenter=[base[0],base[1]+.04,base[2]];
      parts.push({identity:`${vehicle.identity}/turret:${name}:post`,kind:"vehicle-turret-post",label:`${name} post`,
        parent_identity:vehicle.identity,center:[...state.position],center_y:state.position[1],half_extent:[.055,.055],height:postHeight,
        floor_height:postHeight,wall_thickness:.012,palette_role:"rollbar-silver",wall_palette_role:"rollbar-silver",
        geometry_mode:"solid",openings:[],local_center:postCenter,turret_name:name,turret_part:"post",
        appearance:{face_color:model.appearance.colors["rollbar-silver"]},placement:{custody:"placed",elevation:0,rotation:[0,0,0]},
        physics:{enabled:true,collider:"turret-post",mount_authority:"body-assembly-wrench-interface"}});
      parts.push({identity:`${vehicle.identity}/turret:${name}:gimbal`,kind:"vehicle-turret-gimbal",label:`${name} gimbal`,
        parent_identity:vehicle.identity,center:[...state.position],center_y:state.position[1],half_extent:[.13,.12],height:.16,
        floor_height:.16,wall_thickness:.01,palette_role:"engine-accent",wall_palette_role:"engine-accent",
        geometry_mode:"solid",openings:[],local_center:gimbalCenter,turret_name:name,turret_part:"gimbal",
        appearance:{face_color:accent},placement:{custody:"placed",elevation:0,rotation:[0,0,0]},physics:{enabled:false,welded:false}});
      parts.push({identity:`${vehicle.identity}/turret:${name}:weapon`,kind:"vehicle-turret-weapon",label:`${name} weapon`,
        parent_identity:vehicle.identity,center:[...state.position],center_y:state.position[1],half_extent:[.29,.036],height:.072,
        floor_height:.072,wall_thickness:.008,palette_role:"drivetrain-black",wall_palette_role:"drivetrain-black",
        geometry_mode:"solid",openings:[],local_center:gimbalCenter,turret_name:name,turret_part:"weapon",
        appearance:{face_color:steel},placement:{custody:"placed",elevation:0,rotation:[0,0,0]},physics:{enabled:false,welded:false}});
    }
    for(const name of shell.outriggers?.feet||[]){const front=name.startsWith("front")?1:-1,left=name.endsWith("left")?-1:1,
      mount=[front*halfLength*.56,.20,left*halfWidth*.88];
      parts.push({identity:`${vehicle.identity}/outrigger:${name}:actuator`,kind:"vehicle-hydraulic-outrigger",label:`${name} outrigger`,
        parent_identity:vehicle.identity,center:[...state.position],center_y:state.position[1],half_extent:[.38,.032],height:.064,
        floor_height:.064,wall_thickness:.01,palette_role:"actuator-yellow",wall_palette_role:"actuator-yellow",geometry_mode:"solid",
        openings:[],local_center:mount,outrigger_name:name,outrigger_part:"actuator",outrigger_mount_local:mount,
        appearance:{face_color:model.appearance.colors["actuator-yellow"]||accent},placement:{custody:"placed",elevation:0,rotation:[0,0,0]},
        physics:{enabled:false,mount_authority:"body-assembly-wrench-interface"}});
      parts.push({identity:`${vehicle.identity}/outrigger:${name}:reserve`,kind:"vehicle-outrigger-reserve-tube",
        label:`${name} inboard reserve tube`,parent_identity:vehicle.identity,center:[...state.position],center_y:state.position[1],
        half_extent:[Number(shell.outriggers?.inboard_reserve_m||.72)*.5,.044],height:.088,floor_height:.088,wall_thickness:.012,
        palette_role:"rollbar-silver",wall_palette_role:"rollbar-silver",geometry_mode:"solid",openings:[],local_center:mount,
        outrigger_name:name,outrigger_part:"reserve",outrigger_mount_local:mount,
        appearance:{face_color:model.appearance.colors["rollbar-silver"]},placement:{custody:"placed",elevation:0,rotation:[0,0,0]},
        physics:{enabled:true,collider:"inboard-reserve-tube",mount_authority:"body-assembly-wrench-interface"}});
      parts.push({identity:`${vehicle.identity}/outrigger:${name}:foot`,kind:"vehicle-outrigger-foot",label:`${name} welded foot`,
        parent_identity:vehicle.identity,center:[...state.position],center_y:state.position[1],half_extent:[.12,.10],height:.045,
        floor_height:.045,wall_thickness:.01,palette_role:"drivetrain-black",wall_palette_role:"drivetrain-black",geometry_mode:"solid",
        openings:[],local_center:mount,outrigger_name:name,outrigger_part:"foot",outrigger_mount_local:mount,
        appearance:{face_color:steel},placement:{custody:"placed",elevation:0,rotation:[0,0,0]},
        physics:{enabled:true,collider:"terrain-weld-foot",mount_authority:"body-assembly-wrench-interface"}});}
    parts.push({identity:`${vehicle.identity}/turret-carrier:fire-control`,kind:"vehicle-fire-control-computer",
      label:"independent turret fire-control computer",parent_identity:vehicle.identity,center:[...state.position],center_y:state.position[1],
      half_extent:[.20,.16],height:.20,floor_height:.20,wall_thickness:.01,palette_role:"engine-accent",wall_palette_role:"engine-accent",
      geometry_mode:"solid",openings:[],local_center:[-.30,.36,0],appearance:{face_color:accent},
      placement:{custody:"placed",elevation:0,rotation:[0,0,0]},physics:{enabled:true,collider:"equipment-box"}});
    return parts;
  }
  const
    colorRole=shell?.palette_role||vehicle.configuration.presentation.palette_role,specs=[
      ["hood",[halfLength*.48,.32,0],[halfLength*.40,halfWidth*1.08],.30],
      ["cab",[-halfLength*.03,.52,0],[halfLength*.32,halfWidth*1.06],.82],
      ["bed",[-halfLength*.62,.29,0],[halfLength*.35,halfWidth*1.09],.30],
      ["left-fender",[halfLength*.20,.24,-halfWidth*1.07],[halfLength*.74,.065],.24],
      ["right-fender",[halfLength*.20,.24,halfWidth*1.07],[halfLength*.74,.065],.24]];
  return specs.map(([name,localCenter,halfExtent,height])=>({identity:`${vehicle.identity}/cosmetic-shell:${name}`,
    kind:"vehicle-cosmetic-shell",label:name,parent_identity:vehicle.identity,center:[state.position[0],state.position[2]],
    center_y:state.position[1]+localCenter[1],half_extent:halfExtent,height,floor_height:height,wall_thickness:.012,
    palette_role:colorRole,wall_palette_role:colorRole,geometry_mode:"solid",openings:[],local_center:localCenter,
    appearance:{face_color:model.appearance.colors[colorRole],material_identity:shell?.material_identity||null,
      material_profile:shell?.material_profile||"opaque"},presentation_layer:"mounted-breakable-contact-shell",
    placement:{custody:"placed",elevation:state.position[1]+localCenter[1]-height*.5,rotation:[0,0,0]},
    physics:{enabled:true,collider:"subdivided-shell-samples",mass:Number(vehicle.configuration.body_shell.shell_mass_kg||0)/specs.length,
      mount_authority:"mechanical-graph-body-shell-mounts"}}));
}

function ensureVehiclePresentation(vehicle,state) {
  if(vehicleRuntime.box){vehicleRuntime.box.placement.custody="placed";
    if(vehicleRuntime.cabinBox)vehicleRuntime.cabinBox.placement.custody="placed";
    vehicleRuntime.frameBoxes.forEach(member=>member.placement.custody="placed");
    vehicleRuntime.rollCageBoxes.forEach(member=>member.placement.custody="placed");
    vehicleRuntime.mechanicalLinkBoxes.forEach(member=>member.placement.custody="placed");
    vehicleRuntime.suspensionLinkBoxes.forEach(member=>member.placement.custody="placed");
    vehicleRuntime.powertrainBoxes.forEach(member=>member.placement.custody="placed");
    vehicleRuntime.wheelBoxes.forEach(wheel=>wheel.placement.custody="placed");return vehicleRuntime.box;}
  const config=vehicle.configuration,chassis=config.chassis,colorRole=config.presentation.palette_role;
  const box={identity:vehicle.identity,kind:"vehicle",label:vehicle.name,parent_identity:model.identity,
    center:[state.position[0],state.position[2]],center_y:state.position[1],
    half_extent:[Number(chassis.half_length),Number(chassis.half_width)],
    height:.075,floor_height:.075,wall_thickness:.02,
    palette_role:colorRole,wall_palette_role:colorRole,geometry_mode:"solid",openings:[],
    appearance:{face_color:model.appearance.colors[colorRole]},
    placement:{custody:"placed",elevation:state.position[1]-.0375,
      rotation:[state.roll*180/Math.PI,state.yaw*180/Math.PI,state.pitch*180/Math.PI]},
    physics:{body:"dynamic-vehicle",collider:"chassis",enabled:true,welded:true}};
  const presentation=vehicle.configuration.presentation,wheels=vehicle.configuration.wheels,
    wheelPart=vehicle.wheel_parts?.find(item=>item.identity===vehicleRuntime.wheelPart)||vehicle.wheel_parts?.[0];
  vehicleRuntime.wheelBoxes=["front_left","front_right","rear_left","rear_right"].map((name,index)=>({
    identity:`${vehicle.identity}/wheel:${name}`,kind:"vehicle-wheel",label:name,parent_identity:vehicle.identity,
    center:[state.position[0],state.position[2]],center_y:state.position[1],half_extent:[Number(config.tires.radius),Number(config.tires.width)*.5],
    height:Number(config.tires.radius)*2,floor_height:0,wall_thickness:.01,geometry_mode:"vehicle-wheel",openings:[],
    palette_role:presentation.wheel_palette_role||"line",wall_palette_role:presentation.wheel_palette_role||"line",
    appearance:{face_color:model.appearance.colors[presentation.wheel_palette_role||"line"],
      tread_color:model.appearance.colors[presentation.wheel_tread_palette_role||"active"]},
    placement:{custody:"placed",elevation:0,rotation:[0,0,0]},physics:{enabled:false,welded:true},
    wheel_state:{name,index,radius:Number(config.tires.radius),rimRadius:Number(wheels.rim_radius),
      width:Number(config.tires.width),spin:0,steer:0,tireColor:wheelPart?.tire_color,
      treadColor:wheelPart?.tread_color,rimColor:wheelPart?.rim_color,
      carcassProfile:wheelPart?.carcass_profile,rimProfile:wheelPart?.rim_profile,treadPattern:wheelPart?.tread_pattern,
      chassisPosition:[...state.position],chassisRotation:[state.roll,state.yaw,state.pitch],localCenter:[0,0,0]}
  }));
  const frameColor=model.appearance.colors["rollbar-silver"],frameSpecs=[
    ["rail-left",[0,Number(chassis.height)*.72,-Number(wheels.track_half_width)],[Number(chassis.half_length)*.88,.025]],
    ["rail-right",[0,Number(chassis.height)*.72,Number(wheels.track_half_width)],[Number(chassis.half_length)*.88,.025]],
    ["axle-front",[Number(wheels.wheelbase_half_length),Number(chassis.height)*.72,0],[.025,Number(wheels.track_half_width)]],
    ["axle-rear",[-Number(wheels.wheelbase_half_length),Number(chassis.height)*.72,0],[.025,Number(wheels.track_half_width)]]
  ];
  vehicleRuntime.frameBoxes=frameSpecs.map(([name,localCenter,halfExtent])=>({
    identity:`${vehicle.identity}/frame:${name}`,kind:"vehicle-frame-member",label:name,parent_identity:vehicle.identity,
    center:[state.position[0],state.position[2]],center_y:state.position[1],half_extent:halfExtent,
    height:.055,floor_height:.055,wall_thickness:.01,palette_role:"rollbar-silver",wall_palette_role:"rollbar-silver",
    geometry_mode:"solid",openings:[],appearance:{face_color:frameColor},local_center:localCenter,
    placement:{custody:"placed",elevation:state.position[1],rotation:[0,0,0]},physics:{enabled:false,welded:true}}));
  vehicleRuntime.suspensionLinkBoxes=[];
  const cageHalfLength=Number(chassis.half_length)*.68,cageHalfWidth=Number(chassis.half_width)*.76,
    cageFloor=.08,cageRoof=Math.max(.42,Number(chassis.height)+.24),member=.018,
    cageSpecs=[
      ["pillar-front-left",[cageHalfLength,(cageFloor+cageRoof)*.5,-cageHalfWidth],[member,member],cageRoof-cageFloor],
      ["pillar-front-right",[cageHalfLength,(cageFloor+cageRoof)*.5,cageHalfWidth],[member,member],cageRoof-cageFloor],
      ["pillar-rear-left",[-cageHalfLength,(cageFloor+cageRoof)*.5,-cageHalfWidth],[member,member],cageRoof-cageFloor],
      ["pillar-rear-right",[-cageHalfLength,(cageFloor+cageRoof)*.5,cageHalfWidth],[member,member],cageRoof-cageFloor],
      ["roof-left",[0,cageRoof,-cageHalfWidth],[cageHalfLength,member],.045],
      ["roof-right",[0,cageRoof,cageHalfWidth],[cageHalfLength,member],.045],
      ["roof-front",[cageHalfLength,cageRoof,0],[member,cageHalfWidth],.045],
      ["roof-rear",[-cageHalfLength,cageRoof,0],[member,cageHalfWidth],.045]
    ];
  vehicleRuntime.rollCageBoxes=cageSpecs.map(([name,localCenter,halfExtent,height])=>({
    identity:`${vehicle.identity}/roll-cage:${name}`,kind:"vehicle-roll-cage-member",label:name,
    parent_identity:vehicle.identity,center:[...box.center],center_y:state.position[1],half_extent:halfExtent,
    height,floor_height:height,wall_thickness:.008,palette_role:"rollbar-silver",wall_palette_role:"rollbar-silver",
    geometry_mode:"solid",openings:[],appearance:{face_color:frameColor},local_center:localCenter,
    placement:{custody:"placed",elevation:state.position[1],rotation:[0,0,0]},physics:{enabled:false,welded:true}}));
  // Cylindrical members below come directly from the compiler-emitted mechanical graph.
  // Clear the earlier compatibility boxes so there is only one visible load-path authority.
  vehicleRuntime.frameBoxes=[];vehicleRuntime.rollCageBoxes=[];
  const mechanicalGraph=vehicle.physics.mechanical_graph;
  vehicleRuntime.mechanicalLinkBoxes=mechanicalGraph.edges.filter(edge=>
    ["rigid-distance","rigid-offset","spring-damper","steering-link","torque-shaft",
     "constant-velocity-torque-shaft","six-axis-compliant-mount","steering-torque-shaft",
     "universal-joint-steering-shaft","universal-joint-hub-spindle",
     "rack-and-pinion-angle-to-translation","rack-translation","routed-tension-cable",
     "table-actuator-linear-link","tension-limit-strap","routed-energy-line","intake-flow-path","exhaust-flow-path",
     "insulated-copper-wire","pressure-rated-hydraulic-line","pressure-rated-air-line",
     "flexible-hydraulic-hose","flexible-air-line","flexible-air-line-with-rotary-union",
     "steering-assist-torque-coupling"].includes(edge.constraint)).map(edge=>({
      identity:`${vehicle.identity}/mechanical:${edge.identity}`,kind:"vehicle-mechanical-edge",label:edge.identity,
      parent_identity:vehicle.identity,center:[...box.center],center_y:state.position[1],half_extent:[.015,.015],
      height:.03,floor_height:0,wall_thickness:.006,palette_role:edge.palette_role,
      wall_palette_role:edge.palette_role,geometry_mode:"vehicle-link",openings:[],
      appearance:{face_color:model.appearance.colors[edge.palette_role]},mechanical_edge:edge,
      suspension_role:edge.identity.startsWith("suspension.")?edge.constraint:null,
      link_state:{localA:[0,0,0],localB:[0,.1,0],radius:Number(edge.radius||.012),
        chassisPosition:[...state.position],chassisRotation:[state.roll,state.yaw,state.pitch]},
      placement:{custody:"placed",elevation:state.position[1],rotation:[0,0,0]},physics:{enabled:false,welded:true}}));
  const pt=config.powertrain,enginePosition=pt.engine_position.map(Number),wheelbase=Number(wheels.wheelbase_half_length),
    engineScale=Math.cbrt(Math.max(.125,Number(pt.displacement_liters||1.6)/1.6)),
    activePreset=vehicle.power_unit_presets?.find(item=>item.identity===vehicleRuntime.powerUnitPreset)||
      vehicle.power_unit_presets?.find(item=>item.identity===vehicle.power_unit_preset),architecture=activePreset?.architecture||{
        layout:"inline-four",cylinders:4,banks:1},cylinders=Math.max(0,Number(architecture.cylinders||0)),
    cylinderSpecs=Array.from({length:cylinders},(_,index)=>{const banks=Math.max(1,Number(architecture.banks||1)),bank=index%banks,
      bankIndex=Math.floor(index/banks),perBank=Math.ceil(cylinders/banks),x=enginePosition[0]+(bankIndex-(perBank-1)/2)*.055*engineScale,
      side=banks===1?0:(bank===0?-1:1),kind=architecture.layout?.includes("flat")?"vehicle-flat-cylinder":
        architecture.layout?.includes("v8")?"vehicle-v-cylinder":"vehicle-inline-cylinder";
      return [`cylinder-${index+1}`,kind,[x,.13*engineScale,side*.072*engineScale],[.022*engineScale,.026*engineScale],
        .082*engineScale,"engine-metal","engineTorque"];}),
    rotorSpecs=cylinders?[]:Array.from({length:Math.max(1,Number(architecture.banks||1))},(_,index)=>[
      `electric-stator-${index+1}`,"vehicle-electric-stator",[enginePosition[0]+index*.11,.12,(index%2?1:-1)*.045],
      [.06*engineScale,.05*engineScale],.10*engineScale,"engine-metal","engineTorque"]),
    ignitionSpecs=activePreset?.kind==="combustion"?[
      ["ignition-distributor","vehicle-ignition-distributor",[enginePosition[0]-.10,.225,Number(chassis.half_width)*.18],
        [.024,.024],.065,"engine-accent",null],
      ["ignition-coil-left","vehicle-ignition-coil",[enginePosition[0]-.04,.235,-Number(chassis.half_width)*.20],
        [.022,.018],.055,"engine-accent",null],
      ["ignition-coil-right","vehicle-ignition-coil",[enginePosition[0]-.04,.235,Number(chassis.half_width)*.20],
        [.022,.018],.055,"engine-accent",null]
    ]:[],monsterFuelSystem=activePreset?.kind==="combustion"&&String(activePreset?.identity||"").includes("monster"),
    auxiliaryReservoirSpecs=monsterFuelSystem?[
      ["reserve-fuel-cell","vehicle-reserve-fuel-tank",[-Number(chassis.half_length)*.68,.20,-Number(chassis.half_width)*.42],
        [.09,.11],.11,"fuel-tank-candy-red",null,18],
      ["oxidizer-canister","vehicle-oxidizer-canister",[-Number(chassis.half_length)*.68,.20,Number(chassis.half_width)*.42],
        [.07,.08],.16,"oxidizer-canister",null,14]
    ]:[],
    architectureSpecs=[
      ["engine-crank","vehicle-engine-crank",[enginePosition[0],.10,enginePosition[2]],[.13*engineScale,.018*engineScale],.036*engineScale,"engine-accent","engineTorque"],
      ...cylinderSpecs,...rotorSpecs,...ignitionSpecs,...auxiliaryReservoirSpecs,
      ["intake-plenum","vehicle-intake-plenum",[enginePosition[0]-.015,.205,0],[.11*engineScale,.045*engineScale],.055*engineScale,"engine-metal","engineTorque"],
      ["throttle-body","vehicle-throttle-body",[enginePosition[0]+.075,.205,-Number(chassis.half_width)*.13],[.025,.026],.055,"actuator-yellow","engineTorque"],
      ["header-left","vehicle-exhaust-header",[enginePosition[0],.14,-Number(chassis.half_width)*.27],[.10*engineScale,.012],.025,"engine-accent","engineTorque"],
      ["header-right","vehicle-exhaust-header",[enginePosition[0],.14,Number(chassis.half_width)*.27],[.10*engineScale,.012],.025,"engine-accent","engineTorque"],
      ["energy-storage",activePreset?.energy_system?.storage_kind||"baffled-fuel-tank",[-Number(chassis.half_length)*.48,.18,0],
        [.16,Math.min(.24,Number(chassis.half_width)*.65)],.14,activePreset?.kind==="combustion"?
          "fuel-tank-candy-red":"battery-electric-blue",null]
    ],
    powertrainSpecs=[...architectureSpecs,
      ["clutch","vehicle-clutch",[enginePosition[0]+.15,.10,0],[.025,.052],.038,"drivetrain-black","clutchTorque"],
      ["transmission-shaft","vehicle-transmission",[enginePosition[0]+.27,.085,0],[.095,.012],.028,"drivetrain-black","transmissionOutputTorque"],
      ["transfer-case","vehicle-transfer-case",[enginePosition[0]+.39,.065,0],[.045,.055],.06,"drivetrain-black","drivelineTorque"],
      ["center-driveshaft","vehicle-driveshaft",[-.12,.06,0],[Math.max(.12,wheelbase-.25),.010],.024,"drivetrain-black","drivelineTorque"],
      ["front-differential","vehicle-differential",[wheelbase,.065,0],[.04,.07],.038,"drivetrain-black","frontDifferentialTorque"],
      ["rear-differential","vehicle-differential",[-wheelbase,.065,0],[.04,.07],.038,"drivetrain-black","rearDifferentialTorque"],
      ["front-differential-brake","vehicle-differential-brake",[wheelbase+.105,.065,0],[.018,.082],.052,"suspension-yellow","frontDifferentialBrakeTorque"],
      ["rear-differential-brake","vehicle-differential-brake",[-wheelbase-.105,.065,0],[.018,.082],.052,"suspension-yellow","rearDifferentialBrakeTorque"],
      ["steering-servo","vehicle-steering-servo",[.08,.29,-.10],[.035,.045],.075,"active",null,
        Number(config.electrical.steering_servo_mass_kg||3.2)],
      ["engine-mount","vehicle-powertrain-mount",[enginePosition[0],.075,0],[.018,Number(wheels.track_half_width)*.58],.03,"drivetrain-black","engineAccelerationTorque"],
      ["transmission-mount","vehicle-powertrain-mount",[enginePosition[0]+.28,.055,0],[.014,Number(wheels.track_half_width)*.52],.026,"drivetrain-black","transmissionOutputTorque"],
      ["transfer-case-mount","vehicle-powertrain-mount",[enginePosition[0]+.39,.045,0],[.014,Number(wheels.track_half_width)*.45],.024,"drivetrain-black","drivelineTorque"]
    ];
  vehicleRuntime.powertrainBoxes=powertrainSpecs.map(([name,kind,localCenter,halfExtent,height,palette,torqueChannel,explicitMass])=>{
    const cylinder=kind.includes("cylinder"),side=Math.sign(Number(localCenter[2])||1),
      shape=kind.includes("fuel-tank")||kind.includes("canister")?"fuel-tank":kind.includes("battery")?"battery-pack":cylinder?"engine-cylinder":
        kind.includes("differential-brake")?"differential-brake":kind.includes("ignition-coil")?"ignition-coil":
        kind.includes("ignition-distributor")?"distributor":
        kind.includes("throttle-body")?"throttle-body":
        ["vehicle-engine-crank","vehicle-exhaust-header","vehicle-transmission","vehicle-driveshaft"].includes(kind)?"round-shaft":null,
      axis=cylinder?(kind.includes("flat")?[0,0,side]:kind.includes("v-cylinder")?[0,.72,side*.70]:[0,1,0]):
        kind==="vehicle-throttle-body"?[0,0,1]:[1,0,0];
    const size=[Number(halfExtent[0])*2,Number(height),Number(halfExtent[1])*2],declaredMass=Number.isFinite(explicitMass)?Number(explicitMass):
        kind.includes("fuel-tank")||kind.includes("battery")?Number(activePreset?.energy_system?.installed_storage_mass_kg||30):
        kind.includes("differential-brake")?22:cylinder?Math.max(4,12*engineScale):
        kind.includes("ignition")?1.3:kind.includes("differential")?28:kind.includes("transfer")?32:
        kind.includes("transmission")?48:kind.includes("mount")?4:8,
      inertia={roll:declaredMass*(size[1]*size[1]+size[2]*size[2])/12,
        pitch:declaredMass*(size[0]*size[0]+size[2]*size[2])/12,
        yaw:declaredMass*(size[0]*size[0]+size[1]*size[1])/12};
    return {identity:`${vehicle.identity}/powertrain:${name}`,kind,label:name,parent_identity:vehicle.identity,
      center:[...box.center],center_y:state.position[1],half_extent:halfExtent,height,floor_height:height,
      wall_thickness:.008,palette_role:palette,wall_palette_role:palette,
      geometry_mode:shape?"vehicle-powertrain-part":"solid",part_mesh:shape?{shape,axis}:null,openings:[],
      appearance:{face_color:model.appearance.colors[palette]},local_center:localCenter,torque_channel:torqueChannel,
      mass_properties:{mass_kg:declaredMass,center_of_mass_local:[...localCenter],principal_inertia_kg_m2:inertia,
        integration_status:"declared-part-budget-existing-vehicle-lumped-mass-remains-authoritative"},
      placement:{custody:"placed",elevation:state.position[1],rotation:[0,0,0]},physics:{enabled:false,welded:true}};
  });
  vehicleRuntime.bodyShellBoxes=buildVehicleBodyShell(vehicle,state);
  vehicleRuntime.box=box;vehicleRuntime.cabinBox=null;
  shaderViewer.geometry.push(...vehicleRuntime.frameBoxes,...vehicleRuntime.rollCageBoxes,
    ...vehicleRuntime.mechanicalLinkBoxes,...vehicleRuntime.powertrainBoxes);
  updateVehiclePresentation(vehicle,state,0,0);updateVehicleBodyPresentation(vehicle,state);
  rebuildPortableSceneMesh();return box;
}

function solveVehicleMechanicalGraph(vehicle,state){
  const graph=vehicle.physics.mechanical_graph,positions=new Map(graph.nodes.map(node=>[
    node.identity,node.reference_position.map(Number)])),fixed=new Set(graph.nodes.filter(node=>node.fixed_to==="chassis"&&!node.structural_deformable)
      .map(node=>node.identity)),chassis=vehicle.configuration.chassis,wheels=vehicle.configuration.wheels,
    suspension=vehicle.configuration.suspension,damage=vehicleRuntime.damage;
  ["front_left","front_right","rear_left","rear_right"].forEach(corner=>{
    const prefix=`suspension.${corner}`,hubIdentity=`${prefix}.hub`,referenceHub=positions.get(hubIdentity),
      compression=Number(state.compressions[corner]||0),plastic=Number(damage?.springPlasticSet?.[corner]||0),
      leveling=Number(vehicleRuntime.electrical.levelingOffsets?.[corner]||0),
      targetHubY=-Number(chassis.clearance)-(Number(suspension.rest_length)+leveling-plastic)+compression+
        Number(vehicle.configuration.tires.radius),delta=targetHubY-referenceHub[1];
    graph.nodes.filter(node=>node.generalized_coordinate===`compression_${corner}`).forEach(node=>{
      const point=positions.get(node.identity);point[1]+=delta;
    });
  });
  const constraints=graph.edges.filter(edge=>!edge.runtime_failed&&
    (edge.constraint==="rigid-distance"||edge.constraint==="rigid-offset"||edge.constraint==="steering-link"));
  for(let iteration=0;iteration<18;iteration+=1){
    constraints.forEach(edge=>{const a=positions.get(edge.a),b=positions.get(edge.b);if(!a||!b)return;
      const delta=b.map((value,index)=>value-a[index]),length=Math.max(1e-8,Math.hypot(...delta)),
        targetLength=Number(edge.runtime_rest_length??edge.rest_length)+Number(vehicleRuntime.electrical.linkLengthModifiers?.[edge.identity]||0),
        error=(length-targetLength)/length,aFixed=fixed.has(edge.a),bFixed=fixed.has(edge.b),
        aScale=aFixed?0:bFixed?1:.5,bScale=bFixed?0:aFixed?1:.5;
      for(let axis=0;axis<3;axis+=1){a[axis]+=delta[axis]*error*aScale;b[axis]-=delta[axis]*error*bScale;}
    });
    ["front_left","front_right","rear_left","rear_right"].forEach(corner=>{
      const hub=positions.get(`suspension.${corner}.hub`),compression=Number(state.compressions[corner]||0),
        plastic=Number(damage?.springPlasticSet?.[corner]||0),leveling=Number(vehicleRuntime.electrical.levelingOffsets?.[corner]||0),
        target=-Number(chassis.clearance)-(Number(suspension.rest_length)+leveling-plastic)+compression+
          Number(vehicle.configuration.tires.radius);
      hub[1]+=(target-hub[1])*.38;
    });
  }
  ["front_left","front_right","rear_left","rear_right"].forEach(corner=>{
    const hub=positions.get(`suspension.${corner}.hub`),patch=positions.get(`suspension.${corner}.contact_patch`);
    patch[0]=hub[0];patch[1]=hub[1]-Number(vehicle.configuration.tires.radius);patch[2]=hub[2];
  });
  const steeringInput=Number(state.presentationSteering||0),wheelSteer=state.wheelSteerAngles||{};
  [["front_left",Number(wheelSteer.front_left??state.presentationFrontSteering??0)],
   ["front_right",Number(wheelSteer.front_right??state.presentationFrontSteering??0)],
   ["rear_left",Number(wheelSteer.rear_left??state.presentationRearSteering??0)],
   ["rear_right",Number(wheelSteer.rear_right??state.presentationRearSteering??0)]].forEach(([corner,steeringAngle])=>{
    const knuckle=positions.get(`suspension.${corner}.knuckle`),cosine=Math.cos(steeringAngle),sine=Math.sin(steeringAngle),
      rotateAboutKnuckle=identity=>{const point=positions.get(`suspension.${corner}.${identity}`);if(!point||!knuckle)return;
        const dx=point[0]-knuckle[0],dz=point[2]-knuckle[2];point[0]=knuckle[0]+dx*cosine-dz*sine;
        point[2]=knuckle[2]+dx*sine+dz*cosine;};
    // The upright/kingpin is the steering pivot. The hub spindle, wheel-side
    // CV yoke and yellow steering arm articulate about it; the inner halfshaft
    // remains attached to the differential and visibly breaks angle at the yoke.
    ["hub","wheel_rim","tire_carcass","brake_rotor","contact_patch","steering_arm"].forEach(rotateAboutKnuckle);
  });
  const steeringCenter=positions.get("steering.wheel.center"),wheelAngle=Number(
      state.steeringWrench?.columnAngle??(-steeringInput*1.5)),
    wheelCos=Math.cos(wheelAngle),wheelSin=Math.sin(wheelAngle);
  if(steeringCenter)for(let index=0;index<8;index+=1){
    const identity=`steering.wheel.ring_${index}`,point=positions.get(identity),reference=graph.nodes.find(
      node=>node.identity===identity)?.reference_position;if(!point||!reference)continue;
    const dy=reference[1]-steeringCenter[1],dz=reference[2]-steeringCenter[2];
    point[1]=steeringCenter[1]+dy*wheelCos-dz*wheelSin;
    point[2]=steeringCenter[2]+dy*wheelSin+dz*wheelCos;
  }
  [[Number(state.steeringWrench?.frontRackTravel||0),["steering.rack.center","suspension.front_left.steering_rack","suspension.front_right.steering_rack"]],
   [Number(state.steeringWrench?.rearRackTravel||0),["steering.rear_rack.center","suspension.rear_left.steering_rack","suspension.rear_right.steering_rack"]]]
    .forEach(([travel,identities])=>identities.forEach(identity=>{
      const point=positions.get(identity),reference=graph.nodes.find(node=>node.identity===identity)?.reference_position;
      if(point&&reference)point[2]=reference[2]+Number(travel);
    }));
  const throttle=Math.max(0,Math.min(1,Math.abs(Number(state.presentationThrottle||0)))),lever=positions.get(
      "powertrain.intake.throttle_lever"),leverReference=graph.nodes.find(node=>
        node.identity==="powertrain.intake.throttle_lever")?.reference_position;
  if(lever&&leverReference){const table=graph.edges.find(edge=>edge.identity==="actuator.throttle.lever")?.travel_table||[[0,0],[1,.038]];
    let lower=table[0],upper=table.at(-1);for(let index=1;index<table.length;index++)if(throttle<=Number(table[index][0])){
      lower=table[index-1];upper=table[index];break;}const span=Math.max(1e-8,Number(upper[0])-Number(lower[0])),mix=Math.max(0,
      Math.min(1,(throttle-Number(lower[0]))/span)),travel=Number(lower[1])+(Number(upper[1])-Number(lower[1]))*mix;
    lever[0]=Number(leverReference[0])+travel;lever[1]=Number(leverReference[1])-travel*.18;}
  vehicleRuntime.mechanicalNodePositions=positions;return positions;
}

function updateVehiclePresentation(vehicle,state,dt,steering){
  const config=vehicle.configuration,chassis=config.chassis,wheels=config.wheels;
  state.presentationSteering=Number(steering||0);
  state.presentationThrottle=Number(vehicleRuntime.state?.lastThrottle??state.presentationThrottle??0);
  // Rendering consumes the worker's constrained knuckle coordinates. It does
  // not synthesize a second steering animation from the input command.
  state.presentationFrontSteering=Number(state.frontKnuckleSteerAngle||0);
  state.presentationRearSteering=Number(state.rearKnuckleSteerAngle||0);
  const names=["front_left","front_right","rear_left","rear_right"],
    positions=solveVehicleMechanicalGraph(vehicle,state);
  names.forEach((name,index)=>{
    const omega=Number(state.wheelOmegas[name]||0);
    vehicleRuntime.wheelAngles[index]=(vehicleRuntime.wheelAngles[index]+omega*Math.max(0,Math.min(.05,dt)))%(Math.PI*2);
    const compression=Number(state.compressions[name]||0),wheel=vehicleRuntime.wheelBoxes[index];
    if(!wheel)return;
    wheel.center=[state.position[0],state.position[2]];wheel.center_y=state.position[1];
    wheel.wheel_state={...wheel.wheel_state,spin:vehicleRuntime.wheelAngles[index],
      steer:Number(state.wheelSteerAngles?.[name]??(index<2?state.presentationFrontSteering:state.presentationRearSteering)),
      tirePressurePa:Number(vehicleRuntime.electrical.tirePressurePa||config.tires.pressure_pa),
      referenceTirePressurePa:Number(config.tires.reference_pressure_pa||155000),
      chassisPosition:[...state.position],chassisRotation:[state.roll,state.yaw,state.pitch],
      localCenter:[...positions.get(`suspension.${name}.hub`)]};
  });
}

function uploadVehiclePresentationMesh(dt){
  vehicleRuntime.presentationAccumulator+=Math.max(0,dt);
  if(vehicleRuntime.presentationAccumulator<1/30)return;
  vehicleRuntime.presentationAccumulator=0;rebuildPortableSceneMesh({dynamicOnly:true});
}

function updateVehicleBodyPresentation(vehicle,state){
  const chassis=vehicle.configuration.chassis,rotation=[state.roll*180/Math.PI,
    state.yaw*180/Math.PI,state.pitch*180/Math.PI],box=ensureVehiclePresentation(vehicle,state);
  updateVehicleTurretTargeting(vehicle,state);
  box.center=[state.position[0],state.position[2]];box.center_y=state.position[1];
  box.placement.elevation=state.position[1]-Number(box.height)*.5;box.placement.rotation=rotation;
  const cabin=vehicleRuntime.cabinBox;if(cabin){cabin.center=[state.position[0],state.position[2]];
    cabin.center_y=state.position[1]+Number(chassis.height)*.5+Number(cabin.height)*.5;
    cabin.placement.elevation=state.position[1]+Number(chassis.height)*.5;cabin.placement.rotation=[...rotation];}
  vehicleRuntime.frameBoxes.forEach(member=>{
    const local=[...member.local_center];
    const offset=rotateVehiclePresentationVector(local,state,0);
    member.center=[state.position[0]+offset[0],state.position[2]+offset[2]];
    member.center_y=state.position[1]+offset[1];member.placement.elevation=member.center_y-member.height*.5;
    member.placement.rotation=[...rotation];
  });
  const positions=vehicleRuntime.mechanicalNodePositions.size?vehicleRuntime.mechanicalNodePositions:
    solveVehicleMechanicalGraph(vehicle,state);
  vehicleRuntime.mechanicalLinkBoxes.forEach(member=>{
    const edge=member.mechanical_edge,a=[...positions.get(edge.a)],authoredB=[...positions.get(edge.b)],
      b=edge.runtime_failed?a.map((value,index)=>value+(authoredB[index]-value)*.43):authoredB;
    member.appearance.face_color=edge.runtime_failed?"#ff4f62":model.appearance.colors[edge.palette_role];
    if(edge.identity.startsWith("lighting.headlamp."))member.appearance.face_color=
      vehicleRuntime.electrical.headlightsOn&&vehicleRuntime.electrical.lightingCircuitOnline?"#ffe4a3":"#554f42";
    if(edge.identity.startsWith("lighting.tail."))member.appearance.face_color=vehicleRuntime.electrical.brakeLightsOn?
      "#ff2638":vehicleRuntime.electrical.tailLightsOn?"#b81426":"#3b161b";
    if(edge.identity.startsWith("electrical.wire."))member.appearance.face_color=
      vehicleRuntime.electrical.computerOnline?"#d88935":"#34281f";
    let routeLocalPoints=null;
    if(edge.routing==="relaxed-multi-segment-harness"){
      const now=performance.now()*.001,previous=member.link_state.routeLocalPoints,
        elapsed=Math.min(.1,Math.max(0,now-Number(member.link_state.routeUpdatedAt||now))),
        rate=Math.max(.1,Number(edge.relaxation_rate_hz||6)),alpha=1-Math.exp(-rate*elapsed),
        delta=b.map((value,index)=>value-a[index]),span=Math.max(.001,Math.hypot(...delta)),
        slack=Math.max(1,Number(edge.slack_ratio||1.05)),bendRelax=Math.min(1,Math.max(.1,Number(edge.bend_relaxation||.6))),
        minimumBend=Math.max(.005,Number(edge.minimum_bend_radius_m||.02)),
        sag=Math.max(minimumBend*.8*bendRelax,Math.min(span*.28,span*Math.sqrt(Math.max(0,slack*slack-1))*.55*bendRelax)),
        horizontal=Math.hypot(delta[0],delta[2]),side=horizontal>.0001?[-delta[2]/horizontal,0,delta[0]/horizontal]:[1,0,0],
        handedness=[...edge.identity].reduce((sum,character)=>sum+character.charCodeAt(0),0)%2?1:-1,
        target=Array.from({length:7},(_,index)=>{const t=index/6,sine=Math.sin(Math.PI*t),bow=Math.sin(Math.PI*2*t);
          return a.map((value,axis)=>value+delta[axis]*t+(axis===1?-sag*sine:0)+
            side[axis]*handedness*minimumBend*.32*bendRelax*bow);});
      routeLocalPoints=previous?.length===target.length?target.map((point,index)=>
        index===0||index===target.length-1?point:point.map((value,axis)=>
          previous[index][axis]+(value-previous[index][axis])*alpha)):target;
      member.link_state.routeUpdatedAt=now;
    }
    member.link_state={...member.link_state,
      localA:a,localB:b,radius:Number(edge.radius||.012),
      routeLocalPoints:routeLocalPoints||undefined,
      routeUpdatedAt:member.link_state.routeUpdatedAt,
      chassisPosition:[...state.position],chassisRotation:[state.roll,state.yaw,state.pitch]};
  });
  [...vehicleRuntime.rollCageBoxes,...vehicleRuntime.powertrainBoxes,...vehicleRuntime.bodyShellBoxes].forEach(member=>{
    const offset=rotateVehiclePresentationVector(member.local_center,state,0);
    member.center=[state.position[0]+offset[0],state.position[2]+offset[2]];
    member.center_y=state.position[1]+offset[1];member.placement.elevation=member.center_y-member.height*.5;
    member.placement.rotation=[...rotation];
    if(member.outrigger_name){const front=member.outrigger_name.startsWith("front")?1:-1,
      left=member.outrigger_name.endsWith("left")?-1:1,mountOffset=rotateVehiclePresentationVector(member.outrigger_mount_local,state,0),
      mount=state.position.map((value,index)=>value+mountOffset[index]),localDirection=normalized3([front*.18,-.78,left*.60]),
      worldDirection=normalized3(rotateVehiclePresentationVector(localDirection,state,0)),extension=Number(
        vehicleRuntime.electrical.outriggerExtension||0),anchor=vehicleRuntime.electrical.outriggerAnchors?.[member.outrigger_name],
      endpoint=anchor?.position?[...anchor.position]:mount.map((value,index)=>value+worldDirection[index]*(.18+extension)),
      direction=normalized3(endpoint.map((value,index)=>value-mount[index])),yaw=Math.atan2(direction[2],direction[0]),
      pitch=Math.atan2(direction[1],Math.hypot(direction[0],direction[2]));
      if(member.outrigger_part==="actuator"){const center=mount.map((value,index)=>(value+endpoint[index])*.5);
        member.center=[center[0],center[2]];member.center_y=center[1];member.half_extent[0]=Math.max(.08,Math.hypot(...endpoint.map(
          (value,index)=>value-mount[index]))*.5);member.placement.elevation=center[1]-member.height*.5;
        member.placement.rotation=[0,yaw*180/Math.PI,pitch*180/Math.PI];}
      else if(member.outrigger_part==="reserve"){const reserve=.72,center=mount.map((value,index)=>value-direction[index]*reserve*.5);
        member.center=[center[0],center[2]];member.center_y=center[1];member.half_extent[0]=reserve*.5;
        member.placement.elevation=center[1]-member.height*.5;member.placement.rotation=[0,yaw*180/Math.PI,pitch*180/Math.PI];}
      else{member.center=[endpoint[0],endpoint[2]];member.center_y=endpoint[1];member.placement.elevation=endpoint[1]-member.height*.5;
        member.placement.rotation=[0,state.yaw*180/Math.PI,0];member.appearance.face_color=anchor?"#65d89b":model.appearance.colors[member.palette_role];}}
    const turret=member.turret_name?vehicleRuntime.turretSystem.turrets.find(item=>item.name===member.turret_name):null;
    if(turret&&(member.turret_part==="weapon"||member.turret_part==="gimbal")){
      const direction=turret.direction,center=member.turret_part==="weapon"?
        turret.pivot.map((value,index)=>value+direction[index]*.29):turret.pivot;
      member.center=[center[0],center[2]];member.center_y=center[1];member.placement.elevation=center[1]-member.height*.5;
      member.placement.rotation=[0,turret.yaw*180/Math.PI,turret.pitch*180/Math.PI];
      member.appearance.face_color=turret.inhibited?"#6d2028":model.appearance.colors[member.palette_role];
    }
    if(member.torque_channel)member.appearance.face_color=model.appearance.colors[member.palette_role];
  });
  return box;
}

function synchronizeVehicleLookYaw(state){
  if(vehicleRuntime.cameraChassisYaw===null){vehicleRuntime.cameraChassisYaw=state.yaw;return;}
  const delta=Math.atan2(Math.sin(state.yaw-vehicleRuntime.cameraChassisYaw),
    Math.cos(state.yaw-vehicleRuntime.cameraChassisYaw));
  viewportControls.yaw+=delta;vehicleRuntime.cameraChassisYaw=state.yaw;
}

function ensureVehicleWorldMarker(vehicle){
  if(vehicleRuntime.worldMarker)return vehicleRuntime.worldMarker;
  const layer=document.getElementById("entity-layer");if(!layer)return null;
  const marker=document.createElement("button");marker.type="button";marker.className="vehicle-world-marker";
  marker.textContent="◆";marker.title=`${vehicle.name} · click or press V to mount/dismount`;
  marker.setAttribute("aria-label",`${vehicle.name} vehicle marker`);
  marker.addEventListener("pointerdown",event=>event.stopPropagation());
  marker.addEventListener("click",event=>{event.preventDefault();event.stopPropagation();
    if(vehicleRuntime.active)clearActiveVehicle();else setActiveVehicle(vehicle.identity,{placeAtActor:false});});
  layer.append(marker);vehicleRuntime.worldMarker=marker;return marker;
}

function removeVehicleInventoryItem(item){
  if(!item)return;
  const slot=hotbarState.model.slots.find(candidate=>candidate.item===item.identity);
  vehicleRuntime.inventoryItem=item;vehicleRuntime.inventorySlot=slot?.number||item.slot||null;
  if(slot)slot.item=null;
  const index=hotbarState.inventory.items.findIndex(candidate=>candidate.identity===item.identity);
  if(index>=0)hotbarState.inventory.items.splice(index,1);
  if(hotbarState.activeSlot===vehicleRuntime.inventorySlot){hotbarState.activeSlot=null;hotbarState.model.active_slot=null;}
  refreshInventoryCounts();
}

function cloneVehicleState(state){
  return {...state,position:[...state.position],velocity:[...state.velocity],
    wheelOmegas:{...state.wheelOmegas},previousSlips:{...state.previousSlips},
    wheelSteerAngles:{...(state.wheelSteerAngles||{})},steeringWrench:{...(state.steeringWrench||{})},
    compressions:{...state.compressions},defaults:{...state.defaults}};
}

function vehicleTerrainRestPose(vehicle,position,yaw,compressions){
  const config=vehicle.configuration,wheels=config.wheels,chassis=config.chassis,suspension=config.suspension,
    names=["front_left","front_right","rear_left","rear_right"],
    corners=[[wheels.wheelbase_half_length,-wheels.track_half_width],
      [wheels.wheelbase_half_length,wheels.track_half_width],
      [-wheels.wheelbase_half_length,-wheels.track_half_width],
      [-wheels.wheelbase_half_length,wheels.track_half_width]],cy=Math.cos(yaw),sy=Math.sin(yaw),
    samples=corners.map(([forward,right])=>cameraGroundHeight(position[0]+forward*cy-right*sy,
      position[2]+forward*sy+right*cy)),wheelbase=Math.max(1e-6,Number(wheels.wheelbase_half_length)),
    track=Math.max(1e-6,Number(wheels.track_half_width)),forwardSlope=(samples[0]+samples[1]-samples[2]-samples[3])/(4*wheelbase),
    rightSlope=(samples[1]+samples[3]-samples[0]-samples[2])/(4*track),pitch=Math.atan(forwardSlope),
    roll=-Math.atan(rightSlope),pose={roll,pitch,yaw},bodyHeights=corners.map(([forward,right],index)=>{
      const contactLocal=[Number(forward),-Number(chassis.clearance)-Number(suspension.rest_length)+
        Number(compressions[names[index]]||0),Number(right)],offset=rotateVehiclePresentationVector(contactLocal,pose),
        ground=cameraGroundHeight(position[0]+offset[0],position[2]+offset[2]);return ground-offset[1];});
  return {roll,pitch,yaw,bodyY:Math.max(...bodyHeights)+.008};
}

function vehicleSpawnState(vehicle,placeAtActor,{reuseParked=true}={}){
  if(!placeAtActor&&reuseParked&&vehicleRuntime.parkedState)return cloneVehicleState(vehicleRuntime.parkedState);
  const position=placeAtActor?(viewportControls.position||vehicle.pose.position):vehicle.pose.position,
    config=vehicle.configuration;
  const audit=vehicle.physics?.mechanical_graph?.load_audit,corners=audit?.corners||{},
    names=["front_left","front_right","rear_left","rear_right"],compressions=Object.fromEntries(names.map(name=>[
      name,Math.min(Number(config.suspension.travel),Number(corners[name]?.design_spring_compression_m||0))])),
    yaw=Number(placeAtActor?(viewportControls.yaw??vehicle.pose.yaw??0):(vehicle.pose.yaw??0)),
    rest=vehicleTerrainRestPose(vehicle,position,yaw,compressions),
    inheritedVelocity=placeAtActor?[viewportControls.horizontalVelocity[0]||0,0,
      viewportControls.horizontalVelocity[1]||0]:[0,0,0];
  return {identity:vehicle.identity,
    position:[position[0],rest.bodyY,position[2]],
    velocity:inheritedVelocity,
    roll:rest.roll,pitch:rest.pitch,yaw:rest.yaw,
    rollVelocity:0,pitchVelocity:0,yawVelocity:0,
    wheelOmegas:{front_left:0,front_right:0,rear_left:0,rear_right:0},
    wheelSteerAngles:{front_left:0,front_right:0,rear_left:0,rear_right:0},
    previousSlips:{front_left:0,front_right:0,rear_left:0,rear_right:0},
    compressions,defaults:{...vehicle.configuration_defaults}};
}

function setActiveVehicle(identity,{placeAtActor=false,inventoryItem=null}={}) {
  const vehicle=model.vehicle_slot?.vehicles?.find(item=>item.identity===identity);
  if(!vehicle||vehicleRuntime.active)return false;
  if(inventoryItem)removeVehicleInventoryItem(inventoryItem);
  vehicleRuntime.active=vehicle;model.vehicle_slot.active=vehicle.identity;
  vehicleRuntime.transmission={mode:vehicle.configuration.transmission.mode_default,
    gear:Number(vehicle.configuration.transmission.starting_gear),
    displayGear:Number(vehicle.configuration.transmission.starting_gear),torqueReserve:0,reason:"initial-second",
    lowRange:false,transferRange:"high",frontDiffLock:false,rearDiffLock:false,centerDiffLock:false,
    frontDiffMode:"open",rearDiffMode:"open",centerDiffMode:"open",
    frontDriveShare:Number(vehicle.configuration.drivetrain.front_drive_fraction||.5),smoothLaunch:false,
    tractionControlEnabled:true,absEnabled:true,tiltEnabled:true,tractionControlAuthority:1,absAuthority:1};
  vehicleRuntime.brakeLocks={front_left:false,front_right:false,rear_left:false,rear_right:false};
  vehicleRuntime.powerUnitPreset=vehicle.power_unit_preset||vehicle.power_unit_presets?.[0]?.identity||null;
  vehicleRuntime.engineEquationMode=vehicle.engine_kernel_switch?.default_equation_mode||"linear-playable";
  vehicleRuntime.transmissionPreset=vehicle.transmission_preset||vehicle.transmission_presets?.[0]?.identity||null;
  vehicleRuntime.clutchPreset=vehicle.clutch_preset||vehicle.clutch_presets?.[0]?.identity||null;
  vehicleRuntime.chassisProfile=vehicle.chassis_profile||vehicle.chassis_profiles?.[0]?.identity||null;
  vehicleRuntime.wheelPart=vehicle.wheel_part||"balloon-black-current";
  vehicleRuntime.bodyShell=vehicle.body_shell||"clear-polycarbonate-rc";
  const turretAssembly=vehicle.body_shells?.find(item=>item.identity==="five-turret-carrier");
  const turretAmmo=turretAssembly?.ammunition||{},turretCapacity=Math.max(0,Math.min(Number(turretAmmo.capacity_count||0),
    Math.floor(Number(turretAmmo.capacity_mass_kg||0)/Math.max(1e-9,Number(turretAmmo.round_mass_kg||1))),
    Math.floor(Number(turretAmmo.capacity_volume_m3||0)/Math.max(1e-9,Number(turretAmmo.round_volume_m3||1)))));
  vehicleRuntime.turretSystem={fireTakeover:turretAssembly?.fire_control?.primary_fire_takeover_default!==false,
    ammoCount:Math.min(turretCapacity,Number(turretAmmo.initial_count||0)),target:null,targetDistance:0,
    interlockReason:"no-target",turrets:[],outriggers:{commanded:false,extension:0,anchors:{}}};
  vehicleRuntime.electrical={headlightsOn:false,hornOn:false,ignitionOn:true,starterEngaged:false,
    stateOfCharge:Number(vehicle.configuration.electrical?.initial_state_of_charge||1),
    fuelMassKg:Number(vehicle.configuration.fuel_system?.initial_fuel_mass_kg||0),
    totalMassKg:Number(vehicle.configuration.mass||0),fuelIdentity:vehicle.fuel_profile||"pump-gasoline-93",
    ignitionProfileIdentity:vehicle.ignition_profile||"gasoline-distributor",ignitionTimingOffsetCycles:0,
    requestedIgnitionProfileIdentity:vehicle.ignition_profile||"gasoline-distributor",combustionSharpness:1,
    timingErrorDegrees:0,combustionStress:0,computerOnline:true,ecuOnline:true,lightingCircuitOnline:true,
    tailLightsOn:false,brakeLightsOn:false,tirePressurePa:Number(vehicle.configuration.tires?.pressure_pa||155000),
    tirePressureTargetPa:Number(vehicle.tire_pressure_target_pa||vehicle.configuration.tires?.pressure_pa||155000),pneumaticCompressorOn:false,
    pneumaticCompressorPowerW:0,hydraulicPumpOn:false,hydraulicPumpPowerW:0};
  vehicleRuntime.driverAssistance={drivingMode:vehicle.driving_mode||"road",
    governorRpm:Number(vehicle.configuration.powertrain?.redline_rpm||6500),cruiseEnabled:false,
    cruiseTargetSpeedMps:0,cruiseIntegral:0,cruiseThrottle:0,cruiseBrake:0,tiltEnabled:true,
    tiltAuthority:0,tiltRisk:0,tiltGovernorRpm:Number(vehicle.configuration.powertrain?.redline_rpm||6500),
    rearDifferentialBrakeCommand:0};
  vehicleRuntime.chassisLeveling={...(vehicle.chassis_leveling||{}),cornerOffsets:{front_left:0,front_right:0,rear_left:0,rear_right:0}};
  vehicleRuntime.wheelAlignment=JSON.parse(JSON.stringify(vehicle.wheel_alignment||{}));
  vehicleRuntime.steeringSystem={...(vehicle.steering_control||vehicle.configuration.steering_control||{})};
  vehicleRuntime.shockParameters={...vehicle.configuration_defaults};
  vehicleRuntime.damage=null;
  vehicleRuntime.wheelAngles=[0,0,0,0];vehicleRuntime.camera=null;vehicleRuntime.presentationAccumulator=0;
  vehicleRuntime.state=vehicleSpawnState(vehicle,placeAtActor);
  viewportControls.yaw=vehicleRuntime.state.yaw;viewportControls.pitch=0;
  vehicleRuntime.cameraChassisYaw=vehicleRuntime.state.yaw;
  ensureVehiclePresentation(vehicle,vehicleRuntime.state);
  ensureVehicleWorldMarker(vehicle)?.classList.add("active");
  if(vehicleRuntime.contactMonitor)vehicleRuntime.contactMonitor.hidden=shaderViewer.shaderOnly;
  const actorIdentity=viewportControls.policy?.actor;
  if(stateLoopRuntime.ready&&actorIdentity){stateLoopRuntime.worker.postMessage({type:"remove",identity:actorIdentity});
    releasePhysicsSnapshotSlot(actorIdentity);stateLoopRuntime.actorRegistered=false;}
  registerActiveVehiclePhysicsBody();
  cancelEntityNavigation(viewportControls.policy?.actor);setPlacementStatus(
    `${vehicle.name} mounted · JSON physics · four compiled spring lanes`);
  return true;
}

function clearActiveVehicle() {
  if(!vehicleRuntime.active)return;
  void setVehicleHorn(false);
  const state=vehicleRuntime.state,config=vehicleRuntime.active.configuration;
  vehicleRuntime.parkedState=cloneVehicleState(state);
  model.vehicle_slot.active=null;vehicleRuntime.active=null;vehicleRuntime.state=null;
  if(vehicleRuntime.contactMonitor)vehicleRuntime.contactMonitor.hidden=true;
  vehicleRuntime.camera=null;
  vehicleRuntime.cameraChassisYaw=null;
  if(vehicleRuntime.box)vehicleRuntime.box.placement.custody="placed";
  if(vehicleRuntime.cabinBox)vehicleRuntime.cabinBox.placement.custody="placed";
  vehicleRuntime.wheelBoxes.forEach(wheel=>wheel.placement.custody="placed");
  vehicleRuntime.worldMarker?.classList.remove("active");
  // Dismount relinquishes the controls, not the machine.  Its resident solver body,
  // engine, fuel, battery, driveline, contacts, and damage continue to advance.
  if(stateLoopRuntime.ready&&state)stateLoopRuntime.worker.postMessage({type:"vehicle-control",identity:state.identity,
    throttle:0,steering:0,brake:0});
  if(stateLoopRuntime.ready&&state)stateLoopRuntime.worker.postMessage({type:"vehicle-auxiliary",identity:state.identity,
    hornOn:false,starterEngaged:false});
  if(state){
    viewportControls.position=[state.position[0],state.position[1]+Number(config.chassis.camera_height),
      state.position[2]+Number(config.chassis.half_width)+.18];
    viewportControls.horizontalVelocity=[state.velocity[0],state.velocity[2]];
    synchronizePortalBody(viewportControls.policy?.actor,viewportControls.position,
      [state.velocity[0],state.velocity[1],state.velocity[2]]);
  }
  registerPlayerPhysicsBody();
  rebuildPortableSceneMesh();
  setPlacementStatus("Springtail parked in world · platformer physics restored · V remounts");
}

function recoverActiveVehicle(){
  const vehicle=vehicleRuntime.active,state=vehicleRuntime.state;if(!vehicle||!state)return false;
  const lift=Number(vehicle.configuration.tires.radius)+Number(vehicle.configuration.chassis.height)+.12;
  state.position[1]+=lift;state.roll=0;state.pitch=0;state.rollVelocity=0;state.pitchVelocity=0;
  state.yawVelocity=0;state.velocity[1]=Math.max(0,state.velocity[1]);
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-recover",identity:vehicle.identity,lift});
  updateVehiclePresentation(vehicle,state,0,0);updateVehicleBodyPresentation(vehicle,state);
  rebuildPortableSceneMesh({dynamicOnly:true});setPlacementStatus("Springtail recovered upright");return true;
}


function respawnActiveVehicleAtAuthoredPose(){
  const vehicle=vehicleRuntime.active,state=vehicleRuntime.state;if(!vehicle||!state)return false;
  const spawn=vehicleSpawnState(vehicle,false,{reuseParked:false});
  state.position=[...spawn.position];state.velocity=[...spawn.velocity];state.roll=spawn.roll;
  state.pitch=spawn.pitch;state.yaw=spawn.yaw;state.rollVelocity=0;state.pitchVelocity=0;state.yawVelocity=0;
  state.wheelOmegas={...spawn.wheelOmegas};state.previousSlips={...spawn.previousSlips};
  state.compressions={...spawn.compressions};
  vehicleRuntime.damage=null;vehicle.physics.mechanical_graph.edges.forEach(edge=>{
    delete edge.runtime_rest_length;delete edge.runtime_failed;});
  Object.keys(vehicleRuntime.brakeLocks).forEach(name=>vehicleRuntime.brakeLocks[name]=false);
  vehicleRuntime.transmission.frontDifferentialBrake=false;
  vehicleRuntime.transmission.rearDifferentialBrake=false;
  vehicleRuntime.powertrain={engineTorque:0,clutchTorque:0,transmissionOutputTorque:0,drivelineTorque:0,
    frontDifferentialTorque:0,rearDifferentialTorque:0,engineAccelerationTorque:0,
    engineAngularAcceleration:0,reactionTorque:[0,0,0],mountTorque:[0,0,0]};
  vehicleRuntime.camera=null;vehicleRuntime.cameraChassisYaw=state.yaw;viewportControls.yaw=spawn.yaw;
  if(stateLoopRuntime.ready){stateLoopRuntime.worker.postMessage({type:"remove",identity:vehicle.identity});
    releasePhysicsSnapshotSlot(vehicle.identity);registerActiveVehiclePhysicsBody();}
  updateVehiclePresentation(vehicle,state,0,0);updateVehicleBodyPresentation(vehicle,state);
  rebuildPortableSceneMesh({dynamicOnly:true});setPlacementStatus(
    "Springtail respawned on the solid flat apron · drivetrain and brake holds reset");
  updateVehicleTransmissionControls();return true;
}

function respawnViewportActor(){
  if(vehicleRuntime.active)return respawnActiveVehicleAtAuthoredPose();
  const actor=entityState.get(viewportControls.policy?.actor),spawn=actor?.entity?.pose?.position||actor?.record?.pose?.position;
  if(!Array.isArray(spawn))return false;
  const identity=viewportControls.policy.actor;
  if(stateLoopRuntime.ready&&identity){stateLoopRuntime.worker.postMessage({type:"remove",identity});
    releasePhysicsSnapshotSlot(identity);stateLoopRuntime.actorRegistered=false;}
  viewportControls.position=[...spawn];viewportControls.horizontalVelocity=[0,0];physicsRuntime.verticalVelocity=0;
  registerPlayerPhysicsBody();setPlacementStatus("Player respawned at the authored start");return true;
}

function controlVehicleTransmission({mode=null,gearDelta=null,lowRange=null,transferRange=null,frontDiffLock=null,
    rearDiffLock=null,centerDiffLock=null,frontDriveShare=null,smoothLaunch=null,
    frontDiffMode=null,rearDiffMode=null,centerDiffMode=null,
    frontDifferentialBrake=null,rearDifferentialBrake=null,
    tractionControlEnabled=null,absEnabled=null,tiltEnabled=null,tractionControlAuthority=null,absAuthority=null,
    gearset=null,transmissionPreset=null,
    brakeLock=null,releaseAllBrakes=false}={}){
  const vehicle=vehicleRuntime.active;if(!vehicle)return false;
  if(gearset&&Array.isArray(gearset.forward_ratios)){
    vehicle.configuration.transmission={...vehicle.configuration.transmission,...gearset};
    vehicleRuntime.transmissionPreset=transmissionPreset||"custom";
  }
  const transmission=vehicle.configuration.transmission,state=vehicleRuntime.transmission;
  if(gearset){state.gear=Math.max(1,Math.min(transmission.forward_ratios.length,Number(transmission.starting_gear||1)));
    state.displayGear=state.gear;state.engagedRatio=Number(transmission.forward_ratios[state.gear-1]);}
  if(mode==="automatic")state.mode="automatic";
  if(Number.isFinite(gearDelta)){state.mode="manual";state.gear=Math.max(1,Math.min(
    transmission.forward_ratios.length,Math.round(state.gear+Number(gearDelta))));state.displayGear=state.gear;}
  if(["high","l1","l2"].includes(transferRange)){state.transferRange=transferRange;state.lowRange=transferRange!=="high";}
  else if(typeof lowRange==="boolean"){state.lowRange=lowRange;state.transferRange=lowRange?"l2":"high";}
  if(typeof frontDiffLock==="boolean")state.frontDiffLock=frontDiffLock;
  if(typeof rearDiffLock==="boolean")state.rearDiffLock=rearDiffLock;
  if(typeof centerDiffLock==="boolean")state.centerDiffLock=centerDiffLock;
  for(const [key,value] of Object.entries({frontDiffMode,rearDiffMode,centerDiffMode}))if(
      ["open","limited-slip","locked"].includes(value)){state[key]=value;
    state[key.replace("Mode","Lock")]=value==="locked";}
  if(Number.isFinite(frontDriveShare))state.frontDriveShare=Math.max(.05,Math.min(.95,Number(frontDriveShare)));
  if(typeof smoothLaunch==="boolean")state.smoothLaunch=smoothLaunch;
  if(typeof frontDifferentialBrake==="boolean")state.frontDifferentialBrake=frontDifferentialBrake;
  if(typeof rearDifferentialBrake==="boolean")state.rearDifferentialBrake=rearDifferentialBrake;
  if(typeof tractionControlEnabled==="boolean")state.tractionControlEnabled=tractionControlEnabled;
  if(typeof absEnabled==="boolean")state.absEnabled=absEnabled;
  if(typeof tiltEnabled==="boolean")state.tiltEnabled=tiltEnabled;
  if(Number.isFinite(tractionControlAuthority))state.tractionControlAuthority=Math.max(0,Math.min(1,Number(tractionControlAuthority)));
  if(Number.isFinite(absAuthority))state.absAuthority=Math.max(0,Math.min(1,Number(absAuthority)));
  if(brakeLock&&brakeLock.name in vehicleRuntime.brakeLocks)
    vehicleRuntime.brakeLocks[brakeLock.name]=Boolean(brakeLock.locked);
  if(releaseAllBrakes)Object.keys(vehicleRuntime.brakeLocks).forEach(name=>vehicleRuntime.brakeLocks[name]=false);
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-transmission",identity:vehicle.identity,
    mode,gearDelta,lowRange,transferRange,frontDiffLock,rearDiffLock,centerDiffLock,frontDiffMode,rearDiffMode,centerDiffMode,
    frontDriveShare,smoothLaunch,frontDifferentialBrake,rearDifferentialBrake,
    tractionControlEnabled,absEnabled,tiltEnabled,tractionControlAuthority,absAuthority,
    gearset,transmissionPreset,brakeLock,releaseAllBrakes});
  updateVehicleTransmissionControls();
  setPlacementStatus(`${state.mode==="automatic"?"Springtail automatic":`Springtail manual · gear ${state.gear}`} · ${
    `${String(state.transferRange||"high").toUpperCase()} transfer`} · F${Math.round(state.frontDriveShare*100)}/R${Math.round((1-state.frontDriveShare)*100)}`);return true;
}

function controlVehicleAuxiliary(update={}){
  const vehicle=vehicleRuntime.active;if(!vehicle)return false;const state=vehicleRuntime.electrical;
  for(const name of ["headlightsOn","hornOn","ignitionOn","starterEngaged"])
    if(typeof update[name]==="boolean")state[name]=update[name];
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-auxiliary",identity:vehicle.identity,
    headlightsOn:state.headlightsOn,hornOn:state.hornOn,ignitionOn:state.ignitionOn,
    starterEngaged:state.starterEngaged});
  if(typeof update.hornOn==="boolean")void setVehicleHorn(state.hornOn);
  return true;
}

function controlVehicleFuelIgnition(update={}){
  const vehicle=vehicleRuntime.active;if(!vehicle)return false;
  if(typeof update.fuelIdentity==="string"){vehicle.fuel_profile=update.fuelIdentity;vehicleRuntime.electrical.fuelIdentity=update.fuelIdentity;}
  if(typeof update.ignitionProfileIdentity==="string"){vehicle.ignition_profile=update.ignitionProfileIdentity;
    vehicleRuntime.electrical.ignitionProfileIdentity=update.ignitionProfileIdentity;
    vehicleRuntime.electrical.requestedIgnitionProfileIdentity=update.ignitionProfileIdentity;}
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-fuel-ignition",identity:vehicle.identity,
    fuelIdentity:vehicleRuntime.electrical.fuelIdentity,ignitionProfileIdentity:vehicleRuntime.electrical.ignitionProfileIdentity});
  updateVehicleTransmissionControls();return true;
}

function controlVehicleDriverAssistance(update={}){
  const vehicle=vehicleRuntime.active;if(!vehicle)return false;const state=vehicleRuntime.driverAssistance;
  if(typeof update.drivingMode==="string"&&(vehicle.driving_modes||[]).some(item=>item.identity===update.drivingMode))
    state.drivingMode=update.drivingMode;
  if(Number.isFinite(update.governorRpm))state.governorRpm=Math.max(500,Math.min(Number(
    vehicle.configuration.powertrain?.redline_rpm||6500),Number(update.governorRpm)));
  if(typeof update.cruiseEnabled==="boolean"){state.cruiseEnabled=update.cruiseEnabled;
    if(update.cruiseEnabled&&!Number.isFinite(update.cruiseTargetSpeedMps))state.cruiseTargetSpeedMps=Math.hypot(
      Number(vehicleRuntime.state?.velocity?.[0]||0),Number(vehicleRuntime.state?.velocity?.[2]||0));}
  if(Number.isFinite(update.cruiseTargetSpeedMps))state.cruiseTargetSpeedMps=Math.max(0,Number(update.cruiseTargetSpeedMps));
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-driver-assistance",identity:vehicle.identity,
    drivingMode:state.drivingMode,governorRpm:state.governorRpm,cruiseEnabled:state.cruiseEnabled,
    cruiseTargetSpeedMps:state.cruiseTargetSpeedMps});
  updateVehicleTransmissionControls();return true;
}

function selectVehicleTransmissionPreset(presetIdentity){
  const vehicle=vehicleRuntime.active,preset=vehicle?.transmission_presets?.find(item=>item.identity===presetIdentity);
  if(!preset)return false;
  controlVehicleTransmission({gearset:preset.configuration,transmissionPreset:preset.identity});
  setPlacementStatus(`${preset.label} selected · ratios remain live compiled inputs`);return true;
}

function selectVehicleClutchPreset(presetIdentity){
  const vehicle=vehicleRuntime.active,preset=vehicle?.clutch_presets?.find(item=>item.identity===presetIdentity),
    reference=vehicle?.clutch_presets?.find(item=>item.default)||vehicle?.clutch_presets?.[0],
    powerPreset=vehicle?.power_unit_presets?.find(item=>item.identity===vehicleRuntime.powerUnitPreset);
  if(!vehicle||!preset||!reference)return false;
  vehicle.clutch_preset=preset.identity;vehicleRuntime.clutchPreset=preset.identity;
  const baseInertia=Number(powerPreset?.parameters?.engine_rotating_inertia||vehicle.configuration.powertrain.engine_rotating_inertia_kg_m2),
    effectiveInertia=Math.max(.001,baseInertia+Number(preset.driven_inertia_kg_m2)-Number(reference.driven_inertia_kg_m2));
  Object.assign(vehicle.configuration.powertrain,{clutch_stiffness_nm_per_rad_s:Number(preset.stiffness_nm_per_rad_s),
    clutch_maximum_torque_nm:Number(preset.maximum_torque_nm),clutch_efficiency:Number(preset.efficiency),
    engine_rotating_inertia_kg_m2:effectiveInertia});
  Object.assign(vehicle.configuration_defaults,{clutch_stiffness:Number(preset.stiffness_nm_per_rad_s),
    clutch_maximum_torque:Number(preset.maximum_torque_nm),clutch_efficiency:Number(preset.efficiency),
    engine_rotating_inertia:effectiveInertia});vehicleRuntime.state.defaults={...vehicle.configuration_defaults};
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-clutch-preset",identity:vehicle.identity,
    preset:{...preset,reference_mass_kg:Number(reference.mass_kg),effective_engine_inertia_kg_m2:effectiveInertia}});
  updateVehicleTransmissionControls();setPlacementStatus(`${preset.label} · ${preset.maximum_torque_nm} Nm · ${preset.engagement}`);
  return true;
}

function selectVehiclePowerUnit(presetIdentity){
  const vehicle=vehicleRuntime.active,preset=vehicle?.power_unit_presets?.find(item=>item.identity===presetIdentity);
  if(!vehicle||!preset)return false;
  vehicle.power_unit_preset=preset.identity;vehicleRuntime.powerUnitPreset=preset.identity;
  const clutch=vehicle.clutch_presets?.find(item=>item.identity===vehicleRuntime.clutchPreset),
    referenceClutch=vehicle.clutch_presets?.find(item=>item.default)||vehicle.clutch_presets?.[0],
    inertiaDelta=clutch&&referenceClutch?Number(clutch.driven_inertia_kg_m2)-Number(referenceClutch.driven_inertia_kg_m2):0,
    parameters={...preset.parameters,engine_rotating_inertia:Number(preset.parameters.engine_rotating_inertia)+inertiaDelta,
      clutch_stiffness:Number(clutch?.stiffness_nm_per_rad_s??vehicle.configuration_defaults.clutch_stiffness),
      clutch_maximum_torque:Number(clutch?.maximum_torque_nm??vehicle.configuration_defaults.clutch_maximum_torque),
      clutch_efficiency:Number(clutch?.efficiency??vehicle.configuration_defaults.clutch_efficiency)},
    configuration={...preset.configuration,engine_rotating_inertia_kg_m2:Number(
      preset.configuration.engine_rotating_inertia_kg_m2)+inertiaDelta,
      clutch_stiffness_nm_per_rad_s:Number(clutch?.stiffness_nm_per_rad_s??vehicle.configuration.powertrain.clutch_stiffness_nm_per_rad_s),
      clutch_maximum_torque_nm:Number(clutch?.maximum_torque_nm??vehicle.configuration.powertrain.clutch_maximum_torque_nm),
      clutch_efficiency:Number(clutch?.efficiency??vehicle.configuration.powertrain.clutch_efficiency)};
  Object.assign(vehicle.configuration_defaults,parameters);
  Object.assign(vehicle.configuration.powertrain,configuration);
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-power-unit",identity:vehicle.identity,
    preset:{identity:preset.identity,kind:preset.kind,kernelSelector:preset.kernel_selector,
      equationMode:vehicleRuntime.engineEquationMode,parameters,configuration,
      compatibleFuelProfiles:[...(preset.compatible_fuel_profiles||[])],
      fuelCompatibility:{...(preset.fuel_compatibility||{})},
      ignitionCompatibility:{...(preset.ignition_compatibility||{})}}});
  vehicleRuntime.state.defaults={...vehicle.configuration_defaults};
  // The selected architecture owns its cylinder/bank or stator layout and
  // energy-storage presentation. Rebuild only this vehicle's presentation;
  // physics state and the resident compiled buffers remain in place.
  shaderViewer.geometry=shaderViewer.geometry.filter(item=>item.identity!==vehicle.identity&&item.parent_identity!==vehicle.identity);
  vehicleRuntime.box=null;vehicleRuntime.cabinBox=null;vehicleRuntime.frameBoxes=[];vehicleRuntime.rollCageBoxes=[];
  vehicleRuntime.mechanicalLinkBoxes=[];vehicleRuntime.suspensionLinkBoxes=[];vehicleRuntime.powertrainBoxes=[];
  vehicleRuntime.bodyShellBoxes=[];vehicleRuntime.wheelBoxes=[];ensureVehiclePresentation(vehicle,vehicleRuntime.state);
  updateVehicleTransmissionControls();setPlacementStatus(`${preset.label} selected · baked curve ${preset.curve_reference}`);
  return true;
}

function selectVehicleEngineEquationMode(mode){
  if(!["linear-playable","symbolic-fidelity"].includes(mode)||!vehicleRuntime.active)return false;
  vehicleRuntime.engineEquationMode=mode;return selectVehiclePowerUnit(vehicleRuntime.powerUnitPreset);
}

function applyChassisProfileToVehicle(vehicle,profile){
  const reference=vehicle.chassis_profile_reference||{},baseMass=Number(reference.vehicle_mass_kg||vehicle.configuration.mass),
    baseMemberMass=Number(reference.member_mass_kg||0),newMass=Math.max(1,baseMass-baseMemberMass+Number(profile.member_mass_kg||0)),
    massScale=newMass/Math.max(1,baseMass),graph=vehicle.physics?.mechanical_graph;
  for(const edge of graph?.edges||[]){if(!edge.chassis_profile_member)continue;
    edge.radius=Number(profile.outer_diameter_m)/2;
    edge.mass_kg=Number(edge.rest_length)*Number(profile.section_area_m2)*Number(profile.density_kg_m3);
    if(edge.damage)Object.assign(edge.damage,{material:profile.material,section_area_m2:profile.section_area_m2,
      youngs_modulus_pa:profile.youngs_modulus_pa,yield_strength_pa:profile.yield_strength_pa,
      shear_strength_pa:profile.shear_strength_pa,axial_yield_force_n:profile.axial_yield_force_n,
      shear_force_limit_n:profile.shear_force_limit_n,axial_stiffness_n_per_m:Number(profile.youngs_modulus_pa)*
        Number(profile.section_area_m2)/Math.max(1e-9,Number(edge.rest_length))});}
  vehicle.configuration.mass=newMass;
  vehicle.configuration_defaults.inverse_mass=1/newMass;
  for(const axis of ["roll","pitch","yaw"]){const key=`inverse_inertia_${axis}`,
      baseKey=`_chassis_reference_${key}`;
    if(!Number.isFinite(vehicle[baseKey]))vehicle[baseKey]=Number(vehicle.configuration_defaults[key]);
    vehicle.configuration_defaults[key]=Number(vehicle[baseKey])/massScale;}
  return {newMass,massScale};
}

function selectVehicleChassisProfile(profileIdentity){
  const vehicle=vehicleRuntime.active,profile=vehicle?.chassis_profiles?.find(item=>item.identity===profileIdentity);
  if(!vehicle||!profile)return false;
  vehicle.chassis_profile=profile.identity;vehicleRuntime.chassisProfile=profile.identity;
  const physical=applyChassisProfileToVehicle(vehicle,profile);vehicleRuntime.state.defaults={...vehicle.configuration_defaults};
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-chassis-profile",identity:vehicle.identity,
    profile:{...profile,vehicle_mass_kg:physical.newMass}});
  updateVehicleTransmissionControls();setPlacementStatus(`${profile.label} · ${physical.newMass.toFixed(1)} kg vehicle · physical frame limits active`);
  return true;
}

function selectVehicleChassisGeometry({chassisLengthM,wheelbaseM}){
  const vehicle=vehicleRuntime.active,spec=vehicle?.chassis_geometry_parameters,graph=vehicle?.physics?.mechanical_graph;
  if(!vehicle||!spec||!graph)return false;
  const clamp=(value,range)=>Math.max(Number(range[0]),Math.min(Number(range[1]),Number(value))),
    length=clamp(chassisLengthM??spec.chassis_length_m,spec.chassis_length_range_m),
    requestedWheelbase=clamp(wheelbaseM??spec.wheelbase_m,spec.wheelbase_range_m),
    wheelbase=Math.min(requestedWheelbase,length-2*Number(spec.minimum_end_overhang_m||0)),
    halfLength=length/2,halfWheelbase=wheelbase/2,positions=new Map();
  vehicle.configuration.chassis.half_length=halfLength;vehicle.configuration.wheels.wheelbase_half_length=halfWheelbase;
  spec.chassis_length_m=length;spec.wheelbase_m=wheelbase;
  for(const node of graph.nodes){const parameter=node.longitudinal_parameterization||{},point=node.reference_position;
    if(parameter.authority==="wheelbase-half-length")point[0]=Number(parameter.sign)*halfWheelbase+Number(parameter.offset_m||0);
    else if(parameter.authority==="chassis-half-length")point[0]=Number(parameter.fraction||0)*halfLength;
    positions.set(node.identity,point);}
  for(const edge of graph.edges){const a=positions.get(edge.a),b=positions.get(edge.b);if(!a||!b)continue;
    const previous=Math.max(1e-9,Number(edge.rest_length)),next=Math.hypot(...b.map((value,index)=>value-a[index]));
    edge.rest_length=next;if(edge.damage){edge.damage.natural_rest_length=next;
      edge.damage.axial_stiffness_n_per_m=Number(edge.damage.axial_stiffness_n_per_m||0)*previous/Math.max(1e-9,next);}
    const state=vehicleRuntime.damage?.members?.[edge.identity];if(state){state.restLength=next*(1+Number(state.elasticStrain||0)+
      Number(state.plasticStrain||0));edge.runtime_rest_length=state.restLength;}}
  const profile=vehicle.chassis_profiles?.find(item=>item.identity===vehicleRuntime.chassisProfile);
  if(profile){const memberLength=graph.edges.filter(edge=>edge.chassis_profile_member).reduce(
      (sum,edge)=>sum+Number(edge.rest_length),0),dynamic={...profile,member_length_m:memberLength,
      member_mass_kg:memberLength*Number(profile.section_area_m2)*Number(profile.density_kg_m3)};
    applyChassisProfileToVehicle(vehicle,dynamic);}
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-chassis-geometry",identity:vehicle.identity,
    geometry:{chassisHalfLength:halfLength,wheelbaseHalfLength:halfWheelbase,mechanicalGraph:graph,
      mass:Number(vehicle.configuration.mass),defaults:{...vehicle.configuration_defaults}}});
  vehicleRuntime.state.defaults={...vehicle.configuration_defaults};
  shaderViewer.geometry=shaderViewer.geometry.filter(item=>item.identity!==vehicle.identity&&item.parent_identity!==vehicle.identity);
  vehicleRuntime.box=null;vehicleRuntime.cabinBox=null;vehicleRuntime.frameBoxes=[];vehicleRuntime.rollCageBoxes=[];
  vehicleRuntime.mechanicalLinkBoxes=[];vehicleRuntime.suspensionLinkBoxes=[];vehicleRuntime.powertrainBoxes=[];
  vehicleRuntime.bodyShellBoxes=[];vehicleRuntime.wheelBoxes=[];ensureVehiclePresentation(vehicle,vehicleRuntime.state);
  updateVehicleTransmissionControls();setPlacementStatus(`chassis ${length.toFixed(2)} m · wheelbase ${wheelbase.toFixed(2)} m · graph rebuilt`);
  return true;
}

function selectVehicleWheelPart(partIdentity){
  const vehicle=vehicleRuntime.active,part=vehicle?.wheel_parts?.find(item=>item.identity===partIdentity);
  if(!vehicle||!part)return false;
  vehicle.wheel_part=part.identity;vehicleRuntime.wheelPart=part.identity;
  const tires=vehicle.configuration.tires,wheels=vehicle.configuration.wheels,drivetrain=vehicle.configuration.drivetrain,
    reference=vehicle.wheel_part_reference||={radius:Number(tires.radius),width:Number(tires.width),rimRadius:Number(wheels.rim_radius),
      wheelMassKg:Number(drivetrain.wheel_mass_kg),tireMassKg:Number(drivetrain.tire_mass_kg),
      rotationalInertiaScale:Number(drivetrain.rotational_inertia_scale),toroidSectionRadiusM:Number(tires.toroid_section_radius_m),
      effectiveTreadWidthFraction:Number(tires.effective_tread_width_fraction),gasPolytropicExponent:Number(tires.gas_polytropic_exponent),
      radialCarcassLossNsPerM:Number(tires.radial_carcass_loss_n_s_per_m),
      sidewallShearStiffnessLongitudinalNPerM:Number(tires.sidewall_shear_stiffness_longitudinal_n_per_m),
      sidewallShearStiffnessLateralNPerM:Number(tires.sidewall_shear_stiffness_lateral_n_per_m),
      sidewallShearDampingNsPerM:Number(tires.sidewall_shear_damping_n_s_per_m),
      longitudinalModeFrequencyHz:Number(tires.longitudinal_deformation_mode_frequency_hz),
      lateralModeFrequencyHz:Number(tires.lateral_deformation_mode_frequency_hz),
      deformationDampingRatio:Number(tires.sidewall_deformation_damping_ratio),
      maximumSidewallDeformationM:Number(tires.maximum_sidewall_deformation_m)},
    radius=reference.radius*Number(part.radius_scale||1),width=reference.width*Number(part.width_scale||1),
    rimRadius=reference.rimRadius*Number(part.rim_scale||part.radius_scale||1),
    wheelMassKg=Number(part.wheel_mass_kg||reference.wheelMassKg),tireMassKg=Number(part.tire_mass_kg||reference.tireMassKg),
    inertiaScale=Number(part.rotational_inertia_scale||reference.rotationalInertiaScale),
    wheelInertia=inertiaScale*(wheelMassKg*rimRadius**2+.5*tireMassKg*(rimRadius**2+radius**2));
  Object.assign(tires,{radius,width,pressure_pa:Number(part.cold_pressure_kpa||tires.pressure_pa/1000)*1000,
    toroid_section_radius_m:Number(part.toroid_section_radius_m||reference.toroidSectionRadiusM),
    effective_tread_width_fraction:Number(part.effective_tread_width_fraction||reference.effectiveTreadWidthFraction),
    gas_polytropic_exponent:Number(part.gas_polytropic_exponent||reference.gasPolytropicExponent),
    radial_carcass_loss_n_s_per_m:Number(part.radial_carcass_loss_n_s_per_m||reference.radialCarcassLossNsPerM),
    sidewall_shear_stiffness_longitudinal_n_per_m:Number(part.sidewall_shear_stiffness_longitudinal_n_per_m||reference.sidewallShearStiffnessLongitudinalNPerM),
    sidewall_shear_stiffness_lateral_n_per_m:Number(part.sidewall_shear_stiffness_lateral_n_per_m||reference.sidewallShearStiffnessLateralNPerM),
    sidewall_shear_damping_n_s_per_m:Number(part.sidewall_shear_damping_n_s_per_m||reference.sidewallShearDampingNsPerM),
    longitudinal_deformation_mode_frequency_hz:Number(part.longitudinal_deformation_mode_frequency_hz||reference.longitudinalModeFrequencyHz),
    lateral_deformation_mode_frequency_hz:Number(part.lateral_deformation_mode_frequency_hz||reference.lateralModeFrequencyHz),
    sidewall_deformation_damping_ratio:Number(part.sidewall_deformation_damping_ratio||reference.deformationDampingRatio),
    maximum_sidewall_deformation_m:Number(part.maximum_sidewall_deformation_m||reference.maximumSidewallDeformationM)});
  Object.assign(wheels,{rim_radius:rimRadius});Object.assign(drivetrain,{wheel_mass_kg:wheelMassKg,tire_mass_kg:tireMassKg,
    rotational_inertia_scale:inertiaScale});Object.assign(vehicle.configuration_defaults,{wheel_radius:radius,wheel_inertia:wheelInertia,
    tire_longitudinal_deformation_frequency_hz:tires.longitudinal_deformation_mode_frequency_hz,
    tire_lateral_deformation_frequency_hz:tires.lateral_deformation_mode_frequency_hz,
    tire_sidewall_deformation_damping_ratio:tires.sidewall_deformation_damping_ratio,
    tire_maximum_sidewall_deformation:tires.maximum_sidewall_deformation_m});
  vehicleRuntime.wheelBoxes.forEach(wheel=>{
    wheel.height=radius*2;wheel.half_extent=[radius,width*.5];wheel.appearance.face_color=part.tire_color;
    wheel.appearance.tread_color=part.tread_color;wheel.wheel_state={...wheel.wheel_state,radius,
      rimRadius,width,tireColor:part.tire_color,treadColor:part.tread_color,rimColor:part.rim_color,
      coldPressureKpa:Number(part.cold_pressure_kpa||0),compound:part.compound,
      carcassProfile:part.carcass_profile,rimProfile:part.rim_profile,treadPattern:part.tread_pattern,
      dryGripScale:Number(part.dry_grip_scale||1)};});
  vehicleRuntime.state.defaults={...vehicle.configuration_defaults};
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-wheel-part",identity:vehicle.identity,part:{...part}});
  if(Number(part.cold_pressure_kpa)>0)controlVehicleTirePressure(Number(part.cold_pressure_kpa)*1000);
  rebuildPortableSceneMesh();updateVehicleTransmissionControls();setPlacementStatus(`${part.label} · ${radius.toFixed(2)} m radius · ${
    (wheelMassKg+tireMassKg).toFixed(0)} kg each · physical tyre graph rebuilding`);return true;
}

function selectVehicleBodyShell(shellIdentity){
  const vehicle=vehicleRuntime.active,shell=vehicle?.body_shells?.find(item=>item.identity===shellIdentity);
  if(!vehicle||!shell)return false;vehicle.body_shell=shell.identity;vehicleRuntime.bodyShell=shell.identity;
  vehicleRuntime.bodyShellBoxes=buildVehicleBodyShell(vehicle,vehicleRuntime.state);updateVehicleBodyPresentation(vehicle,vehicleRuntime.state);
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-body-shell",identity:vehicle.identity,shellIdentity,
    assembly:shell,ammoCount:shell.identity==="five-turret-carrier"?vehicleRuntime.turretSystem.ammoCount:0});
  rebuildPortableSceneMesh();updateVehicleTransmissionControls();setPlacementStatus(`${shell.label} · breakable mounted collision shell`);
  return true;
}

function controlVehicleOutriggers(deployed){
  const vehicle=vehicleRuntime.active;if(!vehicle||vehicleRuntime.bodyShell!=="five-turret-carrier")return false;
  vehicleRuntime.turretSystem.outriggers.commanded=Boolean(deployed);
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-outriggers",identity:vehicle.identity,deployed:Boolean(deployed)});
  setPlacementStatus(deployed?"outriggers extending · feet weld on first terrain crossing":
    "outriggers withdrawing · terrain welds hold until fully retracted");updateVehicleTransmissionControls();return true;
}

function controlVehicleChassisLeveling(update){
  const vehicle=vehicleRuntime.active;if(!vehicle)return false;
  const leveling=vehicleRuntime.chassisLeveling||={...(vehicle.chassis_leveling||{})};
  if(typeof update.enabled==="boolean")leveling.enabled=update.enabled;
  if(typeof update.mode==="string")leveling.mode=update.mode;
  if(typeof update.active_pose==="string")leveling.active_pose=update.active_pose;
  for(const key of ["target_ride_height_offset_m","target_roll_rad","target_pitch_rad","pose_lerp_rate_m_s"])
    if(Number.isFinite(update[key]))leveling[key]=Number(update[key]);
  if(update.manual_corner_targets_m)leveling.manual_corner_targets_m={...(leveling.manual_corner_targets_m||{}),
    ...Object.fromEntries(Object.entries(update.manual_corner_targets_m).filter(([,value])=>Number.isFinite(value)))};
  if(update.programmed_poses)leveling.programmed_poses={...(leveling.programmed_poses||{}),...update.programmed_poses};
  vehicle.chassis_leveling={...leveling};
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-chassis-leveling",identity:vehicle.identity,
    leveling:{...leveling}});
  updateVehicleTransmissionControls();setPlacementStatus(`chassis leveling ${leveling.enabled?"armed":"off"} · ride ${
    Number(leveling.target_ride_height_offset_m||0).toFixed(2)} m · roll/pitch targets ${
    (Number(leveling.target_roll_rad||0)*180/Math.PI).toFixed(1)}°/${
    (Number(leveling.target_pitch_rad||0)*180/Math.PI).toFixed(1)}°`);return true;
}

function applyVehicleHydraulicPose(identity){
  const leveling=vehicleRuntime.chassisLeveling,pose=(leveling.pose_presets||[]).find(item=>item.identity===identity)||
    leveling.programmed_poses?.[identity];if(!pose)return false;
  return controlVehicleChassisLeveling({enabled:true,mode:"manual-wheel",active_pose:identity,
    manual_corner_targets_m:{...pose.corners}});
}

function programVehicleHydraulicPose(slot){
  const leveling=vehicleRuntime.chassisLeveling;if(!leveling)return false;const corners={...(leveling.manual_corner_targets_m||{})},
    pose={identity:slot,label:slot,corners};controlVehicleChassisLeveling({programmed_poses:{[slot]:pose}});
  saveLivingEdits(null);
  setPlacementStatus(`hydraulic pose ${slot} stored from four wheel-height targets`);return true;
}

function controlVehicleTirePressure(targetPa){
  const vehicle=vehicleRuntime.active;if(!vehicle||!Number.isFinite(targetPa))return false;
  const electrical=vehicle.configuration.electrical,minimum=Number(electrical.minimum_tire_pressure_pa||45000),
    maximum=Number(electrical.maximum_tire_pressure_pa||260000);
  vehicleRuntime.electrical.tirePressureTargetPa=Math.max(minimum,Math.min(maximum,Number(targetPa)));
  vehicle.tire_pressure_target_pa=vehicleRuntime.electrical.tirePressureTargetPa;saveLivingEdits(null);
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-pneumatics",identity:vehicle.identity,
    tirePressureTargetPa:vehicleRuntime.electrical.tirePressureTargetPa});updateVehicleTransmissionControls();return true;
}

function controlVehicleSteeringSystem(update){
  const vehicle=vehicleRuntime.active;if(!vehicle)return false;
  const steeringSystem=vehicleRuntime.steeringSystem||={...(vehicle.steering_control||vehicle.configuration.steering_control||{})};
  if(typeof update.front_axle_enabled==="boolean")steeringSystem.front_axle_enabled=update.front_axle_enabled;
  if(typeof update.rear_axle_enabled==="boolean")steeringSystem.rear_axle_enabled=update.rear_axle_enabled;
  for(const key of ["front_share","rear_phase"])if(Number.isFinite(update[key]))steeringSystem[key]=Number(update[key]);
  vehicle.steering_control={...steeringSystem};vehicle.configuration.steering_control={...steeringSystem};
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-steering-system",identity:vehicle.identity,
    steering:{...steeringSystem}});
  updateVehicleTransmissionControls();setPlacementStatus(`steering · ${steeringSystem.front_axle_enabled?"front":"front free"} / ${
    steeringSystem.rear_axle_enabled?"rear":"rear free"} · ${Math.round(Number(steeringSystem.front_share||0)*100)}% front authority · ${
    Number(steeringSystem.rear_phase||-1)<0?"counter-phase":"same-phase"}`);return true;
}

function controlVehicleShockParameters(parameters){
  const vehicle=vehicleRuntime.active;if(!vehicle)return false;
  const clean={};for(const [name,value] of Object.entries(parameters||{}))if(Number.isFinite(value))clean[name]=Number(value);
  if(!Object.keys(clean).length)return false;Object.assign(vehicle.configuration_defaults,clean);
  vehicleRuntime.shockParameters={...vehicle.configuration_defaults};
  const suspension=vehicle.configuration.suspension,mapping={suspension_rest_length:"rest_length",suspension_travel:"travel",
    spring_stiffness:"stiffness",pneumatic_compression_damping:"pneumatic_compression_damping",
    pneumatic_rebound_damping:"pneumatic_rebound_damping",pneumatic_efficiency:"pneumatic_efficiency",
    active_damping_minimum_scale:"active_damping_minimum_scale",active_damping_maximum_scale:"active_damping_maximum_scale",
    active_damping_body_velocity_gain_s_per_m:"active_damping_body_velocity_gain_s_per_m",
    active_damping_rebound_release_gain_s_per_m:"active_damping_rebound_release_gain_s_per_m"};
  for(const [parameter,key] of Object.entries(mapping))if(Number.isFinite(clean[parameter]))suspension[key]=clean[parameter];
  vehicleRuntime.state.defaults={...vehicle.configuration_defaults};updateVehicleTransmissionControls();
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-parameters",identity:vehicle.identity,parameters:clean});
  setPlacementStatus(`parametric shock control · ${Object.entries(clean).map(([name,value])=>`${name}=${Number(value).toPrecision(4)}`).join(" · ")}`);
  return true;
}

function controlVehicleWheelAlignment({corner,parameter,value,linked=false,calibrationMode=null}={}){
  const vehicle=vehicleRuntime.active,alignment=vehicleRuntime.wheelAlignment;if(!vehicle||!alignment)return false;
  if(calibrationMode){alignment.calibration={...(alignment.calibration||{}),requested:calibrationMode!=="static",
      continuous:calibrationMode==="full-time-auto",settledTicks:0,status:calibrationMode};}
  else if(alignment.corners?.[corner]&&["camber_deg","caster_deg","toe_deg"].includes(parameter)&&Number.isFinite(value)){
    const range=alignment.ranges?.[parameter]||[-12,12],numeric=Math.max(Number(range[0]),Math.min(Number(range[1]),Number(value))),
      targets=linked?Object.keys(alignment.corners):[corner];for(const name of targets)alignment.corners[name][parameter]=numeric;
  }else return false;
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"vehicle-wheel-alignment",identity:vehicle.identity,
    alignment:JSON.parse(JSON.stringify(alignment))});updateVehicleTransmissionControls();
  setPlacementStatus(calibrationMode?`alignment mode · ${calibrationMode}`:
    `${linked?"linked":"corner"} alignment · ${parameter} ${Number(value).toFixed(2)}°`);return true;
}

function updateActiveVehicle(dt,throttle,steering,brake) {
  const vehicle=vehicleRuntime.active,state=vehicleRuntime.state;
  if(!vehicle||!state)return false;
  try{
  state.lastThrottle=Number(throttle||0);
  if(stateLoopRuntime.ready){
    registerActiveVehiclePhysicsBody();
    stateLoopRuntime.worker.postMessage({type:"vehicle-control",identity:vehicle.identity,
      throttle,steering,brake});
    const body=stateLoopRuntime.bodies.get(vehicle.identity);
    if(body){state.position=[...body.position];state.velocity=[...body.velocity];
      state.roll=body.roll||0;state.pitch=body.pitch||0;state.yaw=body.yaw||0;
      state.rollVelocity=body.rollVelocity||0;state.pitchVelocity=body.pitchVelocity||0;state.yawVelocity=body.yawVelocity||0;
      ["front_left","front_right","rear_left","rear_right"].forEach((name,index)=>{
        state.wheelOmegas[name]=body.wheelOmegas?.[index]||0;
        state.compressions[name]=body.compressions?.[index]||0;
      });}
    synchronizeVehicleLookYaw(state);
    const chassis=vehicle.configuration.chassis;updateVehiclePresentation(vehicle,state,dt,steering);
    updateVehicleBodyPresentation(vehicle,state);
    viewportControls.position=[state.position[0],state.position[1]+Number(chassis.camera_height),state.position[2]];
    viewportControls.horizontalVelocity=[state.velocity[0],state.velocity[2]];
    const actor=entityState.get(viewportControls.policy?.actor);
    if(actor){actor.worldPosition=[...viewportControls.position];actor.velocity=[...state.velocity];}
    uploadVehiclePresentationMesh(dt);updateEngineSoundTelemetry(throttle);return true;
  }
  // Initialization owns the detailed pipeline error.  Do not overwrite it on
  // every presentation frame with a generic not-ready symptom.
  return false;
  }catch(error){reportRuntimeFault("vehicle-step",error);return false;}
}

function updateParkedVehicle(dt){
  const vehicle=model.vehicle_slot?.vehicles?.[0],state=vehicleRuntime.parkedState;
  if(vehicleRuntime.active||!vehicle||!state)return false;
  const body=stateLoopRuntime.bodies.get(vehicle.identity);if(body){state.position=[...body.position];state.velocity=[...body.velocity];
    state.roll=body.roll||0;state.pitch=body.pitch||0;state.yaw=body.yaw||0;
    state.rollVelocity=body.rollVelocity||0;state.pitchVelocity=body.pitchVelocity||0;state.yawVelocity=body.yawVelocity||0;}
  updateVehiclePresentation(vehicle,state,dt,0);updateVehicleBodyPresentation(vehicle,state);
  uploadVehiclePresentationMesh(dt);updateEngineSoundTelemetry(0);return true;
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

function registerActiveVehiclePhysicsBody() {
  const vehicle=vehicleRuntime.active,state=vehicleRuntime.state;
  if(!stateLoopRuntime.ready||!vehicle||!state||stateLoopRuntime.bodies.has(vehicle.identity))return;
  const snapshot=reservePhysicsSnapshotSlot(vehicle.identity);state.identity=vehicle.identity;
  snapshot.position=[...state.position];snapshot.velocity=[...state.velocity];snapshot.yaw=state.yaw;
  stateLoopRuntime.worker.postMessage({type:"upsert",body:{identity:vehicle.identity,kind:"vehicle",
    slot:snapshot.slot,generation:snapshot.generation,position:[...state.position],velocity:[...state.velocity],
    radius:Number(vehicle.configuration.chassis.half_width),roll:state.roll,pitch:state.pitch,yaw:state.yaw,
    rollVelocity:state.rollVelocity,pitchVelocity:state.pitchVelocity,yawVelocity:state.yawVelocity,
    wheelOmegas:{...state.wheelOmegas},
    wheelSteerAngles:{...(state.wheelSteerAngles||{})},
    previousSlips:{...state.previousSlips},
    compressions:{...state.compressions},controls:{throttle:0,steering:0,brake:0},bodyShell:vehicleRuntime.bodyShell,
    fuelProfile:vehicle.fuel_profile,ignitionProfile:vehicle.ignition_profile,bodyAssemblies:vehicle.body_shells,
    bodyAssemblyInterface:vehicle.body_assembly_interface,turretAmmoCount:vehicleRuntime.turretSystem.ammoCount,
    fuelProfiles:vehicle.fuel_profiles,ignitionProfiles:vehicle.ignition_profiles,
    powerUnitCompatibleFuels:[...(vehicle.power_unit_presets?.find(item=>item.identity===vehicleRuntime.powerUnitPreset)?.compatible_fuel_profiles||[])],
    powerUnitFuelCompatibility:{...(vehicle.power_unit_presets?.find(item=>item.identity===vehicleRuntime.powerUnitPreset)?.fuel_compatibility||{})},
    powerUnitIgnitionCompatibility:{...(vehicle.power_unit_presets?.find(item=>item.identity===vehicleRuntime.powerUnitPreset)?.ignition_compatibility||{})},
    engineKernelSwitch:vehicle.engine_kernel_switch,
    engineEquationMode:vehicleRuntime.engineEquationMode,
    engineProfileSelector:Number(vehicle.power_unit_presets?.find(item=>item.identity===vehicleRuntime.powerUnitPreset)?.kernel_selector||0)*2+
      (vehicleRuntime.engineEquationMode==="symbolic-fidelity"?1:0),
    drivingMode:vehicleRuntime.driverAssistance.drivingMode,drivingModes:vehicle.driving_modes,
    driverAssistance:{...vehicleRuntime.driverAssistance},
    energySeed:{...vehicleRuntime.electrical},
    transmission:{...vehicleRuntime.transmission},
    brakeLocks:{...vehicleRuntime.brakeLocks},
    config:{...vehicle.configuration,mechanical_graph:vehicle.physics.mechanical_graph,
      chassis_leveling:{...vehicleRuntime.chassisLeveling},wheel_alignment:JSON.parse(JSON.stringify(vehicleRuntime.wheelAlignment)),
      steering_control:{...vehicleRuntime.steeringSystem}},
    wheelAlignment:JSON.parse(JSON.stringify(vehicleRuntime.wheelAlignment)),
    steeringSystem:{...vehicleRuntime.steeringSystem},
    defaults:vehicle.configuration_defaults,overrides:{}}});
  if(model.vehicle_slot?.initial_state?.placement==="authored-world-pose"){
    stateLoopRuntime.worker.postMessage({type:"vehicle-respawn",identity:vehicle.identity,
      position:[...state.position],yaw:state.yaw,roll:state.roll,pitch:state.pitch});
  }
}

function setPhysicsParameter(name, rawValue) {
  const descriptor = model.physics_program?.parameters.find(item => item.name === name);
  let value = Number(rawValue);
  if (!descriptor || !Number.isFinite(value)) return false;
  if (descriptor.minimum !== undefined) value = Math.max(descriptor.minimum, value);
  if (descriptor.maximum !== undefined) value = Math.min(descriptor.maximum, value);
  physicsRuntime.parameters.set(name, value);
  if(name==="linear_drag"&&model.projectiles?.archetype?.physics)
    model.projectiles.archetype.physics.linear_drag=value;
  if (stateLoopRuntime.ready) stateLoopRuntime.worker.postMessage({type: "parameters",
    parameters: {[name]: value}});
  wakeSleepingProjectiles(`physics-field-change:${name}`);
  descriptor.value = value;
  saveLivingEdits(null);
  const actor = viewportControls.policy?.actor || model.identity;
  const destination = `${model.physics_program.identity}/parameters/${name}`;
  const edge = actionEdges.register(actor, "edit-physics-parameter", destination);
  abstractUISystemTimer.issue({actor, type: "edit-physics-parameter", destination,
    edge, parameter: name, value, issued_at: performance.now()});
  return true;
}

function selectWallContact(position, radius, excludedObject = null) {
  let selected = null;
  shaderViewer.colliders.forEach(collider => {
    if (collider.objectIdentity === excludedObject || collider.surface) return;
    const [minimumX, minimumY, minimumZ] = collider.minimum;
    const [maximumX, maximumY, maximumZ] = collider.maximum;
    if (position[1] + radius <= minimumY || position[1] - radius >= maximumY ||
        position[0] < minimumX - radius || position[0] > maximumX + radius ||
        position[2] < minimumZ - radius || position[2] > maximumZ + radius) return;
    const closestX = Math.max(minimumX, Math.min(maximumX, position[0]));
    const closestZ = Math.max(minimumZ, Math.min(maximumZ, position[2]));
    if ((position[0] - closestX) ** 2 + (position[2] - closestZ) ** 2 > radius ** 2) return;
    const faces = [
      {penetration: position[0] + radius - minimumX, normal: [1, 0], plane: minimumX},
      {penetration: maximumX + radius - position[0], normal: [-1, 0], plane: -maximumX},
      {penetration: position[2] + radius - minimumZ, normal: [0, 1], plane: minimumZ},
      {penetration: maximumZ + radius - position[2], normal: [0, -1], plane: -maximumZ},
    ].filter(face => face.penetration >= 0)
      .sort((left, right) => left.penetration - right.penetration);
    const face = faces[0];
    if (face && (!selected || face.penetration < selected.penetration)) {
      selected = {...face, collider};
    }
  });
  return selected;
}

function sampleDeclaredSurface(surface,worldX,worldZ){
  if(surface.kind!=="sampled-height-field")return {height:surface.origin[1]+
    surface.gradient[0]*(worldX-surface.origin[0])+surface.gradient[1]*(worldZ-surface.origin[2]),
    gradient:[...surface.gradient]};
  const [columns,rows]=surface.resolution,[cellX,cellZ]=surface.cell_size,
    u=Math.max(0,Math.min(columns-1,(worldX-surface.origin[0])/cellX)),
    v=Math.max(0,Math.min(rows-1,(worldZ-surface.origin[2])/cellZ)),
    column=Math.min(columns-2,Math.floor(u)),row=Math.min(rows-2,Math.floor(v)),tx=u-column,tz=v-row,
    at=(x,z)=>Number(surface.heights[z*columns+x]),h00=at(column,row),h10=at(column+1,row),
    h01=at(column,row+1),h11=at(column+1,row+1);
  if(tx>=tz)return {height:h00+(h10-h00)*tx+(h11-h10)*tz,
    gradient:[(h10-h00)/cellX,(h11-h10)/cellZ]};
  return {height:h00+(h11-h01)*tx+(h01-h00)*tz,
    gradient:[(h11-h01)/cellX,(h01-h00)/cellZ]};
}

function sampleContactSurface(worldX,worldZ,previousBaseY,candidateBaseY,verticalVelocity,reach=.08) {
  const candidates=[];
  const terrainReplacesFloor=shaderViewer.colliders.some(collider=>{const surface=collider.surface,domain=surface?.domain;
    return surface?.kind==="sampled-height-field"&&worldX>=domain.minimum_x&&worldX<=domain.maximum_x&&
      worldZ>=domain.minimum_z&&worldZ<=domain.maximum_z;});
  if(!terrainReplacesFloor&&verticalVelocity<=.2&&previousBaseY>=-reach&&candidateBaseY<=reach){
    candidates.push({supported:true,height:0,gradient:[0,0],normal:[0,1,0],
      identity:"world-floor",runtimePartId:0});
  }
  shaderViewer.colliders.forEach(collider=>{
    const surface=collider.surface;if(!surface)return;
    const domain=surface.domain;
    if(worldX<domain.minimum_x-1e-7||worldX>domain.maximum_x+1e-7||
       worldZ<domain.minimum_z-1e-7||worldZ>domain.maximum_z+1e-7)return;
    const sampled=sampleDeclaredSurface(surface,worldX,worldZ),height=sampled.height,gradient=sampled.gradient;
    const crossed=verticalVelocity<=.2&&previousBaseY>=height-reach&&candidateBaseY<=height+reach;
    const retained=Math.abs(previousBaseY-height)<=reach*1.5&&candidateBaseY<=height+reach;
    if(!crossed&&!retained)return;
    const normal=normalized3([-gradient[0],1,-gradient[1]]);
    candidates.push({supported:true,height,gradient,normal,
      identity:collider.identity,runtimePartId:collider.runtimePartId});
  });
  candidates.sort((left,right)=>right.height-left.height||left.runtimePartId-right.runtimePartId);
  return candidates[0]||null;
}

function runCompiledPhysicsState(position, velocity, radius, dt,
                                 overrides = {}, excludedObject = null) {
  const instance = physicsRuntime.instance;
  const plugin = physicsRuntime.plugin;
  if (!instance || !plugin?.abi || dt <= 0) return null;
  const contact = selectWallContact(position, radius, excludedObject);
  const dynamic = {
    position_x: position[0], position_y: position[1], position_z: position[2],
    velocity_x: velocity[0], velocity_y: velocity[1], velocity_z: velocity[2],
    dt, obstacle_active: contact ? 1 : 0,
    obstacle_normal_x: contact?.normal[0] || 0,
    obstacle_normal_z: contact?.normal[1] || 0,
    obstacle_plane: contact?.plane || 0,
    radius, ...overrides,
  };
  const memory = new Float64Array(instance.exports.memory.buffer);
  plugin.abi.input_names.forEach((name, index) => {
    const value = dynamic[name] ?? physicsRuntime.parameters.get(name) ?? 0;
    memory[plugin.abi.input_offsets[index] / 8] = value;
  });
  instance.exports[plugin.entrypoint](0);
  const result = {};
  plugin.abi.output_names.forEach((name, index) => {
    result[name] = memory[plugin.abi.output_offsets[index] / 8];
  });
  if (!Object.values(result).every(Number.isFinite)) return null;
  return {...result, contactIdentity: contact?.collider.identity || null,
    contactRuntimePartId: contact?.collider.runtimePartId || 0};
}

function portalOpeningTransform(host, opening) {
  const thickness = Math.max(0.01, Number(host.appearance?.wall_thickness ||
    host.wall_thickness || 0.04));
  const halfX = host.half_extent[0], halfZ = host.half_extent[1];
  const local = {center: [0, 0], normal: [0, 0]};
  if (opening.side === "north") {
    local.center = [opening.offset, halfZ - thickness * 0.5]; local.normal = [0, 1];
  } else if (opening.side === "south") {
    local.center = [opening.offset, -halfZ + thickness * 0.5]; local.normal = [0, -1];
  } else if (opening.side === "east") {
    local.center = [halfX - thickness * 0.5, opening.offset]; local.normal = [1, 0];
  } else {
    local.center = [-halfX + thickness * 0.5, opening.offset]; local.normal = [-1, 0];
  }
  const yaw = boxYawDegrees(host) * Math.PI / 180;
  const cosine = Math.cos(yaw), sine = Math.sin(yaw);
  const rotate = point => [point[0] * cosine - point[1] * sine,
    point[0] * sine + point[1] * cosine];
  const center = rotate(local.center), normal = rotate(local.normal);
  const bottom = Number(host.placement?.elevation || 0) + Number(opening.bottom || 0);
  return {host, opening, position: [host.center[0] + center[0],
    bottom + Number(opening.height) * 0.5, host.center[1] + center[1]],
    normal, tangent: [-normal[1], normal[0]], width: Number(opening.width),
    height: Number(opening.height), bottom};
}

function choosePortalGraphEdge(sourceIdentity) {
  const edges = portalRuntime.graph.edges.filter(edge => edge.source === sourceIdentity);
  if (!edges.length) return null;
  let sample = Math.random();
  for (const edge of edges) {
    sample -= edge.weight;
    if (sample <= 0) return edge;
  }
  return edges.at(-1);
}

function synchronizePortalBody(identity, position, velocity) {
  const body = stateLoopRuntime.bodies.get(identity);
  if (!body) return;
  body.position = [...position];
  if (velocity) body.velocity = [...velocity];
  body.controlGeneration += 1;
  body.lastSubmittedPosition = [position[0], position[2]];
  if (stateLoopRuntime.ready) {
    stateLoopRuntime.worker.postMessage({type: "control", identity,
      position: [position[0], position[2]], generation: body.controlGeneration});
    if (velocity) stateLoopRuntime.worker.postMessage({type: "impulse", identity,
      velocity: [...velocity]});
  }
}

function advancePortalTransit(identity, position, velocity, radius, now) {
  const transit = portalRuntime.transits.get(identity);
  if (!transit) return false;
  const edge = portalRuntime.graph.edges.find(candidate => candidate.identity === transit.edge);
  const source = portalRuntime.splats.find(splat => splat.identity === transit.source);
  const target = portalRuntime.splats.find(splat => splat.identity === transit.target);
  if (!edge || !source || !target) { portalRuntime.transits.delete(identity); return false; }
  const t = Math.max(0, Math.min(1, (now - transit.started) / transit.duration));
  const point = portalCurvePoint(edge, t), frame = portalCurveFrame(edge, source, t);
  const suctionProgress = Math.max(0, Math.min(1, t / 0.22));
  const suction = 1 - suctionProgress * suctionProgress * (3 - 2 * suctionProgress);
  point.forEach((_, axis) => { point[axis] += suction *
    (source.tangent[axis] * transit.local_u + source.bitangent[axis] * transit.local_v); });
  point.forEach((value, axis) => { position[axis] = value; });
  if (velocity) frame.tangent.forEach((value, axis) => {
    velocity[axis] = value * portalRuntime.graph.traversalSpeed;
  });
  if (identity === viewportControls.policy?.actor) {
    const transportedFacing = normalized3(quaternionRotate(frame.orientation,
      transit.entry_facing || [1, 0, 0]));
    viewportControls.yaw = Math.atan2(transportedFacing[2], transportedFacing[0]);
    viewportControls.pitch = Math.asin(Math.max(-1, Math.min(1, transportedFacing[1])));
  }
  if (t < 1) { synchronizePortalBody(identity, position, velocity); return true; }
  const clearance = Math.max(0.09, radius + 0.035);
  position.forEach((_, axis) => { position[axis] = target.center[axis] +
    target.tangent[axis] * transit.local_u + target.bitangent[axis] * transit.local_v +
    target.normal[axis] * clearance; });
  const orientation = quaternionBetween(source.normal.map(value => -value), target.normal);
  if (velocity) {
    const transformed = quaternionRotate(orientation, transit.entry_velocity);
    transformed.forEach((value, axis) => { velocity[axis] = value; });
  }
  if (identity === viewportControls.policy?.actor) {
    cancelEntityNavigation(identity);
  }
  portalRuntime.transits.delete(identity);
  portalRuntime.lastPositions.set(identity, [...position]);
  portalRuntime.cooldowns.set(identity, now + portalRuntime.cooldownMilliseconds);
  synchronizePortalBody(identity, position, velocity);
  return true;
}

function traversePortalBody(identity, position, velocity, radius = 0.001) {
  if (!position) return false;
  const now = performance.now();
  if (portalRuntime.transits.has(identity)) {
    return advancePortalTransit(identity, position, velocity, radius, now);
  }
  const previous = portalRuntime.lastPositions.get(identity);
  portalRuntime.lastPositions.set(identity, [...position]);
  if (!previous || now < (portalRuntime.cooldowns.get(identity) || 0)) return false;
  for (const source of portalRuntime.splats.filter(splat => splat.port_role === "in")) {
    const previousOffset = previous.map((value, axis) => value - source.center[axis]);
    const currentOffset = position.map((value, axis) => value - source.center[axis]);
    const previousDistance = previousOffset.reduce((sum, value, axis) =>
      sum + value * source.normal[axis], 0);
    const currentDistance = currentOffset.reduce((sum, value, axis) =>
      sum + value * source.normal[axis], 0);
    const crossedPlane = previousDistance > 0 && currentDistance <= 0;
    const touchedBlockedPlane = currentDistance > 0 &&
      currentDistance <= Math.max(0.12, radius + 0.06) &&
      previousDistance > currentDistance + 1e-5;
    if (!crossedPlane && !touchedBlockedPlane) continue;
    const denominator = previousDistance - currentDistance;
    const fraction = crossedPlane && Math.abs(denominator) >= 1e-8
      ? previousDistance / denominator : 1;
    const crossing = previous.map((value, axis) => value +
      (position[axis] - value) * fraction);
    const crossingOffset = crossing.map((value, axis) => value - source.center[axis]);
    const localU = crossingOffset.reduce((sum, value, axis) => sum + value * source.tangent[axis], 0);
    const localV = crossingOffset.reduce((sum, value, axis) => sum + value * source.bitangent[axis], 0);
    if (Math.hypot(localU, localV) > source.radius + radius) continue;
    const graphEdge = choosePortalGraphEdge(source.identity);
    if (!graphEdge) { setPlacementStatus("IN splat has no OUT distribution yet"); continue; }
    portalRuntime.transits.set(identity, {edge: graphEdge.identity, source: source.identity,
      target: graphEdge.target, started: now,
      duration: Math.max(280, graphEdge.length / portalRuntime.graph.traversalSpeed * 1000),
      local_u: localU, local_v: localV, entry_velocity: velocity ? [...velocity] : [0, 0, 0],
      entry_facing: identity === viewportControls.policy?.actor
        ? normalized3(shaderViewer.cameraFacing || [1, 0, 0]) : null});
    const registeredEdge = actionEdges.register(source.identity, "traverse-portal-tube", graphEdge.target);
    abstractUISystemTimer.issue({actor: identity, type: "traverse-portal-tube",
      source: source.identity, destination: graphEdge.target,
      backing_graph: portalRuntime.graph.identity, probability: graphEdge.weight,
      path_model: graphEdge.path_model, edge: registeredEdge, issued_at: now});
    return advancePortalTransit(identity, position, velocity, radius, now);
  }
  return false;
}

function updatePortalTraversals() {
  const playerIdentity = viewportControls.policy?.actor;
  if (playerIdentity && viewportControls.position) {
    const body = stateLoopRuntime.bodies.get(playerIdentity);
    const velocity = body?.velocity || [0, physicsRuntime.verticalVelocity, 0];
    if (traversePortalBody(playerIdentity, viewportControls.position, velocity,
        Number(model.viewer.camera.collision_radius || 0.001))) {
      const actor = entityState.get(playerIdentity);
      if (actor) { actor.worldPosition = [...viewportControls.position]; actor.velocity = [...velocity]; }
    }
  }
  projectileState.forEach(state => {
    const radius = Number(state.box.radius || model.projectiles?.archetype?.geometry?.radius || 0.001);
    if (traversePortalBody(state.identity, state.position, state.velocity, radius)) {
      state.box.center = [state.position[0], state.position[2]];
      state.box.center_y = state.position[1];
      state.entityRuntime.worldPosition = [...state.position];
      state.entityRuntime.position = [...state.position];
      state.entityRuntime.velocity = [...state.velocity];
    }
  });
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

function requestRepresentationTransition(position) {
  const boundary = model.document_geometry.representation_boundary;
  if (!boundary || viewportControls.representationTransition) return;
  viewportControls.representationTransition = {
    boundary: boundary.identity, operation: boundary.crossing_operation,
    target: boundary.outside_representation, position: [...position]
  };
  const actor = viewportControls.policy?.actor || model.identity;
  const edge = actionEdges.register(actor, boundary.crossing_operation, boundary.identity);
  abstractUISystemTimer.issue({
    actor, type: boundary.crossing_operation, destination: boundary.identity,
    edge, target_representation: boundary.outside_representation,
    issued_at: performance.now()
  });
  shaderViewer.readout.textContent =
    `representation boundary · ${boundary.outside_representation}`;
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
    horn=viewportInputValue("vehicle-horn",gamepad)>0,starter=viewportInputValue("vehicle-starter",gamepad)>0;
  if(vehicleRuntime.active&&headlights&&!viewportControls.headlightsDown)
    controlVehicleAuxiliary({headlightsOn:!vehicleRuntime.electrical.headlightsOn});
  if(vehicleRuntime.active&&horn!==viewportControls.hornDown)controlVehicleAuxiliary({hornOn:horn});
  if(vehicleRuntime.active&&starter!==viewportControls.starterDown)controlVehicleAuxiliary({starterEngaged:starter});
  viewportControls.headlightsDown=headlights;viewportControls.hornDown=horn;viewportControls.starterDown=starter;
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
    const throttle=mobileClamp(forward*(running?1.18:1));
    updateActiveVehicle(dt,throttle,strafe,jumping?1:0);return;
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

function softwareMaterial(mesh, vertexOffset, depth) {
  const normal = normalized3([mesh[vertexOffset + 3], mesh[vertexOffset + 4], mesh[vertexOffset + 5]]);
  const color = [mesh[vertexOffset + 6], mesh[vertexOffset + 7], mesh[vertexOffset + 8]];
  const celestial=shaderViewer.celestial || celestialState();
  const keyLight = celestial.key === "sun" ? celestial.sunDirection : celestial.moonDirection;
  const fillLight = normalized3([0.72, 0.35, -0.28]);
  const dot = (left, right) => left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
  const key = Math.max(0, dot(normal, keyLight));
  const illumination = (celestial.key === "sun" ? .42 : .24) + 0.72 * key +
    0.28 * Math.max(0, dot(normal, fillLight));
  const fog = Math.exp(-depth * 0.018);
  const sky = colorVector(model.appearance.colors.sky);
  const light = celestial.key === "sun" ? celestial.sunColor : celestial.moonColor;
  const lit = color.map((channel, index) => channel * illumination + light[index] * key * 0.10);
  const mixed = lit.map((channel, index) => sky[index] * (1 - fog) + channel * fog);
  return `rgb(${mixed.map(channel => Math.round(Math.max(0, Math.min(1, channel)) * 255)).join(",")})`;
}

function drawCompiledMeshViewer(context, width, height, cameraPosition, cameraFacing, clearBackground = true) {
  const colors = model.appearance.colors;
  if (clearBackground) {
    const sky = context.createLinearGradient(0, 0, 0, height);
    sky.addColorStop(0, colors.sky); sky.addColorStop(1, colors.ground);
    context.fillStyle = sky; context.fillRect(0, 0, width, height);
  }
  const wasm = shaderViewer.softwareWasm;
  const forward = normalized3(cameraFacing);
  const right = normalized3(cross3(forward, [0, 1, 0]));
  const up = normalized3(cross3(right, forward));
  const values = {
    camera_x: cameraPosition[0], camera_y: cameraPosition[1], camera_z: cameraPosition[2],
    forward_x: forward[0], forward_y: forward[1], forward_z: forward[2],
    right_x: right[0], right_y: right[1], right_z: right[2],
    up_x: up[0], up_y: up[1], up_z: up[2], width, height
  };
  Object.entries(values).forEach(([name, value]) => wasm.arrays[name].fill(value));
  const args = wasm.descriptor.parameters.map(parameter =>
    parameter.role === "extent" ? wasm.count : wasm.offsets[parameter.name]);
  wasm.run(...args);
  const screenX = wasm.arrays.screen_x, screenY = wasm.arrays.screen_y, depth = wasm.arrays.view_z;
  const triangles = [];
  for (let vertex = 0; vertex < wasm.count; vertex += 3) {
    const depths = [depth[vertex], depth[vertex + 1], depth[vertex + 2]];
    const coordinates = [
      [screenX[vertex], screenY[vertex]],
      [screenX[vertex + 1], screenY[vertex + 1]],
      [screenX[vertex + 2], screenY[vertex + 2]]
    ];
    if (depths.some(value => value <= 0.04) || coordinates.flat().some(value => !Number.isFinite(value))) continue;
    const meanDepth = (depths[0] + depths[1] + depths[2]) / 3;
    triangles.push({coordinates, depth: meanDepth, vertexOffset: vertex * 9});
  }
  triangles.sort((left, rightTriangle) => rightTriangle.depth - left.depth);
  shaderViewer.softwareTriangleCount = triangles.length;
  shaderViewer.softwareOnscreenCount = triangles.filter(triangle => {
    const xs = triangle.coordinates.map(point => point[0]);
    const ys = triangle.coordinates.map(point => point[1]);
    return Math.max(...xs) >= 0 && Math.min(...xs) <= width &&
      Math.max(...ys) >= 0 && Math.min(...ys) <= height;
  }).length;
  context.lineJoin = "round";
  triangles.forEach(triangle => {
    context.beginPath();
    context.moveTo(...triangle.coordinates[0]);
    context.lineTo(...triangle.coordinates[1]);
    context.lineTo(...triangle.coordinates[2]);
    context.closePath();
    context.fillStyle = softwareMaterial(shaderViewer.mesh, triangle.vertexOffset, triangle.depth);
    context.fill();
    context.strokeStyle = colors.line + "88";
    context.lineWidth = Math.max(0.7, width / 1400);
    context.stroke();
  });
}

function drawSoftwareViewer(context, width, height, cameraPosition, cameraFacing) {
  const colors = model.appearance.colors;
  const celestial=shaderViewer.celestial || celestialState();
  const sky = context.createLinearGradient(0, 0, 0, height * 0.58);
  sky.addColorStop(0,celestial.key==="sun"?colors["sky-day"]:colors["sky-night"]);
  sky.addColorStop(.72,colors.sky);sky.addColorStop(1,colors["sky-horizon"]);
  context.fillStyle = sky; context.fillRect(0, 0, width, height * 0.58);
  const ground = context.createLinearGradient(0, height * 0.48, 0, height);
  ground.addColorStop(0, colors.courtyard); ground.addColorStop(1, colors.void);
  context.fillStyle = ground; context.fillRect(0, height * 0.58, width, height * 0.42);
  const drawBody=(direction,color,radius)=>{
    const forward=normalized3(cameraFacing),right3=normalized3(cross3(forward,[0,1,0]));
    const up3=normalized3(cross3(right3,forward));
    const view=[direction[0]*right3[0]+direction[1]*right3[1]+direction[2]*right3[2],
      direction[0]*up3[0]+direction[1]*up3[1]+direction[2]*up3[2],
      direction[0]*forward[0]+direction[1]*forward[1]+direction[2]*forward[2]];
    if(view[2]<=.02)return;
    const x=width*(.5+.5*view[0]/(.70*(width/height)*view[2]));
    const y=height*(.5-.5*view[1]/(.70*view[2]));
    const glow=context.createRadialGradient(x,y,0,x,y,radius*3.2);
    glow.addColorStop(0,color);glow.addColorStop(.32,color+"cc");glow.addColorStop(1,color+"00");
    context.fillStyle=glow;context.beginPath();context.arc(x,y,radius*3.2,0,Math.PI*2);context.fill();
  };
  drawBody(celestial.sunDirection,model.appearance.colors.sun,Math.max(4,width*.009));
  drawBody(celestial.moonDirection,model.appearance.colors.moon,Math.max(3,width*.007));
  const planarLength = Math.hypot(cameraFacing[0], cameraFacing[2]) || 1;
  const forward = [cameraFacing[0] / planarLength, cameraFacing[2] / planarLength];
  const right = [-forward[1], forward[0]];
  const focal = width * 0.72;
  const horizon = height * 0.52;
  const projected = shaderViewer.geometry.map(box => {
    const corners = [
      [-box.half_extent[0], -box.half_extent[1]], [box.half_extent[0], -box.half_extent[1]],
      [box.half_extent[0], box.half_extent[1]], [-box.half_extent[0], box.half_extent[1]]
    ].map(offset => {
      const relative = [box.center[0] + offset[0] - cameraPosition[0], box.center[1] + offset[1] - cameraPosition[2]];
      const depth = relative[0] * forward[0] + relative[1] * forward[1];
      const side = relative[0] * right[0] + relative[1] * right[1];
      return {depth, x: width * 0.5 + side / Math.max(0.08, depth) * focal};
    });
    const visible = corners.filter(corner => corner.depth > 0.08);
    if (!visible.length) return null;
    const depth = visible.reduce((sum, corner) => sum + corner.depth, 0) / visible.length;
    const left = Math.max(-width, Math.min(...visible.map(corner => corner.x)));
    const rightEdge = Math.min(width * 2, Math.max(...visible.map(corner => corner.x)));
    const base = horizon + cameraPosition[1] / depth * focal;
    const top = horizon + (cameraPosition[1] - box.height) / depth * focal;
    return {box, depth, left, right: rightEdge, base, top};
  }).filter(Boolean).sort((a, b) => b.depth - a.depth);
  projected.forEach(item => {
    const material = item.box.appearance?.face_color || colors[item.box.palette_role] || colors.room;
    context.fillStyle = material;
    context.strokeStyle = item.box.appearance?.wall_color ||
      colors[item.box.wall_palette_role] || colors.line;
    context.lineWidth = Math.max(1, width / 900);
    context.beginPath();
    context.moveTo(item.left, item.base); context.lineTo(item.left, item.top);
    context.lineTo(item.right, item.top); context.lineTo(item.right, item.base);
    context.closePath(); context.globalAlpha = 0.14; context.fill();
    context.globalAlpha = 1; context.stroke();
    context.fillStyle = colors.active + "55";
    context.fillRect(item.left, item.top, Math.max(0, item.right - item.left), Math.max(1, (item.base - item.top) * 0.13));
  });
}

function drawCanvasCrosshair(context, width, height, hasTarget) {
  const centerX = width * 0.5, centerY = height * 0.5;
  context.save();
  context.strokeStyle = hasTarget ? model.appearance.colors.active : model.appearance.colors.muted;
  context.globalAlpha = 0.92;
  context.lineWidth = Math.max(1, window.devicePixelRatio || 1);
  context.beginPath();
  context.moveTo(centerX - 11, centerY); context.lineTo(centerX - 3, centerY);
  context.moveTo(centerX + 3, centerY); context.lineTo(centerX + 11, centerY);
  context.moveTo(centerX, centerY - 11); context.lineTo(centerX, centerY - 3);
  context.moveTo(centerX, centerY + 3); context.lineTo(centerX, centerY + 11);
  context.stroke();
  context.restore();
}

function drawShaderCrosshair(gl, width, height) {
  const program = shaderViewer.crosshairProgram;
  if (!program) return;
  gl.useProgram(program);
  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.uniform2f(shaderViewer.crosshairLocations.uResolution, width, height);
  gl.uniform3fv(shaderViewer.crosshairLocations.uIdleColor, colorVector(model.appearance.colors.muted));
  gl.uniform3fv(shaderViewer.crosshairLocations.uTargetColor, colorVector(model.appearance.colors.active));
  gl.uniform1f(shaderViewer.crosshairLocations.uHasTarget, shaderViewer.crosshairIdentity ? 1 : 0);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.disable(gl.BLEND);
  gl.enable(gl.DEPTH_TEST);
}

function drawVehicleShaderHud(gl,width,height){
  const program=shaderViewer.vehicleHudProgram,vehicle=vehicleRuntime.active;
  if(!program||!vehicle||!shaderViewer.shaderOnly){vehicleRuntime.shaderHudHitRegions=[];return;}
  vehicleRuntime.shaderHudHitRegions=[];
  const locations=shaderViewer.vehicleHudLocations,ratio=Math.min(2,window.devicePixelRatio||1),
    panelWidth=154*ratio,panelHeight=286*ratio,panelX=width-panelWidth-14*ratio,panelY=62*ratio,
    travel=Math.max(1e-6,Number(vehicle.configuration.suspension.travel));
  const rgba=(color,alpha=1)=>[...colorVector(color),alpha],draw=(x,y,w,h,color,ellipse=false,angle=0)=>{
    if(w<=0||h<=0)return;gl.uniform4f(locations.uRect,x,y,w,h);
    gl.uniform1f(locations.uAngle,angle);
    gl.uniform4fv(locations.uColor,color);gl.uniform1f(locations.uEllipse,ellipse?1:0);
    gl.drawArrays(gl.TRIANGLES,0,6);
  },controlColor=value=>rgba(value>.65?"#ff5f7d":value>.25?"#ffd166":"#54e39b",.88),
    glyphs={A:["010","101","111","101","101"],B:["110","101","110","101","110"],
      C:["011","100","100","100","011"],S:["011","100","010","001","110"],
      T:["111","010","010","010","010"],F:["111","100","110","100","100"],
      H:["101","101","111","101","101"],L:["100","100","100","100","111"],
      R:["110","101","110","101","101"],V:["101","101","101","101","010"]},
    glyphText=(text,x,y,scale,color)=>{let cursor=x;for(const character of text){const rows=glyphs[character]||[];
      rows.forEach((bits,row)=>[...bits].forEach((bit,column)=>{if(bit==="1")draw(cursor+column*scale,y+row*scale,scale,scale,color);}));
      cursor+=4*scale;}},
    dial=(cx,cy,r,value,color)=>{
      const level=Math.max(0,Math.min(1,Number(value)||0)),needle=-2.35+4.7*level;
      draw(cx-r,cy-r,r*2,r*2,rgba("#30443e",.96),true);
      draw(cx-r+1.5*ratio,cy-r+1.5*ratio,r*2-3*ratio,r*2-3*ratio,rgba("#0b1513",.98),true);
      for(let mark=0;mark<9;mark+=1){
        const angle=-2.35+4.7*mark/8,radial=r*.78,
          mx=cx+Math.sin(angle)*radial,my=cy-Math.cos(angle)*radial;
        draw(mx-.65*ratio,my-1.8*ratio,1.3*ratio,3.6*ratio,
          rgba(mark===8?"#ff5f7d":"#a8bbb4",.9),false,angle);
      }
      const nx=cx+Math.sin(needle)*r*.34,ny=cy-Math.cos(needle)*r*.34;
      draw(nx-.8*ratio,ny-r*.34,1.6*ratio,r*.68,color,false,needle);
      draw(cx-2*ratio,cy-2*ratio,4*ratio,4*ratio,color,true);
    };
  gl.useProgram(program);gl.disable(gl.DEPTH_TEST);gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.bindVertexArray(shaderViewer.vehicleHudVao);
  gl.uniform2f(locations.uResolution,width,height);
  draw(panelX,panelY,panelWidth,panelHeight,rgba("#07100f",.72));
  const powertrain=vehicleRuntime.powertrain||{},state=vehicleRuntime.state||{},
    power=vehicle.configuration.powertrain,drivetrain=vehicle.configuration.drivetrain,
    tireRadius=Math.max(.01,Number(vehicle.configuration.tires.radius)),
    redline=Math.max(1,Number(power.redline_rpm)),clutchLimit=Math.max(1,Number(power.clutch_maximum_torque_nm)),
    maximumWheelOmega=Math.max(1,Number(drivetrain.maximum_wheel_speed_rad_s)),
    wheelNames=["front_left","front_right","rear_left","rear_right"],
    wheelOmegas=wheelNames.map(name=>Number(state.wheelOmegas?.[name]||0)),
    roadSpeed=Math.hypot(Number(state.velocity?.[0]||0),Number(state.velocity?.[2]||0)),
    maximumRoadSpeed=maximumWheelOmega*tireRadius;
  // Four classic, label-free drivetrain instruments: crank RPM, clutch load,
  // transfer/final-drive torque, and GPS speed.  All values are passive
  // resident-graph snapshots; drawing them cannot schedule or stall physics.
  dial(panelX+21*ratio,panelY+20*ratio,15*ratio,Math.abs(Number(powertrain.engineRPM||0))/redline,rgba("#ffd166",.98));
  dial(panelX+58*ratio,panelY+20*ratio,15*ratio,Math.abs(Number(powertrain.clutchTorque||0))/clutchLimit,rgba("#ff8fb3",.98));
  dial(panelX+95*ratio,panelY+20*ratio,15*ratio,Math.abs(Number(powertrain.drivelineTorque||0))/(clutchLimit*6),rgba("#e5e7e6",.98));
  dial(panelX+132*ratio,panelY+20*ratio,15*ratio,roadSpeed/maximumRoadSpeed,rgba("#54e39b",.98));
  [0,1,2,3].forEach(index=>{
    const column=index%2,row=Math.floor(index/2),x=panelX+(6+column*73)*ratio,y=panelY+(43+row*47)*ratio,
      compression=Math.max(0,Math.min(1,Number(vehicleRuntime.compressions[index]||0)/travel)),
      damper=Math.max(.5,Math.min(1.5,vehicleRuntime.damperScales[index]??1)),
      tc=1-Math.max(0,Math.min(1,vehicleRuntime.tractionScales[index]??1)),
      abs=1-Math.max(0,Math.min(1,vehicleRuntime.brakeScales[index]??1)),
      wheelLevel=Math.abs(wheelOmegas[index])/maximumWheelOmega,
      wheelNeedle=-2.35+4.7*Math.max(0,Math.min(1,wheelLevel));
    draw(x,y,68*ratio,42*ratio,rgba("#152620",.72));
    draw(x+5*ratio,y+5*ratio,6*ratio,32*ratio,rgba("#31483f",.8));
    draw(x+6*ratio,y+(36-30*compression)*ratio,4*ratio,30*compression*ratio,
      rgba(damper>1.05?"#66d9ff":damper<.96?"#b28cff":compression>.82?"#ff5f7d":compression>.58?"#ffd166":"#fff2a8",.95));
    // Reserved for the compiled per-probe force integral. Penetration probes
    // are deliberately not drawn as force/heat; that would be false telemetry.
    const tireX=x+16*ratio,tireY=y+4*ratio,tireW=48*ratio,tireH=14*ratio;
    draw(tireX,tireY,tireW,tireH,rgba("#0b1513",.98));
    const wheelNeedleCenterX=x+(12+Math.sin(wheelNeedle)*1.6)*ratio,
      wheelNeedleCenterY=y+(10-Math.cos(wheelNeedle)*1.6)*ratio;
    draw(wheelNeedleCenterX-.6*ratio,wheelNeedleCenterY-2.6*ratio,1.2*ratio,5.2*ratio,rgba("#66d9ff",.96),false,wheelNeedle);
    draw(x+18*ratio,y+21*ratio,44*ratio,5*ratio,rgba("#263b34",.9));
    draw(x+18*ratio,y+21*ratio,44*tc*ratio,5*ratio,controlColor(tc));
    for(let segment=0;segment<4;segment+=1){
      const segmentValue=Math.max(0,Math.min(1,abs*4-segment));
      draw(x+(18+segment*11)*ratio,y+31*ratio,9*ratio,5*ratio,rgba("#263b34",.9));
      draw(x+(18+segment*11)*ratio,y+31*ratio,9*segmentValue*ratio,5*ratio,controlColor(abs));
    }
  });
  const transmission=vehicleRuntime.transmission,indicatorY=panelY+133*ratio,
    indicator=(label,x,mode)=>{const color=mode==="locked"?"#1b6b4b":mode==="limited-slip"?"#69551f":"#252d2b",
      glyph=mode==="locked"?"#8effc7":mode==="limited-slip"?"#ffd166":"#71817b";
      draw(x,indicatorY,15*ratio,7*ratio,rgba(color,.96));glyphText(label,x+5*ratio,indicatorY+1*ratio,ratio,rgba(glyph,1));};
  indicator(String(transmission.transferRange||(transmission.lowRange?"l2":"high")).toUpperCase(),panelX+7*ratio,"locked");
  indicator("C",panelX+25*ratio,transmission.centerDiffMode||"open");
  indicator("F",panelX+43*ratio,transmission.frontDiffMode||"open");
  indicator("R",panelX+61*ratio,transmission.rearDiffMode||"open");
  draw(panelX+82*ratio,indicatorY,64*ratio,7*ratio,rgba("#263b34",.9));
  draw(panelX+82*ratio,indicatorY,64*Math.max(.05,Math.min(.95,Number(transmission.frontDriveShare||.5)))*ratio,
    7*ratio,rgba("#66d9ff",.9));
  const buttonY=panelY+141*ratio,buttonH=18*ratio,tcX=panelX+7*ratio,absX=panelX+81*ratio,buttonW=66*ratio,
    tcEnabled=vehicleRuntime.transmission.tractionControlEnabled!==false,
    absEnabled=vehicleRuntime.transmission.absEnabled!==false;
  [[tcX,"TC",tcEnabled,"traction-control"],[absX,"ABS",absEnabled,"abs"]].forEach(([x,label,enabled,action])=>{
    draw(x,buttonY,buttonW,buttonH,rgba(enabled?"#1b6b4b":"#3b2929",.96));
    draw(x+1*ratio,buttonY+1*ratio,buttonW-2*ratio,buttonH-2*ratio,rgba(enabled?"#123f30":"#1b1717",.96));
    glyphText(label,x+(label.length===2?25:23)*ratio,buttonY+4*ratio,2*ratio,rgba(enabled?"#8effc7":"#ff8c9d",1));
    vehicleRuntime.shaderHudHitRegions.push({action,x,y:buttonY,w:buttonW,h:buttonH});
  });
  const authorityY=panelY+163*ratio,authorityRadius=9*ratio;
  [[panelX+26*ratio,"T",Number(transmission.tractionControlAuthority??1),"traction-authority"],
   [panelX+100*ratio,"A",Number(transmission.absAuthority??1),"abs-authority"]].forEach(([cx,label,value,action])=>{
    dial(cx,authorityY+authorityRadius,authorityRadius,value,rgba("#66d9ff",.98));
    glyphText(label,cx+13*ratio,authorityY+5*ratio,ratio,rgba("#a8bbb4",1));
    vehicleRuntime.shaderHudHitRegions.push({action,x:cx-authorityRadius,y:authorityY,
      w:45*ratio,h:authorityRadius*2,valueX:cx-authorityRadius,valueWidth:authorityRadius*2});
  });
  const levelingY=panelY+187*ratio,levelingEnabled=Boolean(vehicleRuntime.chassisLeveling?.enabled);
  draw(panelX+7*ratio,levelingY,140*ratio,17*ratio,rgba(levelingEnabled?"#173f37":"#252d2b",.96));
  glyphText("LVL",panelX+59*ratio,levelingY+3*ratio,2*ratio,rgba(levelingEnabled?"#8effc7":"#71817b",1));
  vehicleRuntime.shaderHudHitRegions.push({action:"chassis-leveling",x:panelX+7*ratio,y:levelingY,w:140*ratio,h:17*ratio});
  // Actual mechanical graph damage, projected top-down. Every edge is drawn;
  // damageable members shade green -> amber -> red, while non-damageable
  // reference members remain muted. This is presentation of worker-owned
  // plastic strain/fracture state, never a DOM approximation.
  const graph=vehicle.physics?.mechanical_graph,nodes=new Map((graph?.nodes||[]).map(node=>[node.identity,
      vehicleRuntime.mechanicalNodePositions.get(node.identity)||node.reference_position])),
    edges=graph?.edges||[],damage=vehicleRuntime.damage||{},damageX=panelX+7*ratio,damageY=panelY+222*ratio,
    damageW=140*ratio,damageH=57*ratio,positions=[...nodes.values()],
    minX=Math.min(-1,...positions.map(value=>Number(value?.[0]||0))),maxX=Math.max(1,...positions.map(value=>Number(value?.[0]||0))),
    minZ=Math.min(-1,...positions.map(value=>Number(value?.[2]||0))),maxZ=Math.max(1,...positions.map(value=>Number(value?.[2]||0)));
  draw(damageX,damageY,damageW,damageH,rgba("#0b1513",.96));
  edges.forEach(edge=>{const a=nodes.get(edge.a),b=nodes.get(edge.b);if(!a||!b)return;
    const ax=damageX+(Number(a[2])-minZ)/Math.max(1e-6,maxZ-minZ)*damageW,
      ay=damageY+(maxX-Number(a[0]))/Math.max(1e-6,maxX-minX)*damageH,
      bx=damageX+(Number(b[2])-minZ)/Math.max(1e-6,maxZ-minZ)*damageW,
      by=damageY+(maxX-Number(b[0]))/Math.max(1e-6,maxX-minX)*damageH,
      dx=bx-ax,dy=by-ay,length=Math.max(.8*ratio,Math.hypot(dx,dy)),state=damage.members?.[edge.identity],
      halfshaftName=["front_left","front_right","rear_left","rear_right"].find(name=>edge.identity.includes(`${name}_halfshaft`)),
      springName=["front_left","front_right","rear_left","rear_right"].find(name=>edge.identity===`suspension.${name}.coilover`),
      fracture=Math.max(1e-8,Number(edge.damage?.fracture_strain||.08)),plastic=state?Number(state.plasticStrain||0)/fracture:0,
      health=halfshaftName?Number(damage.halfshaftHealth?.[halfshaftName]??1):springName?
        Number(damage.springHealth?.[springName]??1):state?.failed?0:state?Math.max(0,1-plastic):null,
      color=health===null?"#3b5049":health<=0?"#ff3f62":health<.45?"#ff8a4c":health<.82?"#ffd166":"#54e39b",
      thickness=Math.max(1.1,Math.min(3.2,Number(edge.radius||.01)*90))*ratio,
      angle=Math.atan2(dx,-dy);
    draw((ax+bx)/2-thickness/2,(ay+by)/2-length/2,thickness,length,rgba(color,.94),false,angle);
  });
  gl.disable(gl.BLEND);gl.enable(gl.DEPTH_TEST);
}

function handleVehicleShaderHudPointer(event){
  if(!shaderViewer.shaderOnly||event.button!==0||!vehicleRuntime.active)return false;
  const canvas=shaderViewer.canvas,bounds=canvas?.getBoundingClientRect();if(!bounds)return false;
  const x=(event.clientX-bounds.left)*canvas.width/Math.max(1,bounds.width),
    y=(event.clientY-bounds.top)*canvas.height/Math.max(1,bounds.height),
    hit=vehicleRuntime.shaderHudHitRegions.find(region=>x>=region.x&&x<=region.x+region.w&&y>=region.y&&y<=region.y+region.h);
  if(!hit)return false;
  if(hit.action==="traction-control")controlVehicleTransmission({tractionControlEnabled:
    vehicleRuntime.transmission.tractionControlEnabled===false});
  else if(hit.action==="abs")controlVehicleTransmission({absEnabled:vehicleRuntime.transmission.absEnabled===false});
  else if(hit.action==="traction-authority")controlVehicleTransmission({tractionControlAuthority:
    Math.max(0,Math.min(1,(x-hit.valueX)/Math.max(1,hit.valueWidth)))});
  else if(hit.action==="abs-authority")controlVehicleTransmission({absAuthority:
    Math.max(0,Math.min(1,(x-hit.valueX)/Math.max(1,hit.valueWidth)))});
  else if(hit.action==="chassis-leveling")controlVehicleChassisLeveling({enabled:!vehicleRuntime.chassisLeveling?.enabled});
  return true;
}

function drawVehicleCanvasHud(context,width,height){
  const vehicle=vehicleRuntime.active,expanded=vehicleRuntime.contactMonitor?.classList.contains("expanded");
  if(!vehicle||!shaderViewer.shaderOnly||expanded)return;
  const ratio=Math.min(2,window.devicePixelRatio||1),panelWidth=154*ratio,panelHeight=104*ratio,
    panelX=width-panelWidth-14*ratio,panelY=62*ratio,
    travel=Math.max(1e-6,Number(vehicle.configuration.suspension.travel)),
    patchColors=["#52635d","#54e39b","#ffd166","#ff5f7d"],controlColor=value=>
      value>.65?"#ff5f7d":value>.25?"#ffd166":"#54e39b";
  context.save();context.globalAlpha=.8;context.fillStyle="#07100f";
  context.fillRect(panelX,panelY,panelWidth,panelHeight);
  [0,1,2,3].forEach(index=>{
    const x=panelX+(6+(index%2)*73)*ratio,y=panelY+(7+Math.floor(index/2)*47)*ratio,
      compression=Math.max(0,Math.min(1,Number(vehicleRuntime.compressions[index]||0)/travel)),
      area=Math.max(0,Number(vehicleRuntime.contactAreas[index]||0)),tires=vehicle.configuration.tires,
      areaRange=Math.max(1e-8,Number(tires.maximum_contact_area)-Number(tires.minimum_contact_area)),
      areaLevel=Math.max(0,Math.min(1,(area-Number(tires.minimum_contact_area))/areaRange)),
      mode=Math.max(0,Math.min(3,Math.round(vehicleRuntime.contactModes[index]||0))),
      damper=Math.max(.5,Math.min(1.5,vehicleRuntime.damperScales[index]??1)),
      tc=1-Math.max(0,Math.min(1,vehicleRuntime.tractionScales[index]??1)),
      abs=1-Math.max(0,Math.min(1,vehicleRuntime.brakeScales[index]??1));
    context.fillStyle="#152620";context.fillRect(x,y,68*ratio,42*ratio);
    context.fillStyle="#31483f";context.fillRect(x+5*ratio,y+5*ratio,6*ratio,32*ratio);
    context.fillStyle=damper>1.05?"#66d9ff":damper<.96?"#b28cff":compression>.82?"#ff5f7d":compression>.58?"#ffd166":"#fff2a8";
    context.fillRect(x+6*ratio,y+(36-30*compression)*ratio,4*ratio,30*compression*ratio);
    const patchWidth=(22+22*areaLevel)*ratio;context.fillStyle=patchColors[mode];context.beginPath();
    context.ellipse(x+43*ratio,y+10*ratio,patchWidth*.5,5*ratio,0,0,Math.PI*2);context.fill();
    context.fillStyle="#263b34";context.fillRect(x+18*ratio,y+21*ratio,44*ratio,5*ratio);
    context.fillStyle=controlColor(tc);context.fillRect(x+18*ratio,y+21*ratio,44*tc*ratio,5*ratio);
    for(let segment=0;segment<4;segment+=1){const segmentValue=Math.max(0,Math.min(1,abs*4-segment));
      const sx=x+(18+segment*11)*ratio;context.fillStyle="#263b34";context.fillRect(sx,y+31*ratio,9*ratio,5*ratio);
      context.fillStyle=controlColor(abs);context.fillRect(sx,y+31*ratio,9*segmentValue*ratio,5*ratio);}
  });
  context.restore();
}

function cameraGroundHeight(worldX,worldZ){
  let height=0;
  const terrainSources=shaderViewer.colliders.length?shaderViewer.colliders:
    (model.document_geometry?.boxes||[]).filter(box=>box.surface);
  terrainSources.forEach(source=>{
    const surface=source.surface,domain=surface?.domain;if(!surface||!domain)return;
    if(worldX<domain.minimum_x||worldX>domain.maximum_x||worldZ<domain.minimum_z||worldZ>domain.maximum_z)return;
    height=Math.max(height,sampleDeclaredSurface(surface,worldX,worldZ).height);
  });
  return height;
}

function updateVehicleChaseCamera(vehicle,state){
  const cues=vehicle.configuration.presentation,now=performance.now(),speed=Math.hypot(...state.velocity);
  const forward=[Math.cos(state.yaw),0,Math.sin(state.yaw)];
  const distance=Number(cues.chase_camera_distance||2.6)+speed*Number(cues.chase_camera_speed_pullback||.045);
  const desired=[state.position[0]-forward[0]*distance,
    state.position[1]+Number(cues.chase_camera_height||1.15),state.position[2]-forward[2]*distance];
  desired[1]=Math.max(desired[1],cameraGroundHeight(desired[0],desired[2])+
    Number(cues.chase_camera_ground_clearance||.3));
  const lookTarget=[state.position[0]+forward[0]*Number(cues.chase_camera_look_ahead||1.4)*.28,
    state.position[1]+Number(vehicle.configuration.chassis.height)*.2,
    state.position[2]+forward[2]*Number(cues.chase_camera_look_ahead||1.4)*.28],
    baseDelta=lookTarget.map((value,index)=>value-desired[index]),
    baseYaw=Math.atan2(baseDelta[2],baseDelta[0]),
    basePitch=Math.atan2(baseDelta[1],Math.hypot(baseDelta[0],baseDelta[2])),
    yawOffset=Math.atan2(Math.sin(viewportControls.yaw-state.yaw),Math.cos(viewportControls.yaw-state.yaw)),
    aimPitch=Math.max(-1.35,Math.min(1.1,basePitch+viewportControls.pitch)),horizontal=Math.cos(aimPitch),
    desiredFacing=normalized3([horizontal*Math.cos(baseYaw+yawOffset),Math.sin(aimPitch),
      horizontal*Math.sin(baseYaw+yawOffset)]);
  if(!vehicleRuntime.camera){vehicleRuntime.camera={position:desired,facing:desiredFacing,time:now};}
  const camera=vehicleRuntime.camera,dt=Math.max(0,Math.min(.05,(now-camera.time)/1000));camera.time=now;
  const positionAlpha=1-Math.exp(-Number(cues.chase_camera_position_response||7)*dt),
    facingAlpha=1-Math.exp(-Number(cues.chase_camera_facing_response||11)*dt);
  camera.position=camera.position.map((value,index)=>value+(desired[index]-value)*positionAlpha);
  const smoothedFacing=camera.facing.map((value,index)=>value+(desiredFacing[index]-value)*facingAlpha);
  camera.facing=normalized3(smoothedFacing);return camera;
}

function updateShaderViewer() {
  const {gl, canvas, program, context2d} = shaderViewer;
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const player = entityState.get(viewportControls.policy?.actor);
  const bounds = shaderViewer.mapElement?.getBoundingClientRect();
  const extent = model.document_geometry.extent;
  const origin = model.document_geometry.origin || [0, 0];
  const eyeHeight = model.viewer.camera.eye_height;
  const courtyard = shaderViewer.geometry.find(box => box.kind === "courtyard");
  let cameraPosition = [
    courtyard?.center[0] ?? origin[0] + extent[0] * 0.5,
    eyeHeight,
    courtyard?.center[1] ?? origin[1] + extent[1] * 0.5
  ];
  let cameraFacing = [0, -0.22, -1];
  if (viewportControls.highlighted && viewportControls.position) {
    cameraPosition = [...viewportControls.position];
    const horizontal = Math.cos(viewportControls.pitch);
    cameraFacing = [
      horizontal * Math.cos(viewportControls.yaw),
      Math.sin(viewportControls.pitch),
      horizontal * Math.sin(viewportControls.yaw)
    ];
    const gamepad = viewportControls.gamepadIdentity
      ? ` · gamepad ${viewportControls.gamepadIdentity}` : " · awaiting gamepad";
    shaderViewer.readout.textContent = `${shaderViewer.backend} · controls captured · WASD + mouse${gamepad}`;
  } else if (player?.worldPosition) {
    cameraPosition = [...player.worldPosition];
    cameraFacing = [player.facing[0], -0.18, player.facing[1]];
    shaderViewer.inhabitedCameraPosition = [...cameraPosition];
    shaderViewer.inhabitedCameraFacing = [...cameraFacing];
    shaderViewer.readout.textContent = `${shaderViewer.backend} · world player · camera ${cameraPosition[0].toFixed(2)}, ${cameraPosition[2].toFixed(2)} · facing ${player.facing[0].toFixed(2)}, ${player.facing[1].toFixed(2)}`;
  }
  if(vehicleRuntime.active&&vehicleRuntime.state){
    try{
      const vehicleCamera=updateVehicleChaseCamera(vehicleRuntime.active,vehicleRuntime.state);
      cameraPosition=vehicleCamera.position;cameraFacing=vehicleCamera.facing;
    }catch(error){reportRuntimeFault("vehicle-camera",error);}
    const speed=Math.hypot(...vehicleRuntime.state.velocity);
    shaderViewer.readout.textContent=`${shaderViewer.backend} · spring chase · ${speed.toFixed(1)} m/s · free look/tools · V dismount`;
  }
  shaderViewer.cameraPosition = [...cameraPosition];
  shaderViewer.cameraFacing = [...cameraFacing];
  pickCrosshairIdentity();
  const celestial=celestialState();shaderViewer.celestial=celestial;publishCelestialStatus(celestial);
  if (gl && program) {
    renderCameraDepthPass(gl,Math.max(1,Math.round(width*.5)),Math.max(1,Math.round(height*.5)),
      cameraPosition,cameraFacing);
    renderShadowPass(gl,celestial);
    gl.viewport(0, 0, width, height);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    drawSkyHalfDome(gl,width,height,cameraFacing,celestial);gl.useProgram(program);
    updateCelestialLighting(gl,program,cameraFacing,celestial);
    gl.uniform2f(shaderViewer.locations.uResolution, width, height);
    gl.uniform3fv(shaderViewer.locations.uCameraPosition, cameraPosition);
    gl.uniform3fv(shaderViewer.locations.uCameraFacing, cameraFacing);
    const headlightState=vehicleRuntime.state||vehicleRuntime.parkedState,
      headlightVehicle=vehicleRuntime.active||model.vehicle_slot?.vehicles?.[0],headlightLocations=shaderViewer.locations;
    if(headlightState&&headlightVehicle){
      const chassis=headlightVehicle.configuration.chassis,lampX=Number(chassis.half_length)+.045,
        tailX=-Number(chassis.half_length)-.035,
        left=rotateVehiclePresentationVector([lampX,.095,-.14],headlightState,0),
        right=rotateVehiclePresentationVector([lampX,.095,.14],headlightState,0),
        tailLeft=rotateVehiclePresentationVector([tailX,.095,-.14],headlightState,0),
        tailRight=rotateVehiclePresentationVector([tailX,.095,.14],headlightState,0),
        forward=rotateVehiclePresentationVector([1,-.025,0],headlightState,0),
        rearward=rotateVehiclePresentationVector([-1,-.01,0],headlightState,0);
      if(headlightLocations.uHeadlightLeft!==null)gl.uniform3fv(headlightLocations.uHeadlightLeft,
        [headlightState.position[0]+left[0],headlightState.position[1]+left[1],headlightState.position[2]+left[2]]);
      if(headlightLocations.uHeadlightRight!==null)gl.uniform3fv(headlightLocations.uHeadlightRight,
        [headlightState.position[0]+right[0],headlightState.position[1]+right[1],headlightState.position[2]+right[2]]);
      if(headlightLocations.uHeadlightForward!==null)gl.uniform3fv(headlightLocations.uHeadlightForward,forward);
      if(headlightLocations.uTailLightLeft!==null)gl.uniform3fv(headlightLocations.uTailLightLeft,
        [headlightState.position[0]+tailLeft[0],headlightState.position[1]+tailLeft[1],headlightState.position[2]+tailLeft[2]]);
      if(headlightLocations.uTailLightRight!==null)gl.uniform3fv(headlightLocations.uTailLightRight,
        [headlightState.position[0]+tailRight[0],headlightState.position[1]+tailRight[1],headlightState.position[2]+tailRight[2]]);
      if(headlightLocations.uVehicleRearward!==null)gl.uniform3fv(headlightLocations.uVehicleRearward,rearward);
      if(headlightLocations.uHeadlightActive!==null)gl.uniform1f(headlightLocations.uHeadlightActive,
        vehicleRuntime.electrical.headlightsOn&&vehicleRuntime.electrical.lightingCircuitOnline&&
          vehicleRuntime.electrical.stateOfCharge>0?1:0);
      const lightingPowered=vehicleRuntime.electrical.lightingCircuitOnline&&
        vehicleRuntime.electrical.stateOfCharge>0;
      if(headlightLocations.uTailLightActive!==null)gl.uniform1f(headlightLocations.uTailLightActive,
        lightingPowered&&vehicleRuntime.electrical.tailLightsOn?1:0);
      if(headlightLocations.uBrakeLightActive!==null)gl.uniform1f(headlightLocations.uBrakeLightActive,
        lightingPowered&&vehicleRuntime.electrical.brakeLightsOn?1:0);
    }else{
      if(headlightLocations.uHeadlightActive!==null)gl.uniform1f(headlightLocations.uHeadlightActive,0);
      if(headlightLocations.uTailLightActive!==null)gl.uniform1f(headlightLocations.uTailLightActive,0);
      if(headlightLocations.uBrakeLightActive!==null)gl.uniform1f(headlightLocations.uBrakeLightActive,0);
    }
    const renderPass=shaderViewer.locations.uRenderPass;
    if(renderPass!==null){
      gl.disable(gl.BLEND);gl.depthMask(true);gl.uniform1i(renderPass,1);drawSceneMeshes(gl);
      gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);
      gl.uniform1i(renderPass,2);drawSceneMeshes(gl);gl.depthMask(true);gl.disable(gl.BLEND);gl.uniform1i(renderPass,0);
    }else drawSceneMeshes(gl);
    drawVehicleWheels(gl,width,height,cameraPosition,cameraFacing,celestial);
    drawVehicleShaderHud(gl,width,height);
    drawShaderCrosshair(gl, width, height);
  } else if (context2d) {
    if (shaderViewer.softwareWasm) {
      drawSoftwareViewer(context2d, width, height, cameraPosition, cameraFacing);
      drawCompiledMeshViewer(context2d, width, height, cameraPosition, cameraFacing, false);
      if (shaderViewer.active) shaderViewer.readout.textContent +=
        ` · ${shaderViewer.softwareOnscreenCount}/${shaderViewer.softwareTriangleCount} triangles onscreen` +
        " · portable geometry floor";
    } else {
      drawSoftwareViewer(context2d, width, height, cameraPosition, cameraFacing);
    }
    drawVehicleCanvasHud(context2d,width,height);
    drawCanvasCrosshair(context2d, width, height, Boolean(shaderViewer.crosshairIdentity));
  }
  shaderViewer.element.classList.toggle("inactive", !shaderViewer.active);
}

const actionEdges = {
  identity: model.action_mezzanine.action_edges.identity,
  time: 0,
  recentWindow: model.action_mezzanine.action_edges.recent_window * 1000,
  rows: new Map(),
  element: null,
  register(source, type, destination) {
    const identity = `${source}::${type}->${destination}`;
    if (!this.rows.has(identity)) {
      const row = {identity, source, type, destination, count: 0, lastIssuedAt: null, element: null};
      this.rows.set(identity, row);
      if (this.element) this.appendRow(row);
    }
    return identity;
  },
  bind(element) {
    this.element = element;
    this.rows.forEach(row => this.appendRow(row));
  },
  appendRow(row) {
    if (row.element) return;
    const element = div("action-edge-row");
    element.dataset.actionEdge = row.identity;
    element.append(div("edge-cell", row.source), div("edge-cell", row.type),
      div("edge-cell", row.destination), div("edge-count", "0"));
    row.element = element;
    this.element.append(element);
  },
  update(actions) {
    actions.forEach(action => {
      const row = this.rows.get(action.edge);
      if (!row) throw new Error(`Issued action names an unknown edge: ${action.edge}`);
      row.count += 1;
      row.lastIssuedAt = action.issued_at;
    });
    this.rows.forEach(row => {
      if (!row.element) return;
      const recent = row.lastIssuedAt !== null && this.time - row.lastIssuedAt <= this.recentWindow;
      row.element.classList.toggle("recent", recent);
      row.element.querySelector(".edge-count").textContent = String(row.count);
    });
  }
};
abstractUISystemTimer.connect(actionEdges);

function div(className, text) {
  const node = document.createElement("div");
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function gridStyle(node) {
  const p = node.position || {column: 0, row: 0, width: 1, height: 1};
  return `grid-column:${p.column + 1}/span ${p.width};grid-row:${p.row + 1}/span ${p.height}`;
}

function geometryBox(identity) {
  return shaderViewer.geometry.find(box => box.identity === identity) ||
    model.document_geometry.boxes.find(box => box.identity === identity) || null;
}

function boxYawDegrees(box) {
  return Number(box?.placement?.rotation?.[1] || 0);
}

function property(name, value) {
  const row = div("property");
  row.append(div("property-name", name));
  const rendered = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  row.append(div("property-value", rendered));
  return row;
}

function inspectNode(node, element) {
  if (element === shaderViewer.element && !viewportControls.highlighted) {
    setViewportControlHighlight(true);
  }
  if (element && element !== shaderViewer.element && viewportControls.highlighted &&
      controlFocus.mode === "game") {
    setViewportControlHighlight(false);
  }
  document.querySelectorAll(".selected").forEach(item => item.classList.remove("selected"));
  if (element) element.classList.add("selected");
  inspector.replaceChildren(div("eyebrow", node.kind), div("inspector-title", node.name));
  ["identity", "parent", "metaphor", "member_kind", "type_name", "parameters", "position",
   "dependencies", "archetype", "principal", "color", "capabilities", "controller",
   "geometry", "form", "material_bindings", "physics", "persistence", "extensions",
   "texture", "traits"]
    .filter(key => node[key] !== undefined && node[key] !== null)
    .forEach(key => inspector.append(property(key, node[key])));
  (node.implied_code || []).forEach(receipt => {
    const group = property(`${receipt.dialect} · ${receipt.operation}`, receipt.explanation);
    group.append(div("code-receipt", receipt.source));
    inspector.append(group);
  });
  status.textContent = `selected ${node.identity}`;
}

function makeSelectable(node, className) {
  index.set(node.identity, node);
  const card = div(className);
  card.dataset.nodeId = node.identity;
  const interaction = node.interaction || {type: "inspect", destination: node.identity};
  const edge = actionEdges.register(node.identity, interaction.type, interaction.destination);
  card.dataset.interaction = interaction.type;
  card.dataset.destination = interaction.destination;
  card.dataset.actionEdge = edge;
  card.tabIndex = 0;
  card.setAttribute("role", "button");
  card.setAttribute("aria-label", `Inspect ${node.kind} ${node.name}`);
  return card;
}

function dispatchInteraction(source) {
  const interaction = source.dataset.interaction;
  const destination = source.dataset.destination;
  const target = index.get(destination);
  if (!target) throw new Error(`Unknown AbstractUI destination: ${destination}`);
  const actor = model.entity_mezzanine?.entities.find(
    entity => entity.controller.kind === "native-input"
  )?.identity || model.identity;
  abstractUISystemTimer.issue({
    actor, type: interaction, destination, edge: source.dataset.actionEdge,
    issued_at: performance.now()
  });
  if (interaction === "inspect") inspectNode(target, source);
  else throw new Error(`Unknown AbstractUI interaction: ${interaction}`);
}

function omnipotentEventHost(event) {
  if (event.type === "click" && event.target.closest(".regions")) {
    const identity=event.target.closest("[data-node-id]")?.dataset.nodeId;
    if (placementToolActive() && geometryBox(identity)) selectPlacementFocus(identity);
    else void autoLocateFromMapClick(event);
  }
  const defaultsControl = event.target.closest("[data-return-defaults]");
  if (defaultsControl) {
    if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return;
    if (event.type === "keydown") event.preventDefault();
    returnLivingMapToDefaults(); return;
  }
  const placementRecipe = event.target.closest("[data-placement-recipe]");
  if (placementRecipe) {
    if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return;
    if (event.type === "keydown") event.preventDefault();
    selectPlacementRecipe(placementRecipe.dataset.placementRecipe); return;
  }
  const placementAction = event.target.closest("[data-placement-action]");
  if (placementAction) {
    if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return;
    if (event.type === "keydown") event.preventDefault();
    if (placementAction.dataset.placementAction === "apply") commitPlacementFocus();
    else cancelPlacement();
    return;
  }
  const dialogueResponse = event.target.closest("[data-dialogue-response]");
  if (dialogueResponse) {
    if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return;
    if (event.type === "keydown") event.preventDefault();
    closeToolDialogue(); return;
  }
  const preset = event.target.closest("[data-aesthetic-preset]");
  if (preset) {
    if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return;
    if (event.type === "keydown") event.preventDefault();
    applyAestheticPreset(preset.dataset.aestheticPreset); return;
  }
  const hotbarSlot = event.target.closest("[data-hotbar-slot]");
  if (hotbarSlot) {
    if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return;
    if (event.type === "keydown") event.preventDefault();
    selectHotbarSlot(Number(hotbarSlot.dataset.hotbarSlot));
    return;
  }
  const source = event.target.closest("[data-interaction][data-destination]");
  if (!source) return;
  if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return;
  if (event.type === "keydown") event.preventDefault();
  dispatchInteraction(source);
}

document.addEventListener("click", omnipotentEventHost);
document.addEventListener("keydown", omnipotentEventHost);
document.addEventListener("input", event => {
  const placementAxis = event.target.closest("[data-placement-axis]");
  if (placementAxis) {
    setPlacementAxis(placementAxis.dataset.placementAxis, placementAxis.value); return;
  }
  const placementSnap = event.target.closest("[data-placement-snap]");
  if (placementSnap) {
    placementState.snapMode = placementSnap.value;
    if (placementState.payload?.box) {
      placementState.payload.box.center = placementPreviewCenter(placementState.payload.box);
      rebuildPortableSceneMesh();
    }
    return;
  }
  const physicsInput = event.target.closest("[data-physics-parameter]");
  if (physicsInput) {
    setPhysicsParameter(physicsInput.dataset.physicsParameter, physicsInput.value);
    return;
  }
  const input = event.target.closest("[data-aesthetic-property][data-aesthetic-target]");
  if (input) applyAestheticValue(input.dataset.aestheticTarget,
    input.dataset.aestheticProperty, input.value);
});

const entityState = new Map();
const artifactState = new Map((model.filesystem?.artifacts || []).map(artifact => {
  const box = model.document_geometry.boxes.find(candidate => candidate.identity === artifact.identity);
  return [artifact.identity, {
    artifact, box, position: box ? [...box.center] : [0, 0], velocity: [0, 0],
    attachment: {...artifact.attachment}
  }];
}));
const projectileState = new Map();
const projectilePickupState = new Map();
let projectileSequence = 0;
let latestControlInput = null;
let entityCycleTime = performance.now();

function artifactOwnerMotion(state) {
  const actor = entityState.get(state.attachment.owner);
  if (actor?.worldPosition) return {position: [actor.worldPosition[0], actor.worldPosition[2]],
    velocity: [actor.velocity[0], actor.velocity[2]]};
  const box = model.document_geometry.boxes.find(candidate =>
    candidate.identity === state.attachment.owner);
  if (box) return {position: box.center, velocity: [0, 0]};
  const envelope = model.document_geometry.boxes.find(candidate =>
    candidate.kind === "world-envelope");
  return {position: envelope?.center || [0, 0], velocity: [0, 0]};
}

function advanceArtifactAttachments(dt) {
  artifactState.forEach((state, identity) => {
    if (state.attachment.state === "welded") return;
    const owner = artifactOwnerMotion(state);
    const distance = Math.hypot(state.position[0] - owner.position[0],
      state.position[1] - owner.position[1]);
    const relativeSpeed = Math.hypot(state.velocity[0] - owner.velocity[0],
      state.velocity[1] - owner.velocity[1]);
    const closeAndSlow = distance <= state.attachment.connection_radius &&
      relativeSpeed <= state.attachment.maximum_connection_speed;
    state.attachment.settle_time = closeAndSlow
      ? state.attachment.settle_time + dt : 0;
    state.attachment.state = !closeAndSlow ? "loose" :
      (state.attachment.settle_time >= state.attachment.required_settle_time
        ? "welded" : "settling");
    if (state.attachment.state === "welded") {
      state.artifact.physics = {...state.artifact.physics, body: "compound-child",
        collider: "owner-compound", welded: true};
      if (state.box) state.box.physics = {...state.artifact.physics};
      const worldObject = model.world.objects.find(candidate => candidate.identity === identity);
      if (worldObject) worldObject.physics = {...worldObject.physics, ...state.artifact.physics};
    }
    const indicator = document.querySelector(`[data-attachment-state="${identity}"]`);
    if (indicator) indicator.textContent = state.attachment.state;
  });
}

function projectileAmmoItem() {
  return hotbarState.inventory?.items.find(item =>
    item.properties?.operation === "projectile-ammunition");
}

function registerProjectilePhysicsMembership(state) {
  if(!stateLoopRuntime.ready||!state||state.sleeping)return false;
  const radius=Number(model.projectiles.archetype.geometry.radius);
  const snapshot=reservePhysicsSnapshotSlot(state.identity);
  snapshot.position[0]=state.position[0];snapshot.position[1]=state.position[1];
  snapshot.position[2]=state.position[2];snapshot.velocity[0]=state.velocity[0];
  snapshot.velocity[1]=state.velocity[1];snapshot.velocity[2]=state.velocity[2];
  stateLoopRuntime.worker.postMessage({type:"upsert",body:{identity:state.identity,
    slot:snapshot.slot,generation:snapshot.generation,position:[...state.position],
    velocity:[...state.velocity],radius,overrides:{
      linear_drag:Number(model.projectiles.archetype.physics.linear_drag),
      minimum_y:0,portal_active:0}}});
  return true;
}

function sleepProjectilePhysics(state,reason="slow-enough") {
  if(!state||state.sleeping)return false;
  state.sleeping=true;state.velocity=[0,0,0];state.settleTime=0;state.age=0;
  state.record.status="sleeping";state.record.physics_membership="dropped";
  state.record.sleep_reason=reason;state.record.slept_at=performance.now();
  state.record.pose={position:[...state.position],velocity:[0,0,0]};
  state.entity.traits.status="sleeping";state.entity.traits.physics_membership="dropped";
  state.entity.pose={...state.entity.pose,position:[...state.position],velocity:[0,0,0]};
  state.entityRuntime.velocity=[0,0,0];
  state.box.physics={...state.box.physics,enabled:false,body:"sleeping-dynamic"};
  const worldObject=model.world.objects.find(item=>item.identity===state.identity);
  if(worldObject)worldObject.physics={...worldObject.physics,enabled:false,body:"sleeping-dynamic"};
  const marker=document.querySelector(`.projectile-entity-marker[data-entity="${state.identity}"]`);
  if(marker)marker.dataset.projectileStatus="sleeping";
  if(stateLoopRuntime.ready)stateLoopRuntime.worker.postMessage({type:"remove",identity:state.identity});
  releasePhysicsSnapshotSlot(state.identity);
  return true;
}

function wakeProjectilePhysics(state,reason="collision-touch",velocity=null) {
  if(!state?.sleeping)return false;
  state.sleeping=false;state.settleTime=0;state.age=0;state.lastContact=null;
  state.velocity=velocity?[...velocity]:[0,0,0];
  state.record.status="active";state.record.physics_membership="active";
  state.record.wake_reason=reason;state.record.woke_at=performance.now();
  state.entity.traits.status="active";state.entity.traits.physics_membership="active";
  state.box.physics={...state.box.physics,enabled:true,body:"dynamic"};
  const worldObject=model.world.objects.find(item=>item.identity===state.identity);
  if(worldObject)worldObject.physics={...worldObject.physics,enabled:true,body:"dynamic"};
  const marker=document.querySelector(`.projectile-entity-marker[data-entity="${state.identity}"]`);
  if(marker)marker.dataset.projectileStatus="active";
  registerProjectilePhysicsMembership(state);
  return true;
}

function wakeSleepingProjectiles(reason="physics-field-change") {
  let count=0;projectileState.forEach(state=>{if(wakeProjectilePhysics(state,reason))count+=1;});
  return count;
}

function expireProjectile(state, reason) {
  state.record.status = "pickup"; state.record.expiry_reason = reason;
  state.record.expired_at = performance.now();
  if (state.entity) {
    state.entity.traits.status = "pickup";
    state.entity.pose = {...state.entity.pose, position: [...state.position],
      velocity: [0,0,0]};
  }
  document.querySelector(`.projectile-entity-marker[data-entity="${state.identity}"]`)?.remove();
  document.querySelector(`.projectile-entity-card[data-node-id="${state.identity}"]`)?.remove();
  projectileState.delete(state.identity);
  projectilePickupState.set(state.identity, state);
  portalRuntime.lastPositions.delete(state.identity);
  portalRuntime.cooldowns.delete(state.identity);
  portalRuntime.transits.delete(state.identity);
  if (stateLoopRuntime.ready) stateLoopRuntime.worker.postMessage({type: "remove", identity: state.identity});
  releasePhysicsSnapshotSlot(state.identity);
  const organization = model.entity_mezzanine.organizations.find(group =>
    group.identity === model.projectiles.organization);
  if (organization) organization.members = organization.members.filter(
    identity => identity !== state.identity);
  entityState.delete(state.identity);
  model.entity_mezzanine.entities = model.entity_mezzanine.entities.filter(
    entity => entity.identity !== state.identity);
  state.box.kind = "projectile-pickup";
  state.box.label = `${state.box.label} pickup`;
  state.box.physics = {...state.box.physics, enabled: false, body: "static-pickup"};
  state.box.placement = {...state.box.placement, custody: "world-pickup",
    placement_kind: "ammunition-pickup"};
  const worldObject=model.world.objects.find(item=>item.identity===state.identity);
  if(worldObject){
    worldObject.kind="projectile-pickup";worldObject.label=state.box.label;
    worldObject.capabilities=["inspect","pick-up"];
    worldObject.physics={...worldObject.physics,enabled:false,body:"static-pickup"};
    worldObject.extensions={...worldObject.extensions,"abstract_ui.pickup":{
      inventory_operation:"projectile-ammunition",quantity:1,status:"available"}};
  }
  rebuildPortableSceneMesh();
}

function firePhysicsBall(exitVelocityScale=1,launchOverride=null) {
  const system = model.projectiles, archetype = system?.archetype;
  const mountedRound=launchOverride?.ammunitionAuthority==="vehicle-turret",ammo=mountedRound?null:projectileAmmoItem();
  if (!system || !archetype || (!mountedRound&&(!ammo || ammo.quantity <= 0))) {
    setPlacementStatus("physics-ball gun is empty"); return false;
  }
  if (!physicsRuntime.instance && !stateLoopRuntime.ready) {
    setPlacementStatus("compiled physics is still initializing"); return false;
  }
  const active = [...projectileState.values()].filter(state=>!state.sleeping)
    .sort((a, b) => a.born - b.born);
  if (active.length >= archetype.maximum_active) sleepProjectilePhysics(active[0], "active-capacity");
  const facing = normalized3(launchOverride?.direction||shaderViewer.cameraFacing||[0,0,-1]);
  const camera = launchOverride?.origin||(vehicleRuntime.active&&vehicleRuntime.state
    ? [vehicleRuntime.state.position[0],vehicleRuntime.state.position[1]+Number(
        vehicleRuntime.active.configuration.chassis.camera_height),vehicleRuntime.state.position[2]]
    : (shaderViewer.cameraPosition || viewportControls.position));
  if (!camera) return false;
  const radius = Number(archetype.geometry.radius),projectilePhysics={...archetype.physics,
    mass:Math.max(.001,Number(launchOverride?.massKg||archetype.physics.mass||.001))};
  ensureMusicRoomAudio().catch(()=>{});
  const projectileColor=spectrumColorHex(projectileSequence*.11);
  const identity = `${system.identity}/instances/ball-${++projectileSequence}`;
  const muzzleClearance=mountedRound ? .12 : radius+.18,position = [camera[0] + facing[0] * muzzleClearance,
    camera[1] + facing[1] * muzzleClearance,camera[2] + facing[2] * muzzleClearance];
  const launchSpeed=Number(launchOverride?.speed||archetype.launch_speed)*Math.max(.1,Number(exitVelocityScale)||1);
  const carrierVelocity=vehicleRuntime.active?.identity&&vehicleRuntime.state
    ? vehicleRuntime.state.velocity:[0,0,0];
  const velocity = facing.map((value,index) => value * launchSpeed+carrierVelocity[index]);
  const box = {identity, kind: "physics-ball", label: `Physics ball ${projectileSequence}`,
    parent_identity: launchOverride?.owner||viewportControls.policy?.actor||model.identity,
    spatial_container: model.document_geometry.boxes[0].identity,
    center: [position[0], position[2]], center_y: position[1],
    half_extent: [radius, radius], radius, height: radius * 2,
    floor_height: 0, wall_thickness: 0, palette_role: archetype.palette_role,
    wall_palette_role: archetype.palette_role, geometry_mode: "sphere", openings: [],
    appearance: {face_color: projectileColor},
    placement: {custody: "placed", placement_kind: "projectile"},
    physics: {...projectilePhysics, enabled: true}};
  const record = {identity, kind: "physics-ball", name: box.label,
    archetype: archetype.identity, owner: box.parent_identity,
    organization: system.organization, status: "active", born_at: performance.now(),
    pose: {position: [...position], velocity: [...velocity]},
    lifetime: archetype.lifetime, geometry: archetype.geometry,
    physics: projectilePhysics};
  const entity = {
    identity, kind: "entity", name: box.label, archetype: "physics-ball-entity",
    geometry: {kind: "sphere", parameters: {radius}},
    texture: {kind: "procedural", reference: "abstract-ui:physics-ball", parameters: {}},
    capabilities: ["inspect", "collide", "expire", "track-on-map"],
    controller: {kind: "compiled-projectile-physics",
      source: archetype.physics.program, parameters: {projectile_system: system.identity}},
    pose: {coordinate_space: "data-world", position: [...position], velocity: [...velocity],
      acceleration: [0,0,0], jerk: [0,0,0], facing: [...facing]},
    principal: box.parent_identity,
    traits: {color: projectileColor,
      projectile: true, marker_scale: archetype.entity_contract.top_down_marker_scale,
      status: "active"},
    color: projectileColor,
    interaction: {type: "inspect", destination: identity}
  };
  system.instances.push(record); index.set(identity, record);
  model.entity_mezzanine.entities.push(entity);
  const organization = model.entity_mezzanine.organizations.find(group =>
    group.identity === system.organization);
  if (organization) organization.members.push(identity);
  model.world.objects.push({identity, kind: "physics-ball", parent: box.parent_identity,
    label: box.label, transform: {position: [...position], coordinate_space: "data-world"},
    form: {recipe: "sphere", radius}, material_bindings: {body: archetype.palette_role},
    capabilities: ["inspect", "collide", "expire"],
    semantic_parts: [{identity: `${identity}/surface:body`, role: "projectile-body",
      material_role: archetype.palette_role}], physics: projectilePhysics,
    persistence: {authority: "projectile-lifecycle", revision: shaderViewer.revision},
    extensions: {"abstract_ui.projectile": record}});
  model.world.object_order.push(identity);
  model.world.dynamic_object_order = [...(model.world.dynamic_object_order || []), identity];
  const entityRuntime = {entity, position: [...position], velocity: [...velocity],
    acceleration: [0,0,0], jerk: [0,0,0], facing: [...facing],
    worldPosition: [...position]};
  entityState.set(identity, entityRuntime);
  const entityGrid = document.querySelector(".entity-grid");
  if (entityGrid) {
    const card = makeSelectable(entity, "entity-card projectile-entity-card");
    card.append(div("kind", entity.controller.kind), div("node-name", entity.name),
      div("metaphor", `physics-ball-entity · ${entity.principal}`));
    entityGrid.append(card);
  }
  const marker = div("entity-sprite projectile-entity-marker");
  marker.dataset.entity = identity; marker.dataset.projectileStatus = "active";
  marker.style.setProperty("--entity-color", entity.color);
  marker.setAttribute("role", "img");
  marker.setAttribute("aria-label", `${box.label} top-down map marker`);
  document.getElementById("entity-layer")?.append(marker);
  const state = {identity, box, record, entity, entityRuntime, position, velocity,
    born: performance.now(), age: 0};
  projectileState.set(identity, state); shaderViewer.geometry.push(box);
  registerProjectilePhysicsMembership(state);
  if(ammo)ammo.quantity-=1;shaderViewer.revision+=1;
  model.scene_mesh.revision = shaderViewer.revision;
  rebuildPortableSceneMesh(); refreshInventoryCounts();
  setPlacementStatus(mountedRound?`turret fired · ${launchSpeed.toFixed(2)} m/s`:
    `fired ${box.label} · ${launchSpeed.toFixed(2)} m/s · ${ammo.quantity} remaining`);
  return true;
}

function projectileContactNormal(state, contactIdentity, radius) {
  if(state.position[1]<=radius+.025)return [0,1,0];
  const collider=shaderViewer.colliders.find(item=>item.identity===contactIdentity);
  if(!collider||collider.role==="projectile-body")return null;
  const candidates=[
    [Math.abs(state.position[0]-collider.minimum[0]),[-1,0,0]],
    [Math.abs(state.position[0]-collider.maximum[0]),[1,0,0]],
    [Math.abs(state.position[2]-collider.minimum[2]),[0,0,-1]],
    [Math.abs(state.position[2]-collider.maximum[2]),[0,0,1]],
    [Math.abs(state.position[1]-collider.maximum[1]),[0,1,0]],
  ];
  return candidates.sort((a,b)=>a[0]-b[0])[0][1];
}

function bounceProjectile(state,radius) {
  const floorContact=state.position[1]<=radius+.025&&state.velocity[1]<0;
  const contact=state.record.contact||(floorContact?"world-floor":null);
  if(!contact){state.lastContact=null;return false;}
  if(state.lastContact===contact)return false;
  state.lastContact=contact;
  const normal=projectileContactNormal(state,contact,radius);if(!normal)return false;
  const incoming=state.velocity.reduce((sum,value,index)=>sum+value*normal[index],0);
  if(incoming>=-.18)return false;
  const restitution=Number(model.projectiles.archetype.physics.restitution||.78);
  const speed=Math.hypot(...state.velocity);
  state.velocity=state.velocity.map((value,index)=>value-(1+restitution)*incoming*normal[index]);
  state.position=state.position.map((value,index)=>value+normal[index]*.012);
  synchronizePortalBody(state.identity,state.position,state.velocity);
  playToyImpact(speed,state.position);return true;
}

function resolveProjectilePairs(radius) {
  const balls=[...projectileState.values()],diameter=radius*2;
  for(let a=0;a<balls.length;a+=1)for(let b=a+1;b<balls.length;b+=1){
    const left=balls[a],right=balls[b],delta=right.position.map((v,i)=>v-left.position[i]);
    let distance=Math.hypot(...delta);if(distance>=diameter||(left.sleeping&&right.sleeping))continue;
    if(left.sleeping)wakeProjectilePhysics(left,"projectile-collision");
    if(right.sleeping)wakeProjectilePhysics(right,"projectile-collision");
    const normal=distance>1e-6?delta.map(v=>v/distance):[1,0,0];
    distance=Math.max(distance,1e-6);const relative=right.velocity.reduce(
      (sum,v,i)=>sum+(v-left.velocity[i])*normal[i],0);
    const overlap=diameter-distance;
    left.position=left.position.map((v,i)=>v-normal[i]*overlap*.51);
    right.position=right.position.map((v,i)=>v+normal[i]*overlap*.51);
    if(relative<0){
      const impulse=-(1+Number(model.projectiles.archetype.physics.ball_restitution||.9))*relative/2;
      left.velocity=left.velocity.map((v,i)=>v-normal[i]*impulse);
      right.velocity=right.velocity.map((v,i)=>v+normal[i]*impulse);
      playToyImpact(Math.abs(relative),left.position);
    }
    synchronizePortalBody(left.identity,left.position,left.velocity);
    synchronizePortalBody(right.identity,right.position,right.velocity);
  }
}

function resolveProjectilePlayerContacts(radius) {
  const player=viewportControls.position,identity=viewportControls.policy?.actor;
  if(!player||!identity)return;
  const playerRadius=Number(model.viewer.camera.collision_radius||.001);
  const eyeHeight=Number(model.viewer.camera.eye_height),foot=player[1]-eyeHeight;
  const capsuleBottom=foot+playerRadius,capsuleTop=player[1];
  const planar=viewportControls.horizontalVelocity;
  projectileState.forEach(state=>{
    const closest=[player[0],Math.max(capsuleBottom,Math.min(capsuleTop,state.position[1])),player[2]];
    const delta=state.position.map((value,index)=>value-closest[index]);
    let distance=Math.hypot(...delta),limit=radius+playerRadius;
    if(distance>=limit)return;
    if(state.sleeping)wakeProjectilePhysics(state,"player-collision");
    const normal=distance>1e-6?delta.map(value=>value/distance):[0,1,0];
    distance=Math.max(distance,1e-6);
    state.position=state.position.map((value,index)=>value+normal[index]*(limit-distance+.002));
    const body=stateLoopRuntime.bodies.get(identity);
    const playerVelocity=[planar[0],body?.velocity[1]||physicsRuntime.verticalVelocity,planar[1]];
    const relative=state.velocity.reduce((sum,value,index)=>sum+(value-playerVelocity[index])*normal[index],0);
    if(relative<0){
      const playerInverseMass=.22,restitution=.58;
      const impulse=-(1+restitution)*relative/(1+playerInverseMass);
      state.velocity=state.velocity.map((value,index)=>value+normal[index]*impulse);
      planar[0]-=normal[0]*impulse*playerInverseMass;
      planar[1]-=normal[2]*impulse*playerInverseMass;
      const vertical=playerVelocity[1]-normal[1]*impulse*playerInverseMass;
      physicsRuntime.verticalVelocity=vertical;
      if(body){body.velocity[1]=vertical;stateLoopRuntime.worker?.postMessage({type:"impulse",identity,
        velocity:[planar[0],vertical,planar[1]]});}
      playToyImpact(Math.abs(relative),state.position);
    }
    synchronizePortalBody(state.identity,state.position,state.velocity);
  });
}

function updatePhysicsBalls(dt) {
  if (!projectileState.size || (!physicsRuntime.instance && !stateLoopRuntime.ready) || dt <= 0) return;
  let changed = false;
  [...projectileState.values()].forEach(state => {
    if(state.sleeping)return;
    state.age += dt;
    if (state.age >= Number(model.projectiles.archetype.lifetime)) {
      sleepProjectilePhysics(state, "awake-lifetime-budget"); changed = true; return;
    }
    const radius = Number(model.projectiles.archetype.geometry.radius);
    const previousY=state.position[1];
    if (stateLoopRuntime.ready) {
      const body = stateLoopRuntime.bodies.get(state.identity);
      if (!body) return;
      state.position = [...body.position]; state.velocity = [...body.velocity];
      state.record.contact = body.contactIdentity; changed = true;
    } else {
    const result = runCompiledPhysicsState(state.position, state.velocity, radius, dt, {
      linear_drag: Number(model.projectiles.archetype.physics.linear_drag),
      minimum_y: 0, portal_active: 0,
    }, state.identity);
    if (!result) return;
    state.position = [result.position_x_next, result.position_y_next, result.position_z_next];
    state.velocity = [result.velocity_x_next, result.velocity_y_next, result.velocity_z_next];
    state.record.contact = result.contactIdentity;
    }
    const support=sampleContactSurface(state.position[0],state.position[2],previousY-radius,
      state.position[1]-radius,state.velocity[1],.035);
    if(support){
      const surfaceY=support.height+radius;
      const contactConstraintVelocity=(surfaceY-previousY)/Math.max(dt,1e-6);
      state.position[1]=surfaceY;
      state.velocity[1]=Math.max(state.velocity[1],contactConstraintVelocity);
      state.record.contact=support.identity;
      synchronizePortalBody(state.identity,state.position,state.velocity);
    }
    bounceProjectile(state,radius);
    state.box.center = [state.position[0], state.position[2]];
    state.box.center_y = state.position[1];
    state.record.pose = {position: [...state.position], velocity: [...state.velocity]};
    state.entity.pose = {...state.entity.pose, position: [...state.position],
      velocity: [...state.velocity]};
    state.entityRuntime.worldPosition = [...state.position];
    state.entityRuntime.position = [...state.position];
    state.entityRuntime.velocity = [...state.velocity];
    const speed3=Math.hypot(...state.velocity);
    const supported=Boolean(state.record.contact)||state.position[1]<=radius+.03;
    state.settleTime=supported&&speed3<.32?(state.settleTime||0)+dt:0;
    if(state.settleTime>=1.15){
      sleepProjectilePhysics(state,"slow-enough");changed=true;return;
    }
    const planarSpeed = Math.hypot(state.velocity[0], state.velocity[2]);
    if (planarSpeed > 1e-6) {
      state.entityRuntime.facing = [state.velocity[0]/planarSpeed,
        state.velocity[2]/planarSpeed,0];
    }
    changed = true;
  });
  resolveProjectilePairs(Number(model.projectiles.archetype.geometry.radius));
  resolveProjectilePlayerContacts(Number(model.projectiles.archetype.geometry.radius));
  if (changed && (!stateLoopRuntime.ready || stateLoopRuntime.appliedSequence !== stateLoopRuntime.latestSequence)) {
    stateLoopRuntime.appliedSequence = stateLoopRuntime.latestSequence;
    rebuildPortableSceneMesh();
  }
}

function collectProjectilePickup(state) {
  const ammo=projectileAmmoItem();
  if(!state||!ammo||ammo.quantity>=ammo.maximum_stack)return false;
  ammo.quantity=Math.min(ammo.maximum_stack,ammo.quantity+1);
  projectilePickupState.delete(state.identity);
  const geometryIndex=shaderViewer.geometry.indexOf(state.box);
  if(geometryIndex>=0)shaderViewer.geometry.splice(geometryIndex,1);
  model.projectiles.instances=model.projectiles.instances.filter(
    record=>record.identity!==state.identity);
  model.world.objects=model.world.objects.filter(item=>item.identity!==state.identity);
  model.world.object_order=model.world.object_order.filter(identity=>identity!==state.identity);
  model.world.dynamic_object_order=(model.world.dynamic_object_order||[]).filter(
    identity=>identity!==state.identity);
  index.delete(state.identity);shaderViewer.revision+=1;
  model.scene_mesh.revision=shaderViewer.revision;
  refreshInventoryCounts();rebuildPortableSceneMesh();
  setPlacementStatus(`picked up ${state.box.label} · ${ammo.quantity} ammunition`);
  return true;
}

function absorbProjectileIntoAmmo(state,reason="attractor-absorption") {
  const ammo=projectileAmmoItem();
  if(!state||!ammo||ammo.quantity>=ammo.maximum_stack)return false;
  if(projectileState.has(state.identity))expireProjectile(state,reason);
  if(!projectilePickupState.has(state.identity))return false;
  return collectProjectilePickup(state);
}

function collectNearbyProjectile(manual=false) {
  const player=shaderViewer.cameraPosition||viewportControls.position;if(!player)return false;
  const candidates=[...projectilePickupState.values()].map(state=>({state,distance:Math.hypot(
    state.position[0]-player[0],state.position[1]-player[1],state.position[2]-player[2])}))
    .sort((left,right)=>left.distance-right.distance);
  const nearest=candidates[0],reach=manual?2.2:.48;
  if(nearest&&nearest.distance<=reach)return collectProjectilePickup(nearest.state);
  if(manual){
    const sleeping=[...projectileState.values()].filter(state=>state.sleeping).map(state=>({state,
      distance:Math.hypot(state.position[0]-player[0],state.position[1]-player[1],
        state.position[2]-player[2])})).sort((left,right)=>left.distance-right.distance)[0];
    if(sleeping&&sleeping.distance<=reach){
      expireProjectile(sleeping.state,"manual-ammunition-transition");
      return collectProjectilePickup(sleeping.state);
    }
    setPlacementStatus("no sleeping physics ball within reach");
  }
  return false;
}

function renderEntityMezzanine() {
  const mezzanine = model.entity_mezzanine;
  if (!mezzanine) return;
  const district = div("entity-district");
  district.append(div("kind", "system-root mezzanine"), div("node-name", "Entity organizations"));
  mezzanine.organizations.forEach(group => {
    district.append(div("metaphor", `${group.name} · ${group.members.length} entities · ${group.cycle}`));
  });
  const actionTable = div("action-table");
  actionTable.append(
    div("metaphor", `${model.action_mezzanine.timer.identity} -> update(actions) -> ${actionEdges.identity}`),
    div("action-table-head")
  );
  actionTable.lastChild.append(
    div("", "source"), div("", "interaction"), div("", "destination"), div("", "count")
  );
  const actionRows = div("action-edge-rows");
  actionEdges.bind(actionRows);
  actionTable.append(actionRows);
  const actionRegistry = document.createElement("details");
  actionRegistry.className = "action-registry";
  const actionSummary = document.createElement("summary");
  actionSummary.textContent = "Registered actions";
  actionRegistry.append(actionSummary, actionTable);
  district.append(actionRegistry);
  const cards = div("entity-grid");
  const layer = document.getElementById("entity-layer");
  const playerSpawn = model.document_geometry.boxes.find(box => box.kind === "courtyard");
  mezzanine.entities.forEach(entity => {
    index.set(entity.identity, entity);
    const state = {
      entity,
      position: [...entity.pose.position],
      velocity: [...entity.pose.velocity],
      acceleration: [...entity.pose.acceleration],
      jerk: [...entity.pose.jerk],
      facing: [...entity.pose.facing],
      worldPosition: entity.controller.kind === "world-player" ? [
        playerSpawn?.center[0] || entity.pose.position[0],
        model.viewer.camera.eye_height,
        playerSpawn ? playerSpawn.center[1] + playerSpawn.half_extent[1] * 0.72 : entity.pose.position[2]
      ] : null
    };
    entityState.set(entity.identity, state);
    const card = makeSelectable(entity, "entity-card");
    const description = div("metaphor");
    const swatch = div("color-swatch");
    swatch.style.background = entity.color;
    description.append(swatch, document.createTextNode(
      `${entity.archetype} · ${entity.principal} · ${entity.color}`
    ));
    card.append(div("kind", entity.controller.kind), div("node-name", entity.name), description);
    cards.append(card);
    const sprite = div("entity-sprite");
    sprite.dataset.entity = entity.identity;
    sprite.style.setProperty("--entity-color", entity.color);
    sprite.setAttribute("aria-hidden", "true");
    layer.append(sprite);
  });
  district.append(cards);
  mapRoot.append(district);
}

function acceptControlInput(event) {
  const previous = latestControlInput?.position;
  const dx = previous ? event.clientX - previous[0] : 0;
  const dy = previous ? event.clientY - previous[1] : -1;
  const length = Math.hypot(dx, dy);
  if (length > 0.01) viewportControls.lastPointerMotion = performance.now();
  viewportControls.pointerButtons = event.buttons;
  const facing = length > 0.01
    ? [dx / length, dy / length, 0]
    : (latestControlInput?.facing || [0, -1, 0]);
  const bounds = shaderViewer.mapElement?.getBoundingClientRect();
  if (!viewportControls.highlighted) {
    shaderViewer.active = Boolean(bounds && event.clientX >= bounds.left && event.clientX <= bounds.right &&
      event.clientY >= bounds.top && event.clientY <= bounds.bottom);
  }
  latestControlInput = {
    controller: "mouse.primary", sequence: event.timeStamp, time: event.timeStamp,
    position: [event.clientX, event.clientY, 0], coordinate_space: "viewport",
    buttons: event.buttons, facing
  };
  if (controlFocus.mode === "projected-pointer" && shaderViewer.canvas) {
    const canvasBounds = shaderViewer.canvas.getBoundingClientRect();
    controlFocus.projectedPosition = [
      Math.max(0, Math.min(canvasBounds.width, event.clientX - canvasBounds.left)),
      Math.max(0, Math.min(canvasBounds.height, event.clientY - canvasBounds.top)),
    ];
    latestControlInput.route = "projected-pointer";
    latestControlInput.projected_position = [...controlFocus.projectedPosition];
    latestControlInput.projected_coordinate_space = "document";
  }
}

function updateVehicleWorldMarker(){
  const marker=vehicleRuntime.worldMarker,state=vehicleRuntime.state||vehicleRuntime.parkedState;
  if(!marker||!state||!shaderViewer.mapElement)return;
  try{const screen=worldToDocumentPoint([state.position[0],state.position[2]]);
    marker.style.transform=`translate3d(${screen[0]}px,${screen[1]}px,0)`;
    marker.classList.toggle("active",Boolean(vehicleRuntime.active));
  }catch(error){reportRuntimeFault("vehicle-map-marker",error);}
}

function viewportGamepad() {
  if (!navigator.getGamepads || !viewportControls.policy?.captures.includes("gamepad")) return null;
  return [...navigator.getGamepads()].find(Boolean) || null;
}

function updateVehicleTransmissionControls(){
  const root=vehicleRuntime.transmissionControls,vehicle=vehicleRuntime.active;if(!root||!vehicle)return;
  const transmission=vehicleRuntime.transmission,config=vehicle.configuration,
    gearLabel=transmission.displayGear<0?"R":String(Math.max(1,Math.round(
      transmission.displayGear||transmission.gear||config.transmission.starting_gear))),
    readout=root.querySelector("[data-transmission-readout]"),
    text=`${transmission.mode==="automatic"?"AUTO":"MANUAL"} · ${gearLabel} · ${String(transmission.transferRange||
      (transmission.lowRange?"l2":"high")).toUpperCase()}`;
  if(readout&&readout.textContent!==text)readout.textContent=text;
  root.querySelectorAll("[data-transmission-mode]").forEach(button=>{const active=
    button.dataset.transmissionMode===transmission.mode;if(button.classList.contains("active")!==active)
      button.classList.toggle("active",active);});
  root.querySelectorAll("[data-drivetrain-toggle]").forEach(button=>{const key=button.dataset.drivetrainToggle,
    active=Boolean(transmission[key]);button.classList.toggle("active",active);
    button.setAttribute("aria-pressed",String(active));});
  root.querySelectorAll("[data-transfer-range]").forEach(button=>{const active=button.dataset.transferRange===(
    transmission.transferRange||(transmission.lowRange?"l2":"high"));button.classList.toggle("active",active);
    button.setAttribute("aria-pressed",String(active));});
  root.querySelectorAll("[data-differential-mode]").forEach(select=>{const key=`${select.dataset.differentialMode}DiffMode`,
    value=transmission[key]||"open";if(document.activeElement!==select&&select.value!==value)select.value=value;});
  root.querySelectorAll("[data-brake-lock]").forEach(button=>{const name=button.dataset.brakeLock,
    active=Boolean(vehicleRuntime.brakeLocks[name]);button.classList.toggle("active",active);
    button.setAttribute("aria-pressed",String(active));});
  const split=root.querySelector("[data-front-drive-share]");if(split){const percent=Math.round(transmission.frontDriveShare*100);
    if(document.activeElement!==split)split.value=String(percent);const value=root.querySelector("[data-torque-split-value]");
    if(value)value.textContent=`F${percent}/R${100-percent}`;}
  const powerUnit=root.querySelector("[data-power-unit-preset]");if(powerUnit&&
      powerUnit.value!==vehicle.power_unit_preset)powerUnit.value=vehicle.power_unit_preset;
  const equationMode=root.querySelector("[data-engine-equation-mode]");if(equationMode&&
      equationMode.value!==vehicleRuntime.engineEquationMode)equationMode.value=vehicleRuntime.engineEquationMode;
  const transmissionPreset=root.querySelector("[data-transmission-preset]");if(transmissionPreset&&
      transmissionPreset.value!==vehicleRuntime.transmissionPreset)transmissionPreset.value=vehicleRuntime.transmissionPreset;
  const chassisProfile=root.querySelector("[data-chassis-profile]");if(chassisProfile&&
      chassisProfile.value!==vehicleRuntime.chassisProfile)chassisProfile.value=vehicleRuntime.chassisProfile;
  const geometry=vehicle.chassis_geometry_parameters||{};root.querySelectorAll("[data-chassis-geometry]").forEach(input=>{
    if(document.activeElement===input)return;const value=input.dataset.chassisGeometry==="chassisLengthM"?
      geometry.chassis_length_m:geometry.wheelbase_m;if(Number.isFinite(Number(value)))input.value=Number(value).toFixed(2);});
  const wheelPart=root.querySelector("[data-wheel-part]");if(wheelPart&&
      wheelPart.value!==vehicleRuntime.wheelPart)wheelPart.value=vehicleRuntime.wheelPart;
  const clutchPreset=root.querySelector("[data-clutch-preset]");if(clutchPreset&&
      clutchPreset.value!==vehicleRuntime.clutchPreset)clutchPreset.value=vehicleRuntime.clutchPreset;
  const bodyShell=root.querySelector("[data-body-shell]");if(bodyShell&&
      bodyShell.value!==vehicleRuntime.bodyShell)bodyShell.value=vehicleRuntime.bodyShell;
  const turretTakeover=root.querySelector("[data-turret-fire-takeover]");if(turretTakeover){
    turretTakeover.checked=Boolean(vehicleRuntime.turretSystem.fireTakeover);turretTakeover.disabled=vehicleRuntime.bodyShell!=="five-turret-carrier";}
  const turretReadout=root.querySelector("[data-turret-readout]");if(turretReadout){const turret=vehicleRuntime.turretSystem;
    turretReadout.textContent=vehicleRuntime.bodyShell==="five-turret-carrier"?`${turret.ammoCount} rounds · ${
      turret.interlockReason||"interlock clear"}`:"carrier not installed";}
  const outrigger=root.querySelector("[data-outrigger-toggle]");if(outrigger){const active=Boolean(
    vehicleRuntime.turretSystem.outriggers.commanded);outrigger.classList.toggle("active",active);
    outrigger.setAttribute("aria-pressed",String(active));outrigger.disabled=vehicleRuntime.bodyShell!=="five-turret-carrier";
    outrigger.textContent=active?"WITHDRAW OUTRIGGERS":"DEPLOY OUTRIGGERS";}
  root.querySelectorAll("[data-vehicle-auxiliary]").forEach(button=>{const key=button.dataset.vehicleAuxiliary,
    active=Boolean(vehicleRuntime.electrical[key]);button.classList.toggle("active",active);
    button.setAttribute("aria-pressed",String(active));});
  const energyReadout=root.querySelector("[data-vehicle-energy-readout]");if(energyReadout){const energy=vehicleRuntime.electrical;
    energyReadout.textContent=`${Number(energy.fuelMassKg||0).toFixed(1)}kg fuel · ${Math.round(Number(energy.stateOfCharge||0)*100)}% battery · ${
      Number(energy.alternatorPowerW||0).toFixed(0)}W alt · timing ${Number(energy.timingErrorDegrees||0)>=0?"+":""}${
      Number(energy.timingErrorDegrees||0).toFixed(0)}° · ECU ${energy.ecuOnline?"ON":"OFF"} · ${Math.round(Number(
        energy.tirePressurePa||0)/1000)}kPa${energy.pneumaticCompressorOn?" AIR":""}${energy.hydraulicPumpOn?" HYD":""}`;}
  root.querySelectorAll("[data-vehicle-chemistry]").forEach(select=>{const value=vehicleRuntime.electrical[
    select.dataset.vehicleChemistry];if(document.activeElement!==select&&value&&select.value!==value)select.value=value;});
  const assistance=vehicleRuntime.driverAssistance,drivingMode=root.querySelector("[data-driving-mode]");
  if(drivingMode&&document.activeElement!==drivingMode&&drivingMode.value!==assistance.drivingMode)
    drivingMode.value=assistance.drivingMode;
  const governor=root.querySelector("[data-governor-rpm]");if(governor&&document.activeElement!==governor)
    governor.value=String(Math.round(Number(assistance.governorRpm||6500)));
  const governorValue=root.querySelector("[data-governor-value]");if(governorValue)
    governorValue.textContent=`${Math.round(Number(assistance.governorRpm||6500))} rpm`;
  const cruise=root.querySelector("[data-cruise-toggle]");if(cruise){cruise.classList.toggle("active",Boolean(
    assistance.cruiseEnabled));cruise.setAttribute("aria-pressed",String(Boolean(assistance.cruiseEnabled)));}
  const cruiseTarget=root.querySelector("[data-cruise-target]");if(cruiseTarget&&document.activeElement!==cruiseTarget)
    cruiseTarget.value=String(Math.round(Number(assistance.cruiseTargetSpeedMps||0)*3.6));
  const cruiseValue=root.querySelector("[data-cruise-value]");if(cruiseValue)cruiseValue.textContent=`${Math.round(Number(
    assistance.cruiseTargetSpeedMps||0)*3.6)} km/h · T${Math.round(Number(assistance.cruiseThrottle||0)*100)} B${Math.round(
    Number(assistance.cruiseBrake||0)*100)}`;
  root.querySelectorAll("[data-shock-parameter]").forEach(input=>{const name=input.dataset.shockParameter,
    value=Number(vehicleRuntime.shockParameters?.[name]??vehicle.configuration_defaults?.[name]);
    if(document.activeElement!==input&&Number.isFinite(value)&&Number(input.value)!==value)input.value=String(value);});
  const alignmentStatus=root.querySelector("[data-alignment-status]");if(alignmentStatus){const calibration=
    vehicleRuntime.wheelAlignment?.calibration||{};alignmentStatus.textContent=`${calibration.continuous?"AUTO · ":""}${
      calibration.status||"static"}${Number.isFinite(calibration.maximumErrorDeg)?` · ${Number(calibration.maximumErrorDeg).toFixed(2)}° max`:""}`;}
  const levelingButton=root.querySelector("[data-chassis-leveling-toggle]"),leveling=vehicleRuntime.chassisLeveling;
  if(levelingButton&&leveling){levelingButton.classList.toggle("active",Boolean(leveling.enabled));
    levelingButton.setAttribute("aria-pressed",String(Boolean(leveling.enabled)));}
  root.querySelectorAll("[data-hydraulic-pose]").forEach(button=>button.classList.toggle("active",
    leveling?.active_pose===button.dataset.hydraulicPose));
  root.querySelectorAll("[data-wheel-height]").forEach(input=>{const value=Number(
    leveling?.manual_corner_targets_m?.[input.dataset.wheelHeight]||0)*1000;
    if(document.activeElement!==input)input.value=String(Math.round(value));});
  const poseRate=root.querySelector("[data-pose-lerp-rate]");if(poseRate&&document.activeElement!==poseRate)
    poseRate.value=String(Math.round(Number(leveling?.pose_lerp_rate_m_s||.055)*1000));
  const poseRateValue=root.querySelector("[data-pose-lerp-value]");if(poseRateValue)
    poseRateValue.textContent=`${Math.round(Number(leveling?.pose_lerp_rate_m_s||.055)*1000)} mm/s`;
  const pressure=root.querySelector("[data-tire-pressure]");if(pressure&&document.activeElement!==pressure)
    pressure.value=String(Math.round(Number(vehicleRuntime.electrical.tirePressureTargetPa||155000)/1000));
  const pressureValue=root.querySelector("[data-tire-pressure-value]");if(pressureValue)pressureValue.textContent=`${Math.round(
    Number(vehicleRuntime.electrical.tirePressurePa||155000)/1000)} kPa${vehicleRuntime.electrical.pneumaticCompressorOn?" · PUMP":""}`;
  const steeringSystem=vehicleRuntime.steeringSystem||{};
  root.querySelectorAll("[data-steering-axle]").forEach(button=>{const key=`${button.dataset.steeringAxle}_axle_enabled`,
      active=steeringSystem[key]!==false;button.classList.toggle("active",active);button.setAttribute("aria-pressed",String(active));});
  const steeringShare=root.querySelector("[data-steering-front-share]");if(steeringShare){const percent=Math.round(
      Number(steeringSystem.front_share??.5)*100);if(document.activeElement!==steeringShare)steeringShare.value=String(percent);
    const value=root.querySelector("[data-steering-share-value]");if(value)value.textContent=`F${percent}/R${100-percent}`;}
  const steeringPhase=root.querySelector("[data-steering-rear-phase]");if(steeringPhase){const counter=Number(
      steeringSystem.rear_phase??-1)<0;steeringPhase.textContent=counter?"REAR COUNTER":"REAR SAME";
    steeringPhase.classList.toggle("active",counter);steeringPhase.setAttribute("aria-pressed",String(counter));}
  const lockedBrakes=Object.values(vehicleRuntime.brakeLocks).filter(Boolean).length;
  const failedShafts=Object.values(vehicleRuntime.damage?.halfshaftHealth||{}).filter(value=>value<=0).length,
    failedMembers=Object.values(vehicleRuntime.damage?.members||{}).filter(value=>value.failed).length;
  const modeGlyph=mode=>mode==="locked"?"✓":mode==="limited-slip"?"≈":"○";
  if(vehicleRuntime.transferHud)vehicleRuntime.transferHud.textContent=`${String(transmission.transferRange||(
    transmission.lowRange?"l2":"high")).toUpperCase()} · ${
    `C${modeGlyph(transmission.centerDiffMode)}`}/${`F${modeGlyph(transmission.frontDiffMode)}`}/${
    `R${modeGlyph(transmission.rearDiffMode)}`} · ${Math.round(transmission.frontDriveShare*100)}/${
    Math.round((1-transmission.frontDriveShare)*100)}${lockedBrakes?` · B${lockedBrakes}`:""}${
      Number(vehicleRuntime.driverAssistance?.tiltAuthority||0)>.01?` · TILT ${Math.round(Number(
        vehicleRuntime.driverAssistance.tiltAuthority)*100)}%`:""}${
      failedShafts||failedMembers?` · DMG ${failedShafts}S/${failedMembers}M`:""}`;
  const dyno=root.querySelector("[data-vehicle-dyno-readout]");if(dyno){const result=vehicleRuntime.dyno;
    dyno.textContent=!result?"DYNO idle":`${result.status==="telemetry"?"LIVE":result.pass?"PASS":"FAULT"} · ${result.compute} · `+
      `${(result.forceY||[]).reduce((sum,value)=>sum+Number(value||0),0).toFixed(0)} N load · `+
      `${(result.forceX||[]).reduce((sum,value)=>sum+Number(value||0),0).toFixed(0)} N pull · `+
      `${(result.wheelTorque||[]).map(value=>Number(value||0).toFixed(0)).join("/")} Nm wheels · `+
      `${Number(result.high?.drivelineTorque||0).toFixed(0)}/${Number(result.ultraLow?.drivelineTorque||0).toFixed(0)} Nm H/L`;
    dyno.dataset.pass=result?.pass?"true":"false";}
}

function requestVehicleDyno(){
  const vehicle=vehicleRuntime.active,worker=stateLoopRuntime.worker;if(!vehicle||!worker||!stateLoopRuntime.ready)return false;
  vehicleRuntime.dyno=null;updateVehicleTransmissionControls();worker.postMessage({type:"vehicle-dyno",
    identity:vehicle.identity,requestId:++vehicleRuntime.dynoRequest});return true;
}

function updateVehicleContactMonitor() {
  const root=vehicleRuntime.contactMonitor;if(!root)return;
  const vehicle=vehicleRuntime.active,config=vehicle?.configuration;root.hidden=!vehicle||shaderViewer.shaderOnly;
  if(!vehicle||!config)return;
  const mode=root.querySelector(".vehicle-contact-mode"),names=["FL","FR","RL","RR"],
    modeNames=["airborne","static grip","at limit","kinetic slide"],
    shortModes=["AIR","GRIP","LIMIT","SLIDE"],
    colors=["#52635d","#54e39b","#ffd166","#ff5f7d"],tires=config.tires,
    travel=Math.max(1e-6,Number(config.suspension.travel)),
    loadAudit=vehicle.physics?.mechanical_graph?.load_audit,
    liveDom=root.classList.contains("expanded");
  if(mode&&liveDom){const nextMode=vehicleRuntime.error
    ?`FAULT · ${String(vehicleRuntime.error).slice(0,160)}`
    :vehicleRuntime.computeMode.replaceAll("-"," ");
    if(mode.textContent!==nextMode)mode.textContent=nextMode;}
  if(!liveDom)return;
  root.querySelectorAll(".vehicle-contact-corner").forEach((corner,index)=>{
    const area=Math.max(0,vehicleRuntime.contactAreas[index]||0),force=Math.max(0,vehicleRuntime.lastSpringForces[index]||0),
      utilization=Math.max(0,vehicleRuntime.frictionUtilizations[index]||0),contactMode=Math.round(vehicleRuntime.contactModes[index]||0),
      compression=Math.max(0,vehicleRuntime.compressions[index]||0),
      tractionScale=Math.max(0,Math.min(1,vehicleRuntime.tractionScales[index]??1)),
      brakeScale=Math.max(0,Math.min(1,vehicleRuntime.brakeScales[index]??1)),
      damperScale=Math.max(0,vehicleRuntime.damperScales[index]??1),
      tractionIntervention=(1-tractionScale)*100,brakeIntervention=(1-brakeScale)*100,
      areaRange=Math.max(1e-8,Number(tires.maximum_contact_area)-Number(tires.minimum_contact_area)),
      areaLevel=Math.max(0,Math.min(1,(area-Number(tires.minimum_contact_area))/areaRange)),
      patchWidth=area>0?Number(tires.width)*(.65+.20*areaLevel):0,
      patchLength=patchWidth>0?area/patchWidth:0,
      springLevel=Math.max(0,Math.min(1,compression/travel)),color=colors[contactMode]||colors[0],
      design=loadAudit?.corners?.[["front_left","front_right","rear_left","rear_right"][index]],
      designMass=Number(design?.design_supported_mass_kg||0),designLoad=Number(design?.design_static_load_n||0);
    corner.style.setProperty("--patch-color",color);corner.style.setProperty("--patch-width",`${55+45*areaLevel}%`);
    corner.style.setProperty("--patch-aspect",String(Math.max(.16,Math.min(.65,patchLength/Math.max(1e-6,patchWidth)))));
    corner.style.setProperty("--spring-level",`${Math.max(4,springLevel*100)}%`);
    corner.dataset.contactMode=modeNames[contactMode]||modeNames[0];
    const readout=corner.querySelector(".vehicle-contact-label");if(readout)readout.textContent=
      `${names[index]} ${shortModes[contactMode]||shortModes[0]} · ${(force/1000).toFixed(1)}kN`;
    const detail=corner.querySelector(".vehicle-contact-detail-line");if(detail)detail.textContent=
      `${modeNames[contactMode]||modeNames[0]} · ${(utilization*100).toFixed(0)}% friction · `+
      `${designMass.toFixed(0)}kg / ${(designLoad/1000).toFixed(2)}kN design · `+
      `${(patchWidth*100).toFixed(0)}×${(patchLength*100).toFixed(1)}cm patch · ${(compression*100).toFixed(1)}cm travel · `+
      `damper ${(damperScale*100).toFixed(0)}% · torque allowed TC ${(tractionScale*100).toFixed(0)}% / ABS ${(brakeScale*100).toFixed(0)}%`;
    [["tc",tractionIntervention],["abs",brakeIntervention]].forEach(([channel,intervention])=>{
      const indicator=corner.querySelector(`[data-${channel}-indicator]`),value=indicator?.querySelector("span");if(!indicator)return;
      indicator.style.setProperty("--control-intervention",`${intervention.toFixed(1)}%`);
      indicator.style.setProperty("--control-color",intervention>65?"#ff5f7d":intervention>25?"#ffd166":"#54e39b");
      indicator.title=`${channel.toUpperCase()} intervention ${intervention.toFixed(1)}%; ${(100-intervention).toFixed(1)}% command allowed`;
      indicator.setAttribute("aria-valuenow",intervention.toFixed(1));
      indicator.setAttribute("aria-valuetext",`${intervention.toFixed(1)}% intervention; ${(100-intervention).toFixed(1)}% command allowed`);
      if(value)value.textContent=`${channel.toUpperCase()} ${intervention.toFixed(0)}%`;
    });
  });
  const nodeColors=[];root.querySelectorAll(".vehicle-chassis-node").forEach((node,index)=>{
    const contactMode=Math.round(vehicleRuntime.contactModes[index]||0),color=colors[contactMode]||colors[0];
    node.setAttribute("fill",color);nodeColors[index]=color;
  });
  const pairs=[[0,1],[1,3],[3,2],[2,0],[0,3],[1,2]];
  root.querySelectorAll(".vehicle-chassis-member").forEach((member,index)=>{
    const [a,b]=pairs[index],load=Math.max(0,Math.min(1,((vehicleRuntime.compressions[a]||0)+
      (vehicleRuntime.compressions[b]||0))/(2*travel)));
    member.setAttribute("stroke",load>.82?"#ff5f7d":load>.58?"#ffd166":"#597269");
    member.setAttribute("stroke-width",String(1.5+2.5*load));
  });
  root.querySelectorAll(".vehicle-torque-value").forEach(value=>{
    const channel=value.dataset.torqueChannel,torque=Number(vehicleRuntime.powertrain[channel]||0);
    value.textContent=`${torque.toFixed(0)} Nm`;
    value.style.color=Math.abs(torque)>650?"#ff5f7d":Math.abs(torque)>180?"#ffd166":"#54e39b";
  });
  const reaction=root.querySelector(".vehicle-reaction-value"),r=vehicleRuntime.powertrain.reactionTorque||[0,0,0];
  if(reaction)reaction.textContent=`reaction ${Math.hypot(...r).toFixed(0)} Nm`;
  const massReadout=root.querySelector("[data-vehicle-mass-readout]");
  if(massReadout&&loadAudit){const cg=loadAudit.center_of_mass||[0,0,0],liveMass=Number(
    vehicleRuntime.electrical.totalMassKg||loadAudit.total_mass_kg);massReadout.textContent=
    `${liveMass.toFixed(1)}kg live · ${Number(vehicleRuntime.electrical.fuelMassKg||0).toFixed(1)}kg fuel · `+
    `CG ${Number(cg[0]).toFixed(2)}, ${Number(cg[1]).toFixed(2)}, ${Number(cg[2]).toFixed(2)}m`;}
}

function deviceSignalValue(source, gamepad, now) {
  if (source === "pointer:relative-motion") {
    return now - viewportControls.lastPointerMotion < 140 ? 1 : 0;
  }
  if (source === "pointer:button-0") return viewportControls.pointerButtons & 1 ? 1 : 0;
  if (source === "pointer:button-2") return viewportControls.pointerButtons & 2 ? 1 : 0;
  if (source.startsWith("keyboard:")) {
    return viewportControls.observedKeys.has(source.slice(9)) ? 1 : 0;
  }
  if (!gamepad) return 0;
  const axes = gamepad.axes;
  if (source === "gamepad:left-y-negative") return Math.max(0, -Number(axes[1] || 0));
  if (source === "gamepad:left-y-positive") return Math.max(0, Number(axes[1] || 0));
  if (source === "gamepad:left-x-negative") return Math.max(0, -Number(axes[0] || 0));
  if (source === "gamepad:left-x-positive") return Math.max(0, Number(axes[0] || 0));
  if (source === "gamepad:right-stick") {
    return Math.min(1, Math.hypot(Number(axes[2] || 0), Number(axes[3] || 0)));
  }
  if (source === "gamepad:button-0") return gamepad.buttons[0]?.value || 0;
  if (source === "gamepad:button-1") return gamepad.buttons[1]?.value || 0;
  return 0;
}

function updateViewportTelemetry(now) {
  // The compact live vehicle instruments are drawn by the viewport shader.
  // The DOM graph is an explicitly opened diagnostic, so do not synchronize
  // it from every animation frame while it is collapsed.
  if(!shaderViewer.shaderOnly&&vehicleRuntime.contactMonitor?.classList.contains("expanded"))updateVehicleContactMonitor();
  const gamepad = viewportGamepad();
  shaderViewer.element?.querySelectorAll("[data-control-source]").forEach(element => {
    const value = deviceSignalValue(element.dataset.controlSource, gamepad, now);
    element.classList.toggle("active", value >= 0.12);
    element.style.setProperty("--signal-level", String(value));
    element.title = `${element.dataset.action} · ${value.toFixed(2)}`;
  });
  shaderViewer.element?.querySelectorAll("[data-device]").forEach(element => {
    const device = element.dataset.device;
    const detected = device === "gamepad" ? Boolean(gamepad) :
      [...element.querySelectorAll(".device-signal")].some(signal => signal.classList.contains("active"));
    element.classList.toggle("detected", detected);
  });
  const actor = entityState.get(model.viewer.dynamics_space?.actor);
  const userValue = shaderViewer.element?.querySelector('[data-dynamics-value="user-dynamics"]');
  if (userValue && actor) {
    const position = actor.worldPosition || actor.position;
    const velocity = actor.worldPosition
      ? Math.hypot(actor.velocity[0], actor.velocity[2])
      : Math.hypot(actor.velocity[0], actor.velocity[1]);
    userValue.textContent = `p ${position[0].toFixed(2)},${position[2].toFixed(2)} · ` +
      `v ${velocity.toFixed(2)}`;
  }
  const worldValue = shaderViewer.element?.querySelector('[data-dynamics-value="world-physics"]');
  if (worldValue) {
    const contact = physicsRuntime.last?.contact_penetration;
    const contactIdentity = physicsRuntime.last?.contactIdentity;
    const gravity = physicsRuntime.parameters.get("gravity_y");
    const sleepingBalls=[...projectileState.values()].filter(state=>state.sleeping).length;
    const activeBalls=projectileState.size-sleepingBalls;
    worldValue.textContent = physicsRuntime.error ? `error · ${physicsRuntime.error}` :
      `${shaderViewer.geometry.length} geometry · contact ${contact === undefined ? "—" : contact.toFixed(4)}` +
      `${contactIdentity ? ` · ${contactIdentity.split("/").at(-1)}` : ""} · g ${gravity}` +
      ` · balls ${activeBalls} active/${sleepingBalls} sleeping`+
      ` · solver ${stateLoopRuntime.engineStage.replaceAll("-"," ")}`+
      `${vehicleRuntime.active?` · vehicle ${vehicleRuntime.active.name} · ${vehicleRuntime.computeMode} · springs ${vehicleRuntime.lastSpringForces.map(value=>Math.round(value)).join("/")} N`:""}`+
      ` · ${stateLoopRuntime.mode} p${stateLoopRuntime.latestSequence}/g${stateLoopRuntime.appliedSequence}`;
  }
  const focusValue = shaderViewer.element?.querySelector("[data-focus-mode]");
  if (focusValue) focusValue.textContent = controlFocus.dialogue
    ? `dialogue · ${controlFocus.dialogue}` : controlFocus.mode.replace("-", " ");
}

function runEntityCycle(now) {
  try{
  const dt = Math.min(0.05, Math.max(0, (now - entityCycleTime) / 1000));
  entityCycleTime = now;
  updateViewportControls(dt);
  updateMusicRoomAnalysis();
  updateEntityNavigation(dt);
  stepCompiledWorldPhysics(dt);
  updateParkedVehicle(dt);
  updatePhysicsBalls(dt);
  updateHeldToolPrimary(dt);
  updateHeldToolSecondary(dt);
  collectNearbyProjectile(false);
  updatePortalTraversals();
  advanceArtifactAttachments(dt);
  if (latestControlInput && !viewportControls.highlighted) {
    entityState.forEach(state => {
      if (state.entity.controller.kind === "native-input" &&
          state.entity.controller.source === latestControlInput.controller) {
        state.position = [...latestControlInput.position];
        state.facing = [...latestControlInput.facing];
      }
    });
  }
  entityState.forEach(state => {
    const orders = {
      "first-order-follow": 1, "second-order-follow": 2,
      "third-order-follow": 3, "fourth-order-follow": 4
    };
    const order = orders[state.entity.controller.kind];
    if (!order) return;
    const parameters = state.entity.controller.parameters;
    const target = entityState.get(parameters.target);
    if (!target) return;
    const omega = Number(parameters.frequency ?? 4);
    const derivatives = [state.position, state.velocity, state.acceleration, state.jerk, [0, 0, 0]];
    const coefficients = {
      1: [1, 1], 2: [1, 2, 1], 3: [1, 3, 3, 1], 4: [1, 4, 6, 4, 1]
    }[order];
    for (let axis = 0; axis < 3; axis += 1) {
      let highest = Math.pow(omega, order) * (target.position[axis] - state.position[axis]);
      for (let derivative = 1; derivative < order; derivative += 1) {
        highest -= coefficients[derivative] * Math.pow(omega, order - derivative) *
          derivatives[derivative][axis];
      }
      derivatives[order][axis] = highest;
    }
    const speed = Math.hypot(state.velocity[0], state.velocity[1]);
    if (speed > 0.001) state.facing = [state.velocity[0] / speed, state.velocity[1] / speed, 0];
    for (let derivative = order - 1; derivative >= 0; derivative -= 1) {
      for (let axis = 0; axis < 3; axis += 1) {
        derivatives[derivative][axis] += derivatives[derivative + 1][axis] * dt;
      }
    }
  });
  entityState.forEach((state, identity) => {
    const sprite = document.querySelector(`.entity-sprite[data-entity="${identity}"]`);
    if (sprite) {
      let screen = state.position;
      if (state.worldPosition && shaderViewer.mapElement) {
        screen = worldToDocumentPoint([state.worldPosition[0], state.worldPosition[2]]);
      } else if (shaderViewer.mapElement) {
        screen = viewportToElementPoint(screen, shaderViewer.mapElement);
      }
      sprite.style.transform = `translate3d(${screen[0]}px,${screen[1]}px,0)`;
    }
  });
  updateVehicleWorldMarker();
  updateShaderViewer();
  updateViewportTelemetry(now);
  updateSourceScopeForPlayer();
  }catch(error){reportRuntimeFault(vehicleRuntime.active?"mounted-frame":"world-frame",error);}
  requestAnimationFrame(runEntityCycle);
}

document.addEventListener("pointermove", acceptControlInput);
document.addEventListener("pointerdown", event => viewportControls.pointerButtons = event.buttons);
document.addEventListener("pointerup", event => {
  viewportControls.pointerButtons=event.buttons;
  if(event.button===0)endViewportPrimary("pointer");
  if(event.button===2)endViewportSecondary("pointer");
});
document.addEventListener("pointercancel", event => {
  viewportControls.pointerButtons=event.buttons||0;
  if(primaryActionState.down)endViewportPrimary("pointer-cancel");
  if(secondaryActionState.down)endViewportSecondary("pointer-cancel");
});
window.addEventListener("blur",()=>{
  if(primaryActionState.down)endViewportPrimary("window-blur");
  if(secondaryActionState.down)endViewportSecondary("window-blur");
});
document.addEventListener("contextmenu",event=>{
  if(shaderViewer.canvas&&(event.target===shaderViewer.canvas||
      document.pointerLockElement===shaderViewer.canvas)&&
      viewportBinding("secondary-action","pointer:button-2"))event.preventDefault();
},true);
document.addEventListener("mousemove", viewportPointerLook);
document.addEventListener("keydown", viewportKeyboardInput);
document.addEventListener("keyup", viewportKeyboardInput);
document.addEventListener("pointerlockchange", () => {
  viewportControls.pointerLocked = document.pointerLockElement === shaderViewer.canvas;
});
document.addEventListener("pointerdown", event => {
  if (shaderViewer.contextMenu && !event.target.closest(".context-menu")) closeSceneContextMenu();
});

function renderRoom(room) {
  const card = makeSelectable(room, "room");
  card.style.cssText = gridStyle(room);
  card.append(div("kind", room.member_kind), div("node-name", room.name), div("metaphor", room.metaphor));
  const observations = turingWorld.containedBy(room.identity)
    .filter(object => object.kind === "performance-observation");
  if (observations.length) {
    const loose = div("performance-observations");
    observations.forEach(observation => {
      const marker = makeSelectable(observation, "performance-observation");
      const label = observation.extensions?.["turing.performance"] || {};
      marker.dataset.inline = label.inline || "neutral";
      marker.dataset.hot = String(Boolean(label.hot_path));
      marker.textContent = `inline ${label.inline || "neutral"}`;
      loose.append(marker);
    });
    card.append(loose);
  }
  return card;
}

function renderBuilding(building) {
  index.set(building.identity, building);
  const node = div("building");
  node.dataset.nodeId = building.identity;
  node.style.cssText = gridStyle(building);
  node.append(div("kind", `${building.source_kind} · ${building.metaphor}`), div("node-name", building.name));
  const rooms = div("rooms");
  rooms.style.gridTemplateColumns = `repeat(${Math.max(1, 1 + Math.max(0, ...building.rooms.map(room => room.position.column)))}, minmax(120px, 1fr))`;
  building.rooms.forEach(room => rooms.append(renderRoom(room)));
  node.append(rooms);
  return node;
}

function renderRegion(region) {
  index.set(region.identity, region);
  const node = div("region");
  node.dataset.nodeId = region.identity;
  node.style.cssText = gridStyle(region);
  node.append(div("kind", "document region"), div("node-name", region.name));
  const buildings = div("buildings");
  buildings.style.gridTemplateColumns = `repeat(${Math.max(1, 1 + Math.max(0, ...region.buildings.map(item => item.position.column)))}, minmax(260px, 1fr))`;
  region.buildings.forEach(building => buildings.append(renderBuilding(building)));
  node.append(buildings);
  return node;
}

function renderDeviceMonitor(monitor) {
  index.set(monitor.identity, monitor);
  const strip = div("device-monitor");
  strip.dataset.nodeId = monitor.identity;
  monitor.groups.forEach(group => {
    const device = div("device-group");
    device.dataset.device = group.device;
    device.append(div("device-name", group.device));
    group.signals.forEach(signal => {
      index.set(signal.identity, signal);
      const light = div("device-signal", signal.label);
      light.dataset.nodeId = signal.identity;
      light.dataset.controlSource = signal.source;
      light.dataset.action = signal.action;
      device.append(light);
    });
    strip.append(device);
  });
  return strip;
}

function renderDynamicsSpace(space) {
  index.set(space.identity, space);
  const root = div("dynamics-space");
  root.dataset.nodeId = space.identity;
  space.lanes.forEach(lane => {
    index.set(lane.identity, lane);
    const node = div("dynamics-lane");
    node.dataset.nodeId = lane.identity;
    const head = div("dynamics-lane-head");
    head.append(div("", lane.kind), div("", lane.phase));
    const channels = div("dynamics-channels");
    lane.channels.forEach(channel => {
      channels.append(div(`dynamics-channel ${channel.status}`, channel.name));
    });
    const value = div("dynamics-value", "awaiting cycle");
    value.dataset.dynamicsValue = lane.kind;
    channels.append(value);
    node.append(head, channels);
    root.append(node);
  });
  if (model.physics_program) {
    const parameters = div("physics-parameters");
    parameters.dataset.nodeId = `${model.physics_program.identity}/parameters`;
    model.physics_program.parameters.filter(item => item.live_editable).forEach(parameter => {
      index.set(parameter.identity, parameter);
      const row = div("physics-parameter");
      row.dataset.nodeId = parameter.identity;
      const label = document.createElement("label");
      label.textContent = `${parameter.name} · ${parameter.unit}`;
      const input = document.createElement("input");
      input.type = "number";
      input.value = String(physicsRuntime.parameters.get(parameter.name));
      input.dataset.physicsParameter = parameter.name;
      input.id = parameter.identity;
      if (parameter.minimum !== undefined) input.min = String(parameter.minimum);
      if (parameter.maximum !== undefined) input.max = String(parameter.maximum);
      if (parameter.step !== undefined) input.step = String(parameter.step);
      label.htmlFor = input.id;
      row.append(label, input);
      parameters.append(row);
    });
    const transmissionRow=div("physics-parameter vehicle-transmission-settings"),
      transmissionLabel=div("","vehicle transmission");
    transmissionRow.append(transmissionLabel,renderVehicleTransmissionSettings());parameters.append(transmissionRow);
    root.append(parameters);
  }
  return root;
}

function renderVehicleTransmissionSettings(){
  const controls=div("vehicle-transmission-controls"),gearReadout=div("vehicle-gear-readout","AUTO · 2");
  gearReadout.dataset.transmissionReadout="true";controls.append(gearReadout);
  [["AUTO","automatic",null],["−",null,-1],["+",null,1]].forEach(([label,mode,gearDelta])=>{
    const button=document.createElement("button");button.type="button";button.className="vehicle-gear-button";
    button.textContent=label;if(mode)button.dataset.transmissionMode=mode;
    button.addEventListener("pointerdown",event=>event.stopPropagation());button.addEventListener("click",event=>{
      event.preventDefault();event.stopPropagation();controlVehicleTransmission({mode,gearDelta});});
    controls.append(button);});
  [["HIGH","high"],["L1","l1"],["L2 CRAWL","l2"]].forEach(([label,range])=>{const button=document.createElement("button");
    button.type="button";button.className="vehicle-gear-button";button.textContent=label;button.dataset.transferRange=range;
    button.setAttribute("aria-pressed",String(range==="high"));button.addEventListener("pointerdown",event=>event.stopPropagation());
    button.addEventListener("click",event=>{event.preventDefault();event.stopPropagation();controlVehicleTransmission({transferRange:range});});
    controls.append(button);});
  [["SMOOTH LAUNCH","smoothLaunch"],
    ["TRACTION CONTROL","tractionControlEnabled"],["ABS","absEnabled"],["TILT","tiltEnabled"]].forEach(([label,key])=>{const button=document.createElement("button");
    button.type="button";button.className="vehicle-gear-button";button.textContent=label;
    button.dataset.drivetrainToggle=key;button.setAttribute("aria-pressed","false");
    button.addEventListener("pointerdown",event=>event.stopPropagation());button.addEventListener("click",event=>{
      event.preventDefault();event.stopPropagation();controlVehicleTransmission({[key]:!vehicleRuntime.transmission[key]});});
    controls.append(button);});
  [["center diff","center"],["front diff","front"],["rear diff","rear"]].forEach(([label,axle])=>{
    const row=div("vehicle-torque-split"),caption=div("",label),select=document.createElement("select"),hint=div("","speed clutch");
    select.className="vehicle-gear-button";select.dataset.differentialMode=axle;
    [["open","OPEN"],["limited-slip","LIMITED SLIP"],["locked","LOCKED"]].forEach(([value,text])=>{
      const option=document.createElement("option");option.value=value;option.textContent=text;select.append(option);});
    select.addEventListener("pointerdown",event=>event.stopPropagation());select.addEventListener("change",event=>{
      event.stopPropagation();controlVehicleTransmission({[`${axle}DiffMode`]:select.value});});
    row.append(caption,select,hint);controls.append(row);
  });
  [["FRONT DIFF BRAKE","frontDifferentialBrake"],["REAR DIFF BRAKE","rearDifferentialBrake"]]
    .forEach(([label,key])=>{const button=document.createElement("button");button.type="button";
      button.className="vehicle-gear-button";button.textContent=label;button.dataset.drivetrainToggle=key;
      button.setAttribute("aria-pressed","false");button.addEventListener("pointerdown",event=>event.stopPropagation());
      button.addEventListener("click",event=>{event.preventDefault();event.stopPropagation();
        controlVehicleTransmission({[key]:!vehicleRuntime.transmission[key]});});controls.append(button);});
  [["FL HOLD","front_left"],["FR HOLD","front_right"],["RL HOLD","rear_left"],["RR HOLD","rear_right"]]
    .forEach(([label,name])=>{const button=document.createElement("button");button.type="button";
      button.className="vehicle-gear-button";button.textContent=label;button.dataset.brakeLock=name;
      button.setAttribute("aria-pressed","false");button.addEventListener("pointerdown",event=>event.stopPropagation());
      button.addEventListener("click",event=>{event.preventDefault();event.stopPropagation();
        controlVehicleTransmission({brakeLock:{name,locked:!vehicleRuntime.brakeLocks[name]}});});controls.append(button);});
  const release=document.createElement("button");release.type="button";release.className="vehicle-gear-button";
  release.textContent="RELEASE BRAKES";release.addEventListener("pointerdown",event=>event.stopPropagation());
  release.addEventListener("click",event=>{event.preventDefault();event.stopPropagation();
    controlVehicleTransmission({releaseAllBrakes:true});});controls.append(release);
  const split=div("vehicle-torque-split"),splitLabel=div("","front / rear"),slider=document.createElement("input"),
    splitValue=div("","F42/R58");slider.type="range";slider.min="5";slider.max="95";slider.step="1";
  slider.value="42";slider.dataset.frontDriveShare="true";splitValue.dataset.torqueSplitValue="true";
  slider.addEventListener("pointerdown",event=>event.stopPropagation());slider.addEventListener("input",event=>{
    event.stopPropagation();controlVehicleTransmission({frontDriveShare:Number(slider.value)/100});});
  split.append(splitLabel,slider,splitValue);controls.append(split);
  const powerUnitRow=div("vehicle-torque-split"),powerUnitLabel=div("","power unit"),
    powerUnitSelect=document.createElement("select"),curveLabel=div("","baked curve");
  powerUnitSelect.dataset.powerUnitPreset="true";powerUnitSelect.className="vehicle-gear-button";
  (model.vehicle_slot?.vehicles?.[0]?.power_unit_presets||[]).forEach(preset=>{const option=document.createElement("option");
    option.value=preset.identity;option.textContent=preset.label;powerUnitSelect.append(option);});
  powerUnitSelect.addEventListener("pointerdown",event=>event.stopPropagation());
  powerUnitSelect.addEventListener("change",event=>{event.stopPropagation();selectVehiclePowerUnit(powerUnitSelect.value);});
  powerUnitRow.append(powerUnitLabel,powerUnitSelect,curveLabel);controls.append(powerUnitRow);
  const equationRow=div("vehicle-torque-split"),equationLabel=div("","engine equations"),equationSelect=document.createElement("select"),
    equationHint=div("","both baked");equationSelect.className="vehicle-gear-button";equationSelect.dataset.engineEquationMode="true";
  for(const [value,label] of [["linear-playable","LINEAR · PLAYABLE"],["symbolic-fidelity","NONLINEAR · FIDELITY"]]){
    const option=document.createElement("option");option.value=value;option.textContent=label;equationSelect.append(option);}
  equationSelect.addEventListener("pointerdown",event=>event.stopPropagation());equationSelect.addEventListener("change",event=>{
    event.stopPropagation();selectVehicleEngineEquationMode(equationSelect.value);});equationRow.append(
      equationLabel,equationSelect,equationHint);controls.append(equationRow);
  const wheelPartRow=div("vehicle-torque-split"),wheelPartLabel=div("","wheel part"),
    wheelPartSelect=document.createElement("select"),wheelPartHint=div("","exclusive");
  wheelPartSelect.dataset.wheelPart="true";wheelPartSelect.className="vehicle-gear-button";
  (model.vehicle_slot?.vehicles?.[0]?.wheel_parts||[]).forEach(part=>{const option=document.createElement("option");
    option.value=part.identity;option.textContent=part.label;wheelPartSelect.append(option);});
  wheelPartSelect.addEventListener("pointerdown",event=>event.stopPropagation());
  wheelPartSelect.addEventListener("change",event=>{event.stopPropagation();selectVehicleWheelPart(wheelPartSelect.value);});
  wheelPartRow.append(wheelPartLabel,wheelPartSelect,wheelPartHint);controls.append(wheelPartRow);
  const clutchRow=div("vehicle-torque-split"),clutchLabel=div("","clutch"),clutchSelect=document.createElement("select"),
    clutchHint=div("","swappable friction pack");clutchSelect.dataset.clutchPreset="true";
  clutchSelect.className="vehicle-gear-button";(model.vehicle_slot?.vehicles?.[0]?.clutch_presets||[]).forEach(preset=>{
    const option=document.createElement("option");option.value=preset.identity;option.textContent=preset.label;clutchSelect.append(option);});
  clutchSelect.addEventListener("pointerdown",event=>event.stopPropagation());clutchSelect.addEventListener("change",event=>{
    event.stopPropagation();selectVehicleClutchPreset(clutchSelect.value);});clutchRow.append(clutchLabel,clutchSelect,clutchHint);controls.append(clutchRow);
  const bodyShellRow=div("vehicle-torque-split"),bodyShellLabel=div("","cosmetic body"),
    bodyShellSelect=document.createElement("select"),bodyShellHint=div("","mounted collider");
  bodyShellSelect.dataset.bodyShell="true";bodyShellSelect.className="vehicle-gear-button";
  (model.vehicle_slot?.vehicles?.[0]?.body_shells||[]).forEach(shell=>{const option=document.createElement("option");
    option.value=shell.identity;option.textContent=shell.label;bodyShellSelect.append(option);});
  bodyShellSelect.addEventListener("pointerdown",event=>event.stopPropagation());bodyShellSelect.addEventListener("change",event=>{
    event.stopPropagation();selectVehicleBodyShell(bodyShellSelect.value);});
  bodyShellRow.append(bodyShellLabel,bodyShellSelect,bodyShellHint);controls.append(bodyShellRow);
  const turretDetails=document.createElement("details"),turretSummary=document.createElement("summary"),
    turretRow=div("vehicle-torque-split"),turretLabel=div("","turret fire takeover"),turretToggle=document.createElement("input"),
    turretReadout=div("","carrier not installed"),outriggerButton=document.createElement("button");
  turretSummary.textContent="WEAPON BODY / FIRE CONTROL";turretSummary.addEventListener("pointerdown",event=>event.stopPropagation());
  turretToggle.type="checkbox";turretToggle.checked=true;turretToggle.dataset.turretFireTakeover="true";
  turretReadout.dataset.turretReadout="true";turretToggle.addEventListener("pointerdown",event=>event.stopPropagation());
  turretToggle.addEventListener("change",event=>{event.stopPropagation();vehicleRuntime.turretSystem.fireTakeover=turretToggle.checked;
    setPlacementStatus(turretToggle.checked?"driver primary fire routed to turret computer":"driver primary fire routed to handheld tool");});
  outriggerButton.type="button";outriggerButton.className="vehicle-gear-button";outriggerButton.textContent="DEPLOY OUTRIGGERS";
  outriggerButton.dataset.outriggerToggle="true";outriggerButton.setAttribute("aria-pressed","false");
  outriggerButton.addEventListener("pointerdown",event=>event.stopPropagation());outriggerButton.addEventListener("click",event=>{
    event.preventDefault();event.stopPropagation();controlVehicleOutriggers(!vehicleRuntime.turretSystem.outriggers.commanded);});
  turretRow.append(turretLabel,turretToggle,turretReadout);turretDetails.append(turretSummary,turretRow,outriggerButton);controls.append(turretDetails);
  for(const [label,items,key] of [["fuel",model.vehicle_slot?.vehicles?.[0]?.fuel_profiles||[],"fuelIdentity"],
      ["ignition timing",model.vehicle_slot?.vehicles?.[0]?.ignition_profiles||[],"ignitionProfileIdentity"]]){
    const row=div("vehicle-torque-split"),caption=div("",label),select=document.createElement("select"),hint=div("","independent");
    select.className="vehicle-gear-button";select.dataset.vehicleChemistry=key;for(const item of items){const option=document.createElement("option");
      option.value=item.identity;option.textContent=item.label;select.append(option);}select.addEventListener("pointerdown",event=>event.stopPropagation());
    select.addEventListener("change",event=>{event.stopPropagation();controlVehicleFuelIgnition({[key]:select.value});});
    row.append(caption,select,hint);controls.append(row);
  }
  const modeRow=div("vehicle-torque-split"),modeLabel=div("","pedal curve"),modeSelect=document.createElement("select"),
    modeHint=div("","solver actuator");modeSelect.className="vehicle-gear-button";modeSelect.dataset.drivingMode="true";
  (model.vehicle_slot?.vehicles?.[0]?.driving_modes||[]).forEach(mode=>{const option=document.createElement("option");
    option.value=mode.identity;option.textContent=mode.label;modeSelect.append(option);});
  modeSelect.addEventListener("pointerdown",event=>event.stopPropagation());modeSelect.addEventListener("change",event=>{
    event.stopPropagation();controlVehicleDriverAssistance({drivingMode:modeSelect.value});});
  modeRow.append(modeLabel,modeSelect,modeHint);controls.append(modeRow);
  const governorRow=div("vehicle-torque-split"),governorLabel=div("","governor"),governor=document.createElement("input"),
    governorValue=div("","6500 rpm");governor.type="range";governor.min="500";governor.max=String(Number(
      model.vehicle_slot?.vehicles?.[0]?.configuration?.powertrain?.redline_rpm||6500));governor.step="50";governor.value=governor.max;
  governor.dataset.governorRpm="true";governorValue.dataset.governorValue="true";
  governor.addEventListener("pointerdown",event=>event.stopPropagation());governor.addEventListener("input",event=>{
    event.stopPropagation();controlVehicleDriverAssistance({governorRpm:Number(governor.value)});});
  governorRow.append(governorLabel,governor,governorValue);controls.append(governorRow);
  const cruiseRow=div("vehicle-torque-split"),cruise=document.createElement("button"),cruiseTarget=document.createElement("input"),
    cruiseValue=div("","0 km/h");cruise.type="button";cruise.className="vehicle-gear-button";cruise.textContent="CRUISE SET";
  cruise.dataset.cruiseToggle="true";cruise.setAttribute("aria-pressed","false");cruiseTarget.type="number";
  cruiseTarget.min="0";cruiseTarget.max="180";cruiseTarget.step="1";cruiseTarget.value="0";
  cruiseTarget.dataset.cruiseTarget="true";cruiseValue.dataset.cruiseValue="true";
  cruise.addEventListener("pointerdown",event=>event.stopPropagation());cruise.addEventListener("click",event=>{
    event.preventDefault();event.stopPropagation();const enable=!vehicleRuntime.driverAssistance.cruiseEnabled,
      speed=Math.hypot(Number(vehicleRuntime.state?.velocity?.[0]||0),Number(vehicleRuntime.state?.velocity?.[2]||0));
    controlVehicleDriverAssistance({cruiseEnabled:enable,cruiseTargetSpeedMps:enable?speed:
      vehicleRuntime.driverAssistance.cruiseTargetSpeedMps});});
  cruiseTarget.addEventListener("pointerdown",event=>event.stopPropagation());cruiseTarget.addEventListener("change",event=>{
    event.stopPropagation();controlVehicleDriverAssistance({cruiseTargetSpeedMps:Math.max(0,Number(cruiseTarget.value))/3.6});});
  cruiseRow.append(cruise,cruiseTarget,cruiseValue);controls.append(cruiseRow);
  const auxiliaryRow=div("vehicle-torque-split"),lights=document.createElement("button"),horn=document.createElement("button"),
    starter=document.createElement("button"),energyReadout=div("","fuel / battery pending");
  for(const [button,label] of [[lights,"LIGHTS · L"],[horn,"HORN · H"],[starter,"START · I"]]){button.type="button";
    button.className="vehicle-gear-button";button.textContent=label;button.addEventListener("pointerdown",event=>event.stopPropagation());}
  lights.dataset.vehicleAuxiliary="headlightsOn";horn.dataset.vehicleAuxiliary="hornOn";starter.dataset.vehicleAuxiliary="starterEngaged";
  lights.addEventListener("click",event=>{event.preventDefault();event.stopPropagation();controlVehicleAuxiliary({
    headlightsOn:!vehicleRuntime.electrical.headlightsOn});updateVehicleTransmissionControls();});
  for(const [eventName,active] of [["pointerdown",true],["pointerup",false],["pointercancel",false],["pointerleave",false]])
    horn.addEventListener(eventName,event=>{event.preventDefault();event.stopPropagation();controlVehicleAuxiliary({hornOn:active});});
  for(const [eventName,active] of [["pointerdown",true],["pointerup",false],["pointercancel",false],["pointerleave",false]])
    starter.addEventListener(eventName,event=>{event.preventDefault();event.stopPropagation();controlVehicleAuxiliary({starterEngaged:active});});
  energyReadout.dataset.vehicleEnergyReadout="true";auxiliaryRow.append(lights,horn,starter,energyReadout);controls.append(auxiliaryRow);
  const transmissionPresetRow=div("vehicle-torque-split"),transmissionPresetLabel=div("","gearset"),
    transmissionPresetSelect=document.createElement("select"),ratioLabel=div("","live ratios");
  transmissionPresetSelect.dataset.transmissionPreset="true";transmissionPresetSelect.className="vehicle-gear-button";
  (model.vehicle_slot?.vehicles?.[0]?.transmission_presets||[]).forEach(preset=>{const option=document.createElement("option");
    option.value=preset.identity;option.textContent=preset.label;transmissionPresetSelect.append(option);});
  transmissionPresetSelect.addEventListener("pointerdown",event=>event.stopPropagation());
  transmissionPresetSelect.addEventListener("change",event=>{event.stopPropagation();
    selectVehicleTransmissionPreset(transmissionPresetSelect.value);});
  transmissionPresetRow.append(transmissionPresetLabel,transmissionPresetSelect,ratioLabel);controls.append(transmissionPresetRow);
  const chassisProfileRow=div("vehicle-torque-split"),chassisProfileLabel=div("","chassis tubes"),
    chassisProfileSelect=document.createElement("select"),chassisPhysicsLabel=div("","mass + limits");
  chassisProfileSelect.dataset.chassisProfile="true";chassisProfileSelect.className="vehicle-gear-button";
  (model.vehicle_slot?.vehicles?.[0]?.chassis_profiles||[]).forEach(profile=>{const option=document.createElement("option");
    option.value=profile.identity;option.textContent=profile.label;chassisProfileSelect.append(option);});
  chassisProfileSelect.addEventListener("pointerdown",event=>event.stopPropagation());
  chassisProfileSelect.addEventListener("change",event=>{event.stopPropagation();
    selectVehicleChassisProfile(chassisProfileSelect.value);});
  chassisProfileRow.append(chassisProfileLabel,chassisProfileSelect,chassisPhysicsLabel);controls.append(chassisProfileRow);
  const geometrySpec=model.vehicle_slot?.vehicles?.[0]?.chassis_geometry_parameters||{},geometryRow=div("vehicle-torque-split"),
    geometryLabel=div("","chassis / wheelbase"),lengthInput=document.createElement("input"),wheelbaseInput=document.createElement("input");
  for(const [input,key,value,range] of [[lengthInput,"chassisLengthM",geometrySpec.chassis_length_m,
      geometrySpec.chassis_length_range_m],[wheelbaseInput,"wheelbaseM",geometrySpec.wheelbase_m,geometrySpec.wheelbase_range_m]]){
    input.type="number";input.step="0.02";input.min=String(range?.[0]||.5);input.max=String(range?.[1]||4);
    input.value=Number(value||1).toFixed(2);input.dataset.chassisGeometry=key;input.addEventListener("pointerdown",event=>event.stopPropagation());
    input.addEventListener("change",event=>{event.stopPropagation();selectVehicleChassisGeometry({
      chassisLengthM:Number(lengthInput.value),wheelbaseM:Number(wheelbaseInput.value)});});}
  geometryRow.append(geometryLabel,lengthInput,wheelbaseInput);controls.append(geometryRow);
  const shockDetails=document.createElement("details"),shockSummary=document.createElement("summary");
  shockSummary.textContent="SHOCK / SPRING PARAMETERS";shockSummary.addEventListener("pointerdown",event=>event.stopPropagation());
  shockDetails.append(shockSummary);
  [["spring N/m","spring_stiffness",5000,180000,500],
   ["compression N·s/m","pneumatic_compression_damping",100,20000,50],
   ["rebound N·s/m","pneumatic_rebound_damping",100,24000,50],
   ["rest length m","suspension_rest_length",.1,.8,.005],
   ["travel m","suspension_travel",.08,.8,.005],
   ["pneumatic efficiency","pneumatic_efficiency",.1,1,.01],
   ["active minimum","active_damping_minimum_scale",.1,1,.01],
   ["active maximum","active_damping_maximum_scale",1,3,.01],
   ["body velocity gain","active_damping_body_velocity_gain_s_per_m",0,4,.01],
   ["rebound release gain","active_damping_rebound_release_gain_s_per_m",0,4,.01]]
    .forEach(([label,name,minimum,maximum,step])=>{const row=div("vehicle-torque-split"),caption=div("",label),
        input=document.createElement("input"),value=div("","");input.type="number";input.min=String(minimum);input.max=String(maximum);
      input.step=String(step);input.value=String((vehicleRuntime.active?.configuration_defaults||
        model.vehicle_slot?.vehicles?.[0]?.configuration_defaults||{})[name]??minimum);input.dataset.shockParameter=name;
      value.textContent=name;input.addEventListener("pointerdown",event=>event.stopPropagation());input.addEventListener("change",event=>{
        event.stopPropagation();const numeric=Math.max(minimum,Math.min(maximum,Number(input.value)));input.value=String(numeric);
        controlVehicleShockParameters({[name]:numeric});});row.append(caption,input,value);shockDetails.append(row);});
  controls.append(shockDetails);
  const alignmentDetails=document.createElement("details"),alignmentSummary=document.createElement("summary"),
    alignmentHeader=div("vehicle-torque-split"),cornerSelect=document.createElement("select"),linked=document.createElement("input"),
    alignmentStatus=div("","static compiled settings");alignmentSummary.textContent="WHEEL ALIGNMENT";
  alignmentSummary.addEventListener("pointerdown",event=>event.stopPropagation());cornerSelect.className="vehicle-gear-button";
  for(const corner of ["front_left","front_right","rear_left","rear_right"]){const option=document.createElement("option");
    option.value=corner;option.textContent=corner.replace("_"," ").toUpperCase();cornerSelect.append(option);}
  linked.type="checkbox";linked.checked=true;linked.title="edit all four wheels in unison";alignmentStatus.dataset.alignmentStatus="true";
  cornerSelect.addEventListener("pointerdown",event=>event.stopPropagation());linked.addEventListener("pointerdown",event=>event.stopPropagation());
  alignmentHeader.append(cornerSelect,linked,alignmentStatus);alignmentDetails.append(alignmentSummary,alignmentHeader);
  const alignmentInputs={};for(const [label,key] of [["camber °","camber_deg"],["caster °","caster_deg"],["toe-in °","toe_deg"]]){
    const row=div("vehicle-torque-split"),caption=div("",label),input=document.createElement("input"),hint=div("","real link trim"),
      range=model.vehicle_slot?.vehicles?.[0]?.wheel_alignment?.ranges?.[key]||[-10,10];input.type="number";input.min=String(range[0]);
    input.max=String(range[1]);input.step="0.05";input.value=String(model.vehicle_slot?.vehicles?.[0]?.wheel_alignment?.corners?.front_left?.[key]||0);
    input.addEventListener("pointerdown",event=>event.stopPropagation());input.addEventListener("change",event=>{event.stopPropagation();
      controlVehicleWheelAlignment({corner:cornerSelect.value,parameter:key,value:Number(input.value),linked:linked.checked});});
    alignmentInputs[key]=input;row.append(caption,input,hint);alignmentDetails.append(row);}
  cornerSelect.addEventListener("change",()=>{const corner=vehicleRuntime.wheelAlignment?.corners?.[cornerSelect.value]||{};
    for(const [key,input] of Object.entries(alignmentInputs))input.value=String(Number(corner[key]||0));});
  const calibrationRow=div("vehicle-wheel-controls");for(const [label,mode] of [["STATIC","static"],["FREE CAL","free-calibrate"],
      ["FULL AUTO","full-time-auto"]]){const button=document.createElement("button");button.type="button";button.className="vehicle-gear-button";
    button.textContent=label;button.addEventListener("pointerdown",event=>event.stopPropagation());button.addEventListener("click",event=>{
      event.preventDefault();event.stopPropagation();controlVehicleWheelAlignment({calibrationMode:mode});});calibrationRow.append(button);}
  alignmentDetails.append(calibrationRow);controls.append(alignmentDetails);
  const steeringRow=div("vehicle-torque-split"),steeringLabel=div("","steering authority"),
    steeringSlider=document.createElement("input"),steeringValue=div("","F50/R50");
  steeringSlider.type="range";steeringSlider.min="0";steeringSlider.max="100";steeringSlider.step="1";steeringSlider.value="50";
  steeringSlider.dataset.steeringFrontShare="true";steeringValue.dataset.steeringShareValue="true";
  steeringSlider.title="front/rear steering authority; center gives full travel to both axles";
  steeringSlider.addEventListener("pointerdown",event=>event.stopPropagation());steeringSlider.addEventListener("input",event=>{
    event.stopPropagation();controlVehicleSteeringSystem({front_share:Number(steeringSlider.value)/100});});
  steeringRow.append(steeringLabel,steeringSlider,steeringValue);controls.append(steeringRow);
  [["FRONT STEER","front"],["REAR STEER","rear"]].forEach(([label,axle])=>{const button=document.createElement("button");
    button.type="button";button.className="vehicle-gear-button active";button.textContent=label;button.dataset.steeringAxle=axle;
    button.setAttribute("aria-pressed","true");button.addEventListener("pointerdown",event=>event.stopPropagation());
    button.addEventListener("click",event=>{event.preventDefault();event.stopPropagation();const key=`${axle}_axle_enabled`;
      controlVehicleSteeringSystem({[key]:vehicleRuntime.steeringSystem?.[key]===false});});controls.append(button);});
  const steeringPhase=document.createElement("button");steeringPhase.type="button";steeringPhase.className="vehicle-gear-button active";
  steeringPhase.textContent="REAR COUNTER";steeringPhase.dataset.steeringRearPhase="true";steeringPhase.setAttribute("aria-pressed","true");
  steeringPhase.addEventListener("pointerdown",event=>event.stopPropagation());steeringPhase.addEventListener("click",event=>{
    event.preventDefault();event.stopPropagation();controlVehicleSteeringSystem({rear_phase:Number(
      vehicleRuntime.steeringSystem?.rear_phase??-1)<0?1:-1});});controls.append(steeringPhase);
  const levelingRow=div("vehicle-torque-split"),levelingButton=document.createElement("button"),
    rideSlider=document.createElement("input"),levelingLabel=div("","slow 2nd-order");
  levelingButton.type="button";levelingButton.className="vehicle-gear-button";levelingButton.textContent="LVL";
  levelingButton.dataset.chassisLevelingToggle="true";levelingButton.addEventListener("pointerdown",event=>event.stopPropagation());
  levelingButton.addEventListener("click",event=>{event.preventDefault();event.stopPropagation();
    controlVehicleChassisLeveling({enabled:!vehicleRuntime.chassisLeveling?.enabled});});
  rideSlider.type="range";rideSlider.min="-80";rideSlider.max="100";rideSlider.step="5";rideSlider.value="0";
  rideSlider.title="preferred chassis ride-height trim, millimetres";rideSlider.addEventListener("pointerdown",event=>event.stopPropagation());
  rideSlider.addEventListener("input",event=>{event.stopPropagation();
    controlVehicleChassisLeveling({enabled:true,mode:"derived-pose",active_pose:"derived",
      target_ride_height_offset_m:Number(rideSlider.value)/1000});});
  levelingRow.append(levelingButton,rideSlider,levelingLabel);controls.append(levelingRow);
  const hydraulicDetails=document.createElement("details"),hydraulicSummary=document.createElement("summary");
  hydraulicSummary.textContent="HYDRAULIC POSES / PNEUMATICS";hydraulicSummary.addEventListener("pointerdown",event=>event.stopPropagation());
  hydraulicDetails.append(hydraulicSummary);
  const poseButtons=div("vehicle-torque-split"),poseLabel=div("","pose presets"),poseGroup=div("vehicle-wheel-controls"),
    poseHint=div("","four corner targets");
  (model.vehicle_slot?.vehicles?.[0]?.chassis_leveling?.pose_presets||[]).forEach(pose=>{const button=document.createElement("button");
    button.type="button";button.className="vehicle-gear-button";button.textContent=pose.label;button.dataset.hydraulicPose=pose.identity;
    button.addEventListener("pointerdown",event=>event.stopPropagation());button.addEventListener("click",event=>{
      event.preventDefault();event.stopPropagation();applyVehicleHydraulicPose(pose.identity);});poseGroup.append(button);});
  poseButtons.append(poseLabel,poseGroup,poseHint);hydraulicDetails.append(poseButtons);
  const programRow=div("vehicle-torque-split"),programLabel=div("","programmable"),programGroup=div("vehicle-wheel-controls"),
    programHint=div("","apply / save");
  for(const slot of model.vehicle_slot?.vehicles?.[0]?.chassis_leveling?.programmable_slots||[]){
    const apply=document.createElement("button"),save=document.createElement("button");for(const button of [apply,save]){
      button.type="button";button.className="vehicle-gear-button";button.addEventListener("pointerdown",event=>event.stopPropagation());}
    apply.textContent=slot;apply.dataset.hydraulicPose=slot;apply.addEventListener("click",event=>{event.preventDefault();event.stopPropagation();
      applyVehicleHydraulicPose(slot);});save.textContent=`SAVE ${slot}`;save.addEventListener("click",event=>{event.preventDefault();
      event.stopPropagation();programVehicleHydraulicPose(slot);});programGroup.append(apply,save);}
  programRow.append(programLabel,programGroup,programHint);hydraulicDetails.append(programRow);
  for(const [corner,label] of [["front_left","FL wheel"],["front_right","FR wheel"],["rear_left","RL wheel"],["rear_right","RR wheel"]]){
    const row=div("vehicle-torque-split"),caption=div("",label),slider=document.createElement("input"),value=div("","-120 / +620 mm");
    slider.type="range";slider.min="-120";slider.max="620";slider.step="2";slider.value="0";slider.dataset.wheelHeight=corner;
    slider.addEventListener("pointerdown",event=>event.stopPropagation());slider.addEventListener("input",event=>{event.stopPropagation();
      controlVehicleChassisLeveling({enabled:true,mode:"manual-wheel",active_pose:"manual",
        manual_corner_targets_m:{[corner]:Number(slider.value)/1000}});});row.append(caption,slider,value);hydraulicDetails.append(row);}
  const rateRow=div("vehicle-torque-split"),rateLabel=div("","pose lerp"),rate=document.createElement("input"),
    rateValue=div("","55 mm/s");rate.type="range";rate.min="5";rate.max="180";rate.step="5";rate.value="55";
  rate.dataset.poseLerpRate="true";rateValue.dataset.poseLerpValue="true";rate.addEventListener("pointerdown",event=>event.stopPropagation());
  rate.addEventListener("input",event=>{event.stopPropagation();controlVehicleChassisLeveling({pose_lerp_rate_m_s:Number(rate.value)/1000});});
  rateRow.append(rateLabel,rate,rateValue);hydraulicDetails.append(rateRow);
  const pressureRow=div("vehicle-torque-split"),pressureLabel=div("","tire pressure"),pressure=document.createElement("input"),
    pressureValue=div("","155 kPa");pressure.type="range";pressure.min="45";pressure.max="260";pressure.step="5";pressure.value="155";
  pressure.dataset.tirePressure="true";pressureValue.dataset.tirePressureValue="true";
  pressure.addEventListener("pointerdown",event=>event.stopPropagation());pressure.addEventListener("input",event=>{
    event.stopPropagation();controlVehicleTirePressure(Number(pressure.value)*1000);});
  pressureRow.append(pressureLabel,pressure,pressureValue);hydraulicDetails.append(pressureRow);controls.append(hydraulicDetails);
  [["preferred roll","target_roll_rad"],["preferred pitch","target_pitch_rad"]].forEach(([label,key])=>{
    const row=div("vehicle-torque-split"),name=div("",label),slider=document.createElement("input"),value=div("","±12°");
    slider.type="range";slider.min="-12";slider.max="12";slider.step="0.5";slider.value="0";
    slider.addEventListener("pointerdown",event=>event.stopPropagation());slider.addEventListener("input",event=>{
      event.stopPropagation();controlVehicleChassisLeveling({enabled:true,mode:"derived-pose",active_pose:"derived",
        [key]:Number(slider.value)*Math.PI/180});});
    row.append(name,slider,value);controls.append(row);});
  const dynoButton=document.createElement("button");dynoButton.type="button";dynoButton.className="vehicle-gear-button";
  dynoButton.textContent="DYNO";dynoButton.addEventListener("pointerdown",event=>event.stopPropagation());
  dynoButton.addEventListener("click",event=>{event.preventDefault();event.stopPropagation();requestVehicleDyno();});
  const dynoReadout=div("vehicle-gear-readout","DYNO idle");dynoReadout.dataset.vehicleDynoReadout="true";
  controls.append(dynoButton,dynoReadout);vehicleRuntime.transmissionControls=controls;return controls;
}

function renderVehicleContactMonitor() {
  const root=div("vehicle-contact-monitor");root.hidden=true;
  root.setAttribute("role","status");root.setAttribute("aria-live","polite");
  const head=div("vehicle-contact-head"),title=div("vehicle-contact-title","springs / patches"),
    transferHud=div("vehicle-transfer-hud","HIGH · C○/F○/R○ · 42/58"),
    details=document.createElement("button"),recover=document.createElement("button"),
    respawn=document.createElement("button");recover.type="button";respawn.type="button";
  details.type="button";details.className="vehicle-detail-button";details.textContent="STATS";
  details.setAttribute("aria-expanded","false");details.addEventListener("pointerdown",event=>event.stopPropagation());
  details.addEventListener("click",event=>{event.preventDefault();event.stopPropagation();const expanded=root.classList.toggle("expanded");
    details.setAttribute("aria-expanded",String(expanded));details.textContent=expanded?"COMPACT":"STATS";updateVehicleContactMonitor();});
  recover.className="vehicle-recover-button";recover.textContent="RIGHT CAR";
  recover.addEventListener("pointerdown",event=>event.stopPropagation());recover.addEventListener("click",event=>{
    event.preventDefault();event.stopPropagation();recoverActiveVehicle();});
  respawn.className="vehicle-recover-button";respawn.textContent="RESPAWN";
  respawn.addEventListener("pointerdown",event=>event.stopPropagation());respawn.addEventListener("click",event=>{
    event.preventDefault();event.stopPropagation();respawnActiveVehicleAtAuthoredPose();});
  head.append(title,transferHud,div("vehicle-contact-mode","not mounted"),details,recover,respawn);root.append(head);
  vehicleRuntime.transferHud=transferHud;
  const massReadout=div("vehicle-mass-readout","mass model pending");massReadout.dataset.vehicleMassReadout="true";root.append(massReadout);
  const grid=div("vehicle-contact-grid"),labels=["FL","FR","RL","RR"];
  labels.forEach((label,index)=>{const corner=div("vehicle-contact-corner");corner.dataset.wheelIndex=String(index);
    const gauge=div("vehicle-spring-gauge"),fill=div("vehicle-spring-fill");gauge.append(fill);
    const patch=div("vehicle-contact-patch"),readout=div("vehicle-contact-label",`${label} airborne`),
      controls=div("vehicle-wheel-controls"),detail=div("vehicle-contact-detail-line","design load pending");
    [["tc","TC 0%"],["abs","ABS 0%"]].forEach(([channel,text])=>{const indicator=div("vehicle-control-indicator"),
      value=document.createElement("span");value.textContent=text;indicator.dataset[`${channel}Indicator`]="true";
      indicator.setAttribute("role","progressbar");indicator.setAttribute("aria-valuemin","0");indicator.setAttribute("aria-valuemax","100");
      indicator.setAttribute("aria-label",`${label} ${channel.toUpperCase()} intervention`);indicator.append(value);controls.append(indicator);});
    corner.append(gauge,patch,readout,controls,detail);grid.append(corner);});
  root.append(grid);
  const ns="http://www.w3.org/2000/svg",structure=document.createElementNS(ns,"svg");
  structure.setAttribute("class","vehicle-chassis-structure");structure.setAttribute("viewBox","0 0 120 64");
  structure.setAttribute("aria-label","force-bearing stick and ball chassis structure");
  const points=[[28,14],[92,14],[28,50],[92,50]],pairs=[[0,1],[1,3],[3,2],[2,0],[0,3],[1,2]];
  pairs.forEach(([a,b])=>{const line=document.createElementNS(ns,"line");line.setAttribute("class","vehicle-chassis-member");
    line.setAttribute("x1",String(points[a][0]));line.setAttribute("y1",String(points[a][1]));
    line.setAttribute("x2",String(points[b][0]));line.setAttribute("y2",String(points[b][1]));structure.append(line);});
  points.forEach(([x,y],index)=>{const node=document.createElementNS(ns,"circle");node.setAttribute("class","vehicle-chassis-node");
    node.setAttribute("cx",String(x));node.setAttribute("cy",String(y));node.setAttribute("r","5");
    node.dataset.wheelIndex=String(index);structure.append(node);});root.append(structure);
  const torqueGraph=div("vehicle-torque-graph");
  [["engine","engineTorque"],["clutch","clutchTorque"],["gearbox","transmissionOutputTorque"],
    ["final drive","drivelineTorque"],["front diff","frontDifferentialTorque"],
    ["rear diff","rearDifferentialTorque"]].forEach(([label,channel])=>{
      const node=div(`vehicle-torque-node${label.includes("diff")?" vehicle-torque-wide":""}`,label),
        value=div("vehicle-torque-value","0 Nm");value.dataset.torqueChannel=channel;node.append(value);torqueGraph.append(node);});
  torqueGraph.append(div("vehicle-reaction-value vehicle-torque-wide","reaction 0 Nm"));root.append(torqueGraph);
  const legend=div("vehicle-contact-legend");[["grip","#54e39b"],["limit","#ffd166"],
    ["slide","#ff5f7d"],["air","#52635d"]].forEach(([label,color])=>{const key=div("vehicle-contact-key",label);
      key.style.setProperty("--key-color",color);legend.append(key);});root.append(legend);
  vehicleRuntime.contactMonitor=root;updateVehicleTransmissionControls();return root;
}

function renderHotbar(hotbar) {
  index.set(hotbar.identity, hotbar);
  index.set(model.inventory.identity, model.inventory);
  model.inventory.items.forEach(item => index.set(item.identity, item));
  const root = div("hotbar");
  root.dataset.nodeId = hotbar.identity;
  const focus = div("focus-mode", controlFocus.mode);
  focus.dataset.focusMode = "true";
  const modeButton=document.createElement("button");modeButton.type="button";
  modeButton.className="tool-mode-control";modeButton.addEventListener("pointerdown",event=>event.stopPropagation());
  modeButton.addEventListener("click",event=>{event.preventDefault();event.stopPropagation();cycleActiveToolMode();});
  toolModeState.button=modeButton;root.append(focus,modeButton);refreshToolModeControl();
  hotbar.slots.forEach(slot => {
    const item = model.inventory.items.find(candidate => candidate.identity === slot.item);
    const element = div(`hotbar-slot${item ? " occupied" : ""}${slot.number === hotbarState.activeSlot ? " active" : ""}`);
    element.dataset.hotbarSlot = String(slot.number);
    element.tabIndex = 0;
    element.setAttribute("role", "button");
    element.setAttribute("aria-label", item ? `Slot ${slot.label}: ${item.name}` : `Empty slot ${slot.label}`);
    const count = div("hotbar-count",
      item && item.maximum_stack > 1 ? String(item.quantity) : "");
    count.dataset.inventoryCount = item?.identity || "";
    element.append(div("hotbar-key", slot.label), div("hotbar-item", item?.name || "—"), count);
    root.append(element);
  });
  return root;
}

function renderPlacementPanel() {
  if (!model.placement) return null;
  index.set(model.placement.identity, model.placement);
  const root = div("placement-panel");
  root.dataset.nodeId = model.placement.identity;
  const stock = div("placement-stock");
  model.placement.recipes.forEach(recipe => {
    index.set(recipe.identity, recipe);
    const item = div("placement-recipe", `${recipe.name} ×${recipe.stock}`);
    item.dataset.placementRecipe = recipe.identity;
    item.tabIndex = 0; item.setAttribute("role", "button");
    stock.append(item);
  });
  const gimbal = div("placement-gimbal");
  [["x", -4, 4, .05], ["y", 0, 2, .05], ["z", -4, 4, .05],
   ["yaw", -180, 180, 5]].forEach(([axis, minimum, maximum, step]) => {
    const label = document.createElement("label"); label.className = "placement-axis";
    label.textContent = axis;
    const input = document.createElement("input"); input.type = "range";
    input.min = String(minimum); input.max = String(maximum); input.step = String(step);
    input.value = "0"; input.dataset.placementAxis = axis;
    label.append(input); gimbal.append(label);
  });
  const snapLabel = document.createElement("label"); snapLabel.className = "placement-axis";
  snapLabel.textContent = "snap";
  const snap = document.createElement("select"); snap.dataset.placementSnap = "true";
  model.placement.snap_modes.forEach(mode => {
    const option = document.createElement("option"); option.value = mode;
    option.textContent = mode; snap.append(option);
  });
  snap.value = "object-face"; placementState.snapMode = snap.value;
  snapLabel.append(snap); gimbal.append(snapLabel);
  const statusLine = div("placement-status",
    "placement idle · choose stock or use slot 2 to pick a focused object");
  statusLine.setAttribute("aria-live", "polite"); placementState.statusElement = statusLine;
  const focusCard = div("placement-focus-card");
  focusCard.dataset.hasFocus = "false";
  const focusLabel = div("placement-focus-label", "No mesh focused · aim or click a map object");
  const focusActions = div("placement-actions");
  const apply = document.createElement("button"); apply.type="button";
  apply.className="placement-action"; apply.dataset.placementAction="apply"; apply.textContent="Apply gimbal";
  const cancel = document.createElement("button"); cancel.type="button";
  cancel.className="placement-action"; cancel.dataset.placementAction="cancel"; cancel.textContent="Cancel";
  focusActions.append(apply,cancel); focusCard.append(focusLabel,focusActions);
  placementState.focusElement = focusCard;
  const defaults = document.createElement("button");
  defaults.type = "button"; defaults.className = "return-defaults";
  defaults.dataset.returnDefaults = "true";
  defaults.textContent = "Return to defaults";
  defaults.setAttribute("aria-label", "Return living map to compiled defaults");
  root.append(stock, gimbal, focusCard, statusLine, defaults);
  return root;
}

function renderShaderViewport() {
  const viewer = model.viewer;
  index.set(viewer.identity, viewer);
  index.set(viewer.camera.identity, viewer.camera);
  const port = makeSelectable(viewer, "viewer-port inactive");
  const head = div("viewer-head");
  const shaderHeading = div("");
  shaderHeading.append(div("kind", "system-root / shader viewport"), div("node-name", viewer.name));
  const shaderSelect = document.createElement("select");
  shaderSelect.className = "viewer-shader-select";
  shaderSelect.setAttribute("aria-label", "Viewport shader");
  VIEWPORT_SHADER_CHOICES.forEach(choice => {
    const option = document.createElement("option");
    option.value = choice.identity; option.textContent = choice.label;
    shaderSelect.append(option);
  });
  shaderSelect.addEventListener("pointerdown", event => event.stopPropagation());
  shaderSelect.addEventListener("click", event => event.stopPropagation());
  shaderSelect.addEventListener("change", event => {
    event.stopPropagation();
    const previous = shaderViewer.shaderChoice;
    try {
      activateViewportShader(shaderSelect.value);
      shaderViewer.readout.textContent = `${shaderViewer.backend} · shader selected`;
    } catch (error) {
      try { activateViewportShader(previous); } catch (_) {}
      shaderSelect.value = previous;
      shaderViewer.readout.textContent = `shader selection failed: ${error.message}`;
      console.error(error);
    }
  });
  shaderHeading.append(shaderSelect);
  const viewerTelemetry=div("viewer-telemetry");
  const viewerReadout=div("viewer-readout", "initializing fragment chain…");
  const celestialStatus=div("celestial-status", "celestial half-dome initializing…");
  viewerTelemetry.append(viewerReadout,celestialStatus);
  head.append(shaderHeading,viewerTelemetry);
  const canvas = document.createElement("canvas");
  canvas.className = "viewer-surface";
  canvas.setAttribute("aria-label", "First-person shader view of the living data map");
  const shaderOnlyToggle = document.createElement("button");
  shaderOnlyToggle.type = "button";
  shaderOnlyToggle.className = "viewer-shader-only-toggle";
  shaderOnlyToggle.textContent = "Shader only";
  shaderOnlyToggle.title = "Hide the page and fill it with the first-person shader";
  shaderOnlyToggle.setAttribute("aria-label", "Toggle full-page shader view");
  shaderOnlyToggle.setAttribute("aria-pressed", "false");
  shaderOnlyToggle.addEventListener("pointerdown", event => event.stopPropagation());
  shaderOnlyToggle.addEventListener("click", event => {
    event.stopPropagation();
    setShaderOnlyMode(!shaderViewer.shaderOnly);
  });
  viewerTelemetry.append(shaderOnlyToggle);
  const musicButton=document.createElement("button");
  musicButton.type="button";musicButton.className="music-room-control";
  musicButton.textContent="Play music room";
  musicButton.title="Play the original loop and analyze it through the embedded C→IR→WebAssembly FFT";
  musicButton.addEventListener("pointerdown",event=>event.stopPropagation());
  musicButton.addEventListener("click",async event=>{
    event.stopPropagation();musicButton.disabled=true;
    try{await toggleMusicRoom();}catch(error){musicButton.textContent=`music unavailable · ${error.message}`;console.error(error);}
    finally{musicButton.disabled=false;}
  });
  musicRoomRuntime.button=musicButton;
  const musicFileInput=document.createElement("input");
  musicFileInput.type="file";musicFileInput.accept="audio/*";musicFileInput.hidden=true;
  musicFileInput.setAttribute("aria-label","Choose a music file for the FFT room");
  const musicLoadButton=document.createElement("button");
  musicLoadButton.type="button";musicLoadButton.className="music-room-control music-file-control";
  musicLoadButton.textContent="Load music file";
  musicLoadButton.title="Choose audio from this device; playback and the embedded FFT stay synchronized";
  musicLoadButton.addEventListener("pointerdown",event=>event.stopPropagation());
  musicLoadButton.addEventListener("click",event=>{event.stopPropagation();musicFileInput.click();});
  musicFileInput.addEventListener("change",async()=>{
    const file=musicFileInput.files?.[0];if(!file)return;
    musicLoadButton.disabled=true;
    try{await loadMusicRoomFile(file);await toggleMusicRoom();}
    catch(error){musicLoadButton.textContent=`load failed · ${error.message}`;console.error(error);}
    finally{musicLoadButton.disabled=false;musicFileInput.value="";}
  });
  musicRoomRuntime.loadButton=musicLoadButton;
  const focusTooltip = div("viewer-focus-tooltip");
  focusTooltip.hidden = true;
  focusTooltip.setAttribute("role", "status");
  focusTooltip.setAttribute("aria-live", "polite");
  const placementOverlay = div("placement-bbox-overlay");
  placementOverlay.hidden = true;
  placementOverlay.append(div("placement-bbox-label"), div("placement-gizmo-x"),
    div("placement-gizmo-z"), div("placement-gizmo-origin"));
  const mobileControls = renderMobileControls(),vehicleContactMonitor=renderVehicleContactMonitor();
  port.append(head, musicButton, musicLoadButton, musicFileInput, canvas, focusTooltip,
    placementOverlay, mobileControls, vehicleContactMonitor);
  if (model.navigation) {
    const navigationStatus = div("navigation-status", "loading navigation assembly…");
    navigationStatus.setAttribute("aria-live", "polite");
    navigationRuntime.statusElement = navigationStatus;
    port.append(navigationStatus);
    navigationRuntime.ready?.then(() => {
      navigationStatus.dataset.plannerThread = "dedicated-worker";
      navigationStatus.textContent =
        `${model.navigation.kernels.length} assembly kernel ready in navigation worker · ` +
        "click the top-down map to auto-locate";
    }).catch(error => navigationStatus.textContent = `navigation unavailable · ${error.message}`);
  }
  if (viewer.device_monitor) port.append(renderDeviceMonitor(viewer.device_monitor));
  if (viewer.dynamics_space) port.append(renderDynamicsSpace(viewer.dynamics_space));
  if (model.hotbar) port.append(renderHotbar(model.hotbar));
  const placementPanel = renderPlacementPanel();
  if (placementPanel) port.append(placementPanel);
  shaderViewer.element = port;
  shaderViewer.canvas = canvas;
  shaderViewer.focusTooltip = focusTooltip;
  shaderViewer.placementOverlay = placementOverlay;
  shaderViewer.readout = viewerReadout;
  shaderViewer.celestialStatus = celestialStatus;
  shaderViewer.shaderSelect = shaderSelect;
  shaderViewer.shaderOnlyToggle = shaderOnlyToggle;
  shaderViewer.telemetry = viewerTelemetry;
  port.addEventListener("click", event => {
    if (event.target.closest("input,select,.placement-panel,.hotbar,.device-monitor,.mobile-controls")) return;
    if (!controlFocus.dialogue) requestViewportControls();
  });
  canvas.addEventListener("pointerdown", event => {
    if(handleVehicleShaderHudPointer(event)){
      event.preventDefault();event.stopPropagation();return;
    }
    if (event.button === 0 && viewportBinding("primary-action", "pointer:button-0")) {
      beginViewportPrimary(null,"pointer");
    }
    if (event.button === 2 && viewportBinding("secondary-action", "pointer:button-2")) {
      event.preventDefault();shaderViewer.active=true;
      const bounds=canvas.getBoundingClientRect();
      const x=viewportControls.pointerLocked?bounds.left+bounds.width/2:event.clientX;
      const y=viewportControls.pointerLocked?bounds.top+bounds.height/2:event.clientY;
      beginViewportSecondary([x,y],"pointer");
    }
  });
  canvas.addEventListener("contextmenu", event => {
    if (viewportBinding("secondary-action", "pointer:button-2")) {
      event.preventDefault();event.stopPropagation();
    }
  });
  port.addEventListener("blur", () => {
    if (document.pointerLockElement !== canvas && controlFocus.mode === "game" &&
        !controlFocus.dialogue) setViewportControlHighlight(false);
  });
  requestAnimationFrame(initializeShaderViewer);
  return port;
}

function renderWorld() {
  index.set(model.identity, model);
  const world = div("world");
  world.append(div("kind", "system root"), div("node-name", model.name));
  world.append(renderShaderViewport());
  const regions = div("regions");
  regions.style.gridTemplateColumns = `repeat(${Math.max(1, 1 + Math.max(0, ...model.regions.map(item => item.position.column)))}, minmax(300px, 1fr))`;
  model.regions.forEach(region => regions.append(renderRegion(region)));
  const entityLayer = document.getElementById("entity-layer");
  if (entityLayer) regions.append(entityLayer);
  world.append(regions);
  shaderViewer.mapElement = regions;
  documentWorldSync.resizeObserver = new ResizeObserver(() => {
    documentWorldSync.dirty = true;
  });
  documentWorldSync.resizeObserver.observe(regions);
  regions.querySelectorAll("[data-node-id]").forEach(element =>
    documentWorldSync.resizeObserver.observe(element));
  window.addEventListener("resize", () => { documentWorldSync.dirty = true; });
  requestAnimationFrame(() => resyncDocumentWorldMap());
  mapRoot.append(world);
}

function renderFilesystemDistrict() {
  const filesystem = model.filesystem;
  if (!filesystem) return;
  index.set(filesystem.identity, filesystem);
  filesystem.nodes.forEach(node => index.set(node.identity, node));
  const district = div("filesystem-district");
  district.append(div("kind", "human artifacts / logical filesystem"),
    div("node-name", "Ownership, paths, and placement remain independent"));
  const contracts = Object.keys(filesystem.backend_contracts).join(" · ");
  district.append(div("filesystem-summary",
    `${filesystem.nodes.length} path nodes · ${filesystem.artifacts.length} artifacts · ${contracts}`));
  const cards = div("filesystem-grid");
  filesystem.artifacts.forEach(artifact => {
    index.set(artifact.identity, artifact);
    const card = makeSelectable(artifact, "artifact-card");
    card.style.setProperty("--artifact-color",
      model.appearance.colors[artifact.palette_role] || model.appearance.colors.active);
    const state = div("attachment-state", artifact.attachment.state);
    state.dataset.attachmentState = artifact.identity;
    card.append(state, div("kind", artifact.kind), div("node-name", artifact.name),
      div("artifact-path", artifact.path),
      div("artifact-relations", `owned by ${artifact.owner}\nplaced in ${artifact.placed_in}`));
    cards.append(card);
  });
  district.append(cards);
  mapRoot.append(district);
}

function semanticSourceClosures() {
  const closures = [];
  model.regions.forEach(region => {
    closures.push({
      identity: region.identity, parent: model.identity, kind: "region", name: region.name,
      opening: `namespace ${region.name} {`
    });
    region.buildings.forEach(building => {
      const buildingKind = building.source_kind || "class";
      closures.push({
        identity: building.identity, parent: region.identity, kind: "building", name: building.name,
        opening: `${buildingKind} ${building.name}:`
      });
      building.rooms.forEach(room => {
        const source = room.implied_code?.[0]?.source || "";
        const opening = source.split("\n").find(line => line.trim()) ||
          `${room.member_kind || "member"} ${room.name}`;
        closures.push({
          identity: room.identity, parent: building.identity, kind: "room", name: room.name,
          opening
        });
      });
    });
  });
  return closures;
}

function playerSourceClosure() {
  const actorIdentity = model.viewer.control_policy?.actor || model.viewer.dynamics_space?.actor;
  const actor = entityState.get(actorIdentity);
  const position = actor?.worldPosition;
  if (!position) return sourceScopeRuntime.closures[0] || null;
  const containing = shaderViewer.geometry.filter(box => {
    if (!sourceScopeRuntime.byIdentity.has(box.identity)) return false;
    return Math.abs(position[0] - box.center[0]) <= box.half_extent[0] &&
      Math.abs(position[2] - box.center[1]) <= box.half_extent[1];
  }).sort((left, right) =>
    (left.half_extent[0] * left.half_extent[1]) -
    (right.half_extent[0] * right.half_extent[1]));
  return sourceScopeRuntime.byIdentity.get(containing[0]?.identity) ||
    sourceScopeRuntime.closures[0] || null;
}

function markSourceClosureEntered(closure) {
  let current = closure;
  while (current) {
    sourceScopeRuntime.visited.add(current.identity);
    current = sourceScopeRuntime.byIdentity.get(current.parent);
  }
}

function renderSourceScopeRows() {
  const grid = sourceScopeRuntime.grid;
  if (!grid) return;
  const currentIdentity = sourceScopeRuntime.current?.identity;
  const visible = sourceScopeRuntime.closures.filter(closure =>
    closure.identity === currentIdentity || !sourceScopeRuntime.visited.has(closure.identity));
  visible.sort((left, right) => {
    if (left.identity === currentIdentity) return -1;
    if (right.identity === currentIdentity) return 1;
    return sourceScopeRuntime.closures.indexOf(left) - sourceScopeRuntime.closures.indexOf(right);
  });
  grid.replaceChildren();
  visible.forEach(closure => {
    const isCurrent = closure.identity === currentIdentity;
    const identity = `source-scope:${closure.identity}`;
    const node = {
      identity, kind: "source-scope-opening", name: closure.name,
      parent: closure.parent, closure_identity: closure.identity,
      interaction: {type: "inspect", destination: identity},
      implied_code: [{
        operation: isCurrent ? "current-scope" : "unentered-scope",
        dialect: closure.kind, source: closure.opening, executable: false,
        explanation: isCurrent
          ? "Opening of the closure currently containing the player."
          : "Opening of a closure the player has not entered yet."
      }]
    };
    const card = makeSelectable(node, "source-line source-scope-opening");
    card.dataset.currentScope = String(isCurrent);
    card.dataset.closureIdentity = closure.identity;
    card.dataset.closureState = isCurrent ? "current" : "unentered";
    card.append(div("line-number", isCurrent ? "▶" : "○"),
      div("line-code", closure.opening),
      div("source-scope-state", isCurrent ? "current scope" : `unentered ${closure.kind}`));
    grid.append(card);
  });
  if (sourceScopeRuntime.statusElement) {
    sourceScopeRuntime.statusElement.textContent =
      `${visible.length} closure openings mounted · ${sourceScopeRuntime.visited.size} entered`;
  }
  status.textContent = `${index.size} inspectable graph objects · ${visible.length} source closures mounted`;
}

function updateSourceScopeForPlayer() {
  if (!sourceScopeRuntime.grid) return;
  const current = playerSourceClosure();
  if (!current || current.identity === sourceScopeRuntime.current?.identity) return;
  sourceScopeRuntime.current = current;
  markSourceClosureEntered(current);
  renderSourceScopeRows();
}

function renderJavaScriptDistrict() {
  const district = div("javascript-district");
  district.append(div("kind", "location-scoped source"), div("node-name", "Source in player scope"));
  const sourceStatus = div("metaphor", "following player containment…");
  sourceStatus.setAttribute("aria-live", "polite");
  const sourceGrid = div("source-grid");
  sourceScopeRuntime.closures = semanticSourceClosures();
  sourceScopeRuntime.byIdentity = new Map(
    sourceScopeRuntime.closures.map(closure => [closure.identity, closure]));
  sourceScopeRuntime.district = district;
  sourceScopeRuntime.grid = sourceGrid;
  sourceScopeRuntime.statusElement = sourceStatus;
  district.append(sourceStatus, sourceGrid);
  mapRoot.append(district);
  updateSourceScopeForPlayer();
}

renderWorld();
renderEntityMezzanine();
renderFilesystemDistrict();
renderJavaScriptDistrict();
inspectNode(model, null);
initializeVehicleFirstExperience();
initializeWorldPhysicsWasm();
armEngineSoundOnFirstGesture();
window.addEventListener("beforeunload", () => {
  stateLoopRuntime.worker?.postMessage({type: "stop"});
  if (stateLoopRuntime.workerUrl) URL.revokeObjectURL(stateLoopRuntime.workerUrl);
  engineSoundRuntime.context?.close?.();
});
requestAnimationFrame(runEntityCycle);
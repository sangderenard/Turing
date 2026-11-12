"""Canonical rigid closed-box metadata and shared host geometric queries.

Basis vectors are world-space columns. Native realization owns solid_shape;
world minimum/maximum are only its conservative broadphase bounds. The packed
layout uses padded vec3 groups so JS and WGSL share explicit scalar offsets.
"""

CLOSED_SOLID_SCHEMA = "oriented-box"
CLOSED_SOLID_STRIDE = 32
CLOSED_SOLID_OFFSETS = {
    "minimum": 0, "maximum": 4, "origin": 8,
    "axis_x": 12, "axis_y": 16, "axis_z": 20,
    "world_minimum": 24, "world_maximum": 28,
}


def emit_closed_solid_javascript() -> str:
    """Emit geometry only, without force laws, ownership filtering or state."""
    import json
    header = (f"const CLOSED_SOLID_STRIDE={CLOSED_SOLID_STRIDE};\n"
              f"const CLOSED_SOLID_OFFSETS={json.dumps(CLOSED_SOLID_OFFSETS)};\n")
    return header + r'''
function closedSolidPack(collider) {
  validateClosedSolid(collider);
  const shape=closedSolidShape(collider),values=Array(CLOSED_SOLID_STRIDE).fill(0),
    groups={minimum:shape.minimum,maximum:shape.maximum,origin:shape.origin,
      axis_x:shape.axes[0],axis_y:shape.axes[1],axis_z:shape.axes[2],
      world_minimum:collider.minimum,world_maximum:collider.maximum};
  for(const [name,vector] of Object.entries(groups))for(let i=0;i<3;i++)values[CLOSED_SOLID_OFFSETS[name]+i]=vector[i];
  return values;
}
function validateClosedSolid(collider) {
  const vector=v=>Array.isArray(v)&&v.length===3&&v.every(Number.isFinite);
  if(!vector(collider.minimum)||!vector(collider.maximum)||collider.minimum.some((v,i)=>v>collider.maximum[i]))
    throw new Error("Invalid closed-solid world bounds: "+collider.identity);
  const shape=collider.solid_shape;if(!shape)return collider;
  if(shape.kind!=="oriented-box"||!vector(shape.origin)||!vector(shape.minimum)||!vector(shape.maximum)||
      shape.minimum.some((v,i)=>v>shape.maximum[i])||!Array.isArray(shape.axes)||shape.axes.length!==3||!shape.axes.every(vector))
    throw new Error("Invalid closed-solid shape: "+collider.identity);
  const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0),a=shape.axes;
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)if(Math.abs(dot(a[i],a[j])-(i===j?1:0))>1e-6)
    throw new Error("Non-orthonormal closed-solid basis: "+collider.identity);
  const cross=[a[0][1]*a[1][2]-a[0][2]*a[1][1],a[0][2]*a[1][0]-a[0][0]*a[1][2],a[0][0]*a[1][1]-a[0][1]*a[1][0]];
  if(dot(cross,a[2])<1-1e-6)throw new Error("Reflected closed-solid basis: "+collider.identity);
  return collider;
}
function closedSolidShape(collider) {
  if(collider.solid_shape?.kind === "oriented-box")return collider.solid_shape;
  return {origin:[0,0,0],minimum:collider.minimum,maximum:collider.maximum,
    axes:[[1,0,0],[0,1,0],[0,0,1]]};
}
function closedSolidLocalVector(collider,vector) {
  return closedSolidShape(collider).axes.map(axis=>axis.reduce((sum,value,i)=>sum+value*vector[i],0));
}
function closedSolidLocalPoint(collider,point) {
  const origin=closedSolidShape(collider).origin;
  return closedSolidLocalVector(collider,point.map((value,i)=>value-origin[i]));
}
function closedSolidWorldVector(collider,vector) {
  const axes=closedSolidShape(collider).axes;
  return [0,1,2].map(i=>axes.reduce((sum,axis,j)=>sum+axis[i]*vector[j],0));
}
function closedSolidWorldPoint(collider,point) {
  const origin=closedSolidShape(collider).origin;
  return closedSolidWorldVector(collider,point).map((value,i)=>value+origin[i]);
}
function closedSolidClosest(collider,worldPoint) {
  const shape=closedSolidShape(collider),p=closedSolidLocalPoint(collider,worldPoint),
    point=p.map((value,i)=>Math.max(shape.minimum[i],Math.min(shape.maximum[i],value))),
    delta=p.map((value,i)=>value-point[i]),length=Math.hypot(...delta);
  let normal=length>1e-6?delta.map(value=>value/length):[0,1,0],signedDistance=length;
  if(length<=1e-6){
    let faceDistance=p[0]-shape.minimum[0];normal=[-1,0,0];point[0]=shape.minimum[0];
    for(const [axis,sign] of [[0,1],[1,-1],[1,1],[2,-1],[2,1]]){
      const distance=sign<0?p[axis]-shape.minimum[axis]:shape.maximum[axis]-p[axis];
      if(distance<faceDistance){faceDistance=distance;normal=[0,0,0];normal[axis]=sign;
        for(let i=0;i<3;i++)point[i]=p[i];point[axis]=sign<0?shape.minimum[axis]:shape.maximum[axis];}
    }signedDistance=-faceDistance;
  }
  return {point:closedSolidWorldPoint(collider,point),normal:closedSolidWorldVector(collider,normal),signedDistance};
}
function closedSolidRay(collider,worldOrigin,worldDirection,minT=0,maxT=Infinity) {
  const shape=closedSolidShape(collider),origin=closedSolidLocalPoint(collider,worldOrigin),
    direction=closedSolidLocalVector(collider,worldDirection);
  let enter=-Infinity,exit=Infinity,enterNormal=null,exitNormal=null;
  for(let axis=0;axis<3;axis++){
    if(Math.abs(direction[axis])<1e-10){if(origin[axis]<shape.minimum[axis]||origin[axis]>shape.maximum[axis])return null;continue;}
    let near=(shape.minimum[axis]-origin[axis])/direction[axis],far=(shape.maximum[axis]-origin[axis])/direction[axis],
      nearNormal=[0,0,0],farNormal=[0,0,0];nearNormal[axis]=-1;farNormal[axis]=1;
    if(near>far){[near,far]=[far,near];[nearNormal,farNormal]=[farNormal,nearNormal];}
    if(near>enter){enter=near;enterNormal=nearNormal;}if(far<exit){exit=far;exitNormal=farNormal;}
    if(enter>exit)return null;
  }
  if(exit<minT||enter>maxT)return null;
  const entering=enter>=minT,fraction=entering?enter:exit,localNormal=entering?enterNormal:exitNormal;
  if(!localNormal||fraction<minT||fraction>maxT)return null;
  return {fraction,entering,point:worldOrigin.map((value,i)=>value+worldDirection[i]*fraction),
    normal:closedSolidWorldVector(collider,localNormal)};
}
function closedSolidSegment(collider,start,finish) {
  const hit=closedSolidRay(collider,start,finish.map((value,i)=>value-start[i]),0,1);
  return hit?.entering?hit:null;
}
function closedSolidVerticalSupport(collider,x,z,queryY,reach) {
  const hit=closedSolidRay(collider,[x,queryY,z],[0,-1,0],0,reach);
  return hit&&hit.normal[1]>0?{...hit,height:hit.point[1]}:null;
}
'''


def emit_closed_solid_wgsl() -> str:
    """Same local-box query contract against the canonical packed buffer.

    The owning geometry shader supplies wall_colliders. Geometry-only output:
    callers keep their own tire/cage radii, support gates and force laws.
    """
    source = r'''
struct ClosedSolidClosest {
  point: vec3<f32>,
  normal: vec3<f32>,
  signed_distance: f32,
};
struct ClosedSolidSegment {
  point: vec3<f32>,
  normal: vec3<f32>,
  fraction: f32,
  valid: u32,
};
fn closed_solid_read(index: u32, offset: u32) -> vec3<f32> {
  let base = index * @@STRIDE@@u + offset;
  return vec3<f32>(wall_colliders[base], wall_colliders[base + 1u], wall_colliders[base + 2u]);
}
fn closed_solid_local_vector(index: u32, v: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(v, closed_solid_read(index, @@AXIS_X@@u)),
    dot(v, closed_solid_read(index, @@AXIS_Y@@u)), dot(v, closed_solid_read(index, @@AXIS_Z@@u)));
}
fn closed_solid_local_point(index: u32, p: vec3<f32>) -> vec3<f32> {
  return closed_solid_local_vector(index, p - closed_solid_read(index, @@ORIGIN@@u));
}
fn closed_solid_world_vector(index: u32, v: vec3<f32>) -> vec3<f32> {
  return closed_solid_read(index, @@AXIS_X@@u) * v.x
    + closed_solid_read(index, @@AXIS_Y@@u) * v.y + closed_solid_read(index, @@AXIS_Z@@u) * v.z;
}
fn closed_solid_world_point(index: u32, p: vec3<f32>) -> vec3<f32> {
  return closed_solid_read(index, @@ORIGIN@@u) + closed_solid_world_vector(index, p);
}
fn closed_solid_closest(world_point: vec3<f32>, index: u32) -> ClosedSolidClosest {
  let minimum = closed_solid_read(index, @@MINIMUM@@u);
  let maximum = closed_solid_read(index, @@MAXIMUM@@u);
  let p = closed_solid_local_point(index, world_point);
  var point = clamp(p, minimum, maximum);
  let delta = p - point; let distance = length(delta);
  var normal = vec3<f32>(0.0f, 1.0f, 0.0f); var signed_distance = distance;
  if (distance > 1.0e-6f) { normal = delta / distance; }
  else {
    var face_distance = p.x - minimum.x;
    normal = vec3<f32>(-1.0f, 0.0f, 0.0f); point.x = minimum.x;
    // Stable face order matches the JS query, including equal-distance ties.
    for (var face = 1u; face < 6u; face += 1u) {
      let axis = face / 2u; let positive = (face % 2u) == 1u;
      let face_value = select(minimum[axis], maximum[axis], positive);
      let candidate_distance = select(p[axis] - face_value, face_value - p[axis], positive);
      if (candidate_distance < face_distance) {
        face_distance = candidate_distance; normal = vec3<f32>(0.0f);
        normal[axis] = select(-1.0f, 1.0f, positive); point = p; point[axis] = face_value;
      }
    }
    signed_distance = -face_distance;
  }
  return ClosedSolidClosest(closed_solid_world_point(index, point),
    closed_solid_world_vector(index, normal), signed_distance);
}
fn closed_solid_segment(start: vec3<f32>, finish: vec3<f32>, index: u32) -> ClosedSolidSegment {
  let origin = closed_solid_local_point(index, start);
  let direction = closed_solid_local_vector(index, finish - start);
  let minimum = closed_solid_read(index, @@MINIMUM@@u);
  let maximum = closed_solid_read(index, @@MAXIMUM@@u);
  var enter = -1.0e30f; var exit = 1.0e30f;
  var normal = vec3<f32>(0.0f);
  for (var axis = 0u; axis < 3u; axis += 1u) {
    if (abs(direction[axis]) < 1.0e-10f) {
      if (origin[axis] < minimum[axis] || origin[axis] > maximum[axis]) {
        return ClosedSolidSegment(start, normal, 0.0f, 0u);
      }
    } else {
      let a = (minimum[axis] - origin[axis]) / direction[axis];
      let b = (maximum[axis] - origin[axis]) / direction[axis];
      let near = min(a, b); let far = max(a, b);
      if (near > enter) {
        enter = near; normal = vec3<f32>(0.0f); normal[axis] = select(-1.0f, 1.0f, a > b);
      }
      exit = min(exit, far);
      if (enter > exit) { return ClosedSolidSegment(start, normal, 0.0f, 0u); }
    }
  }
  if (enter < 0.0f || enter > 1.0f || exit < 0.0f) { return ClosedSolidSegment(start, normal, 0.0f, 0u); }
  return ClosedSolidSegment(start + (finish - start) * enter,
    closed_solid_world_vector(index, normal), enter, 1u);
}
'''
    replacements = {"STRIDE": CLOSED_SOLID_STRIDE, **{key.upper(): value for key, value in CLOSED_SOLID_OFFSETS.items()}}
    for name, value in replacements.items():
        source = source.replace(f"@@{name}@@", str(value))
    assert "@@" not in source
    return source

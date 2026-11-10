// CPU-only geometry checks. No browser, GPU, scene mutation, or server needed.
import assert from 'node:assert/strict';
import {studioRoomBounds, studioRoomFaces} from '../examples/meowdoku/web/marble_room.js';

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-9,
  `${actual} != ${expected}`);
const subtract = (a, b) => a.map((x, i) => x - b[i]);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const dot = (a, b) => a.reduce((sum, x, i) => sum + x*b[i], 0);
const material = (prefix, a, b) => ({wall: 10, floor: 20, ceiling: 30}[prefix]) + ((a+b)&1);
const glassMaterial = 99;

assert.deepEqual(studioRoomBounds(3), studioRoomBounds(3));
assert.deepEqual(studioRoomBounds(1), {
  leftX: -4, rightX: 4, backZ: -3.1, frontZ: 4, floorY: -1.02, ceilingY: -1.02 + 4.02,
});
for (const scale of [NaN, Infinity, -Infinity, 0, 0.999, 8.001, '3', null, {}, true]) {
  assert.throws(() => studioRoomBounds(scale), RangeError);
  assert.throws(() => studioRoomFaces({scale, material}), RangeError);
}
assert.throws(() => studioRoomFaces(), TypeError);
for(const floorDrop of [-1,10.01,NaN,Infinity,'3',null])
  assert.throws(()=>studioRoomBounds(3,floorDrop),RangeError);
assert.throws(() => studioRoomFaces({material, windowOn: 1}), TypeError);
for (const invalid of [undefined, NaN, -1, 0.5, '99'])
  assert.throws(() => studioRoomFaces({material, windowOn: true, glassMaterial: invalid}), RangeError);

for (const scale of [1, 1.25, 2, 3, 4.5, 6, 8]) for (const windowOn of [false, true]) for(const floorDrop of [0,.28,3,9.98,10]) {
  const b = studioRoomBounds(scale,floorDrop), faces = studioRoomFaces({scale, floorDrop, material, windowOn, glassMaterial});
  const width = b.rightX-b.leftX, depth = b.frontZ-b.backZ, height = b.ceilingY-b.floorY;
  const mid = [(b.leftX+b.rightX)/2, (b.floorY+b.ceilingY)/2, (b.backZ+b.frontZ)/2];
  const expected = new Map([
    ['0,0,1', {axis:2, position:b.backZ, area:width*height, prefix:'wall', uv:[0,1]}],
    ['0,0,-1', {axis:2, position:b.frontZ, area:width*height, prefix:'wall', uv:[0,1]}],
    ['1,0,0', {axis:0, position:b.leftX, area:depth*height, prefix:'wall', uv:[2,1]}],
    ['-1,0,0', {axis:0, position:b.rightX, area:depth*height, prefix:'wall', uv:[2,1]}],
    ['0,1,0', {axis:1, position:b.floorY, area:width*depth, prefix:'floor', uv:[0,2]}],
    ['0,-1,0', {axis:1, position:b.ceilingY, area:width*depth, prefix:'ceiling', uv:[0,2]}],
  ]);
  const edges = new Map(), vertices = new Set(), areas = new Map(), counts = new Map();
  const lows = [b.leftX,b.floorY,b.backZ], highs = [b.rightX,b.ceilingY,b.frontZ];
  let glassArea = 0;
  for (const [a, c, d, id, normals] of faces) {
    const n = normals[0], key = n.join(','), surface = expected.get(key);
    assert.ok(surface, 'Every face belongs to one of six inward-facing surfaces');
    assert.deepEqual(normals, [n,n,n]);
    const areaVector = cross(subtract(c,a), subtract(d,a));
    assert.ok(dot(areaVector,n) > 0, 'Winding agrees with the inward normal');
    assert.ok(dot(subtract(mid,a),n) > 0, 'Normal faces the room interior');
    const area = Math.hypot(...areaVector)/2;
    areas.set(key, (areas.get(key)||0)+area);
    counts.set(key, (counts.get(key)||0)+1);
    for (const p of [a,c,d]) {
      assert.equal(p[surface.axis], surface.position);
      p.forEach((x,i) => assert.ok(Number.isFinite(x) && x >= lows[i] && x <= highs[i]));
      vertices.add(p.join(','));
    }
    for (const [p,q] of [[a,c],[c,d],[d,a]]) {
      const pKey=p.join(','), qKey=q.join(',');
      const edgeKey=[pKey,qKey].sort().join('|'), sign=pKey < qKey ? 1 : -1;
      const edge=edges.get(edgeKey)||{count:0, orientation:0};
      edge.count++; edge.orientation+=sign; edges.set(edgeKey,edge);
    }
    if (id === glassMaterial) {
      assert.equal(windowOn,true); assert.equal(key,'0,0,1');
      for (const p of [a,c,d]) {
        assert.ok(p[0]>=-1.2 && p[0]<=1.2 && p[1]>=0.3 && p[1]<=2.3);
      }
      glassArea+=area;
    } else {
      const center=a.map((x,i)=>(x+c[i]+d[i])/3);
      const [u,v]=surface.uv.map(i=>Math.floor(8*(center[i]-lows[i])/(highs[i]-lows[i]))-4);
      assert.equal(id,material(surface.prefix,u,v), 'Subdivisions preserve the eight-cell checker pattern');
    }
  }
  for (const [key, surface] of expected) {
    close(areas.get(key),surface.area);
    if (!windowOn) assert.equal(counts.get(key),8*8*2);
  }
  for (const edge of edges.values()) assert.deepEqual(edge,{count:2,orientation:0},
    'Every edge has exactly two oppositely directed owners; no gaps or T-junctions');
  assert.equal(vertices.size-edges.size+faces.length,2, 'Closed cuboid Euler characteristic');
  close(glassArea,windowOn ? 2.4*2 : 0);
  assert.equal(b.floorY,-1.02-floorDrop);
  if (!windowOn) assert.equal(faces.length,6*8*8*2);
}

// Every base material coordinate is passed as an integer in the original
// -4..3 range, independently of room size and optional geometric subdivisions.
const seen = new Map();
studioRoomFaces({scale:3, material:(prefix,a,b)=>{
  assert.ok(Number.isInteger(a)&&a>=-4&&a<=3);
  assert.ok(Number.isInteger(b)&&b>=-4&&b<=3);
  seen.set(prefix,(seen.get(prefix)||0)+1);
  return 0;
}});
assert.deepEqual(Object.fromEntries(seen),{wall:4*64,floor:64,ceiling:64});
console.log('Studio room: six closed surfaces, inward winding, scaled bounds, fixed exact window and material patterns passed');

assert.deepEqual(studioRoomBounds(),studioRoomBounds(6));

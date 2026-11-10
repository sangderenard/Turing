// Read-only view of MechanicalCreature's existing projectile records, after
// updatePhysicsBalls. No worker ABI decoder, new motion law, or material copy.
import { PACKING, decodeMeta } from './particle_styles.js';
import { rotationMatrix } from './marble_roll.js';

const vector = (value, length, name) => {
  if (!value || value.length !== length || !Array.from(value).every(Number.isFinite))
    throw new TypeError(`${name} must contain ${length} finite numbers`);
  return Array.from(value);
};
const positive = (value, name) => {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${name} must be positive`);
  return value;
};

// Studio's rotateQuaternion uses this same wxyz rotationMatrix helper. The
// caller supplies a fixed authored orientation in world xyz, not law xyz.
// MechanicalCreature's projectile producer publishes no quaternion. In
// particular, its vehicle Euler snapshot columns are not projectile spin.
function authoredRotation(quaternionXYZW) {
  const q = vector(quaternionXYZW, 4, 'fixedQuaternionXYZW');
  if (Math.abs(Math.hypot(...q) - 1) > 8 * Number.EPSILON)
    throw new RangeError('fixedQuaternionXYZW must be unit length; it is not normalized silently');
  return { quaternionXYZW: q, matrix: rotationMatrix([q[3], q[0], q[1], q[2]]) };
}

function modelMatrix(position, radius, center, scale, rotation) {
  // Column-major T(world position) R(authored) S(radius * mesh.scale)
  // T(-mesh.center). Float64 retains original world precision until GL upload.
  const matrix = new Float64Array(16), size = radius * scale;
  for (let column = 0; column < 3; column++)
    for (let row = 0; row < 3; row++) matrix[column * 4 + row] = rotation[row][column] * size;
  for (let row = 0; row < 3; row++)
    matrix[12 + row] = position[row] - center.reduce((sum, v, column) => sum + matrix[column * 4 + row] * v, 0);
  matrix[15] = 1;
  if (!Array.from(matrix).every(Number.isFinite)) throw new RangeError('authored world model exceeds float64 range');
  return matrix;
}

/**
 * Inputs are the actual projectileState and projectilePickupState Maps and
 * full stateLoopRuntime (including slotRecords), plus the
 * already-loaded source {designedMesh, geometry, materials}. All three source
 * resources are returned by reference. Their validation/upload remains owned
 * by the existing Studio loader. This adapter makes no shape-admission claim.
 *
 * fixedQuaternionXYZW is mandatory and explicitly authored. It is NOT an
 * engine orientation and is NOT ParticleStyles' instance-index hash rotation.
 *
 * Optional packing = {map:[cx,cy,worldMetresPerPackedUnit], board:[w,h],
 * planeHeight}. board/planeHeight are in the same metre world as the game.
 * There is no default scale. Without packing, use the per-body model matrices
 * with Studio's normal per-object draw. No original physics data is written.
 * appearanceAssetIdentity optionally selects only that exact explicit shot
 * tag. Omission keeps the original all-projectiles view. The asset loader
 * owns grouping and admission; untagged default shots never match a filter.
 */
export function adaptMechanicalCreatureProjectiles({
  projectileState, projectilePickupState = null, stateLoopRuntime, source, fixedQuaternionXYZW, packing = null,
  appearanceAssetIdentity = null,
}) {
  if (!projectileState || typeof projectileState.entries !== 'function' ||
      !stateLoopRuntime?.bodies || typeof stateLoopRuntime.bodies.get !== 'function')
    throw new TypeError('original projectileState and stateLoopRuntime maps are required');
  if (appearanceAssetIdentity !== null && (typeof appearanceAssetIdentity !== 'string' || !appearanceAssetIdentity))
    throw new TypeError('appearanceAssetIdentity must be an explicit nonempty identity or null');
  const mesh = source?.designedMesh;
  if (!mesh || !source.geometry || !source.materials)
    throw new TypeError('the loaded source designedMesh, geometry and materials are required');
  const center = vector(mesh.center, 3, 'designedMesh.center');
  const scale = positive(mesh.scale, 'designedMesh.scale');
  const rotation = authoredRotation(fixedQuaternionXYZW);
  const instances = [], seen = new Set();
  const maps = [[projectileState, 'projectileState']];
  if (projectilePickupState !== null) {
    if (typeof projectilePickupState.entries !== 'function') throw new TypeError('projectilePickupState must be the original map');
    maps.push([projectilePickupState, 'projectilePickupState']);
  }
  for (const [states, stateSource] of maps) for (const [identity, state] of states.entries()) {
    if (appearanceAssetIdentity !== null && state?.appearanceAssetIdentity !== appearanceAssetIdentity) continue;
    if (seen.has(identity)) throw new Error(`${identity}: belongs to both projectile lifecycle maps`);
    seen.add(identity);
    const pickup = stateSource === 'projectilePickupState';
    if (typeof identity !== 'string' || !identity || state?.identity !== identity ||
        state.box?.identity !== identity || state.record?.identity !== identity)
      throw new Error('projectile identity disagrees with its existing box or record');
    const radiusM = positive(state.record.geometry?.radius, 'projectile record radius');
    if (state.box.radius !== radiusM || state.record.geometry.kind !== 'sphere')
      throw new Error(`${identity}: projectile sphere geometry disagrees with its record`);
    const positionM = vector(state.position, 3, `${identity} position`);
    const velocityMPerS = vector(state.velocity, 3, `${identity} velocity`);
    const sleeping = state.sleeping === true;
    const body = stateLoopRuntime.bodies.get(identity);
    if (body && (body.identity !== identity || !Number.isInteger(body.slot) || body.slot < 0 ||
        !Number.isInteger(body.generation) || body.generation <= 0 ||
        stateLoopRuntime.slotRecords?.[body.slot] !== body))
      throw new Error(`${identity}: physics snapshot membership is stale or inconsistent`);
    if ((sleeping || pickup) && body) throw new Error(`${identity}: sleeping or pickup projectile still owns physics membership`);
    if (!sleeping && !pickup && stateLoopRuntime.ready && !body)
      throw new Error(`${identity}: active projectile has no physics snapshot membership`);
    if (['quaternion', 'orientation'].some(name => name in state || (body && name in body)))
      throw new Error(`${identity}: projectile orientation contract changed; do not discard a newly published pose`);
    instances.push({ identity, stateSource, positionM, velocityMPerS, radiusM, sleeping, pickup,
      appearanceAssetIdentity: state.appearanceAssetIdentity,
      ammunitionItemIdentity: state.ammunitionItemIdentity,
      status: state.record.status,
      membership: body ? { slot: body.slot, generation: body.generation } : null,
      physicsQuaternionXYZW: null,
      fixedQuaternionXYZW: rotation.quaternionXYZW.slice(),
      model: modelMatrix(positionM, radiusM, center, scale, rotation.matrix),
    });
  }
  const result = { instances, identities: instances.map(body => body.identity), n: instances.length,
    snapshotSequence: stateLoopRuntime.latestSequence,
    source, designedMesh: mesh, geometry: source.geometry, materials: source.materials,
    orientation: { physics: 'not-published-by-projectile-physics',
      model: 'explicit-fixed-authored-world-orientation',
      particleStyles: 'unchanged-designed-path-uses-slot-hash-not-authored-model' },
    quaternions: null, colorMode: 'material', packed: null };
  if (packing !== null) result.packed = packInstances(instances, packing);
  return result;
}

function packInstances(instances, packing) {
  const map = vector(packing.map, 3, 'packing.map');
  positive(map[2], 'world metres per packed unit');
  const board = vector(packing.board, 2, 'packing.board'), y = packing.planeHeight;
  if (!Number.isFinite(y)) throw new TypeError('packing.planeHeight must be finite');
  const data = new Float32Array(instances.length * PACKING.stride / 4);
  const words = new Uint32Array(data.buffer), roundtrip = [];
  instances.forEach((body, index) => {
    const offset = index * PACKING.stride / 4, p = body.positionM;
    const radius = body.radiusM / map[2], moving = !body.sleeping && !body.pickup, flag = moving ? 100 : 200;
    // The existing decimal metadata contains radius only in its low two
    // decimal digits. Direct .11 m packing would decode .01 m and another
    // movement flag. A caller must declare a compatible unit transform.
    const meta = Math.fround(flag + radius * 1000), decoded = decodeMeta(meta);
    if (!(radius > 0 && radius < .1) || decoded.moving !== moving || decoded.radius <= 0)
      throw new RangeError(`${body.identity}: radius cannot be represented by the declared PACKING unit transform`);
    data[offset] = map[0] + (p[0] - board[0] / 2) / map[2];
    data[offset + 1] = map[1] + (board[1] / 2 - p[2]) / map[2];
    // instanceCenter adds radius itself; subtract the decoded radius here,
    // so that float32 metadata error cannot add a second vertical error.
    data[offset + 2] = (p[1] - y) / map[2] - decoded.radius;
    words[offset + PACKING.color.offset / 4] = 0xffffffff; // ignored: material's own rows remain authoritative
    data[offset + PACKING.meta.offset / 4] = meta;
    if (![data[offset], data[offset + 1], data[offset + 2]].every(Number.isFinite))
      throw new RangeError(`${body.identity}: world position exceeds the packed float32 range`);
    const positionM = [board[0] / 2 + (data[offset] - map[0]) * map[2],
      y + decoded.radius * map[2] + data[offset + 2] * map[2],
      board[1] / 2 - (data[offset + 1] - map[1]) * map[2]];
    roundtrip.push({ identity: body.identity, radiusM: decoded.radius * map[2], positionM,
      radiusErrorM: decoded.radius * map[2] - body.radiusM,
      positionErrorM: positionM.map((value, axis) => value - p[axis]) });
  });
  return { data, n: instances.length, stride: PACKING.stride, map, board, y,
    worldMetresPerPackedUnit: map[2], roundtrip,
    orientation: 'unchanged-designed-path-uses-slot-hash-not-authored-model' };
}

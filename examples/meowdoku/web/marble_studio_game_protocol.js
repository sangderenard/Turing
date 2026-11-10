// Admission checks around the existing Studio/game bridge, not a transport,
// solver or clock. A visible scene receipt and a worker-applied barrier are
// deliberately different facts.
const fail=(ok,message)=>{if(!ok)throw Error(message);};
const text=value=>typeof value==='string'&&value.length>0;
const count=value=>Number.isSafeInteger(value)&&value>=0;
const freeze=value=>{if(value&&typeof value==='object'){for(const child of Object.values(value))freeze(child);Object.freeze(value);}return value;};

export function createStudioGameProtocol(epoch) {
  fail(text(epoch),'Original game session epoch is required.');
  return freeze({schema:'mechanical-creature-studio-host-v1',snapshotSchema:'mechanical-creature-studio-state-v1',epoch,
    dtAuthority:'original-viewport-slice',
    owners:{environment:'studio',bodies:'original-game-worker',camera:'original-game-controls'},
    capabilities:{workerAppliedBarrier:false,dynamicFixtureBodies:false,rayBodyInsertion:false}});
}

export function validateStudioGameProtocol(value) {
  fail(text(value?.epoch),'Original game session epoch is required.');
  const expected=createStudioGameProtocol(value.epoch);
  for(const key of ['schema','snapshotSchema','dtAuthority'])fail(value[key]===expected[key],`Unsupported Studio game ${key}.`);
  for(const group of ['owners','capabilities'])for(const [key,wanted] of Object.entries(expected[group]))
    fail(value[group]?.[key]===wanted,`Unsupported Studio game ${group}.${key}.`);
  // Keep a detached immutable admitted contract, not a caller-owned descriptor.
  return expected;
}

export function validateStudioGameReceipt(receipt,{protocol,assetIdentity,classIdentity,revision,sequence}) {
  const admitted=validateStudioGameProtocol(protocol),environment=receipt?.environment,building=receipt?.building;
  fail(text(assetIdentity)&&text(classIdentity)&&count(revision)&&Number.isSafeInteger(sequence)&&sequence>0,'Exact requested Studio identity and sequence are required.');
  fail(receipt?.epoch===admitted.epoch,'Studio acknowledgment belongs to a different game session.');
  fail(receipt.state==='visible'&&receipt.identity===assetIdentity&&receipt.sourceRevision===revision&&receipt.sequence===sequence,
    'Studio acknowledgment does not match the requested visible scene.');
  fail(receipt.sourceIdentity===classIdentity&&Number.isSafeInteger(receipt.renderFrame)&&receipt.renderFrame>0,
    'Studio acknowledgment lacks its actual source and completed render frame.');
  fail(text(building?.identity)&&building.assetIdentity===assetIdentity&&Array.isArray(building.placement)&&
    building.placement.length===16&&building.placement.every(Number.isFinite),'Studio acknowledgment lacks its registered building placement.');
  fail(environment?.buildingIdentity===building.identity&&environment.assetIdentity===assetIdentity&&
    environment.revision===revision&&environment.sequence===sequence&&count(environment.geometryRevision),
    'Studio acknowledgment disagrees with its committed environment.');
  fail(['native-scene-ready-worker-pending','posted-to-original-worker'].includes(environment.colliderPublication)&&
    receipt.colliderPublication===environment.colliderPublication,'Studio acknowledgment overstates its collider publication.');
  fail(receipt.physics===admitted.owners.bodies&&receipt.camera===admitted.owners.camera,'Studio acknowledgment changes original game ownership.');
  return freeze(structuredClone(receipt));
}

// Browser-local studio presets. Validation belongs to the studio state owner.
export const STORAGE_KEY = 'meowdoku.studio.lighting-profiles.v1';
export const QUICK_STORAGE_KEY = 'meowdoku.studio.quick-lighting.v1';
export const MAX_PROFILES = 30;
export const MAX_NAME_LENGTH = 80;

function profileName(name) {
  if (typeof name !== 'string' || !name.trim() || name.length > MAX_NAME_LENGTH) {
    throw new Error('Enter a profile name of 1–80 characters.');
  }
  return name.trim();
}

function storageOrThrow(storage) {
  if (!storage || typeof storage.getItem !== 'function' || typeof storage.setItem !== 'function') {
    throw new Error('Browser storage is unavailable. Profiles cannot be saved or loaded.');
  }
  return storage;
}

function checkedState(state, validate) {
  if (typeof validate !== 'function') throw new Error('Lighting validation is unavailable.');
  return validate(state);
}

export function readLightingProfiles(storage, validate) {
  let raw;
  try { raw = storageOrThrow(storage).getItem(STORAGE_KEY); }
  catch (error) { throw new Error(`Cannot read browser profiles: ${error.message}`); }
  if (raw === null) return [];
  let root;
  try { root = JSON.parse(raw); }
  catch { throw new Error('Saved lighting profiles are corrupt. Existing data was preserved.'); }
  if (!root || typeof root !== 'object' || Array.isArray(root) || root.version !== 1) {
    throw new Error('Saved lighting profiles have an unsupported format/version. Existing data was preserved.');
  }
  if (!Array.isArray(root.profiles) || root.profiles.length > MAX_PROFILES) {
    throw new Error('Saved lighting profile list is invalid. Existing data was preserved.');
  }
  const names = new Set();
  return root.profiles.map(profile => {
    if (!profile || typeof profile !== 'object' || Array.isArray(profile)) throw new Error('Invalid saved lighting profile.');
    const name = profileName(profile.name);
    if (name !== profile.name || names.has(name)) throw new Error('Invalid or duplicate saved lighting profile name.');
    names.add(name);
    return {name, state: checkedState(profile.state, validate)};
  });
}

export function saveLightingProfile(storage, name, state, validate) {
  name = profileName(name);
  // Always read first: corrupt or newer-version data must never be overwritten.
  const profiles = readLightingProfiles(storage, validate);
  const profile = {name, state: checkedState(state, validate)};
  const index = profiles.findIndex(item => item.name === name);
  if (index < 0) {
    if (profiles.length >= MAX_PROFILES) throw new Error('Only 30 lighting profiles can be saved. Update an existing name.');
    profiles.push(profile);
  } else profiles[index] = profile;
  // Validate the serialized representation too, before writing (e.g. a validator
  // must not allow values that JSON loses to slip into a supposedly valid save).
  const serialized = JSON.stringify({version: 1, profiles});
  readLightingProfiles({getItem: () => serialized, setItem() {}}, validate);
  try { storageOrThrow(storage).setItem(STORAGE_KEY, serialized); }
  catch (error) { throw new Error(`Profile was not saved: browser storage is unavailable or full. ${error.message}`); }
  return profiles;
}

export function loadLightingProfile(storage, name, validate) {
  const profiles = readLightingProfiles(storage, validate);
  const profile = profiles.find(item => item.name === profileName(name));
  if (!profile) throw new Error('The selected lighting profile no longer exists.');
  return profile.state;
}

export function createLightingProfiles(container, {capture, validate, apply, onChange, storage, onSelectChannel} = {}) {
  const doc = container.ownerDocument;
  const element = doc.createElement('div');
  element.className = 'lighting-profiles';
  const name = doc.createElement('input');
  name.type = 'text'; name.maxLength = MAX_NAME_LENGTH; name.placeholder = 'Lighting profile name';
  name.setAttribute('aria-label', 'Lighting profile name');
  const selection = doc.createElement('select');
  selection.setAttribute('aria-label', 'Saved lighting profiles');
  const save = doc.createElement('button');
  save.type = 'button'; save.textContent = 'Save/update profile';
  const load = doc.createElement('button');
  load.type = 'button'; load.textContent = 'Load selected';
  const status = doc.createElement('span');
  status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
  element.append(name, selection, save, load, status);
  container.append(element);
  let storageError;
  if (storage === undefined) {
    try { storage = globalThis.localStorage; }
    catch (error) { storageError = error; }
  }
  const getStorage = () => {
    if (storageError) throw new Error('Browser storage is unavailable. Profiles cannot be saved or loaded.');
    return storageOrThrow(storage);
  };
  const render = (profiles, selected = selection.value) => {
    selection.replaceChildren();
    for (const profile of profiles) {
      const option = doc.createElement('option');
      option.value = profile.name; option.textContent = profile.name;
      selection.append(option);
    }
    if (profiles.some(profile => profile.name === selected)) selection.value = selected;
    load.disabled = profiles.length === 0;
  };
  function refresh() {
    try {
      const profiles = readLightingProfiles(getStorage(), validate);
      render(profiles);
      status.textContent = profiles.length ? 'Profiles saved in this browser only.' : 'No saved profiles in this browser.';
      return true;
    } catch (error) {
      render([]); status.textContent = error.message; return false;
    }
  }
  selection.addEventListener('change', () => { name.value = selection.value; });
  save.addEventListener('click', () => {
    try {
      const profiles = saveLightingProfile(getStorage(), name.value, capture(), validate);
      render(profiles, name.value.trim());
      status.textContent = `Saved “${name.value.trim()}” in this browser.`;
    } catch (error) { status.textContent = error.message; }
  });
  load.addEventListener('click', () => {
    try {
      const selected = selection.value;
      const state = loadLightingProfile(getStorage(), selected, validate);
      // The owner's apply commits the fully validated state atomically.
      apply(state);
      name.value = selected;
      status.textContent = `Loaded “${selected}”.`;
      if (onChange) onChange(state);
    } catch (error) { status.textContent = `Profile load failed: ${error.message}`; }
  });
  // Reuse named-profile serialization and validation under a separate key.
  const quickStorage = {
    getItem: () => getStorage().getItem(QUICK_STORAGE_KEY),
    setItem: (_, value) => getStorage().setItem(QUICK_STORAGE_KEY, value),
  };
  const quick = doc.createElement('fieldset'); quick.className = 'grid';
  const legend = doc.createElement('legend'); legend.textContent = onSelectChannel ? 'Lighting configurations' : 'Quick lighting A / B / C / D';
  const help = doc.createElement('p'); help.className = 'wide'; help.textContent = onSelectChannel ? 'A drives ray exposure. B follows A with individually unlocked Phong-only overrides. C and D store free configurations in this browser.' : 'Store the current lighting and room in a slot, then tap its letter to recall it. Browser-local, separate from named profiles.';
  quick.append(legend, help);
  const slotButtons = [];
  const refreshSlots = () => {
    const slots = readLightingProfiles(quickStorage, validate);
    slotButtons.forEach(({slot, recall}) => {
      recall.disabled = !slots.some(profile => profile.name === slot);
      recall.title = recall.disabled ? `Slot ${slot} is empty. Use Store ${slot}.` : `Recall lighting ${slot}`;
    });
  };
  if(onSelectChannel) {
    const channelButtons=[];
    for(const [slot,label] of [['A','A (exposure)'],['B','B (room)']]) {
      const button=doc.createElement('button');button.type='button';button.id=`lighting-quick-${slot}`;button.textContent=label;
      button.setAttribute('aria-pressed',String(slot==='A'));
      button.addEventListener('click',()=>{onSelectChannel(slot);for(const [id,b]of channelButtons)b.setAttribute('aria-pressed',String(id===slot));});
      channelButtons.push([slot,button]);quick.append(button);
    }
  }
  for (const slot of (onSelectChannel ? ['C','D'] : ['A', 'B', 'C', 'D'])) {
    const row = doc.createElement('div');
    const recall = doc.createElement('button'); recall.type = 'button'; recall.textContent = slot;
    recall.id = `lighting-quick-${slot}`; recall.setAttribute('aria-label', `Recall lighting ${slot}`);
    const store = doc.createElement('button'); store.type = 'button'; store.textContent = `Store ${slot}`;
    store.id = `lighting-store-${slot}`;
    slotButtons.push({slot, recall}); row.append(recall, store); quick.append(row);
    recall.addEventListener('click', () => {
      try {
        const state = loadLightingProfile(quickStorage, slot, validate);
        apply(state); if (onChange) onChange(state);
        status.textContent = `Recalled lighting ${slot}.`;
      } catch (error) {status.textContent = `Slot ${slot} load failed: ${error.message}`;}
    });
    store.addEventListener('click', () => {
      try {
        saveLightingProfile(quickStorage, slot, capture(), validate); refreshSlots();
        status.textContent = `Stored current lighting in ${slot}.`;
      } catch (error) {status.textContent = `Slot ${slot} was not saved: ${error.message}`;}
    });
  }
  container.append(quick);
  // A/B used to be stored snapshots. Preserve access without rewriting them or
  // confusing those old snapshots with the new channel selectors.
  if(onSelectChannel) {
    try {
      const former=readLightingProfiles(quickStorage,validate).filter(p=>['A','B'].includes(p.name));
      if(former.length) {
        const details=doc.createElement('details'), summary=doc.createElement('summary');summary.textContent='Earlier saved A/B snapshots';details.append(summary);
        for(const profile of former) {
          const button=doc.createElement('button');button.type='button';button.textContent=`Load former ${profile.name} configuration`;
          button.addEventListener('click',()=>{try{const state=loadLightingProfile(quickStorage,profile.name,validate);apply(state);if(onChange)onChange(state);status.textContent=`Loaded former ${profile.name} configuration.`;}catch(e){status.textContent=e.message;}});details.append(button);
        }
        container.append(details);
      }
    }catch { /* Existing refresh reports corrupt/unavailable storage below. */ }
  }
  refresh();
  try {refreshSlots();} catch (error) {slotButtons.forEach(({recall}) => {recall.disabled = true;}); status.textContent = error.message;}
  return {element, refresh};
}


// Room is a derived Phong view of the exposure profile. Each override names a
// real profile leaf; array items with identities use their IDs, never positions.
const ROOM_ROOTS = ['controls','poses','backdrops','table','scale','gobos','fixtureSizes','lightTools','lightStands','shopLights'];
const copyProfile = value => JSON.parse(JSON.stringify(value));
export function roomSettingLeaves(profile) {
  const leaves=[];
  const visit=(value,path) => {
    if (Array.isArray(value)) {
      value.forEach((item,i)=>visit(item,[...path,item && typeof item==='object' && 'id' in item ? '@'+item.id : String(i)]));
    } else if (value && typeof value==='object') {
      for(const [key,item] of Object.entries(value)) if(!['version','id'].includes(key)&&!(profile.fixtureSizes?.underlightFit&&path.join('/')==='fixtureSizes/lights/@underlight'&&['width','height'].includes(key))&&!(path[0]==='shopLights'&&key==='exposureMode')&&!(path[0]==='lightTools'&&['lightId','kind'].includes(key))&&!(profile.lightTools&&path[0]==='fixtureSizes'&&/^(snoot|snootLength|grid|gridCellsX|gridCellsY|gridDepth|gridThickness)$/.test(key))) visit(item,[...path,key]);
    } else if (['string','number','boolean'].includes(typeof value)) leaves.push({path:path.join('/'),value});
  };
  for(const root of ROOM_ROOTS) visit(profile[root],[root]);
  // Optional legacy backdrop fields still have real default geometry and must
  // be individually overridable without mutating the exposure profile shape.
  (profile.backdrops??[]).forEach((layer,i)=>{
    for(const [key,value]of [['shape','rectangle'],['centerY',0]])
      if(layer[key]===undefined)leaves.push({path:`backdrops/${i}/${key}`,value});
  });
  // Legacy scale omits its fixed unit; allow authoring a fixed room unit without
  // changing the radius-linked exposure scale.
  if(!leaves.some(p=>p.path==='scale/unitMm')) leaves.push({path:'scale/unitMm',value:10});
  return leaves;
}
function setRoomLeaf(profile,path,value) {
  const parts=path.split('/');let owner=profile;
  for(const part of parts.slice(0,-1)) {
    owner=Array.isArray(owner)&&part.startsWith('@') ? owner.find(item=>String(item.id)===part.slice(1)) : owner[part];
    if(owner===undefined) return false;
  }
  owner[parts.at(-1)]=value;return true;
}
export function validateRoomOverrides(value, base, validate) {
  if(value===undefined) return {version:1,values:[]};
  if(!value||value.version!==1||!Array.isArray(value.values)||value.values.length>2000) throw Error('Invalid Room overrides');
  const available=new Map(roomSettingLeaves(base).map(item=>[item.path,item.value])), seen=new Set();
  const values=value.values.map(item=>{
    if(!item||typeof item.path!=='string'||(!available.has(item.path)&&!/^gobos\/gobos\/@[1-9][0-9]*\/(enabled|shape|width|height|x|y|z|pitch|yaw|roll|front|back)$/.test(item.path))||seen.has(item.path)) throw Error('Room override no longer names a valid setting');
    seen.add(item.path);
    if((available.has(item.path)&&typeof item.value!==typeof available.get(item.path))||!['string','number','boolean'].includes(typeof item.value)||(typeof item.value==='number'&&!Number.isFinite(item.value))) throw Error('Invalid Room override value');
    return {path:item.path,value:item.value};
  });
  const candidate=copyProfile(base);for(const item of values)setRoomLeaf(candidate,item.path,item.value);
  // A derived Room geometry override is custom, not a claim that the full
  // concentric authoring recipe still matches. Exposure metadata stays intact.
  if(candidate.surfaces && values.some(item=>/^(backdrops\/|table\/|controls\/(room-scale|backdrop-))/.test(item.path)))candidate.surfaces.concentric=false;
  // Existing profile validators are the authority for bounds/materials/geometry.
  validate(candidate);
  return {version:1,values};
}
export function resolveRoomProfile(base, overrides, validate) {
  const candidate=copyProfile(base);
  for(const item of overrides?.values??[])setRoomLeaf(candidate,item.path,item.value);
  if(candidate.surfaces && (overrides?.values??[]).some(item=>/^(backdrops\/|table\/|controls\/(room-scale|backdrop-))/.test(item.path)))candidate.surfaces.concentric=false;
  return validate(candidate);
}

// The original sliders remain the authoritative Studio level. A null exposure
// override means linked, including every profile written before this feature.
export const STUDIO_LIGHTS = [
  {id: 'key-left', label: 'Key left', strengthId: 'light-strength-left'},
  {id: 'key-right', label: 'Key right', strengthId: 'light-strength-right'},
  {id: 'background', label: 'Background', strengthId: 'light-bg-strength'},
  {id: 'kicker-left', label: 'Kicker left', strengthId: 'light-kicker-left-strength'},
  {id: 'kicker-right', label: 'Kicker right', strengthId: 'light-kicker-right-strength'},
  {id: 'fill-lower', label: 'Fill lower', strengthId: 'light-fill-lower-strength'},
  {id: 'fill-upper', label: 'Fill upper', strengthId: 'light-fill-upper-strength'},
  {id: 'table-left', label: 'Table stage left', strengthId: 'light-table-left-strength'},
  {id: 'table-right', label: 'Table stage right', strengthId: 'light-table-right-strength'},
  {id: 'underlight', label: 'Gridded underlight', strengthId: 'light-under-strength'},
];

export function validateLightStrengths(value, fields) {
  if (value === undefined) return {version: 1,
    lights: STUDIO_LIGHTS.map(({id}) => ({id, exposure: null}))};
  if (!value || value.version !== 1 || !Array.isArray(value.lights) || ![7, STUDIO_LIGHTS.length].includes(value.lights.length))
    throw new Error('Invalid Studio/exposure light levels');
  return {version: 1, lights: STUDIO_LIGHTS.map(({id, strengthId}, index) => {
    const light = value.lights[index] ?? (value.lights.length === 7 && index >= 7 ? {id, exposure: null} : null), field = fields[strengthId];
    if (!light || light.id !== id || !field || (light.exposure !== null &&
        (!Number.isFinite(light.exposure) || light.exposure < Number(field.min) / 10 ||
          light.exposure > Number(field.max) / 10))) throw new Error(`Invalid exposure level for ${id}`);
    return {id, exposure: light.exposure};
  })};
}

export function createLightStrengthControls(container, fields, onExposureChange = () => {}, onRoomChange = () => {}) {
  const doc = container.ownerDocument;
  let state = validateLightStrengths(undefined, fields);
  // Solo is a reversible audition mask, never a destructive slider rewrite.
  let solo = null, roomSolo, soloChannel="A";
  const effectiveSolo=channel=>channel==='exposure'?solo:roomSolo===undefined?solo:roomSolo;
  const displayedSolo=()=>effectiveSolo(soloChannel==='B'?'studio':'exposure');
  const soloButtons = [];
  const soloPanel = doc.createElement('div'); soloPanel.className = 'grid';
  const soloHelp = doc.createElement('p');
  soloHelp.textContent = 'Solo one light in both preview and exposure. Click it again, or All lights, to restore the rig. Saved profiles retain the full rig.';
  const allLights = doc.createElement('button'); allLights.type = 'button';
  allLights.id = 'light-solo-clear'; allLights.textContent = 'All lights';
  container.append(soloHelp, soloPanel, allLights);
  const element = doc.createElement('details'); element.id = 'light-strength-levels';
  const add = (tag, text, parent = element) => {
    const node = doc.createElement(tag);
    if (text !== undefined) node.textContent = text;
    parent.appendChild(node); return node;
  };
  const summary = add('summary', 'Separate Studio and exposure light levels');
  add('p', 'Linked by default. Uncheck a light to set its Studio preview and photo exposure independently. The original sliders always set Studio. Linking again uses the current Studio level for both. Live ray view uses exposure levels.');
  const controls = [];
  const ownedIds = new Set(STUDIO_LIGHTS.map(light => light.strengthId));
  const studioValue = index => Number(fields[STUDIO_LIGHTS[index].strengthId].value) / 10;
  const exposureValue = index => state.lights[index].exposure ?? studioValue(index);
  const status = doc.createElement('p'); status.setAttribute('role', 'status');
  const render = () => {
    soloButtons.forEach((button, index) => {
      const active = displayedSolo() === STUDIO_LIGHTS[index].id;
      button.setAttribute('aria-pressed', String(active));
      button.textContent = `${active ? 'Unsolo' : 'Solo'} ${STUDIO_LIGHTS[index].label}`;
    });
    allLights.disabled = soloChannel==='B'?roomSolo===undefined:solo===null;
    controls.forEach(({link, studio, exposure}, index) => {
      link.checked = state.lights[index].exposure === null;
      studio.value = String(studioValue(index));
      exposure.value = String(exposureValue(index));
      exposure.disabled = link.checked;
    });
    const split = state.lights.filter(light => light.exposure !== null).length;
    summary.textContent = split ? `Studio / exposure light levels (${split} unlinked)` : 'Separate Studio and exposure light levels';
  };
  STUDIO_LIGHTS.forEach(({id, label, strengthId}, index) => {
    const source = fields[strengthId];
    const soloButton = doc.createElement('button'); soloButton.type = 'button';
    soloButton.id = `${id}-solo`; soloButton.setAttribute('aria-label', `Solo ${label}`);
    soloButton.addEventListener('click', () => {
      if(soloChannel==='B'){roomSolo=displayedSolo()===id?null:id;render();onRoomChange();}
      else {solo=solo===id?null:id;render();onExposureChange();}
    });
    soloPanel.appendChild(soloButton); soloButtons.push(soloButton); ownedIds.add(soloButton.id);
    const panel = add('fieldset'); panel.className = 'light-strength-level';
    add('legend', label, panel);
    const linkLabel = add('label', undefined, panel); linkLabel.className = 'lighting-pose-lock';
    const link = add('input', undefined, linkLabel); link.type = 'checkbox'; link.id = `${id}-levels-linked`;
    link.setAttribute('aria-label', `${label}: link Studio and exposure`);
    add('span', 'Link Studio and exposure', linkLabel);
    const grid = add('div', undefined, panel); grid.className = 'grid';
    const numeric = channel => {
      const wrapper = add('label', channel === 'studio' ? 'Studio preview' : 'Photo exposure', grid);
      const input = add('input', undefined, wrapper); input.type = 'number'; input.id = `${id}-${channel}-strength`;
      input.min = String(Number(source.min) / 10); input.max = String(Number(source.max) / 10);
      input.step = String(Number(source.step || 1) / 10);
      input.setAttribute('aria-label', `${label} ${channel === 'studio' ? 'Studio preview' : 'photo exposure'} level`);
      input.addEventListener('input', () => {
        if (channel === 'exposure' && state.lights[index].exposure === null) return;
        const value = input.value.trim() === '' ? NaN : Number(input.value);
        if (!Number.isFinite(value) || value < Number(input.min) || value > Number(input.max)) {
          status.textContent = `${label} level must be between ${input.min} and ${input.max}.`; return;
        }
        const before = exposureValue(index);
        if (channel === 'studio') {
          source.value = String(value * 10);
          const output = doc.getElementById(strengthId + '-out');
          if (output) output.value = studioValue(index).toFixed(2);
        } else state.lights[index].exposure = value;
        status.textContent = ''; render();
        if (before !== exposureValue(index)) onExposureChange();
      });
      input.addEventListener('change', () => { status.textContent = ''; render(); });
      ownedIds.add(input.id); return input;
    };
    const studio = numeric('studio'), exposure = numeric('exposure');
    controls.push({link, studio, exposure}); ownedIds.add(link.id);
    link.addEventListener('change', () => {
      const before = exposureValue(index);
      state.lights[index].exposure = link.checked ? null : studioValue(index);
      status.textContent = ''; render();
      if (before !== exposureValue(index)) onExposureChange();
    });
    source.addEventListener('input', () => {
      status.textContent = ''; render();
      if (state.lights[index].exposure === null) onExposureChange();
    });
  });
  allLights.addEventListener('clear-all-solos',()=>{if(solo!==null||roomSolo!==undefined){solo=null;roomSolo=undefined;render();onExposureChange();}});
  ownedIds.add(allLights.id);
  allLights.addEventListener('click', () => {
    if(soloChannel==='B'){if(roomSolo!==undefined){roomSolo=undefined;render();onRoomChange();}}
    else if(solo!==null){solo=null;render();onExposureChange();}
  });
  element.appendChild(status); container.appendChild(element); render();
  return {
    element,
    ownsControl: id => ownedIds.has(id),
    getStrengths: (channel = 'studio') => STUDIO_LIGHTS.map(({id}, i) => effectiveSolo(channel) !== null && effectiveSolo(channel) !== id ? 0 : channel === 'exposure' ? exposureValue(i) : studioValue(i)),
    getSolo: (channel='studio') => effectiveSolo(channel),
    setSoloChannel(channel) {soloChannel=channel;allLights.textContent=channel==='B'?'Restore linked solo':'All lights';soloHelp.textContent=channel==='B'?'Room solo affects Phong only. Restore linked solo to follow exposure again.':'Solo one exposure light; linked Room follows. Click again or All lights to restore. Profiles retain the full rig.';render();},
    getState: () => validateLightStrengths(state, fields),
    setState(value) { state = validateLightStrengths(value, fields); solo = null; roomSolo=undefined; status.textContent = ''; render(); },
  };
}

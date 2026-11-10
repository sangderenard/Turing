/* Imported collection items use the existing inventory, launcher and pickup
 * lifecycle. Only the original ammo slot changes its view when selected.
 * No worker, collision law, body radius, launch speed or tool mode is replaced.
 */

function mechanicalCreatureBallLauncher() {
  return hotbarState.inventory?.items.find(item => item.is_tool &&
    item.properties?.operation === "fire-projectile");
}

function appendMechanicalCreatureAmmoIcon(element, item) {
  const texture = item?.properties?.appearance_icon;
  if (!texture) return;
  const image = document.createElement("img");
  image.className = "hotbar-ammo-icon";
  image.src = texture; image.alt = item.name;
  image.width = 32; image.height = 32;
  image.style.objectFit = "contain"; image.style.pointerEvents = "none";
  element.append(image);
}

function appendMechanicalCreatureAmmoPicker(element, item) {
  if (item?.properties?.operation !== "projectile-ammunition") return;
  const items = globalThis.mechanicalCreatureAmmo.items();
  if (items.length < 2) return;
  element.style.position = "relative";
  const toggle = document.createElement("button"), choices = document.createElement("div");
  toggle.type = "button"; toggle.textContent = "Ammo ▾";
  toggle.setAttribute("aria-label", "Choose ball launcher ammunition");
  toggle.setAttribute("aria-expanded", "false");
  choices.className = "mechanical-creature-ammo-picker"; choices.hidden = true;
  choices.setAttribute("role", "group"); choices.setAttribute("aria-label", "Ball launcher ammunition");
  Object.assign(choices.style, {position: "absolute", bottom: "100%", right: "0", zIndex: "20",
    width: "220px", maxWidth: "70vw", maxHeight: "240px", overflowY: "auto", padding: "6px",
    background: "#13221f", border: "1px solid #739b89", borderRadius: "6px"});
  for (const held of items) {
    const choice = document.createElement("button"); choice.type = "button";
    choice.dataset.ammunitionItem = held.identity;
    choice.setAttribute("aria-pressed", String(held.identity === item.identity));
    Object.assign(choice.style, {display: "flex", alignItems: "center", gap: "6px", width: "100%",
      minHeight: "44px", textAlign: "left"});
    appendMechanicalCreatureAmmoIcon(choice, held);
    const label = document.createElement("span"); label.textContent = held.name;
    const count = document.createElement("span"); count.dataset.inventoryCount = held.identity;
    count.textContent = String(held.quantity); choice.append(label, count);
    choice.addEventListener("click", event => {
      event.preventDefault(); event.stopPropagation(); globalThis.mechanicalCreatureAmmo.select(held.identity);
    });
    choices.append(choice);
  }
  for (const target of [toggle, choices]) target.addEventListener("pointerdown", event => event.stopPropagation());
  choices.addEventListener("click", event => event.stopPropagation());
  toggle.addEventListener("click", event => {
    event.preventDefault(); event.stopPropagation(); choices.hidden = !choices.hidden;
    toggle.setAttribute("aria-expanded", String(!choices.hidden));
  });
  choices.addEventListener("keydown", event => {
    if (event.key === "Escape") {
      event.preventDefault(); event.stopPropagation(); choices.hidden = true;
      toggle.setAttribute("aria-expanded", "false"); toggle.focus();
    }
  });
  element.append(toggle, choices);
}

function refreshMechanicalCreatureAmmoHotbar() {
  // Reuse original construction and delegated click routing, not new controls.
  const current = shaderViewer.element?.querySelector(".hotbar");
  if (current) current.replaceWith(renderHotbar(hotbarState.model));
  refreshInventoryCounts();
}

function tagMechanicalCreatureProjectileAsset(state, ammo) {
  // Newer original games also launch vehicle-turret rounds without inventory.
  if (!ammo) return;
  // Retain the source stack even for default rounds. Collection/vacuum must
  // return each shot to its own stack after the user selects different ammo.
  state.ammunitionItemIdentity = ammo.identity;
  const assetIdentity = ammo.properties?.appearance_asset_identity;
  if (!assetIdentity) return;
  state.appearanceAssetIdentity = assetIdentity;
  state.record.appearance_asset_identity = assetIdentity;
  state.record.ammunition_item_identity = ammo.identity;
  state.box.appearance.asset_identity = assetIdentity;
  state.entity.traits.appearance_asset_identity = assetIdentity;
  state.entity.texture = {...state.entity.texture, kind: "studio-asset",
    reference: assetIdentity, parameters: {...state.entity.texture?.parameters,
      icon: ammo.properties.appearance_icon}};
}

globalThis.mechanicalCreatureAmmo = {
  items() {
    return (hotbarState.inventory?.items || []).filter(item => item.properties?.operation === "projectile-ammunition");
  },
  selected() { return projectileAmmoItem(); },
  addAsset({assetIdentity, name, iconUrl, quantity = 1, maximumStack = 128}) {
    if (typeof assetIdentity !== "string" || !assetIdentity.trim()) throw new TypeError("assetIdentity is required");
    if (typeof name !== "string" || !name.trim()) throw new TypeError("ammo name is required");
    if (typeof iconUrl !== "string" || !iconUrl) throw new TypeError("rendered ammo icon is required");
    if (!Number.isInteger(quantity) || quantity < 0 || !Number.isInteger(maximumStack) || maximumStack < 1 || quantity > maximumStack)
      throw new RangeError("ammo quantity must fit its integer maximum stack");
    const inventory = hotbarState.inventory, launcher = mechanicalCreatureBallLauncher();
    const original = inventory?.items.find(item => item.properties?.operation === "projectile-ammunition" &&
      !item.properties?.appearance_asset_identity);
    if (!original || !launcher) throw new Error("original game ammo and launcher are required");
    const identity = `${inventory.identity}/studio-ammo:${encodeURIComponent(assetIdentity)}`;
    let item = inventory.items.find(candidate => candidate.identity === identity);
    if (item) {
      if (item.properties?.appearance_asset_identity !== assetIdentity) throw new Error("ammo identity collision");
      // Refreshing an asset never refills spent ammunition or changes selection.
      item.name = name; item.properties.appearance_icon = iconUrl;
    } else {
      item = {...original, identity, name, slot: null, quantity, maximum_stack: maximumStack,
        stack_key: assetIdentity, properties: {...original.properties, tool: launcher.entity,
          appearance_asset_identity: assetIdentity, appearance_icon: iconUrl}};
      inventory.items.push(item); index.set(identity, item);
    }
    refreshMechanicalCreatureAmmoHotbar();
    return item;
  },
  select(identity) {
    const inventory = hotbarState.inventory;
    const item = inventory?.items.find(candidate => candidate.identity === identity &&
      candidate.properties?.operation === "projectile-ammunition");
    if (!item) throw new Error("ammo item is not in this inventory");
    const slot = hotbarState.model?.slots.find(candidate => inventory.items.some(held =>
      held.identity === candidate.item && held.properties?.operation === "projectile-ammunition"));
    if (!slot) throw new Error("original ammunition hotbar slot is unavailable");
    const old = inventory.items.find(candidate => candidate.identity === slot.item);
    if (old && old !== item) old.slot = null;
    slot.item = item.identity; item.slot = slot.number;
    inventory.active_ammunition = item.identity;
    refreshMechanicalCreatureAmmoHotbar();
    selectHotbarSlot(slot.number);
    return item;
  },
};

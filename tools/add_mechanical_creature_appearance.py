"""Add an appearance-only draw hook to the exact restored playable page.

No compiler import, model reserialization, worker relink, or publication. The
input hash is the preserved load-dependent-tire game. Narrow draw and inventory
seams import selectable asset ammo; source helpers are retained verbatim.
Studio's actual renderer/assets are installed separately by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAYABLE_SHA256 = "aa6cb6fb897d24e6bf769ca2dbfde3a0d3d0dbcdd64ddb87fd30173bc0f0d5e2"
NEWER_SHA256 = "2cd5c5e23f1491243346dd46586f5ebc10a4391187a4e991fa8f151f8dfd4ef4"
HOOK_SOURCE = ROOT / "examples/meowdoku/web/mechanical_creature_appearance_hook.js"
AMMO_SOURCE = ROOT / "examples/meowdoku/web/mechanical_creature_asset_ammo.js"
DRAW_ORIGINAL = """function drawSceneMeshes(gl){
  gl.bindVertexArray(shaderViewer.vao);gl.drawArrays(gl.TRIANGLES,0,shaderViewer.vertexCount);
  if(shaderViewer.vehicleVao&&shaderViewer.vehicleVertexCount){
    gl.bindVertexArray(shaderViewer.vehicleVao);gl.drawArrays(gl.TRIANGLES,0,shaderViewer.vehicleVertexCount);
  }
}"""
DRAW_REPLACEMENT = """function drawSceneMeshes(gl,hiddenIdentities=null){
  drawMechanicalCreatureBaseAppearance(gl,hiddenIdentities);
  if(shaderViewer.vehicleVao&&shaderViewer.vehicleVertexCount){
    gl.bindVertexArray(shaderViewer.vehicleVao);gl.drawArrays(gl.TRIANGLES,0,shaderViewer.vehicleVertexCount);
  }
}"""
AUTHORED_WORLD_DRAW = """  if (shaderViewer.authoredWorldVao && shaderViewer.authoredWorldSurface) {
    gl.bindVertexArray(shaderViewer.authoredWorldVao);
    gl.drawArrays(gl.TRIANGLES, 0, shaderViewer.authoredWorldSurface.mesh.length / 9);
  }
"""
AUTHORED_DRAW_ORIGINAL = DRAW_ORIGINAL[:-1] + AUTHORED_WORLD_DRAW + "}"
AUTHORED_DRAW_REPLACEMENT = DRAW_REPLACEMENT[:-1] + AUTHORED_WORLD_DRAW + "}"
COLOR_ORIGINAL = """    drawSceneMeshes(gl);
    drawVehicleWheels(gl,width,height,cameraPosition,cameraFacing,celestial);"""
COLOR_REPLACEMENT = """    const importedAppearance=prepareMechanicalCreatureImportedAppearance(
      gl,cameraPosition,cameraFacing,width,height);
    drawSceneMeshes(gl,importedAppearance?.identities);
    drawMechanicalCreatureImportedAppearance(importedAppearance);
    drawVehicleWheels(gl,width,height,cameraPosition,cameraFacing,celestial);"""
AMMO_ORIGINAL = """function projectileAmmoItem() {
  return hotbarState.inventory?.items.find(item =>
    item.properties?.operation === "projectile-ammunition");
}"""
AMMO_REPLACEMENT = """function projectileAmmoItem(state=null) {
  const items=hotbarState.inventory?.items||[];
  const identity=state ? state.ammunitionItemIdentity : hotbarState.inventory?.active_ammunition;
  const selected=identity ? items.find(item=>item.identity===identity&&
    item.properties?.operation==="projectile-ammunition") : null;
  if(state&&identity)return selected||null;
  return selected ||
    items.find(item=>item.properties?.operation==="projectile-ammunition"&&!item.properties?.appearance_asset_identity);
}"""
SELECT_ORIGINAL = """  const routedTool = item?.properties?.tool || (item?.is_tool ? item.entity : null);"""
SELECT_REPLACEMENT = """  const isAmmunition=item?.properties?.operation==="projectile-ammunition";
  if(isAmmunition)hotbarState.inventory.active_ammunition=item.identity;
  const routedTool = item?.properties?.tool || (item?.is_tool ? item.entity : null) ||
    (isAmmunition ? mechanicalCreatureBallLauncher()?.entity : null);"""
TAG_ORIGINAL = """  projectileState.set(identity, state); shaderViewer.geometry.push(box);"""
TAG_REPLACEMENT = """  tagMechanicalCreatureProjectileAsset(state,ammo);
  projectileState.set(identity, state); shaderViewer.geometry.push(box);"""
PICKUP_ORIGINAL = """function collectProjectilePickup(state) {
  const ammo=projectileAmmoItem();"""
PICKUP_REPLACEMENT = """function collectProjectilePickup(state) {
  const ammo=projectileAmmoItem(state);"""
ABSORB_ORIGINAL = """function absorbProjectileIntoAmmo(state,reason="attractor-absorption") {
  const ammo=projectileAmmoItem();"""
ABSORB_REPLACEMENT = """function absorbProjectileIntoAmmo(state,reason="attractor-absorption") {
  const ammo=projectileAmmoItem(state);"""
ICON_ORIGINAL = """    element.append(div("hotbar-key", slot.label), div("hotbar-item", item?.name || "\u2014"), count);"""
ICON_REPLACEMENT = ICON_ORIGINAL + "\n    appendMechanicalCreatureAmmoIcon(element,item);\n    appendMechanicalCreatureAmmoPicker(element,item);"
NEWER_COLOR_ORIGINAL = """    const renderPass=shaderViewer.locations.uRenderPass;
    if(renderPass!==null){
      gl.disable(gl.BLEND);gl.depthMask(true);gl.uniform1i(renderPass,1);drawSceneMeshes(gl);
      gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);
      gl.uniform1i(renderPass,2);drawSceneMeshes(gl);gl.depthMask(true);gl.disable(gl.BLEND);gl.uniform1i(renderPass,0);
    }else drawSceneMeshes(gl);
    drawVehicleWheels(gl,width,height,cameraPosition,cameraFacing,celestial);"""
NEWER_COLOR_REPLACEMENT = """    const importedAppearance=prepareMechanicalCreatureImportedAppearance(
      gl,cameraPosition,cameraFacing,width,height);
    const renderPass=shaderViewer.locations.uRenderPass;
    if(renderPass!==null){
      gl.disable(gl.BLEND);gl.depthMask(true);gl.uniform1i(renderPass,1);drawSceneMeshes(gl,importedAppearance?.identities);
      drawMechanicalCreatureImportedAppearance(importedAppearance,"opaque");
      gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);
      gl.uniform1i(renderPass,2);drawSceneMeshes(gl,importedAppearance?.identities);
      drawMechanicalCreatureImportedAppearance(importedAppearance,"transparent");
      gl.depthMask(true);gl.disable(gl.BLEND);gl.uniform1i(renderPass,0);
    }else{drawSceneMeshes(gl,importedAppearance?.identities);drawMechanicalCreatureImportedAppearance(importedAppearance);}
    drawVehicleWheels(gl,width,height,cameraPosition,cameraFacing,celestial);"""

# The pinned original HTML has CRLF bytes. Match only those original fragments;
# never normalize the saved HTML/model/worker or any untouched behavior bytes.
PATCHES = tuple((before.replace("\n", "\r\n"), after.replace("\n", "\r\n")) for before, after in (
    (DRAW_ORIGINAL, DRAW_REPLACEMENT), (COLOR_ORIGINAL, COLOR_REPLACEMENT),
    (AMMO_ORIGINAL, AMMO_REPLACEMENT), (SELECT_ORIGINAL, SELECT_REPLACEMENT),
    (TAG_ORIGINAL, TAG_REPLACEMENT), (PICKUP_ORIGINAL, PICKUP_REPLACEMENT),
    (ABSORB_ORIGINAL, ABSORB_REPLACEMENT), (ICON_ORIGINAL, ICON_REPLACEMENT),
))


def behavior_bounds(document: str) -> tuple[int, int]:
    model = document.index('id="abstract-ui-model"')
    model_end = document.index("</script>", model)
    start = document.index("<script>", model_end) + len("<script>")
    return start, document.index("</script>", start)


def patches_for_page(behavior: str) -> tuple[tuple[str, str], ...]:
    separator = "\r\n" if "\r\n" in behavior else "\n"
    normalized = behavior.replace("\r\n", "\n")
    pairs = [(before.replace("\r\n", "\n"), after.replace("\r\n", "\n")) for before, after in PATCHES]
    if AUTHORED_DRAW_ORIGINAL in normalized:
        pairs[0] = (AUTHORED_DRAW_ORIGINAL, AUTHORED_DRAW_REPLACEMENT)
    if NEWER_COLOR_ORIGINAL in normalized:
        pairs[1] = (NEWER_COLOR_ORIGINAL, NEWER_COLOR_REPLACEMENT)
    return tuple((before.replace("\n", separator), after.replace("\n", separator)) for before, after in pairs)


def patch_playable_page(source: bytes, hook_source: str | None = None, *, expected_source_sha256: str | None = None) -> tuple[bytes, dict]:
    digest = hashlib.sha256(source).hexdigest()
    if ((expected_source_sha256 is not None and digest != expected_source_sha256.lower()) or
            (expected_source_sha256 is None and digest not in {PLAYABLE_SHA256, NEWER_SHA256})):
        raise ValueError(f"refusing unverified playable page SHA256 {digest}")
    document = source.decode("utf-8")
    start, end = behavior_bounds(document)
    behavior = document[start:end]
    patches = patches_for_page(behavior)
    hook = (HOOK_SOURCE.read_bytes().decode("utf-8") + "\n" + AMMO_SOURCE.read_bytes().decode("utf-8")) if hook_source is None else hook_source
    if "</script" in hook.lower():
        raise ValueError("appearance hook contains an unsafe script terminator")
    for fragment, _replacement in patches:
        if behavior.count(fragment) != 1:
            raise ValueError("playable appearance insertion seam is not unique")
    modified = behavior
    for before, after in patches:
        modified = modified.replace(before, after, 1)
    modified = modified.replace(patches[0][1], hook + "\n" + patches[0][1], 1)
    # This reversibility assertion pins every other byte, including all input,
    # HUD, physics, time, picking, depth and shadow behavior to the source page.
    recovered = modified.replace(hook + "\n" + patches[0][1], patches[0][1], 1)
    for before, after in reversed(patches):
        recovered = recovered.replace(after, before, 1)
    if recovered != behavior:
        raise AssertionError("appearance patch changed unrelated game behavior")
    output = (document[:start] + modified + document[end:]).encode("utf-8")
    return output, {
        "schema": "mechanical-creature-appearance-hook-v1",
        "source_sha256": digest,
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "hook_sha256": hashlib.sha256(hook.encode("utf-8")).hexdigest(),
        "original_behavior_sha256": hashlib.sha256(behavior.encode("utf-8")).hexdigest(),
        "original_behavior_recovered_exactly": True,
        "unchanged_model_worker_wasm": True,
        "unchanged_depth_shadow_picking_colliders": True,
        "compile_performed": False,
        "published": False,
        "appearance_adapter_included": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    protected = {source, (ROOT / "build/meowdoku/web/mechanical_creature_play.html").resolve(),
                 (ROOT / "build/dot-living-map-runtime-20261110/matched/index.html").resolve(),
                 (ROOT / "build/meowdoku/web/mechanical_creature_studio.html").resolve(),
                 (ROOT / "build/dot-playable-tires-20261110/index.html").resolve()}
    if output in protected:
        parser.error("output must be a separate candidate; refusing to overwrite the playable game")
    candidate, receipt = patch_playable_page(source.read_bytes(), expected_source_sha256=args.expected_source_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create also refuses to overwrite another worker's candidate.
    with output.open("xb") as stream:
        stream.write(candidate)
    if args.receipt:
        with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
    print(json.dumps({"output": str(output), **receipt}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

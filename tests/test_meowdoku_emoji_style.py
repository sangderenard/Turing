"""The particle field's "emoji" style (examples/meowdoku/web/particle_styles.js): the pure parts, run by node -- the atlas layout
and UVs, the assignment hash, the fallback choice -- and the atlas the export bakes (examples/meowdoku/web_export.py)."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "examples" / "meowdoku" / "web"

JS_EMOJI = """
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const [dir] = process.argv.slice(2);
const P = await import(pathToFileURL(`${dir}/particle_styles.js`).href);
const full = {floatRender: 'float', maxPointSize: 256}, none = {floatRender: null, maxPointSize: 256};

// ---- registry: both variants are page styles (the server's list is untouched), after the existing ones
assert.ok(P.STYLE_IDS.includes('emoji') && P.STYLE_IDS.includes('emoji_mesh'));
assert.ok(!P.STYLES.some((s) => s.id.startsWith('emoji')));
assert.deepEqual(P.EMOJI_ASSIGN.map(([id]) => id), ['id', 'species', 'size']);

// ---- the assignment hash: a stable mix of the slot (known values pin the GLSL twin's constants too)
assert.equal(P.emojiHash(0), 0);
assert.equal(P.emojiHash(1), 0x688990c0);
assert.equal(P.emojiHash(2), 0xd1132181);
assert.equal(P.emojiHash(0xffffffff), P.emojiHash(-1));
assert.equal(P.emojiHash(7), P.emojiHash(7));
const K = 13, counts = new Array(K).fill(0);
for (let i = 0; i < 13000; i++) counts[P.emojiKind('id', i, 0.01, K, [], [0, 1])]++;
assert.ok(Math.min(...counts) > 800 && Math.max(...counts) < 1200, counts.join());       // an even mix, every kind present
// stable: a slot's kind depends on the slot alone, not on the radius or the other grains
for (let i = 0; i < 200; i++) assert.equal(P.emojiKind('id', i, 0.009, K, [], [0, 1]), P.emojiKind('id', i, 0.014, K, [], [0, 1]));

// ---- by species and by size (species from the radius, spread over the kinds; size from the ramp)
const table = P.speciesTable(null), bands = P.speciesBands(table);
assert.equal(P.speciesCountOf(bands), 4);
const bySpecies = table.map((s) => P.emojiKind('species', 5, s.radius, K, bands, [0, 1]));
assert.deepEqual(bySpecies, [1, 4, 8, 11]);
assert.equal(new Set(bySpecies).size, 4); assert.deepEqual([...bySpecies].sort((a, b) => a - b), bySpecies);
const range = [0.0081, 1 / (0.0147 - 0.0081)];
let last = -1;
for (let r = 0.0081; r <= 0.0147; r += 0.0001) { const k = P.emojiKind('size', 0, r, K, bands, range); assert.ok(k >= last && k < K); last = k; }
assert.equal(P.emojiKind('size', 0, 0.0001, K, bands, range), 0); assert.equal(P.emojiKind('size', 0, 1, K, bands, range), K - 1);

// ---- the atlas: cells and UVs (row 0 is the image's top, like gl_PointCoord)
const meta = {cell: 128, cols: 4, rows: 4, ink: 0.875, kinds: Array.from({length: 13}, (_, i) => ({name: `k${i}`, mesh: `emoji_k${i}`, color: [0.5, 0.5, 0.5]}))};
assert.equal(P.checkAtlasMeta(meta), '');
assert.deepEqual(P.atlasRect(0, meta), [0, 0, 0.25, 0.25]);
assert.deepEqual(P.atlasRect(5, meta), [0.25, 0.25, 0.25, 0.25]);
assert.deepEqual(P.atlasRect(12, meta), [0, 0.75, 0.25, 0.25]);
const rects = P.atlasRects(meta);
assert.equal(rects.length, P.MAX_EMOJI * 4);
for (let i = 0; i < 13; i++) { const [u, v, du, dv] = P.atlasRect(i, meta); assert.ok(u >= 0 && v >= 0 && u + du <= 1 && v + dv <= 1); assert.deepEqual(Array.from(rects.slice(i * 4, i * 4 + 4)), P.atlasRect(i, meta)); }
const cells = new Set(Array.from({length: 13}, (_, i) => P.atlasRect(i, meta).join())); assert.equal(cells.size, 13);       // no two kinds share a cell
assert.match(P.checkAtlasMeta({kinds: []}), /no kinds/);
assert.match(P.checkAtlasMeta({cols: 2, rows: 2, kinds: Array(5).fill({})}), /grid/);
assert.match(P.checkAtlasMeta({cols: 8, rows: 8, kinds: Array(P.MAX_EMOJI + 1).fill({})}), /holds/);
assert.equal(P.packRgba8([1, 0.5, 0]) >>> 24, 255); assert.equal(P.packRgba8([1, 0.5, 0]) & 0xff, 255); assert.equal((P.packRgba8([1, 0.5, 0]) >> 8) & 0xff, 128);
assert.equal(P.emojiNative([0, 0, 0, 0], {mode: 0}), true);        // simulation colour, native palette: the emoji's own colours
assert.equal(P.emojiNative([1, 0, 0, 1], {mode: 0}), false);       // a solid / material colour tints
assert.equal(P.emojiNative([0, 0, 0, 0], {mode: 4}), false);       // a palette tints
assert.equal(P.emojiNative(null, null), true);

// ---- fallbacks: both variants say why they are points
for (const id of ['emoji', 'emoji_mesh']) {
  assert.equal(P.chooseStyle(id, full).style, id);                                    // (no scene: nothing to refuse)
  const c = P.chooseStyle(id, none); assert.equal(c.style, 'points'); assert.match(c.reason, /float/);
  assert.equal(P.chooseStyle(id, {floatRender: 'float', maxPointSize: 1}).style, id === 'emoji' ? 'points' : id);     // (the meshes need no point sprites)
  assert.equal(P.chooseStyle(id, full, {[id]: 'atlas did not load'}).style, id === 'emoji' ? 'points' : 'emoji');
}
assert.match(P.chooseStyle('emoji', full, {emoji: 'atlas did not load'}).reason, /atlas did not load/);
// too small on screen: the image variant is points; the mesh variant is points too (its fallback, the image variant, is as small)
const speck = {ready: true, count: 5000, radiusPx: 0.9};
let c = P.chooseStyle('emoji', full, {}, speck); assert.equal(c.style, 'points'); assert.match(c.reason, /too small/);
c = P.chooseStyle('emoji_mesh', full, {}, {...speck, radiusPx: 0.3}); assert.equal(c.style, 'points'); assert.match(c.reason, /emoji_mesh unavailable/);
c = P.chooseStyle('emoji_mesh', full, {}, speck); assert.deepEqual([c.style, c.lod], ['emoji_mesh', 1]);      // a 0.9 px mesh is still drawn (the material style's own floor is 0.4 px)
c = P.chooseStyle('emoji', full, {}, {ready: false, count: 5000, radiusPx: 5}); assert.equal(c.style, 'emoji');        // the sprite variant needs no scene shader
// the mesh variant: needs the scene shader; falls back to the image variant, which draws when the grains are big enough
c = P.chooseStyle('emoji_mesh', full, {}, {ready: false, count: 100, radiusPx: 6}); assert.equal(c.style, 'emoji'); assert.match(c.reason, /scene's material shader/);
// budget: the full mesh while it fits and the grains are big; its front cap when they are small or many; the image variant beyond that
c = P.chooseStyle('emoji_mesh', full, {}, {ready: true, count: 1000, radiusPx: 8}); assert.deepEqual([c.style, c.lod], ['emoji_mesh', 0]);
c = P.chooseStyle('emoji_mesh', full, {}, {ready: true, count: 1000, radiusPx: 2}); assert.deepEqual([c.style, c.lod], ['emoji_mesh', 1]);
c = P.chooseStyle('emoji_mesh', full, {}, {ready: true, count: 8000, radiusPx: 8}); assert.deepEqual([c.style, c.lod], ['emoji_mesh', 1]);
c = P.chooseStyle('emoji_mesh', full, {}, {ready: true, count: 65536, radiusPx: 8}); assert.equal(c.style, 'emoji'); assert.match(c.reason, /frame holds|meshes hold/);
c = P.chooseStyle('emoji_mesh', full, {}, {ready: true, count: 200000, radiusPx: 8}); assert.equal(c.style, 'emoji'); assert.match(c.reason, /meshes hold/);
c = P.chooseStyle('emoji_mesh', full, {}, {ready: true, count: 1000, radiusPx: 8, tris: [4000, 1000]}); assert.equal(c.lod, 0 + 1 * (1000 * 4000 > 2.6e6));   // the loaded meshes' own triangle counts are used
const plan = P.emojiMeshPlan({ready: true, count: 65536, radiusPx: 8, force: true}); assert.equal(plan.ok, true); assert.equal(plan.lod, 0);              // forced: the budget is ignored (the grain size still picks the mesh)
assert.equal(P.emojiMeshPlan({ready: true, count: 65536, radiusPx: 2, force: true}).lod, 1);
assert.equal(P.emojiSpritePlan({radiusPx: 1.2}).ok, false); assert.equal(P.emojiSpritePlan({radiusPx: 1.2, force: true}).ok, true); assert.equal(P.emojiSpritePlan({radiusPx: 6}).ok, true);

// ---- the instanced shader's emoji additions keep the original colour path and the sphere path
const vert = 'layout(location = 0) in vec3 aPos;layout(location = 1) in vec3 aNorm;uniform mat4 uMV;uniform mat4 uMVP;out vec3 vPosV;out vec3 vNormV;out vec3 vPosObj;\\nvoid main() { vPosV = vec3(0.0); }';
const frag = 'out vec4 FragColor;\\nvoid main() { int id = 0; vec3 albedo = mat_albedo(id); vec3 inner = ph_inner(id); FragColor = vec4(albedo, 1.0); }';
const ext = P.extendForInstancing(vert, frag);
assert.match(ext.vert, /uInstRot/); assert.match(ext.vert, /aInstId/); assert.match(ext.vert, /main_scene\\(\\)/);
assert.ok(ext.vert.indexOf('if (!uInstanced) return;') < ext.vert.indexOf('uInstRot > 0.5'));     // pieces never reach the emoji rotation
console.log('ok');
"""


def _node(tmp_path, script):
    shutil.copyfile(WEB / "particle_styles.js", tmp_path / "particle_styles.js")
    shutil.copyfile(WEB / "world_curve.js", tmp_path / "world_curve.js")
    shutil.copyfile(WEB / "world_zoom.js", tmp_path / "world_zoom.js")
    (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
    done = subprocess.run(["node", str(tmp_path / "run.mjs"), str(tmp_path)], capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout + done.stderr)[-2500:]
    return done.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node runs the page's particle style code")
def test_the_emoji_style_assignment_atlas_layout_and_fallbacks(tmp_path):
    assert _node(tmp_path, JS_EMOJI).strip().endswith("ok")


def test_the_export_bakes_one_atlas_cell_per_animal_from_the_emoji_font(tmp_path):
    """web_export.bake_emoji_atlas: every kind has an opaque glyph inside its own cell, margins stay transparent, the colour of
    the ink spreads into the transparent texels (no black fringe in the mips), and kinds name their exported meshes."""
    pytest.importorskip("PIL")
    import numpy as np
    sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
    from meowdoku import web_export as we
    if not Path(we.EMOJI_FONT).exists():
        pytest.skip("Segoe UI Emoji is not installed here")
    kinds = we.emoji_kinds()
    names = [k["name"] for k in kinds]
    assert len(set(names)) == len(names) <= 16
    assert {"frog", "cat", "fox", "owl", "panda", "rabbit", "bear", "dog", "snail", "bat", "wolf"} <= set(names)
    assert all(k["mesh"] == f"emoji_{k['name']}" for k in kinds)
    atlas, meta = we.bake_emoji_atlas(kinds[:5], cell=64, cols=3)
    assert atlas.shape == (128, 192, 4) and meta["rows"] == 2 and meta["size"] == [192, 128]
    for index in range(5):
        r, c = divmod(index, 3)
        cell = atlas[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64]
        ys, xs = np.nonzero(cell[..., 3] > 127)
        assert len(ys) > 200, "the glyph is drawn"
        margin = round((1 - meta["ink"]) * 64 / 2) - 1
        assert xs.min() >= margin and xs.max() < 64 - margin and ys.min() >= margin and ys.max() < 64 - margin, "ink stays inside its margin"
        assert max(xs.max() - xs.min(), ys.max() - ys.min()) + 1 >= meta["ink"] * 64 - 2, "its longer side fills the ink share"
        opaque = cell[..., 3] > 0
        beside = ~opaque & (np.roll(opaque, 1, 0) | np.roll(opaque, -1, 0) | np.roll(opaque, 1, 1) | np.roll(opaque, -1, 1))
        assert beside.any() and cell[beside][:, :3].any(axis=1).mean() > 0.95, "texels beside the ink are transparent but carry its colour, not black"
    assert not atlas[64:, 128:].any(), "unused cells stay empty"

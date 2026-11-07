// Node check of web/wordle_layout.js: the rack's rows centred by their own key count, the entry bar centred over the board, and the
// hit tests agreeing with the tile centres exactly.
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../examples/meowdoku/web");
const W = await import(pathToFileURL(path.join(web, "wordle_layout.js")).href);
const near = (a, b, m = "") => assert.ok(Math.abs(a - b) < 1e-9, `${m} ${a} != ${b}`);

for (const cols of [4, 5, 6, 7]) {
  const rows = 6, lens = [10, 9, 10];
  const R = W.rackLayout(cols, rows, 3, 10, lens), B = W.barLayout(cols, cols + 2, 10), G = W.giveLayout(cols, rows);
  const mid = cols / 2;
  // every rack row is centred on the board's middle line by its own key count
  for (let r = 0; r < 3; r++) {
    const first = W.rackCentre(R, r, 0)[0], last = W.rackCentre(R, r, lens[r] - 1)[0];
    near((first + last) / 2, mid, `rack row ${r} (cols ${cols})`);
  }
  const nine = W.rackRowLeft(R, 1) - W.rackRowLeft(R, 0);
  near(nine, R.size / 2, "the 9-key row sits half a key in");
  // the bar is centred, above the board, clear of it
  near((W.barCentre(B, 0)[0] + W.barCentre(B, B.cols - 1)[0]) / 2, mid, "bar");
  assert.ok(B.origin[1] + B.zsize < 0, "the bar ends above the board's first row");
  assert.ok(R.origin[1] > rows, "the rack starts below the board");
  // the rack's keys are 1.5x the board's width in all, spaced wider than the board's tiles
  near(R.span, cols * W.RACK_WIDE);
  assert.ok(R.gapX / R.size > 0.03 && R.gapZ / R.size > 0.03);
  // hit tests: each tile's centre hits itself; just past a row's last key is a miss; the 9-key row has no tenth key
  for (let r = 0; r < 3; r++) for (let c = 0; c < lens[r]; c++) {
    const [x, z] = W.rackCentre(R, r, c);
    assert.deepEqual(W.rackHit(R, x, z), { r, c });
    assert.deepEqual(W.rackHit(R, x - R.size * 0.49, z - R.zsize * 0.49), { r, c });
    assert.deepEqual(W.rackHit(R, x + R.size * 0.49, z + R.zsize * 0.49), { r, c });
  }
  const [x9, z9] = W.rackCentre(R, 1, 8);
  assert.equal(W.rackHit(R, x9 + R.size, z9), null, "past the 9-key row's end");
  assert.equal(W.rackHit(R, W.rackRowLeft(R, 1) - 0.01, z9), null, "before the 9-key row's start");
  assert.equal(W.rackHit(R, mid, R.origin[1] - 0.01), null);
  assert.equal(W.rackHit(R, mid, R.origin[1] + 3 * R.zsize + 0.01), null);
  for (let c = 0; c < B.cols; c++) {
    const [x, z] = W.barCentre(B, c);
    assert.deepEqual(W.barHit(B, x, z), { c });
  }
  assert.equal(W.barHit(B, mid, 0.5), null, "the board is not the bar");
  assert.equal(W.barHit(B, B.origin[0] - 0.01, B.origin[1] + 0.1), null);
  assert.deepEqual([0, 1, B.cols - 1].map((c) => W.barKey(B, c)), ["enter", "slot", "back"]);
  // the whole layout frames symmetrically about the board's middle (left + width / 2) and holds the bar, rack and give-up tile
  const X = W.wordleExtent(cols, rows, R, B, G);
  near(X.left + X.width / 2, mid, "extent centre");
  assert.ok(X.left <= B.origin[0] && X.left + X.width >= G.origin[0] + G.size);
  near(X.z0, B.origin[1]); near(X.depth, R.origin[1] + 3 * R.zsize);
}
console.log("wordle layout ok");

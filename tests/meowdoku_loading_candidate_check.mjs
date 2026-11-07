// The Fountain button's compact capture must use live compute snapshots and
// submit an MLA1 stream to the temporary endpoint without a physics replay.
import assert from 'node:assert/strict';
import { beginCandidate } from '../examples/meowdoku/web/loading_candidate.js';
import { decodeStream } from '../examples/meowdoku/web/loading_anim.js';

let upload = null;
globalThis.fetch = async (path, init) => {
  upload = { path, init };
  return { ok: true, json: async () => ({ stored: { expires_at: 'tomorrow' } }) };
};
const compute = {
  ready: true, rec: { n: 4 }, manifest: { law: { dt: 1 / 120 } },
  snap: 1, product: {}, onSnapshot: null, caseAngleAt: () => 0,
};
let status = '', done = 0;
const capture = beginCandidate({ compute, game: { game: 'fountain3d', compute: { size: '4k', view: [-1, 1, -1, 1] } },
  onStatus: (s) => { status = s; }, onDone: () => { done++; } });
for (let k = 1; k <= 25; k++) {
  const frame = new Float32Array(20);
  for (let i = 0; i < 4; i++) {
    frame[5 * i] = -0.6 + i * 0.2 + k * 0.001;
    frame[5 * i + 1] = -0.4 + i * 0.2;
    frame[5 * i + 4] = 100.025;
  }
  compute.onSnapshot(k, frame);
}
await capture.stop();
assert.equal(done, 1);
assert.equal(compute.onSnapshot, null);
assert.equal(upload.path, '/api/loading_candidate');
assert.equal(upload.init.headers['Content-Type'], 'application/octet-stream');
const params = JSON.parse(decodeURIComponent(upload.init.headers['X-Meowdoku-Params']));
assert.equal(params.source, 'fountain3d');
assert.equal(params.size, '4k');
assert.equal(params.visual_only, true);
assert.equal(params.projected_3d, true);
const anim = decodeStream(upload.init.body);
assert.equal(anim.x.length, 3);
assert.equal(anim.x[0].length, 4);
assert.match(status, /^submitted 3 frames/);

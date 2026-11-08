import assert from "node:assert/strict";
import { packMarblePhoto, photoCameraFromStudio } from "../examples/meowdoku/web/marble_photo.js";

const geometry = { interior: [], glass: [] };
const digest = "a".repeat(64);
const config = (options) => {
  const bytes = packMarblePhoto(geometry, digest, options);
  const view = new DataView(bytes);
  return JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, 16, view.getUint32(4, true))));
};
const builtIn = config({ resolution: 512 });
assert.equal(builtIn.camera.fov_degrees, 18);
assert.deepEqual(builtIn.camera.eye_m, [0, 0, .07]);
assert.equal(builtIn.camera.focus_distance_m, .11);

const orbit = photoCameraFromStudio(-.5, .2, 3.45);
const current = config({ ...orbit, resolution: 512 });
assert.equal(current.camera.fov_degrees, 18);
assert.equal(current.camera.focus_distance_m, .11);
assert.ok(current.camera.eye_m[0] > 0 && current.camera.eye_m[1] > 0 && current.camera.eye_m[2] > .18);
assert.deepEqual(current.checker.center_m, [0, 0, .06]);
assert.equal(current.checker.normal_sign, 1);
assert.ok(photoCameraFromStudio(0, 0, 2.3).camera.focus_distance_m < .11);
assert.throws(() => photoCameraFromStudio(0, 0, 0), RangeError);
console.log("marble photo camera framing: ok");

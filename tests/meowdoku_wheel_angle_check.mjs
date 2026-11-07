import assert from "node:assert/strict";
import { Recording, recordedAngle } from "../examples/meowdoku/web/recording.js";

const tape = new Recording({ snapEvery: 6, frameBytes: 4, capBytes: 12 });
tape.push(1, new Float32Array([1]), { body_th: 3.1, body_om: 2 });
tape.push(2, new Float32Array([2]), { body_th: -3.1, body_om: 2 });
const middle = recordedAngle(tape, 1.5, "body_th");
assert.ok(Math.abs(middle - Math.PI) < 0.001, `wrapped midpoint ${middle}`);
assert.equal(recordedAngle(tape, 2, "body_th"), -3.1);
tape.push(3, new Float32Array([3]), { body_th: -3.0, body_om: 2 });
tape.push(4, new Float32Array([4]), { body_th: -2.9, body_om: 2 });
assert.equal(tape.getMeta(1), null); // sliding window drops the matching angle
assert.equal(tape.getMeta(4).body_th, -2.9);
tape.truncateAfter(3);
assert.equal(tape.getMeta(4), null); // branching replay drops future angles
console.log("wheel angle follows per-frame law samples across wrap and recording window");

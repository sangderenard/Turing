import assert from "node:assert/strict";
import { EmojiTumble, EMOJI_TUMBLE_HISTORY, EMOJI_TUMBLE_LIMIT } from "../examples/meowdoku/web/emoji_tumble.js";

const tumble = new EmojiTumble();
const omega = new Float32Array([0, 0, Math.PI, Math.PI, 0, 0]);
const first = tumble.add(1, omega, 0.5);
assert.equal(first.length, 8);
assert.ok(Math.abs(first[2] - Math.SQRT1_2) < 1e-6);
assert.ok(Math.abs(first[3] - Math.SQRT1_2) < 1e-6);
const next = tumble.add(2, omega, 0.5);
assert.ok(Math.abs(next[2] - 1) < 1e-6);
assert.ok(Math.abs(next[3]) < 1e-6);
assert.strictEqual(tumble.get(1), first);
tumble.truncateAfter(1);
assert.equal(tumble.get(2), null);

const long = new EmojiTumble();
const field = new Float32Array((EMOJI_TUMBLE_LIMIT + 10) * 3);
for (let i = 1; i <= EMOJI_TUMBLE_HISTORY + 2; i++) long.add(i, field, 0.01);
assert.equal(long.get(1), null);
assert.equal(long.get(EMOJI_TUMBLE_HISTORY + 2).length, EMOJI_TUMBLE_LIMIT * 4);
console.log("emoji tumble: physical quaternion integration, replay, and bounds OK");

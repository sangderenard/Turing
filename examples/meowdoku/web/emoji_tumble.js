// Physical pose of the displayed emoji meshes. The 3-D law provides angular
// velocity; presentation integrates only the bounded displayed prefix. A
// replay frame keeps its pose, while the source simulation stays untouched.
import { integrateAngularField } from "./marble_rotation.js";

export const EMOJI_TUMBLE_LIMIT = 8192;
export const EMOJI_TUMBLE_HISTORY = 64;

export class EmojiTumble {
  constructor() { this.frames = new Map(); }
  clear() { this.frames.clear(); }
  add(frame, omega, seconds, tapeLo = 0) {
    if (!omega || !omega.length) return null;
    const count = Math.min(EMOJI_TUMBLE_LIMIT, Math.floor(omega.length / 3));
    const pose = new Float32Array(count * 4);
    const previous = this.frames.get(frame - 1);
    integrateAngularField(previous && previous.length === pose.length ? previous : null, pose, omega, seconds, count);
    this.frames.set(frame, pose);
    while (this.frames.size && (this.frames.keys().next().value < tapeLo || this.frames.size > EMOJI_TUMBLE_HISTORY))
      this.frames.delete(this.frames.keys().next().value);
    return pose;
  }
  get(frame) { return this.frames.get(frame) || null; }
  truncateAfter(frame) { for (const f of this.frames.keys()) if (f > frame) this.frames.delete(f); }
}

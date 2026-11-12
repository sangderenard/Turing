import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const source = await readFile(new URL('../examples/meowdoku/web/access.js', import.meta.url), 'utf8');
async function access(reply, fail = false) {
  globalThis.window = {};
  globalThis.fetch = async () => {
    if (fail) throw new Error('offline');
    return { ok: true, json: async () => reply };
  };
  await import('data:text/javascript;base64,' + Buffer.from(source + '\n//' + Math.random()).toString('base64'));
  return window.meowdokuAccess;
}
const guest = await access({ origin: 'dns', tier: 'shader', shader: true, role: 'guest', signed_in: false, capabilities: ['view_site', 'photo_read'] });
assert.equal(guest.role, 'guest');
assert.equal(guest.signed_in, false);
assert.equal(guest.shader, true);
assert.deepEqual(guest.capabilities, ['view_site', 'photo_read']);
assert.ok(Object.isFrozen(guest.capabilities));
assert.ok(Object.isFrozen(guest));
const member = await access({ origin: 'dns', tier: 'play', role: 'member', signed_in: true, capabilities: ['photo_read', 'photo_submit'] });
assert.equal(member.role, 'member');
assert.equal(member.signed_in, true);
assert.ok(member.capabilities.includes('photo_submit'));
const fallback = await access(null, true);
assert.equal(fallback.role, 'guest');
assert.equal(fallback.signed_in, false);
assert.deepEqual(fallback.capabilities, []);
console.log('Guest and member access metadata, immutable capabilities, offline fallback: passed');

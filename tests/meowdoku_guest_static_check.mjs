import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
const dir=path.resolve(process.argv[2]??'build/meowdoku-guest-static');
const manifest=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));
assert.deepEqual(fs.readdirSync(dir).sort(),[...Object.keys(manifest.files),'manifest.json'].sort());
for(const [name,entry]of Object.entries(manifest.files)){
 const bytes=fs.readFileSync(path.join(dir,name));assert.equal(bytes.length,entry.bytes);
 assert.equal(createHash('sha256').update(bytes).digest('hex'),entry.sha256);
 if(name.endsWith('.js'))for(const match of bytes.toString().matchAll(/^import .*?from ["']\.\/([^"']+)["']/gm))assert.ok(manifest.files[match[1]],`Missing dependency ${match[1]}`);
}
// Import the complete renderer in static mode with startup disabled. Every
// network call throws: loading access.js or probing the live API fails here.
globalThis.document={documentElement:{dataset:{meowdokuStatic:'shader'}}};
globalThis.fetch=()=>{throw Error('Cold export must not request an API');};
await import(pathToFileURL(path.join(dir,'loading_panel.js')));
const {decodePacket}=await import(pathToFileURL(path.join(dir,'loading_anim.js')));
const animation=decodePacket(new Uint8Array(fs.readFileSync(path.join(dir,'loading_anim.mla'))));
assert.equal(animation.x.length,manifest.animation.frames);assert.equal(animation.radius.length,manifest.animation.grains);
assert.match(fs.readFileSync(path.join(dir,'index.html'),'utf8'),/data-meowdoku-static="shader"/);
assert.equal(fs.existsSync(path.join(dir,'access.js')),false);
console.log('PASS isolated static dependency closure, hashes, server-free import and baked animation decode');

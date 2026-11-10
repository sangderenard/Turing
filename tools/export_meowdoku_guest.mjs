// Export only the existing guest renderer and its baked animation, never the
// live build directory (which also holds private server state).
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {decodePacket} from '../examples/meowdoku/web/loading_anim.js';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const output=path.resolve(process.argv[2]??path.join(root,'build/meowdoku-guest-static'));
if(fs.existsSync(output)&&fs.readdirSync(output).length)throw Error('Output must be a new or empty isolated directory');
const web=path.join(root,'examples/meowdoku/web');
const packet=fs.readFileSync(path.join(root,'build/meowdoku/loading_anim.mla'));
const anim=decodePacket(new Uint8Array(packet));
const files=['loading_panel.js','loading_anim.js','octet_codec.js','particle_styles.js','world_curve.js','world_zoom.js'];
fs.mkdirSync(output,{recursive:true});
for(const name of files)fs.copyFileSync(path.join(web,name),path.join(output,name));
fs.copyFileSync(path.join(web,'shader_static.html'),path.join(output,'index.html'));
fs.writeFileSync(path.join(output,'loading_anim.mla'),packet);
const manifest={version:1,renderer:'existing loading_panel.js, standalone shader mode',
 animation:{frames:anim.x.length,grains:anim.radius.length,fps:anim.fps},
 files:Object.fromEntries([...files,'index.html','loading_anim.mla'].map(name=>{
 const bytes=fs.readFileSync(path.join(output,name));return [name,{bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')}];}))};
fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify({output,...manifest.animation,files:Object.keys(manifest.files).length}));

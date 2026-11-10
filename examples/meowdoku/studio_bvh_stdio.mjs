// JSON-lines compute adapter. Lease ownership and commit remain with Nodus/engine.
import {readFile} from 'node:fs/promises';
import {resolve,sep} from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {createInterface} from 'node:readline';
import {StudioBvhCompute} from './web/studio_bvh_compute.js';
if(!process.argv[2])throw Error('Compiled artifact directory required');
const root=resolve(process.argv[2]),base=pathToFileURL(root+sep);
globalThis.fetch=async url=>{const path=resolve(fileURLToPath(new URL(url)));if(!path.startsWith(root+sep))throw Error('Artifact outside selected directory');const b=await readFile(path);return {ok:true,arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)}};
const backend=await StudioBvhCompute.create(base,{backend:'wasm'});
for await(const line of createInterface({input:process.stdin,crlfDelay:Infinity})){
 try{const request=JSON.parse(line);if(request.query_kernel&&request.query_kernel!==backend.manifest.query_kernel_sha256)throw Error('Query kernel identity mismatch');backend.setScene(request);const rays=new Float32Array(Array.isArray(request.rays[0])?request.rays.flat():request.rays);const result=await backend.query(rays,request.revision);process.stdout.write(JSON.stringify({...result,hits:Array.from(result.hits)})+'\n');}
 catch(error){process.stdout.write(JSON.stringify({error:error.message})+'\n');}
}
backend.close();

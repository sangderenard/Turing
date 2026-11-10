// Local-only browser QA. All APIs and non-exported paths are rejected; this
// process has no Python dependency and never forwards to the live server.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
const dir=path.resolve(process.argv[2]??'build/meowdoku-guest-static');
const manifest=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));
const allowed=new Set(Object.keys(manifest.files));
http.createServer((req,res)=>{
 const name=new URL(req.url,'http://localhost').pathname.slice(1)||'index.html';
 console.log(`${req.method} /${name}`);
 if(req.method!=='GET'||!allowed.has(name)){res.writeHead(404);res.end();return;}
 const mime=name.endsWith('.js')?'text/javascript':name.endsWith('.html')?'text/html':'application/octet-stream';
 res.writeHead(200,{'Content-Type':mime,'Cache-Control':'no-store'});fs.createReadStream(path.join(dir,name)).pipe(res);
}).listen(8897,'127.0.0.1',()=>console.log('Cold static QA at http://127.0.0.1:8897, all APIs denied'));

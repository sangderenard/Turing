import http from 'node:http';import fs from 'node:fs';import path from 'node:path';
const root=path.resolve('build/meowdoku/web');
http.createServer((req,res)=>{
 const pathname=new URL(req.url,'http://localhost').pathname;
 if(pathname==='/phong-probe.js'){res.writeHead(200,{'Content-Type':'text/javascript'});res.end(fs.readFileSync('tests/meowdoku_phong_probe.js'));return;}
 if(pathname==='/assets.json'){const assets=JSON.parse(fs.readFileSync(path.join(root,'assets.json'),'utf8'));assets.emitterProfiles=JSON.parse(fs.readFileSync('build/dot-named-emitters-20261110/emitter_profiles.json','utf8'));res.writeHead(200,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify(assets));return;}
 const chosenRoot=['/marble_studio.js','/marble_studio_lighting.js'].includes(pathname)?path.resolve('build/dot-shop-registration'):['/marble_shop_lights.js','/marble_shop_light_controls.js'].includes(pathname)?path.resolve('examples/meowdoku/web'):root;
 const file=path.resolve(chosenRoot,'.'+(pathname==='/'?'/marble_studio.html':pathname));
 if(!file.startsWith(chosenRoot+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()){res.writeHead(404);res.end();return;}
 const mime=file.endsWith('.js')?'text/javascript':file.endsWith('.html')?'text/html':file.endsWith('.json')?'application/json':'application/octet-stream';
 res.writeHead(200,{'Content-Type':mime,'Cache-Control':'no-store'});
 if(file.endsWith('marble_studio.html'))res.end(fs.readFileSync(file,'utf8').replace('<head>','<head><script src="/phong-probe.js"></script>'));
 else fs.createReadStream(file).pipe(res);
}).listen(8898,'127.0.0.1',()=>console.log('Independent Phong probe http://127.0.0.1:8898'));


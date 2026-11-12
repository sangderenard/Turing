"""Bounded real Chrome check of the existing static Phong renderer."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import base64, hashlib, json, shutil, subprocess, tempfile, threading

ROOT=Path(__file__).resolve().parents[1]
CHECK=r'''<script>
(async()=>{try{
 const host=window.livingDataStaticHost; await host.ready;
 const snapshot=host.snapshot();
 const a=await host.render({position:[32,2,-10],facing:[0,0,-1],width:320,height:240});
 const b=await host.render({position:[28,4,-8],facing:[1,-.2,-1],width:320,height:240});
 const changed=[...turingWorld.objects.values()][0], oldLabel=changed.label;
 changed.label=String(changed.label)+' test-only mutation';
 let snapshotRejected=false,renderRejected=false,restoredRejected=false;
 try{host.snapshot();}catch(e){snapshotRejected=String(e).includes('Static map changed');}
 try{await host.render({position:[32,2,-10],facing:[0,0,-1],width:320,height:240});}catch(e){renderRejected=String(e).includes('Static map changed');}
 changed.label=oldLabel;
 try{host.snapshot();}catch(e){restoredRejected=String(e).includes('Static map changed');}
 if(!snapshotRejected||!renderRejected||!restoredRejected)throw Error('Static source guard did not latch refusal');
  await fetch('/receipt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
 ok:true,staticMutationRejected:true,workers:window.staticTestWorkers,snapshot,a,b})});
}catch(error){await fetch('/receipt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ok:false,error:String(error),stack:error.stack})});}})();
</script>'''


def run_static_browser_check(stage):
 stage=Path(stage); page=(stage/'index.html').read_bytes()
 worker=b'<script>window.staticTestWorkers=[];const StaticNativeWorker=window.Worker;window.Worker=class extends StaticNativeWorker{constructor(...args){super(...args);window.staticTestWorkers.push(String(args[0]));}};</script>'
 page=page.replace(b'<head>',b'<head>'+worker,1).replace(b'</body>',CHECK.encode()+b'</body>',1)
 received=[];event=threading.Event()
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_GET(self):
   self.send_response(200);self.send_header('Content-Type','text/html');self.send_header('Content-Length',str(len(page)));self.end_headers();self.wfile.write(page)
  def do_POST(self):
   size=int(self.headers.get('Content-Length','0'))
   if self.path!='/receipt' or size>8*1024*1024:self.send_error(400);return
   received.append(json.loads(self.rfile.read(size)));self.send_response(200);self.end_headers();event.set()
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
 browser=next(p for p in [r'C:\Program Files\Google\Chrome\Application\chrome.exe',r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'] if Path(p).exists())
 try:
  with tempfile.TemporaryDirectory(prefix='static-phong-profile-') as profile:
   process=subprocess.Popen([browser,'--headless=new','--no-first-run','--disable-extensions','--use-angle=swiftshader','--enable-unsafe-swiftshader',f'--user-data-dir={profile}',f'http://127.0.0.1:{server.server_port}/'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   try:
    if not event.wait(45):raise TimeoutError('Static Phong browser did not return a receipt')
   finally:
    process.terminate();process.wait(timeout=10)
 except PermissionError:pass # Windows profile cleanup can lag renderer process exit.
 finally:server.shutdown();server.server_close()
 if not received:raise RuntimeError('No static Phong receipt')
 result=received[0]
 if not result.get('ok'):raise RuntimeError(result)
 assert result['workers']==[],result['workers']
 assert result['snapshot']['physics_started'] is False
 assert all('extensions' in node and 'material_bindings' in node and 'physics' in node and 'persistence' in node for node in result['snapshot']['map']['world']['objects'])
 assert result['a']['renderer_identity'].endswith('/pluck-phong')
 assert result['a']['data_url']!=result['b']['data_url']
 summary={'static_mutation_rejected':result['staticMutationRejected'],'workers_started':0,'objects':len(result['snapshot']['map']['world']['objects']),
  'source_sha256':result['snapshot']['source_sha256'],'renderer_identity':result['a']['renderer_identity'],'frames':[]}
 for key in ['a','b']:
  value=result[key];data=base64.b64decode(value['data_url'].split(',',1)[1]);file=stage/f'phong-{key}.png';file.write_bytes(data)
  assert data.startswith(b'\x89PNG\r\n\x1a\n')
  summary['frames'].append({'path':str(file),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'camera':value['camera']})
 (stage/'static-browser-receipt.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
 return summary

if __name__=='__main__':print(json.dumps(run_static_browser_check(ROOT/'build/dot-static-phong-rest-20261110'),indent=2))

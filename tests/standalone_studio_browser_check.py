"""Real local Chrome WebGL2 check of the no-controls full Studio runner."""
from pathlib import Path
import base64,hashlib,json,subprocess,tempfile,threading
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[1]
PAGE='''<!doctype html><meta charset="utf-8"><canvas id="view"></canvas><script type="module">
import {createStandaloneStudioScene} from '/examples/meowdoku/web/standalone_studio.js';
try{
 const asset=await fetch('/build/dot-studio-game-transfer-20261110/complete-studio.scene-asset.json').then(r=>r.json());
 const canvas=document.querySelector('canvas');
 let workers=0,frames=0,listeners=0;
 const W=window.Worker;window.Worker=class extends W{constructor(...a){workers++;super(...a);}};
 const raf=window.requestAnimationFrame;window.requestAnimationFrame=(...a)=>{frames++;return raf(...a);};
 const listen=EventTarget.prototype.addEventListener;EventTarget.prototype.addEventListener=function(...a){listeners++;return listen.apply(this,a);};
 const runner=await createStandaloneStudioScene({canvas,sceneAsset:asset});
 const view=asset.render.referenceCamera.view, f=1/Math.tan((asset.render.referenceCamera.fovDegrees||50)*Math.PI/360);
 const camera={view:[...view],projection:[f,0,0,0,0,f,0,0,0,0,-30.01/29.99,-1,0,0,-.6/29.99,0],width:320,height:320};
 const first=runner.render(camera);let rejected=false;
 try{runner.render({...camera,width:NaN});}catch(e){rejected=true;}
 const stable=runner.render(camera);
 const changed={...camera,view:[...view]};changed.view[12]+=.4;
 const second=runner.render(changed);
 if(!rejected||stable.data_url!==first.data_url)throw Error('Malformed camera poisoned valid render');
 if(first.data_url===second.data_url)throw Error('Camera change did not alter pixels');
 if(workers||frames||listeners)throw Error('Runner installed animation or input machinery');
 runner.dispose();
 await fetch('/receipt',{method:'POST',body:JSON.stringify({ok:true,workers,frames,listeners,rejected,asset:asset.identity,images:[first,second]})});
}catch(e){await fetch('/receipt',{method:'POST',body:JSON.stringify({ok:false,error:String(e),stack:e.stack})});}
</script>'''

def run(output_directory=None):
 received=[];done=threading.Event()
 class Handler(SimpleHTTPRequestHandler):
  def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT),**kwargs)
  def do_GET(self):
   if self.path!='/check':return super().do_GET()
   self.send_response(200);self.send_header('Content-Type','text/html');self.end_headers();self.wfile.write(PAGE.encode())
  def do_POST(self):
   if self.path!='/receipt':self.send_error(404);return
   received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))));self.send_response(200);self.end_headers();done.set()
 httpd=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=httpd.serve_forever,daemon=True).start()
 with tempfile.TemporaryDirectory(prefix='standalone-studio-',ignore_cleanup_errors=True) as profile:
  browser=r'C:\Program Files\Google\Chrome\Application\chrome.exe'
  p=subprocess.Popen([browser,'--headless=new','--no-first-run','--disable-extensions','--use-angle=swiftshader','--enable-unsafe-swiftshader',f'--user-data-dir={profile}',f'http://127.0.0.1:{httpd.server_port}/check'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  try:
   if not done.wait(90):raise TimeoutError('Studio check did not finish')
  finally:p.terminate();p.wait(timeout=10);httpd.shutdown();httpd.server_close()
 result=received[0]
 if not result['ok']:raise AssertionError(result)
 directory=Path(output_directory) if output_directory else ROOT/'build/dot-standalone-studio-20261111';directory.mkdir(exist_ok=True)
 for index,frame in enumerate(result['images']):
  data=base64.b64decode(frame.pop('data_url').split(',')[1]);path=directory/f'studio-camera-{index}.png';path.write_bytes(data)
  frame.update(path=str(path),sha256=hashlib.sha256(data).hexdigest())
 (directory/'receipt.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 return result
if __name__=='__main__':print(json.dumps(run(),indent=2))

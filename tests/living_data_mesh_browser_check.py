"""Capture the actual static map's uploaded mesh in real Chrome, without gameplay."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import subprocess
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
CHECK = b'''<script>
(async()=>{try{
 const host=window.livingDataStaticHost;await host.ready;
 const mesh=host.meshSnapshot();
 const again=host.meshSnapshot();
 if(JSON.stringify(mesh)!==JSON.stringify(again))throw Error('Mesh snapshot is not stable');
 if(mesh.world.objects.length!==33)throw Error('Expected the actual 33-object Storybook world');
 if(!mesh.world_identity.endsWith('StorybookEnvironment'))throw Error('Wrong world');
 const original=shaderViewer.mesh[0];shaderViewer.mesh[0]+=1;
 let refused=false;try{host.meshSnapshot();}catch(e){refused=String(e).includes('mesh changed');}
 shaderViewer.mesh[0]=original;
 let latched=false;try{host.meshSnapshot();}catch(e){latched=String(e).includes('Static map changed');}
 if(!refused||!latched)throw Error('Mesh mutation guard did not latch');
 await fetch('/mesh',{method:'POST',body:JSON.stringify({ok:true,workers:window.meshTestWorkers,
  mesh,mutation_refused:refused,refusal_latched:latched})});
}catch(error){await fetch('/mesh',{method:'POST',body:JSON.stringify({ok:false,error:String(error),stack:error.stack})});}})();
</script>'''


def capture_storybook_mesh(stage):
    stage = Path(stage)
    page = (stage / 'index.html').read_bytes()
    worker = b'<script>window.meshTestWorkers=[];const MeshNativeWorker=window.Worker;window.Worker=class extends MeshNativeWorker{constructor(...args){super(...args);window.meshTestWorkers.push(String(args[0]));}};</script>'
    page = page.replace(b'<head>', b'<head>' + worker, 1).replace(b'</body>', CHECK + b'</body>', 1)
    received, event = [], threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self):
            size = int(self.headers.get('Content-Length', '0'))
            if self.path != '/mesh' or not 0 < size <= 64 * 1024 * 1024:
                self.send_error(400)
                return
            received.append(json.loads(self.rfile.read(size)))
            self.send_response(200)
            self.end_headers()
            event.set()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    browser = next(p for p in [r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                              r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'] if Path(p).exists())
    try:
        with tempfile.TemporaryDirectory(prefix='living-data-mesh-', ignore_cleanup_errors=True) as profile:
            process = subprocess.Popen([browser, '--headless=new', '--no-first-run',
                '--disable-extensions', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
                f'--user-data-dir={profile}', f'http://127.0.0.1:{server.server_port}/'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                if not event.wait(45):
                    raise TimeoutError('Native Storybook mesh capture did not return')
            finally:
                process.terminate()
                process.wait(timeout=10)
    finally:
        server.shutdown()
        server.server_close()
    result = received[0]
    if not result.get('ok'):
        raise RuntimeError(result)
    assert result['workers'] == []
    assert not result['mesh']['physics_started']
    path = stage / 'realized-mesh.json'
    path.write_text(json.dumps(result['mesh'], separators=(',', ':')), encoding='utf-8')
    return {'mesh_path': str(path), 'vertices': len(result['mesh']['vertices']) // 9,
            'objects': len(result['mesh']['world']['objects']), 'workers': 0,
            'mutation_refused': result['mutation_refused'], 'refusal_latched': result['refusal_latched']}


if __name__ == '__main__':
    print(json.dumps(capture_storybook_mesh(ROOT / 'build/dot-storybook-static-phong-20261110'), indent=2))

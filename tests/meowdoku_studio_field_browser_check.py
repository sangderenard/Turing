"""Isolated real Studio/API fixture for manual browser field validation.

No alternate renderer or API responses: current source web assets overlay the
built material assets, and real Sessions owns private test state and ray jobs.
Run with the service Python, then use its printed localhost URL in Chrome.
"""
from pathlib import Path
from http.server import ThreadingHTTPServer
import json
import sys
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'examples')]
from meowdoku.server import Sessions, make_handler
from meowdoku.state_paths import StatePaths

def main():
    fixture = ROOT / 'build/dot-studio-field-20261111'
    fixture.mkdir(parents=True, exist_ok=True)
    built = ROOT / 'build/meowdoku/web'
    source = ROOT / 'examples/meowdoku/web'
    # The deployed catalogue predates ribbon authoring. Build current tables
    # through the canonical exporter without rebuilding unrelated glyph meshes.
    from meowdoku.web_export import _spectral, register_material_catalog
    _, _, MaterialDatabase = _spectral()
    db = MaterialDatabase()
    index, extra = register_material_catalog(db)
    tensors = db.build_tensors()
    assets = json.loads((built / 'assets.json').read_text(encoding='utf8'))
    assets.update(material_index=index, material_extras=extra, materials={
        name: {'floats_per_record': int(tensors[key].shape[1]),
               'records': int(tensors[key].shape[0]),
               'data': tensors[key].reshape(-1).tolist()}
        for name, key in [('pbr','pbr'),('phong','phong_compat'),
                          ('enamel','enamel'),('texstack','texture_stack')]})
    (fixture / 'assets.json').write_text(json.dumps(assets), encoding='utf8')
    shader = ROOT.parent / 'spectral-analyzer/repo-cache/phong-field-20261111/base_material.webgl.shadow.frag.glsl'
    assert shader.is_file()
    sessions = Sessions(built, state=StatePaths.legacy_at(fixture / 'state'))
    # A disposable local-only account, never production credentials.
    sessions.accounts.ensure_user('field_fixture', 'local-field-fixture-only')
    base = make_handler(sessions)
    class Handler(base):
        def do_GET(self):
            if urllib.parse.urlsplit(self.path).path == '/marble_studio.html':
                data = (source / 'marble_studio.html').read_text(encoding='utf8')
                probe = '''<script>
const originalGetContext=HTMLCanvasElement.prototype.getContext;
HTMLCanvasElement.prototype.getContext=function(...args){
 const gl=originalGetContext.apply(this,args);
 if(this.id==='view'&&args[0]==='webgl2'&&!gl.__fieldProbe){
  gl.__fieldProbe=true; const draw=gl.drawArrays.bind(gl),upload=gl.texImage3D.bind(gl);
  let uploaded=null,last=0;
  gl.texImage3D=function(...a){if(a[3]>1&&a[4]>1&&a[5]>1)uploaded={dimensions:a.slice(3,6),samples:a[9]?.length,nonzero:a[9]?.some?.(v=>v!==0)};return upload(...a);};
  gl.drawArrays=function(...a){
   if(performance.now()-last>250){const p=gl.getParameter(gl.CURRENT_PROGRAM),u=gl.getUniformLocation(p,'uFieldGain');
    if(u!==null){last=performance.now();let out=document.getElementById('field-test-observation');
     if(!out){out=document.createElement('output');out.id='field-test-observation';out.style='position:fixed;bottom:0;left:0;z-index:9999;background:#000;color:#fff;font:11px monospace';document.body.append(out);}
     const get=n=>gl.getUniform(p,gl.getUniformLocation(p,n));
     out.textContent=JSON.stringify({gain:get('uFieldGain'),useTransform:get('uFieldUseTransform'),unit:get('uFieldVolume'),matrix:Array.from(get('uFieldFromView')),uploaded});}}
   return draw(...a);
  };
 }return gl;
};
</script>'''
                data = data.replace('<head>', '<head>' + probe).encode('utf8')
                self.send_response(200); self.send_header('Content-Type','text/html'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
                return
            return super().do_GET()
        def translate_path(self, path):
            relative = urllib.parse.unquote(urllib.parse.urlsplit(path).path).lstrip('/')
            if relative == 'assets.json' and (fixture / 'assets.json').is_file():
                return str(fixture / 'assets.json')
            if relative == 'shaders/base_material.webgl.shadow.frag.glsl':
                return str(shader)
            candidate = (source / relative).resolve()
            if candidate.is_relative_to(source.resolve()) and candidate.is_file():
                return str(candidate)
            return super().translate_path(path)
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    receipt = {'url': f'http://127.0.0.1:{httpd.server_port}/login.html',
               'state': str(sessions.state.root), 'shader': str(shader)}
    (fixture / 'server.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    try:
        httpd.serve_forever()
    finally:
        sessions.marble_photos.close()
        httpd.server_close()

if __name__ == '__main__':
    main()

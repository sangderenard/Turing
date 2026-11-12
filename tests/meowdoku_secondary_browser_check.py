"""Isolated real Studio/API fixture for manual browser field validation.

No alternate renderer or API responses: current source web assets overlay the
built material assets, and real Sessions owns private test state and ray jobs.
Run with the service Python, then use its printed localhost URL in Chrome.
"""
from pathlib import Path
from http.server import ThreadingHTTPServer
import json
import os
import sys
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'examples')]
from meowdoku.server import Sessions, make_handler
from meowdoku.state_paths import StatePaths

def main():
    fixture = ROOT / os.environ.get('FIELD_FIXTURE_DIR', 'build/dot-secondary-browser-20261111')
    fixture.mkdir(parents=True, exist_ok=True)
    built = ROOT / 'build/meowdoku/web'
    source = ROOT / os.environ.get('FIELD_FIXTURE_WEB', 'examples/meowdoku/web')
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
    shader = ROOT / os.environ.get('FIELD_FIXTURE_SHADER', 'build/dot-secondary-field-20261111/shaders/base_material.webgl.shadow.frag.glsl')
    assert shader.is_file()
    sessions = Sessions(built, state=StatePaths.legacy_at(fixture / 'state'))
    # A disposable local-only account, never production credentials.
    sessions.accounts.ensure_user('field_fixture', 'local-field-fixture-only')
    base = make_handler(sessions)
    class Handler(base):
        def log_message(self, fmt, *args):
            # Local fixture audit only: verifies the UI did not request ray images.
            with (fixture / 'requests.log').open('a', encoding='utf8') as stream:
                stream.write((fmt % args) + '\n')
            return super().log_message(fmt, *args)
        def do_GET(self):
            if urllib.parse.urlsplit(self.path).path == '/marble_studio.html':
                data = (source / 'marble_studio.html').read_text(encoding='utf8')
                probe = '<script src="secondary_probe.js"></script>'
                data = data.replace('<head>', '<head>' + probe).encode('utf8')
                self.send_response(200); self.send_header('Content-Type','text/html'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
                return
            return super().do_GET()
        def translate_path(self, path):
            relative = urllib.parse.unquote(urllib.parse.urlsplit(path).path).lstrip('/')
            if relative == 'secondary_probe.js':
                return str(ROOT / 'tests/meowdoku_secondary_probe.js')
            if relative == 'assets.json' and (fixture / 'assets.json').is_file():
                return str(fixture / 'assets.json')
            if relative in ('shaders/base_material.webgl.shadow.frag.glsl', 'shaders/base_material.webgl.vert.glsl'):
                return str(shader.parent / Path(relative).name)
            candidate = (source / relative).resolve()
            if candidate.is_relative_to(source.resolve()) and candidate.is_file():
                return str(candidate)
            return super().translate_path(path)
    httpd = ThreadingHTTPServer(('127.0.0.1', int(os.environ.get('FIELD_FIXTURE_PORT', '0'))), Handler)
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


"""Read-only loopback QA for the staged optional layer and unchanged live assets."""
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'build/dot-studio-sync-20261110'
LIVE = ROOT / 'build/meowdoku/web'

class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, '.js':'text/javascript', '.mjs':'text/javascript'}

    def translate_path(self, path):
        relative = Path(unquote(urlsplit(path).path).lstrip('/') or 'marble_studio_layer.html')
        if relative.is_absolute() or '..' in relative.parts:
            return str(STAGE / 'not-found')
        candidate = STAGE / relative
        return str(candidate if candidate.is_file() else LIVE / relative)

    def do_GET(self):
        if urlsplit(self.path).path.startswith('/api/'):
            self.send_error(404, 'Read-only QA: production APIs are not proxied')
            return
        for name in ('If-Modified-Since','If-None-Match'):
            if name in self.headers:
                del self.headers[name]
        super().do_GET()

    def do_POST(self):
        self.send_error(405, 'Read-only QA: no production mutations')

    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        super().end_headers()

if __name__ == '__main__':
    print('Read-only Studio/original-game layer QA: http://127.0.0.1:8898/marble_studio_layer.html', flush=True)
    ThreadingHTTPServer(('127.0.0.1',8898), Handler).serve_forever()

"""Isolated headless Chrome, local static fixtures, no production app mutation."""
from functools import partial
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import subprocess,tempfile,threading

root=Path(__file__).resolve().parents[1]
class Quiet(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(root)))
threading.Thread(target=server.serve_forever,daemon=True).start()
try:
    with tempfile.TemporaryDirectory(prefix='studio-rangefinder-chrome-',dir=root/'build') as profile:
        result=subprocess.run([r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe','--headless=new',
            '--disable-background-networking','--no-first-run','--no-default-browser-check',
            '--user-data-dir='+profile,'--virtual-time-budget=10000','--dump-dom',
            f'http://127.0.0.1:{server.server_port}/tests/meowdoku_rangefinder_browser.html'],
            capture_output=True,text=True,timeout=60)
        html=result.stdout
        (root/'build'/'studio-rangefinder-browser-receipt.html').write_text(html,encoding='utf-8')
        start=html.find('<pre id="result">');end=html.find('</pre>',start)
        print(html[start:end+6] if start>=0 else result.stderr[-3000:])
        if result.returncode or '<title>PASS</title>' not in html:raise SystemExit(1)
finally:server.shutdown();server.server_close()

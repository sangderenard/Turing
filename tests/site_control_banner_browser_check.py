"""Offline browser checks of the shared control banner (no live accounts/requests)."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
BROWSER = next((p for p in [r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe', r'C:\Program Files\Google\Chrome\Application\chrome.exe'] if Path(p).exists()), None)

class BannerTests(unittest.TestCase):
    @unittest.skipUnless(BROWSER, 'Chrome or Edge is required')
    def test_account_banner(self):
        script = (ROOT / 'examples/meowdoku/web/site-nav.js').read_bytes()
        html = b'''<!doctype html><html><body><button id="outside">Outside</button>
<script>
const attach = Element.prototype.attachShadow;
Element.prototype.attachShadow = function(init) { const root = attach.call(this,init); window.testRoot=root; return root; };
const signed = location.search !== '?guest';
window.fetch = async path => ({ok:true,status:200,json:async()=>path==='/api/whoami'?{signed_in:signed,username:'test_operator',role:'operator',capabilities:signed?['play','dev_ui']:[]}:{route:'a',dev_route:'b',dev_available:false}});
</script>
<script type="module" src="/site-nav.js" data-site-nav="allow"></script>
<script type="module">
const check=(value,label)=>{if(!value)throw Error(label);};
try {
 for(let i=0;i<50 && !window.testRoot;i++)await new Promise(r=>setTimeout(r,20));
 await new Promise(r=>setTimeout(r,50));
 const root=window.testRoot,nav=root.querySelector('nav');
 check(nav.getAttribute('aria-label')==='Site controls','named control region');
 const box=nav.getBoundingClientRect();check(box.right<=innerWidth && box.left>=0,'mobile bounds');
 if(!signed)check(root.querySelector('a').getAttribute('href')==='/login.html','guest sign in');
 else {
  const button=root.querySelector('button[aria-haspopup]'),menu=root.querySelector('.menu');
  check(menu.hidden && getComputedStyle(menu).display==='none','initially closed');
  button.click();check(!menu.hidden,'opens');
  check([...menu.querySelectorAll('button')].some(b=>b.textContent==='Development server' && b.disabled),'unavailable dev disabled');
  nav.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}));check(menu.hidden,'escape closes');
  button.click();document.getElementById('outside').dispatchEvent(new PointerEvent('pointerdown',{bubbles:true}));check(menu.hidden,'outside closes');
  button.click();button.click();check(menu.hidden,'repeat toggles');
 }
 document.body.append(Object.assign(document.createElement('pre'),{id:'result',textContent:'PASS'}));
} catch(error) {document.body.append(Object.assign(document.createElement('pre'),{id:'result',textContent:'FAIL '+error.message}));}
</script></body></html>'''
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                content, kind = (script, 'text/javascript') if self.path == '/site-nav.js' else (html, 'text/html')
                self.send_response(200); self.send_header('Content-Type', kind); self.end_headers(); self.wfile.write(content)
            def log_message(self,*args): pass
        server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            for state in ['guest','member']:
                with self.subTest(state=state), tempfile.TemporaryDirectory(prefix='banner-browser-') as profile:
                    result=subprocess.run([BROWSER,'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',f'--user-data-dir={profile}','--window-size=390,844','--virtual-time-budget=2500','--dump-dom',f'http://127.0.0.1:{server.server_port}/?{state}'],capture_output=True,text=True,timeout=30)
                    self.assertIn('<pre id="result">PASS</pre>',result.stdout,result.stdout[-3000:]+result.stderr[-1000:])
        finally:
            server.shutdown();server.server_close()

if __name__=='__main__': unittest.main()

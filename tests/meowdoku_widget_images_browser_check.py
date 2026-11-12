"""Local HTTPS browser image opt-in regression, not collected by pytest.

Run: python tests/meowdoku_widget_images_browser_check.py
Uses existing stdlib CDP harness pattern, Pillow, and Git's OpenSSL. The temporary
certificate is trusted only by the fixture HTTPS client and pinned browser process.
No installed trust store, production settings, or user profile is modified.
"""
from __future__ import annotations

import ssl, hashlib, io
import base64, http.client, json, os, secrets, shutil, socket, struct, subprocess, sys, tempfile, threading, time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku import server as srv  # noqa: E402

BROWSERS = ["C:/Program Files/Google/Chrome/Application/chrome.exe", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"]
RESULTS = []
TLS_CONTEXT = None


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (" -- " + str(detail) if detail != "" else ""), flush=True)


def call(port, path, method="GET", cookie="", body=None):
    conn = http.client.HTTPSConnection("127.0.0.1", port, timeout=40, context=TLS_CONTEXT)
    headers = {"Cookie": cookie} if cookie else {}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=headers)
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    try:
        return r.status, json.loads(raw), r
    except ValueError:
        return r.status, raw.decode("utf-8", "replace"), r


def login(port, name, pw):
    s, b, r = call(port, "/api/account/login", "POST", "", {"username": name, "password": pw})
    return next(c.split(";")[0].split("=", 1)[1] for c in r.msg.get_all("Set-Cookie") if c.startswith(srv.ACCOUNT_COOKIE + "="))


class WS:
    def __init__(self, port, path):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.s.recv(1)
        assert b" 101 " in buf.split(b"\r\n")[0], buf
        self.s.settimeout(None)
        self.lock = threading.Lock()
        self.id = 0
        self.pending = {}
        self.events = []
        self.cv = threading.Condition()
        threading.Thread(target=self._read, daemon=True).start()

    def _recv(self, n):
        b = b""
        while len(b) < n:
            c = self.s.recv(n - len(b))
            if not c:
                raise EOFError
            b += c
        return b

    def _read(self):
        try:
            while True:
                h = self._recv(2)
                op = h[0] & 15
                ln = h[1] & 127
                if ln == 126:
                    ln = struct.unpack(">H", self._recv(2))[0]
                elif ln == 127:
                    ln = struct.unpack(">Q", self._recv(8))[0]
                data = self._recv(ln)
                if op != 1:
                    continue
                m = json.loads(data)
                with self.cv:
                    if "id" in m:
                        self.pending[m["id"]] = m
                    else:
                        self.events.append((time.time(), m))
                    self.cv.notify_all()
        except Exception:
            pass

    def cmd(self, method, params=None, timeout=30):
        with self.lock:
            self.id += 1
            i = self.id
            data = json.dumps({"id": i, "method": method, "params": params or {}}).encode()
            mask = os.urandom(4)
            n = len(data)
            if n < 126:
                hdr = bytes([0x81, 0x80 | n])
            elif n < 65536:
                hdr = bytes([0x81, 0x80 | 126]) + struct.pack(">H", n)
            else:
                hdr = bytes([0x81, 0x80 | 127]) + struct.pack(">Q", n)
            self.s.sendall(hdr + mask + bytes(b ^ mask[k % 4] for k, b in enumerate(data)))
        end = time.time() + timeout
        with self.cv:
            while i not in self.pending:
                if time.time() > end:
                    raise TimeoutError(method)
                self.cv.wait(0.2)
            r = self.pending.pop(i)
        if "error" in r:
            raise RuntimeError(f"{method}: {r['error']}")
        return r["result"]


def main():
    global TLS_CONTEXT
    tmp = Path(tempfile.mkdtemp(prefix="mwidget_images_"))
    web = tmp / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}))
    (web / "index.html").write_text("<!doctype html><title>Local image opt-in test</title><script type=module src=/messenger-widget.js data-messenger=allow></script>")
    shutil.copy(ROOT / "examples/meowdoku/web/messenger-widget.js", web / "messenger-widget.js")
    openssl = "C:/Program Files/Git/usr/bin/openssl.exe"
    cert, key = tmp / "cert.pem", tmp / "key.pem"
    subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-keyout", str(key), "-out", str(cert), "-subj", "/CN=localhost", "-addext", "subjectAltName=IP:127.0.0.1,DNS:localhost"], check=True, capture_output=True)
    pub = subprocess.run([openssl, "x509", "-in", str(cert), "-pubkey", "-noout"], check=True, capture_output=True).stdout
    der = subprocess.run([openssl, "pkey", "-pubin", "-outform", "DER"], input=pub, check=True, capture_output=True).stdout
    spki = base64.b64encode(hashlib.sha256(der).digest()).decode()
    TLS_CONTEXT = ssl.create_default_context(cafile=str(cert))
    sessions = srv.Sessions(web, None)
    sessions.lifecycle = SimpleNamespace(token="t" * 40, restore_for=lambda token: None)
    sessions.access.lookup = lambda url: None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(cert, key)
    httpd.socket = tls.wrap_socket(httpd.socket, server_side=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    pw = {n: secrets.token_urlsafe(16) for n in ("mia", "pat")}
    for n in pw: sessions.accounts.ensure_user(n, pw[n])
    ck = {n: login(port, n, pw[n]) for n in pw}
    ch = {n: srv.ACCOUNT_COOKIE + "=" + ck[n] for n in pw}
    def api(user, path, body=None):
        status, data, _ = call(port, path, "GET" if body is None else "POST", ch[user], body)
        assert status == 200, (status, data)
        return data
    api("pat", "/api/messages/contacts", {"username":"mia", "action":"request"})
    api("mia", "/api/messages/contacts", {"username":"pat", "action":"accept"})
    handles = {n:api(n, "/api/messages/conversations")["contacts"][0]["conversation_handle"] for n in pw}
    from PIL import Image
    png = tmp / "upload.png"
    Image.new("RGB", (40, 30), "#e09030").save(png)
    image = {"data":base64.b64encode(png.read_bytes()).decode()}
    api("pat", "/api/messages/images", {"conversation_handle":handles["pat"], "enabled":True})
    receipt = api("pat", "/api/messages/send", {"conversation_handle":handles["pat"], "body":"Unlock this existing image", "image":image, "idempotency_key":"local-test-image-1"})
    base = f"https://127.0.0.1:{port}"
    exe = next(b for b in BROWSERS if os.path.exists(b))
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); dport=sock.getsockname()[1]; sock.close()
    proc = subprocess.Popen([exe, "--headless=new", f"--remote-debugging-port={dport}", f"--user-data-dir={tmp / 'profile'}", f"--ignore-certificate-errors-spki-list={spki}", "--no-first-run", "--no-default-browser-check", "--disable-gpu", "--window-size=1000,850", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pid=proc.pid
    try:
        tgt = None
        for _ in range(60):
            try:
                c = http.client.HTTPConnection("127.0.0.1", dport, timeout=3)
                c.request("GET", "/json")
                lst = json.loads(c.getresponse().read())
                c.close()
                tgt = next((t for t in lst if t["type"] == "page"), None)
                if tgt:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        ws = WS(dport, "/" + tgt["webSocketDebuggerUrl"].split("/", 3)[3])
        for d in ("Page", "Network", "Runtime", "DOM"):
            ws.cmd(d + ".enable")
        ws.cmd("Network.setCacheDisabled", {"cacheDisabled": True})

        def ev(expr):
            r = ws.cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True})
            return r.get("result", {}).get("value")

        def goto(path):
            ws.cmd("Page.navigate", {"url": base + path})
            time.sleep(0.5)
            for _ in range(60):
                if ev("document.readyState") == "complete":
                    break
                time.sleep(0.2)
            time.sleep(1.5)

        def shadow():
            doc = ws.cmd("DOM.getDocument", {"depth": -1, "pierce": True})["root"]

            def walk(n):
                for sr in n.get("shadowRoots", []):
                    return sr
                for c_ in n.get("children", []):
                    r = walk(c_)
                    if r:
                        return r
            return walk(doc)

        def find(sel):
            root = shadow()
            if not root:
                return None
            nid = ws.cmd("DOM.querySelector", {"nodeId": root["nodeId"], "selector": sel})["nodeId"]
            return ws.cmd("DOM.describeNode", {"nodeId": nid})["node"]["backendNodeId"] if nid else None

        def fn(nid, body):
            oid = ws.cmd("DOM.resolveNode", {"backendNodeId": nid})["object"]["objectId"]
            r = ws.cmd("Runtime.callFunctionOn", {"objectId": oid, "functionDeclaration": "function(){" + body + "}", "returnByValue": True})
            return r.get("result", {}).get("value")

        def wait(pred, t=8):
            end = time.time() + t
            while time.time() < end:
                try:
                    v = pred()
                    if v:
                        return v
                except Exception:
                    pass
                time.sleep(0.25)
            return None

        def inflight_since():
            started, done = {}, set()
            for _, m in list(ws.events):
                p = m.get("params", {})
                if m["method"] == "Network.requestWillBeSent" and "/api/messages/since" in p["request"]["url"]:
                    started[p["requestId"]] = p["request"]["url"]
                elif m["method"] in ("Network.loadingFinished", "Network.loadingFailed"):
                    done.add(p["requestId"])
            return [u for r, u in started.items() if r not in done]

        def msgs():
            return fn(find(".body"), "return Array.from(this.querySelectorAll('.msg')).map(m=>m.firstChild.textContent)") or []

        def button(text):
            panel=find('.panel')
            return fn(panel, "const b=[...this.querySelectorAll('button')].find(b=>b.textContent==="+json.dumps(text)+"); if(b){b.click(); return true;} return false;") if panel else False
        def content():
            panel=find('.panel')
            return fn(panel, "return this.textContent") if panel else ""
        def loaded():
            node=find('img.attachment')
            return bool(node and fn(node, "return this.complete && this.naturalWidth===40"))
        def image_requests():
            return [m for _,m in list(ws.events) if m['method']=='Network.requestWillBeSent' and '/api/messages/image?' in m['params']['request']['url']]
        ws.cmd("Network.setCookie", {"name":srv.ACCOUNT_COOKIE, "value":ck['mia'], "url":base+'/', "path":"/", "secure":True})
        goto('/')
        assert wait(lambda:ev('!!window.MeowdokuMessenger'))
        ev('window.MeowdokuMessenger.open()')
        assert wait(lambda:find('button.item'))
        fn(find('button.item'), 'this.click()')
        assert wait(lambda:'Image locked' in content())
        check('locked image has no image fetch', not image_requests() and not find('img'))
        check('chooser disabled before opt-in', fn(find('input[type=file]'), 'return this.disabled'))
        assert wait(lambda:button('Enable images'))
        check('same existing message unlocks as decoded PNG', bool(wait(loaded)), content())
        thread=api('mia','/api/messages/thread?peer=pat')['messages']
        check('unlock kept original message id and count', len(thread)==1 and thread[0]['id']==receipt['id'])
        artifact=ROOT/'build'/'messenger-image-browser.png';artifact.parent.mkdir(exist_ok=True)
        artifact.write_bytes(base64.b64decode(ws.cmd('Page.captureScreenshot', {'format':'png'})['data']))
        print('Screenshot:', artifact)
        assert wait(lambda:button('Disable images'))
        check('opt-out re-hides existing image', bool(wait(lambda:'Image locked' in content() and not find('img'))))
        check('opt-out disables chooser', fn(find('input[type=file]'), 'return this.disabled'))
        assert wait(lambda:button('Enable images'))
        assert wait(loaded)
        ws.cmd('DOM.setFileInputFiles', {'files':[str(png)], 'backendNodeId':find('input[type=file]')})
        assert wait(lambda:'PNG attached' in content())
        fn(find('input[type=text]'), "this.value='Uploaded from browser'; this.dispatchEvent(new Event('input',{bubbles:true}));")
        assert button('Send')
        def uploaded():
            return [m for m in api('pat','/api/messages/thread?peer=mia')['messages'] if m['body']=='Uploaded from browser' and m.get('image',{}).get('locked') is False]
        check('file chooser sends real PNG with caption', bool(wait(uploaded)))
        # Delay an actual fetch response to exercise stale async close/back guards.
        ev("window.realFetch=window.fetch; window.fetch=async(...a)=>{const r=await window.realFetch(...a); if(String(a[0]).startsWith('/api/messages/thread'))await new Promise(resolve=>setTimeout(resolve,1000)); return r;}")
        button('Back'); assert wait(lambda:find('button.item'))
        fn(find('button.item'),'this.click()');button('Back');time.sleep(1.5)
        check('Back ignores delayed thread response', bool(find('button.item')) and not find('.msg'))
        fn(find('button.item'),'this.click()');ev('window.MeowdokuMessenger.close()');time.sleep(1.5)
        check('Close ignores delayed thread response', not find('.panel'))
        ev('window.fetch=window.realFetch;window.MeowdokuMessenger.open()')
        assert wait(lambda:find('button.item'));fn(find('button.item'),'this.click()')
        check('reopen works after stale response', bool(wait(loaded)))
    finally:
        subprocess.run(['taskkill','/PID',str(pid),'/T','/F'],capture_output=True)
        httpd.shutdown();httpd.server_close()
        # Only the exact disposable fixture directory, never user data.
        assert tmp.resolve().parent == Path(tempfile.gettempdir()).resolve() and tmp.name.startswith('mwidget_images_')
        shutil.rmtree(tmp, ignore_errors=True)
    print('SUMMARY:',sum(ok for _,ok in RESULTS),'/',len(RESULTS),'passed')
    return 0 if all(ok for _,ok in RESULTS) else 1

if __name__ == '__main__':
    sys.exit(main())

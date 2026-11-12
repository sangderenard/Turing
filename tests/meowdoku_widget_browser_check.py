"""Headless-browser check of examples/meowdoku/web/messenger-widget.js against a LOCAL test server (not collected by pytest).

Run: python tests/meowdoku_widget_browser_check.py
Stdlib only; drives Chrome/Edge over the DevTools protocol, headless, one browser process, killed by Windows pid at the end.
"""
from __future__ import annotations

import base64, http.client, json, os, secrets, shutil, socket, struct, subprocess, sys, tempfile, threading, time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku import server as srv  # noqa: E402

BROWSERS = ["C:/Program Files/Google/Chrome/Application/chrome.exe", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"]
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (" -- " + str(detail) if detail != "" else ""), flush=True)


def call(port, path, method="GET", cookie="", body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=40)
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
    tmp = Path(tempfile.mkdtemp(prefix="mwidget_"))
    web = tmp / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    (web / "index.html").write_text("<html>GAME</html>", encoding="utf-8")
    shutil.copy(ROOT / "examples/meowdoku/web/messenger-widget.js", web / "messenger-widget.js")
    hostile = "body * { display:none !important; color:red } button{position:fixed;top:0}"
    (web / "host.html").write_text(f"<!doctype html><html><head><meta charset=utf-8><style>{hostile}</style></head><body><p>host</p><button>hostbtn</button>"
                                   "<script type=module src=/messenger-widget.js data-messenger=allow></script></body></html>", encoding="utf-8")
    (web / "plain.html").write_text("<!doctype html><html><head><meta charset=utf-8></head><body><p>host</p>"
                                    "<script type=module src=/messenger-widget.js data-messenger=allow></script></body></html>", encoding="utf-8")
    (web / "noopt.html").write_text("<!doctype html><html><head><meta charset=utf-8></head><body><p>host</p>"
                                    "<script type=module src=/messenger-widget.js></script></body></html>", encoding="utf-8")
    sessions = srv.Sessions(web, None)
    sessions.lifecycle = SimpleNamespace(token="t" * 40, restore_for=lambda token: None)
    sessions.access.lookup = lambda url: None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    pw = {n: secrets.token_urlsafe(16) for n in ("mia", "pat")}
    for n in pw:
        sessions.accounts.ensure_user(n, pw[n])
    ck = {n: login(port, n, pw[n]) for n in pw}
    ch = {n: srv.ACCOUNT_COOKIE + "=" + ck[n] for n in pw}
    assert call(port, "/api/messages/contacts", "POST", ch["pat"], {"username": "mia", "action": "request"})[0] == 200
    assert call(port, "/api/messages/contacts", "POST", ch["mia"], {"username": "pat", "action": "accept"})[0] == 200
    XSS = '<img src=x onerror=window.__xss=1><a href="https://example.invalid">link</a>'
    for t in ("first", XSS, "third"):
        assert call(port, "/api/messages/send", "POST", ch["pat"], {"to": "mia", "body": t})[0] == 200
    base = f"http://127.0.0.1:{port}"

    exe = next((b for b in BROWSERS if os.path.exists(b)), None)
    print("browser:", exe, "(headless=new, CDP over stdlib websocket)")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dport = s.getsockname()[1]
    s.close()
    prof = tmp / "profile"
    proc = subprocess.Popen([exe, "--headless=new", f"--remote-debugging-port={dport}", f"--user-data-dir={prof}", "--no-first-run",
                             "--no-default-browser-check", "--disable-gpu", "--window-size=1000,700", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pid = proc.pid
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

        # (a) no opt-in, signed in
        ws.cmd("Network.setCookie", {"name": srv.ACCOUNT_COOKIE, "value": ck["mia"], "url": base + "/", "path": "/"})
        goto("/noopt.html")
        check("a. no data-messenger=allow -> no launcher", ev("document.body.querySelectorAll('div').length") == 0 and shadow() is None)
        # (b) signed out
        ws.cmd("Network.deleteCookies", {"name": srv.ACCOUNT_COOKIE, "url": base + "/"})
        goto("/host.html")
        check("b. signed-out visitor -> nothing rendered",
              ev("document.body.querySelectorAll('div').length") == 0 and shadow() is None and ev("typeof window.MeowdokuMessenger") == "undefined")
        # (c) signed in
        ws.cmd("Network.setCookie", {"name": srv.ACCOUNT_COOKIE, "value": ck["mia"], "url": base + "/", "path": "/"})
        ws.events.clear()
        goto("/host.html")
        L = wait(lambda: find("button.launch"))
        badge = find(".badge")
        btxt = fn(badge, "return this.textContent+'|'+this.hasAttribute('hidden')") if badge else None
        closed = bool(L) and fn(L, "return this.getAttribute('aria-expanded')") == "false" and not find(".panel")
        infl = inflight_since()
        check("c. launcher closed, badge=3, no long poll while closed", bool(L) and closed and btxt == "3|false" and not infl,
              f"badge={btxt} inflight_since={len(infl)}")
        # (h) style isolation
        if L:
            st = fn(L, "const c=getComputedStyle(this),r=this.getBoundingClientRect();return {d:c.display,p:c.position,col:c.color,w:r.width,h:r.height,top:r.top}")
            check("h. hostile host CSS does not hide/restyle launcher",
                  bool(st) and st["d"] != "none" and st["p"] == "fixed" and st["w"] == 48 and st["h"] == 48 and st["top"] > 100, st)
        else:
            check("h. hostile host CSS does not hide/restyle launcher", False, "no launcher found")
            print("   host div display:", ev("(()=>{const d=document.body.querySelector('div');return d?getComputedStyle(d).display:'no host div'})()"))
        goto("/plain.html")
        L = wait(lambda: find("button.launch"))
        if L:
            ws.events.clear()
            ev("window.MeowdokuMessenger && window.MeowdokuMessenger.open()")
            items = wait(lambda: find("button.item"))
            txt = fn(items, "return this.textContent") if items else ""
            check("d1. open: contact list shows peer", bool(items) and "pat" in txt, txt)
            if items:
                fn(items, "this.click()")
                wait(lambda: len(msgs()) == 3)
                time.sleep(3)
                nthread = sum(1 for _, m in list(ws.events) if m["method"] == "Network.requestWillBeSent" and "/api/messages/thread" in m["params"]["request"]["url"])
                nread = sum(1 for _, m in list(ws.events) if m["method"] == "Network.requestWillBeSent" and "/api/messages/read" in m["params"]["request"]["url"])
                check("d0. no thread/read request storm after selecting a peer", nthread <= 2 and nread <= 2, f"thread={nthread} read={nread} in ~3s")
                texts = msgs()
                check("d2. thread shows 3 messages", len(texts) == 3, len(texts))
                check("d3. hostile body literal text", XSS in texts, texts)
                check("d4. window.__xss undefined", ev("typeof window.__xss") == "undefined")
                inner = fn(find(".panel"), "return this.querySelectorAll('img,a,iframe,script').length")
                check("d5. no img/a elements inside widget", inner == 0, inner)
                bad = [m["params"]["request"]["url"] for _, m in list(ws.events)
                       if m["method"] == "Network.requestWillBeSent" and "example.invalid" in m["params"]["request"]["url"]]
                check("d6. no request to example.invalid", not bad, bad)
                # (e) send
                inp = find(".panel input")
                ws.cmd("DOM.focus", {"backendNodeId": inp})
                ws.cmd("Input.insertText", {"text": "hello from mia"})
                for t in ("rawKeyDown", "char", "keyUp"):
                    ws.cmd("Input.dispatchKeyEvent", {"type": t, "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13,
                                                      "text": "\r" if t != "keyUp" else ""})
                got = wait(lambda: [m for m in call(port, "/api/messages/thread?peer=mia", cookie=ch["pat"])[1]["messages"] if m["body"] == "hello from mia"])
                check("e. Enter sends (seen via API as pat)", bool(got))
                # (f) incoming
                call(port, "/api/messages/send", "POST", ch["pat"], {"to": "mia", "body": "live from pat"})
                t0 = time.time()
                seen = wait(lambda: "live from pat" in msgs(), 5)
                check("f. incoming message appears within 5 s", bool(seen), f"{time.time()-t0:.1f}s")
                check("f2. long poll in flight while open", len(inflight_since()) >= 1, len(inflight_since()))
                # (g) Escape
                ws.cmd("DOM.focus", {"backendNodeId": find(".panel input")})
                ws.events.clear()
                for t in ("rawKeyDown", "keyUp"):
                    ws.cmd("Input.dispatchKeyEvent", {"type": t, "key": "Escape", "code": "Escape", "windowsVirtualKeyCode": 27})
                closed = wait(lambda: find(".panel") is None, 3)
                time.sleep(1)
                aborted = any(m["method"] == "Network.loadingFailed" and m["params"].get("canceled") for _, m in list(ws.events))
                infl = inflight_since()
                check("g. Escape closes panel and aborts long poll", bool(closed) and not infl and aborted,
                      f"closed={bool(closed)} inflight={len(infl)} canceled={aborted}")
    finally:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        httpd.shutdown()
        httpd.server_close()
        time.sleep(1)
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        print("browser pid", pid, "remaining:", "yes" if str(pid) in out else "no")
        shutil.rmtree(tmp, ignore_errors=True)
    print("SUMMARY:", sum(ok for _, ok in RESULTS), "/", len(RESULTS), "passed")


if __name__ == "__main__":
    main()

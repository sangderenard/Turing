import json, ssl, sys, time, sqlite3, subprocess, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT = Path("C:/dev/Powershell/turing")
sys.path[:0] = [str(ROOT), str(ROOT / "examples"), "C:/dev/Powershell/nodus/tools"]
from meowdoku import authz, federation_peer as fp
import contract_service_peer as csp
DB = "C:/dev/Powershell/srv/state/sql/accounts.db"
CANVAS = r"C:\dev\Powershell\turing\build\meowdoku\canvas\messenger.canvas"
WEB = ROOT / "build/meowdoku/web"
CA = "C:/dev/Powershell/srv/state/secrets/tls/ca.crt"
URL = "wss://stichka.duckdns.org/ws/contract/gui.proof"
import socket
_ga = socket.getaddrinfo
def _pin(host, port, *a, **k):
    return _ga("127.0.0.1" if host == "stichka.duckdns.org" else host, port, *a, **k)
socket.getaddrinfo = _pin
T0 = time.monotonic()
rec = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "url": URL, "steps": []}
def step(name, ok, **kw):
    rec["steps"].append({"step": name, "ok": ok, "t": round(time.monotonic() - T0, 4), **kw}); print(name, ok, kw, flush=True)
def pids():
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    return {p: sorted({l.split()[-1] for l in out.splitlines() if f":{p} " in l and "LISTENING" in l}) for p in (47611, 47612)}
rec["pids_before"] = pids()
ctx = ssl.create_default_context()
ids = []
try:
    conn = sqlite3.connect(DB); uid = conn.execute("SELECT id FROM users WHERE username=?", ("svc-nodus-messenger",)).fetchone()[0]; conn.close()
    e_id, e_sec = authz.grant_peer(DB, "proof-engine", "gui.proof", "engine", "submit,decision", 1, "operator:claude-proof", service_user="svc-nodus-messenger"); ids.append(e_id)
    h_id, h_sec = authz.grant_peer(DB, "proof-host", "gui.proof", "host", "accept,result", 1, "operator:claude-proof", service_user="svc-meowdoku-a"); ids.append(h_id)
    step("1 grants", True, engine_grant=e_id, host_grant=h_id, engine_uid=uid)
    digests = fp.identity_digests(CANVAS, WEB)
    step("2 digests", True, graph=digests["graph_definition_digest"], api_count=len(digests["api_digests"]))
    t = time.monotonic()
    es = fp.connect(URL, e_id, e_sec, "engine", digests, ssl_context=ctx); hs = fp.connect(URL, h_id, h_sec, "host", digests, ssl_context=ctx)
    step("3a connect via apache (CA-verified)", True, seconds=round(time.monotonic() - t, 3), engine_hello=es.hello_id, host_hello=hs.hello_id)
    ev = {"admitted": [], "routed": [], "receipts": [], "offers": []}; times = {}
    def mark(k):
        times.setdefault(k, round(time.monotonic() - started, 4))
    class P:
        def admitted(self, i): ev["admitted"].append(i); mark("Admitted")
        def route(self, r): ev["routed"].append(list(r)); mark("Route"); return 1
    def dispatch(loan): mark("Offer"); ev["offers"].append(str(loan)); return 555
    def receipt(loan, acc): ev["receipts"].append(acc); mark("Receipt")
    started = time.monotonic()
    try:
        eng = csp.EnginePeer(es, P())
        with ThreadPoolExecutor(max_workers=1) as ex:
            host = csp.ServicePeer(hs, ex, dispatch, receipt, announce=False)
            sub = eng.submit((100, uid, 3, 4)); mark("Submit-sent")
            while time.monotonic() < started + 20:
                eng.poll(); host.poll()
                if ev["admitted"] and ev["routed"] and ev["receipts"]: break
                time.sleep(0.002)
            el = time.monotonic() - started
            ok = (sub and eng.error is None and host.error is None and ev["admitted"] == [100] and len(ev["routed"]) == 1
                  and ev["routed"][0][7] == 555 and ev["receipts"] == [True] and el < 20)
            step("3b round trip", bool(ok), elapsed=round(el, 4), frame_times=times, admitted=ev["admitted"], result=ev["routed"][0][7] if ev["routed"] else None,
                 receipts=ev["receipts"], engine_error=str(eng.error), host_error=str(host.error))
    finally:
        es.close(); hs.close()
    time.sleep(0.5)
    try:
        fp.connect(URL, e_id, e_sec + "x", "engine", digests, ssl_context=ctx); step("4a wrong secret refused", False, note="connected")
    except fp.FederationRefused as x: step("4a wrong secret refused", True, reason=x.reason)
    r = authz.revoke_peer(DB, e_id); step("4b revoke engine grant", bool(r))
    try:
        fp.connect(URL, e_id, e_sec, "engine", digests, ssl_context=ctx); step("4c revoked reconnect refused", False, note="connected")
    except fp.FederationRefused as x: step("4c revoked reconnect refused", True, reason=x.reason)
except BaseException as x:
    import traceback; step("ERROR", False, error=traceback.format_exc())
finally:
    for i in ids:
        try: authz.revoke_peer(DB, i)
        except Exception as x: print("revoke err", x)
    lp = [p for p in authz.list_peers(DB) if p["grant_id"] in ids] if ids else []
    step("5 final revoke", bool(ids) and all(p.get("revoked_at") for p in lp), peers=[{k: (v if k != "secret_hash" else "..") for k, v in p.items()} for p in lp])
    rec["pids_after"] = pids()
    step("6 host untouched", rec["pids_before"] == rec["pids_after"] and all(rec["pids_after"].values()), before=rec["pids_before"], after=rec["pids_after"])
    rec["overall"] = all(s["ok"] for s in rec["steps"])
    out = ROOT / f"build/meowdoku/receipts/federation-live-apache-{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8"); print("RECEIPT", out, "overall", rec["overall"])

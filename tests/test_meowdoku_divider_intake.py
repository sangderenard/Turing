"""Divider intake (examples/meowdoku/divider_intake.py, server.py [DIVIDER]) over an ephemeral loopback port and a temp state root.

No worker is started, nothing is compiled or fetched: jobs are planned by AST triage and their attempts stay queued."""
from __future__ import annotations

import http.client
import io
import json
import os
import socket
import sys
import threading
import time
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import divider_intake as di  # noqa: E402
from meowdoku import server as srv  # noqa: E402
from meowdoku import source_divider as sd  # noqa: E402

LOCAL = {"Host": "127.0.0.1"}
THREE = "def b(x: float) -> float:\n    return x * 2.0\n\ndef a(x: float) -> float:\n    return b(x) + 1.0\n\ndef c(x):\n    return x\n"


def _call(port, path, method="GET", headers=None, body=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    hdrs = {**LOCAL, **(headers or {})}
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=data, headers=hdrs)
    r = conn.getresponse()
    payload = r.read()
    info = SimpleNamespace(status=r.status, cookies=r.msg.get_all("Set-Cookie") or [], raw=payload, cache=r.getheader("Cache-Control"))
    conn.close()
    try:
        info.json = json.loads(payload)
    except ValueError:
        info.json = None
    return info


@pytest.fixture()
def live(tmp_path):
    web = Path(tmp_path) / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    sessions = srv.Sessions(web, None)
    sessions.lifecycle = SimpleNamespace(token="t" * 40, restore_for=lambda token: None)
    sessions.access.lookup = lambda url: None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield SimpleNamespace(port=httpd.server_address[1], sessions=sessions, root=Path(tmp_path))
    finally:
        httpd.shutdown()
        httpd.server_close()


def _signup(port, name):
    r = _call(port, "/api/account/signup", "POST", body={"username": name, "password": "correcthorse1"})
    assert r.status == 201
    return {"Cookie": next(c.split(";")[0] for c in r.cookies if c.startswith(srv.ACCOUNT_COOKIE))}


def _upload(live, who, name, data):
    return _call(live.port, "/api/divider/jobs", "POST", {**who, "Content-Type": "application/octet-stream", "X-Filename": name}, raw=data)


def _zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in entries:
            if isinstance(item, zipfile.ZipInfo):
                zf.writestr(item, "x")
            else:
                zf.writestr(item[0], item[1])
    return buf.getvalue()


def _files(root):
    return {p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file() and "sql" not in p.relative_to(root).parts}


def test_guest_is_refused_naming_compile_submit(live):
    r = _call(live.port, "/api/divider/jobs", "POST", {"Content-Type": "application/octet-stream", "X-Filename": "a.py"}, raw=b"x = 1\n")
    assert r.status == 403 and "compile_submit" in r.json["error"]
    assert _call(live.port, "/api/divider/jobs").status == 403


def test_single_file_job_has_the_expected_pieces(live):
    who = _signup(live.port, "div1")
    r = _upload(live, who, "m.py", THREE.encode())
    assert r.status == 201, r.json
    assert r.cache == "no-store"
    job = _call(live.port, f"/api/divider/jobs/{r.json['job_id']}", headers=who).json
    names = [p["names"] for p in job["pieces_detail"]]
    assert sorted(sum(names, [])) == ["a", "b", "c"]
    by_name = {p["names"][0]: p for p in job["pieces_detail"]}
    assert by_name["a"]["deps"] == [by_name["b"]["piece_id"]]
    assert by_name["a"]["annotated"] and not by_name["c"]["annotated"]
    assert by_name["c"]["attempts"][0]["state"] == "needs-annotation"
    assert by_name["b"]["state"] == "queued" and by_name["b"]["attempts"][0]["backend"] == "wasm"
    assert job["upload"] == {"kept": 1, "skipped": 0}
    listed = _call(live.port, "/api/divider/jobs", headers=who).json["jobs"]
    assert [j["job_id"] for j in listed] == [job["job_id"]] and listed[0]["active"]


def test_zip_job_keeps_source_suffixes_and_skips_the_rest(live):
    who = _signup(live.port, "div2")
    data = _zip([("pkg/a.py", THREE), ("pkg/readme.md", "hi"), ("pkg/data.bin", "zz"), ("pkg/logo.png", "zz"), ("setup.cfg", "[x]")])
    r = _upload(live, who, "repo.zip", data)
    assert r.status == 201, r.json
    area = live.sessions.state.jobs_dir / "divider" / r.json["job_id"] / "src"
    assert sorted(p.relative_to(area).as_posix() for p in area.rglob("*") if p.is_file()) == ["pkg/a.py", "pkg/readme.md", "setup.cfg"]
    assert r.json["upload"] == {"kept": 3, "skipped": 2}
    assert (live.sessions.state.jobs_dir / r.json["job_id"] / "source" / "pkg" / "a.py").is_file()


def _symlink():
    info = zipfile.ZipInfo("link.py")
    info.external_attr = 0o120777 << 16
    return info


@pytest.mark.parametrize("label,entries", [
    ("dotdot", [("ok.py", "x = 1\n"), ("../evil.py", "x = 2\n")]),
    ("absolute", [("/abs.py", "x = 1\n")]),
    ("drive", [("C:/evil.py", "x = 1\n")]),
    ("backslash-dotdot", [("a\\..\\..\\evil.py", "x = 1\n")]),
    ("symlink", [_symlink()]),
    ("too-many", [(f"n{i}.txt", "") for i in range(2001)]),
    ("bomb", [("a.py", "#" * 400000)]),
    ("oversize-entry", [("big.py", os.urandom(5 * 1024 * 1024))]),
    ("no-python", [("a.md", "hi")]),
])
def test_bad_zips_are_refused_and_write_nothing(live, label, entries):
    who = _signup(live.port, "div3" + label.replace("-", "")[:8])
    before = _files(live.root)
    stored = zipfile.ZIP_STORED if label == "oversize-entry" else zipfile.ZIP_DEFLATED
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", stored) as zf:
        for item in entries:
            if isinstance(item, zipfile.ZipInfo):
                zf.writestr(item, "x")
            else:
                zf.writestr(item[0], item[1])
    r = _upload(live, who, "x.zip", buf.getvalue())
    assert r.status == 400, (label, r.status, r.json)
    assert _files(live.root) == before
    assert not (live.root / "evil.py").exists()
    jobs = live.sessions.state.jobs_dir
    assert not jobs.exists() or not [p for p in jobs.rglob("*") if p.is_file()]      # no job tree left behind
    assert _call(live.port, "/api/divider/jobs", headers=who).json["jobs"] == []


def test_upload_type_and_size_rules(live):
    who = _signup(live.port, "div4")
    assert _upload(live, who, "notes.txt", b"hi").status == 400
    assert _upload(live, who, "x.zip", b"not a zip").status == 400
    assert _upload(live, who, "big.py", b"#" * (512 * 1024 + 1)).status == 413
    assert _call(live.port, "/api/divider/jobs", "POST", {**who, "Content-Type": "text/plain", "X-Filename": "a.py"}, raw=b"x").status == 415


def test_oversize_content_length_is_refused_before_the_body_is_read(live):
    who = _signup(live.port, "div5")
    for name, length in (("a.zip", 30 * 1024 * 1024), ("a.py", 600 * 1024)):
        conn = http.client.HTTPConnection("127.0.0.1", live.port, timeout=10)     # a read of the missing body would time out
        conn.putrequest("POST", "/api/divider/jobs", skip_host=True)
        for k, v in {**LOCAL, **who, "Content-Type": "application/octet-stream", "X-Filename": name, "Content-Length": str(length)}.items():
            conn.putheader(k, v)
        conn.endheaders(b"PK")
        r = conn.getresponse()
        r.read()
        assert r.status == 413, name
        conn.close()


@pytest.mark.parametrize("url,ref,ok", [
    ("https://github.com/owner/repo", None, True),
    ("https://github.com/owner/repo.git", "main", True),
    ("https://github.com/Own-er/re.po_x", "release/1.2-rc", True),
    ("http://github.com/owner/repo", None, False),
    ("https://gitlab.com/owner/repo", None, False),
    ("https://github.com.evil.com/owner/repo", None, False),
    ("https://github.com/owner", None, False),
    ("https://github.com/owner/repo/tree/main", None, False),
    ("https://user@github.com/owner/repo", None, False),
    ("https://github.com/owner/repo?x=1", None, False),
    ("https://github.com/owner/..", None, False),
    ("https://github.com/owner/repo", "bad ref", False),
    ("https://github.com/owner/repo", "a..b", False),
    ("https://github.com/owner/repo", "x" * 101, False),
    ("https://github.com/owner/repo", "--upload-pack=x", False),
    ("file:///etc/passwd", None, False),
    ("", None, False),
])
def test_repo_link_validation_makes_no_network_call(tmp_path, monkeypatch, url, ref, ok):
    def refuse(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    paths = sd.StatePaths(tmp_path / "state")
    if ok:
        job_id = di.accept_repo_link(paths, "account:1", url, ref)
        job = di.get_job(paths, job_id, "account:1")
        assert job["state"] == "awaiting_fetch" and job["pieces"] == 0 and "not fetched" in job["note"]
        assert job["repo"] == {"url": url[:-4] if url.endswith(".git") else url, "ref": ref, "fetched": False}
        assert not (paths.jobs_dir / job_id).exists()                         # nothing fetched, no source tree
    else:
        with pytest.raises(di.DividerError):
            di.accept_repo_link(paths, "account:1", url, ref)


def test_repo_link_over_http_records_a_job(live):
    who = _signup(live.port, "div6")
    r = _call(live.port, "/api/divider/jobs", "POST", who, {"repo": "https://github.com/owner/repo", "ref": "main"})
    assert r.status == 201 and r.json["state"] == "awaiting_fetch"
    assert _call(live.port, "/api/divider/jobs", "POST", who, {"repo": "https://evil.example/x"}).status == 400


def test_jobs_are_visible_to_their_owner_only(live):
    a, b = _signup(live.port, "div7"), _signup(live.port, "div8")
    job_id = _upload(live, a, "m.py", THREE.encode()).json["job_id"]
    assert _call(live.port, f"/api/divider/jobs/{job_id}", headers=b).status == 404
    assert _call(live.port, f"/api/divider/jobs/{job_id}/cancel", "POST", b).status == 404
    assert _call(live.port, "/api/divider/jobs", headers=b).json["jobs"] == []
    assert _call(live.port, f"/api/divider/jobs/{job_id}", headers=a).status == 200
    assert _call(live.port, "/api/divider/jobs/ffffffffffff", headers=a).status == 404
    owner = di.list_jobs(live.sessions.state, "nobody", admin=True)                 # an admin sees every job
    assert [j["job_id"] for j in owner] == [job_id]


def test_cancel_blocks_queued_attempts(live):
    who = _signup(live.port, "div9")
    job_id = _upload(live, who, "m.py", THREE.encode()).json["job_id"]
    r = _call(live.port, f"/api/divider/jobs/{job_id}/cancel", "POST", who)
    assert r.status == 200 and r.json["state"] == "cancelled" and not r.json["active"]
    assert {a["state"] for p in r.json["pieces_detail"] for a in p["attempts"]} <= {"blocked", "needs-annotation"}
    assert _call(live.port, f"/api/divider/jobs/{job_id}/cancel", "POST", who).status == 409


def test_limits_five_queued_and_twenty_per_day(live, tmp_path):
    who = _signup(live.port, "div10")
    for _ in range(5):
        assert _upload(live, who, "m.py", THREE.encode()).status == 201
    r = _upload(live, who, "m.py", THREE.encode())
    assert r.status == 429 and "5" in r.json["error"]
    assert _call(live.port, "/api/divider/jobs", "POST", who, {"repo": "https://github.com/o/r"}).status == 429
    # twenty per day: finished jobs still count
    paths = sd.StatePaths(tmp_path / "other")
    db = sd.connect(paths.accounts_db)
    for i in range(20):
        db.execute("INSERT INTO divider_jobs VALUES (?,?,?,?,?,?,?)", (f"{i:012x}", "account:9", "x", "o", time.time() - 60, "done", "wasm"))
    db.close()
    with pytest.raises(di.DividerError) as caught:
        di.accept_upload(paths, "account:9", "m.py", THREE.encode())
    assert caught.value.status == 429 and "20" in caught.value.message
    assert di.accept_upload(paths, "account:10", "m.py", THREE.encode())            # another account is unaffected

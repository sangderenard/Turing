"""/browse/<root>/<path...> (examples/meowdoku/listing.py, server.py [LISTING]) over an ephemeral loopback port and temp roots."""
from __future__ import annotations

import hashlib
import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import listing  # noqa: E402
from meowdoku import math_jobs  # noqa: E402
from meowdoku import server as srv  # noqa: E402

LOCAL = {"Host": "127.0.0.1"}                                  # a local direct caller: the edit tier (dev_ui), role guest
PUBLIC = {"X-Forwarded-For": "8.8.8.8", "X-Forwarded-Proto": "https", "Host": "cats.example.org"}   # dns origin: shader tier
JSON = {"Accept": "application/json"}
API_YAML = ("schema: turing-compiled-program-api-v1\nmodule: demo_mod\nlanguage: wasm\nentry: demo_entry\n"
            "entry_points:\n  - name: demo_entry\n    symbol: demo_entry\n    kind: numerical\n")
ABI = {"schema": "turing.mobile-wasm-abi.v1", "inputs": [{"name": "x"}, {"name": "y"}], "outputs": ["r"]}
ABOUT = "# Demo assets\n\nThese are hand-made test files.\n\n- first item\n- second `code` item\n"


def _call(port, path, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("GET", path, headers={**LOCAL, **(headers or {})})
    r = conn.getresponse()
    raw = r.read()
    out = SimpleNamespace(status=r.status, raw=raw, headers={k.lower(): v for k, v in r.getheaders()}, cookies=r.msg.get_all("Set-Cookie") or [])
    conn.close()
    try:
        out.json = json.loads(raw)
    except ValueError:
        out.json = None
    return out


def _signup(port, name):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("POST", "/api/account/signup", body=json.dumps({"username": name, "password": "correcthorse1"}).encode(),
                 headers={**LOCAL, "Content-Type": "application/json"})
    r = conn.getresponse()
    r.read()
    assert r.status == 201
    cookie = next(c.split(";")[0] for c in (r.msg.get_all("Set-Cookie") or []) if c.startswith(srv.ACCOUNT_COOKIE))
    conn.close()
    return {"Cookie": cookie}


@pytest.fixture()
def world(tmp_path):
    web = Path(tmp_path) / "build" / "meowdoku" / "web"
    (web / "sub").mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    (web / "hello.txt").write_bytes(b"hello listing\n")
    (web / "demo.api.yaml").write_text(API_YAML, encoding="utf-8")
    (web / "demo.abi.json").write_text(json.dumps(ABI), encoding="utf-8")
    (web / "demo.wasm").write_bytes(b"\0asm\1\0\0\0")
    (web / "_about.md").write_text(ABOUT, encoding="utf-8")
    (web / "page.html").write_text("<script>alert(1)</script>", encoding="utf-8")
    (web / "sub" / "inner.txt").write_text("inner", encoding="utf-8")
    sessions = srv.Sessions(web, None)
    sessions.lifecycle = SimpleNamespace(token="t" * 40, restore_for=lambda token: None)
    sessions.access.lookup = lambda url: None
    listing.LIMITER = listing.RateLimiter(rate=1000.0)        # the rate-limit test builds its own bucket
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield SimpleNamespace(port=httpd.server_address[1], sessions=sessions, web=web)
    httpd.shutdown()
    httpd.server_close()


def test_guest_and_shader_tier_get_403_for_every_root(world):
    for root in ("", "site/", "math/", "docs/"):
        r = _call(world.port, f"/browse/{root}", PUBLIC)
        assert r.status == 403, root
    assert _call(world.port, "/browse/site/hello.txt", PUBLIC).status == 403
    assert _call(world.port, "/browse/", PUBLIC).headers["content-type"] == "application/json"


def test_operator_lists_site_as_json_with_hashes_and_declared_descriptions(world):
    r = _call(world.port, "/browse/site/", JSON)
    assert r.status == 200 and r.headers["cache-control"] == "no-store" and r.headers["x-content-type-options"] == "nosniff"
    doc = r.json
    assert doc["schema"] == "turing.listing.v1" and doc["root"] == "site" and doc["path"] == ""
    assert doc["capabilities_required"] == ["dev_ui"] and doc["truncated"] is False
    by = {e["name"]: e for e in doc["entries"]}
    assert by["hello.txt"]["sha256"] == hashlib.sha256(b"hello listing\n").hexdigest() and by["hello.txt"]["bytes"] == 14
    assert by["hello.txt"]["content_type"] == "text/plain" and by["hello.txt"]["href"] == "/browse/site/hello.txt"
    assert by["sub"]["kind"] == "dir" and by["sub"]["href"] == "/browse/site/sub/"
    api = by["demo.api.yaml"]["describe"]
    assert api["module"] == "demo_mod" and api["language"] == "wasm" and api["entry"] == "demo_entry" and api["entry_points"] == ["demo_entry"]
    assert by["demo.abi.json"]["describe"] == {"kind": "abi", "schema": "turing.mobile-wasm-abi.v1", "inputs": ["x", "y"], "outputs": ["r"]}
    assert by["hello.txt"]["describe"] is None                  # no guessing
    assert doc["narration"]["source"] == "_about.md" and "Demo assets" in doc["narration"]["text"]
    speak = doc["narration"]["speak"]
    assert "#" not in speak and "`" not in speak and "hand-made test files" in speak
    assert doc["parents"] == [{"name": "browse", "href": "/browse/"}]
    sub = _call(world.port, "/browse/site/sub/", JSON).json
    assert [p["href"] for p in sub["parents"]] == ["/browse/", "/browse/site/"]


def test_text_format_lines_parse_and_next_section_lists_subdirectories(world):
    r = _call(world.port, "/browse/site/?format=text")
    assert r.status == 200 and r.headers["content-type"].startswith("text/plain")
    lines = r.raw.decode("utf-8").splitlines()
    assert lines[0] == "# site"
    rows = [ln.split("\t") for ln in lines if ln.startswith(("file\t", "dir\t"))]
    hello = next(row for row in rows if row[3] == "hello.txt")
    assert hello[:3] == ["file", "14", hashlib.sha256(b"hello listing\n").hexdigest()[:12]] and len(hello) == 5
    assert any(row[3] == "demo.api.yaml" and "demo_mod" in row[4] for row in rows)
    nxt = lines[lines.index("next:") + 1:]
    assert "json\t/browse/site/sub/?format=json" in nxt and "text\t/browse/site/sub/?format=text" in nxt
    assert _call(world.port, "/browse/site/", {"Accept": "text/plain"}).raw.startswith(b"# site")      # Accept default
    assert _call(world.port, "/browse/site/?format=nope").status == 400


def test_html_returns_the_page_and_files_are_served_with_their_types(world):
    page = _call(world.port, "/browse/site/")
    assert page.status == 200 and page.headers["content-type"].startswith("text/html") and b"/browse.js" in page.raw
    assert "script-src 'self'" in page.headers["content-security-policy"]
    txt = _call(world.port, "/browse/site/hello.txt")
    assert txt.status == 200 and txt.raw == b"hello listing\n" and txt.headers["content-type"].startswith("text/plain")
    assert txt.headers["content-disposition"] == "inline"
    wasm = _call(world.port, "/browse/site/demo.wasm")
    assert wasm.headers["content-type"] == "application/wasm" and wasm.raw[:4] == b"\0asm"
    assert _call(world.port, "/browse/site/boards.json").headers["content-type"] == "application/json"
    assert _call(world.port, "/browse/site/page.html").headers["content-type"].startswith("text/plain")    # html is never rendered from here
    assert _call(world.port, "/browse/site/sub/inner.txt").raw == b"inner"
    assert _call(world.port, "/browse/site/missing.txt").status == 404


def test_traversal_and_escapes_are_refused(world, tmp_path):
    (Path(tmp_path) / "secret.txt").write_text("secret", encoding="utf-8")
    for bad in ("/browse/site/..%2fsecret.txt", "/browse/site/%2e%2e/secret.txt", "/browse/site/../secret.txt",
                "/browse/site//secret.txt", "/browse/site/C:/Windows/win.ini", "/browse/site/hello.txt:stream",
                "/browse/site/hello.txt::$DATA", "/browse/site/sub%5c..%5c..%5csecret.txt", "/browse/site/%2fetc%2fpasswd",
                "/browse/%2e%2e/", "/browse/site/hello.txt."):
        r = _call(world.port, bad)
        assert r.status in (403, 404) and b"secret" not in r.raw, bad
    assert _call(world.port, "/browse/site/..%2fsecret.txt").status == 403
    assert _call(world.port, "/browse/etc/").status == 404
    link = world.web / "outside"
    try:
        link.symlink_to(Path(tmp_path), target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted here")
    assert _call(world.port, "/browse/site/outside/secret.txt").status == 404
    names = [e["name"] for e in _call(world.port, "/browse/site/", JSON).json["entries"]]
    assert "outside" not in names


def _make_job(world, member, text="r = x + y"):
    conn = http.client.HTTPConnection("127.0.0.1", world.port, timeout=30)
    conn.request("POST", "/api/mathlab/jobs", body=json.dumps({"text": text, "targets": ["numpy"]}).encode(),
                 headers={**LOCAL, **member, "Content-Type": "application/json"})
    r = conn.getresponse()
    job_id = json.loads(r.read())["job_id"]
    conn.close()
    directory = math_jobs.job_dir(world.sessions.state, job_id)
    (directory / "native").mkdir(parents=True)
    (directory / "law.math.txt").write_text(text, encoding="utf-8")
    (directory / "law.numpy.py").write_text("def user_law(x, y):\n    return x + y\n", encoding="utf-8")
    (directory / "law.native.piece").write_bytes(b"piece")
    (directory / "native" / "user_law.dll").write_bytes(b"MZ")
    (directory / "products.json").write_text(json.dumps({
        "schema": "turing.math-lab-products.v1", "name": "law", "equations": [text], "inputs": ["x", "y"], "outputs": ["r"],
        "files": {"numpy": "law.numpy.py"},
        "targets": {"numpy": {"state": "complete", "seconds": 0.5, "file": "law.numpy.py"},
                    "native": {"state": "complete", "seconds": 2.0, "file": "native/user_law.dll", "piece_file": "law.native.piece"}}}), encoding="utf-8")
    return job_id


def test_member_lists_only_their_own_math_jobs_and_native_files_stay_hidden(world):
    a, b = _signup(world.port, "cat1"), _signup(world.port, "cat2")
    job_a, job_b = _make_job(world, a), _make_job(world, b, "q = x * y")
    root = _call(world.port, "/browse/math/", {**JSON, **a}).json
    assert [e["name"] for e in root["entries"]] == [job_a] and root["capabilities_required"] == ["compile_submit"]
    job = _call(world.port, f"/browse/math/{job_a}/", {**JSON, **a}).json
    names = {e["name"] for e in job["entries"]}
    assert {"products.json", "law.numpy.py"} <= names and not any(n.endswith((".piece", ".dll")) or n == "native" for n in names)
    assert "law.math.txt" not in names                          # not a served file, exactly as the file endpoint
    products = next(e for e in job["entries"] if e["name"] == "products.json")["describe"]
    assert products["kind"] == "products" and products["inputs"] == ["x", "y"] and products["targets"] == {"numpy": "complete", "native": "complete"}
    assert job["narration"]["source"] == "generated:products.json+law.math.txt"
    assert "r = x + y" in job["narration"]["text"] and "native: complete in 2.0 seconds" in job["narration"]["text"] and job["narration"]["speak"]
    assert _call(world.port, f"/browse/math/{job_a}/law.numpy.py", a).status == 200
    for name in ("law.native.piece", "native", "law.math.txt"):
        assert _call(world.port, f"/browse/math/{job_a}/{name}", a).status == 404, name
    assert _call(world.port, f"/browse/math/{job_b}/", {**JSON, **a}).status == 404            # another account's job
    assert _call(world.port, f"/browse/math/{job_b}/law.numpy.py", a).status == 404
    assert _call(world.port, f"/browse/math/{job_a}/", JSON).status == 403                     # no login: no compile_submit


def test_docs_root_describes_each_markdown_file(world):
    doc = _call(world.port, "/browse/docs/", JSON).json
    by = {e["name"]: e for e in doc["entries"]}
    assert "README.md" in by and by["README.md"]["describe"]["kind"] == "document" and by["README.md"]["describe"]["heading"]
    assert "deploy" in by and by["deploy"]["kind"] == "dir"
    assert all(e["name"].endswith(".md") or e["kind"] == "dir" for e in doc["entries"])
    assert _call(world.port, "/browse/docs/README.md").headers["content-type"].startswith("text/markdown")
    assert _call(world.port, "/browse/docs/server.py").status == 404


def test_truncation_hash_budget_and_rate_limit(world, monkeypatch):
    big = world.web / "many"
    big.mkdir()
    for i in range(2100):
        (big / f"f{i:04d}.txt").write_bytes(b"x")
    doc = _call(world.port, "/browse/site/many/", JSON).json
    assert doc["truncated"] is True and len(doc["entries"]) == 2000
    monkeypatch.setattr(listing, "HASH_BUDGET", 10)
    monkeypatch.setattr(listing, "HASHES", listing.HashCache())
    small = world.web / "budget"
    small.mkdir()
    for i in range(30):
        (small / f"b{i:02d}.txt").write_bytes(b"0123456789")
    # the handler's Context holds the module-level cache by default argument; a fresh one makes the budget bite
    doc = _call(world.port, "/browse/site/budget/", JSON).json
    deferred = [e for e in doc["entries"] if e.get("hash_deferred")]
    assert deferred and all(e["sha256"] is None for e in deferred) and doc["hash_deferred"] is True
    assert any(e["sha256"] for e in doc["entries"])
    bucket = listing.RateLimiter(rate=10.0)
    assert sum(bucket.allow("c") for _ in range(25)) == 10 and bucket.allow("other")
    listing.LIMITER = listing.RateLimiter(rate=1.0)
    statuses = [_call(world.port, "/browse/site/", JSON).status for _ in range(3)]
    assert statuses[0] == 200 and 429 in statuses


def test_static_directory_request_redirects_for_dev_ui_and_files_are_unchanged(world):
    r = _call(world.port, "/sub/")
    assert r.status == 302 and r.headers["location"] == "/browse/site/sub/"
    assert _call(world.port, "/hello.txt").raw == b"hello listing\n"
    assert _call(world.port, "/sub/inner.txt").raw == b"inner"
    assert _call(world.port, "/sub/", PUBLIC).status != 200

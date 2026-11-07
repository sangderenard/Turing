"""Meowdoku server lifecycle: session dumps and restores, the admin gate, hashed session files, the page's scrubbed copy, the supervisor."""

import http.client
import json
import shutil
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku import lifecycle  # noqa: E402
from meowdoku.puzzle import generate  # noqa: E402
from meowdoku import server as srv  # noqa: E402


def _board(n=5, seed=1):
    p = generate(n, seed)
    return {"id": "t5", "n": n, "level": "small", "type": "t", "region": p.region.astype(int).tolist(),
            "solution": p.solution.astype(int).tolist(), "hues": [k / n for k in range(n)]}


@pytest.fixture()
def web(tmp_path):
    web = tmp_path / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [_board()], "levels": {"small": 1}}), encoding="utf-8")
    return web


def _table(web):
    return srv.Table(web, None)


# ---------------------------------------------------------------- snapshot round trips
def test_meowdoku_round_trip_replays_through_the_law(web):
    a = _table(web)
    a.new({"game": "meowdoku", "id": "t5", "palette": "Pastel"})
    solution = np.asarray(_board()["solution"]).reshape(5, 5)
    cats = [tuple(int(v) for v in rc) for rc in np.argwhere(solution > 0)]
    blank = tuple(int(v) for v in np.argwhere(solution == 0)[0])
    a.game.cell(*blank, "double")                        # a wrong cat: costs a heart
    a.game.cell(*cats[0], "double")
    a.game.cell(*cats[1], "double")
    other = [tuple(int(v) for v in rc) for rc in np.argwhere(solution == 0)][3]
    a.game.cell(*other, "secondary")                     # an X
    a.game.power("mouse")                                # mice, and a spent power-up
    dump = lifecycle.dump_table(a)
    assert dump["mode"] == "exact" and dump["kind"] == "meowdoku"
    b = _table(web)
    assert lifecycle.restore_table(b, json.loads(json.dumps(dump))) == "exact"
    assert np.array_equal(a.game.session.cat, b.game.session.cat)
    assert np.array_equal(a.game.session.x_mark, b.game.session.x_mark)
    assert a.game.session.hearts == b.game.session.hearts == 2
    assert (a.game.moves, a.game.stock, a.game.mouse_cells, a.game.palette_name) == (b.game.moves, b.game.stock, b.game.mouse_cells, b.game.palette_name)
    assert lifecycle.dump_rev(lifecycle.dump_table(b)) == lifecycle.dump_rev(dump)


def test_meowdoku_lost_and_won_states_restore(web):
    a = _table(web)
    a.new({"game": "meowdoku", "id": "t5"})
    blank = tuple(int(v) for v in np.argwhere(np.asarray(_board()["solution"]).reshape(5, 5) == 0)[0])
    for _ in range(3):
        a.game.cell(*blank, "double")
    assert a.game.session.lost
    b = _table(web)
    lifecycle.restore_table(b, lifecycle.dump_table(a))
    assert b.game.session.lost and b.game.session.hearts == 0


def test_fountain_round_trip_is_the_server_side_configuration(web):
    a = _table(web)
    a.new({"game": "fountain3d", "aspect": 1.6})
    for payload in ({"type": "fountain_size", "size": "small"}, {"type": "fountain_choice", "name": "shape", "value": "hexagon"},
                    {"type": "fountain_choice", "name": "fluid", "value": "glycerine"}, {"type": "fountain_mode", "mode": 3},
                    {"type": "fountain_speed", "speed": 2}, {"type": "fountain_param", "name": "stiffness", "value": 700}):
        try:
            a.game.do(payload)
        except ValueError:
            pass
    a.game.mix = 0.25
    dump = lifecycle.dump_table(a)
    assert dump["mode"] == "exact"
    b = _table(web)
    lifecycle.restore_table(b, json.loads(json.dumps(dump)))
    for field in ("size", "mix", "mode", "speed", "strata", "params", "shape", "fluid", "inlet", "run", "aspect", "phase", "epoch"):
        assert getattr(a.game, field) == getattr(b.game, field), field


def test_wordle_chess_and_fresh_games(web):
    a = _table(web)
    a.new({"game": "wordle", "seed": 7})
    a.game.text("crane"); a.game.submit()
    a.game.text("sl")
    b = _table(web)
    lifecycle.restore_table(b, lifecycle.dump_table(a))
    assert (b.game.answer, b.game.guesses, b.game.entry) == (a.game.answer, a.game.guesses, a.game.entry)
    a.new({"game": "chess"})
    a.game.cell(6, 4, "primary"); a.game.cell(4, 4, "primary")
    b = _table(web)
    lifecycle.restore_table(b, lifecycle.dump_table(a))
    assert b.game.pieces == a.game.pieces and b.game.turn == a.game.turn == "black" and b.game.rights == a.game.rights and b.game.ep == a.game.ep
    a.new({"game": "space_invaders"})
    dump = lifecycle.dump_table(a)
    assert dump["mode"] == "fresh"
    b = _table(web)
    assert lifecycle.restore_table(b, dump) == "fresh" and b.game.game == "space_invaders"


def test_restore_refuses_forged_or_malformed_copies(web):
    t = _table(web)
    good = {"v": 1, "kind": "meowdoku", "mode": "exact", "new": {"game": "meowdoku", "id": "t5"}, "game": {"cats": [9999]}}
    for bad in (None, [], {"v": 2}, {**good, "game": {"cats": [9999]}}, {**good, "kind": "chess"}, {**good, "new": {"game": "meowdoku", "id": "nope"}},
                {**good, "game": {"cats": "x"}}, {**good, "kind": "Bad Kind!"}):
        with pytest.raises(lifecycle.LifecycleError):
            lifecycle.restore_table(t, bad)
    with pytest.raises(lifecycle.LifecycleError) as big:
        lifecycle.validate_dump({**good, "game": {"pad": "x" * 70000}})
    assert big.value.status == 413


# ---------------------------------------------------------------- hashed files, scrubbing
def test_session_files_are_named_by_hash_and_hold_no_cookie(web, tmp_path):
    cookie = "RAW-COOKIE-VALUE-abcdefghijklmnop"
    t = _table(web)
    t.player, t.player_token = "alice", "PLAYER-TOKEN-xyz"
    t.new({"game": "meowdoku", "id": "t5"})
    store = lifecycle.SessionStore(tmp_path / "sessions")
    dump = lifecycle.dump_table(t)
    assert store.save(cookie, dump) and not store.save(cookie, dump)           # unchanged: not rewritten
    files = list((tmp_path / "sessions").iterdir())
    assert [f.name for f in files] == [lifecycle.cookie_hash(cookie) + ".json"]
    blob = files[0].read_text(encoding="utf-8")
    for secret in (cookie, "PLAYER-TOKEN-xyz", "alice"):
        assert secret not in blob and secret not in files[0].name
    assert store.load(cookie) == dump and store.load("another-cookie") is None


def test_dump_strips_credentialish_keys():
    dump = lifecycle.validate_dump({"v": 1, "kind": "wordle", "mode": "exact", "new": {"game": "wordle", "player": "x"},
                                    "game": {"guesses": [], "session_token": "s", "nested": {"api_key": "k", "ok": 1}}, "player_token": "t"})
    assert "player_token" not in dump and "player" not in dump["new"]
    assert dump["game"] == {"guesses": [], "nested": {"ok": 1}}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_copy_validation_in_node(tmp_path):
    script = tmp_path / "check.mjs"
    script.write_text(f"""
import {{ validateCopy, stripSecrets, formatCountdown }} from {json.dumps((ROOT / 'examples/meowdoku/web/lifecycle.js').as_uri())};
const ok = {{ v: 1, rev: "abc", savedAt: 1, dump: {{ v: 1, kind: "wordle", mode: "exact", new: {{ game: "wordle", player: "bob" }},
  game: {{ guesses: [], session: "s", deep: {{ player_token: "t", fine: 1 }} }}, player: "bob", stranger: 1 }} }};
const c = validateCopy(JSON.stringify(ok));
const text = JSON.stringify(c);
if (!c || /bob|session|player|stranger|"t"/.test(text)) throw new Error("not scrubbed: " + text);
if (c.dump.game.deep.fine !== 1) throw new Error("lost a good key");
for (const bad of [null, "{{", "[]", JSON.stringify({{ ...ok, v: 2 }}), JSON.stringify({{ ...ok, dump: {{ ...ok.dump, kind: "Bad!" }} }}),
   JSON.stringify({{ ...ok, dump: {{ ...ok.dump, game: {{ pad: "x".repeat(70000) }} }} }})]) if (validateCopy(bad) !== null) throw new Error("accepted " + String(bad).slice(0, 40));
if (formatCountdown(45) !== "0:45" || formatCountdown(61.2) !== "1:02") throw new Error("countdown");
console.log("ok");
""", encoding="utf-8")
    done = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and "ok" in done.stdout, done.stdout + done.stderr


# ---------------------------------------------------------------- the admin gate
def test_admin_authorize_rejects_non_localhost_proxies_and_bad_tokens():
    token = "t" * 43
    auth = lifecycle.admin_authorize
    assert auth("127.0.0.1", "127.0.0.1:8766", False, token, token)[0]
    assert auth("::1", "[::1]:8766", False, token, token)[0]
    assert auth("192.168.1.9", "127.0.0.1:8766", False, token, token)[1] == 403
    assert auth("127.0.0.1", "example.com", False, token, token)[1] == 403            # a proxy's Host
    assert auth("127.0.0.1", "localhost:8766", True, token, token)[1] == 403          # forwarding headers
    assert auth("127.0.0.1", "localhost:8766", False, "wrong", token)[1] == 401
    assert auth("127.0.0.1", "localhost:8766", False, None, token)[1] == 401
    assert auth("127.0.0.1", "localhost:8766", False, "", "")[1] == 401


def test_admin_endpoint_over_http(web):
    sessions = srv.Sessions(web, None)
    lc = lifecycle.Lifecycle(sessions, web.parent, 8851, lambda: _table(web))
    sessions.lifecycle = lc
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]

    def call(headers, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("POST", "/api/admin/lifecycle", json.dumps(body or {"op": "status"}), headers)
        r = conn.getresponse()
        return r.status, json.loads(r.read())
    try:
        assert call({})[0] == 401
        assert call({"X-Meowdoku-Admin": "nope"})[0] == 401
        assert call({"X-Meowdoku-Admin": lc.token, "X-Forwarded-For": "8.8.8.8"})[0] == 403
        status, out = call({"X-Meowdoku-Admin": lc.token}, {"op": "restart", "in": 600})
        assert status == 200 and 590 < out["seconds"] <= 600
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", "/api/lifecycle")                                          # public, no token: the countdown only
        public = json.loads(conn.getresponse().read())
        assert public["restart_at"] and "token" not in json.dumps(public)
        assert call({"X-Meowdoku-Admin": lc.token}, {"op": "restart", "in": 99999})[0] == 400
        assert call({"X-Meowdoku-Admin": lc.token}, {"op": "cancel"})[1]["restart_at"] is None
        assert lc.token not in (web.parent / "launch_8851.json").read_text() if (web.parent / "launch_8851.json").exists() else True
    finally:
        httpd.shutdown()
        lc.stopping = True


# ---------------------------------------------------------------- the supervisor
def test_supervisor_relaunches_the_recorded_argv_and_env(tmp_path):
    log = tmp_path / "runs.log"
    child = tmp_path / "child.py"
    child.write_text(
        "import os, sys\n"
        f"log = open({str(log)!r}, 'a')\n"
        "log.write(repr((sys.argv[1:], os.environ.get('MEOWDOKU_FOUNTAIN_WASM'), os.getcwd())) + '\\n'); log.close()\n"
        f"n = len(open({str(log)!r}).read().splitlines())\n"
        "sys.exit({1: 75, 2: 3}.get(n, 0))\n", encoding="utf-8")
    cwd = tmp_path / "work"
    cwd.mkdir()
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"argv": [sys.executable, str(child), "--port", "9999"], "cwd": str(cwd), "env": {"MEOWDOKU_FOUNTAIN_WASM": "0"}}), encoding="utf-8")
    lines, sleeps = [], []
    code = lifecycle.supervise(launch, out=lines.append, sleep=sleeps.append, relaunch_delay=0.1)
    assert code == 0
    runs = [eval(line) for line in log.read_text().splitlines()]
    assert len(runs) == 3 and all(r[0] == ["--port", "9999"] and r[1] == "0" and Path(r[2]).resolve() == cwd.resolve() for r in runs)
    assert sleeps == [0.1, 2.0]                           # a scheduled exit relaunches at once; a crash backs off
    assert any("scheduled exit" in l for l in lines) and any("crash (exit 3)" in l for l in lines)


def test_record_launch_keeps_the_configuration_not_secrets(web, tmp_path, monkeypatch):
    monkeypatch.setenv("MEOWDOKU_FOUNTAIN_WASM", "0")
    monkeypatch.setenv("MEOWDOKU_LICHESS_TOKEN", "sekrit")
    paths = lifecycle.state_paths(tmp_path, 8851)
    record = lifecycle.record_launch(paths, 8851, "127.0.0.1", web)
    saved = json.loads(paths["launch"].read_text())
    assert saved["env"]["MEOWDOKU_FOUNTAIN_WASM"] == "0" and "MEOWDOKU_LICHESS_TOKEN" not in saved["env"]
    assert saved["port"] == 8851 and saved["python"] == sys.executable and saved["argv"] and saved["cwd"] == record["cwd"]

"""The manual's hard copy as a refresh check, and (opt-in) the cron tool asking this server's host to run it.

The first tests are pure: a map is baked into a folder, an unchanged map writes nothing, a changed map replaces the
folder. The last starts the real Nodus canvas host with the cron tool on a temporary canvas when
NODUS_HEADLESS_TEST_DLL names the built library; the schedule is every minute, so it waits up to about 80 s.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku import rest_api, scheduled_tasks, studio_product_nodus  # noqa: E402
from meowdoku.living_data_query import GraphView  # noqa: E402

NODUS = ROOT.parent / "nodus"


def small_map(extra=False):
    regions = [{"identity": "m/a", "kind": "room", "name": "A", "buildings": []},
               {"identity": "m/b", "kind": "room", "name": "B", "buildings": []}]
    tracks = [{"source": "m/a", "target": "m", "relationship": "contained-by"},
              {"source": "m/b", "target": "m", "relationship": "contained-by"}]
    if extra:
        regions.append({"identity": "m/c", "kind": "room", "name": "C", "buildings": []})
        tracks.append({"source": "m/c", "target": "m", "relationship": "contained-by"})
        regions.pop(0)
        tracks.pop(0)
    return {"schema": "abstract-ui-div-map-v0", "identity": "m", "name": "Small", "regions": regions, "tracks": tracks}


def refresh(web, document):
    return scheduled_tasks.refresh_hard_copy(web, name="hard", document=document, site=rest_api.site, graph_view=GraphView)


def test_the_hard_copy_is_written_once_and_left_alone_while_the_map_is_unchanged(tmp_path):
    changed, sentence = refresh(tmp_path, small_map())
    assert changed and "rebuilt" in sentence
    folder = tmp_path / "hard"
    stamp = json.loads((folder / scheduled_tasks.STAMP).read_text(encoding="utf-8"))
    assert stamp["pages"] == 4 and stamp["digest"] == rest_api.site.digest(small_map())
    assert (folder / "index.html").is_file() and len(list(folder.glob("*.html"))) == 4
    before = {p.name: p.stat().st_mtime_ns for p in folder.iterdir()}
    changed, sentence = refresh(tmp_path, small_map())
    assert not changed and "already matches" in sentence
    assert {p.name: p.stat().st_mtime_ns for p in folder.iterdir()} == before, "an unchanged map writes nothing"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["hard"], "no working folders are left behind"


def test_a_changed_map_replaces_the_whole_folder(tmp_path):
    refresh(tmp_path, small_map())
    gone = rest_api.site.address("m/a") + ".html"
    assert (tmp_path / "hard" / gone).is_file()
    changed, _sentence = refresh(tmp_path, small_map(extra=True))
    assert changed
    assert not (tmp_path / "hard" / gone).exists(), "the page of a place that no longer exists goes away"
    assert (tmp_path / "hard" / (rest_api.site.address("m/c") + ".html")).is_file()
    (tmp_path / "hard" / "index.html").unlink()
    changed, _sentence = refresh(tmp_path, small_map(extra=True))
    assert changed, "a damaged copy is rebuilt even when the map is unchanged"


def test_the_manual_itself_bakes(tmp_path):
    if rest_api.manual is None:
        pytest.skip("the SpeakToMe checkout has no manual")
    changed, sentence = scheduled_tasks.refresh_hard_copy(tmp_path, name=scheduled_tasks.MANUAL_FOLDER,
                                                          document=rest_api.manual.as_map(), site=rest_api.site,
                                                          graph_view=GraphView)
    assert changed and (tmp_path / "manual" / "index.html").is_file(), sentence


def test_the_nightly_backup_is_text_and_databases_with_no_rolling_window(tmp_path):
    """The owner's policy: "just the non binary files and the database nightly no rolling window"."""
    import sqlite3
    from meowdoku import state_backup
    state = tmp_path / "state"
    (state / "sql").mkdir(parents=True)
    db = sqlite3.connect(state / "sql" / "accounts.db")
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)")
    db.execute("INSERT INTO users (username) VALUES ('ann')")
    db.commit()
    db.close()
    photos = state / "content" / "marble_photos" / "job"
    photos.mkdir(parents=True)
    (photos / "exposure.npz").write_bytes(b"PK\x03\x04" + bytes(range(256)) * 16)
    (photos / "preview.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    (photos / "status.json").write_text('{"state": "complete"}', encoding="utf-8")
    (state / "content" / "profiles.json").write_text('{"name": "caf\u00e9"}', encoding="utf-8")
    (state / "jobs" / "cron").mkdir(parents=True)
    (state / "jobs" / "cron" / "cron.table").write_text("x * * * * *\n", encoding="utf-8")
    assert state_backup.is_text_file(photos / "status.json") and not state_backup.is_text_file(photos / "preview.png")
    out = tmp_path / "backups"
    sentences = []
    for _round in range(2):
        sentences.append(scheduled_tasks.backup_state(state, out, keep=1, exclude=(), text_only=True,
                                                      state_backup=state_backup))
        time.sleep(1.1)                                  # snapshots are named by the second
    kept = sorted(p.name for p in out.iterdir() if p.is_dir())
    assert len(kept) == 1, "no rolling window: each night's copy replaces the last"
    newest = out / kept[0]
    names = sorted(p.relative_to(newest).as_posix() for p in newest.rglob("*") if p.is_file())
    assert names == ["content/marble_photos/job/status.json", "content/profiles.json", "jobs/cron/cron.table",
                     "manifest.json", "sql/accounts.db"]
    manifest = json.loads((newest / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["text_only"] is True and manifest["binary_left_out"]["files"] == 2
    assert manifest["binary_left_out"]["bytes"] == (photos / "exposure.npz").stat().st_size + (photos / "preview.png").stat().st_size
    copied = sqlite3.connect(newest / "sql" / "accounts.db")
    assert copied.execute("SELECT username FROM users").fetchall() == [("ann",)]
    copied.close()
    assert "binary left out: 2 files" in sentences[-1] and "1 old removed, keeping 1" in sentences[-1]
    assert state_backup.verify(newest)["ok"]


def test_a_backup_can_also_leave_a_named_folder_out(tmp_path):
    from meowdoku import state_backup
    state = tmp_path / "state"
    (state / "sql").mkdir(parents=True)
    (state / "content" / "big" / "job").mkdir(parents=True)
    (state / "content" / "big" / "job" / "notes.txt").write_text("text, but in the folder left out", encoding="utf-8")
    (state / "content" / "big_index.txt").write_text("not inside the folder", encoding="utf-8")
    manifest = state_backup.snapshot(state, tmp_path / "backups", keep=1, exclude=("content/big",))
    names = sorted(entry["rel"] for entry in manifest["files"])
    assert names == ["content/big_index.txt"], "a name that merely begins like the folder is not left out"
    assert manifest["excluded"] == ["content/big"] and manifest["text_only"] is False


def test_a_task_is_offered_only_when_what_it_needs_was_given():
    class Statuses:
        DONE, UNCHANGED = 1, 2
    assert scheduled_tasks.handlers(Statuses) == {}
    assert sorted(scheduled_tasks.handlers(Statuses, web_dir="w", manual=object(), site=None, graph_view=None)) == ["manual.hardcopy"]
    both = scheduled_tasks.handlers(Statuses, web_dir="w", manual=object(), site=None, graph_view=None,
                                    backup={"state_dir": "s", "backup_dir": "b", "keep": 1, "exclude": (), "text_only": True, "module": None})
    assert sorted(both) == ["manual.hardcopy", "state.backup"]


def test_the_worker_bakes_at_its_own_start_and_idles_without_a_cron_port(tmp_path):
    if rest_api.manual is None:
        pytest.skip("the SpeakToMe checkout has no manual")
    web = tmp_path / "web"
    web.mkdir()
    worker = subprocess.Popen([sys.executable, str(ROOT / "examples" / "meowdoku" / "deploy" / "task_worker.py"),
                               "--state-dir", str(tmp_path / "state"), "--web-dir", str(web),
                               "--cron-jobs-dir", str(tmp_path / "jobs"), "--federation-port", "1",
                               "--federation-canvas", "ignored"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        first = worker.stdout.readline()
        second = worker.stdout.readline()
        assert "[worker] at start: the manual hard copy was rebuilt" in first, first
        assert "no cron port was given" in second, second
        assert (web / "manual" / "index.html").is_file()
        time.sleep(1.5)
        assert worker.poll() is None, "it idles; it does not exit and get restarted in a loop"
    finally:
        worker.kill()
        worker.wait(timeout=10)


def test_the_web_server_no_longer_bakes_or_hosts_scheduled_tasks():
    source = (ROOT / "examples" / "meowdoku" / "server.py").read_text(encoding="utf-8")
    assert "scheduled_tasks" not in source and "--cron-port" not in source


def test_the_cron_tool_asks_and_this_host_performs_when_the_library_is_named(tmp_path):
    if not os.environ.get("NODUS_HEADLESS_TEST_DLL") or rest_api.manual is None:
        pytest.skip("Set NODUS_HEADLESS_TEST_DLL (and PATH for its dependencies) to test the real cron round trip")
    blank, canvas, jobs, web = tmp_path / "blank.canvas", tmp_path / "cron.canvas", tmp_path / "jobs", tmp_path / "web"
    jobs.mkdir()
    web.mkdir()
    blank.write_text("CANVAS V4\n1 1 56\nOFFSET 0 0\n\n", encoding="utf-8")
    port = 47697
    subprocess.run([sys.executable, str(NODUS / "tools" / "add_contract_port.py"), "--canvas", str(blank), "--out", str(canvas),
                    "--prefix", "cron", "--port", str(port), "--loan", "3600"], check=True, capture_output=True, timeout=120)
    (jobs / "cron.table").write_text("manual.hardcopy * * * * * missed=once\n", encoding="utf-8")
    host = subprocess.Popen([sys.executable, str(NODUS / "tools" / "run_canvas.py"), "--canvas", str(canvas), "--rate", "60",
                             "--acquire", "cron:4:3600", "--cron", f"cron:7201:{jobs}", "--ticks", "6000"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    performer = None
    try:
        for line in host.stdout:
            if "contract_ports" in line:
                break
        lines = []
        performer = scheduled_tasks.start(port=port, jobs_dir=jobs, web_dir=web,
                                          native=studio_product_nodus.nodus_modules()[0],
                                          nodus_tools=studio_product_nodus.DEFAULT_NODUS_TOOLS, manual=rest_api.manual,
                                          site=rest_api.site, graph_view=GraphView, log=lines.append)
        end = time.time() + 80
        while not performer.performed and time.time() < end:
            time.sleep(0.25)
        assert performer.performed, f"the cron tool never asked: {lines}"
        contract, task, status = performer.performed[0]
        assert task == "manual.hardcopy" and status == 1, lines
        assert (web / "manual" / "index.html").is_file()
        result = json.loads((jobs / f"{contract}.result.json").read_text(encoding="utf-8"))
        assert result["status_name"] == "done" and "rebuilt" in result["detail"]
    finally:
        if performer is not None:
            performer.close()
        host.kill()
        host.wait(timeout=10)

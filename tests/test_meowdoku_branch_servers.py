"""[LOAN] Every turing or speaktome checkout lends its owner a server; the supervisor runs exactly the ready loans."""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import branch_gate, branch_review, branch_servers as B, central_checkout, instance_control  # noqa: E402

ANN, BOB = 2, 3
STUB_EXPORT = ("import argparse, pathlib\np = argparse.ArgumentParser(); p.add_argument('--out'); p.add_argument('--skip-boards', "
               "action='store_true')\na = p.parse_args(); pathlib.Path(a.out, 'index.html').write_text('page', encoding='utf-8')\n")


def run(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def world(tmp_path):
    remote, work = tmp_path / "remote.git", tmp_path / "turing"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(work)], check=True)
    for key, value in (("user.name", "t"), ("user.email", "t@example.invalid"), ("commit.gpgsign", "false")):
        run(work, "config", key, value)
    run(work, "remote", "add", "origin", str(remote))
    (work / "examples" / "meowdoku").mkdir(parents=True)
    (work / "examples" / "meowdoku" / "web_export.py").write_text(STUB_EXPORT, encoding="utf-8")
    run(work, "add", ".")
    run(work, "commit", "-q", "-m", "base")
    run(work, "push", "-q", "origin", "main")
    db = sqlite3.connect(tmp_path / "accounts.db", isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL)")
    db.execute("CREATE TABLE account_roles (user_id INTEGER PRIMARY KEY, role TEXT NOT NULL)")
    db.executemany("INSERT INTO users VALUES (?,?)", [(ANN, "ann"), (BOB, "bob"), (9, "svc-clean")])
    policy = {**branch_gate.DEFAULTS, "repositories": {"turing": {"path": str(work), "default": "main"},
                                                       "nodus": {"path": str(work), "default": "main"}}}
    boards = tmp_path / "dev-web" / "boards.json"
    boards.parent.mkdir()
    boards.write_text('{"boards": [], "levels": {}}', encoding="utf-8")
    cfg = {"state_root": str(tmp_path / "loans"), "dev_turing": str(tmp_path / "dev-turing"),
           "dev_speaktome": str(tmp_path / "dev-speaktome"), "accounts_db": str(tmp_path / "dev" / "sql" / "accounts.db"),
           "args": ["--require-cap", "dev_ui"], "boards_from": str(boards)}
    return {"db": db, "policy": policy, "root": tmp_path / "state" / "checkouts", "cfg": cfg, "tmp": tmp_path}


def call(world, operation, user=ANN, username="ann", repository="turing", **body):
    return central_checkout.run(world["db"], world["policy"], world["root"], user_id=user, username=username,
                                operation=operation, body={"repository": repository, **body})


def test_a_checkout_lends_a_server_within_the_limits_and_closing_ends_it(world):
    first = call(world, "open", branch="ann/one")["server"]
    assert first == {"instance": "loan-8769", "port": 8769, "state": "pending", "note": first["note"]}
    assert call(world, "open", branch="ann/one")["server"]["port"] == 8769, "reopening keeps the same loan"
    assert call(world, "open", branch="ann/two")["server"]["port"] == 8770
    assert "already have 2" in call(world, "open", branch="ann/three")["server"]["none"], "two per account"
    assert call(world, "open", user=BOB, username="bob", branch="bob/one")["server"]["port"] == 8771
    assert "has no server" in call(world, "open", repository="nodus", user=BOB, username="bob", branch="bob/n")["server"]["none"]
    assert call(world, "close", branch="ann/one")["server"] == {"ended": "loan-8769"}
    assert call(world, "open", branch="ann/three")["server"]["port"] == 8769, "a freed port is lent again"


def test_refresh_exports_then_the_supervisor_spec_appears_and_a_commit_restarts(world):
    db, cfg = world["db"], world["cfg"]
    call(world, "open", branch="ann/one")
    assert B.specs(db, cfg) == [], "a pending loan has no pages yet, so it does not run"
    [done] = B.refresh(db, cfg, requester=(9, "svc-clean"), log=lambda line: None)
    assert done["action"] == "exported" and "restart" not in done["reason"]
    assert (Path(cfg["state_root"]) / "8769" / "web" / "boards.json").exists(), "the dev boards are copied in"
    [spec] = B.specs(db, cfg)
    folder = db.execute("SELECT path FROM checkouts").fetchone()[0]
    assert spec["name"] == "loan-8769" and spec["port"] == 8769 and spec["bind"] == "127.0.0.1"
    assert spec["code_root"] == folder and spec["speaktome_root"] == cfg["dev_speaktome"]
    assert spec["accounts_db"] == cfg["accounts_db"]
    sup_mod = _supervisor_module()
    instances = world["tmp"] / "instances.json"
    instances.write_text(json.dumps({"instances": [], "loans": cfg}), encoding="utf-8")
    sup = sup_mod.Supervisor(world["tmp"] / "live.db", instances, world["tmp"] / "state")
    command = sup_mod.Instance(spec, sup).command()
    assert command[command.index("--accounts-db") + 1] == cfg["accounts_db"]
    assert str(sup.db_path) not in command
    assert spec["args"][-2:] == ["--web-dir", str(Path(cfg["state_root"]) / "8769" / "web")]
    assert B.refresh(db, cfg, requester=(9, "svc-clean"), log=lambda line: None) == [], "nothing new, nothing done"
    branch_review.set_identity(db, user_id=ANN, name="ann", email="ann@example.invalid")
    call(world, "write", branch="ann/one", path="x.txt", text="x\n")
    call(world, "commit", branch="ann/one", message="x")
    [done] = B.refresh(db, cfg, requester=(9, "svc-clean"), known=["loan-8769"], log=lambda line: None)
    assert done["action"] == "exported" and "restart requested" in done["reason"]
    [row] = instance_control.pending(db)
    assert row["instance"] == "loan-8769" and row["action"] == "restart"


def test_an_unused_loan_idles_and_wakes_and_a_vanished_checkout_ends_it(world):
    db, cfg = world["db"], world["cfg"]
    call(world, "open", branch="ann/one")
    B.refresh(db, cfg, requester=(9, "svc-clean"), log=lambda line: None)
    later = time.time() + (B.IDLE_HOURS + 1) * 3600
    [done] = B.refresh(db, cfg, requester=(9, "svc-clean"), now=later, log=lambda line: None)
    assert done["action"] == "idle" and B.specs(db, cfg) == [], "an idle loan does not run"
    db.execute("UPDATE checkout_calls SET at = ?", (later,))
    [done] = B.refresh(db, cfg, requester=(9, "svc-clean"), now=later, log=lambda line: None)
    assert done["action"] == "woken" and len(B.specs(db, cfg)) == 1
    db.execute("DELETE FROM checkouts")
    [done] = B.refresh(db, cfg, requester=(9, "svc-clean"), now=later, log=lambda line: None)
    assert done["action"] == "ended" and B.loans(db) == []


def test_only_the_owner_may_restart_their_loan_and_only_restart_or_start(world):
    db = world["db"]
    call(world, "open", branch="ann/one")
    assert B.may_restart(db, "loan-8769", ANN, "restart") and B.may_restart(db, "loan-8769", ANN, "start")
    assert not B.may_restart(db, "loan-8769", ANN, "shutdown")
    assert not B.may_restart(db, "loan-8769", BOB, "restart")
    assert not B.may_restart(db, "meowdoku-a", ANN, "restart") and not B.may_restart(db, "loan-x", ANN, "restart")
    assert "/" not in json.dumps(B.public(db)).replace("ann/one", ""), "the menu listing carries no folder path"


def _supervisor_module():
    spec = importlib.util.spec_from_file_location("apache_supervisor_t", ROOT / "examples" / "meowdoku" / "deploy" / "apache_supervisor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_supervisor_runs_exactly_the_ready_loans(world, monkeypatch):
    sup_mod = _supervisor_module()
    db, cfg, tmp = world["db"], world["cfg"], world["tmp"]
    instances = tmp / "instances.json"
    instances.write_text(json.dumps({"instances": [], "loans": cfg}), encoding="utf-8")
    begun = []
    stopped = []
    monkeypatch.setattr(sup_mod.Instance, "begin", lambda self: (begun.append(self.name), setattr(self, "state", "running")))
    monkeypatch.setattr(sup_mod.Instance, "force_stop", lambda self, wait=15.0: (stopped.append(self.name), "stopped")[1])
    sup = sup_mod.Supervisor(tmp / "accounts.db", instances, tmp / "state")
    call(world, "open", branch="ann/one")
    sup.reconcile_loans()
    assert begun == [] and sup.instances == [], "a pending loan is not started"
    B.refresh(db, cfg, requester=(9, "svc-clean"), log=lambda line: None)
    sup.reconcile_loans()
    sup.reconcile_loans()
    assert begun == ["loan-8769"] and [i.name for i in sup.instances] == ["loan-8769"], "started once"
    call(world, "close", branch="ann/one")
    sup.reconcile_loans()
    assert sup.instances == [] and stopped == ["loan-8769"], "an ended loan's instance is stopped and dropped"
    instances.write_text(json.dumps({"instances": []}), encoding="utf-8")
    call(world, "open", branch="ann/two")
    B.refresh(db, cfg, requester=(9, "svc-clean"), log=lambda line: None)
    sup.reconcile_loans()
    assert sup.instances == [], "without a loans block no loan server runs"


def test_loopback_is_the_default_not_the_only_option(world):
    db, cfg = world["db"], world["cfg"]
    call(world, "open", branch="ann/one")
    B.refresh(db, cfg, requester=(9, "svc-clean"), log=lambda line: None)
    [spec] = B.specs(db, cfg)
    assert spec["bind"] == "127.0.0.1" and "--trusted-proxy" not in spec["args"], "the default is loopback and trusts no one else"
    remote = dict(cfg, bind="192.168.1.232", trusted_proxies=["192.168.1.232"])
    [spec] = B.specs(db, remote)
    assert spec["bind"] == "192.168.1.232"
    assert spec["args"][-2:] == ["--trusted-proxy", "192.168.1.232"], "a network-bound loan believes only the named dev Apache"

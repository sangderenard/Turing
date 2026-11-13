"""Paper-order admission and retry behavior without contacting Alpaca."""
import sys
import time
from pathlib import Path
from decimal import Decimal
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from meowdoku import paper_arena, paper_broker, paper_mock  # noqa: E402
from meowdoku.living_data_query import GraphView  # noqa: E402


@pytest.fixture
def db(tmp_path):
    connection = paper_arena.connect(tmp_path / "paper.sqlite")
    yield connection
    connection.close()


def seat(db, owner=1, request_id="seat"):
    return paper_arena.create_strategy(db, owner, {"request_id": request_id, "name": "Test", "risk": "cautious"})["id"]


def order(strategy_id, request_id="decision"):
    return {"request_id": request_id, "strategy_id": strategy_id,
            "symbol": "BTC/USD", "side": "buy", "amount": "5"}


def test_strategy_and_order_idempotency_and_separation(db, monkeypatch):
    calls = []
    monkeypatch.setattr(paper_broker, "submit", lambda keys, payload: calls.append(payload) or {
        "status": "accepted", "filled_qty": "0", "client_order_id": payload["client_order_id"]})
    strategy = seat(db)
    assert seat(db) == strategy
    first = paper_arena.submit(db, 1, ("key", "secret"), order(strategy))
    replay = paper_arena.submit(db, 1, ("key", "secret"), order(strategy))
    assert first["client_order_id"] == replay["client_order_id"]
    assert replay["replayed"] is True and len(calls) == 1
    assert paper_arena.intents(db, 2) == []
    with pytest.raises(paper_arena.ArenaError, match="already used"):
        paper_arena.submit(db, 1, ("key", "secret"), {**order(strategy), "amount": "4"})


def test_risk_limit_and_gross_budget(db, monkeypatch):
    monkeypatch.setattr(paper_broker, "submit", lambda keys, payload: {"status": "accepted"})
    strategy = seat(db)
    with pytest.raises(paper_arena.ArenaError, match="risk limit"):
        paper_arena.submit(db, 1, ("key", "secret"), {**order(strategy), "amount": "5.01"})
    for number in range(10):
        paper_arena.submit(db, 1, ("key", "secret"), order(strategy, f"buy{number}"))
    with pytest.raises(paper_arena.ArenaError, match="budget"):
        paper_arena.submit(db, 1, ("key", "secret"), order(strategy, "buy11"))


def test_uncertain_submit_is_retained_and_reconciled_without_resubmission(db, monkeypatch):
    strategy = seat(db)
    attempts = []

    def uncertain(keys, payload):
        attempts.append(payload)
        raise paper_broker.PaperBrokerError("outcome uncertain", uncertain=True)

    monkeypatch.setattr(paper_broker, "submit", uncertain)
    with pytest.raises(paper_arena.ArenaError) as raised:
        paper_arena.submit(db, 1, ("key", "secret"), order(strategy))
    assert raised.value.status == 503
    pending = paper_arena.intents(db, 1)[0]
    assert pending["state"] == "uncertain"
    assert paper_arena.submit(db, 1, ("key", "secret"), order(strategy))["replayed"] is True
    assert len(attempts) == 1
    monkeypatch.setattr(paper_broker, "order_by_client_id", lambda keys, cid: {
        "status": "filled", "filled_qty": "0.0001", "filled_avg_price": "50000"})
    found = paper_arena.reconcile(db, 1, ("key", "secret"), pending["id"])
    assert found["broker_found"] is True and found["broker_status"] == "filled"
    assert found["fill_delta_qty"] == "0.0001"
    assert paper_arena.reconcile(db, 1, ("key", "secret"), pending["id"])["fill_delta_qty"] == "0"


def test_living_map_projects_only_owned_strategy_and_order_nodes(db, monkeypatch):
    monkeypatch.setattr(paper_broker, "submit", lambda keys, payload: {"status": "accepted"})
    mine, other = seat(db, 1, "mine"), seat(db, 2, "other")
    paper_arena.submit(db, 1, ("key", "secret"), order(mine))
    paper_arena.submit(db, 2, ("key", "secret"), order(other))
    graph = GraphView(paper_arena.living_map(db, 1), "world")
    assert f"strategy:{mine}" in graph.nodes
    assert f"strategy:{other}" not in graph.nodes
    assert graph.query("paper:1")["total"] >= 8


def test_broker_transport_uses_only_paper_host_and_treats_server_error_as_uncertain(monkeypatch):
    urls = []

    def fail(request, timeout):
        urls.append(request.full_url)
        raise HTTPError(request.full_url, 503, "unavailable", {}, None)

    monkeypatch.setattr(paper_broker.urllib.request, "urlopen", fail)
    with pytest.raises(paper_broker.PaperBrokerError) as raised:
        paper_broker.submit(("key", "secret"), {"symbol": "BTC/USD"})
    assert raised.value.uncertain is True
    assert urls == ["https://paper-api.alpaca.markets/v2/orders"]


def test_bot_registry_presence_and_trainer_cursor(db):
    body = {"request_id": "bot1", "name": "Tensor bot", "kind": "compiled", "source": {"ref": "product:" + "a" * 64}}
    created = paper_arena.register_bot(db, 1, body)
    assert paper_arena.register_bot(db, 1, body)["replayed"] is True
    with pytest.raises(paper_arena.ArenaError, match="different bot"):
        paper_arena.register_bot(db, 1, {**body, "name": "Changed"})
    paper_arena.record_activity(db, 1, created["id"], "mcp_reported", "get /api/paper/account")
    assert paper_arena.presence(db)["bots"][0]["id"] == created["id"]
    page = paper_arena.trainer_events(db, 0)
    assert page["events"][0]["channel"] == "mcp_reported"
    assert paper_arena.trainer_events(db, page["next_cursor"])["events"] == []
    with pytest.raises(paper_arena.ArenaError, match="bot not found"):
        paper_arena.record_activity(db, 2, created["id"], "rest", "look")


def test_qualification_is_bound_to_bot_source_and_risk(db):
    bot = paper_arena.register_bot(db, 1, {
        "request_id": "qualified", "name": "Candidate", "kind": "compiled",
        "source": {"ref": "product:" + "a" * 64}})
    report = paper_arena.record_historical_qualification(
        db, 1, bot["id"], "cautious", "b" * 64,
        ["0.10", "0.12", "0.08"], [1, 1, 1])
    assert report["passed"] is True
    assert paper_arena.bot_qualified(db, 1, bot["id"], "cautious")
    assert not paper_arena.bot_qualified(db, 1, bot["id"], "standard")
    assert not paper_arena.bot_qualified(db, 2, bot["id"], "cautious")
    db.execute("UPDATE paper_bots SET source=? WHERE id=?",
               ('{"registration":{"ref":"product:' + "c" * 64 + '"}}', bot["id"]))
    assert not paper_arena.bot_qualified(db, 1, bot["id"], "cautious")


def test_presence_pages_every_active_bot(db):
    now = time.time()
    db.executemany("INSERT INTO paper_bots VALUES(?,?,?,?,?,?,?,?)", [
        (f"{i:032x}", 1, f"Robot {i}", "agent", "{\"registration\":{},\"digest\":\"x\"}",
         f"bot{i}", now, now) for i in range(501)])
    first = paper_arena.presence(db)
    second = paper_arena.presence(db, offset=first["next_offset"])
    assert len(first["bots"]) == 500 and len(second["bots"]) == 1
    assert second["next_offset"] is None


def test_internal_mock_cash_holdings_fees_and_replay(db):
    strategy = seat(db)
    buy = order(strategy, "mockbuy")
    filled = paper_mock.place(db, 1, buy, "50000")
    assert filled["venue"] == "internal_mock" and filled["fee"] == "0.0025"
    assert paper_mock.place(db, 1, buy, "90000")["replayed"] is True
    assert len(paper_mock.orders(db, 1)["orders"]) == 1
    holdings = paper_mock.ledger(db, 1, strategy)
    assert holdings["cash_usd"] == "44.9975"
    assert Decimal(holdings["holdings"][0]["quantity"]) > 0
    sell = {**buy, "request_id": "mocksell", "side": "sell", "amount": "4"}
    sold = paper_mock.place(db, 1, sell, "51000")
    assert sold["side"] == "sell"
    assert Decimal(paper_mock.ledger(db, 1, strategy)["realized_pnl_usd"]) > 0
    with pytest.raises(paper_arena.ArenaError, match="lacks holdings"):
        paper_mock.place(db, 1, {**sell, "request_id": "too_much", "amount": "5"}, "51000")

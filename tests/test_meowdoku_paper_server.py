"""Authenticated paper routes and credential choice, with no broker network calls."""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku import alpaca_credentials, market_data, paper_broker

TESTS = Path(__file__).with_name("test_meowdoku_server_authz.py")
spec = importlib.util.spec_from_file_location("paper_http_helpers", TESTS)
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
spec.loader.exec_module(helpers)


def test_admin_shared_paper_account_and_dev_own_account(server, monkeypatch):
    port = server.port
    admin_cookie = helpers._signup(port, "paper_admin")
    holder_cookie = helpers._signup(port, "paper_holder")
    dev_cookie = helpers._signup(port, "paper_dev")
    local = {"Host": "127.0.0.1"}
    assert helpers._call(port, "/api/admin/account/grant", "POST",
                         {**local, "X-Meowdoku-Admin": helpers.TOKEN}, {"username": "paper_admin"})[0] == 200
    admin_id = helpers._call(port, "/api/account/me", headers={"Cookie": admin_cookie})[1]["account"]["id"]
    holder_id = helpers._call(port, "/api/account/me", headers={"Cookie": holder_cookie})[1]["account"]["id"]
    dev_id = helpers._call(port, "/api/account/me", headers={"Cookie": dev_cookie})[1]["account"]["id"]
    alpaca_credentials.save_paper(server.sessions.state.secrets_dir, holder_id, "holder-key-id", "holder-secret")
    alpaca_credentials.save_paper(server.sessions.state.secrets_dir, dev_id, "dev-key-id", "dev-secret")
    admin_headers = {**helpers.PUBLIC_XFF, "Cookie": admin_cookie}
    dev_headers = {**helpers.PUBLIC_XFF, "Cookie": dev_cookie}
    assert helpers._call(port, "/api/paper/shared-account", "POST", admin_headers,
                         {"username": "paper_holder"})[0] == 200
    assert helpers._call(port, "/api/account/authority/grant", "POST", admin_headers,
                         {"username": "paper_dev", "group": "dev", "enabled": True})[0] == 200
    seen = []
    monkeypatch.setattr(paper_broker, "account", lambda keys: seen.append(keys[0]) or {
        "status": "ACTIVE", "buying_power": "100", "equity": "100", "cash": "100", "currency": "USD",
        "trading_blocked": False})
    assert helpers._call(port, "/api/paper/account", headers=admin_headers)[0] == 200
    assert helpers._call(port, "/api/paper/account", headers=dev_headers)[0] == 200
    assert seen == ["holder-key-id", "dev-key-id"]
    assert admin_id != holder_id != dev_id

    registered = helpers._call(port, "/api/paper/bots", "POST", dev_headers,
                               {"request_id": "bot1", "name": "Test robot", "kind": "agent", "source": {}})
    assert registered[0] == 201
    bot_id = registered[1]["id"]
    assert helpers._call(port, "/api/paper/bots", "POST", dev_headers,
                         {"request_id": "badpython", "name": "Unowned", "kind": "python",
                          "source": {"ref": "divider:aaaaaaaaaaaa"}})[0] == 404
    assert helpers._call(port, "/api/paper/bots", "POST", dev_headers,
                         {"request_id": "remote1", "name": "Remote", "kind": "remote_api",
                          "source": {"url": "https://example.org/market-bot"}})[0] == 201
    assert any(bot["id"] == bot_id for bot in helpers._call(port, "/api/paper/presence", headers=admin_headers)[1]["bots"])
    assert helpers._call(port, "/api/paper/trainer-events", headers=dev_headers)[0] == 403
    events = helpers._call(port, "/api/paper/trainer-events?after=0", headers=admin_headers)[1]["events"]
    assert any(event["bot_id"] == bot_id for event in events)

    submissions = []
    monkeypatch.setattr(paper_broker, "submit", lambda keys, payload: submissions.append((keys[0], payload)) or {
        "status": "filled", "filled_qty": "0.0001", "filled_avg_price": "50000",
        "client_order_id": payload["client_order_id"]})
    bot_headers = {**dev_headers, "X-Meowdoku-Bot": bot_id}
    created = helpers._call(port, "/api/paper/strategies", "POST", bot_headers,
                            {"request_id": "model1", "name": "Model 1", "risk": "cautious", "bot_id": bot_id})
    assert created[0] == 201
    command = {"request_id": "decision1", "strategy_id": created[1]["id"],
               "symbol": "BTC/USD", "side": "buy", "amount": "5"}
    assert helpers._call(port, "/api/paper/orders", "POST", bot_headers, command)[0] == 201
    assert helpers._call(port, "/api/paper/orders", "POST", bot_headers, command)[1]["replayed"] is True
    assert len(submissions) == 1 and submissions[0][0] == "dev-key-id"
    assert len(helpers._call(port, "/api/paper/orders", headers=dev_headers)[1]["orders"]) == 1
    assert helpers._call(port, "/api/paper/orders", headers=admin_headers)[1]["orders"] == []
    graph = helpers._call(port, "/api/paper/map", headers=dev_headers)[1]
    assert graph["schema"] == "abstract-ui-div-map-v0"
    assert any(node["identity"].startswith("strategy:") for node in graph["world"]["objects"])
    assert helpers._call(port, "/api/paper/map/query", headers=dev_headers)[1]["schema"] == "living-data-query.v1"
    dm = helpers._call(port, "/api/paper/dm", "POST", dev_headers,
                       {"verb": "examine", "target": "node", "input": {"at": graph["identity"], "radius": 1}})
    assert dm[0] == 200 and dm[1]["schema"] == "living-data-query.v1"
    assert helpers._call(port, "/api/paper/dm", "POST", dev_headers,
                         {"verb": "look", "target": "orders", "input": {}})[1]["orders"][0]["id"]
    monkeypatch.setattr(market_data, "latest_prices", lambda directory, account: {"BTC/USD": 51000})
    scoreboard = helpers._call(port, "/api/paper/dm", "POST", dev_headers,
                               {"verb": "look", "target": "scoreboard", "input": {}})
    assert scoreboard[0] == 200 and scoreboard[1]["strategies"][0]["pnl_before_fees_usd"] == "0.1000"
    mock_command = {**command, "request_id": "internal1"}
    mock = helpers._call(port, "/api/paper/dm", "POST", bot_headers,
                         {"verb": "make", "target": "mock_order", "input": mock_command})
    assert mock[0] == 201 and mock[1]["venue"] == "internal_mock"
    ledger = helpers._call(port, f"/api/paper/mock/ledger?strategy_id={created[1]['id']}", headers=dev_headers)
    assert ledger[0] == 200 and ledger[1]["cash_usd"] == "44.9975"
    events = helpers._call(port, "/api/paper/trainer-events?after=0", headers=admin_headers)[1]["events"]
    assert any(event["interaction"] == "fill internal mock order" and
               json.loads(event["detail"])["venue"] == "internal_mock" for event in events)
    assert any(event["detail"] == "alpaca_paper" for event in events)
    assert len([event for event in events if event["interaction"] == "alpaca paper fill"]) == 1


# Reuse the tested ephemeral server fixture, rather than launching a second kind of server.
server = helpers.server

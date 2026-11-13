"""Account-owned paper strategy seats and durable Alpaca order receipts.

One broker account may back several strategy ledgers. Every submitted order
retains its strategy identity through Alpaca's client_order_id. A failed network
call is uncertain and is never retried as a new order.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
from decimal import Decimal, InvalidOperation

from meowdoku import market_data, paper_broker, schema_version

RISK = {
    "cautious": (Decimal("5"), Decimal("50")),
    "standard": (Decimal("25"), Decimal("250")),
    "explore": (Decimal("100"), Decimal("1000")),
}
POLICIES = ("manual", "momentum", "mean_reversion", "tensor", "external")
OUTCOMES = ("fixed_horizon", "closed_trade", "fill_quality")
SYMBOLS = frozenset((*market_data.ETFS, *market_data.WILDCARDS, market_data.CRYPTO))
RID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SID = re.compile(r"^[0-9a-f]{32}$")


class ArenaError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def connect(path) -> sqlite3.Connection:
    db = sqlite3.connect(str(path), timeout=10, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    schema_version.declare(db, "paper_arena", 3)
    db.execute("CREATE TABLE IF NOT EXISTS paper_bots ("
               "id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL, "
               "source TEXT NOT NULL, request_id TEXT NOT NULL, created REAL NOT NULL, last_seen REAL, "
               "UNIQUE(owner_id,request_id))")
    db.execute("CREATE TABLE IF NOT EXISTS paper_activity ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, owner_id INTEGER NOT NULL, bot_id TEXT, "
               "channel TEXT NOT NULL, interaction TEXT NOT NULL, strategy_id TEXT, "
               "detail TEXT NOT NULL, created REAL NOT NULL)")
    db.execute("CREATE INDEX IF NOT EXISTS paper_activity_recent ON paper_activity(created DESC)")
    db.execute("CREATE TABLE IF NOT EXISTS paper_bot_strategies ("
               "strategy_id TEXT PRIMARY KEY, bot_id TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS paper_bot_qualifications ("
               "bot_id TEXT NOT NULL, risk TEXT NOT NULL, owner_id INTEGER NOT NULL, source_sha256 TEXT NOT NULL, "
               "dataset_sha256 TEXT NOT NULL, report TEXT NOT NULL, passed INTEGER NOT NULL, "
               "created REAL NOT NULL, PRIMARY KEY(bot_id,risk))")
    db.execute("CREATE TABLE IF NOT EXISTS paper_strategy_modules ("
               "strategy_id TEXT PRIMARY KEY, policy TEXT NOT NULL, outcomes TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS paper_shared_account ("
               "singleton INTEGER PRIMARY KEY CHECK(singleton=1), owner_id INTEGER NOT NULL, "
               "set_by INTEGER NOT NULL, updated REAL NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS paper_strategies ("
               "id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, name TEXT NOT NULL, risk TEXT NOT NULL, "
               "request_id TEXT NOT NULL, request_digest TEXT NOT NULL, created REAL NOT NULL, "
               "UNIQUE(owner_id,request_id))")
    db.execute("CREATE TABLE IF NOT EXISTS paper_intents ("
               "id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, strategy_id TEXT NOT NULL, "
               "request_id TEXT NOT NULL, request_digest TEXT NOT NULL, client_order_id TEXT NOT NULL UNIQUE, "
               "symbol TEXT NOT NULL, side TEXT NOT NULL, amount TEXT NOT NULL, state TEXT NOT NULL, "
               "broker_order TEXT, created REAL NOT NULL, updated REAL NOT NULL, "
               "UNIQUE(owner_id,request_id))")
    db.execute("CREATE INDEX IF NOT EXISTS paper_intents_strategy ON paper_intents(strategy_id,created)")
    return db


def shared_owner(db) -> int | None:
    row = db.execute("SELECT owner_id FROM paper_shared_account WHERE singleton=1").fetchone()
    return None if row is None else row[0]


def set_shared_owner(db, owner_id: int, actor_id: int) -> None:
    if type(owner_id) is not int or owner_id < 1 or type(actor_id) is not int or actor_id < 1:
        raise ArenaError("invalid paper account owner")
    db.execute("INSERT INTO paper_shared_account(singleton,owner_id,set_by,updated) VALUES(1,?,?,?) "
               "ON CONFLICT(singleton) DO UPDATE SET owner_id=excluded.owner_id,set_by=excluded.set_by,updated=excluded.updated",
               (owner_id, actor_id, time.time()))


BOT_KINDS = frozenset(("python", "symbolic", "compiled", "remote_api", "agent"))
CHANNELS = frozenset(("dm", "rest", "mcp_reported", "kernel"))


def register_bot(db, owner: int, body: dict) -> dict:
    if not isinstance(body, dict) or set(body) != {"request_id", "name", "kind", "source"}:
        raise ArenaError("provide request_id, name, kind and source")
    request_id = _request_id(body["request_id"])
    name, kind, source = body["name"], body["kind"], body["source"]
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80 or name != name.strip() or kind not in BOT_KINDS:
        raise ArenaError("invalid bot name or kind")
    try:
        source_bytes = json.dumps(source, allow_nan=False).encode("utf-8") if isinstance(source, dict) else b""
    except (TypeError, ValueError):
        source_bytes = b""
    if not source_bytes or len(source_bytes) > 2048:
        raise ArenaError("source must be a bounded object")
    if kind == "remote_api":
        from urllib.parse import urlsplit
        url = source.get("url")
        try:
            parsed = urlsplit(url) if isinstance(url, str) else None
        except ValueError:
            parsed = None
        if parsed is None or parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ArenaError("remote API registration requires an HTTPS URL without embedded credentials")
    elif kind in ("python", "symbolic", "compiled"):
        expected = {"python": r"divider:[0-9a-f]{12}",
                    "symbolic": r"mathlab:[0-9a-f]{16}",
                    "compiled": r"product:[0-9a-f]{64}"}[kind]
        if not isinstance(source.get("ref"), str) or not re.fullmatch(expected, source["ref"]):
            raise ArenaError(f"{kind} source.ref must use {expected}")
    digest = _digest(body)
    db.execute("BEGIN IMMEDIATE")
    try:
        row = db.execute("SELECT * FROM paper_bots WHERE owner_id=? AND request_id=?", (owner, request_id)).fetchone()
        if row:
            if json.loads(row["source"]) != {"registration": source, "digest": digest}:
                raise ArenaError("request_id was already used for a different bot", 409)
            result = {"id": row["id"], "name": row["name"], "kind": row["kind"], "replayed": True}
        else:
            if db.execute("SELECT COUNT(*) FROM paper_bots WHERE owner_id=?", (owner,)).fetchone()[0] >= 3000:
                raise ArenaError("bot capacity reached", 429)
            identity = uuid.uuid4().hex
            db.execute("INSERT INTO paper_bots VALUES(?,?,?,?,?,?,?,NULL)",
                       (identity, owner, name, kind, json.dumps({"registration": source, "digest": digest}),
                        request_id, time.time()))
            result = {"id": identity, "name": name, "kind": kind, "replayed": False}
        db.execute("COMMIT")
        return result
    except Exception:
        db.execute("ROLLBACK")
        raise


def record_activity(db, owner: int, bot_id: str | None, channel: str, interaction: str,
                    strategy_id: str | None = None, detail: str = "") -> None:
    if channel not in CHANNELS or not isinstance(interaction, str) or len(interaction) > 80:
        raise ArenaError("invalid activity")
    if bot_id is not None:
        if not isinstance(bot_id, str) or not SID.fullmatch(bot_id):
            raise ArenaError("bot not found", 404)
        row = db.execute("SELECT 1 FROM paper_bots WHERE id=? AND owner_id=?", (bot_id, owner)).fetchone()
        if row is None:
            raise ArenaError("bot not found", 404)
    if strategy_id is not None:
        _strategy(db, owner, strategy_id)
    now = time.time()
    db.execute("INSERT INTO paper_activity(owner_id,bot_id,channel,interaction,strategy_id,detail,created) "
               "VALUES(?,?,?,?,?,?,?)", (owner, bot_id, channel, interaction, strategy_id, detail[:240], now))
    if bot_id is not None:
        db.execute("UPDATE paper_bots SET last_seen=? WHERE id=?", (now, bot_id))
    db.execute("DELETE FROM paper_activity WHERE created<?", (now - 30 * 86400,))


def presence(db, owner: int | None = None, *, since_seconds: int = 300, offset: int = 0) -> dict:
    if type(offset) is not int or offset < 0:
        raise ArenaError("invalid presence offset")
    cutoff = time.time() - since_seconds
    where = "AND owner_id=?" if owner is not None else ""
    args = (cutoff, owner, 501, offset) if owner is not None else (cutoff, 501, offset)
    rows = list(db.execute(f"SELECT id,owner_id,name,kind,last_seen FROM paper_bots "
                           f"WHERE last_seen>=? {where} ORDER BY last_seen DESC,id LIMIT ? OFFSET ?", args))
    bots = []
    for row in rows[:500]:
        activity = db.execute("SELECT channel,interaction,strategy_id,created FROM paper_activity "
                              "WHERE bot_id=? ORDER BY id DESC LIMIT 1", (row["id"],)).fetchone()
        bots.append({"id": row["id"], "owner_id": row["owner_id"], "name": row["name"],
                     "kind": row["kind"], "last_seen": row["last_seen"],
                     "activity": dict(activity) if activity else None})
    return {"bots": bots, "fade_seconds": since_seconds, "asof": time.time(),
            "next_offset": offset + 500 if len(rows) > 500 else None}


def bots(db, owner: int) -> dict:
    rows = db.execute("SELECT id,name,kind,source,created,last_seen FROM paper_bots "
                      "WHERE owner_id=? ORDER BY created DESC LIMIT 3000", (owner,))
    return {"bots": [{"id": row["id"], "name": row["name"], "kind": row["kind"],
                      "source": json.loads(row["source"])["registration"],
                      "created": row["created"], "last_seen": row["last_seen"],
                      "qualified_risks": [risk for risk in RISK if bot_qualified(db, owner, row["id"], risk)]}
                     for row in rows]}


def _bot_source_hash(source: str) -> str:
    registration = json.loads(source)["registration"]
    return hashlib.sha256(json.dumps(registration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def bot_qualified(db, owner: int, bot_id: str, risk: str) -> bool:
    row = db.execute("SELECT source FROM paper_bots WHERE id=? AND owner_id=?", (bot_id, owner)).fetchone()
    if row is None:
        return False
    proof = db.execute("SELECT source_sha256,passed FROM paper_bot_qualifications "
                       "WHERE bot_id=? AND risk=? AND owner_id=?", (bot_id, risk, owner)).fetchone()
    return bool(proof and proof["passed"] == 1 and proof["source_sha256"] == _bot_source_hash(row["source"]))


def record_historical_qualification(db, owner: int, bot_id: str, risk: str,
                                    dataset_sha256: str, window_pnl_usd: list[str],
                                    sell_fill_counts: list[int],
                                    *, margin: Decimal = Decimal("0.01")) -> dict:
    """Internal judge entry; no client route accepts reported returns or calls this function."""
    bot = db.execute("SELECT source FROM paper_bots WHERE id=? AND owner_id=?", (bot_id, owner)).fetchone()
    if bot is None:
        raise ArenaError("bot not found", 404)
    if risk not in RISK:
        raise ArenaError("unknown qualification risk tier")
    if not isinstance(dataset_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", dataset_sha256):
        raise ArenaError("invalid historical dataset digest")
    if not isinstance(window_pnl_usd, list) or len(window_pnl_usd) < 3:
        raise ArenaError("at least three independent historical windows required")
    if (not isinstance(sell_fill_counts, list) or len(sell_fill_counts) != len(window_pnl_usd)
            or any(type(value) is not int or value < 0 for value in sell_fill_counts)):
        raise ArenaError("sell fill count required for every historical window")
    try:
        pnl = [Decimal(value) for value in window_pnl_usd]
    except (InvalidOperation, TypeError):
        raise ArenaError("invalid historical P&L") from None
    if any(not value.is_finite() for value in pnl) or not margin.is_finite() or margin <= 0:
        raise ArenaError("finite P&L and positive margin required")
    returns = [value / RISK[risk][0] for value in pnl]
    mean = sum(returns, Decimal(0)) / len(returns)
    passed = mean >= margin and all(value > 0 for value in returns) and all(value >= 1 for value in sell_fill_counts)
    report = {"window_net_pnl_usd": [str(value) for value in pnl],
              "window_net_returns": [str(value) for value in returns],
              "mean_net_return": str(mean), "required_margin": str(margin),
              "basis": "net P&L divided by maximum per-order capital at risk", "risk": risk,
              "sell_fill_counts": sell_fill_counts,
              "windows": len(returns), "passed": passed}
    db.execute("INSERT INTO paper_bot_qualifications VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(bot_id,risk) DO UPDATE SET "
               "owner_id=excluded.owner_id,source_sha256=excluded.source_sha256,"
               "dataset_sha256=excluded.dataset_sha256,report=excluded.report,"
               "passed=excluded.passed,created=excluded.created",
               (bot_id, risk, owner, _bot_source_hash(bot["source"]), dataset_sha256,
                json.dumps(report, separators=(",", ":")), int(passed), time.time()))
    return report


def require_qualified_order(db, owner: int, strategy_id: str, bot_id: str | None) -> None:
    """Automated strategy orders require a qualified, matching bot identity."""
    _strategy(db, owner, strategy_id)
    row = db.execute("SELECT b.bot_id,m.policy,s.risk FROM paper_strategies s "
                     "LEFT JOIN paper_bot_strategies b ON b.strategy_id=s.id "
                     "LEFT JOIN paper_strategy_modules m ON m.strategy_id=s.id WHERE s.id=?", (strategy_id,)).fetchone()
    assigned = row["bot_id"]
    if assigned is None and (row["policy"] or "manual") == "manual" and bot_id is None:
        return
    if assigned is None or bot_id != assigned or not bot_qualified(db, owner, assigned, row["risk"]):
        raise ArenaError("bot must pass server-judged historical qualification before ordering", 403)


def trainer_events(db, after: int, limit: int = 200) -> dict:
    if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 500:
        raise ArenaError("invalid trainer cursor")
    rows = db.execute("SELECT id,owner_id,bot_id,channel,interaction,strategy_id,detail,created FROM paper_activity "
                      "WHERE id>? ORDER BY id LIMIT ?", (after, limit)).fetchall()
    events = [dict(row) for row in rows]
    return {"events": events, "next_cursor": events[-1]["id"] if events else after}


def _request_id(value):
    if not isinstance(value, str) or not RID.fullmatch(value):
        raise ArenaError("request_id must be 1-64 letters, digits, underscores or hyphens")
    return value


def _digest(body: dict) -> str:
    try:
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise ArenaError("finite JSON data required") from None
    return hashlib.sha256(encoded.encode()).hexdigest()


def _amount(value: str) -> Decimal:
    if not isinstance(value, str) or not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,8})?", value):
        raise ArenaError("amount must be a positive decimal string with up to 8 decimal places")
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise ArenaError("invalid amount") from None
    if number <= 0:
        raise ArenaError("amount must be positive")
    return number


def _strategy(db, owner: int, strategy_id: str):
    if not isinstance(strategy_id, str) or not SID.fullmatch(strategy_id):
        raise ArenaError("strategy not found", 404)
    row = db.execute("SELECT * FROM paper_strategies WHERE owner_id=? AND id=?", (owner, strategy_id)).fetchone()
    if row is None:
        raise ArenaError("strategy not found", 404)
    return row


def strategies(db, owner: int) -> list[dict]:
    rows = db.execute("SELECT s.*, b.bot_id, m.policy, m.outcomes FROM paper_strategies s "
                      "LEFT JOIN paper_bot_strategies b ON b.strategy_id=s.id "
                      "LEFT JOIN paper_strategy_modules m ON m.strategy_id=s.id "
                      "WHERE s.owner_id=? ORDER BY s.created", (owner,))
    return [{"id": r["id"], "name": r["name"], "risk": r["risk"], "created": r["created"],
             "bot_id": r["bot_id"], "policy": r["policy"] or "manual",
             "outcomes": json.loads(r["outcomes"]) if r["outcomes"] else list(OUTCOMES),
             "max_order_usd": str(RISK[r["risk"]][0]),
             "gross_buy_budget_usd": str(RISK[r["risk"]][1])} for r in rows]


def create_strategy(db, owner: int, body: dict) -> dict:
    if not isinstance(body, dict) or not {"request_id", "name", "risk"} <= set(body) or set(body) - {"request_id", "name", "risk", "bot_id", "policy", "outcomes"}:
        raise ArenaError("provide request_id, name, risk and optional bot_id, policy and outcomes")
    request_id = _request_id(body["request_id"])
    name, risk = body["name"], body["risk"]
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80 or name != name.strip():
        raise ArenaError("strategy name must be 1-80 characters")
    if risk not in RISK:
        raise ArenaError("risk must be cautious, standard or explore")
    policy = body.get("policy", "manual")
    outcomes = body.get("outcomes", list(OUTCOMES))
    if (not isinstance(policy, str) or policy not in POLICIES or not isinstance(outcomes, list)
            or not outcomes or not all(isinstance(x, str) and x in OUTCOMES for x in outcomes)
            or len(outcomes) != len(set(outcomes))):
        raise ArenaError("invalid policy or outcomes")
    bot_id = body.get("bot_id")
    if bot_id is not None and db.execute("SELECT 1 FROM paper_bots WHERE id=? AND owner_id=?", (bot_id, owner)).fetchone() is None:
        raise ArenaError("bot not found", 404)
    digest = _digest(body)
    db.execute("BEGIN IMMEDIATE")
    try:
        old = db.execute("SELECT * FROM paper_strategies WHERE owner_id=? AND request_id=?",
                         (owner, request_id)).fetchone()
        if old:
            if old["request_digest"] != digest:
                raise ArenaError("request_id was already used for a different strategy", 409)
            result = {"id": old["id"], "name": old["name"], "risk": old["risk"], "created": old["created"], "replayed": True}
        else:
            if db.execute("SELECT COUNT(*) FROM paper_strategies WHERE owner_id=?", (owner,)).fetchone()[0] >= 3000:
                raise ArenaError("strategy capacity reached", 429)
            identity = hashlib.sha256(f"strategy:{owner}:{request_id}".encode()).hexdigest()[:32]
            created = time.time()
            db.execute("INSERT INTO paper_strategies VALUES (?,?,?,?,?,?,?)",
                       (identity, owner, name, risk, request_id, digest, created))
            db.execute("INSERT INTO paper_strategy_modules VALUES(?,?,?)",
                       (identity, policy, json.dumps(outcomes)))
            if bot_id is not None:
                db.execute("INSERT INTO paper_bot_strategies VALUES(?,?)", (identity, bot_id))
            result = {"id": identity, "name": name, "risk": risk, "created": created, "replayed": False}
        db.execute("COMMIT")
        return result
    except Exception:
        db.execute("ROLLBACK")
        raise


def intents(db, owner: int, strategy_id: str | None = None) -> list[dict]:
    if strategy_id is not None:
        _strategy(db, owner, strategy_id)
    rows = db.execute("SELECT * FROM paper_intents WHERE owner_id=? AND (? IS NULL OR strategy_id=?) "
                      "ORDER BY created DESC LIMIT 500", (owner, strategy_id, strategy_id))
    return [_public(row) for row in rows]


def living_map(db, owner: int) -> dict:
    """Project this account's monitored markets and strategy receipts into the canonical graph map."""
    symbols = (*market_data.ETFS, *market_data.WILDCARDS, market_data.CRYPTO)
    seats = strategies(db, owner)
    orders = intents(db, owner)
    root_id = f"paper:{owner}"
    objects = [{"identity": f"market:{symbol}", "kind": "market", "name": symbol,
                "dependencies": []} for symbol in symbols]
    objects += [{"identity": f"strategy:{seat['id']}", "kind": "paper-strategy",
                 "name": seat["name"], "risk": seat["risk"],
                 "dependencies": [{"target": f"order:{o['id']}", "relationship": "submitted"}
                                  for o in orders if o["strategy_id"] == seat["id"]]}
                for seat in seats]
    objects += [{"identity": f"order:{o['id']}", "kind": "paper-order",
                 "name": f"{o['side']} {o['symbol']} {o['amount']}",
                 "state": o["state"], "broker_status": o["broker_status"],
                 "dependencies": [{"target": f"market:{o['symbol']}", "relationship": "trades"}]}
                for o in orders]
    return {"schema": "abstract-ui-div-map-v0", "identity": root_id, "kind": "paper-arena",
            "name": "Paper trading arena", "dependencies": [
                *({"target": f"market:{symbol}", "relationship": "monitors"} for symbol in symbols),
                *({"target": f"strategy:{seat['id']}", "relationship": "owns"} for seat in seats)],
            "world": {"objects": objects}}


def scoreboard(db, owner: int, prices: dict) -> dict:
    """Mark reconciled strategy buys; report missing marks rather than guessing them."""
    seats = strategies(db, owner)
    orders = intents(db, owner)
    rows = []
    for seat in seats:
        holdings: dict[str, dict] = {}
        unreconciled = 0
        for order in orders:
            if order["strategy_id"] != seat["id"]:
                continue
            if order["state"] in ("pending", "uncertain"):
                unreconciled += 1
            qty, average = order["filled_qty"], order["filled_avg_price"]
            if order["side"] != "buy" or qty is None or average is None:
                continue
            try:
                quantity, price = Decimal(qty), Decimal(average)
            except (InvalidOperation, TypeError):
                continue
            if quantity <= 0 or price <= 0:
                continue
            holding = holdings.setdefault(order["symbol"], {"quantity": Decimal(0), "cost_usd": Decimal(0)})
            holding["quantity"] += quantity
            holding["cost_usd"] += quantity * price
        positions = []
        total_cost = Decimal(0)
        total_value = Decimal(0)
        complete_marks = True
        for symbol, holding in sorted(holdings.items()):
            mark = prices.get(symbol)
            try:
                mark = Decimal(str(mark)) if mark is not None else None
            except InvalidOperation:
                mark = None
            if mark is None or not mark.is_finite() or mark <= 0:
                complete_marks = False
                value = None
            else:
                value = holding["quantity"] * mark
                total_value += value
            total_cost += holding["cost_usd"]
            positions.append({"symbol": symbol, "quantity": str(holding["quantity"]),
                              "cost_usd": str(holding["cost_usd"]),
                              "mark": str(mark) if value is not None else None,
                              "value_usd": str(value) if value is not None else None})
        pnl = total_value - total_cost if complete_marks else None
        rows.append({"strategy_id": seat["id"], "name": seat["name"], "risk": seat["risk"],
                     "positions": positions, "cost_usd": str(total_cost),
                     "marked_value_usd": str(total_value) if complete_marks else None,
                     "pnl_before_fees_usd": str(pnl) if pnl is not None else None,
                     "return_on_spend_pct": str(pnl * 100 / total_cost) if pnl is not None and total_cost else None,
                     "unreconciled_orders": unreconciled})
    return {"strategies": rows, "basis": "reconciled buy fills marked at latest available bars; fees and spread are excluded"}


def _public(row) -> dict:
    order = json.loads(row["broker_order"]) if row["broker_order"] else None
    return {"id": row["id"], "strategy_id": row["strategy_id"], "request_id": row["request_id"],
            "client_order_id": row["client_order_id"], "symbol": row["symbol"], "side": row["side"],
            "amount": row["amount"], "state": row["state"], "broker_status": order.get("status") if order else None,
            "filled_qty": order.get("filled_qty") if order else None,
            "filled_avg_price": order.get("filled_avg_price") if order else None,
            "created": row["created"], "updated": row["updated"]}


def submit(db, owner: int, keys: tuple[str, str], body: dict) -> dict:
    if not isinstance(body, dict) or set(body) != {"request_id", "strategy_id", "symbol", "side", "amount"}:
        raise ArenaError("provide request_id, strategy_id, symbol, side and amount")
    request_id = _request_id(body["request_id"])
    strategy = _strategy(db, owner, body["strategy_id"])
    symbol, side = body["symbol"], body["side"]
    if symbol not in SYMBOLS or side not in ("buy", "sell"):
        raise ArenaError("unsupported paper order symbol or side")
    amount = _amount(body["amount"])
    if side != "buy":
        raise ArenaError("strategy sells require reconciled holdings; not enabled yet", 409)
    max_order, budget = RISK[strategy["risk"]]
    if amount > max_order:
        raise ArenaError("order exceeds this strategy's risk limit", 403)
    digest = _digest(body)
    db.execute("BEGIN IMMEDIATE")
    try:
        old = db.execute("SELECT * FROM paper_intents WHERE owner_id=? AND request_id=?", (owner, request_id)).fetchone()
        if old:
            if old["request_digest"] != digest:
                raise ArenaError("request_id was already used for a different order", 409)
            db.execute("COMMIT")
            return dict(_public(old), replayed=True)
        spent = db.execute("SELECT amount FROM paper_intents WHERE strategy_id=? AND side='buy' "
                           "AND state IN ('pending','uncertain','submitted')", (strategy["id"],))
        if amount + sum((Decimal(r["amount"]) for r in spent), Decimal(0)) > budget:
            raise ArenaError("strategy gross buy budget reached", 403)
        identity = hashlib.sha256(f"intent:{owner}:{request_id}".encode()).hexdigest()[:32]
        client_id = "mdk-" + identity
        now = time.time()
        db.execute("INSERT INTO paper_intents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (identity, owner, strategy["id"], request_id, digest, client_id,
                    symbol, side, str(amount), "pending", None, now, now))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    order = {"symbol": symbol, "notional": str(amount), "side": "buy", "type": "market",
             "time_in_force": "gtc" if symbol == market_data.CRYPTO else "day",
             "client_order_id": client_id}
    try:
        receipt = paper_broker.submit(keys, order)
    except paper_broker.PaperBrokerError as error:
        state = "uncertain" if error.uncertain else "rejected"
        db.execute("UPDATE paper_intents SET state=?,updated=? WHERE id=?", (state, time.time(), identity))
        raise ArenaError(str(error), 503 if error.uncertain else error.status) from None
    db.execute("UPDATE paper_intents SET state='submitted',broker_order=?,updated=? WHERE id=?",
               (json.dumps(receipt, separators=(",", ":")), time.time(), identity))
    return _public(db.execute("SELECT * FROM paper_intents WHERE id=?", (identity,)).fetchone())


def reconcile(db, owner: int, keys: tuple[str, str], intent_id: str) -> dict:
    if not isinstance(intent_id, str) or not SID.fullmatch(intent_id):
        raise ArenaError("paper order not found", 404)
    row = db.execute("SELECT * FROM paper_intents WHERE owner_id=? AND id=?", (owner, intent_id)).fetchone()
    if row is None:
        raise ArenaError("paper order not found", 404)
    order = paper_broker.order_by_client_id(keys, row["client_order_id"])
    if order is None:
        return dict(_public(row), broker_found=False)
    prior = json.loads(row["broker_order"]) if row["broker_order"] else {}
    try:
        prior_qty = Decimal(str(prior.get("filled_qty") or "0"))
        current_qty = Decimal(str(order.get("filled_qty") or "0"))
        delta = max(Decimal(0), current_qty - prior_qty)
    except (InvalidOperation, TypeError):
        delta = Decimal(0)
    db.execute("UPDATE paper_intents SET state='submitted',broker_order=?,updated=? WHERE id=?",
               (json.dumps(order, separators=(",", ":")), time.time(), intent_id))
    return dict(_public(db.execute("SELECT * FROM paper_intents WHERE id=?", (intent_id,)).fetchone()),
                broker_found=True, fill_delta_qty=str(delta))

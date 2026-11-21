"""Fast internal paper venue with explicit, separate simulated accounting.

Fills use cached latest bars plus fixed spread, markup and fee assumptions.
They are never Alpaca receipts and must not be labeled real order success.
"""
from __future__ import annotations

import hashlib
import time
from decimal import Decimal, InvalidOperation

from meowdoku import paper_arena

SPREAD_BPS = Decimal("10")
MARKUP_BPS = Decimal("5")
FEE_BPS = Decimal("5")
DENOM = Decimal("10000")


def prepare(db):
    db.execute("CREATE TABLE IF NOT EXISTS paper_mock_accounts ("
               "strategy_id TEXT PRIMARY KEY, cash TEXT NOT NULL, realized_pnl TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS paper_mock_holdings ("
               "strategy_id TEXT NOT NULL, symbol TEXT NOT NULL, quantity TEXT NOT NULL, cost TEXT NOT NULL, "
               "PRIMARY KEY(strategy_id,symbol))")
    db.execute("CREATE TABLE IF NOT EXISTS paper_mock_orders ("
               "id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, strategy_id TEXT NOT NULL, "
               "request_id TEXT NOT NULL, digest TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL, "
               "amount TEXT NOT NULL, quantity TEXT NOT NULL, execution_price TEXT NOT NULL, "
               "fee TEXT NOT NULL, cash_after TEXT NOT NULL, created REAL NOT NULL, "
               "UNIQUE(owner_id,request_id))")


def _digest(body):
    return paper_arena._digest(body)


def _public(row):
    return {key: row[key] for key in ("id", "strategy_id", "symbol", "side", "amount", "quantity",
                                     "execution_price", "fee", "cash_after", "created")}


def replay(db, owner, body):
    prepare(db)
    if not isinstance(body, dict) or not isinstance(body.get("request_id"), str):
        return None
    row = db.execute("SELECT * FROM paper_mock_orders WHERE owner_id=? AND request_id=?",
                     (owner, body["request_id"])).fetchone()
    if row is None:
        return None
    if row["digest"] != _digest(body):
        raise paper_arena.ArenaError("request_id was already used for a different mock order", 409)
    return dict(_public(row), replayed=True, venue="internal_mock")


def place(db, owner: int, body: dict, mark) -> dict:
    prepare(db)
    if not isinstance(body, dict) or set(body) != {"request_id", "strategy_id", "symbol", "side", "amount"}:
        raise paper_arena.ArenaError("provide request_id, strategy_id, symbol, side and amount")
    request_id = paper_arena._request_id(body["request_id"])
    strategy = paper_arena._strategy(db, owner, body["strategy_id"])
    symbol, side = body["symbol"], body["side"]
    if symbol not in paper_arena.SYMBOLS or side not in ("buy", "sell"):
        raise paper_arena.ArenaError("unsupported mock order symbol or side")
    amount = paper_arena._amount(body["amount"])
    max_order, initial_cash = paper_arena.RISK[strategy["risk"]]
    if amount > max_order:
        raise paper_arena.ArenaError("mock order exceeds strategy risk limit", 403)
    try:
        mark = Decimal(str(mark))
    except (InvalidOperation, TypeError):
        raise paper_arena.ArenaError("market mark unavailable", 503) from None
    if not mark.is_finite() or mark <= 0:
        raise paper_arena.ArenaError("market mark unavailable", 503)
    digest = _digest(body)
    db.execute("BEGIN IMMEDIATE")
    try:
        old = replay(db, owner, body)
        if old is not None:
            db.execute("COMMIT")
            return old
        account = db.execute("SELECT cash,realized_pnl FROM paper_mock_accounts WHERE strategy_id=?",
                             (strategy["id"],)).fetchone()
        cash = Decimal(account["cash"]) if account else initial_cash
        realized = Decimal(account["realized_pnl"]) if account else Decimal(0)
        holding = db.execute("SELECT quantity,cost FROM paper_mock_holdings WHERE strategy_id=? AND symbol=?",
                             (strategy["id"], symbol)).fetchone()
        held_qty = Decimal(holding["quantity"]) if holding else Decimal(0)
        held_cost = Decimal(holding["cost"]) if holding else Decimal(0)
        direction = Decimal(1) if side == "buy" else Decimal(-1)
        execution = mark * (1 + direction * (SPREAD_BPS + MARKUP_BPS) / DENOM)
        quantity = amount / execution
        fee = amount * FEE_BPS / DENOM
        if side == "buy":
            if cash < amount + fee:
                raise paper_arena.ArenaError("mock strategy cash exhausted", 403)
            cash -= amount + fee
            held_qty += quantity
            held_cost += amount + fee
        else:
            if quantity > held_qty:
                raise paper_arena.ArenaError("mock strategy lacks holdings for this sell", 403)
            released_cost = held_cost * quantity / held_qty
            held_qty -= quantity
            held_cost -= released_cost
            cash += amount - fee
            realized += amount - fee - released_cost
        db.execute("INSERT INTO paper_mock_accounts VALUES(?,?,?) ON CONFLICT(strategy_id) DO UPDATE "
                   "SET cash=excluded.cash,realized_pnl=excluded.realized_pnl",
                   (strategy["id"], str(cash), str(realized)))
        db.execute("INSERT INTO paper_mock_holdings VALUES(?,?,?,?) ON CONFLICT(strategy_id,symbol) DO UPDATE "
                   "SET quantity=excluded.quantity,cost=excluded.cost",
                   (strategy["id"], symbol, str(held_qty), str(held_cost)))
        identity = hashlib.sha256(f"mock:{owner}:{request_id}".encode()).hexdigest()[:32]
        db.execute("INSERT INTO paper_mock_orders VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (identity, owner, strategy["id"], request_id, digest, symbol, side, str(amount),
                    str(quantity), str(execution), str(fee), str(cash), time.time()))
        row = db.execute("SELECT * FROM paper_mock_orders WHERE id=?", (identity,)).fetchone()
        db.execute("COMMIT")
        return dict(_public(row), replayed=False, venue="internal_mock",
                    assumptions={"spread_bps": str(SPREAD_BPS), "markup_bps": str(MARKUP_BPS),
                                 "fee_bps": str(FEE_BPS), "mark": str(mark)})
    except Exception:
        db.execute("ROLLBACK")
        raise


def ledger(db, owner: int, strategy_id: str) -> dict:
    strategy = paper_arena._strategy(db, owner, strategy_id)
    prepare(db)
    row = db.execute("SELECT cash,realized_pnl FROM paper_mock_accounts WHERE strategy_id=?",
                     (strategy_id,)).fetchone()
    holdings = db.execute("SELECT symbol,quantity,cost FROM paper_mock_holdings WHERE strategy_id=? "
                          "AND quantity!='0' ORDER BY symbol", (strategy_id,)).fetchall()
    return {"strategy_id": strategy_id, "venue": "internal_mock",
            "cash_usd": row["cash"] if row else str(paper_arena.RISK[strategy["risk"]][1]),
            "realized_pnl_usd": row["realized_pnl"] if row else "0",
            "holdings": [dict(item) for item in holdings]}


def orders(db, owner: int) -> dict:
    prepare(db)
    rows = db.execute("SELECT * FROM paper_mock_orders WHERE owner_id=? ORDER BY created DESC LIMIT 500", (owner,))
    return {"orders": [_public(row) for row in rows], "venue": "internal_mock"}

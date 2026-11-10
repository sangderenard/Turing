"""Bounded, cookie-table action batches; no queue or cross-table routing authority.

Receipts are process-local. A new/replaced game changes epoch, so reconnects must
never replay an old batch into a new table. A failure can have mutated its action;
only the successful prefix is known, and nothing is rolled back or retried.
"""
from __future__ import annotations

import copy
import hashlib
import json
import secrets

MAX_BYTES = 65536
MAX_ACTIONS = 32


def validate(payload):
    if not isinstance(payload, dict):
        raise ValueError("batch must be an object")
    if payload == {"op": "describe"}:
        return None
    if set(payload) != {"table_epoch", "sequence", "actions"}:
        raise ValueError("expected table_epoch, sequence and actions only")
    if not isinstance(payload["table_epoch"], str) or len(payload["table_epoch"]) != 32:
        raise ValueError("invalid table_epoch")
    if type(payload["sequence"]) is not int or not 0 <= payload["sequence"] < 2**53:
        raise ValueError("invalid sequence")
    actions = payload["actions"]
    if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_ACTIONS:
        raise ValueError("batch needs 1 to 32 actions")
    for action in actions:
        if not isinstance(action, dict) or not isinstance(action.get("type"), str):
            raise ValueError("every action needs a string type")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("batch exceeds 64 KiB")
    return hashlib.sha256(raw).hexdigest()


def execute(table, payload):
    """Caller has resolved/authenticated table and preflighted ALL edit actions.

    One table lock, existing Table.action semantics, no transaction guarantee.
    Only latest receipt retained; older sequence numbers are rejected forever
    within this epoch, rather than executing again after cache eviction.
    """
    signature = validate(payload)
    with table.lock:
        ledger = getattr(table, "_action_batch", None)
        if ledger is None or ledger["game"] is not table.game:
            ledger = {"game": table.game, "epoch": secrets.token_hex(16), "next": 0, "last": None}
            table._action_batch = ledger
        base = {"table_epoch": ledger["epoch"], "next_sequence": ledger["next"],
                "max_actions": MAX_ACTIONS, "atomic": False, "durable_receipts": False}
        if signature is None:
            return base
        if payload["table_epoch"] != ledger["epoch"]:
            raise ValueError("table epoch changed; inspect state, do not replay")
        seq = payload["sequence"]
        last = ledger["last"]
        if last is not None and seq == last[0]:
            if signature != last[1]:
                raise ValueError("sequence reused with different actions")
            return copy.deepcopy(last[2])
        if seq != ledger["next"]:
            raise ValueError("stale or out-of-order sequence; do not replay")
        # Reserve before dispatch: even a failed action must never auto-replay.
        ledger["next"] += 1
        receipt = {**base, "sequence": seq, "next_sequence": ledger["next"],
                   "completed": 0, "status": "running", "state": None}
        ledger["last"] = (seq, signature, receipt)
        for index, action in enumerate(payload["actions"]):
            try:
                receipt["state"] = copy.deepcopy(table.action(action))
                receipt["completed"] += 1
            except Exception as error:
                receipt.update(status="stopped", failed_index=index,
                               failed_action_may_have_mutated=True,
                               error=f"{type(error).__name__}: {error}"[:512])
                break
        else:
            receipt["status"] = "completed"
        return copy.deepcopy(receipt)

"""Alpaca paper-account transport. The host is fixed; no live-order URL is accepted.

This module transports account, position and order state. Strategy attribution,
idempotency and risk admission belong to the paper arena before submit is called.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

PAPER_HOST = "https://paper-api.alpaca.markets"
MAX_RESPONSE = 2 * 1024 * 1024


class PaperBrokerError(Exception):
    def __init__(self, message: str, status: int = 502, uncertain: bool = False):
        super().__init__(message)
        self.status = status
        self.uncertain = uncertain


def _request(path: str, keys: tuple[str, str], method: str = "GET", body: dict | None = None):
    if not path.startswith("/v2/") or "://" in path or method not in ("GET", "POST", "DELETE"):
        raise PaperBrokerError("invalid paper broker request", 400)
    payload = None if body is None else json.dumps(body, separators=(",", ":"), allow_nan=False).encode("utf-8")
    headers = {"APCA-API-KEY-ID": keys[0], "APCA-API-SECRET-KEY": keys[1],
               "Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(PAPER_HOST + path, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read(MAX_RESPONSE + 1)
            if len(raw) > MAX_RESPONSE:
                raise PaperBrokerError("paper broker response exceeded size limit")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as error:
        raise PaperBrokerError(f"paper broker returned HTTP {error.code}",
                               404 if error.code == 404 else 429 if error.code == 429 else 502,
                               uncertain=method != "GET" and error.code >= 500) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise PaperBrokerError("paper broker request outcome is uncertain", uncertain=method != "GET") from None
    except (ValueError, UnicodeError):
        raise PaperBrokerError("paper broker response was invalid", uncertain=method != "GET") from None


def account(keys: tuple[str, str]) -> dict:
    result = _request("/v2/account", keys)
    if not isinstance(result, dict):
        raise PaperBrokerError("paper broker account response was invalid")
    return result


def positions(keys: tuple[str, str]) -> list[dict]:
    result = _request("/v2/positions", keys)
    if not isinstance(result, list):
        raise PaperBrokerError("paper broker positions response was invalid")
    return result


def orders(keys: tuple[str, str], status: str = "all") -> list[dict]:
    if status not in ("all", "open", "closed"):
        raise PaperBrokerError("invalid order status", 400)
    result = _request("/v2/orders?" + urllib.parse.urlencode({"status": status, "limit": 500}), keys)
    if not isinstance(result, list):
        raise PaperBrokerError("paper broker orders response was invalid")
    return result


def order_by_client_id(keys: tuple[str, str], client_order_id: str) -> dict | None:
    if not isinstance(client_order_id, str) or not 1 <= len(client_order_id) <= 128 or not client_order_id.isascii():
        raise PaperBrokerError("invalid client order id", 400)
    path = "/v2/orders:by_client_order_id?" + urllib.parse.urlencode({"client_order_id": client_order_id})
    try:
        result = _request(path, keys)
    except PaperBrokerError as error:
        if error.status == 404:
            return None
        raise
    if not isinstance(result, dict):
        raise PaperBrokerError("paper broker order response was invalid")
    return result


def submit(keys: tuple[str, str], order: dict) -> dict:
    result = _request("/v2/orders", keys, "POST", order)
    if not isinstance(result, dict):
        raise PaperBrokerError("paper broker order response was invalid", uncertain=True)
    return result

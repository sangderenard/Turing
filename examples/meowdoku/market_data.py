"""Bounded, read-only Alpaca market data for the Meowdoku paper-market view.

One account's own paper key is used server-side. The browser receives bars and
book levels only. No trading endpoint is present in this module.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from meowdoku import alpaca_credentials

DATA_HOST = "https://data.alpaca.markets"
ETFS = ("SPY", "QQQ", "IWM")
WILDCARDS = ("AAPL", "MSFT", "NVDA", "TSLA")
CRYPTO = "BTC/USD"
HISTORY_SECONDS = 300
NOW_SECONDS = 15
MAX_RESPONSE = 2 * 1024 * 1024
MAX_HISTORY_PAGES = 24
_lock = threading.Lock()
_cache: dict[tuple, tuple[float, dict]] = {}
_key_locks: dict[tuple, threading.Lock] = {}


def invalidate(account_id: int) -> None:
    with _lock:
        for key in tuple(_cache):
            if key[0] == account_id:
                _cache.pop(key, None)


class MarketError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def _stock(stock: str) -> str:
    if stock not in WILDCARDS:
        raise MarketError("unsupported stock selection", 400)
    return stock


def _fetch(path: str, params: dict, keys: tuple[str, str]) -> tuple[dict, dict]:
    url = DATA_HOST + path + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"APCA-API-KEY-ID": keys[0],
                                                  "APCA-API-SECRET-KEY": keys[1],
                                                  "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read(MAX_RESPONSE + 1)
            if len(raw) > MAX_RESPONSE:
                raise MarketError("Alpaca response was too large")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise MarketError("Alpaca response was invalid")
            quota = {"limit": response.headers.get("X-RateLimit-Limit"),
                     "remaining": response.headers.get("X-RateLimit-Remaining")}
            return data, quota
    except urllib.error.HTTPError as error:
        status = error.code
        raise MarketError(f"Alpaca market data returned HTTP {status}", 429 if status == 429 else 502) from None
    except (urllib.error.URLError, TimeoutError, ValueError):
        raise MarketError("Alpaca market data is unavailable") from None


def _history_pages(path: str, params: dict, keys: tuple[str, str]) -> tuple[dict, dict]:
    """Collect Alpaca's symbol-first pages without silently dropping later symbols."""
    bars: dict[str, list] = {}
    seen: set[str] = set()
    page_token = None
    quota = {}
    for _ in range(MAX_HISTORY_PAGES):
        page, quota = _fetch(path, {**params, **({"page_token": page_token} if page_token else {})}, keys)
        for symbol, series in (page.get("bars") or {}).items():
            if not isinstance(series, list):
                raise MarketError("Alpaca historical bars response was invalid")
            bars.setdefault(symbol, []).extend(series)
        page_token = page.get("next_page_token")
        if not page_token:
            return {"bars": bars}, quota
        if not isinstance(page_token, str) or page_token in seen:
            raise MarketError("Alpaca historical pagination was invalid")
        seen.add(page_token)
    raise MarketError("Alpaca historical bars exceeded the page limit")


def _cached(account_id: int, kind: str, stock: str, seconds: int, load) -> dict:
    key = (account_id, kind, stock)
    with _lock:
        key_lock = _key_locks.setdefault(key, threading.Lock())
    with key_lock:
        with _lock:
            prior = _cache.get(key)
            if prior and prior[0] > time.monotonic():
                return prior[1]
        value = load()
        with _lock:
            _cache[key] = (time.monotonic() + seconds, value)
        return value


def history(secrets_dir, account_id: int, stock: str = "AAPL") -> dict:
    stock = _stock(stock)
    keys = alpaca_credentials.load_paper(secrets_dir, account_id)

    def load():
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)
        window = {"start": start.isoformat().replace("+00:00", "Z"),
                  "end": end.isoformat().replace("+00:00", "Z"),
                  "timeframe": "1Hour", "limit": "10000"}
        stock_end = end - timedelta(minutes=20)
        equities, equity_quota = _history_pages("/v2/stocks/bars", {**window,
            "end": stock_end.isoformat().replace("+00:00", "Z"),
            "symbols": ",".join((*ETFS, stock)), "feed": "iex"}, keys)
        crypto, crypto_quota = _history_pages("/v1beta3/crypto/us/bars", {**window, "symbols": CRYPTO}, keys)
        return {"asof": end.isoformat(), "stock": stock, "etf_feed": "iex",
                "bars": {**{symbol: (equities.get("bars") or {}).get(symbol, []) for symbol in (*ETFS, stock)},
                         CRYPTO: (crypto.get("bars") or {}).get(CRYPTO, [])},
                "rate_limit": {"stocks": equity_quota, "crypto": crypto_quota}}

    return _cached(account_id, "history", stock, HISTORY_SECONDS, load)


def now(secrets_dir, account_id: int, stock: str = "AAPL") -> dict:
    stock = _stock(stock)
    keys = alpaca_credentials.load_paper(secrets_dir, account_id)

    def load():
        equities, equity_quota = _fetch("/v2/stocks/bars/latest", {"symbols": ",".join((*ETFS, stock)), "feed": "iex"}, keys)
        crypto, crypto_quota = _fetch("/v1beta3/crypto/us/latest/bars", {"symbols": CRYPTO}, keys)
        book = None
        book_note = None
        try:
            books, book_quota = _fetch("/v1beta3/crypto/us/latest/orderbooks", {"symbols": CRYPTO}, keys)
            book = (books.get("orderbooks") or {}).get(CRYPTO)
        except MarketError as error:
            book_note = str(error)
            book_quota = None
        return {"asof": datetime.now(timezone.utc).isoformat(), "stock": stock, "etf_feed": "iex",
                "bars": {**(equities.get("bars") or {}), **(crypto.get("bars") or {})},
                "book": book, "book_note": book_note,
                "rate_limit": {"stocks": equity_quota, "crypto": crypto_quota, "book": book_quota}}

    return _cached(account_id, "now", stock, NOW_SECONDS, load)

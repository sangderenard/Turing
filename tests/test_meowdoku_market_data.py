"""Read-only Alpaca request shapes and cache boundaries."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import alpaca_credentials, market_data  # noqa: E402


class MarketDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.secrets = Path(self.temp.name)
        alpaca_credentials.save_paper(self.secrets, 1, "paper-key-123", "paper-secret-123")
        market_data.invalidate(1)

    def test_history_uses_selected_stock_and_account_cache(self):
        calls = []

        def fetch(path, params, keys):
            calls.append((path, params, keys))
            symbol = "BTC/USD" if "crypto" in path else "SPY"
            return {"bars": {symbol: [{"t": "2026-01-01T00:00:00Z", "c": 1}]}}, {"remaining": "199"}

        with patch.object(market_data, "_fetch", side_effect=fetch):
            one = market_data.history(self.secrets, 1, "MSFT")
            self.assertEqual(one, market_data.history(self.secrets, 1, "MSFT"))
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][1]["feed"], "iex")
            self.assertEqual(calls[0][1]["symbols"], "SPY,QQQ,IWM,MSFT")
            self.assertEqual(calls[1][1]["symbols"], "BTC/USD")
            self.assertEqual(one["bars"]["BTC/USD"][0]["c"], 1)
            with self.assertRaises(market_data.MarketError):
                market_data.history(self.secrets, 1, "../OTHER")
            market_data.invalidate(1)
            market_data.history(self.secrets, 1, "MSFT")
            self.assertEqual(len(calls), 4)

    def test_latest_book_failure_keeps_bars(self):
        def fetch(path, params, keys):
            if "orderbooks" in path:
                raise market_data.MarketError("book unavailable")
            symbol = "BTC/USD" if "crypto" in path else "SPY"
            return {"bars": {symbol: {"t": "2026-01-01T00:00:00Z", "c": 1}}}, {}

        with patch.object(market_data, "_fetch", side_effect=fetch):
            result = market_data.now(self.secrets, 1)
        self.assertIsNone(result["book"])
        self.assertEqual(result["book_note"], "book unavailable")
        self.assertIn("BTC/USD", result["bars"])

    def test_latest_prices_share_one_cached_mark_set(self):
        calls = []

        def fetch(path, params, keys):
            calls.append((path, params["symbols"]))
            symbols = params["symbols"].split(",")
            return {"bars": {symbol: {"c": 100} for symbol in symbols}}, {}

        with patch.object(market_data, "_fetch", side_effect=fetch):
            first = market_data.latest_prices(self.secrets, 1)
            self.assertEqual(first, market_data.latest_prices(self.secrets, 1))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], "SPY,QQQ,IWM,AAPL,MSFT,NVDA,TSLA")
        self.assertEqual(first["BTC/USD"], 100)

    def test_history_collects_later_symbol_and_crypto_pages(self):
        calls = []

        def fetch(path, params, keys):
            calls.append((path, params.get("page_token")))
            if "stocks" in path:
                if not params.get("page_token"):
                    return {"bars": {"AAPL": [{"c": 1}]}, "next_page_token": "next-stock"}, {}
                return {"bars": {"SPY": [{"c": 2}], "QQQ": [{"c": 3}]}}, {}
            if not params.get("page_token"):
                return {"bars": {"BTC/USD": [{"c": 4}]}, "next_page_token": "next-crypto"}, {}
            return {"bars": {"BTC/USD": [{"c": 5}]}}, {}

        with patch.object(market_data, "_fetch", side_effect=fetch):
            result = market_data.history(self.secrets, 1)
        self.assertEqual([bar["c"] for bar in result["bars"]["BTC/USD"]], [4, 5])
        self.assertEqual(result["bars"]["SPY"][0]["c"], 2)
        self.assertEqual(result["bars"]["QQQ"][0]["c"], 3)
        self.assertEqual(calls, [("/v2/stocks/bars", None), ("/v2/stocks/bars", "next-stock"),
                                 ("/v1beta3/crypto/us/bars", None), ("/v1beta3/crypto/us/bars", "next-crypto")])


if __name__ == "__main__":
    unittest.main()

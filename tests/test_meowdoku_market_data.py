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


if __name__ == "__main__":
    unittest.main()

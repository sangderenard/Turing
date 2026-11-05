"""Player profiles and score records, shared by every session.

A profile is a display name (no password: it is a label, like an arcade
cabinet's initials) and, per game, the finished results recorded under it.
A session picks its profile with ``POST /api/profile``; the page remembers
the name and sends it again after a reload.  Results come from the games
themselves: a game that has finished returns ``result()`` (its ``score`` --
higher is better -- and what it means), and the table records it once.

The store is one JSON file under ``build/meowdoku`` (ignored by git), written
whole after each change; one lock serializes every session's writes.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

NAME = re.compile(r"^[A-Za-z0-9 _.-]{1,24}$")
RECENT = 20
TOP = 10


class Profiles:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {"profiles": {}}

    @staticmethod
    def clean(name: str) -> str:
        name = " ".join(str(name).split())
        if not NAME.match(name):
            raise ValueError("a name is 1-24 letters, digits, spaces, _ . or -")
        return name

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def ensure(self, name: str) -> str:
        name = self.clean(name)
        with self.lock:
            if name not in self.data["profiles"]:
                self.data["profiles"][name] = {"created": time.time(), "games": {}}
                self._save()
        return name

    def record(self, name: str, game: str, result: dict[str, Any]) -> dict[str, Any]:
        """Add one finished result; returns the game's record for ``name``."""
        entry = {"score": float(result["score"]), "detail": str(result.get("detail", "")), "at": time.time()}
        with self.lock:
            games = self.data["profiles"].setdefault(name, {"created": time.time(), "games": {}})["games"]
            book = games.setdefault(game, {"plays": 0, "best": None, "recent": []})
            book["plays"] += 1
            if book["best"] is None or entry["score"] > book["best"]["score"]:
                book["best"] = entry
            book["recent"] = ([entry] + book["recent"])[:RECENT]
            self._save()
            return json.loads(json.dumps(book))

    def profile(self, name: str | None) -> dict[str, Any] | None:
        if not name:
            return None
        with self.lock:
            found = self.data["profiles"].get(name)
            return None if found is None else {"name": name, **json.loads(json.dumps(found))}

    def best(self, name: str | None, game: str) -> dict[str, Any] | None:
        if not name:
            return None
        with self.lock:
            book = self.data["profiles"].get(name, {}).get("games", {}).get(game)
            return None if book is None else {"plays": book["plays"], "best": book["best"]}

    def leaderboard(self, game: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = [{"name": name, "plays": book["plays"], **book["best"]}
                    for name, profile in self.data["profiles"].items()
                    for g, book in profile["games"].items() if g == game and book["best"]]
        return sorted(rows, key=lambda row: -row["score"])[:TOP]

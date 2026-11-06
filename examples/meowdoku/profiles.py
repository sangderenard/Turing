"""Player profiles and score records, shared by every session.

A profile is a display name (no password: it is a label, like an arcade
cabinet's initials) and, per game, the finished results recorded under it.
A session picks its profile with ``POST /api/profile``; the page remembers
the name and sends it again after a reload.  Results come from the games
themselves: a game that has finished returns ``result()`` (its ``score`` --
higher is better -- and what it means), and the table records it once.

The store is one JSON file under ``build/meowdoku`` (ignored by git), written
whole after each change; one lock serializes every session's writes.

IDENTITY: a browser keeps a long-lived player token (a cookie); ``tokens``
maps it to a profile name.  A first visit gets a generated name, so a player
who just opens the page and plays is already somebody; typing a name re-binds
the token to it.

INVENTORY (per profile): ``points`` earned by finishing games -- a solved
puzzle pays its lives left times the success multiplier (1 + the run of
consecutive solves, at most MULTIPLIER_MAX), Wordle its unused guesses times
the same, the arcade games their score over ARCADE_RATE; points buy SHOP
items: banked puzzle power-ups (added to a puzzle's tray, spent after the
free ones) and an ONGOING power-up that lasts several puzzles.  Also saved:
the page's settings and each game's completed boards.  Everything rolls over
from visit to visit.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

NAME = re.compile(r"^[A-Za-z0-9 _.-]{1,24}$")
ADJECTIVES = ("Calico", "Tabby", "Ginger", "Velvet", "Comet", "Misty", "Pepper", "Mocha", "Biscuit", "Shadow",
              "Sunny", "Pixel", "Maple", "Clover", "Nimbus", "Echo", "Juniper", "Saffron", "Indigo", "Cosmo")
ANIMALS = ("Cat", "Kitten", "Lynx", "Ocelot", "Panther", "Tiger", "Puma", "Serval", "Caracal", "Jaguar")
MULTIPLIER_MAX = 5
ARCADE_RATE = 500.0
STREAK_GAMES = ("meowdoku", "wordle")
#: item -> (label, cost in points, what it is)
SHOP = {
    "cat": ("Cat power-up", 3, "bank"),
    "mouse": ("Mouse power-up", 2, "bank"),
    "hint": ("Hint bulb", 2, "bank"),
    "lucky": ("Lucky run: +1 of every power-up for the next 5 puzzles", 10, "ongoing"),
}
LUCKY_PUZZLES = 5
SETTING_KEYS = ("camMode", "pads", "bombStyle", "wordle", "palette")
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

    # identity ------------------------------------------------------------
    def by_token(self, token: str | None) -> str | None:
        if not token:
            return None
        with self.lock:
            name = self.data.setdefault("tokens", {}).get(token)
            return name if name in self.data["profiles"] else None

    def bind(self, token: str, name: str) -> None:
        with self.lock:
            self.data.setdefault("tokens", {})[token] = name
            self._save()

    def generated_name(self) -> str:
        """A fresh name nobody has: two words and a number."""
        import secrets as _s
        with self.lock:
            taken = set(self.data["profiles"])
        for _ in range(200):
            name = f"{_s.choice(ADJECTIVES)} {_s.choice(ANIMALS)} {_s.randbelow(90) + 10}"
            if name not in taken:
                return self.ensure(name)
        return self.ensure(f"Cat {_s.token_hex(3)}")

    # inventory ------------------------------------------------------------
    @staticmethod
    def _inventory(profile: dict) -> dict:
        inv = profile.setdefault("inventory", {})
        inv.setdefault("points", 0)
        inv.setdefault("streak", {})
        inv.setdefault("bank", {"cat": 0, "mouse": 0, "hint": 0})
        inv.setdefault("ongoing", None)
        inv.setdefault("settings", {})
        inv.setdefault("completed", {})
        return inv

    def award(self, name: str, game: str, result: dict) -> dict:
        """Points for one finished game (see the module notes); the streak
        grows with each success in a streak game."""
        with self.lock:
            inv = self._inventory(self.data["profiles"].setdefault(name, {"created": time.time(), "games": {}}))
            streak = int(inv["streak"].get(game, 0))
            if game in STREAK_GAMES:
                streak += 1
                inv["streak"][game] = streak
            multiplier = min(MULTIPLIER_MAX, streak) if game in STREAK_GAMES else 1
            if game == "meowdoku":
                points = int(result.get("lives", 0)) * multiplier
            elif game == "wordle":
                points = int(result.get("score", 0)) * multiplier
            else:
                points = int(float(result.get("score", 0.0)) // ARCADE_RATE)
            inv["points"] += points
            if result.get("board"):
                done = inv["completed"].setdefault(game, [])
                if result["board"] not in done:
                    done.append(result["board"])
            self._save()
            return {"points": points, "multiplier": multiplier, "streak": streak, "game": game}

    def break_streak(self, name: str, game: str) -> None:
        with self.lock:
            inv = self._inventory(self.data["profiles"].setdefault(name, {"created": time.time(), "games": {}}))
            if inv["streak"].get(game):
                inv["streak"][game] = 0
                self._save()

    def buy(self, name: str, item: str) -> None:
        if item not in SHOP:
            raise ValueError(f"no such item {item!r}")
        label, cost, kind = SHOP[item]
        with self.lock:
            inv = self._inventory(self.data["profiles"][name])
            if inv["points"] < cost:
                raise ValueError(f"{label} costs {cost} points; you have {inv['points']}")
            inv["points"] -= cost
            if kind == "bank":
                inv["bank"][item] = int(inv["bank"].get(item, 0)) + 1
            else:
                left = int((inv["ongoing"] or {}).get("puzzles", 0)) if (inv["ongoing"] or {}).get("item") == item else 0
                inv["ongoing"] = {"item": item, "puzzles": left + LUCKY_PUZZLES}
            self._save()

    def start_puzzle(self, name: str) -> tuple[dict, dict]:
        """At a puzzle's start: the banked power-ups (still banked until
        spent) and the ongoing power-up's free extras (one puzzle used)."""
        with self.lock:
            inv = self._inventory(self.data["profiles"][name])
            bonus = {"cat": 0, "mouse": 0, "hint": 0}
            ongoing = inv["ongoing"]
            if ongoing and ongoing.get("puzzles", 0) > 0:
                bonus = {"cat": 1, "mouse": 1, "hint": 1}
                ongoing["puzzles"] -= 1
                if ongoing["puzzles"] <= 0:
                    inv["ongoing"] = None
                self._save()
            return dict(inv["bank"]), bonus

    def set_bank(self, name: str, kind: str, count: int) -> None:
        with self.lock:
            inv = self._inventory(self.data["profiles"][name])
            inv["bank"][kind] = max(0, int(count))
            self._save()

    def save_settings(self, name: str, values: dict) -> None:
        with self.lock:
            inv = self._inventory(self.data["profiles"][name])
            for key in SETTING_KEYS:
                if key in values and len(json.dumps(values[key])) < 400:
                    inv["settings"][key] = values[key]
            self._save()

    def inventory(self, name: str | None) -> dict | None:
        if not name:
            return None
        with self.lock:
            profile = self.data["profiles"].get(name)
            if profile is None:
                return None
            inv = json.loads(json.dumps(self._inventory(profile)))
        inv["shop"] = [{"item": k, "label": label, "cost": cost} for k, (label, cost, _) in SHOP.items()]
        inv["multiplier_max"] = MULTIPLIER_MAX
        return inv

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

    def level(self, name: str | None, game: str) -> int:
        """The highest level ``name`` has defeated in ``game`` (0: none)."""
        if not name:
            return 0
        with self.lock:
            return int(self.data["profiles"].get(name, {}).get("games", {}).get(game, {}).get("level", 0))

    def defeated(self, name: str, game: str, level: int) -> None:
        """Record that ``name`` defeated ``level`` (kept as the maximum)."""
        with self.lock:
            games = self.data["profiles"].setdefault(name, {"created": time.time(), "games": {}})["games"]
            book = games.setdefault(game, {"plays": 0, "best": None, "recent": []})
            book["level"] = max(int(book.get("level", 0)), int(level))
            self._save()

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

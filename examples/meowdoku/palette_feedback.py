"""Per-participant site color experiment, using the existing game swatches.

The profile/browser or account key is supplied by server._job_identity. A rank
sets the initial color order; explicit good/bad feedback changes the next
candidate. Feedback is retained as data for later training, never pooled into
another participant's live theme.
"""
from __future__ import annotations

import json
import math
import secrets
import sqlite3
import time

from meowdoku.games import PALETTES

def _srgb(channel):
    return 12.92 * channel if channel <= 0.0031308 else 1.055 * channel ** (1 / 2.4) - 0.055


SWATCHES = tuple("#" + "".join(f"{round(255 * _srgb(channel)):02x}" for channel in rgb)
                 for rgb in PALETTES["Colour-blind safe"])
COUNT = len(SWATCHES)


class PaletteError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _connect(path):
    db = sqlite3.connect(str(path), timeout=10, isolation_level=None)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS palette_participants (
            owner TEXT PRIMARY KEY, ranking TEXT NOT NULL, candidate INTEGER,
            nonce TEXT, updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS palette_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL,
            color INTEGER NOT NULL, rating INTEGER NOT NULL, created REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS palette_feedback_owner_color
            ON palette_feedback(owner, color);
        CREATE INDEX IF NOT EXISTS palette_feedback_owner_time
            ON palette_feedback(owner, created);
    """)
    return db


def _row(db, owner):
    row = db.execute("SELECT ranking,candidate,nonce FROM palette_participants WHERE owner=?", (owner,)).fetchone()
    return (json.loads(row[0]), row[1], row[2]) if row else (None, None, None)


def _counts(db, owner):
    answer = {i: [0, 0] for i in range(COUNT)}
    for color, rating, n in db.execute("SELECT color,rating,COUNT(*) FROM palette_feedback WHERE owner=? GROUP BY color,rating", (owner,)):
        if color in answer:
            answer[color][0 if rating == 1 else 1] = n
    return answer


def _offer(db, owner, ranking, previous=None):
    counts = _counts(db, owner)
    def score(color):
        good, bad = counts[color]
        trials = good + bad
        prior = (COUNT - ranking.index(color)) * 0.3
        observed = 1.5 * (good - bad) / math.sqrt(trials + 1)
        explore = 1.0 / math.sqrt(trials + 1)
        return prior + observed + explore - (0.3 if color == previous else 0)
    candidate = max(ranking, key=score)
    nonce = secrets.token_urlsafe(18)
    db.execute("UPDATE palette_participants SET candidate=?,nonce=?,updated=? WHERE owner=?",
               (candidate, nonce, time.time(), owner))
    return candidate, nonce


def _state(db, owner):
    ranking, candidate, nonce = _row(db, owner)
    return {"swatches": [{"id": i, "color": c} for i, c in enumerate(SWATCHES)],
            "ranking": ranking, "candidate": ({"id": candidate, "color": SWATCHES[candidate],
                                                   "secondary": SWATCHES[ranking[1]], "nonce": nonce}
                                                  if candidate is not None and ranking else None),
            "feedback": {str(i): {"good": v[0], "bad": v[1]} for i, v in _counts(db, owner).items()}}


def route(handler, path, method, *, accounts_db):
    keys, _ = handler._job_identity()
    owner = keys[0]
    db = _connect(accounts_db)
    try:
        if method == "GET" and path == "/api/palette":
            return handler._json(_state(db, owner))
        if method != "POST" or path not in ("/api/palette/rank", "/api/palette/feedback"):
            raise PaletteError("not found", 404)
        if (handler.headers.get("Content-Type") or "").split(";", 1)[0].lower() != "application/json":
            raise PaletteError("JSON only", 415)
        body = handler._body_json(2048)
        if body is None:
            return
        if not isinstance(body, dict):
            raise PaletteError("expected an object")
        db.execute("BEGIN IMMEDIATE")
        try:
            ranking, candidate, nonce = _row(db, owner)
            if path == "/api/palette/rank":
                order = body.get("ranking")
                if not isinstance(order, list) or len(order) != COUNT or any(type(i) is not int for i in order) or set(order) != set(range(COUNT)):
                    raise PaletteError("ranking must list each swatch id exactly once")
                db.execute("INSERT INTO palette_participants(owner,ranking,updated) VALUES (?,?,?) "
                           "ON CONFLICT(owner) DO UPDATE SET ranking=excluded.ranking,updated=excluded.updated",
                           (owner, json.dumps(order), time.time()))
                _offer(db, owner, order)
            else:
                rating = body.get("rating")
                if ranking is None or candidate is None or not isinstance(nonce, str):
                    raise PaletteError("rank colors before giving feedback", 409)
                if body.get("nonce") != nonce or rating not in ("good", "bad"):
                    raise PaletteError("feedback must match the current palette and say good or bad", 409)
                daily = db.execute("SELECT COUNT(*) FROM palette_feedback WHERE owner=? AND created>?",
                                   (owner, time.time() - 86400)).fetchone()[0]
                if daily >= 60:
                    raise PaletteError("feedback limit reached for today", 429)
                db.execute("INSERT INTO palette_feedback(owner,color,rating,created) VALUES (?,?,?,?)",
                           (owner, candidate, 1 if rating == "good" else -1, time.time()))
                _offer(db, owner, ranking, previous=candidate)
            db.commit()
        except BaseException:
            db.rollback()
            raise
        return handler._json(_state(db, owner))
    except PaletteError as error:
        return handler._json({"error": str(error)}, error.status)
    finally:
        db.close()

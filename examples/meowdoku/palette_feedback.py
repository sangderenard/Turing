"""Per-participant site color experiment, using the existing game swatches.

The profile/browser or account key is supplied by server._job_identity. A rank
sets the initial color order; explicit good/bad feedback changes the next
candidate. Feedback is retained as data for later training, never pooled into
another participant's live theme.

Every design a participant is shown (accent plus secondary) is saved once in
palette_saved, and palette_active names the one their site wears. Select makes
any saved design active again; it becomes the candidate the next feedback rates.
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
        CREATE TABLE IF NOT EXISTS palette_saved (
            id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL,
            color INTEGER NOT NULL, secondary INTEGER NOT NULL, created REAL NOT NULL,
            UNIQUE(owner, color, secondary));
        CREATE TABLE IF NOT EXISTS palette_active (
            owner TEXT PRIMARY KEY, saved INTEGER NOT NULL, updated REAL NOT NULL);
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


def _save(db, owner, color, secondary):
    """Keeps this design (once) and makes it the one the participant's site wears."""
    now = time.time()
    db.execute("INSERT OR IGNORE INTO palette_saved(owner,color,secondary,created) VALUES (?,?,?,?)",
               (owner, color, secondary, now))
    saved = db.execute("SELECT id FROM palette_saved WHERE owner=? AND color=? AND secondary=?",
                       (owner, color, secondary)).fetchone()[0]
    db.execute("INSERT INTO palette_active(owner,saved,updated) VALUES (?,?,?) "
               "ON CONFLICT(owner) DO UPDATE SET saved=excluded.saved,updated=excluded.updated", (owner, saved, now))
    return saved


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
    _save(db, owner, candidate, _secondary(ranking, candidate))
    return candidate, nonce


def _secondary(ranking, color):
    """The participant's most preferred color other than the accent."""
    return next(c for c in ranking if c != color)


def _active(db, owner):
    """(saved id, color, secondary) of the design this participant's site wears, or None."""
    row = db.execute("SELECT s.id,s.color,s.secondary FROM palette_active a JOIN palette_saved s ON s.id=a.saved "
                     "WHERE a.owner=? AND s.owner=?", (owner, owner)).fetchone()
    if row:
        return row
    ranking, candidate, _ = _row(db, owner)
    if ranking and candidate is not None:          # a participant from before designs were saved
        secondary = _secondary(ranking, candidate)
        return _save(db, owner, candidate, secondary), candidate, secondary
    return None


def _state(db, owner):
    ranking, candidate, nonce = _row(db, owner)
    active = _active(db, owner)
    saved = [{"id": i, "color": SWATCHES[c], "secondary": SWATCHES[s], "created": t}
             for i, c, s, t in db.execute("SELECT id,color,secondary,created FROM palette_saved WHERE owner=? ORDER BY id",
                                          (owner,)) if 0 <= c < COUNT and 0 <= s < COUNT]
    return {"swatches": [{"id": i, "color": c} for i, c in enumerate(SWATCHES)],
            "ranking": ranking, "candidate": ({"id": active[1], "color": SWATCHES[active[1]],
                                               "secondary": SWATCHES[active[2]], "nonce": nonce, "saved": active[0]}
                                              if active and ranking and candidate is not None else None),
            "saved": saved, "active": active[0] if active else None,
            "feedback": {str(i): {"good": v[0], "bad": v[1]} for i, v in _counts(db, owner).items()}}


def route(handler, path, method, *, accounts_db):
    keys, _ = handler._job_identity()
    owner = keys[0]
    db = _connect(accounts_db)
    try:
        if method == "GET" and path == "/api/palette":
            return handler._json(_state(db, owner))
        if method != "POST" or path not in ("/api/palette/rank", "/api/palette/feedback", "/api/palette/select"):
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
            elif path == "/api/palette/select":
                chosen = body.get("id")
                if type(chosen) is not int:
                    raise PaletteError("id must be one of your saved palettes")
                row = db.execute("SELECT color,secondary FROM palette_saved WHERE id=? AND owner=?", (chosen, owner)).fetchone()
                if row is None or ranking is None:
                    raise PaletteError("no saved palette with that id", 404)
                db.execute("UPDATE palette_participants SET candidate=?,nonce=?,updated=? WHERE owner=?",
                           (row[0], secrets.token_urlsafe(18), time.time(), owner))
                _save(db, owner, row[0], row[1])
            else:
                rating = body.get("rating")
                if ranking is None or candidate is None or not isinstance(nonce, str):
                    raise PaletteError("rank colors before giving feedback", 409)
                if body.get("nonce") != nonce or rating not in ("good", "bad"):
                    raise PaletteError("feedback must match the current palette and say good or bad", 409)
                daily = db.execute("SELECT COUNT(*) FROM palette_feedback WHERE owner=? AND created>?",
                                   (owner, time.time() - 86400)).fetchone()[0]
                from meowdoku import rate_limits
                if rate_limits.enabled() and daily >= 60:
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

"""The color game: random designs, ranked and scored, teaching a native network.

A round is ``ROUND`` designs drawn at random (``palette_designs.random_design``:
one of the 148 named CSS colors with a random bevel and shadow), fresh for every
round and every participant.  Ranking a round stores every ordered pair (the
preferred design first).  The site then wears a proposed design -- the best
of a random pool as the network scores it for this participant -- and the
participant scores it on a continuous slider from 0 to 1.  Rankings and
scores train ``palette_model`` continuously: everyone's responses train the
shared network (the site default), each participant's own responses their
personal layer, which only nudges that default.  The comparison view shows
the pool's best design for everyone beside the best for this participant.

The owner key is the existing ``server._job_identity`` browser/account key.
``try again`` (``/api/palette/round``) draws a new round without recording
anything; ``clear`` (``/api/palette/clear``) forgets this participant's
responses, designs and personal layer.  Every design the site has worn is
kept (``saved``) and ``select`` wears one again.  The tables of the earlier
fixed-swatch game (``palette_participants``, ``palette_feedback``,
``palette_saved``, ``palette_active``) are left as they were.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import time

from . import palette_designs as designs

ROUND = 8              # designs in one ranking round
POOL = 48              # random designs scored for each proposal
EXPLORE = 0.2          # chance a proposal is drawn from the pool's top eight instead of its best
DAILY_LIMIT = 240      # responses (rankings + scores) per participant per day
_RNG = secrets.SystemRandom()


class PaletteError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _connect(path):
    db = sqlite3.connect(str(path), timeout=10, isolation_level=None)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS palette_round (
            owner TEXT PRIMARY KEY, nonce TEXT NOT NULL, designs TEXT NOT NULL, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS palette_pair (
            id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL,
            preferred TEXT NOT NULL, other TEXT NOT NULL, created REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS palette_pair_owner ON palette_pair(owner, id);
        CREATE TABLE IF NOT EXISTS palette_rating (
            id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL,
            design TEXT NOT NULL, secondary TEXT NOT NULL, score REAL NOT NULL, created REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS palette_rating_owner ON palette_rating(owner, id);
        CREATE TABLE IF NOT EXISTS palette_offer (
            owner TEXT PRIMARY KEY, nonce TEXT NOT NULL, design TEXT NOT NULL,
            secondary TEXT NOT NULL, scores TEXT NOT NULL, updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS palette_design_saved (
            id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL, design TEXT NOT NULL,
            secondary TEXT NOT NULL, key TEXT NOT NULL, created REAL NOT NULL, UNIQUE(owner, key));
        CREATE TABLE IF NOT EXISTS palette_design_active (
            owner TEXT PRIMARY KEY, saved INTEGER NOT NULL, updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS palette_personal (
            owner TEXT PRIMARY KEY, weights TEXT NOT NULL, steps INTEGER NOT NULL, updated REAL NOT NULL);
    """)
    return db


# -- the model ---------------------------------------------------------------
def _model(accounts_db):
    from . import palette_model
    return palette_model.model_for(accounts_db)


def _learner(accounts_db):
    """The model's readers and writer of this server's tables."""

    def responses(owner):
        db = _connect(accounts_db)
        try:
            def pairs(rows):
                return [(designs.features(json.loads(p)), designs.features(json.loads(o))) for p, o in rows]

            def ratings(rows):
                return [(designs.features(json.loads(d)), float(s)) for d, s in rows]

            own_pairs = pairs(db.execute("SELECT preferred,other FROM palette_pair WHERE owner=? ORDER BY id DESC LIMIT 64",
                                         (owner,)).fetchall()[::-1])
            own_ratings = ratings(db.execute("SELECT design,score FROM palette_rating WHERE owner=? ORDER BY id DESC LIMIT 16",
                                             (owner,)).fetchall()[::-1])
            all_pairs = pairs(db.execute("SELECT preferred,other FROM palette_pair ORDER BY id DESC LIMIT 64").fetchall()[::-1])
            all_ratings = ratings(db.execute("SELECT design,score FROM palette_rating ORDER BY id DESC LIMIT 16").fetchall()[::-1])
            return own_pairs, own_ratings, all_pairs, all_ratings, _personal(db, owner)
        finally:
            db.close()

    def store_personal(owner, weights, steps):
        db = _connect(accounts_db)
        try:
            db.execute("INSERT INTO palette_personal(owner,weights,steps,updated) VALUES (?,?,?,?) "
                       "ON CONFLICT(owner) DO UPDATE SET weights=excluded.weights,"
                       "steps=palette_personal.steps+excluded.steps,updated=excluded.updated",
                       (owner, json.dumps(weights), int(steps), time.time()))
        finally:
            db.close()

    return responses, store_personal


def _personal(db, owner):
    row = db.execute("SELECT weights FROM palette_personal WHERE owner=?", (owner,)).fetchone()
    return json.loads(row[0]) if row else None


# -- rounds, proposals, saved designs ----------------------------------------
def _new_round(db, owner):
    round_designs = designs.random_round(_RNG, ROUND)
    nonce = secrets.token_urlsafe(18)
    db.execute("INSERT INTO palette_round(owner,nonce,designs,created) VALUES (?,?,?,?) "
               "ON CONFLICT(owner) DO UPDATE SET nonce=excluded.nonce,designs=excluded.designs,created=excluded.created",
               (owner, nonce, json.dumps(round_designs), time.time()))
    return nonce, round_designs


def _round(db, owner):
    row = db.execute("SELECT nonce,designs FROM palette_round WHERE owner=?", (owner,)).fetchone()
    return (row[0], json.loads(row[1])) if row else _new_round(db, owner)


def _distinct(one, two):
    """Two designs a site can wear together: different colors, apart in OKLab."""
    a, b = designs.oklab(one["color"]), designs.oklab(two["color"])
    return one["color"] != two["color"] and sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5 > 0.1


def _choose(pool, scores, rng):
    order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
    first = order[0] if rng.random() >= EXPLORE else rng.choice(order[:8])
    second = next(i for i in order if i != first and _distinct(pool[first], pool[i]))
    return first, second


def _draft(model, personal):
    """A random pool and its network scores, for this participant and for everyone.

    Run before the request's database transaction: scoring waits for the
    model, and training (which holds the model) writes to the database."""
    pool = [designs.random_design(_RNG) for _ in range(POOL)]
    rows = [designs.features(design) for design in pool]
    yours = model.score(rows, personal) if model.ready else None
    everyone = model.score(rows, None) if model.ready else None
    return pool, yours, everyone


def _propose(db, owner, draft):
    """Offer the draft's best design for this participant."""
    pool, yours, everyone = draft
    if yours is None:
        # The network has not compiled yet: a random design, said plainly in ``model``.
        yours = everyone = [0.0] * len(pool)
        first = 0
        second = next(i for i in range(1, len(pool)) if _distinct(pool[0], pool[i]))
    else:
        first, second = _choose(pool, yours, _RNG)
    shared_best = max(range(len(pool)), key=lambda i: everyone[i])
    shared_second = next(i for i in sorted(range(len(pool)), key=lambda i: everyone[i], reverse=True)
                         if i != shared_best and _distinct(pool[shared_best], pool[i]))
    scores = {"yours": yours[first], "everyone": everyone[first],
              "everyone_best": {"design": pool[shared_best], "secondary": pool[shared_second],
                                "everyone": everyone[shared_best], "yours": yours[shared_best]}}
    nonce = secrets.token_urlsafe(18)
    db.execute("INSERT INTO palette_offer(owner,nonce,design,secondary,scores,updated) VALUES (?,?,?,?,?,?) "
               "ON CONFLICT(owner) DO UPDATE SET nonce=excluded.nonce,design=excluded.design,"
               "secondary=excluded.secondary,scores=excluded.scores,updated=excluded.updated",
               (owner, nonce, json.dumps(pool[first]), json.dumps(pool[second]), json.dumps(scores), time.time()))
    _save(db, owner, pool[first], pool[second])


def _key(design, secondary):
    return json.dumps([design, secondary], sort_keys=True)


def _save(db, owner, design, secondary):
    now = time.time()
    key = _key(design, secondary)
    db.execute("INSERT OR IGNORE INTO palette_design_saved(owner,design,secondary,key,created) VALUES (?,?,?,?,?)",
               (owner, json.dumps(design), json.dumps(secondary), key, now))
    saved = db.execute("SELECT id FROM palette_design_saved WHERE owner=? AND key=?", (owner, key)).fetchone()[0]
    db.execute("INSERT INTO palette_design_active(owner,saved,updated) VALUES (?,?,?) "
               "ON CONFLICT(owner) DO UPDATE SET saved=excluded.saved,updated=excluded.updated", (owner, saved, now))
    return saved


def _view(design):
    return {"name": design["name"], "color": design["color"], "square": designs.square_css(design)}


def _wear(design, secondary):
    return {**designs.site_css(design, secondary), "design": _view(design), "secondary_design": _view(secondary)}


def _state(db, owner, model):
    nonce, round_designs = _round(db, owner)
    offer = db.execute("SELECT nonce,design,secondary,scores FROM palette_offer WHERE owner=?", (owner,)).fetchone()
    active = db.execute("SELECT saved FROM palette_design_active WHERE owner=?", (owner,)).fetchone()
    saved = []
    for saved_id, design, secondary, created in db.execute(
            "SELECT id,design,secondary,created FROM palette_design_saved WHERE owner=? ORDER BY id", (owner,)):
        design, secondary = json.loads(design), json.loads(secondary)
        css = designs.site_css(design, secondary)
        saved.append({"id": saved_id, "color": css["color"], "secondary": css["secondary"],
                      "names": [design["name"], secondary["name"]], "created": created})
    candidate = comparison = None
    if offer:
        design, secondary, scores = json.loads(offer[1]), json.loads(offer[2]), json.loads(offer[3])
        candidate = {**_wear(design, secondary), "nonce": offer[0], "saved": active[0] if active else None,
                     "score": {"yours": scores["yours"], "everyone": scores["everyone"]}}
        best = scores["everyone_best"]
        comparison = {"yours": {**_wear(design, secondary), "score": {"yours": scores["yours"], "everyone": scores["everyone"]}},
                      "everyone": {**_wear(best["design"], best["secondary"]),
                                   "score": {"yours": best["yours"], "everyone": best["everyone"]}}}
    pairs, rounds = db.execute("SELECT COUNT(*), COUNT(DISTINCT created) FROM palette_pair WHERE owner=?", (owner,)).fetchone()
    ratings, mean = db.execute("SELECT COUNT(*), AVG(score) FROM palette_rating WHERE owner=?", (owner,)).fetchone()
    steps = db.execute("SELECT steps FROM palette_personal WHERE owner=?", (owner,)).fetchone()
    return {"round": {"nonce": nonce, "designs": [{"id": i, **_view(d)} for i, d in enumerate(round_designs)]},
            "swatches": [{"id": i, "color": d["color"]} for i, d in enumerate(round_designs)],
            "candidate": candidate, "comparison": comparison, "saved": saved,
            "active": active[0] if active else None,
            "history": {"rounds": rounds, "pairs": pairs, "ratings": ratings,
                        "mean_rating": None if mean is None else round(mean, 4),
                        "personal_steps": steps[0] if steps else 0},
            "model": model.status()}


def _daily(db, owner):
    since = time.time() - 86400
    return sum(db.execute(f"SELECT COUNT(*) FROM {table} WHERE owner=? AND created>?", (owner, since)).fetchone()[0]
               for table in ("palette_rating",)) + db.execute(
        "SELECT COUNT(DISTINCT created) FROM palette_pair WHERE owner=? AND created>?", (owner, since)).fetchone()[0]


def _limit(db, owner):
    from meowdoku import rate_limits
    if rate_limits.enabled() and _daily(db, owner) >= DAILY_LIMIT:
        raise PaletteError("response limit reached for today", 429)


ROUTES = ("/api/palette/rank", "/api/palette/feedback", "/api/palette/select",
          "/api/palette/round", "/api/palette/clear")


def route(handler, path, method, *, accounts_db):
    keys, _ = handler._job_identity()
    owner = keys[0]
    model = _model(accounts_db)
    db = _connect(accounts_db)
    learn = False
    try:
        # The model is consulted before the transaction opens (see ``_draft``).
        if method == "GET" and path == "/api/palette":
            draft = (_draft(model, _personal(db, owner))
                     if db.execute("SELECT 1 FROM palette_offer WHERE owner=?", (owner,)).fetchone() is None
                     else None)
            db.execute("BEGIN IMMEDIATE")
            try:
                if draft is not None and db.execute(
                        "SELECT 1 FROM palette_offer WHERE owner=?", (owner,)).fetchone() is None:
                    _propose(db, owner, draft)
                state = _state(db, owner, model)
                db.commit()
            except BaseException:
                db.rollback()
                raise
            return handler._json(state)
        if method != "POST" or path not in ROUTES:
            raise PaletteError("not found", 404)
        if (handler.headers.get("Content-Type") or "").split(";", 1)[0].lower() != "application/json":
            raise PaletteError("JSON only", 415)
        body = handler._body_json(4096)
        if body is None:
            return
        if not isinstance(body, dict):
            raise PaletteError("expected an object")
        draft = selected = None
        if path in ("/api/palette/rank", "/api/palette/feedback"):
            draft = _draft(model, _personal(db, owner))
        elif path == "/api/palette/clear":
            draft = _draft(model, None)
        elif path == "/api/palette/select" and type(body.get("id")) is int:
            chosen_row = db.execute("SELECT design FROM palette_design_saved WHERE id=? AND owner=?",
                                    (body["id"], owner)).fetchone()
            if chosen_row is not None:
                selected = _selected_scores(model, db, owner, json.loads(chosen_row[0]))
        db.execute("BEGIN IMMEDIATE")
        try:
            if path == "/api/palette/rank":
                nonce, round_designs = _round(db, owner)
                order = body.get("ranking")
                if body.get("nonce") not in (None, nonce):
                    raise PaletteError("that round is over; rank the current one", 409)
                if (not isinstance(order, list) or len(order) != len(round_designs)
                        or any(type(i) is not int for i in order) or set(order) != set(range(len(round_designs)))):
                    raise PaletteError("ranking must list each design of the round exactly once, favorite first")
                _limit(db, owner)
                now = time.time()
                db.executemany("INSERT INTO palette_pair(owner,preferred,other,created) VALUES (?,?,?,?)",
                               [(owner, json.dumps(round_designs[order[i]]), json.dumps(round_designs[order[j]]), now)
                                for i in range(len(order)) for j in range(i + 1, len(order))])
                _new_round(db, owner)
                _propose(db, owner, draft)
                learn = True
            elif path == "/api/palette/feedback":
                offer = db.execute("SELECT nonce,design,secondary FROM palette_offer WHERE owner=?", (owner,)).fetchone()
                rating = body.get("rating")
                if rating in ("good", "bad"):
                    rating = 1.0 if rating == "good" else 0.0
                if offer is None:
                    raise PaletteError("there is no design to score yet", 409)
                if (body.get("nonce") != offer[0] or isinstance(rating, bool)
                        or not isinstance(rating, (int, float)) or not 0.0 <= float(rating) <= 1.0):
                    raise PaletteError("score the current design with a number from 0 to 1", 409)
                _limit(db, owner)
                db.execute("INSERT INTO palette_rating(owner,design,secondary,score,created) VALUES (?,?,?,?,?)",
                           (owner, offer[1], offer[2], float(rating), time.time()))
                _propose(db, owner, draft)
                learn = True
            elif path == "/api/palette/select":
                chosen = body.get("id")
                if type(chosen) is not int:
                    raise PaletteError("id must be one of your saved palettes")
                row = db.execute("SELECT design,secondary FROM palette_design_saved WHERE id=? AND owner=?",
                                 (chosen, owner)).fetchone()
                if row is None or selected is None:
                    raise PaletteError("no saved palette with that id", 404)
                db.execute("INSERT INTO palette_offer(owner,nonce,design,secondary,scores,updated) VALUES (?,?,?,?,?,?) "
                           "ON CONFLICT(owner) DO UPDATE SET nonce=excluded.nonce,design=excluded.design,"
                           "secondary=excluded.secondary,scores=excluded.scores,updated=excluded.updated",
                           (owner, secrets.token_urlsafe(18), row[0], row[1], json.dumps(selected), time.time()))
                _save(db, owner, json.loads(row[0]), json.loads(row[1]))
            elif path == "/api/palette/round":
                _new_round(db, owner)
            else:
                for table in ("palette_round", "palette_pair", "palette_rating", "palette_offer",
                              "palette_design_saved", "palette_design_active", "palette_personal"):
                    db.execute(f"DELETE FROM {table} WHERE owner=?", (owner,))
                _propose(db, owner, draft)
            state = _state(db, owner, model)
            db.commit()
        except BaseException:
            db.rollback()
            raise
        if learn:
            model.learn(owner, *_learner(accounts_db))
        return handler._json(state)
    except PaletteError as error:
        return handler._json({"error": str(error)}, error.status)
    finally:
        db.close()


def _selected_scores(model, db, owner, design):
    """The scores of a design worn again, with the comparison it was offered beside."""
    row = designs.features(design)
    personal = _personal(db, owner)
    yours = model.score([row], personal) if model.ready else None
    everyone = model.score([row], None) if model.ready else None
    previous = db.execute("SELECT scores FROM palette_offer WHERE owner=?", (owner,)).fetchone()
    best = json.loads(previous[0])["everyone_best"] if previous else {
        "design": design, "secondary": design, "everyone": 0.0, "yours": 0.0}
    return {"yours": yours[0] if yours else 0.0, "everyone": everyone[0] if everyone else 0.0, "everyone_best": best}

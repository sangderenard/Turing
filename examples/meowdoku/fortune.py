"""fortune(): the I Ching style reading from the owner's fortune teller (Documents/AI Projects/Project Bootstrap/ft3.py).

Stages 1-4 of ft3.py run here as written; stage 5, the narrative, is the asker's (an LLM through the MCP, or a
person): the reply carries the reading and the order in which ft3.py says to weigh it.

    Stage 1  hash the question: sha256, the low 12 bits
    Stage 2  two pseudorandom 6-bit patterns
    Stage 3  XOR them in, split into two hexagrams, a concept fortune for each, the elemental (trigram) readout of the
             12-bit pattern read circularly, and the trigrams ordered by angle
    Stage 4  replace runs of the ordered trigrams with super elements
    Stage 5  (the asker) the story

    reading(question)                          -> the reading, no state
    ask(db, user_id=, question=, config=)      -> the reading, recorded; FortuneError(429) for a repeat inside the cool-down
    readings(db, user_id=, limit=)             -> the account's past readings, newest first

A repeated question waits: the same account asking the same question (case, spacing and surrounding punctuation
ignored) inside ``repeat_seconds`` is refused with the time left and the reading it already got. A new question
always gets a reading.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sqlite3
import time

DEFAULTS = {"repeat_seconds": 24 * 3600, "max_question": 2000}
DDL = (
    "CREATE TABLE IF NOT EXISTS fortune_readings (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, "
    "question_key TEXT NOT NULL, question TEXT NOT NULL, asked_at REAL NOT NULL, reading TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS fortune_readings_key ON fortune_readings(user_id, question_key, asked_at)",
)

_RNG = random.SystemRandom()


class FortuneError(Exception):
    def __init__(self, message: str, status: int = 400, **extra):
        super().__init__(message)
        self.message, self.status, self.extra = message, status, extra


# ---------------------------------------------------------------- ft3.py, stages 1-4

def hash_question(input_string):
    hex_dig = hashlib.sha256(input_string.encode()).hexdigest()
    return int(hex_dig, 16) & 0xfff                      # reduce to 12 bits


def generate_pseudorandom_patterns(rng=_RNG):
    return rng.randint(0, 63), rng.randint(0, 63)


def combine_patterns(deterministic_pattern, pattern1, pattern2):
    return (deterministic_pattern ^ ((pattern1 << 6) | pattern2)) & 0xfff


PI_SYMBOLS = ["🌅", "🌇", "⬆️", "⬇️"]
CATEGORY_SYMBOLS = {
    "00": ("Computorial", "🔒📡"),
    "01": ("Biologic", "🔒🌊"),
    "10": ("Symbolic", "🌐📡"),
    "11": ("Signal", "🌐🌊"),
}
ID_SYMBOL = {"1": "🔺", "0": "🔻"}
SF_SYMBOL = {"1": "⭐", "0": "💫"}
CONCEPTS = [
    "Concrete", "Abstract", "Analysis", "Synthesis", "Deduction", "Induction",
    "Certainty", "Doubt", "Individual", "Collective", "Independence", "Interdependence",
    "Self-Preservation", "Altruism", "Empathy", "Apathy", "Imagination", "Realism",
    "Innovation", "Tradition", "Spontaneity", "Planning", "Chaos", "Order",
    "Passion", "Pragmatism", "Optimism", "Pessimism", "Forgiveness", "Resentment",
    "Impulse", "Inhibition",
]


def calculate_concept_index(bp):
    # As in ft3.py: the category offset is computed but the index returned is the pattern mod 32.
    co = ["01", "10", "00", "11"]
    bi = co.index(bp[2:4]) * 8
    op = int(bp, 2)
    if op >= 32:
        bi += 32
    return op % 32


def generate_output(bp):
    pi, ci, id_, sf = bp[0:2], bp[2:4], bp[4], bp[5]
    concept = CONCEPTS[calculate_concept_index(bp)]
    category_name, category_symbol = CATEGORY_SYMBOLS[ci]
    symbols = PI_SYMBOLS[int(pi, 2)] + category_symbol + ID_SYMBOL[id_] + SF_SYMBOL[sf]
    return {"text": f"{concept}:{category_name},{symbols}", "concept": concept, "category": category_name,
            "symbols": symbols}


GUA = {'000': '☷', '001': '☲', '010': '☵', '011': '☳', '100': '☶', '101': '☴', '110': '☱', '111': '☰'}
GUA_NAMES = {'☷': "Earth", '☲': "Fire", '☵': "Water", '☳': "Thunder", '☶': "Mountain", '☴': "Wind/Wood",
             '☱': "Lake", '☰': "Heaven"}


def analyze_single_pattern_for_angles_only(pattern, sub_pattern_length=3):
    pattern_analysis = {}
    circular_pattern = pattern + pattern[:sub_pattern_length - 1]
    for sub_pattern, gua_symbol in GUA.items():
        thetas = [2 * math.pi * start / len(pattern) for start in range(len(pattern))
                  if circular_pattern[start:start + sub_pattern_length] == sub_pattern]
        if thetas:
            pattern_analysis[gua_symbol] = thetas
    return pattern_analysis


def reconstruct_gua_order_from_theta(elemental_readout):
    flattened = [(gua, theta) for gua, thetas in elemental_readout.items() for theta in thetas]
    return [gua for gua, _theta in sorted(flattened, key=lambda x: x[1])]


COMBINATIONS = [
    (['☲', '☵', '☶', '☳', '☰', '☷', '☴', '☱'], "Unity of all forces"),
    (['☲', '☵', '☴'], "Evaporation or Steam"),
    (['☲', '☵', '☶'], "Mud or Lava"),
    (['☵', '☴', '☶'], "Mist or Fog"),
    (['☲', '☴', '☶'], "Dust or Ash"),
    (['☲', '☵'], "Transformation and Balance"),
    (['☲', '☴'], "Inspiration and Energy"),
    (['☲', '☶'], "Creation and Destruction"),
    (['☵', '☴'], "Communication and Flexibility"),
    (['☵', '☶'], "Growth and Nurturing"),
    (['☴', '☶'], "Materialization and Reality"),
]


def replace_with_super_elements(pattern, combinations):
    extended_pattern = list(pattern) * 2
    super_elements = []
    replaced_indices = set()
    for combo in combinations:
        super_element = "Super" + ''.join(combo)
        i = 0
        while i < len(pattern):
            if all(extended_pattern[i + j] == combo[j] for j in range(len(combo))) and \
                    all(i + j not in replaced_indices for j in range(len(combo))):
                for k in range(i, i + len(combo)):
                    extended_pattern[k] = super_element
                    replaced_indices.add(k)
                if super_element not in super_elements:
                    super_elements.append(super_element)
                i += len(combo)
            else:
                i += 1
    updated_pattern = [extended_pattern[i] for i in range(len(pattern)) if i not in replaced_indices]
    unmatched_elements = [element for i, element in enumerate(pattern) if i not in replaced_indices]
    return updated_pattern, unmatched_elements, super_elements


NARRATIVE = ("Tell a story that prioritizes influence from the fortune in this order: the super element implications, "
             "with minor association with the super element's name or process and more with the cultural implications; "
             "the concept words and their categories; the icon set of the two fortunes; the unmatched elements.")


def reading(question, rng=_RNG):
    """Stages 1-4 of ft3.py for one question."""
    deterministic_pattern = hash_question(question)
    pattern1, pattern2 = generate_pseudorandom_patterns(rng)
    combined = format(combine_patterns(deterministic_pattern, pattern1, pattern2), '012b')
    hexagrams = [int(combined[:6], 2), int(combined[6:], 2)]
    fortunes = [generate_output(format(h, '06b')) for h in hexagrams]
    readout = analyze_single_pattern_for_angles_only(combined, 3)
    ordered = reconstruct_gua_order_from_theta(readout)
    named = {"Super" + ''.join(combo): name for combo, name in COMBINATIONS}
    _updated, unmatched, supers = replace_with_super_elements(ordered, [c for c, _n in COMBINATIONS])
    return {
        "question": question,
        "pattern": combined,
        "hexagrams": hexagrams,
        "fortunes": fortunes,
        "elemental_readout": {gua: [round(t, 2) for t in thetas] for gua, thetas in readout.items()},
        "elemental_pattern": ordered,
        "trigrams": {gua: GUA_NAMES[gua] for gua in dict.fromkeys(ordered)},
        "super_elements": [{"element": s, "meaning": named[s]} for s in supers],
        "unmatched_elements": unmatched,
        "narrative": NARRATIVE,
    }


# ---------------------------------------------------------------- the service: one reading per question per cool-down

def question_key(question: str) -> str:
    folded = re.sub(r"\s+", " ", question.casefold()).strip().strip(" ?!.,;:\"'")
    return hashlib.sha256(folded.encode("utf-8")).hexdigest()


def _ensure(db):
    for statement in DDL:
        db.execute(statement)


def ask(db, *, user_id, question, config=DEFAULTS, now=None, rng=_RNG):
    if not isinstance(question, str) or not question.strip():
        raise FortuneError("ask a question")
    question = question.strip()
    if len(question) > config["max_question"]:
        raise FortuneError(f"a question is at most {config['max_question']} characters")
    now = time.time() if now is None else now
    key = question_key(question)
    _ensure(db)
    last = db.execute("SELECT asked_at, reading FROM fortune_readings WHERE user_id=? AND question_key=? "
                      "ORDER BY asked_at DESC LIMIT 1", (user_id, key)).fetchone()
    if last is not None and now - last[0] < config["repeat_seconds"]:
        wait = int(math.ceil(config["repeat_seconds"] - (now - last[0])))
        raise FortuneError("you asked this already; ask something new, or ask this again later", 429,
                           retry_after_seconds=wait, previous=json.loads(last[1]), asked_at=last[0])
    made = reading(question, rng)
    made["asked_at"] = now
    cursor = db.execute("INSERT INTO fortune_readings (user_id, question_key, question, asked_at, reading) VALUES (?,?,?,?,?)",
                        (user_id, key, question, now, json.dumps(made, ensure_ascii=False)))
    made["id"] = cursor.lastrowid
    return made


def readings(db, *, user_id, limit=20):
    _ensure(db)
    limit = max(1, min(int(limit), 200))
    rows = db.execute("SELECT id, reading FROM fortune_readings WHERE user_id=? ORDER BY asked_at DESC, id DESC LIMIT ?",
                      (user_id, limit)).fetchall()
    return [{**json.loads(body), "id": row_id} for row_id, body in rows]


def http_route(self, path, method, *, accounts_db, config=DEFAULTS):
    """fortune() over REST, for any signed-in account.

        POST /api/fortune            {question}     a reading; 429 with retry_after_seconds and the previous reading
                                                    for a repeat inside the cool-down
        GET  /api/fortune/readings?limit=N          your past readings, newest first
    """
    from urllib.parse import parse_qs
    self._set_cookie = []
    if method == "POST":
        self.close_connection = True
    principal = self._principal()
    if principal.user_id is None:
        return self._json({"error": "sign in first"}, 401)
    db = sqlite3.connect(accounts_db, timeout=10, isolation_level=None)
    try:
        if method == "GET":
            if path != "/api/fortune/readings":
                return self._json({"error": "not found"}, 404)
            raw = (parse_qs(self.path.partition("?")[2]).get("limit") or ["20"])[0]
            if not raw.isdigit():
                raise FortuneError("limit must be a whole number")
            return self._json({"readings": readings(db, user_id=principal.user_id, limit=int(raw))})
        if path != "/api/fortune":
            return self._json({"error": "not found"}, 404)
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return self._json({"error": "JSON only"}, 415)
        body = self._body_json(config["max_question"] * 4 + 1024)
        if body is None:
            return
        if not isinstance(body, dict):
            raise FortuneError("a JSON object is required")
        return self._json(ask(db, user_id=principal.user_id, question=body.get("question"), config=config))
    except FortuneError as error:
        return self._json({"error": error.message, **error.extra}, error.status)
    finally:
        db.close()

"""fortune(): ft3.py's reading and the per-account cool-down for a repeated question, on a real SQLite database."""
import random
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku import fortune  # noqa: E402

ANN, BOB = 2, 3


def test_reading_follows_ft3():
    made = fortune.reading("hello", random.Random(7))
    assert len(made["pattern"]) == 12 and len(made["hexagrams"]) == 2
    assert int(made["pattern"], 2) ^ fortune.hash_question("hello") < 4096
    assert len(made["elemental_pattern"]) == 12            # one trigram starts at each of the 12 circular positions
    for item in made["fortunes"]:
        assert item["concept"] in fortune.CONCEPTS
    supers = {s["element"] for s in made["super_elements"]}
    assert all(e.startswith("Super") for e in supers)
    assert set(made["unmatched_elements"]) <= set(fortune.GUA.values())


def test_super_elements_as_written():
    pattern = ['☲', '☵', '☴', '☷', '☲', '☴']
    updated, unmatched, supers = fortune.replace_with_super_elements(pattern, [c for c, _ in fortune.COMBINATIONS])
    assert supers == ["Super☲☵☴", "Super☲☴"]
    assert unmatched == ['☷']


def test_repeat_waits_new_question_always_read(tmp_path):
    db = sqlite3.connect(tmp_path / "accounts.db", isolation_level=None)
    config = {**fortune.DEFAULTS, "repeat_seconds": 100}
    first = fortune.ask(db, user_id=ANN, question="Will it rain?", config=config, now=1000.0)
    with pytest.raises(fortune.FortuneError) as refused:
        fortune.ask(db, user_id=ANN, question="  will IT   rain ", config=config, now=1050.0)
    assert refused.value.status == 429
    assert refused.value.extra["retry_after_seconds"] == 50
    assert refused.value.extra["previous"]["pattern"] == first["pattern"]
    fortune.ask(db, user_id=ANN, question="Will it snow?", config=config, now=1051.0)   # a new question is read
    fortune.ask(db, user_id=BOB, question="Will it rain?", config=config, now=1052.0)   # another account is read
    fortune.ask(db, user_id=ANN, question="Will it rain?", config=config, now=1101.0)   # after the cool-down
    assert [r["question"] for r in fortune.readings(db, user_id=ANN)] == ["Will it rain?", "Will it snow?", "Will it rain?"]
    with pytest.raises(fortune.FortuneError):
        fortune.ask(db, user_id=ANN, question="   ", config=config)

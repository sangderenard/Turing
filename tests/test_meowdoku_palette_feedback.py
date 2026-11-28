"""The color game: random rounds, rankings, slider scores and the native network that learns from them."""
import http.client
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "examples"), str(Path(__file__).resolve().parent)]
from test_meowdoku_access import tiered  # noqa: F401,E402

from meowdoku import palette_designs as designs  # noqa: E402
from meowdoku import palette_model  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def compiled_model(tmp_path_factory):
    """One compiled network for the module: every test server shares its directory."""
    directory = tmp_path_factory.mktemp("palette_model")
    previous = os.environ.get(palette_model.DIRECTORY_ENVIRONMENT)
    os.environ[palette_model.DIRECTORY_ENVIRONMENT] = str(directory)
    yield directory
    if previous is None:
        os.environ.pop(palette_model.DIRECTORY_ENVIRONMENT, None)
    else:
        os.environ[palette_model.DIRECTORY_ENVIRONMENT] = previous


def _client(port):
    def call(method, path, body=None, cookie=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        headers = {"Content-Type": "application/json"}
        if cookie:
            headers["Cookie"] = cookie
        conn.request(method, path, json.dumps(body) if body is not None else None, headers)
        response = conn.getresponse()
        payload = json.loads(response.read())
        issued = response.getheader("Set-Cookie")
        conn.close()
        return response.status, payload, issued.split(";", 1)[0] if issued else None
    return call


def _ready(call, cookie, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, state, _ = call("GET", "/api/palette", cookie=cookie)
        assert status == 200
        assert state["model"]["state"] != "unavailable", state["model"]
        if state["model"]["state"] == "ready":
            return state
        time.sleep(1.0)
    raise AssertionError("the palette network did not compile")


def _model(directory):
    return palette_model._MODELS[str(Path(directory).resolve())]


def test_rounds_are_random_rankings_and_scores_are_recorded(tiered, compiled_model):
    port, _, _ = tiered
    call = _client(port)
    status, first, a = call("GET", "/api/palette")
    assert status == 200 and a
    state = _ready(call, a)
    round_one = state["round"]
    assert len(round_one["designs"]) == 8
    assert len({d["colors"][0] for d in round_one["designs"]}) == 8, "no two designs of a round start with one color"
    for d in round_one["designs"]:
        assert 1 <= len(d["colors"]) <= 6 and len(set(d["colors"])) == len(d["colors"])
        assert all(designs.NAMED[n] == c for n, c in zip(d["names"], d["colors"]))
        assert d["style"] in designs.STYLES and d["scheme"] in designs.SCHEMES
        assert set(designs.ELEMENTS) <= set(d["preview"]) and isinstance(d["preview"]["surface_shadow"], str)
    # Try again draws a new random round and records nothing.
    status, again, _ = call("POST", "/api/palette/round", {}, a)
    assert status == 200 and again["round"]["nonce"] != round_one["nonce"]
    assert [d["names"] for d in again["round"]["designs"]] != [d["names"] for d in round_one["designs"]]
    assert again["history"]["pairs"] == 0
    # A stale round is refused; the current one ranks into every ordered pair.
    order = [d["id"] for d in again["round"]["designs"]][::-1]
    assert call("POST", "/api/palette/rank", {"nonce": round_one["nonce"], "ranking": order}, a)[0] == 409
    assert call("POST", "/api/palette/rank", {"nonce": again["round"]["nonce"], "ranking": [0] * 8}, a)[0] == 400
    status, ranked, _ = call("POST", "/api/palette/rank", {"nonce": again["round"]["nonce"], "ranking": order}, a)
    assert status == 200 and ranked["history"]["pairs"] == 28 and ranked["history"]["rounds"] == 1
    assert ranked["round"]["nonce"] != again["round"]["nonce"]
    candidate = ranked["candidate"]
    assert candidate and all(n in designs.NAMED for n in candidate["design"]["names"])
    assert candidate["color"].startswith("#") and candidate["variables"]["--palette-fg"].startswith("light-dark(#")
    assert candidate["variables"]["--palette-surface-radius"].endswith("px")
    preview = candidate["design"]["preview"]        # readability is enforced, not learned
    assert designs.contrast(preview["text"], preview["surface"]) >= 4.5
    assert designs.contrast(preview["link"], preview["surface"]) >= 4.5
    assert set(ranked["comparison"]) == {"yours", "everyone"}
    # The slider score: a number from 0 to 1 for the current design.
    nonce = candidate["nonce"]
    for bad in (1.5, -0.1, "nice", True, None):
        assert call("POST", "/api/palette/feedback", {"nonce": nonce, "rating": bad}, a)[0] == 409
    status, scored, _ = call("POST", "/api/palette/feedback", {"nonce": nonce, "rating": 0.8}, a)
    assert status == 200 and scored["history"]["ratings"] == 1 and scored["history"]["mean_rating"] == 0.8
    assert scored["candidate"]["nonce"] != nonce
    assert call("POST", "/api/palette/feedback", {"nonce": nonce, "rating": 0.3}, a)[0] == 409, "a design is scored once"
    status, scored, _ = call("POST", "/api/palette/feedback", {"nonce": scored["candidate"]["nonce"], "rating": "bad"}, a)
    assert status == 200 and scored["history"]["ratings"] == 2, "good and bad still mean 1 and 0"
    # Another browser has its own rounds and history.
    status, other, b = call("GET", "/api/palette")
    assert status == 200 and b != a and other["history"]["pairs"] == 0
    assert [d["names"] for d in other["round"]["designs"]] != [d["names"] for d in scored["round"]["designs"]]
    # Saved designs are kept and can be worn again.
    saved = scored["saved"]
    assert len(saved) >= 3 and scored["active"] == saved[-1]["id"]
    status, worn, _ = call("POST", "/api/palette/select", {"id": saved[0]["id"]}, a)
    assert status == 200 and worn["active"] == saved[0]["id"]
    assert worn["candidate"]["design"]["colors"] == saved[0]["colors"]
    assert call("POST", "/api/palette/select", {"id": saved[0]["id"]}, b)[0] == 404
    # Clear forgets this participant only.
    _model(compiled_model).drain()
    status, cleared, _ = call("POST", "/api/palette/clear", {}, a)
    assert status == 200 and cleared["history"] == {"rounds": 0, "pairs": 0, "ratings": 0, "mean_rating": None,
                                                    "personal_steps": 0}
    assert len(cleared["saved"]) == 1, "clearing starts again from one fresh proposal"


def test_the_network_learns_a_preference_continuously(tiered, compiled_model):
    port, _, _ = tiered
    call = _client(port)
    _, _, a = call("GET", "/api/palette")
    state = _ready(call, a)
    model = _model(compiled_model)
    shared_before = model.status()["shared_steps"]
    # A participant who always prefers lighter cards (the OKLab lightness of the surface, one of the features).
    # The slider spans the surface's dark lightness band (0 at its darkest, 1 at its lightest), as a person would.
    low, high = designs.ELEMENTS["surface"][2]

    def surface_lightness(view):
        return min(1.0, max(0.0, (designs.oklab(view["preview"]["surface"])[0] - low) / (high - low)))

    for _round in range(6):
        ranked = sorted(state["round"]["designs"], key=surface_lightness, reverse=True)
        status, state, _ = call("POST", "/api/palette/rank",
                                {"nonce": state["round"]["nonce"], "ranking": [d["id"] for d in ranked]}, a)
        assert status == 200
        lightness = surface_lightness(state["candidate"]["design"])
        status, state, _ = call("POST", "/api/palette/feedback",
                                {"nonce": state["candidate"]["nonce"], "rating": round(lightness, 3)}, a)
        assert status == 200
    assert model.drain(), "every queued training pass ran"
    _, state, _ = call("GET", "/api/palette", cookie=a)
    assert state["model"]["shared_steps"] > shared_before
    assert state["history"]["personal_steps"] > 0
    assert "error" not in state["model"], state["model"]     # no training pass failed (e.g. a locked database)
    # The trained network (shared plus this participant's layer) now ranks held-out designs by lightness.
    import random
    rng = random.Random(7)
    held_out = [designs.random_design(rng) for _ in range(64)]
    rows = [designs.features(d) for d in held_out]
    import sqlite3
    weights = json.loads(sqlite3.connect(str(tiered[2].accounts_db)).execute(
        "SELECT weights FROM palette_personal").fetchone()[0])
    scores = model.score(rows, weights)
    light = [designs.oklab(designs.preview(d)["surface"])[0] for d in held_out]
    agree = np.mean([(scores[i] > scores[j]) == (light[i] > light[j])
                     for i in range(64) for j in range(i + 1, 64) if light[i] != light[j]])
    assert agree > 0.75, agree

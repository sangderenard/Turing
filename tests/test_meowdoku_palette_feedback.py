"""Personal palette state and feedback stay with the browser identity."""
import http.client
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "examples"), str(Path(__file__).resolve().parent)]
from test_meowdoku_access import tiered  # noqa: F401


def test_ranking_feedback_and_browser_isolation(tiered):
    port, _, sessions = tiered

    def call(method, path, body=None, cookie=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        headers = {"Content-Type": "application/json"}
        if cookie:
            headers["Cookie"] = cookie
        conn.request(method, path, json.dumps(body) if body is not None else None, headers)
        response = conn.getresponse()
        payload = json.loads(response.read())
        issued = response.getheader("Set-Cookie")
        conn.close()
        return response.status, payload, issued.split(";", 1)[0] if issued else None

    status, first, a = call("GET", "/api/palette")
    assert status == 200 and a and first["candidate"] is None
    order = [s["id"] for s in first["swatches"]][::-1]
    status, first, _ = call("POST", "/api/palette/rank", {"ranking": order}, a)
    assert status == 200 and first["ranking"] == order and first["candidate"]
    nonce = first["candidate"]["nonce"]
    color = first["candidate"]["id"]
    status, changed, _ = call("POST", "/api/palette/feedback", {"rating": "bad", "nonce": nonce}, a)
    assert status == 200 and changed["feedback"][str(color)]["bad"] == 1
    assert changed["candidate"]["nonce"] != nonce
    assert changed["candidate"]["id"] != color, "bad feedback should change this participant's design"
    assert call("POST", "/api/palette/feedback", {"rating": "good", "nonce": nonce}, a)[0] == 409
    status, second, b = call("GET", "/api/palette")
    assert status == 200 and b != a and second["ranking"] is None
    assert all(v == {"good": 0, "bad": 0} for v in second["feedback"].values())
    status, second, _ = call("POST", "/api/palette/rank", {"ranking": order}, b)
    same = second["candidate"]["id"]
    status, second, _ = call("POST", "/api/palette/feedback",
                             {"rating": "good", "nonce": second["candidate"]["nonce"]}, b)
    assert status == 200 and second["candidate"]["id"] == same, "good feedback should keep a satisfying design"
    assert call("POST", "/api/palette/rank", {"ranking": [0] * len(order)}, a)[0] == 400

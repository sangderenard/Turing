"""The tip jar (examples/meowdoku/tipjar.py): vout summing, the size mapping, watch-only refusal, the poller's cache and fail-soft,
BIP84 derivation against the published vectors, and a deterministic jar for the committed DEMO fixture.  No network (the poller is
handed a fake ``fetch``)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku import tipjar as T  # noqa: E402

ZPUB = "zpub6rFR7y4Q2AijBEqTUquhVz398htDFrtymD9xYYfG1m4wAcvPhXNfE3EfH1r1ADqtfSdVCToUG868RvUUkgDKf31mGDtKsAYz2oz2AGutZYs"   # BIP84, "abandon ... about", account 0
BIP84_RECEIVE = ["bc1qcr8te4kr609gcawutmrza0j4xv80jy8z306fyu", "bc1qnjg0jd8228aq7egyzacy8cys3knf9xvrerkf9g"]
BIP84_CHANGE0 = "bc1q8c6fshw2dlwun7ekn9qwf37cu2rn755upcp6el"


# ------------------------------------------------------------------ vouts
def _tx(vout, vin_addr="SENDER", kind="v0_p2wpkh", confirmed=True, t=1_700_000_000, txid="aa"):
    return {"txid": txid, "vin": [{"prevout": {"scriptpubkey_address": vin_addr, "scriptpubkey_type": kind}}],
            "vout": [{"scriptpubkey_address": a, "value": v} for a, v in vout], "status": {"confirmed": confirmed, **({"block_time": t} if confirmed else {})}}


def test_received_is_the_sum_of_the_vouts_paying_the_jar_and_change_is_not_ours():
    mine = {"JAR"}
    assert T.donation_of(_tx([("JAR", 5000), ("SENDER_CHANGE", 90000)]), mine)["sats"] == 5000          # the sender's change is not ours
    assert T.donation_of(_tx([("JAR", 5000), ("OTHER", 1), ("JAR", 700)]), mine)["sats"] == 5700         # a multi-output payment to the jar
    assert T.donation_of(_tx([("OTHER", 5000)]), mine) is None
    assert T.donation_of(_tx([("JAR", 123), ("X", 5)], vin_addr="JAR"), mine) is None                     # the wallet spending its own coin: its change is no donation
    two = T.donation_of(_tx([("JAR0", 10), ("JAR1", 32)]), {"JAR0", "JAR1"}, lambda a: int(a[-1]))
    assert two["sats"] == 42 and two["slot"] == 0
    d = T.donation_of(_tx([("JAR", 1)], confirmed=False, kind="v1_p2tr"), mine)
    assert d["confirmed"] is False and d["time"] is None and d["kind"] == "v1_p2tr"


def test_the_demo_fixture_has_change_multi_output_unconfirmed_and_a_self_spend():
    data = json.loads(T.FIXTURE_PATH.read_text(encoding="utf-8"))
    mine = {data["address"]}
    donations = [d for d in (T.donation_of(tx, mine) for tx in data["txs"]) if d]
    assert len(donations) == len(data["txs"]) - 1                                  # exactly the self-spend is dropped
    assert sum(not d["confirmed"] for d in donations) == 3
    assert any(sum(v["value"] for v in tx["vout"] if v["scriptpubkey_address"] in mine) != next(v["value"] for v in tx["vout"] if v["scriptpubkey_address"] in mine)
               for tx in data["txs"][:-1])                                          # a transaction with two outputs to the jar
    assert data["address"] == "DEMO_JAR" and "bc1" not in json.dumps(data)         # no real address


# ------------------------------------------------------------------ sizes
@pytest.mark.parametrize("sats", [[1000, 2000, 8000], [546, 10_000, 3_000_000, 100_000_000], [5, 5], [7]])
def test_the_radius_mapping_is_monotone_and_clamped(sats):
    lo, hi = T.R_MIN, 0.014
    radii, method = T.radius_map(sats, lo, hi)
    assert all(lo - 1e-12 <= r <= hi + 1e-12 for r in radii)
    order = np.argsort(sats, kind="stable")
    assert all(radii[order[i]] <= radii[order[i + 1]] + 1e-12 for i in range(len(sats) - 1))
    assert method == {0: "cbrt", 1: "log", 2: "single", 3: "single"}[[[1000, 2000, 8000], [546, 10_000, 3_000_000, 100_000_000], [5, 5], [7]].index(sats)]


def test_cbrt_is_volume_proportional_when_the_range_fits():
    radii, method = T.radius_map([1000, 8000], 0.005, 0.014)
    assert method == "cbrt" and radii[1] / radii[0] == pytest.approx(2.0)        # 8x the sats: 8x the volume: 2x the radius


def test_ids_are_assigned_so_no_grain_exceeds_its_species_radius():
    id_radius = np.array([0.009, 0.0105, 0.012, 0.014, 0.009, 0.014])
    radii = [0.013, 0.005, 0.0095, 0.014]
    ids = T.assign_ids(radii, id_radius)
    assert len(set(ids.tolist())) == 4 and all(id_radius[i] >= r - 1e-12 for i, r in zip(ids, radii))


# ------------------------------------------------------------------ watch-only
def _wif() -> str:
    return T.b58encode_check(b"\x80" + bytes(range(1, 33)) + b"\x01")


def _xprv() -> str:
    return T.b58encode_check(bytes.fromhex("0488ade4") + b"\x00" * 9 + bytes(range(32)) + b"\x00" + bytes(range(1, 33)))


@pytest.mark.parametrize("secret", [_wif(), _xprv(), " ".join(["abandon"] * 11 + ["about"]), " ".join(["word"] * 24), "ab" * 32])
def test_private_keys_and_seeds_are_refused(tmp_path, capsys, secret):
    assert T.looks_secret(secret)
    path = tmp_path / "tipjar.json"
    path.write_text(json.dumps({"address": "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4", "note": secret}), encoding="utf-8")
    cfg = T.load_config(path)
    assert not cfg.configured and cfg.refused
    assert "REFUSED" in capsys.readouterr().out


def test_public_config_is_accepted_and_missing_means_demo(tmp_path):
    path = tmp_path / "tipjar.json"
    path.write_text(json.dumps({"address": "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4", "xpub": ZPUB, "show_txid": True}), encoding="utf-8")
    cfg = T.load_config(path)
    assert cfg.configured and cfg.show_txid and cfg.xpub == ZPUB
    assert not T.load_config(tmp_path / "absent.json").configured
    assert T.looks_secret("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4") is None and T.looks_secret(ZPUB) is None


# ------------------------------------------------------------------ the poller
def _fake(history):
    calls = []

    def fetch(path):
        calls.append(path)
        if path.startswith("/v1/prices"):
            return {"USD": 50000.0}
        if path == "/address/JAR":
            return {"chain_stats": {"funded_txo_sum": 9000, "spent_txo_sum": 0, "tx_count": 1}, "mempool_stats": {"funded_txo_sum": 500, "spent_txo_sum": 0, "tx_count": 1}}
        if path == "/address/JAR/txs":
            return history
        raise AssertionError(path)
    return fetch, calls


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def test_one_upstream_refresh_per_interval_however_many_viewers():
    fetch, calls = _fake([_tx([("JAR", 9000)], txid="a1"), _tx([("JAR", 500)], confirmed=False, txid="a2")])
    clock = Clock()
    poller = T.Poller(T.Config(address="JAR"), fetch=fetch, clock=clock, interval=60.0)
    for _ in range(25):
        snap = poller.snapshot()
    assert poller.refreshes == 1 and len(calls) == 3                      # address totals, txs, prices: ONE refresh
    assert snap["balance"] == 9500 and [d["sats"] for d in snap["donations"]] == [9000, 500]       # confirmed first, then the mempool one
    clock.t += 59.0
    poller.snapshot()
    assert poller.refreshes == 1
    clock.t += 2.0
    poller.snapshot()
    assert poller.refreshes == 2 and len(calls) == 6


def test_a_failed_poll_keeps_the_last_good_list():
    fetch, _calls = _fake([_tx([("JAR", 9000)], txid="a1")])
    clock = Clock()
    poller = T.Poller(T.Config(address="JAR"), fetch=fetch, clock=clock, interval=60.0)
    good = poller.snapshot()
    assert good["error"] is None and len(good["donations"]) == 1

    def down(path):
        raise OSError("offline")
    poller.fetch = down
    clock.t += 61.0
    soft = poller.snapshot()
    assert soft["donations"] == good["donations"] and soft["balance"] == good["balance"] and "offline" in soft["error"]


def test_the_demo_never_touches_the_network():
    def boom(path):
        raise AssertionError("network")
    poller = T.Poller(T.Config(), fetch=boom)
    snap = poller.snapshot()
    assert snap["demo"] and snap["address"] is None and len(snap["donations"]) == 48


def test_http_helper_is_the_only_urllib_use(monkeypatch):
    seen = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": 1}'
    monkeypatch.setattr(T.urllib.request, "urlopen", lambda req, timeout=0: seen.append(req.full_url) or Resp())
    assert T.http_get_json("/address/x") == {"ok": 1} and seen == ["https://mempool.space/api/address/x"]


def test_xpub_polling_derives_addresses_with_a_gap_limit_and_offers_a_fresh_one():
    used = {BIP84_RECEIVE[0]: [_tx([(BIP84_RECEIVE[0], 7000)], txid="b1")]}
    calls = []

    def fetch(path):
        calls.append(path)
        parts = path.split("/")
        if path == "/v1/prices":
            return {"USD": 1.0}
        addr = parts[2]
        if path.endswith("/txs"):
            return used.get(addr, [])
        n = len(used.get(addr, []))
        return {"chain_stats": {"funded_txo_sum": 7000 * n, "spent_txo_sum": 0, "tx_count": n}, "mempool_stats": {"funded_txo_sum": 0, "spent_txo_sum": 0, "tx_count": 0}}
    poller = T.Poller(T.Config(xpub=ZPUB), fetch=fetch, interval=60.0)
    snap = poller.snapshot()
    assert snap["balance"] == 7000 and snap["address"] == BIP84_RECEIVE[1]            # the first unused address is shown to donors
    assert sum(1 for c in calls if c.count("/") == 2 and c.startswith("/address/")) >= 1 + T.GAP_LIMIT


# ------------------------------------------------------------------ BIP84 and the hashes
def test_ripemd160_and_bech32_against_published_vectors():
    assert T.ripemd160(b"").hex() == "9c1185a5c5e9fc54612808977ee8f548b2258d31"
    assert T.ripemd160(b"abc").hex() == "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc"
    assert T.ripemd160(b"a" * 100).hex() == __import__("hashlib").new("ripemd160", b"a" * 100).hexdigest() if "ripemd160" in __import__("hashlib").algorithms_available else True
    pub = bytes.fromhex("0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")          # BIP173's example key
    assert T.bip84_address(pub) == "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"


def test_bip84_receiving_addresses_match_the_published_vectors():
    assert [T.derive_receive(ZPUB, i) for i in range(2)] == BIP84_RECEIVE
    assert T.derive_receive(ZPUB, 0, chain_index=1) == BIP84_CHANGE0


def test_an_extended_private_key_is_not_an_xpub():
    with pytest.raises(ValueError):
        T.parse_xpub(_xprv())


# ------------------------------------------------------------------ colours
def test_colours_encode_source_and_age_and_decode_back():
    for kind in ("p2pkh", "p2sh", "v0_p2wpkh", "v0_p2wsh", "v1_p2tr"):
        for slot in (0, 1, 2):
            for age in (0.0, 0.5, 1.0):
                back = T.decode_rgb(T.grain_rgb(kind, slot, age, True))
                assert back["kind"] == kind and back["confirmed"] and abs(back["age01"] - age) < 0.03 and back["slot_step"] == (slot % 3) - 1
            pending = T.decode_rgb(T.grain_rgb(kind, slot, 0.3, False))
            assert pending["kind"] == kind and not pending["confirmed"]


# ------------------------------------------------------------------ the jar (deterministic, for the fixture)
def test_the_demo_jar_is_a_pure_function_of_the_tx_list():
    snap = T.Poller(T.Config()).snapshot()
    a = T.build_jar(snap["donations"], 256)
    b = T.build_jar(list(snap["donations"]), 256)
    assert a.digest == b.digest == T.tx_hash(snap["donations"])
    assert a.payload() == b.payload()
    assert len(set(a.ids.tolist())) == a.count == 48 and 0 in a.ids                  # id 0's emission time is 0: it must be a grain
    cols = a.columns(256)
    placed = np.zeros(256, bool)
    placed[a.ids] = True
    assert np.all(cols["x"][~placed] == 0) and np.all(cols["y"][~placed] == 0) and np.all(cols["z"][~placed] == 0)      # spare ids parked
    assert np.all(cols["age"][placed] > 1e6) and np.all(cols["age"][~placed] == 0)
    assert float(cols["emit_rate"]) * 1.0 > 1.0e3                                      # no spare id's emission time ever comes
    assert cols["rs"][placed].min() >= T.R_MIN - 1e-6
    # chronological from the top: a later donation starts no lower than an earlier one (in layers)
    order = {int(i): k for k, i in enumerate(a.ids)}
    ys = np.array([cols["y"][a.ids[k]] for k in range(a.count)])
    assert ys[-1] >= ys[0]
    # the radius follows the amount, and never exceeds the species radius of the id (the law's own cap)
    sats = np.array([d["sats"] for d in a.donations])
    rs = np.array([cols["rs"][i] for i in a.ids])
    assert np.all(np.diff(rs[np.argsort(sats, kind="stable")]) >= -1e-6)
    assert order


def test_a_new_donation_changes_the_jar_and_the_tail_is_the_newest():
    snap = T.Poller(T.Config()).snapshot()
    more = snap["donations"] + [{"txid": "ff" * 32, "sats": 250_000, "confirmed": False, "time": None, "kind": "v1_p2tr", "slot": 0}]
    a, b = T.build_jar(snap["donations"], 256), T.build_jar(more, 256)
    assert a.digest != b.digest and b.count == a.count + 1
    assert b.donations[-1]["txid"] == "ff" * 32


def test_the_payload_decodes_like_the_page_does():
    snap = T.Poller(T.Config()).snapshot()
    jar = T.build_jar(snap["donations"], 256)
    cols = T.decode_start(jar.payload(), 256)
    assert np.allclose(cols["rs"][jar.ids], jar.radii, atol=2e-6) and np.all(cols["ir"][jar.ids] <= 1.0)


def test_the_game_state_carries_the_demo_panel_and_no_address():
    g = T.TipJarGameClass()(aspect=1.0, poller=T.Poller(T.Config()))
    st = g.state()
    t = st["tipjar"]
    assert t["demo"] and t["address"] is None and t["uri"] is None and "DEMO" in st["status"]
    assert st["compute"]["start"]["ids"] and st["compute"]["named"] == ["ir", "ig", "ib"] and st["compute"]["uniforms"]["use_image"] == 1.0
    assert all("txid" not in grain for grain in t["grains"])                      # the txid is only shared when the config enables it
    epoch = g.epoch
    g.state()
    assert g.epoch == epoch                                                       # an unchanged tx list does not restart the jar


# ------------------------------------------------------------------ the page's QR (Node)
@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_qr_code_helpers_in_node():
    done = subprocess.run(["node", str(ROOT / "tests" / "meowdoku_tipjar_check.mjs")], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_qr_wall_layout_in_node():
    """[QRWALL] the module -> cube layout, material assignment, cube geometry and the head-on camera (web/qr_wall.js)."""
    done = subprocess.run(["node", str(ROOT / "tests" / "meowdoku_qrwall_check.mjs")], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr


def test_qr_wall_is_exported_with_the_page():
    """[QRWALL] web_export copies qr_wall.js beside the page (app.js imports it)."""
    assert '"qr_wall.js"' in (ROOT / "examples" / "meowdoku" / "web_export.py").read_text(encoding="utf-8")

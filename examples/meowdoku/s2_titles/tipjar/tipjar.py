"""TIP JAR: a read-only Bitcoin piggybank on the 3-D fountain.  Every incoming transaction is one grain; the jar is the wallet.

WATCH-ONLY.  ``build/meowdoku/tipjar.json`` (gitignored) holds ``{"address": "...", "xpub": "...", "show_txid": false}``; only an address and/or
an account-level public key (xpub / zpub) are ever accepted.  Anything that looks like a private key or a seed (WIF, xprv-family, 12/15/18/21/24
words, 64 hex digits) is refused with a log line and the jar stays unconfigured.  Unconfigured, the game runs a clearly labelled DEMO from a committed
fixture of sample transactions (``tipjar_fixture.json``, no real address).

POLLER.  One refresh of the mempool.space REST API per ``POLL_SECONDS`` (cached; the viewer count changes nothing), fail soft (the last good list
is kept).  Received amount of a transaction = the sum of its outputs that pay the jar; a transaction that spends from the jar is the wallet's
own movement and is not a donation.  With an xpub the BIP84 receiving addresses are derived (gap limit 20) and summed, and a fresh unused address
is offered to donors.

THE ANTIBLENDER.  The jar is the 3-D fountain law, unedited.  Each donation is a grain whose attributes ride channels that are truly per grain:
  * SIZE   -> the per-id state column ``rs`` (the grain's current radius; the species table is untouched, so a grain can only be as large as the
              species radius of the id it is given: ids are assigned by size, see ``assign_ids``).
  * HUE    -> the per-id image colours ``ir ig ib`` (``use_image`` = 1): the sender's input script type (p2pkh / p2sh / p2wpkh / p2wsh / p2tr).
  * LIGHT  -> the same colour: lightness is the age, newer brighter; an unconfirmed transaction is near white ("glowing").
Not per grain in the law or the page: finish / material (the render look is global) and a true emissive flag (the glow look is global too), so
"unconfirmed" is carried by the very light colour and a tier by nothing.  Position is not encoded: the pile mixes, and a tap decodes the grain it
hits through its id.

SIZE MAPPING.  Volume is proportional to sats (radius ~ cbrt(sats)), clamped to the legal range ``[R_MIN, R_MAX]`` (R_MAX is the largest species
radius in the id table; R_MIN the param minimum 0.005).  The law's range is narrow (ratio ~2.8, volume ratio ~22), so when the list's own cbrt
range does not fit, the radius follows the logarithm of the amount across the whole range instead (still monotone and clamped).

HAND-OFF.  The grains start loose in the closed jar, oldest lowest and newest on top, at rest (the page's stop-at-rest run settles them; a
new donation changes the tx list and starts the jar again, the newest grain dropping last).  Spare ids stay parked (their emission time never
comes).  The payload is the glyph start's (``glyph_start.GlyphStart``) plus ``ids`` (which ids), ``rgb`` and ``age``.
"""
from __future__ import annotations

import base64
import colorsys
import hashlib
import hmac
import json
import math
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CONFIG_PATH = ROOT / "build" / "meowdoku" / "tipjar.json"
FIXTURE_PATH = HERE / "tipjar_fixture.json"
API = "https://mempool.space/api"
POLL_SECONDS = 60.0
GAP_LIMIT = 20
MAX_PAGES = 8                      # 25 confirmed txs per page: 200 per address is more than the 4096 ids ever show for a tip jar
SATS = 100_000_000


def log(message: str) -> None:
    print(f"[meowdoku] tipjar: {message}", flush=True)


# ---------------------------------------------------------------- hashing: ripemd160 (OpenSSL 3 may not offer it), base58, bech32
def _rol(x: int, n: int) -> int:
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


_RL = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8, 3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
       1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2, 4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13)
_RR = (5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12, 6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2, 15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
       8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14, 12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11)
_SL = (11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8, 7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12, 11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
       11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12, 9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6)
_SR = (8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6, 9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11, 9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
       15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8, 8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11)
_KL = (0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E)
_KR = (0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000)


def _f(j: int, x: int, y: int, z: int) -> int:
    if j < 16:
        return x ^ y ^ z
    if j < 32:
        return (x & y) | (~x & 0xFFFFFFFF & z)
    if j < 48:
        return (x | (~y & 0xFFFFFFFF)) ^ z
    if j < 64:
        return (x & z) | (y & (~z & 0xFFFFFFFF))
    return x ^ (y | (~z & 0xFFFFFFFF))


def ripemd160(data: bytes) -> bytes:
    """RIPEMD-160 (hashlib's may be missing under OpenSSL 3): used only for the public-key hash of a derived address."""
    h = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0]
    msg = data + b"\x80" + b"\x00" * ((55 - len(data)) % 64) + (8 * len(data)).to_bytes(8, "little")
    for off in range(0, len(msg), 64):
        x = [int.from_bytes(msg[off + 4 * i: off + 4 * i + 4], "little") for i in range(16)]
        al, bl, cl, dl, el = h
        ar, br, cr, dr, er = h
        for j in range(80):
            t = (_rol((al + _f(j, bl, cl, dl) + x[_RL[j]] + _KL[j // 16]) & 0xFFFFFFFF, _SL[j]) + el) & 0xFFFFFFFF
            al, el, dl, cl, bl = el, dl, _rol(cl, 10), bl, t
            t = (_rol((ar + _f(79 - j, br, cr, dr) + x[_RR[j]] + _KR[j // 16]) & 0xFFFFFFFF, _SR[j]) + er) & 0xFFFFFFFF
            ar, er, dr, cr, br = er, dr, _rol(cr, 10), br, t
        t = (h[1] + cl + dr) & 0xFFFFFFFF
        h = [t, (h[2] + dl + er) & 0xFFFFFFFF, (h[3] + el + ar) & 0xFFFFFFFF, (h[4] + al + br) & 0xFFFFFFFF, (h[0] + bl + cr) & 0xFFFFFFFF]
    return b"".join(v.to_bytes(4, "little") for v in h)


def hash160(data: bytes) -> bytes:
    return ripemd160(hashlib.sha256(data).digest())


B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58decode_check(text: str) -> bytes:
    n = 0
    for ch in text:
        n = n * 58 + B58.index(ch)                      # ValueError on a character outside the alphabet
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    raw = b"\x00" * (len(text) - len(text.lstrip("1"))) + raw
    body, check = raw[:-4], raw[-4:]
    if hashlib.sha256(hashlib.sha256(body).digest()).digest()[:4] != check:
        raise ValueError("bad base58 checksum")
    return body


def b58encode_check(body: bytes) -> str:
    raw = body + hashlib.sha256(hashlib.sha256(body).digest()).digest()[:4]
    n = int.from_bytes(raw, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = B58[r] + out
    return "1" * (len(raw) - len(raw.lstrip(b"\x00"))) + out


_BECH = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _polymod(values: list[int]) -> int:
    gen = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)
    chk = 1
    for v in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ v
        for i in range(5):
            chk ^= gen[i] if (top >> i) & 1 else 0
    return chk


def bech32_encode(hrp: str, version: int, program: bytes) -> str:
    """A segwit address (BIP173 bech32 for version 0, BIP350 bech32m for the later ones)."""
    acc, bits, data = 0, 0, [version]
    for byte in program:
        acc, bits = (acc << 8) | byte, bits + 8
        while bits >= 5:
            bits -= 5
            data.append((acc >> bits) & 31)
    if bits:
        data.append((acc << (5 - bits)) & 31)
    const = 1 if version == 0 else 0x2BC830A3
    values = [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]
    mod = _polymod(values + data + [0] * 6) ^ const
    checksum = [(mod >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp + "1" + "".join(_BECH[d] for d in data + checksum)


# ---------------------------------------------------------------- BIP32 public derivation + BIP84 addresses
_P = 2 ** 256 - 2 ** 32 - 977
_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798, 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)


def _add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a[0] == b[0] and (a[1] + b[1]) % _P == 0:
        return None
    if a == b:
        m = 3 * a[0] * a[0] * pow(2 * a[1], -1, _P) % _P
    else:
        m = (b[1] - a[1]) * pow(b[0] - a[0], -1, _P) % _P
    x = (m * m - a[0] - b[0]) % _P
    return x, (m * (a[0] - x) - a[1]) % _P


def _mul(k: int, point=_G):
    out = None
    while k:
        if k & 1:
            out = _add(out, point)
        point = _add(point, point)
        k >>= 1
    return out


def _compress(point) -> bytes:
    return bytes([2 + (point[1] & 1)]) + point[0].to_bytes(32, "big")


def _decompress(raw: bytes):
    x = int.from_bytes(raw[1:], "big")
    y = pow((x ** 3 + 7) % _P, (_P + 1) // 4, _P)
    if (y & 1) != (raw[0] & 1):
        y = _P - y
    if y * y % _P != (x ** 3 + 7) % _P:
        raise ValueError("not a point on the curve")
    return x, y


XPUB_VERSIONS = {bytes.fromhex("0488b21e"): "xpub", bytes.fromhex("04b24746"): "zpub"}


def parse_xpub(text: str) -> tuple[bytes, bytes]:
    """(compressed public key, chain code) of an account-level extended PUBLIC key (xpub or zpub)."""
    body = b58decode_check(text.strip())
    if len(body) != 78 or body[:4] not in XPUB_VERSIONS:
        raise ValueError("not an extended public key (xpub / zpub)")
    key, chain = body[45:78], body[13:45]
    _decompress(key)
    return key, chain


def ckd_pub(key: bytes, chain: bytes, index: int) -> tuple[bytes, bytes]:
    """BIP32 non-hardened public child derivation."""
    if index >= 0x80000000:
        raise ValueError("a public key derives only non-hardened children")
    digest = hmac.new(chain, key + index.to_bytes(4, "big"), hashlib.sha512).digest()
    il = int.from_bytes(digest[:32], "big")
    if il >= _N:
        raise ValueError("invalid child")
    child = _add(_mul(il), _decompress(key))
    if child is None:
        raise ValueError("invalid child")
    return _compress(child), digest[32:]


def bip84_address(key: bytes, hrp: str = "bc") -> str:
    """Native segwit (p2wpkh) address of a compressed public key."""
    return bech32_encode(hrp, 0, hash160(key))


def derive_receive(xpub: str, index: int, chain_index: int = 0) -> str:
    """Address ``index`` of the receiving chain (m/.../0/index) of an account-level key."""
    key, chain = parse_xpub(xpub)
    key, chain = ckd_pub(key, chain, chain_index)
    key, _ = ckd_pub(key, chain, index)
    return bip84_address(key)


# ---------------------------------------------------------------- the config: watch-only, refuse anything that looks secret
_WORD = re.compile(r"^[a-z]{3,8}$")


def looks_secret(text: str) -> str | None:
    """The reason a string looks like a private key or a seed, else None."""
    s = str(text or "").strip()
    if not s:
        return None
    words = s.split()
    if len(words) in (12, 15, 18, 21, 24) and all(_WORD.match(w) for w in words):
        return "a recovery phrase (12/24 words)"
    if re.match(r"^[xyztuv]prv[1-9A-HJ-NP-Za-km-z]{100,}$", s):
        return "an extended PRIVATE key"
    if re.fullmatch(r"[0-9a-fA-F]{64}", s):
        return "a raw 256-bit hex key"
    if re.fullmatch(r"[5KLc9][1-9A-HJ-NP-Za-km-z]{50,51}", s):
        try:
            body = b58decode_check(s)
            if body[0] in (0x80, 0xEF) and len(body) in (33, 34):
                return "a WIF private key"
        except ValueError:
            pass
    return None


@dataclass
class Config:
    address: str | None = None
    xpub: str | None = None
    show_txid: bool = False
    refused: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.address or self.xpub)


_ADDRESS = re.compile(r"^(bc1[a-z0-9]{11,87}|[13][1-9A-HJ-NP-Za-km-z]{25,34})$")


def load_config(path: Path | None = None) -> Config:
    """The watch-only config.  A missing file, bad JSON or any secret-looking value gives an unconfigured jar (the reason is logged)."""
    path = Path(path) if path else CONFIG_PATH
    if not path.is_file():
        return Config()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("the config must be a JSON object")
    except Exception as error:          # noqa: BLE001
        log(f"config {path.name} unreadable ({type(error).__name__}); showing the DEMO")
        return Config(refused="unreadable config")
    for key, value in raw.items():
        why = looks_secret(value if isinstance(value, str) else json.dumps(value))
        if why:
            log(f"REFUSED: field {key!r} looks like {why}.  This jar is watch-only: put only a public address or an xpub/zpub in the config.")
            return Config(refused=f"{key} looks like {why}")
    cfg = Config(show_txid=bool(raw.get("show_txid", False)))
    address, xpub = raw.get("address"), raw.get("xpub")
    if address:
        if not isinstance(address, str) or not _ADDRESS.match(address.strip()):
            log("the configured address is not a bitcoin address; showing the DEMO")
            return Config(refused="bad address")
        cfg.address = address.strip()
    if xpub:
        try:
            parse_xpub(str(xpub))
        except Exception as error:      # noqa: BLE001
            log(f"the configured xpub is not an extended public key ({error}); showing the DEMO")
            return Config(refused="bad xpub")
        cfg.xpub = str(xpub).strip()
    return cfg


# ---------------------------------------------------------------- transactions -> donations
def donation_of(tx: dict, mine: set[str], slot_of: Callable[[str], int] | None = None) -> dict | None:
    """The donation a transaction makes to the jar, or None.  ``sats`` = the sum of the outputs paying one of ``mine`` (change and other
    outputs of the sender are not ours).  A transaction that SPENDS from the jar (an input of ours) is the wallet moving its own coins."""
    for vin in tx.get("vin", []):
        if ((vin.get("prevout") or {}).get("scriptpubkey_address")) in mine:
            return None
    sats = sum(int(v.get("value", 0)) for v in tx.get("vout", []) if v.get("scriptpubkey_address") in mine)
    if sats <= 0:
        return None
    status = tx.get("status") or {}
    first = (tx.get("vin") or [{}])[0].get("prevout") or {}
    paid = [v.get("scriptpubkey_address") for v in tx.get("vout", []) if v.get("scriptpubkey_address") in mine]
    return {"txid": str(tx["txid"]), "sats": sats, "confirmed": bool(status.get("confirmed")),
            "time": int(status["block_time"]) if status.get("block_time") else None,
            "kind": str(first.get("scriptpubkey_type") or "unknown"),
            "slot": slot_of(paid[0]) if slot_of else 0}


def tx_hash(donations: list[dict]) -> str:
    """The identity of a jar: its donations in order (the jar is a pure function of it)."""
    h = hashlib.sha256()
    for d in donations:
        h.update(f"{d['txid']}:{d['sats']}:{int(d['confirmed'])}:{d.get('time')}:{d['kind']}:{d['slot']};".encode())
    return h.hexdigest()[:16]


def order_donations(donations: list[dict], first_seen: dict[str, float] | None = None) -> list[dict]:
    """Chronological: confirmed by block time, then the unconfirmed by when they were first seen; the txid breaks ties."""
    seen = first_seen or {}
    key = lambda d: (d["time"] if d["time"] is not None else seen.get(d["txid"], 4.0e9), d["txid"])  # noqa: E731
    return sorted(donations, key=key)


# ---------------------------------------------------------------- the poller
def http_get_json(path: str) -> Any:
    request = urllib.request.Request(API + path, headers={"User-Agent": "meowdoku-tipjar/1 (watch-only)"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


class Poller:
    """The jar's source of truth: one cached snapshot, refreshed at most once per ``interval`` however many viewers ask.

    ``fetch(path)`` returns parsed JSON for a mempool.space ``/api`` path (default: urllib).  Every failure keeps the last good snapshot
    (``error`` says why).  The DEMO (no config) never touches the network."""

    def __init__(self, config: Config, *, fetch: Callable[[str], Any] | None = None, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time, interval: float = POLL_SECONDS, fixture: Path | None = None):
        self.config, self.fetch, self.clock, self.wall, self.interval = config, fetch or http_get_json, clock, wall, interval
        self.fixture = fixture or FIXTURE_PATH
        self.lock = threading.Lock()
        self.due = None                         # clock() after which the next refresh is allowed
        self.refreshes = 0
        self.first_seen: dict[str, float] = {}
        self.snap: dict[str, Any] = {"demo": not config.configured, "donations": [], "balance": 0, "address": None, "price": None,
                                     "error": None, "updated": None}
        self.used: dict[str, int] = {}          # xpub: address -> index, for the addresses with history
        self.next_index = 0

    # -- the public read
    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            now = self.clock()
            if self.due is None or now >= self.due:
                self.due = now + self.interval
                self._refresh()
            return self.snap

    # -- one refresh
    def _refresh(self) -> None:
        self.refreshes += 1
        if not self.config.configured:
            self.snap = self._demo()
            return
        try:
            snap = self._live()
        except Exception as error:      # noqa: BLE001  (fail soft: the last good list stays)
            self.snap = {**self.snap, "error": f"{type(error).__name__}: {error}"}
            log(f"poll failed ({self.snap['error']}); keeping the last list")
            return
        self.snap = snap

    def _demo(self) -> dict[str, Any]:
        data = json.loads(self.fixture.read_text(encoding="utf-8"))
        mine = {data["address"]}
        donations = [d for d in (donation_of(tx, mine) for tx in data["txs"]) if d]
        balance = sum(d["sats"] for d in donations)
        return {"demo": True, "donations": order_donations(donations), "balance": balance, "address": None,
                "price": data.get("price"), "error": None, "updated": self.wall()}

    def _addresses(self) -> tuple[list[str], str]:
        """The addresses to watch and the one to show donors."""
        if self.config.address and not self.config.xpub:
            return [self.config.address], self.config.address
        key, chain = parse_xpub(self.config.xpub)
        rkey, rchain = ckd_pub(key, chain, 0)
        watch, index, idle = [], 0, 0
        shown = None
        while idle < GAP_LIMIT:
            ckey, _ = ckd_pub(rkey, rchain, index)
            address = bip84_address(ckey)
            info = self.fetch(f"/address/{address}")
            count = int(info["chain_stats"]["tx_count"]) + int(info["mempool_stats"]["tx_count"])
            watch.append(address)
            if count:
                idle = 0
                self.used[address] = index
            else:
                idle += 1
                shown = shown or address
            index += 1
        self.next_index = index
        extra = [self.config.address] if self.config.address else []
        return extra + [a for a in watch if a in self.used], shown or watch[-1]

    def _history(self, address: str) -> list[dict]:
        txs = list(self.fetch(f"/address/{address}/txs"))
        confirmed = [t for t in txs if (t.get("status") or {}).get("confirmed")]
        pages = 1
        while len(confirmed) >= 25 * pages and pages < MAX_PAGES:
            more = self.fetch(f"/address/{address}/txs/chain/{confirmed[-1]['txid']}")
            if not more:
                break
            txs += more
            confirmed += more
            pages += 1
        return txs

    def _live(self) -> dict[str, Any]:
        watch, shown = self._addresses()
        mine = set(watch) | ({self.config.address} if self.config.address else set())
        slots = {a: self.used.get(a, 0) for a in mine}
        by_id: dict[str, dict] = {}
        balance = 0
        for address in watch:
            info = self.fetch(f"/address/{address}")
            c, m = info["chain_stats"], info["mempool_stats"]
            balance += int(c["funded_txo_sum"]) - int(c["spent_txo_sum"]) + int(m["funded_txo_sum"]) - int(m["spent_txo_sum"])
            for tx in self._history(address):
                by_id[tx["txid"]] = tx
        now = self.wall()
        donations = []
        for tx in by_id.values():
            d = donation_of(tx, mine, lambda a: slots.get(a, 0))
            if d:
                if not d["confirmed"]:
                    self.first_seen.setdefault(d["txid"], now)
                donations.append(d)
        price = None
        try:
            price = self.fetch("/v1/prices")
        except Exception:               # noqa: BLE001  (the fiat estimate is optional)
            price = (self.snap or {}).get("price")
        return {"demo": False, "donations": order_donations(donations, self.first_seen), "balance": balance, "address": shown, "price": price,
                "error": None, "updated": now}


# ---------------------------------------------------------------- sizes: sats -> radius
R_MIN = 0.005                      # the law's smallest legal species radius (fountain3d.PARAMS r<s>)


def radius_map(sats: list[int], r_lo: float, r_hi: float) -> tuple[list[float], str]:
    """Radius per donation: volume proportional to sats (radius ~ cbrt) when the list's own range fits ``[r_lo, r_hi]``, else the logarithm of
    the amount across the whole range.  Monotone in sats and clamped.  Returns (radii, 'cbrt' | 'log' | 'single')."""
    if not sats:
        return [], "single"
    lo, hi = min(sats), max(sats)
    if lo == hi:
        return [math.sqrt(r_lo * r_hi)] * len(sats), "single"
    if (hi / lo) ** (1.0 / 3.0) * r_lo <= r_hi:
        return [min(r_hi, max(r_lo, r_lo * (s / lo) ** (1.0 / 3.0))) for s in sats], "cbrt"
    span = math.log(hi / lo)
    return [min(r_hi, max(r_lo, r_lo * (r_hi / r_lo) ** (math.log(s / lo) / span))) for s in sats], "log"


def assign_ids(radii: list[float], id_radius: np.ndarray) -> np.ndarray:
    """An id for every donation: the smallest unused id whose species radius covers the wanted radius (the grain's ``rs`` can only be smaller than
    its id's species radius), the largest left over when none does.  Largest donations choose first; deterministic."""
    order = np.argsort(-np.asarray(radii), kind="stable")
    free = sorted(range(len(id_radius)), key=lambda i: (float(id_radius[i]), i))          # ascending radius
    out = np.zeros(len(radii), dtype=np.int64)
    for k in order:
        pick = next((j for j, i in enumerate(free) if id_radius[i] >= radii[k] - 1.0e-12), len(free) - 1)
        out[k] = free.pop(pick)
    return out


# ---------------------------------------------------------------- colours: the source and the age
KIND_HUE = {"p2pkh": 35.0, "p2sh": 270.0, "v0_p2wpkh": 150.0, "v0_p2wsh": 195.0, "v1_p2tr": 320.0, "unknown": 0.0}
KIND_LABEL = {"p2pkh": "legacy (P2PKH)", "p2sh": "wrapped / multisig (P2SH)", "v0_p2wpkh": "native segwit (P2WPKH)", "v0_p2wsh": "segwit script (P2WSH)",
              "v1_p2tr": "taproot (P2TR)", "unknown": "unknown or coinbase"}
SLOT_SHIFT = 10.0                  # degrees of hue per receiving-address slot (mod 3: -1, 0, +1 steps), the xpub branch
L_OLD, L_NEW, L_UNCONFIRMED = 0.30, 0.72, 0.93
SAT = 0.85


def grain_rgb(kind: str, slot: int, age01: float, confirmed: bool) -> tuple[int, int, int]:
    """The grain's colour bytes: hue = source (input script type, receiving slot), lightness = age (1 = newest), near white while unconfirmed."""
    hue = (KIND_HUE.get(kind, 0.0) + SLOT_SHIFT * ((int(slot) % 3) - 1)) % 360.0
    light = L_UNCONFIRMED if not confirmed else L_OLD + (L_NEW - L_OLD) * min(1.0, max(0.0, age01))
    sat = 0.0 if kind == "unknown" else SAT
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, light, sat)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def decode_rgb(rgb: tuple[int, int, int]) -> dict[str, Any]:
    """The inverse the page performs on a tapped grain: (kind, slot offset, confirmed, age01) from its colour."""
    h, l, s = colorsys.rgb_to_hls(*(c / 255.0 for c in rgb))
    hue = h * 360.0
    if s < 0.15:
        kind, base = "unknown", 0.0
    else:
        kind = min(KIND_HUE, key=lambda k: min(abs(hue - KIND_HUE[k]) % 360.0, 360.0 - abs(hue - KIND_HUE[k]) % 360.0) if k != "unknown" else 1e9)
        base = KIND_HUE[kind]
    diff = (hue - base + 180.0) % 360.0 - 180.0
    confirmed = l < 0.5 * (L_NEW + L_UNCONFIRMED)
    return {"kind": kind, "slot_step": int(round(diff / SLOT_SHIFT)), "confirmed": confirmed,
            "age01": (l - L_OLD) / (L_NEW - L_OLD) if confirmed else 1.0}


# ---------------------------------------------------------------- the jar as a glyph-start state
def build_jar(donations: list[dict], n_ids: int, *, mix: float = 1.0, params: dict | None = None):
    """The ``JarStart`` of a list of donations (oldest first) for ``n_ids`` ids.  Pure function of (donations, n_ids, mix, params)."""
    from meowdoku.s2_titles.fountain import fountain3d as F
    from meowdoku.s2_titles.fountain import glyph_start as GS
    donations = donations[-n_ids:]                                      # more donations than ids: the newest fill the jar
    count = len(donations)
    p = F.resolve_params(mix, params)
    u = {k: float(v) for k, v in F.uniforms(n=n_ids, mix=mix, params=params, shape="box", inlet="pour").items()}
    _species, id_radius = GS.grain_radii(n_ids, u)
    r_hi = float(id_radius.max())
    radii, method = radius_map([d["sats"] for d in donations], R_MIN, r_hi)
    ids = assign_ids(radii, id_radius) if count else np.zeros(0, dtype=np.int64)
    if count and 0 not in ids:               # id 0 has emission time 0 whatever the rate: it must be a placed grain or it would pour from the nozzle
        small = int(np.argmin(radii))
        ids[small] = 0
    rs = np.minimum(np.array(radii), id_radius[ids]) if count else np.zeros(0)
    # colours
    times = [d["time"] for d in donations if d["time"] is not None]
    if times and max(times) > min(times):
        age = [((d["time"] if d["time"] is not None else max(times)) - min(times)) / (max(times) - min(times)) for d in donations]
    else:
        age = [1.0] * count
    rgb = np.array([grain_rgb(d["kind"], d["slot"], a, d["confirmed"]) for d, a in zip(donations, age)], dtype=np.uint8).reshape(count, 3)
    # the jar: nx x nz grains across, a loose lattice oldest lowest, newest on top; a seeded sway keeps the columns from stacking dead straight
    cell = 2.0 * F.rmax_of(p, True)
    s = 1.12 * cell
    nx = nz = 4
    layers = max(1, math.ceil(count / (nx * nz)))
    margin = F.MARGIN_CELLS * cell if hasattr(F, "MARGIN_CELLS") else 2.0 * cell
    hx, hz = 0.5 * nx * s + margin, 0.5 * nz * s + margin
    hy = 0.5 * (1.0 + layers) * s * 1.5 + margin
    seed = int.from_bytes(hashlib.sha256(tx_hash(donations).encode()).digest()[:8], "little")
    sway = np.random.default_rng(seed).uniform(-0.12, 0.12, (max(count, 1), 2)) * s
    pos = np.zeros((count, 3))
    for k in range(count):
        layer, rest = divmod(k, nx * nz)
        i, j = rest % nx, rest // nx
        pos[k] = ((i - 0.5 * (nx - 1)) * s + sway[k, 0], margin + (layer + 1.0) * s * 0.9 + 2.0 * s, (j - 0.5 * (nz - 1)) * s + sway[k, 1])
    return JarStartClass().make(donations, n_ids, ids, rs, rgb, pos, hx, hy, hz, cell, p, method, r_hi, mix)


def _jar_class():
    from meowdoku.s2_titles.fountain import fountain3d as F
    from meowdoku.s2_titles.fountain import glyph_start as GS

    @dataclass
    class JarStart(GS.GlyphStart):
        ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
        rgb: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.uint8))
        count: int = 0
        method: str = ""
        donations: list = field(default_factory=list)
        digest: str = ""

        AGE_LIVE = 1.0e9                       # the placed grains' age: past every emission time, so they are live from round zero
        EMIT_NEVER = 1.0e4                     # seconds per id: a spare id's emission time (id * this) never comes

        @classmethod
        def make(cls, donations, n_ids, ids, rs, rgb, pos, hx, hy, hz, cell, p, method, r_hi, mix):
            count = len(donations)
            rc = 0.5 * cell
            cols, rows = int(math.ceil(2.0 * hx / cell)), int(math.ceil(2.0 * hy / cell)) + 1
            dh = hz + F.Z_MARGIN_CELLS * cell
            over = {"half": hx, "cols": cols, "rows": rows, "lays": max(1, int(math.ceil(2.0 * dh / cell - 1.0e-9))), "cell": cell, "dh": dh,
                    "emit_rate": cls.EMIT_NEVER, "nozzle_x": 0.0, "nozzle_y": 2.0 * hy - 2.0 * cell, "nozzle_z": 0.0, "bar_half": 0.0, "bar_z": 0.0,
                    "spread": 0.0}
            over.update(F.slot_uniforms([F._slot("box", "fluid", 0.0, hy, 0.0, 0.0, hx, hy, hz, rc)]))
            over.update({name: 0.0 for name in F.POLYGON_UNIFORMS})
            bounds = (-hx, hx, 0.0, 2.0 * hy, -hz, hz)
            lo, hi = np.array(bounds[0::2]), np.array(bounds[1::2])
            q = lo + np.rint((pos - lo) / (hi - lo) * GS.QUANT) / GS.QUANT * (hi - lo) if count else pos
            vol = 8.0 * hx * hy * hz
            return cls(text="tipjar", n=n_ids, crystal=False, positions=q.astype(np.float32).astype(np.float64), radii=F._f32(rs), species=np.zeros(count, dtype=int),
                       scale=2.0 * hy, spacing=cell, k=1.0, phi=0.0, phi_target=0.0, volume=vol, jitter=0.0, sites=count,
                       box={"hx": hx, "hy": hy, "hz": hz, "margin": 2.0 * cell, "cell": cell, "rc": rc}, overrides=over, bounds=bounds, swell=True,
                       ids=np.asarray(ids, dtype=np.int64), rgb=rgb, count=count, method=method, donations=list(donations), digest=tx_hash(donations))

        def stats(self):
            return {"n": self.count, "height": self.scale, "spacing": self.spacing, "k": 1.0, "phi": 0.0, "phi_target": 0.0, "sites": self.count,
                    "vacancies": 0, "jitter": 0.0, "volume": self.volume}

        def columns(self, pad, **kwargs):
            cols = F.initial_columns(pad, mix=kwargs.get("mix", 1.0), mode=kwargs.get("mode", 0), use_image=1.0, active=self.n,
                                     strata=kwargs.get("strata", F.STRATA), params=kwargs.get("params"), shape="box", inlet="pour")
            cols.update(self.uniforms(**{**kwargs, "use_image": 1.0}))
            d = decode_start(self.payload(), pad)
            for name, col in d.items():
                cols[name][:] = col
            return cols

        def _encode(self):
            lo, hi = np.array(self.bounds[0::2]), np.array(self.bounds[1::2])
            q = np.rint((self.positions - lo) / (hi - lo) * GS.QUANT).astype("<u2")
            rmax = float(1.001 * max(float(self.radii.max()) if self.count else 0.0, 1.0e-6))
            rq = np.rint(self.radii / rmax * GS.QUANT).astype("<u2")
            b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")  # noqa: E731
            return {"n": self.count, "bounds": [float(b) for b in self.bounds], "xyz": b64(q), "rmax": rmax, "rs": b64(rq),
                    "ids": b64(self.ids.astype("<u2")), "rgb": b64(self.rgb), "age": float(self.AGE_LIVE)}

        def inside_glyphs(self, points):
            return np.ones(len(points), dtype=bool)

    return JarStart


_JAR = None


def JarStartClass():
    """The ``JarStart`` class (built on first use: it subclasses ``glyph_start.GlyphStart``, which pulls in the law module)."""
    global _JAR
    _JAR = _JAR or _jar_class()
    return _JAR


def decode_start(payload: dict, pad: int) -> dict[str, np.ndarray]:
    """The page's decode of the jar payload (compute.js ``decodeStart``): columns x y z rs age ir ig ib of length ``pad``, the placed rows scattered to
    their ids; every other id is the exact zero state (parked)."""
    n = int(payload["n"])
    out = {name: np.zeros(pad, dtype=np.float32) for name in ("x", "y", "z", "rs", "age", "ir", "ig", "ib")}
    if not n:
        return out
    ids = np.frombuffer(base64.b64decode(payload["ids"]), dtype="<u2").astype(np.int64)
    q = np.frombuffer(base64.b64decode(payload["xyz"]), dtype="<u2").reshape(n, 3).astype(np.float64)
    b = np.array(payload["bounds"])
    for c, name in enumerate(("x", "y", "z")):
        out[name][ids] = (b[2 * c] + q[:, c] / 65535.0 * (b[2 * c + 1] - b[2 * c])).astype(np.float32)
    out["rs"][ids] = (np.frombuffer(base64.b64decode(payload["rs"]), dtype="<u2").astype(np.float64) / 65535.0 * float(payload["rmax"])).astype(np.float32)
    out["age"][ids] = np.float32(payload["age"])
    rgb = np.frombuffer(base64.b64decode(payload["rgb"]), dtype=np.uint8).reshape(n, 3)
    for c, name in enumerate(("ir", "ig", "ib")):
        out[name][ids] = (rgb[:, c] / 255.0).astype(np.float32)
    return out


# ---------------------------------------------------------------- the page's numbers
def page_summary(snap: dict, config: Config, grains: list[dict]) -> dict[str, Any]:
    """What the page's panel shows (no donor data beyond amount, date, status, and the coarse source type)."""
    bal = int(snap["balance"])
    price = snap.get("price") or {}
    usd = price.get("USD")
    address = None if snap["demo"] else snap.get("address")
    return {
        "demo": bool(snap["demo"]), "configured": config.configured, "refused": config.refused,
        "address": address, "uri": f"bitcoin:{address}" if address else None,
        "balance_sats": bal, "balance_btc": f"{bal / SATS:.8f}", "count": len(snap["donations"]),
        "fiat": {"USD": round(bal / SATS * float(usd), 2)} if usd else None,
        "updated": snap.get("updated"), "error": snap.get("error"), "show_txid": bool(config.show_txid),
        "grains": grains, "kinds": [[k, KIND_LABEL[k], KIND_HUE[k]] for k in KIND_HUE],
        "legend": {"size": "volume proportional to the amount (log-scaled when the amounts span more than the grains can)",
                   "hue": "the sender's input script type (and the receiving address slot, a small shift)",
                   "lightness": "age: newer is lighter", "unconfirmed": "nearly white until confirmed",
                   "light": [L_OLD, L_NEW, L_UNCONFIRMED], "slot_shift": SLOT_SHIFT},
    }


# ---------------------------------------------------------------- the game
_POLLER: Poller | None = None
_POLLER_LOCK = threading.Lock()


def shared_poller() -> Poller:
    """The process-wide poller (one upstream refresh per interval whatever the number of games or viewers)."""
    global _POLLER
    with _POLLER_LOCK:
        if _POLLER is None:
            config = load_config()
            if not config.configured:
                log("tip jar not configured: showing the DEMO" + (f" ({config.refused})" if config.refused else ""))
            _POLLER = Poller(config)
        return _POLLER


def reset_shared_poller(poller: Poller | None = None) -> None:
    global _POLLER
    with _POLLER_LOCK:
        _POLLER = poller


HIDDEN_CONTROLS = {"image", "camera", "replay", "verify", "mix", "shape", "inlet", "glyph_text", "glyph_crystal", "depth", "spread"}


def _game_class():
    from meowdoku.s2_titles.fountain.fountain_game import GLYPH_SHAPE, FountainGame

    class TipJarGame(FountainGame):
        """The tip jar: the 3-D fountain with the jar's donations as its starting grains (see the module docstring)."""

        title = "Tip jar"

        def __init__(self, board_id: str = "", *, aspect: float | None = None, poller: Poller | None = None):
            super().__init__(board_id, size="small", aspect=aspect)
            self.poller = poller or shared_poller()
            self.mix, self.shape, self.run, self.phase = 1.0, GLYPH_SHAPE, "continuous", "replay"
            self.size_locked = False
            self.jar = None
            self._digest = None
            self.snap: dict[str, Any] = {}
            self._sync()

        def reset(self) -> None:
            self.verified = None
            self.epoch += 1

        def do(self, payload: dict) -> None:
            kind = str(payload.get("type"))
            if kind in ("fountain_replay", "fountain_image", "fountain_glyph", "fountain_verify"):
                raise ValueError(f"the tip jar takes no {kind}")
            if kind == "fountain_choice" and str(payload.get("name")) in ("shape", "inlet"):
                raise ValueError("the tip jar is a closed jar")
            if kind == "fountain_size":
                self.size_locked = True
            super().do(payload)
            self.shape, self.phase = GLYPH_SHAPE, "replay"
            self._sync(force=True)

        def _sync(self, force: bool = False) -> None:
            snap = self.poller.snapshot()
            donations = snap["donations"]
            if not self.size_locked:
                want = "small" if len(donations) <= 256 else "4k"
                if want != self.size:
                    self.size, force = want, True
            digest = tx_hash(donations[-self.n:]) + f"|{self.size}|{self.mix}|{sorted(self.params.items())}"
            self.snap = snap
            if self.jar is None or force or digest != self._digest:
                if self.jar is not None and (force or digest != self._digest):
                    self.epoch += 1                       # a new donation (or a new setting): the jar starts again, the newest grain dropping last
                self._digest = digest
                self.jar = build_jar(list(donations), self.n, mix=self.mix, params=self.params)

        def glyph(self):
            self._sync()
            return self.jar

        def controls_ui(self) -> list[dict[str, Any]]:
            return [c for c in super().controls_ui() if c["id"] not in HIDDEN_CONTROLS]

        def state(self) -> dict[str, Any]:
            self._sync()
            st = super().state()
            jar, snap = self.jar, self.snap
            grains = []
            for d, i in zip(jar.donations, jar.ids):
                g = {"id": int(i), "sats": d["sats"], "time": d["time"], "confirmed": d["confirmed"], "kind": d["kind"], "slot": d["slot"]}
                if self.poller.config.show_txid:
                    g["txid"] = d["txid"]
                grains.append(g)
            summary = page_summary(snap, self.poller.config, grains)
            summary["size_method"] = jar.method
            summary["radius_range"] = [R_MIN, float(jar.radii.max()) if jar.count else R_MIN]
            st["tipjar"] = summary
            label = "DEMO (sample transactions, no real address)" if snap["demo"] else "watch-only"
            note = "tip jar not configured: " if snap["demo"] else ""
            st["status"] = (f"{note}tip jar {label} · {summary['balance_btc']} BTC ({summary['balance_sats']:,} sats) · {summary['count']} donations · "
                            f"one grain each, size by amount ({jar.method}), hue by source, lighter is newer, near white = unconfirmed"
                            + (" · last poll failed, showing the last list" if snap.get("error") else ""))
            return st

    return TipJarGame


_GAME = None


def TipJarGameClass():
    global _GAME
    _GAME = _GAME or _game_class()
    return _GAME

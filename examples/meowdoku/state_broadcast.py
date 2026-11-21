"""Regional state broadcast: one writer, every enrolled region a full encrypted copy of the hidden state.

Owner, 2026-11-21 (ticket e2f19532): "each region should be able to grow a new entire system inside itself if there's
room, so the database needs to be broadcast written everywhere ... regularly"; "connected systems should use a single
writer many reader model"; "of course nothing handled in the clear". Hidden data -- the state root, its secrets, what
git does not carry -- moves only through canvas-owned back channels; code and tool versions come from git.

A GENERATION is one consistent snapshot of the writer's state root (state_backup.snapshot: SQLite online backup plus
integrity_check, secrets included, the region's own keys excluded), packed as a stream of RECORDS:

    record 0   header: magic, then the manifest JSON, then the writer's Ed25519 signature over the manifest bytes
    record 1.. chunks: the tar of the snapshot, gzip, sealed with ChaCha20-Poly1305 in chunks of CHUNK_BYTES

The manifest names the generation number (rising, per writer), the writer region, when, the tool commits that wrote
it, the chunk count and the sha256 of every sealed chunk, and one wrapped content key per recipient: X25519 between a
fresh ephemeral key and the recipient's box key, HKDF-SHA256, then ChaCha20-Poly1305 over the content key. Each chunk's
associated data binds the generation, the chunk index and whether it is the last, so chunks cannot be reordered,
swapped between generations or cut short. The records are transport-agnostic: the Nodus link bulk row carries each as
one U8 tensor (<= 256 MiB); nothing here opens a socket.

A reader accepts a generation only when the signature verifies against the writer key it PINNED, the number is above
every generation it already accepted from that writer, every chunk's hash matches, and the unsealed snapshot passes
state_backup.verify. It keeps the sealed records, never the plaintext. Taking over (becoming the writer from a held
generation) is a separate, explicit act: unseal and state_backup.restore into an empty root.

Keys and files (per region, under <state>/secrets/region/, mode 0600; never broadcast):
    self.json   {"region": name}            sign.key  Ed25519 private (32 bytes)     box.key  X25519 private (32 bytes)
    pins.json   {writer: sign_pub_b64}      accepted.json {writer: highest accepted generation}
    generation  the writer's last packed generation number
Enrollment (who a generation is sealed to) is a row in the accounts SQL (table region_enrollment), so it travels inside
the generations themselves.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import struct
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.exceptions import InvalidSignature, InvalidTag

FORMAT = "speaktome.state-broadcast.v1"
MAGIC = b"STMBCAST1\n"
CHUNK_BYTES = 8 << 20               # 8 MiB of plaintext per sealed chunk; far under the link's 256 MiB bulk limit
REGION_DIR = Path("secrets") / "region"
_RAW = serialization.Encoding.Raw
_PRIV = (serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())

SCHEMA = """
CREATE TABLE IF NOT EXISTS region_enrollment (
    region      TEXT PRIMARY KEY,
    sign_pub    TEXT NOT NULL,          -- Ed25519 public key, base64
    box_pub     TEXT NOT NULL,          -- X25519 public key, base64
    kind        TEXT NOT NULL DEFAULT 'region',     -- 'region' or 'recovery' (the owner's offline key; never a writer)
    enrolled_by INTEGER NOT NULL REFERENCES users(id),
    enrolled_at REAL NOT NULL,
    revoked_at  REAL
)"""


class BroadcastError(Exception):
    """A refusal: the generation, key or request is not acceptable. The message says which and why."""


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str, size: int, what: str) -> bytes:
    raw = base64.b64decode(text.encode("ascii"), validate=True)
    if len(raw) != size:
        raise BroadcastError(f"{what} is {len(raw)} bytes, not {size}")
    return raw


def _write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.replace(tmp, path)


def _read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _write_json(path: Path, value) -> None:
    _write_private(path, json.dumps(value, indent=1, sort_keys=True).encode("utf-8"))


# ---- region keys ------------------------------------------------------------------------------------------------------

class Region:
    """One region's own identity, read from <state>/secrets/region/."""

    def __init__(self, state_dir):
        self.dir = Path(state_dir) / REGION_DIR
        meta = _read_json(self.dir / "self.json", None)
        if meta is None:
            raise BroadcastError(f"no region identity in {self.dir}: run keygen first")
        self.name = str(meta["region"])
        self.sign = Ed25519PrivateKey.from_private_bytes((self.dir / "sign.key").read_bytes())
        self.box = X25519PrivateKey.from_private_bytes((self.dir / "box.key").read_bytes())

    def public(self) -> dict:
        return {"region": self.name, "sign_pub": _b64(self.sign.public_key().public_bytes(_RAW, serialization.PublicFormat.Raw)),
                "box_pub": _b64(self.box.public_key().public_bytes(_RAW, serialization.PublicFormat.Raw))}


def keygen(state_dir, region: str) -> dict:
    """Make this region's keys. Refuses to replace existing ones (a new key is a new enrollment, decided by a person)."""
    d = Path(state_dir) / REGION_DIR
    if (d / "self.json").exists():
        raise BroadcastError(f"{d} already holds a region identity; keys are never replaced in place")
    _write_private(d / "sign.key", Ed25519PrivateKey.generate().private_bytes(*_PRIV))
    _write_private(d / "box.key", X25519PrivateKey.generate().private_bytes(*_PRIV))
    _write_json(d / "self.json", {"region": region})
    return Region(state_dir).public()


def recovery_keygen(out_file) -> dict:
    """The owner's offline recovery key: an X25519 private key written to ``out_file`` (move it offline). It can open
    every generation it is enrolled for and can never write one."""
    out = Path(out_file)
    if out.exists():
        raise BroadcastError(f"{out} exists; a recovery key is never overwritten")
    key = X25519PrivateKey.generate()
    _write_private(out, key.private_bytes(*_PRIV))
    return {"box_pub": _b64(key.public_key().public_bytes(_RAW, serialization.PublicFormat.Raw))}


def pin_writer(state_dir, writer: str, sign_pub: str) -> None:
    """Trust ``writer``'s signing key for generations this region accepts. Re-pinning a different key refuses."""
    _unb64(sign_pub, 32, "sign_pub")
    path = Path(state_dir) / REGION_DIR / "pins.json"
    pins = _read_json(path, {})
    if pins.get(writer) not in (None, sign_pub):
        raise BroadcastError(f"{writer} is already pinned to a different key; unpinning is a manual act")
    pins[writer] = sign_pub
    _write_json(path, pins)


# ---- enrollment (accounts SQL) ----------------------------------------------------------------------------------------

def enroll(accounts_db, region: str, sign_pub: str, box_pub: str, by_username: str, kind: str = "region") -> None:
    if kind not in ("region", "recovery"):
        raise BroadcastError(f"unknown enrollment kind {kind!r}")
    _unb64(sign_pub, 32, "sign_pub") if sign_pub else None
    _unb64(box_pub, 32, "box_pub")
    with sqlite3.connect(str(accounts_db)) as conn:
        conn.execute(SCHEMA)
        row = conn.execute("SELECT id FROM users WHERE username = ? COLLATE NOCASE", (by_username,)).fetchone()
        if row is None:
            raise BroadcastError(f"no account {by_username!r} to record as the enroller")
        conn.execute("INSERT INTO region_enrollment (region, sign_pub, box_pub, kind, enrolled_by, enrolled_at) "
                     "VALUES (?, ?, ?, ?, ?, ?)", (region, sign_pub or "", box_pub, kind, int(row[0]), time.time()))


def revoke(accounts_db, region: str) -> None:
    with sqlite3.connect(str(accounts_db)) as conn:
        conn.execute(SCHEMA)
        done = conn.execute("UPDATE region_enrollment SET revoked_at = ? WHERE region = ? AND revoked_at IS NULL",
                            (time.time(), region)).rowcount
    if not done:
        raise BroadcastError(f"no live enrollment for {region!r}")


def recipients(accounts_db) -> list[dict]:
    with sqlite3.connect(str(accounts_db)) as conn:
        conn.execute(SCHEMA)
        rows = conn.execute("SELECT region, box_pub, kind FROM region_enrollment WHERE revoked_at IS NULL "
                            "ORDER BY region").fetchall()
    return [{"region": r, "box_pub": b, "kind": k} for r, b, k in rows]


# ---- sealing ----------------------------------------------------------------------------------------------------------

def _kek(shared: bytes, generation: int, region: str) -> bytes:
    info = f"{FORMAT} key-wrap generation={generation} region={region}".encode("utf-8")
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(shared)


def _chunk_aad(generation: int, writer: str, index: int, last: bool) -> bytes:
    return f"{FORMAT}|{writer}|{generation}|{index}|{int(last)}".encode("utf-8")


def _nonce(prefix: bytes, index: int) -> bytes:
    return prefix + struct.pack(">Q", index)


def _tool_commits(repos) -> dict:
    commits = {}
    for name, path in (repos or {}).items():
        try:
            out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
            commits[name] = out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            commits[name] = None
    return commits


def pack(state_dir, accounts_db, work_dir, out_dir, repos=None, exclude=()) -> dict:
    """Pack the next generation of ``state_dir`` into ``out_dir/<writer>-<generation>/`` as record files
    (000000.rec = header, then one file per sealed chunk). ``work_dir`` holds the short-lived plaintext snapshot and
    must be outside the state root; the plaintext is removed before this returns. Memory stays at one chunk."""
    import state_backup

    region = Region(state_dir)
    targets = recipients(accounts_db)
    if not targets:
        raise BroadcastError("no enrolled regions: a generation sealed to nobody is refused")
    gen_path = region.dir / "generation"
    generation = int(gen_path.read_text().strip()) + 1 if gen_path.is_file() else 1
    out = Path(out_dir)
    final = out / f"{region.name}-{generation:08d}"
    work_out = out / f".partial-{region.name}-{generation:08d}"
    if final.exists():
        raise BroadcastError(f"{final} already exists")
    shutil.rmtree(work_out, ignore_errors=True)
    work_out.mkdir(parents=True)

    key, prefix = os.urandom(32), os.urandom(4)
    aead = ChaCha20Poly1305(key)
    hashes_out, files = [], []
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=work_dir) as scratch,                 tempfile.TemporaryDirectory(dir=region.dir) as private:          # the archive sits in the restricted folder
            snap = state_backup.snapshot(state_dir, Path(scratch) / "snapshots", keep=1, include_secrets=True,
                                         exclude=tuple(exclude) + (REGION_DIR.as_posix(),))
            files = [{"rel": f["rel"], "kind": f["kind"], "size": f["size"]} for f in snap["files"]]
            archive = Path(private) / "generation.tar.gz"
            with gzip.GzipFile(archive, mode="wb", compresslevel=3, mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
                tar.add(snap["snapshot"], arcname="snapshot")
            shutil.rmtree(snap["snapshot"], ignore_errors=True)
            size = archive.stat().st_size
            count = max(1, -(-size // CHUNK_BYTES))
            with open(archive, "rb") as source:
                for index in range(count):
                    sealed = aead.encrypt(_nonce(prefix, index), source.read(CHUNK_BYTES),
                                          _chunk_aad(generation, region.name, index, index == count - 1))
                    (work_out / f"{index + 1:06d}.rec").write_bytes(sealed)
                    hashes_out.append(hashlib.sha256(sealed).hexdigest())
            archive.unlink()
        wrapped = []
        for target in targets:
            ephemeral = X25519PrivateKey.generate()
            shared = ephemeral.exchange(X25519PublicKey.from_public_bytes(_unb64(target["box_pub"], 32, "box_pub")))
            nonce = os.urandom(12)
            wrapped.append({"region": target["region"], "kind": target["kind"],
                            "epk": _b64(ephemeral.public_key().public_bytes(_RAW, serialization.PublicFormat.Raw)),
                            "nonce": _b64(nonce),
                            "key": _b64(ChaCha20Poly1305(_kek(shared, generation, target["region"])).encrypt(nonce, key, None))})
        manifest = {"format": FORMAT, "generation": generation, "writer": region.name,
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "tools": _tool_commits(repos), "files": files, "chunk_bytes": CHUNK_BYTES,
                    "nonce_prefix": _b64(prefix), "chunks": len(hashes_out), "chunk_sha256": hashes_out,
                    "recipients": wrapped}
        body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        (work_out / "000000.rec").write_bytes(MAGIC + struct.pack(">I", len(body)) + body + region.sign.sign(body))
        os.replace(work_out, final)
    except BaseException:
        shutil.rmtree(work_out, ignore_errors=True)
        raise
    gen_path.write_text(str(generation))
    return {"writer": region.name, "generation": generation, "chunks": len(hashes_out),
            "recipients": [t["region"] for t in targets], "dir": str(final)}


# ---- reading ----------------------------------------------------------------------------------------------------------

def read_header(header: bytes) -> tuple[dict, bytes, bytes]:
    if not header.startswith(MAGIC):
        raise BroadcastError("not a state-broadcast header")
    (length,) = struct.unpack(">I", header[len(MAGIC):len(MAGIC) + 4])
    body = header[len(MAGIC) + 4:len(MAGIC) + 4 + length]
    signature = header[len(MAGIC) + 4 + length:]
    if len(body) != length or len(signature) != 64:
        raise BroadcastError("header is cut short")
    manifest = json.loads(body.decode("utf-8"))
    if manifest.get("format") != FORMAT:
        raise BroadcastError(f"unknown format {manifest.get('format')!r}")
    return manifest, body, signature


def _records(generation_dir) -> list[Path]:
    paths = sorted(Path(generation_dir).glob("*.rec"))
    if not paths or paths[0].name != "000000.rec":
        raise BroadcastError(f"{generation_dir} holds no header record")
    return paths


def _check_chunks(paths: list[Path], manifest: dict) -> None:
    chunks = paths[1:]
    if len(chunks) != int(manifest["chunks"]):
        raise BroadcastError(f"{len(chunks)} chunk(s) arrived, the manifest names {manifest['chunks']}")
    for index, chunk in enumerate(chunks):
        if hashlib.sha256(chunk.read_bytes()).hexdigest() != manifest["chunk_sha256"][index]:
            raise BroadcastError(f"chunk {index} does not match its hash")


def _unseal_to(paths: list[Path], box: X25519PrivateKey, me: str, manifest: dict, archive: Path) -> None:
    """Open the content key with this region's box key, then unseal the chunks one at a time into ``archive``."""
    entry = next((r for r in manifest["recipients"] if r["region"] == me), None)
    if entry is None:
        raise BroadcastError(f"generation {manifest['generation']} is not sealed to {me}")
    shared = box.exchange(X25519PublicKey.from_public_bytes(_unb64(entry["epk"], 32, "epk")))
    generation, writer = int(manifest["generation"]), str(manifest["writer"])
    chunks = paths[1:]
    try:
        key = ChaCha20Poly1305(_kek(shared, generation, me)).decrypt(_unb64(entry["nonce"], 12, "nonce"),
                                                                    base64.b64decode(entry["key"]), None)
        prefix = _unb64(manifest["nonce_prefix"], 4, "nonce_prefix")
        aead = ChaCha20Poly1305(key)
        with open(archive, "wb") as sink:
            for index, chunk in enumerate(chunks):
                sink.write(aead.decrypt(_nonce(prefix, index), chunk.read_bytes(),
                                        _chunk_aad(generation, writer, index, index == len(chunks) - 1)))
    except InvalidTag:
        archive.unlink(missing_ok=True)
        raise BroadcastError(f"generation {generation} does not open with {me}'s key (tampered, or not for this key)") from None


def _extract(archive: Path, into: Path) -> Path:
    with gzip.GzipFile(archive, mode="rb") as gz, tarfile.open(fileobj=gz, mode="r") as tar:
        tar.extractall(into, filter="data")
    return into / "snapshot"


def _verify_signature(region: "Region", manifest: dict, body: bytes, signature: bytes) -> None:
    writer = str(manifest["writer"])
    pins = _read_json(region.dir / "pins.json", {})
    if writer not in pins:
        raise BroadcastError(f"writer {writer} is not pinned here")
    try:
        Ed25519PublicKey.from_public_bytes(_unb64(pins[writer], 32, "pin")).verify(signature, body)
    except InvalidSignature:
        raise BroadcastError(f"generation {manifest['generation']}: signature does not verify against "
                             f"{writer}'s pinned key") from None


def accept(state_dir, incoming_generation_dir, held_dir) -> dict:
    """A reader region's arrival (the record files of one generation): verify the signature against the pinned writer,
    require a generation above every one accepted from it, check every chunk hash, unseal into this region's private
    folder, state_backup.verify, remove the plaintext, then keep only the sealed records in ``held_dir``."""
    import state_backup

    region = Region(state_dir)
    paths = _records(incoming_generation_dir)
    manifest, body, signature = read_header(paths[0].read_bytes())
    _verify_signature(region, manifest, body, signature)
    writer, generation = str(manifest["writer"]), int(manifest["generation"])
    accepted_path = region.dir / "accepted.json"
    accepted = _read_json(accepted_path, {})
    if generation <= int(accepted.get(writer, 0)):
        raise BroadcastError(f"generation {generation} from {writer} is not above {accepted[writer]}, already accepted")
    _check_chunks(paths, manifest)
    with tempfile.TemporaryDirectory(dir=region.dir) as private:
        archive = Path(private) / "generation.tar.gz"
        _unseal_to(paths, region.box, region.name, manifest, archive)
        result = state_backup.verify(_extract(archive, Path(private)))
    if not result["ok"]:
        raise BroadcastError(f"generation {generation} unseals but does not verify: {result['bad'][:3]}")
    held = Path(held_dir)
    held.mkdir(parents=True, exist_ok=True)
    target, work = held / f"{writer}-{generation:08d}", held / f".partial-{writer}-{generation:08d}"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(incoming_generation_dir, work)
    os.replace(work, target)
    accepted[writer] = generation
    _write_json(accepted_path, accepted)
    return {"accepted": True, "writer": writer, "generation": generation, "files": result["checked"], "held": str(target)}


def take_over(state_dir, held_generation_dir, target_state_dir, writer_stopped: bool) -> dict:
    """Become the writer from a held generation: unseal it and state_backup.restore it into an EMPTY root. Refuses
    unless the caller states the old writer is stopped (one writer, always), and unless this is the newest generation
    held from that writer."""
    import state_backup

    if not writer_stopped:
        raise BroadcastError("take-over refused: state that the current writer is stopped (writer_stopped=True)")
    region = Region(state_dir)
    paths = _records(held_generation_dir)
    manifest, body, signature = read_header(paths[0].read_bytes())
    _verify_signature(region, manifest, body, signature)
    writer, generation = str(manifest["writer"]), int(manifest["generation"])
    newest = int(_read_json(region.dir / "accepted.json", {}).get(writer, 0))
    if generation != newest:
        raise BroadcastError(f"take-over refused: {held_generation_dir} is generation {generation}, the newest held "
                             f"from {writer} is {newest}")
    _check_chunks(paths, manifest)
    with tempfile.TemporaryDirectory(dir=region.dir) as private:
        archive = Path(private) / "generation.tar.gz"
        _unseal_to(paths, region.box, region.name, manifest, archive)
        restored = state_backup.restore(_extract(archive, Path(private)), target_state_dir, into_empty=True)
    if not restored.get("ok"):
        raise BroadcastError(f"restore did not verify: {restored.get('bad')}")
    # The new writer continues the numbering, so every reader keeps seeing a rising generation.
    (region.dir / "generation").write_text(str(generation))
    return {"took_over_from": writer, "generation": generation, "restored": restored["count"], "into": str(target_state_dir)}

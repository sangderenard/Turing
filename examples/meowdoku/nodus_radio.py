"""Read the configured Nodus Radio's existing relay inbox; no socket or audio capture.

NRAD v1 is the native Radio's RMS/band telemetry, not PCM audio. Only an
operator-selected file-source station is exposed; arbitrary relay inboxes and
system-output capture are never published by this endpoint.
"""
from __future__ import annotations
from contextlib import closing
import json
import math
import sqlite3
import struct
import time
from meowdoku import relay

HEADER = struct.Struct('<4sIIIQf')
MAX_BANDS = 128


def decode_packet(data):
    if len(data) < HEADER.size:
        raise ValueError('short NRAD packet')
    magic, version, count, rate, frame, rms = HEADER.unpack_from(data)
    if magic != b'NRAD' or version != 1 or not 1 <= count <= MAX_BANDS:
        raise ValueError('unsupported NRAD header')
    if len(data) != HEADER.size + 4 * count or not 8000 <= rate <= 384000:
        raise ValueError('invalid NRAD shape or sample rate')
    bands = struct.unpack_from('<' + 'f' * count, data, HEADER.size)
    if not math.isfinite(rms) or rms < 0 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in bands):
        raise ValueError('invalid NRAD levels')
    return {'sample_rate': rate, 'frame': str(frame), 'rms': rms, 'bands': list(bands)}


def snapshot(config_path, accounts_db, *, now=None, station=None):
    now = time.time() if now is None else now
    empty = {'status': 'unavailable', 'title': 'Nodus Radio', 'bands': [], 'audio': False}
    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return empty
    # Publishing must be intentional and limited to a dedicated authored source.
    if config.get('enabled') is not True or config.get('source') != 'file':
        return empty
    try:
        audio = station is not None and config['jukebox']['native']['hls']['enabled'] is True
    except (KeyError, TypeError):
        audio = False
    address, owner, peer = config.get('address'), config.get('owner_id'), config.get('peer')
    # Trusted in-process native source metadata, never taken from HTTP inputs.
    if station is not None:
        peer = station['peer']
        config['title'] = station['title']
    if type(address) is not int or type(owner) is not int or not isinstance(peer, str) or not peer:
        raise ValueError('invalid Radio station configuration')
    state = {**empty, 'title': str(config.get('title', 'Nodus Radio'))[:160], 'status': 'waiting', 'audio': audio}
    # Read-only view of the existing bounded relay table. Validate the real
    # owner's open address, then inspect only its newest retained record.
    uri = accounts_db.resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True, timeout=2)) as db:
        relay._owned(db, address, owner)
        rows = db.execute("SELECT peer, kind, body, created_at FROM relay_packages WHERE address=? AND direction='in' ORDER BY id DESC LIMIT 8", (address,)).fetchall()
    for sender, kind, body, created in rows:
        try:
            packets = relay.unpack_record(body) if kind == relay.CONGLOMERATE else [(sender, body)]
            for sender, data in reversed(packets):
                if sender != peer:
                    continue
                try:
                    result = decode_packet(data)
                except ValueError:
                    continue
                age = max(0, now - created)
                return {**state, **result, 'status': 'live' if age < 2 else 'stale', 'age_s': round(age, 3)}
        except (ValueError, IndexError, struct.error, OSError):
            continue
    return state

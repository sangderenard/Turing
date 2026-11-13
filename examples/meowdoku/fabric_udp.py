"""Collect complete NFAB frames from the native Nodus spring row.

The producer owns physics. This receiver only reassembles its bounded UDP
publication so Studio can display a coherent state, skipping missing chunks.
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time


HEADER = struct.Struct("<IHHQQHHHH")
PAIR = struct.Struct("<Ifff")
MAGIC = 0x4241464E


class FabricUdpReceiver:
    def __init__(self, host="127.0.0.1", port=48768):
        self.host, self.port = host, port
        self._lock = threading.Lock()
        self._partial = {}
        self._latest = {}
        self._socket = None

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        multicast = 224 <= int(self.host.split('.')[0]) <= 239
        sock.bind(("0.0.0.0" if multicast else self.host, self.port))
        if multicast:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                            socket.inet_aton(self.host) + socket.inet_aton("0.0.0.0"))
        sock.settimeout(0.5)
        self._socket = sock
        threading.Thread(target=self._receive, name="nodus-fabric-udp", daemon=True).start()
        return self

    def _receive(self):
        while self._socket is not None:
            try:
                packet, _ = self._socket.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            self.accept(packet)

    def accept(self, packet):
        if len(packet) < HEADER.size:
            return False
        magic, version, header_size, frame, identity, chunk, chunks, count, state = HEADER.unpack_from(packet)
        if (magic != MAGIC or version != 1 or header_size != HEADER.size or not identity or
                not 1 <= chunks <= 6 or chunk >= chunks or count > 48 or
                len(packet) != HEADER.size + count * PAIR.size or state & ~1):
            return False
        pairs = {}
        for offset in range(HEADER.size, len(packet), PAIR.size):
            node, x, y, z = PAIR.unpack_from(packet, offset)
            if node >= 256 or node in pairs or not all(map(math.isfinite, (x, y, z))):
                return False
            pairs[node] = [x, y, z]
        now = time.monotonic()
        with self._lock:
            for key, value in list(self._partial.items()):
                if now - value[0] > 0.5:
                    del self._partial[key]
            key = identity, frame
            old = self._latest.get(identity)
            if old and frame <= old["frame"]:
                return False
            entry = self._partial.setdefault(key, (now, chunks, state, {}))
            if entry[1:3] != (chunks, state) or chunk in entry[3]:
                return False
            entry[3][chunk] = pairs
            if len(entry[3]) != chunks:
                return False
            merged = {}
            for part in range(chunks):
                for node, point in entry[3][part].items():
                    if node in merged:
                        del self._partial[key]
                        return False
                    merged[node] = point
            self._latest[identity] = {"identity": identity, "frame": frame,
                                      "frozen": bool(state & 1), "positions": merged,
                                      "received_at": now}
            del self._partial[key]
            if len(self._latest) > 32:
                oldest = min(self._latest, key=lambda ident: self._latest[ident]["received_at"])
                del self._latest[oldest]
            return True

    def latest(self, identity):
        with self._lock:
            value = self._latest.get(identity)
            if value is None:
                return None
            return {key: item for key, item in value.items() if key != "received_at"}

    def close(self):
        sock, self._socket = self._socket, None
        if sock is not None:
            sock.close()

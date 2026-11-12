"""Trusted backend subscription to native Nodus Radio PCM via udp_router.

No HTTP listener, SQL recording, audio device, or authentication bypass. The
operator supplies an allocated UDP port and a pinned producer endpoint/SSRC.
The consumer clocks take() at 48000/256 Hz; missing blocks become silence.
"""
from __future__ import annotations
from array import array
from dataclasses import dataclass
import socket
import struct
import threading
from meowdoku.relay import FRAME, ENTRY, OPEN, OPENED, BATCH, NOTE, READY

SAMPLES = 256
CHANNELS = 2
RATE = 48000
BLOCK_BYTES = SAMPLES * CHANNELS * 2
SILENCE = bytes(BLOCK_BYTES)


@dataclass(frozen=True)
class PcmPacket:
    sequence: int
    timestamp: int
    ssrc: int
    pcm_s16le: bytes


def decode_rtp(data: bytes) -> PcmPacket:
    # This is the explicitly negotiated fixed Radio profile, not all RTP.
    if len(data) != 12 + BLOCK_BYTES:
        raise ValueError('Radio RTP packet must contain 256 stereo L16 frames')
    first, second, seq, timestamp, ssrc = struct.unpack_from('!BBHII', data)
    if first != 0x80 or second & 0x7f != 96:
        raise ValueError('Radio requires RTP v2/PT96 without CSRC, extension or padding')
    samples = array('h'); samples.frombytes(data[12:])
    samples.byteswap()  # Swap the network-order bytes to little-endian bytes.
    return PcmPacket(seq, timestamp, ssrc, samples.tobytes())


def distance(seq, base):
    return ((seq - base + 32768) & 65535) - 32768


class JitterBuffer:
    """Bounded by packets, caller-clocked. Never synthesizes energy for a loss."""
    def __init__(self, *, ssrc, capacity=32, prefill=3):
        if not 1 <= prefill <= capacity <= 512 or not 0 <= ssrc <= 0xffffffff:
            raise ValueError('invalid buffer policy or SSRC')
        self.ssrc, self.capacity, self.prefill = ssrc, capacity, prefill
        self.lock = threading.Lock()
        self.pending = {}
        self.expected = self.timestamp = None
        self.started = False
        self.received = self.lost = self.late = self.rejected = self.resyncs = 0

    def put(self, packet):
        with self.lock:
            if packet.ssrc != self.ssrc:
                self.rejected += 1; return False
            if self.expected is None:
                self.expected, self.timestamp = packet.sequence, packet.timestamp
            delta = distance(packet.sequence, self.expected)
            if delta < 0:
                self.late += 1; return False
            if packet.timestamp != ((self.timestamp + delta * SAMPLES) & 0xffffffff):
                self.rejected += 1; return False
            if delta >= self.capacity:
                # A paused consumer must not retain an unbounded backlog or
                # reject fresh audio forever. Drop backlog and prebuffer anew.
                self.pending.clear(); self.started = False
                self.expected, self.timestamp = packet.sequence, packet.timestamp
                self.resyncs += 1
            if packet.sequence in self.pending:
                self.rejected += 1; return False
            self.pending[packet.sequence] = packet.pcm_s16le
            self.received += 1
            return True

    def take(self):
        with self.lock:
            if self.expected is None or (not self.started and len(self.pending) < self.prefill):
                return None
            self.started = True
            block = self.pending.pop(self.expected, None)
            if block is None:
                self.lost += 1; block = SILENCE
            self.expected = (self.expected + 1) & 65535
            self.timestamp = (self.timestamp + SAMPLES) & 0xffffffff
            return block

    def status(self):
        with self.lock:
            return {'queued': len(self.pending), 'capacity': self.capacity, 'received': self.received,
                    'lost': self.lost, 'late': self.late, 'rejected': self.rejected, 'resyncs': self.resyncs}


def unpack_batch(data):
    if len(data) < 4:
        raise ValueError('short router batch')
    count, = struct.unpack_from('<I', data)
    if count > (len(data) - 4) // ENTRY.size:
        raise ValueError('invalid router count')
    at = 4
    result = []
    for _ in range(count):
        if at + ENTRY.size > len(data):
            raise ValueError('short router entry')
        ip, port, size = ENTRY.unpack_from(data, at); at += ENTRY.size
        if at + size > len(data):
            raise ValueError('short router payload')
        result.append(((socket.inet_ntoa(ip), port), data[at:at + size])); at += size
    if at != len(data):
        raise ValueError('trailing router bytes')
    return result


class Subscription:
    """One actual router-owned UDP port. Closing TCP releases it in the router.

    Construction is only for a trusted backend caller after its service has
    authorized and allocated the subscription. It grants no account capability.
    """
    def __init__(self, *, router_port, listen_port, peer, ssrc, capacity=32, prefill=3):
        if not 0 < router_port < 65536 or not 0 < listen_port < 65536:
            raise ValueError('invalid port')
        socket.inet_aton(peer[0])
        if not 0 < peer[1] < 65536:
            raise ValueError('invalid producer port')
        self.port, self.peer = listen_port, tuple(peer)
        self.buffer = JitterBuffer(ssrc=ssrc, capacity=capacity, prefill=prefill)
        self.error = None; self.invalid = self.foreign = 0
        self.ready = threading.Event(); self.stopped = threading.Event()
        self.socket = socket.create_connection(('127.0.0.1', router_port), timeout=3)
        self.socket.settimeout(None)
        self.thread = threading.Thread(target=self._run, name='nodus-radio-pcm', daemon=True)
        self.thread.start()

    def _read(self, length):
        data = bytearray()
        while len(data) < length:
            part = self.socket.recv(length - len(data))
            if not part:
                raise EOFError('Nodus router disconnected')
            data.extend(part)
        return bytes(data)

    def _run(self):
        try:
            while not self.stopped.is_set():
                op, port, _ip, _peer_port, size = FRAME.unpack(self._read(FRAME.size))
                if size > 2 * 1024 * 1024:
                    raise ValueError('oversize router frame')
                body = self._read(size)
                if op == READY:
                    # Native router must be bound on loopback for this private
                    # PCM experiment. Broader publication requires service policy.
                    if body != b'127.0.0.1':
                        raise ValueError('PCM experiment requires loopback-bound router')
                    self.socket.sendall(FRAME.pack(OPEN, self.port, bytes(4), 0, 0))
                elif op == OPENED and port == self.port:
                    self.ready.set()
                elif op == NOTE and port == self.port:
                    raise OSError(body.decode('utf-8', errors='replace')[:256])
                elif op == BATCH and port == self.port:
                    for peer, packet in unpack_batch(body):
                        if peer != self.peer:
                            self.foreign += 1; continue
                        try:
                            self.buffer.put(decode_rtp(packet))
                        except ValueError:
                            self.invalid += 1
        except (OSError, EOFError, ValueError, struct.error) as error:
            if not self.stopped.is_set():
                self.error = str(error)
        finally:
            self.stopped.set()
            self.socket.close()

    def close(self):
        self.stopped.set()
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError('Radio subscription did not stop')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

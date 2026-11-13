"""Trusted backend subscription to native Nodus Radio PCM via udp_router.

No HTTP listener, SQL recording, audio device, or authentication bypass. The
operator supplies an allocated UDP port and a pinned producer endpoint/SSRC.
Use take_due() for bounded adaptive readback (perf_counter clock), or take()
when an external audio clock owns pacing. Missing blocks remain counted silence.
"""
from __future__ import annotations
from array import array
from dataclasses import dataclass
import socket
import struct
import threading
import time
import math
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
    def __init__(self, *, ssrc, capacity=32, prefill=8,
                 max_rate_offset=.005, slew_per_second=.002, momentum_seconds=.5,
                 clock=time.perf_counter):
        if not 1 <= prefill <= capacity <= 512 or not 0 <= ssrc <= 0xffffffff:
            raise ValueError('invalid buffer policy or SSRC')
        self.ssrc, self.capacity, self.prefill = ssrc, capacity, prefill
        if not (0 < max_rate_offset <= .02 and 0 < slew_per_second <= .1
                and 0 < momentum_seconds <= 10):
            raise ValueError('invalid readback policy')
        self.lock = threading.RLock()
        self.pending = {}
        self.expected = self.timestamp = None
        self.started = False
        self.received = self.lost = self.late = self.rejected = self.resyncs = 0
        self.clock = clock
        self.max_rate_offset = max_rate_offset
        self.slew_per_second = slew_per_second
        self.momentum_seconds = momentum_seconds
        self.rate_offset = self.filtered_error = 0.
        self.deadline = self.last_read = self.last_poll = None
        self.end_sequence = None
        self.ended = False
        self.read_blocks = self.clock_rebuffers = 0

    def finish(self, last_sequence):
        """Trusted control-plane EOF, NOT inferred from a quiet UDP stream.

        Call after delivering the final batch. The final sequence bounds any
        missing tail blocks; after that, readback stops without inventing loss.
        """
        with self.lock:
            if not 0 <= last_sequence <= 65535 or self.expected is None:
                raise ValueError('EOF requires a started sequence and uint16 end')
            if self.end_sequence is not None:
                if self.end_sequence != last_sequence:
                    raise ValueError('conflicting EOF')
                return
            delta = distance(last_sequence, self.expected)
            if delta < -1 or delta >= self.capacity:
                raise ValueError('EOF outside bounded current sequence window')
            if any(distance(seq, last_sequence) > 0 for seq in self.pending):
                raise ValueError('EOF precedes queued PCM')
            self.end_sequence = last_sequence
            self.ended = delta == -1

    def take_due(self, now=None):
        """Return one unchanged PCM block when due; None means wait or ended.

        Queue depth drives a low-pass (momentum) rate target with a bounded
        rate-of-change (slew). The clock is perf_counter, not Windows' coarse
        GetTickCount clock. No sample interpolation, repetition or hidden loss
        recovery occurs here. A fixed-rate DAC still needs its own resampler.
        Poll at next_read_at; only one consumer may own this readback clock.
        """
        with self.lock:
            now = self.clock() if now is None else now
            if not math.isfinite(now) or (self.last_poll is not None and now < self.last_poll):
                raise ValueError('readback clock must be finite and monotonic')
            self.last_poll = now
            if self.ended:
                return None
            period = SAMPLES / RATE
            # A stalled consumer must rebuffer, not burst many overdue blocks.
            if self.deadline is not None and now - self.deadline > self.capacity * period:
                self.started = False
                self.deadline = self.last_read = None
                self.rate_offset = self.filtered_error = 0.
                self.clock_rebuffers += 1
            if self.deadline is not None and now < self.deadline:
                return None
            block = self.take()
            if block is None:
                return None
            dt = period if self.last_read is None else now - self.last_read
            # Depth after a read targets prefill-1. Positive means speed up.
            error = len(self.pending) - (self.prefill - 1)
            alpha = -math.expm1(-dt / self.momentum_seconds)
            self.filtered_error += alpha * (error - self.filtered_error)
            target = max(-self.max_rate_offset, min(self.max_rate_offset,
                         self.filtered_error * .002))
            step = self.slew_per_second * dt
            self.rate_offset += max(-step, min(step, target - self.rate_offset))
            anchor = now if self.deadline is None else self.deadline
            self.deadline = anchor + period / (1. + self.rate_offset)
            self.last_read = now
            self.read_blocks += 1
            return block

    def put(self, packet):
        with self.lock:
            if self.ended or (self.end_sequence is not None and
                              distance(packet.sequence, self.end_sequence) > 0):
                self.rejected += 1; return False
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
                self.deadline = self.last_read = None
                self.rate_offset = self.filtered_error = 0.
            if packet.sequence in self.pending:
                self.rejected += 1; return False
            self.pending[packet.sequence] = packet.pcm_s16le
            self.received += 1
            return True

    def take(self):
        with self.lock:
            if self.ended or self.expected is None or (not self.started and
                    len(self.pending) < self.prefill and self.end_sequence is None):
                return None
            self.started = True
            block = self.pending.pop(self.expected, None)
            if block is None:
                self.lost += 1; block = SILENCE
            self.expected = (self.expected + 1) & 65535
            self.timestamp = (self.timestamp + SAMPLES) & 0xffffffff
            if self.end_sequence is not None and self.expected == ((self.end_sequence + 1) & 65535):
                self.ended = True
            return block

    def status(self):
        with self.lock:
            return {'queued': len(self.pending), 'capacity': self.capacity, 'received': self.received,
                    'lost': self.lost, 'late': self.late, 'rejected': self.rejected, 'resyncs': self.resyncs,
                    'read_rate': 1. + self.rate_offset, 'next_read_at': self.deadline,
                    'read_blocks': self.read_blocks, 'clock_rebuffers': self.clock_rebuffers,
                    'end_sequence': self.end_sequence, 'ended': self.ended}


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
    def __init__(self, *, router_port, listen_port, peer, ssrc, capacity=32, prefill=8):
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

    def take_due(self, now=None):
        return self.buffer.take_due(now)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

import pathlib, sys, struct, unittest, socket, threading
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'examples'))
from meowdoku.nodus_radio_pcm import *

def packet(seq=1,ts=100,ssrc=42):
    return struct.pack('!BBHII',128,96,seq,ts,ssrc)+struct.pack('!512h',*([1234,-2345]*256))

class PcmTests(unittest.TestCase):
    def test_decode_and_reject(self):
        p=decode_rtp(packet());self.assertEqual(p.pcm_s16le[:4],struct.pack('<hh',1234,-2345))
        for data in (b'', packet()[:-1],bytes([129])+packet()[1:],packet()[:1]+bytes([97])+packet()[2:]):
            with self.assertRaises(ValueError):decode_rtp(data)
    def test_order_loss_and_wrap(self):
        b=JitterBuffer(ssrc=42,prefill=2)
        self.assertTrue(b.put(decode_rtp(packet(65535,0xffffff00))))
        self.assertIsNone(b.take())
        b.put(decode_rtp(packet(1,256)));b.put(decode_rtp(packet(0,0)))
        for _ in range(3):self.assertNotEqual(b.take(),SILENCE)
        self.assertEqual(b.take(),SILENCE);self.assertEqual(b.lost,1)
        self.assertFalse(b.put(decode_rtp(packet(0,0))))
        self.assertFalse(b.put(decode_rtp(packet(3,768,43))))
    def test_bound_and_timestamp(self):
        b=JitterBuffer(ssrc=42,capacity=4,prefill=1)
        for n in range(20):b.put(decode_rtp(packet(n,n*256)))
        self.assertLessEqual(b.status()['queued'],4);self.assertGreater(b.resyncs,0)
        self.assertFalse(b.put(decode_rtp(packet(20,1))))
    def test_batch(self):
        p=packet();batch=struct.pack('<I',1)+ENTRY.pack(socket.inet_aton('127.0.0.1'),1234,len(p))+p
        self.assertEqual(unpack_batch(batch),[(('127.0.0.1',1234),p)])
        for bad in (b'',batch[:-1],batch+b'x'):
            with self.assertRaises(ValueError):unpack_batch(bad)
    def test_control_stream_fragmentation(self):
        listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen()
        router_port=listener.getsockname()[1]; outcome=[]
        def server():
            conn,_=listener.accept()
            with conn:
                hello=FRAME.pack(READY,0,bytes(4),0,9)+b'127.0.0.1'
                for part in (hello[:2],hello[2:9],hello[9:]):conn.sendall(part)
                data=b''
                while len(data)<FRAME.size:data+=conn.recv(FRAME.size-len(data))
                outcome.append(FRAME.unpack(data)[:2]);conn.sendall(FRAME.pack(OPENED,49123,bytes(4),0,0))
                p=packet();body=struct.pack('<I',1)+ENTRY.pack(socket.inet_aton('127.0.0.1'),1234,len(p))+p
                conn.sendall(FRAME.pack(BATCH,49123,bytes(4),0,len(body))+body)
                while conn.recv(1):pass
        t=threading.Thread(target=server);t.start()
        with Subscription(router_port=router_port,listen_port=49123,peer=('127.0.0.1',1234),ssrc=42,prefill=1) as sub:
            self.assertTrue(sub.ready.wait(2))
            for _ in range(100):
                if sub.buffer.status()['received']:break
                threading.Event().wait(.01)
            self.assertEqual(sub.take_due(),decode_rtp(packet()).pcm_s16le)
        t.join(2);listener.close();self.assertFalse(t.is_alive());self.assertEqual(outcome,[(OPEN,49123)])
class ReadbackTests(unittest.TestCase):
    def test_standard_terminal_rtcp_and_short_final_block(self):
        import struct
        b = JitterBuffer(ssrc=42)
        b.put(decode_rtp(packet(65535, 0xffffff00)))
        short = struct.pack('!BBHII', 128, 96, 0, 0, 42) + struct.pack('!14h', *range(14))
        p = decode_rtp(short)
        self.assertEqual(len(p.pcm_s16le), 28)
        terminal = (struct.pack('!BBH6I', 128, 200, 6, 42, 0, 0, 7, 2, 1052) +
                    struct.pack('!BBHI', 129, 202, 3, 42) + b'\x01\x05radio\0' +
                    struct.pack('!BBHI', 129, 203, 1, 42))
        endpoint = decode_end(terminal, 42)
        self.assertEqual(endpoint, 7)
        b.finish_timestamp(endpoint)  # RTCP may overtake the final UDP packet.
        self.assertTrue(b.put(p))
        self.assertEqual(len(b.take_due(0)), 1024)
        self.assertEqual(b.take_due(b.deadline), p.pcm_s16le)
        self.assertTrue(b.ended)
        self.assertIsNone(b.take_due(b.deadline + 1))
        b.finish_timestamp(endpoint)  # Repeated BYE is idempotent.
        self.assertEqual(b.lost, 0)
        for bad in (terminal[:-1], terminal+b'x', terminal[:28]+b'x'+terminal[29:]):
            with self.assertRaises(ValueError): decode_end(bad, 42)
        with self.assertRaises(ValueError): decode_end(terminal, 43)
        with self.assertRaises(ValueError): b.finish_timestamp(8)

    def test_lost_short_final_block_has_exact_endpoint_length(self):
        b = JitterBuffer(ssrc=42)
        b.put(decode_rtp(packet(0, 0)))
        b.finish_timestamp(SAMPLES + 7)
        b.take_due(0)
        self.assertEqual(b.take_due(b.deadline), bytes(28))
        self.assertEqual(b.lost, 1)
        self.assertTrue(b.ended)

    def test_high_resolution_clock_and_policy(self):
        now = [0.]
        b = JitterBuffer(ssrc=42, prefill=1, clock=lambda: now[0])
        b.put(decode_rtp(packet(0, 0)))
        self.assertIsNotNone(b.take_due())
        self.assertIsNone(b.take_due())
        now[0] = SAMPLES / RATE
        self.assertEqual(b.take_due(), SILENCE)
        self.assertEqual(b.lost, 1)
        for value in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError): b.take_due(value)
        for args in ({'max_rate_offset': 2}, {'slew_per_second': 0}, {'momentum_seconds': 0}):
            with self.assertRaises(ValueError): JitterBuffer(ssrc=42, **args)
        self.assertIs(JitterBuffer(ssrc=42).clock, time.perf_counter)

    def test_slew_and_momentum_bounds(self):
        b = JitterBuffer(ssrc=42, prefill=3)
        for i in range(20): b.put(decode_rtp(packet(i, i * SAMPLES)))
        prior, t = 1., 0.
        for i in range(10):
            self.assertIsNotNone(b.take_due(t))
            rate = b.status()['read_rate']
            dt = SAMPLES / RATE if i == 0 else t - last
            self.assertLessEqual(abs(rate - prior), b.slew_per_second * dt + 1e-12)
            self.assertLessEqual(abs(rate - 1), b.max_rate_offset)
            self.assertGreater(rate, prior)
            self.assertLess(b.filtered_error, 17)
            prior, last = rate, t
            t = b.status()['next_read_at']

    def test_skew_and_jitter_preserve_every_block(self):
        # Deterministic 60-second producer, +/-2000ppm and bounded arrival jitter.
        # Sequence/timestamp wrap is exercised; no audio device or real sleep.
        import random
        for skew in (-.002, .002):
            rng = random.Random(42)
            count = 11250
            b = JitterBuffer(ssrc=42, prefill=8)
            events = sorted((i * SAMPLES / RATE / (1 + skew) +
                             (0 if i == 0 else rng.uniform(0, .004)), i)
                            for i in range(count))
            at = emitted = 0
            t = 0.
            while not b.ended and t < 62:
                while at < count and events[at][0] <= t:
                    i = events[at][1]
                    b.put(PcmPacket((65500 + i) & 65535,
                                    (0xffffff00 + i * SAMPLES) & 0xffffffff,
                                    42, struct.pack('<I', i)))
                    at += 1
                    if at == count: b.finish((65500 + count - 1) & 65535)
                value = b.take_due(t)
                if value is not None:
                    self.assertEqual(value, struct.pack('<I', emitted))
                    emitted += 1
                t += .0005
            self.assertEqual(emitted, count)
            self.assertEqual((b.lost, b.late, b.resyncs), (0, 0, 0))
            self.assertTrue(b.ended)

    def test_actual_loss_is_counted_and_eof_stops_tail(self):
        b = JitterBuffer(ssrc=42, prefill=8)
        for i in (0, 2): b.put(decode_rtp(packet(i, i * SAMPLES)))
        b.finish(3)  # Both an internal and final block genuinely missing.
        b.finish(3)
        outputs = []
        t = 0.
        for _ in range(4):
            outputs.append(b.take_due(t)); t = b.deadline
        self.assertEqual([x == SILENCE for x in outputs], [False, True, False, True])
        self.assertEqual(b.lost, 2)
        self.assertTrue(b.ended)
        self.assertIsNone(b.take_due(t + 30))
        self.assertEqual(b.lost, 2)
        self.assertFalse(b.put(decode_rtp(packet(4, 4 * SAMPLES))))
        with self.assertRaises(ValueError): b.finish(4)

    def test_stalled_consumer_rebuffers_without_catchup_burst(self):
        b = JitterBuffer(ssrc=42, prefill=2)
        for i in range(3): b.put(decode_rtp(packet(i, i * SAMPLES)))
        b.take_due(0)
        self.assertIsNotNone(b.take_due(1))
        self.assertIsNone(b.take_due(1))
        self.assertEqual(b.clock_rebuffers, 1)
        self.assertEqual(b.lost, 0)

    def test_no_eof_guessed_from_silence_and_bad_eof_refused(self):
        b = JitterBuffer(ssrc=42, prefill=1)
        with self.assertRaises(ValueError): b.finish(0)
        b.put(decode_rtp(packet(0, 0)))
        b.put(decode_rtp(packet(1, SAMPLES)))
        with self.assertRaises(ValueError): b.finish(0)
        with self.assertRaises(ValueError): b.finish(400)
        b.take_due(0)
        b.take_due(b.deadline)
        b.take_due(b.deadline)
        self.assertEqual(b.lost, 1)
        self.assertFalse(b.ended)

if __name__=='__main__':unittest.main()

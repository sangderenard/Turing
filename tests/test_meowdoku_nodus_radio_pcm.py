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
            self.assertEqual(sub.buffer.take(),decode_rtp(packet()).pcm_s16le)
        t.join(2);listener.close();self.assertFalse(t.is_alive());self.assertEqual(outcome,[(OPEN,49123)])
if __name__=='__main__':unittest.main()

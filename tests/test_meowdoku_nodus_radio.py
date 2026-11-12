import sys, pathlib, importlib.util, struct, tempfile, sqlite3, json, unittest, socket
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'examples'))
from meowdoku import nodus_radio as r
class RadioTests(unittest.TestCase):
 def packet(self,frame=55,levels=(.1,.2,.3,.4,.5)):
  return struct.pack('<4sIIIQf',b'NRAD',1,len(levels),48000,frame,.25)+struct.pack('<'+'f'*len(levels),*levels)
 def test_native_contract(self):
  p=r.decode_packet(self.packet(frame=2**60)); self.assertEqual(p['frame'],str(2**60)); self.assertEqual(len(p['bands']),5)
 def test_reject(self):
  p=self.packet()
  for bad in (b'',p[:-1],p+b'0',b'FAIL'+p[4:],self.packet(levels=(float('nan'),)),self.packet(levels=(2.,))):
   with self.assertRaises(ValueError):r.decode_packet(bad)
 def test_relay_view(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d); dbp=root/'accounts.db'; cfg=root/'radio.json'
   db=sqlite3.connect(dbp);r.relay.ensure(db)
   db.execute('INSERT INTO relay_addresses(address,user_id,username,opened_at) VALUES(47701,7,"dot",0)')
   packet=self.packet(); batch=struct.pack('<I',1)+r.relay.ENTRY.pack(socket.inet_aton('127.0.0.1'),23999,len(packet))+packet
   db.execute('INSERT INTO relay_packages(address,direction,peer,kind,body,created_at) VALUES(?,?,?,?,?,?)',(47701,'in','router','datagrams',batch,100));db.commit()
   self.assertEqual(r.snapshot(cfg,dbp,now=100)['status'],'unavailable')
   config={'enabled':True,'source':'file','owner_id':7,'address':47701,'peer':'127.0.0.1:23999','title':'Tempest'}
   cfg.write_text(json.dumps(config)); self.assertEqual(r.snapshot(cfg,dbp,now=101)['status'],'live')
   self.assertEqual(r.snapshot(cfg,dbp,now=103)['status'],'stale')
   config['peer']='127.0.0.1:11111';cfg.write_text(json.dumps(config));self.assertEqual(r.snapshot(cfg,dbp,now=101)['status'],'waiting')
   config['source']='system_output';cfg.write_text(json.dumps(config));self.assertEqual(r.snapshot(cfg,dbp,now=101)['status'],'unavailable')
   config['source']='file';config['owner_id']=8;cfg.write_text(json.dumps(config))
   with self.assertRaises(r.relay.RelayError):r.snapshot(cfg,dbp,now=101)
   db.close()
if __name__=='__main__':unittest.main()

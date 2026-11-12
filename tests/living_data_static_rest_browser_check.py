"""Real Chrome -> actual authenticated REST -> native static Phong round trip."""
from pathlib import Path
import base64, hashlib, json, subprocess, sys, tempfile, threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'examples')]
from meowdoku import server as srv, authz

CHECK=r'''<script>
(async()=>{try{
 const calls=[];
 async function call(path,body,method='POST'){
  const response=await fetch(path,{method,credentials:'same-origin',headers:{'Content-Type':'application/json'},...(body===null?{}:{body:JSON.stringify(body)})});
  const value=await response.json();calls.push({path,status:response.status});
  if(!response.ok)throw Error(path+' '+response.status+' '+JSON.stringify(value));return value;
 }
 await call('/api/account/login',{username:'static_phong_operator',password:'temporary-local-fixture-917'});
 const host=window.livingDataStaticHost;await host.ready;const snapshot=host.snapshot();
 const base='/api/living-data/dm/worlds';
 const attached=await call(base,{map:snapshot.map,source_sha256:snapshot.source_sha256,renderer_identity:snapshot.renderer_identity,request_id:'attach-static'});
 const url=base+'/'+attached.world_id;
 const objectId=snapshot.map.world.objects[0].identity;
 const inspected=await call(url+'/commands',{request_id:'inspect1',operation:'inspect',at:objectId});
 const storybook=snapshot.map.identity==='introspection-world:python:meowdoku.storybook_environment.StorybookEnvironment';
 let viewpoints=[
  {position:[32,2,-10],facing:[0,0,-1],width:320,height:240},
  {position:[28,4,-8],facing:[1,-.2,-1],width:320,height:240}];
 if(storybook){
  const classes=snapshot.map.regions.flatMap(r=>r.buildings.map(b=>b.name));
  for(const name of ['StorybookEnvironment','LanternCottage','WhisperingWood','MoonBridge','ClockworkTower'])
   if(!classes.includes(name))throw Error('Missing authored storybook class '+name);
  if(['entity_mezzanine','vehicle_slot','physics_program','loop_deployment','engine_audio','music_room'].some(k=>k in model))
   throw Error('Static producer included game/physics fields');
  if(JSON.stringify(snapshot.map.world.objects).includes('MechanicalCreature'))throw Error('Mechanical world content leaked');
  const room=snapshot.map.regions.flatMap(r=>r.buildings).flatMap(b=>b.rooms).find(r=>r.metaphor);
  if(!room)throw Error('Authored story metaphor missing from snapshot');
  const semantic=await call(url+'/commands',{request_id:'inspect-story-room',operation:'inspect',layer:'layout',at:room.identity});
  if(semantic.result.object.metaphor!==room.metaphor)throw Error('REST dropped authored metaphor');
  const ids=['python:meowdoku.storybook_environment.LanternCottage','python:meowdoku.storybook_environment.ClockworkTower'];
  viewpoints=ids.map(id=>{const box=snapshot.geometry.boxes.find(b=>b.identity===id);
   if(!box||!box.center||!box.half_extent)throw Error('Missing native storybook geometry '+id);
   return {position:[box.center[0],Math.max(1,box.height*.6),box.center[1]-box.half_extent[1]-3],
    facing:[0,0,1],width:320,height:240};});
 }
 const frames=[];
 for(const [index,camera] of viewpoints.entries()){
  const request_id='render'+index;
  await call(url+'/commands',{request_id,operation:'render-phong',...camera});
  const work=await call(url+'/host',{host_token:attached.host_token,action:'poll'});
  if(work.request_id!==request_id)throw Error('Wrong render command');
  const rendered=await host.render(work.camera);
  await call(url+'/host',{host_token:attached.host_token,action:'complete',request_id,result:{
   png_base64:rendered.data_url.split(',')[1],width:rendered.width,height:rendered.height,
   source_sha256:rendered.source_sha256,renderer_identity:rendered.renderer_identity,
   position:work.camera.position,facing:work.camera.facing,actual_camera:rendered.camera}});
  const receipt=await call(url+'/commands/'+request_id,null,'GET');frames.push(receipt);
 }
 const changed=[...turingWorld.objects.values()][0], oldLabel=changed.label;
 changed.label=String(changed.label)+' test-only mutation';
 let snapshotRejected=false,renderRejected=false,restoredRejected=false;
 try{host.snapshot();}catch(e){snapshotRejected=String(e).includes('Static map changed');}
 try{await host.render({position:[32,2,-10],facing:[0,0,-1],width:320,height:240});}catch(e){renderRejected=String(e).includes('Static map changed');}
 changed.label=oldLabel;
 try{host.snapshot();}catch(e){restoredRejected=String(e).includes('Static map changed');}
 if(!snapshotRejected||!renderRejected||!restoredRejected)throw Error('Static source guard did not latch refusal');
  await fetch('/__static_receipt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ok:true,calls,storybook,viewpoints,world_identity:snapshot.map.identity,staticMutationRejected:true,workers:window.staticTestWorkers,inspected,frames,objectId,source_sha256:snapshot.source_sha256,renderer_identity:snapshot.renderer_identity})});
}catch(error){await fetch('/__static_receipt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ok:false,error:String(error),stack:error.stack})});}})();
</script>'''


def run(stage):
 stage=Path(stage);page=(stage/'index.html').read_bytes()
 worker=b'<script>window.staticTestWorkers=[];const StaticNativeWorker=window.Worker;window.Worker=class extends StaticNativeWorker{constructor(...args){super(...args);window.staticTestWorkers.push(String(args[0]));}};</script>'
 page=page.replace(b'<head>',b'<head>'+worker,1).replace(b'</body>',CHECK.encode()+b'</body>',1)
 received=[];done=threading.Event()
 with tempfile.TemporaryDirectory(prefix='living-static-rest-',ignore_cleanup_errors=True) as root:
  root=Path(root);web=root/'web';web.mkdir();(web/'index.html').write_bytes(page)
  (web/'boards.json').write_text('{"boards":[],"levels":{}}',encoding='utf8')
  sessions=srv.Sessions(web,None);sessions.lifecycle=SimpleNamespace(token='t'*40,restore_for=lambda token:None)
  sessions.access.lookup=lambda url:None
  sessions.accounts.ensure_user('static_phong_operator','temporary-local-fixture-917')
  authz.grant_role(sessions.accounts_db,'static_phong_operator','operator','operator:test')
  class Handler(srv.make_handler(sessions)):
   def do_POST(self):
    if self.path!='/__static_receipt':return super().do_POST()
    size=int(self.headers.get('Content-Length','0'))
    if not 0<size<8*1024*1024:self.send_error(400);return
    received.append(json.loads(self.rfile.read(size)));self.send_response(200);self.end_headers();done.set()
  httpd=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=httpd.serve_forever,daemon=True).start()
  browser=next(p for p in [r'C:\Program Files\Google\Chrome\Application\chrome.exe',r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'] if Path(p).exists())
  try:
   profile=root/'profile';process=subprocess.Popen([browser,'--headless=new','--no-first-run','--disable-extensions','--use-angle=swiftshader','--enable-unsafe-swiftshader',f'--user-data-dir={profile}',f'http://127.0.0.1:{httpd.server_port}/index.html'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   try:
    if not done.wait(50):raise TimeoutError('No static REST browser receipt')
   finally:process.terminate();process.wait(timeout=10)
  finally:httpd.shutdown();httpd.server_close();sessions.accounts._conn.close()
  assert not sessions.tables,'Static REST created game Tables'
 if not received or not received[0].get('ok'):raise RuntimeError(received)
 result=received[0];assert result['workers']==[]
 assert result['inspected']['state']=='complete'
 assert result['inspected']['result']['object']['identity']==result['objectId']
 assert all(key in result['inspected']['result']['object'] for key in ['extensions','material_bindings','physics','persistence'])
 summaries=[]
 for index,frame in enumerate(result['frames']):
  assert frame['state']=='complete';value=frame['result'];data=base64.b64decode(value['png_base64'])
  assert value['actual_camera']['position']==value['position']
  assert value['actual_camera']['width']==value['width'] and value['actual_camera']['height']==value['height']
  import math
  magnitude=math.hypot(*value['facing'])
  assert all(math.isclose(actual,requested/magnitude,rel_tol=1e-12,abs_tol=1e-12) for actual,requested in zip(value['actual_camera']['facing'],value['facing']))
  assert hashlib.sha256(data).hexdigest()==value['png_sha256']
  path=stage/f'rest-phong-{index}.png';path.write_bytes(data)
  summaries.append({k:v for k,v in value.items() if k!='png_base64'}|{'path':str(path),'bytes':len(data)})
 assert summaries[0]['png_sha256']!=summaries[1]['png_sha256']
 result['frames']=summaries
 (stage/'static-rest-browser-receipt.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 return result

if __name__=='__main__':print(json.dumps(run(ROOT/'build/dot-static-phong-rest-20261110'),indent=2))

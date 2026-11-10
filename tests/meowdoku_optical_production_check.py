"""Local-production optical photo API smoke; archives only its own new fixture object."""
import base64,hashlib,json,ssl,struct,subprocess,time,urllib.request,urllib.error,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    source=(ROOT/'tests/meowdoku_studio_optical_check.mjs').read_text().split('const originalJSON')[0]
    source=source.replace("'../examples/","'./examples/")
    source+='''\noriginal.name='Optical pipeline verification (temporary)';
original.design.optical_authoring={version:1,mode:'glass_rod',provenance:OPTICAL_PROVENANCE,materials:records};
const p=await prepareObjectSnapshot(original),f=[...p.geometry.interior,...p.geometry.glass];
const bytes=a=>Buffer.from(a.buffer).toString('base64');
console.log(JSON.stringify({value:p.value,v:bytes(new Float32Array(f.flatMap(x=>x.slice(0,3).flat()))),n:bytes(new Float32Array(f.flatMap(x=>x[4].flat()))),ids:bytes(new Uint32Array(f.map(x=>x[3])))}));'''
    data=json.loads(subprocess.run(['node','--input-type=module','-e',source],cwd=ROOT,check=True,capture_output=True,text=True,timeout=30).stdout)
    context=ssl.create_default_context(cafile=str(ROOT/'build/meowdoku/tls/ca.crt'))
    def request(path,body=None,kind='application/json'):
        req=urllib.request.Request('https://127.0.0.1:8766'+path,data=body,headers={'Content-Type':kind,'X-Meowdoku-Marble':'1'})
        with urllib.request.urlopen(req,context=context,timeout=90) as response:return response.status,json.load(response)
    status,created=request('/api/marble_objects',json.dumps(data['value']).encode());assert status==201
    object_digest=created['digest'];record={'object':object_digest,'new_object':not created['duplicate']}
    v=np.frombuffer(base64.b64decode(data['v']),dtype='<f4');n=np.frombuffer(base64.b64decode(data['n']),dtype='<f4');ids=np.frombuffer(base64.b64decode(data['ids']),dtype='<u4')
    base=json.loads((ROOT.parent/'spectral-analyzer/build/rod-bundle-worker-gate-optimized-20261109/config.json').read_text())
    config={k:base[k] for k in ('transport','camera','fixtures','max_bounces','bounce_mode')}
    config.update(object_digest=object_digest,exposure_seconds=.25,sphere_coverage='mesh')
    def wire(c):
        cfg=json.dumps(c).encode();return b'MRP1'+struct.pack('<III',len(cfg),len(ids),0)+cfg+v.tobytes()+n.tobytes()+ids.tobytes()
    try:
        # Optical LIVE is now admitted; this smoke remains a still-only test.
        # Dedicated isolated live tests verify progression and camera reload.
        status,started=request('/api/marble_photo',wire(config),'application/octet-stream');assert status==202
        digest=started['digest'];record['photo']=digest;begin=time.monotonic()
        while time.monotonic()-begin<180:
            _,state=request('/api/marble_photo?digest='+digest)
            if state['state'] in ('complete','error','canceled'):break
            time.sleep(1)
        record['status']=state;record['wait_seconds']=time.monotonic()-begin
        assert state['state']=='complete',state
        assert state['rod_bundle_diagnostic']['unresolved_samples']==0,state
        req=urllib.request.urlopen('https://127.0.0.1:8766/api/marble_photo/image?digest='+digest+'&name='+state['image'],context=context,timeout=20)
        image=req.read();assert image.startswith(b'\x89PNG')
        (out/'production-photo.png').write_bytes(image)
        record['image_bytes']=len(image)
    finally:
        # Never remove or modify a pre-existing object. Archive this test's
        # newly-created immutable fixture, preserving the full record locally.
        if not created['duplicate']:
            p=ROOT/'build/meowdoku/marble_objects'/f'{object_digest}.json'
            if p.is_file() and json.loads(p.read_text())['name']==data['value']['name']:
                shutil.move(str(p),str(out/f'{object_digest}.json'));record['test_object_archived']=True
        (out/'receipt.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({'photo':record['photo'],'state':state['state'],'unresolved_samples':state['rod_bundle_diagnostic']['unresolved_samples'],'wait_seconds':record['wait_seconds'],'test_object_archived':record.get('test_object_archived')},indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out.resolve())

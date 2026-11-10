from pathlib import Path
import argparse,hashlib,json,re,shutil
root=Path(__file__).resolve().parents[1];target=root/'build/meowdoku/web';source=root/'examples/meowdoku/web'
p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');args=p.parse_args()
stage=root/'build/dot-live-sharing-4k-20261110/ui';stage.mkdir(parents=True,exist_ok=True)
old={name:(target/name).read_bytes() for name in ['marble_photo.js','marble_studio.html','marble_studio.js','marble_live.js']}
photo=old['marble_photo.js'].decode('utf-8-sig')
assert 'clientIntersections: participation.requested()' in photo and 'participation.connect(value, options.clientIntersections)' in photo
photo,n=re.subn(r'Device intersections require an RGB still[^"\n]*', 'Device intersections require RGB transport, one frame, and a listed resolution.',photo);assert n==1
html=old['marble_studio.html'].decode('utf-8-sig');sourcehtml=(source/'marble_studio.html').read_text(encoding='utf-8')
for id in ['photo-resolution','live-resolution']:
 pattern='<select id="'+id+'">.*?</select>';replacement=re.search(pattern,sourcehtml,re.S).group();html,n=re.subn(pattern,lambda m:replacement,html,flags=re.S);assert n==1
# Update only the participation section, preserving all unrelated deployed controls.
start='    <label class="wide"><input id="photo-client-intersections"';end='    <button id="photo-start">'
a=sourcehtml.index(start);b=sourcehtml.index(end,a);x=html.index(start);y=html.index(end,x);html=html[:x]+sourcehtml[a:b]+html[y:]
start='    <label class="wide"><input type="checkbox" id="live-client-intersections"';end='    <output class="wide" id="live-status"'
a=sourcehtml.index(start);b=sourcehtml.index(end,a)
if 'id="live-client-intersections"' in html:
 x=html.index(start);y=html.index(end,x);html=html[:x]+sourcehtml[a:b]+html[y:]
else:assert html.count(end)==1;html=html.replace(end,sourcehtml[a:b]+end,1)
bootstrap=old['marble_studio.js'].decode('utf-8-sig');bootstrap=bootstrap.replace('import { mountClientRayDiagnostic } from "./client_ray_diagnostic.js";\n','')
if '    mountClientRayDiagnostic({getScene:()=>{' in bootstrap:
 a=bootstrap.index('    mountClientRayDiagnostic({getScene:()=>{');b=bootstrap.index('    frame();',a);bootstrap=bootstrap[:a]+bootstrap[b:]
assert 'mountClientRayDiagnostic' not in bootstrap
# This file's source/deployed diff was inspected: only cooperative Live hooks,
# all-size admission and device-limit error handling differ.
live=(source/'marble_live.js').read_text(encoding='utf-8')
changes={'marble_photo.js':photo,'marble_studio.html':html,'marble_studio.js':bootstrap,'marble_live.js':live}
for name,text in changes.items():(stage/name).write_text(text,encoding='utf-8')
modules=['marble_photo_participation.js','marble_live_participation.js','client_ray_participation.js','client_ray_settings.js','client_ray_diagnostic.js','client_ray_kernel.js','client_triangle_kernel.js','studio_bvh_compute.js','studio_triangle_compute.js','studio_slab_compute.js','studio_compiled_law_batch.js']
for name in modules:assert (source/name).is_file(),name
assert 'LIVE_CLIENT_INTERSECTIONS_READY = true' in (source/'marble_live_participation.js').read_text()
assert '&& !live' not in (source/'marble_photo_participation.js').read_text()
if args.publish:
 for name,data in old.items():assert (target/name).read_bytes()==data,'Concurrent edit: '+name
 for name,data in old.items():(stage/(name+'.before')).write_bytes(data)
 for name in modules:shutil.copy2(source/name,target/name)
 for name,text in changes.items():(target/name).write_text(text,encoding='utf-8')
receipts={name:hashlib.sha256((target/name if args.publish else stage/name).read_bytes()).hexdigest() for name in changes}
if args.publish:receipts.update({name:hashlib.sha256((target/name).read_bytes()).hexdigest() for name in modules})
(stage/'receipts.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
print('PUBLISHED' if args.publish else 'DRY RUN');print(json.dumps(receipts,indent=2))

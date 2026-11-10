from pathlib import Path
import hashlib, shutil
root=Path(__file__).resolve().parents[1]
source=root/'examples/meowdoku/web'
target=root/'build/meowdoku/web'
for dependency in ('marble_photo_participation.js','client_ray_participation.js','studio_bvh_compute.js'):
    assert (target/dependency).is_file(), 'Publish participation dependencies first: '+dependency
p=target/'marble_studio.html'; original=p.read_bytes(); text=original.decode('utf-8-sig')
section='    <details class="wide"><summary>My jobs</summary><section id="marble-jobs"></section></details>\n'
script='<script type="module" src="marble_jobs.js"></script>'
if 'id="marble-jobs"' not in text:
    needle='    <output id="photo-status"'
    assert text.count(needle)==1
    text=text.replace(needle, section+needle,1)
if script not in text:
    needle='<script type="module" src="marble_studio.js"></script>'
    assert text.count(needle)==1
    text=text.replace(needle,needle+script,1)
backup=root/'build/dot-jobs-ui-release-backup';backup.mkdir(exist_ok=True)
(backup/'marble_studio.before.html').write_bytes(original)
assert p.read_bytes()==original,'Concurrent Studio HTML edit; abort release'
for name in ('marble_jobs.js','marble_jobs.html'):
    shutil.copy2(source/name,target/name)
p.write_text(text,encoding='utf-8')
for name in ('marble_jobs.js','marble_jobs.html','marble_studio.html'):
    print(name,hashlib.sha256((target/name).read_bytes()).hexdigest())
print('Only jobs assets and scoped Studio HTML insertion published; no server or bootstrap changes')

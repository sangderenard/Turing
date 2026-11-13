import base64,importlib.util,json,os,subprocess,sys
from pathlib import Path
root=Path(os.environ['STUDIO_SPECTRAL_TEST_ROOT']);sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location('ancestry_fixture',root/'tests/test_studio_forward_ancestry_transport.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s,meta,rows=m.fixture();s.consume_forward(meta,rows);manifest,payload=s.snapshot()
subprocess.run(['node',str(Path(__file__).with_name('meowdoku_forward_ancestry_check.mjs'))],
 input=json.dumps(dict(manifest=manifest,base64=base64.b64encode(payload).decode(),words=rows.ravel().tolist())),text=True,check=True)

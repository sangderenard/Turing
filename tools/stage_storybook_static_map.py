"""Build the authored StorybookEnvironment through native static map producers."""
from pathlib import Path
import hashlib,json,sys,re
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'examples')]
from meowdoku.storybook_environment import build_storybook_world
from src.compiler.abstract_ui_div_map import project_world_to_div_map
from tools.add_living_data_static_host import patch_static_host


def stage_storybook(directory):
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
 ui=project_world_to_div_map(build_storybook_world(),title='The Lantern Road - a Living Data storybook',static_only=True)
 assert not set(ui.model)&{'entity_mezzanine','vehicle_slot','physics_program','loop_deployment','engine_audio','music_room'}
 assert all('MechanicalCreature' not in json.dumps(node) for node in ui.model['world']['objects'])
 source=ui.html.encode('utf8');digest=hashlib.sha256(source).hexdigest()
 (directory/'native-static-source.html').write_bytes(source)
 (directory/'model.json').write_text(json.dumps(ui.model,ensure_ascii=False,indent=2),encoding='utf8')
 output=source
 receipt={"schema":"living-data-static-projection.v1","source_sha256":re.search(r"const livingDataStaticSource = '([0-9a-f]{64})'",ui.html).group(1),
  "source_hash_scope":"native-javascript-packet-before-static-adapter","output_sha256":digest,"physics_started":False,"published":False}
 (directory/'index.html').write_bytes(output)
 receipt.update(world_identity=ui.model['identity'],objects=len(ui.model['world']['objects']),
  geometry_boxes=len(ui.model['document_geometry']['boxes']),producer='world_div_map_model(static_only=True)',
  class_source='examples/meowdoku/storybook_environment.py')
 (directory/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
 return receipt

if __name__=='__main__':print(json.dumps(stage_storybook(ROOT/'build/dot-storybook-static-phong-20261110'),indent=2),flush=True)

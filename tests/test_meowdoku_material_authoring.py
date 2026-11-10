from pathlib import Path
import copy,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
sys.path.insert(0,str(ROOT.parent/'spectral-analyzer'))
from meowdoku.marble_material_authoring import validate_material_authoring
from canonical_material_documents import read_material_documents

def test_all_native_documents_survive_immutable_contract():
    docs=read_material_documents()
    value={'version':1,'records':[{'material':i,'source':doc.snapshot(),'overrides':[]} for i,doc in enumerate(docs)]}
    before=copy.deepcopy(value);restored=validate_material_authoring(value,len(docs))
    assert value==before
    for i,doc in enumerate(docs):
        assert restored[i].source_bytes==doc.source_bytes
        assert restored[i].definition==doc.definition

def test_hash_mismatch_duplicate_id_and_bad_edit_rejected():
    doc=read_material_documents()[0]
    base={'version':1,'records':[{'material':0,'source':doc.snapshot(),'overrides':[]}]}
    for mutate in [lambda v:v['records'].append(copy.deepcopy(v['records'][0])),lambda v:v['records'][0]['source'].update(revision='0'*64),lambda v:v['records'][0]['overrides'].append({'path':['not-present'],'value':3})]:
        bad=copy.deepcopy(base);mutate(bad)
        with pytest.raises((ValueError,KeyError)):validate_material_authoring(bad,1)

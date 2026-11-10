"""Source-preserving export of the existing native material library for Studio.

This is a derived view, not another material registry. Unresolved source profile
references remain listed but unavailable; no fallback material is invented.
"""
from pathlib import Path
import sys
import numpy as np


def export_native_material_library(spectral_root=None):
    root=Path(spectral_root) if spectral_root else Path(__file__).resolve().parents[3]/'spectral-analyzer'
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from canonical_material_documents import read_material_documents,native_payload
    from material_db import MaterialDatabase
    from spectral_library import load_default_library
    load_default_library();db=MaterialDatabase();records=[]
    for document in read_material_documents(root/'configs'/'materials'):
        record={'key':document.key,'source':document.snapshot(),'definition':document.definition}
        record['profile_dependencies']={k:v for k,v in document.definition.items() if k in ('color_profile_name','emit_profile_name','emission_profile_name','remit_profile_name')}
        try:
            compiled=native_payload(document)
            record.update(native_compiles=True,native_index=db.register(document.key,compiled.payload),
                          uncompiled_fields=list(compiled.uncompiled_fields),warnings=list(compiled.warnings))
        except (ValueError,KeyError,TypeError) as error:
            record.update(native_compiles=False,error=str(error))
        records.append(record)
    tensors=db.build_tensors()
    chunks=('pbr','phong_compat','enamel','texture_stack','spectral','raymat_compat')
    for record in records:
        if record['native_compiles']:
            index=record.pop('native_index')
            if not all(np.isfinite(tensors[name][index]).all() for name in chunks):
                record.update(native_compiles=False,error='projected native material buffer is nonfinite');continue
            record['compiled']={name:tensors[name][index].tolist() for name in chunks}
            record['compiled']['n_bands']=int(tensors['n_bands'][index])
    return {'version':1,'authority':'spectral-analyzer/configs/materials',
            'projection_contract':'derived inspection view; profile indices are process-local and must be resolved from named source dependencies before runtime use',
            'self_contained_runtime_buffers':False,'records':records}


def export_named_emitter_profiles(spectral_root=None):
    root=Path(spectral_root) if spectral_root else Path(__file__).resolve().parents[3]/'spectral-analyzer'
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from camera_designer.studio_named_emitters import export_named_emitters
    return export_named_emitters()

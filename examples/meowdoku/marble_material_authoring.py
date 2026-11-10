"""Validation for source-preserving canonical material references in Studio saves.

Packed tables remain required for older clients. Source records preserve the
original disk profile and targeted edits, and are compiled by the native adapter.
"""
from __future__ import annotations
import sys
from pathlib import Path

def validate_material_authoring(value, material_count, spectral_root=None):
    if not isinstance(value,dict) or set(value)!={'version','records'} or value['version']!=1:
        raise ValueError('invalid material authoring contract')
    if not isinstance(value['records'],list) or len(value['records'])>128:
        raise ValueError('invalid material authoring records')
    root=Path(spectral_root) if spectral_root else Path(__file__).resolve().parents[3]/'spectral-analyzer'
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from canonical_material_documents import restore_material_document
    seen=set();documents={}
    for record in value['records']:
        if not isinstance(record,dict) or set(record)!={'material','source','overrides'}:
            raise ValueError('invalid material source record')
        mid=record['material']
        if type(mid) is not int or not 0<=mid<material_count or mid in seen:
            raise ValueError('invalid or repeated material source ID')
        if not isinstance(record['overrides'],list) or len(record['overrides'])>256:
            raise ValueError('too many material overrides')
        source=record['source']
        if not isinstance(source,dict) or not isinstance(source.get('source_yaml'),str) or len(source['source_yaml'].encode('utf-8'))>65536:
            raise ValueError('material source exceeds document limit')
        documents[mid]=restore_material_document(source,record['overrides']);seen.add(mid)
    return documents

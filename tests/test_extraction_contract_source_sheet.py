"""Fresh project workers retain the audit's exact source declarations."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml
from src.compiler.extraction_contract import ExtractionContract


def test_sheet_sources_round_trip_in_fresh_worker(tmp_path):
    root = Path(__file__).resolve().parents[1]
    original = root / 'extraction_contracts/program_extraction.yaml'
    raw = yaml.safe_load(original.read_text())
    raw['roots'] = {'authored': [], 'repository': []}
    module = tmp_path / 'external_admitted.py'
    module.write_text('def value():\n    return 7\n')
    raw['sources'] = [{'name': 'external_admitted', 'path': module.name}]
    sheet = tmp_path / 'policy.yaml'
    sheet.write_text(yaml.safe_dump(raw))
    policy = ExtractionContract(sheet)
    receipt = [{'name': 'external_admitted', 'path': str(module.resolve())}]
    assert policy.sources_receipt() == receipt
    digest = hashlib.sha256(sheet.read_text().encode()).hexdigest()
    canonical = json.dumps(receipt, sort_keys=True, separators=(',', ':'))
    assert policy.fingerprint == hashlib.sha256(f'{digest}:sources:{canonical}'.encode()).hexdigest()
    script = '''import importlib,json,sys
from src.compiler.extraction_contract import ExtractionContract
sys.path.insert(0, sys.argv[2])
p=ExtractionContract(sys.argv[1])
f=importlib.import_module('external_admitted').value
d=p.decide(f)
print(json.dumps([p.sources_receipt(),p.fingerprint,d.subject.classification,d.ingest_parent]))
'''
    result = subprocess.run([sys.executable, '-c', script, str(sheet), str(tmp_path)],
                            cwd=root, text=True, capture_output=True, check=True)
    fresh = json.loads(result.stdout.strip().splitlines()[-1])
    assert fresh == [receipt, policy.fingerprint, 'authored_python', True]
    derived = policy.with_program_abi(policy.program_abi.receipt())
    assert derived.sources_receipt() == receipt

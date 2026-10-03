"""Authored return occurrences are distinct from the object they return."""

import pickle
import subprocess
import sys

import pytest

from tools.compiler_probes import probe_record_return_merge as probe
from src.compiler.identity_concordance import CorrelationTable, concordance_report
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.fixture(scope='module')
def lowered():
    return probe.lower()


def test_same_object_returns_keep_three_control_edges(lowered):
    module, _, _ = lowered
    root = probe.root_function(module)
    edges = [block.instrs[-1] for block in root.blocks.values()
             if block.instrs and 'return_source_value_ids' in
             (block.instrs[-1].attributes or {})]
    assert len(edges) == 3, 'One authored return was replaced by a loop backedge'
    assert len({edge.attributes['return_site_cell'] for edge in edges}) == 3
    assert len({edge.attributes['return_source_value_ids'] for edge in edges}) == 1


def test_unwritten_return_field_keeps_incoming_formal(lowered):
    module, _, _ = lowered
    root = probe.root_function(module)
    inputs = [arg for arg in root.args if
              (arg.accounting or {}).get('program_abi_field') == 'hard_failure']
    assert len(inputs) == 1, 'The input flag was replaced by another return site write'
    flag_phis = [op for block in root.blocks.values() for op in block.instrs
                 if op.op == 'Phi' and
                 (op.attributes or {}).get('record_field') == 'hard_failure' and
                 (op.attributes or {}).get('binding') == 'return_merge']
    assert len(flag_phis) == 1
    assert sum(arg.id == inputs[0].id for arg in flag_phis[0].args) == 2


def _assert_native_returns(lowered, tmp_path, optimization="O0"):
    module, outputs, _ = lowered
    root = probe.root_function(module)
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / 'native', optimization=optimization)
    payload = tmp_path / 'return-sites.pkl'
    payload.write_bytes(pickle.dumps((artifact, root, outputs[root.name], probe.SOURCE)))
    result = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
artifact, root, outputs, source = pickle.load(open(sys.argv[1], 'rb'))
namespace = {}
exec(source, namespace)
named = dict(root.metadata['parameter_names'])
definitions = {op.res.id: op for block in root.blocks.values()
               for op in block.instrs if op.res is not None}
for initial in (False, True):
    for rejected, value in ((True, 2.0), (True, .5), (False, 2.0), (False, .5)):
        expected = namespace['step'](namespace['Metrics'](initial, value), rejected)
        feeds = {}
        for arg in root.args:
            field = (arg.accounting or {}).get('program_abi_field')
            if field == 'hard_failure':
                feeds[arg.id] = np.asarray(initial, dtype=np.bool_)
            elif field == 'value':
                feeds[arg.id] = np.asarray(value, dtype=np.float64)
            elif arg.id == named['rejected']:
                feeds[arg.id] = np.asarray(rejected, dtype=np.bool_)
        execution = artifact.prepare_execution(feeds).run()
        actual = {}
        for output in outputs:
            definition = definitions.get(output.id)
            field = ((definition.attributes or {}).get('record_field') if definition else None)
            field = field or (output.accounting or {}).get('program_abi_field')
            actual[field] = execution.buffers[output.id].item()
        assert actual == {'hard_failure': expected.hard_failure, 'value': expected.value}, (
            initial, rejected, value, actual, expected)
        print(initial, rejected, value, actual, flush=True)
''', str(payload)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_return_site_receipts_survive_pickle_and_republication(lowered):
    from src.compiler.ssa_record_return_state import publish_scalar_record_return_fields

    module = pickle.loads(pickle.dumps(lowered[0]))
    root = probe.root_function(module)
    before = [(op.res.id, tuple(arg.id for arg in op.args))
              for block in root.blocks.values() for op in block.instrs if op.op == 'Phi']
    assert publish_scalar_record_return_fields(module) == 0
    assert publish_scalar_record_return_fields(module) == 0
    assert before == [(op.res.id, tuple(arg.id for arg in op.args))
                      for block in root.blocks.values() for op in block.instrs if op.op == 'Phi']
    assert not [finding for finding in CorrelationTable.build(module).findings(module)
                if finding.kind == 'return-site-edge-disagreement']


@pytest.mark.parametrize('corruption', ['missing-site', 'wrong-slots'])
def test_concordance_detects_return_site_disagreement(lowered, corruption):
    module = pickle.loads(pickle.dumps(lowered[0]))
    root = probe.root_function(module)
    edge = next(block.instrs[-1] for block in root.blocks.values()
                if block.instrs and 'return_source_value_ids' in block.instrs[-1].attributes)
    if corruption == 'missing-site':
        edge.attributes.pop('return_site_cell')
    else:
        edge.attributes['return_source_value_ids'] = (987654321,)
    findings = CorrelationTable.build(module).findings(module)
    assert len([finding for finding in findings
                if finding.kind == 'return-site-edge-disagreement']) == 1


@pytest.mark.parametrize("optimization", ["O0", "O2"])
def test_all_three_return_sites_match_native_python(lowered, tmp_path, optimization):
    _assert_native_returns(lowered, tmp_path, optimization)


RETURN_VARIANTS = {
    'nested_loop': '''
from dataclasses import dataclass
@dataclass
class Metrics:
    hard_failure: bool = False
    value: float = 0.0

def step(m: Metrics, rejected: bool) -> Metrics:
    while True:
        while True:
            if rejected:
                m.hard_failure = True
                return m
            if m.value > 1.0:
                m.value = m.value * 0.25
                return m
            return m
''',
    'for_loop': '''
from dataclasses import dataclass
@dataclass
class Metrics:
    hard_failure: bool = False
    value: float = 0.0

def step(m: Metrics, rejected: bool) -> Metrics:
    for turn in range(2):
        if rejected:
            m.hard_failure = True
            return m
        if m.value > 1.0:
            m.value = m.value * 0.25
            return m
        return m
    return m
''',
    'zero_trip': '''
from dataclasses import dataclass
@dataclass
class Metrics:
    hard_failure: bool = False
    value: float = 0.0

def step(m: Metrics, rejected: bool) -> Metrics:
    while rejected:
        if m.value > 1.0:
            m.hard_failure = True
            return m
        return m
    return m
''',
    'else_arm': '''
from dataclasses import dataclass
@dataclass
class Metrics:
    hard_failure: bool = False
    value: float = 0.0

def step(m: Metrics, rejected: bool) -> Metrics:
    while True:
        if rejected:
            m.hard_failure = True
            return m
        else:
            if m.value > 1.0:
                m.value = m.value * 0.25
                return m
            return m
''',
    'alias_receiver': '''
from dataclasses import dataclass
@dataclass
class Metrics:
    hard_failure: bool = False
    value: float = 0.0

def step(m: Metrics, rejected: bool) -> Metrics:
    result = m
    while True:
        if rejected:
            result.hard_failure = True
            return result
        if result.value > 1.0:
            result.value = result.value * 0.25
            return result
        return result
''',
    'static_branch': '''
from dataclasses import dataclass
@dataclass
class Metrics:
    hard_failure: bool = False
    value: float = 0.0

def step(m: Metrics, rejected: bool) -> Metrics:
    while True:
        if False:
            m.hard_failure = True
            return m
        if rejected:
            m.value = m.value * 0.25
            return m
        return m
''',
    'field_reassignment': '''
from dataclasses import dataclass
@dataclass
class Metrics:
    hard_failure: bool = False
    value: float = 0.0

def step(m: Metrics, rejected: bool) -> Metrics:
    while True:
        if rejected:
            m.hard_failure = True
            return m
        if m.value > 1.0:
            m.hard_failure = False
            m.value = m.value * 0.25
            return m
        return m
''',
}

@pytest.mark.parametrize('variant', [
    pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=(
        'HELD (port onto 854e145c): a declared scalar field with no live '
        'read or write is not a record output on this head (same as a field '
        'never mentioned). The proposal projected every declared scalar '
        'parameter field at entry to make it one; that ABI change was not '
        'ported. See docs/concordance_census/CONTINUATION_return_site_port.md.'
    ))) if name == 'static_branch' else name
    for name in RETURN_VARIANTS
])
def test_nested_and_fallthrough_returns_match_native_python(variant, tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'SOURCE', RETURN_VARIANTS[variant])
    _assert_native_returns(probe.lower(), tmp_path)


def test_each_live_return_field_has_a_sourced_exact_selection(lowered):
    """Every live selection is a sourced cell: the site's posted field
    version, or -- at a site that records no write of the field -- the
    entered version, which is the field's ProgramABI formal (the scope
    ladder; ``scalar_return_field_versions.entered_version``)."""
    from src.compiler.concordance_declarations import (
        RECORD_RETURN_FIELD_SELECTION, SSA_FIELD_VERSION, SSA_VALUE,
    )
    from src.compiler.identity_concordance import Ref, identity_book

    module = lowered[0]
    root = probe.root_function(module)
    live = {block.name for block in root.blocks.values() if block.instrs
            and 'return_site_cell' in block.instrs[-1].attributes}
    book = identity_book(module)
    page = book.pages[RECORD_RETURN_FIELD_SELECTION.name]
    rows = [row for row in page.rows() if row[-1] in live]
    assert len(rows) == 6
    for row in rows:
        selection = page.latest(row)
        assert isinstance(selection, Ref)
        assert selection.page == SSA_FIELD_VERSION or (
            selection.page == SSA_VALUE
            and int(selection.row[1]) in {int(arg.id) for arg in root.args}
        ), selection
        selected_by = book.latest_ref(RECORD_RETURN_FIELD_SELECTION, row)
        assert book.edges_into(selected_by)

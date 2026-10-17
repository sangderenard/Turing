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

RETURN_VARIANTS['unmentioned_field'] = RETURN_VARIANTS['static_branch'].replace(
    '        if False:\n            m.hard_failure = True\n            return m\n', '',
)

@pytest.mark.parametrize('variant', RETURN_VARIANTS)
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


EARLY_RETURN_SOURCE = '''
def hex_tokens(value: int) -> list[int]:
    result: list[int] = []
    if value == 0:
        result.append(48)
        return result
    if value < 0:
        result.append(45)
        return result
    result.append(49)
    return result


def root(value: int) -> list[int]:
    return hex_tokens(value)
'''


def test_return_sites_cross_the_worker_boundary_with_their_cells(tmp_path):
    """``prefixed_lower_hex_integer_glyph_tokens`` cut to seconds: three
    ``return result`` sites, two of them early returns inside ``if`` arms,
    compiled as one planned unit and linked by another (the census's SSA
    composition).  The worker receives the reducer's receipts as plain
    span-keyed dicts and the ``Name`` occurrence nodes that carried
    ``return_site_cell`` are gone, so the site identity must be the graph's
    own ``return_site_spans`` record and the site rows must be in the scoped
    input's book: every early-return edge then names its own site and the
    linking unit's audit (``_return_site_edge_findings``) accepts the
    product instead of refusing it with ``return-site-edge-disagreement``."""
    from src.compiler import project_compilation_product as product
    from src.compiler.concordance_declarations import RETURN_SITE_SLOT
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.identity_concordance import Ref

    contract = ExtractionContract(
        "extraction_contracts/program_extraction.yaml",
    ).with_program_abi({
        "values": [{"function": name, "parameter": "value", "storage": "scalar",
                    "dtype": "int64", "rank": 0, "shape": [],
                    "python_type": "builtins.int"}
                   for name in ("hex_tokens", "root")],
        "records": {}, "bindings": [],
    })
    planned = product.compile_partitioned_source(
        EARLY_RETURN_SOURCE, tmp_path / "growth", extraction_contract=contract,
        ssa_composition=True, plan_only=True, max_unit_source_nodes=64,
    )
    manifest = product.compile_resolved_process_graph_plan(
        planned["resolved_process_graph"], planned["process_graph_unit_plan"],
        tmp_path / "growth/product", jobs=1,
        worker_resident_reservation_bytes=256 << 20,
    )
    units = {unit["qualified_names"][0]: unit for unit in manifest["units"]}
    assert units["hex_tokens"]["status"] == "compiled-unverified", units["hex_tokens"]
    assert "return-site-edge-disagreement" not in str(units["root"].get("error", ""))
    module, _outputs, _exports = product._load_pickle_artifact(
        tmp_path / "growth/product/units"
        / f'unit_{units["hex_tokens"]["unit_index"]:03d}' / units["hex_tokens"]["artifact"]
    )
    assert CorrelationTable.build(module).findings(module) == []
    function = module.functions["hex_tokens__hex_tokens"]
    edges = [block.instrs[-1].attributes for block in function.blocks.values()
             if block.instrs and "return_source_value_ids" in (block.instrs[-1].attributes or {})]
    assert len(edges) == 2
    sites = [edge.get("return_site_cell") for edge in edges]
    assert all(isinstance(site, Ref) for site in sites) and len(set(sites)) == 2
    scope = tuple(function.metadata["record_return_state_scope"])
    rows = module.metadata["identity_book"].pages[RETURN_SITE_SLOT.name].scope_rows(scope)
    assert {row[1] for row in rows} >= set(sites)

"""Retained If ownership survives numerical graph folding."""

import pytest
from types import SimpleNamespace

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    Arm,
    CANONICAL_VALUE,
    CONTROL_BLOCK,
    CONTROL_BLOCK_PLACEMENT,
    DEPLOYMENT_REGION_MEMBER,
)
from src.compiler.control_activity import binding, conditional_memberships
from src.compiler.control_source import ConditionalBlock, StatementBlock
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, current_identity_book
import src.compiler.glsl_deployment_strategy as glsl


SOURCE = '''def pair(gram: AbstractTensor, octets: AbstractTensor):
    return gram * 0 + 1, gram * 0

def root(octets: AbstractTensor, grams: int):
    token = octets * 0 + 1
    match_length = octets * 0
    match_distance = octets * 0
    if grams > 1:
        gram = octets[:grams] * 0
        for offset in range(2):
            gram = gram * 256 + octets[offset:offset + grams]
        length, distance = pair(gram, octets)
        has = (length > 0).to_dtype("int64")
        pad = octets[grams:] * 0
        token = AbstractTensor.cat((length * has + (1 - has), pad + 1), dim=0)
        match_length = AbstractTensor.cat((length, pad), dim=0)
        match_distance = AbstractTensor.cat((distance, pad), dim=0)
    for attempt in range(2):
        token = token + match_length * 0
        match_length = match_length + match_distance * 0
        match_distance = match_distance + token * 0
    return (token + match_length + match_distance).sum().item()
'''


class _Captured(BaseException):
    pass


def _real_control_slice(monkeypatch):
    original = glsl._overlay_control_or_require_subdivision
    captured = {}

    def capture(*args, **kwargs):
        program = original(*args, **kwargs)
        if not args[1]:
            return program
        graph = args[0]
        controls = tuple(
            block for block in program.root.blocks
            if isinstance(block, ConditionalBlock) and block.source_node_id is not None
        )
        if controls:
            captured.update(graph=graph, program=program, control=controls[0],
                            book=current_identity_book())
            raise _Captured
        return program

    monkeypatch.setattr(glsl, '_overlay_control_or_require_subdivision', capture)
    contract = ExtractionContract(
        'extraction_contracts/program_extraction.yaml'
    ).with_program_abi({
        'values': [
            {'function': 'root', 'parameter': 'octets', 'storage': 'span',
             'dtype': 'int64', 'rank': 1,
             'python_type': 'src.common.tensors.abstraction.AbstractTensor'},
            {'function': 'root', 'parameter': 'grams', 'storage': 'scalar',
             'dtype': 'int64', 'rank': 0, 'python_type': 'builtins.int'},
        ], 'records': {}, 'bindings': [],
    })
    with pytest.raises(_Captured):
        lower_ast_source_to_ssa(
            SOURCE, 'root', name='retained_control_activity_placement',
            python_bindings={'AbstractTensor': AbstractTensor},
            extraction_contract=contract,
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    return captured


def test_retired_if_owner_places_three_cat_regions_in_body_and_rejects_forged_owner(monkeypatch):
    specimen = _real_control_slice(monkeypatch)
    graph, program, control, book = (
        specimen[key] for key in ('graph', 'program', 'control', 'book')
    )
    scope = graph.G.graph['lexical_read_scope']
    planning_scope = graph.G.graph['planning_scope']
    owner_id = int(control.source_node_id)
    assert owner_id not in graph.G
    owner = book.latest_ref(CANONICAL_VALUE, (scope, owner_id))
    assert owner is not None
    control_rows = tuple(
        row for row in book.page(CONTROL_BLOCK).rows()
        if row[0] == scope and row[1].value == 'conditional'
        and book.page(CONTROL_BLOCK).latest(row).predicate
        == book.latest_ref(CANONICAL_VALUE, (scope, control.predicate_value_id))
    )
    assert len(control_rows) == 1
    control_cell = book.latest_ref(CONTROL_BLOCK, control_rows[0])
    body_regions = {
        int(line[len('__scheduled_region_'):-2])
        for block in control.body.blocks if isinstance(block, StatementBlock)
        for line in block.lines if line.startswith('__scheduled_region_')
    }
    cat_nodes = {
        int(node) for node, data in graph.G.nodes(data=True)
        if str(data.get('type')).casefold() == 'cat'
        and (owner_id, 'body') in conditional_memberships(graph, book=book).get(int(node), ())
    }
    assert len(cat_nodes) == 3
    assert {int(arm) for arm, _other, _initial, _merged in control.carried_aliases} == cat_nodes
    members = book.page(DEPLOYMENT_REGION_MEMBER)
    cat_regions = {
        int(row[1]) for row in members.rows()
        if row[0] == planning_scope and row[2] in cat_nodes
    }
    assert cat_regions <= body_regions
    assert len(cat_regions) == 3
    placements = book.page(CONTROL_BLOCK_PLACEMENT)
    checked_placements = set()
    for row in placements.rows():
        block = row[1]
        if row[0] != scope or block.row[1].value != 'statement':
            continue
        region = block.row[2]
        if region.page.name != 'deployment_region' or region.row[1] not in cat_regions:
            continue
        fact = placements.latest(row)
        assert fact.parent == control_cell and fact.arm is Arm.BODY
        placement_cell = book.latest_ref(CONTROL_BLOCK_PLACEMENT, row)
        direct = {source for source, _stage in book.edges_into(placement_cell)}
        assert owner in direct
        for node in cat_nodes:
            member = book.latest_ref(
                DEPLOYMENT_REGION_MEMBER, (planning_scope, region.row[1], node),
            )
            if member is None:
                continue
            _points, activity = binding(graph, node, book=book)
            assert member in direct and activity in direct
        checked_placements.add(int(region.row[1]))
    assert checked_placements == cat_regions
    audit_module = SimpleNamespace(metadata={'identity_book': book})
    audit_findings = [finding for finding in CorrelationTable._control_activity_findings(
        audit_module,
    ) if finding.kind == 'control-arm-placement']
    assert not audit_findings

    target_placement = next(
        book.latest_ref(CONTROL_BLOCK_PLACEMENT, row)
        for row in placements.rows()
        if row[0] == scope and row[1].row[1].value == 'statement'
        and row[1].row[2].page.name == 'deployment_region'
        and row[1].row[2].row[1] in cat_regions
    )
    original_edges = type(book).edges_into
    missing_activity = next(
        source for source, _stage in original_edges(book, target_placement)
        if source.page.name == 'source_activity_binding'
    )
    with monkeypatch.context() as patch:
        patch.setattr(type(book), 'edges_into',
                      lambda self, ref: tuple(
                          (source, stage) for source, stage in original_edges(self, ref)
                          if source != missing_activity
                      ) if self is book and ref == target_placement
                      else original_edges(self, ref))
        assert any(finding.kind == 'control-arm-placement'
                   for finding in CorrelationTable._control_activity_findings(
                       audit_module,
                   ))

    # The same public graph remains live. Poison only the exact retained
    # canonical ancestry read by the incumbent activity projector.
    original_edges = type(book).edges_into
    with monkeypatch.context() as patch:
        patch.setattr(type(book), 'edges_into',
                      lambda self, ref: () if self is book and ref == owner
                      else original_edges(self, ref))
        with pytest.raises(ValueError, match='lacks activity binding'):
            conditional_memberships(graph, book=book)

    foreign = book.latest_ref(CANONICAL_VALUE, (scope, next(iter(cat_nodes))))
    foreign_causes = original_edges(book, foreign)
    with monkeypatch.context() as patch:
        patch.setattr(type(book), 'edges_into',
                      lambda self, ref: foreign_causes if self is book and ref == owner
                      else original_edges(self, ref))
        with pytest.raises(ValueError, match='disagrees with activity kind'):
            conditional_memberships(graph, book=book)

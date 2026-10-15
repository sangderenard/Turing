"""Representation-neutral tests shared by Python and whole native core probes.

These exercise the real book writer and storage transitions. They deliberately
include values a narrowed integer span, object-pointer slab or tensor layout
must represent exactly, or explicitly decline before changing the book.
"""
import copy
import pickle

import pytest

from src.compiler import identity_concordance as core
from src.compiler.concordance_declarations import ScopeFork, SCOPE_ORIGIN
from src.compiler.live_tiles import LiveTileArena, _ACTIVE_ARENA


# Protocol-4 archives emitted by the untouched dict-backed implementation.
_LEGACY_AUDIT_PICKLES = {
    "Claim": "80049553000000000000008c217372632e636f6d70696c65722e6964656e746974795f636f6e636f7264616e6365948c05436c61696d9493942981947d94288c046b696e649468058c036b65799468068c06736f7572636594680775622e",
    "ValueRow": "80049590000000000000008c217372632e636f6d70696c65722e6964656e746974795f636f6e636f7264616e6365948c0856616c7565526f779493942981947d94288c0866756e6374696f6e948c0166948c0876616c75655f6964944b018c056474797065944e8c0969735f666f726d616c94898c06636c61696d73945d948c0b646566696e6974696f6e73945d948c0475736573945d9475622e",
    "Finding": "80049569000000000000008c217372632e636f6d70696c65722e6964656e746974795f636f6e636f7264616e6365948c0746696e64696e679493942981947d94288c046b696e649468058c0866756e6374696f6e948c0166948c0876616c75655f6964944b018c0664657461696c94680975622e",
}


@pytest.fixture(params=("python", "compressed", "spill", "live-ram", "live-disk"))
def storage(request, tmp_path, monkeypatch):
    arena = None
    token = None
    if request.param.startswith("live"):
        arena = LiveTileArena(tile_rows=4, bucket_rows=4, hot_tiles=1, directory=tmp_path)
        token = _ACTIVE_ARENA.set(arena)
    monkeypatch.setenv("TURING_IDENTITY_SPILL_DIR", str(tmp_path))
    registry = core.Registry()
    declared = registry.declare_page("storage_contract", (
        core.RowField("scope", core.RowFieldKind.SCOPE),
        core.RowField("identity", core.RowFieldKind.VALUE_ID)), object)
    origin = registry.declare_page(SCOPE_ORIGIN.name, SCOPE_ORIGIN.row_fields, SCOPE_ORIGIN.fact_type)
    stage = registry.declare_stage("storage_contract")
    registry.declare_stage("read_scope_fork")
    transform = registry.declare_transform("storage_contract_root", 0)
    book = core.IdentityBook(registry=registry)

    def transition():
        if request.param == "compressed":
            result = book.compress_pages((declared.name,), tile_rows=4)
            assert declared.name in result.compressed, result
        elif request.param == "spill":
            result = book.spill_pages((declared.name,))
            assert book.page(declared).spilled, result
        elif request.param == "live-ram":
            arena.compress_resident_ram()
        elif request.param == "live-disk":
            arena.offload_to_disk()

    try:
        yield book, declared, origin, stage, transform, transition
    finally:
        if token is not None:
            _ACTIVE_ARENA.reset(token)


def post(book, page, row, fact, stage, transform, sources=(), mode=None):
    return book.post(page, row, fact, stage=stage,
                     provenance=core.Derived(sources) if sources else core.Novel(transform, ()),
                     mode=core.Mode.CONCORD if mode is None else mode)


def test_typed_storage_retains_fork_snapshot_and_exact_registry_leaves(storage):
    book, declared, origin, stage, transform, transition = storage
    # The incumbent clock is an exact Python integer. A native span cannot
    # silently wrap it just because its common-case storage uses uint64.
    book.clock[0] = (1 << 100) + 3
    source = post(book, declared, ("origin", 0), "before", stage, transform)
    wide = (1 << 100) + 17
    fact = (False, -wide, wide, None, "λ", b"\0\xff", 1.25)
    child = post(book, declared, ("origin", wide), fact, stage, transform, (source,))
    marker = post(book, origin, ("fork",), ScopeFork("origin", "contract", copy_on_read=True),
                  stage, transform, (source,))
    book.register_read_through("fork", "origin", marker)
    source_stamp, child_stamp = book.stamp_of(source), book.stamp_of(child)
    transition()
    page = book.page(declared)
    assert page.clock is book.clock
    assert page.latest(("origin", wide)) == fact
    assert page.stamp_at(source.row, source.column) == source_stamp
    assert page.stamp_at(child.row, child.column) == child_stamp
    ((predecessor, edge_stage),) = book.edges_into(child)
    assert predecessor == source and predecessor.page is declared
    assert edge_stage is stage
    assert (child, stage) in book.edges_out_of(source)
    assert book.mint_of(source)[0] is transform
    assert page.is_virtual(("fork", 0))
    assert page.latest(("fork", 0)) == "before"
    assert page.history(("fork", 0)) == ((0, "before"),)
    before = book.clock[0]
    updated = post(book, declared, source.row, "after", stage, transform, (child,), core.Mode.REVISE)
    assert book.clock[0] == before + 1
    assert page.history(source.row) == ((0, "before"), (1, "after"))
    assert page.latest(("fork", 0)) == "before"
    forked = post(book, declared, ("fork", 0), "local", stage, transform, (updated,), core.Mode.REVISE)
    assert page.history(forked.row) == ((0, "before"), (1, "local"))
    ((inherited, inherited_stage),) = book.edges_into(core.Ref(declared, forked.row, 0))
    assert inherited == source and inherited.page is declared
    assert inherited_stage.name == "read_scope_fork"
    restored = pickle.loads(pickle.dumps(book))
    restored_page = restored.registry.pages[declared.name]
    assert restored.page(restored_page).history(forked.row) == page.history(forked.row)
    assert restored.page(restored_page).clock is restored.clock
    assert restored.edges_into(child)[0][0].page is restored_page


def test_integer_layout_does_not_narrow_or_sort_authored_columns(storage):
    book, declared, _origin, stage, transform, transition = storage
    page = book.page(declared)
    columns = (-9, (1 << 100) + 1, 4)
    for column in columns:
        page.set(("scope", -((1 << 100) + 1)), column, column)
    transition()
    assert page.history(("scope", -((1 << 100) + 1))) == tuple((value, value) for value in columns)
    assert page.latest_column(("scope", -((1 << 100) + 1))) == 4
    assert page.latest(("scope", -((1 << 100) + 1))) == 4
    book.latch = core.Latch.CLOSED
    with pytest.raises(core.ConcordanceRefusal):
        page.set(("scope", 3), 0, 1)
    with pytest.raises(core.ConcordanceRefusal):
        post(book, declared, ("scope", True), 1, stage, transform)


@pytest.mark.parametrize("record", (
    core.Claim("kind", "key", "source"), core.ValueRow("f", 1),
    core.Finding("kind", "f", 1, "detail"),
))
def test_fixed_field_audit_records_keep_copy_and_legacy_archive_semantics(record):
    assert copy.copy(record) == record
    assert copy.deepcopy(record) == record
    assert pickle.loads(pickle.dumps(record)) == record
    assert pickle.loads(bytes.fromhex(_LEGACY_AUDIT_PICKLES[type(record).__name__])) == record
    from dataclasses import fields
    state = {field.name: getattr(record, field.name) for field in fields(record)}
    restored = copy.copy(record)
    restored.__setstate__(state)
    assert restored == record
    if isinstance(record, core.ValueRow):
        claim = core.Claim("alias", "1", "contract")
        restored.claims.append(claim)
        assert record.claims[-1] is claim  # same lists; no reconstructed writer
        record.claims.pop()


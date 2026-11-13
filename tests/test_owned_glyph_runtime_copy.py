from dataclasses import replace
import numpy as np
import pytest
from src.compiler.runtime_abi import OwnedGlyphSequenceCopy, copy_owned_sequence_results
from src.compiler.string_table import string_token


def fixture():
    # Physical padding makes even malformed metadata probes non-dangerous.
    handles = np.array([2, 0, 1, 2, 0, 0, 0, 0], dtype=np.int64)
    lengths = np.array([2, 0, 1, 0, 0, 0, 0, 0], dtype=np.int64)
    glyphs = np.zeros(32, dtype=np.int64)
    glyphs[:2] = [string_token('é'), string_token('a')]
    glyphs[8] = string_token('🙂')
    owner = np.zeros(1, dtype=np.int64)
    values = [handles.ctypes.data, 4, 4, owner.ctypes.data,
              glyphs.ctypes.data, lengths.ctypes.data, 16, 4]
    buffers = {i: np.array([v], dtype=np.uintp if i in (0,3,4,5) else np.int64)
               for i,v in enumerate(values)}
    return buffers, (handles, lengths, glyphs, owner), OwnedGlyphSequenceCopy(*range(8))


def copy(buffers, spec):
    return copy_owned_sequence_results(buffers, [(0,1,2,3,'int64')], glyph_sequences=(spec,))


def test_deep_glyph_rows_reorder_repeat_empty_and_source_overwrite():
    buffers, backing, spec = fixture()
    result = copy(buffers, spec)
    assert isinstance(result[0], tuple)
    expected = [[string_token('🙂')], [string_token('é'),string_token('a')], [], [string_token('🙂')]]
    assert [a.tolist() for a in result[0]] == expected
    for a in backing: a.fill(0)
    assert [a.tolist() for a in result[0]] == expected
    result[0][0][0] = 123
    assert result[0][3][0] == string_token('🙂')
    buffers[1][0] = 0
    assert result[1][0] == 4
    assert all(a.dtype == np.int64 for a in result[0])


def test_primitive_default_unchanged():
    buffers, backing, spec = fixture()
    result = copy_owned_sequence_results(buffers, [(0,1,2,3,'int64')])
    assert isinstance(result[0], np.ndarray)
    assert result[0].tolist() == [2,0,1,2]


def test_empty_pool_requires_no_native_read(monkeypatch):
    buffers = {i: np.zeros(1,dtype=np.int64) for i in range(8)}
    monkeypatch.setattr(np.ctypeslib, 'as_array', lambda *a,**k: pytest.fail('no native read'))
    assert copy(buffers,OwnedGlyphSequenceCopy(*range(8)))[0] == ()


@pytest.mark.parametrize('cell,value', [(1,-1),(1,5),(2,-1),(6,-1),(7,-1),(6,15)])
def test_invalid_scalar_metadata_refuses_before_array_read(monkeypatch,cell,value):
    buffers, backing, spec = fixture()
    buffers[cell][0] = value
    monkeypatch.setattr(np.ctypeslib, 'as_array', lambda *a,**k: pytest.fail('must validate first'))
    with pytest.raises(ValueError): copy(buffers,spec)


@pytest.mark.parametrize('handle', [-1,4])
def test_invalid_handles_refuse_before_reading_lengths(handle):
    buffers, backing, spec = fixture()
    backing[0][0] = handle
    with pytest.raises(ValueError,match='handle'): copy(buffers,spec)


@pytest.mark.parametrize('length', [-1,5])
def test_invalid_row_length_refuses_before_payload_copy(length):
    buffers, backing, spec = fixture()
    backing[1][2] = length
    with pytest.raises(ValueError,match='row length'): copy(buffers,spec)


def test_unused_row_lengths_not_read():
    buffers, backing, spec = fixture()
    backing[1][3] = -900
    assert len(copy(buffers,spec)[0]) == 4


def test_duplicate_roles_and_missing_or_duplicate_spec_refuse():
    buffers, backing, spec = fixture()
    with pytest.raises(ValueError,match='distinct'): copy(buffers,replace(spec,glyph_stride_id=6))
    with pytest.raises(ValueError,match='match'): copy_owned_sequence_results(buffers,[],glyph_sequences=[spec])
    with pytest.raises(ValueError,match='duplicate'): copy_owned_sequence_results(buffers,[(0,1,2,3,'int64')],glyph_sequences=[spec,spec])


def test_metadata_shape_and_unsigned_overflow_refuse():
    buffers, backing, spec = fixture()
    buffers[7] = np.array([4,4],dtype=np.int64)
    with pytest.raises(ValueError,match='one integer'): copy(buffers,spec)
    buffers[7] = np.array([np.iinfo(np.uint64).max],dtype=np.uint64)
    with pytest.raises(ValueError,match='capacity'): copy(buffers,spec)


def test_same_data_or_metadata_output_collision_refuses():
    buffers, backing, spec = fixture()
    for other in ((0,8,9,3,'int64'),(8,0,9,3,'int64'),(8,9,1,3,'int64')):
        with pytest.raises(ValueError,match='collides'):
            copy_owned_sequence_results(buffers,[(0,1,2,3,'int64'),other],glyph_sequences=[spec])


def test_empty_outer_does_not_authorize_overflowing_stride():
    buffers = {i:np.zeros(1,dtype=np.int64) for i in range(8)}
    buffers[7] = np.array([np.iinfo(np.uint64).max],dtype=np.uint64)
    with pytest.raises(ValueError,match='capacity'):
        copy(buffers,OwnedGlyphSequenceCopy(*range(8)))


@pytest.mark.parametrize('capacity', [1, 4, 16])
def test_allocator_shaped_empty_reserved_outer_has_no_child_arenas(monkeypatch, capacity):
    outer = np.zeros(capacity, dtype=np.int64)
    owner = np.zeros(1, dtype=np.int64)
    values = [outer.ctypes.data, 0, capacity, owner.ctypes.data, 0, 0, 0, 0]
    buffers = {i:np.array([v],dtype=np.uintp if i in (0,3,4,5) else np.int64)
               for i,v in enumerate(values)}
    monkeypatch.setattr(np.ctypeslib, 'as_array', lambda *a,**k: pytest.fail('empty must not read'))
    result = copy(buffers, OwnedGlyphSequenceCopy(*range(8)))
    assert result[0] == () and result[1][0] == 0 and result[2][0] == capacity
    # The identical null child storage is invalid as soon as a row is live.
    buffers[1][0] = 1
    with pytest.raises(ValueError,match='null'):
        copy(buffers, OwnedGlyphSequenceCopy(*range(8)))


@pytest.mark.parametrize('bad', [[], '0', -1, True])
def test_invalid_port_identity_refuses_at_spec_creation(bad):
    ids = list(range(8)); ids[0] = bad
    with pytest.raises(ValueError,match='distinct'):
        OwnedGlyphSequenceCopy(*ids)

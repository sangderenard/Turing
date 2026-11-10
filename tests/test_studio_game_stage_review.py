import hashlib

import pytest

from tools.stage_studio_game_layer import reviewed_game, stage


def test_alternate_game_requires_exact_reviewed_bytes(tmp_path):
    candidate = tmp_path / 'game.html'
    candidate.write_bytes(b'reviewed candidate')
    expected = hashlib.sha256(candidate.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='explicitly reviewed'):
        reviewed_game(candidate)
    with pytest.raises(ValueError, match='64 lowercase'):
        reviewed_game(candidate, 'not-a-hash')
    with pytest.raises(ValueError, match='rebased'):
        reviewed_game(candidate, '0' * 64)
    path, body, receipt_hash = reviewed_game(candidate, expected)
    assert path == candidate.resolve()
    assert body == b'reviewed candidate'
    assert receipt_hash == expected
    candidate.write_bytes(b'changed after review')
    with pytest.raises(ValueError, match='rebased'):
        reviewed_game(candidate, expected)


def test_stage_never_overwrites_frozen_candidate(tmp_path):
    sentinel = tmp_path / 'publication.json'
    sentinel.write_bytes(b'preserved')
    with pytest.raises(ValueError, match='already exists'):
        stage(tmp_path)
    assert sentinel.read_bytes() == b'preserved'

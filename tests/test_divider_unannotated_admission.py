"""Authored queue admission only: source is never compiled or given invented types."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'examples')]
from meowdoku import source_divider as sd
from meowdoku.state_paths import StatePaths

SOURCE = b'def identity(value):\n    return value\n'


@pytest.mark.parametrize('allow', [False, True])
def test_unannotated_submit_preserves_source_and_annotation_truth(tmp_path, allow):
    state = StatePaths.coerce(tmp_path)
    db = sd.connect(state.accounts_db)
    try:
        intake = sd._new_job(state, '123456789abc', 'test')
        sd._store(intake, 'm.py', SOURCE)
        sd.submit(db, state, intake, owner_key='account:1', allow_unannotated=allow)
        pieces = sd.pieces(db, intake.job_id)
        assert pieces and all(p['annotated'] is False for p in pieces)
        attempts = db.execute('SELECT state,stage FROM divider_attempts WHERE job_id=?', (intake.job_id,)).fetchall()
        assert [tuple(row) for row in attempts] == [('queued', None) if allow else ('needs-annotation', 'plan')]
        assert (intake.job_dir / 'source' / 'm.py').read_bytes() == SOURCE
        # Idempotent replanning never silently promotes an already refused plan.
        sd.plan(db, state, intake.job_id, allow_unannotated=not allow)
        assert [tuple(row) for row in db.execute('SELECT state,stage FROM divider_attempts WHERE job_id=?', (intake.job_id,))] == [tuple(row) for row in attempts]
    finally:
        db.close()


@pytest.mark.parametrize('bad', ['true', 1, None])
def test_invalid_plan_and_submit_option_rejected_before_database_access(bad):
    with pytest.raises(ValueError, match='boolean'):
        sd.plan(None, None, '123456789abc', allow_unannotated=bad)
    with pytest.raises(ValueError, match='boolean'):
        sd.submit(None, None, None, owner_key='account:1', allow_unannotated=bad)

from __future__ import annotations

from tools.profile_repository_compression import main, profile


def test_profile_reports_pickle_and_real_repository_levels(capsys):
    records = profile(row_count=32, tile_rows=16, levels=(0, 1), repeats=1)
    assert tuple(record[0] for record in records) == (
        "pickle", "pickle+zlib-1", "pickle+zlib-9", "pickle+lzma-6",
        "repository-0", "repository-1",
    )
    assert all(record[1] > 0 and record[3] >= 0 and record[4] >= 0
               for record in records)

    assert main([
        "--rows", "16", "--tile-rows", "8", "--levels", "0",
    ]) == 0
    output = capsys.readouterr().out
    assert "payload" in output and "pickle/x" in output
    assert "repository-0" in output

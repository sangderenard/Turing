"""Removed duplicate ingestion must not silently build traversal products."""
import pytest
from examples.meowdoku.studio_traversal_compute import build


def test_full_traversal_refuses_without_writing_artifacts(tmp_path):
    output = tmp_path / "artifacts"
    with pytest.raises(NotImplementedError, match="lower_ast_source_to_ssa"):
        build(output, node_capacity=3, triangle_capacity=2)
    assert not output.exists()

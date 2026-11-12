from pathlib import Path
import shutil
import subprocess
import pytest


def test_field_category_lineage_and_transparent_order():
    node=shutil.which("node")
    if not node:
        pytest.skip("Node required for renderer grouping checks")
    script=Path(__file__).with_name("meowdoku_field_categories_check.mjs")
    subprocess.run([node,str(script)],check=True,timeout=30,cwd=script.parent.parent)

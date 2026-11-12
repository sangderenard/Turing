"""Rendered native prisms retain exact local contact geometry."""
from pathlib import Path
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
def test_prism_mesh_and_collider_share_rigid_transform():
    node=shutil.which("node")
    if not node: pytest.skip("Node.js required")
    result=subprocess.run([node,str(ROOT/"tests/closed_solid_realization_check.cjs")],cwd=ROOT,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr


def test_authored_bridge_slabs_register_in_geometry_container():
    import textwrap
    from types import SimpleNamespace
    source=(ROOT/"src/compiler/abstract_ui_div_map.py").read_text(encoding="utf-8")
    start=source.index("        slab_records = []")
    end=source.index("        # A separate low-speed pleasure road",start)
    context={"world":SimpleNamespace(identity="test-world"),"course_parent":"parent",
             "yard_center_x":0.,"course_z":0.,"course_top":4.,"course_thickness":.58,
             "geometry":{"boxes":[]}}
    exec(textwrap.dedent(source[start:end]),context)
    assert len(context["geometry"]["boxes"])==6
    assert len(context["slab_records"])==6
    assert all(box["geometry_mode"]=="solid" for box in context["geometry"]["boxes"])

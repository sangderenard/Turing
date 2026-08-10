from __future__ import annotations

import sys
from pathlib import Path

import build_homepage
import build_site_page
import build_wasm_compiler_page
from src.compiler.site_bundle import DEFAULT_PUBLISH_ROOT


def test_gallery_generators_default_to_parent_publish_root(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_homepage.py"])
    assert build_homepage._arguments().destination == DEFAULT_PUBLISH_ROOT

    monkeypatch.setattr(
        sys,
        "argv",
        ["build_site_page.py", "--source", "example.py"],
    )
    assert build_site_page._arguments().destination == DEFAULT_PUBLISH_ROOT

    monkeypatch.setattr(sys, "argv", ["build_wasm_compiler_page.py"])
    assert build_wasm_compiler_page._arguments().destination == DEFAULT_PUBLISH_ROOT


def test_infer_python_package(tmp_path: Path):
    source = tmp_path / "src" / "compiler" / "example.py"
    source.parent.mkdir(parents=True)
    (tmp_path / "src" / "__init__.py").touch()
    (source.parent / "__init__.py").touch()
    source.touch()

    assert build_site_page._infer_python_package(source) == "src.compiler"

    standalone = tmp_path / "standalone" / "example.py"
    standalone.parent.mkdir()
    standalone.touch()
    assert build_site_page._infer_python_package(standalone) is None


def test_homepage_generator_refuses_turing_as_destination():
    try:
        build_homepage.build(build_homepage.TURING_ROOT)
    except ValueError as error:
        assert "parent workspace root, not Turing" in str(error)
    else:
        raise AssertionError("homepage builder accepted the Turing repository")

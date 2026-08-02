from __future__ import annotations

import sys

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


def test_homepage_generator_refuses_turing_as_destination():
    try:
        build_homepage.build(build_homepage.TURING_ROOT)
    except ValueError as error:
        assert "parent workspace root, not Turing" in str(error)
    else:
        raise AssertionError("homepage builder accepted the Turing repository")

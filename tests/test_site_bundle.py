from __future__ import annotations

import json
from pathlib import Path

from src.compiler.site_bundle import (
    BUNDLE_SCHEMA,
    DEFAULT_PUBLISH_ROOT,
    TURING_REPOSITORY_ROOT,
    build_program_bundle,
    build_source_inspection_bundle,
    build_source_inspection_page,
    discover_source_contract,
    resolve_publish_root,
    slugify,
)


def test_default_publish_root_is_the_parent_workspace():
    assert DEFAULT_PUBLISH_ROOT == TURING_REPOSITORY_ROOT.parent
    assert resolve_publish_root(DEFAULT_PUBLISH_ROOT) == DEFAULT_PUBLISH_ROOT


def test_gallery_refuses_to_publish_into_the_turing_source_repository():
    import pytest

    with pytest.raises(ValueError, match="parent workspace root, not Turing"):
        resolve_publish_root(TURING_REPOSITORY_ROOT)


class _InspectionCompiler:
    table_size: int = 8

    def encode(self, value):
        self.last_value = value
        return value


SOURCE = """
TURING_PAGE = {
    "title": "Affine Field",
    "slug": "affine-field",
    "feeds": {"gain": {"values": [2.0, 2.0, 2.0, 2.0]}},
    "feed_expressions": {"x": "x / max(1, w - 1)"},
    "width": 16,
    "height": 8,
}

def helper(value):
    return value + 1.0

def kernel(x, gain):
    return helper(x * gain)
"""


def test_literal_source_contract_is_inspected_without_importing_source():
    contract = discover_source_contract(SOURCE)

    assert contract.entrypoint == "kernel"
    assert contract.title == "Affine Field"
    assert contract.slug == "affine-field"
    assert contract.feeds["gain"]["values"] == [2.0] * 4
    assert contract.feed_expressions == {"x": "x / max(1, w - 1)"}
    assert (contract.width, contract.height) == (16, 8)


def test_website_builder_accepts_a_compiler_class_without_embedding_wasm(tmp_path):
    page = build_source_inspection_page(_InspectionCompiler, tmp_path)
    html = page.read_text(encoding="utf-8")

    assert page.name == "index.html"
    assert "_InspectionCompiler.encode" in html
    assert "Class map and navigation LUT" in html
    assert "turing.class.resolve_member" in html
    assert "window.TuringClassNavigation" in html
    assert 'id="picker"' in html
    assert "WASM_BASE64 = null" in html


def test_ast_inspection_is_published_as_a_standard_gallery_bundle(tmp_path):
    bundle = build_source_inspection_bundle(
        _InspectionCompiler,
        tmp_path,
        title="Inspection Compiler",
        slug="inspection-compiler",
    )
    html = bundle.page_path.read_text(encoding="utf-8")

    assert bundle.manifest["schema"] == BUNDLE_SCHEMA
    assert bundle.manifest["program"]["kind"] == "python-ast-inspection"
    assert bundle.url.startswith(
        "/site/programs/inspection-compiler/versions/v1-"
    )
    assert "_InspectionCompiler.encode" in html
    assert '"mode": "python-ast-inspection"' in bundle.manifest_path.read_text()
    assert (
        bundle.directory / "source/python_source/test_site_bundle.py"
    ).is_file()
    callable_pages = list((bundle.directory / "callables").glob("*/index.html"))
    assert callable_pages
    assert any("inspectioncompiler-encode" in page.parent.name for page in callable_pages)
    assert "Callable run systems" in html
    assert 'data-callable="_InspectionCompiler.encode"' in html
    assert "Open generated callable page" in html
    assert 'const RESOURCE_ROUTE = "./"' in html
    assert 'const STATIC_GALLERY = [' in html
    assert 'const BROWSER_PYTHON = {"script_url": "https://cdn.jsdelivr.net/pyodide/' in html
    assert 'href="callables/inspectioncompiler-encode/index.html"' in html
    assert '"python_source_url": "source/python_source/test_site_bundle.py"' in html


def test_file_scope_precedes_classes_and_lists_outer_symbols(tmp_path):
    source_path = tmp_path / "outer_scope.py"
    source_path.write_text(SOURCE, encoding="utf-8")

    bundle = build_source_inspection_bundle(
        source_path,
        tmp_path / "published",
        title="Outer Scope",
        slug="outer-scope",
    )
    html = bundle.page_path.read_text(encoding="utf-8")

    assert 'data-callable-owner-tab="file-scope" aria-selected="true"' in html
    assert 'data-callable="helper"' in html
    assert 'data-callable="kernel"' in html
    assert 'data-callable-view="file-symbols" hidden' in html
    assert "TURING_PAGE" in html
    assert 'class="python-callable-run"' in html
    assert 'fetch(serverURL("/api/run")' in html


def test_request_values_override_the_literal_source_contract():
    contract = discover_source_contract(
        SOURCE,
        title="Uploaded title",
        slug="UPLOAD 42",
        probes={"x": [1.0, 2.0]},
    )

    assert contract.title == "Uploaded title"
    assert contract.slug == "upload-42"
    assert contract.feeds["x"] == [1.0, 2.0]


def test_slugify_never_emits_path_syntax():
    assert slugify(" ../../My Neat_Kernel.py ") == "my-neat-kernel-py"


def test_program_bundle_owns_page_source_wasm_manifest_and_inventory(tmp_path: Path):
    bundle = build_program_bundle(
        SOURCE,
        tmp_path,
        source_filename="affine.py",
        include_backends=False,
        include_mathematics=False,
    )

    assert bundle.manifest["schema"] == BUNDLE_SCHEMA
    assert bundle.page_path.is_file()
    assert (bundle.directory / "source/python_source/affine.py").is_file()
    assert list((bundle.directory / "wasm").glob("*.wasm"))
    paths = {item["path"] for item in bundle.manifest["artifacts"]}
    assert "index.html" in paths
    assert "source/python_source/affine.py" in paths
    assert any(path.startswith("wasm/") for path in paths)
    assert bundle.url.startswith("/site/programs/affine-field/versions/v1-")
    assert json.loads(bundle.manifest_path.read_text())["page"]["url"] == bundle.url

    # Content versioning is idempotent: the same source/config returns the
    # same complete directory rather than making timestamp-named duplicates.
    repeated = build_program_bundle(
        SOURCE,
        tmp_path,
        source_filename="affine.py",
        include_backends=False,
        include_mathematics=False,
    )
    assert repeated.directory == bundle.directory

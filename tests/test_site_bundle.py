from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from src.compiler.site_bundle import (
    BUNDLE_SCHEMA,
    DEFAULT_WASM_CARD_OPERATIONS,
    DEFAULT_PUBLISH_ROOT,
    TURING_REPOSITORY_ROOT,
    _compile_feed_values,
    _content_version,
    _shader_execution_descriptor,
    _write_program_origin,
    build_program_bundle,
    build_source_inspection_bundle,
    build_source_inspection_page,
    discover_source_contract,
    publish_prebuilt_program_bundle,
    resolve_publish_root,
    slugify,
)
from src.compiler.ssa_fortran_backend import fortran_compiler


def test_default_publish_root_is_the_parent_workspace():
    assert DEFAULT_PUBLISH_ROOT == TURING_REPOSITORY_ROOT.parent


def test_prebuilt_program_uses_common_versioned_bundle_and_refreshes_gallery(tmp_path):
    (tmp_path / "index.html").write_text(
        '<script>const STATIC_GALLERY = [];</script>', encoding="utf-8",
    )
    bundle = publish_prebuilt_program_bundle(
        destination=tmp_path,
        slug="Dream Machine",
        title="Dream Machine",
        entrypoint="run",
        html="<!doctype html><title>machine</title>",
        source_filename="dream/machine.dream",
        source="/* dream */",
        artifacts={"subject/demo.pe": b"MZ\0\0"},
        runtime={"snapshot_abi": "TMSNAP01"},
        refresh_gallery=True,
    )

    assert bundle.directory.parent.name == "versions"
    assert bundle.manifest["schema"] == BUNDLE_SCHEMA
    assert bundle.manifest["runtime"]["snapshot_abi"] == "TMSNAP01"
    assert (bundle.directory / "subject" / "demo.pe").read_bytes() == b"MZ\0\0"
    assert all(item["sha256"] for item in bundle.manifest["artifacts"])
    assert '"slug": "dream-machine"' in (tmp_path / "index.html").read_text()
    assert publish_prebuilt_program_bundle(
        destination=tmp_path,
        slug="Dream Machine",
        title="Dream Machine",
        entrypoint="run",
        html="<!doctype html><title>machine</title>",
        source_filename="dream/machine.dream",
        source="/* dream */",
        artifacts={"subject/demo.pe": b"MZ\0\0"},
        runtime={"snapshot_abi": "TMSNAP01"},
    ).directory == bundle.directory


def test_only_published_browser_shader_graduates_a_bundle_page():
    desktop_only = [{
        "language": "glsl",
        "available": True,
        "url": "source/glsl/glsl.comp.glsl",
    }]
    assert _shader_execution_descriptor(desktop_only) is None

    descriptor = _shader_execution_descriptor(desktop_only + [{
        "language": "webgl",
        "available": True,
        "url": "source/webgl/webgl.frag.glsl",
    }])
    assert descriptor is None

    webgl_only = desktop_only + [{
        "language": "webgl",
        "role": "shader-surface",
        "available": True,
        "url": "source/roles/shader-surface/webgl.frag.glsl",
    }]
    descriptor = _shader_execution_descriptor(webgl_only)
    assert descriptor == {
        "url": "source/roles/shader-surface/webgl.frag.glsl",
        "language": "webgl2-glsl-es",
        "stage": "fragment",
        "role": "shader-surface",
        "autostart": True,
        "execution": {
            "continuous": True,
            "prefer_contiguous": True,
        },
        "candidates": [{
            "url": "source/roles/shader-surface/webgl.frag.glsl",
            "language": "webgl2-glsl-es",
            "stage": "fragment",
            "role": "shader-surface",
        }],
    }
    io = {"requirements": {"requests": [{"capability": "pointer"}]}}
    assert _shader_execution_descriptor(desktop_only + [{
        "language": "webgl",
        "role": "shader-surface",
        "available": True,
        "url": "surface.frag.glsl",
    }], io)["io"] == io
    assert resolve_publish_root(DEFAULT_PUBLISH_ROOT) == DEFAULT_PUBLISH_ROOT


def test_webgpu_candidate_takes_priority_and_canvas2d_is_the_last_resort():
    sources = [
        {
            "language": "webgpu",
            "role": "shader-surface",
            "available": True,
            "url": "source/roles/shader-surface/program.compute.wgsl",
        },
        {
            "language": "webgl",
            "role": "shader-surface",
            "available": True,
            "url": "source/roles/shader-surface/program.frag.glsl",
        },
    ]
    io = {"requirements": {"requests": [{"capability": "pointer"}]}}
    descriptor = _shader_execution_descriptor(sources, io)

    assert descriptor["language"] == "wgsl"
    assert descriptor["url"] == "source/roles/shader-surface/program.compute.wgsl"
    languages = [item["language"] for item in descriptor["candidates"]]
    assert languages == ["wgsl", "webgl2-glsl-es", "canvas2d"]
    assert descriptor["candidates"][-1]["url"] is None

    # No shell_io means the canvas2d last resort has nothing to paint, so it
    # is withheld rather than offered as a candidate with nothing to show.
    without_io = _shader_execution_descriptor(sources)
    assert [item["language"] for item in without_io["candidates"]] == [
        "wgsl", "webgl2-glsl-es",
    ]


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
    assert contract.bake_mode == "whole_program"
    assert contract.final_fused_reduction is True
    assert contract.file_parameters == {}
    assert contract.schedule_preference == "alap"
    assert contract.constant_map == {}
    assert (contract.width, contract.height) == (16, 8)


def test_configured_constant_map_is_validated_before_compilation():
    configured = SOURCE.replace(
        '"width": 16,', '"constants": {"gain": 2.0},\n    "width": 16,'
    )
    contract = discover_source_contract(configured)

    assert contract.constant_map == {"gain": 2.0}
    with pytest.raises(ValueError, match="unknown parameters"):
        discover_source_contract(configured.replace("gain\": 2.0", "missing\": 2.0"))


def test_state_feedback_parameters_cannot_be_declared_static():
    configured = SOURCE.replace(
        '"width": 16,',
        '"state_feedback": {"gain": "result"},\n'
        '    "constants": {"gain": 2.0},\n    "width": 16,',
    )

    with pytest.raises(ValueError, match="mutable.*cannot.*constants"):
        discover_source_contract(configured)


def test_bake_mode_override_is_explicit_and_validated():
    contract = discover_source_contract(
        SOURCE,
        bake_mode="one-shot",
        schedule_preference="ASAP",
    )

    assert contract.bake_mode == "one_shot"
    assert contract.schedule_preference == "asap"
    with pytest.raises(ValueError, match="one_shot.*whole_program"):
        discover_source_contract(SOURCE, bake_mode="numeric-ish")
    with pytest.raises(ValueError, match="asap.*alap"):
        discover_source_contract(SOURCE, schedule_preference="whenever")




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


def test_runtime_file_probe_does_not_identify_or_serialize_the_program(tmp_path: Path):
    source = '''
TURING_PAGE = {
    "file_parameters": {
        "subject": {"name": "subject-binary", "accept": ".exe"},
    },
}

def load_subject(subject: bytes):
    return len(subject)
'''
    first = discover_source_contract(source, probes={"subject": b"MZ-first"})
    second = discover_source_contract(source, probes={"subject": b"MZ-second"})

    assert first.mutable_parameters == ("subject",)
    assert _compile_feed_values(
        first, ("subject",), frozenset()
    ) == {}
    assert _content_version(source, first)[0] == _content_version(source, second)[0]

    origin_path = _write_program_origin(
        tmp_path,
        first,
        source,
        "loader.py",
        backend_targets=None,
        include_backends=False,
        include_mathematics=False,
    )
    origin = json.loads(origin_path.read_text(encoding="utf-8"))

    assert origin["probes"] == {}
    assert origin["runtime_file_parameters"] == {
        "subject": {"name": "subject-binary", "accept": ".exe"},
    }


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
    assert "origin" not in bundle.manifest  # origin is program-level, not per-version
    origin_path = bundle.directory.parent.parent / "origin.json"
    origin = json.loads(origin_path.read_text(encoding="utf-8"))
    assert origin["entrypoint"] == "kernel"
    assert origin["slug"] == "affine-field"
    assert origin["source"] == SOURCE
    assert origin["probes"]["gain"] == {"values": [2.0] * 4}
    assert origin["include_backends"] is False
    assert origin["include_mathematics"] is False
    assert origin["backend_targets"] is None
    assert bundle.page_path.is_file()
    assert (bundle.directory / "source/python_source/affine.py").is_file()
    assert list((bundle.directory / "wasm").glob("*.wasm"))
    paths = {item["path"] for item in bundle.manifest["artifacts"]}
    assert "index.html" in paths
    assert "source/python_source/affine.py" in paths
    assert any(path.startswith("wasm/") for path in paths)
    card_prefix = f"wasm/size-{DEFAULT_WASM_CARD_OPERATIONS}/"
    assert any(path.startswith(card_prefix) and path.endswith(".wasm") for path in paths)
    assert card_prefix + "class-inventory.json" in paths
    assert f'"{DEFAULT_WASM_CARD_OPERATIONS}":' in bundle.page_path.read_text()
    html = bundle.page_path.read_text(encoding="utf-8")
    assert '"supports_ranges": false' in html
    assert '"contiguous": null' in html
    assert bundle.manifest["compiler"]["bake_mode"] == "whole_program"
    assert bundle.manifest["compiler"]["schedule_preference"] == "alap"
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


def test_program_bundle_can_skip_final_fused_reduction(tmp_path: Path):
    bundle = build_program_bundle(
        SOURCE,
        tmp_path,
        source_filename="affine.py",
        final_fused_reduction=False,
        include_backends=False,
        include_mathematics=False,
    )

    html = bundle.page_path.read_text(encoding="utf-8")
    wasm_paths = {
        item["path"]
        for item in bundle.manifest["artifacts"]
        if item["path"].endswith(".wasm")
    }

    assert bundle.manifest["compiler"]["final_fused_reduction"] is False
    assert any(path.endswith("kernel_control_2000.wasm") for path in wasm_paths)
    assert '"contiguous": null' in html


def test_force_new_version_appends_instead_of_reusing(tmp_path: Path):
    import re

    first = build_program_bundle(
        SOURCE, tmp_path, source_filename="affine.py",
        include_backends=False, include_mathematics=False,
    )
    forced = build_program_bundle(
        SOURCE, tmp_path, source_filename="affine.py",
        include_backends=False, include_mathematics=False,
        force_new_version=True,
    )
    forced_again = build_program_bundle(
        SOURCE, tmp_path, source_filename="affine.py",
        include_backends=False, include_mathematics=False,
        force_new_version=True,
    )

    assert first.directory != forced.directory != forced_again.directory
    versions_dir = first.directory.parent
    assert first.directory.parent == forced.directory.parent  # same program
    match = re.match(r"^v(\d+)\.(\d{3})-(\d{8})-([0-9a-f]{16})$", forced.directory.name)
    assert match, forced.directory.name
    next_match = re.match(r"^v(\d+)\.(\d{3})-", forced_again.directory.name)
    assert next_match
    assert int(next_match.group(2)) == int(match.group(2)) + 1

    # A plain (non-forced) build still gets today's idempotent lookup and
    # does not add yet another version.
    plain_again = build_program_bundle(
        SOURCE, tmp_path, source_filename="affine.py",
        include_backends=False, include_mathematics=False,
    )
    assert plain_again.directory == first.directory
    assert len(list(versions_dir.iterdir())) == 3


def test_backend_targets_restricts_published_source_tabs(tmp_path: Path):
    bundle = build_program_bundle(
        SOURCE,
        tmp_path,
        source_filename="affine.py",
        backend_targets=("glsl",),
    )

    languages = {
        item["language"] for item in bundle.manifest["page"].get("shader", {}) or {}
    }
    html = bundle.page_path.read_text(encoding="utf-8")
    assert '"language": "glsl"' in html
    assert '"language": "webgl"' not in html
    assert '"language": "fortran"' not in html

    region_capabilities = bundle.manifest["compiler"]["region_target_capabilities"]
    assert region_capabilities
    assert all(isinstance(item, list) for item in region_capabilities.values())

    # A different backend_targets selection over the identical source must
    # not silently reuse the first build's directory/tabs.
    narrower = build_program_bundle(
        SOURCE,
        tmp_path,
        source_filename="affine.py",
        backend_targets=("fortran",),
    )
    assert narrower.directory != bundle.directory
    narrower_html = narrower.page_path.read_text(encoding="utf-8")
    assert '"language": "fortran"' in narrower_html
    assert '"language": "glsl"' not in narrower_html


RGB_SOURCE = """
TURING_PAGE = {
    "title": "RGB Passthrough",
    "slug": "rgb-passthrough",
    "width": 4,
    "height": 4,
}

def kernel(input_red, input_green, input_blue):
    red = input_red + 0.0
    green = input_green + 0.0
    blue = input_blue + 0.0
    return red, green, blue
"""


def test_default_passthrough_becomes_shader_surface_with_a_webgpu_candidate(tmp_path: Path):
    bundle = build_program_bundle(
        RGB_SOURCE,
        tmp_path,
        source_filename="rgb.py",
        include_backends=False,
        include_mathematics=False,
    )

    manifest = json.loads(bundle.manifest_path.read_text())
    shader = manifest["page"]["shader"]
    assert manifest["page"]["mode"] == "shader-execution"
    languages = [item["language"] for item in shader["candidates"]]
    assert languages[0] == "wgsl"
    assert "webgl2-glsl-es" in languages

    wgsl_path = bundle.directory / shader["candidates"][0]["url"]
    assert wgsl_path.is_file()
    wgsl_source = wgsl_path.read_text(encoding="utf-8")
    assert "@compute @workgroup_size(" in wgsl_source
    assert wgsl_source.count("var<storage, read_write>") == 3

    html = bundle.page_path.read_text(encoding="utf-8")
    assert '"language": "wgsl"' in html
    assert "window.TuringPassthroughShader" in html


def test_one_shot_bundle_packages_the_discovery_numeric_trace(tmp_path: Path):
    whole = build_program_bundle(
        SOURCE,
        tmp_path,
        source_filename="affine.py",
        include_backends=False,
        include_mathematics=False,
    )
    one_shot = build_program_bundle(
        SOURCE,
        tmp_path,
        source_filename="affine.py",
        include_backends=False,
        include_mathematics=False,
        bake_mode="one_shot",
    )

    assert one_shot.directory != whole.directory
    assert one_shot.manifest["compiler"]["bake_mode"] == "one_shot"
    html = one_shot.page_path.read_text(encoding="utf-8")
    assert '"supports_ranges": true' in html
    assert '"contiguous": {' in html


@pytest.mark.skipif(
    fortran_compiler() is None, reason="no Fortran compiler installed"
)
def test_program_bundle_compiles_fortran_and_records_output_fidelity(tmp_path: Path):
    bundle = build_program_bundle(
        SOURCE,
        tmp_path,
        source_filename="affine.py",
        include_backends=True,
        include_mathematics=False,
    )

    fortran_path = bundle.directory / "source/fortran/fortran.f90"
    api_path = bundle.directory / "source/fortran/fortran.api.yaml"
    proof_path = bundle.directory / "verification/fortran-fidelity.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))

    assert fortran_path.is_file()
    assert api_path.is_file()
    assert "intent(out)" in fortran_path.read_text(encoding="utf-8")
    assert proof["schema"] == "turing-fortran-fidelity-v1"
    assert proof["passed"] is True
    assert proof["case_count"] == 3
    assert all(case["passed"] for case in proof["cases"])
    assert all(
        output["fortran"] == output["reference"]
        for case in proof["cases"]
        for output in case["outputs"]
    )
    assert proof["source_sha256"] == hashlib.sha256(
        fortran_path.read_bytes()
    ).hexdigest()
    assert (
        bundle.directory / "native/fortran" / proof["native_library"]
    ).is_file()
    assert bundle.manifest["compiler"]["fortran_verification"] == {
        "passed": True,
        "case_count": 3,
        "source_sha256": proof["source_sha256"],
        "url": "verification/fortran-fidelity.json",
    }


def test_progress_sink_receives_live_build_records_that_also_reach_the_page(tmp_path: Path):
    """The telemetry channel used to be created and never written to, so a
    published page always showed an empty build log. build_program_bundle
    must both call a live progress_sink as it compiles and still bake the
    same records into the page's own console log."""

    seen = []
    bundle = build_program_bundle(
        SOURCE,
        tmp_path,
        source_filename="affine.py",
        include_backends=False,
        include_mathematics=False,
        progress_sink=seen.append,
    )

    kinds_and_paths = {(record.kind, record.path) for record in seen}
    assert ("log", "contract") in kinds_and_paths
    assert ("profile", "process_graph") in kinds_and_paths
    assert ("profile", "aot") in kinds_and_paths
    assert ("log", "wasm") in kinds_and_paths
    aot_start = next(
        index
        for index, record in enumerate(seen)
        if record.path == "aot" and record.message == "AOT compile starting"
    )
    graph_recovery = next(
        index
        for index, record in enumerate(seen)
        if record.path == "process_graph"
        and record.message == "recover process graph from AOT deployment"
    )
    assert aot_start < graph_recovery
    assert not any(
        record.path == "process_graph"
        and "build_from_ast" in record.message
        for record in seen
    )
    # "bundle published" is necessarily emitted after the page HTML is
    # already serialized -- a page cannot describe its own publish event --
    # so it is only observable live, not baked into this page's own log.
    assert ("log", "bundle") in kinds_and_paths

    html = bundle.page_path.read_text(encoding="utf-8")
    assert "AOT compile" in html
    assert "wasm module emitted" in html

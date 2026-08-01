import pytest

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.machine_targets import emit
from src.compiler.wasm_html_shell import emit_html_shell, shell_for_artifact


def _artifact(name="demo"):
    left, right, s0, s1, s2 = 1, 2, 3, 4, 5
    program = FusedProgram(
        version=1,
        feeds={left, right},
        steps=[
            OpStep(step_id=0, op_name="sub", input_ids=[left, right], attrs={}, result_id=s0),
            OpStep(step_id=1, op_name="abs", input_ids=[s0], attrs={}, result_id=s1),
            OpStep(
                step_id=2, op_name="add", input_ids=[s1],
                attrs={"right_scalar": 1.0}, result_id=s2,
            ),
        ],
        outputs={"result": s2},
    )
    return emit(program, "wasm", name=name)


def test_the_page_is_generated_from_the_descriptor_not_the_program():
    """The controls are whatever the parameters are -- compile something
    else and the page reshapes itself."""

    html = shell_for_artifact(_artifact()).html
    for parameter in ("count", "feed0", "feed1", "out0"):
        assert parameter in html
    # One input field per feed, none for the output.
    assert 'id="in_feed0"' in html
    assert 'id="in_feed1"' in html
    assert 'id="in_out0"' not in html


def test_without_a_binary_the_page_offers_a_picker_and_says_why():
    """A browser cannot assemble WAT. The page must say that plainly rather
    than looking broken."""

    shell = shell_for_artifact(_artifact())
    assert shell.embedded is False
    assert 'id="picker"' in shell.html
    assert "wat2wasm" in shell.html
    assert 'id="run" disabled' in shell.html


def test_with_a_binary_the_page_is_self_contained():
    shell = shell_for_artifact(_artifact(), wasm_bytes=b"\x00asm\x01\x00\x00\x00")
    assert shell.embedded is True
    assert 'id="picker"' not in shell.html
    assert "AGFzbQEAAAA=" in shell.html  # base64 of the header above
    assert "self-contained" in shell.html


def test_the_emitted_source_travels_with_the_page_for_reading():
    shell = shell_for_artifact(_artifact())
    assert "f64.sub" in shell.html
    assert "f64.abs" in shell.html


def test_no_layout_engine_creeps_in():
    """Layout belongs to a different subrepo. This is a stack of labelled
    rows; if it grows a grid engine or a component model, it should be handed
    over rather than extended here."""

    html = shell_for_artifact(_artifact()).html
    assert "<table" not in html
    assert "grid-template" not in html
    # And no third-party anything: the page must open with no network.
    assert "http://" not in html and "https://" not in html
    assert "<script src" not in html


def test_an_artifact_without_a_descriptor_is_refused():
    artifact = _artifact()
    stripped = type(artifact)(
        target=artifact.target, name=artifact.name, source=artifact.source,
        complete=artifact.complete, shortfalls=artifact.shortfalls,
        api=None, extension=artifact.extension, module=artifact.module,
    )
    with pytest.raises(ValueError, match="no API descriptor"):
        shell_for_artifact(stripped)


def test_the_page_writes_beside_its_artifact(tmp_path):
    shell = shell_for_artifact(_artifact(name="written"))
    path = shell.write(tmp_path)
    assert path.name == "written_shell.html"
    assert path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_a_mapping_is_accepted_as_well_as_a_descriptor_object():
    artifact = _artifact()
    shell = emit_html_shell(artifact.api.to_mapping(), source=artifact.source)
    assert "feed0" in shell.html


# --- output views and diagnostics -----------------------------------------


def test_output_views_are_tabs_over_the_same_numbers():
    """How to look at a result is the caller's question, not a property of
    the program, so it is a tab rather than a second compilation."""

    html = shell_for_artifact(_artifact()).html
    assert 'data-view="raw"' in html
    assert 'data-view="image"' in html
    assert "<canvas" in html
    assert "renderWebGLPalette" in html
    assert 'canvas.getContext("webgl2"' in html
    assert "const anyNetwork" in html
    assert "anyExpression || anyGaussian || anyNetwork" in html
    assert "raw scalar field rendered into RGB canvas pixels" in html
    assert "toDataURL(\"image/jpeg\"" not in html
    # Geometry is stated once, on the domain, and the image view follows it
    # -- two places to type a width is two places for them to disagree.
    assert 'id="dom_w"' in html and 'id="dom_h"' in html
    assert 'id="img_w"' not in html


def test_the_diagnostics_bootstrap_is_a_separate_script():
    """A handler defined inside the program script cannot catch that
    script's own parse error -- nothing in it has run yet. Two script tags
    is what makes a dead shell announce itself instead of looking inert."""

    html = shell_for_artifact(_artifact()).html
    assert html.count("<script>") == 2
    boot, program = html.split("<script>")[1], html.split("<script>")[2]
    assert 'addEventListener("error"' in boot
    assert "const API =" in program
    # The banner the handler reveals must exist before either script runs.
    assert html.index('id="fatal"') < html.index("<script>")


def test_the_call_itself_is_logged_not_just_the_result():
    """Argument order and the memory offsets are the two things most likely
    to be wrong and the least visible from a wrong answer alone."""

    html = shell_for_artifact(_artifact()).html
    assert 'log("call"' in html
    assert "offsets: offsets" in html
    assert 'log("error"' in html
    assert 'id="copylog"' in html


def test_the_javascript_has_no_stray_real_newline_inside_a_string_literal():
    """Twice, a JS escape written into a non-raw Python string became a real
    newline and killed the whole shell at parse time. _JS is raw now; this
    pins it, because the symptom (a page that renders but does nothing) is
    far from the cause."""

    from src.compiler import wasm_html_shell

    for name in ("_BOOT_JS", "_JS"):
        source = getattr(wasm_html_shell, name)
        for number, line in enumerate(source.splitlines(), 1):
            # An odd number of quotes on a line means a string was opened and
            # not closed on that line.
            unescaped = line.replace('\\"', "")
            assert unescaped.count('"') % 2 == 0, f"{name} line {number}: {line!r}"


def test_the_shell_receives_telemetry_progress_graph_and_both_sources():
    from src.compiler.shell_telemetry import TelemetryChannel
    from src.compiler.wasm_html_shell import emit_html_shell

    channel = TelemetryChannel(name="build")
    with channel.stepped("compiling", 2) as advance:
        channel.log("frontend done", path="frontend", nodes=83)
        advance("graph")
        channel.profile("emission", nanoseconds=1234, path="wasm")
        advance("wasm")

    artifact = _artifact()
    shell = emit_html_shell(
        artifact.api,
        source=artifact.source,
        wasm_bytes=b"\x00asm\x01\x00\x00\x00",
        telemetry=channel,
        process_graph={"nodes": 83, "edges": 88, "histogram": {"Load": 34},
                       "table": [], "truncated": False},
        origin_source="def kernel(a, b):\n    return a - b\n",
    )
    html = shell.html

    # Build records travel with the page, so the timeline starts before it.
    assert "frontend done" in html and "compiling" in html
    assert '"nodes": 83' in html or '"nodes":83' in html
    # Progress drives the bar from the same records shown in the pane.
    assert 'id="barfill"' in html and 'setProgress(' in html
    # Both sources, and the descriptor, are readable from the page.
    assert "def kernel(a, b):" in html
    assert "API descriptor" in html and 'id="apiyaml"' in html
    assert "schema: turing-compiled-program-api-v1" in html


def test_segmented_shell_keeps_one_public_api_and_runs_full_arrays():
    from src.compiler.wasm_html_shell import emit_html_shell

    artifact = _artifact()
    class_graph = {
        "modules": [{
            "name": "private_region_0",
            "wasm_base64": "AGFzbQEAAAA=",
            "entry": "kernel__0",
            "reserved_bytes": 24,
            "inputs": ["feed0"],
            "outputs": ["value_2"],
            "value_type": "f64",
            "element_bytes": 8,
            "shared_memory_import": {"module": "env", "field": "memory"},
        }],
        "edges": [],
        "logical_inputs": {"feed0": [["private_region_0", "feed0"]]},
        "logical_outputs": {"result": ["private_region_0", "value_2"]},
        "root_module": "private_region_0",
        "root_outputs": ["value_2"],
        "shared_memory": True,
        "shared_static_bytes": 24,
        "schedule": {
            "nodes": [{"id": "private_region_0", "level": 0,
                       "operation_count": 1, "is_root": True}],
            "levels": [{"level": 0, "modules": ["private_region_0"]}],
        },
    }
    html = emit_html_shell(
        artifact.api,
        class_graph=class_graph,
        process_graph={"nodes": 1, "edges": 0, "histogram": {},
                       "table": [], "truncated": False},
    ).html

    assert "new WebAssembly.Memory" in html
    assert "WebAssembly.instantiate(moduleBinary, imports)" in html
    assert "No live tensor is copied through" in html
    assert "shared-memory slot" in html
    assert "Live deployment schedule:" in html
    assert "await fetch(spec.url)" in html
    assert "one element per call today" not in html


def test_versioned_sources_are_not_embedded_or_fetched_before_a_click():
    html = emit_html_shell(
        _artifact().api,
        backend_sources=[{
            "language": "fortran", "title": "Fortran", "available": True,
            "source": "SECRET SOURCE BODY", "lines": 1,
            "url": "site/v2/source/render/fortran/render.f90",
            "filename": "render.f90",
        }],
    ).html

    assert "SECRET SOURCE BODY" not in html
    assert "site/v2/source/render/fortran/render.f90" in html
    assert "The file is fetched only after this button is clicked" in html
    assert 'button.addEventListener("click", async () =>' in html
    assert "await fetch(descriptor.url)" in html


def test_sympy_mathematics_is_rendered_separately_from_lazy_sources():
    mathml = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">'
        "<mrow><mi>y</mi><mo>=</mo><mi>x</mi></mrow></math>"
    )
    html = emit_html_shell(
        _artifact().api,
        mathematics={
            "target": "sympy",
            "projection": "process_graph_to_sympy_relations",
            "node_count": 3,
            "equation_count": 2,
            "constraint_count": 0,
            "uninterpreted": [],
            "program_relation": {
                "head": "And", "arity": 2, "arguments": "equations[*]",
            },
            "outputs": [{
                "name": "result", "node_id": 5, "mathml": mathml,
            }],
            "url": "site/v3/math/render/sympy-process-model.json",
        },
    ).html

    assert "Math is programming is math" in html
    assert "existing SymPy target" in html
    assert mathml in html
    assert "site/v3/math/render/sympy-process-model.json" in html
    assert "Load and render the symbolic program" in html
    assert "One SymPy <code>And</code>" in html
    assert "<mo>⋀</mo>" in html
    assert "await fetch(MATHEMATICS.url)" in html
    assert "DOMParser" in html


def test_graph_phosphor_integrates_profile_pulses_with_decay():
    html = shell_for_artifact(_artifact()).html
    assert "function phosphorColor(node, now)" in html
    assert "Math.exp(-age / decay)" in html
    assert "pulseGraphNodes(spec.node_ids, elapsedMs)" in html
    assert 'id="graph-decay"' in html or "graph-decay" in html
    assert 'id="graph-edges" type="checkbox"' in html
    assert "showAllEdges || graphSelectedNode !== null" in html
    assert "Math.atan2(vector[1], vector[0])" in html


def test_an_edited_descriptor_does_not_pretend_to_apply():
    """Applying an edited descriptor is not wired up; a control that looks
    live but is not is worse than one that says so."""

    html = shell_for_artifact(_artifact()).html
    assert 'id="applyapi" disabled' in html
    assert "not wired up" in html


def test_feeds_can_be_generated_from_the_grid_rather_than_typed():
    """A kernel's feeds are a function of position. Pasting a quarter of a
    million numbers into a text field is a workaround, not a control."""

    from src.compiler.wasm_html_shell import emit_html_shell

    artifact = _artifact()
    html = emit_html_shell(
        artifact.api,
        source=artifact.source,
        feed_expressions={"feed0": "-2.2 + 3.0 * x / (w - 1)"},
        default_width=480,
        default_height=300,
    ).html

    assert 'id="mode_feed0"' in html
    assert "-2.2 + 3.0 * x / (w - 1)" in html
    # The one with an expression defaults to it; the other stays literal.
    assert 'id="expr_feed1"' in html
    assert 'value="480"' in html and 'value="300"' in html


def test_compiled_in_parameters_are_shown_but_not_editable():
    """An unrolled loop count is part of the emitted instructions. Offering
    it as an input would be a lie -- it needs a recompile."""

    from src.compiler.wasm_html_shell import emit_html_shell

    artifact = _artifact()
    html = emit_html_shell(
        artifact.api,
        source=artifact.source,
        build_parameters={"iterations (unrolled)": 48, "steps": 720},
    ).html

    assert "iterations (unrolled)" in html and ">48<" in html
    assert "needs a recompile" in html
    assert 'id="in_iterations (unrolled)"' not in html


def test_every_backend_gets_a_tab_including_the_ones_that_refused():
    """Which languages a program can reach is a real property of the
    program. A tab that quietly vanished would hide it."""

    from src.compiler.wasm_html_shell import emit_html_shell

    artifact = _artifact()
    html = emit_html_shell(
        artifact.api,
        source=artifact.source,
        backend_sources=[
            {"language": "fortran", "title": "Fortran", "source": "module k\nend",
             "available": True, "reason": "", "highlight": "fortran", "lines": 2},
            {"language": "spirv", "title": "SPIR-V", "source": "",
             "available": False, "reason": "no SPIR-V type for dtype 'x'",
             "highlight": "text", "lines": 0},
        ],
    ).html

    assert "What made this" in html
    assert 'data-lang="fortran"' in html and 'data-lang="spirv"' in html
    assert "module k" in html
    # The refusal is shown, with its reason, rather than dropped.
    assert "no SPIR-V type" in html
    assert "&middot; n/a" in html


def test_inputs_can_be_drawn_from_a_gaussian():
    html = shell_for_artifact(_artifact()).html
    assert '<option value="gaussian">' in html
    assert 'id="mean_feed0"' in html and 'id="sigma_feed0"' in html
    assert "function gaussian()" in html
    # Box-Muller keeps its spare rather than discarding half the work.
    assert "spareNormal" in html


def test_the_program_can_be_looped_for_a_steady_state_measurement():
    """One call measures instantiation and first-touch as much as the
    kernel; the spread over repeats is what says how fast it is."""

    html = shell_for_artifact(_artifact()).html
    assert 'id="repeats"' in html
    assert "median" in html and "Melem/s" in html
    assert 'log("profile", "steady state over ' in html


def test_a_repeat_is_also_a_frame_so_the_picture_moves():
    """Repeating with identical inputs measures speed and nothing else. Each
    repeat regenerates the feeds, so a gaussian redraws and an expression
    sees a new t -- that is what makes the output change over time, and it
    is one mechanism rather than a separate animate button doing the same
    thing."""

    html = shell_for_artifact(_artifact()).html
    assert "frameIndex = r;" in html
    assert "feedValues(p, count, d, frameIndex)" in html
    # Painted per frame, or only the last frame would ever be seen.
    assert "requestAnimationFrame" in html
    assert "fps" in html
    # t is offered to expressions and documented where they are entered.
    assert '"i", "x", "y", "w", "h", "t"' in html
    assert "expression over i, x, y, w, h, t" in html
    # Only the kernel call is timed, not the feed regeneration around it.
    assert html.index("const t0 = frameStarted;") < html.index("fn(...args);")


def test_animation_is_driven_by_repeats_not_a_second_control():
    html = shell_for_artifact(_artifact()).html
    assert 'id="animate"' not in html
    assert 'id="frames"' not in html


def test_feedback_network_contract_is_executable():
    from src.compiler.wasm_html_shell import emit_html_shell

    manifest = {
        "name": "future scorer",
        "module": {"api": {"entry_points": []}, "wasm_base64": "AGFzbQ=="},
        "feedback": {"candidate_offsets": [0.0, 0.45, 0.9], "fps": 120, "travel_feed": "feed0"},
        "routes": [{"feed": "feed0", "effect": "future scores to speed"}],
    }
    html = emit_html_shell(_artifact().api, network_manifest=manifest).html
    assert "advanceFeedback" in html
    assert "candidate_offsets" in html
    assert "feedbackState.speed" in html
    assert "WebAssembly.instantiate(bytes" in html

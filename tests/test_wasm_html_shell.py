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
    assert "toDataURL(\"image/jpeg\"" in html
    # Geometry is the caller's to state; a run does not know its own shape.
    assert 'id="img_w"' in html and 'id="img_h"' in html


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

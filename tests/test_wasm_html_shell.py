import base64
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.machine_targets import emit
from src.compiler.wasm_html_shell import emit_html_shell, shell_for_artifact
from src.compiler.compiled_program_api import CompiledProgramAPI, EntryPoint, Parameter
from src.compiler.shell_io import (
    ShellIOManifest, ShellIORequest, SystemPort, VirtualFileSystemContract,
    VirtualMount, attach_shell_io,
)


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


def _file_port_api(*, domain=None):
    parameters = (
        Parameter("t4", "input", "u8", "uint8_t", "c_uint8", "reference", source_name="subject_bytes"),
        Parameter("t5", "input", "i64", "int64_t", "c_int64", "value", source_name="subject_length"),
    )
    requests = [ShellIORequest.create("files")]
    ports = [SystemPort.create(
        "subject", "file", "input", entry_point="load_subject",
        fields={"data": "subject_bytes", "length": "subject_length"},
        attributes={"accept": ".exe,.dll"},
    )]
    if domain is not None:
        requests.append(ShellIORequest.create(
            "bundle_references" if domain == "bundle" else "host_references"
        ))
        ports.append(SystemPort.create(
            "decoder", "external_reference", "call",
            external_domain=domain,
            attributes={"bundle": "decoder-bundle", "export": "decode"},
        ))
    return attach_shell_io(CompiledProgramAPI(
        "machine", "wasm", "load_subject",
        (EntryPoint("load_subject", "load_subject", "control", parameters),),
    ), ShellIOManifest(tuple(requests), system_ports=tuple(ports)))


def test_html_renders_file_system_port_instead_of_numeric_parameter_fields():
    html = emit_html_shell(_file_port_api(domain="bundle")).html

    assert 'data-system-file-port="subject"' in html
    assert 'accept=".exe,.dll"' in html
    assert 'id="in_t4"' not in html
    assert 'id="in_t5"' not in html
    assert "window.TuringSystemPorts = systemPorts" in html
    assert "publishFile(name, file)" in html
    assert "bundle · decoder-bundle :: decode" in html
    assert "logicalInputs[logicalName] = file.bytes" in html
    assert "required file input is not loaded:" in html
    assert "loadedFileLengths" in html


def test_html_accepts_host_system_external_reference_ports():
    # Host-system capability ports are the general support structure for a
    # compiled executor living inside this page by simulation: the shell
    # (not the program) owns whatever handler actually resolves each named
    # capability. Only "bundle" and "host_system" domains are accepted.
    html = emit_html_shell(_file_port_api(domain="host_system")).html
    assert "registerHostCapability(name, handler)" in html
    assert "resolveHostCapability(name, request)" in html
    assert "host_system · " in html


def test_html_still_rejects_other_external_reference_domains():
    with pytest.raises(ValueError, match="Turing bundles or"):
        emit_html_shell(_file_port_api(domain="guest_binary"))


@pytest.mark.skipif(
    not Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe").exists(),
    reason="Chrome is not installed",
)
def test_host_system_capability_registers_and_resolves_in_a_real_browser(tmp_path):
    html = emit_html_shell(_file_port_api(domain="host_system")).html
    injected = html.replace(
        "</body>",
        """
<script>
window.TuringSystemPorts.registerHostCapability("decoder", async (request) => {
  return { echoed: request.value * 2 };
});
window.TuringSystemPorts.resolveHostCapability("decoder", { value: 21 }).then(result => {
  document.body.setAttribute("data-test-result", "RESOLVED " + JSON.stringify(result));
}).catch(error => {
  document.body.setAttribute("data-test-result", "ERROR " + error.message);
});
</script>
</body>""",
    )
    page = tmp_path / "host_capability.html"
    page.write_text(injected, encoding="utf-8")
    completed = subprocess.run(
        [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "--headless=new", "--disable-gpu", "--no-sandbox",
            "--virtual-time-budget=5000", "--dump-dom", page.as_uri(),
        ],
        capture_output=True, text=True, timeout=30, check=True,
    )
    assert 'data-test-result="RESOLVED {&quot;echoed&quot;:42}"' in completed.stdout


@pytest.mark.skipif(
    not Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe").exists(),
    reason="Chrome is not installed",
)
def test_required_host_system_capability_fails_closed_without_a_handler(tmp_path):
    html = emit_html_shell(_file_port_api(domain="host_system")).html
    injected = html.replace(
        "</body>",
        """
<script>
window.TuringSystemPorts.resolveHostCapability("decoder", { value: 1 }).then(result => {
  document.body.setAttribute("data-test-result", "RESOLVED " + JSON.stringify(result));
}).catch(error => {
  document.body.setAttribute("data-test-result", "ERROR " + error.message);
});
</script>
</body>""",
    )
    page = tmp_path / "host_capability_unregistered.html"
    page.write_text(injected, encoding="utf-8")
    completed = subprocess.run(
        [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "--headless=new", "--disable-gpu", "--no-sandbox",
            "--virtual-time-budget=5000", "--dump-dom", page.as_uri(),
        ],
        capture_output=True, text=True, timeout=30, check=True,
    )
    assert 'data-test-result="ERROR required host-system capability has no simulation registered: decoder"' in completed.stdout


def test_html_shell_hydrates_persistent_mounts_and_bridges_virtual_devices():
    api = attach_shell_io(
        CompiledProgramAPI(
            "machine", "wasm", "run",
            (EntryPoint("run", "run", "control", ()),),
        ),
        ShellIOManifest(
            (
                ShellIORequest.create("files"),
                ShellIORequest.create("system_devices"),
            ),
            system_ports=(
                SystemPort.create(
                    "terminal_input", "device", "input",
                    attributes={"device": "console.input"},
                ),
                SystemPort.create(
                    "terminal_output", "device", "output",
                    attributes={"device": "console.output"},
                ),
            ),
            virtual_filesystem=VirtualFileSystemContract(mounts=(
                VirtualMount.create("/", "memory", access="read_write"),
                VirtualMount.create(
                    "/database", "indexed_db", access="read_write",
                    source="machine-runtime",
                ),
                VirtualMount.create(
                    "/origin", "opfs", access="read_write",
                    source="machine/runtime",
                ),
            )),
        ),
    )

    html = emit_html_shell(api).html

    assert 'indexedDB.open("turing-vfs:" + mount.source, 1)' in html
    assert "navigator.storage.getDirectory()" in html
    assert "systemPorts.ready = systemPorts.initializeVirtualFilesystem()" in html
    assert "await systemPorts.ready" in html
    assert "writeVirtualFileAsync(path, bytes)" in html
    assert "registerDeviceHandler(name, handler)" in html
    assert 'device === "console.input"' in html
    assert "runtime.injectDeviceBytes(device, bytes, options)" in html


@pytest.mark.skipif(
    not Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe").exists(),
    reason="Chrome is unavailable",
)
def test_persistent_virtual_mounts_execute_in_browser(tmp_path):
    api = attach_shell_io(
        CompiledProgramAPI(
            "storage_probe", "wasm", "run",
            (EntryPoint("run", "run", "control", ()),),
        ),
        ShellIOManifest(
            (ShellIORequest.create("files"),),
            virtual_filesystem=VirtualFileSystemContract(mounts=(
                VirtualMount.create("/", "memory", access="read_write"),
                VirtualMount.create(
                    "/database", "indexed_db", access="read_write",
                    source="browser-probe",
                ),
                VirtualMount.create(
                    "/origin", "opfs", access="read_write",
                    source="browser/probe",
                ),
            )),
        ),
    )
    probe = r"""<script>
(async () => {
  try {
    const ports = window.TuringSystemPorts;
    await ports.ready;
    await ports.writeVirtualFileAsync("/database/one.bin", new Uint8Array([1, 2, 3]));
    await ports.writeVirtualFileAsync("/origin/two.bin", new Uint8Array([4, 5, 6]));
    ports.virtualFiles.delete("/database/one.bin");
    ports.virtualFiles.delete("/origin/two.bin");
    await ports.hydrateIndexedDB(ports.virtualMount("/database/one.bin"));
    await ports.hydrateOPFS(ports.virtualMount("/origin/two.bin"));
    const left = Array.from(ports.readVirtualFile("/database/one.bin"));
    const right = Array.from(ports.readVirtualFile("/origin/two.bin"));
    document.body.textContent = JSON.stringify(left) === "[1,2,3]" &&
      JSON.stringify(right) === "[4,5,6]" ? "PASS PERSISTENT VFS" : "FAIL VALUES";
  } catch (error) { document.body.textContent = "FAIL " + String(error); }
})();
</script>"""
    (tmp_path / "index.html").write_text(
        emit_html_shell(api).html + probe, encoding="utf-8",
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.2)
        completed = subprocess.run(
            [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                "--headless=new", "--disable-gpu", "--no-sandbox",
                "--virtual-time-budget=10000", "--dump-dom",
                f"--user-data-dir={tmp_path / 'chrome-profile'}",
                f"http://127.0.0.1:{port}/index.html",
            ],
            capture_output=True, text=True, timeout=30, check=True,
        )
    finally:
        server.terminate()
        server.wait(timeout=5)
    assert "PASS PERSISTENT VFS" in completed.stdout


def test_published_webgl_shader_graduates_page_to_execution_surface():
    shell = shell_for_artifact(
        _artifact(),
        shader_execution={
            "url": "source/webgl/webgl.frag.glsl",
            "language": "webgl2-glsl-es",
            "stage": "fragment",
            "role": "shader-surface",
            "autostart": True,
            "execution": {"continuous": True, "prefer_contiguous": True},
        },
    )
    html = shell.html

    assert '<canvas id="shader-surface" tabindex="0"' in html
    assert 'width: 100%;' in html
    assert 'height: 100%;' in html
    assert 'canvas.getContext("webgl2"' in html
    assert 'fetch(new URL(activeCandidate.url, document.baseURI)' in html
    assert "canvas.setPointerCapture(event.pointerId)" in html
    assert 'canvas.addEventListener("keydown"' in html
    assert "window.TuringShaderSurface" in html
    assert "window.TuringShaderLiaison" in html
    assert "window.TuringWasmRuntime" in html
    assert "outputFrame()" in html
    assert "uploadOutputTexture()" in html
    assert "channels[0][index] * channelScale" in html
    assert "wasm: window.TuringWasmRuntime || null" in html
    assert "start({continuous = true, preferContiguous = true} = {})" in html
    assert "preferContiguous && contiguousRunner" in html
    assert "liaison.wasm.start({" in html

    # The opaque shader overlays the inspector without display:none, because
    # the browser must still calculate the hidden document's real layout.
    assert '<body class="shader-execution">' in html
    assert "position: fixed" in html
    assert "z-index: 2147483647" in html
    assert "display: none !important" not in html
    assert 'schema: "turing-dom-layout"' in html
    assert "element.getBoundingClientRect()" in html
    assert "view.getComputedStyle(element)" in html
    assert "new Float32Array(elements.length * 8)" in html
    assert "await settledLayout(document)" in html
    assert "async loadHTML(source" in html
    assert "async loadFile(file)" in html
    assert 'canvas.addEventListener("drop"' in html
    assert "snapshot.viewport.height - (element.y + element.height * 0.5)" in html
    assert "Local publisher and gallery" in html
    assert "Process graph" in html
    assert 'id="run"' in html
    assert 'id="canvas"' in html


def test_canvas_presentation_needs_no_published_shader_url():
    shell = shell_for_artifact(
        _artifact(),
        shader_execution={
            "url": None,
            "language": "canvas2d",
            "stage": "none",
            "role": "shader-surface",
            "autostart": True,
            "configuration": {
                "channels": ["red", "green", "blue"],
                "channel_scale": 255.0,
            },
        },
    )

    assert '"language": "canvas2d"' in shell.html
    assert "channels[0][index] * channelScale" in shell.html
    assert "context2d.imageSmoothingEnabled = false" in shell.html
    assert "0, 0, canvas.width, canvas.height" in shell.html


def test_the_emitted_source_travels_with_the_page_for_reading():
    shell = shell_for_artifact(_artifact())
    assert "f64.sub" in shell.html
    assert "f64.abs" in shell.html


def test_no_component_or_third_party_layout_engine_creeps_in():
    """Layout belongs to a different subrepo. This is a stack of labelled
    rows; if it grows a grid engine or a component model, it should be handed
    over rather than extended here."""

    html = shell_for_artifact(_artifact()).html
    assert "<table" not in html
    # The live graph uses the browser's native CSS grid, not a JS layout
    # engine or component framework.
    assert "process-graph-grid" in html
    # And no third-party anything: the page must open with no network.
    # The one URL is the explicitly configurable loopback publisher, not a
    # third-party dependency. The page still carries all of its own UI code.
    without_local_server = html.replace("http://localhost:8787", "")
    assert "http://" not in without_local_server and "https://" not in html
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


def test_local_publisher_uses_configurable_server_and_bundle_resource_route():
    html = emit_html_shell(
        _artifact().api,
        resource_route="/site/programs/demo/versions/v1-abc/",
        static_gallery=[{"slug": "demo", "url": "site/programs/demo/index.html"}],
    ).html

    assert 'id="server-address" value="http://localhost:8787"' in html
    assert 'id="python-source"' in html
    assert 'id="generate-page"' in html
    assert 'id="gallery"' in html
    assert 'fetch(serverURL("/api/generate")' in html
    assert 'fetch(serverURL("/api/gallery")' in html
    assert 'const RESOURCE_ROUTE = "/site/programs/demo/versions/v1-abc/"' in html
    assert 'const STATIC_GALLERY = [{"slug": "demo"' in html
    assert 'renderGallery(STATIC_GALLERY)' in html
    assert 'const itemURL = item => fromServer ? serverURL(item.url) : resourceURL(item.url)' in html
    assert "function resourceURLs(path)" in html
    assert "add(relative);" in html
    assert 'const siteIndex = window.location.pathname.indexOf("/site/")' in html
    assert "add(serverURL(pageDirectory + relative))" in html
    assert 'add(serverURL("/" + relative))' in html
    assert "resource not found in any configured location" in html
    assert 'const pagesPrefix = routeIndex >= 0' in html
    assert "fetchResource(descriptor.url)" in html
    assert "const programs = new Map()" in html
    assert "versions.find(item => item.latest) || versions[0]" in html
    assert 'selector.className = "gallery-version"' in html
    assert "program(s) · " in html


def test_html_shell_reports_class_lut_and_exposes_semantic_navigation_methods():
    map_ir = {
        "class_navigation": {
            "classes": [{
                "identity": "Vault",
                "permissions": ["vault:enter"],
                "instantiation_functions": [4],
                "members": [{
                    "name": "read",
                    "identity": "Vault.read",
                    "kind": "method",
                    "storage": None,
                    "function_reference": 7,
                    "permissions": ["vault:read"],
                }],
            }],
        },
        "semantic_methods": [
            {"function": "turing.class.resolve_member", "operations": ["Const", "Eq", "And", "LAnd", "Select", "Ret"]},
        ],
    }

    html = emit_html_shell(_artifact().api, map_ir=map_ir).html

    assert "Class map and navigation LUT" in html
    assert 'id="class-map"' in html
    assert "Vault.read" in html
    assert "turing.class.resolve_member" in html
    assert "const MAP_IR =" in html
    assert "window.TuringClassNavigation" in html
    assert "resolveClassMember" in html
    assert "resolveClassInstantiation" in html
    assert "class navigation requires a permission evaluator" in html


def test_callable_run_systems_are_grouped_and_packed_into_hidden_div_tabs():
    map_ir = {
        "callable_systems": {
            "classes": [{
                "identity": "Curve",
                "methods": [
                    {
                        "identity": "Curve.derive",
                        "name": "derive",
                        "signature": "derive(self, iterations: int)",
                        "parameters": [
                            {"name": "self", "role": "input", "dtype": "Curve", "passing": "value"},
                            {"name": "iterations", "role": "input", "dtype": "int", "passing": "value"},
                        ],
                        "page_url": "/curve/derive/index.html",
                        "function_reference": 3,
                    },
                    {
                        "identity": "Curve.trace",
                        "name": "trace",
                        "signature": "trace(self, iterations: int)",
                        "parameters": [],
                        "page_url": "/curve/trace/index.html",
                        "function_reference": 4,
                    },
                ],
            }],
            "file_scope": {
                "functions": [{
                    "identity": "helper",
                    "name": "helper",
                    "signature": "helper(value)",
                    "parameters": [],
                    "page_url": "/helper/index.html",
                    "function_reference": 5,
                }],
                "symbols": [{
                    "name": "SCALE",
                    "kind": "binding",
                    "expression": "SCALE = 2.0",
                }],
            },
            "functions": [{
                "identity": "helper",
                "name": "helper",
                "signature": "helper(value)",
                "parameters": [],
                "page_url": "/helper/index.html",
                "function_reference": 5,
            }],
        }
    }

    html = emit_html_shell(_artifact().api, map_ir=map_ir).html

    assert "Callable run systems" in html
    assert html.index("file scope") < html.index("class Curve")
    assert 'class="callable-owner-tab"' in html
    assert 'class="callable-owner-view" data-callable-owner-view="file-scope"' in html
    assert 'data-callable-owner-view="class-Curve" hidden' in html
    assert 'class="callable-tab"' in html
    assert 'class="callable-tabview" data-callable-view="Curve-trace" hidden' in html
    assert 'data-callable="Curve.derive"' in html
    assert 'id="callable-Curve-derive-iterations"' in html
    assert 'data-callable-view="file-symbols" hidden' in html
    assert "SCALE = 2.0" in html
    assert "Open generated callable page" in html
    assert "function wireCallableTabs()" in html
    assert "activateOwner" in html
    assert 'data-class-map-view="raw" hidden' in html
    assert "function wireClassMapTabs()" in html


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
    is what makes a dead shell announce itself instead of looking inert.
    (A third, later tag drives the always-present text transcript and is
    independent of this boot/program pair.)"""

    html = shell_for_artifact(_artifact()).html
    assert html.count("<script>") == 3
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
    assert "window.TuringSharedClassMemory" in html
    assert "turingStorageReference" in html
    assert "residentOutputs" in html
    assert "redirectStorageOffset" in html
    assert "lastExecutionMs" in html
    assert "residentValues" in html
    assert "queueDeploymentProfile" in html
    assert "requestAnimationFrame" in html
    assert "WebAssembly.compile(moduleBinary)" in html
    assert "WebAssembly.instantiate(module, imports)" in html
    assert "No live tensor is copied through" in html
    assert "shared-memory slot" in html
    assert "Live deployment schedule:" in html
    assert "await fetchResource(spec.url)" in html
    assert "one element per call today" not in html


def test_parallel_wasm_plan_uses_aligned_bounded_worker_tiles_and_join():
    from src.compiler.wasm_html_shell import emit_html_shell

    artifact = _artifact()
    class_graph = {
        "modules": [{
            "name": "lane_0", "wasm_base64": "AGFzbQEAAAA=",
            "entry": "kernel", "inputs": ["x"], "outputs": ["y"],
            "value_type": "f64", "element_bytes": 8,
            "shared_memory_import": {"module": "env", "field": "memory"},
        }],
        "edges": [], "logical_inputs": {"feed0": [["lane_0", "x"]]},
        "logical_outputs": {"result": ["lane_0", "y"]},
        "shared_memory": True, "shared_static_bytes": 0,
        "class_inventory": {
            "field_slots": [{"index": 0, "key": "in::feed0"},
                            {"index": 1, "key": "out::lane_0::y"}],
            "storage_redirects": [],
            "methods": [{"index": 0, "module": "lane_0", "entry": "kernel",
                         "input_slots": [0], "output_slots": [1]}],
        },
        "coordinator": {"entry": "run_range", "method_count": 1},
        "thread_deployment": {
            "abi": "turing.wasm-thread-deployment.v1",
            "tile_alignment": 8, "tiles_per_worker": 2,
            "root": {"kind": "deploy", "scale": 1,
                     "join": {"mode": "barrier"},
                     "lanes": [{"kind": "call", "method": 0}]},
        },
    }
    html = emit_html_shell(artifact.api, class_graph=class_graph).html

    assert "navigator.hardwareConcurrency" in html
    assert "Math.min(taskCount, 8" in html
    assert "tile_alignment" in html
    assert "new Worker(this.tileWorkerURL)" in html
    assert "executeThreadDeployment" in html
    assert "vertically fused WebAssembly tiles" in html
    assert "window.TuringWasmThreads" in html
    assert 'contract.extent_effect !== "collective"' in html
    assert "whole-extent Wasm coordinator retained" in html
    assert "await Promise.all" in html
    assert "Join: all WebAssembly tiles committed" in html
    assert "replaying serial Wasm schedule" in html


@pytest.mark.skipif(
    not Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe").exists(),
    reason="Chrome is unavailable",
)
def test_wasm_tile_worker_executes_kernel_and_crosses_join_in_browser(tmp_path):
    from src.compiler.fused_program_wasm_backend import emit_wasm_module
    from src.compiler.wasm_binary import WasmImport
    from src.compiler.wasm_html_shell import _JS

    program = FusedProgram(
        version=1,
        feeds={1},
        steps=[OpStep(0, "mul", [1], {"right_scalar": 2.0}, 2)],
        outputs={"result": 2},
    )
    module = emit_wasm_module(
        program,
        name="tile_kernel",
        imports=(WasmImport(
            module="env", field="memory", kind="memory", memory_pages=1,
        ),),
    )
    worker_function = _JS.split(
        "function wasmTileWorkerSource() {", 1
    )[1].split("class ClassGraphRunner", 1)[0]
    worker_function = "function wasmTileWorkerSource() {" + worker_function
    encoded = base64.b64encode(module.binary).decode("ascii")
    page = f"""<!doctype html><body>WAIT<script>
{worker_function}
const manifest = {{
  modules: [{{name: "lane", entry: "tile_kernel", value_type: "f64",
    element_bytes: 8, wasm_base64: "{encoded}",
    shared_memory_import: {{module: "env", field: "memory"}}}}],
  shared_static_bytes: 0
}};
const inventory = {{field_slots: [{{index:0}}, {{index:1}}], methods: [{{
  index: 0, module: "lane", entry: "tile_kernel",
  input_slots: [0], output_slots: [1]
}}]}};
const url = URL.createObjectURL(new Blob([wasmTileWorkerSource()], {{type:"text/javascript"}}));
const worker = new Worker(url);
worker.onmessage = event => {{
  if (event.data.type === "configured") {{
    worker.postMessage({{type:"run", taskId:1, methodIds:[0], count:3,
      fields:{{0:new Float64Array([1,2,3])}}, resultSlots:[1]}});
    return;
  }}
  const values = Array.from(event.data.outputs[1] || []);
  document.body.textContent = !event.data.error && JSON.stringify(values) === "[2,4,6]"
    ? "PASS JOIN [2,4,6]" : "FAIL " + JSON.stringify(event.data);
  worker.terminate();
}};
const raw = atob("{encoded}");
const bytes = new Uint8Array(raw.length);
for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
WebAssembly.compile(bytes).then(compiled => worker.postMessage({{
  type:"configure", manifest, inventory, compiledModules:[["lane", compiled]]
}}));
</script></body>"""
    page_path = tmp_path / "worker.html"
    page_path.write_text(page, encoding="utf-8")
    completed = subprocess.run(
        [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "--headless=new", "--disable-gpu", "--no-sandbox",
            "--virtual-time-budget=5000", "--dump-dom", page_path.as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert "PASS JOIN [2,4,6]" in completed.stdout


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
    assert "await fetchResource(descriptor.url)" in html


def test_sympy_mathematics_is_rendered_separately_from_lazy_sources():
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
            "depiction": {
                "kind": "function", "name": "demo",
                "inputs": ["x", "parameter"], "outputs": ["result"],
            },
            "outputs": [{
                "name": "result", "node_id": 5,
            }],
            "url": "site/v3/math/render/sympy-process-model.json",
        },
    ).html

    assert "Math is programming is math" in html
    assert "Deterministic numeric map" in html
    assert "<mi>demo</mi>" in html
    assert "<mi>result</mi>" in html
    assert "site/v3/math/render/sympy-process-model.json" in html
    assert "Download exact SymPy model" in html
    assert "await fetchResource(MATHEMATICS.url)" in html
    assert "relation 1" not in html
    assert "math-equations" not in html


@pytest.mark.parametrize(
    ("kind", "heading"),
    [
        ("predicate", "Boolean predicate"),
        ("transition", "State transition"),
        ("relation", "Implicit relation"),
    ],
)
def test_sympy_chalkboard_uses_the_program_semantic_shape(kind, heading):
    html = emit_html_shell(
        _artifact().api,
        mathematics={
            "target": "sympy",
            "node_count": 1,
            "equation_count": 1,
            "constraint_count": 0,
            "uninterpreted": [],
            "depiction": {
                "kind": kind, "name": "demo",
                "inputs": ["x"], "outputs": ["result"],
            },
            "url": "sympy-model.json",
        },
    ).html

    assert heading in html
    assert "Download exact SymPy model" in html
    assert "math-equations" not in html


def test_graph_phosphor_integrates_profile_pulses_with_decay():
    html = shell_for_artifact(_artifact()).html
    assert '<canvas id="process-graph-canvas"' not in html
    assert "process-graph-grid" in html
    assert "graph-indicator" in html
    assert 'return "graph-node-" + viewName' in html
    assert "function drawProcessGraph" not in html
    assert "function phosphorColor" not in html
    assert "previousEnergy * Math.exp" in html
    assert 'element.style.setProperty("--node-opacity"' in html
    assert 'element.style.setProperty("--node-scale"' in html
    assert 'element.style.setProperty("--node-blur"' in html
    assert "@keyframes graph-phosphor" not in html
    assert "data-pulse" not in html
    assert "transition:" not in html
    assert "animation:" not in html
    assert "graph-profile-stats" in html
    assert 'root.style.setProperty("--profile-median-ms"' in html
    assert 'root.style.setProperty("--profile-normalizer-us"' in html
    assert "perNodeUs / normalizerUs" in html
    assert "phosphor scale p95" in html
    assert "document.getElementById(graphIndicatorId(activeGraphView, nodeId))" in html
    assert 'event.target.closest(".graph-indicator")' in html
    assert 'id="graph-decay"' in html or "graph-decay" in html
    assert 'id="graph-edges"' not in html
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


def test_transcript_is_present_and_linked_regardless_of_shader_or_graph():
    """The plain-text transcript is not gated behind the shader body class or
    a supplied process graph -- a bare inspection page still gets one, and a
    reader following raw HTML (no JS, no canvas) can reach every node."""

    html = emit_html_shell(_artifact().api).html
    assert 'id="program-transcript"' in html
    assert 'class="shader-execution"' not in html
    assert '?node=graph-index' in html
    assert '?node=log' in html
    assert '?node=network' in html
    assert '?node=shader' in html
    assert "No shader execution surface attached" in html


def test_transcript_graph_nodes_link_to_their_parents():
    """The process graph's own parent edges are the navigable structure --
    each node section links every parent by the same ?node= scheme."""

    graph = {
        "nodes": 2,
        "edges": 1,
        "truncated": False,
        "histogram": {"add": 1, "sub": 1},
        "table": [
            {"id": 1, "type": "sub", "label": "left - right", "parents": []},
            {"id": 2, "type": "add", "label": "abs(...) + 1", "parents": [1]},
        ],
    }
    html = emit_html_shell(_artifact().api, process_graph=graph).html
    assert 'data-node="graph-1"' in html
    assert 'data-node="graph-2"' in html
    assert '<a href="?node=graph-1">' in html
    assert '<a href="?node=graph-1">node 1 (sub)</a>' in html


def test_transcript_telemetry_renders_as_readable_log_text():
    """Build-time telemetry records show up as plain list text in the
    transcript, not only as JSON handed to the diagnostics script."""

    telemetry = {
        "records": [
            {"kind": "log", "message": "compiled entry", "path": "backend_sources"},
            {"kind": "error", "message": "boom", "path": ""},
        ]
    }
    html = emit_html_shell(_artifact().api, telemetry=telemetry).html
    assert "[log] compiled entry (backend_sources)" in html
    assert "[error] boom" in html


def test_transcript_survives_with_no_optional_data_at_all():
    """Every input the transcript reads is optional; omitting all of them
    must still produce a coherent, non-crashing transcript."""

    html = emit_html_shell(_artifact().api).html
    assert 'id="program-transcript"' in html
    assert "No feedback network attached" in html


def test_machine_snapshot_liaison_has_bounded_live_transport_and_input_port():
    html = emit_html_shell(
        _artifact().api,
        shader_execution={
            "url": "display.frag.glsl", "language": "webgl2-glsl-es",
            "stage": "fragment", "role": "shader-surface",
        },
    ).html

    assert 'connect(endpoint = "/snapshot", options = {})' in html
    assert '"after=" + this.generation' in html
    assert 'cache: "no-store"' in html
    assert "this.disconnect();" in html
    assert "async sendTerminalInput(value)" in html
    assert "async sendControl(action, value = null)" in html
    assert "async loadSubject(value)" in html
    assert 'String(options.inputEndpoint || "/input")' in html
    assert 'String(options.controlEndpoint || "/control")' in html
    assert 'String(options.subjectEndpoint || "/subject")' in html

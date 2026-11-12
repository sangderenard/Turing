# Internal captured Studio Phong execution

`StudioAgentProducts.build` accepts a new explicit `render-captured` operation
when trusted startup supplies `phong_host=StudioPhongHost(...)`. This runs the
existing `standalone_studio.js` full-scene renderer, including its captured room,
fixtures and subject. It returns an integrity-checked `image/png` product in the
same base64 envelope as the existing scene/ray product transport.

```python
from pathlib import Path
from meowdoku.studio_agent import StudioAgentProducts, product_bytes
from meowdoku.studio_phong import StudioPhongHost

host = StudioPhongHost(Path("trusted-full-studio.scene-asset.json"),
    chrome=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
products = StudioAgentProducts(Path("assets.json"), phong_host=host)
# Use this callback only in the existing authenticated StudioProductDispatcher.
# Camera matrices are column-major, with an affine view; both are explicit.
product = products.build({"schema": "turing.studio-agent-request.v1",
    "operation": "render-captured", "camera": camera})
png = product_bytes(product, content_type="image/png")
provenance_and_limits = host.describe()
```

`scene` and `ray-payload` are unchanged. A `render-captured` request accepts only
schema, operation and camera. It cannot replace the host's scene, executable,
shader, URL, profile, or module paths, and it rejects settings instead of silently
rendering an unrelated captured baseline. The enclosing service still must check
the caller's account and authorization before dispatch. The result is bytes,
not an owned PhotoJobs job. `commit_studio_ray` rejects PNG products.

## Execution and provenance

The host reads a trusted full-scene asset and the exact six existing transitive
renderer modules once at startup. The scene hydrator verifies the existing
content identity and validates geometry, material tables, shader declarations,
uniforms, resource state and triangle ownership before drawing. Actual captured
vertex and fragment GLSL strings are compiled unchanged by WebGL2. `describe()`
reports their SHA-256 hashes, scene bytes/identity, module hashes and limits.
There is no new GLSL copy, translation, alternate shader or independent draw loop.

One render is admitted at a time per host instance. Each render uses a fresh
headless Chrome profile, a temporary loopback server with a random 256-bit path
capability, and only allowlisted renderer modules and in-memory input bytes.
There is no directory listing or general file server. Exact Host and POST Origin
checks reject cross-origin requests; CSP restricts scripts/connections to this
origin. The instance does not attach to a user's browser or use their cookies.
Chrome uses SwiftShader explicitly, leaving the existing native GPU job owner
alone. No browser is installed or downloaded automatically.

Input scene bytes are limited to 32 MiB, output to 32 MiB, frame dimensions to
16..2048, and runtime to a trusted 0..120-second timeout (60 seconds by default).
The host disposes the loopback server, profile and its own browser process tree.
PNG dimensions and returned scene/camera identity are checked before returning.
This is bounded trusted-code execution, not a sandbox for untrusted GLSL assets.

## Honest limits and remaining integration

- The existing captured-scene renderer rejects active lightfield, shadows and
  thermal resources. It does not reproduce the new live secondary/DOF passes.
- Rendering uses a fixed captured scene with explicit cameras, not arbitrary
  authoring settings, live simulation, video encoding or UI screenshot capture.
- Software-WebGL executes the same shaders, but cross-driver bitwise equality
  with the user's live hardware renderer is not claimed.
- The plain Node command remains scene/ray preparation. The optional Python
  product host is the executable internal API seam; the generic authorized Nodus
  dispatcher can carry its PNG envelope without a new transport.
- `StudioPictures` and its HTTP route remain ray-job-only. They do not accept
  `render-captured`, and no account artifact store, public endpoint, service
  startup configuration or production deployment has been added.
- To support arbitrary settings faithfully, extract the actual live Studio
  frame's uniform/resource binding and draw orchestration into a shared module,
  including the existing scene builders and new field/DOF consumers. Merely
  copying GLSL does not supply those bindings. Do not replace that work with
  fabricated captured uniforms or a reduced subject-only renderer.

## Measured validation

`STUDIO_PHONG_GL_CHECK=1` enables the actual local Chrome test in
`tests/test_meowdoku_studio_phong.py`. On Windows, two 320x320 camera renders and
two contract tests passed in 8.96 seconds. The first frame's PNG SHA-256 was
`906d8b20548ee95467f7e9a3e2c8240c79e25d6039f8e86fabcdc2781b261d96`, identical
to the earlier standalone runner receipt, and the shifted-camera PNG differed.
Visual inspection of that reference PNG shows a dark exterior room wall at its
recorded camera. This establishes captured-renderer replay, not a usable subject
composition or parity with the current live Studio frame. Choose a meaningful
explicit camera when consuming this captured asset; no walls were removed to
make the validation image look better.
The captured source asset identity was
`studio-scene:131e3cfc8de620cb8aea2d89a49cb7a0364eb83f1a5c3c3544aaa2617d4da44f`.
The actual shaders were vertex `ed478595ec3ff33c8cfd9375522ec306463f284879034ef64aab2493d7c9c0ec`
and fragment `e34c45ce966fc789d43b1ba8212baa417a779617fda8d0777b69a87a06783eef`.
The six existing `test_meowdoku_studio_agent.py` tests also passed, including
real Python-to-Node scene preparation and unchanged ray-job authorization.
The final rerun passed all four host tests, including loopback security/cleanup,
in 5.97 seconds. The generic Nodus adapter tests passed 9 with 1 optional native
test skipped in 0.33 seconds. Production Nodus/HTTP rollout was not tested.

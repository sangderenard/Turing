# Explicit captured Studio Phong execution, source only

The current user explicitly requested self-hosted Phong rendering. This is a
different product boundary from the earlier settings-to-scene-only request
recorded in AGENT_OBSTINANCE. Existing scene/ray operations remain unchanged.

Added `StudioPhongHost` and an optional `render-captured` product operation.
The host uses the unchanged full-scene `standalone_studio.js` renderer and exact
captured GLSL, uniform/material tables and scene ownership. It does not invent a
new renderer or execute user-supplied shaders, URLs or paths. Trusted startup
selects one full-scene asset and Chrome binary; requests supply camera only.
Rendering uses isolated software WebGL, not the photo service's native GPU.

Measured on local Windows: actual Chrome generated two different 320x320 PNGs
from the two explicit cameras in 8.96 seconds including two contract tests.
First PNG SHA-256 `906d8b20548ee95467f7e9a3e2c8240c79e25d6039f8e86fabcdc2781b261d96`
matches the earlier standalone runner receipt. Six existing Studio agent tests
passed, including Python-to-Node scene preparation. Dedicated tests also check
camera bounds, explicit opt-in, rejected settings/path/shader fields, PNG refusal
at ray commit, loopback capability/Host/Origin restrictions and server cleanup.

This is not arbitrary settings-to-pixels, active-field/DOF replay or video.
The existing capture contract rejects active field/shadow/thermal state. It is
not integrated into the ray-only StudioPictures HTTP route and saves no account
artifact. The existing authorized generic Nodus dispatcher can transport the
PNG product; startup configuration/ownership integration remains host work.
No production restart or deployment occurred. See docs/STUDIO_PHONG_HOST.md for
source hashes, invocation, execution limits and the required future shared live
draw-binding extraction. Do not represent this bounded captured scene as a render
of user settings it does not consume.

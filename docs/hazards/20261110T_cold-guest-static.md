# Cold guest shader export

Build: node tools/export_meowdoku_guest.mjs [new-empty-output-directory]
Verify: node tests/meowdoku_guest_static_check.mjs [output-directory]
Default output: build/meowdoku-guest-static.

Serve only this isolated directory as static files. It contains index.html, the original loading-panel renderer and complete decoder dependency closure, loading_anim.mla, and a SHA-256 manifest. Never serve the parent build/meowdoku folder: it contains private server state. No Python, PHP, access.json, database, API proxy, credentials or server token belongs in the publication directory. JavaScript modules need normal JavaScript MIME handling. WebGL 2 is required. Browsers still compile their own GLSL programs at load; the animation data is prebaked, not a server-side simulation.

Standalone mode skips access.js and readiness/animation API requests, ignores animation/phase URL controls and parent steering, and does not install the game readiness API or fetch instrumentation. Same original shaders, interpolation, seam and drawing code. No time-of-day/Studio light baking is relevant to this guest loading animation.

Verification on 2026-11-10: existing loading-panel timing regression passed. Static test passed exact file inventory/hashes, dependency closure, import with network forbidden and decoded packet (265 frames, 256 grains, 12 fps). Chrome cold-loaded from a separate loopback Node file server with all APIs denied. Requests: index.html, six JS modules, loading_anim.mla, favicon.ico (404); zero API requests. Full-screen animation visibly rendered, zero controls and hidden caption. After stopping that file server, another screenshot showed advanced particle positions: playback continues without a server. Main Python service and original user Studio page were untouched. No Apache configuration or public deployment performed. Artifact is ready for an isolated Apache document root.

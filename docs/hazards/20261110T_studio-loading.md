# Studio reuses server loading panel

Marble Studio imports the existing loading_panel.js directly. Its boot hold and fetch counter cover shader/material downloads, then boot enters its initializing phase. Finally releases after first frame/setup or caught failure, including absent WebGL. Renderer, animation selection, codec, physics and sessions unchanged.

Source and served bootstrap patched independently; no wholesale export. Only served marble_studio.js replaced after byte guard. Backup and receipt: build/dot-studio-loading-20261110. No server restart.

Both syntax checks and VM no-WebGL/download/material-failure lifecycle checks pass. First-frame release ordering assertion and existing stage timing test pass. HTTPS localhost unreachable during verification: server delivery and browser end-to-end remain unverified. Covers initial boot only, not later object loads/background exposures. Python codec tests launched but tool failed before completion could be collected.

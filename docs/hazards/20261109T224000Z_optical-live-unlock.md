# Optical Live admission and visible errors

New designs defaulted to Glass rod, but both createLiveRayView and decode_upload still contained the earlier PHOTO-FIRST performance/QA rejection. This prevented every new optical design from starting Live before any network request, while the reason appeared below a long controls block. The user explicitly requested unlocking both optical modes.

Removed those explicit client/server optical-mode gates. This does not convert materials or remove validation: Live now receives the same saved region_media, optical_media and analytic sphere data through the shared accumulator path. The existing single-job GPU ownership and actual error/invalid-image diagnostics remain. Snapshot-before-exposure remains still-only; Live still uses the normal saved immutable design.

The live-status output now follows its toggle immediately and is its accessible description. Unexpected start exceptions and unreadable server responses show a useful reason including HTTP status. Stale photo-only labels changed in the mode selector and optical panel.

Verification: 81 Python sphere/live tests pass; actual live UI lifecycle tests pass with both optical modes admitted, malformed HTTP error preserved, and status placed before other settings. Optical authoring/control suite passes. Dedicated actual GPU rod+dye live/revision tests are owned by material worker. Do not claim production delivery until build/dot-live-optical-unlock-20261109/verified.json exists. Scope is three static assets and the backend admission change, with pinned publication hashes and an idle-only retained-products detached restart.


Released on explicit user direction without retaining the optional performance/QA gate. Actual GPU live smoke reached scene setup but its tool run was interrupted before a frame; this is unverified, not a transport failure or a pass. Subsequent snapshot/photo regression12tests also passed. Trusted HTTPS verified3 changed assets and226unchanged; server PID53476 boot775d6760ce04. Idle check found no active photo before restart. Initial default detached CreateProcess returned WinError5; explicit approved escalated execution of the exact retained launcher succeeded. No security settings changed. Material worker continues isolated live frame testing and owns subsequent unified editor edits.

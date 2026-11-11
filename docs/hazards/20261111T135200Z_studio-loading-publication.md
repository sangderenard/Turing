2026-11-11 STUDIO LOADING AND SHARED SQL ISSUE HEADER (Windows, measured):

Apache access log for the user's 07:32 local Studio request recorded 404 for
marble_imported_object.js, marble_chase_camera.js, marble_studio_phong.js and
marble_lighting_racks.js. Recursive inspection found marble_studio_transition.js
also absent. The relative import URLs were correct. One existing shop-light
module lacked the box-face export consumed by the scene builder's rack module.
The aborted module graph also prevented the imported loading panel from starting.

After restoring those dependencies in an isolated fixture, actual Chrome boot
reported LIGHTING_PROFILE_FIELDS is not defined. Studio's initial scene-builder
capture used that existing authored field list without importing it. The source
and deployed bootstrap now import it beside setupStudioLighting.

Published 10 narrowly scoped static files in build/meowdoku/web without a service
restart. Publication was byte-guarded, with before-copies and hashes in
build/dot-studio-load-20261111/publication-receipt.json. Existing shop-light models,
limits and controls were preserved; only its already-authored box-face export
was added to the deployed copy. No renderer, shader, database, auth, network or
TLS implementation changed. Added the missing transitive modules to the explicit
web_export.py copy list, with an import-closure regression test.

The existing shared header is now v4, with the existing SQL issue page/assets.
Its Report issue always opens /issues.html so a frozen game cannot redirect it
to the legacy file-ticket form. The game's own capture form is not rewritten.

Verified: all 94 static modules link using Node's real ES-module linker. Chrome
against a read-only localhost fixture rendered the actual 72,960-triangle Pluck
Studio scene; its loading panel became visible and released normally. The
existing recorded particle animation was visibly rendered in loading_demo mode.
Header account/Tickets/Report issue appeared; Report issue opened the SQL issue
page, and Cancel/reopen worked. Account/API responses in this fixture were mocks;
no production sessions, SQL or HTTP were used. UI errors from a browser ad-block
extension were observed separately; no final Studio error remained.

Focused Python: 16 tests and 10 subtests pass (issue ownership/HTTP plus static
export closure). Apache fixture: 12 tests pass when MEOWDOKU_TEST_HTTPD points to
the installed httpd; a combined run without that setting skipped those 12.
Node: loading stage timing, control-suite lifecycle, hidden Phong partition,
initial lighting field capture and Studio failure-release checks pass. The older
loading test failed before changes solely because it required a comment at byte
zero despite a newly prepended scene-builder import; it now checks the comment
exists and retains all lifecycle assertions. Syntax and git diff --check pass.
No full-suite claim.

Apache automatic header injection is prepared and tested separately, not active:
see deploy/apache/SITE_NAV.md and the Apache-specific hazard receipt. Live
C:/Apache24/conf/meowdoku-proxy.conf was not modified. Public service HTTP check
was denied by network policy; the Windows active-worker CIM read was unavailable.
No alternate route was attempted. Since the sanctioned stop script can force-kill
remaining workers, no live reload was attempted without a safe lifecycle check.
Public HTTPS, real-admin end-to-end and Apache-wide activation remain unverified.

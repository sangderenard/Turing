2026-11-10 GUARDED PRIVATE LAYER UPGRADES (MEASURED, offline Windows).

Extended only tools/publish_studio_game_layer.py and its focused test file.
Default publication still refuses differing existing bytes. Explicit
--prior-publication accepts a successful HTTPS-verified receipt only when its
manifest matches its adjacent frozen stage receipt and every prior staged file
still has its recorded hash. New receipts also bind the stage-receipt bytes.
Legacy pose-stage receipts remain supported. An upgrade must use a separate
fresh stage, and only installed hashes equal to that prior manifest can be
replaced. Primary marble_studio.js/html and the original game are never targets.
Original game bytes are now verified using original_game_path when present or
the reviewed canonical newer-game source for legacy stage receipts.

All staged bytes, primary Studio sources, original game, prior receipt/stage,
backup hashes and target hashes are rechecked immediately before each install.
Replaced bytes are exclusively written and fsynced under the new stage's
publication-backup directory before any live write. A durable write-ahead
publication-in-progress.json records planned hashes and backups. Updates use
same-directory atomic replace; additions use exclusive hard links so a file
created after the absence check cannot be overwritten. The entry is last.
Failed installation/HTTPS verification restores only our still-matching written
hashes and removes only our still-matching additions. Conflicting work is left
in place, with rollback conflicts/errors in publication.json and stdout. Receipt
write failure is also reported to stdout; no automatic retry is attempted.

Trusted HTTPS remains mandatory. --system-trust uses system roots;
--ca-file uses an explicit CA bundle. The previous private-CA default remains.
No TLS/security setting, server lifecycle, live file, frozen stage, commit or
push was changed. This worker did not execute a live publication.

Verification: original publisher tests passed 3/3 before editing. Final focused
python -m pytest -q tests/test_studio_game_layer_publish.py passed 33/33 in
1.95 seconds. Tests cover upgrade, prior/stage/installed tampering, missing
prior, source/manifest/backup changes during preparation, failed verification,
partial-mutation failure, concurrent replacement/create, rollback conflict,
unrelated-file preservation, temporary-cleanup and receipt-write failures,
forbidden destinations and both trust options. CLI --help and scoped git diff
--check passed. Full repository tests, browser QA and live HTTPS were not run.
The hazards helper still rejects existing malformed
20261109T_material-library-paused-checkpoint.md; newest ten receipts and the
legacy baseline were read directly before testing.

Limits: these local receipts are integrity/provenance evidence, not signed
attestations against someone controlling the entire stage. Per-file operations
are atomic, but the bundle is not a multi-file transaction; already-open pages
can observe mixed dependency generations. Hash check followed by replacement
or rollback is not an OS compare-and-swap: non-cooperating writers must be
quiesced for publication. Rechecks catch observed changes and the rollback tests
prove those are preserved, but do not promise safety against an adversarial
write in the final syscall gap. Process/host crashes do not automatically roll
back; durable backups and journal require deliberate recovery. Directory fsync
is unavailable on Windows, though all backup and temporary file bytes are
flushed with fsync. A denied temporary cleanup may leave a private random temp
file, reported as the publication failure; installed layer tracking still
allows rollback. Existing empty directories are not removed during rollback.

Final review correction: the host is Python 3.11, where Path.is_junction does
not exist. Path validation now uses lstat st_file_attributes and
stat.FILE_ATTRIBUTE_REPARSE_POINT, and visits ancestors before children. This
rejects junctions and other reparse points without first traversing them. The
live snapshot now validates each directory before descent instead of letting
rglob follow an unrelated redirected subtree. Four additional tests simulate
Windows reparse attributes on root/directory/file paths and forbid traversal or
reads beneath an unrelated marked directory. They create no actual redirects.
Final focused result: 37 passed in 1.93 seconds; scoped git diff --check passed.
No live publication was performed after this correction.

Frozen final SHA256:
- tools/publish_studio_game_layer.py:
  ed334e8740751d09df2089c5661a93c8470605da05c1853d1b554f117291a56d
- tests/test_studio_game_layer_publish.py:
  77c829ac833e322db27229f97252ec63b7d6910f0983e25bd7f58394905093a7

Connection-routing continuation after parent-reported public NAT/hairpin
verification timeout: the failed authorized publication restored all three
replacements, with rollback.complete true in
build/dot-studio-entry-20261110/publication.json. This worker read that failure
receipt but did not execute a live request or publication.

Added optional --connect-host/--connect-port for HTTPS verification only.
The existing HTTPConnection socket factory routes TCP to the supplied address;
stdlib HTTPSConnection still wraps using the base URL hostname for SNI and
certificate hostname validation, and HTTP Host remains the origin authority.
The same create_default_context system/custom-CA context is required, with
check_hostname true and verify_mode CERT_REQUIRED. Direct override connections
do not use proxy environment settings or follow redirects. Any non-200 response
fails verification. No DNS, system trust, TLS/security setting, or server change
is made. Publication receipts now record origin and explicit TCP destination.

55 focused tests passed in 3.02 seconds. The new tests execute real stdlib HTTP
request/response parsing over mocked sockets, observing origin SNI, Host, normal
verification flags, destination port selection, cleanup and certificate-error
propagation. They reject HTTP, credential URLs, insecure contexts, redirects and
bad destination configuration, and exercise override preflight plus every file
verification in a temporary publication. No socket reaches a live service in
these tests. CLI --help and scoped git diff --check passed. Parent owns fresh
retry staging and any real TLS/publication validation.

Final connection-routing SHA256 (supersedes the earlier freeze):
- tools/publish_studio_game_layer.py:
  cc8955c7088e165db221d863a5f28c0d80c6ac4b205f0037ff3697e473dad75b
- tests/test_studio_game_layer_publish.py:
  a4063322e036be0d046b69bb1dc9f7378181d4e6c5f0a063f4fb8f486aa0737e

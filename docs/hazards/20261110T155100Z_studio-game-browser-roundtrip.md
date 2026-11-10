# Actual Studio/newer-game browser roundtrip

The optional same-origin layer ran in Chrome against the unchanged newer
MechanicalCreature base 2cd5c5e23f1491243346dd46586f5ebc10a4391187a4e991fa8f151f8dfd4ef4.
This supersedes the earlier browser-unavailable note for the measured run only.

Observed in the actual browser:
- Complete scene GPU fixture passed with zero differing pixels against canonical
  Studio drawing: 37 parts, 5,184 static and 72,960 subject triangles. The derived
  game entrance uses 5,197 static triangles. Fixture emission, split passes,
  glass ordering within the transferred scene and GL-state restoration passed.
- Optional activation captured the displayed scene, installed the original
  game's registered room/table geometry and exact Studio appearance, and
  received the original lifecycle's copied frames.
- Selecting Current Studio marble and using the original fire control created
  ball-1 with the captured appearance. Returned data contained its real worker
  slot/generation, position, velocity and the inverse Studio transform.
- Editing Studio table width from 8 to 8.5 sent newer authored revisions while
  keeping ball-1 and the same game epoch. No ammunition refill occurred.
- Original vehicle-marker dismount and top-down map navigation reached the
  imported room. Original touch-look worked at a 390 by 844 viewport.
- Return to Studio hid the same iframe; reopen retained its epoch and ball.
  Stop removed the iframe and cleared the receiver, leaving Studio present.

Evidence: build/dot-studio-sync-20261110/browser-roundtrip-receipt.json and
phone-room-proof.jpg. The screenshot is desktop Chrome at phone dimensions,
not physical iPhone validation. It shows the real imported ceiling grid and
original touch controls, with over 5,800 received frames. The original player
eye is about 0.2675 m above the floor. Its approach looks into the black table
skirt; looking up reveals the fixture. No player-scale or lighting alteration
was made to obtain that view.

The successful QA stage had diagnostic-only GET notifications at phase changes.
The current clean stage removes those notifications. Final source tests passed:
12 controller lifecycle checks, actual original launcher/bridge and return
consumer checks, room update/cancellation gates, and 15 targeted pytest cases.
A clean-stage reload/reopen then became unresponsive to browser inspection.
Screenshot and DOM calls timed out. Only that owned QA tab was closed. A fresh
owned tab of the same exact clean stage then activated successfully, acknowledged
revision 9, and launched the captured marble through the original control. Its
actual slot 2/generation 1 body and position/velocity returned to Studio.
clean-launch-proof.jpg visibly includes that marble, original car/HUD, selected
ammo with quantity 11 and frame counter 178. clean-browser-receipt.json records
the clean session at 210 returned frames. No cause is inferred for the reload
timeout, and no speculative engine or scheduling change was made. Fresh launch
is verified; reload/startup responsiveness still needs a bounded investigation.

At publication preflight there was no listener on production port 8766. Only
the owned read-only QA servers 8897 and 8898 were listening. Per the user's DLL
coordination instruction, no server was restarted and nothing was published.
The 23-file additive candidate remains under build/dot-studio-sync-20261110;
stage-receipt.json records exact clean hashes. Existing Studio and original game
pages are untouched. Recheck the clean candidate and availability before release.

Limits remain unchanged: room/table contacts use original admitted shapes;
fixture meshes are not reciprocal dynamic light bodies. Returned projectiles
are the original .11 m sphere body with no published spin. No ray-job insertion,
native solver-dt synchronization or worker-applied environment barrier is claimed.

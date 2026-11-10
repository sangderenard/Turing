# 2026-11-10 ISOLATED SHOP-LIGHT CONTROLS

Added marble_shop_light_controls.js and its focused fake-DOM Node test only.
No main, profiles, shared material/lighting code, browser, publication, restart,
commit or push. The parent owns integration and any override-path pruning.

The existing shop-light schema remains authoritative. Fresh editors create its
two scale-aware defaults; setState(undefined) restores the empty legacy rig.
Material selectors use the supplied catalogue. Stable fixture records expose
enabled/model/source, position, yaw, ceiling drop in feet, dimmer and RGB filter.
Add/remove retains unique IDs, and both model buttons enforce the eight-light
cap. Ordinary field changes do not detach the focused fixture DOM subtree.

state() and callback arguments are detached snapshots. validateChange(next,
previous) runs before commit; onChange(next, previous) runs with the new state
visible. A throw or false result restores this editor's state and displays the
reason. setState validates atomically without invoking onChange. Callbacks are
synchronous; external side effects remain the integrating caller's responsibility.

No burst control was added. Ordinary records display Photos: off with the
original timing integration requirement. A restored burst request is retained
unchanged and labelled unavailable, not silently converted into steady emission.

Passed Node syntax checks for both new files, the focused control suite and the
incumbent shop geometry suite. Covers isolated defaults, scale, every field,
catalogue rejection, blank/nonfinite/range rejection, validation and notification
rollback, failed add/remove, legacy restoration, model choices, maximum count,
non-reused removed IDs, valid unusual IDs without DOM collisions and detached
previous/new snapshots for override pruning. Fixed control nodes survive edits.
Browser/UI appearance and full repository tests were not run.

hazards_log.py show rejects the pre-existing malformed
20261109T_material-library-paused-checkpoint.md filename. Read newest receipts
and the legacy log opening directly before verification; did not alter that file.

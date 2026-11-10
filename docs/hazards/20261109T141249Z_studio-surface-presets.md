2026-11-09 SIMPLE STUDIO SURFACES AND CONCENTRIC MATS (MEASURED, local Windows source):

The simple room panel has a White/Black x Matte/Gloss matrix. It writes the
existing floor/wall/ceiling pattern and A/B fields to solid white plaster,
white gloss paint, existing black felt, or black gloss paint. The three new
catalog materials are opaque, nonmetallic and nonemissive; solid paint gloss
does not impersonate the thin-film enamel coating. Existing material entries
are unchanged. Assets have NOT been refreshed. Until the material snapshot
contains those entries, unavailable choices are disabled with an explanation.
The material-index preservation worker owns export and saved-object migration;
new catalogue rows must not be naively inserted at legacy synthetic IDs 32..34.

Individual material, pattern and layer controls are inside collapsed Advanced.
The first explicit preset choice retains the preceding custom room fields.
Concentric on retains the complete prior Advanced backdrop; off restores it.
Dedicated restore buttons expose both backups. Editing an active default mat
switches to Advanced without replacing other layer values. Optional profile
surfaces state stores backups and inversion; missing legacy state means custom,
not an automatically applied preset. Initialization never writes browser storage
or changes any room/backdrop value. Profile validation completes before writes.

Six nested circular mats are real triangles, sharing the existing backdrop
geometry path for Phong, live rays and photos. There are 96 triangles per disc;
successive layers advance 0.012 Studio units toward the subject. Diameter falls
linearly from 0.98 times the rear-wall diagonal to the rear-wall height, with
the centre at the rear-wall vertical midpoint. The innermost mat is white
plaster by default; inversion swaps all six materials with black felt. Natural
room clipping leaves every ring some visible area near diagonal wall regions.
At scale 3 with the table off: inner 12.06, outer 26.32, centre Y 5.01. With the
table's 3-unit floor drop: inner 15.06, outer 27.77, centre Y 3.51. Active defaults
follow room scale/table changes; custom Advanced mat dimensions remain fixed.
Legacy rectangles remain the default shape. Optional shape/centre fields retain
legacy profile structure. Advanced sizes now reach 64 units to cover scale 6;
circular mats do not inherit the rectangular floor sweep.

Measured checks: Studio component tests passed explicit four-way mapping,
repeated click idempotence, Advanced restore, inversion with/off the rig,
actual dimensions/depth/material order and visible bands for scales 1/3/6 with
both table states, profile roundtrips, corrupt-state rejection, exact legacy
layers and missing-catalog disabled behavior. The actual photo scene/wire
packer preserved every concentric triangle and material row for both inversions.
Existing room, profile and split-strength focused tests passed. Real
MaterialDatabase checks passed 3 tests in 0.24s, including the opaque finishes
and zero enamel thickness. JS syntax and scoped whitespace checks passed.
These are component/CPU tests, not real-browser or GPU visual verification.
Table and ten-light work continue separately and require a final combined gate.
No export, restart, exposure, commit, push or unrelated file restoration ran.

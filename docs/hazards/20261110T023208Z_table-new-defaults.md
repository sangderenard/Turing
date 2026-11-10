# Table new-set defaults 8 x 4 x 4

Source-only staged change: new Studio tables are width8, length4, floor-to-top height4 in Studio units. The existing default scale is10mm/unit, making80 x40 x40mm; legacy radius-linked profiles keep their original scale. UI help states both units and conversion.

Preserved explicit saved dimensions. Introduced a separate legacy migration record5.2 x3 x3.02 so missing dimensions and absent old tables do not inherit the enlarged new defaults. A raw missing-height table still produces floor drop3. Top contact remains Y=-1; new height4 gives floorY=-5. Skirt/material settings retained.

Passed focused Node tests: table geometry/defaults/migration, lighting UI/profile lifecycle, gobo full scene, scale, Studio scene capture, table lights. New-default mesh bounds verified x[-4,4],z[-2,2],y[-5,-1]. Fresh UI fields verified8/4/4. Old scene regression fixtures now explicitly request their original dimensions; expected physical results were not changed to follow new defaults. Syntax/scoped diff checks pass. Browser UI not tested.

Only production source changed: examples/meowdoku/web/marble_studio_table.js. Stage build/dot-table-defaults-20261110/marble_studio_table.js; manifest records source/published-before hashes. Not published, no restart, no user-tab or saved configuration mutation, no commit/push. Parent coordinates publication after priority load-system release.

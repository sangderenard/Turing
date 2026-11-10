# Studio defaults and concentric outlines

New pages initialize floor/four walls/ceiling to solid white plaster and the product table to the existing marble_milk_glass material. Explicit saved materials remain authoritative. Profiles with no table still load disabled clear-glass furniture metadata, retaining legacy behavior. Milk glass is the existing artistic preset, not a newly validated physical scattering model.

Concentric authoring exposes Ovoid/Rectangular outlines using the existing ellipse/rectangle backdrop geometry. The shape is profile-persistent; missing legacy shape defaults to ellipse. Advanced per-layer width/height remain available for tall pillar patterns; editing those exits the simple concentric recipe as before. No saved browser configuration is rewritten.

Regression uncovered that Room geometry overrides caused a valid concentric Exposure profile to fail full validation. Derived Room geometry now drops the recipe-match claim while preserving actual validated layers and Exposure metadata. Bounds/material checks remain active.

Passed focused Node checks: Studio lighting/profile lifecycle and rectangular save/load, profile storage, table geometry/payload, gobo scene isolation, scale, table lights. Four Python table/gobo/scale tests passed. Syntax and scoped diff checks passed. Isolated Chrome startup failed; installed Edge attempt ended in a tool/network failure. New browser UI remains unverified, no certificate/security bypass. This is not a full repository suite or new optical-material accuracy proof.

Five scoped static assets published with hash guards, HTTPS using configured CA verified; 224 unrelated files unchanged. No restart, no saved-object mutation, no commit/push. Main JS/HTML changes were narrowly applied to the current published versions, preserving other released work. Receipt: build/dot-studio-defaults-20261109/publication.json.

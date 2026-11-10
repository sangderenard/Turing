# Composable marble confetti and independent ribbon materials

Source implements explicit additive confetti for ribbons, spirals, rings and clear shells. Existing decorations are unchanged; conservative exclusion boxes prevent confetti intersecting them. New mixed recipes also enforce box separation between chips for the existing optical admission contract. Legacy omitted-flag geometry remains identical in all six modes.

Seven stable material authoring rows are appended via the existing spectral MaterialDatabase registration, preserving previous IDs and every existing exported row. Dynamic Ribbon 1..N editor targets select independent buffers. Old saves retain the original shared materials until an individual ribbon is selected for editing; it then copies all four material table rows before binding independently. Saved files are not rewritten.

The default axial Glass rod path keeps its existing bundle ownership and permits explicit closed four-face confetti outside the bundle. Existing topology, dielectric and pairwise overlap admission remains active. No other shapes enter the axial ownership exception.

Validation: additive composition test (500 chips, all four base decoration types, rotated/unrotated); dynamic control VM tests for both source and staged served JS (1/2/7/0 ribbons, selection fallback, isolation); existing mesh/material/bubble/optical JS tests; targeted Python ownership/material tests. Legacy exact parity checked against live unchanged mesh. Existing D6 distant-proxy assertion fails identically in source and untouched served baseline, unrelated to this change.

Release staged at build/dot-marble-composition-20261110. Scoped staging preserves unrelated unpublished OBJ, chase-camera, optical-dice changes. Assets append only seven generated rows, leaving prior 41 rows and other payloads unchanged. Publish script compares current live hashes, backs up all six files, replaces them atomically, and verifies bytes. Backend module reload must be coordinated before publication. No server restart, public deployment, or user saved-object changes performed by this task yet. Visual browser QA pending.

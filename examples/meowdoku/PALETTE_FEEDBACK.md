# Personal color preference game

`/palette.html` lets any visitor choose among the ten existing Meowdoku
colour-blind-safe swatches in the moment, in the style of a classic colour
test: a mid-alpha black overlay covers and disables the site while ten
borderless one-inch squares hang in space with ample margin, and each square
disappears when chosen. The order of choices is that participant's ranking.
`POST /api/palette/rank` stores it and selects a personal site accent.
`POST /api/palette/feedback` accepts `good` or `bad` with the current
candidate nonce, stores the outcome, and selects the next candidate. The
nonce prevents a stale page from rating a different design. The browser's
existing player token or an account's existing identity is the owner key.
Other visitors' palettes never change.

Every design a participant is shown (accent and secondary) is saved once in
`palette_saved`; `palette_active` names the one they wear. `GET /api/palette`
lists them as `saved` with the `active` id, and `POST /api/palette/select`
with a saved `id` wears that design again and makes it the candidate the next
feedback rates. The color game lists the saved designs and steps through them.

`palette-theme.js` puts the active design on every page that loads
`site-nav.js`, plus the home feed and the color game: the `--palette-accent`
and `--palette-secondary` variables, and the color game's accent rules for
links, button hover and focus, selection, form controls and a stripe along the
top of the page. Element rules only, so a page's own class rules still win. A
design chosen or rated on the color game reaches the participant's other open
pages at once through a `BroadcastChannel`.

The current selector combines rank position, that participant's good/bad
history, and a small exploration term. Positive feedback tends to retain a
liked color; negative feedback tries another. The SQLite event table keeps
each observation separately with an owner and time index, so later training
can use real choices rather than guesses. This first version is an adaptive
rule, not a trained neural network or a claim of production-scale learning.
Sixty feedback events per participant per day limit accidental loops.

The server uses the swatches from `games.PALETTES` and the existing
`_job_identity()` account/browser boundary. The matching SpeakToMe catalog
offers `palette_state`, `palette_rank`, `palette_feedback`, and `palette_select` to agents,
including no-signup preview agents. Palette play is never a guest
qualification requirement. Deploy the matching revisions and export the web
assets before using the public page; source review alone does not do that.

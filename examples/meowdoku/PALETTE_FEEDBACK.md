# Personal color preference game

`/palette.html` lets any visitor choose among the ten existing Meowdoku
colour-blind-safe swatches in the moment, in the style of a classic colour
test: a mid-alpha black overlay covers and disables the site while ten
borderless one-inch squares hang in space with ample margin, and each square
disappears when chosen. The order of choices is that participant's ranking.
`POST /api/palette/rank` stores it and selects a personal site accent. The
page and the home feed show it for that participant.
`POST /api/palette/feedback` accepts `good` or `bad` with the current
candidate nonce, stores the outcome, and selects the next candidate. The
nonce prevents a stale page from rating a different design. The browser's
existing player token or an account's existing identity is the owner key.
Other visitors' palettes never change.

The current selector combines rank position, that participant's good/bad
history, and a small exploration term. Positive feedback tends to retain a
liked color; negative feedback tries another. The SQLite event table keeps
each observation separately with an owner and time index, so later training
can use real choices rather than guesses. This first version is an adaptive
rule, not a trained neural network or a claim of production-scale learning.
Sixty feedback events per participant per day limit accidental loops.

The server uses the swatches from `games.PALETTES` and the existing
`_job_identity()` account/browser boundary. The matching SpeakToMe catalog
offers `palette_state`, `palette_rank`, and `palette_feedback` to agents,
including no-signup preview agents. Palette play is never a guest
qualification requirement. Deploy the matching revisions and export the web
assets before using the public page; source review alone does not do that.

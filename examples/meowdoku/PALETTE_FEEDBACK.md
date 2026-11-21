# Personal color preference game

`/palette.html` lets any visitor order the ten existing Meowdoku
colour-blind-safe swatches. `POST /api/palette/rank` stores that ordering and
selects a personal site accent. The page and the home feed show it for that
participant. `POST /api/palette/feedback` accepts `good` or `bad` with the
current candidate nonce, stores the outcome, and selects the next candidate.
The nonce prevents a stale page from rating a different design. The browser's
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

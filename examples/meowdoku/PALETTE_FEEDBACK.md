# The color game: random designs and a native learning network

`/palette.html` shows a round of eight designs drawn at random for each
participant and each round. A design is one of the 148 named CSS colors
(`palette_designs.NAMED`) with its own bevel (depth, light angle) and shadow
(offset, blur, spread, darkness). No two squares of a round share a color. A
mid-alpha black layer covers and disables the site while the squares hang in
space; each square disappears when chosen, and the order of choices is the
participant's ranking of that round.

The site then wears a design the network proposes. The participant scores it
on a slider from 0 to 1. **Try again** draws a new round without recording
anything; **Clear history** forgets that participant's rankings, scores, saved
palettes and personal layer. **Yours beside everyone's** shows, from the same
random pool, the best design for this participant and the best for the shared
network alone, each with both networks' scores. Every design the site has worn
is kept, and choosing one (or stepping through them) wears it again.

## The network

`palette_native_training.py` defines it on an `SSATensorProgram`, as
`examples/pattern_project/native_image_training.py` does, and compiles it with
`compile_native_training_schedule`: forward, loss and the backward derived from
`BACKWARD_RULES`, as one native LLVM library with a stepping entry per
parameter group.

- A design's features (`palette_designs.features`) are eleven numbers in
  [0, 1]: the color in OKLab, the bevel depth and the sine and cosine of its
  light angle, and the five shadow numbers.
- Its score is a shared two-layer network (`shared.w1`, `shared.b1`,
  `shared.w2`, `shared.b2`) plus a personal linear layer `personal.u` weighted
  0.25. The personal layer is pulled toward zero, so it only nudges the shared
  default.
- The loss: each ranked pair wants `sigmoid(score(preferred) - score(other))`
  near 1; each slider score wants `sigmoid(score(design))` near the slider;
  plus the pull on the personal layer.

## Continuous training

`palette_model.py` owns the compiled schedule (compiled once, in the background,
into the model directory beside the accounts database, or the directory named by
`MEOWDOKU_PALETTE_MODEL_DIR`). Until it is ready the game offers random designs
and says so. Every ranking (all 28 ordered pairs of a round) and every slider
score queues a training pass on one worker thread: native steps of the `shared`
group on everyone's newest responses, then of the participant's `personal`
group on their own. The shared parameters are kept in `shared.npz`; each
personal layer is kept with its owner in `palette_personal`. Proposals are
scored by the same compiled function at learning rate zero, before the request's
database transaction opens (training, which holds the model, writes to the
database).

## Endpoints

- `GET /api/palette`: the round, the proposed design (`candidate`), the
  comparison, saved palettes, history and the network's state.
- `POST /api/palette/rank` `{nonce, ranking}`: every design id of the current
  round once, favorite first. A stale round is refused (409).
- `POST /api/palette/feedback` `{nonce, rating}`: a number from 0 to 1 for the
  current design (`good`/`bad` still mean 1 and 0).
- `POST /api/palette/round`: a new random round (Try again).
- `POST /api/palette/clear`: forget this participant's history.
- `POST /api/palette/select` `{id}`: wear a saved palette again.

The SpeakToMe catalog offers the same endpoints to agents (`palette_state`,
`palette_rank`, `palette_feedback`, `palette_round`, `palette_clear`,
`palette_select`). The owner key is the existing `_job_identity()`
browser/account key; one participant's answers change only their personal layer
and their share of the shared network's training. Responses (rankings and
scores) are limited to 240 per participant per day. The tables of the earlier
fixed-swatch game are left in place, unread.

`palette-theme.js` puts the worn design on every page that loads `site-nav.js`:
the readable accent and secondary (moved in OKLab lightness to meet 4.5:1 and
3:1 contrast on the site background) as `--palette-accent` and
`--palette-secondary`.

# The color game: random designs and a native learning network

`/palette.html` shows a round of eight designs drawn at random for each
participant and each round, each drawn as a small page. A design
(`palette_designs.random_design`) is:

- a palette of 1 to 6 of the 148 named CSS colors (`palette_designs.NAMED`),
  its size drawn from a normal curve centred on 4 (`SIZE_WEIGHTS`), arranged by
  a harmony scheme (`SCHEMES`: monochrome, analogous, complementary,
  split-complementary, triadic, tetradic, or free), with neutrals mixed in as a
  decorator would;
- a token for each of the page's sixteen elements (`ELEMENTS`: page, surface,
  raised controls, nav bar, header, input, lines, text, muted text, links,
  primary and secondary buttons, badges, the secondary color, the focus ring,
  the top stripe): which palette color it takes, where its lightness sits in
  that element's band, how much chroma it keeps, and for boxes the corner
  radius, border width, bevel and shadow, all lit from the design's one light
  angle. Tokens are drawn around a style family (`STYLES`: flat, material,
  beveled, brutalist, soft, glass, editorial, or free) so distinct styles appear.

Colors take decorating roles (`roles`): the most neutral color dominates the
page's surfaces, the most chromatic is the accent, the next the secondary; an
element sometimes takes another palette color. Readability is not learned: text,
muted text, links, the focus ring and the secondary are moved in OKLab lightness
until they meet their contrast on the surface behind them (`CONTRAST`).

No two designs of a round start with the same color. A mid-alpha black layer
covers and disables the site while the designs hang in space; each disappears
when chosen, and the order of choices is the participant's ranking of that
round.

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

- A design's features (`palette_designs.features`, `FEATURES` = 294 numbers in
  [0, 1]) are a state tensor of the page: for each of the sixteen elements, the
  palette color it takes (OKLab), its tone and chroma, its resolved color on the
  dark page (OKLab), its radius, border, bevel, shadow depth, blur and darkness,
  and the light angle; then the palette's harmony scores (`harmony`: size,
  affinity for monochrome, analogous, complementary, triadic, split and
  tetradic arrangements, chroma-weighted hue spread, hue entropy, share of
  neutrals, lightness and chroma spread, warmth, distinctness); then the page's
  structure (text, link and button contrasts, lightness steps between page,
  surface and controls). Color theory measures; the network learns what is good.
- Its score is a shared two-layer network (`shared.w1`, `shared.b1`,
  `shared.w2`, `shared.b2`, 64 hidden units) plus a personal linear layer
  `personal.u` weighted 0.25. The personal layer is pulled toward zero, so it
  only nudges the shared default.
- The schedule is compiled for one layout of the network (`layout.json` beside
  it); a model directory holding another layout compiles again, and shared or
  personal parameters of another layout start over. Rankings and scores stored
  before palettes are read as palettes of their colors (`upgrade`).
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

`palette-theme.js` puts the worn design on every page that loads `site-nav.js`
(`palette_designs.wear`): every element's color as `light-dark(light, dark)`
(`COLOR_VARIABLES`: `--palette-bg`, `-surface`, `-raised`, `-line`, `-fg`,
`-muted`, `-ink`, `-accent`, `-secondary`, `-nav-bg`, `-header-bg`,
`-input-bg`, `-button2-bg`, `-badge-bg`, `-focus`, `-stripe`), the text on the
accent (`--palette-on-accent`), and each box's `--palette-<box>-radius`,
`-border` and `-shadow`. Pages read them with their own colors as fallbacks,
so a page without a design looks as it did.

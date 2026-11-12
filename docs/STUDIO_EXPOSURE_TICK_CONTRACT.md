# Studio exposure tick contract (2026-11-11)

The Studio still-photo shutter has a model duration `exposure_seconds` and an
integer `exposure_substeps`. Its model `dt` is their quotient, at most 0.1 s.
The browser advances one substep after each completed Studio display tick. If
the display takes longer, the exposure's model time waits; it does not skip a
light state to catch up to wall time. The Studio preview runtime receives the
same model clock while the shutter is open. The original clock resumes after
capture. The current timed-photo route fixes camera and fixture geometry;
only emission and named emitter gain/filter may vary.

The MRP1 photo configuration carries `exposure_substeps` (at most 216000) and
may carry `exposure_iso` (25..6400) and `highlight_clip_percent` (0..25).
Requests without ISO or clip retain the legacy display transform. ISO 100 is
unit display gain. The RGB display uses the existing linear luminance curve
with a white reference at the `(100 - highlight_clip_percent)` percentile of
positive profiled luminance. Values above that white clip in the display;
the raw linear sensor HDR stays untouched. ISO applies gain after that white
reference. The monochrome display applies the corresponding linear rule.

An `exposure_state_frames` bundle contains contiguous `[t0,t1)` intervals
covering the shutter. Every boundary lies on the contracted tick grid. Each
frame carries the complete fixture PBR and named emitter profile state; the
server admits changes only to declared light emission and profile power/filter.
The worker prepares native sources once, installs each state at a complete
sensor boundary, and leaves one accumulator open. It requests sensor sweeps
according to the interval's tick count. A trigger can split a tick into
additional cutoff/flash intervals so each nonempty sensor interval is sampled.

REST `POST /api/show/pictures` accepts these settings inside `photo` as
`exposureSubsteps`, `exposureIso`, and `highlightClipPercent`. Optional
`lightTimeline` is an ordered array of `{tick, lighting}` changes after tick
zero; each `lighting` is a complete resolved Studio lighting record. The
initial state is `settings.lighting` (or the host's configured default).
The existing Nodus product builder creates the MRP1 bundle and photo job.
The SpeakToMe MCP `show` tool forwards the same `photo` object and its
generated `photo_picture_ask` tool exposes that REST request.

For a host with configured Studio defaults, an MCP `show` call can ask for a
five-second exposure with `photo: {"seconds":5,"exposureSubsteps":300,
"exposureIso":200,"highlightClipPercent":25}`. The tool returns a picture
receipt; calling `show` again with its `picture` ID reads job progress.

This is a capture-then-trace contract. GPU work starts after the browser has
recorded the bundle, so GPU backpressure does not yet pause a browser shutter
that is still recording. The browser's own rendering cost does pause its
model clock. No GPU exposure receipt was run because `AGENT_OBSTINANCE.md`
currently prohibits new renders until its field-information condition is met.

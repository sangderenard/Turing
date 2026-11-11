# Shared header control suites

The existing `examples/meowdoku/web/site-nav.js` header accepts page-owned
control suites. It copies the Living Data shell's declarative presentation
pattern, not its vehicle, physics, capture or input engine. Account, ticket,
and server-switch controls keep their existing behavior.

Opt in with the existing script tag:

```html
<script type="module" src="/site-nav.js" data-site-nav="allow"></script>
```

A page can import `registerControlSuite` from the same module or use
`window.MeowdokuSiteNav.registerControlSuite` after `site-controls-ready`.
Registering before header mount is supported. A unique policy identity owns
one suite; duplicate identities are rejected. The returned function removes
its UI/listeners and cancels its telemetry frame. It is safe to call twice.

```js
const dispose = registerControlSuite({
  identity: policy.identity,
  label: "Controls",
  policy, // Existing ViewportControlPolicy data: {bindings: [{action, inputs}]}
  sample: action => sampler.sampleAction(action), // Existing TuringControlSampler
  actions: [{label: "Reset view", run: resetView}], // Existing page callbacks
  onOpen: releasePageInput,
});
```

Sampling is optional. When provided, it updates action lamps only while the
panel is open. Each row names the semantic action and its declared inputs;
the lamp represents the action, not an individual physical input. This UI
never attaches input listeners or requests device access. Sampling exceptions
are shown, not replaced with simulated input. `actions[].close: true` closes
the panel before invoking the action, preserving the direct user gesture for
page-owned pointer capture. Callback failures are displayed in the panel.

`abstract_ui_div_map.py` registers Living Data's actual viewer policy,
`viewportInputValue`/`viewportGamepad`, and existing capture and shader-only
callbacks when the host has opted into the shared header. Opening the suite
releases the existing viewport capture; closing it does not silently recapture
input. Standalone generated projections do not fetch the site header or gain
an account-service dependency. To use the adapter in a hosted generated page,
its host must include the opt-in script above and serve the module.

Verification: `node tests/site_control_suite_check.mjs` runs the real module
against an offline DOM boundary and checks validation, registration, duplicate
rejection, callbacks, telemetry, Escape, outside click, repeated toggles,
unregistration, and account API failure independence. It is not browser layout
or pointer-lock verification. Browser coverage belongs to the banner's browser
checks and the actual hosted page. No generated/live assets are published by
this source change.

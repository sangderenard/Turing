# Meowdoku page contract: publication boundary

## Current boundary

The public root is the feed of **author-published** pages. Built-in games retain
their direct addresses and the separate Games link; they are not silently
attributed to an account. `authored_pages.py` currently has two states: `dev`
(owner-readable on the development host) and `release` (publicly readable on
stable). Uploaded HTML/text is an explicit public copy. Turing site bundles are
immutable, inventoried versions registered from an owned, measured workspace.
The creator's Bake button makes a separate release record and retains the dev
page. The older `/stage` route remains for legacy clients, while baked release
records cannot be returned to dev.

The visual feed card may use a static poster declared by a bundle. It must not
load a game, shader, script, or iframe merely to create a thumbnail. If no
poster is available, the feed shows the title-based placeholder. A poster is
presentation data, never evidence that the page or its compute works.

## One versioned descriptor

Every generator should add a `page_contract` object to the existing
`turing-program-bundle-v1` `bundle.json`, or supply the same object with an
explicit HTML/text upload. It is metadata *inside the existing bundle and
account-owned publication path*, not a second site bundle format or a new
authorization system. A missing object on an older page means `unknown`, not
`offline`, `safe`, or `requires nothing`.

```json
{
  "schema": "meowdoku.page.v1",
  "presentation": {
    "title": "Chess arena",
    "summary": "A player and a network compete",
    "poster": "poster.webp",
    "surface": "canvas"
  },
  "execution": {
    "standalone": "unknown",
    "network": "optional",
    "compute": ["browser.webgpu"],
    "site_endpoints": []
  },
  "access": {
    "required_capabilities": []
  },
  "controls": {
    "local": "site-nav.registerControlSuite",
    "server": null
  }
}
```

`surface` is one of `dom`, `canvas`, `webgl`, `webgpu`, or `mixed` and describes
the visible page, not its authority. `standalone` is currently `requires-site`
or `unknown`; a `verified` claim is reserved until the build can attach an
offline test receipt. `network` is `none`, `optional`, `required`, or `unknown`. `compute` names
browser mechanisms or registered server operations. `site_endpoints` names
declared API catalog operations, rather than arbitrary URL strings. Each server
operation carries its capability from the existing API catalog; the descriptor
may show that requirement to a visitor but cannot grant it. A page may declare
no controls. If it does declare local controls, its JS calls the *existing*
`registerControlSuite` function with bindings and callbacks. If it declares
server controls, they refer to a Living Data map/control contract, and the
server checks the signed-in principal on every forwarded action. The descriptor
contains no executable callbacks, secret values, or caller-supplied grants.

The producer must keep the contract, its poster, JS, shaders, and compute
artifacts in the bundle inventory. The publication service validates their
paths and digests before exposing them. The feed returns only bounded display
fields and requirement labels; opening the page reads the exact inventoried
version. Legacy uploads and bundles display requirements as unknown until they
are rebaked with a contract.

`GET /api/pages/{id}/contract` returns the sealed descriptor through the same
stable/dev visibility gate as `GET /p/{id}/`. `POST /api/pages/{id}/bake` takes
`{}` and returns a new release ID; repeating it returns the same release. The
page upload request may include the descriptor as `page_contract`. A plain
HTML/text upload has no inventoried poster asset, so its poster must be null.
These descriptors display requirements; they do not grant access or execute a
control.

## Bake and release

1. A creator edits in an owned dev workspace. Each generator uses the existing
   Turing bundle publisher (`build_program_bundle` or
   `publish_prebuilt_program_bundle`), producing one complete version with
   `bundle.json` written last. It emits the same page contract regardless of
   whether the interior came from a shader, game, Dream document, or another
   generator.
2. The existing workspace ledger measures the output. The trusted registration
   path binds that bundle version to the creator; a browser does not submit a
   server filesystem path. Dev preview uses the normal page route and the dev
   host's owner check.
3. Bake validates the inventory and contract, checks the declared poster, and
   creates a **new release record** pointing at
   that immutable version. The dev record stays available for the next edit.
   A plain upload likewise creates a distinct release copy after the owner
   explicitly confirms public publication. Releasing is not an in-place edit
   to a mutable draft.
4. Stable lists and serves only release records. Dev shows the creator's drafts
   alongside releases. Both hosts enforce the same account authorization and
   immutable-content checks. A broken or missing asset fails closed; the feed
   must not claim that a failed bake is public.

The next data migration separates **channel** (`dev` or `stable`) from
**audience** (`private`, later a shared group, or `public`). The current two
states map to private/dev and public/stable. This allows private stable pages
and public development previews without overloading a single `stage` field.
Private encrypted uploads still need the owner's browser to decrypt and send a
separate readable publication copy; the server never acquires their private
file key.

## Feed and page shell

The root uses one responsive card layout for public releases. The development
host adds an owner-only draft section; the card distinguishes dev/release and
upload/bundle. A contract-backed static poster, capability chips,
compute chip, and standalone indicator occupy the same card positions. Author
links group the author's published pages. The shared `site-nav.js` remains the
single account, route, and local-control surface; each page owns its interior.
The old game selector remains a direct-page directory while generators migrate
to the common publication contract.

## Remaining implementation gates

- Migrate the two-axis audience/channel model without exposing existing dev
  pages. Bundle release records still point to the sealed, owned workspace
  version, whose inventory is checked on reads; durable release storage and
  workspace lifecycle coupling remain to be completed.
- Give authors a structured contract editor, and verify declared API operation
  names and compute requirements against the installed catalogs at bake.
- Have each generator emit truthful artifact and runtime requirements, with an
  offline verification receipt where it claims standalone operation.
- Wire the declared Living Data control contract through the existing site
  controls and authenticated API catalog; no generic client-supplied REST relay.
- Publish the source assets through the managed deployment path and verify
  stable and dev browser views with a real author account.

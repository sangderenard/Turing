# Account-authored pages

The old root game selector is retained as `play.html` for existing links. The
exported root `index.html` is an author feed. `web_export.py` also emits one
`play-<game>.html` entry for each existing game and a `games.html` directory.
Those games are site products and never appear in the author feed by default.

`authored_pages` v1 lives in the **shared Accounts database**, the same account
authority used by stable and development. Each page has an account owner and
one of two explicit stages:

| Stage | Development server | Stable server |
|---|---|---|
| `dev` | Owner can preview the page and see it in their feed. | No feed entry or direct page access. |
| `release` | Visible to readers. | Visible in the public feed, author listing, and direct page URL. |

Returning a page to `dev` immediately removes stable reads, including its
direct URL. The stable service does not rely on the development content root:
small uploaded HTML/text copies live in the shared Accounts database, and a
registered compiler bundle stays in the shared external account workspace
already measured by `workspace_accounting`. Public page responses use CSP
sandboxing, which gives their scripts an opaque origin.

`GET /api/pages[?author=<username>|?mine=1]` lists pages; `GET /p/<id>/` and
`GET /p/<id>/<artifact-path>` read them. An authenticated same-origin POST to
`/api/pages` creates a dev page from an explicitly confirmed UTF-8 HTML/text
copy (512 KiB per page, 4 MiB and 32 pages per author). The existing encrypted
private-file store is not decrypted by the server. Its owner can download and
decrypt a file in their browser, then choose the readable file for an explicit
public copy. `POST /api/pages/<id>/stage` accepts only `dev` or `release` and
rechecks the account owner. Published pages are not editable in place; a new
publication has a new id.

## Compiled site bundle handoff

The supported compiled page is the existing immutable
`turing-program-bundle-v1` from `src.compiler.site_bundle`, with an artifact
inventory. The Nodus `SiteCompiler` currently emits **row-tool products**, not
these page bundles. The trusted builder can build a Turing site bundle beneath
`<external-workspace-root>/<workspace_id>/`, have the existing workspace ledger
measure those bytes, then run:

```text
python examples/meowdoku/register_site_bundle.py --accounts-db <accounts.db> \
  --workspace-id <workspace_id> --bundle-dir <workspace>/site/programs/<slug>/versions/<version>
```

The command derives the owner from `workspace_leases`, measures the workspace
through `workspace_accounting`, verifies the completed bundle and its inventory,
and registers it as a dev page. The author then previews and marks it release
from the feed. HTTP clients cannot submit a filesystem path or claim a bundle
owner. Workspace release refuses while any page references its bundle. The
external workspace must be readable by both managed web service accounts.

This handoff requires a completed bundle already in an account workspace. The
site has no self-service compiler-to-page job yet. It must be added through the
existing source divider, Turing site-bundle builder, and workspace ledger, with
the same owner and retention checks; a row-tool result is not a page bundle.
The existing bundle shell's absolute `/site/programs/<slug>/versions/<version>/`
asset URLs resolve through the same stage and owner gate. The trusted builder
must choose a unique account-specific slug; registration refuses route collisions.

## Deployment

`web_export.py` must publish the new static files to each instance's web root.
The backend must then be restarted through the managed lifecycle so both
instances use the same Accounts database and page route code. Source edits
alone do not change the running site. Do not restart an active instance without
the existing operator coordination and session-preserving procedure.

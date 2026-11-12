# Licensing

## Scope

The MIT grant in [LICENSE](LICENSE) applies to the first-party code and
documentation for which sangderenard holds licensing rights. Existing
copyright and permission notices must be retained. Third-party software,
submodules, dependencies, assets and independently authored material retain
their own terms. This grant does not claim ownership of those works or
supply missing permission for unverified material.

The owner selected this first-party MIT policy on 2026-11-11 to express the
projects' continuity with SpeakToMe. Repository history and author names were
checked as provenance evidence; they are not proof of ownership of every
imported file. This document is a scoped inventory, not a complete dependency
audit or a statement that every possible combined distribution is compatible.

## Preserved notices and imported components

- `src/compiler/vendor/acorn.js` identifies Acorn 8.16.0 and its upstream
  authors. It is third-party MIT software, not a sangderenard copyright.
  The full upstream notice and author list are in
  `src/compiler/vendor/acorn-notices/LICENSE` and `AUTHORS`.
  Their exact upstream source is recorded in that directory's README.
- `src/cells/bath/hybrid_fluid.py` and `surface_animator.py` already contain
  MIT declarations. They remain unchanged.
- Material originating in SpeakToMe retains its original notices. Its existing
  MIT grant is not replaced by the new root copyright line.
- Installed compiler/runtime packages and external services retain their own
  terms. They have not all been inventoried here.

## Shared REST extraction

SpeakToMe commit `91add50378fae73baeefe4505a57007fc9a9b0a3` extracted HTTP
handlers from Turing; Turing commit `763582a0` supplies the matching adapter.
First-party extracted code follows the owner's MIT direction. This does not
license backend dependencies or a third party's independently authored work.
Keep both source revisions and their notices in the deployment receipt.
This license update does not itself merge or deploy that extraction.

# SpeakToMe REST extraction

Matching SpeakToMe source: `91add50378fae73baeefe4505a57007fc9a9b0a3`
on `codex/rest-api-offload-20261111` (local commit; remote publication blocked).

The existing Meowdoku listener now delegates its messaging, Living Data query,
graph-navigation and static DM HTTP handlers to `speaktome_api.routes` in the
SpeakToMe repository. It passes the same handler, authenticated principals,
database path, messaging module, graph services and StaticWorlds instance.
Only HTTP contract ownership moved. Account/state/schema/compiler behavior,
URLs, permissions, payloads and responses did not change.

`examples/meowdoku/rest_api.py` loads that package from the sibling `speaktome`
checkout, or the explicitly selected `SPEAKTOME_REPO`. It does not put the
other repository on sys.path or import its numerical packages. This is a hard
deployment dependency: update SpeakToMe before this Turing adapter. Startup
refuses missing source or an incompatible host binding instead of silently
running a second implementation. Keep both repository revisions together in
the deployment receipt. See `speaktome_api/README.md` for the host contract.

Nodus remains the canvas/contract-port runtime. Its existing federation socket,
negotiators and tick scheduling are unchanged. The `nogodsnomasters`
`engine_toy` Machine/PartPort/MachineSystem identities remain the machine-side
authority. Neither is a newly implemented REST backend in this extraction.
DM still means the existing owner-scoped static inspect/render command surface,
not a new account role or live machine game simulation.

No process was restarted and no live state, credentials, privileges, trust or
network configuration was changed. Use the existing Apache lifecycle and
coordinate the deployment with the federation/identity lane. Do not launch
a duplicate server on its live port.

Focused gate, with `SPEAKTOME_REPO` set to the matching SpeakToMe checkout:

`python -m pytest tests/test_meowdoku_messaging.py tests/test_meowdoku_messenger_client.py tests/test_living_data_query.py tests/test_living_data_cursor.py tests/test_living_data_static.py tests/test_speaktome_rest_dependency.py tests/test_meowdoku_authz.py tests/test_meowdoku_proxy_trust.py tests/test_meowdoku_server_authz.py tests/test_meowdoku_public_login.py -q`

This is a focused offline/loopback gate, not the full repository suite, public
TLS proof, deployment proof or browser-render proof.

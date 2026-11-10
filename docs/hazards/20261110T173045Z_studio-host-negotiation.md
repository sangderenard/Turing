2026-11-10 Original game host negotiation (MEASURED, offline Windows)

The optional host uses the shared marble_studio_game_protocol.js descriptor
and receipt validator. readProtocol() returns one immutable session contract.
Visible acknowledgments retain the actual nested environment receipt, including
native-scene-ready-worker-pending. The previous host unconditionally labeled
the top level posted-to-original-worker. The host now mirrors only the nested
publication, validates identity/revision/sequence/building/epoch, and stores a
detached frozen receipt before advancing its last accepted sequence. No
worker-applied barrier is advertised by this protocol.

Each connection owns its disconnect closure. Old disconnects and retained old
callbacks cannot detach or deliver into a replacement connection. Closed
connect/update calls refuse, closed snapshot reads return null, disposal is
idempotent, and queued native updates compose host-open with room isCurrent.
Request identity/revision scalars are captured before asynchronous work.

The collection's array admission now permits zero to eight assets. A scene
without supported sphere ammunition can install its empty appearance holder
and transfer the environment. Explicit malformed assets still fail validation;
later environment updates never reinstall or refill the collection.

Offline results: 19 host lifecycle cases pass, including actual bridge
setSink/readSnapshot/publish calls; collection and room module checks pass
with mocked GPU boundaries; 18 shared protocol negative gates pass; two
source syntax checks pass; host pytest wrapper and Studio patch checks pass
3/3 in 0.65 seconds. Host installer timing is explicitly controlled by test
doubles. No native physics, GPU, browser, server, stage, deployment, commit
or push operation was performed in this task. No claim about the earlier
browser reload stall follows from these tests.

Host SHA256: 07abb3f641fcf064add359ea94bee8801adc9294eee26ea59d36a0679cfc4476
Collection SHA256: 1109d0d8c19976f85c7d988025ec147d4a4c167a7726f8786c0a808bd2a6ba72

# Backend PCM subscription for native Nodus Radio

`nodus_radio_pcm.Subscription` consumes the existing native `udp_router` TCP
control/BATCH protocol using `relay.FRAME` and `relay.ENTRY`. It does not route
PCM through outgoing REST send limits or SQL history. It opens only its own
router-owned UDP port after verifying the router's READY bind is loopback.
Closing the control connection releases the port in the native router.

The current negotiated audio profile is RTP v2, dynamic payload type 96,
L16/48000/2, exactly 256 stereo sample frames per packet (1024 payload bytes).
The producer endpoint and SSRC are pinned by the trusted backend caller. No
account authorization is inferred from UDP or SSRC. There is no public route,
new broker, audio device, or automatic subscription registry in this change.
Service authorization/port allocation must precede constructing a Subscription.

The bounded, caller-clocked buffer handles sequence/timestamp wrap, reordering,
duplicates, late packets and loss. Missing blocks become zero PCM; a consumer
that falls beyond its capacity drops the old backlog and prebuffers fresh
frames. The consumer calls buffer.take() at 48000/256 Hz; None means initial
prefill, bytes are little-endian interleaved signed16. Capacity defaults to32
packets (~171ms maximum stored audio); prefill3 (~16ms). No timing guarantee.
Queue counts and loss/rejection/resync counters remain observable.

Verification: five unittest cases pass including a real local TCP fixture
fragmenting READY and delivering the exact BATCH protocol. This is a protocol
fixture, not a proof of the native producer/router; native integration evidence
is to be supplied separately. No audible playback, live deployment or restart.

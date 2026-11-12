2026-11-10 Original worker collider barrier (MEASURED, offline Windows)

The pinned original worker has no collider revision acknowledgment. Its
engine-state message is not a barrier: colliders call armEngine, which emits
nothing when already in full dynamics. A first subsequent snapshot is also
insufficient; it can already be queued before the collider update is consumed.

The actual protocol permits only one snapshot in flight despite its three
buffer pool. If the main decoder has consumed sequence N when the native
collider post occurs, N+2 requires recycling N+1. That recycle is posted on the
same Worker channel after the collider post. FIFO therefore proves that the
worker CPU collider assignment occurred before publication of N+2. It does not
identify the solver tick, prove GPU completion, or guarantee later updates have
not superseded that field. No stronger workerAppliedBarrier capability is set.

mechanical_creature_studio_bridge.js now adds workerColliderBarrier evidence to
the existing receipt while retaining its pending/posted colliderPublication
semantics. A passive listener observes the exact original Worker's messages
after its existing decoder; mutable global latestSequence alone is not proof.
Rollback reposts rearm the barrier. Closing an epoch removes the observer;
reopening resets its observation. Worker replacement, not-ready state or a
sequence reset invalidates the old receipt and removes the old listener.

The focused test executes the complete original worker and exact original main
message handler with controlled transport queues and real structured-clone
buffer transfers. It proves the pre-consumption N+1 counterexample, post-recycle
N+2, exact collider payload, rollback reposting, stale worker/epoch rejection,
listener cleanup and immutable older receipts. It does not run a physics tick,
timer, GPU or browser. Existing native registry/collider/lifecycle proofs remain.

Offline gates passed on both pinned originals: the Node bridge checks, bridge
syntax and six patcher pytest checks (1.78 seconds). Source patch reversal still
recovers every original byte. The newer game remains
2cd5c5e23f1491243346dd46586f5ebc10a4391187a4e991fa8f151f8dfd4ef4.
Bridge SHA256 at this receipt:
73c0f6982716c716b68fdf0604fc719564535bed1e9bb45022bd89733d54699b.
No compiler, worker, physics, dynamics, server, live page or publication changed.
No commit or push was performed. Full repository tests were not run.

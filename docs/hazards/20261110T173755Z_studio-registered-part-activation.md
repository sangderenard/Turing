2026-11-10 Registered Studio part activation (MEASURED, offline Windows)

The bridge now uses the building helper's pure reconciliation of startup-
registered parts. A complete descriptor retains every registered ID and the
last admitted geometry for disabled parts. activeProxyIdentities separately
names active members. The room renderer receives only those active proxy IDs.

Application resolves the original model document boxes and their exact existing
world-object extension references. It removes inactive original boxes only from
shaderViewer.geometry and reinserts the same refs when re-enabled, then calls
the unchanged native rebuildPortableSceneMesh. Registry, model and runtime IDs
are retained. No custody transition into player inventory is invented. This
avoids the original span-index assumption exposed when a hidden custody box
remains in the active geometry input. Rollback restores the full prior active
array/order and rearms the prior worker collider publication barrier.

Focused actual bridge/native tests passed on both pinned originals. Table
contacts change 15 to 6 to 15, and hiding one middle table part gives 14. Tests
check exact active span identity indices, unchanged player/fired projectile,
same document/world/runtime refs, retained moved hidden pose rather than startup
pose, failed re-enable rollback, active renderer admission, and rejection of
unknown IDs/missing building/custody changes. Source preparation uses a small
declared room fixture; it is not claimed as a live Studio capture or browser run.
All prior exact worker FIFO/recycle tests still pass. Six patcher pytest checks
pass in 1.46 seconds, including complete original byte recovery.

Bridge SHA256: ff08ba75b3ebc497b3f40a8586c3c198ad99819b57c4c2acdd29dc7ab35825b6.
Original newer game SHA256 remains
2cd5c5e23f1491243346dd46586f5ebc10a4391187a4e991fa8f151f8dfd4ef4.
This slice does not admit new structural identities or extend the registry.
That separately authorized canonical registry work is still in progress.
No compiler, physics, server, live page, publication, commit or push changed.
Full repository tests and browser/GPU tests were not run.

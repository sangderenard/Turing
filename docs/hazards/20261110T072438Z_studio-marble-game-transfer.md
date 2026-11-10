# First Studio marble in the original playable game

Published an additive /mechanical_creature_studio.html with 17 private-subdirectory
assets. The preserved /mechanical_creature_play.html and 280 existing files are
byte-identical. Trusted TLS verified every new hash. Backend boot 21637d216896
stayed unchanged. No restart, compiler rebuild, model serialization or worker
replacement occurred. Receipt: build/dot-studio-game-transfer-20261110/publication.json.

The actual captured spherical blue-green cat-eye has 72,960 triangles. Existing
76-byte float32 triangle transport preserves all 1,313,280 GPU attributes, material
IDs, semantic groups, shader sources and four material tables. The original
game owns radius .11 m, mass, contacts, controls, timer, inventory, launcher,
pickup and vacuum. Fixed authored orientation is explicit because the original
projectile snapshot does not publish quaternion state. Source rig positions and
emitter areas scale consistently; this is captured Studio Phong lighting, not a
ray-traced or geodesic bake and not reciprocal world illumination.

Selected assets use original inventory stacks and slot 8 routing. A rendered
icon picker also stays accessible in default shader-only view, including phone
layout. Untagged balls retain their normal color draw. Shadow/depth/picking
proxies and colliders remain original. Exact transferred glass is drawn in
opaque/transparent phases on the original game GL context. Transparency uses
center ordering across objects, not an exact intersecting-glass solution.

Passed: original lifecycle functions for both preserved/newer matched runtime;
27 asset rejection cases; collection read-only/concurrency/observer failure
contracts; four patch-reversibility pytest checks; actual original launcher,
selection, pickup and vacuum functions. GPU comparison against current Studio
reported zero pixel differences; actual FBO thumbnail had 10,668 visible pixels,
no clipped edges, identical repeats and restored GL state.

Browser tested exact candidate/release modules at 1280x799 and 390x844. Selected
cat-eye, launched with existing fire/use, observed exact glass moving and later
sleeping with original body identities and snapshot sequence. Default ammo
decremented 128 to 127 independently. Screenshots are under the build receipt
directory. Phone layout is desktop browser testing, not physical iPhone QA.

This release contains one verified captured example, not arbitrary chosen-set
export UI or persistent collection sync. Those remain next work. The full
repository suite was not run. Original page remains available separately.

Post-publication Chrome navigation to the loopback HTTPS URL was blocked by
ERR_CERT_AUTHORITY_INVALID. No warning bypass was attempted. Trusted TLS
verification used the existing local CA and matched all published bytes to the
browser-tested release. Screenshots were uploaded as requested; IDs and local
paths are retained in the build screenshot-library receipt. Temporary local QA
server was stopped.

# Shop continuous ray-exposure override

Added a per-fixture continuous light override, default off, using existing exposureMode and shared emission rows. Continuous mode exports the enabled A fixture dimmer times RGB filter into ray-scene PBR emission. It does not simulate shutter bursts. Saved burst requests retain the authoritative timing-packet error. Disabled fixtures and zero dimmers remain dark; geometry and Studio buffers are unchanged.

Passed Node suites: meowdoku_marble_shop_lights_check, meowdoku_marble_shop_light_controls_check, meowdoku_marble_shop_bridge_check, meowdoku_marble_lighting_profiles_check, meowdoku_marble_studio_lighting_check. Added continuous emission, sibling isolation, disabled/zero handling, profile roundtrip, rollback and actual sceneForPhoto export assertions. Published only marble_shop_lights.js and marble_shop_light_controls.js, with matching source/build hashes.

Fresh live Chrome QA verified both unchecked controls, checked one and verified continuous-not-flash status, then unchecked it. Existing user page, stored profiles, materials and renderer process were not changed. Fresh-page initialization took time; no app initialization error was established. No full ray-render image was run in this task.

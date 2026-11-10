# Key shape and tool compatibility guard

Source-only narrow follow-up. Reproduced A key-shape shrink with an attached32-cell grid: changing bare to snoot retained snoot, leaving an invalid pitch. Added document-capture input/change validation before the Studio ancestor capture reader, restoring the accepted shape and displaying the validation error on failure. The accepted rollback target follows profile loads. Valid changes still work; pitch displays resync without altering attachment settings. Only marble_studio_lighting.js and marble_light_tools.js production source changed.

applyLightTools now immediately checks the actual emitter role shape before applying an enabled grid or snoot, in addition to the existing fixture-ID compatibility validation. Rectangular key roles remain rectangular regardless of the legacy shape-menu label.

Passed focused Node suites: studio_lighting (invalid shrink on both event kinds, exact state/tools/profile storage persistence, valid shrink, loaded-profile rollback target, document-capture ordering); light_tools (actual emitter shape rejection, exact legacy mesh and prior taper/seal checks); gobo_scene; lighting_profiles; table_lights; fixture_sizes. Both source modules pass syntax checks. Browser UI not tested. No publication/restart by this worker. Parent owns release.

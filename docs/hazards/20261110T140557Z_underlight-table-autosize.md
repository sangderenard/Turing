# Underlight table autosize, scoped static release

New rigs store underlightFit shortMargin/longMargin, with 0.05/0 Studio-unit defaults. Margins apply per side to the entire can/apron, not merely the emitter. Short-axis limit is the table skirt interior. Long-axis maximum is between inner leg faces: half-width minus 0.25 inset minus 0.09 leg half-width. Existing 1.25 housing expansion and 0.002 apron extension are included. Default whole assembly is 7.32 X by 3.90 Z inside the 8 X by 4 Z table. Vertical dimensions/pose and original material/emission paths are unchanged.

Existing saved fixed sizes remain exact and expose an explicit Fit saved underlight to table conversion. Conversion removes obsolete B underlight width/height overrides; other settings remain. Fitted mode exposes only the two margins and rejects off-centre or manually tilted poses rather than silently moving a rig. Existing fixed-mode poses remain unrestricted. Table resize resolves the effective dimensions before shared geometry; impossible margins/grid pitch reject. Profile restore sets the table before validating its dependent attachments.

Changed source: marble_fixture_sizes.js, marble_studio_lighting.js, marble_lighting_profiles.js. Live fixture sizes module was byte-identical before changes; other live modules received checked narrow substitutions only, retaining unrelated unpublished source differences. No server restart or session edit.

Release backup, candidates and verified HTTPS byte hashes: build/dot-underlight-autosize-20261110T140435Z/receipt.json.

PASS: meowdoku_underlight_fit_check.mjs (actual original can and apron vertices at four table sizes and three margin pairs, saved identity, invalid sizes, Room leaves); meowdoku_marble_fixture_sizes_check.mjs; meowdoku_marble_light_tools_check.mjs; meowdoku_marble_studio_lighting_check.mjs (actual control handlers, resizing, invalid rollback, save/load). Updated initial mock underlight pose to the actual centered/upward default. Candidate syntax checks pass.

Not green: broader meowdoku_marble_table_lights_check.mjs / meowdoku_marble_studio_table_check.mjs encounter existing scene-packer VM binding omissions shopEmissionProfiles / shopLightEmissionRecords. No actual phone/GPU visual test claimed. Browser sessions untouched. Existing tabs must load the updated modules to see the controls.

2026-11-10 Current game renderer resource contract repaired (MEASURED)

Current matched game4ff14dc8aa23ff7e809944abae88edaeabaf5aff7d7bc3aea512c66fc11de472 included Pluck ThermalChunk/thermal_k in shader resource bindings. configurePluckPhongResources supplied only pbr/phong/enamel/texstack; undefined reached createScalarTexture.values.length. Thermal recommended unit16 also exceeded the portable16-unit fragment range.

src/compiler/abstract_ui_div_map.py now binds the unauthored thermal channel to an explicit inactive[0] texture with uThermalEmission/uThermalGain0, matching existing Studio policy rather than inventing temperatures. Material samplers map to5..9, disjoint from identity0..4 and shadow12. Unknown future bindings fail with a named error before allocation; scalar uploads require nonempty finite values.

New tests/test_pluck_viewport_resources.py executes the actual JavaScript resource setup with a recording GL contract. It covers all five current bindings, exact lengths, inactive thermal data/uniforms, noncolliding sampler units despite manifest recommendations16+, unknown-binding rejection before allocation, invalid data rejection.1test passed0.28s; renderer Node syntax passed.

Guarded current candidate build/dot-current-renderer-repair-20261110/index.html SHA25696bd5ef1fce0553a4aa7f1aa3a65384e9ebcba114fc2aa73cd7b30ed663c3223 substitutes only the resource-function slice. Reverse replacement exactly restores original bytes. Model, physics, qualification and kernels unchanged. Receipt adjacent. Candidate is not a normal full rebuild; next build must capture source.

SQL-managed independent localhost8901 QA showed qualification running with the original20second protocol and no renderer resource error in console. Only previously seen Adblock extension checksum errors remain. Qualification was not bypassed and this does not establish vehicle qualification success or full game visual acceptance. Tire worker owns the separate telemetry/qualification bug. No live state, Apache,8766,8767or Nodus changes.

Final browser gate at20.00s: configuration failed physical qualification, living map withheld; tire crossing510.0mm exceeds72.3mm,2/4contacts,22J,epsilon.308. No bypass. Own browser closed and QA session54016 stopped. Combined renderer/audio regression8passed2.06s with existing cffi deprecation warning.

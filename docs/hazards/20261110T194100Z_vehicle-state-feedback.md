2026-11-10 VEHICLE STATE FEEDBACK CONTRACT (measured, Windows)

Exact matched current worker drive probe at5seconds: commanded throttle.3, effective/applied.196840135, engine94.014Nm, clutch102.350Nm, driveline995.181Nm. Differential shaft inputs stayed0 while compiled outputs were front15.69743355rad/s and rear21.67740823rad/s. Both outputs were discarded by browser handoff, so engaged hubs never received a rotating shaft. GPU commit map omitted them too. This is state persistence loss, not a tire/engine law repair.

Existing GPU direct feedback map moved to one explicit VEHICLE_STATE_FEEDBACK declaration in abstract_ui_vehicles.py. It includes all90 authored next-state outputs and corresponding existing machine inputs. Added previously omitted differential shaft speeds, differential/hub locker wear and glaze, traction battery charge, clutch temperature/health/wear/glaze, alternator CVT ratio/wear/glaze, accessory battery charge, air reserve gas mass/temperature, and bypass engagement/tooth health. The worker emitter serializes the same declaration; both capture paths consume it while preserving unrelated active state. Missing old-version output fields leave previous state intact. GPU commit uses that same mapping, retaining its finite_or policy.

No runtime suffix guessing, unit conversions, replacement solver, compiler-core changes or physical law edits. Existing body projections for position, wheels and suspension still provide the next-feed overlay. Worker source has no alternate pre-step assignments to the newly retained charge/thermal/reserve/CVT state names. energy.batteryChargeWh is a separate ordinary electrical bookkeeping field, not a conversion authority for traction_battery_charge_fraction or accessory_battery_cube_charge_fraction. Broader electrical-model semantics were not changed or claimed unified.

Gate python -m pytest tests/test_vehicle_differential_state_handoff.py -q:2passed in33.89s. Verifies completeness against all authored next outputs, every existing input identity and GPU commit slot, actual Wasm output handoff to next input object for all90 pairs, retained unrelated active state and missing older fields, GPU capture hook and shaft reupload. GPU hardware execution and resulting drive stability not yet established. A full fresh matched build is needed for GPU adapter changes; no mismatched worker-only live publication.

Secondary finding, not changed: engine speed uses _c2_clamp(raw,0,1.03*redline,.5), whose lower smooth positive part maps exactly0 to about.25rad/s. Reapplying injects speed even with ignition/starter off until drag balances. Combustion and idle torque both multiply engine_enabled. This explains a source-level off-crank drift mechanism; no tuning/fix performed.

Full explicit feedback pairs:
position_x_next -> position_x
position_y_next -> position_y
position_z_next -> position_z
velocity_x_next -> velocity_x
velocity_y_next -> velocity_y
velocity_z_next -> velocity_z
roll_next -> roll
pitch_next -> pitch
yaw_next -> yaw
roll_velocity_next -> roll_velocity
pitch_velocity_next -> pitch_velocity
yaw_velocity_next -> yaw_velocity
wheel_omega_front_left_next -> wheel_omega_front_left
wheel_omega_front_right_next -> wheel_omega_front_right
wheel_omega_rear_left_next -> wheel_omega_rear_left
wheel_omega_rear_right_next -> wheel_omega_rear_right
wheel_angle_front_left_next -> wheel_angle_front_left
wheel_angle_front_right_next -> wheel_angle_front_right
wheel_angle_rear_left_next -> wheel_angle_rear_left
wheel_angle_rear_right_next -> wheel_angle_rear_right
slip_longitudinal_front_left_next -> previous_slip_longitudinal_front_left
slip_longitudinal_front_right_next -> previous_slip_longitudinal_front_right
slip_longitudinal_rear_left_next -> previous_slip_longitudinal_rear_left
slip_longitudinal_rear_right_next -> previous_slip_longitudinal_rear_right
slip_sensor_velocity_front_left_next -> slip_sensor_velocity_front_left
slip_sensor_velocity_front_right_next -> slip_sensor_velocity_front_right
slip_sensor_velocity_rear_left_next -> slip_sensor_velocity_rear_left
slip_sensor_velocity_rear_right_next -> slip_sensor_velocity_rear_right
friction_utilization_front_left_next -> friction_utilization_front_left
friction_utilization_front_right_next -> friction_utilization_front_right
friction_utilization_rear_left_next -> friction_utilization_rear_left
friction_utilization_rear_right_next -> friction_utilization_rear_right
friction_utilization_sensor_velocity_front_left_next -> friction_utilization_sensor_velocity_front_left
friction_utilization_sensor_velocity_front_right_next -> friction_utilization_sensor_velocity_front_right
friction_utilization_sensor_velocity_rear_left_next -> friction_utilization_sensor_velocity_rear_left
friction_utilization_sensor_velocity_rear_right_next -> friction_utilization_sensor_velocity_rear_right
tire_deformation_longitudinal_front_left_next -> tire_deformation_longitudinal_front_left
tire_deformation_longitudinal_front_right_next -> tire_deformation_longitudinal_front_right
tire_deformation_longitudinal_rear_left_next -> tire_deformation_longitudinal_rear_left
tire_deformation_longitudinal_rear_right_next -> tire_deformation_longitudinal_rear_right
tire_deformation_velocity_longitudinal_front_left_next -> tire_deformation_velocity_longitudinal_front_left
tire_deformation_velocity_longitudinal_front_right_next -> tire_deformation_velocity_longitudinal_front_right
tire_deformation_velocity_longitudinal_rear_left_next -> tire_deformation_velocity_longitudinal_rear_left
tire_deformation_velocity_longitudinal_rear_right_next -> tire_deformation_velocity_longitudinal_rear_right
tire_deformation_lateral_front_left_next -> tire_deformation_lateral_front_left
tire_deformation_lateral_front_right_next -> tire_deformation_lateral_front_right
tire_deformation_lateral_rear_left_next -> tire_deformation_lateral_rear_left
tire_deformation_lateral_rear_right_next -> tire_deformation_lateral_rear_right
tire_deformation_velocity_lateral_front_left_next -> tire_deformation_velocity_lateral_front_left
tire_deformation_velocity_lateral_front_right_next -> tire_deformation_velocity_lateral_front_right
tire_deformation_velocity_lateral_rear_left_next -> tire_deformation_velocity_lateral_rear_left
tire_deformation_velocity_lateral_rear_right_next -> tire_deformation_velocity_lateral_rear_right
compression_front_left_next -> compression_front_left
compression_front_right_next -> compression_front_right
compression_rear_left_next -> compression_rear_left
compression_rear_right_next -> compression_rear_right
compression_velocity_front_left_next -> compression_velocity_front_left
compression_velocity_front_right_next -> compression_velocity_front_right
compression_velocity_rear_left_next -> compression_velocity_rear_left
compression_velocity_rear_right_next -> compression_velocity_rear_right
engine_angular_speed_next -> engine_angular_speed
differential_wrench_shaft_omega_front_next -> differential_wrench_shaft_omega_front
differential_wrench_shaft_omega_rear_next -> differential_wrench_shaft_omega_rear
traction_battery_charge_fraction_next -> traction_battery_charge_fraction
clutch_temperature_k_next -> clutch_temperature_k
clutch_health_next -> clutch_health
clutch_wear_next -> clutch_wear
clutch_glaze_next -> clutch_glaze
alternator_cvt_ratio_state_next -> alternator_cvt_ratio_state
alternator_cvt_wear_next -> alternator_cvt_wear
alternator_cvt_glaze_next -> alternator_cvt_glaze
accessory_battery_cube_charge_fraction_next -> accessory_battery_cube_charge_fraction
air_mix_reserve_gas_mass_kg_next -> air_mix_reserve_gas_mass_kg
air_mix_reserve_temperature_k_next -> air_mix_reserve_temperature_k
direct_drive_bypass_engagement_next -> direct_drive_bypass_engagement
direct_drive_bypass_tooth_health_next -> direct_drive_bypass_tooth_health
hub_locker_wear_front_left_next -> hub_locker_wear_front_left
hub_locker_wear_front_right_next -> hub_locker_wear_front_right
hub_locker_wear_rear_left_next -> hub_locker_wear_rear_left
hub_locker_wear_rear_right_next -> hub_locker_wear_rear_right
hub_locker_glaze_front_left_next -> hub_locker_glaze_front_left
hub_locker_glaze_front_right_next -> hub_locker_glaze_front_right
hub_locker_glaze_rear_left_next -> hub_locker_glaze_rear_left
hub_locker_glaze_rear_right_next -> hub_locker_glaze_rear_right
differential_locker_wear_front_next -> differential_locker_wear_front
differential_locker_wear_rear_next -> differential_locker_wear_rear
differential_locker_wear_center_next -> differential_locker_wear_center
differential_locker_glaze_front_next -> differential_locker_glaze_front
differential_locker_glaze_rear_next -> differential_locker_glaze_rear
differential_locker_glaze_center_next -> differential_locker_glaze_center

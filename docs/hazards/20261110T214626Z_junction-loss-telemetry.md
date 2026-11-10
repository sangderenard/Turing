# Junction loss telemetry startup correction (measured)

Browser generic bushing coefficients feed estimated loss telemetry, not equal/opposite mechanical damping. Initial previousLength used authored rest length, so first projected pose generated a false finite-difference heat impulse. The new lossMeasurementInitialized flag suppresses only the first estimated linear loss sample, then uses the existing rate/history. Geometry profile reset clears that flag. Existing relativeLinearSpeedMps remains unchanged because alignment strain-relief consumes it; that initial actuator-rate behavior remains unaudited and is not claimed repaired.

Added explicit estimated-constitutive-loss metadata with mechanicalEnergyRemoved=false and angular-rate proxy warning. Existing telemetry field names remain compatibility aliases. No tire, joint-force, suspension or drivetrain law changed.

Actual authored worker-function tests verify initial displaced zero-rate sample, stationary second sample, known subsequent displacement, 100x coefficient scaling without mechanical state change, metadata and reset contract. Command: python -m pytest tests/test_vehicle_junction_loss_telemetry.py -q. Result: 2 passed in 0.28 seconds. python -m py_compile src/compiler/state_loop_deployment.py passed. Full matched build/runtime and GPU checks belong to parent worker and are not claimed here.

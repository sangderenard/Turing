"""The existing qualification gate must compare length to length, not radians."""
from pathlib import Path
import subprocess


def test_qualification_reads_peak_depth_not_arc_angle():
    source = Path('src/compiler/abstract_ui_div_map.py').read_text(encoding='utf-8')
    start = source.index('function vehiclePeakTirePenetration(')
    end = source.index('\nfunction ', start + 1)
    function = source[start:end]
    code = function + "\nconst assert=require('assert');assert.equal(vehiclePeakTirePenetration({radialProbePenetrations:[[.51,.003,.006,.012],[.7,.004,.008,.02]]}),.02);assert.equal(vehiclePeakTirePenetration({radialProbePenetrations:[[0,0,0,0]]}),0);"
    subprocess.run(['node', '-e', code], check=True, capture_output=True, text=True)


def test_both_contact_lanes_publish_peak_depth_in_reserved_slot():
    worker = Path('src/compiler/state_loop_deployment.py').read_text(encoding='utf-8')
    shader = Path('src/compiler/abstract_ui_vehicles.py').read_text(encoding='utf-8')
    assert 'probes[3]=best.penetration' in worker
    assert 'radial_probes[lane * 15u + 3u] = tire_radial_compression;' in shader
    assert 'radialProbeActiveCounts=body.radialProbePenetrations.map(samples=>Number(samples[3]>0))' in worker


def test_qualification_returns_regular_game_with_ignition_off():
    source = Path('src/compiler/abstract_ui_div_map.py').read_text(encoding='utf-8')
    start = source.index('function finishVehicleQualification(')
    end = source.index('\nfunction ', start + 1)
    assert 'ignitionOn:false,starterEngaged:false' in source[start:end]
    assert 'ignitionOn:q.passed' not in source[start:end]

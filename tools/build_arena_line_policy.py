"""Build a five-hidden-neuron line follower through the existing graph compiler.

Uploads contain weights and graph masks only, never source or executable modules.
The complimentary weights are an uncalibrated starting policy, not a trained one.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'examples')]


def starter_params():
    import numpy as np
    from meowdoku.pilot_law import graph_param_shapes
    params = {k: np.zeros(v) for k, v in graph_param_shapes(15, 3, 5).items()}
    params['GM1'][:] = 1
    params['GM2'][:] = 1
    # Sensor order is four local xyz points / 10, forward/lateral speed / 10,
    # yaw rate. Hidden gates encode near/far lateral error and current motion.
    for feature, neuron, gain in [(2, 0, 5), (11, 1, 3), (12, 2, 4), (13, 3, 3), (14, 4, 1)]:
        params['GW1'][feature, neuron] = gain
    params['GW2'][:, 0] = [-1.2, -1.8, 0, .3, .4]
    params['Gb2'][0, 0] = 1.15
    params['GW2'][2, 1] = -1.5
    params['Gb2'][0, 1] = 1.25
    params['Gb2'][0, 2] = -1
    return params


def build(destination):
    import json
    from meowdoku.graph_wasm import build_pilot
    destination.mkdir(parents=True, exist_ok=True)
    result = build_pilot(destination, params=starter_params())
    (destination / 'turing-wasm-shell.js').write_bytes(
        (ROOT / 'examples/mobile_reference_app/web/turing-wasm-shell.js').read_bytes())
    print(json.dumps({'features': result['features'], 'outputs': result['outputs_width'],
                     'weights': result['weights'], 'destination': str(destination)}), flush=True)
    return result


def write_starter_circuits(destination, model_path):
    import json, math
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    terrain = next(b for b in model["document_geometry"]["boxes"]
                   if b["identity"].endswith("/outer-field-offroad-playground"))
    cx, cz = terrain["center"]
    circuits = []
    for name, color, offset in [("arena-blue", "#38bdf8", -.35), ("arena-amber", "#fbbf24", .35)]:
        points = [[cx+(16+offset)*math.cos(2*math.pi*i/48),
                   cz+(12+offset)*math.sin(2*math.pi*i/48)] for i in range(49)]
        points[-1] = points[0]
        circuits.append({"id": name, "color": color, "segments": [
            {"support": terrain["identity"]+"/surface:sampled-terrain", "from": a, "to": b}
            for a, b in zip(points, points[1:])]})
    weights = json.loads((destination/"pilot.weights.json").read_text())
    packet = {"schema": "turing.arena-line-driver.v1", "network": {
        "architecture": "graph-15-5-3-tanh-gates-v1", "weights": weights}, "circuits": circuits}
    (destination/"starter-circuits.json").write_text(json.dumps(packet, indent=2), encoding="utf-8")
    return packet


if __name__ == '__main__':
    destination = Path(sys.argv[1])
    build(destination)
    if len(sys.argv) > 2:
        write_starter_circuits(destination, Path(sys.argv[2]))

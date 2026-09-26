"""Wrap the emitted contact law in extended precision and see what 64-bit buys.

The law leaves sympy as AbstractTensor code (``abstract_tensor_source`` in the
vehicle program).  It is a pure scalar expression over + - * / ** sqrt tanh,
which is exactly the operator surface ``Precision`` carries, so the same source
runs unmodified at float32, float64 and a double-double stack of float64 limbs.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.extended_precision import Precision, limb_element_facts

DEMO = Path(r'C:/dev/Powershell/site/demos/living-data-map/index.html')


def emitted_source():
    raw = max(DEMO.read_text(encoding='utf-8').split('\n'), key=len)
    model = json.loads(raw[raw.index('>') + 1:raw.rindex('</script>')])
    return model['vehicle_slot']['programs'][0]['tensor_contact_precompile']


def load_law(source):
    namespace = {}
    exec(source, namespace)
    return namespace['abstract_ui_wheel_contact_tensor']


# The working point the geometry kernel actually writes into the contact feed:
# every constant is the literal from the kernel, every dynamic value is a loaded
# wheel rolling with a little slip.
WORKING_POINT = {
    'active_damping_body_velocity_gain_s_per_m': 0.22,
    'active_damping_maximum_scale': 1.18,
    'active_damping_minimum_scale': 0.88,
    'active_damping_rebound_release_gain_s_per_m': 0.08,
    'attachment_x': 0.6448, 'attachment_y': -0.2894, 'attachment_z': -0.78,
    'chassis_velocity_y': -0.14,
    'corner_front_sign': 1.0, 'corner_side_sign': -1.0,
    'corner_weight': 1459.7121862800002,
    'dt': 1.0 / 120.0,
    'forward_x': 0.99861, 'forward_y': 0.04382, 'forward_z': 0.02771,
    'geometric_compression': 0.0869,
    'lateral_stiffness': 14000.0,
    'linkage_motion_ratio': 1.0,
    'load_sensitivity': 0.075,
    'longitudinal_stiffness': 9200.0,
    'maximum_compression_speed': 1.25,
    'maximum_contact_area': 0.06,
    'minimum_contact_area': 0.008,
    'mu_kinetic': 0.92, 'mu_static': 1.18,
    'normal_x': -0.0431, 'normal_y': 0.99835, 'normal_z': -0.0376,
    'pitch_velocity': 0.21,
    'pneumatic_compression_damping': 3200.0,
    'pneumatic_efficiency': 0.96,
    'pneumatic_rebound_damping': 4100.0,
    'previous_compression': 0.0841,
    'right_x': -0.02705, 'right_y': -0.03864, 'right_z': 0.99889,
    'roll_velocity': -0.13,
    'slip_lateral': 0.0472, 'slip_longitudinal': -0.3118,
    'slip_transition_speed': 0.42,
    'spring_stiffness': 26000.0,
    'support': 1.0,
    'suspension_alignment': 0.99127,
    'suspension_travel': 0.26,
    'tire_pressure': 155000.0,
    'tire_radial_compression': 0.0091,
    'tire_radial_damping': 2600.0,
    'tire_radial_stiffness': 32000.0,
    'tire_radial_velocity': -0.0624,
    'track_half_width': 0.78,
    'wheelbase_half_length': 0.62,
}


def install_tanh_range_reduction():
    """Give the wide tanh core the reduction it asks for.

    The core is proven on +-0.5 and refuses outside it rather than returning a
    plausible wrong answer.  The law's argument is a slip magnitude over a
    friction capacity, which at load runs past 0.75 and can go far higher, so
    the argument is halved until it lands inside the interval and the result is
    doubled back out with tanh(2u) = 2 tanh(u) / (1 + tanh(u)^2).  That identity
    is exact and uses only the operators Precision already carries, so nothing
    is approximated outside the core's own domain.
    """

    core = Precision.tanh
    depth = {'halvings': 0}

    def peak(value):
        flat = value.collapse().tolist()
        while isinstance(flat, list):
            if not flat:
                return 0.0
            flat = max(flat, key=lambda item: abs(item) if isinstance(item, float) else 0.0) \
                if all(isinstance(item, float) for item in flat) else flat[0]
        return abs(float(flat))

    def reduced(self):
        magnitude, halvings, value = peak(self), 0, self
        while magnitude > 0.5 and halvings < 60:
            value = value * 0.5
            magnitude *= 0.5
            halvings += 1
        depth['halvings'] = max(depth['halvings'], halvings)
        result = core(value)
        for _ in range(halvings):
            result = (result * 2.0) / (result * result + 1.0)
        return result

    Precision.tanh = reduced
    return depth


def build(names, count, dtype, limbs=None):
    """One tensor per parameter, all of them the same width."""
    values = {}
    for name in names:
        base = float(WORKING_POINT[name])
        tensor = AbstractTensor.get_tensor([base] * count)
        tensor = tensor.to_dtype(dtype)
        values[name] = Precision.of(tensor, limbs) if limbs else tensor
    return values


def as_floats(result):
    out = []
    for item in result:
        if isinstance(item, Precision):
            terms = item.to_float_lists()
            out.append(sum(term[0] for term in terms))
        else:
            out.append(float(item.tolist()[0]))
    return out


def exact(result):
    """Full value of a wide result: every limb summed in python floats."""
    out = []
    for item in result:
        terms = item.to_float_lists()
        out.append(sum(term[0] for term in terms))
    return out


def run(law, values, repeats):
    start = time.perf_counter()
    for _ in range(repeats):
        result = law(**values)
    return result, (time.perf_counter() - start) / repeats


def main():
    block = emitted_source()
    law = load_law(block['abstract_tensor_source'])
    depth = install_tanh_range_reduction()
    names = block['inputs']
    outputs = block['outputs']
    missing = [name for name in names if name not in WORKING_POINT]
    assert not missing, missing
    print('law            :', law.__name__)
    print('inputs/outputs :', len(names), '->', len(outputs))
    print('limb facts     :', limb_element_facts())
    print()

    LANES = (
        ('float32', 'float32', None),
        ('float64', 'float64', None),
        ('float64 x2 limbs', 'float64', 2),
        ('float64 x3 limbs', 'float64', 3),
    )

    # Accuracy needs one working point, not a batch; width is what is expensive.
    lanes = {}
    for label, dtype, limbs in LANES:
        values = build(names, 1, dtype, limbs)
        result, seconds = run(law, values, 1)
        lanes[label] = {'value': exact(result) if limbs else as_floats(result)}
        print('%-18s evaluated  (%7.1f ms)' % (label, seconds * 1e3))

    # Throughput is measured separately, on a batch, at the widths that can
    # plausibly carry one.
    print()
    for label, dtype, limbs in LANES:
        count = 4096 if limbs is None else 256
        repeats = 20 if limbs is None else 1
        values = build(names, count, dtype, limbs)
        _, seconds = run(law, values, repeats)
        per_element = seconds / count * 1e9
        lanes[label]['seconds'] = seconds
        lanes[label]['ns'] = per_element
        print('%-18s %10.3f ms for %5d elements  = %9.1f ns/element'
              % (label, seconds * 1e3, count, per_element))

    print()
    truth = lanes['float64 x3 limbs']['value']
    print('%-18s %-22s %-12s %s' % ('output', 'reference (3 limbs)', 'f64 rel err', 'f32 rel err'))
    for index, name in enumerate(outputs):
        reference = truth[index]
        scale = abs(reference) if abs(reference) > 1e-30 else 1.0
        f64 = abs(lanes['float64']['value'][index] - reference) / scale
        f32 = abs(lanes['float32']['value'][index] - reference) / scale
        print('%-18s %-22.12g %-12.3e %.3e' % (name, reference, f64, f32))

    print()
    base = lanes['float64']['ns']
    for label in lanes:
        print('%-18s %8.1fx float64' % (label, lanes[label]['ns'] / base))
    print()
    print('tanh range reduction: %d halvings to reach the core interval' % depth['halvings'])


if __name__ == '__main__':
    main()

import numpy as np
from src.hardware.analog_spec import (
    FRAME_SAMPLES,
    generate_bit_wave,
    sigma_L,
    sigma_R,
    BiosHeader,
    BIOS_HEADER_STRUCT,
    MAGIC_ID,
    header_frames,
    pack_bios_header,
    unpack_bios_header,
    LANES,
    BIT_FRAME_MS,
    WRITE_BIAS,
)
from src.turing_machine.tape_map import TapeMap, create_register_tapes
from src.hardware.cassette_tape import CassetteTapeBackend
from src.compiler.bitops import BitOps
from src.turing_machine.turing_provenance import ProvenanceGraph, ProvNode, ProvEdge
from src.compiler.ssa_builder import graph_to_ssa_with_loops
import src.hardware.nand_wave as nw
from src.turing_machine.turing_ssa import ProvenanceTM, SSAView


def run_test(name, func):
    print(f"=== {name} ===")
    try:
        func()
        print(f"{name}: PASS\n")
        return True
    except AssertionError as e:
        print(f"{name}: FAIL - {e}\n")
        return False
    except Exception as e:
        print(f"{name}: ERROR - {e}\n")
        return False


def test_analog_spec():
    print("Generating bit waves for 1 and 0 and checking shapes and amplitudes.")
    one = generate_bit_wave(1, 0)
    zero = generate_bit_wave(0, 0)
    print(f"one shape {one.shape}, zero shape {zero.shape}")
    print(f"max amplitude one {np.max(np.abs(one))}, zero {np.max(np.abs(zero))}")
    assert one.shape == (FRAME_SAMPLES,)
    assert zero.shape == (FRAME_SAMPLES,)
    assert np.max(np.abs(one)) > 0.0
    assert np.max(np.abs(zero)) == 0.0

    print("Testing sigma_L then sigma_R on bit waves.")
    frames = [generate_bit_wave(1, 0)] * 2
    appended = sigma_L(frames, 1)
    print(f"sigma_L length {len(appended)}, last frame max {np.max(np.abs(appended[-1]))}")
    assert len(appended) == 3
    assert np.max(np.abs(appended[-1])) == 0.0
    trimmed = sigma_R(appended, 2)
    print(f"sigma_R length {len(trimmed)} after trimming")
    assert len(trimmed) == 1

    print("Checking envelope edges for sigma ops.")
    frames = [np.ones(FRAME_SAMPLES, dtype='f4') for _ in range(2)]
    left = sigma_L(frames, 0)
    right = sigma_R(frames, 0)
    print(f"left start {left[0][0]} end {left[-1][-1]}; right start {right[0][0]} end {right[-1][-1]}")
    assert abs(left[0][0]) < 1e-3 and abs(left[-1][-1]) < 1e-3
    assert abs(right[0][0]) < 1e-3 and abs(right[-1][-1]) < 1e-3


def test_cassette_tape():
    print("Testing read/write emits audio.")
    tape = CassetteTapeBackend(tape_length_inches=0.02, op_sine_coeffs={'read': {440.0:1.0}, 'write': {880.0:1.0}, 'motor': {60.0:0.5}})
    frame0 = generate_bit_wave(1,0)
    tape.write_wave(0,0,0,frame0)
    out0 = tape.read_wave(0,0,0)
    frame1 = generate_bit_wave(1,1)
    tape.write_wave(0,1,1,frame1)
    out1 = tape.read_wave(0,1,1)
    print(f"out0 max {np.max(np.abs(out0))}, out1 max {np.max(np.abs(out1))}, audio cursor {tape._audio_cursor}")
    assert np.max(np.abs(out0)) > 0.0
    assert np.max(np.abs(out1)) > 0.0
    assert tape._audio_cursor > 0
    tape.close()

    print("Testing move_head_to_bit and write_bit wrappers.")
    tape = CassetteTapeBackend(tape_length_inches=0.02)
    tape.move_head_to_bit(5)
    tape.write_bit(0,0,5,1)
    rb1 = tape.read_bit(0,0,5)
    tape.write_bit(0,1,6,1)
    rb2 = tape.read_bit(0,1,6)
    rb3 = tape.read_bit(0,0,6)
    print(f"read bits -> addr5 {rb1}, track1bit6 {rb2}, track0bit6 {rb3}")
    assert rb1 == 1 and rb2 == 1 and rb3 == 0
    tape.close()

    print("Testing head gating on speed.")
    tape = CassetteTapeBackend(tape_length_inches=0.02)
    frame = generate_bit_wave(1,0)
    tape._tape_frames[(0,0,0)] = frame
    tape.move_head_to_bit(0)
    tape._head.enqueue_read(0,0,0)
    wrong = tape._head.activate(0, 'read', 0.5)
    correct = tape._head.activate(0, 'read', tape.read_write_speed_ips)
    print(f"wrong speed result {wrong}, correct speed amplitude {np.max(np.abs(correct)) if correct is not None else None}")
    assert wrong is None
    assert np.max(np.abs(correct)) > 0
    tape.close()

    print("Testing that write adds bias tone.")
    tape = CassetteTapeBackend(tape_length_inches=0.02)
    frame = generate_bit_wave(1,0)
    tape.write_wave(0,0,0,frame)
    out = tape.read_wave(0,0,0)
    t = np.linspace(0, BIT_FRAME_MS / 1000.0, FRAME_SAMPLES, endpoint=False)
    bias_wave = 0.1 * np.sin(2 * np.pi * WRITE_BIAS * t)
    diff = out - frame
    dot = abs(np.dot(diff, bias_wave))
    print(f"bias tone correlation {dot}")
    assert np.max(np.abs(diff)) > 0.0
    assert dot > 100.0
    tape.close()


def test_bitops_translator():
    print("Testing BitOps translated operations.")
    a, b, width = 0b1010, 0b0110, 4
    ops = BitOps(bit_width=width, encoding='binary')
    print(f"bit_and -> {ops.bit_and(a,b)} expected {a & b}")
    print(f"bit_or -> {ops.bit_or(a,b)} expected {a | b}")
    print(f"bit_xor -> {ops.bit_xor(a,b)} expected {a ^ b}")
    print(f"bit_not -> {ops.bit_not(a)} expected {(~a) & ((1<<width)-1)}")
    print(f"shift_left -> {ops.bit_shift_left(a,1)} expected {(a << 1) & ((1<<width)-1)}")
    print(f"shift_right -> {ops.bit_shift_right(a,1)} expected {a >> 1}")
    print(f"add -> {ops.bit_add(a,b)} expected {(a + b) & ((1<<width)-1)}")
    print(f"sub -> {ops.bit_sub(a,b)} expected {(a - b) & ((1<<width)-1)}")
    print(f"mul -> {ops.bit_mul(a,b)} expected {(a * b) & ((1<<width)-1)}")
    print(f"div -> {ops.bit_div(a,b)} expected {(a // b) & ((1<<width)-1)}")
    print(f"mod -> {ops.bit_mod(a,b)} expected {(a % b) & ((1<<width)-1)}")
    assert ops.bit_and(a,b) == (a & b)
    assert ops.bit_or(a,b) == (a | b)
    assert ops.bit_xor(a,b) == (a ^ b)
    assert ops.bit_not(a) == ((~a) & ((1<<width)-1))
    assert ops.bit_shift_left(a,1) == ((a << 1) & ((1<<width)-1))
    assert ops.bit_shift_right(a,1) == (a >> 1)
    assert ops.bit_add(a,b) == ((a + b) & ((1<<width)-1))
    assert ops.bit_sub(a,b) == ((a - b) & ((1<<width)-1))
    assert ops.bit_mul(a,b) == ((a * b) & ((1<<width)-1))
    assert ops.bit_div(a,b) == ((a // b) & ((1<<width)-1))
    assert ops.bit_mod(a,b) == ((a % b) & ((1<<width)-1))
    nodes = len(ops.translator.graph.nodes)
    edges = len(ops.translator.graph.edges)
    print(f"translator graph nodes {nodes}, edges {edges}")
    assert nodes and edges


def bits_to_bytes(frames):
    bits = [b for frame in frames for b in frame]
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i+8]:
            byte = (byte << 1) | bit
        out.append(byte)
    return bytes(out)


def test_header_layout():
    print("Testing BIOS header pack/unpack and tape map layout.")
    h = BiosHeader(calib_fast_ms=1.0, calib_read_ms=2.0, drift_ms=0.1, inputs=[0,1], outputs=[2], instr_start_addr=1234)
    packed = pack_bios_header(h)
    print(f"packed length {len(packed)} expected {BIOS_HEADER_STRUCT.size}")
    assert len(packed) == BIOS_HEADER_STRUCT.size
    unpacked = unpack_bios_header(packed)
    print(f"unpacked inputs {unpacked.inputs}, outputs {unpacked.outputs}, instr_start_addr {unpacked.instr_start_addr}")
    assert unpacked.calib_fast_ms == h.calib_fast_ms
    assert unpacked.inputs == h.inputs
    assert unpacked.outputs == h.outputs
    assert unpacked.instr_start_addr == h.instr_start_addr

    frames = header_frames(h)
    print(f"header frames count {len(frames)}, frame width {len(frames[0]) if frames else 0}")
    reconstructed = bits_to_bytes(frames)[:len(MAGIC_ID)]
    print(f"reconstructed magic id {reconstructed}")
    assert reconstructed == MAGIC_ID

    tmap = TapeMap(h, instruction_frames=10)
    print(f"tmap instr_start {tmap.instr_start}, data_start {tmap.data_start}")
    assert tmap.instr_start == len(header_frames(h))
    assert tmap.data_start == tmap.instr_start + 10

    tmap = TapeMap(h, instruction_frames=0)
    frames = tmap.encode_bios()
    decoded = TapeMap.decode_bios(frames)
    print(f"decoded instr_start_addr {decoded.instr_start_addr}")
    assert decoded.instr_start_addr == h.instr_start_addr

    regs = create_register_tapes(h)
    print(f"register keys {regs.keys()}")
    assert set(regs.keys()) == {0,1,2}
    for r in regs.values():
        assert r.instr_start == len(header_frames(h))
        assert r.data_start == r.instr_start


def test_loops():
    print("Testing SSA loop builder for phi nodes.")
    pg = ProvenanceGraph()
    pg._nodes = [ProvNode(0,'init',(),{},0), ProvNode(1,'cmp',(),{},1), ProvNode(2,'inc',(),{},2)]
    pg._edges = [ProvEdge(0,1,0), ProvEdge(1,2,0), ProvEdge(2,1,0)]
    ssa = graph_to_ssa_with_loops(pg)
    print(f"SSA output:\n{ssa}")
    assert '%n1.phi0' in ssa and 'phi' in ssa

    pg = ProvenanceGraph()
    pg._nodes = [ProvNode(0,'start',(),{},0), ProvNode(1,'check',(),{},1), ProvNode(2,'rot',(),{},2)]
    pg._edges = [ProvEdge(0,1,0), ProvEdge(1,2,0), ProvEdge(2,1,0)]
    ssa = graph_to_ssa_with_loops(pg)
    print(f"SSA rotate:\n{ssa}")
    assert '%n1.phi0' in ssa

    pg = ProvenanceGraph()
    pg._nodes = [
        ProvNode(0,'i0',(),{},0), ProvNode(1,'outer_chk',(),{},1),
        ProvNode(2,'j0',(),{},2), ProvNode(3,'inner_chk',(),{},3),
        ProvNode(4,'inner_inc',(),{},4), ProvNode(5,'outer_inc',(),{},5),
    ]
    pg._edges = [
        ProvEdge(0,1,0), ProvEdge(1,5,0), ProvEdge(5,1,0),
        ProvEdge(1,3,0), ProvEdge(2,3,0), ProvEdge(3,4,0), ProvEdge(4,3,0),
    ]
    ssa = graph_to_ssa_with_loops(pg)
    print(f"SSA two loops:\n{ssa}")
    assert '%n1.phi0' in ssa and '%n3.phi0' in ssa


def test_motor_profile():
    print("Testing motor speed profile integration.")
    tape = CassetteTapeBackend(time_scale_factor=0.0)
    distance = 2.0
    profile = tape._generate_speed_profile(distance, tape.seek_speed_ips)
    dt = 1.0 / tape.sample_rate_hz
    travelled = float(np.sum(profile)) * dt
    print(f"travelled {travelled} expected {distance}")
    assert abs(travelled - distance) / distance < 1e-3
    assert profile[0] < 1e-2 and profile[-1] < 1e-2
    tape.close()

    print("Testing simulate movement writes audio.")
    tape = CassetteTapeBackend(time_scale_factor=0.0)
    tape._simulate_movement(1.0, tape.seek_speed_ips, 1, 'seek')
    print(f"audio cursor {tape._audio_cursor}, any audio {np.any(tape._audio_buffer[:tape._buffer_cursor])}")
    assert tape._audio_cursor > 0
    assert np.any(tape._audio_buffer[:tape._buffer_cursor])
    tape.close()


def test_nand_wave():
    print("Testing parallel XOR copy via NAND wave.")
    lane = 5
    x = nw.generate_bit_wave(lane)
    y = np.zeros(nw.FRAME_SAMPLES, dtype='f4')
    out = nw.nand_wave(x, y, mode='parallel', lane_mask=1<<lane, energy_thresh=0.01)
    corr = np.corrcoef(out, x)[0,1]
    print(f"correlation {corr}")
    assert abs(corr - 1.0) <= 2e-2

    print("Testing parallel NOR generates fresh one.")
    lane = 2
    x = np.zeros(nw.FRAME_SAMPLES, dtype='f4')
    y = np.zeros_like(x)
    out = nw.nand_wave(x, y, mode='parallel', lane_mask=1<<lane, energy_thresh=0.01)
    expected = nw.generate_bit_wave(lane)
    corr = np.corrcoef(out, expected)[0,1]
    print(f"correlation {corr}")
    assert abs(corr - 1.0) <= 1e-2

    print("Testing parallel both-on gives silence.")
    lane = 4
    x = nw.generate_bit_wave(lane)
    y = nw.generate_bit_wave(lane)
    out = nw.nand_wave(x, y, mode='parallel', lane_mask=1<<lane, energy_thresh=0.01)
    max_amp = np.max(np.abs(out))
    print(f"max amplitude {max_amp}")
    assert max_amp < 1e-6

    print("Testing dominant XOR envelope replay.")
    src_lane = 3
    target_lane = 7
    x = nw.generate_bit_wave(src_lane)
    y = np.zeros(nw.FRAME_SAMPLES, dtype='f4')
    out = nw.nand_wave(x, y, mode='dominant', target_lane=target_lane, energy_thresh=0.01)
    env = nw.extract_lane(x, src_lane)
    expected = nw.replay_envelope(env, target_lane)
    corr = np.corrcoef(out, expected)[0,1]
    print(f"correlation {corr}")
    assert abs(corr - 1.0) <= 1e-2

    print("Testing dominant NOR generates fresh one.")
    target_lane = 1
    x = np.zeros(nw.FRAME_SAMPLES, dtype='f4')
    y = np.zeros_like(x)
    out = nw.nand_wave(x, y, mode='dominant', target_lane=target_lane, energy_thresh=0.01)
    expected = nw.generate_bit_wave(target_lane)
    corr = np.corrcoef(out, expected)[0,1]
    print(f"correlation {corr}")
    assert abs(corr - 1.0) <= 1e-2

    print("Testing dominant both-on gives silence.")
    lane = 6
    x = nw.generate_bit_wave(lane)
    y = nw.generate_bit_wave(lane)
    out = nw.nand_wave(x, y, mode='dominant', target_lane=lane, energy_thresh=0.01)
    max_amp = np.max(np.abs(out))
    print(f"max amplitude {max_amp}")
    assert max_amp < 1e-6


def test_turing_ssa():
    print("Testing tape and head views with patching.")
    tm = ProvenanceTM(width=4)
    tm.init_tape('0101')
    tm.init_head(1)
    tape0 = tm.tape_view(0)
    head0 = tm.head_view(0)
    print(f"initial tape {[v.value for v in tape0]}, head {head0.value}")
    old_id = tape0[1].node_id
    new_val = tape0[1].value ^ 1
    tm.patch_cell(0,1,new_val)
    tape1 = tm.tape_view(1)
    print(f"patched tape {[v.value for v in tape1]}")
    assert tape1[1].value == new_val and tape1[1].node_id != old_id
    assert tm.graph.nodes

    print("Testing init_tape from various sources.")
    tm = ProvenanceTM(width=8)
    tm.init_tape(0b1011)
    print(f"from int -> {[v.value for v in tm.tape_view(0)]}")
    tm.init_tape(b'\xf0')
    print(f"from bytes -> {[v.value for v in tm.tape_view(0)]}")
    tm.init_tape([1,0,1])
    print(f"from list -> {[v.value for v in tm.tape_view(0)][:3]}")
    assert [v.value for v in tm.tape_view(0)][:3] == [1,0,1]


def main():
    tests = [
        ("Analog Spec", test_analog_spec),
        ("Cassette Tape Backend", test_cassette_tape),
        ("BitOps Translator", test_bitops_translator),
        ("Header and Tape Layout", test_header_layout),
        ("SSA Loop Builder", test_loops),
        ("Motor Profile", test_motor_profile),
        ("NAND Wave", test_nand_wave),
        ("Turing SSA", test_turing_ssa),
    ]
    passed = 0
    for name, func in tests:
        if run_test(name, func):
            passed += 1
    print(f"Summary: {passed}/{len(tests)} test groups passed")


if __name__ == '__main__':
    main()

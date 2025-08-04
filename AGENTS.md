# Zero-Ambiguity Action Plan

*(Everything below is mandatory. Any omission, digital shortcut, or silent deviation is "total failure." Agent = the person or program that will edit the existing repo.)*

---

## 1  Global Parameters — set **once** before any run

* **LANES:** 32 (32 carrier bins)
* **TRACKS:** 2 (per tape or register; L = data+instr, R = data+bias)
* **REGISTERS:** 3 (default IDs = R0 R1 R2, each a 2-track tape-sim)
* **BIT_FRAME:** 500 ms (complete ADSR per bit)
* **FS:** 44 100 Hz
* **BASE_FREQ:** 110 Hz (lane 0)
* **SEMI_RATIO:** 2^(1/12) (lane i = BASE × ratio^i)
* **MOTOR_CARRIER:** 60 Hz
* **WRITE_BIAS:** 150 Hz
* **DATA_ADSR:** (50 A, 50 D, 0.8 S, 100 R ms)
* **MOTOR_ENV_UP/DN:** 250 ms ramp (each direction)

---

## 2  Tape Structure (immutable contract)

1. **BIOS header** (always at physical start, duplicated every *N* feet on loop media)

   * magic 8-byte ID
   * motor-calibration block (fast-wind time, read-speed time, drift)
   * **inputs** — parallel frames (all active lanes)
   * **outputs** — mirrors inputs, pre-filled with silence
   * start address of instruction table
2. **Instruction table** — sequence of 16-bit *machine words*; each word appears in **one parallel frame** across lanes 0-15 of Track-0.
3. **Data zones** — arbitrarily allocated bit-frame ranges, mutable at runtime.
4. **End-stop marker** — continuous silence then fixed “stop” tone on MOTOR lane; reaching it cuts motor gain to 0 until a new SEEK is commanded.

---

## 3  Instruction Word (16 bits, one frame, parallel-encoded)

```
bits 15-12 : OPCODE
bits 11-10 : REG-A   (00 R0  01 R1  10 R2  11 reserved)
bits  9- 8 : REG-B
bits  7- 6 : DEST
bits  5- 0 : PARAM  (length, shift-k, etc.)
```

**Opcode map (hex):**
0x0 SEEK  0x1 READ  0x2 WRITE  0x3 NAND  0x4 SIGL  0x5 SIGR  0x6 CONCAT  0x7 SLICE  0x8 MU  0x9 LENGTH  0xA ZEROS  0xF HALT

---

## 4  Register Behaviour

* Each register occupies two fixed tracks of main tape.
* **Track-0:** data/instruction carriers.
* **Track-1:** write-bias tone when writing; data otherwise.
* Operations sequence (always):
  1  Motor **SEEK** envelope to target address.
  2  Continuous **read** or **write** sweep over *n* bit-frames (no stop-start per bit).
  3  Optional rewind if media is end-to-end type.

Registers persist across instructions; nothing is auto-cleared. All register moves must be logged as PCM and decoded back to digital before use.

---

## 5  Lane & Frame Encoding Rules

* **Bit 1:** tone at lane-freq with DATA_ADSR in full 500 ms frame.
* **Bit 0:** absolute silence on that lane in same frame.
* **Parallel mode:** multiple lanes active simultaneously in one frame → whole word.
* **Serial mode:** one active lane per successive frame → bit stream.
* Frames align exactly to sample boundaries so FFT(N = FS·BIT_FRAME) yields perfect bin peaks.

---

## 6  Analog Implementation of **ALL** Primitive Operators

(No digital fallback permitted; stubbing with digital math = failure.)

| Operator          | Mandatory analog realisation                                                                                                                                    |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NAND**          | *Both operands use the same lane.* Sum amplitudes → 0, A, 2A. Threshold ≥1.5A inverts (output 0 only for 2A). Result tone written on result lane in next frame. |
| **σ_L(k)**       | Copy X frames to output, then write *k* silent frames.                                                                                                          |
| **σ_R(k)**       | Read X, drop last *k* frames, write remainder.                                                                                                                  |
| **CONCAT**        | Sequentially write X then Y frames to output address.                                                                                                           |
| **SLICE(i,j)**    | Seek to X+i, read (j-i) frames, write to output.                                                                                                                |
| **MU (selector)** | Selector lane amplitude gates VCA switching between X-lane and Y-lane; chosen lane passes to output frame.                                                      |
| **LENGTH**        | Mechanically time run from start to end-marker lane; encode elapsed frame-count as binary value in output frames.                                               |
| **ZEROS(n)**      | Write n silent frames with write-bias engaged.                                                                                                                  |

*If an operator cannot yet be modelled with full analog maths, agent must provide a placeholder **PCM waveform that evidences the intended amplitude-threshold or gating effect** and document exact missing physics for later refinement.*

---

## 7  Motor Control Simulation

* **Dedicated MOTOR lane** (Track-0) carries 1 kHz carrier; envelope amplitude = motor gain.
* Equation: gain(t) → integrate τ_motor → speed(t) → integrate → position(t).
* Calibration: run full-length wind & read-speed crawl at start-up; store times in BIOS calibration block.
* All SEEK envelopes must be trapezoidal (accel-coast-decel) derived from calibration constants.
* End-stop kills motor; restart requires fresh SEEK instruction.

---

## 8  Audio Event Intermediate Representation (**MIDI-centric**)

* Every action (data tone, motor envelope, bias tone, logic-result tone) = **MidiEvent**
  `(start_ms, duration_ms, channel=track#, note=MIDI_from_freq, velocity, ADSR, pan)`
* PCM buffers are rendered **only** from MidiEvents (sin-sum or external synth).
* FFT analysis of registers is performed **on PCM**; decoded magnitudes convert back to MidiEvents for downstream ops.
* Two deliverables per run:

  1. **Tape-state PCM** (raw carriers)
  2. **Execution PCM** (tape + motor + head sounds)

---

## 9  Execution Modes

1. **Logic-leading:** Turing graph triggers tape ops.
2. **Tape-leading:** Instruction table drives operations autonomously.
3. **Nested / Parallel:** Either simulator may wrap the other; hooks must allow one to pause while the other runs.

All modes must share the same primitive op definitions; only orchestration differs.

---

## 10  Header & Metadata (no JSON)

* Fixed-length binary struct only.
* Fields (order fixed):
  magic ID | calib_fast | calib_read | num_inputs | inputs… | num_outputs | outputs… | instr_start_addr | reserved.
* Encoded across **all tracks & lanes in parallel** for first few frames.
* Sanity check = Hamming-distance test on magic ID; failure aborts run.

---

## 11  Testing & Failure Criteria

* Every primitive op executed on test patterns must reproduce correct digital result **after round-trip PCM→FFT→bits**.
* Any zero-motion read/write, missing audible frame, or digital shortcut counts as **FAIL**.
* End-to-end test: multiply 5 × 3 via NAND-based adder; verify audio contains correct lane frames and output region decodes to “15”.
* All existing unit tests in repo must still pass; extend with new analog-logic tests.

---

## 12  Stub Policy

* If true analog modelling is temporarily infeasible, agent may provide **explicit placeholder waveform** that follows the amplitude/threshold rule and leaves TODO comment describing missing physics.
* Silent stubs or direct-bit digital ops are **not allowed**.

---

### **Deliver this plan to any agent altering the repo. Deviation = total failure.**


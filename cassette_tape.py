from __future__ import annotations

"""
CassetteTapeBackend – High-Fidelity Physical Simulation (v2)
============================================================

This version implements a simulation-grade physical model where audio is an
emergent property of the machine's state.

Upgrades:
- Dynamic Motor Sound: Pitch and timbre of the motor hum scale with
  instantaneous speed and direction.
- Continuous Instruction Track: Commands like 'seek' or 'read' are
  represented as a continuous carrier signal mixed with the motor's
  audio response for the full duration of the physical action. This
  simulates an instruction being actively fed to the motor controller.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import threading
import queue
import time

from analog_spec import generate_bit_wave, FRAME_SAMPLES
from tape_head import TapeHead

try:
    import numpy as np
    _Vec = np.ndarray
except ModuleNotFoundError: # pragma: no cover
    np = None
    _Vec = List[float]

try:
    import sounddevice as _sd
except Exception: # pragma: no cover
    _sd = None

@dataclass
class CassetteTapeBackend:
    # -------------------------- Physical Constants ----------------------------- #
    tape_length_inches: float = 3600.0 * 12 # 3600 feet
    bits_per_inch: int = 800

    read_write_speed_ips: float = 1.875
    seek_speed_ips: float = 30.0
    motor_acceleration_ips2: float = 100.0

    # -------------------------- Simulation & Audio ----------------------------- #
    sample_rate_hz: int = 44100
    time_scale_factor: float = 1.0
    
    attack_ms: float = 2.0
    decay_ms: float = 2.0
    sustain_level: float = 0.7
    release_ms: float = 5.0

    op_sine_coeffs: Dict[str, Dict[float, float]] = field(default_factory=dict)
    motor_speed_pitch_coeff: float = 0.05 # How much speed affects pitch
    motor_direction_coeff: float = 1.02 # Slight pitch change for reverse
    motor_friction_coeff: float = 0.02 # Fractional drag per second

    # If provided, override computed bits; otherwise derive from tape_length_inches
    tape_length: Optional[int] = None  # Total bits on tape (optional override)

    # -------------------------- Runtime State ---------------------------------- #
    _head_pos_inches: float = 0.0
    _tape_frames: Dict[Tuple[int, int, int], _Vec] = field(default_factory=dict)
    _gate: threading.Lock = field(default_factory=threading.Lock)

    _audio_cursor: int = 0
    _audio_buffer: _Vec | None = None
    _audio_queue: "queue.Queue[_Vec | None]" | None = None
    _audio_thread: threading.Thread | None = None

    def __post_init__(self):
        if np is None:
            raise ImportError("Numpy is required for the physical simulation.")
        self._attack_s = self.attack_ms / 1000.0 * self.time_scale_factor
        self._decay_s = self.decay_ms / 1000.0 * self.time_scale_factor
        self._release_s = self.release_ms / 1000.0 * self.time_scale_factor
        # If total bits override provided, infer missing physical properties
        if self.tape_length is not None:
            # default tape_length_inches constant
            default_inches = 3600.0 * 12
            if self.tape_length_inches == default_inches:
                # infer physical length from bits and density
                self.tape_length_inches = self.tape_length / self.bits_per_inch
            else:
                # infer density from bits and physical length
                self.bits_per_inch = int(round(self.tape_length / self.tape_length_inches))
        self._init_audio_system()
        self._head = TapeHead(self)

    @property
    def total_bits(self) -> int:
        """
        Total bits on tape. Use override if provided, otherwise compute from physical length.
        """
        if self.tape_length is not None:
            return int(self.tape_length)
        return int(self.tape_length_inches * self.bits_per_inch)

    def _init_audio_system(self):
        self._audio_buffer = np.zeros(self.sample_rate_hz * 2, dtype="f4")
        self._audio_queue = queue.Queue()
        self._audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self._audio_thread.start()

    def _ensure_audio_capacity(self, required_samples: int):
        if self._audio_cursor + required_samples > len(self._audio_buffer):
            self._audio_queue.put(self._audio_buffer[:self._audio_cursor].copy())
            self._audio_buffer.fill(0)
            self._audio_cursor = 0
            if required_samples > len(self._audio_buffer):
                 self._audio_buffer = np.zeros(required_samples * 2, dtype="f4")

    def move_head_to_bit(self, target_bit_idx: int):
        target_pos_inches = target_bit_idx / self.bits_per_inch
        distance_inches = target_pos_inches - self._head_pos_inches
        if abs(distance_inches) < 1e-6:
            return
        
        direction = 1 if distance_inches > 0 else -1
        self._simulate_movement(abs(distance_inches), self.seek_speed_ips, direction, 'seek')
        self._head_pos_inches = target_pos_inches

    # ---------------------------- Bit / Frame Access --------------------------- #

    def read_wave(self, track: int, lane: int, bit_idx: int) -> _Vec:
        """Return the PCM frame at ``(track, lane, bit_idx)`` after seek and scan."""
        with self._gate:
            self.move_head_to_bit(bit_idx)
            current_idx = int(round(self._head_pos_inches * self.bits_per_inch))
            if current_idx != bit_idx:
                raise RuntimeError("head misaligned for read")
            bit_width_inches = 1.0 / self.bits_per_inch
            self._head.enqueue_read(track, lane, bit_idx)
            self._simulate_movement(bit_width_inches, self.read_write_speed_ips, 1, 'read')
            frame = self._head.activate(track, 'read', self.read_write_speed_ips)
            self._head_pos_inches += bit_width_inches
            if frame is None:
                raise RuntimeError("read head not engaged at correct speed")
            return frame

    def write_wave(self, track: int, lane: int, bit_idx: int, frame: _Vec):
        """Write a PCM frame to ``(track, lane, bit_idx)`` respecting movement."""
        with self._gate:
            self.move_head_to_bit(bit_idx)
            current_idx = int(round(self._head_pos_inches * self.bits_per_inch))
            if current_idx != bit_idx:
                raise RuntimeError("head misaligned for write")
            bit_width_inches = 1.0 / self.bits_per_inch
            self._head.enqueue_write(track, lane, bit_idx, frame)
            self._simulate_movement(bit_width_inches, self.read_write_speed_ips, 1, 'write')
            self._head.activate(track, 'write', self.read_write_speed_ips)
            self._head_pos_inches += bit_width_inches

    # Digital convenience wrappers ------------------------------------------------ #

    def read_bit(self, track: int, lane: int, bit_idx: int) -> int:
        wave = self.read_wave(track, lane, bit_idx)
        return 1 if float(np.max(np.abs(wave))) > 0.5 else 0

    def write_bit(self, track: int, lane: int, bit_idx: int, value: int):
        frame = generate_bit_wave(1 if value else 0, lane)
        self.write_wave(track, lane, bit_idx, frame)

    # ------------------------------------------------------------------
    def _generate_speed_profile(self, distance: float, target_speed: float) -> _Vec:
        """Return an instantaneous speed profile for the given travel distance."""
        dt = 1.0 / self.sample_rate_hz
        accel = self.motor_acceleration_ips2
        speed = 0.0
        pos = 0.0
        speeds: List[float] = []
        while True:
            remaining = distance - pos
            braking = (speed ** 2) / (2 * accel)
            if remaining <= 0 and speed <= 0:
                break
            if remaining <= braking:
                speed = max(0.0, speed - accel * dt)
            else:
                speed = min(target_speed, speed + accel * dt)
            speed *= 1.0 - self.motor_friction_coeff * dt
            pos += speed * dt
            speeds.append(speed)
            if len(speeds) > 10_000_000:
                # Safety break in pathological cases
                break
        return np.array(speeds, dtype="f4")

    def _simulate_movement(self, distance_inches: float, target_speed_ips: float, direction: int, op_name: str):
        # Generate speed profile via physical integration
        speed_profile = self._generate_speed_profile(distance_inches, target_speed_ips)
        num_samples = len(speed_profile)
        if num_samples == 0:
            return
        self._ensure_audio_capacity(num_samples)

        t = np.arange(num_samples) / self.sample_rate_hz
        instruction_track = np.zeros(num_samples, dtype="f4")
        op_coeffs = self.op_sine_coeffs.get(op_name, {})
        for freq, amp in op_coeffs.items():
            instruction_track += amp * np.sin(2 * np.pi * freq * t)

        motor_hum = np.zeros(num_samples, dtype="f4")
        motor_base_coeffs = self.op_sine_coeffs.get('motor', {60.0: 0.5})
        for i in range(num_samples):
            speed = speed_profile[i]
            for base_freq, amp in motor_base_coeffs.items():
                dir_coeff = self.motor_direction_coeff if direction == -1 else 1.0
                freq = base_freq * dir_coeff * (1 + speed * self.motor_speed_pitch_coeff)
                amp_mod = amp * (0.5 + 0.5 * (speed / max(target_speed_ips, 1e-6)))
                motor_hum[i] += amp_mod * np.sin(2 * np.pi * freq * t[i])

        final_mix = instruction_track + motor_hum
        duration_sec = num_samples / self.sample_rate_hz
        master_env = self._get_adsr_envelope(duration_sec, num_samples)
        final_mix *= master_env

        self._audio_buffer[self._audio_cursor : self._audio_cursor + num_samples] += final_mix
        self._audio_cursor += num_samples
        time.sleep(duration_sec * self.time_scale_factor)

    def _get_adsr_envelope(self, duration_s: float, n_samples: int) -> _Vec:
        env = np.zeros(n_samples, dtype="f4")
        n_atk = min(n_samples, int(self._attack_s * self.sample_rate_hz))
        n_dec = min(n_samples - n_atk, int(self._decay_s * self.sample_rate_hz))
        n_rel = min(n_samples - n_atk - n_dec, int(self._release_s * self.sample_rate_hz))
        n_sus = max(0, n_samples - n_atk - n_dec - n_rel)
        if n_atk > 0: env[:n_atk] = np.linspace(0, 1, n_atk)
        if n_dec > 0: env[n_atk:n_atk+n_dec] = np.linspace(1, self.sustain_level, n_dec)
        if n_sus > 0: env[n_atk+n_dec : n_atk+n_dec+n_sus] = self.sustain_level
        if n_rel > 0: env[-n_rel:] = np.linspace(self.sustain_level, 0, n_rel)
        return env

    def close(self):
        if self._audio_cursor > 0:
            self._audio_queue.put(self._audio_buffer[:self._audio_cursor].copy())
        self._audio_queue.put(None)
        if self._audio_thread is not None:
            self.op_sine_coeffs = {}
            self._audio_thread.join(timeout=2.0)

    def _audio_worker(self):
        while True:
            chunk = self._audio_queue.get()
            if chunk is None: break
            if _sd is not None:
                try: _sd.play(chunk, self.sample_rate_hz, blocking=True)
                except Exception as e: print(f"Audio playback error: {e}")

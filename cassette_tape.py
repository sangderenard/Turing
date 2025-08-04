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
from typing import Dict, List, Tuple
import threading
import queue
import time

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

    # -------------------------- Runtime State ---------------------------------- #
    _head_pos_inches: float = 0.0
    _tape_data: Dict[int, int] = field(default_factory=dict)

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
        self._init_audio_system()

    @property
    def total_bits(self) -> int:
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

    def read_bit(self, bit_idx: int) -> int:
        self.move_head_to_bit(bit_idx)
        bit_width_inches = 1.0 / self.bits_per_inch
        self._simulate_movement(bit_width_inches, self.read_write_speed_ips, 1, 'read')
        self._head_pos_inches += bit_width_inches
        return self._tape_data.get(bit_idx, 0)

    def write_bit(self, bit_idx: int, value: int):
        self.move_head_to_bit(bit_idx)
        bit_width_inches = 1.0 / self.bits_per_inch
        self._simulate_movement(bit_width_inches, self.read_write_speed_ips, 1, 'write')
        self._head_pos_inches += bit_width_inches
        self._tape_data[bit_idx] = 1 if value else 0
        
    def _simulate_movement(self, distance_inches: float, target_speed_ips: float, direction: int, op_name: str):
        # --- 1. Calculate Time and Speed Profile ---
        time_to_accel = target_speed_ips / self.motor_acceleration_ips2
        dist_accel = 0.5 * self.motor_acceleration_ips2 * (time_to_accel ** 2)

        if 2 * dist_accel >= distance_inches:
            time_to_peak = (distance_inches / self.motor_acceleration_ips2)**0.5
            total_time_sec = 2 * time_to_peak
            time_coast = 0.0
        else:
            dist_coast = distance_inches - 2 * dist_accel
            time_coast = dist_coast / target_speed_ips
            total_time_sec = 2 * time_to_accel + time_coast

        scaled_time_sec = total_time_sec * self.time_scale_factor
        num_samples = int(scaled_time_sec * self.sample_rate_hz)
        if num_samples == 0: return
        self._ensure_audio_capacity(num_samples)
        
        # --- 2. Generate Continuous Instruction Track ---
        t = np.arange(num_samples) / self.sample_rate_hz
        instruction_track = np.zeros(num_samples, dtype="f4")
        op_coeffs = self.op_sine_coeffs.get(op_name, {})
        for freq, amp in op_coeffs.items():
            instruction_track += amp * np.sin(2 * np.pi * freq * t)

        # --- 3. Generate Dynamic Motor Hum based on Speed Profile ---
        motor_hum = np.zeros(num_samples, dtype="f4")
        motor_base_coeffs = self.op_sine_coeffs.get('motor', {60.0: 0.5})
        
        # Create a vector of instantaneous speed over the movement duration
        speed_profile = np.zeros(num_samples, dtype="f4")
        n_accel = int(time_to_accel * self.time_scale_factor * self.sample_rate_hz)
        n_coast = int(time_coast * self.time_scale_factor * self.sample_rate_hz)
        if n_accel > 0:
            accel_speeds = np.linspace(0, target_speed_ips, n_accel)
            speed_profile[:n_accel] = accel_speeds
            speed_profile[num_samples-n_accel:] = accel_speeds[::-1]
        if n_coast > 0:
            speed_profile[n_accel : n_accel + n_coast] = target_speed_ips

        # Generate audio sample by sample based on speed
        for i in range(num_samples):
            speed = speed_profile[i]
            for base_freq, amp in motor_base_coeffs.items():
                dir_coeff = self.motor_direction_coeff if direction == -1 else 1.0
                freq = base_freq * dir_coeff * (1 + speed * self.motor_speed_pitch_coeff)
                motor_hum[i] += amp * np.sin(2 * np.pi * freq * t[i])

        # --- 4. Mix and Store ---
        final_mix = instruction_track + motor_hum
        
        # Apply a master ADSR envelope to the entire operation
        master_env = self._get_adsr_envelope(scaled_time_sec, num_samples)
        final_mix *= master_env

        self._audio_buffer[self._audio_cursor : self._audio_cursor + num_samples] += final_mix
        self._audio_cursor += num_samples

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

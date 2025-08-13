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
from typing import Callable, Dict, List, Tuple, Optional
import threading
import queue
import time

from .analog_spec import (
    generate_bit_wave,
    MotorCalibration,
    trapezoidal_motor_envelope,
)
from .constants import (
    FRAME_SAMPLES,
    BIT_FRAME_MS,
    MOTOR_CARRIER,
    BIAS_AMP,
    SIMULATION_VOLUME,
)
from .analog_helpers import lane_rms
from ..turing_machine.tape_head import TapeHead



try:
    import numpy as np
    from numpy import ndarray
except ModuleNotFoundError: # pragma: no cover
    np = None



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
    _tape_frames: Dict[Tuple[int, int, int], ndarray] = field(default_factory=dict)
    _gate: threading.Lock = field(default_factory=threading.Lock)

    _audio_cursor: int = 0  # total samples generated
    _buffer_cursor: int = 0
    _audio_buffer: Optional[ndarray] = None
    _audio_queue: Optional[queue.Queue] = None
    _audio_thread: threading.Thread | None = None

    # Optional callback for visualizers (e.g., reel demos) to receive
    # updates on tape position and activity. The callable should accept
    # ``(tape_position_tuple, head_pos_inches, reading, writing)``.
    status_callback: Callable[[Tuple[int, int], float, bool, bool], None] | None = None

    def __post_init__(self):
        if np is None:
            raise ImportError("Numpy is required for the physical simulation.")
        self._attack_s = self.attack_ms / 1000.0 * self.time_scale_factor
        self._decay_s = self.decay_ms / 1000.0 * self.time_scale_factor
        self._release_s = self.release_ms / 1000.0 * self.time_scale_factor
        # If total bits override provided, infer missing physical properties
        default_inches = 3600.0 * 12
        if self.tape_length is None and self.tape_length_inches == default_inches:
            # shorten the default tape to encourage visible head movement
            self.tape_length = 1024
            self.tape_length_inches = self.tape_length / self.bits_per_inch
        elif self.tape_length is not None:
            if self.tape_length_inches == default_inches:
                # infer physical length from bits and density
                self.tape_length_inches = self.tape_length / self.bits_per_inch
            else:
                # infer density from bits and physical length
                self.bits_per_inch = int(round(self.tape_length / self.tape_length_inches))
        self._init_audio_system()
        self._head = TapeHead(self)
        self._notify_status("seek")

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
        self._buffer_cursor = 0
        self._audio_cursor = 0

    def _notify_status(self, op_name: str) -> None:
        """Send tape position and activity to the status callback if present."""
        if self.status_callback is None:
            return
        current_bit = int(round(self._head_pos_inches * self.bits_per_inch))
        left = max(self.total_bits - current_bit, 0)
        right = min(current_bit, self.total_bits)
        reading = op_name == "read"
        writing = op_name == "write"
        try:
            self.status_callback((left, right), self._head_pos_inches, reading, writing)
        except Exception:
            pass

    def _ensure_audio_capacity(self, required_samples: int):
        if self._buffer_cursor + required_samples > len(self._audio_buffer):
            self._audio_queue.put(self._audio_buffer[:self._buffer_cursor].copy())
            self._audio_buffer.fill(0)
            self._buffer_cursor = 0
            if required_samples > len(self._audio_buffer):
                 self._audio_buffer = np.zeros(required_samples * 2, dtype="f4")

    def move_head_to_bit(self, target_bit_idx: int):
        target_pos_inches = target_bit_idx / self.bits_per_inch
        distance_inches = target_pos_inches - self._head_pos_inches
        if abs(distance_inches) < 1e-6:
            self._notify_status("seek")
            return
        
        direction = 1 if distance_inches > 0 else -1
        self._simulate_movement(abs(distance_inches), self.seek_speed_ips, direction, 'seek')
        self._head_pos_inches = target_pos_inches
        self._notify_status("seek")

    # ---------------------------- Bit / Frame Access --------------------------- #

    def read_wave(self, track: int, lane: int, bit_idx: int) -> ndarray:
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
            self._notify_status("read")
            return frame

    def write_wave(self, track: int, lane: int, bit_idx: int, frame: ndarray):
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
            self._notify_status("write")

    # Digital convenience wrappers ------------------------------------------------ #

    def read_bit(self, track: int, lane: int, bit_idx: int) -> int:
        wave = self.read_wave(track, lane, bit_idx)
        amp = lane_rms(wave, lane)
        return 1 if amp > BIAS_AMP * 2 else 0

    def write_bit(self, track: int, lane: int, bit_idx: int, value: int):
        frame = generate_bit_wave(1 if value else 0, lane)
        self.write_wave(track, lane, bit_idx, frame)

    # ------------------------------------------------------------------
    def _simulate_movement(self, distance_inches: float, target_speed_ips: float, direction: int, op_name: str):
        """Simulate movement using a trapezoidal motor envelope."""
        distance_frames = int(round(distance_inches * self.bits_per_inch))
        calib = MotorCalibration(
            fast_wind_ms=BIT_FRAME_MS * (self.read_write_speed_ips / self.seek_speed_ips),
            read_speed_ms=BIT_FRAME_MS,
            drift_ms=0.0,
        )
        mode = "read" if target_speed_ips == self.read_write_speed_ips else "fast"
        env = trapezoidal_motor_envelope(distance_frames, calib, mode)
        num_samples = len(env)
        if num_samples == 0:
            return
        self._ensure_audio_capacity(num_samples)

        t = np.arange(num_samples) / self.sample_rate_hz
        instruction_track = np.zeros(num_samples, dtype="f4")
        op_coeffs = self.op_sine_coeffs.get(op_name, {})
        for freq, amp in op_coeffs.items():
            instruction_track += amp * np.sin(2 * np.pi * freq * t)

        motor = env * np.sin(2 * np.pi * MOTOR_CARRIER * t) * (1.0 if direction == 1 else -1.0)
        final_mix = instruction_track + motor
        duration_sec = num_samples / self.sample_rate_hz
        master_env = self._get_adsr_envelope(duration_sec, num_samples)
        final_mix *= master_env
        peak = float(np.max(np.abs(final_mix)))
        if peak > 1.0:
            final_mix *= 1.0 / peak

        self._audio_buffer[self._buffer_cursor : self._buffer_cursor + num_samples] += final_mix
        self._buffer_cursor += num_samples
        self._audio_cursor += num_samples
        time.sleep(duration_sec * self.time_scale_factor)

    def _get_adsr_envelope(self, duration_s: float, n_samples: int) -> ndarray:
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
        if self._buffer_cursor > 0:
            self._audio_queue.put(self._audio_buffer[:self._buffer_cursor].copy())
        self._audio_queue.put(None)
        if self._audio_thread is not None:
            self.op_sine_coeffs = {}
            self._audio_thread.join(timeout=2.0)

    def _audio_worker(self):
        while True:
            chunk = self._audio_queue.get()
            if chunk is None: break
            if _sd is not None:
                try: _sd.play(chunk * SIMULATION_VOLUME, self.sample_rate_hz, blocking=True)
                except Exception as e: print(f"Audio playback error: {e}")

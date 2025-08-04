from __future__ import annotations
"""Cassette-style analogue backend (with ADSR envelope) for Transmogrifier Turing stack.

Key upgrades vs. first draft
---------------------------
* **ADSR envelopes** – Each logical action (read‑enable / write‑enable) now
  drives the amplifier gain pin with an Attack–Decay–Sustain–Release envelope
  derived from first principles of RC‑charging/discharging (linear slope
  approximation is default; exponential optional).
* **Envelope template** is generated once per instance and stamped into the
  analogue tracks at every event, replacing the previous flat‑top pulses.
* **Public API is unchanged** (`read_bit`, `write_bit`, `move_head`,
  `export_audio`), so *Turing* classes can swap back‑ends without code
  changes.  Additional envelope parameters are exposed as constructor
  defaults (attack_ms, decay_ms, sustain_level, release_ms, shape='linear').

⚠️  Analogue mode can now consume more than one logical frame worth of audio
per event if the sum of envelope segments exceeds *frame_duration_ms*.

"""

from dataclasses import dataclass
from typing import List, Protocol, Tuple

try:
    import numpy as np

    _Vector = np.ndarray
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# Minimal protocol Turing expects
# ---------------------------------------------------------------------------
class TapeHookProtocol(Protocol):
    def read_bit(self, idx: int) -> int: ...
    def write_bit(self, idx: int, bit: int) -> None: ...
    def move_head(self, delta: int) -> None: ...
    @property
    def head_pos(self) -> int: ...


@dataclass
class CassetteTapeBackend(TapeHookProtocol):
    """Fixed‑length virtual tape with optional analogue signal logging.

    When *analogue_mode* is True every logical operation is emitted to three
    tracks:
        motor_voltage      – capstan drive (V)
        read_head_enable   – ADSR‑shaped gate (V)
        write_head_enable  – ADSR‑shaped gate (V)

    Envelope parameters control the shape in milliseconds; the template is
    resampled at *sample_rate_hz*.
    """

    # --- configuration ---------------------------------------------------- #
    tape_length: int = 4096
    sample_rate_hz: int = 44_100
    frame_duration_ms: float = 5.0   # 5 ms logical step ≈ 220 samples @ 44.1 kHz

    # Motor & head voltages
    motor_idle_voltage: float = 2.0   # V (halt)
    motor_run_voltage: float = 5.0    # V (play fwd)
    head_active_voltage: float = 5.0  # V (gate on)

    # ADSR parameters (ms / scalar)
    attack_ms: float = 0.4
    decay_ms: float = 0.4
    sustain_level: float = 0.8        # proportion of peak (0‑1)
    release_ms: float = 0.4
    envelope_shape: str = "linear"    # or "exp"

    analogue_mode: bool = False  # set True to log audio

    # --- runtime state ---------------------------------------------------- #
    _tape: List[int] | _Vector = None  # type: ignore
    _head: int = 0

    _motor_signal: List[float] | _Vector = None  # type: ignore
    _read_en_signal: List[float] | _Vector = None  # type: ignore
    _write_en_signal: List[float] | _Vector = None  # type: ignore

    _cursor: int = 0   # write pointer into signal tracks (samples)
    _env: _Vector | None = None  # ADSR template scaled to head_active_voltage

    # --------------------------------------------------------------------- #
    def __post_init__(self) -> None:
        self._init_storage()
        self._build_envelope_template()

    # Storage for tape + tracks ------------------------------------------- #
    def _init_storage(self):
        if np is None:  # pure‑python fallback, minimal
            self._tape = [0] * self.tape_length
            self._motor_signal, self._read_en_signal, self._write_en_signal = ([], [], [])
            return
        self._tape = np.zeros(self.tape_length, dtype="i1")
        if self.analogue_mode:
            # Pre‑allocate 10 k logical steps worth of samples; auto‑grow.
            cap = int(10_000 * self.frame_samples)
            self._motor_signal = np.full(cap, self.motor_idle_voltage, dtype="f4")
            self._read_en_signal = np.zeros(cap, dtype="f4")
            self._write_en_signal = np.zeros(cap, dtype="f4")
        else:
            self._motor_signal, self._read_en_signal, self._write_en_signal = (
                [],
                [],
                [],
            )

    # Envelope ------------------------------------------------------------- #
    def _build_envelope_template(self):
        if not self.analogue_mode or np is None:
            return
        sr = self.sample_rate_hz
        atk = int(self.attack_ms * sr / 1_000)
        dec = int(self.decay_ms * sr / 1_000)
        rel = int(self.release_ms * sr / 1_000)
        sus = max(0, self.frame_samples - (atk + dec + rel))
        if self.envelope_shape == "exp":
            def _seg(n0, n1, start, end):
                return start * (end / start) ** (np.linspace(0, 1, n1 - n0, False))
        else:  # linear
            def _seg(n0, n1, start, end):
                return np.linspace(start, end, n1 - n0, False)
        env = np.concatenate(
            [
                _seg(0, atk, 0.0, 1.0),
                _seg(0, dec, 1.0, self.sustain_level),
                np.full(sus, self.sustain_level),
                _seg(0, rel, self.sustain_level, 0.0),
            ]
        )
        if env.size < self.frame_samples:  # pad
            env = np.pad(env, (0, self.frame_samples - env.size))
        self._env = env.astype("f4") * self.head_active_voltage

    # Convenience ---------------------------------------------------------- #
    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate_hz * self.frame_duration_ms / 1_000)

    # ------------------------------------------------------------------ #
    #                Public API                                         #
    # ------------------------------------------------------------------ #
    def read_bit(self, idx: int) -> int:
        self._ensure_idx(idx)
        if self.analogue_mode:
            self._emit_signals(read=True)
        return int(self._tape[idx])

    def write_bit(self, idx: int, bit: int) -> None:
        self._ensure_idx(idx)
        if self.analogue_mode:
            self._emit_signals(write=True)
        self._tape[idx] = 1 if bit else 0

    def move_head(self, delta: int) -> None:
        # Clamp inside tape
        new = max(0, min(self.tape_length - 1, self._head + delta))
        self._head = new
        if self.analogue_mode:
            mv = self.motor_run_voltage if delta >= 0 else 0.0
            self._emit_signals(motor_voltage=mv)

    # ------------------------------------------------------------------ #
    @property
    def head_pos(self) -> int:
        return self._head

    # ------------------------------------------------------------------ #
    #                Audio export                                     #
    # ------------------------------------------------------------------ #
    def export_audio(self) -> Tuple[_Vector, _Vector, _Vector]:
        if not self.analogue_mode or np is None:
            raise RuntimeError("Analogue mode disabled or NumPy missing")
        used = self._cursor
        return (
            self._motor_signal[:used].copy(),
            self._read_en_signal[:used].copy(),
            self._write_en_signal[:used].copy(),
        )

    # ------------------------------------------------------------------ #
    #                 Internal helpers                                  #
    # ------------------------------------------------------------------ #
    def _emit_signals(self, *, read=False, write=False, motor_voltage=None):
        if np is None:
            # Fallback – store scalars per logical step
            self._motor_signal.append(motor_voltage or self.motor_idle_voltage)
            self._read_en_signal.append(self.head_active_voltage if read else 0.0)
            self._write_en_signal.append(self.head_active_voltage if write else 0.0)
            self._cursor += 1
            return
        samples = self._env if (read or write) else np.full(self.frame_samples, 0.0, dtype="f4")
        end = self._cursor + self.frame_samples
        if end > self._motor_signal.shape[0]:
            self._grow_buffers(end * 2)
        # Motor track: constant voltage per frame
        self._motor_signal[self._cursor:end] = (
            motor_voltage if motor_voltage is not None else self.motor_idle_voltage
        )
        # Gate tracks: place envelope if requested
        if read:
            self._read_en_signal[self._cursor:end] = samples
        if write:
            self._write_en_signal[self._cursor:end] = samples
        self._cursor = end

    def _grow_buffers(self, new):
        self._motor_signal = np.resize(self._motor_signal, new)
        self._read_en_signal = np.resize(self._read_en_signal, new)
        self._write_en_signal = np.resize(self._write_en_signal, new)

    def _ensure_idx(self, idx: int):
        if idx < 0 or idx >= self.tape_length:
            raise IndexError(idx)

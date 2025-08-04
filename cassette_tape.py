from __future__ import annotations

"""
CassetteTapeBackend – Multilane Audio‑IR Edition
================================================

This backend turns every logical event (read / write gate, motor step) into a
**multidimensional audio IR** suitable for driving hardware that splits the
spectrum into parallel lanes.  The core signal tensor has shape::

    (n_lanes, n_samples)

* **lane 0** is always present and contains a single‑band amplitude gate
  (works on any deck with no splitter hardware).
* Additional lanes are optional; each lane has its **own carrier spectrum**
  (defined in FFT bin‑gain space) and an optional parametric‑EQ metadata
  dictionary (center frequency, gain, Q).  This metadata is *not* rendered to
  waveform—it is stored alongside the IR so that downstream encoders or
  hardware description layers can program DSP blocks directly.

No MP3/WAV writing is performed here—the IR remains entirely in numpy space
so a media‑encoder board can consume it without ever materialising PCM.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Tuple
import threading
import queue

try:  # optional real-time playback
    import sounddevice as _sd  # pragma: no cover - optional dependency
except Exception:  # pragma: no cover - missing audio library
    _sd = None

try:
    import numpy as np

    _Vec = np.ndarray
except ModuleNotFoundError:  # pragma: no cover – pure‑python fallback
    np = None  # type: ignore
    _Vec = List[float]  # type: ignore


class TapeHook(Protocol):
    def read_bit(self, idx: int) -> int: ...

    def write_bit(self, idx: int, bit: int) -> None: ...

    def move_head(self, delta: int) -> None: ...

    @property
    def head_pos(self) -> int: ...


@dataclass
class CassetteTapeBackend(TapeHook):
    # -------------------------- user‑config ----------------------------- #
    tape_length: int = 4096
    sample_rate_hz: int = 44_100
    frame_ms: float = 5.0

    motor_idle_v: float = 2.0
    motor_run_v: float = 5.0

    # ADSR envelope parameters
    attack_ms: float = 0.4
    decay_ms: float = 0.4
    sustain_level: float = 0.8
    release_ms: float = 0.4
    env_shape: str = "linear"  # or "exp"

    # Lane definitions -------------------------------------------------- #
    lane_band_gains: Dict[int, Dict[int, float]] = field(
        default_factory=lambda: {0: {1: 1.0}}
    )
    lane_eq_params: Dict[int, Dict[str, float]] = field(default_factory=dict)
    # e.g. {1: {"fo": 1000.0, "gain_db": 6.0, "Q": 1.0}}

    analogue_mode: bool = False

    # Sine coefficient tables per operation (freq_hz -> amplitude)
    op_sine_coeffs: Dict[str, Dict[float, float]] = field(
        default_factory=lambda: {
            "read": {440.0: 1.0},
            "write": {880.0: 1.0},
            "motor": {220.0: 0.5},
        }
    )

    # Frequency bins committed to the data and instruction buses
    data_bus_bins: List[int] = field(default_factory=list)
    instr_bus_bins: List[int] = field(default_factory=list)

    # -------------------------- runtime -------------------------------- #
    _tape: _Vec | List[int] = None  # type: ignore
    _head: int = 0
    _cursor: int = 0

    _lanes: _Vec | List[List[float]] = None  # shape (n_lanes, n_samples)
    _motor: _Vec | List[float] = None

    _env: _Vec | None = None  # ADSR vector * 1.0 (unit)
    _carriers: Dict[int, _Vec] = field(default_factory=dict)  # lane→carrier
    _data_bus_wave: _Vec | None = None
    _instr_bus_wave: _Vec | None = None

    _audio_queue: "queue.Queue[_Vec | None]" | None = None
    _audio_thread: threading.Thread | None = None
    _audio_frames: List[_Vec] | None = None

    # ------------------------------------------------------------------ #
    def __post_init__(self):
        self._init_tape()
        if self.analogue_mode and np is not None:
            self._ensure_capacity(int(self.frame_samples * 10_000))
            self._build_env()
            for lane in self.lane_band_gains:
                self._build_carrier(lane)
            self._build_bus_wave("data")
            self._build_bus_wave("instr")
            self._audio_frames = []
            self._audio_queue = queue.Queue()
            self._audio_thread = threading.Thread(
                target=self._audio_worker, daemon=True
            )
            self._audio_thread.start()

    # -------------------------- properties ----------------------------- #
    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate_hz * self.frame_ms / 1_000)

    @property
    def head_pos(self) -> int:
        return self._head

    # -------------------------- hook API ------------------------------- #
    def read_bit(self, idx: int) -> int:
        self._check_idx(idx)
        if self.analogue_mode:
            self._emit(read=True)
        return int(self._tape[idx])

    def write_bit(self, idx: int, bit: int) -> None:
        self._check_idx(idx)
        if self.analogue_mode:
            self._emit(write=True)
        self._tape[idx] = 1 if bit else 0

    def move_head(self, delta: int) -> None:
        self._head = max(0, min(self.tape_length - 1, self._head + delta))
        if self.analogue_mode:
            mv = self.motor_run_v if delta >= 0 else 0.0
            self._emit(motor_v=mv)

    # -------------------------- export -------------------------------- #
    def export_ir(self) -> Tuple[_Vec, _Vec, Dict[int, Dict[str, float]]]:
        """Return (lanes, motor, lane_eq_params).  `lanes` shape::

            (n_lanes, n_samples_used)
        """
        if not self.analogue_mode or np is None:
            raise RuntimeError("Analogue mode disabled or numpy missing")
        used = self._cursor
        return (
            self._lanes[:, :used].copy(),
            self._motor[:used].copy(),
            self.lane_eq_params,
        )

    # -------------------------- init helpers --------------------------- #
    def _init_tape(self):
        if np is None:
            self._tape = [0] * self.tape_length
            self._motor, self._lanes = ([], [])
            return
        self._tape = np.zeros(self.tape_length, dtype="i1")
        n_lanes = max(self.lane_band_gains.keys()) + 1
        self._lanes = np.zeros((n_lanes, 1), dtype="f4")
        self._motor = np.full(1, self.motor_idle_v, dtype="f4")

    def _ensure_capacity(self, size: int):
        if np is None:
            return
        if size <= self._motor.shape[0]:
            return
        self._motor = np.resize(self._motor, size)
        self._lanes = np.resize(self._lanes, (self._lanes.shape[0], size))

    # -------------------------- envelope ------------------------------ #
    def _build_env(self):
        if np is None:
            return
        sr = self.sample_rate_hz
        atk = int(self.attack_ms * sr / 1_000)
        dec = int(self.decay_ms * sr / 1_000)
        rel = int(self.release_ms * sr / 1_000)
        sus = max(0, self.frame_samples - (atk + dec + rel))
        def seg(n0, n1, a, b):
            if self.env_shape == "exp":
                return a * ((b / a) ** np.linspace(0, 1, n1 - n0, False))
            return np.linspace(a, b, n1 - n0, False)
        env = np.concatenate([
            seg(0, atk, 0.0, 1.0),
            seg(0, dec, 1.0, self.sustain_level),
            np.full(sus, self.sustain_level, dtype="f4"),
            seg(0, rel, self.sustain_level, 0.0),
        ])
        if env.size < self.frame_samples:
            env = np.pad(env, (0, self.frame_samples - env.size))
        self._env = env.astype("f4")

    # -------------------------- carrier per lane ---------------------- #
    def _build_carrier(self, lane: int):
        if np is None:
            return
        gains = self.lane_band_gains.get(lane, {1: 1.0})
        n = self.frame_samples
        spec = np.zeros(n, dtype=np.complex64)
        for k, amp in gains.items():
            spec[k % n] = amp
            if k != 0:
                spec[-k % n] = amp.conjugate()
        car = np.fft.ifft(spec).real
        peak = np.max(np.abs(car)) or 1.0
        self._carriers[lane] = (car / peak).astype("f4")

    # -------------------------- emit packet --------------------------- #
    def _emit(self, *, read=False, write=False, motor_v=None):
        if np is None:
            # scalar fallback: only lane0 amplitude log
            if len(self._lanes) == 0:
                self._lanes.append([])  # type: ignore
            self._lanes[0].append(float(read or write))  # type: ignore
            self._motor.append(motor_v or self.motor_idle_v)  # type: ignore
            self._cursor += 1
            return
        end = self._cursor + self.frame_samples
        self._ensure_capacity(end * 2)
        # motor track
        self._motor[self._cursor:end] = motor_v if motor_v is not None else self.motor_idle_v
        op = "read" if read else "write" if write else "motor"
        op_wave = self._generate_op_wave(op)
        bus_wave = 1.0
        if read or write:
            if self._data_bus_wave is not None:
                bus_wave = self._data_bus_wave
            for lane, car in self._carriers.items():
                pkt = self._env * car * op_wave * bus_wave
                self._lanes[lane, self._cursor:end] = pkt
        audio_mix = self._env * op_wave * bus_wave
        if self._audio_queue is not None:
            self._audio_queue.put(audio_mix.copy())
        self._cursor = end

    # Instruction bus emit -------------------------------------------------
    def execute_instruction(self):
        if self.analogue_mode:
            self._emit_instr()

    def _emit_instr(self):
        if np is None:
            return
        end = self._cursor + self.frame_samples
        self._ensure_capacity(end * 2)
        self._motor[self._cursor:end] = self.motor_idle_v
        op_wave = self._generate_op_wave("instr")
        bus_wave = self._instr_bus_wave if self._instr_bus_wave is not None else 1.0
        audio_mix = self._env * op_wave * bus_wave
        for lane, car in self._carriers.items():
            pkt = self._env * car * op_wave * bus_wave
            self._lanes[lane, self._cursor:end] = pkt
        if self._audio_queue is not None:
            self._audio_queue.put(audio_mix.copy())
        self._cursor = end

    # -------------------------- helpers ------------------------------- #
    def _check_idx(self, idx: int):
        if idx < 0 or idx >= self.tape_length:
            raise IndexError(idx)

    # dynamic reconfigure --------------------------------------------------
    def set_lane_band_gain(self, lane: int, bin_idx: int, gain: float):
        self.lane_band_gains.setdefault(lane, {})[bin_idx] = gain
        if self.analogue_mode and np is not None:
            self._build_carrier(lane)

    def set_lane_eq(self, lane: int, eq_params: Dict[str, float]):
        self.lane_eq_params[lane] = eq_params

    def configure_bus_width(self, data_bins: List[int], instr_bins: List[int]):
        self.data_bus_bins = list(data_bins)
        self.instr_bus_bins = list(instr_bins)
        if self.analogue_mode and np is not None:
            self._build_bus_wave("data")
            self._build_bus_wave("instr")

    # -------------------------- op waveform --------------------------- #
    def _generate_op_wave(self, op: str) -> _Vec:
        if np is None:
            return []  # pragma: no cover - pure python fallback
        coeffs = self.op_sine_coeffs.get(op, {})
        t = np.arange(self.frame_samples, dtype="f4") / self.sample_rate_hz
        wave = np.zeros(self.frame_samples, dtype="f4")
        for freq, amp in coeffs.items():
            wave += amp * np.sin(2 * np.pi * freq * t)
        peak = np.max(np.abs(wave)) or 1.0
        return (wave / peak).astype("f4")

    # -------------------------- bus wave ------------------------------- #
    def _build_bus_wave(self, which: str):
        if np is None:
            return
        bins = self.data_bus_bins if which == "data" else self.instr_bus_bins
        if not bins:
            if which == "data":
                self._data_bus_wave = None
            else:
                self._instr_bus_wave = None
            return
        n = self.frame_samples
        spec = np.zeros(n, dtype=np.complex64)
        for k in bins:
            spec[k % n] = 1.0
            if k != 0:
                spec[-k % n] = 1.0
        wave = np.fft.ifft(spec).real
        peak = np.max(np.abs(wave)) or 1.0
        if which == "data":
            self._data_bus_wave = (wave / peak).astype("f4")
        else:
            self._instr_bus_wave = (wave / peak).astype("f4")

    # -------------------------- audio thread ------------------------- #
    def _audio_worker(self):  # pragma: no cover - realtime side effect
        if np is None:
            return
        while True:
            frame = self._audio_queue.get()
            if frame is None:
                break
            if self._audio_frames is not None:
                self._audio_frames.append(frame)
            if _sd is not None:
                try:
                    _sd.play(frame, self.sample_rate_hz, blocking=True)
                except Exception:
                    pass

    def close(self):
        if self._audio_queue is not None:
            self._audio_queue.put(None)
        if self._audio_thread is not None:
            self._audio_thread.join(timeout=1.0)

    def __del__(self):  # pragma: no cover - best effort cleanup
        self.close()

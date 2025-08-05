"""Analog helper functions for per-lane FFT operations."""
import numpy as np
from analog_spec import FS, FRAME_SAMPLES, lane_frequency, LANES, DATA_ADSR

# Precompute FFT bin map for lanes
FREQS = np.fft.rfftfreq(FRAME_SAMPLES, 1 / FS)
BIN_IDX = {ln: int(np.argmin(np.abs(FREQS - lane_frequency(ln)))) for ln in range(LANES)}


def _lane_bin(lane: int) -> int:
    """Return the FFT bin index corresponding to the lane frequency."""
    freqs = np.fft.rfftfreq(FRAME_SAMPLES, 1 / FS)
    target = lane_frequency(lane)
    return int(np.argmin(np.abs(freqs - target)))
 
def lane_band(lane: int, half_bw: int = 3) -> slice:
    """Return FFT-bin slice around the lane's centre bin."""
    c = BIN_IDX[lane]
    return slice(max(c - half_bw, 0), c + half_bw + 1)


def extract_lane(frame: np.ndarray, lane: int) -> np.ndarray:
    """Extract the PCM waveform of a single lane component from a frame."""
    fft = np.fft.rfft(frame)
    bin_idx = _lane_bin(lane)
    single = np.zeros_like(fft)
    single[bin_idx] = fft[bin_idx]
    wave = np.fft.irfft(single)
    return wave.astype(frame.dtype)
 
def lane_rms(wave: np.ndarray, lane: int, half_bw: int = 3) -> float:
    """RMS energy of ``wave`` on ``lane``."""
    spec = np.fft.rfft(wave.astype("f4"))
    band = lane_band(lane, half_bw)
    return float(np.sqrt(np.mean(np.abs(spec[band]) ** 2)))

def track_rms(wave: np.ndarray) -> float:
    """Whole-track RMS energy."""
    return float(np.sqrt(np.mean(wave.astype("f4") ** 2)))


def mix_fft_lane(orig: np.ndarray, new: np.ndarray, lane: int) -> np.ndarray:
    """Mix a single lane component from 'new' into 'orig' via FFT bin replacement."""
    orig_fft = np.fft.rfft(orig)
    new_fft = np.fft.rfft(new)
    bin_idx = _lane_bin(lane)
    orig_fft[bin_idx] = new_fft[bin_idx]
    mixed = np.fft.irfft(orig_fft)
    return mixed.astype(orig.dtype)
 
def replay_envelope(env_wave: np.ndarray, lane: int) -> np.ndarray:
    """Replay ``env_wave``'s envelope on ``lane`` starting at phase 0."""
    env = np.abs(env_wave)
    t = np.arange(FRAME_SAMPLES, dtype="f4") / FS
    carrier = np.sin(2 * np.pi * lane_frequency(lane) * t)
    return (env * carrier).astype("f4")

def generate_bit_wave(lane: int) -> np.ndarray:
    """Return a 500 ms ADSR tone for ``lane`` starting at phase 0."""
    a_ms, d_ms, sustain_level, r_ms = DATA_ADSR
    a_n = int(FS * (a_ms / 1000.0))
    d_n = int(FS * (d_ms / 1000.0))
    r_n = int(FS * (r_ms / 1000.0))
    s_n = FRAME_SAMPLES - (a_n + d_n + r_n)
    env = np.concatenate(
        [
            np.linspace(0.0, 1.0, a_n, endpoint=False),
            np.linspace(1.0, sustain_level, d_n, endpoint=False),
            np.full(s_n, sustain_level),
            np.linspace(sustain_level, 0.0, r_n, endpoint=True),
        ]
    )
    t = np.arange(FRAME_SAMPLES) / FS
    carrier = np.sin(2 * np.pi * lane_frequency(lane) * t)
    return (env * carrier).astype("f4")

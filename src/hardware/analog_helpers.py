"""Analog helper functions for per-lane FFT operations."""
import numpy as np
from .constants import FS, FRAME_SAMPLES, lane_frequency, LANES, DATA_ADSR

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
    wave = np.fft.irfft(single, n=len(frame))
    return wave.astype(frame.dtype)
 
def lane_rms(wave: np.ndarray, lane: int, half_bw: int = 3) -> float:
    """Return a length-independent spectral amplitude for ``lane``.

    Raw FFT magnitudes scale with the number of samples.  Comparing those
    magnitudes with physical amplitude thresholds made the write-bias floor
    look like a logical one.  The band norm is therefore converted back to a
    sinusoidal amplitude before it reaches either decoding or NAND logic.
    """
    spec = np.fft.rfft(wave.astype("f4"))
    band = lane_band(lane, half_bw)
    band_energy = np.sqrt(np.sum(np.abs(spec[band]) ** 2))
    return float(2.0 * band_energy / len(wave))

def track_rms(wave: np.ndarray) -> float:
    """Whole-track RMS energy."""
    return float(np.sqrt(np.mean(wave.astype("f4") ** 2)))


def mix_fft_lane(orig: np.ndarray, new: np.ndarray, lane: int) -> np.ndarray:
    """Mix a single lane component from 'new' into 'orig' via FFT bin replacement."""
    orig_fft = np.fft.rfft(orig)
    new_fft = np.fft.rfft(new)
    bin_idx = _lane_bin(lane)
    orig_fft[bin_idx] = new_fft[bin_idx]
    # ``FRAME_SAMPLES`` is odd at the current sample rate.  Without an
    # explicit length NumPy reconstructs 2*(bins-1), silently dropping one
    # sample and making the next lane combination non-broadcastable.
    mixed = np.fft.irfft(orig_fft, n=len(orig))
    return mixed.astype(orig.dtype)
 
def replay_envelope(env_wave: np.ndarray, lane: int) -> np.ndarray:
    """Replay ``env_wave``'s envelope on ``lane`` starting at phase 0."""
    env = np.abs(env_wave)
    if len(env) != FRAME_SAMPLES:
        env = np.pad(env, (0, max(0, FRAME_SAMPLES - len(env))))[:FRAME_SAMPLES]
    t = np.arange(FRAME_SAMPLES, dtype="f4") / FS
    carrier = np.sin(2 * np.pi * lane_frequency(lane) * t)
    return (env * carrier).astype("f4")

def generate_bit_wave(lane: int) -> np.ndarray:
    """Return an ADSR tone for ``lane`` spanning one frame."""
    a_ratio, d_ratio, s_ratio, r_ratio, attack_level, sustain_level = DATA_ADSR
    total = a_ratio + d_ratio + s_ratio + r_ratio
    if total <= 0:
        env = np.zeros(FRAME_SAMPLES, dtype="f4")
    else:
        a_n = int(FRAME_SAMPLES * (a_ratio / total))
        d_n = int(FRAME_SAMPLES * (d_ratio / total))
        s_n = int(FRAME_SAMPLES * (s_ratio / total))
        r_n = FRAME_SAMPLES - (a_n + d_n + s_n)
        env = np.concatenate([
            np.linspace(0.0, attack_level, a_n, endpoint=False) if a_n > 0 else np.array([], dtype="f4"),
            np.linspace(attack_level, sustain_level, d_n, endpoint=False) if d_n > 0 else np.array([], dtype="f4"),
            np.full(s_n, sustain_level, dtype="f4") if s_n > 0 else np.array([], dtype="f4"),
            np.linspace(sustain_level, 0.0, r_n, endpoint=True) if r_n > 0 else np.array([], dtype="f4"),
        ])
        if len(env) < FRAME_SAMPLES:
            env = np.pad(env, (0, FRAME_SAMPLES - len(env)), mode="constant")
        elif len(env) > FRAME_SAMPLES:
            env = env[:FRAME_SAMPLES]
    t = np.arange(FRAME_SAMPLES) / FS
    carrier = np.sin(2 * np.pi * lane_frequency(lane) * t)
    return (env * carrier / LANES).astype("f4")

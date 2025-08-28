
# random.py
from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Callable, Union
import os, time, hmac, hashlib, struct, socket, tempfile

# ======================================================================================
# Enums (as given)
# ======================================================================================

class PRNG_SYS(Enum):
    OS_URANDOM          = auto()  # generic OS CSPRNG (getrandom/SecRandom/CNG)
    LINUX_GETRANDOM     = auto()
    MACOS_SECRANDOM     = auto()
    WIN_CNG             = auto()
    X86_RDRAND          = auto()
    X86_RDSEED          = auto()
    ARM_RNDR            = auto()
    JITTER_ENTROPY      = auto()  # user-space jitter collector (TRNG-ish; then extract)
    # Optional device-fed:
    CAMERA_NOISE        = auto()
    MIC_NOISE           = auto()
    DISK_TIMING         = auto()
    NET_TIMING          = auto()

class PRNG_ALGO(Enum):  # fast, non-crypto
    LCG_LEHMER64        = auto()
    MCG_64              = auto()
    PCG32_XSH_RR        = auto()
    PCG64_DXSM          = auto()
    XORSHIFT32          = auto()
    XORSHIFT64STAR      = auto()
    XORSHIFT128PLUS     = auto()
    XOROSHIRO128SS      = auto()
    XOSHIRO256SS        = auto()
    SPLITMIX64          = auto()   # great seeder / mixer
    JSF64               = auto()   # Jenkins Small Fast
    SFC64               = auto()   # Small Fast Chaotic
    MSWS                = auto()   # Middle-square Weyl sequence
    MT19937             = auto()
    WELL512             = auto()
    WELL1024            = auto()
    LFSR113             = auto()
    KISS99              = auto()
    LXM_128_64          = auto()   # LXM family (Blackman–Vigna)
    # Counter-based (fast, parallel-friendly; not crypto by default):
    THREEFRY_2X64_20    = auto()
    PHILOX_4X32_10      = auto()

class CSPRNG_ALGO(Enum):  # cryptographic
    CHACHA20            = auto()
    AES128_CTR_DRBG     = auto()   # SP 800-90A
    AES256_CTR_DRBG     = auto()
    HMAC_DRBG_SHA256    = auto()
    HASH_DRBG_SHA256    = auto()
    ISAAC               = auto()   # historical, not best practice today
    BBS                 = auto()   # Blum Blum Shub (slow, academic)

class RNG_ALGO(Enum):  # true/randomness collectors (need conditioning)
    TRNG_OS_POOL        = auto()   # OS entropy pool abstraction
    TRNG_JITTER         = auto()   # timer jitter collector (software-only)
    TRNG_RING_OSC       = auto()   # ring-oscillator (needs HW)
    TRNG_AVALANCHE      = auto()   # zener/avalanche diode noise (HW)
    TRNG_SRAM_STARTUP   = auto()   # SRAM power-up bias (HW/embedded)
    TRNG_CAMERA_NOISE   = auto()
    TRNG_MIC_NOISE      = auto()


# ======================================================================================
# Bit helpers
# ======================================================================================

MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1

def u32(x: int) -> int: return x & MASK32
def u64(x: int) -> int: return x & MASK64
def rol32(x: int, r: int) -> int: return u32((x << r) | (x >> (32 - r)))
def rolu64(x: int, r: int) -> int: return u64((x << r) | (x >> (64 - r)))

# ======================================================================================
# Entropy collectors (TRNG-ish) + OS adapters
# ======================================================================================

def _os_urandom(n: int) -> bytes:
    # portable: Linux getrandom/urandom, macOS SecRandom, Windows CNG behind the scenes
    return os.urandom(n)

def _linux_getrandom(n: int) -> bytes:
    if hasattr(os, "getrandom"):
        return os.getrandom(n)
    return _os_urandom(n)

def _macos_secrandom(n: int) -> bytes:
    # In Python, os.urandom() already uses SecRandom on macOS.
    return _os_urandom(n)

def _win_cng(n: int) -> bytes:
    # os.urandom uses BCryptGenRandom on Windows.
    return _os_urandom(n)

def _cpu_rdrand(n: int) -> bytes:
    # Python cannot portably issue RDRAND; fall back to OS pool.
    return _os_urandom(n)

def _cpu_rdseed(n: int) -> bytes:
    # Same story as RDRAND.
    return _os_urandom(n)

def _arm_rndr(n: int) -> bytes:
    # ARMv8.5 RNDR not accessible from Python; fallback.
    return _os_urandom(n)

def _jitter_entropy(n: int) -> bytes:
    # Basic software-only jitter collector + HKDF-like expansion.
    # Collect timer deltas while stirring state; then hash & expand.
    def _vn(bits: int, k: int = 64):
        out = 0; j = 0
        for i in range(0, k - 1, 2):
            b1 = (bits >> i) & 1
            b2 = (bits >> (i + 1)) & 1
            if b1 == b2:
                continue
            # map 10 -> 1, 01 -> 0
            out |= ((b1 & (~b2 & 1)) & 1) << j
            j += 1
        return out, j

    h = hashlib.sha256()
    last = time.perf_counter_ns()
    acc = 0
    # Gather until we have at least one digest worth of material
    while True:
        # micro-perturb: do some arithmetic with data dep to induce jitter
        x = 0
        for _ in range(257):
            x = (x * 1103515245 + 12345) & MASK32
        now = time.perf_counter_ns()
        delta = now - last
        last = now
        acc ^= (delta + (x << 17)) & MASK64
        v, bits = _vn(acc, 64)
        h.update(struct.pack("<Q", u64(v)) + struct.pack("<I", bits))
        if len(h.digest()) >= 32:
            break
    key = h.digest()
    out = b""
    ctr = 0
    while len(out) < n:
        ctr += 1
        out += hmac.new(key, b"JITTER"+ctr.to_bytes(4,"little"), hashlib.sha256).digest()
    return out[:n]

def _disk_timing_entropy(n: int) -> bytes:
    # time small writes + fsync as jitter source (slow; dev/diagnostic use)
    h = hashlib.sha256()
    with tempfile.NamedTemporaryFile(delete=True) as f:
        for i in range(32):
            t0 = time.perf_counter_ns()
            f.write(os.urandom(64))
            f.flush()
            os.fsync(f.fileno())
            t1 = time.perf_counter_ns()
            h.update(struct.pack("<QQ", t0, t1))
    key = h.digest()
    return hmac.new(key, b"DISK", hashlib.sha256).digest()[:n] if n <= 32 else (key + _disk_timing_entropy(n-32))

def _net_timing_entropy(n: int) -> bytes:
    # round-trip timing jitter to localhost UDP
    h = hashlib.sha256()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.02)
    try:
        for i in range(128):
            payload = struct.pack("<Q", i) + os.urandom(8)
            t0 = time.perf_counter_ns()
            try:
                s.sendto(payload, ("127.0.0.1", 9))  # discard port
            except Exception:
                pass
            t1 = time.perf_counter_ns()
            h.update(struct.pack("<QQ", t0, t1))
    finally:
        s.close()
    key = h.digest()
    out = b""
    ctr = 0
    while len(out) < n:
        ctr += 1
        out += hmac.new(key, b"NET"+ctr.to_bytes(4,"little"), hashlib.sha256).digest()
    return out[:n]

def _camera_entropy(n: int) -> bytes:
    raise NotImplementedError("Camera noise collector is not wired in this environment.")

def _mic_entropy(n: int) -> bytes:
    raise NotImplementedError("Microphone noise collector is not wired in this environment.")

def get_entropy(n: int, source: PRNG_SYS) -> bytes:
    """Switch over system/TRNG sources."""
    match source:
        case PRNG_SYS.OS_URANDOM:      return _os_urandom(n)
        case PRNG_SYS.LINUX_GETRANDOM: return _linux_getrandom(n)
        case PRNG_SYS.MACOS_SECRANDOM: return _macos_secrandom(n)
        case PRNG_SYS.WIN_CNG:         return _win_cng(n)
        case PRNG_SYS.X86_RDRAND:      return _cpu_rdrand(n)
        case PRNG_SYS.X86_RDSEED:      return _cpu_rdseed(n)
        case PRNG_SYS.ARM_RNDR:        return _arm_rndr(n)
        case PRNG_SYS.JITTER_ENTROPY:  return _jitter_entropy(n)
        case PRNG_SYS.DISK_TIMING:     return _disk_timing_entropy(n)
        case PRNG_SYS.NET_TIMING:      return _net_timing_entropy(n)
        case PRNG_SYS.CAMERA_NOISE:    return _camera_entropy(n)
        case PRNG_SYS.MIC_NOISE:       return _mic_entropy(n)
        case _:                        return _os_urandom(n)

# Convenience mapping for RNG_ALGO → collector
def collect_rng(n: int, algo: RNG_ALGO) -> bytes:
    match algo:
        case RNG_ALGO.TRNG_OS_POOL:      return get_entropy(n, PRNG_SYS.OS_URANDOM)
        case RNG_ALGO.TRNG_JITTER:       return get_entropy(n, PRNG_SYS.JITTER_ENTROPY)
        case RNG_ALGO.TRNG_RING_OSC:     raise NotImplementedError("Ring oscillator TRNG requires hardware.")
        case RNG_ALGO.TRNG_AVALANCHE:    raise NotImplementedError("Avalanche diode TRNG requires hardware.")
        case RNG_ALGO.TRNG_SRAM_STARTUP: raise NotImplementedError("SRAM startup bias TRNG requires hardware.")
        case RNG_ALGO.TRNG_CAMERA_NOISE: return get_entropy(n, PRNG_SYS.CAMERA_NOISE)
        case RNG_ALGO.TRNG_MIC_NOISE:    return get_entropy(n, PRNG_SYS.MIC_NOISE)


# ======================================================================================
# Seed mixers and counter expanders
# ======================================================================================

class SplitMix64:
    def __init__(self, seed: int):
        self.state = u64(seed)

    def next_u64(self) -> int:
        self.state = u64(self.state + 0x9E3779B97F4A7C15)
        z = self.state
        z = u64((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9)
        z = u64((z ^ (z >> 27)) * 0x94D049BB133111EB)
        return u64(z ^ (z >> 31))

def seed_from_bytes(b: bytes) -> int:
    h = hashlib.sha256(b).digest()
    return int.from_bytes(h[:8], "little")

def ensure_seed64(seed: Optional[Union[int, bytes]] = None) -> int:
    if seed is None:
        cs = make_csprng(CSPRNG_ALGO.CHACHA20)
        return int.from_bytes(cs.random_bytes(8), "little")
    if isinstance(seed, bytes):
        return seed_from_bytes(seed)
    return u64(seed)

# ======================================================================================
# PRNGs (non-crypto), unified interface
# ======================================================================================

@dataclass
class PRNG:
    """Unified adapter: exposes next_u32/next_u64/next_float/next_bytes."""
    next_u64_fn: Callable[[], int]

    def next_u64(self) -> int:
        return self.next_u64_fn()

    def next_u32(self) -> int:
        return u32(self.next_u64() >> 32)

    def next_float(self) -> float:
        # 53-bit precision uniform in [0,1)
        x = self.next_u64() >> 11
        return x / float(1 << 53)

    def next_bytes(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            out.extend(self.next_u64().to_bytes(8, "little"))
        return bytes(out[:n])

# -- Implementations ---------------------------------------------------------

def make_lcg_lehmer64(seed: int) -> PRNG:
    # a(n+1) = a(n) * 48271 mod (2^31 - 1) is classic Lehmer (Park-Miller).
    # We'll use a 64-bit Lehmer multiplicative: a(n+1) = a(n) * A mod 2^64 (not prime modulus; OK for toy use).
    A = 2862933555777941757  # will give multiplicative LCG behavior
    state = u64(seed or 1)
    def next_u64():
        nonlocal state
        state = u64(state * A + 3037000493)  # add Weyl to improve
        return state
    return PRNG(next_u64)

def make_mcg_64(seed: int) -> PRNG:
    # Multiplicative congruential (no increment), add Weyl for full period.
    A = 6364136223846793005
    weyl = 0xB5AD4ECEDA1CE2A9
    state = u64(seed or 1)
    def next_u64():
        nonlocal state
        state = u64(state * A)
        return u64(state + weyl)
    return PRNG(next_u64)

def make_pcg32_xsh_rr(seed: int, inc: int = 1442695040888963407) -> PRNG:
    # PCG32 XSH RR (O'Neill). 64-bit state, 32-bit output.
    state = u64(seed)
    inc = u64((inc << 1) | 1)
    def next_u64():
        nonlocal state
        old = state
        state = u64(old * 6364136223846793005 + inc)
        xorshifted = u32(((old >> 18) ^ old) >> 27)
        rot = (old >> 59) & 31
        out32 = u32((xorshifted >> rot) | (xorshifted << ((-rot) & 31)))
        # pack two calls per 64 bits
        low = out32
        # second 32
        old2 = state
        state = u64(old2 * 6364136223846793005 + inc)
        x2 = u32(((old2 >> 18) ^ old2) >> 27)
        r2 = (old2 >> 59) & 31
        hi = u32((x2 >> r2) | (x2 << ((-r2) & 31)))
        return (hi << 32) | low
    return PRNG(next_u64)

def make_pcg64_dxsm(seed: int) -> PRNG:
    # Minimal PCG64 + DXSM output transform (approx).
    state = u128 = seed | (1 << 63)  # ensure odd-ish; simple seed trick
    inc = 0xDA3E39CB94B95BDB
    def step(x):
        return u64(x * 6364136223846793005 + inc)
    def dxsm(x):
        x ^= (x >> 32)
        x *= 0xd2b74407b1ce6e93 & MASK64
        x ^= (x >> 29)
        x *= 0x9fb21c651e98df25 & MASK64
        x ^= (x >> 32)
        return u64(x)
    def next_u64():
        nonlocal state
        state = step(state)
        return dxsm(state)
    return PRNG(next_u64)

def make_xorshift32(seed: int) -> PRNG:
    state = u32(seed or 1)
    def next_u64():
        nonlocal state
        x = state
        x ^= (x << 13) & MASK32
        x ^= (x >> 17)
        x ^= (x << 5) & MASK32
        state = u32(x)
        # widen to 64 by stepping twice
        y = state
        y ^= (y << 13) & MASK32
        y ^= (y >> 17)
        y ^= (y << 5) & MASK32
        state = u32(y)
        return (y << 32) | x
    return PRNG(next_u64)

def make_xorshift64star(seed: int) -> PRNG:
    state = u64(seed or 1)
    def next_u64():
        nonlocal state
        x = state
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        state = u64(x)
        return u64(x * 0x2545F4914F6CDD1D)
    return PRNG(next_u64)

def make_xorshift128plus(seed: int) -> PRNG:
    sm = SplitMix64(seed)
    s0 = sm.next_u64() or 1
    s1 = sm.next_u64() or 2
    def next_u64():
        nonlocal s0, s1
        x = s0; y = s1
        s0 = y
        x ^= (x << 23) & MASK64
        s1 = u64(x ^ y ^ (x >> 18) ^ (y >> 5))
        return u64(s1 + y)
    return PRNG(next_u64)

def make_xoroshiro128ss(seed: int) -> PRNG:
    sm = SplitMix64(seed)
    s0 = sm.next_u64()
    s1 = sm.next_u64()
    def next_u64():
        nonlocal s0, s1
        res = rolu64(s0 * 5, 7) * 9
        t = s1 ^ s0
        s0 = rolu64(s0, 24) ^ t ^ ((t << 16) & MASK64)
        s1 = rolu64(t, 37)
        return u64(res)
    return PRNG(next_u64)

def make_xoshiro256ss(seed: int) -> PRNG:
    sm = SplitMix64(seed)
    s = [sm.next_u64(), sm.next_u64(), sm.next_u64(), sm.next_u64()]
    def next_u64():
        nonlocal s
        result = rolu64(s[1] * 5, 7) * 9
        t = u64(s[1] << 17)
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]
        s[2] ^= t
        s[3] = rolu64(s[3], 45)
        return u64(result)
    return PRNG(next_u64)

def make_sfc64(seed: int) -> PRNG:
    a = SplitMix64(seed).next_u64()
    b = SplitMix64(seed ^ 0xDEADBEEF).next_u64()
    c = SplitMix64(seed ^ 0xCAFEBABE).next_u64()
    counter = 1
    def next_u64():
        nonlocal a, b, c, counter
        res = u64(a + b + counter)
        counter = u64(counter + 1)
        b ^= (b >> 11)
        c = u64(c + (c << 3))
        a = u64((a ^ (a << 24)) + res)
        a = u64(a)
        return res
    return PRNG(next_u64)

def make_jsf64(seed: int) -> PRNG:
    a = 0xf1ea5eed
    b = c = d = SplitMix64(seed).next_u64()
    for _ in range(20):  # warm up
        e = a - rol32(b, 27)
        a = b ^ rol32(c, 17)
        b = c + d
        c = d + e
        d = a + e
    def next_u64():
        nonlocal a, b, c, d
        e = a - rol32(b, 27)
        a = b ^ rol32(c, 17)
        b = c + d
        c = d + e
        d = a + e
        # widen (compose two 32-bit steps)
        e2 = a - rol32(b, 27)
        a = b ^ rol32(c, 17)
        b = c + d
        c = d + e2
        d = a + e2
        return (u64(d) << 32) | (u64(c) & MASK32)
    return PRNG(next_u64)

def make_msws(seed: int) -> PRNG:
    x = u64(seed)
    w = 0
    s = 0xb5ad4eceda1ce2a9
    def next_u64():
        nonlocal x, w
        x = u64(x * x + (w := u64(w + s)))
        x = u64((x >> 32) | (x << 32))
        return x
    return PRNG(next_u64)

# Counter-based: Threefry2x64/20 (Salmon et al.)
def _threefry2x64_round(x0, x1, k0, k1, k2, r):
    x0 = u64(x0 + x1); x1 = rolu64(x1, r) ^ x0
    return x0, x1
def make_threefry_2x64_20(seed: int) -> PRNG:
    # Use SplitMix64 to derive keys and counter
    sm = SplitMix64(seed)
    k0, k1 = sm.next_u64(), sm.next_u64()
    ctr0, ctr1 = 0, 0
    R = [14,16,52,57,23,40,5,37]  # classic rotation constants
    def block(c0, c1):
        x0, x1 = c0 + k0, c1 + k1
        k2 = u64(0x1BD11BDAA9FC1A22 ^ k0 ^ k1)
        # 20 rounds in 4x key injection schedule
        for i in range(20):
            r = R[i % 8]
            x0, x1 = _threefry2x64_round(x0, x1, k0, k1, k2, r)
            if (i % 4) == 3:
                # inject key + tweak (here tweak is round index)
                add = [k0, k1, k2][(i//4) % 3]
                x0 = u64(x0 + add + i//4)
                x1 = u64(x1 + [k1, k2, k0][(i//4) % 3])
        return x0, x1
    def next_u64():
        nonlocal ctr0, ctr1
        x0, x1 = block(ctr0, ctr1)
        ctr0 = u64(ctr0 + 1)
        if ctr0 == 0: ctr1 = u64(ctr1 + 1)
        return x0
    return PRNG(next_u64)

# Counter-based: Philox 4x32-10 (Salmon et al.)
def _mulhilo32(a,b):
    p = a * b
    return (p & MASK32, (p >> 32) & MASK32)
def make_philox_4x32_10(seed: int) -> PRNG:
    sm = SplitMix64(seed)
    k0 = sm.next_u64() & MASK32
    k1 = sm.next_u64() & MASK32
    ctr = [0,0,0,0]
    M0, M1 = 0xD2511F53, 0xCD9E8D57
    W0, W1 = 0x9E3779B9, 0xBB67AE85
    def round4x(c, k0, k1):
        (lo0, hi0) = _mulhilo32(M0, c[0])
        (lo1, hi1) = _mulhilo32(M1, c[2])
        return [
            hi1 ^ c[1] ^ k0,
            lo1,
            hi0 ^ c[3] ^ k1,
            lo0
        ]
    def block(c, k0, k1):
        x = c[:]
        kk0, kk1 = k0, k1
        for _ in range(10):
            x = round4x(x, kk0, kk1)
            kk0 = (kk0 + W0) & MASK32
            kk1 = (kk1 + W1) & MASK32
        return x
    def next_u64():
        nonlocal ctr
        out = block(ctr, k0, k1)
        # increment 128-bit counter
        ctr[0] = (ctr[0] + 1) & MASK32
        if ctr[0] == 0:
            ctr[1] = (ctr[1] + 1) & MASK32
            if ctr[1] == 0:
                ctr[2] = (ctr[2] + 1) & MASK32
                if ctr[2] == 0:
                    ctr[3] = (ctr[3] + 1) & MASK32
        return (out[1] << 32) | out[0]
    return PRNG(next_u64)

# Simple LXM_128_64: SplitMix64 as L, xoroshiro128** as X, mix
def make_lxm_128_64(seed: int) -> PRNG:
    sm = SplitMix64(seed)
    # X part
    s0, s1 = sm.next_u64(), sm.next_u64()
    # L part (LCG)
    a = 0xd1342543de82ef95
    lstate = sm.next_u64()
    def next_u64():
        nonlocal s0, s1, lstate
        # xoroshiro128** output
        resx = rolu64(s0 * 5, 7) * 9
        t = s1 ^ s0
        s0 = rolu64(s0, 24) ^ t ^ ((t << 16) & MASK64)
        s1 = rolu64(t, 37)
        # LCG step
        lstate = u64(lstate * a + 1)
        # mix
        return u64(resx + lstate)
    return PRNG(next_u64)

def make_prng(algo: PRNG_ALGO, seed: Optional[Union[int, bytes]] = None) -> PRNG:
    s = ensure_seed64(seed)
    match algo:
        case PRNG_ALGO.SPLITMIX64:         return PRNG(SplitMix64(s).next_u64)
        case PRNG_ALGO.XORSHIFT32:         return make_xorshift32(s)
        case PRNG_ALGO.XORSHIFT64STAR:     return make_xorshift64star(s)
        case PRNG_ALGO.XORSHIFT128PLUS:    return make_xorshift128plus(s)
        case PRNG_ALGO.XOROSHIRO128SS:     return make_xoroshiro128ss(s)
        case PRNG_ALGO.XOSHIRO256SS:       return make_xoshiro256ss(s)
        case PRNG_ALGO.SFC64:              return make_sfc64(s)
        case PRNG_ALGO.JSF64:              return make_jsf64(s)
        case PRNG_ALGO.MSWS:               return make_msws(s)
        case PRNG_ALGO.LCG_LEHMER64:       return make_lcg_lehmer64(s)
        case PRNG_ALGO.MCG_64:             return make_mcg_64(s)
        case PRNG_ALGO.PCG32_XSH_RR:       return make_pcg32_xsh_rr(s)
        case PRNG_ALGO.PCG64_DXSM:         return make_pcg64_dxsm(s)
        case PRNG_ALGO.THREEFRY_2X64_20:   return make_threefry_2x64_20(s)
        case PRNG_ALGO.PHILOX_4X32_10:     return make_philox_4x32_10(s)
        # heavy ones not implemented here:
        case PRNG_ALGO.MT19937:            raise NotImplementedError("MT19937 placeholder; use Python random if needed.")
        case PRNG_ALGO.WELL512:            raise NotImplementedError("WELL512 not implemented.")
        case PRNG_ALGO.WELL1024:           raise NotImplementedError("WELL1024 not implemented.")
        case PRNG_ALGO.LFSR113:            raise NotImplementedError("LFSR113 not implemented.")
        case PRNG_ALGO.KISS99:             raise NotImplementedError("KISS99 not implemented.")
        case PRNG_ALGO.LXM_128_64:         return make_lxm_128_64(s)
        case _:                            return make_xoroshiro128ss(s)

# ======================================================================================
# CSPRNGs
# ======================================================================================

class ChaCha20:
    # Simple ChaCha20 (IETF variant, 32-byte key, 12-byte nonce, 32-bit counter)
    def __init__(self, key: bytes, nonce: bytes):
        if len(key) != 32: raise ValueError("ChaCha20 key must be 32 bytes")
        if len(nonce) != 12: raise ValueError("ChaCha20 nonce must be 12 bytes")
        self.key = key
        self.nonce = nonce
        self.counter = 0

    @staticmethod
    def _qr(a,b,c,d):
        a = (a + b) & MASK32; d ^= a; d = rol32(d, 16)
        c = (c + d) & MASK32; b ^= c; b = rol32(b, 12)
        a = (a + b) & MASK32; d ^= a; d = rol32(d, 8)
        c = (c + d) & MASK32; b ^= c; b = rol32(b, 7)
        return a,b,c,d

    def _block(self, counter: int) -> bytes:
        def w32(b,i): return int.from_bytes(b[4*i:4*i+4], "little")
        state = [
            0x61707865,0x3320646e,0x79622d32,0x6b206574,
            w32(self.key,0), w32(self.key,1), w32(self.key,2), w32(self.key,3),
            w32(self.key,4), w32(self.key,5), w32(self.key,6), w32(self.key,7),
            counter & MASK32,
            int.from_bytes(self.nonce[0:4], "little"),
            int.from_bytes(self.nonce[4:8], "little"),
            int.from_bytes(self.nonce[8:12], "little")
        ]
        x = state[:]
        for _ in range(10):  # 20 rounds (2 per loop)
            x[0],x[4],x[8],x[12]   = self._qr(x[0],x[4],x[8],x[12])
            x[1],x[5],x[9],x[13]   = self._qr(x[1],x[5],x[9],x[13])
            x[2],x[6],x[10],x[14]  = self._qr(x[2],x[6],x[10],x[14])
            x[3],x[7],x[11],x[15]  = self._qr(x[3],x[7],x[11],x[15])
            x[0],x[5],x[10],x[15]  = self._qr(x[0],x[5],x[10],x[15])
            x[1],x[6],x[11],x[12]  = self._qr(x[1],x[6],x[11],x[12])
            x[2],x[7],x[8],x[13]   = self._qr(x[2],x[7],x[8],x[13])
            x[3],x[4],x[9],x[14]   = self._qr(x[3],x[4],x[9],x[14])
        for i in range(16):
            x[i] = (x[i] + state[i]) & MASK32
        return b"".join(int(x[i]).to_bytes(4, "little") for i in range(16))

    def keystream(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            out.extend(self._block(self.counter))
            self.counter = (self.counter + 1) & MASK32
        return bytes(out[:n])

class HMAC_DRBG_SHA256:
    # NIST SP800-90A-ish HMAC-DRBG (no personalization/additional_input for brevity)
    def __init__(self, seed_material: bytes):
        self.K = b"\x00" * 32
        self.V = b"\x01" * 32
        self._update(seed_material)

    def _hmac(self, key, data): return hmac.new(key, data, hashlib.sha256).digest()

    def _update(self, provided_data: Optional[bytes]):
        self.K = self._hmac(self.K, self.V + b"\x00" + (provided_data or b""))
        self.V = self._hmac(self.K, self.V)
        if provided_data:
            self.K = self._hmac(self.K, self.V + b"\x01" + provided_data)
            self.V = self._hmac(self.K, self.V)

    def reseed(self, seed_material: bytes):
        self._update(seed_material)

    def random_bytes(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            self.V = self._hmac(self.K, self.V)
            out.extend(self.V)
        self._update(None)
        return bytes(out[:n])

class HASH_DRBG_SHA256:
    # Simple hash-DRBG: state = SHA256(state || counter)
    def __init__(self, seed_material: bytes):
        self.state = hashlib.sha256(b"seed"+seed_material).digest()
        self.counter = 0

    def random_bytes(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            self.counter += 1
            h = hashlib.sha256(self.state + self.counter.to_bytes(8,"little")).digest()
            out.extend(h)
            self.state = h
        return bytes(out[:n])

class BlumBlumShub:
    # Academic CSPRNG; slow but simple. Choose n = p*q where p,q ≡ 3 (mod 4).
    def __init__(self, seed: int, p: int = 1000003, q: int = 1000033):
        if (p % 4 != 3) or (q % 4 != 3):
            raise ValueError("p and q must be 3 mod 4")
        self.n = p*q
        x = (seed % self.n)
        self.x = (x*x) % self.n

    def next_bit(self) -> int:
        self.x = (self.x * self.x) % self.n
        return self.x & 1

    def random_bytes(self, n: int) -> bytes:
        out = 0
        for i in range(n*8):
            out |= (self.next_bit() << i)
        return out.to_bytes(n, "little")

@dataclass
class CSPRNG:
    random_bytes_fn: Callable[[int], bytes]
    def random_bytes(self, n: int) -> bytes:
        return self.random_bytes_fn(n)

def make_csprng(algo: CSPRNG_ALGO, seed_material: Optional[bytes] = None) -> CSPRNG:
    if seed_material is None:
        seed_material = get_entropy(48, PRNG_SYS.OS_URANDOM)

    match algo:
        case CSPRNG_ALGO.CHACHA20:
            key = hashlib.sha256(b"key"+seed_material).digest()   # 32
            nonce = hashlib.sha256(b"nonce"+seed_material).digest()[:12]
            ch = ChaCha20(key, nonce)
            return CSPRNG(ch.keystream)

        case CSPRNG_ALGO.HMAC_DRBG_SHA256:
            drbg = HMAC_DRBG_SHA256(seed_material)
            return CSPRNG(drbg.random_bytes)

        case CSPRNG_ALGO.HASH_DRBG_SHA256:
            drbg = HASH_DRBG_SHA256(seed_material)
            return CSPRNG(drbg.random_bytes)

        case CSPRNG_ALGO.BBS:
            s = seed_from_bytes(seed_material)
            bbs = BlumBlumShub(s)
            return CSPRNG(bbs.random_bytes)

        case CSPRNG_ALGO.AES128_CTR_DRBG | CSPRNG_ALGO.AES256_CTR_DRBG:
            raise NotImplementedError("AES-CTR DRBG requires an AES primitive (add PyCryptodome or OpenSSL).")

        case CSPRNG_ALGO.ISAAC:
            raise NotImplementedError("ISAAC not implemented here (historical; avoid in new designs).")

        case _:
            # default to ChaCha20
            key = hashlib.sha256(b"key"+seed_material).digest()
            nonce = hashlib.sha256(b"nonce"+seed_material).digest()[:12]
            ch = ChaCha20(key, nonce)
            return CSPRNG(ch.keystream)


# ======================================================================================
# High-level: unified random generator interface
# ======================================================================================

from enum import Enum, auto

class RANDOM_KIND(Enum):
    SYSTEM = auto()      # OS CSPRNG (default)
    CSPRNG = auto()      # Cryptographic PRNG (ChaCha20, HMAC_DRBG, etc)
    PRNG = auto()        # Fast non-crypto PRNG (Xoroshiro, PCG, etc)

def random_generator(kind=RANDOM_KIND.SYSTEM, algo=None, seed=None, dtype="float", batch_size=1, distribution="uniform"):
    """
    Returns a generator yielding random numbers of the requested type and distribution.
    kind: RANDOM_KIND enum (SYSTEM, CSPRNG, PRNG)
    algo: PRNG_ALGO or CSPRNG_ALGO (optional, for PRNG/CSPRNG kinds)
    seed: int or bytes (optional)
    dtype: "float" (default) for [0,1), "int" for 64-bit unsigned ints
    batch_size: number of values per yield
    distribution: "uniform" (default), "normal", or a callable/histogram
    """
    import math
    if seed is None and kind != RANDOM_KIND.PRNG:
        kind = RANDOM_KIND.CSPRNG
    if kind == RANDOM_KIND.PRNG and seed is None:
        cs = make_csprng(CSPRNG_ALGO.CHACHA20)
        seed = int.from_bytes(cs.random_bytes(8), "little")
    def get_uniforms():
        if kind == RANDOM_KIND.SYSTEM:
            def sys_iter():
                while True:
                    vals = []
                    for _ in range(batch_size):
                        if dtype == "int":
                            x = int.from_bytes(get_entropy(8, PRNG_SYS.OS_URANDOM), "little")
                            vals.append(x)
                        else:
                            x = int.from_bytes(get_entropy(8, PRNG_SYS.OS_URANDOM), "little") >> 11
                            vals.append(x / float(1 << 53))
                    yield vals if batch_size > 1 else vals[0]
            return sys_iter()
        elif kind == RANDOM_KIND.CSPRNG:
            csprng = make_csprng(algo or CSPRNG_ALGO.CHACHA20, seed)
            def csprng_iter():
                while True:
                    vals = []
                    for _ in range(batch_size):
                        if dtype == "int":
                            x = int.from_bytes(csprng.random_bytes(8), "little")
                            vals.append(x)
                        else:
                            x = int.from_bytes(csprng.random_bytes(8), "little") >> 11
                            vals.append(x / float(1 << 53))
                    yield vals if batch_size > 1 else vals[0]
            return csprng_iter()
        elif kind == RANDOM_KIND.PRNG:
            prng = make_prng(algo or PRNG_ALGO.XOROSHIRO128SS, seed)
            def prng_iter():
                while True:
                    vals = []
                    for _ in range(batch_size):
                        if dtype == "int":
                            x = prng.next_u64()
                            vals.append(x)
                        else:
                            x = prng.next_u64() >> 11
                            vals.append(x / float(1 << 53))
                    yield vals if batch_size > 1 else vals[0]
            return prng_iter()
        else:
            raise ValueError("Unknown RANDOM_KIND for random_generator")

    uniform_batches = get_uniforms()

    def uniform_scalars():
        for batch in uniform_batches:
            if isinstance(batch, list):
                for v in batch:
                    yield v
            else:
                yield batch

    uniforms = uniform_scalars()

    def normal_batch():
        # Box-Muller for normal distribution
        vals = []
        n = batch_size
        while len(vals) < n:
            u1 = next(uniforms)
            u2 = next(uniforms)
            z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            vals.append(z0)
            if len(vals) < n:
                z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
                vals.append(z1)
        return vals[:n]

    if callable(distribution):
        def custom_iter():
            while True:
                vals = [distribution() for _ in range(batch_size)]
                if dtype == "int":
                    vals = [int(v) for v in vals]
                yield vals if batch_size > 1 else vals[0]
        return custom_iter()
    elif isinstance(distribution, (list, tuple)):
        import random
        def hist_iter():
            while True:
                vals = random.choices(distribution, k=batch_size)
                if dtype == "int":
                    vals = [int(v) for v in vals]
                yield vals if batch_size > 1 else vals[0]
        return hist_iter()
    elif distribution == "normal":
        def normal_iter():
            while True:
                vals = normal_batch()
                if dtype == "int":
                    vals = [int(v) for v in vals]
                yield vals if batch_size > 1 else vals[0]
        return normal_iter()
    elif distribution == "uniform":
        def uniform_iter():
            while True:
                vals = [next(uniforms) for _ in range(batch_size)]
                if dtype == "int":
                    vals = [int(v) for v in vals]
                yield vals if batch_size > 1 else vals[0]
        return uniform_iter()
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# --- Python-random-like interface using project PRNG infra ---
class Random:
    """
    Python random-like interface using the project's random_generator system.
    By default uses fast PRNG (Xoroshiro128**), but can be seeded and configured.
    """
    def __init__(self, kind=None, algo=None, seed=None):
        from .random import RANDOM_KIND, PRNG_ALGO
        self.kind = kind or RANDOM_KIND.PRNG
        self.algo = algo or PRNG_ALGO.XOROSHIRO128SS
        self.seed = seed
        self._reset_gen()

    def _reset_gen(self):
        self._gen = random_generator(kind=self.kind, algo=self.algo, seed=self.seed, dtype="float", batch_size=1)

    def seed(self, seed=None):
        self.seed = seed
        self._reset_gen()

    def random(self):
        return next(self._gen)

    def randint(self, a, b):
        # Uniform integer in [a, b]
        r = next(random_generator(kind=self.kind, algo=self.algo, seed=self.seed, dtype="float", batch_size=1))
        return a + int(r * ((b - a) + 1))

    def uniform(self, a, b):
        r = next(self._gen)
        return a + (b - a) * r

    def choice(self, seq):
        if not seq:
            raise IndexError("Cannot choose from an empty sequence")
        idx = int(next(self._gen) * len(seq))
        return seq[idx]

    def shuffle(self, x):
        # Fisher-Yates shuffle
        for i in reversed(range(1, len(x))):
            j = int(next(self._gen) * (i + 1))
            x[i], x[j] = x[j], x[i]

    def sample(self, population, k):
        if k > len(population):
            raise ValueError("Sample larger than population")
        pool = list(population)
        self.shuffle(pool)
        return pool[:k]

    def gauss(self, mu: float = 0.0, sigma: float = 1.0):
        """Gaussian (normal) variate via Box–Muller.

        Defaults to standard normal (mu=0, sigma=1), a scientifically
        conventional baseline for measurement noise and aggregated effects
        by the central limit theorem.
        """
        
        # Guard u1 in (0,1] to avoid log(0); u2 in [0,1)
        u1 = max(next(self._gen), 1e-12)
        u2 = next(self._gen)
        import math
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + z0 * sigma

    def normalvariate(self, mu, sigma):
        return self.gauss(mu, sigma)

    # Energy-distribution helpers ------------------------------------------------
    def sample_energy(self, kind: str = "thermal_boltzmann", **params):
        """Sample a physically meaningful energy-like random variable.

        Kinds:
        - thermal_boltzmann: E ≥ 0 with P(E) ∝ exp(-E/kT)
            params: kT (default 1.0)
        - maxwell_boltzmann_energy: 3D kinetic energy with Gamma(3/2, kT)
            params: kT (default 1.0)
        - gaussian_energy: measurement-like around mean energy
            params: mu (default 1.0), sigma (default 0.1)
        - lognormal_energy: multiplicative processes; E = exp(N(mu_log, sigma_log))
            params: mu_log (0.0), sigma_log (1.0)
        """
        import math
        kind = (kind or "").lower()
        if kind == "thermal_boltzmann":
            kT = float(params.get("kT", 1.0))
            # Exponential with mean kT: E = -kT * ln(1-u)
            u = max(next(self._gen), 1e-12)
            return -kT * math.log(1.0 - u)
        elif kind in ("maxwell_boltzmann_energy", "maxwell-boltzmann-energy", "mb_energy"):
            kT = float(params.get("kT", 1.0))
            # If Z_i ~ N(0,1), S = sum Z_i^2 ~ ChiSquare(3) = Gamma(1.5, 2)
            # Then E ~ Gamma(1.5, kT) by scaling: E = (kT/2) * S
            z1 = self.gauss()
            z2 = self.gauss()
            z3 = self.gauss()
            S = z1*z1 + z2*z2 + z3*z3
            return 0.5 * kT * S
        elif kind in ("gaussian_energy", "gaussian"):
            mu = float(params.get("mu", 1.0))
            sigma = float(params.get("sigma", 0.1))
            return self.gauss(mu, sigma)
        elif kind in ("lognormal_energy", "lognormal"):
            mu_log = float(params.get("mu_log", 0.0))
            sigma_log = float(params.get("sigma_log", 1.0))
            return math.exp(self.gauss(mu_log, sigma_log))
        else:
            raise ValueError(f"Unknown energy distribution kind: {kind}")

    # Add more methods as needed for your use case


    # ------------------------------------------------------------------------------
    # Human-friendly test main for quick validation and demonstration
    # ------------------------------------------------------------------------------
    if __name__ == "__main__":
        print("[random.py] Human-friendly test main: Random tensor creation demo\n")
        import sys
        import importlib
        # Lazy import AbstractTensor and creation helpers
        

        # Helper to call random_tensor with universal kwargs support
        def call_random_tensor(*args, **kwargs):
            from .creation import random_tensor
            import inspect
            sig = inspect.signature(random_tensor)
            # Only pass kwargs that are accepted by random_tensor
            filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return random_tensor(*args, **filtered)

        print("Random float tensor (uniform, shape=(3, 4)):")
        t1 = call_random_tensor((3, 4), dtype="float", distribution="uniform", seed=42)
        print(t1)

        print("\nRandom int tensor (uniform, shape=(3, 4)):")
        t2 = call_random_tensor((3, 4), dtype="int", distribution="uniform", seed=42)
        print(t2)

        print("\nRandom float tensor (normal, shape=(3, 4)):")
        t3 = call_random_tensor((3, 4), dtype="float", distribution="normal", seed=42)
        print(t3)

        print("\nRandom int tensor (normal, shape=(3, 4)):")
        t4 = call_random_tensor((3, 4), dtype="int", distribution="normal", seed=42)
        print(t4)

        print("\nRandom tensor from histogram [10, 20, 30] (int, shape=(2, 5)):")
        t5 = call_random_tensor((2, 5), dtype="int", distribution=[10, 20, 30], seed=42)
        print(t5)

        print("\nRandom tensor from custom callable (lambda: 99), shape=(2, 2):")
        t6 = call_random_tensor((2, 2), dtype="int", distribution=lambda: 99, seed=42)
        print(t6)

        print("\nAll done!\n")

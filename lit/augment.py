"""Acoustic augmentation for training clips: make clean read speech look like phone voice notes.

All functions take and return float32 mono 16 kHz arrays in [-1, 1]. Nothing here uses external
data: babble is other training clips, reverb is a synthetic impulse response.
"""

from __future__ import annotations

import subprocess

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt

SR = 16000


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-9)


def _bounded(y: np.ndarray) -> np.ndarray:
    peak = float(np.abs(y).max()) if len(y) else 0.0
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def gain(y: np.ndarray, db: float) -> np.ndarray:
    return _bounded(y * 10 ** (db / 20))


def add_babble(y: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(y):
        noise = np.tile(noise, len(y) // len(noise) + 1)
    noise = noise[: len(y)]
    scale = _rms(y) / (_rms(noise) * 10 ** (snr_db / 20))
    return _bounded(y + scale * noise)


def reverb(y: np.ndarray, rt60: float, rng: np.random.Generator) -> np.ndarray:
    # exponentially decaying noise burst: a generic room, -60 dB after rt60 seconds
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0  # direct path
    rir /= np.sqrt(np.sum(rir ** 2))
    out = fftconvolve(y, rir)[: len(y)]
    out *= _rms(y) / _rms(out)
    return _bounded(out)


_BANDPASS = butter(4, [300, 3400], btype="band", fs=SR, output="sos")


def bandpass(y: np.ndarray) -> np.ndarray:
    return _bounded(sosfiltfilt(_BANDPASS, y))


def mp3_roundtrip(y: np.ndarray, kbps: int) -> np.ndarray:
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    enc = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                          "-c:a", "libmp3lame", "-b:a", f"{kbps}k", "-f", "mp3", "pipe:1"],
                         input=pcm, capture_output=True, check=True).stdout
    dec = subprocess.run(["ffmpeg", "-v", "error", "-f", "mp3", "-i", "pipe:0", "-f", "s16le", "-ar", str(SR),
                          "-ac", "1", "pipe:1"], input=enc, capture_output=True, check=True).stdout
    out = np.frombuffer(dec, dtype="<i2").astype(np.float32) / 32767
    # the codec pads the start; realign by cross-correlation on the first second, then crop
    n = min(len(out), len(y), SR)
    lag = int(np.argmax(np.correlate(out[: n + SR // 10], y[:n], mode="valid")))
    out = out[lag: lag + len(y)]
    return _bounded(out)


class AcousticAugment:
    """With probability p, apply 1-3 random degradations to a clip. Seeded, so runs are repeatable."""

    OPS = ("mp3", "babble", "reverb", "gain", "bandpass")

    def __init__(self, babble_pool: list[np.ndarray], p: float = 0.6, seed: int = 0):
        self.pool, self.p, self.rng = babble_pool, p, np.random.default_rng(seed)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        rng = self.rng
        if rng.random() >= self.p:
            return y
        ops = [op for op in self.OPS if op != "babble" or self.pool]
        for op in rng.choice(ops, size=rng.integers(1, 4), replace=False):
            if op == "mp3":
                y = mp3_roundtrip(y, kbps=int(rng.choice([16, 24, 32, 48, 64])))
            elif op == "babble":
                y = add_babble(y, self.pool[rng.integers(len(self.pool))], snr_db=float(rng.uniform(5, 20)))
            elif op == "reverb":
                y = reverb(y, rt60=float(rng.uniform(0.2, 0.8)), rng=rng)
            elif op == "gain":
                y = gain(y, db=float(rng.uniform(-6, 6)))
            elif op == "bandpass":
                y = bandpass(y)
        return _bounded(y)

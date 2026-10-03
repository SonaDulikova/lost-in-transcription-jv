import numpy as np
import pytest

from lit import augment as A

SR = 16000


def tone(seconds: float = 2.0, hz: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y ** 2)))


def test_gain_scales_by_db():
    y = tone()
    assert rms(A.gain(y, 6.0)) == pytest.approx(rms(y) * 10 ** (6 / 20), rel=1e-3)
    assert rms(A.gain(y, -6.0)) == pytest.approx(rms(y) / 10 ** (6 / 20), rel=1e-3)


def test_babble_hits_the_requested_snr():
    y, noise = tone(), tone(hz=1000.0, amp=0.05)
    out = A.add_babble(y, noise, snr_db=10.0)
    added = out - y
    assert 20 * np.log10(rms(y) / rms(added)) == pytest.approx(10.0, abs=0.1)
    assert len(out) == len(y)


def test_babble_tiles_a_short_noise_and_crops_a_long_one():
    y = tone(3.0)
    assert len(A.add_babble(y, tone(0.5), 10.0)) == len(y)
    assert len(A.add_babble(y, tone(9.0), 10.0)) == len(y)


def test_reverb_keeps_length_and_adds_a_tail():
    y = tone(1.0)
    y[SR // 2:] = 0.0  # silence in the second half
    out = A.reverb(y, rt60=0.5, rng=np.random.default_rng(0))
    assert len(out) == len(y)
    assert rms(out[SR // 2: SR // 2 + SR // 10]) > 0.01  # energy spilled into the silence
    assert np.abs(out).max() <= 1.0


def test_bandpass_removes_out_of_band_energy():
    low, mid = tone(hz=100.0), tone(hz=1000.0)
    assert rms(A.bandpass(low)) < 0.2 * rms(low)
    assert rms(A.bandpass(mid)) > 0.7 * rms(mid)


def test_mp3_roundtrip_keeps_length_and_content():
    y = tone(1.5)
    out = A.mp3_roundtrip(y, kbps=24)
    assert abs(len(out) - len(y)) <= SR // 20  # codec padding under 50 ms
    n = min(len(out), len(y))
    corr = np.corrcoef(out[:n], y[:n])[0, 1]
    assert corr > 0.9


def test_acoustic_augment_is_seeded_and_applies_with_probability():
    y = tone(2.0)
    aug = A.AcousticAugment(babble_pool=[tone(hz=800.0, amp=0.1)], p=1.0, seed=3)
    a = aug(y.copy())
    b = A.AcousticAugment(babble_pool=[tone(hz=800.0, amp=0.1)], p=1.0, seed=3)(y.copy())
    assert np.array_equal(a, b) and not np.array_equal(a, y)
    off = A.AcousticAugment(babble_pool=[], p=0.0, seed=3)
    assert off(y.copy()) is not None and np.array_equal(off(y.copy()), y)


def test_acoustic_augment_output_is_bounded():
    y = tone(2.0, amp=0.9)
    aug = A.AcousticAugment(babble_pool=[tone(hz=800.0, amp=0.5)], p=1.0, seed=0)
    for _ in range(10):
        out = aug(y.copy())
        assert out.dtype == np.float32 and np.abs(out).max() <= 1.0 and len(out) > 0

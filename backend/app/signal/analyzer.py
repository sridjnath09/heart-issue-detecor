"""Signal-quality and rhythm metrics for a heart sound recording."""
import numpy as np
from scipy.signal import find_peaks

from . import features as F
from .preprocessing import bandpass_filter, envelope

TARGET_SR = 2000  # heart sounds live below ~500 Hz; keeps the analysis cheap


def _clean(o):
    """Replace NaN/inf (e.g. SD2 from near-constant intervals) so the result is JSON-safe."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def cycle_period(env: np.ndarray, sr: int, lo_bpm: float = 35, hi_bpm: float = 170):
    """Length of one full heart cycle (S1 to next S1) in samples, from envelope autocorrelation.

    Returns (period_samples, strength 0-1) or (None, 0). Picks the shortest strong lag so a
    double-length lag (two cycles) is not chosen by mistake.
    """
    e = env - np.mean(env)
    if np.allclose(e, 0):
        return None, 0.0
    n = len(e)
    f = np.fft.rfft(e, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(sr * 60 / hi_bpm), min(n - 1, int(sr * 60 / lo_bpm))
    if hi <= lo + 2:
        return None, 0.0
    cand, _ = find_peaks(ac[lo:hi])
    if len(cand) == 0:
        return None, 0.0
    cand = cand + lo
    best = ac[cand].max()
    lag = int(next(c for c in cand if ac[c] >= 0.8 * best))
    return lag, float(ac[lag])


def cycle_starts(peaks: np.ndarray, period: int | None) -> np.ndarray:
    """One peak per cycle (S1). Diastole is longer than systole, so S1 follows the longer gap."""
    if len(peaks) < 3:
        return peaks
    gaps = np.diff(peaks)
    sel = [peaks[0] if gaps[0] <= np.median(gaps) else None]
    for i in range(1, len(peaks) - 1):
        if gaps[i - 1] > gaps[i]:
            sel.append(peaks[i])
    sel = np.array([x for x in sel if x is not None], dtype=int)
    # If the sounds were not paired (single peak per cycle), keep every peak.
    if period and len(sel) >= 2 and abs(np.median(np.diff(sel)) / period - 1) > 0.3:
        alt = abs(np.median(gaps) / period - 1)
        return peaks if alt < 0.3 else sel
    return sel if len(sel) >= 2 else peaks


def analyze_signal(y: np.ndarray, sr: int) -> dict:
    """Return JSON-safe metrics plus downsampled waveform/envelope for plotting."""
    y = np.asarray(y, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    y = y / (np.max(np.abs(y)) + 1e-9)
    filt = bandpass_filter(y, sr, 25.0, 400.0)
    env = envelope(filt, sr)
    peaks, _ = F.detect_peaks_from_envelope(env, sr)
    period, strength = cycle_period(env, sr)
    cycles = cycle_starts(peaks, period)
    intervals = F.compute_intervals(cycles, sr)

    metrics = {
        "duration_s": round(len(y) / sr, 2),
        "n_beats_detected": int(len(cycles)),
        "n_sounds_detected": int(len(peaks)),
        "heart_rate_bpm": round(60 * sr / period, 1) if period and strength >= 0.2 else None,
        "rhythm_regularity": round(strength, 2),
        "snr_db": F.estimate_snr(filt, peaks, sr),
        "hrv": F.compute_hrv_metrics(intervals),
        "energy": F.energy_distribution(filt, sr),
        "murmur_band_energy": F.frequency_band_energy(filt, sr, (150, 400)),
        "s1_s2": F.s1_s2_amplitude_ratio(env, peaks),
        "extra_peaks": F.detect_extra_peaks_per_cycle(peaks, sr, intervals),
        "rhythm": F.irregular_spacing_stats(intervals),
    }
    snr = metrics["snr_db"]
    metrics["quality"] = (
        "poor" if len(peaks) < 3 or snr is None or snr < 3
        else "fair" if snr < 8 else "good"
    )
    step = max(1, len(filt) // 1500)
    metrics["waveform"] = [round(float(v), 4) for v in filt[::step]]
    metrics["envelope"] = [round(float(v), 4) for v in env[::step]]
    metrics["peak_idx"] = [int(p // step) for p in peaks]
    metrics["cycle_idx"] = [int(p // step) for p in cycles]
    return _clean(metrics)

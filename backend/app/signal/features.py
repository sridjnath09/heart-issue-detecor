# signal_processing/features.py
import numpy as np
import scipy.signal as signal
from typing import Dict, Any


# ---------- Peak Detection ----------
def detect_peaks_from_envelope(env: np.ndarray, fs: int,
                            min_bpm: float = 30, max_bpm: float = 220,
                            prominence_factor: float = 0.6):
    min_interval_s = 60.0 / max_bpm
    min_distance = int(min_interval_s * fs * 0.8)
    height = np.percentile(env, 50) + (np.max(env) - np.percentile(env, 50)) * 0.3
    peaks, props = signal.find_peaks(
        env,
        distance=min_distance,
        height=height,
        prominence=prominence_factor * np.std(env)
    )
    return peaks, props


# ---------- Interval / HRV ----------
def compute_intervals(peaks: np.ndarray, fs: int) -> np.ndarray:
    if len(peaks) < 2:
        return np.array([])
    return np.diff(peaks) / fs


def compute_hrv_metrics(intervals_s: np.ndarray) -> Dict[str, Any]:
    out = {}
    if intervals_s.size == 0:
        return out
    ibi_ms = intervals_s * 1000.0
    out["mean_IBI_ms"] = float(np.mean(ibi_ms))
    out["median_IBI_ms"] = float(np.median(ibi_ms))
    out["bpm_mean"] = float(60000.0 / np.mean(ibi_ms))
    out["SDNN_ms"] = float(np.std(ibi_ms, ddof=1))
    diff_ms = np.diff(ibi_ms)
    out["RMSSD_ms"] = float(np.sqrt(np.mean(diff_ms ** 2))) if diff_ms.size > 0 else None
    out["pNN50_pct"] = float(np.sum(np.abs(diff_ms) > 50.0) / diff_ms.size * 100.0) if diff_ms.size > 0 else None
    out["CV"] = float(out["SDNN_ms"] / out["mean_IBI_ms"]) if out["mean_IBI_ms"] > 0 else None
    if diff_ms.size > 0:
        sd1 = np.sqrt(0.5 * np.var(diff_ms, ddof=1))
        sd2 = np.sqrt(2 * np.var(ibi_ms, ddof=1) - 0.5 * np.var(diff_ms, ddof=1))
        out["SD1_ms"] = float(sd1)
        out["SD2_ms"] = float(sd2)
    return out


# ---------- Signal Quality ----------
def estimate_snr(signal_data: np.ndarray, peaks: np.ndarray, fs: int, window_ms: int = 100) -> float:
    w = max(1, int(fs * window_ms / 1000.0))
    sig_powers = []
    mask = np.ones_like(signal_data, dtype=bool)
    for p in peaks:
        start = max(0, p - w)
        end = min(len(signal_data), p + w)
        mask[start:end] = False
        sig_powers.append(np.mean(signal_data[start:end] ** 2))
    if not sig_powers:
        return None
    noise_power = np.mean(signal_data[mask] ** 2) if np.any(mask) else 1e-12
    signal_power = np.mean(sig_powers)
    return float(10.0 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12)))


def energy_distribution(signal_data: np.ndarray, fs: int, cutoff_hz: float = 200.0):
    f, Pxx = signal.welch(signal_data, fs=fs, nperseg=min(4096, len(signal_data)))
    total_energy = np.trapezoid(Pxx, f)
    below_ix = f <= cutoff_hz
    energy_below = np.trapezoid(Pxx[below_ix], f[below_ix])
    pct_below = float(100.0 * energy_below / (total_energy + 1e-12))
    centroid = float(np.sum(f * Pxx) / (np.sum(Pxx) + 1e-12))
    return {"pct_energy_below_{}Hz".format(int(cutoff_hz)): pct_below,
            "spectral_centroid_hz": centroid}


def s1_s2_amplitude_ratio(env: np.ndarray, peaks: np.ndarray):
    if len(peaks) < 2:
        return None
    amps = env[peaks]
    s1 = amps[::2]
    s2 = amps[1::2]
    if np.mean(s2) == 0:
        return None
    return {"S1_mean": float(np.mean(s1)),
            "S2_mean": float(np.mean(s2)),
            "S1_to_S2_ratio": float(np.mean(s1) / np.mean(s2))}


# ---------- Abnormalities ----------
def detect_extra_peaks_per_cycle(peaks: np.ndarray, fs: int, intervals_s: np.ndarray) -> Dict[str, Any]:
    res = {"cycles_checked": 0, "cycles_with_extra_peaks": 0}
    if len(peaks) < 2 or intervals_s.size == 0:
        return res
    mean_cycle = np.mean(intervals_s)
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end_time = (peaks[i] / fs) + mean_cycle
        end = int(round(end_time * fs))
        in_cycle = peaks[(peaks >= start) & (peaks <= end)]
        if len(in_cycle) > 2:
            res["cycles_with_extra_peaks"] += 1
        res["cycles_checked"] += 1
    return res


def irregular_spacing_stats(intervals_s: np.ndarray) -> Dict[str, Any]:
    out = {}
    if intervals_s.size == 0:
        return out
    out["intervals_mean_s"] = float(np.mean(intervals_s))
    out["intervals_std_s"] = float(np.std(intervals_s, ddof=1))
    out["intervals_cv"] = out["intervals_std_s"] / out["intervals_mean_s"]
    q1, q3 = np.percentile(intervals_s, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = intervals_s[(intervals_s < lower) | (intervals_s > upper)]
    out["n_outliers"] = int(len(outliers))
    return out


def frequency_band_energy(signal_data: np.ndarray, fs: int, band=(150, 500)):
    f, Pxx = signal.welch(signal_data, fs=fs, nperseg=min(4096, len(signal_data)))
    band_ix = (f >= band[0]) & (f <= band[1])
    band_energy = np.trapezoid(Pxx[band_ix], f[band_ix]) if np.any(band_ix) else 0.0
    total_energy = np.trapezoid(Pxx, f)
    return {"band_hz": band,
            "band_energy_pct": float(100.0 * band_energy / (total_energy + 1e-12))}

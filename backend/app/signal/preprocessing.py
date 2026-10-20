# signal_processing/preprocessing.py
import numpy as np
import scipy.signal as signal


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(x: np.ndarray, fs: int, lowcut=20.0, highcut=500.0, order=4):
    """
    Apply a bandpass filter to keep only heart sound frequencies.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, x)
    return y


def envelope(x: np.ndarray, fs: int, win_ms: float = 50.0):
    """
    Compute smoothed envelope of signal using moving RMS.
    """
    win = int(fs * win_ms / 1000.0)
    win = max(1, win)
    sq = x ** 2
    kernel = np.ones(win) / win
    env = np.sqrt(np.convolve(sq, kernel, mode="same"))
    return env

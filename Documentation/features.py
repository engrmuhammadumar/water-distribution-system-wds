import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def zero_crossing_rate(x):
    x = np.asarray(x)
    return np.mean(np.diff(np.sign(x)) != 0)


def extract_frequency_features(segment, fs=12000):
    """
    Extract simple frequency-domain features from one signal segment.
    """
    x = np.asarray(segment)
    n = len(x)

    # FFT (real-valued frequency spectrum)
    fft_vals = np.fft.rfft(x)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n, d=1/fs)

    # Avoid divide-by-zero
    mag_sum = np.sum(fft_mag) + 1e-12

    dominant_freq = freqs[np.argmax(fft_mag)]
    spectral_centroid = np.sum(freqs * fft_mag) / mag_sum
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_mag) / mag_sum)
    spectral_rms = np.sqrt(np.mean(fft_mag ** 2))

    total_energy = np.sum(fft_mag ** 2) + 1e-12

    # Energy in bands
    band_0_1k = np.sum(fft_mag[(freqs >= 0) & (freqs < 1000)] ** 2) / total_energy
    band_1k_3k = np.sum(fft_mag[(freqs >= 1000) & (freqs < 3000)] ** 2) / total_energy
    band_3k_6k = np.sum(fft_mag[(freqs >= 3000) & (freqs <= 6000)] ** 2) / total_energy

    return {
        "dominant_freq": dominant_freq,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rms": spectral_rms,
        "band_energy_0_1k": band_0_1k,
        "band_energy_1k_3k": band_1k_3k,
        "band_energy_3k_6k": band_3k_6k,
    }


def extract_time_features(segment):
    """
    Extract basic time-domain features from one signal segment.
    """
    x = np.asarray(segment)

    mean_val = np.mean(x)
    std_val = np.std(x)
    rms_val = rms(x)
    peak_val = np.max(np.abs(x))
    ptp_val = np.ptp(x)
    skew_val = skew(x)
    kurt_val = kurtosis(x)

    abs_mean = np.mean(np.abs(x)) + 1e-12
    sqrt_abs_mean_sq = (np.mean(np.sqrt(np.abs(x))) ** 2) + 1e-12

    crest_factor = peak_val / (rms_val + 1e-12)
    shape_factor = rms_val / abs_mean
    impulse_factor = peak_val / abs_mean
    clearance_factor = peak_val / sqrt_abs_mean_sq

    return {
        "mean": mean_val,
        "std": std_val,
        "rms": rms_val,
        "peak": peak_val,
        "peak_to_peak": ptp_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "crest_factor": crest_factor,
        "shape_factor": shape_factor,
        "impulse_factor": impulse_factor,
        "clearance_factor": clearance_factor,
        "zero_crossing_rate": zero_crossing_rate(x),
    }


def extract_all_features(segment, fs=12000):
    """
    Combine time-domain and frequency-domain features.
    """
    feats = {}
    feats.update(extract_time_features(segment))
    feats.update(extract_frequency_features(segment, fs=fs))
    return feats


def extract_feature_matrix(segments, fs=12000):
    """
    Convert many signal segments into a DataFrame of features.
    """
    features = [extract_all_features(seg, fs=fs) for seg in segments]
    return pd.DataFrame(features)
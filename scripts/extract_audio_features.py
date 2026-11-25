"""Time-frequency, periodicity and speech-level measurements."""
from functools import lru_cache

import numpy as np
from scipy.fft import dct
from scipy.signal import stft


@lru_cache(maxsize=8)
def mel_matrix(rate, n_fft):
    hz = np.fft.rfftfreq(n_fft, 1/rate)
    centers = 700 * (10**(np.linspace(0, 2595*np.log10(1+(rate/2)/700), 28)/2595)-1)
    return np.maximum(0, np.minimum((hz[None,:]-centers[:-2,None])/(centers[1:-1,None]-centers[:-2,None]),
                                    (centers[2:,None]-hz[None,:])/(centers[2:,None]-centers[1:-1,None])))


def spectral_features(values, rate):
    signal = np.mean(values, axis=1)
    if len(signal) < 1024:
        signal = np.pad(signal, (0, 1024-len(signal)))
    n_fft = 1024 if rate >= 32000 else 512
    frequencies, times, z = stft(signal, fs=rate, nperseg=n_fft, noverlap=n_fft//2, boundary=None)
    amplitude = np.abs(z).astype(np.float64)
    power = amplitude**2+1e-15
    total = power.sum(axis=0)
    flatness = np.exp(np.log(power).mean(axis=0))/power.mean(axis=0)
    centroid = (frequencies[:,None]*power).sum(axis=0)/total
    high = power[frequencies >= min(6000, 0.32*rate)].sum(axis=0)/total
    normalized = amplitude / np.maximum(1e-9, amplitude.sum(axis=0))
    flux = np.r_[0, np.maximum(0, np.diff(normalized, axis=1)).sum(axis=0)]
    log_mel = np.log(np.maximum(mel_matrix(rate, n_fft) @ power, 1e-12))
    mfcc = dct(log_mel, axis=0, norm="ortho")[1:9].T
    return {"times": times, "power": power, "frequencies": frequencies, "rms_db": 10*np.log10(total),
            "flatness": flatness, "centroid": centroid, "high_ratio": high,
            "flux": flux, "mfcc": mfcc}


def voice_features(values, rate, compute_praat=True):
    signal = np.mean(values, axis=1)
    spectrum = spectral_features(values, rate)
    if not len(signal):
        raise ValueError("Empty audio window.")
    envelope = spectrum["rms_db"]
    speech = envelope > max(-55, float(np.quantile(envelope, 0.75))-27)
    used = speech if speech.any() else np.ones(len(speech), dtype=bool)
    result = {
        "rms_dbfs": float(20*np.log10(max(1e-8,np.sqrt(np.mean(signal**2))))),
        "peak_dbfs": float(20*np.log10(max(1e-8,np.max(np.abs(values))))),
        "dynamic_range_db": float(np.quantile(envelope[used], .9)-np.quantile(envelope[used], .1)),
        "centroid": float(np.median(spectrum["centroid"][used])),
        "flatness": float(np.median(spectrum["flatness"][used])),
        "flux": float(np.median(spectrum["flux"][used])),
        "zcr": float(np.mean(np.diff(np.signbit(signal)))) if len(signal) > 1 else 0.0,
        "speech_fraction_energy_vad": float(np.mean(speech)),
        "mfcc": [float(value) for value in np.median(spectrum["mfcc"][used], axis=0)],
    }
    result["crest_db"] = result["peak_dbfs"]-result["rms_dbfs"]
    if not compute_praat or len(signal)/rate < 0.12:
        return result, spectrum
    import parselmouth

    sound = parselmouth.Sound(signal, sampling_frequency=rate)
    pitch = sound.to_pitch(time_step=.02, pitch_floor=70, pitch_ceiling=550)
    f0 = pitch.selected_array["frequency"]
    valid = f0[f0 > 0]
    hnr = sound.to_harmonicity_cc(time_step=.04).values.ravel()
    hnr = hnr[hnr > -100]
    result.update({"voiced_fraction": float(np.count_nonzero(f0)/len(f0)),
                   "pitch_median": float(np.median(valid)) if valid.size else None,
                   "pitch_spread_hz": float(np.quantile(valid,.9)-np.quantile(valid,.1)) if valid.size else None,
                   "hnr_db": float(np.median(hnr)) if hnr.size else None})
    return result, spectrum

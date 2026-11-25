"""Measure reproducible, uncalibrated acoustic fingerprints on decoded audio."""
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.signal import correlate, find_peaks

from scripts.extract_audio_features import spectral_features
from scripts.segment_audio import positive_runs


NOVELTY_SETTINGS = {"window_s": 0.72, "minimum_audible_db": -51, "minimum_spacing_s": 0.55,
                    "reference_exclusion_s": 3, "reference_window_s": 10}


def novelty(values, rate):
    spec = spectral_features(values, rate)
    if len(spec["times"]) < 9:
        return spec, np.zeros(len(spec["times"])), {}
    hop = spec["times"][1] - spec["times"][0]
    activity = spec["rms_db"] > NOVELTY_SETTINGS["minimum_audible_db"]
    brightness = np.log1p(100 * spec["high_ratio"])
    level = np.abs(np.diff(spec["rms_db"], prepend=spec["rms_db"][0]))
    timbre = np.linalg.norm(np.diff(spec["mfcc"], axis=0, prepend=spec["mfcc"][[0]]), axis=1)
    signals = np.stack([level, spec["flux"], brightness, timbre])
    # Compare with a local neighborhood while limiting each cue's leverage.
    noise = gaussian_filter1d(signals, sigma=max(2, round(NOVELTY_SETTINGS["window_s"] / hop)), axis=1)
    deviations = np.maximum(0, signals - noise)
    normalizer = np.array([5.0, .16, .85, 4.0])[:, None]
    standardized = np.minimum(deviations / normalizer, 8)
    score = gaussian_filter1d(np.sum(standardized, axis=0), 1.1)
    score[~maximum_filter1d(activity, size=3)] = 0
    return spec, score, {name: value for name, value in zip(("level", "flux", "brightness", "mfcc_change"), signals)}


def local_peaks(values, rate, origin):
    spectrum, scores, traces = novelty(values, rate)
    if len(scores) < 9:
        return []
    times = spectrum["times"]
    hop = times[1] - times[0]
    peaks, _ = find_peaks(scores, distance=max(1, round(NOVELTY_SETTINGS["minimum_spacing_s"] / hop)))
    events = []
    for i in peaks:
        if scores[i] < 1:
            continue
        left, right = i, i
        while left > max(0, i-12) and scores[left-1] > .65 * scores[i]:
            left -= 1
        while right < min(i+12, len(scores)-1) and scores[right+1] > .65 * scores[i]:
            right += 1
        events.append({
            "start": max(origin, origin + float(times[left] - hop)),
            "end": min(origin + len(values)/rate, origin + float(times[right] + hop)),
            "peak_s": origin + float(times[i]), "score": float(scores[i]),
            "cues_at_peak": {name: float(trace[i]) for name, trace in traces.items()},
        })
    return events


def nearest_activity(values, rate, origin, point, max_offset=2.5):
    peaks = [row for row in local_peaks(values, rate, origin) if abs(row["peak_s"] - point) <= max_offset]
    if not peaks:
        return point - .18, point + .18, []
    peaks.sort(key=lambda row: (-row["score"], abs(row["peak_s"] - point)))
    first = peaks[0]
    return first["start"], first["end"], peaks


def acoustic_curves(values, rate):
    import librosa
    import parselmouth

    mono = values.mean(axis=1)
    spec = spectral_features(values, rate)
    n_fft = (len(spec["frequencies"]) - 1) * 2
    mel = librosa.filters.mel(sr=rate, n_fft=n_fft, n_mels=48, fmax=min(10000, rate/2), norm=None)
    energy_t = np.arange(0, len(mono), max(1, rate//100))
    energy = np.array([20 * np.log10(max(1e-8, np.sqrt(np.mean(mono[i:i+rate//100]**2)))) for i in energy_t])
    sound = parselmouth.Sound(mono, sampling_frequency=rate)
    pitch = sound.to_pitch(time_step=.01, pitch_floor=70, pitch_ceiling=550)
    f0 = pitch.selected_array["frequency"]
    strengths = pitch.selected_array["strength"]
    hnr_track = sound.to_harmonicity_cc(time_step=.02, minimum_pitch=70)
    hnr = hnr_track.values.ravel()
    hnr[hnr < -100] = np.nan
    formants = sound.to_formant_burg(time_step=.01, max_number_of_formants=5, maximum_formant=5500)
    ft = formants.xs()
    f = np.array([[formants.get_value_at_time(n, t) for t in ft] for n in (1, 2, 3)])
    bands = [[0, 500], [500, 1500], [1500, 3500], [3500, 6000], [6000, 10000], [10000, rate/2]]
    by_band = np.array([spec["power"][(spec["frequencies"] >= lo) & (spec["frequencies"] < hi)].sum(axis=0)
                        for lo, hi in bands])
    band_ratio = by_band / spec["power"].sum(axis=0)
    return {**spec, "mel_db": 10*np.log10(np.maximum(mel @ spec["power"], 1e-15)),
            "f0_times": pitch.xs(), "f0": np.where(f0 > 0, f0, np.nan), "periodicity": strengths,
            "hnr_times": hnr_track.xs(), "hnr": hnr, "formant_times": ft, "formants": f,
            "energy_times": energy_t/rate, "energy_dbfs": energy,
            "band_edges_hz": np.array(bands), "band_ratio": band_ratio}


def finite_median(values):
    usable = np.asarray(values)
    usable = usable[np.isfinite(usable)]
    return float(np.median(usable)) if len(usable) else None


def fingerprint(values, rate, curves, start, end):
    section = values[max(0, round(start*rate)):min(len(values), round(end*rate))]
    if len(section) == 0:
        raise ValueError("Empty fingerprint interval.")
    select = (curves["times"] >= start) & (curves["times"] <= end)
    if not np.any(select):
        select[np.argmin(abs(curves["times"] - (start + end)/2))] = True

    def under(times):
        return (times >= start) & (times < end)

    pitch = curves["f0"][under(curves["f0_times"])]
    valid = np.isfinite(pitch)
    pitch_delta = np.abs(np.diff(12 * np.log2(pitch)))
    pitch_delta = pitch_delta[np.isfinite(pitch_delta)]
    energy = curves["energy_dbfs"][under(curves["energy_times"])]
    low_runs = [round((b-a)*.01, 3) for a, b in positive_runs(energy < -55)]
    mono = section.mean(axis=1)
    lag_low, lag_high = round(.04 * rate), min(round(.30 * rate), len(mono)//2)
    repeat = None
    if lag_high > lag_low and np.sqrt(np.mean(mono**2)) > .004:
        centered = mono - mono.mean()
        autocorrelation = correlate(centered, centered, mode="full", method="fft")[len(mono)-1:]
        lag = lag_low + int(np.argmax(autocorrelation[lag_low:lag_high]))
        repeat = {"lag_s": lag / rate, "normalized_autocorrelation": float(autocorrelation[lag] / autocorrelation[0])}
    return {
        "duration_s": len(section)/rate, "peak_abs": float(np.max(np.abs(section))),
        "peak_step_abs": float(np.max(np.abs(np.diff(section, axis=0)))) if len(section)>1 else 0,
        "rms_dbfs": float(20*np.log10(max(1e-8, np.sqrt(np.mean(section**2))))),
        "stereo_side_rms_dbfs": (float(20*np.log10(max(1e-8, np.sqrt(np.mean(((section[:,0]-section[:,1])/2)**2)))))
                                 if section.shape[1] == 2 else None),
        "centroid_hz": finite_median(curves["centroid"][select]),
        "flatness": finite_median(curves["flatness"][select]),
        "flux_peak": float(np.max(curves["flux"][select])),
        "higher_band_ratio_peak": float(np.max(curves["high_ratio"][select])),
        "band_median_ratios": np.median(curves["band_ratio"][:, select], axis=1).tolist(),
        "mfcc_median": np.median(curves["mfcc"][select], axis=0).tolist(),
        "f0_hz": finite_median(pitch), "voiced_fraction": float(np.mean(valid)) if len(valid) else None,
        "adjacent_f0_step_max_st": float(np.max(pitch_delta)) if len(pitch_delta) else None,
        "formants_hz": [finite_median(row[under(curves["formant_times"])] ) for row in curves["formants"]],
        "hnr_db": finite_median(curves["hnr"][under(curves["hnr_times"])]),
        "low_energy_longest_s": max(low_runs, default=0.), "repetition_proxy": repeat,
    }


def plots(directory, values, rate, curves, start, end, origin):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9, 7.6), layout="constrained")
    axes = fig.subplots(5, 1, sharex=True)
    timeline = origin + np.arange(len(values))/rate
    step = max(1, len(timeline)//11000)
    axes[0].plot(timeline[::step], values[::step].mean(axis=1), color="#145d61", lw=.55)
    spectrum_db = 10*np.log10(curves["power"])
    top = float(np.max(spectrum_db))
    axes[1].pcolormesh(origin+curves["times"], curves["frequencies"], spectrum_db, shading="auto", cmap="magma",
                       vmin=top-66, vmax=top)
    axes[1].set_ylim(0, min(11000, rate/2))
    mel_top = float(np.max(curves["mel_db"]))
    axes[2].pcolormesh(origin+curves["times"], np.arange(curves["mel_db"].shape[0]), curves["mel_db"],
                       shading="auto", cmap="magma", vmin=mel_top-65, vmax=mel_top)
    axes[3].plot(origin+curves["f0_times"], curves["f0"], ".", color="#145d61", markersize=2)
    axes[4].plot(origin+curves["energy_times"], curves["energy_dbfs"], color="#145d61", lw=.85)
    for axis, label in zip(axes, ("Waveform", "Spectrogram · Hz", "Mel band", "Praat F0 · Hz", "Energy · dBFS")):
        axis.set_ylabel(label, fontsize=9)
        axis.axvspan(start, end, color="#23856b", alpha=.27)
    axes[-1].set_xlim(timeline[0], timeline[-1])
    axes[-1].set_xlabel("Seconds from start of decoded recording")
    fig.savefig(Path(directory)/"acoustic.png", dpi=130)

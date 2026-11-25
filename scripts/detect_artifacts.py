"""Run signal detectors on speech and non-speech alike."""
from collections import defaultdict

import numpy as np

from .detect_clicks import clicks
from .detect_clipping import clipping
from .detect_dropouts import dropouts
from .extract_audio_features import voice_features
from .segment_audio import positive_runs


def time_events(raw, origin, rate):
    return [{"category": category, "start": (origin+int(start))/rate, "end": (origin+int(end))/rate,
             "detector": "digital_signal", "evidence": evidence, "timing": "decoded_sample_clock"}
            for category, start, end, evidence in raw]


def spectral_events(spectrum, rate, maximum, sensitivity):
    power = spectrum["rms_db"]
    times = spectrum["times"]
    if len(times) < 4:
        return []
    quiet = np.median(power)
    high = spectrum["high_ratio"]
    flat = spectrum["flatness"]
    flux = spectrum["flux"]
    cutoff = .28 if sensitivity == "recall" else .47 if sensitivity == "precision" else .37
    evidence = (high > cutoff) & (flat > 0.16) & (power > max(-42, quiet + 10))
    candidates = []
    for lo, hi in positive_runs(evidence):
        count = hi-lo
        if not 2 <= count <= 12:
            continue
        a, b = max(0, lo-4), min(len(high), hi+4)
        context = np.r_[high[a:lo],high[hi:b]]
        if not len(context) or float(np.median(context)) >= cutoff*0.55:
            continue
        start = round(times[lo]*rate)
        end = min(maximum, round((times[hi-1]+(times[1]-times[0]))*rate))
        if end > start:
            candidates.append(("high_frequency_burst", start, end, {
                "band_power_ratio": float(np.max(high[lo:hi])), "spectral_flatness": float(np.max(flat[lo:hi])),
                "spectral_flux": float(np.max(flux[lo:hi])), "possible_natural_fricative": True,
            }))
    return candidates


def scan_signal(values, rate, origin=0, sensitivity="balanced", deep=False):
    if len(values) < 8:
        return [], {}
    rail = clipping(values, rate)
    impulsive = clicks(values, rate, sensitivity)
    digital_dropouts = dropouts(values, rate)
    blockers = {}
    for _, start, end, evidence in rail:
        blockers.setdefault(evidence["channel"], []).append((start, end))
    impulsive = [row for row in impulsive if not any(
        row[1] - 3 <= end and row[2] + 3 >= start
        for start, end in blockers.get(row[3]["channel"], ())
    ) and not any(row[1] - 3 <= end and row[2] + 3 >= start for _, start, end, _ in digital_dropouts)]
    features, spectrum = voice_features(values, rate, compute_praat=deep)
    spectral = spectral_events(spectrum, rate, len(values), sensitivity)
    raw = rail + impulsive + digital_dropouts + spectral
    return time_events(raw, origin, rate), features

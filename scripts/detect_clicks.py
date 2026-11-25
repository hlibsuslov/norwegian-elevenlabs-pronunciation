"""Detect impulsive steps and short peaks against nearby channel energy."""
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from .segment_audio import positive_runs


def clicks(samples, rate, sensitivity):
    events = []
    window = max(11, int(rate*0.014) | 1)
    excursion = 0.13 if sensitivity == "recall" else 0.22 if sensitivity == "precision" else 0.17
    for channel, signal in enumerate(samples.T):
        step = np.abs(np.diff(signal, prepend=signal[0]))
        rms = np.sqrt(np.maximum(1e-9, uniform_filter1d(step**2, size=window)))
        spikes = (step > excursion) & (step > rms * (5 if sensitivity == "recall" else 7))
        merged = maximum_filter1d(spikes, size=max(3, (rate // 700) | 1))
        for lo, hi in positive_runs(merged):
            support = np.flatnonzero(spikes[lo:hi]) + lo
            if not len(support):
                continue
            first, last = max(0, support[0]-2), min(len(signal), support[-1]+3)
            if (last-first)/rate > 0.010:
                continue
            peak = int(support[np.argmax(step[support])])
            events.append(("click_or_pop", first, last, {"channel": channel, "step": float(step[peak]),
                           "local_step_ratio": float(step[peak]/rms[peak]), "samples": int(last-first)}))
    return events

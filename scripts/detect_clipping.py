"""Locate repeated same-polarity samples near the digital rails."""
import numpy as np

from .segment_audio import positive_runs


def clipping(samples, rate):
    events = []
    for channel, signal in enumerate(samples.T):
        for polarity in (1, -1):
            levels = polarity * signal
            for start, end in positive_runs(levels >= 0.995):
                if end-start >= 3 and np.ptp(levels[start:end]) <= 0.007:
                    events.append(("clipping", start, end, {
                        "channel": channel, "peak": float(np.max(levels[start:end])), "rail_samples": int(end-start),
                    }))
    return events

"""Splice regenerated audio at checked low-energy boundaries with linear fades."""
import argparse
import io
import wave
from pathlib import Path

import numpy as np

from .analyze_audio import sound_file
from .common import atomic_bytes, config


def dbfs(values):
    if not values.size:
        return -240.0
    return float(20 * np.log10(max(1e-12, np.sqrt(np.mean(values ** 2)))))


def quiet_cut(values, rate, first, last, window):
    low = max(window, round(first * rate))
    high = min(round(last * rate), values.shape[1] - window)
    if first == 0 and last == 0:
        return 0
    if first == values.shape[1] / rate and last == first:
        return values.shape[1]
    if high < low:
        raise ValueError("There is insufficient aligned silence for a crossfade.")
    # A short RMS window rejects a zero crossing inside voiced speech.
    candidates = sorted(set(range(low, high + 1, max(1, rate // 1000))) | {high})
    at = min(candidates, key=lambda i: dbfs(values[:, i-window:i+window]))
    if dbfs(values[:, at-window:at+window]) > config("quality_thresholds")["quiet_boundary_dbfs"]:
        raise ValueError("No low-energy splice point in the aligned gap.")
    return at


def wav_bytes(values, rate):
    clipped = np.rint(values.T * 32768).clip(-32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setparams((values.shape[0], 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(clipped.tobytes())
    return buffer.getvalue()


def splice(original, replacement, destination, left, right):
    source, insert = sound_file(original), sound_file(replacement)
    rate = int(source.sampling_frequency)
    if insert.n_channels != source.n_channels or insert.sampling_frequency != source.sampling_frequency:
        raise ValueError("Splicing requires matching sample rate and channel count.")
    limits = config("quality_thresholds")
    samples = source.values
    if not (0 <= left[0] <= left[1] < right[0] <= right[1] <= samples.shape[1] / rate + 0.000001):
        raise ValueError("Splice bounds must be ordered, disjoint, and inside the source.")
    fade = max(1, round(rate * limits["crossfade_seconds"]))
    start = quiet_cut(samples, rate, *left, fade)
    end = quiet_cut(samples, rate, *right, fade)
    candidate = insert.values
    ratio = candidate.shape[1] / max(1, end - start)
    max_ratio = limits["maximum_segment_duration_ratio"]
    if not 1 / max_ratio <= ratio <= max_ratio:
        raise ValueError("Replacement duration changed too much for an automatic splice.")
    fade_in = min(fade, start)
    fade_out = min(fade, samples.shape[1] - end)
    if candidate.shape[1] <= fade_in + fade_out:
        raise ValueError("Replacement is shorter than the crossfades.")
    parts = [samples[:, :start-fade_in]]
    if fade_in:
        if dbfs(candidate[:, :fade_in]) > limits["quiet_boundary_dbfs"]:
            raise ValueError("Replacement starts in speech without a quiet margin.")
        ramp = np.linspace(0, 1, fade_in)
        parts.append(samples[:, start-fade_in:start] * (1-ramp) + candidate[:, :fade_in] * ramp)
    parts.append(candidate[:, fade_in:candidate.shape[1]-fade_out])
    if fade_out:
        if dbfs(candidate[:, -fade_out:]) > limits["quiet_boundary_dbfs"]:
            raise ValueError("Replacement ends in speech without a quiet margin.")
        ramp = np.linspace(0, 1, fade_out)
        parts.append(candidate[:, -fade_out:] * (1-ramp) + samples[:, end:end+fade_out] * ramp)
    parts.append(samples[:, end+fade_out:])
    merged = np.concatenate(parts, axis=1)
    if np.max(np.abs(merged)) > 1.0001:
        raise ValueError("Splice would clip.")
    atomic_bytes(destination, wav_bytes(merged, rate))
    return {"source_start_s": start / rate, "source_end_s": end / rate, "crossfade_seconds": fade / rate}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("replacement", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--left", nargs=2, type=float, required=True)
    parser.add_argument("--right", nargs=2, type=float, required=True)
    args = parser.parse_args()
    print(splice(args.original, args.replacement, args.out, args.left, args.right))

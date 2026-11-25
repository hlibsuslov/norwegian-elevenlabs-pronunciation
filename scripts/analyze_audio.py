"""Measure acoustic features in declared word and phone intervals with Praat."""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import parselmouth

from .alignment import textgrid, word_mapping
from .common import config, file_digest, key, read_json, write_json

SILENCE = {"sil", "sp", "spn", "<unk>"}
SYLLABIC_CONSONANTS = {"LX", "MX", "NX", "RLX", "RNX", "RX", "SX"}


def sound_file(path):
    try:
        return parselmouth.Sound(str(path))
    except parselmouth.PraatError:
        if not shutil.which("ffmpeg"):
            raise ValueError("Audio decoding requires FFmpeg on PATH or a readable WAV.")
        with tempfile.TemporaryDirectory(prefix="nelp-audio-") as temporary:
            wav = Path(temporary) / "decoded.wav"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-c:a",
                            "pcm_s16le", str(wav)], check=True, capture_output=True, timeout=180)
            return parselmouth.Sound(str(wav))


def bark(hz):
    return float(6 * np.arcsinh(hz / 600))


def median_or_none(values):
    finite = [value for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else None


def interval_samples(times, values, begin, end):
    return values[(times >= begin) & (times <= end)]


def timed_word_rows(analysis, alignment):
    # A TextGrid fixes boundaries, not the identity of any actually spoken phone.
    aligned = alignment["words"]
    expected = [token["text"] for token in analysis["tokens"]]
    indices, differences = word_mapping(expected, [word["text"] for word in aligned])
    return {i: aligned[index] for i, index in indices.items()}, differences


def validate_intervals(items, duration):
    previous = 0.0
    for item in items:
        start, end = item["start"], item["end"]
        if not np.isfinite([start, end]).all() or start < previous - 0.002 or end <= start or end > duration + 0.020:
            raise ValueError("Alignment intervals must be ordered, non-overlapping and inside the audio.")
        previous = end


def phone_breakdown(interval, phones, formants, threshold):
    indexed = []
    start, end = interval["start"], interval["end"]
    for phone in phones:
        label = phone["text"].strip().upper()
        if label.lower() in SILENCE:
            continue
        lo, hi = phone["start"], phone["end"]
        if lo < start - 0.004 or hi > end + 0.004:
            continue
        stress = int(label[-1]) if label[-1:] in ("0", "1", "2", "3") else None
        symbol = label[:-1] if stress is not None else label
        row = {"phone": symbol, "start": lo, "end": hi, "stress_code": stress, "duration_s": hi - lo}
        if stress is not None and symbol not in SYLLABIC_CONSONANTS and hi - lo >= threshold:
            for index in (1, 2):
                values = [formants.get_value_at_time(index, time) for time in np.linspace(lo + (hi-lo)*0.30, lo + (hi-lo)*0.70, 5)]
                value = median_or_none(values)
                row[f"f{index}_bark"] = bark(value) if value is not None and value > 0 else None
        indexed.append(row)
    nuclei = [p for p in indexed if p["stress_code"] is not None]
    syllables = []
    if nuclei:
        # NoFA encodes nuclei. The inter-nuclear midpoint is a duration proxy.
        boundaries = [start, *((a["end"] + b["start"]) / 2 for a, b in zip(nuclei, nuclei[1:])), end]
        for i, nucleus in enumerate(nuclei):
            syllables.append({"nucleus": nucleus, "start": boundaries[i], "end": boundaries[i + 1],
                              "duration_s": boundaries[i + 1] - boundaries[i]})
    return indexed, syllables


def segmental_matches(target, measured):
    if not target or not measured:
        return False
    expected = [phone.rstrip("0123") for phone in target["nofabet"].split()]
    return expected == [phone["phone"] for phone in measured]


def pitch_shape(weighted_times, pitch, start, end, median_pitch):
    if not median_pitch:
        return []
    edges = np.linspace(start, end, 6)
    contour = []
    for lo, hi in zip(edges, edges[1:]):
        values = interval_samples(weighted_times, pitch, lo, hi)
        frequency = median_or_none(values[values > 0])
        contour.append(float(12 * np.log2(frequency / median_pitch)) if frequency else None)
    return contour


def alignment_gaps(alignment, threshold):
    if not alignment:
        return []
    return [
        {"type": "long_interword_pause", "start": earlier["end"], "end": later["start"],
         "confidence": "heuristic", "previous_word": earlier["text"], "next_word": later["text"]}
        for earlier, later in zip(alignment["words"], alignment["words"][1:])
        if later["start"] - earlier["end"] >= threshold
    ]


def transcription_findings(analysis, asr):
    if not asr:
        return None
    tokens = analysis["tokens"]
    expected = [t["text"] for t in tokens]
    words = asr["words"]
    _, differences = word_mapping(expected, [word["text"] for word in words])
    issues = []
    count = 0
    for edit in differences:
        a, b = edit["expected"]
        c, d = edit["observed"]
        target, observed = expected[a:b], [word["text"] for word in words[c:d]]
        if any(word[:1].isdecimal() for word in target + observed):
            edit["type"] = "written_spoken_normalization_review"
        else:
            edit["type"] = {"delete": "possible_omission", "insert": "possible_insertion",
                            "replace": "possible_substitution"}[edit["kind"]]
            if observed and not target:
                neighbors = [words[index]["text"] for index in (c - 1, d) if 0 <= index < len(words)]
                if any(key(word) == key(neighbor) for word in observed for neighbor in neighbors):
                    edit["type"] = "possible_repetition"
            count += max(b-a, d-c)
        edit.update({"expected_words": target, "observed_words": observed, "confidence": "asr_candidate"})
        cues = words[c:d]
        if not cues:
            cues = words[max(0, c-1):min(len(words), c+1)]
        if cues:
            edit.update({"start": cues[0]["start"], "end": cues[-1]["end"]})
        issues.append(edit)
    return {"source": asr["source"], "issues": issues, "lexical_edit_count": count,
            "observed_word_count": len(words), "expected_word_count": len(expected)}


def analyze_recording(path, analysis, alignment=None, asr=None):
    thresholds = config("quality_thresholds")
    sound = sound_file(path)
    mono = sound.convert_to_mono() if sound.n_channels > 1 else sound
    pitch = mono.to_pitch(time_step=thresholds["frame_step_seconds"], pitch_floor=thresholds["pitch_floor_hz"],
                          pitch_ceiling=thresholds["pitch_ceiling_hz"])
    frequencies = pitch.selected_array["frequency"]
    center = median_or_none(frequencies[frequencies > 0])
    intensity = mono.to_intensity(minimum_pitch=thresholds["pitch_floor_hz"])
    intensity_median = median_or_none(intensity.values[0])
    timings = {}
    mismatches = []
    if alignment:
        validate_intervals(alignment["words"], sound.duration)
        validate_intervals(alignment["phones"], sound.duration)
        timings, mismatches = timed_word_rows(analysis, alignment)
    formants = mono.to_formant_burg() if alignment and alignment["phones"] else None
    rows = []
    phones = alignment["phones"] if alignment else []
    phone_cursor = 0
    for token in analysis["tokens"]:
        interval = timings.get(token["id"])
        if not interval:
            continue
        start, end = interval["start"], interval["end"]
        while phone_cursor < len(phones) and phones[phone_cursor]["end"] <= start:
            phone_cursor += 1
        stop = phone_cursor
        while stop < len(phones) and phones[stop]["start"] < end:
            stop += 1
        measured, syllables = phone_breakdown(interval, phones[phone_cursor:stop], formants,
                                               thresholds["minimum_phone_seconds"])
        level = median_or_none(interval_samples(intensity.xs(), intensity.values[0], start, end))
        for syllable in syllables:
            lo, hi = syllable["start"], syllable["end"]
            values = interval_samples(pitch.xs(), frequencies, lo, hi)
            syllable["median_f0_hz"] = median_or_none(values[values > 0])
            syllable["median_intensity_db"] = median_or_none(interval_samples(intensity.xs(), intensity.values[0], lo, hi))
        rows.append({
            "id": token["id"], "text": token["text"], "sentence": token["sentence"],
            "start": start, "end": end, "duration_s": end - start,
            "relative_intensity_db": level - intensity_median if level is not None and intensity_median is not None else None,
            "f0_shape_st": pitch_shape(pitch.xs(), frequencies, start, end, center),
            "phones": measured, "syllables": syllables,
            "phones_match_lexical_target": segmental_matches(token["target"], measured),
        })
    by_sentence = {}
    for row in rows:
        by_sentence.setdefault(row["sentence"], []).append(row)
    for sequence in by_sentence.values():
        typical = float(np.median([row["duration_s"] for row in sequence]))
        for row in sequence:
            row["relative_duration_log"] = float(np.log(row["duration_s"] / typical)) if len(sequence) > 1 else None
    return {
        "audio_sha256": file_digest(path), "duration_s": float(sound.duration), "channels": sound.n_channels,
        "peak_dbfs": float(20 * np.log10(max(1e-12, float(np.max(np.abs(sound.values)))))),
        "median_f0_hz": center, "median_intensity_db": intensity_median, "voiced_frame_fraction": float(np.mean(frequencies > 0)),
        "f0_hz": [float(value) if value > 0 else None for value in frequencies],
        "f0_times_s": [float(value) for value in pitch.xs()], "aligned_words": rows,
        "alignment_source": alignment["source"] if alignment else None,
        "alignment_words": alignment["words"] if alignment else [],
        "alignment_edits": mismatches, "pauses": alignment_gaps(alignment, thresholds["pause_seconds"]),
        "asr": transcription_findings(analysis, asr),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("analysis")
    parser.add_argument("--textgrid", type=Path)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_json(args.out, analyze_recording(args.audio, read_json(args.analysis), textgrid(args.textgrid) if args.textgrid else None))

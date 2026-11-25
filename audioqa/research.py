"""Build a recording-level QA dataset from a frozen blind scan and review evidence."""
import argparse
import csv
import shutil
from collections import Counter
from contextlib import closing
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

from scripts.common import atomic_bytes, digest, file_digest, read_json, write_json
from scripts.generate_review_dashboard import dashboard, export_report
from scripts.lookup_pronunciation import Lexicon
from scripts.segment_audio import read_samples, windows
from .research_benchmark import evaluate
from .research_features import NOVELTY_SETTINGS, acoustic_curves, fingerprint, local_peaks, nearest_activity, plots
from .research_models import Yamnet, neighbor_distances


PARTITIONS = (
    "raw_audio", "annotations", "confirmed_artifacts", "confirmed_pronunciation_errors", "language_interference",
    "clean_controls", "hard_negatives", "ambiguous_cases", "repaired_examples", "acoustic_features", "embeddings", "metadata",
)
PRIORITY = {"cross_language_pronunciation": 95, "pronunciation": 90, "artifact": 80, "embedding_outlier": 55}
PARTICLE_PHRASES = ("stå opp", "gi opp", "slå av", "slå på", "holde på", "komme tilbake", "gå ut", "passe på")


def labels(path, duration):
    if not path:
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        parts = [int(value) for value in row["timestamp"].split(":")]
        point = sum(value * 60**index for index, value in enumerate(reversed(parts)))
        if not 0 <= point < duration:
            raise ValueError(f"Approximate point outside source: {row['id']}.")
        row["point_s"] = point
        row["word"] = None if row["word"] in {"", "unverified"} else row["word"]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate manual annotation IDs.")
    return rows


def report_words(paths, sha):
    models, languages = {}, []
    for path in paths:
        report = read_json(path)
        if report["source"]["sha256"] != sha:
            raise ValueError("The ASR report belongs to a different source.")
        models.update(report["reverse_transcript"])
        languages.extend(report["window_languages"])
    return models, languages


def nearby_words(words, start, end):
    return [row for row in words if row["start"] < end and row["end"] > start]


def excerpt(words):
    return " ".join(row["text"] for row in words)


def audio_bounds(proxy, rate, count, start, end):
    if end <= start:
        raise ValueError("End must follow start on the decoded sample clock.")
    lo = max(0, min(count - 1, round(start * rate)))
    hi = max(lo + 1, min(count, round(end * rate)))
    return read_samples(proxy, lo, hi), lo/rate, hi/rate


def speech_span(words, start, end, duration):
    context = nearby_words(words, max(0, start-12), min(end+12, duration))
    blocks = []
    for word in context:
        if not blocks or word["start"] - blocks[-1][-1]["end"] > .50:
            blocks.append([])
        blocks[-1].append(word)
    matching = [block for block in blocks if block[0]["start"] < end + .20 and block[-1]["end"] > start - .20]
    if matching:
        block_start = max(start - 12, matching[0][0]["start"] - .20)
        block_end = min(end + 12, matching[-1][-1]["end"] + .25, duration)
        return min(start, block_start), max(end, block_end), "ASR word-gap block estimate"
    return max(0, start-9), min(duration, end+9), "bounded audio fallback"


def candidate_report(proxy, report, threshold):
    candidates = []
    rate = report["sample_rate"]
    for window in windows(proxy):
        values = read_samples(proxy, window.start, window.end)
        for row in local_peaks(values, rate, window.start/rate):
            if not window.core_start/rate <= row["peak_s"] < window.core_end/rate:
                continue
            if row["score"] < threshold:
                continue
            identity = digest([report["source"]["sha256"], "local_acoustic_novelty_v1", round(row["peak_s"], 3)])[:14]
            candidates.append({**row, "id": identity, "category": "local_acoustic_novelty", "severity": "LOW",
                               "timing": "short_window_estimate", "status": "SUSPECTED",
                               "clip_start": max(0, row["start"]-1.35),
                               "clip_end": min(report["duration_s"], row["end"]+1.35)})
    return candidates


def embedding_scan(proxy, report, model, root, name):
    rate = report["sample_rate"]
    duration = report["duration_s"]
    count = round(duration * rate)
    vectors, starts, nclasses = [], [], []
    hop, length, steps = .48, .975, 64
    for batch in range(0, int(duration / hop) + 1, steps):
        start = batch * hop
        end = min(duration, start + (steps-1)*hop + length)
        if end - start < length:
            break
        values = read_samples(proxy, round(start*rate), min(count, round(end*rate)))
        scores, embeddings = model.frames(values, rate)
        keep = min(steps, len(embeddings), int((duration - start - length)/hop) + 1)
        vectors.extend(embeddings[:keep:2])
        nclasses.extend(scores[:keep:2, 0])
        starts.extend(start + i*hop for i in range(0, keep, 2))
    starts, vectors = np.asarray(starts), np.asarray(vectors)
    distances = neighbor_distances(vectors, starts)
    events = []
    for i in np.argsort(np.nan_to_num(distances, nan=-1))[::-1]:
        if not np.isfinite(distances[i]) or len(events) >= np.ceil(duration / 120):
            break
        if any(abs(starts[i] - event["start"]) < 5 for event in events):
            continue
        start = float(starts[i])
        events.append({
            "id": digest([report["source"]["sha256"], "yamnet_knn_v1", start])[:14], "category": "embedding_outlier",
            "start": start, "end": min(start+length, duration), "timing": "embedding_frame_estimate",
            "score": float(distances[i]), "speech_class_score": float(nclasses[i]), "status": "SUSPECTED",
            "clip_start": max(0, start-1.35), "clip_end": min(duration, start+length+1.35),
        })
    np.savez_compressed(root/"embeddings"/f"{name}-track.npz", start_s=starts, vectors=vectors, distances=distances)
    return {"events": events, "model_sha256": model.sha256, "frames": len(starts),
            "method": "cosine distance to five nearest embeddings outside 30 seconds; at most one event per two minutes on average"}


def make_case(event, report, proxy, words_by_model, language_windows, root, name, lexicon, embedding):
    rate, duration = report["sample_rate"], report["duration_s"]
    count = round(rate*duration)
    start, end = event["start"], event["end"]
    base = root/"review"/"clips"/event["id"]
    result_dir = root/"acoustic_features"/event["id"]
    base.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    primary = next(iter(words_by_model.values()), [])
    extended_start, extended_end, scope = speech_span(primary, start, end, duration)
    local, origin, local_end = audio_bounds(proxy, rate, count, start-5, end+5)
    micro, start, end = audio_bounds(proxy, rate, count, start, end)
    extended, extended_start, extended_end = audio_bounds(proxy, rate, count, extended_start, extended_end)
    clips = {"defect": ("micro.wav", micro), "context": ("local.wav", local), "extended": ("extended.wav", extended)}
    saved = base/"evidence.json"
    data_path = result_dir/"curves.npz"
    recorded = read_json(saved) if saved.exists() else {}
    same_interval = recorded.get("start") == start and recorded.get("end") == end
    for key, (filename, samples) in clips.items():
        if key == "extended" or not same_interval or not (base/filename).exists():
            sf.write(base/filename, samples, rate, subtype="FLOAT")
    if same_interval and data_path.exists() and (base/"acoustic.png").exists():
        with np.load(data_path, allow_pickle=False) as stored:
            curves = {key: stored[key] for key in stored.files}
    else:
        curves = acoustic_curves(local, rate)
        plots(base, local, rate, curves, start, end, origin)
        np.savez_compressed(data_path, origin_s=origin, **curves)
    observed = fingerprint(local, rate, curves, start-origin, end-origin)
    controls = comparable_intervals(local, rate, curves, origin, start, end, observed)
    statistics = {}
    for feature in ("rms_dbfs", "centroid_hz", "flux_peak", "higher_band_ratio_peak", "hnr_db",
                    "f0_hz", "adjacent_f0_step_max_st", "peak_step_abs"):
        usable = [row["features"][feature] for row in controls if row["features"][feature] is not None]
        if usable and observed[feature] is not None:
            statistics[feature] = observed[feature] - float(np.median(usable))
    word = event.get("word")
    targets = lexicon.lookup(word) if word else []
    language_votes = [row for row in language_windows if row.get("start", duration) <= (start+end)/2 < row.get("end", 0)]
    text = {key: excerpt(nearby_words(words, extended_start, extended_end)) for key, words in words_by_model.items()}
    asset_base = f"clips/{event['id']}/"
    assets = {key: asset_base + filename for key, (filename, _) in clips.items()}
    assets.update({"spectrogram": asset_base + "acoustic.png", "evidence": asset_base + "evidence.json"})
    case = {
        **event, "start": start, "end": end, "clip_start": origin, "clip_end": local_end,
        "extended_start": extended_start, "extended_end": extended_end, "extended_scope": scope,
        "audio_file": report["source"]["path"], "source_sha256": report["source"]["sha256"],
        "recording_id": name, "voice_id": None, "model_id": None, "generation_settings": None,
        "expected_language": "nb-NO" if event["category"] in {"pronunciation", "cross_language_pronunciation"} else None,
        "context_language_evidence": language_votes, "expected_pronunciation": targets or None,
        "sentence": next(iter(text.values()), None), "sentence_origin": "reverse_transcription",
        "asr_excerpt": next(iter(text.values()), None), "reverse_transcripts": text,
        "fingerprint": observed, "same_recording_contrasts": statistics, "reference_intervals": controls,
        "assets": assets, "accepted_repair": None,
        "manual_status": "APPROXIMATE_USER_LABEL" if "annotation_id" in event else "UNREVIEWED",
        "confidence": "exploratory", "mechanism": None, "source_generation_cost": None,
        "working_hypothesis": signal_hypothesis(event, observed, statistics),
    }
    if embedding:
        embedding_path = root/"embeddings"/f"{event['id']}.npy"
        case["embedding_model_sha256"] = embedding.sha256
        if (same_interval and recorded.get("embedding_model_sha256") == embedding.sha256 and
                embedding_path.is_file()):
            case["audioset_scores"] = recorded.get("audioset_scores")
        else:
            vector, descriptors = embedding.describe(local[round(max(0,start-origin-.45)*rate):
                                                           round(min(local_end-origin,end-origin+.45)*rate)], rate)
            np.save(embedding_path, vector)
            case["audioset_scores"] = descriptors
        from .research_review import find_examples

        case["reviewed_neighbors"] = find_examples(root, embedding_path, case["source_sha256"], 5, embedding.sha256)
    write_json(base/"evidence.json", case)
    return case


def comparable_intervals(local, rate, curves, origin, start, end, observed):
    references = []
    length = end-start
    for shift in (-4, -3, -2, -1.2, 1.2, 2, 3, 4):
        a, b = start+shift, end+shift
        if a < origin or b > origin + len(local)/rate or abs(shift) < length+.25:
            continue
        values = fingerprint(local, rate, curves, a-origin, b-origin)
        if values["rms_dbfs"] < -55:
            continue
        match = abs(values["rms_dbfs"]-observed["rms_dbfs"])/12
        if observed["voiced_fraction"] is not None and values["voiced_fraction"] is not None:
            match += abs(values["voiced_fraction"]-observed["voiced_fraction"])
        references.append({"start": a, "end": b, "matching_cost": match, "features": values,
                           "status": "unreviewed_local_comparator"})
    return sorted(references, key=lambda row: row["matching_cost"])[:3]


def signal_hypothesis(event, profile, differences):
    if event["category"] in {"pronunciation", "cross_language_pronunciation"}:
        return {"description": "Review the source word and compare independent phones to dictionary hypotheses.",
                "possible_detection": "Local multilingual ASR and competing phone-sequence scores.",
                "possible_repair": "Sentence regeneration or a scoped dictionary trial after native review."}
    if event["category"] == "control_candidate":
        return {"description": "Same-source listening comparator awaiting review."}
    if profile["higher_band_ratio_peak"] > .18 and differences.get("higher_band_ratio_peak", 0) > .06:
        cue = "Upper-band transient; a fricative, release or breath can show the same pattern."
    elif profile["voiced_fraction"] and differences.get("hnr_db", 0) < -5:
        cue = "Lower harmonicity than local comparators; voiced irregularity or a mixed phone window is possible."
    elif differences.get("peak_step_abs", 0) > .09:
        cue = "Large sample-step contrast; inspect decoder ringing, an impulse or a stop release."
    else:
        cue = "Local spectral or energy change; physical subtype unresolved."
    return {"description": cue, "possible_detection": "Inspect local waveform, spectral and periodicity contrast.",
            "possible_repair": "Review before considering a bounded impulse repair or sentence regeneration."}


def manual_events(targets, baseline, proxy, words, match, studies):
    by_label = {pair["annotation_id"]: pair["event_id"] for pair in match["typed_matches"]}
    by_id = {row["id"]: row for row in baseline["events"]}
    duration, rate = baseline["duration_s"], baseline["sample_rate"]
    for label in targets:
        point = label["point_s"]
        event = by_id.get(by_label.get(label["id"]))
        hits = [item for item in nearby_words(words, point-5, point+5)
                if item["text"].casefold() == (label["word"] or "").casefold()]
        peaks = []
        if label["category"] != "artifact":
            word = min(hits, key=lambda row: abs((row["start"]+row["end"])/2-point)) if hits else None
            lo, hi = (word["start"], word["end"]) if word else (point-1.5, point+1.5)
            timing = "unverified_ASR_word_estimate" if word else "unlocalized_pronunciation_listening_window"
            study = studies.get(label["id"])
            if study:
                lo, hi = study["scope"]
                timing = "targeted_ASR_and_phone_comparison_scope"
        else:
            local, origin, _ = audio_bounds(proxy, rate, round(duration*rate), point-4, point+4)
            lo, hi, peaks = nearest_activity(local, rate, origin, point)
            if event:
                lo, hi = event["start"], event["end"]
            timing = "DSP_hypothesis_near_approximate_point" if peaks or event else "unlocalized_artifact_listening_window"
        yield {
            **label, "annotation_id": label["id"], "id": "annotation-"+digest([baseline["source"]["sha256"], label["id"]])[:14],
            "start": max(0, lo), "end": min(duration, hi), "timing": timing, "status": "REVIEW_REQUIRED",
            "severity": "HIGH" if label["category"] != "artifact" else "MEDIUM",
            "hypothesis_peaks": peaks[:5], "baseline_match": event["id"] if event else None,
            "evidence": event["evidence"] if event else [],
            "word": (studies[label["id"]]["word"] if label["id"] in studies else label["word"]),
            "reported_word": label["word"], "phonetic_study": studies.get(label["id"]),
        }


def control_events(targets, events, proxy, rate, duration, sha):
    blocked = [row["point_s"] for row in targets] + [(row["start"]+row["end"])/2 for row in events]
    for i, hint in enumerate((.08, .20, .35, .47, .62, .74, .87, .95), 1):
        center = hint * duration
        for offset in (0, 12, -12, 23, -23, 36, -36):
            point = center + offset
            if any(abs(point - marker) < 6 for marker in blocked) or not 6 < point < duration - 6:
                continue
            values, _, _ = audio_bounds(proxy, rate, round(duration*rate), point-.35, point+.35)
            if np.sqrt(np.mean(values**2)) < .014:
                continue
            yield {"id": "control-"+digest([sha, i])[:14], "start": point-.35, "end": point+.35,
                   "category": "control_candidate", "severity": "INFO", "timing": "selected_recording_interval",
                   "status": "REVIEW_REQUIRED", "evidence": [],
                   "selection": "audible sample at a fixed relative position, away from approximate annotations and blind events"}
            break


def prioritize(event, discrepancies):
    category = event["category"]
    if category in PRIORITY:
        score = PRIORITY[category]
    else:
        score = 40 if event.get("severity") != "INFO" else 0
    reason = "Approximate human report" if event.get("annotation_id") else "Independent candidate for adjudication"
    if event["id"] in discrepancies:
        score, reason = max(score, 85), "Detector category disagrees with a nearby approximate report"
    if category == "control_candidate":
        reason = "Comparator selected for human review"
    return {**event, "review_priority": score, "review_priority_reason": reason}


def prepare(args):
    began = perf_counter()
    root, name = args.dataset, args.recording
    for path in PARTITIONS:
        (root/path).mkdir(parents=True, exist_ok=True)
    baseline = read_json(args.baseline)
    sha = baseline["source"]["sha256"]
    guard = args.baseline.with_suffix(".sha256")
    if guard.is_file() and guard.read_text(encoding="utf-8").split()[0] != file_digest(args.baseline):
        raise ValueError("Frozen baseline SHA-256 mismatch.")
    if file_digest(args.audio) != sha:
        raise ValueError("Source audio and baseline SHA-256 differ.")
    copied = root/"raw_audio"/(name+args.audio.suffix.lower())
    if not copied.exists():
        staging = copied.with_suffix(copied.suffix+".part")
        shutil.copyfile(args.audio, staging)
        staging.replace(copied)
    if file_digest(copied) != sha:
        raise ValueError("Dataset original does not match the scan.")
    rate, duration = baseline["sample_rate"], baseline["duration_s"]
    with sf.SoundFile(args.proxy) as proxy:
        if proxy.samplerate != rate or len(proxy) != round(duration*rate):
            raise ValueError("Decoded audio does not match the baseline sample clock.")
    words, language_windows = report_words(args.asr, sha)
    primary = next(iter(words.values()), [])
    targets = labels(args.annotations, duration)
    tolerances = (1, 2.5, 5)
    baseline_eval = [evaluate(targets, baseline["events"], duration, value) for value in tolerances]
    optional_channels = []
    selected = [evaluate(targets, baseline["events"], duration, 2.5)]
    started = perf_counter()
    novelty_output = root/"metadata"/f"{name}-experimental-dsp.json"
    existing_dsp = read_json(novelty_output) if args.reuse_scans and novelty_output.exists() else {}
    if existing_dsp and existing_dsp["candidate_floor"] == args.novelty_floor and existing_dsp["settings"] == NOVELTY_SETTINGS:
        novelty = existing_dsp["events"]
        novelty_seconds = existing_dsp["seconds"]
    else:
        novelty = candidate_report(args.proxy, baseline, args.novelty_floor)
        novelty_seconds = perf_counter()-started
    novelty_sweep = {str(threshold): evaluate(targets, [row for row in novelty if row["score"] >= threshold],
                                               duration, 2.5) for threshold in (args.novelty_floor, 7.5, 8.5, 9.5)
                     if threshold >= args.novelty_floor}
    write_json(novelty_output, {"events": novelty, "settings": NOVELTY_SETTINGS,
               "candidate_floor": args.novelty_floor, "seconds": novelty_seconds, "threshold_sweep": novelty_sweep})
    embedding = Yamnet(args.yamnet) if args.yamnet else None
    discrepancies = {row["event_id"] for row in baseline_eval[1]["category_disagreements"]}
    if embedding:
        started = perf_counter()
        embedding_path = root/"metadata"/f"{name}-embedding-scan.json"
        embedding_result = read_json(embedding_path) if args.reuse_scans and embedding_path.exists() else {}
        if embedding_result.get("model_sha256") != embedding.sha256:
            embedding_result = embedding_scan(args.proxy, baseline, embedding, root, name)
            embedding_result["processing_seconds"] = perf_counter()-started
        embedding_eval = evaluate(targets, embedding_result["events"], duration, 2.5)
        write_json(embedding_path, {**embedding_result, "evaluation": embedding_eval})
        discrepancies.update(row["event_id"] for row in embedding_eval["category_disagreements"])
        optional_channels.extend(embedding_result["events"])
    if args.relaxed:
        relaxed = read_json(args.relaxed)
        if relaxed["source"]["sha256"] != sha:
            raise ValueError("Relaxed scan came from a different original.")
        selected[0] = {"baseline": selected[0], "relaxed": evaluate(targets, relaxed["events"], duration, 2.5),
                       "relaxed_processing_s": relaxed.get("processing_seconds_this_run"),
                       "relaxed_events": relaxed["events"]}
    studies = {row["annotation_id"]: row for row in read_json(args.phone_study)["cases"]} if args.phone_study else {}
    reviewed = {row["id"]: row for row in read_json(args.reviewer_reports)["results"]} if args.reviewer_reports else {}
    repairs = {row["id"]: row for row in read_json(args.repair_trials)["repair_trials"]} if args.repair_trials else {}
    events = list(manual_events(targets, baseline, args.proxy, primary, baseline_eval[1], studies))
    known_matches = {row["baseline_match"] for row in events if row.get("baseline_match")}
    events.extend(row for row in baseline["events"] if row["id"] not in known_matches)
    events.extend(row for row in novelty if row["score"] >= args.review_novelty_floor)
    events.extend(optional_channels)
    events.extend(control_events(targets, baseline["events"], args.proxy, rate, duration, sha))
    events = [prioritize(row, discrepancies) for row in events]
    events.sort(key=lambda row: (-row["review_priority"], -row.get("score", 0), row["start"]))
    cases = []
    with closing(Lexicon("e_written")) as lexicon:
        for event in events:
            case = make_case(event, baseline, args.proxy, words, language_windows, root, name, lexicon, embedding)
            if case["id"] in reviewed:
                case["independent_audio_model_review"] = reviewed[case["id"]]
            if case["id"] in repairs:
                repair = repairs[case["id"]]
                case["repair_experiments"] = repair["experiments"]
                available = [row for row in repair["experiments"] if row.get("review_asset")]
                if available:
                    case["assets"]["repaired_original"] = f"clips/{case['id']}/repair-original.wav"
                    case["assets"]["repaired"] = available[0]["review_asset"]
            if case["id"] in reviewed or case["id"] in repairs:
                write_json(root/"review"/"clips"/case["id"]/"evidence.json", case)
            cases.append(case)
        vocabulary = ({word for phrase in PARTICLE_PHRASES for word in phrase.split()} | {"slo", "holder", "å"})
        write_json(root/"metadata"/f"{name}-lexicon.json", {
            "profile": "NB Uttale e_written",
            "word_forms": {word: lexicon.lookup(word) for word in sorted(vocabulary)},
            "particle_verbs": {f"å {phrase}": lexicon.lookup(phrase) for phrase in PARTICLE_PHRASES},
        })
    manifest = {
        "schema_version": 1, "recording_id": name, "source": {**baseline["source"], "dataset_copy": str(copied)},
        "sample_rate": rate, "duration_s": duration, "baseline_sha256": file_digest(args.baseline),
        "manuscript_available": bool(baseline.get("captions_authoritative")), "annotation_photo_available": None,
        "generation": {"voice_id": None, "model_id": None, "settings": None}, "cases": cases,
        "channels": {"signal": "frozen baseline and exploratory scan", "reverse_asr": list(words),
                     "embedding": "YAMNet" if embedding else None, "phonetics": bool(studies), "audio_model": bool(reviewed)},
        "case_counts": dict(Counter(row["category"] for row in cases)),
        "processing_seconds": perf_counter()-began, "api_credits": 0,
    }
    write_json(root/"metadata"/f"{name}-manifest.json", manifest)
    write_json(root/"metadata"/f"{name}-benchmark.json", {"baseline": baseline_eval,
               "sensitivity_experiment": selected[0], "novelty_threshold_sweep": novelty_sweep,
               "combined_exploratory": evaluate(targets, baseline["events"] + optional_channels +
                                               [row for row in novelty if row["score"] >= args.review_novelty_floor],
                                               duration, 2.5),
               "source_duration_s": duration, "seconds_baseline": baseline["processing_seconds_this_run"],
               "independent_recordings": [], "api_credits": 0})
    report = {**baseline, "events": cases, "review_run": name, "dataset_review": True,
              "limitations": ["Approximate labels and automated intervals require listening. Control examples are pending review.",
                              "The local, micro and extended players use the decoded source. Time fields use absolute seconds."]}
    export_report(root, report)
    write_json(root/"reports"/f"{name}.json", report)
    atomic_bytes(root/"review"/f"{name}.html", dashboard(report).encode())
    print(f"{name}: {len(targets)} source labels, {len(cases)} review items, {perf_counter()-began:.1f} seconds.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/elevenlabs_audio_qa"))
    parser.add_argument("--recording", required=True, type=safe_recording)
    for name in ("audio", "baseline", "proxy"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("annotations", "relaxed", "yamnet", "phone-study", "reviewer-reports", "repair-trials"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--asr", type=Path, action="append", default=[])
    parser.add_argument("--novelty-floor", type=float, default=6.5)
    parser.add_argument("--review-novelty-floor", type=float, default=8.5)
    parser.add_argument("--reuse-scans", action="store_true")
    prepare(parser.parse_args())


def safe_recording(value):
    if not value or not value.isascii() or not value.replace("-", "").replace("_", "").isalnum():
        raise argparse.ArgumentTypeError("Use an ASCII recording ID with letters, digits, hyphens or underscores.")
    return value


if __name__ == "__main__":
    main()

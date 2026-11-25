"""Compare frozen candidates with approximate or adjudicated annotations."""
from collections import Counter
from math import inf

import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.benchmark_detectors import union_duration
from scripts.evidence_fusion import DIGITAL, LANGUAGE, WORD_EVENTS


def broad(category):
    if category in DIGITAL or category in {
        "artifact", "transient_outlier", "local_acoustic_novelty", "possible_timbre_or_metallic_shift",
        "possible_unstable_or_robotic_voicing", "possible_gain_or_noise_shift", "voice_spectral_anomaly",
        "possible_stretched_word", "embedding_outlier", "possible_speech_interruption",
    }:
        return "artifact"
    if category in {"possible_phonetic_interference", "possible_unexpected_language", "cross_language_pronunciation"}:
        return "cross_language_pronunciation"
    if category in LANGUAGE or category == "pronunciation":
        return "pronunciation"
    if category in WORD_EVENTS:
        return "speech_content"
    return category


def assign(targets, candidates, tolerance, *, typed, points):
    if not targets or not candidates:
        return []
    if not 0 <= tolerance < inf:
        raise ValueError("Tolerance must be a finite nonnegative number.")
    # Any eligible edge costs less than the loss from giving up a match.
    penalty = (max(len(targets), len(candidates)) + 1) * (tolerance + 1)
    costs = np.full((len(targets), len(candidates)), penalty)
    for i, label in enumerate(targets):
        for j, event in enumerate(candidates):
            if typed and broad(label["category"]) != broad(event["category"]):
                continue
            distance = (abs((event["start"] + event["end"]) / 2 - label["point_s"]) if points else
                        max(0, label["start"] - event["end"], event["start"] - label["end"]))
            if distance <= tolerance:
                costs[i, j] = distance
    rows, cols = linear_sum_assignment(costs)
    return [(i, j, float(costs[i, j])) for i, j in zip(rows, cols) if costs[i, j] < penalty]


def evaluate(targets, events, duration, tolerance, *, points=True, candidate_decisions=None, complete=False):
    candidates = [event for event in events if event.get("severity") != "INFO"]
    if points and (complete or candidate_decisions):
        raise ValueError("Provisional points cannot be a complete adjudicated benchmark.")
    if complete and any(item.get("status") not in {"CONFIRMED", "VERIFIED"} for item in targets):
        raise ValueError("A complete benchmark requires reviewed target intervals.")
    timing = assign(targets, candidates, tolerance, typed=False, points=points)
    typed = assign(targets, candidates, tolerance, typed=True, points=points)
    found_targets, found_candidates = {i for i, _, _ in typed}, {j for _, j, _ in typed}
    timing_candidates = {j for _, j, _ in timing}
    correct = Counter(broad(targets[i]["category"]) for i, _, _ in typed)
    expected = Counter(broad(row["category"]) for row in targets)
    decisions = candidate_decisions or {}
    unreviewed = [row["id"] for j, row in enumerate(candidates) if j not in found_candidates and
                  decisions.get(row["id"]) not in {"FALSE_POSITIVE", "HARD_NEGATIVE"}]
    dismissed = len(candidates) - len(found_candidates) - len(unreviewed)
    tp, fn = len(typed), len(targets) - len(typed)
    fp = dismissed if complete and not unreviewed else None
    timing_errors = [dist for _, _, dist in typed]
    start_errors = [abs(targets[i]["start"] - candidates[j]["start"]) for i, j, _ in typed] if not points else []
    end_errors = [abs(targets[i]["end"] - candidates[j]["end"]) for i, j, _ in typed] if not points else []
    return {
        "status": "provisional_point_overlap" if points else "reviewed_intervals",
        "tolerance_s": tolerance, "target_count": len(targets), "candidate_count": len(candidates),
        "typed_matches": [{"annotation_id": targets[i]["id"], "event_id": candidates[j]["id"],
                           "distance_s": d, "predicted_category": candidates[j]["category"]} for i, j, d in typed],
        "category_disagreements": [
            {"annotation_id": targets[i]["id"], "event_id": candidates[j]["id"], "expected": targets[i]["category"],
             "predicted": candidates[j]["category"], "distance_s": d}
            for i, j, d in timing if broad(targets[i]["category"]) != broad(candidates[j]["category"])
        ],
        "unmatched_annotations": [row["id"] for i, row in enumerate(targets) if i not in found_targets],
        "unmatched_timing_candidates": [row["id"] for j, row in enumerate(candidates) if j not in timing_candidates],
        "unadjudicated_candidates": unreviewed,
        "annotation_overlap_recall": tp / len(targets) if targets else None,
        "recall": tp / len(targets) if complete and targets else None,
        "category_recall": {category: correct[category] / count for category, count in sorted(expected.items())},
        "confirmed_false_positives": dismissed, "false_negatives": fn if complete else None,
        "precision": tp / (tp + fp) if fp is not None and tp + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if fp is not None and 2 * tp + fp + fn else None,
        "false_positives_per_hour": fp * 3600 / duration if fp is not None and duration else None,
        "mean_approximate_point_distance_s": float(np.mean(timing_errors)) if points and timing_errors else None,
        "start_mae_s": float(np.mean(start_errors)) if start_errors else None,
        "end_mae_s": float(np.mean(end_errors)) if end_errors else None,
        "manual_review_seconds": union_duration(
            (event.get("clip_start", event["start"]), event.get("clip_end", event["end"])) for event in candidates),
    }


def main():
    import argparse
    from pathlib import Path

    from scripts.common import read_json, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("targets", "predictions", "decisions", "out"):
        parser.add_argument(f"--{label}", type=Path, required=label!="decisions")
    parser.add_argument("--tolerance", type=float, default=.30)
    parser.add_argument("--complete", action="store_true", help="All source defects and unmatched predictions were adjudicated.")
    args = parser.parse_args()
    targets = [row for row in read_json(args.targets) if row.get("status") in {"CONFIRMED", "VERIFIED"}]
    predictions = read_json(args.predictions)
    decisions = {row["id"]:row["status"] for row in read_json(args.decisions)} if args.decisions else {}
    metrics = evaluate(targets, predictions["events"], predictions["duration_s"], args.tolerance,
                       points=False, candidate_decisions=decisions, complete=args.complete)
    write_json(args.out, {**metrics, "processing_seconds": predictions.get("processing_seconds_this_run"),
                          "api_credits": predictions.get("api", {}).get("credits_used")})


if __name__ == "__main__":
    main()

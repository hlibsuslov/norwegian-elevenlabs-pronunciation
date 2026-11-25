"""Accept a candidate only with comparable target improvement and regression checks."""
import argparse

import numpy as np

from .common import config, read_json, write_json
from .detect_stress_errors import independent_cues


def compare(before, after, before_features, after_features, targets):
    thresholds = config("quality_thresholds")
    baseline = {int(i): value for i, value in before["scores"].items()}
    revised = {int(i): value for i, value in after["scores"].items()}
    reasons = []
    if not targets or not set(targets).issubset(baseline.keys() & revised.keys()):
        return {"accepted": False, "reasons": ["missing_comparable_target_acoustics"], "comparison": {}}
    if not set(baseline).issubset(revised):
        reasons.append("lost_reference_coverage")
    before_ids = {row["id"] for row in before_features["aligned_words"]}
    after_ids = {row["id"] for row in after_features["aligned_words"]}
    if not before_ids.issubset(after_ids):
        reasons.append("lost_word_alignment")
    if after_features["peak_dbfs"] >= -0.1 and after_features["peak_dbfs"] > before_features["peak_dbfs"] + 0.01:
        reasons.append("near_clipping")
    if len(after_features["pauses"]) > len(before_features["pauses"]):
        reasons.append("additional_long_pause")
    before_asr, after_asr = before_features["asr"], after_features["asr"]
    if not before_asr or not after_asr or before_asr["source"] != after_asr["source"]:
        reasons.append("independent_asr_required_for_auto_acceptance")
    elif after_asr["lexical_edit_count"] > before_asr["lexical_edit_count"]:
        reasons.append("transcript_regression")
    else:
        prior = {(tuple(i["expected"]), tuple(i["observed_words"]), i["type"]) for i in before_asr["issues"]}
        if any((tuple(i["expected"]), tuple(i["observed_words"]), i["type"]) not in prior
               for i in after_asr["issues"] if i["type"] != "written_spoken_normalization_review"):
            reasons.append("new_transcript_edits")
    comparison = {}
    for token_id in sorted(baseline.keys() & revised.keys()):
        a, b = baseline[token_id]["features"], revised[token_id]["features"]
        shared = a.keys() & b.keys()
        if not a.keys() <= b.keys() or independent_cues(shared) < thresholds["minimum_compare_features"]:
            reasons.append(f"lost_acoustic_features:{token_id}")
            continue
        before_mean = float(np.mean([a[field] for field in shared]))
        after_mean = float(np.mean([b[field] for field in shared]))
        comparison[token_id] = {"before": before_mean, "after": after_mean,
                                "target": token_id in targets, "features": sorted(shared)}
        if token_id in targets:
            if before_mean - after_mean < thresholds["minimum_improvement"]:
                reasons.append(f"insufficient_target_improvement:{token_id}")
            if any(b[field] - a[field] > thresholds["maximum_feature_regression"] for field in shared):
                reasons.append(f"target_acoustic_regression:{token_id}")
        elif after_mean - before_mean > thresholds["maximum_nontarget_regression"]:
            reasons.append(f"non_target_regression:{token_id}")
    return {"accepted": not reasons, "reasons": reasons, "comparison": comparison}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--target", required=True, type=int, action="append")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    before, after = read_json(args.before), read_json(args.after)
    write_json(args.out, compare(before["detection"], after["detection"], before["features"], after["features"], args.target))

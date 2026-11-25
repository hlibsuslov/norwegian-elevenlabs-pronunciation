"""Compare multiple acoustic cues against matched, independently reviewed audio."""
import argparse
import math

import numpy as np

from .common import config, read_json, write_json


def scaled_delta(candidate, reference, unit):
    if candidate is None or reference is None:
        return None
    return abs(candidate - reference) / unit


def median(values):
    valid = [value for value in values if value is not None]
    return float(np.median(valid)) if valid else None


def feature_distances(word, reference, units):
    deltas = {name: scaled_delta(word.get(name), reference.get(name), units[name])
              for name in ("relative_duration_log", "relative_intensity_db")}
    contour = [scaled_delta(a, b, units["f0_shape_st"]) for a, b in zip(word["f0_shape_st"], reference["f0_shape_st"])]
    if sum(value is not None for value in contour) >= 3:
        deltas["f0_shape_st"] = median(contour)
    if word["phones_match_lexical_target"] and reference["phones_match_lexical_target"]:
        phone_pairs = list(zip(word["phones"], reference["phones"]))
        if len(word["phones"]) == len(reference["phones"]) and all(a["phone"] == b["phone"] for a, b in phone_pairs):
            for kind in ("vowel", "consonant"):
                name = f"{kind}_duration_log"
                deltas[name] = median([
                    abs(math.log(a["duration_s"] / b["duration_s"])) / units[name] for a, b in phone_pairs
                    if ("f1_bark" in a and "f1_bark" in b if kind == "vowel"
                        else a["stress_code"] is None and b["stress_code"] is None)
                ])
            for name in ("f1_bark", "f2_bark"):
                deltas[name] = median([scaled_delta(a.get(name), b.get(name), units[name]) for a, b in phone_pairs])
            if len(word["syllables"]) == len(reference["syllables"]):
                deltas["syllable_duration_log"] = median([
                    abs(math.log(a["duration_s"] / b["duration_s"])) / units["syllable_duration_log"]
                    for a, b in zip(word["syllables"], reference["syllables"])
                ])
    return {name: value for name, value in deltas.items() if value is not None}


def independent_cues(features):
    groups = (
        {"f0_shape_st"}, {"relative_intensity_db"}, {"f1_bark", "f2_bark"},
        {"relative_duration_log", "vowel_duration_log", "consonant_duration_log", "syllable_duration_log"},
    )
    return sum(bool(group.intersection(features)) for group in groups)


def detect(analysis, measured, reference=None):
    thresholds = config("quality_thresholds")
    reference_words = {row["id"]: row for row in reference["aligned_words"]} if reference else {}
    lexemes = {row["id"]: row for row in analysis["tokens"]}
    scores = {}
    defects = list(measured["pauses"])
    if measured["asr"]:
        defects.extend(measured["asr"]["issues"])
    for observed in measured["aligned_words"]:
        token_id = observed["id"]
        expected = reference_words.get(token_id)
        if not expected:
            continue
        features = feature_distances(observed, expected, thresholds["reference_units"])
        if independent_cues(features) < thresholds["minimum_compare_features"]:
            continue
        score = float(np.mean(list(features.values())))
        scores[token_id] = {"mean": score, "features": features}
        if score >= thresholds["warning_reference_distance"]:
            token = lexemes[token_id]
            target = token["target"]
            reason = [name for name, value in features.items() if value >= 1.0]
            defects.append({"type": "matched_reference_deviation", "token_id": token_id,
                            "word": observed["text"], "start": observed["start"], "end": observed["end"],
                            "sentence": observed["sentence"], "confidence": "probabilistic",
                            "possible_causes": reason, "observed_ipa": None,
                            "target_ipa": target["ipa"] if target else None,
                            "target_tonemes": target["tonemes"] if target else [],
                            "target_primary_stress": target["primary_stress_phone_indices"] if target else [],
                            "target_secondary_stress": target["secondary_stress_phone_indices"] if target else []})
    return {
        "defects": defects, "scores": scores, "confirmed_errors": 0,
        "status": "matched_reference_comparison" if reference else "no_matched_acoustic_reference",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis")
    parser.add_argument("audio_features")
    parser.add_argument("--reference-features")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    reference = read_json(args.reference_features) if args.reference_features else None
    write_json(args.out, detect(read_json(args.analysis), read_json(args.audio_features), reference))

"""Regenerate only the aligned sentence containing a reviewed candidate defect."""
from .common import config
from .generate_audio import APIError, pcm_wav


def windows(segment, analysis, features):
    ids = [token["id"] for token in analysis["tokens"] if token["sentence"] == segment["id"]]
    words = features["aligned_words"]
    by_id = {word["id"]: word for word in words}
    if not ids or not all(i in by_id for i in ids):
        raise ValueError("The repair sentence lacks complete word boundaries.")
    first, last = by_id[ids[0]], by_id[ids[-1]]
    if features["alignment_source"] != "NoFA_TextGrid":
        raise ValueError("Repair needs word boundaries from the Norwegian forced aligner.")
    aligned = features["alignment_words"]
    previous_end = max((w["end"] for w in aligned if w["end"] <= first["start"] + 1e-7), default=0)
    next_start = min((w["start"] for w in aligned if w["start"] >= last["end"] - 1e-7),
                     default=features["duration_s"])
    limit = config("quality_thresholds")["boundary_search_seconds"]
    left, right = (max(previous_end, first["start"] - limit), first["start"]), (
        last["end"], min(next_start, last["end"] + limit)
    )
    if previous_end == 0 and ids[0] == 0:
        left = (0, 0)
    if next_start == features["duration_s"] and ids[-1] == len(analysis["tokens"]) - 1:
        right = (next_start, next_start)
    return left, right


def repair_jobs(plan, analysis, detection):
    by_sentence = {segment["id"]: segment for segment in plan["segments"]}
    by_token = {token["id"]: token for token in analysis["tokens"]}
    jobs = {}
    for defect in detection["defects"]:
        if defect["type"] != "matched_reference_deviation":
            continue
        segment = by_sentence[defect["sentence"]]
        target = by_token[defect["token_id"]]
        rule = next((
            row for row in segment["rules"] if row["word"] == defect["word"]
            or ("source_span" in row and row["source_span"][0] <= target["start"]
                and target["end"] <= row["source_span"][1])
        ), None)
        if not rule or segment["over_model_limit"]:
            continue
        job = jobs.setdefault(segment["id"], {"segment": segment, "target_ids": [], "rules": []})
        job["target_ids"].append(defect["token_id"])
        if rule not in job["rules"]:
            job["rules"].append(rule)
    return list(jobs.values())


def regenerate(api, plan, job, settings, seed, attempt):
    pcm, metadata = api.generate(plan["model"], plan["voice_id"], job["segment"], job["rules"],
                                 settings=settings, seed=(seed + attempt) % (2**32), take=attempt)
    return pcm_wav(pcm), metadata


if __name__ == "__main__":
    import sys

    from .workflow import main

    main(["repair", *sys.argv[1:]])

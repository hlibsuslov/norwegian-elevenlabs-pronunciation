"""Merge repeated detections and preserve independent evidence and clock provenance."""
from collections import defaultdict

from .common import digest

DIGITAL={"click_or_pop","clipping","dropout","high_frequency_burst","crackle"}
LANGUAGE={"possible_unexpected_language","possible_phonetic_interference","possible_phone_mismatch",
          "possible_stress_mismatch","possible_tone_mismatch"}
WORD_EVENTS={"possible_omission","possible_insertion","possible_substitution","possible_repetition",
             "transcript_unmatched_block","unclear_speech"}
INFO={"interword_pause","transition_pause","spoken_normalization"}
OTHER_TIME={"window_estimate", "asr_estimate", "reviewer_estimate"}
ACTION={"click_or_pop":"localized_declick","crackle":"review_local_declick","clipping":"review_or_regenerate",
        "dropout":"review_or_regenerate"}


def threshold(category):
    return .035 if category in DIGITAL else .30


def source_name(detector):
    if detector.startswith("asr:"):
        return detector.split(":")[1]
    return detector


def fuse(events, audio, duration):
    grouped = defaultdict(list)
    for event in events:
        if 0 <= event["start"] < event["end"] <= duration+.060:
            grouped[event["category"]].append({**event,"end":min(event["end"],duration)})
    rows=[]
    for category, items in grouped.items():
        current=None
        for event in sorted(items,key=lambda e:(e["start"],e["end"])):
            if current is None or event["start"]-current["end"]>threshold(category) or (
                event.get("word") and current.get("word") and event["word"].casefold()!=current["word"].casefold()
            ):
                current={**event,"evidence":[evidence_entry(event)]}
                rows.append(current)
                continue
            current["end"]=max(current["end"],event["end"])
            current["evidence"].append(evidence_entry(event))
            for field in ("word","sentence","expected_language","detected_language","pronunciation_language",
                          "expected_pronunciation","observed_pronunciation","intentional_code_switch"):
                if current.get(field) is None and event.get(field) is not None:
                    current[field]=event[field]
            if event["timing"]=="decoded_sample_clock":
                current["timing"]=event["timing"]
    for row in rows:
        category=row["category"]
        names=sorted({e["detector"] for e in row["evidence"]})
        families={source_name(n) for n in names}
        distinct={digest(e):e for e in row["evidence"]}
        independent=len(families)
        severity=("HIGH" if category in LANGUAGE else "MEDIUM" if category in DIGITAL else
                  "INFO" if category in INFO else "LOW")
        confidence=("multi_signal" if independent>=2 else "signal_pattern" if category in DIGITAL else "heuristic")
        status=("LIKELY" if category in {"clipping","dropout"} or independent>=2 else
                "HUMAN_REVIEW_REQUIRED" if category in LANGUAGE else "SUSPECTED")
        for name in ("word","sentence","expected_language","detected_language","pronunciation_language",
                     "expected_pronunciation","observed_pronunciation"):
            row.setdefault(name,None)
        row.update({"id":digest([audio["sha256"],category,round(row["start"],3),round(row["end"],3)])[:14],
                    "detector_names": names, "confidence":confidence,"severity":severity,"status":status,
                    "evidence":list(distinct.values())[:16],"evidence_count":len(distinct),"audio_file":audio["path"],
                    "recommended_repair":ACTION.get(category,"review_context" if category not in INFO else "none")})
    return sorted(rows,key=lambda row:(row["start"],row["end"],row["category"]))


def evidence_entry(event):
    return {"detector":event["detector"],"timing":event["timing"],"data":event["evidence"]}

"""Flag transcript disagreements without converting ASR confidence into phonetic facts."""
from difflib import SequenceMatcher

from .common import TOKEN, key


def comparison(expected, observed, label, limit=10):
    left = [key(item["text"]) for item in expected]
    right = [key(item["text"]) for item in observed]
    match = SequenceMatcher(None, left, right, autojunk=False)
    events, matches = [], []
    for kind, a, b, c, d in match.get_opcodes():
        if kind == "equal":
            matches.extend(zip(range(a,b), range(c,d)))
            continue
        reference = expected[a:b]
        found = observed[c:d]
        time_source = found or observed[max(0,c-1):min(len(observed),c+1)]
        if not time_source or max(b-a,d-c) > limit:
            # Long unmatched passages need one navigation event, not false per-word confidence.
            if found:
                events.append({"category": "transcript_unmatched_block", "start": found[0]["start"],
                               "end": found[-1]["end"], "detector": label,
                               "evidence": {"expected_count": b-a, "observed_count": d-c}, "timing": "asr_estimate"})
            continue
        if any(w["text"][:1].isdecimal() for w in (*reference,*found)):
            category = "spoken_normalization"
        else:
            category = {"delete": "possible_omission", "insert": "possible_insertion",
                        "replace": "possible_substitution"}[kind]
        if category == "possible_insertion" and found:
            neighbors = [right[i] for i in (c-1,d) if 0 <= i < len(right)]
            if any(key(row["text"]) in neighbors for row in found):
                category = "possible_repetition"
        events.append({"category": category, "start": time_source[0]["start"], "end": time_source[-1]["end"],
                       "detector": label, "word": " ".join(w["text"] for w in reference) or None,
                       "sentence": reference[0].get("sentence") if reference else None,
                       "expected_language": reference[0].get("expected_language") if reference else None,
                       "evidence": {"expected": [w["text"] for w in reference], "asr": [w["text"] for w in found]},
                       "timing": "asr_estimate"})
    return events, matches


def tokenized(text):
    return [{"text": m.group(), "offset": m.start()} for m in TOKEN.finditer(text)]


def low_confidence(words, detector, cutoff):
    return [{"category": "unclear_speech", "start": w["start"], "end": w["end"], "word": w["text"],
             "detector": detector, "evidence": {"asr_token_score": w["confidence"]}, "timing": "asr_estimate"}
            for w in words if w.get("confidence") is not None and w["confidence"] < cutoff]

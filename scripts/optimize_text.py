"""Prepare transparent, model-compatible pronunciation dictionary plans."""
import argparse
import re
from collections import Counter, defaultdict
from contextlib import closing

from .analyze_text import analyze
from .build_dictionary import record_rule, verified_records
from .common import config, key, model_config, read_text, write_json
from .lookup_pronunciation import Lexicon


def select_rule(token, sentence, analysis, model, voice, records, trials):
    word = token["text"]
    candidates = [row for row in records if key(row["word"]) == key(word)
                  and row["dialect"] == analysis["dialect"] and row["model"] == model
                  and row["voice_id"] in (None, voice)
                  and (row["scope"] == "global" or key(row["context"]) == key(sentence))]
    # A global rule would also affect other meanings of an ambiguous spelling.
    if token["resolution"] in ("unresolved_homograph", "contextual_pos_candidate",
                                "invalid_or_conflicting_context_id", "verified_context_id",
                                "attested_register_variants"):
        candidates = [row for row in candidates if row["scope"] == "sentence"]
    if candidates:
        selected = max(candidates, key=lambda row: (row["scope"] == "sentence", bool(row["voice_id"])))
        return record_rule(selected, word)
    flags = {"secondary_stress", "compound_review", "proper_name_or_toponym", "lexical_tone_or_sense_review"}
    target = token["target"]
    if (trials and target and model_config(model)["norwegian_ipa"] and flags.intersection(token["flags"])
            and token["resolution"] in ("lexical_consensus", "verified_context_id")):
        return {"word": word, "type": "phoneme", "value": target["ipa"],
                "evidence": "attested_ipa_trial", "source": target["source"]}
    return None


def phrase_rules(sentence, analysis, model, voice, records):
    matches = []
    for row in records:
        word = row["word"]
        if (not re.search(r"\s", word) or row["dialect"] != analysis["dialect"] or row["model"] != model
                or row["voice_id"] not in (None, voice)
                or (row["scope"] == "sentence" and key(row["context"]) != key(sentence))):
            continue
        occurrences = list(re.finditer(r"(?<!\w)" + re.escape(word) + r"(?!\w)", sentence, re.IGNORECASE))
        if len(occurrences) != 1:
            continue
        match = occurrences[0]
        matches.append((match.start(), match.end(), row))
    matches.sort(key=lambda item: (item[1] - item[0], item[2]["scope"] == "sentence", bool(item[2]["voice_id"])),
                 reverse=True)
    chosen = []
    for lo, hi, row in matches:
        if all(hi <= earlier or lo >= later for earlier, later, _ in chosen):
            chosen.append((lo, hi, record_rule(row, sentence[lo:hi])))
    return chosen


def prepare(analysis, model, voice=None, records=(), trials=False):
    specification = model_config(model)
    grouped = defaultdict(list)
    for token in analysis["tokens"]:
        grouped[token["sentence"]].append(token)
    segments = []
    for sentence in analysis["utterances"]:
        current = grouped[sentence["id"]]
        counts = Counter(key(token["text"]) for token in current)
        phrases = phrase_rules(sentence["text"], analysis, model, voice, records)
        proposals = defaultdict(list)
        for token in current:
            overlaps_phrase = "start" in token and any(
                token["start"] < sentence["start"] + end and token["end"] > sentence["start"] + begin
                for begin, end, _ in phrases
            )
            unsafe_repeat = counts[key(token["text"])] > 1 and (token["resolution"] != "lexical_consensus")
            rule = None if overlaps_phrase or unsafe_repeat else select_rule(
                token, sentence["text"], analysis, model, voice, records, trials
            )
            proposals[token["text"]].append(rule)
        rules = [
            {**rule, "source_span": [sentence["start"] + lo, sentence["start"] + hi]}
            for lo, hi, rule in phrases
        ]
        conflicts = []
        for word, options in proposals.items():
            active = {(rule["type"], rule["value"]) for rule in options if rule}
            if len(active) == 1 and all(options):
                rules.append(next(rule for rule in options if rule))
            elif active:
                conflicts.append(word)
        segments.append({
            **sentence, "rules": rules, "conflicts": conflicts,
            "previous_text": analysis["source"][max(0, sentence["start"] - 100):sentence["start"]],
            "next_text": analysis["source"][sentence["end"]:sentence["end"] + 100],
            "over_model_limit": len(sentence["text"]) > specification["limit"],
        })
    return {
        "model": model, "voice_id": voice, "transport": specification["transport"],
        "norwegian_supported": specification["norwegian"], "dialect": analysis["dialect"], "language": "nb",
        "synthesis_language_code": "no" if specification["norwegian"] else None,
        "source": analysis["source"], "synthesis_text": analysis["source"], "segments": segments,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=config("models")["default"])
    parser.add_argument("--voice")
    parser.add_argument("--dialect", default=config("dialects")["default"])
    parser.add_argument("--trial-ipa", action="store_true")
    args = parser.parse_args()
    with closing(Lexicon(args.dialect)) as lexicon:
        analysis = analyze(read_text(args.text), lexicon)
    write_json(args.out, prepare(analysis, args.model, args.voice, verified_records(), args.trial_ipa))

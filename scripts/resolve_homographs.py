"""Resolve variants using documented source IDs or conservative POS evidence."""
from .common import key

# Universal POS -> NB Uttale's core tag inventory. Unknown mappings never eliminate a candidate.
POS_MATCH = {
    "ADJ": {"JJ"}, "ADP": {"PP"}, "ADV": {"AB"}, "AUX": {"VB"}, "CCONJ": {"KN"},
    "DET": {"DT"}, "INTJ": {"IE", "UH"}, "NOUN": {"NN"}, "NUM": {"CD"},
    "PRON": {"PN"}, "PROPN": {"PM"}, "SCONJ": {"SN"}, "VERB": {"VB"},
}


def lexical_pos(row):
    return row["pos"].partition("|")[0]


def signature(row):
    return row["nb_ipa"], row["nofabet"]


def resolve(word, candidates, morphology, sentence, dialect, overrides=()):
    if not candidates:
        return None, "out_of_vocabulary"
    if len({signature(row) for row in candidates}) == 1:
        return candidates[0], "lexical_consensus"

    manual = {
        str(item["wordform_id"]) for item in overrides
        if key(item["word"]) == key(word) and item["dialect"] == dialect
        and key(item["sentence"]) == key(sentence) and item.get("source")
    }
    if manual:
        matched = [row for row in candidates if manual.intersection(row["wordform_ids"])]
        if matched and len({signature(row) for row in matched}) == 1:
            return matched[0], "verified_context_id"
        return None, "invalid_or_conflicting_context_id"

    tags = POS_MATCH.get(morphology.get("pos"), set())
    matches = [row for row in candidates if lexical_pos(row) in tags]
    unresolved = [row for row in candidates if not lexical_pos(row)]
    if matches and not unresolved and len({signature(row) for row in matches}) == 1:
        return matches[0], "contextual_pos_candidate"
    comparable = matches or candidates
    if len(comparable) > 1 and set.intersection(*(set(row["wordform_ids"]) for row in comparable)):
        return None, "attested_register_variants"
    return None, "unresolved_homograph"


if __name__ == "__main__":
    import argparse
    import json

    from .common import read_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", help="The text_analysis.json from PREVENT.")
    parser.add_argument("--token", type=int, required=True)
    args = parser.parse_args()
    rows = [row for row in read_json(args.analysis)["tokens"] if row["id"] == args.token]
    if not rows:
        parser.error("Unknown token ID.")
    print(json.dumps(rows[0], ensure_ascii=False, indent=2))

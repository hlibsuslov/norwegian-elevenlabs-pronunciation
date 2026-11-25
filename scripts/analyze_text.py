"""Analyze an unchanged Norwegian manuscript with source and occurrence offsets."""
import argparse
import re
from contextlib import closing
from functools import lru_cache

from .common import ROOT, TOKEN, config, key, read_json, read_text, sentences, write_json
from .lookup_pronunciation import Lexicon
from .resolve_homographs import resolve


@lru_cache(maxsize=1)
def nlp_model():
    import spacy

    return spacy.load("nb_core_news_sm")


@lru_cache(maxsize=1)
def language_detector():
    from lingua import Language, LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_languages(
        Language.BOKMAL, Language.NYNORSK, Language.SWEDISH, Language.DANISH,
        Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.RUSSIAN,
        Language.POLISH, Language.SPANISH, Language.ITALIAN, Language.UKRAINIAN,
    ).with_low_accuracy_mode().with_minimum_relative_distance(0.15).build()


def language_assessment(source, requested):
    if requested != "auto":
        return {"target": requested, "detected": None, "method": "explicit"}
    if len(re.findall(r"[^\W\d_]+", source)) < 8:
        return {"target": "nb", "detected": None, "method": "short_input_default"}
    detected = language_detector().detect_language_of(source)
    code = detected.iso_code_639_1.name.lower() if detected else None
    return {"target": "nb", "detected": code, "method": "lingua_low_accuracy"}


def compound_candidates(surface, lookup, limit=4):
    normalized = key(surface)
    splits = []
    for at in range(3, len(normalized) - 2):
        left, right = normalized[:at], normalized[at:]
        for stem, link in ((left, ""), (left[:-1], left[-1:])):
            if (link in ("", "s", "e", "-") and len(stem) >= 3
                    and lookup(stem) and lookup(right)):
                splits.append([stem, link, right])
                if len(splits) == limit:
                    return splits
    return splits


def analyze(source, lexicon, language="auto", exceptions=None):
    if not source.strip():
        raise ValueError("The manuscript is empty.")
    exceptions = exceptions if exceptions is not None else read_json(ROOT / "dictionaries/exceptions.json")
    profile = language_assessment(source, language)
    doc = nlp_model()(source)
    morphology = {
        (t.idx, t.idx + len(t)): {
            "lemma": t.lemma_ or None, "pos": t.pos_ or None, "feats": t.morph.to_dict(),
            "dependency": t.dep_ or None, "entity": t.ent_type_ or None,
        }
        for t in doc if not t.is_space
    }
    tokens = []
    phrases = []
    utterances = []
    review_words = set(map(key, exceptions["review_words"]))
    known_compounds = set(map(key, exceptions["compound_review"]))
    for sentence_id, (start, end) in enumerate(sentences(source)):
        utterance = source[start:end]
        utterances.append({"id": sentence_id, "start": start, "end": end, "text": utterance})
        occurrences = list(TOKEN.finditer(source, start, end))
        for i, match in enumerate(occurrences):
            surface = match.group()
            morph = morphology.get(match.span(), {})
            spoken_form = not surface[0].isdecimal()
            candidates = lexicon.lookup(surface) if spoken_form else []
            target, resolution = resolve(
                surface, candidates, morph, utterance, lexicon.dialect, exceptions["sense_overrides"]
            )
            flags = []
            if not spoken_form:
                resolution = "model_text_normalization"
                flags.append("number_date_time_currency_or_amount")
            if surface.isupper() and len(surface) >= 2:
                flags.append("abbreviation_or_initialism")
            has_proper_row = any(row["pos"].startswith("PM") for row in candidates)
            if (has_proper_row and (morph.get("pos") == "PROPN" or not morph.get("pos"))
                    or morph.get("pos") == "PROPN" and i > 0):
                flags.append("proper_name_or_toponym")
            if morph.get("entity") in ("PER", "LOC", "GPE_LOC"):
                flags.append("named_entity")
            if morph.get("pos") == "X" or (spoken_form and not candidates and not morph.get("entity")):
                flags.append("unknown_spelling_or_possible_loan")
            if resolution in ("contextual_pos_candidate", "unresolved_homograph", "invalid_or_conflicting_context_id"):
                flags.append(resolution)
            if key(surface) in review_words:
                flags.append("lexical_tone_or_sense_review")
            if any(row["secondary_stress_phone_indices"] for row in candidates):
                flags.append("secondary_stress")
            compound = []
            if spoken_form and (key(surface) in known_compounds or (len(surface) >= 14 and not candidates)):
                flags.append("compound_review")
                if not candidates:
                    compound = compound_candidates(surface, lexicon.lookup)
            if i + 1 < len(occurrences):
                for final in occurrences[i + 1:i + 4]:
                    phrase = source[match.start():final.end()]
                    words = phrase.split()
                    if len(words) < 2 or any(not word[0].isalpha() for word in words):
                        break
                    phrase_rows = lexicon.lookup(" ".join(words))
                    if phrase_rows:
                        phrases.append({"start": match.start(), "end": final.end(),
                                        "text": phrase, "candidates": phrase_rows})
            tokens.append({
                "id": len(tokens), "text": surface, "start": match.start(), "end": match.end(),
                "sentence": sentence_id, "morphology": morph, "candidates": candidates,
                "target": target, "resolution": resolution, "flags": flags,
                "compound_split_candidates": compound,
            })
    return {
        "language": profile, "dialect": lexicon.dialect, "source": source,
        "utterances": utterances, "tokens": tokens, "phrase_candidates": phrases,
        "unresolved": [
            t["id"] for t in tokens
            if (t["target"] is None and t["resolution"] != "attested_register_variants")
            or set(t["flags"]).difference({"secondary_stress"})
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--out", required=True)
    parser.add_argument("--language", choices=["auto", "nb"], default="auto")
    parser.add_argument("--dialect", default=config("dialects")["default"])
    args = parser.parse_args()
    with closing(Lexicon(args.dialect)) as lexicon:
        write_json(args.out, analyze(read_text(args.text), lexicon, args.language))

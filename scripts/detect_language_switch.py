"""Route manuscript spans and report tentative audio-language disagreements."""
import re
from functools import lru_cache

from .common import TOKEN, key, sentences
from .reverse_transcribe import ASR_LOCALES

ANNOTATION = re.compile(r"\[\[(nb-NO|uk-UA|en-US|en-GB|en)\|([^\]]+)\]\]")
QUOTED = re.compile(r"[«“\"]([^»”\"\n]+)[»”\"]")
CYRILLIC = re.compile(r"[\u0400-\u052f]")
NORWEGIAN = re.compile(r"[æøå]", re.IGNORECASE)
HINTS = {"норвезьк": "nb-NO", "українськ": "uk-UA", "англійськ": "en",
         "norsk": "nb-NO", "ukrainsk": "uk-UA", "engelsk": "en",
         "norwegian": "nb-NO", "ukrainian": "uk-UA", "english": "en"}


@lru_cache(maxsize=1)
def detector():
    from lingua import Language, LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_languages(
        Language.BOKMAL, Language.NYNORSK, Language.ENGLISH, Language.UKRAINIAN,
        Language.RUSSIAN, Language.SWEDISH, Language.DANISH, Language.GERMAN,
    ).with_low_accuracy_mode().with_minimum_relative_distance(.20).build()


def locale_of(text, english, short=False):
    words = list(TOKEN.finditer(text))
    if len(words) < 4:
        if NORWEGIAN.search(text) and not CYRILLIC.search(text):
            return "nb-NO", "orthographic_hint"
        if short and CYRILLIC.search(text) and re.search(r"[іїєґ]", text, re.I):
            return "uk-UA", "orthographic_hint"
        return None, "insufficient_context"
    detected = detector().detect_language_of(text)
    code = detected.iso_code_639_1.name.lower() if detected else None
    return (english if code=="en" else ASR_LOCALES.get(code)), "context_text_lid"


def unpack_markup(text, english):
    clean = []
    spans = []
    last, length = 0, 0
    for match in ANNOTATION.finditer(text):
        prefix, content = text[last:match.start()], match[2]
        clean.extend([prefix, content])
        length += len(prefix)
        spans.append({"start": length, "end": length+len(content),
                      "language": english if match[1]=="en" else match[1], "method": "explicit_markup"})
        length += len(content)
        last = match.end()
    clean.append(text[last:])
    return "".join(clean), spans


def role_hints(text, english):
    spans = []
    for match in QUOTED.finditer(text):
        before = key(text[max(0, match.start()-55):match.start()])
        before = re.split(r"[»”\"\n]", before)[-1]
        hints = [(before.rfind(prefix), locale) for prefix, locale in HINTS.items() if prefix in before]
        if hints:
            word_locale = max(hints)[1]
            spans.append({"start": match.start(1), "end": match.end(1),
                          "language": english if word_locale=="en" else word_locale, "method": "quotation_context"})
    return spans


def route(text, fallback="nb-NO", english="en-US", overrides=()):
    spoken, explicit = unpack_markup(text, english)
    doc, doc_method = locale_of(spoken, english)
    blocks = [{"start": m.start(), "end": m.end(), "language": locale_of(m.group(), english)[0]}
              for m in re.finditer(r"\S(?:(?!\n\s*\n)[\s\S])*", spoken)]
    explicit = [{**p, "method": "explicit_offsets"} for p in overrides] + explicit
    for item in explicit:
        if item["language"] not in {"nb-NO", "uk-UA", "en-US", "en-GB"} or not 0 <= item["start"] < item["end"] <= len(spoken):
            raise ValueError("Language spans must use supported locales and valid spoken-text offsets.")
    hints = role_hints(spoken, english)
    tokens, sections, transitions = [], [], []
    for sentence_id, (start, end) in enumerate(sentences(spoken)):
        original = spoken[start:end]
        lexical = locale_of(original, english)
        parent = next((b["language"] for b in blocks if b["start"] <= start < b["end"]), None)
        base = lexical[0] or parent or doc or fallback
        sections.append({"sentence": sentence_id, "start": start, "end": end, "text": original,
                         "expected_language": base})
        for match in TOKEN.finditer(spoken, start, end):
            spans = [p for p in explicit+hints if p["start"] <= match.start() and match.end() <= p["end"]]
            marked = spans[0] if spans else None
            if marked and any(p["language"] != marked["language"] for p in spans
                              if p["method"] in {"explicit_offsets", "explicit_markup"}
                              and marked["method"] in {"explicit_offsets", "explicit_markup"}):
                raise ValueError("Conflicting explicit language spans.")
            language, method = (marked["language"], marked["method"]) if marked else (base, lexical[1] if lexical[0] else "inherited_context")
            script_locale, _ = locale_of(match.group(), english, short=True)
            # A distinct script or Norwegian character can recover a short switch within its context.
            if not marked and script_locale and script_locale != language:
                language, method = script_locale, "orthographic_hint"
            row = {"id": len(tokens), "text": match.group(), "start": match.start(), "end": match.end(),
                   "sentence_id": sentence_id, "sentence": original, "expected_language": language,
                   "detected_language": None, "pronunciation_language": None,
                   "language_confidence": "explicit" if method.startswith("explicit") else "contextual",
                   "intentional_code_switch": marked is not None, "language_method": method}
            if tokens and tokens[-1]["expected_language"] != language:
                previous = tokens[-1]
                transitions.append({"from": previous["expected_language"], "to": language,
                                    "before_id": previous["id"], "after_id": row["id"],
                                    "intentional": previous["intentional_code_switch"] or row["intentional_code_switch"],
                                    "offset": row["start"]})
            tokens.append(row)
    return {"expected_language": doc or fallback, "method": doc_method, "text": spoken, "paragraphs": blocks,
            "sentences": sections, "tokens": tokens, "transitions": transitions}


def unexpected_window(window, route_tokens, matches, threshold=.85):
    findings = []
    in_window = {item["id"] for item in route_tokens}
    expected = {item["expected_language"].split("-")[0] for item in route_tokens}
    for model in window.get("transcripts", []):
        detected = ASR_LOCALES.get(model.get("detected_language"))
        if not detected or len(model["words"]) < 5:
            continue
        # Single-word disagreements, mixed-language cues and inferred quotations are never decisive.
        simple = detected.split("-")[0]
        if len(expected) != 1 or simple in expected or (model.get("language_confidence") or 0) < threshold:
            continue
        supported = [i for i in in_window if i in matches]
        if len(supported) < 4:
            continue
        if any(row["intentional_code_switch"] for row in route_tokens):
            continue
        findings.append({
            "category":"possible_unexpected_language", "start":model["words"][0]["start"], "end":model["words"][-1]["end"],
            "detector":"asr_language_id", "expected_language":route_tokens[0]["expected_language"],
            "detected_language":detected, "timing":"asr_estimate",
            "evidence":{"provider":model["provider"],"language_score":model["language_confidence"],
                        "duration_scope":"window","matched_tokens":len(supported)},
        })
    return findings


def transition_events(routing, timings, audio_features):
    result = []
    for marker in routing["transitions"]:
        before, after = timings.get(marker["before_id"]), timings.get(marker["after_id"])
        if not before or not after:
            continue
        gap = after["start"]-before["end"]
        if gap < .65:
            continue
        token = routing["tokens"][marker["after_id"]]
        result.append({"category":"transition_pause", "start":before["end"], "end":after["start"],
                       "word":token["text"], "sentence":token["sentence"], "expected_language":marker["to"],
                       "intentional_code_switch":marker["intentional"], "timing":"asr_estimate",
                       "detector":"language_transition", "evidence":{"pause_s":gap,"from":marker["from"], "to":marker["to"]}})
    return result

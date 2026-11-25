"""Validate, store and export reviewed pronunciations and scoped ElevenLabs rules."""
import argparse
import csv
import io
import math
import re
import sqlite3
from contextlib import closing
from xml.etree import ElementTree as ET

from .common import ROOT, atomic_bytes, config, data_dir, key, model_config, read_json, write_json

PLS = "http://www.w3.org/2005/01/pronunciation-lexicon"
XML = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("", PLS)
FIELDS = (
    "word", "lemma", "context", "dialect", "ipa", "primary_stress", "secondary_stress",
    "toneme", "source", "error_type", "method", "model", "voice_id", "confidence",
    "acoustic_review", "native_review", "replacement", "scope",
)
LANGUAGE_FIELDS = ("word_language", "context_language", "alternate_pronunciations",
                   "interference_type", "previous_language", "next_language")
LOCALES = {"nb-NO", "uk-UA", "en-US", "en-GB"}
UNSUPPORTED_IPA_MARKERS = re.compile(r"""[<>&/"'¹²0123456789]""")


def validate_record(record):
    missing = set(FIELDS).difference(record)
    if missing:
        raise ValueError(f"Missing review fields: {', '.join(sorted(missing))}.")
    for name in ("word", "lemma", "ipa", "source", "error_type"):
        if not isinstance(record[name], str) or not record[name].strip():
            raise ValueError(f"{name} must be nonempty text.")
    for name in ("context", "voice_id"):
        if record[name] is not None and (not isinstance(record[name], str) or not record[name].strip()):
            raise ValueError(f"{name} must be nonempty text or null.")
    if record["scope"] not in ("global", "sentence") or (
        (record["scope"] == "sentence") != bool(record["context"])
    ):
        raise ValueError("Sentence scope requires the exact context sentence. Global scope requires null context.")
    locale = record.get("word_language", "nb-NO")
    if locale not in LOCALES or any(record.get(name) not in LOCALES | {None}
            for name in ("context_language", "previous_language", "next_language")):
        raise ValueError("Unknown correction locale.")
    if (locale == "nb-NO" and record["dialect"] not in config("dialects")["dialects"]
            or locale != "nb-NO" and record["dialect"] != locale):
        raise ValueError("Unknown dialect.")
    alternatives = record.get("alternate_pronunciations", [])
    if not isinstance(alternatives, list) or any(
        not isinstance(item, str) or not item.strip() or UNSUPPORTED_IPA_MARKERS.search(item) for item in alternatives
    ):
        raise ValueError("Alternate pronunciations must be standard segmental/stress IPA strings.")
    if record.get("interference_type") is not None and not isinstance(record["interference_type"], str):
        raise ValueError("interference_type must be a text description or null.")
    if locale != "nb-NO" and (record["scope"] != "sentence" or record["toneme"] is not None):
        raise ValueError("A multilingual correction requires sentence scope and no Norwegian toneme.")
    if record["toneme"] not in (None, 1, 2) or isinstance(record["toneme"], bool):
        raise ValueError("toneme must be 1, 2 or null.")
    for name in ("primary_stress", "secondary_stress"):
        value = record[name]
        if not isinstance(value, list) or any(type(index) is not int or index < 0 for index in value):
            raise ValueError(f"{name} must contain zero-based integer phone indices.")
    confidence = record["confidence"]
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be a documented number between 0 and 1.")
    model = model_config(record["model"])
    if not model["norwegian"]:
        raise ValueError("Choose a model with official Norwegian support before learning a Norwegian correction.")
    for assessment in ("acoustic_review", "native_review"):
        item = record[assessment]
        if not isinstance(item, dict) or item.get("status") != "confirmed" or not item.get("evidence"):
            raise ValueError(f"{assessment} requires confirmed status and an evidence reference.")
    if UNSUPPORTED_IPA_MARKERS.search(record["ipa"]):
        raise ValueError("Use standard segmental/stress IPA and store lexical tone separately.")
    if record["method"] == "alias":
        if not model["aliases"] or not record["voice_id"]:
            raise ValueError("An A/B-tested alias requires a compatible model and the tested voice.")
        value = record["replacement"]
        if not isinstance(value, str) or not value.strip() or re.search(r"[<>\r\n]", value):
            raise ValueError("An alias requires a nonempty, single-line spoken replacement.")
    elif record["method"] == "phoneme":
        if not model["norwegian_ipa"] or record["replacement"] is not None:
            raise ValueError("A phoneme rule requires documented Norwegian IPA and null replacement.")
    else:
        raise ValueError("method must be phoneme or alias.")
    return {**{name: record[name] for name in FIELDS},
            **{name: record[name] for name in LANGUAGE_FIELDS if name in record}}


def identity(record):
    original = tuple(key(record[name] or "") for name in ("word", "dialect", "model", "voice_id", "context"))
    locale = record.get("word_language", "nb-NO")
    neighbors = tuple(record.get(name) or "" for name in ("context_language", "previous_language", "next_language"))
    return original if locale == "nb-NO" and not any(neighbors) else original + (locale, *neighbors)


def verified_records(seed=None):
    seed = seed if seed is not None else ROOT / "dictionaries/corrections.json"
    indexed = {identity(row): validate_record(row) for row in read_json(seed)}
    db_path = data_dir() / "corrections.sqlite3"
    if db_path.exists():
        with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            import json

            for (serialized,) in db.execute("SELECT record FROM corrections ORDER BY id"):
                record = validate_record(json.loads(serialized))
                indexed[identity(record)] = record
    return list(indexed.values())


def learn(record):
    import json

    record = validate_record(record)
    path = data_dir() / "corrections.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=20)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS corrections (id TEXT PRIMARY KEY, record TEXT NOT NULL)")
        db.execute("INSERT OR REPLACE INTO corrections VALUES (?, ?)",
                   (json.dumps(identity(record), ensure_ascii=False), json.dumps(record, ensure_ascii=False)))
    return record


def make_pls(rules, language="nb-NO"):
    grouped = {}
    for rule in rules:
        name = rule["word"]
        value = rule["type"], rule["value"]
        if name in grouped and grouped[name] != value:
            raise ValueError(f"Conflicting dictionary rules for {name!r}.")
        if not name or not value[1]:
            raise ValueError("PLS rules require a grapheme and value.")
        grouped[name] = value
    tree = ET.Element(f"{{{PLS}}}lexicon", {"version": "1.0", "alphabet": "ipa", f"{{{XML}}}lang": language})
    for word, (kind, phonetic) in sorted(grouped.items()):
        if kind not in ("alias", "phoneme"):
            raise ValueError(f"Unsupported PLS rule type {kind!r}.")
        lexeme = ET.SubElement(tree, f"{{{PLS}}}lexeme")
        ET.SubElement(lexeme, f"{{{PLS}}}grapheme").text = word
        ET.SubElement(lexeme, f"{{{PLS}}}{kind}").text = phonetic
    ET.indent(tree)
    return ET.tostring(tree, encoding="utf-8", xml_declaration=True) + b"\n"


def record_rule(record, word=None):
    kind = record["method"]
    return {"word": word or record["word"], "type": kind,
            "value": record["ipa"] if kind == "phoneme" else record["replacement"],
            "evidence": "verified", "source": record["source"]}


def export(records, destination, fmt, model=None, dialect=None, voice=None):
    if fmt == "json":
        write_json(destination, records)
    elif fmt == "csv":
        import json

        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=FIELDS + LANGUAGE_FIELDS)
        writer.writeheader()
        for row in records:
            writer.writerow({name: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                             else value for name, value in row.items()})
        atomic_bytes(destination, buffer.getvalue().encode())
    elif fmt == "pls":
        if not model or not dialect:
            raise ValueError("PLS export requires --model and --dialect.")
        selected = [row for row in records if row["scope"] == "global" and row["model"] == model
                    and row["dialect"] == dialect and row["voice_id"] in (None, voice)]
        by_word = {}
        for row in sorted(selected, key=lambda row: bool(row["voice_id"])):
            by_word[key(row["word"])] = record_rule(row)
        atomic_bytes(destination, make_pls(list(by_word.values())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("learn")
    add.add_argument("record")
    out = commands.add_parser("export")
    out.add_argument("--format", required=True, choices=["json", "csv", "pls"])
    out.add_argument("--out", required=True)
    out.add_argument("--model")
    out.add_argument("--dialect")
    out.add_argument("--voice")
    args = parser.parse_args()
    if args.command == "learn":
        learn(read_json(args.record))
    else:
        export(verified_records(), args.out, args.format, args.model, args.dialect, args.voice)

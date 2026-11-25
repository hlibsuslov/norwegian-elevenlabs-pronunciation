"""Paths and deterministic local I/O shared by the skill commands."""
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*", re.UNICODE)
TOKEN = re.compile(
    r"\d{1,4}(?:[./:-]\d{1,4}){1,3}|\d+(?:[ ,]\d+)?[%€$]?|" + WORD.pattern
)


def config(name):
    return yaml.safe_load((ROOT / "config" / f"{name}.yaml").read_text(encoding="utf-8"))


def data_dir():
    return Path(os.environ.get("NELP_DATA_DIR", ROOT / ".data")).expanduser().resolve()


def read_text(path):
    # Preserve source newlines; offsets refer to Unicode code points.
    return Path(path).read_bytes().decode("utf-8-sig")


def read_json(path):
    return json.loads(read_text(path))


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def key(value):
    return unicodedata.normalize("NFC", value).casefold()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sentences(source):
    """Conservative boundaries. Retain exact positions and protect common abbreviations."""
    begin = 0
    for match in re.finditer(r"[.!?]+[»”\"')\]]*(?=\s|$)|(?:\r?\n){2,}", source):
        punctuation = match.group()
        previous = re.search(r"(\w+)$", source[begin:match.start()])
        if punctuation == "." and previous:
            word = previous.group(1)
            abbreviations = {"bl", "a", "f", "eks", "ca", "kl", "kr", "nr", "jf", "dr", "mm", "osv", "dvs"}
            if key(word) in abbreviations or len(word) == 1 or word.isdecimal():
                continue
        end = match.start() if "\n" in punctuation else match.end()
        fragment = source[begin:end]
        if fragment.strip():
            yield begin + len(fragment) - len(fragment.lstrip()), begin + len(fragment.rstrip())
        begin = match.end()
    fragment = source[begin:]
    if fragment.strip():
        yield begin + len(fragment) - len(fragment.lstrip()), begin + len(fragment.rstrip())


def model_config(model):
    models = config("models")["models"]
    if model not in models:
        raise ValueError(f"Unknown model {model!r}. Choose from {', '.join(models)}.")
    return models[model]

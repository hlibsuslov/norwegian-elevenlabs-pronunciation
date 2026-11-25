"""Read source-attested pronunciations from a local NB Uttale index."""
import argparse
import json
import re
import sqlite3
from contextlib import closing
from functools import lru_cache

from .common import config, data_dir, key
from .setup_lexicon import NB


def pronunciation(row, dialect):
    raw = row["ipa_transcription"]
    nuclei = [
        {"phone_index": i, "phone": match[1], "stress": int(match[2])}
        for i, phone in enumerate(row["nofabet_transcription"].split())
        if (match := re.fullmatch(r"([A-Z]+)([0-3])", phone))
    ]
    primary = [n["phone_index"] for n in nuclei if n["stress"] in (1, 2)]
    return {
        "wordform": row["wordform"], "pos": row["pos"], "features": row["feats"],
        "wordform_ids": row["wordform_id"].split("|"), "source_update": row["update_info"],
        "source": NB, "dialect": dialect, "nb_ipa": raw,
        "ipa": raw.replace("'", "ˈ").replace('"', "ˈ").replace("_", " "),
        "nofabet": row["nofabet_transcription"], "xsampa": row["sampa_transcription"],
        "nuclei": nuclei, "primary_stress_phone_indices": primary,
        "secondary_stress_phone_indices": [n["phone_index"] for n in nuclei if n["stress"] == 3],
        "tonemes": [n["stress"] for n in nuclei if n["stress"] in (1, 2)],
        "long_vowel_markers": raw.count("ː"),
        "retroflex_segments": [phone for phone in "ɖɭɳʂʈ" if phone in raw],
    }


class Lexicon:
    def __init__(self, dialect, path=None):
        if dialect not in config("dialects")["dialects"]:
            raise ValueError(f"Unknown NB Uttale dialect {dialect!r}.")
        self.dialect = dialect
        self.path = path or data_dir() / f"{dialect}.sqlite3"
        if not self.path.is_file():
            raise ValueError(f"Run python -m scripts.setup_lexicon --dialect {dialect}. Missing {self.path}")
        self.db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        if self.db.execute("SELECT dialect FROM provenance").fetchone()[0] != dialect:
            self.close()
            raise ValueError("The index provenance does not match the chosen dialect.")

    def close(self):
        self._lookup.cache_clear()
        self.db.close()

    @lru_cache(maxsize=8192)
    def _lookup(self, normalized):
        rows = self.db.execute("SELECT * FROM entries WHERE lookup = ? ORDER BY rowid", (normalized,))
        return [pronunciation(row, self.dialect) for row in rows if row["ipa_transcription"]]

    def lookup(self, word):
        return self._lookup(key(word).replace(" ", "_"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("word", nargs="+")
    parser.add_argument("--dialect", default=config("dialects")["default"])
    args = parser.parse_args()
    with closing(Lexicon(args.dialect)) as lexicon:
        print(json.dumps(lexicon.lookup(" ".join(args.word)), ensure_ascii=False, indent=2))

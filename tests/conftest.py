import sqlite3

import pytest

from scripts.common import key
from scripts.lookup_pronunciation import Lexicon
from scripts.setup_lexicon import COLUMNS, NB, NB_SHA


@pytest.fixture
def synthetic_lexicon(tmp_path):
    # This wholly synthetic record exists for offsets and scoping tests only.
    path = tmp_path / "synthetic.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE entries (lookup TEXT, " + ", ".join(c + " TEXT" for c in COLUMNS) + ")")
        db.execute("CREATE TABLE provenance (dialect TEXT, sha256 TEXT, license TEXT, source TEXT)")
        db.execute("INSERT INTO provenance VALUES (?, ?, ?, ?)", ("e_written", NB_SHA, "synthetic test fixture", NB))
        db.execute("INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (key("Testsymbol"), "Testsymbol", "NN", "", "T1", "synthetic fixture",
                    "T EE1 S T AH3", "'teːst.ˌɑ", ""))
    db.close()
    lexicon = Lexicon("e_written", path)
    yield lexicon
    lexicon.close()

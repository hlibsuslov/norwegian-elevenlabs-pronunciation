"""Download immutable CC0 resources and build an indexed NB Uttale dialect."""
import argparse
import csv
import io
import os
import shutil
import sqlite3
import tarfile
from contextlib import closing
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from filelock import FileLock

from .common import config, data_dir, file_digest, key

NB = "https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-79/"
NB_URL = "https://www.nb.no/sbfil/uttaleleksikon/nb_uttale_leksika.zip"
NB_SHA = "6e87e9e99a6081a5fc4164077750652880f4a663c860f50c3d45b18ba003bc80"
NOFA_URL = "https://www.nb.no/sbfil/verktoy/nofa/NoFA-1_0.tar.gz"
NOFA_SHA = "ae23a5f0f79fdb3c4f78be5da5f960fcfd4bfa83f144baeeaa8101c99ccd17b8"
COLUMNS = (
    "wordform", "pos", "feats", "wordform_id", "update_info",
    "nofabet_transcription", "ipa_transcription", "sampa_transcription",
)


def verified(path, sha):
    if file_digest(path) != sha:
        raise ValueError(f"SHA-256 mismatch for {path}. Use the documented release.")


def download(url, destination, sha):
    if not destination.exists():
        temporary = destination.with_suffix(".part")
        try:
            with urlopen(url, timeout=60) as src, temporary.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            verified(temporary, sha)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    verified(destination, sha)
    return destination


def index_lexicon(archive, dialect, target):
    member = f"nb_uttale_leksika/{dialect}_pronunciation_lexicon.csv"
    temporary = target.with_suffix(".part")
    temporary.unlink(missing_ok=True)
    try:
        with ZipFile(archive) as zip_file, zip_file.open(member) as raw:
            with closing(sqlite3.connect(temporary)) as db, db:
                rows = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
                if tuple(rows.fieldnames or ()) != COLUMNS:
                    raise ValueError(f"Unexpected NB Uttale CSV schema in {member}.")
                db.execute("CREATE TABLE entries (lookup TEXT, " + ", ".join(column + " TEXT" for column in COLUMNS) + ")")
                db.executemany(
                    "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ((key(row["wordform"]), *(row[column] for column in COLUMNS)) for row in rows),
                )
                db.execute("CREATE INDEX lookup_index ON entries (lookup)")
                db.execute("CREATE TABLE provenance (dialect TEXT, sha256 TEXT, license TEXT, source TEXT)")
                db.execute("INSERT INTO provenance VALUES (?, ?, ?, ?)", (dialect, NB_SHA, "CC0-1.0", NB))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def setup(dialects, archive=None, nofa=False, nofa_archive=None):
    root = data_dir()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / "setup.lock", timeout=600):
        if dialects:
            if archive:
                verified(archive, NB_SHA)
            else:
                archive = download(NB_URL, root / "nb_uttale_leksika.zip", NB_SHA)
            for dialect in dialects:
                target = root / f"{dialect}.sqlite3"
                if not target.exists():
                    index_lexicon(archive, dialect, target)
                print(target)
        if nofa:
            if nofa_archive:
                verified(nofa_archive, NOFA_SHA)
            else:
                nofa_archive = download(NOFA_URL, root / "NoFA-1_0.tar.gz", NOFA_SHA)
            nofa_dir = root / "nofa"
            nofa_dir.mkdir(exist_ok=True)
            with tarfile.open(nofa_archive) as tar:
                for name in ("NoFA/model/NoFA_1-0.zip", "NoFA/lexicon/NoFA-lex.dict"):
                    destination = nofa_dir / Path(name).name
                    if not destination.exists():
                        temporary = destination.with_suffix(".part")
                        with tar.extractfile(name) as src, temporary.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                        os.replace(temporary, destination)
                    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialect", action="append", choices=config("dialects")["dialects"])
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--nofa", action="store_true")
    parser.add_argument("--nofa-only", action="store_true")
    parser.add_argument("--nofa-archive", type=Path)
    args = parser.parse_args()
    dialects = [] if args.nofa_only else args.dialect or [config("dialects")["default"]]
    setup(dialects, args.archive, args.nofa or args.nofa_only, args.nofa_archive)

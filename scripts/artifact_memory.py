"""Keep only explicitly confirmed artifact examples, with review and audio references."""
import json
import sqlite3
from contextlib import closing

from .common import data_dir, digest


def store(event, decision, original_clip, repaired_clip=None, metadata=None):
    if decision.get("status") not in {"CONFIRMED","VERIFIED","FALSE_POSITIVE"} or not decision.get("reviewed_at"):
        raise ValueError("Artifact memory requires an explicit dated review decision.")
    root=data_dir()/"artifact_memory"
    root.mkdir(exist_ok=True,parents=True)
    source=original_clip.read_bytes()
    replacement=repaired_clip.read_bytes() if repaired_clip else None
    identity=digest([event["id"],event["audio_file"],decision.get("status")])
    with closing(sqlite3.connect(root/"examples.sqlite3",timeout=20)) as database,database:
        database.execute("CREATE TABLE IF NOT EXISTS examples (id TEXT PRIMARY KEY, record TEXT, before BLOB, after BLOB)")
        serialized=json.dumps({"event":event,"decision":decision,"generation":metadata or {}},ensure_ascii=False,allow_nan=False)
        database.execute("INSERT OR REPLACE INTO examples VALUES (?,?,?,?)",(identity,serialized,source,replacement))
    return identity

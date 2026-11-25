"""Validate imported independent phoneme evidence before using it for assessment."""
import math

from .common import read_json


def observed_phones(path):
    if not path:
        return []
    payload = read_json(path)
    records = []
    for row in payload:
        start, end = row["start"],row["end"]
        if (not math.isfinite(start) or not math.isfinite(end) or end<=start or start<0
                or not row.get("model") or not row.get("alphabet") or not row.get("phones")):
            raise ValueError("Independent phoneme evidence needs timing, model, alphabet and phone sequence.")
        records.append(row)
    return records

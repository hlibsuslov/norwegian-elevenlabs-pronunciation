"""Persist versioned human decisions and the audio actually selected by the reviewer."""
import argparse
from collections import defaultdict
from datetime import datetime
from math import isfinite
from pathlib import Path
import re

import soundfile as sf
from filelock import FileLock

from scripts.common import digest, read_json, write_json
from scripts.segment_audio import read_samples
from .research_benchmark import broad
from .research_models import normalized_vectors


LABEL = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
IDENTITY = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
STATUSES = {"CONFIRMED", "FALSE_POSITIVE", "CLEAN", "HARD_NEGATIVE", "AMBIGUOUS"}


def partition(status, category):
    if status == "CONFIRMED":
        return {"artifact": "confirmed_artifacts", "pronunciation": "confirmed_pronunciation_errors",
                "cross_language_pronunciation": "language_interference"}.get(broad(category), "ambiguous_cases")
    return {"CLEAN": "clean_controls", "HARD_NEGATIVE": "hard_negatives",
            "FALSE_POSITIVE": "hard_negatives", "AMBIGUOUS": "ambiguous_cases"}[status]


class ReviewDataset:
    def __init__(self, root, report, proxy):
        self.root = Path(root)
        self.report = report
        self.proxy = Path(proxy)
        self.recording = report.get("review_run")
        if not report.get("dataset_review") or not self.recording or not IDENTITY.fullmatch(self.recording):
            raise ValueError("A research report with a valid recording identifier is required.")
        manifest = read_json(self.root/"metadata"/f"{self.recording}-manifest.json")
        if report["source"]["sha256"] != manifest["source"]["sha256"]:
            raise ValueError("Report and dataset refer to different sources.")
        self.directory = self.root/"annotations"/"decisions"/self.recording
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cases = {row["id"]: row for row in report["events"]}
        with sf.SoundFile(self.proxy) as source:
            self.rate, self.count = source.samplerate, len(source)
        if self.rate != report["sample_rate"] or self.count != round(report["duration_s"]*self.rate):
            raise ValueError("Review proxy does not match the dataset sample clock.")

    def records(self):
        return [read_json(path) for path in self.directory.glob("*.json")]

    def decisions(self):
        return {row["id"]: {**row["history"][-1], "persisted": True} for row in self.records()}

    def save(self, request):
        identity = request.get("id")
        status, category = request.get("status"), request.get("category")
        if not isinstance(identity, str) or not IDENTITY.fullmatch(identity) or identity not in self.cases:
            raise ValueError("Unknown dataset case.")
        if request.get("recording_id") != self.recording:
            raise ValueError("Decision recording ID does not match this dataset.")
        if status not in STATUSES or not isinstance(category, str) or not LABEL.fullmatch(category):
            raise ValueError("Use a supported status and a lowercase category.")
        timestamp = request.get("reviewed_at")
        if not isinstance(timestamp, str) or datetime.fromisoformat(timestamp.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("Use a timezone-aware ISO timestamp.")
        start, end = request.get("start"), request.get("end")
        if any(type(value) not in (int, float) or not isfinite(value) for value in (start, end)):
            raise ValueError("The interval must contain two finite numbers in seconds.")
        first, last = round(start*self.rate), round(end*self.rate)
        if not 0 <= first < last <= self.count or (last-first)/self.rate > 120:
            raise ValueError("Choose an interval within the source of at most 120 seconds.")
        comments = request.get("comments", "")
        if not isinstance(comments, str) or len(comments) > 4000:
            raise ValueError("Review notes must contain at most 4000 characters.")
        case = self.cases[identity]
        revision = {"id": identity, "status": status, "category": category, "recording_id": self.recording,
                    "start": first/self.rate, "end": last/self.rate, "reviewed_at": timestamp, "comments": comments}
        destination = self.directory/f"{identity}.json"
        with FileLock(self.directory/".review.lock", timeout=15):
            history = read_json(destination)["history"] if destination.exists() else []
            if revision not in history:
                location = self.root/"review"/"clips"/identity
                audio = location/f"decision-{digest(revision)[:14]}.wav"
                location.mkdir(parents=True, exist_ok=True)
                if not audio.exists():
                    temp = audio.with_suffix(".part.wav")
                    sf.write(temp, read_samples(self.proxy, first, last), self.rate, subtype="FLOAT")
                    temp.replace(audio)
                revision["audio"] = audio.relative_to(self.root).as_posix()
                if not any({k: v for k, v in prior.items() if k != "audio"} ==
                           {k: v for k, v in revision.items() if k != "audio"} for prior in history):
                    history.append(revision)
                    history.sort(key=lambda row: datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00")))
            current = history[-1]
            same_interval = (round(case["start"]*self.rate), round(case["end"]*self.rate)) == (
                round(current["start"]*self.rate), round(current["end"]*self.rate))
            record = {"id": identity, "source_sha256": case["source_sha256"], "recording_id": self.recording,
                      "category": current["category"], "status": current["status"],
                      "start": current["start"], "end": current["end"],
                      "partition": partition(current["status"], current["category"]),
                      "source": case["audio_file"], "text": case.get("sentence"),
                      "language": case.get("expected_language"), "dictionary_targets": case.get("expected_pronunciation"),
                      "voice_id": case.get("voice_id"), "model_id": case.get("model_id"),
                      "generation_settings": case.get("generation_settings"),
                      "measurements": case.get("fingerprint") if same_interval else None,
                      "embedding": f"embeddings/{identity}.npy" if same_interval and
                      (self.root/"embeddings"/f"{identity}.npy").is_file() else None,
                      "embedding_model_sha256": case.get("embedding_model_sha256") if same_interval else None,
                      "detector_results": case.get("evidence"), "repairs": case.get("repair_experiments", []),
                      "history": history}
            write_json(destination, record)
            self.reindex()
        return record

    def reindex(self):
        groups = defaultdict(list)
        for record in self.records():
            groups[record["partition"]].append(record)
        for name in ("confirmed_artifacts", "confirmed_pronunciation_errors", "language_interference",
                     "clean_controls", "hard_negatives", "ambiguous_cases"):
            write_json(self.root/name/f"{self.recording}-reviewed.json", sorted(groups[name], key=lambda row: row["start"]))


def find_examples(root, vector_path, source, limit, model_sha=None):
    import numpy as np

    query = np.load(vector_path, allow_pickle=False)
    candidates, vectors = [], []
    for path in (root/"annotations"/"decisions").glob("*/*.json"):
        record = read_json(path)
        if record["status"] not in {"CONFIRMED", "CLEAN", "HARD_NEGATIVE"} or record["source_sha256"] == source:
            continue
        if model_sha and record.get("embedding_model_sha256") != model_sha:
            continue
        filename = record.get("embedding")
        if filename and (root/filename).is_file():
            candidate = np.load(root/filename, allow_pickle=False)
            if candidate.shape == query.shape:
                vectors.append(candidate)
                candidates.append(record)
    if not vectors:
        return []
    similarity = normalized_vectors(np.asarray(vectors)) @ normalized_vectors(query[None, :])[0]
    return [{**candidates[i], "cosine_similarity": float(similarity[i])}
            for i in np.argsort(similarity)[::-1][:limit]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    for name in ("dataset", "report", "proxy", "decisions"):
        ingest.add_argument(f"--{name}", type=Path, required=True)
    similar = sub.add_parser("similar")
    similar.add_argument("--dataset", type=Path, required=True)
    similar.add_argument("--embedding", type=Path, required=True)
    similar.add_argument("--exclude-source-sha", required=True)
    similar.add_argument("--model-sha", required=True)
    similar.add_argument("--limit", type=int, default=5)
    similar.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "similar":
        write_json(args.out, find_examples(args.dataset, args.embedding, args.exclude_source_sha, args.limit, args.model_sha))
    else:
        dataset = ReviewDataset(args.dataset, read_json(args.report), args.proxy)
        rows = read_json(args.decisions)
        for row in rows:
            dataset.save(row)
        print(f"Imported {len(rows)} decisions for {dataset.recording}.")


if __name__ == "__main__":
    main()

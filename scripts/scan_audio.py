"""Compute resumable window observations before evidence fusion and review export."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from filelock import FileLock

from .common import digest, file_digest, read_json, write_json
from .detect_artifacts import scan_signal
from .reverse_transcribe import globalize, model_instance, scribe, transcript
from .segment_audio import decode, probe, read_samples, windows


def source_signature(path):
    return {"path": str(path.resolve()), "sha256": file_digest(path)}


def model_signature(location):
    path = Path(location).resolve()
    if not path.exists():
        raise ValueError(f"Local ASR model does not exist: {path}")
    files = list(path.iterdir()) if path.is_dir() else [path]
    return [str(path), sorted((f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files if f.is_file())]


def scan_raw(path, output, *, providers=(), device="cpu", workers=1, deep=False, sensitivity="balanced",
             signal=True, expected_locale=None, resume=False, scribe_key=None, cloud_seconds=0):
    path = Path(path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / "scan.lock", timeout=0):
        return _scan_raw(path, output, providers, device, workers, deep, sensitivity, signal,
                         expected_locale, resume, scribe_key, cloud_seconds)


def _scan_raw(path, output, providers, device, workers, deep, sensitivity, signal,
              expected_locale, resume, scribe_key, cloud_seconds):
    source = source_signature(path)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    manifest = cache / "manifest.json"
    identity = {"algorithm": 1, "source": source, "models": [[kind, model_signature(location)] for kind, location in providers],
                "deep": deep, "sensitivity": sensitivity, "signal": signal,
                "device": device, "locale": expected_locale, "scribe": bool(scribe_key), "cloud_seconds": cloud_seconds}
    prior = read_json(manifest) if manifest.is_file() else None
    if prior and not resume:
        raise ValueError("Run directory exists. Use --resume with the same options or choose a new one.")
    if prior and prior["identity"] != read_compatible(identity):
        raise ValueError("Cached run settings or inputs changed. Choose a new run directory.")
    proxy = cache / "decoded.wav"
    if not proxy.is_file():
        decode(path, proxy)
    with sf.SoundFile(proxy) as input_audio:
        count, rate = len(input_audio), input_audio.samplerate
    if not count:
        raise ValueError("The audio stream contains no decoded samples.")
    schedule = list(windows(proxy))
    if not prior:
        info = probe(path)
        write_json(manifest, {"identity": identity, "source": source, "media": info,
                              "samples": count, "sample_rate": rate, "windows": [w.data() for w in schedule]})
    else:
        info = prior["media"]
    for kind, location in providers:
        model_instance(kind, str(location), device)
    cloud_indices = set()
    accumulated = 0.0
    if scribe_key:
        for window in schedule:
            duration = (window.end-window.start)/rate
            if accumulated + duration > cloud_seconds:
                break
            accumulated += duration
            cloud_indices.add(window.index)

    def process(window):
        file = cache / f"window-{window.index:06d}.json"
        cloud_attempt = cache / f"cloud-{window.index:06d}.json"
        if file.is_file():
            return
        values = read_samples(proxy, window.start, window.end)
        events, features = scan_signal(values, rate, window.start, sensitivity, deep) if signal else ([], {})
        transcriptions = []
        global_locale = None
        for kind, location in providers:
            specialist = kind in {"nb", "nb-verbatim", "nb-whisperx"}
            if specialist and global_locale not in {None, "no", "nb", "nn"}:
                continue
            if specialist and not global_locale and expected_locale not in {None, "nb-NO"}:
                continue
            model_result = transcript(kind, location, values, rate, "no" if specialist else None, device)
            global_locale = global_locale or model_result["detected_language"]
            transcriptions.append(model_result)
        if window.index in cloud_indices:
            if cloud_attempt.is_file():
                prior_cloud = read_json(cloud_attempt)
                if prior_cloud.get("state") == "done":
                    transcriptions.append(prior_cloud["result"])
            else:
                # A partially completed paid request must not be repeated on resume.
                write_json(cloud_attempt, {"state": "attempted"})
                model_result = scribe(values, rate, scribe_key, "character" if deep else "word")
                write_json(cloud_attempt, {"state": "done", "result": model_result})
                transcriptions.append(model_result)
        for row in transcriptions:
            row["words"] = globalize(row["words"], window.start, window.core_start, window.core_end, rate)
            row["audio_events"] = [
                {**event, "start": event["start"]+window.start/rate, "end": event["end"]+window.start/rate}
                for event in row.get("audio_events", []) if event.get("start") is not None and event.get("end") is not None
            ]
        write_json(file, {**window.data(), "events": events, "features": features, "transcripts": transcriptions})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done in pool.map(process, schedule):
            pass
    return {"source": source, "proxy": proxy, "media": info, "sample_rate": rate, "samples": count,
            "windows": [read_json(cache / f"window-{window.index:06d}.json") for window in schedule],
            "cloud_duration_requested": accumulated,
            "cloud_unknown_outcomes": sum(read_json(path).get("state") != "done" for path in cache.glob("cloud-*.json"))}


def read_compatible(value):
    import json
    return json.loads(json.dumps(value))

"""Run independent local reverse ASR, preserving word and window provenance."""
import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import soundfile as sf

from scripts.common import file_digest, read_json, write_json
from scripts.reverse_transcribe import audio_16k, globalize, item_words, reconcile_words
from scripts.segment_audio import read_samples, windows
from .research import audio_bounds, labels


def cuda_libraries():
    folder = Path(sys.prefix)/"Lib/site-packages/nvidia"
    directories = list(folder.glob("*/bin"))
    if os.name != "nt" or not directories:
        return []
    os.environ["PATH"] = os.pathsep.join([*(str(item) for item in directories), os.environ["PATH"]])
    return [os.add_dll_directory(str(item)) for item in directories]


def recognize(args):
    from faster_whisper import WhisperModel

    begin = perf_counter()
    baseline = read_json(args.baseline)
    rate = baseline["sample_rate"]
    duration = baseline["duration_s"]
    with sf.SoundFile(args.proxy) as audio:
        count = len(audio)
        if rate != audio.samplerate or count != round(rate * duration):
            raise ValueError("Decoded audio does not match the baseline sample clock.")
    handles = cuda_libraries() if args.device == "cuda" else []
    try:
        model = WhisperModel(str(args.model), device=args.device, compute_type=args.compute_type,
                             cpu_threads=args.cpu_threads, local_files_only=True)
        points = labels(args.focus_annotations, duration) if args.focus_annotations else []
        scopes = [(float(row["point_s"])-5, float(row["point_s"])+5, None, row["id"]) for row in points]
        if not args.focus_annotations:
            scopes = [(window.start/rate, window.end/rate, (window.core_start, window.core_end), str(window.index))
                      for window in windows(args.proxy)]
        checkpoint = args.out.with_suffix("") / "windows"
        checkpoint.mkdir(parents=True, exist_ok=True)
        model_assets = {path.name: file_digest(path) for path in args.model.iterdir() if path.is_file()}
        settings = {"device": args.device, "compute_type": args.compute_type, "model_assets_sha256": model_assets,
                    "model_revision": args.revision, "language_conditioning": args.language, "beam_size": 2,
                    "source_sha256": baseline["source"]["sha256"], "scopes": scopes}
        manifest = checkpoint.parent/"manifest.json"
        if manifest.exists():
            from scripts.scan_audio import read_compatible

            if read_json(manifest) != read_compatible(settings):
                raise ValueError("ASR settings changed. Choose a new output path.")
        else:
            write_json(manifest, settings)
        rows, observed_languages = [], []
        for index, (a, b, core, identity) in enumerate(scopes):
            cached = checkpoint/f"{index:04d}.json"
            if cached.exists():
                result = read_json(cached)
            else:
                values, start, end = audio_bounds(args.proxy, rate, count, a, b)
                signal = audio_16k(values, rate)
                segments, info = model.transcribe(signal, language=args.language, beam_size=2, vad_filter=True,
                                                  word_timestamps=True, condition_on_previous_text=False)
                local = [item for segment in segments for word in segment.words or []
                         for item in item_words(word.word, word.start, word.end, float(word.probability))]
                origin = round(start*rate)
                left, right = core or (origin, round(end*rate))
                result = {"words": globalize(local, origin, left, right, rate), "language": info.language,
                          "language_probability": float(info.language_probability) if not args.language else None,
                          "language_conditioning": args.language, "start": start, "end": end, "scope": identity}
                write_json(cached, result)
            rows.extend(result["words"])
            observed_languages.append({key: value for key, value in result.items() if key != "words"})
            if (index+1) % 10 == 0:
                print(f"{args.name} {index+1}/{len(scopes)}", flush=True)
        write_json(args.out, {"source": baseline["source"], "reverse_transcript": {args.name: reconcile_words(rows)},
                              "window_languages": observed_languages, "windows": len(scopes), "settings": settings,
                              "targeted_annotation_windows": bool(args.focus_annotations), "api_credits": 0,
                              "processing_seconds": perf_counter()-begin})
    finally:
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("proxy", "baseline", "model", "out"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--focus-annotations", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--language")
    parser.add_argument("--cpu-threads", type=int, default=4)
    recognize(parser.parse_args())


if __name__ == "__main__":
    main()

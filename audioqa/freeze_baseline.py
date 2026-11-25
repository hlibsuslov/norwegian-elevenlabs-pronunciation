"""Freeze a blind report and its reproducibility metadata without overwriting prior evidence."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

from scripts.common import file_digest, read_json
from scripts.segment_audio import ff


def freeze(report_path, cache, out, command, model_revision):
    report = read_json(report_path)
    manifest = read_json(cache)
    if manifest["source"]["sha256"] != report["source"]["sha256"]:
        raise ValueError("Run manifest and scan report refer to different sources.")
    if report["captions_authoritative"]:
        report["baseline_input_note"] = "The scan used an authoritative manuscript or subtitles."
    sources = ("scan_audio", "detect_artifacts", "detect_clicks", "detect_clipping", "detect_dropouts",
               "refine_signal", "extract_audio_features", "evidence_fusion", "segment_audio", "reverse_transcribe")
    root = Path(__file__).resolve().parents[1]
    code = [*(root/"scripts"/f"{name}.py" for name in sources), root/"audioqa"/"report.py"]
    report["baseline_protocol"] = {
        "command": command,
        "settings": manifest["identity"], "model_revision": model_revision,
        "code_sha256": {path.relative_to(root).as_posix(): file_digest(path) for path in code},
        "packages": {name: version(name) for name in ("numpy", "scipy", "praat-parselmouth", "faster-whisper", "soundfile")},
        "ffmpeg": ff(["ffmpeg","-version"],timeout=15).stdout.decode("utf-8").splitlines()[0],
        "annotation_input": None, "original_report_sha256": file_digest(report_path),
        "point_matching_protocol": {"primary_tolerance_s": 2.5, "sensitivity_tolerances_s": [1, 5],
                                    "distance": "predicted interval midpoint to approximate point",
                                    "assignment": "maximum cardinality one-to-one, then minimum total distance"},
    }
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+"\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    guard = out.with_suffix(".sha256")
    if guard.exists():
        raise FileExistsError(guard)
    with out.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    with guard.open("x", encoding="utf-8") as handle:
        handle.write(f"{file_digest(out)}  {out.name}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "cache-manifest", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--model-revision", action="append", default=[])
    args = parser.parse_args()
    freeze(args.report, args.cache_manifest, args.out, args.command, args.model_revision)


if __name__ == "__main__":
    main()

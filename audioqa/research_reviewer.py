"""Audition blinded clips through a local llama.cpp audio model."""
import argparse
import base64
import io
from pathlib import Path
from time import perf_counter

import requests
import soundfile as sf

from scripts.common import read_json, write_json
from scripts.reverse_transcribe import audio_16k
from .research import audio_bounds


PROMPT = (
    "Transcribe the short recording in its original language(s). Then say whether a clear audible glitch is "
    "present, absent, or uncertain, with a brief reason. Natural speech sounds and language switches are valid."
)


def review_clip(session, url, values, rate):
    wav = io.BytesIO()
    sf.write(wav, audio_16k(values, rate), 16000, format="WAV", subtype="PCM_16")
    speech = base64.b64encode(wav.getvalue()).decode()
    payload = {"model": "local", "temperature": 0, "seed": 37, "max_tokens": 230,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": [
                   {"type": "input_audio", "input_audio": {"data": speech, "format": "wav"}},
                   {"type": "text", "text": PROMPT}]}]}
    response = session.post(url, json=payload, timeout=180)
    response.raise_for_status()
    result = response.json()
    choice = result["choices"][0]
    return {"answer": choice["message"].get("content"), "finish_reason": choice["finish_reason"],
            "runtime": result.get("system_fingerprint"), "usage": result.get("usage")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8913)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--controls", type=int, default=3)
    for key in ("manifest", "proxy", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    by_annotation = {row["annotation_id"]: row for row in manifest["cases"] if "annotation_id" in row}
    subset = [by_annotation[number] for number in args.case]
    subset += [row for row in manifest["cases"] if row["category"] == "control_candidate"][:args.controls]
    with sf.SoundFile(args.proxy) as source:
        rate, count = source.samplerate, len(source)
    rows = []
    start = perf_counter()
    with requests.Session() as session:
        session.trust_env = False
        for case in sorted(subset, key=lambda row: row["id"]):
            center = (case["start"] + case["end"]) / 2
            values, a, b = audio_bounds(args.proxy, rate, count, center-2.6, center+2.6)
            result = review_clip(session, f"http://127.0.0.1:{args.port}/v1/chat/completions", values, rate)
            rows.append({"id": case["id"], "annotation_id": case.get("annotation_id"), "interval": [a,b], **result})
            write_json(args.out, {"source_sha256": manifest["source"]["sha256"], "prompt": PROMPT,
                                  "model": "Gemma 4 E2B IT Q4_0 with Q8_0 multimodal projector",
                                  "model_revision": "2a8538c2e2d4687953a86c422c7c623cfd022880",
                                  "results": rows, "processed": len(rows), "selected": len(subset),
                                  "processing_seconds": perf_counter()-start, "api_credits": 0})
            print(f"Local audio review {len(rows)}/{len(subset)}", flush=True)


if __name__ == "__main__":
    main()

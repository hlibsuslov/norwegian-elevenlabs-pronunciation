"""Decode locally and schedule bounded, overlapping media windows in sample coordinates."""
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

MEDIA = {".wav", ".mp3", ".flac", ".m4a", ".mp4", ".aac", ".ogg", ".webm"}


def ff(command, **options):
    try:
        return subprocess.run(command, check=True, capture_output=True, **options)
    except FileNotFoundError as exc:
        raise ValueError(f"Install {command[0]} and add it to PATH.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace").splitlines()[-1:] if exc.stderr else []
        raise ValueError(f"{command[0]} failed: {' '.join(detail)}") from exc


def probe(path):
    result = ff(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], timeout=45)
    metadata = json.loads(result.stdout)
    audio = next((stream for stream in metadata["streams"] if stream["codec_type"] == "audio"), None)
    if audio is None:
        raise ValueError(f"No audio track in {path.name}.")
    return {name: audio.get(name) for name in ("codec_name", "sample_rate", "channels", "start_time", "duration")}


def decode(path, proxy):
    proxy.parent.mkdir(parents=True, exist_ok=True)
    temp = proxy.with_name(f"{proxy.stem}.part.wav")
    try:
        ff(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path), "-map", "0:a:0", "-vn",
            "-c:a", "pcm_f32le", "-rf64", "auto", str(temp)], timeout=7200)
        temp.replace(proxy)
    finally:
        temp.unlink(missing_ok=True)


def read_samples(path, start, end):
    with sf.SoundFile(path) as audio:
        audio.seek(start)
        return audio.read(end-start, dtype="float32", always_2d=True)


@dataclass(frozen=True)
class Window:
    index: int
    start: int
    end: int
    core_start: int
    core_end: int

    def data(self):
        return asdict(self)


def db_level(values):
    return 20 * np.log10(np.maximum(1e-8, np.sqrt(np.mean(values**2, axis=0))))


def windows(path, target_seconds=24, overlap_seconds=3):
    with sf.SoundFile(path) as audio:
        count, rate = len(audio), audio.samplerate
    start = 0
    index = 0
    margin = round(overlap_seconds * rate / 2)
    while start < count:
        end = min(count, start + round(target_seconds * rate))
        if count-end <= 10 * rate:
            end = count if count-start <= 30 * rate else min(end, count-10*rate)
        if end < count:
            search = round(3 * rate)
            lo, hi = max(start+10*rate, end-search), min(count-10*rate, end+search)
            probe_values = read_samples(path, lo, hi)
            frame = round(0.05*rate)
            quiet = []
            for at in range(frame, len(probe_values)-frame, frame):
                level = float(np.max(db_level(probe_values[at-frame:at+frame])))
                if level < -42:
                    quiet.append((abs(lo+at-end), lo+at))
            if quiet:
                end = min(quiet)[1]
        yield Window(index, max(0, start-margin), min(count, end+margin), start, end)
        index += 1
        start = end


def positive_runs(mask):
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    boundaries = np.flatnonzero(np.diff(padded))
    return zip(boundaries[::2], boundaries[1::2])


def discover_audio(path):
    path = Path(path).resolve()
    if path.is_file() and path.suffix.lower() in MEDIA:
        return [path]
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in MEDIA)
    raise ValueError(f"No supported local media at {path}.")

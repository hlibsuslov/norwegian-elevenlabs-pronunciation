"""Norwegian Bokmål NoFA/MFA 1 TextGrid and independent Whisper timestamps."""
import shutil
import subprocess
import tempfile
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from .common import TOKEN, data_dir, key


def normalized(text):
    return key(text).strip("-'’")


def word_mapping(expected, observed):
    left = [normalized(t) for t in expected]
    right = [normalized(t) for t in observed]
    mapping = {}
    differences = []
    matcher = SequenceMatcher(a=left, b=right, autojunk=False)
    for kind, start, end, first, last in matcher.get_opcodes():
        if kind == "equal":
            mapping.update(zip(range(start, end), range(first, last)))
        else:
            differences.append({"kind": kind, "expected": [start, end], "observed": [first, last]})
    return mapping, differences


def textgrid(path):
    import parselmouth
    from parselmouth.praat import call

    grid = parselmouth.Data.read(str(path))
    tiers = {}
    for index in range(1, call(grid, "Get number of tiers") + 1):
        if not call(grid, "Is interval tier", index):
            continue
        name = call(grid, "Get tier name", index).lower().strip()
        values = []
        for item in range(1, call(grid, "Get number of intervals", index) + 1):
            label = call(grid, "Get label of interval", index, item).strip()
            if label and label.lower() not in {"sil", "sp"}:
                values.append({"text": label, "start": call(grid, "Get start time of interval", index, item),
                               "end": call(grid, "Get end time of interval", index, item)})
        tiers[name.rsplit(" - ", 1)[-1]] = values
    if not tiers.get("words"):
        raise ValueError(f"{path} requires a nonempty words interval tier.")
    return {"words": tiers["words"], "phones": tiers.get("phones", []), "source": "NoFA_TextGrid"}


def run_nofa(audio, source, command):
    root = data_dir() / "nofa"
    lexicon, model = root / "NoFA-lex.dict", root / "NoFA_1-0.zip"
    if not lexicon.is_file() or not model.is_file():
        raise ValueError("Run python -m scripts.setup_lexicon --nofa-only before running NoFA.")
    with tempfile.TemporaryDirectory(prefix="nelp-nofa-") as job:
        incoming, outgoing = Path(job) / "input", Path(job) / "output"
        incoming.mkdir()
        if not shutil.which("ffmpeg"):
            raise ValueError("NoFA conversion requires FFmpeg.")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(Path(audio).resolve()), "-ac", "1",
             "-ar", "16000", "-c:a", "pcm_s16le", str(incoming / "sample.wav")],
            check=True, capture_output=True, timeout=120,
        )
        words = []
        for token in TOKEN.finditer(source):
            surface = token.group()
            # Unknown written numbers remain unmatched and are flagged in the report.
            words.append(surface.lower())
        (incoming / "sample.lab").write_text(" ".join(words), encoding="utf-8")
        try:
            subprocess.run([command, str(incoming), str(lexicon), str(model), str(outgoing),
                            "-t", str(Path(job) / "working"), "-j", "1", "-q"],
                           check=True, capture_output=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"NoFA failed ({type(exc).__name__}). Use an MFA 1.0 runner or provide its TextGrid.") from exc
        output = list(outgoing.rglob("sample.TextGrid"))
        if len(output) != 1:
            raise ValueError("NoFA did not produce exactly one sample.TextGrid.")
        return textgrid(output[0])


@lru_cache(maxsize=2)
def whisper_model(directory):
    from faster_whisper import WhisperModel

    return WhisperModel(directory, device="cpu", compute_type="int8", local_files_only=True)


def transcribe(audio, model_directory):
    import numpy as np

    directory = Path(model_directory).resolve()
    if not directory.is_dir():
        raise ValueError("--whisper-model expects a locally downloaded multilingual CTranslate2 Whisper directory.")
    if not shutil.which("ffmpeg"):
        raise ValueError("Whisper audio decoding requires FFmpeg.")
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(audio), "-vn", "-ac", "1",
         "-ar", "16000", "-f", "f32le", "pipe:1"],
        check=True, capture_output=True, timeout=180,
    )
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not samples.size:
        raise ValueError("The input contains no audio for Whisper.")
    segments, info = whisper_model(str(directory)).transcribe(
        samples, language="no", word_timestamps=True, condition_on_previous_text=False,
    )
    words = []
    for segment in segments:
        for word in segment.words or ():
            matches = list(TOKEN.finditer(word.word))
            for match in matches:
                words.append({"text": match.group(), "start": word.start, "end": word.end,
                              "probability": word.probability})
    return {"words": words, "language": info.language, "source": "multilingual_whisper"}

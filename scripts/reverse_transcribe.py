"""Local CT2/OpenAI/Transformers Whisper and opt-in ElevenLabs Scribe adapters."""
import io
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly

from .common import TOKEN, key
from .generate_audio import BASE

LOCALES = {"nb-NO": "no", "uk-UA": "uk", "en-US": "en", "en-GB": "en"}
ASR_LOCALES = {"no": "nb-NO", "nn": "nb-NO", "nb": "nb-NO", "nor": "nb-NO", "nob": "nb-NO",
               "uk": "uk-UA", "ukr": "uk-UA", "en": "en", "eng": "en"}
FAMILIES = {"faster": "whisper", "openai": "whisper", "transformers": "whisper",
            "nb": "nb_whisper", "nb-verbatim": "nb_whisper", "nb-whisperx": "nb_whisper"}


def audio_16k(values, rate):
    mono = values.mean(axis=1).astype(np.float32)
    if rate == 16000:
        return mono
    divisor = math.gcd(rate, 16000)
    return resample_poly(mono, 16000//divisor, rate//divisor).astype(np.float32)


@lru_cache(maxsize=8)
def model_instance(kind, location, device):
    if kind == "openai":
        import whisper
        return whisper.load_model(location, device=device)
    if kind in {"transformers", "nb", "nb-verbatim", "nb-whisperx"} and not (Path(location) / "model.bin").is_file():
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        model = AutoModelForSpeechSeq2Seq.from_pretrained(location, local_files_only=True)
        processor = AutoProcessor.from_pretrained(location, local_files_only=True)
        return pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                        feature_extractor=processor.feature_extractor, device=0 if device=="cuda" else -1)
    from faster_whisper import WhisperModel

    return WhisperModel(location, device=device, compute_type="float16" if device=="cuda" else "int8",
                        cpu_threads=2, local_files_only=True)


def item_words(text, start, end, probability=None):
    return [{"text": token.group(), "start": float(start), "end": float(end), "confidence": probability}
            for token in TOKEN.finditer(text) if start is not None and end is not None and end > start]


def transcript(kind, location, values, rate, language, device):
    model = model_instance(kind, str(location), device)
    audio = audio_16k(values, rate)
    speech = []
    detected, confidence = None, None
    if kind == "openai":
        result = model.transcribe(audio, language=language, word_timestamps=True,
                                  fp16=device=="cuda", condition_on_previous_text=False)
        detected = result.get("language")
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                speech.extend(item_words(word["word"], word["start"], word["end"], word.get("probability")))
    elif kind in {"transformers", "nb", "nb-verbatim", "nb-whisperx"} and not (Path(location) / "model.bin").is_file():
        kwargs = {"task": "transcribe"}
        if language:
            kwargs["language"] = language
        result = model({"array": audio, "sampling_rate": 16000}, return_timestamps="word", generate_kwargs=kwargs)
        for word in result.get("chunks", []):
            speech.extend(item_words(word["text"], *word["timestamp"]))
    else:
        segments, meta = model.transcribe(audio, language=language, word_timestamps=True, vad_filter=True,
                                          condition_on_previous_text=False, beam_size=2)
        detected = meta.language if language is None else None
        confidence = float(meta.language_probability) if detected else None
        for segment in segments:
            for word in segment.words or []:
                speech.extend(item_words(word.word, word.start, word.end, float(word.probability)))
    if language is not None:
        detected = None
    return {"provider": kind, "family": FAMILIES[kind], "words": speech, "detected_language": detected,
            "language_confidence": confidence, "language_conditioning": language}


def scribe(values, rate, key_value, granularity="word"):
    audio = io.BytesIO()
    sf.write(audio, audio_16k(values, rate), 16000, format="WAV", subtype="PCM_16")
    response = requests.post(BASE + "/speech-to-text", headers={"xi-api-key": key_value}, timeout=(10, 120),
                             allow_redirects=False, files={"file": ("clip.wav", audio.getvalue(), "audio/wav")},
                             data={"model_id": "scribe_v2", "timestamps_granularity": granularity,
                                   "tag_audio_events": "true", "diarize": "false"})
    if response.status_code != 200:
        raise RuntimeError(f"Scribe returned HTTP {response.status_code}.")
    items = response.json()
    result = []
    events = []
    for token in items.get("words", []):
        if token.get("type") == "word":
            for word in item_words(token["text"], token.get("start"), token.get("end")):
                result.append({**word, "characters": token.get("characters")})
        elif token.get("type") == "audio_event":
            events.append({k: token.get(k) for k in ("start", "end", "text")})
    return {"provider": "scribe_v2", "family": "scribe", "words": result, "audio_events": events,
            "detected_language": items.get("language_code"), "language_confidence": items.get("language_probability")}


def globalize(words, start, core_start, core_end, rate):
    accepted = []
    for word in words:
        w_start = max(start/rate, start/rate + word["start"])
        w_end = start/rate + word["end"]
        center = (w_start+w_end)/2
        if core_start/rate <= center < core_end/rate:
            row = {**word, "start": w_start, "end": min(w_end, (core_end/rate)+2)}
            if word.get("characters"):
                row["characters"] = [
                    {**char, "start": char["start"] + start/rate, "end": char["end"] + start/rate}
                    for char in word["characters"] if char.get("start") is not None and char.get("end") is not None
                ]
            accepted.append(row)
    return accepted


def reconcile_words(words):
    result = []
    for word in sorted(words, key=lambda row: (row["start"], row["end"])):
        if result:
            prior = result[-1]
            intersection = min(prior["end"], word["end"])-max(prior["start"], word["start"])
            shortest = min(prior["end"]-prior["start"], word["end"]-word["start"])
            if key(prior["text"]) == key(word["text"]) and intersection >= 0.4*shortest:
                if (word.get("confidence") or 0) > (prior.get("confidence") or 0):
                    result[-1] = word
                continue
        result.append(word)
    return result

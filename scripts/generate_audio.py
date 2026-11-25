"""Budgeted, cached ElevenLabs synthesis through the documented model transports."""
import argparse
import base64
import hashlib
import io
import json
import os
import re
import time
import wave
from collections import defaultdict
from dataclasses import dataclass, field
from urllib.parse import quote, urlencode

import requests
from dotenv import dotenv_values
from filelock import FileLock
from websockets.sync.client import connect

from .build_dictionary import make_pls
from .common import ROOT, atomic_bytes, config, data_dir, digest, model_config, read_json, write_json

BASE = "https://api.elevenlabs.io/v1"
WS = "wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input"
SAMPLE_RATE = 24000


class APIError(RuntimeError):
    pass


class BudgetExceeded(APIError):
    pass


def credentials(env_file=None):
    values = dotenv_values(env_file or ROOT / ".env", interpolate=False)
    file_key = values.get("ELEVENLABS_API_KEY") or values.get("XI_API_KEY")
    environment_key = os.getenv("ELEVENLABS_API_KEY") or os.getenv("XI_API_KEY")
    api_key = file_key if env_file else environment_key or file_key
    voice = (values.get("ELEVENLABS_VOICE_ID") if env_file
             else os.getenv("ELEVENLABS_VOICE_ID") or values.get("ELEVENLABS_VOICE_ID"))
    return api_key, voice


@dataclass
class Budget:
    max_chars: int
    max_requests: int
    attempted_chars: int = 0
    synthesis_requests: int = 0
    dictionary_uploads: int = 0
    reported_character_costs: list = field(default_factory=list)
    unknown_cost_requests: int = 0
    cache_hits: int = 0

    def check(self, chars, requests_count=1):
        if (self.attempted_chars + chars > self.max_chars
                or self.synthesis_requests + requests_count > self.max_requests):
            raise BudgetExceeded("The configured synthesis character or request budget would be exceeded.")

    def reserve(self, chars):
        self.check(chars)
        self.attempted_chars += chars
        self.synthesis_requests += 1

    def report(self):
        costs = self.reported_character_costs
        return {
            "reserved_character_units": self.attempted_chars, "synthesis_requests": self.synthesis_requests,
            "dictionary_uploads": self.dictionary_uploads, "cache_hits": self.cache_hits,
            "provider_reported_character_cost": sum(costs) if costs else None,
            "requests_without_reported_cost": self.unknown_cost_requests,
            "credits_used": 0 if not self.synthesis_requests else None,
        }


def units(text):
    return len(text.encode("utf-16-le")) // 2


def query_chars(segment, rules, model):
    text = segment["text"]
    if model_config(model)["transport"] != "dialogue_ws":
        text += segment.get("previous_text", "").strip() + segment.get("next_text", "").strip()
    count = units(text)
    for rule in rules:
        # Overestimates matching by allowing substring occurrences and using the entire replacement.
        count += text.count(rule["word"]) * units(rule["value"])
    return count


def pcm_wav(pcm):
    if not pcm or len(pcm) % 2:
        raise APIError("ElevenLabs did not return valid 16-bit PCM audio.")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        out.writeframes(pcm)
    return buffer.getvalue()


def locator(data):
    if not data.get("id") or not data.get("version_id"):
        raise APIError("The pronunciation dictionary response lacked a version locator.")
    return {"pronunciation_dictionary_id": data["id"], "version_id": data["version_id"]}


def request_body(model, voice, segment, locators, settings, seed):
    transport = model_config(model)["transport"]
    body = {"model_id": model, "seed": seed, "apply_text_normalization": "auto"}
    language = segment.get("language_code", "no" if model_config(model)["norwegian"] else None)
    if language:
        body["language_code"] = language
    if locators:
        body["pronunciation_dictionary_locators"] = locators
    if transport == "dialogue":
        body["inputs"] = [{"text": segment["text"], "voice_id": voice}]
        if settings:
            body["settings"] = settings
    else:
        body["text"] = segment["text"]
        if settings:
            body["voice_settings"] = settings
    if segment.get("previous_text", "").strip():
        body["previous_text"] = segment["previous_text"].strip()
    if segment.get("next_text", "").strip():
        body["future_text" if transport == "dialogue" else "next_text"] = segment["next_text"].strip()
    return body


class ElevenLabs:
    def __init__(self, budget, api_key=None, session=None, connect_ws=connect):
        if api_key is None:
            api_key, _ = credentials()
        if not api_key:
            raise APIError("Set ELEVENLABS_API_KEY before running live synthesis.")
        self.headers = {"xi-api-key": api_key}
        self.cache = data_dir() / "elevenlabs" / hashlib.sha256(api_key.encode()).hexdigest()[:24]
        self.cache.mkdir(parents=True, exist_ok=True)
        self.http = session or requests.Session()
        self.connect_ws = connect_ws
        self.budget = budget

    def post(self, path, **kwargs):
        try:
            return self.http.post(BASE + path, headers=self.headers, timeout=(10, 120),
                                  allow_redirects=False, **kwargs)
        except requests.RequestException as exc:
            raise APIError(f"ElevenLabs {type(exc).__name__}. Request outcome may be unknown.") from None

    @staticmethod
    def json_response(response):
        if response.status_code != 200:
            raise APIError(f"ElevenLabs returned HTTP {response.status_code}. Check model access, quota and voice settings.")
        try:
            result = response.json()
        except ValueError:
            raise APIError("ElevenLabs returned an invalid JSON response.") from None
        if not isinstance(result, dict):
            raise APIError("ElevenLabs returned an unexpected response object.")
        return result

    def dictionary(self, rules, language="nb-NO"):
        name = digest([language, [(r["word"], r["type"], r["value"]) for r in rules]])
        file = self.cache / f"pls-{name}.json"
        with FileLock(file.with_suffix(".lock"), timeout=180):
            if file.exists():
                return read_json(file)
            response = self.post(
                "/pronunciation-dictionaries/add-from-file",
                data={"name": f"norwegian-pronunciation-{name[:12]}"},
                files={"file": ("pronunciation.pls", make_pls(rules, language), "application/pls+xml")},
            )
            result = locator(self.json_response(response))
            self.budget.dictionary_uploads += 1
            write_json(file, result)
            return result

    def websocket(self, model, voice, segment, locators, settings):
        parameters = {"model_id": model, "sync_alignment": "true",
                      "output_format": config("models")["output_format"]}
        language = segment.get("language_code", "no")
        if language:
            parameters["language_code"] = language
        query = urlencode(parameters)
        initial = {"voices": [voice]}
        if settings:
            initial["voice_settings"] = settings
        if locators:
            initial["pronunciation_dictionary_locators"] = locators
        output = bytearray()
        events = []
        deadline = time.monotonic() + 120
        try:
            with self.connect_ws(
                WS + "?" + query, additional_headers=self.headers, open_timeout=10, close_timeout=3,
            ) as connection:
                connection.send(json.dumps(initial))
                connection.send(json.dumps({"inputs": [{"text": segment["text"], "voice_id": voice}]}))
                connection.send(json.dumps({"close_socket": True}))
                while time.monotonic() < deadline:
                    message = json.loads(connection.recv(timeout=min(25, max(0.01, deadline - time.monotonic()))))
                    if message.get("error"):
                        raise APIError("ElevenLabs rejected the dialogue WebSocket request.")
                    if message.get("audio"):
                        output.extend(base64.b64decode(message["audio"], validate=True))
                    if message.get("alignment"):
                        events.append(message["alignment"])
                    if message.get("is_final"):
                        return bytes(output), {"alignment_chunks": events, "request_id": None}
        except APIError:
            raise
        except Exception as exc:
            raise APIError(f"ElevenLabs WebSocket {type(exc).__name__}. Request outcome may be unknown.") from None
        raise APIError("ElevenLabs dialogue WebSocket timed out.")

    def cache_path(self, model, voice, segment, rules, settings, seed, take=0):
        request = {"text": segment["text"], "language_code": segment.get("language_code", "no")}
        if model_config(model)["transport"] != "dialogue_ws":
            request.update({name: segment.get(name, "").strip() for name in ("previous_text", "next_text")})
        values = [(row["word"], row["type"], row["value"], row.get("language", "nb-NO")) for row in rules]
        identity = digest([3, model, voice, request, values, settings or {}, seed, take, config("models")["output_format"]])
        return self.cache / f"audio-{identity}.json"

    def estimate(self, model, voice, segment, rules, settings, seed, take=0):
        if self.cache_path(model, voice, segment, rules, settings, seed, take).is_file():
            return 0, 0
        return query_chars(segment, rules, model), 1

    def generate(self, model, voice, segment, rules, settings=None, seed=0, take=0):
        spec = model_config(model)
        if not spec["norwegian"]:
            raise ValueError(f"{model} does not officially support Norwegian.")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", voice or ""):
            raise ValueError("A valid ElevenLabs voice ID is required.")
        if not segment["text"].strip() or units(segment["text"]) > spec["limit"]:
            raise ValueError("The synthesis segment is empty or exceeds the model request limit.")
        if any(rule["type"] not in ("alias", "phoneme") or (
            rule["type"] == "phoneme" and not spec["norwegian_ipa"]
        ) or (rule["type"] == "alias" and (not spec["aliases"] or rule.get("evidence") != "verified"))
            for rule in rules
        ):
            raise ValueError("The rules contain an unsupported phoneme or an unverified alias.")
        chars = query_chars(segment, rules, model)
        cache_path = self.cache_path(model, voice, segment, rules, settings, seed, take)
        with FileLock(cache_path.with_suffix(".lock"), timeout=180):
            if cache_path.exists():
                cached = read_json(cache_path)
                self.budget.cache_hits += 1
                return base64.b64decode(cached.pop("pcm"), validate=True), {**cached, "cached": True}
            self.budget.check(chars)
            groups = defaultdict(list)
            for rule in rules:
                groups[rule.get("language", "nb-NO")].append(rule)
            if len(groups) > 3:
                raise ValueError("ElevenLabs accepts at most three pronunciation dictionaries per request.")
            locators = [self.dictionary(rows, language) for language, rows in groups.items()]
            self.budget.reserve(chars)
            if spec["transport"] == "dialogue_ws":
                self.budget.unknown_cost_requests += 1
                pcm, info = self.websocket(model, voice, segment, locators, settings)
            else:
                path = ("/text-to-dialogue/with-timestamps" if spec["transport"] == "dialogue"
                        else f"/text-to-speech/{quote(voice, safe='')}/with-timestamps")
                try:
                    response = self.post(path, params={"output_format": config("models")["output_format"]},
                                         json=request_body(model, voice, segment, locators, settings, seed))
                except APIError:
                    self.budget.unknown_cost_requests += 1
                    raise
                charge = response.headers.get("character-cost", "")
                if str(charge).isdigit():
                    self.budget.reported_character_costs.append(int(charge))
                else:
                    self.budget.unknown_cost_requests += 1
                result = self.json_response(response)
                try:
                    pcm = base64.b64decode(result["audio_base64"], validate=True)
                except (KeyError, TypeError, ValueError):
                    raise APIError("ElevenLabs returned audio in an unexpected format.") from None
                info = {"alignment": result.get("alignment"), "normalized_alignment": result.get("normalized_alignment"),
                        "request_id": response.headers.get("request-id")}
            pcm_wav(pcm)
            write_json(cache_path, {**info, "pcm": base64.b64encode(pcm).decode("ascii")})
            return pcm, {**info, "cached": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("--out", required=True)
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-characters", type=int)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--env-file")
    args = parser.parse_args()
    plan = read_json(args.plan)
    if not 0 <= args.segment < len(plan["segments"]):
        parser.error("--segment is outside the plan.")
    segment = plan["segments"][args.segment]
    if not args.live:
        write_json(args.out, {"mode": "dry_run", "transport": plan["transport"],
                              "reserved_character_units": query_chars(segment, segment["rules"], plan["model"]),
                              "segment": segment, "credits_used": 0})
    else:
        if not args.max_characters or args.max_characters < 1 or not args.max_requests or args.max_requests < 1:
            parser.error("--live requires positive --max-characters and --max-requests.")
        api_key, env_voice = credentials(args.env_file)
        api = ElevenLabs(Budget(args.max_characters, args.max_requests), api_key=api_key or "")
        voice = plan.get("voice_id") or env_voice
        pcm, _ = api.generate(plan["model"], voice, segment, segment["rules"])
        atomic_bytes(args.out, pcm_wav(pcm))

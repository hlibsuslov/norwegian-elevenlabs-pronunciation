import base64
import json
import wave
from io import BytesIO
from xml.etree import ElementTree as ET

import pytest

from scripts.build_dictionary import PLS, export, learn, make_pls, verified_records
from scripts.generate_audio import Budget, BudgetExceeded, ElevenLabs, pcm_wav, request_body


def record(**updates):
    row = {
        "word": "X & Y", "lemma": "x & y", "context": None, "dialect": "e_written",
        "ipa": "ˈɑː", "primary_stress": [0], "secondary_stress": [], "toneme": 1,
        "source": "synthetic test", "error_type": "stress", "method": "phoneme",
        "model": "eleven_v4", "voice_id": None, "confidence": 0.8,
        "acoustic_review": {"status": "confirmed", "evidence": "synthetic A/B"},
        "native_review": {"status": "confirmed", "evidence": "synthetic review"},
        "replacement": None, "scope": "global",
    }
    return {**row, **updates}


def test_learning_gates_and_exports(tmp_path, monkeypatch):
    monkeypatch.setenv("NELP_DATA_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="native_review"):
        learn(record(native_review={"status": "unreviewed"}))
    with pytest.raises(ValueError, match="Norwegian IPA"):
        learn(record(model="eleven_v3"))
    with pytest.raises(ValueError, match="tested voice"):
        learn(record(model="eleven_flash_v2_5", method="alias", replacement="X og Y"))
    learn(record())
    learn(record(scope="sentence", context="X & Y."))
    learn(record(confidence=0.9))
    records = verified_records()
    assert len(records) == 2
    for fmt in ("json", "csv", "pls"):
        export(records, tmp_path / f"rules.{fmt}", fmt, "eleven_v4", "e_written")
    assert len(json.loads((tmp_path / "rules.json").read_text(encoding="utf8"))) == 2
    assert "confidence" in (tmp_path / "rules.csv").read_text(encoding="utf8")
    assert ET.parse(tmp_path / "rules.pls").findtext(f".//{{{PLS}}}grapheme") == "X & Y"
    assert len(ET.parse(tmp_path / "rules.pls").getroot()) == 1


def test_pls_rejects_competing_values():
    with pytest.raises(ValueError, match="Conflicting"):
        make_pls([{"word": "x", "type": "alias", "value": value} for value in ("a", "b")])


class Response:
    status_code = 200
    headers = {"character-cost": "9"}

    def __init__(self, value):
        self.value = value

    def json(self):
        return self.value


class MockSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("add-from-file"):
            assert kwargs["files"]["file"][0] == "pronunciation.pls"
            return Response({"id": "dictionary", "version_id": "version"})
        assert kwargs["json"]["pronunciation_dictionary_locators"] == [
            {"pronunciation_dictionary_id": "dictionary", "version_id": "version"}]
        return Response({"audio_base64": base64.b64encode(b"\0" * 4800).decode("ascii")})


def test_http_dictionary_cache_and_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("NELP_DATA_DIR", str(tmp_path))
    http = MockSession()
    budget = Budget(50, 1)
    api = ElevenLabs(budget, "unit-test", session=http)
    sentence = {"text": "X og Y."}
    rules = [{"word": "X", "type": "phoneme", "value": "ˈɑː", "evidence": "attested_ipa_trial"}]
    raw, source = api.generate("eleven_v4", "voice", sentence, rules)
    assert not source["cached"]
    assert api.generate("eleven_v4", "voice", sentence, rules)[1]["cached"]
    assert len(http.calls) == 2
    assert http.calls[1][0].endswith("/text-to-dialogue/with-timestamps")
    assert http.calls[1][1]["json"]["inputs"] == [{"voice_id": "voice", "text": "X og Y."}]
    with wave.open(BytesIO(pcm_wav(raw))) as audio:
        assert audio.getframerate() == 24000 and audio.getnframes() == 2400
    with pytest.raises(BudgetExceeded):
        api.generate("eleven_v4", "voice", {"text": "Unntak."}, rules)
    assert budget.report()["provider_reported_character_cost"] == 9


def test_documented_payload_for_flash():
    payload = request_body("eleven_flash_v2_5", "voice", {"text": "Hei.", "next_text": "Hallo."}, [], {}, 42)
    assert payload["text"] == "Hei."
    assert payload["next_text"] == "Hallo."
    assert payload["language_code"] == "no"
    assert "inputs" not in payload


def test_dialogue_websocket(tmp_path, monkeypatch):
    monkeypatch.setenv("NELP_DATA_DIR", str(tmp_path))
    messages = []
    class Socket:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def send(self, text):
            messages.append(json.loads(text))
        def recv(self, timeout):
            return '{"audio": "AAAAAA==", "is_final": true}'
    def connect(url, **kwargs):
        assert "model_id=eleven_v4_turbo" in url
        return Socket()
    api = ElevenLabs(Budget(10, 1), "unit-test", connect_ws=connect)
    audio, _ = api.generate("eleven_v4_turbo", "voice", {"text": "Hei."}, [])
    assert audio == b"\0" * 4
    assert messages == [{"voices": ["voice"]}, {"inputs": [{"text": "Hei.", "voice_id": "voice"}]},
                        {"close_socket": True}]

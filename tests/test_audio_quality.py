from copy import deepcopy

import numpy as np
import pytest

from scripts.alignment import textgrid
from scripts.analyze_audio import analyze_recording, transcription_findings
from scripts.detect_stress_errors import detect
from scripts.merge_audio import splice, wav_bytes
from scripts.validate_audio import compare


def source_audio(path, hz=180, seconds=2):
    rate = 24000
    x = np.arange(int(seconds * rate)) / rate
    values = (0.25 * np.sin(2*np.pi*hz*x) * ((x >= 0.2) & (x <= seconds - 0.2)))[None, :]
    path.write_bytes(wav_bytes(values, rate))


def expected_analysis():
    return {"tokens": [{"id": 0, "text": "test", "sentence": 0, "target": None},
                       {"id": 1, "text": "lyd", "sentence": 0, "target": None}]}


def test_praat_textgrid_and_acoustic_measurements(tmp_path):
    import parselmouth
    from parselmouth.praat import call

    path = tmp_path / "original.wav"
    source_audio(path)
    file = tmp_path / "test.TextGrid"
    grid = call("Create TextGrid", 0, 2, "words phones", "")
    call(grid, "Insert boundary", 1, 1.0)
    call(grid, "Set interval text", 1, 1, "test")
    call(grid, "Set interval text", 1, 2, "lyd")
    call(grid, "Insert boundary", 2, 1.0)
    call(grid, "Set interval text", 2, 1, "EE1")
    call(grid, "Set interval text", 2, 2, "YY2")
    call(grid, "Save as text file", str(file))
    timing = textgrid(file)
    output = analyze_recording(path, expected_analysis(), timing)
    assert output["median_f0_hz"] == pytest.approx(180, rel=0.025)
    assert output["aligned_words"][0]["duration_s"] == 1
    assert output["aligned_words"][0]["syllables"][0]["nucleus"]["phone"] == "EE"
    assert output["aligned_words"][1]["syllables"][0]["nucleus"]["stress_code"] == 2
    assert detect(expected_analysis(), output)["confirmed_errors"] == 0


def test_transcript_flags_repetition_but_not_stress():
    observed = {"source": "mock_asr", "words": [{"text": name, "start": i, "end": i+1}
                for i, name in enumerate(["test", "test", "lyd"])]}
    report = transcription_findings(expected_analysis(), observed)
    assert report["issues"][0]["type"] == "possible_repetition"
    assert report["lexical_edit_count"] == 1


def test_compare_rejects_failed_or_incomplete_edits():
    baseline = {"scores": {0: {"features": {"relative_duration_log": 2, "f0_shape_st": 2}},
                            1: {"features": {"relative_duration_log": 0, "f0_shape_st": 0}}}}
    revised = deepcopy(baseline)
    revised["scores"][0]["features"] = {"relative_duration_log": 1, "f0_shape_st": 1}
    signal = {"peak_dbfs": -2, "pauses": [], "aligned_words": [{"id": 0}, {"id": 1}],
              "asr": {"source": "mock_asr", "lexical_edit_count": 0, "issues": []}}
    assert compare(baseline, revised, signal, signal, [0])["accepted"]
    assert not compare(baseline, revised, signal, {**signal, "asr": None}, [0])["accepted"]
    worse = deepcopy(revised)
    worse["scores"][1]["features"]["relative_duration_log"] = 1
    assert not compare(baseline, worse, signal, signal, [0])["accepted"]
    worse = deepcopy(revised)
    del worse["scores"][1]
    assert not compare(baseline, worse, signal, signal, [0])["accepted"]


def test_splice_checks_quiet_gaps_and_preserves_source(tmp_path):
    recording, replacement, output = [tmp_path / f"{name}.wav" for name in ("baseline", "patch", "final")]
    source_audio(recording)
    source_audio(replacement, hz=205, seconds=1.7)
    original = recording.read_bytes()
    assert splice(recording, replacement, output, (0, 0), (1.82, 1.94))["crossfade_seconds"] > 0
    assert output.read_bytes() != original
    assert recording.read_bytes() == original
    with pytest.raises(ValueError, match="low-energy"):
        splice(recording, replacement, output, (0.4, 0.5), (1.82, 1.94))

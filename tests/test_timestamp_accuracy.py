from scripts.benchmark_detectors import benchmark
from scripts.forced_alignment import subtitle_cues
from scripts.generate_review_dashboard import timestamp
from scripts.reverse_transcribe import globalize


def test_decode_clock_offsets_preserve_character_positions():
    speech=[{"text":"hei","start":1.10,"end":1.3,"characters":[{"text":"h","start":1.1,"end":1.14}]}]
    found,=globalize(speech,start=10000,core_start=10000,core_end=15000,rate=1000)
    assert found["characters"][0]["start"]==found["start"]==11.1
    assert timestamp(found["start"],True)=="00:00:11,100"


def test_one_truth_only_matches_one_event_by_category():
    predicted=[{"id":"one","start":10,"end":10.2,"category":"clipping"},
               {"id":"two","start":10.05,"end":10.21,"category":"clipping"},
               {"id":"three","start":11,"end":11.3,"category":"possible_stress_mismatch"}]
    truth=[{"start":10.01,"end":10.15,"category":"clipping"},{"start":15,"end":15.15,"category":"dropout"}]
    result=benchmark(predicted,truth)
    assert (result["tp"],result["fp"],result["fn"])==(1,2,1)
    assert result["recall"]==.5
    assert result["f1"]==.4


def test_webvtt_settings_track_time_without_loading_html(tmp_path):
    path=tmp_path/"captions.vtt"
    path.write_text("WEBVTT\n\n00:01.000 --> 00:03.100 align:start\n<i>Hei.</i>\n",encoding="utf-8")
    cues=subtitle_cues(path)
    assert cues==[{"start":1,"end":3.1,"text":"Hei."}]

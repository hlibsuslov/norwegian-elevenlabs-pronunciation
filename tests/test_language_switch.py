from audioqa.mixed import segments
from scripts.common import ROOT, read_text
from scripts.detect_language_switch import route, unexpected_window
from scripts.generate_audio import request_body


def test_annotated_mixed_corpus_preserves_shared_surface_words_with_different_languages():
    routing=route(read_text(ROOT/"examples/mixed_input.txt"))
    gift=[w for w in routing["tokens"] if w["text"]=="gift"]
    assert [(w["expected_language"],w["intentional_code_switch"]) for w in gift]==[("nb-NO",True),("en-US",True)]
    assert {w["expected_language"] for w in routing["tokens"]}=={"nb-NO","uk-UA","en-GB","en-US"}
    assert all(w["pronunciation_language"] is None for w in routing["tokens"])
    assert "[[" not in routing["text"]
    assert len(routing["transitions"])>=12
    all_text=segments(routing,"single","eleven_v4","voice")
    assert all_text[0]["language_code"] is None
    assert "language_code" not in request_body("eleven_v4","voice",all_text[0],[],None,0)
    split=segments(routing,"language","eleven_v4","voice")
    assert len(split)>len(all_text)
    assert {chunk["language_code"] for chunk in split}>={"no","uk","en"}


def test_intentional_english_quote_does_not_trigger_an_unplanned_switch():
    routing=route("Сьогодні це пояснення «[[en-GB|I can read it here]]».")
    words=routing["tokens"]
    p={"words":[{"start":0,"end":1}]*7,"detected_language":"en","language_confidence":.99,"provider":"fixture"}
    assert not unexpected_window({"transcripts":[p]},words,{w["id"]:w for w in words})

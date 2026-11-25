from pathlib import Path

import pytest

from scripts.analyze_text import analyze, language_assessment
from scripts.common import ROOT, data_dir, key, read_text, sentences
from scripts.lookup_pronunciation import Lexicon
from scripts.workflow import baseline_segments
from scripts.optimize_text import prepare


def test_original_offsets_and_numbers_are_preserved(synthetic_lexicon):
    source = "Testsymbol. \r\nDen 17. mai kl. 08.30 kostet det 1 250 kr."
    report = analyze(source, synthetic_lexicon, language="nb")
    assert report["source"] == source
    assert len(report["utterances"]) == 2
    assert all(source[t["start"]:t["end"]] == t["text"] for t in report["tokens"])
    assert any(t["text"] == "08.30" and t["resolution"] == "model_text_normalization" for t in report["tokens"])
    assert report["tokens"][0]["morphology"]["lemma"]
    assert report["tokens"][0]["target"]["wordform_ids"] == ["T1"]
    assert all(t["target"] is None for t in report["tokens"] if t["text"][0].isdecimal())


def test_language_detector_distinguishes_non_norwegian():
    report = language_assessment("This sentence contains enough words for a preliminary language comparison with Norwegian.", "auto")
    assert report["detected"] == "en"
    assert language_assessment("Kort tekst.", "auto")["detected"] is None


def test_oversized_manuscript_splits_at_boundaries():
    source = ("Dette er en prøve med mange ord. " * 180).strip()
    utterances = [{"id": i, "start": a, "end": b, "text": source[a:b]} for i, (a, b) in enumerate(sentences(source))]
    plan = prepare({"tokens": [], "utterances": utterances, "source": source, "dialect": "e_written"}, "eleven_v4")
    chunks = list(baseline_segments(plan))
    assert len(chunks) >= 3
    assert " ".join(chunk["text"] for chunk in chunks) == source
    assert all(chunk["text"].endswith(".") and len(chunk["text"]) <= 2000 for chunk in chunks)


def test_official_corpus_coverage():
    if not (data_dir() / "e_written.sqlite3").is_file():
        pytest.skip("Install the official CC0 e_written index with scripts.setup_lexicon.")
    lexicon = Lexicon("e_written")
    try:
        report = analyze(read_text(ROOT / "examples/input.txt"), lexicon, language="nb")
        observed = {key(row["text"]): row for row in report["tokens"]}
        required = """
        arbeidsmiljø arbeidsledighet universitetet informasjon kommunikasjon økonomi samfunnsutvikling
        sannsynlighet forsikringsselskap helsepersonell menneskerettigheter selvstendighet offentlighet
        tilgjengelighet gjennomføring selvfølge selvfølgelig egentlig forskjellige anledning
        pålitelighet bønder bønner
        """.split()
        assert all(observed[word]["candidates"] for word in required)
        assert lexicon.lookup("universitet")
        assert not lexicon.lookup("qzzxwøkasdfab")
        assert observed["tanken"]["resolution"] == "unresolved_homograph"
        assert observed["tanken"]["target"] is None
    finally:
        lexicon.close()

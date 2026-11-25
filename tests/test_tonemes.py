import pytest

from scripts.common import data_dir
from scripts.lookup_pronunciation import Lexicon, pronunciation


def test_two_source_markers_map_to_stress_without_ipa_tone_claim():
    source = {"wordform": "synthetic", "pos": "NN", "feats": "", "wordform_id": "A", "update_info": "",
              "nofabet_transcription": "T EE2", "ipa_transcription": '"teː', "sampa_transcription": ""}
    value = pronunciation(source, "e_written")
    assert value["ipa"] == "ˈteː"
    assert value["tonemes"] == [2]
    assert value["nb_ipa"] == '"teː'


def test_attested_bonder_minimal_tone_pair():
    if not (data_dir() / "e_written.sqlite3").is_file():
        pytest.skip("Requires the official CC0 index.")
    lexicon = Lexicon("e_written")
    try:
        first = lexicon.lookup("bønder")
        second = lexicon.lookup("bønner")
        assert {tuple(row["tonemes"]) for row in first} == {(1,)}
        assert {tuple(row["tonemes"]) for row in second} == {(2,)}
        assert {row["ipa"] for row in first} == {row["ipa"] for row in second}
    finally:
        lexicon.close()

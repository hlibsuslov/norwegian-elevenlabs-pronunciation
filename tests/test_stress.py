from scripts.optimize_text import prepare
from scripts.resolve_homographs import resolve


def test_source_stress_indices_are_preserved(synthetic_lexicon):
    target, = synthetic_lexicon.lookup("testsymbol")
    assert target["primary_stress_phone_indices"] == [1]
    assert target["secondary_stress_phone_indices"] == [4]
    assert target["ipa"] == "ˈteːst.ˌɑ"


def test_pos_evidence_and_attested_context_override():
    variants = [
        {"nb_ipa": "phonetic-A", "nofabet": "A", "pos": "NN", "wordform_ids": ["1"]},
        {"nb_ipa": "phonetic-B", "nofabet": "B", "pos": "VB", "wordform_ids": ["2"]},
        {"nb_ipa": "phonetic-C", "nofabet": "C", "pos": "NN", "wordform_ids": ["3"]},
    ]
    assert resolve("x", variants, {"pos": "VERB"}, "Jeg x.", "e_written")[0] == variants[1]
    assert resolve("x", variants, {"pos": "NOUN"}, "En x.", "e_written")[0] is None
    reviewed = {"word": "x", "wordform_id": "3", "sentence": "En x.", "source": "test", "dialect": "e_written"}
    assert resolve("x", variants, {"pos": "NOUN"}, "En x.", "e_written", [reviewed])[0] == variants[2]
    assert resolve("x", variants, {"pos": "NOUN"}, "Et annet x.", "e_written", [reviewed])[0] is None


def test_context_collision_never_applies_to_plain_text():
    choice = {"text": "x", "target": {"ipa": "a", "source": "synthetic"}, "flags": ["compound_review"],
              "resolution": "verified_context_id", "sentence": 0}
    unknown = {**choice, "resolution": "unresolved_homograph", "target": None}
    text = "x x."
    analysis = {"tokens": [choice, unknown], "source": text, "dialect": "e_written",
                "utterances": [{"id": 0, "start": 0, "end": len(text), "text": text}]}
    plan = prepare(analysis, "eleven_v4", trials=True)
    assert plan["synthesis_text"] == text
    assert not plan["segments"][0]["rules"]

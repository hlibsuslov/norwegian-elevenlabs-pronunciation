from scripts.detect_mispronunciation import pronunciation_events
from scripts.locale_lexicons import LexicalRouter, ipa_units
from scripts.pronunciation_assessment import assess


def test_cmu_primary_stress_and_no_fabricated_ipa():
    with LexicalRouter() as lexicons:
        gift=lexicons.lookup("gift","en-US")
    assert any(row["stress"]==[1] for row in gift)
    assert all(row["ipa"] is None for row in gift)
    assert ipa_units("ˈbœ.nər")==["b","œ","n","ə","r"]


def test_phoneme_model_evidence_is_compared_to_all_attested_variants():
    rows=[{"alphabet":"IPA","phones":["f","a"]},{"alphabet":"IPA","phones":["f","ɛ"]}]
    obs={"model":"fixture","alphabet":"IPA","phones":["f","ɛ"]}
    assert assess(obs,rows)["distance"]==0
    assert assess({**obs,"alphabet":"unknown"},rows) is None


def test_unvalidated_observation_cannot_assign_foreign_pronunciation():
    class Lexicons:
        def lookup(self,word,locale):
            return [{"alphabet":"IPA","phones":["g","ɪ","f","t"] if locale=="en-GB" else ["j","ɪ","f","t"]}]
    routed={"tokens":[{"id":0,"text":"gift","expected_language":"nb-NO","sentence":"gift",
                       "intentional_code_switch":False}]}
    observed={"start":2.0,"end":2.3,"alphabet":"IPA","model":"fixture",
              "phones":["g","ɪ","f","t","s"]}
    timing={0:{"start":2.0,"end":2.3}}
    result=pronunciation_events(routed,timing,Lexicons(),[observed])
    assert not any(row.get("pronunciation_language") for row in result)

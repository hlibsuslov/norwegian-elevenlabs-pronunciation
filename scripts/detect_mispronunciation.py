"""Generate tentative lexical, stress and cross-language candidates."""
from .locale_lexicons import ipa_units
from .pronunciation_assessment import assess, phone_distance


def pronunciation_events(routed, timings, lexicons, evidence, options=None):
    events = []
    for token in routed["tokens"]:
        time = timings.get(token["id"])
        if not time:
            continue
        locale = token["expected_language"]
        targets = lexicons.lookup(token["text"], locale)
        if not targets:
            continue
        overlaps = [row for row in evidence if min(row["end"], time["end"]) -
                    max(row["start"], time["start"]) >= .5*(time["end"]-time["start"])]
        for observation in overlaps:
            if observation["alphabet"]=="IPA":
                observation={**observation,"phones":[p for phone in observation["phones"] for p in ipa_units(phone)]}
            result = assess(observation,targets)
            if result is None:
                continue
            foreign = []
            other_locales = {"nb-NO","uk-UA","en-US","en-GB"} - {locale}
            if locale.startswith("en-"):
                other_locales -= {"en-US","en-GB"}
            for alternative in sorted(other_locales):
                for option in lexicons.lookup(token["text"],alternative):
                    if option["alphabet"] == observation["alphabet"]:
                        foreign.append((phone_distance(observation["phones"],option["phones"]),alternative))
            alternate = min(foreign, default=(1,None))
            compatible=[target for target in targets if target["alphabet"]==observation["alphabet"]]
            if (result["distance"]>=.35 and alternate[0]+.25<result["distance"]
                    and result["validated_for_locale"] and not token["intentional_code_switch"]):
                category = "possible_phonetic_interference"
            elif result["distance"]>=.35:
                category = "possible_phone_mismatch"
            elif (result["observed_stress"] is not None and all(target.get("stress") is not None for target in compatible)
                  and not any(target["stress"]==result["observed_stress"] for target in compatible)):
                category = "possible_stress_mismatch"
            elif (locale=="nb-NO" and result["observed_toneme"] in (1,2)
                  and all(target.get("tonemes") for target in compatible)
                  and not any(result["observed_toneme"] in target["tonemes"] for target in compatible)):
                category = "possible_tone_mismatch"
            else:
                continue
            events.append({"category":category,"start":time["start"],"end":time["end"],"word":token["text"],
                           "sentence":token["sentence"],"expected_language":locale,
                           "pronunciation_language":alternate[1] if category.endswith("interference") else None,
                           "expected_pronunciation":result["expected"],"observed_pronunciation":result["observed"],
                           "timing":"word_alignment_estimate","detector":"independent_phoneme_observation",
                           "evidence":result,"intentional_code_switch":token["intentional_code_switch"]})
    return events

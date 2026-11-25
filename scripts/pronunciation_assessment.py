"""Compare independent phone observations to all compatible lexical variants."""
from difflib import SequenceMatcher
from collections import defaultdict

import numpy as np


def phone_distance(observed, expected):
    return 1 - SequenceMatcher(None, observed, expected, autojunk=False).ratio()


def assess(observation, candidates):
    valid = [candidate for candidate in candidates if candidate["alphabet"] == observation.get("alphabet")]
    if not valid:
        return None
    best = min(valid,key=lambda candidate: phone_distance(observation["phones"],candidate["phones"]))
    score = phone_distance(observation["phones"],best["phones"])
    return {"distance":score,"expected":best,"matching_variants":len(valid),
            "observed":observation["phones"],"model":observation["model"],
            "validated_for_locale":bool(observation.get("locale_validation_evidence")),
            "observed_stress":observation.get("stress"),"observed_toneme":observation.get("toneme")}


def repeated_word_timing(alignments, lexicons):
    """Flag a large duration departure among comparable instances of the same attested word."""
    by_word=defaultdict(list)
    for alignment in alignments:
        if not alignment.get("phones"):
            continue
        locale=alignment["locale"]
        for word in alignment["words"]:
            surface=word["text"]
            candidates=lexicons.lookup(surface,locale)
            if len([c for c in candidates if c["alphabet"]=="NoFAbet"])!=1:
                continue
            phones=[p for p in alignment["phones"] if p["start"]>=word["start"]-.004 and p["end"]<=word["end"]+.004]
            sequence=[p["text"].upper() for p in phones]
            expected=next(row for row in candidates if row["alphabet"]=="NoFAbet")
            if sequence!=expected["phones"]:
                continue
            durations=[p["end"]-p["start"] for p in phones]
            by_word[(locale,surface.casefold(),tuple(sequence))].append((word,durations,expected))
    events=[]
    for (locale,surface,sequence), items in by_word.items():
        if len(items)<4:
            continue
        for index,(word,durations,expected) in enumerate(items):
            rest=np.asarray([row[1] for i,row in enumerate(items) if i!=index])
            typical=np.median(rest,axis=0)
            dispersion=np.maximum(.018,1.4826*np.median(np.abs(rest-typical),axis=0))
            scores=(np.asarray(durations)-typical)/dispersion
            longest=int(np.argmax(scores))
            if scores[longest]<4 or durations[longest]<max(.12,1.7*typical[longest]):
                continue
            events.append({"category":"possible_phone_stretching", "start":word["start"],"end":word["end"],
                           "word":word["text"],"expected_language":locale,"expected_pronunciation":expected,
                           "detector":"within_word_acoustic_timing","timing":"forced_alignment_estimate",
                           "evidence":{"phone_label":sequence[longest],"observed_duration_s":durations[longest],
                                       "comparison_median_s":float(typical[longest]),"comparison_instances":len(items)-1,
                                       "local_emphasis_possible":True}})
    return events

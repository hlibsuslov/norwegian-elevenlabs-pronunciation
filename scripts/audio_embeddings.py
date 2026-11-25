"""Robust, local acoustic vectors for voice-profile comparisons."""
import numpy as np


COLUMNS = ("rms_dbfs","dynamic_range_db","centroid","flatness","flux","zcr","crest_db",
           "pitch_median","pitch_spread_hz","hnr_db","voiced_fraction")


def vector(features):
    return {
        **{name: features[name] for name in COLUMNS if features.get(name) is not None},
        **{f"mfcc_{i+1}": value for i,value in enumerate(features.get("mfcc", []))},
        **{name: features[name] for name in ("words_per_second", "median_pause_s","median_phone_s")
           if features.get(name) is not None},
    }


def profile(rows, metadata):
    vectors = [vector(row) for row in rows if row.get("speech_fraction_energy_vad",0)>.2]
    if len(vectors)<3:
        raise ValueError("At least three speech-bearing reference excerpts are required.")
    fields = sorted(set.union(*(set(row) for row in vectors)))
    output={}
    for field in fields:
        values = np.array([row[field] for row in vectors if field in row])
        if len(values) < 3:
            continue
        center = float(np.median(values))
        spread = float(1.4826*np.median(np.abs(values-center)))
        default_scale = 60 if field=="centroid" else .02 if field in {"flatness","flux","zcr","voiced_fraction"} else 1
        output[field] = {"median":center,"scale":max(spread, default_scale)}
    return {"version":1,"method":"robust_acoustic_feature_distance","count":len(vectors),
            "metadata":metadata,"fields":output}


def distances(features, reference):
    observed = vector(features)
    return {field:float((observed[field]-row["median"])/row["scale"])
            for field,row in reference["fields"].items() if field in observed}

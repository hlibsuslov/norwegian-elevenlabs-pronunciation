"""Provide candidate anomaly classes from orthogonal acoustic and text evidence."""
import numpy as np

from .audio_embeddings import distances
from .reverse_transcribe import reconcile_words


def ranked_voice_events(windows, rate, reference, sensitivity):
    if not reference:
        return []
    cutoff = 3.0 if sensitivity=="recall" else 5.0 if sensitivity=="precision" else 4.0
    candidates=[]
    for window in windows:
        features=window["features"]
        if features.get("speech_fraction_energy_vad",0)<.25:
            continue
        deviation=distances(features,reference)
        highlighted={name:value for name,value in deviation.items() if abs(value)>=cutoff}
        spectral=[value for name,value in highlighted.items() if name in {"centroid","flatness","zcr"} or name.startswith("mfcc_")]
        level = {"rms_dbfs","crest_db","dynamic_range_db"} & highlighted.keys()
        voicing = {"pitch_median","pitch_spread_hz","hnr_db","voiced_fraction"} & highlighted.keys()
        if len(spectral)>=3 and voicing:
            label = "possible_timbre_or_metallic_shift"
        elif len(voicing)>=2:
            label = "possible_unstable_or_robotic_voicing"
        elif level and spectral:
            label = "possible_gain_or_noise_shift"
        elif len(spectral)>=4:
            label = "voice_spectral_anomaly"
        else:
            continue
        candidates.append({"category":label,"start":window["core_start"]/rate,"end":window["core_end"]/rate,
                           "detector":"voice_profile","timing":"window_estimate",
                           "evidence":{"deviation_units":highlighted,"reference_clips":reference["count"]}})
    return candidates


def speech_events(transcripts, silence_min=.8):
    result=[]
    for label, words in transcripts.items():
        words = reconcile_words(words)
        for i, item in enumerate(words):
            duration=item["end"]-item["start"]
            if duration > max(1.35, len(item["text"])*.18):
                result.append({"category":"possible_stretched_word","start":item["start"],"end":item["end"],
                               "word":item["text"],"detector":label,"timing":"asr_estimate",
                               "evidence":{"word_duration_s":duration,"requires_phone_alignment":True}})
            if not i:
                continue
            prior=words[i-1]
            gap=item["start"]-prior["end"]
            if gap>=silence_min and gap<=4:
                result.append({"category":"interword_pause","start":prior["end"],"end":item["start"],
                               "detector":label,"timing":"asr_estimate",
                               "evidence":{"duration_s":gap,"natural_sentence_pause_possible":True}})
    return result

import numpy as np

from scripts.detect_artifacts import scan_signal
from scripts.evidence_fusion import fuse


def test_click_rail_and_dropout_are_localized_without_asr():
    rate=16000
    t=np.arange(rate*3)/rate
    values=(.24*np.sin(2*np.pi*231*t))[:,None].astype("float32")
    values[rate//2,0]=.9
    values[rate:rate+15,0]=.999
    first=5*rate//3+7
    values[first:first+rate//20,0]=0
    events,features=scan_signal(values,rate,origin=3*rate,sensitivity="recall")
    actual={row["category"] for row in events}
    assert {"click_or_pop","clipping","dropout"} <= actual
    assert any(e["category"]=="click_or_pop" and e["start"]<=3.5<=e["end"] for e in events)
    assert features["peak_dbfs"]<0


def test_clean_sine_and_ordinary_silence_are_not_digital_events():
    rate=24000
    t=np.arange(rate*2)/rate
    sound=(.18*np.sin(2*np.pi*200*t)*np.sin(np.pi*t/2)**2)[:,None]
    events,_=scan_signal(sound,rate,sensitivity="recall")
    assert events==[]


def test_fusion_uses_distinct_evidence_families_and_preserves_uncertainty():
    prototypes=[{"category":"possible_phone_mismatch","start":2,"end":2.17,"word":"gift",
                 "timing":"word_alignment_estimate","detector":source,"evidence":{"test":True}}
                for source in ("asr:whisper:base","asr:whisper:small")]
    result,=fuse(prototypes,{"path":"test.wav","sha256":"test"},10)
    assert result["confidence"]=="heuristic"
    assert result["status"]=="HUMAN_REVIEW_REQUIRED"
    assert len(result["detector_names"])==2

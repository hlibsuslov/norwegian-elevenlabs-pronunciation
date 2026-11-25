import numpy as np
import soundfile as sf

from scripts.detect_artifacts import scan_signal
from scripts.evidence_fusion import fuse
from scripts.segment_audio import read_samples, windows


def test_segments_overlap_and_impulse_in_boundary_has_one_id(tmp_path):
    rate=8000
    signal=np.zeros((45*rate,1),dtype="float32")
    signal[23*rate+120,0]=.72
    path=tmp_path/"ticks.wav"
    sf.write(path,signal,rate,subtype="FLOAT")
    coverage=list(windows(path))
    assert all(a.core_end==b.core_start and a.end>b.start for a,b in zip(coverage,coverage[1:]))
    assert coverage[0].start==0 and coverage[-1].end==len(signal)
    events=[]
    for window in coverage:
        local=read_samples(path,window.start,window.end)
        candidates,_=scan_signal(local,rate,window.start,"recall")
        events.extend(e for e in candidates if e["category"]=="click_or_pop")
    result=fuse(events,{"path":"test","sha256":"sha"},45)
    assert len(result)==1
    assert result[0]["start"]<=23.015<=result[0]["end"]

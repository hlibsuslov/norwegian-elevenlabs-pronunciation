import numpy as np
import pytest
import soundfile as sf

from scripts.scan_audio import scan_raw
from scripts.segment_audio import windows


def test_forty_minute_schedule_covers_entire_recording_with_bounded_windows(tmp_path):
    path=tmp_path/"long.wav"
    rate=1000
    with sf.SoundFile(path,mode="w",samplerate=rate,channels=1,subtype="PCM_16") as audio:
        for _ in range(41):
            audio.write(np.zeros(60*rate,dtype="float32"))
    slices=list(windows(path))
    assert len(slices)>85 and slices[-1].core_end==41*60*rate
    assert sum(w.core_end-w.core_start for w in slices)==41*60*rate
    assert all(w.end-w.start<=34*rate for w in slices)


def test_cached_scan_resumes_and_rejects_changed_settings(tmp_path):
    path=tmp_path/"short.wav"
    sf.write(path,np.zeros(4000,dtype="float32"),8000)
    out=tmp_path/"run"
    first=scan_raw(path,out,signal=False)
    second=scan_raw(path,out,signal=False,resume=True)
    assert first["windows"]==second["windows"]
    with pytest.raises(ValueError,match="changed"):
        scan_raw(path,out,signal=False,deep=True,resume=True)

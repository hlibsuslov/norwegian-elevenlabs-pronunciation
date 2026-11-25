import numpy as np
import pytest
import soundfile as sf

from scripts.repair_artifacts import declick
from scripts.seamless_replace import apply_replacements, fit_clip, subtitle_offsets
from scripts.validate_repaired_audio import parameter_check


def test_interpolation_patches_only_damaged_samples_and_keeps_video_clock(tmp_path):
    rate=24000
    time=np.arange(rate)/rate
    signal=(.15*np.sin(2*np.pi*217*time))[:,None].astype("float32")
    damages=4321
    signal[damages,0]=.9
    repaired,stats=declick(signal[damages-400:damages+401],rate,398,403)
    assert stats["detected_signal_events_after"]<stats["detected_signal_events_before"]
    source,dest=tmp_path/"source.wav",tmp_path/"final.wav"
    sf.write(source,signal,rate,subtype="FLOAT")
    apply_replacements(source,dest,[{"start_sample":damages-2,"end_sample":damages+3,"values":repaired[398:403]}],rate,1)
    modified,_=sf.read(dest,dtype="float32",always_2d=True)
    assert np.array_equal(modified[:damages-2],signal[:damages-2])
    assert np.array_equal(modified[damages+3:],signal[damages+3:])
    assert parameter_check(source,dest,True)["duration_delta_s"]==0


def test_video_locked_rejects_large_stretch_and_subtitles_shift():
    with pytest.raises(ValueError,match="8%"):
        fit_clip(np.zeros((24000,1)),24000,12000,1,"VIDEO_LOCKED",None)
    patch={"start_sample":1000,"end_sample":2000,"values":np.zeros((1500,1))}
    assert subtitle_offsets([{"start":1.5,"end":3,"text":"tekst"}],[patch],1000)[0]=={
        "start":1.75,"end":3.5,"text":"tekst"}

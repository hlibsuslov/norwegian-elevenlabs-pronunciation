"""Check assembled media on the global decoded clock and compare digital artifacts."""
from collections import Counter

import soundfile as sf

from .detect_artifacts import scan_signal
from .segment_audio import read_samples, windows


def parameter_check(original, repaired, locked):
    with sf.SoundFile(original) as base, sf.SoundFile(repaired) as fixed:
        if (base.samplerate,base.channels)!=(fixed.samplerate,fixed.channels):
            raise ValueError("Output audio format differs from the decoded source.")
        if locked and len(base)!=len(fixed):
            raise ValueError("Locked output drifted on the decoded sample clock.")
        return {"original_frames":len(base),"output_frames":len(fixed),"sample_rate":base.samplerate,
                "channels":base.channels,"duration_delta_s":(len(fixed)-len(base))/base.samplerate}


def global_digital_scan(path, sensitivity="balanced"):
    with sf.SoundFile(path) as metadata:
        rate=metadata.samplerate
    events=[]
    for window in windows(path):
        values=read_samples(path,window.start,window.end)
        found,_=scan_signal(values,rate,window.start,sensitivity)
        events.extend(event for event in found if window.core_start/rate <= event["start"] < window.core_end/rate)
    return dict(Counter(e["category"] for e in events))

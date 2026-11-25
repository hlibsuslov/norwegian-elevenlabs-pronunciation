"""Regenerate a reviewed context span through the shared ElevenLabs client."""
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .generate_audio import SAMPLE_RATE
from .repair_artifacts import digital_signature
from .seamless_replace import fit_clip, fades
from .segment_audio import read_samples


def regenerate_event(api, media, rate, event, metadata, mode, seed=0):
    context=event.get("repair_context")
    if not context or not context.get("text_authoritative"):
        raise ValueError("Sentence regeneration requires an authoritative manuscript or approved subtitle timing.")
    voice=metadata.get("voice_id")
    if not voice or not metadata.get("model"):
        raise ValueError("Original voice_id and model are required for regenerated edits.")
    start,end=round(context["start"]*rate),round(context["end"]*rate)
    before=read_samples(media,start,end)
    if len(before)!=end-start:
        raise ValueError("This repair does not have a complete source interval.")
    if before.shape[1]>1:
        difference=float(np.mean((before[:,0]-before[:,1])**2))
        level=float(np.mean(before**2))
        if before.shape[1]!=2 or level<=0 or difference/level>.025:
            raise ValueError("TTS splicing requires mono or a near-identical stereo voice pair.")
    from math import gcd

    request={"text":context["text"],"previous_text":context.get("previous_text",""),
             "next_text":context.get("next_text",""),"language_code":context.get("language_code")}
    pcm,info=api.generate(metadata["model"],voice,request,context.get("rules",[]),
                          settings=metadata.get("voice_settings"),seed=seed)
    signal=np.frombuffer(pcm,dtype="<i2").astype(np.float32)/32768
    divider=gcd(rate,SAMPLE_RATE)
    candidate=np.repeat(resample_poly(signal,rate//divider,SAMPLE_RATE//divider)[:,None],before.shape[1],axis=1)
    with tempfile.TemporaryDirectory(prefix="nelp-repair-") as directory:
        fitted=fit_clip(candidate,rate,len(before),before.shape[1],mode,Path(directory))
    fixed=fades(before,fitted,rate)
    if digital_signature(fixed,rate)>digital_signature(before,rate):
        raise ValueError("A new digital artifact appeared in the candidate.")
    return {"start_sample":start,"end_sample":end,"values":fixed},info

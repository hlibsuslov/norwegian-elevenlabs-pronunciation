"""Export context clips, narrow clips, waveform plots and evidence per event."""
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import write_json
from .extract_audio_features import spectral_features
from .segment_audio import read_samples


def figures(base, values, rate, event_lo, event_hi):
    from matplotlib.figure import Figure

    def axes(title):
        figure=Figure(figsize=(6,1.55),layout="constrained")
        ax=figure.subplots()
        ax.set_title(title,fontsize=10,loc="left")
        ax.tick_params(axis="both",labelsize=8)
        return figure,ax

    t=np.arange(len(values))/rate
    mono=values.mean(axis=1)
    stride=max(1,len(values)//5500)
    figure,ax=axes("Waveform")
    ax.plot(t[::stride],mono[::stride],color="#1a645e", linewidth=.65)
    ax.axvspan(event_lo,event_hi,color="#b7491e",alpha=.24)
    ax.set_xlim(0,t[-1])
    ax.set_ylim(-max(.07,float(np.abs(mono).max())),max(.07,float(np.abs(mono).max())))
    ax.set_xlabel("Seconds in clip",fontsize=8)
    figure.savefig(base/"waveform.png",dpi=120)

    spec=spectral_features(values, rate)
    figure,ax=axes("Spectrogram")
    power=10*np.log10(spec["power"])
    high=float(np.max(power))
    ax.pcolormesh(spec["times"],spec["frequencies"],power,shading="auto",cmap="magma",
                  vmin=high-65,vmax=high)
    ax.axvspan(event_lo,event_hi,color="#b7491e",alpha=.25)
    ax.set_xlim(0,t[-1])
    ax.set_ylim(0,min(10000,rate/2))
    ax.set_ylabel("Hz",fontsize=8)
    ax.set_xlabel("Seconds in clip",fontsize=8)
    figure.savefig(base/"spectrogram.png",dpi=120)


def extract(proxy, events, directory, rate, total, *, pictures=True, context=1.35):
    root=Path(directory)
    root.mkdir(parents=True,exist_ok=True)
    for event in events:
        base=root/event["id"]
        base.mkdir(exist_ok=True)
        start=max(0,round((event["start"]-context)*rate))
        preview_end=min(event["end"], event["start"]+24)
        end=min(total,round((preview_end+context)*rate))
        values=read_samples(proxy,start,end)
        at, until=event["start"]-start/rate, preview_end-start/rate
        frame=min(total,max(1,round(at*rate))),min(len(values),max(1,round(until*rate)))
        a,b=max(0,frame[0]-round(.035*rate)),min(len(values),frame[1]+round(.035*rate))
        completed=(base/"evidence.json").exists()
        if not completed or not (base/"context.wav").is_file() or not (base/"defect.wav").is_file():
            sf.write(base/"context.wav",values,rate,subtype="PCM_16")
            sf.write(base/"defect.wav",values[a:b],rate,subtype="PCM_16")
        if pictures and (not completed or not (base/"waveform.png").is_file() or not (base/"spectrogram.png").is_file()):
            figures(base,values,rate,at,until)
        event["assets"]={"context":f"clips/{event['id']}/context.wav","defect":f"clips/{event['id']}/defect.wav",
                         "evidence":f"clips/{event['id']}/evidence.json",
                         "waveform":f"clips/{event['id']}/waveform.png" if pictures else None,
                         "spectrogram":f"clips/{event['id']}/spectrogram.png" if pictures else None}
        event["clip_start"],event["clip_end"]=start/rate,end/rate
        write_json(base/"evidence.json",event)
    return events

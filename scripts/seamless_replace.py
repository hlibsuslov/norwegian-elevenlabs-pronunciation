"""Fit and splice replacements into decoded audio with bounded timing and drift checks."""
import numpy as np
import soundfile as sf

from .segment_audio import db_level, ff, read_samples


def fit_clip(raw_clip, rate, target_length, channels, mode, directory):
    duration=len(raw_clip)/rate
    target=target_length/rate
    if duration<=0 or target<=0:
        raise ValueError("Both repair intervals need audio.")
    ratio=duration/target
    if mode=="VIDEO_LOCKED":
        if abs(ratio-1)>.08:
            raise ValueError("Video-locked replacement needs more than 8% tempo change.")
        if abs(ratio-1)>.002:
            source=directory/"tempo-input.wav"
            dest=directory/"tempo-output.wav"
            sf.write(source,raw_clip,rate,subtype="FLOAT")
            ff(["ffmpeg","-nostdin","-v","error","-y","-i",str(source),"-af",
                f"atempo={ratio:.7f}","-c:a","pcm_f32le",str(dest)],timeout=120)
            raw_clip,_=sf.read(dest,dtype="float32",always_2d=True)
        if len(raw_clip)-target_length>round(.035*rate) or target_length-len(raw_clip)>round(.035*rate):
            raise ValueError("Video-locked tempo correction left excessive residual duration.")
        raw_clip=raw_clip[:target_length]
        if len(raw_clip)<target_length:
            raw_clip=np.pad(raw_clip,((0,target_length-len(raw_clip)),(0,0)))
    if raw_clip.shape[1]!=channels:
        raise ValueError("Automatic replacement requires matching channel configuration.")
    return raw_clip


def fades(before, candidate, rate, boundary_dbfs=-35):
    if len(candidate)<.10*rate:
        raise ValueError("Replacement segment is too short.")
    fade=round(rate*.012)
    if max(float(np.max(db_level(x))) for x in (
        before[:fade], before[-fade:], candidate[:fade],candidate[-fade:],
    )) > boundary_dbfs:
        raise ValueError("The proposed cut touches audible speech. Review the context boundaries.")
    original=float(np.mean(db_level(before)))
    proposed=float(np.mean(db_level(candidate)))
    adjustment=original-proposed
    if abs(adjustment)>6:
        raise ValueError("Replacement level differs by more than 6 dB.")
    adjusted=candidate.copy()*10**(adjustment/20)
    ramp=np.linspace(0,1,fade,dtype=np.float32)[:,None]
    adjusted[:fade]=before[:fade]*(1-ramp)+adjusted[:fade]*ramp
    adjusted[-fade:]=adjusted[-fade:]*(1-ramp)+before[-fade:]*ramp
    if np.max(np.abs(adjusted))>=.995:
        raise ValueError("Replacement would create clipping.")
    return adjusted


def apply_replacements(proxy, output, patches, rate, channels):
    """Stream the untouched samples through, holding only one patch and a copy buffer."""
    patches=sorted(patches,key=lambda row:row["start_sample"])
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_name(f".{output.stem}.part.wav")
    try:
        with sf.SoundFile(proxy) as source, sf.SoundFile(temp,mode="w",samplerate=rate,channels=channels,
                                                        format="RF64",subtype="FLOAT") as destination:
            cursor=0
            for patch in patches:
                begin,end=patch["start_sample"],patch["end_sample"]
                if begin<cursor or end<=begin or end>len(source):
                    raise ValueError("Repair intervals overlap or are out of range.")
                transfer(source,destination,begin-cursor)
                source.seek(end)
                destination.write(patch["values"])
                cursor=end
            transfer(source,destination,len(source)-cursor)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)


def transfer(source,destination,count):
    while count:
        take=min(65536,count)
        destination.write(source.read(take,dtype="float32",always_2d=True))
        count-=take


def subtitle_offsets(cues, patches, rate):
    ordered=sorted(patches,key=lambda row:row["start_sample"])

    def adjusted(t):
        samples=round(t*rate)
        shift=0
        for patch in ordered:
            start,end=patch["start_sample"],patch["end_sample"]
            size=len(patch["values"])
            if samples<=start:
                break
            if samples<end:
                return (shift+start+size*(samples-start)/(end-start))/rate
            shift+=size-(end-start)
        return (samples+shift)/rate

    return [{**cue,"start":adjusted(cue["start"]),"end":adjusted(cue["end"])} for cue in cues]

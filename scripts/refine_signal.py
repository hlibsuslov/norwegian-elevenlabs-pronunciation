"""Find short spectral/level discontinuities relative to the same local recording."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import find_peaks

from .common import read_json, write_json
from .extract_audio_features import spectral_features
from .segment_audio import read_samples


def transient_candidates(values, rate, origin, core_start, core_end, sensitivity):
    spectrum=spectral_features(values,rate)
    delta=np.abs(np.diff(spectrum["rms_db"],prepend=spectrum["rms_db"][0]))
    flux=spectrum["flux"]
    high=spectrum["high_ratio"]
    frames=spectrum["times"]
    if len(frames)<12:
        return []
    median=float(np.median(spectrum["rms_db"]))
    level=np.maximum(spectrum["rms_db"],np.r_[spectrum["rms_db"][0],spectrum["rms_db"][:-1]])
    voiced_or_audible=level > max(-53, median+8)
    excursion=maximum_filter1d(delta,size=3)
    brightness=maximum_filter1d(high,size=3)
    novelty=flux*np.sqrt(excursion/8)*np.sqrt(np.maximum(.0001,brightness))
    eligible=voiced_or_audible & (excursion>7) & (flux>.25) & (brightness>.012)
    event_floor={"recall":.075,"balanced":.105,"precision":.135}[sensitivity]
    novelty[~eligible]=0
    peaks,_=find_peaks(novelty,height=event_floor,distance=max(1,round(.28/(frames[1]-frames[0]))))
    rows=[]
    for point in peaks:
        sample=int(origin+frames[point]*rate)
        if not core_start<=sample<core_end:
            continue
        rows.append({"category":"transient_outlier","start":max(origin/rate,sample/rate-.018),
                     "end":min((origin+len(values))/rate,sample/rate+.045),
                     "detector":"relative_transient_shape","timing":"short_window_estimate",
                     "evidence":{"score":float(novelty[point]),"spectral_flux":float(flux[point]),
                                 "local_level_change_db":float(excursion[point]),"upper_band_ratio":float(brightness[point]),
                                 "natural_consonant_possible":True}})
    limit={"recall":2,"balanced":1,"precision":1}[sensitivity]
    return sorted(rows,key=lambda event:event["evidence"]["score"],reverse=True)[:limit]


def refined(raw, sensitivity, cache, workers=1):
    rate=raw["sample_rate"]

    def work(window):
        name=cache/f"refine-v1-{sensitivity}-{window['index']:06d}.json"
        if name.is_file():
            return read_json(name)
        values=read_samples(raw["proxy"],window["start"],window["end"])
        events=transient_candidates(values,rate,window["start"],window["core_start"],window["core_end"],sensitivity)
        write_json(name,events)
        return events

    with ThreadPoolExecutor(max_workers=workers) as pool:
        candidates = [event for result in pool.map(work,raw["windows"]) for event in result]
    if len(candidates) < 12:
        return candidates
    scores = np.asarray([event["evidence"]["score"] for event in candidates])
    center = float(np.median(scores))
    scale = max(.03, float(1.4826*np.median(np.abs(scores-center))))
    cutoff = center + {"recall":1.05, "balanced":1.65, "precision":2.5}[sensitivity]*scale
    return [event for event in candidates if event["evidence"]["score"]>=cutoff]

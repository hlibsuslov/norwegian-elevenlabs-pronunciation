"""Produce narrowly scoped declick candidates and refuse acoustically unsafe edits."""
import numpy as np

from .detect_clicks import clicks
from .detect_clipping import clipping
from .detect_dropouts import dropouts


def digital_signature(values,rate):
    return len(clicks(values,rate,"recall"))+len(clipping(values,rate))+len(dropouts(values,rate))


def declick(values, rate, start, end, channels=None):
    channels=range(values.shape[1]) if not channels else sorted(set(channels))
    if start<2 or end+2>=len(values) or end-start>round(.012*rate) or end<=start:
        raise ValueError("Declick needs a short bounded sample interval and neighbors on both sides.")
    output=values.copy()
    for channel in channels:
        signal=values[:,channel]
        output[start:end,channel]=np.linspace(signal[start-1],signal[end],end-start+2)[1:-1]
    before=digital_signature(values,rate)
    after=digital_signature(output,rate)
    if before==0 or after>=before or np.max(np.abs(output[start:end]))>.995:
        raise ValueError("Local interpolation did not remove the signal defect without regression.")
    return output,{"detected_signal_events_before":before,"detected_signal_events_after":after}

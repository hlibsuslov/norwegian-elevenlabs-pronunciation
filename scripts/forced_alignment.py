"""Read subtitles and adapt trusted alignment tools to language-specific audio spans."""
import re
import tempfile
from pathlib import Path

import soundfile as sf

from .alignment import run_nofa, textgrid
from .common import read_text
from .segment_audio import ff

CLOCK = re.compile(r"(?:(\d+):)?(\d{2}):(\d{2})[,.](\d{3})")


def clock(text):
    match = CLOCK.fullmatch(text.strip())
    if not match:
        raise ValueError(f"Invalid subtitle timestamp {text!r}.")
    hours, minutes, seconds, millis = (int(x or 0) for x in match.groups())
    if minutes > 59 or seconds > 59:
        raise ValueError(f"Invalid subtitle timestamp {text!r}.")
    return hours*3600+minutes*60+seconds+millis/1000


def subtitle_cues(path):
    rows = []
    for block in re.split(r"\n\s*\n", read_text(path).replace("\r\n","\n").strip()):
        lines = block.splitlines()
        index = next((i for i,line in enumerate(lines) if "-->" in line),-1)
        if index < 0 or not lines[index+1:]:
            continue
        a,b = lines[index].split("-->",1)
        start,end = clock(a),clock(b.strip().split()[0])
        if end <= start:
            raise ValueError("Subtitle cue end must follow its start.")
        text = re.sub(r"<[^>]*>", "", " ".join(lines[index+1:]))
        rows.append({"start":start,"end":end,"text":text})
    if not rows:
        raise ValueError("No valid subtitle cues.")
    return rows


def align_span(values, rate, transcript, locale, nofa=None, mfa=None, models=None):
    if locale=="nb-NO" and not nofa or locale!="nb-NO" and not (mfa and models and locale in models):
        return None
    with tempfile.TemporaryDirectory(prefix="nelp-locale-") as temporary:
        root = Path(temporary)
        audio = root/"clip.wav"
        sf.write(audio, values, rate, subtype="PCM_16")
        if locale=="nb-NO":
            return run_nofa(audio,transcript,nofa)
        lexicon, acoustic = models[locale]
        lab = root/"clip.lab"
        lab.write_text(transcript, encoding="utf-8")
        output = root/"clip.TextGrid"
        ff([mfa,"align_one",str(audio),str(lab),lexicon,acoustic,str(output)],timeout=600)
        result = textgrid(output)
        result["source"]="MFA_"+locale
        return result

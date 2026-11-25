"""Evaluate competing phones on the same audio with an independent CTC recognizer."""
import argparse
from contextlib import closing
from pathlib import Path
from time import perf_counter

import cmudict
import numpy as np
import soundfile as sf

from scripts.common import read_json, write_json
from scripts.lookup_pronunciation import Lexicon
from .research import audio_bounds, nearby_words
from .research_features import acoustic_curves, fingerprint
from .research_models import Wav2vecPhones


def analyze_scope(model, proxy, rate, count, scope, hypotheses, acoustic=True):
    start, end = scope
    values, origin, end = audio_bounds(proxy, rate, count, start, end)
    log_probs = model.posteriors(values, rate)
    observations = model.observe(log_probs, origin)
    shifts = {}
    # This tests sensitivity to uncertain word crops using exactly the same crop for each competing sequence.
    for context in (0, .06, .12):
        padded, _, _ = audio_bounds(proxy, rate, count, start-context, end+context)
        shifts[f"padding_{context:.2f}_s"] = model.compare(
            log_probs if context == 0 else model.posteriors(padded, rate), hypotheses)
    output = {"scope": [origin, end], "observed_ctc_runs": observations, "competing_ctc_masses": shifts,
              "ctc_frame_step_s": model.step_s}
    if acoustic:
        curves = acoustic_curves(values, rate)
        f = fingerprint(values, rate, curves, 0, end-origin)
        for row in observations:
            if row["end"] > end:
                continue
            t = (row["start"] + row["end"])/2-origin
            row["voiced_formant_measurements_hz"] = [
                float(np.interp(t, curves["formant_times"], formant)) if len(formant) else None
                for formant in curves["formants"]
            ] if len(curves["f0"]) and np.any(np.isfinite(curves["f0"][abs(curves["f0_times"]-t)<.021])) else None
            if row["voiced_formant_measurements_hz"] is not None:
                row["voiced_formant_measurements_hz"] = [x if x is not None and np.isfinite(x) else None
                                                          for x in row["voiced_formant_measurements_hz"]]
        output["span_features"] = f
    return output


def phrase_occurrences(words, lexicon, proxy, rate, count):
    from .research import PARTICLE_PHRASES

    output = {}
    for phrase in PARTICLE_PHRASES:
        verb, particle = phrase.split()
        candidates = [(first, second) for first, second in zip(words, words[1:])
                      if first["text"].casefold() == verb and second["text"].casefold() == particle and
                      .02 < second["end"]-first["start"] < 3 and -.2 < second["start"]-first["end"] < .5]
        rows = []
        for first, second in candidates[:4]:
            start, end = first["start"], second["end"]
            local, origin, _ = audio_bounds(proxy, rate, count, start-.24, end+.30)
            curves = acoustic_curves(local, rate)
            observed = {}
            for token, word in (("verb", first), ("particle", second)):
                observed[token] = fingerprint(local, rate, curves, word["start"]-origin, word["end"]-origin)
            rows.append({"interval": [start, end], "reverse_context": [w["text"] for w in nearby_words(words, start-2, end+2)],
                         "asr_timing_and_acoustics": observed})
        output[f"å {phrase}"] = {"phrase_dictionary_results": lexicon.lookup(phrase),
                                "verb_dictionary_results": lexicon.lookup(verb),
                                "particle_dictionary_results": lexicon.lookup(particle),
                                "asr_adjacent_matches": len(candidates), "examples": rows}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("hypotheses", "model", "proxy", "out"):
        parser.add_argument(f"--{label}", type=Path, required=True)
    parser.add_argument("--asr", type=Path)
    args = parser.parse_args()
    began = perf_counter()
    definitions = read_json(args.hypotheses)
    phonemes = Wav2vecPhones(args.model)
    with sf.SoundFile(args.proxy) as source:
        rate, count = source.samplerate, len(source)
    primary = next(iter(read_json(args.asr)["reverse_transcript"].values())) if args.asr else []
    cases = []
    with closing(Lexicon("e_written")) as lexicon:
        for row in definitions["cases"]:
            case = {**row, "dictionary": lexicon.lookup(row["word"]) if row["word"] else [],
                    "evidence": analyze_scope(phonemes, args.proxy, rate, count, row["scope"], row["hypotheses"])}
            if "alternate_window" in row:
                case["alternate_evidence"] = analyze_scope(phonemes, args.proxy, rate, count, row["alternate_window"],
                                                           row.get("alternate_hypotheses", {}))
            if row["word"] == "holder":
                case["cmudict"] = cmudict.dict().get("holder")
                control_words = [word for word in primary if word["text"].casefold() == "holder" and
                                 abs(word["start"]-row["scope"][0])>6 and .13<word["end"]-word["start"]<1.25]
                case["same_recording_review_controls"] = [
                    {"asr_word": word, "evidence": analyze_scope(phonemes, args.proxy, rate, count,
                     (word["start"], word["end"]), row["hypotheses"], acoustic=False)} for word in control_words[:6]]
            cases.append(case)
        phrases = phrase_occurrences(primary, lexicon, args.proxy, rate, count)
    write_json(args.out, {"cases": cases, "particle_phrases": phrases, "model": "facebook/wav2vec2-lv-60-espeak-cv-ft",
                         "model_assets_sha256": phonemes.model_asset, "processing_seconds": perf_counter()-began,
                         "limitations": ["CTC runs are labels on 20 ms frames; inactive frames leave phone durations unresolved.",
                                         "Scores are model likelihoods for hand-mapped token strings on the same crop.",
                                         "This model has not been calibrated for Norwegian–Ukrainian switching or Urban East Norwegian.",
                                         "F0 and amplitude in connected speech do not verify lexical or particle stress alone."]})


if __name__ == "__main__":
    main()

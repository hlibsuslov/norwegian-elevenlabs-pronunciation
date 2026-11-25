"""Run local repair auditions and measure contrasts against nearby speech."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import butter, sosfiltfilt
from scipy.stats import rankdata

from scripts.common import read_json, write_json
from scripts.detect_clicks import clicks
from scripts.repair_artifacts import declick, digital_signature
from scripts.segment_audio import ff
from .research import audio_bounds
from .research_features import acoustic_curves, fingerprint, local_peaks


def comparable_scores(proxy, rate, count, cases):
    rows = []
    for case in cases:
        point = case["point_s"]
        sections = []
        for shift in (0, -14, -7, 7, 14):
            center = point + shift
            if shift and any(abs(other["point_s"]-center) < 5 for other in cases):
                continue
            values, origin, _ = audio_bounds(proxy, rate, count, center-4, center+4)
            possible = [event for event in local_peaks(values, rate, origin) if abs(event["peak_s"]-center)<=2.5]
            peak = max((event["score"] for event in possible), default=0.0)
            sections.append({"offset_s": shift, "score": peak})
        rows.append({"annotation_id": case["annotation_id"], "comparison_windows": sections,
                     "anchor_score": sections[0]["score"],
                     "baseline_match": case.get("baseline_match")})
    pos, controls = np.array([row["anchor_score"] for row in rows]), np.array(
        [window["score"] for row in rows for window in row["comparison_windows"] if window["offset_s"]])
    combined = rankdata(np.r_[pos, controls])
    auc = (combined[:len(pos)].sum() - len(pos)*(len(pos)+1)/2)/(len(pos)*len(controls)) if len(pos)*len(controls) else None
    return {"cases": rows, "rank_auc_exploratory": auc, "anchor_count": len(pos), "unreviewed_control_count": len(controls),
            "anchor_median": float(np.median(pos)) if len(pos) else None,
            "comparator_median": float(np.median(controls)) if len(controls) else None,
            "review_status": "Approximate positive windows and unlabeled same-track comparators."}


def cluster_cases(cases):
    fields = ("peak_step_abs", "rms_dbfs", "centroid_hz", "flatness", "flux_peak", "higher_band_ratio_peak")
    matrix = np.array([[case["fingerprint"][name] for name in fields] for case in cases])
    if len(matrix) < 3 or not np.isfinite(matrix).all():
        return []
    medians = np.median(matrix, axis=0)
    scales = np.maximum(np.quantile(matrix,.75,axis=0)-np.quantile(matrix,.25,axis=0), 1e-9)
    groups = fcluster(linkage((matrix-medians)/scales, method="ward"), t=3, criterion="maxclust")
    return [{"cluster": int(group), "annotation_ids": [row["annotation_id"] for row, label in zip(cases, groups) if label==group],
             "medians": dict(zip(fields, np.median(matrix[groups==group],axis=0).tolist()))} for group in np.unique(groups)]


def run_repairs(proxy, rate, count, root, cases):
    results = []
    ffmpeg = ff(["ffmpeg", "-version"], timeout=15).stdout.decode().splitlines()[0]
    for case in cases:
        if case["category"] != "artifact":
            continue
        identity = case["id"]
        start, end = case["start"], case["end"]
        values, origin, _ = audio_bounds(proxy, rate, count, start-.30, end+.30)
        a,b = round((start-origin)*rate), round((end-origin)*rate)
        repairs = []
        cue = case["fingerprint"]
        before_fingerprint = fingerprint(values, rate, acoustic_curves(values, rate), a/rate,b/rate)
        before_count = digital_signature(values, rate)
        for method in ("sample_interpolation", "ffmpeg_adeclick", "tapered_spectral_attenuation"):
            reason, candidate = None, None
            if method == "sample_interpolation":
                impulses = [item for item in clicks(values, rate, "recall") if item[1]<b and item[2]>a]
                if impulses:
                    try:
                        candidate, _ = declick(values, rate, min(row[1] for row in impulses), max(row[2] for row in impulses),
                                               [row[3]["channel"] for row in impulses])
                    except ValueError as exc:
                        reason = str(exc)
                else:
                    reason = "No sample-clock click in the provisional interval at recall sensitivity."
            elif method == "ffmpeg_adeclick":
                if not case.get("baseline_match") and cue["peak_step_abs"] < .11:
                    reason = "No sufficiently strong step or frozen transient for an impulse-repair audition."
                else:
                    converted = ff(["ffmpeg","-nostdin","-v","error","-f","f32le","-ar",str(rate),"-ac",str(values.shape[1]),
                                    "-i","pipe:0","-af","adeclick=window=55:overlap=75:threshold=2:burst=2",
                                    "-f","f32le","pipe:1"], input=values.astype("<f4").tobytes(), timeout=20)
                    candidate = np.frombuffer(converted.stdout, dtype="<f4").reshape(-1, values.shape[1])
            elif cue["higher_band_ratio_peak"] < .12 or rate <= 11800:
                reason = "High-band reduction was screened out for this candidate."
            else:
                candidate = values.copy()
                filtered = sosfiltfilt(butter(4, 5900, fs=rate, output="sos"), values, axis=0)
                taper = np.sin(np.pi*np.arange(b-a)/(max(1,b-a-1)))**2
                candidate[a:b] += .55*taper[:,None]*(filtered[a:b]-candidate[a:b])
            result = {"method": method, "status": "SCREENED_OUT" if reason else "UNAPPROVED_TRIAL",
                      "reason": reason, "api_credits": 0, "manual_assessment": None,
                      "parameters": {"window_ms":55,"overlap_pct":75,"threshold":2,"burst":2,"ffmpeg":ffmpeg}
                      if method=="ffmpeg_adeclick" else {"lowpass_hz":5900,"amount":.55,"order":4}
                      if method=="tapered_spectral_attenuation" else {"sensitivity":"recall","maximum_patch_s":.012}}
            if candidate is not None:
                if candidate.shape != values.shape:
                    result.update(status="REJECTED", reason="Length or number of channels changed.")
                else:
                    narrowed = candidate.copy()
                    narrowed[:a] = values[:a]
                    narrowed[b:] = values[b:]
                    delta = narrowed[a:b]-values[a:b]
                    target_rms_delta_db = 20*np.log10(max(1e-8,np.sqrt(np.mean(narrowed[a:b]**2))) /
                                                      max(1e-8,np.sqrt(np.mean(values[a:b]**2))))
                    after_count = digital_signature(narrowed, rate)
                    after_fingerprint = fingerprint(narrowed, rate, acoustic_curves(narrowed,rate), a/rate,b/rate)
                    changed = bool(np.any(abs(delta) > 1e-7))
                    result["comparison"] = {
                        "same_sample_count": True, "outside_interval_peak_change": 0.0,
                        "inside_interval_peak_change": float(np.max(abs(delta))), "target_rms_change_db": float(target_rms_delta_db),
                        "digital_candidates_before": before_count, "digital_candidates_after": after_count,
                        "upper_band_ratio_before": before_fingerprint["higher_band_ratio_peak"],
                        "upper_band_ratio_after": after_fingerprint["higher_band_ratio_peak"],
                    }
                    if not changed:
                        result.update(status="NO_CHANGE", reason="No change above float round-off within the interval.")
                    elif after_count > before_count:
                        result.update(status="REJECTED", reason="Detected more digital candidates after processing.")
                    else:
                        destination = root/"repaired_examples"/identity
                        destination.mkdir(exist_ok=True, parents=True)
                        sf.write(destination/"original.wav", values, rate, subtype="FLOAT")
                        sf.write(destination/f"{method}.wav", narrowed, rate, subtype="FLOAT")
                        review = root/"review"/"clips"/identity/f"{method}.wav"
                        sf.write(review, narrowed, rate, subtype="FLOAT")
                        sf.write(review.with_name("repair-original.wav"), values, rate, subtype="FLOAT")
                        result["original"] = (destination/"original.wav").relative_to(root).as_posix()
                        result["audio"] = (destination/f"{method}.wav").relative_to(root).as_posix()
                        result["review_asset"] = f"clips/{identity}/{method}.wav"
                        if before_count == 0:
                            result["reason"] = "No validated digital defect to compare; this trial requires matched audition."
            repairs.append(result)
        results.append({"id":identity,"annotation_id":case["annotation_id"],"scope":[origin,origin+len(values)/rate],
                        "experiments":repairs,"accepted":False})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset", "manifest", "proxy", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    began = perf_counter()
    manifest = read_json(args.manifest)
    cases = [row for row in manifest["cases"] if "annotation_id" in row and row["category"]=="artifact"]
    with sf.SoundFile(args.proxy) as source:
        rate, count = source.samplerate, len(source)
    write_json(args.out, {"source":manifest["source"],"local_score_contrast":comparable_scores(args.proxy,rate,count,cases),
                         "clusters":cluster_cases(cases),"repair_trials":run_repairs(args.proxy,rate,count,args.dataset,cases),
                         "processing_seconds":perf_counter()-began,"api_credits":0})


if __name__ == "__main__":
    main()

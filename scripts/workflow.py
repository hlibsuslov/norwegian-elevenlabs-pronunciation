"""Run PREVENT, DETECT, REPAIR, FULL AUTO, or LEARN for a Norwegian manuscript."""
import argparse
import re
import secrets
import subprocess
from contextlib import closing
from pathlib import Path

from .alignment import run_nofa, textgrid, transcribe
from .analyze_audio import analyze_recording
from .analyze_text import analyze
from .build_dictionary import export, learn, make_pls, verified_records
from .common import atomic_bytes, config, digest, file_digest, model_config, read_json, write_json
from .detect_stress_errors import detect
from .generate_audio import APIError, Budget, BudgetExceeded, ElevenLabs, credentials, pcm_wav, query_chars, units
from .lookup_pronunciation import Lexicon
from .merge_audio import splice
from .optimize_text import prepare
from .regenerate_segment import regenerate, repair_jobs, windows
from .validate_audio import compare


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prevent", "detect", "repair", "full-auto", "learn"))
    parser.add_argument("--text", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--alignment", type=Path, help="Existing NoFA TextGrid for the original audio.")
    parser.add_argument("--nofa-command", help="MFA 1.0 mfa_align executable or compatible wrapper.")
    parser.add_argument("--whisper-model", type=Path, help="Local multilingual faster-whisper model directory.")
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--reference-alignment", type=Path)
    parser.add_argument("--reference-review", type=Path, help="Reviewed, matched reference metadata JSON.")
    parser.add_argument("--record", type=Path, help="Reviewed correction for LEARN.")
    parser.add_argument("--out", type=Path, required=True, help="New output directory.")
    parser.add_argument("--language", choices=("auto", "nb"), default="auto")
    parser.add_argument("--dialect", choices=config("dialects")["dialects"], default=config("dialects")["default"])
    parser.add_argument("--model", choices=config("models")["models"], default=config("models")["default"])
    parser.add_argument("--voice")
    parser.add_argument("--env-file", type=Path, help="Optional local file for the ElevenLabs key and voice ID.")
    parser.add_argument("--voice-settings", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trial-ipa", action="store_true", help="Allow sourced segmental IPA A/B candidates.")
    parser.add_argument("--live", action="store_true", help="Authorize paid ElevenLabs synthesis with a budget.")
    parser.add_argument("--ab", action="store_true", help="Write blinded sentence pairs in FULL AUTO.")
    parser.add_argument("--max-characters", type=int, default=0, help="Upper limit for locally reserved character units.")
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=config("quality_thresholds")["max_repair_attempts"])
    args = parser.parse_args(argv)
    if (args.mode == "learn" and not args.record) or (args.mode != "learn" and not args.text):
        parser.error("LEARN requires --record. Other modes require --text.")
    if args.mode in {"detect", "repair"} and not args.audio:
        parser.error("DETECT and REPAIR require --audio.")
    if args.ab and args.mode != "full-auto":
        parser.error("--ab is available in FULL AUTO.")
    if args.live:
        if args.mode not in {"full-auto", "repair"}:
            parser.error("Live synthesis is available in FULL AUTO and REPAIR.")
        if args.max_characters <= 0 or args.max_requests <= 0:
            parser.error("--live requires positive --max-characters and --max-requests.")
    if not 0 <= args.seed < 2**32 or not 0 <= args.max_attempts <= config("quality_thresholds")["max_repair_attempts"]:
        parser.error("Seed must be uint32. Attempts must be within the configured limit.")
    if bool(args.reference_audio) != bool(args.reference_review):
        parser.error("--reference-audio and --reference-review must be supplied together.")
    if args.reference_alignment and not args.reference_audio:
        parser.error("--reference-alignment requires --reference-audio.")
    if args.alignment and args.mode == "full-auto":
        parser.error("Use --nofa-command to align newly generated audio.")
    return args


def source_from(path, output):
    original = path.read_bytes()
    source = original.decode("utf-8-sig")
    atomic_bytes(output / "original.txt", original)
    return source


def settings_from(args):
    if not args.voice_settings:
        return {}
    settings = read_json(args.voice_settings)
    transport = model_config(args.model)["transport"]
    fields = {"stability"} if transport == "dialogue" else {
        "stability", "similarity_boost", "style", "speed", "use_speaker_boost"
    }
    if not isinstance(settings, dict) or not settings.keys() <= fields:
        raise ValueError(f"Use these voice settings for this transport: {', '.join(sorted(fields))}.")
    for name, value in settings.items():
        if name == "use_speaker_boost":
            valid = type(value) is bool
        else:
            lower, upper = (0.7, 1.2) if name == "speed" else (0.0, 1.0)
            valid = type(value) in (float, int) and lower <= value <= upper
        if not valid:
            raise ValueError(f"Unsupported voice settings value for {name}.")
    return settings


def baseline_segments(plan):
    source = plan["source"]
    limit = model_config(plan["model"])["limit"]
    start = 0
    ends = {row["end"] for row in plan["segments"]}
    previous = 0
    for match in re.finditer(r"\S+\s*", source):
        last = match.end()
        if units(source[start:last]) > limit:
            possible = [end for end in ends if start < end <= previous]
            cut = max(possible, default=previous)
            if cut <= start or units(source[cut:last]) > limit:
                cut = previous
            if cut <= start or units(source[cut:last]) > limit:
                raise ValueError("A single manuscript chunk exceeds the model limit. Supply a natural boundary.")
            yield make_chunk(source, start, cut)
            start = cut
        previous = last
    if source[start:].strip():
        yield make_chunk(source, start, len(source))


def make_chunk(source, start, end):
    return {"text": source[start:end].strip(), "start": start, "end": end,
            "previous_text": source[max(0, start-100):start],
            "next_text": source[end:end+100]}


def get_features(args, path, analysis, declared_alignment=None, use_asr=True):
    alignment = textgrid(declared_alignment) if declared_alignment else None
    if not alignment and args.nofa_command:
        alignment = run_nofa(path, analysis["source"], args.nofa_command)
    asr = transcribe(path, args.whisper_model) if use_asr and args.whisper_model else None
    return analyze_recording(path, analysis, alignment, asr)


def reviewed_reference(args, analysis):
    if not args.reference_audio:
        return None
    record = read_json(args.reference_review)
    if (record.get("dialect") != args.dialect
        or record.get("source_sha256") != digest(analysis["source"])
        or record.get("audio_sha256") != file_digest(args.reference_audio)
        or record.get("status") != "reviewed_matched_reference"
        or not record.get("native_reviewer_evidence")
        or (not args.reference_alignment and not args.nofa_command)):
        raise ValueError("Reference review must match the exact manuscript, audio and dialect and supply native evidence and alignment.")
    return get_features(args, args.reference_audio, analysis, args.reference_alignment, use_asr=False)


def build_baseline(args, plan, budget, api, settings):
    chunks = list(baseline_segments(plan))
    if not api:
        return chunks, None
    estimates = [api.estimate(plan["model"], plan["voice_id"], chunk, [], settings, args.seed) for chunk in chunks]
    budget.check(sum(chars for chars, _ in estimates), sum(calls for _, calls in estimates))
    audio = bytearray()
    metadatas = []
    for chunk in chunks:
        pcm, metadata = api.generate(plan["model"], plan["voice_id"], chunk, [], settings, args.seed)
        metadatas.append({"start": chunk["start"], "end": chunk["end"], **metadata})
        audio.extend(pcm)
    baseline = args.out / "baseline.wav"
    atomic_bytes(baseline, pcm_wav(audio))
    write_json(args.out / "generation_alignment.json", metadatas)
    return chunks, baseline


def blind_pairs(args, api, plan, analysis, settings):
    pairs = []
    metrics = []
    directory = args.out / "ab"
    for segment in plan["segments"]:
        if not segment["rules"] or segment["over_model_limit"]:
            continue
        estimates = [api.estimate(plan["model"], plan["voice_id"], segment, rules, settings, args.seed)
                     for rules in ([], segment["rules"])]
        try:
            api.budget.check(sum(chars for chars, _ in estimates), sum(calls for _, calls in estimates))
            baseline, _ = api.generate(plan["model"], plan["voice_id"], segment, [], settings, args.seed)
            optimized, _ = api.generate(plan["model"], plan["voice_id"], segment, segment["rules"], settings, args.seed)
        except BudgetExceeded:
            break
        labels = ["A", "B"]
        secrets.SystemRandom().shuffle(labels)
        sentence_analysis = {"tokens": [row for row in analysis["tokens"] if row["sentence"] == segment["id"]]}
        measurements = {}
        for label, pcm in zip(labels, [baseline, optimized]):
            audio = directory / f"{segment['id']:03d}-{label}.wav"
            atomic_bytes(audio, pcm_wav(pcm))
            asr = transcribe(audio, args.whisper_model) if args.whisper_model else None
            features = analyze_recording(audio, sentence_analysis, asr=asr)
            measurements[label] = {field: features[field] for field in (
                "duration_s", "median_f0_hz", "voiced_frame_fraction", "peak_dbfs", "asr",
            )}
        pairs.append({"sentence": segment["id"], "baseline": labels[0], "optimized": labels[1]})
        metrics.append({"sentence": segment["id"], "text": segment["text"], "measurements": measurements})
        write_json(args.out / "ab_key.json", pairs)
        write_json(args.out / "ab_metrics.json", metrics)
    return len(pairs)


def quality(path, detection, audio_features, reference=None):
    write_json(path, {"detection": detection, "features": audio_features, "reference": reference})


def repair(args, api, plan, analysis, baseline, before, original_features, reference, settings):
    current_path, current, current_features = baseline, before, original_features
    attempts = []
    if not (api and reference is not None and args.nofa_command and args.whisper_model):
        return current_path, current, current_features, attempts
    for job in repair_jobs(plan, analysis, before):
        for attempt in range(1, args.max_attempts + 1):
            sentence = job["segment"]["id"]
            row = {"sentence": sentence, "attempt": attempt, "target_ids": job["target_ids"]}
            try:
                left, right = windows(job["segment"], analysis, current_features)
                candidate = args.out / f"candidate-{sentence}-{attempt}.wav"
                replacement = args.out / f"segment-{sentence}-{attempt}.wav"
                audio, metadata = regenerate(api, plan, job, settings, args.seed, attempt)
                row["cached"] = metadata["cached"]
                atomic_bytes(replacement, audio)
                row["splice"] = splice(current_path, replacement, candidate, left, right)
                features = get_features(args, candidate, analysis)
                detection = detect(analysis, features, reference)
                row.update(compare(current, detection, current_features, features, job["target_ids"]))
                if row["accepted"]:
                    current_path, current, current_features = candidate, detection, features
            except BudgetExceeded as exc:
                attempts.append({**row, "accepted": False, "reason": str(exc)})
                return current_path, current, current_features, attempts
            except (APIError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                row.update({"accepted": False, "reason": str(exc)})
            attempts.append(row)
            write_json(args.out / "repair_log.json", attempts)
            if row["accepted"] or "cached" not in row:
                break
    return current_path, current, current_features, attempts


def execute(args, report, budget, api_key=None):
    if args.mode == "learn":
        saved = learn(read_json(args.record))
        export(verified_records(), args.out / "corrections.json", "json")
        export(verified_records(), args.out / "corrections.csv", "csv")
        export(verified_records(), args.out / "pronunciation.pls", "pls", saved["model"], saved["dialect"], saved["voice_id"])
        return {**report, "status": "reviewed_correction_saved", "record": saved}

    source = source_from(args.text, args.out)
    with closing(Lexicon(args.dialect)) as lexicon:
        analysis = analyze(source, lexicon, args.language)
    plan = prepare(analysis, args.model, args.voice, verified_records(), args.trial_ipa)
    write_json(args.out / "text_analysis.json", analysis)
    write_json(args.out / "synthesis_plan.json", plan)
    for segment in plan["segments"]:
        if segment["rules"]:
            atomic_bytes(args.out / "dictionaries" / f"{segment['id']:03d}.pls", make_pls(segment["rules"]))
    report.update({
        "language": analysis["language"], "dialect": args.dialect, "model": args.model,
        "voice_id": args.voice, "analyzed_words": len(analysis["tokens"]),
        "text_review_count": len(analysis["unresolved"]), "confirmed_errors": 0,
        "uncertainties": {
            "text_review_token_ids": analysis["unresolved"], "observed_ipa": None,
            "toneme_realization": "requires_matched_native_evidence",
        },
        "native_review_token_ids": analysis["unresolved"],
        "source_sha256": digest(source), "phoneme_rules_are_segmental": True,
    })
    if args.mode == "prevent":
        return {**report, "status": "text_analysis_complete", "dry_run": True}
    spec = model_config(args.model)
    if args.live and (not spec["norwegian"] or analysis["language"]["detected"] not in (None, "nb", "no")):
        raise ValueError("Live Norwegian synthesis requires a Norwegian-supported model and Bokmål input or an explicit --language nb.")
    if args.live and not args.voice:
        raise ValueError("Provide --voice or set ELEVENLABS_VOICE_ID for live synthesis.")
    settings = settings_from(args)
    api = ElevenLabs(budget, api_key=api_key or "") if args.live else None
    reference = reviewed_reference(args, analysis)
    if args.mode == "full-auto":
        chunks, baseline = build_baseline(args, plan, budget, api, settings)
        report["baseline_plan"] = {"chunks": len(chunks),
                                   "reserved_character_units": sum(query_chars(c, [], args.model) for c in chunks)}
        if baseline is None:
            return {**report, "status": "dry_run", "dry_run": True}
    else:
        baseline = args.audio
    original_features = get_features(args, baseline, analysis, args.alignment)
    before = detect(analysis, original_features, reference)
    quality(args.out / "baseline_analysis.json", before, original_features, reference)
    report.update({"baseline_potential_errors": len(before["defects"]), "baseline_defects": before["defects"]})
    if args.mode in {"repair", "full-auto"}:
        report["proposed_repairs"] = [
            {"sentence": job["segment"]["id"], "target_ids": job["target_ids"], "rules": job["rules"],
             "reserved_character_units": query_chars(job["segment"], job["rules"], args.model)}
            for job in repair_jobs(plan, analysis, before)
        ]
    current_path, final, final_features, attempts = baseline, before, original_features, []
    if args.mode in {"repair", "full-auto"}:
        current_path, final, final_features, attempts = repair(
            args, api, plan, analysis, baseline, before, original_features, reference, settings,
        )
        destination = args.out / ("final" + current_path.suffix.lower())
        atomic_bytes(destination, current_path.read_bytes())
        report["audio"] = destination.name
    if args.ab and api:
        report["blind_ab_pairs"] = blind_pairs(args, api, plan, analysis, settings)
    quality(args.out / "final_analysis.json", final, final_features)
    accepted = [a for a in attempts if a["accepted"]]
    corrected = [
        {"token_id": i, "word": analysis["tokens"][i]["text"],
         "target": analysis["tokens"][i]["target"], "method": "matched_acoustic_comparison"}
        for attempt in accepted for i in attempt["target_ids"]
    ]
    review_ids = set(report["native_review_token_ids"])
    review_ids.update(d["token_id"] for d in final["defects"] if d.get("token_id") is not None)
    notes = []
    if reference is None:
        notes.append("No reviewed matched acoustic reference was supplied.")
    if not original_features["alignment_source"]:
        notes.append("Word, syllable and phone intervals require an alignment.")
    if not original_features["asr"]:
        notes.append("Omissions and repeats require independently recognized speech.")
    if args.mode in {"repair", "full-auto"} and args.live and not (reference and args.nofa_command and args.whisper_model):
        notes.append("Automatic acceptance requires a matched reference, NoFA and independent Whisper.")
    return {
        **report, "dry_run": not args.live, "status": final["status"], "confidence": "probabilistic",
        "potential_errors": len(final["defects"]), "defects": final["defects"],
        "native_review_token_ids": sorted(review_ids),
        "regenerations": sum("cached" in attempt for attempt in attempts), "attempts": attempts,
        "accepted_repairs": len(accepted), "corrected_targets": corrected, "remaining_uncertainties": notes,
    }


def main(argv=None):
    args = arguments(argv)
    api_key, env_voice = credentials(args.env_file) if args.live else (None, None)
    args.voice = args.voice or env_voice
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "mode": args.mode, "dry_run": not args.live}
    budget = Budget(args.max_characters, args.max_requests)
    try:
        report.update(execute(args, report, budget, api_key))
    except (APIError, OSError, RuntimeError, ValueError, subprocess.SubprocessError, ImportError) as exc:
        report.update({"status": "failed", "error": str(exc)})
        write_json(args.out / "quality_report.json", {**report, "api": budget.report()})
        raise SystemExit(str(exc)) from None
    write_json(args.out / "quality_report.json", {**report, "api": budget.report()})


if __name__ == "__main__":
    main()

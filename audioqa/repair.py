"""Run approved offline repairs and prepare bounded live replacement candidates."""
from pathlib import Path

import soundfile as sf
from filelock import FileLock

from scripts.artifact_memory import store
from scripts.common import file_digest, read_json, write_json, atomic_bytes
from scripts.forced_alignment import subtitle_cues
from scripts.generate_audio import Budget, ElevenLabs, credentials
from scripts.generate_review_dashboard import timestamp
from scripts.regenerate_defects import regenerate_event
from scripts.repair_artifacts import declick
from scripts.seamless_replace import apply_replacements, fit_clip, fades, subtitle_offsets
from scripts.segment_audio import decode, read_samples
from scripts.validate_repaired_audio import global_digital_scan, parameter_check


def verified_source(path, report):
    if file_digest(path)!=report["source"]["sha256"]:
        raise ValueError("The media differs from the scanned source.")


def make_api(args):
    if getattr(args,"local_only",False):
        raise ValueError("Live regeneration requires cloud access.")
    if args.max_api_budget<=0 or args.max_regenerations<=0:
        raise ValueError("Set positive --max-api-budget and --max-regenerations for live regeneration.")
    key,_=credentials(args.env_file)
    return ElevenLabs(Budget(args.max_api_budget,args.max_regenerations),api_key=key or "")


def click_patch(proxy, rate, frame_count, event):
    margin=round(.05*rate)
    first,last=round(event["start"]*rate),round(event["end"]*rate)
    lo,hi=max(0,first-margin),min(frame_count,last+margin)
    values=read_samples(proxy,lo,hi)
    channels=[e["data"]["channel"] for e in event["evidence"] if "channel" in e["data"]]
    repaired,metrics=declick(values,rate,first-lo,last-lo,channels)
    return {"start_sample":first,"end_sample":last,"values":repaired[first-lo:last-lo]},metrics


def repair(args):
    scan_root=args.report.resolve().parents[1]
    dest=(args.out or scan_root/"output").resolve()
    if args.dry_run:
        return run_repair(args,scan_root,dest)
    dest.mkdir(exist_ok=True,parents=True)
    with FileLock(dest/"repair.lock",timeout=0):
        return run_repair(args,scan_root,dest)


def checked_replacement(args, row, proxy, rate, mode, path):
    import tempfile

    from scripts.compare_transcripts import comparison, tokenized
    from scripts.reverse_transcribe import transcript
    from scripts.repair_artifacts import digital_signature

    context=row.get("repair_context")
    if not context or not context["text_authoritative"] or not args.whisper_model:
        raise ValueError("A finished spoken repair needs the authoritative text and local Whisper QA.")
    candidate,candidate_rate=sf.read(path,dtype="float32",always_2d=True)
    if candidate_rate!=rate:
        raise ValueError("Reviewed replacement must use the decoded audio sample rate.")
    start,end=round(context["start"]*rate),round(context["end"]*rate)
    baseline=read_samples(proxy,start,end)
    with tempfile.TemporaryDirectory(prefix="nelp-reviewed-") as temporary:
        fitted=fit_clip(candidate,rate,len(baseline),baseline.shape[1],mode,Path(temporary))
    adjusted=fades(baseline,fitted,rate)
    if digital_signature(adjusted,rate)>digital_signature(baseline,rate):
        raise ValueError("The replacement introduced a digital artifact.")
    expected=tokenized(context["text"])
    locale=context.get("language_code")
    baseline_asr=transcript("faster",args.whisper_model,baseline,rate,locale,"cpu")
    fixed_asr=transcript("faster",args.whisper_model,adjusted,rate,locale,"cpu")
    prior,_=comparison(expected,baseline_asr["words"],"baseline")
    after,_=comparison(expected,fixed_asr["words"],"candidate")
    edits=lambda events:{(e["category"],e.get("word"),tuple(e["evidence"].get("asr",[]))) for e in events}
    if not fixed_asr["words"] or not edits(after)<=edits(prior):
        raise ValueError("The replacement caused a new transcript disagreement.")
    return {"start_sample":start,"end_sample":end,"values":adjusted},{"before":prior,"after":after}


def run_repair(args,scan_root,dest):
    report=read_json(args.report)
    source=Path(args.audio).resolve()
    verified_source(source,report)
    mode="VIDEO_LOCKED" if args.video_locked else "SUBTITLE_SYNC" if args.subtitle_sync else "FREE_TIMING"
    decisions={row["id"]:row for row in (read_json(args.decisions) if args.decisions else [])}
    replacements=read_json(args.replacement_manifest) if args.replacement_manifest else {}
    selected={}
    for row in report["events"]:
        decision=decisions.get(row["id"],{})
        if decision.get("status") in {"CONFIRMED","REGENERATE","VERIFIED"}:
            selected[row["id"]]=(row,decision)
        elif not (args.approved_only or args.human_review) and args.automatic_digital and row["category"]=="click_or_pop":
            selected[row["id"]]=(row,{})
    if args.dry_run:
        print(f"{len(selected)} events eligible for candidate repairs. No files or API requests.")
        return
    dest.mkdir(exist_ok=True,parents=True)
    proxy=scan_root/"cache/decoded.wav"
    if not proxy.is_file():
        proxy=dest/"decoded.wav"
        decode(source,proxy)
    api=make_api(args) if args.live else None
    meta=read_json(args.generation_metadata) if args.generation_metadata else {}
    with sf.SoundFile(proxy) as f:
        rate,frames,channels=f.samplerate,len(f),f.channels
    attempts,patches=[],[]
    for row,decision in selected.values():
        item={"event_id":row["id"],"category":row["category"]}
        try:
            if decision.get("status")=="VERIFIED" and row["id"] in replacements:
                path=(args.replacement_manifest.parent/replacements[row["id"]]).resolve()
                patch,metrics=checked_replacement(args,row,proxy,rate,mode,path)
                if any(patch["end_sample"]>p["start_sample"] and patch["start_sample"]<p["end_sample"] for p in patches):
                    raise ValueError("This patch overlaps an earlier repair.")
                patches.append(patch)
                item.update({"status":"REPAIRED","method":"reviewed_sentence_replacement","asr":metrics})
            elif row["category"]=="click_or_pop" and decision.get("status")!="REGENERATE":
                patch,metrics=click_patch(proxy,rate,frames,row)
                if any(patch["end_sample"]>p["start_sample"] and patch["start_sample"]<p["end_sample"] for p in patches):
                    raise ValueError("This patch overlaps an earlier repair.")
                patches.append(patch)
                item.update({"status":"REPAIRED","method":"sample_interpolation",**metrics})
            elif api and decision.get("status")=="REGENERATE":
                patch,provider=regenerate_event(api,proxy,rate,row,meta,mode)
                file=dest/f"{row['id']}-candidate.wav"
                sf.write(file,patch["values"],rate,subtype="FLOAT")
                item.update({"status":"HUMAN_REVIEW_REQUIRED","method":"elevenlabs_sentence_candidate",
                             "candidate":file.name,"cached":provider["cached"]})
                if args.whisper_model:
                    from scripts.compare_transcripts import comparison, tokenized
                    from scripts.reverse_transcribe import transcript

                    samples,sample_rate=sf.read(file,dtype="float32",always_2d=True)
                    actual=transcript("faster",args.whisper_model,samples,sample_rate,
                                      row["repair_context"].get("language_code"),"cpu")
                    findings,_=comparison(tokenized(row["repair_context"]["text"]),actual["words"],"independent_whisper")
                    item["candidate_transcript_findings"]=findings
            else:
                raise ValueError("This event needs a suitable repair method and reviewed regeneration context.")
        except (ValueError,OSError,RuntimeError,ImportError) as exc:
            item.update({"status":"HUMAN_REVIEW_REQUIRED","reason":str(exc)})
        attempts.append(item)
        if decision.get("status") in {"CONFIRMED","VERIFIED"} and row.get("assets"):
            original=scan_root/"review"/row["assets"]["defect"]
            if original.is_file() and decision.get("reviewed_at"):
                store(row,decision,original,dest/item["candidate"] if item.get("candidate") else None,meta)
    final=None
    validation=None
    if patches:
        candidate=dest/"repaired.wav"
        if candidate==source:
            raise ValueError("Choose an output that differs from the original media.")
        apply_replacements(proxy,candidate,patches,rate,channels)
        validation=parameter_check(proxy,candidate,mode=="VIDEO_LOCKED")
        before=global_digital_scan(proxy)
        after=global_digital_scan(candidate)
        accepted=all(after.get(k,0)<=before.get(k,0) for k in after)
        validation.update({"before":before,"after":after,"accepted":accepted})
        if accepted:
            final=candidate.name
            if args.subtitle_sync:
                cues=subtitle_offsets(subtitle_cues(args.subtitle_sync),patches,rate)
                blocks=[f"{i}\n{timestamp(c['start'],True)} --> {timestamp(c['end'],True)}\n{c['text']}\n"
                        for i,c in enumerate(cues,1)]
                atomic_bytes(dest/"updated-subtitles.srt","\n".join(blocks).encode())
        else:
            candidate.rename(dest/"rejected-candidate.wav")
            for item in attempts:
                if item["status"]=="REPAIRED":
                    item["status"]="HUMAN_REVIEW_REQUIRED"
                    item["reason"]="Global QC rejected the assembled candidate."
    if report["events"]:
        for row in report["events"]:
            decision=decisions.get(row["id"],{})
            if decision.get("status")=="FALSE_POSITIVE" and decision.get("reviewed_at") and row.get("assets"):
                original=scan_root/"review"/row["assets"]["defect"]
                if original.is_file():
                    store(row,decision,original,metadata=meta)
    write_json(dest/"repair_report.json",{"original":str(source),"final":final,"timing_mode":mode,
               "attempts":attempts,"validation":validation,"api":api.budget.report() if api else {"credits_used":0}})
    print(dest/"repair_report.json")

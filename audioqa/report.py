"""Join manuscript, ASR, audio and independent evidence on the decoded timeline."""
import bisect
import os
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audio_reviewer import GeminiReviewer
from scripts.benchmark_detectors import union_duration
from scripts.common import key, read_json, read_text, write_json
from scripts.compare_transcripts import comparison, low_confidence
from scripts.detect_language_switch import route, transition_events, unexpected_window
from scripts.detect_mispronunciation import pronunciation_events
from scripts.detect_voice_anomalies import ranked_voice_events, speech_events
from scripts.evidence_fusion import fuse
from scripts.extract_defect_clips import extract
from scripts.forced_alignment import align_span, subtitle_cues
from scripts.locale_lexicons import LexicalRouter
from scripts.phoneme_alignment import observed_phones
from scripts.pronunciation_assessment import repeated_word_timing
from scripts.refine_signal import refined
from scripts.reverse_transcribe import ASR_LOCALES, LOCALES, reconcile_words
from scripts.segment_audio import read_samples

SEVERITY={"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3,"INFO":4}
LOCALE_CODE={"nb-NO":"no","uk-UA":"uk","en-US":"en","en-GB":"en"}


def attach_asr(raw, extra):
    if not extra:
        return
    cache=Path(extra)
    if (cache/"cache").is_dir():
        cache=cache/"cache"
    manifest=read_json(cache/"manifest.json")
    if manifest["source"]["sha256"]!=raw["source"]["sha256"] or manifest["sample_rate"]!=raw["sample_rate"]:
        raise ValueError("Supplemental ASR evidence must use the same source and decoded clock.")
    for window in raw["windows"]:
        fragment=cache/f"window-{window['index']:06d}.json"
        if not fragment.exists():
            continue
        local=read_json(fragment)
        if any(local[name]!=window[name] for name in ("start","end","core_start","core_end")):
            raise ValueError("Supplemental ASR evidence has a mismatched window schedule.")
        window["transcripts"] += [{**t,"provider":t["provider"]+"_supplement"} for t in local["transcripts"]]


def transcript_rows(raw, rate):
    rows=defaultdict(list)
    counts=Counter()
    for window in raw["windows"]:
        duration=(window["core_end"]-window["core_start"])/rate
        features=window["features"]
        for provider in window["transcripts"]:
            label=f"asr:{provider['family']}:{provider['provider']}"
            rows[label].extend(provider["words"])
            counts[(label,provider.get("detected_language") or "und")] += 1
        if window["transcripts"] and duration and features:
            model=window["transcripts"][0]
            features["words_per_second"]=len(model["words"])/duration
            gaps=[b["start"]-a["end"] for a,b in zip(model["words"],model["words"][1:]) if b["start"]>a["end"]+.12]
            if gaps:
                features["median_pause_s"]=sorted(gaps)[len(gaps)//2]
    return {name:reconcile_words(words) for name,words in rows.items()}, [
        {"provider":p,"language":language,"windows":count} for (p,language),count in sorted(counts.items())
    ]


def manuscript(args, fallback, reverse, raw):
    provided=args.reference
    cues=[]
    if provided and provided.suffix.lower() in {".srt",".vtt"}:
        cues=subtitle_cues(provided)
        text="\n\n".join(cue["text"] for cue in cues)
    elif provided:
        text=read_text(provided)
    else:
        breaks={w["core_start"]/raw["sample_rate"] for w in raw["windows"]}
        text_parts=[]
        prior=-1
        markers=sorted(breaks)
        for word in reverse:
            group=bisect.bisect_right(markers,(word["start"]+word["end"])/2)
            text_parts.append(("\n\n" if group!=prior else " ") + word["text"])
            prior=group
        text="".join(text_parts).lstrip()
    overrides=read_json(args.language_spans) if args.language_spans else []
    routing=route(text,fallback,args.english,overrides) if text else {"text":"","tokens":[],"transitions":[],"sentences":[],"paragraphs":[]}
    return routing,cues


def vicinity(words, starts, begin, end, padding=1.35):
    at=max(0,bisect.bisect_left(starts,begin-padding)-1)
    upto=bisect.bisect_right(starts,end+padding)
    return [word for word in words[at:upto] if word["end"]>=begin-padding]


def force_routed(args, raw, routed, timed):
    if not (args.nofa_command or args.mfa_command) or not timed or not args.max_alignments:
        return []
    rate=raw["sample_rate"]
    models=read_json(args.mfa_models) if args.mfa_models else {}
    results=[]
    group=[]
    groups=[]
    for token in routed["tokens"]:
        interval=timed.get(token["id"])
        if not interval or (group and (token["expected_language"]!=group[-1][0]["expected_language"]
                                      or interval["end"]-group[0][1]["start"]>22
                                      or interval["start"]-group[-1][1]["end"]>1.5)):
            if group:
                groups.append(group)
                group=[]
        if interval:
            group.append((token,interval))
    if group:
        groups.append(group)
    eligible=[group for group in groups if len(group)>=3 and (
        (group[0][0]["expected_language"]=="nb-NO" and args.nofa_command)
        or (args.mfa_command and group[0][0]["expected_language"] in models)
    )]
    if len(eligible)>args.max_alignments:
        eligible=[eligible[i*len(eligible)//args.max_alignments] for i in range(args.max_alignments)]
    for group in eligible:
        first, last=group[0][1]["start"],group[-1][1]["end"]
        start=max(0,round((first-.18)*rate)); stop=min(raw["samples"],round((last+.18)*rate))
        locale=group[0][0]["expected_language"]
        try:
            aligned=align_span(read_samples(raw["proxy"],start,stop),rate," ".join(t["text"] for t,_ in group),locale,
                               args.nofa_command,args.mfa_command,models)
        except (OSError,ValueError,RuntimeError) as exc:
            results.append({"start":start/rate,"end":stop/rate,"locale":locale,"error":str(exc)})
            continue
        if not aligned:
            continue
        for kind in ("words","phones"):
            for row in aligned[kind]:
                row.update({"start":row["start"]+start/rate,"end":row["end"]+start/rate})
        results.append({"start":start/rate,"end":stop/rate,"locale":locale,**aligned})
    return results


def report_selection(events, maximum):
    if maximum is None or len(events)<=maximum:
        return events
    result=sorted(events,key=lambda item:(SEVERITY[item["severity"]],item["start"]))[:maximum]
    return sorted(result,key=lambda item:item["start"])


def content_events(args, raw, transcripts, routed, primary, timed):
    rate=raw["sample_rate"]
    events=[]
    label=next(iter(transcripts),None)
    if not label or args.artifact_only:
        return events
    for name, words in transcripts.items():
        if args.reference:
            differences,matches=comparison(routed["tokens"],words,name)
            events.extend(differences)
            if name==label:
                timed.update({routed["tokens"][a]["id"]:words[b] for a,b in matches})
        else:
            events.extend(low_confidence(words,name,.15 if args.high_recall else .08 if args.high_precision else .11))
        if name!=label and not name.endswith("_supplement"):
            differences,_=comparison(primary,words,name)
            for event in differences:
                if event["category"]!="transcript_unmatched_block":
                    events.append({**event,"category":"asr_disagreement","evidence":{**event["evidence"],"primary":label}})
    for window in raw["windows"]:
        if not window["transcripts"]:
            continue
        main=window["transcripts"][0]
        for secondary in window["transcripts"][1:]:
            if not secondary["provider"].endswith("_supplement") or not secondary["words"]:
                continue
            label=f"asr:{secondary['family']}:{secondary['provider']}"
            differences,_=comparison(main["words"],secondary["words"],label,limit=4)
            for event in differences:
                if event["category"]!="transcript_unmatched_block" and event["end"]-event["start"]<=1.6:
                    events.append({**event,"category":"asr_disagreement","evidence":{**event["evidence"],"primary":main["provider"]}})
    if not args.reference and len(primary)==len(routed["tokens"]):
        timed.update({row["id"]:word for row,word in zip(routed["tokens"],primary)})
    if args.reference and timed:
        timing_rows=[(timed[row["id"]]["start"],row) for row in routed["tokens"] if row["id"] in timed]
        for window in raw["windows"]:
            start,end=window["core_start"]/rate,window["core_end"]/rate
            local=[row for timestamp,row in timing_rows if start<=timestamp<end]
            events.extend(unexpected_window(window,local,timed))
        events.extend(transition_events(routed,timed,None))
    phones=observed_phones(args.phones)
    if phones:
        with LexicalRouter(args.dialect,args.ukrainian_stress) as lexicons:
            events.extend(pronunciation_events(routed,timed,lexicons,phones))
    for window in raw["windows"]:
        for entry in window["transcripts"]:
            for vocal in entry.get("audio_events",[]):
                events.append({"category":"possible_vocalization","start":vocal["start"],"end":vocal["end"],
                               "detector":"asr:scribe:audio_events","timing":"asr_estimate","evidence":vocal})
    return events


def attach_context(events, routed, primary, timed, cues, trusted, waveform_duration):
    starts=[item["start"] for item in primary]
    sentences=[]
    takings={row["sentence_id"] for row in routed["tokens"] if row["id"] in timed}
    for section in routed["sentences"]:
        if section["sentence"] not in takings:
            continue
        members=[row for row in routed["tokens"] if row["sentence_id"]==section["sentence"]]
        if not members or not (members[0]["id"] in timed and members[-1]["id"] in timed):
            continue
        words=[timed[row["id"]] for row in members if row["id"] in timed]
        if len(words)/len(members)<.8 or words[-1]["end"]-words[0]["start"]>25:
            continue
        locales={row["expected_language"] for row in members}
        lo,hi=words[0]["start"],words[-1]["end"]
        sentences.append({"text":section["text"],"start":max(0,lo-.10),"end":min(waveform_duration,hi+.10),
                          "language_code":LOCALE_CODE[next(iter(locales))] if len(locales)==1 else None,
                          "text_authoritative":trusted,"clock":"asr_estimate"})
    if cues:
        sentences=[{**cue,"text_authoritative":trusted,"language_code":None,"clock":"subtitle_clock"} for cue in cues]
    for row in events:
        local=vicinity(primary,starts,row["start"],row["end"])
        event_mid=(row["start"]+row["end"])/2
        row["asr_excerpt"]=" ".join(w["text"] for w in local[:45])
        matching=next((section for section in sentences if section["start"]<=event_mid<=section["end"]),None)
        if matching:
            row["sentence"]=row.get("sentence") or matching["text"]
            if trusted and row["severity"]!="INFO":
                row["repair_context"]=matching
        if not trusted and (matching or local):
            row["sentence_origin"]="reverse_transcription"
    return events


def produce(raw,dest,args,fallback,sensitivity):
    attach_asr(raw,args.extra_asr_cache)
    rate=raw["sample_rate"]
    duration=raw["samples"]/rate
    transcripts, languages=transcript_rows(raw,rate)
    primary=next(iter(transcripts.values()),[])
    routed,cues=manuscript(args,fallback,primary,raw)
    timed={}
    source_events=content_events(args,raw,transcripts,routed,primary,timed)
    alignments=force_routed(args,raw,routed,timed)
    if args.deep and alignments:
        with LexicalRouter(args.dialect,args.ukrainian_stress) as lexicons:
            source_events+=repeated_word_timing(alignments,lexicons)
    if not args.pronunciation_only:
        signal=[row for window in raw["windows"] for row in window["events"]]
        if args.deep:
            signal+=refined(raw,sensitivity,Path(dest)/"cache",args.workers)
        clean=read_json(args.voice_profile) if args.voice_profile else None
        metadata=read_json(args.generation_metadata) if args.generation_metadata else {}
        if clean and clean.get("metadata",{}).get("voice_id") != metadata.get("voice_id"):
            raise ValueError("Voice profile requires matching source voice_id metadata.")
        acoustic=signal+ranked_voice_events(raw["windows"],rate,clean,sensitivity)
        if not args.reference:
            starts=sorted(event["start"] for event in acoustic)
            source_events=[event for event in source_events if event["category"]!="asr_disagreement" or
                           bisect.bisect_left(starts,event["start"]-.25)<bisect.bisect_right(starts,event["end"]+.25)]
        source_events+=acoustic
        if not args.artifact_only and (args.reference or args.deep):
            source_events+=speech_events(transcripts,1.3 if args.high_recall else 1.55)
    events=fuse(source_events,raw["source"],duration)
    total=len(events)
    events=report_selection(events,args.max_candidates)
    attach_context(events,routed,primary,timed,cues,bool(args.reference),duration)
    extract(raw["proxy"],events,Path(dest)/"review/clips",rate,raw["samples"],pictures=not args.no_plots)
    limitations=[
        "Events require review. Sample-clock, window and ASR locations have different precision.",
        "Forced alignment places expected phones and does not certify pronunciation.",
    ]
    if not args.reference:
        limitations.append("No source manuscript was supplied. Reverse transcript snippets are hypotheses.")
    if not args.phones:
        limitations.append("No independent phoneme recognizer evidence was supplied.")
    if not transcripts:
        limitations.append("Run with a local ASR model to analyze speech content.")
    if not args.voice_profile:
        limitations.append("No reviewed acoustic voice profile was supplied.")
    if len(events)!=total:
        limitations.append(f"{total-len(events)} candidates were omitted by the explicit export limit.")
    defect_counts=Counter((row.get("expected_language") or "und",row["category"]) for row in events)
    return {
        "schema_version":1,"source":raw["source"],"media":raw["media"],"duration_s":duration,"sample_rate":rate,
        "windows":len(raw["windows"]),"events":events,"all_candidate_count":total,"captions_authoritative":bool(args.reference),
        "reverse_transcript":transcripts,"window_languages":languages,"language_strategy":fallback,
        "language_routes":routed,"alignment_results":alignments,
        "analysis_by_language":[{"language":language,"category":cat,"candidates":count}
                                for (language,cat),count in sorted(defect_counts.items())],
        "acoustic_windows":[{**w["features"],"start":w["core_start"]/rate,"end":w["core_end"]/rate} for w in raw["windows"]],
        "review_listening_minutes":union_duration((row["clip_start"],row["clip_end"]) for row in events)/60,
        "candidate_listening_minutes":union_duration((row["clip_start"],row["clip_end"])
                                                     for row in events if row["severity"]!="INFO")/60,
        "validation_metrics":None,"native_confirmed_defects":0,"limitations":limitations,
        "api":{"scribe_audio_seconds_reserved":raw["cloud_duration_requested"],"unknown_provider_outcomes":raw["cloud_unknown_outcomes"],
               "credits_used":0 if raw["cloud_duration_requested"]==0 else None},
    }


def review_cloud(report,args,dest):
    reviewer=GeminiReviewer(args.audio_reviewer,os.environ.get("GEMINI_API_KEY"))
    consumed=0
    for row in sorted(report["events"],key=lambda event:SEVERITY[event["severity"]])[:args.max_review_calls]:
        key=Path(dest)/"cache"/f"review-{row['id']}-{args.audio_reviewer}.json"
        if key.exists():
            prior=read_json(key)
            if "observations" in prior:
                row["multimodal_review"]=prior["observations"]
            continue
        write_json(key,{"state":"attempted"})
        context={f:row.get(f) for f in ("sentence","asr_excerpt","expected_language","expected_pronunciation","evidence")}
        context["dialect"]=args.dialect
        values=reviewer.review(Path(dest)/"review"/row["assets"]["context"],context)
        duration=row["clip_end"]-row["clip_start"]
        localized=[{**entry,"global_start_estimate":row["clip_start"]+entry["start"],
                    "global_end_estimate":row["clip_start"]+entry["end"],"confirmed":False}
                   for entry in values if 0<=entry["start"]<entry["end"]<=duration+.05]
        consumed+=1
        row["multimodal_review"]=localized
        write_json(key,{"observations":localized})
        write_json(Path(dest)/"review"/row["assets"]["evidence"],row)
    report["api"]["multimodal_calls"]=consumed
    if consumed:
        report["api"]["credits_used"]=None

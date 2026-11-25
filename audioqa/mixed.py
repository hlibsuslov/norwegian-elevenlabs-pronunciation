"""Plan or audition bounded mixed-language synthesis strategies."""
from collections import defaultdict
import re

from scripts.build_dictionary import record_rule, verified_records
from scripts.common import atomic_bytes, key, model_config, read_json, read_text, write_json
from scripts.detect_language_switch import route
from scripts.generate_audio import Budget, ElevenLabs, credentials, pcm_wav, units
from .report import LOCALE_CODE


def language_boundaries(routing, preserve_short=False):
    tokens=routing["tokens"]
    active=[]
    for token in tokens:
        locale=token["expected_language"]
        if not active or active[-1][-1]["expected_language"]!=locale:
            active.append([])
        active[-1].append(token)
    points=[0]
    for i, span in enumerate(active):
        if i==0:
            continue
        if preserve_short:
            gap=routing["text"][active[i-1][-1]["end"]:span[0]["start"]]
            if len(span)<5 or not re.search(r"[.!?;]|\n\n",gap):
                continue
        cut=span[0]["start"]
        if not preserve_short:
            gap=routing["text"][active[i-1][-1]["end"]:cut]
            quote_start=re.search(r"[«“\"]\s*$",gap)
            if quote_start:
                cut=active[i-1][-1]["end"]+quote_start.start()
        points.append(cut)
    return points+[len(routing["text"])]


def boundaries(routing,strategy):
    text=routing["text"]
    if strategy=="single":
        return [0,len(text)]
    if strategy=="blocks":
        points=[row["start"] for row in routing["paragraphs"]]
        return sorted({0,len(text),*points})
    return language_boundaries(routing,preserve_short=strategy=="hybrid")


def scoped_rules(routing,tokens,records,model,voice):
    all_tokens=routing["tokens"]
    sentences={row["sentence"]:row for row in routing["sentences"]}
    proposals=defaultdict(list)
    for token in tokens:
        locale=token["expected_language"]
        adjacent=[
            all_tokens[i]["expected_language"] if 0<=i<len(all_tokens) else None
            for i in (token["id"]-1, token["id"]+1)
        ]
        sentence=sentences[token["sentence_id"]]
        eligible=[
            row for row in records
            if row["model"]==model and row["voice_id"] in (None,voice)
            and row.get("word_language","nb-NO")==locale
            and (locale!="nb-NO" or row["dialect"]=="e_written")
            and row.get("context_language") in (None,sentence["expected_language"])
            and row.get("previous_language") in (None,adjacent[0]) and row.get("next_language") in (None,adjacent[1])
            and key(row["word"])==key(token["text"]) and row["scope"]=="sentence" and key(row["context"])==key(token["sentence"])
        ]
        record=max(eligible,key=lambda row:sum(bool(row.get(name)) for name in
                    ("voice_id","context_language","previous_language","next_language"))) if eligible else None
        proposals[token["text"]].append((locale,record))
    rules=[]
    for surface,entries in proposals.items():
        if any(record is None for locale,record in entries):
            continue
        if len({(locale,record["method"],record["ipa"],record["replacement"]) for locale,record in entries})!=1:
            continue
        locale,record=entries[0]
        rules.append({**record_rule(record,surface),"language":"en" if locale.startswith("en-") else locale})
    return rules


def segments(routing,strategy,model,voice,records=()):
    points=boundaries(routing,strategy)
    source=routing["text"]
    limit=model_config(model)["limit"]
    result=[]
    for start,end in zip(points,points[1:]):
        text=source[start:end].strip()
        if not text:
            continue
        if units(text)>limit:
            raise ValueError(f"{strategy} yielded a segment larger than the model limit. Add paragraph boundaries.")
        # End checks are also needed for a standalone English phrase split from an annotated context.
        tokens=[row for row in routing["tokens"] if start<=row["start"]<end]
        locales={row["expected_language"] for row in tokens}
        code=LOCALE_CODE[next(iter(locales))] if len(locales)==1 else None
        result.append({"start":start,"end":end,"text":text,"language_code":code,
                       "locales":sorted(locales),"previous_text":source[max(0,start-100):start],
                       "next_text":source[end:end+100],"rules":scoped_rules(routing,tokens,records,model,voice)})
    if not result:
        raise ValueError("The mixed manuscript is empty.")
    return result


def mixed(args):
    text=read_text(args.text)
    spans=read_json(args.spans) if args.spans else []
    routing=route(text,english=args.english,overrides=spans)
    secret,env_voice=credentials(args.env_file) if args.live else (None,None)
    voice=args.voice or env_voice
    result=segments(routing,args.strategy,args.model,voice,verified_records())
    args.out.mkdir(parents=True,exist_ok=False)
    atomic_bytes(args.out/"original.txt",args.text.read_bytes())
    write_json(args.out/"mixed_plan.json",{"strategy":args.strategy,"model":args.model,"voice_id":voice,"routing":routing,
                                         "segments":result,"dry_run":not args.live})
    if not args.live:
        return
    if args.max_api_budget<=0 or args.max_regenerations<=0 or not voice:
        raise ValueError("Explicit voice and live character/request budgets are required.")
    budget=Budget(args.max_api_budget,args.max_regenerations)
    api=ElevenLabs(budget,api_key=secret or "")
    candidates=[api.estimate(args.model,voice,s,s["rules"],None,0) for s in result]
    budget.check(sum(chars for chars,_ in candidates),sum(calls for _,calls in candidates))
    audio=bytearray()
    timings=[]
    try:
        for segment in result:
            pcm,details=api.generate(args.model,voice,segment,segment["rules"])
            audio.extend(pcm)
            timings.append({"start":segment["start"],"end":segment["end"],"cached":details["cached"]})
        atomic_bytes(args.out/"candidate.wav",pcm_wav(audio))
    finally:
        write_json(args.out/"mixed_generation.json",{"api":budget.report(),"segments_generated":timings,
                                                    "accepted":False,"status":"native_review_pending"})

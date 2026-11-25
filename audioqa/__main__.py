"""Scan, review, repair and validate multilingual educational voiceovers."""
import argparse
import os
import re
import shutil
from pathlib import Path
from time import perf_counter

from scripts.audio_embeddings import profile
from scripts.benchmark_detectors import benchmark
from scripts.common import read_json, read_text, write_json
from scripts.detect_language_switch import route
from scripts.generate_audio import credentials
from scripts.generate_review_dashboard import export_report
from scripts.reverse_transcribe import FAMILIES
from scripts.scan_audio import scan_raw
from scripts.segment_audio import discover_audio, ff
from .report import produce, review_cloud
from .repair import repair
from .review_server import serve

LOCALE={"no":"nb-NO","nb":"nb-NO","nb-NO":"nb-NO","uk":"uk-UA","uk-UA":"uk-UA",
        "en":"en-US","en-US":"en-US","en-GB":"en-GB","auto":None}


def scan_options(command):
    command.add_argument("--audio", required=True)
    command.add_argument("--out", type=Path, default=Path("runs/audioqa"))
    command.add_argument("--reference", type=Path, help="Exact manuscript TXT, SRT or VTT.")
    command.add_argument("--language", choices=LOCALE, default="no", help="Expected script fallback. ASR still detects window language.")
    command.add_argument("--english", choices=["en-US","en-GB"], default="en-US")
    command.add_argument("--language-spans", type=Path)
    command.add_argument("--dialect", default="e_written")
    command.add_argument("--ukrainian-stress", type=Path)
    command.add_argument("--phones", type=Path, help="Independent phoneme observations JSON.")
    command.add_argument("--voice-profile", type=Path)
    command.add_argument("--generation-metadata", type=Path)
    command.add_argument("--deep", action="store_true")
    command.add_argument("--max-alignments", type=int, default=16)
    command.add_argument("--nofa-command")
    command.add_argument("--mfa-command")
    command.add_argument("--mfa-models", type=Path, help="Map locale to local [dictionary, acoustic model].")
    command.add_argument("--asr", action="append", default=[], metavar="ENGINE=MODEL_PATH")
    command.add_argument("--extra-asr-cache", type=Path)
    command.add_argument("--scribe", action="store_true")
    command.add_argument("--audio-reviewer", metavar="GEMINI_MODEL")
    command.add_argument("--max-review-calls", type=int, default=3)
    command.add_argument("--max-cloud-seconds", type=float, default=0)
    command.add_argument("--allow-cloud", action="store_true")
    command.add_argument("--local-only", action="store_true")
    command.add_argument("--env-file", type=Path)
    command.add_argument("--resume", action="store_true")
    command.add_argument("--workers", type=int, default=2)
    command.add_argument("--max-candidates", type=int, default=None, help="Explicit limit on exported review candidates.")
    command.add_argument("--no-plots", action="store_true")
    choice=command.add_mutually_exclusive_group()
    choice.add_argument("--gpu", action="store_true")
    choice.add_argument("--cpu", action="store_true")
    choice=command.add_mutually_exclusive_group()
    choice.add_argument("--high-recall", action="store_true")
    choice.add_argument("--high-precision", action="store_true")
    choice=command.add_mutually_exclusive_group()
    choice.add_argument("--artifact-only", action="store_true")
    choice.add_argument("--pronunciation-only", action="store_true")
    command.add_argument("--dry-run", action="store_true")


def repair_options(command):
    command.add_argument("--decisions", type=Path)
    command.add_argument("--approved-only", action="store_true")
    command.add_argument("--automatic-digital", action="store_true")
    command.add_argument("--human-review", action="store_true")
    command.add_argument("--live", action="store_true")
    command.add_argument("--max-api-budget", type=int, default=0)
    command.add_argument("--max-regenerations", type=int, default=0)
    command.add_argument("--video-locked", action="store_true")
    command.add_argument("--subtitle-sync", type=Path)
    command.add_argument("--whisper-model", type=Path, help="Local multilingual CT2 model for regenerated candidates.")
    command.add_argument("--replacement-manifest", type=Path, help="Map VERIFIED event IDs to reviewed replacement WAVs.")


def make_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    for mode in ("scan","validate","full-auto"):
        command=commands.add_parser(mode)
        scan_options(command)
        if mode=="full-auto":
            repair_options(command)
    command=commands.add_parser("repair")
    command.add_argument("--audio", type=Path, required=True)
    command.add_argument("--report", type=Path, required=True)
    command.add_argument("--out", type=Path)
    command.add_argument("--env-file", type=Path)
    command.add_argument("--generation-metadata", type=Path)
    command.add_argument("--local-only", action="store_true")
    command.add_argument("--dry-run", action="store_true")
    repair_options(command)
    command=commands.add_parser("review")
    command.add_argument("--report",type=Path,required=True)
    command.add_argument("--serve",action="store_true")
    command.add_argument("--env-file",type=Path)
    command.add_argument("--generation-metadata",type=Path)
    command.add_argument("--port",type=int,default=8765)
    command.add_argument("--max-api-budget",type=int,default=0)
    command.add_argument("--max-regenerations",type=int,default=0)
    command=commands.add_parser("benchmark")
    command.add_argument("--report",type=Path,required=True)
    command.add_argument("--annotations",type=Path,required=True)
    command.add_argument("--out",type=Path,required=True)
    command.add_argument("--tolerance",type=float,default=.3)
    command=commands.add_parser("voice-profile")
    command.add_argument("--reports",type=Path,nargs="+",required=True)
    command.add_argument("--metadata",type=Path,required=True)
    command.add_argument("--out",type=Path,required=True)
    command=commands.add_parser("plan-mixed")
    command.add_argument("--text",type=Path,required=True)
    command.add_argument("--spans",type=Path)
    command.add_argument("--english",choices=["en-US","en-GB"],default="en-US")
    command.add_argument("--strategy",choices=["single","blocks","language","hybrid"],default="hybrid")
    command.add_argument("--model",default="eleven_v4")
    command.add_argument("--voice")
    command.add_argument("--env-file",type=Path)
    command.add_argument("--max-api-budget",type=int,default=0)
    command.add_argument("--max-regenerations",type=int,default=0)
    command.add_argument("--live",action="store_true")
    command.add_argument("--out",type=Path,required=True)
    return parser


def model_flags(specs):
    providers=[]
    if not specs:
        default=os.environ.get("AUDIOQA_WHISPER_MODEL")
        specs=["faster="+default] if default else []
    for spec in specs:
        kind,delimiter,filename=spec.partition("=")
        if not delimiter or kind not in FAMILIES or not Path(filename).exists():
            raise ValueError("--asr expects a supported ENGINE=LOCAL_MODEL_PATH.")
        providers.append((kind,Path(filename)))
    return providers


def media_inputs(location, output):
    if not location.startswith("rclone:"):
        return discover_audio(location)
    remote=location.removeprefix("rclone:")
    if not re.fullmatch(r"[^:]+:.+\.(wav|mp3|flac|m4a|mp4)",remote,re.I):
        raise ValueError("Use rclone:REMOTE:path/to/file with an authorized, configured Drive remote.")
    directory=output/"cache/drive"
    directory.mkdir(exist_ok=True,parents=True)
    local=directory/Path(remote.split(":",1)[1]).name
    ff(["rclone","copyto","--",remote,str(local)],timeout=7200)
    return [local]


def scan(args):
    if not 1<=args.workers<=16 or args.max_alignments<0 or args.max_review_calls<0 or (
        args.max_candidates is not None and args.max_candidates<0
    ):
        raise ValueError("Invalid worker or candidate limits.")
    if (args.scribe or args.audio_reviewer) and (not args.allow_cloud or args.local_only):
        raise ValueError("Cloud review requires --allow-cloud and a separate explicit provider selection.")
    if args.scribe and args.max_cloud_seconds<=0:
        raise ValueError("Scribe requires a positive --max-cloud-seconds.")
    if args.command=="full-auto" and args.live and args.local_only:
        raise ValueError("--local-only is incompatible with live synthesis.")
    if args.dry_run:
        print("Dry run. No audio decoded, generated or uploaded.")
        return []
    if args.command=="full-auto" and args.live and (args.max_api_budget<=0 or args.max_regenerations<=0):
        raise ValueError("Set both the TTS character budget and request limit.")
    media=media_inputs(args.audio,args.out)
    if not media:
        raise ValueError("No supported media found in the directory.")
    if len(media)>1 and (args.reference or args.extra_asr_cache or args.phones):
        raise ValueError("Use one scan per media file when supplying file-specific evidence.")
    if len(media)>1 and (args.scribe or args.audio_reviewer or (args.command=="full-auto" and args.live)):
        raise ValueError("Cloud budgets apply to one file per run.")
    providers=[] if args.artifact_only else model_flags(args.asr)
    scribe_key=None
    if args.scribe:
        scribe_key,_=credentials(args.env_file)
        if not scribe_key:
            raise ValueError("Scribe requires an ElevenLabs key.")
    sensitivity="recall" if args.high_recall else "precision" if args.high_precision else "balanced"
    results=[]
    for index,path in enumerate(media):
        start=perf_counter()
        dest=args.out if len(media)==1 else args.out / f"{index+1:03d}-{path.stem}"
        raw=scan_raw(path,dest,providers=providers,device="cuda" if args.gpu else "cpu",workers=args.workers,
                     deep=args.deep,sensitivity=sensitivity,signal=not args.pronunciation_only,expected_locale=LOCALE[args.language],
                     resume=args.resume,scribe_key=scribe_key,cloud_seconds=args.max_cloud_seconds)
        report=produce(raw,dest,args,LOCALE[args.language] or "nb-NO",sensitivity)
        report["processing_seconds_this_run"]=perf_counter()-start
        if args.audio_reviewer:
            review_cloud(report,args,dest)
        export_report(dest,report)
        results.append((path,dest,report))
        print(f"{path.name}: {len(report['events'])} candidates. {dest/'review/index.html'}",flush=True)
    return results


def main():
    parser=make_parser()
    args=parser.parse_args()
    try:
        if args.command in {"scan","validate","full-auto"}:
            for path,dest,report in scan(args):
                if args.command=="full-auto" and not args.human_review:
                    if not (args.decisions or args.automatic_digital):
                        continue
                    args.report=dest/"reports/defects.json"
                    args.audio=path
                    args.out=dest/"output"
                    repair(args)
        elif args.command=="review":
            report=read_json(args.report)
            if args.serve:
                serve(args,report)
            else:
                print((args.report.resolve().parents[1]/"review/index.html").as_uri())
        elif args.command=="repair":
            repair(args)
        elif args.command=="benchmark":
            report=read_json(args.report)
            metrics=benchmark(report["events"],read_json(args.annotations),args.tolerance)
            write_json(args.out,{**metrics,"processing_seconds":report.get("processing_seconds_this_run"),
                                 "api":report.get("api")})
        elif args.command=="voice-profile":
            records=[read_json(path) for path in args.reports]
            meta=read_json(args.metadata)
            if not meta.get("clean_review_evidence") or not meta.get("voice_id"):
                raise ValueError("Voice profile metadata needs voice_id and clean_review_evidence.")
            write_json(args.out,profile([chunk for record in records for chunk in record["acoustic_windows"]],meta))
        else:
            from .mixed import mixed
            mixed(args)
    except (ValueError,OSError,RuntimeError,ImportError) as exc:
        parser.exit(1,str(exc)+"\n")


if __name__=="__main__":
    main()

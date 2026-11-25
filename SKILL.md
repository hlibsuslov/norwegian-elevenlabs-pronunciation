---
name: norwegian-elevenlabs-pronunciation
description: Scan ElevenLabs audio for artifacts, Norwegian pronunciation problems, and language interference. Use for long MP3 or WAV recordings, review and targeted repair, Norwegian Bokmål stress and lexical tone, pronunciation dictionaries, and Bokmål, Ukrainian and English educational voiceovers.
---

# Norwegian ElevenLabs pronunciation

Default to Bokmål with the NB Uttale `e_written` profile for Eastern Norwegian read speech. Ask for a narrower dialect only when it affects a consequential pronunciation decision. Review [installation and examples](README.md) if the local index or environment is missing.

Run `python -m scripts.workflow MODE` from this directory. Use a new output directory per run.

| Mode | Use |
| --- | --- |
| `prevent` | Analyze the manuscript and produce a model-specific plan and scoped PLS files. |
| `detect` | Examine an existing recording with Praat features and, when supplied, NoFA, Whisper and a matched reference. |
| `repair` | Propose or regenerate flagged sentences and accept a splice only after local validation. |
| `full-auto` | Plan a generation or synthesize the baseline, run QA and apply independently validated repairs. Use `--ab` for blinded experimental samples. |
| `learn` | Store and export a correction with documented source, compatibility, native review and acoustic evidence. |

- Keep `original.txt` untouched. Prefer source-attested targets and mark ambiguous or unavailable pronunciations for review. Consult [phonology](references/norwegian_phonology.md), [stress](references/stress_rules.md), [tonal accents](references/tonal_accents.md), or [compounds](references/compound_words.md) as needed.
- Use [model compatibility](references/elevenlabs_compatibility.md) to choose the transport and eligible rules. Accent digits and lexical tone labels belong in the report. Scope pronunciation rules to the appropriate context and voice. Keep phonetic spellings out of the manuscript.
- Run offline until live synthesis is authorized. Set both the request cap and character budget before enabling `--live`. Treat `--trial-ipa` as an experiment and report which takes were actually measured.
- Use the [acoustic and learning workflow](references/acoustic_workflow.md) for alignment, comparison and review. Report candidate defects, accepted edits, uncertainties and provider-reported cost from the output artifacts. Retain the prior audio whenever the comparison is inconclusive.

For an existing recording, batches, or mixed Bokmål–Ukrainian–English material, use `python -m audioqa` and read the [audio QA guide](references/audioqa.md). Start with a local scan and an explicitly multilingual ASR model. Add a manuscript or subtitles when available, and keep expected locale, recognized language and observed pronunciation distinct. Use the review dashboard for candidate clips, and require verified evidence before attributing a lecture error or accepting a regenerated take.

When invoked with an audio file, run the scan from the directory containing this skill. Prefer the repository's `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on Unix when available. Run `python -m audioqa scan --audio "<file>" --deep --language auto --local-only --out "<new run directory>"`. Add `--asr faster=models/asr/whisper-small` when that local multilingual model is installed, or point to another installed multilingual model. If none is available, use `--artifact-only` and report that only signal candidates were checked. Return the review page and identify which evidence requires listening.

For a labeled recording or an iterative audio QA study, follow the [recording research protocol](references/real_audio_qa.md). Freeze a separate label-free scan before opening the annotation input. Keep approximate points, detector candidates and adjudicated intervals distinct. Save review decisions by recording and source SHA-256. Check [prior experiments](knowledge/experiments.jsonl) and [known limits](knowledge/known_limitations.md) before changing detector thresholds or pronunciation rules.

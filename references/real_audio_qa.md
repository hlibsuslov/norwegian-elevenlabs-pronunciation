# Research on real ElevenLabs recordings

Use a separate process for the initial scan. Give it the audio and an authoritative manuscript when available. Keep manual labels, category counts and prior results out of that process. Freeze its report before using the labels. Include the exact configuration and local model revisions.

```powershell
python -m audioqa scan --audio input.mp3 --deep --artifact-only --language auto --local-only --no-plots --out runs/blind
python -m audioqa.freeze_baseline --report runs/blind/reports/defects.json --cache-manifest runs/blind/cache/manifest.json --out datasets/elevenlabs_audio_qa/metadata/RECORDING/baseline_predictions.json --command "audioqa scan, deep balanced DSP"
python -m audioqa.research --recording RECORDING --audio input.mp3 --baseline datasets/elevenlabs_audio_qa/metadata/RECORDING/baseline_predictions.json --proxy runs/blind/cache/decoded.wav --annotations manual-points.csv
```

Omit `--annotations` when scanning a new, unlabeled recording. The CSV accepts `id,timestamp,category,word`; use `HH:MM:SS` or `MM:SS`. A matching candidate is assigned once by minimum midpoint distance at maximum match count. Apply the principal tolerance of 2.5 seconds and report sensitivity at 1 and 5 seconds. Evaluate category agreement independently of timing agreement. Use `python -m audioqa.research_benchmark` with reviewed intervals and adjudicated unmatched candidates for final metrics.

## Independent evidence

- The baseline freezes the signal report. Run the same process with a local multilingual ASR model and manuscript when available. Use `python -m audioqa.research_asr --help` for independent timed reverse transcripts. Set an explicit model revision, use `--focus-annotations` only after the baseline, and retain all channel outputs. Pass repeated `--asr` inputs to `audioqa.research`. Put the whole-track multilingual transcript first.
- `audioqa.research` exports waveform, linear and mel spectrograms, Praat F0, energy, harmonicity, formants and local comparisons. For YAMNet supply a local ONNX export and AudioSet class map through `--yamnet`. The within-recording nearest-neighbor scan is for prioritization. It ranks novelty without verified clean controls. Pass `--relaxed` with a distinct higher-sensitivity scan to compare its review cost.
- Use `python -m audioqa.research_phones --help` to score competing phone sequences over the same audio crop. Store a short JSON of interval and word hypotheses with detector provenance, following the first recording's example. Use an independent wav2vec2 phoneme ONNX and its matching vocabulary. Compare several crops around uncertain word boundaries. Interpret CTC runs as frame activity. Use a matched alignment and native reference to confirm vowel length, stress and tone. Attach the file with `--phone-study`.
- Use `python -m audioqa.research_reviewer --help` with a local llama.cpp endpoint and an audio-capable model. Pass comparable cases and controls, and keep the reviewer blind to their labels. Record the response and evidence limitations. Attach it with `--reviewer-reports`.
- Use `python -m audioqa.research_experiments --help` for restricted local DSP auditions and within-source score comparisons. Inspect the original and alternate versions side by side. Attach the experiment record with `--repair-trials`. An audition remains provisional pending review and post-repair checks.

## Review and update

```powershell
python -m audioqa review --serve --report datasets/elevenlabs_audio_qa/reports/RECORDING.json --dataset datasets/elevenlabs_audio_qa --dataset-proxy runs/blind/cache/decoded.wav
```

The server binds to loopback. A saved decision carries recording ID, category, status, exact interval, review notes, timestamp and the source digest. A later decision adds a revision. For offline review open `review/RECORDING.html`, export decisions and import them using `python -m audioqa.research_review ingest --help`. Use the smallest interval supported by the audio and retain context. A confirmed natural release or legal language switch belongs in hard negatives when it triggered a detector. Keep a candidate unresolved if the exact word or cause needs native review.

Search reviewed examples with `python -m audioqa.research_review similar --help`. Compare compatible embedding versions on other source digests. For benchmark experiments separate all samples of the same original and stratify by voice and provider model where possible. Re-run the unchanged detectors and the new mode. Advance a threshold after comparing recall, review time and errors on an independent recording. Keep rule tuning, additions to the example library, prompt evaluation and classifier training distinct in `knowledge/experiments.jsonl`.

For Norwegian particle verbs, establish the intended meaning, word order, intervening material and contrast. Corpus data show particle accent, verb primary stress with particle secondary stress, and separate accents. Use local IPA and matched recordings to compare the full prosodic unit. [Tengesdal, Larsson and Lundquist](https://ojs.novus.no/index.php/MOM/article/download/2316/2288/3335)

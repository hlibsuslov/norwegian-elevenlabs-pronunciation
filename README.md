# Norwegian ElevenLabs Pronunciation Optimizer

Agent Skill and local CLI for Norwegian Bokmål synthesis. It preserves the manuscript, consults the NB Uttale dialect lexicon, prepares scoped ElevenLabs dictionaries, analyzes audio with Praat, and can repair individual sentences when acoustic evidence supports a splice.

The `audioqa` commands also scan long recordings for digital and speech candidates, generate an interactive review bundle, and route mixed Bokmål, Ukrainian and English manuscripts. See [installation and workflows for audio QA](references/audioqa.md).

## Install

Use Python 3.11–3.13. FFmpeg on PATH is needed for Whisper, automatic NoFA input preparation and audio formats that Praat cannot open. NoFA uses Montreal Forced Aligner 1.0 separately.

~~~sh
python -m venv .venv
# Unix
source .venv/bin/activate
# Windows PowerShell equivalent
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.setup_lexicon
~~~

The index setup downloads the official CC0 NB Uttale v1.0 archive, verifies its SHA-256 and streams the selected dialect into SQLite. Use `--dialect w_written` or any profile in [dialects.yaml](config/dialects.yaml). Use `--archive path/to/nb_uttale_leksika.zip` for an offline install with a predownloaded official archive. The index and account-scoped synthesis cache live under `.data/` by default. Set `NELP_DATA_DIR` to place working data elsewhere. The archive is about 158 MB and each indexed profile takes additional disk space.

For discovery across local Codex projects, put this directory at `~/.agents/skills/norwegian-elevenlabs-pronunciation` or link that location to the checkout where the models and dataset live. Invoke `$norwegian-elevenlabs-pronunciation` with the audio file. When the checkout is open as a project, [AGENTS.md](AGENTS.md) also routes requests to the skill. For Claude Code, place the folder under `~/.claude/skills/` or the project’s `.claude/skills/`. Run module commands from the skill directory with the selected Python environment.

For an existing recording, install `requirements-audioqa.txt`. A local DSP scan works without model weights. For transcription add an installed multilingual Whisper checkpoint via `--asr faster=LOCAL_MODEL_DIR`. The scan returns `reports/defects.json` and `review/index.html`.

## Workflows

~~~sh
# Analyze the corpus and propose source-attested IPA trials without using the API.
python -m scripts.workflow prevent --text examples/input.txt --trial-ipa --out runs/prevent

# Inspect a recording using an existing Norwegian NoFA TextGrid.
python -m scripts.workflow detect --text manuscript.txt --audio recording.wav --alignment recording.TextGrid --out runs/detect

# Print a dry-run generation plan. Live requests require an explicit budget.
python -m scripts.workflow full-auto --text manuscript.txt --out runs/plan

# Generate a short baseline with an authorized Norwegian voice and produce blind A/B pairs.
python -m scripts.workflow full-auto --text short.txt --model eleven_v4 --voice YOUR_VOICE_ID --trial-ipa --live --ab --max-requests 4 --max-characters 600 --out runs/live

# Store a correction after reviewing the evidence.
python -m scripts.workflow learn --record reviewed-correction.json --out runs/learn
~~~

Supply the API key through the environment, through an ignored `.env` based on [.env.example](.env.example), or through `--env-file` pointing to an existing local file. The optional file accepts the ElevenLabs variable names only for purposes of this workflow. Use a voice authorized for your account and matching the target dialect. The engine receives exact sentence text and dictionary locators. It does not add spellings, stress tricks or SSML to the manuscript.

The output directory contains `text_analysis.json`, `synthesis_plan.json` and `quality_report.json`. Audio runs additionally store measurements and available timing. If acoustic evidence is inadequate, the final file keeps the baseline or the last accepted edit. The live budget counts locally reserved characters, including context and an allowance for rule expansion, and limits synthesis requests. It does not express an account credit cap. The report records the provider's `character-cost` header when available. Billing can depend on the model, voice and plan. A cache hit uses no new generation request.

For automatic repair, review the [NoFA, Whisper and reference setup](references/acoustic_workflow.md). See the [compatibility matrix](references/elevenlabs_compatibility.md) for the model and transport limitations, including Multilingual v2.

## Verify

~~~sh
python -m pip install pytest
python -m pytest -q
python -m scripts.setup_lexicon
python -m pytest -q
~~~

Tests without the official index use an explicitly synthetic fixture for mechanics only. Index-gated tests check corpus coverage and documented distinctions against released records. Acoustic tests use analytically generated WAVs and local TextGrid fixtures. Mock API tests exercise response shape, dictionary upload, sockets, cache and budget. See [validation status](references/validation.md) for executed checks and remaining evidence.

NB Uttale v1.0 and NoFA are published by Språkbanken under CC0. The spaCy Bokmål model is MIT licensed and pulled from its upstream wheel. Other linguistic references guide review and are linked in [phonology](references/norwegian_phonology.md).

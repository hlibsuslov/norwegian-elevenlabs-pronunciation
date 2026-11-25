# Acoustic analysis, repair and review

## Align and transcribe

- For word and phone boundaries use NoFA v1.0, trained for Norwegian Bokmål. Run `python -m scripts.setup_lexicon --nofa-only` and provide the MFA **1.0** `mfa_align` executable or a wrapper with the same argument interface. The four positional arguments are the input folder (mono WAV and LAB), NoFA dictionary, model ZIP and output folder. The adapter also provides `-t`, `-j 1` and `-q`. Alternatively pass an existing Praat TextGrid with `words` and optional `phones` interval tiers. [NoFA](https://www.nb.no/sprakbanken/ressurskatalog/en/oai-nb-no-sbr-59/) · [MFA 1.0.1 distribution](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/tag/v1.0.1)
- NoFA uses its own lexicon and may miss obscure words, normalized numerals and variants. The alignment can assign expected phone labels to different sounds. Inspect the coverage and use it for segmentation and measurement. Supplying a compatible TextGrid supports DETECT on any platform with Python and Praat bindings.
- Install `requirements-asr.txt` and download a **multilingual** CTranslate2 Whisper model. For example, run `python -c "from faster_whisper.utils import download_model; download_model('base', output_dir='whisper-base')"` once, then pass `--whisper-model whisper-base`. Word recognition is independent of the forced aligner. Review ASR alternatives for omissions, repeats or substitutions and pay attention to written versus spoken normalization. [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)
- Praat measures voiced F0, contours, intensity, phone and word duration, and vowel F1/F2. The syllable measurement estimates boundaries between adjacent nuclei. Formant differences are an articulation cue. Voice identity, emotional delivery and timbre changes require listening alongside the measured cues.

## Supply an independently reviewed reference

Record or license a matched native rendition with the same manuscript, dialect, speaking style and similar vocal range. Review its segment boundaries. Provide `--reference-audio` and either `--reference-alignment` or `--nofa-command`. Add `--reference-review review.json` with four fields.

| Field | Value |
| --- | --- |
| `status` | `reviewed_matched_reference` |
| `dialect` | An exact profile ID, for example `e_written` |
| `native_reviewer_evidence` | Reference to the review of the audio, dialect, delivery and context |
| `source_sha256` | The value from the PREVENT report for the exact manuscript |
| `audio_sha256` | Standard SHA-256 digest of the reviewed reference audio |

The tool compares relative pitch contours, intensity and duration, along with vowel or consonant timing and formant distance where compatible phone boundaries exist. It requires multiple independent feature groups and labels deviations as probabilistic. Thresholds are engineering defaults and need calibration for the intended voice, context and dialect. Whisper flags words, while human review can identify the intended focus or verify tone, articulation and timbre.

## Repair and acceptance

Use the same voice ID and settings as the original. Start with a dry repair. Add `--live`, a character and request budget, and `--trial-ipa` if experimental source-attested candidates are appropriate. Supply compatible source audio at 24 kHz mono for automatic splicing, NoFA and local Whisper. Whole-manuscript QA runs on each candidate, while only the affected sentence is sent to ElevenLabs. The splicer requires low-energy boundaries before applying short fades and checks duration and clipping. The final choice requires improvement at the target, retained reference and alignment coverage, and no significant acoustic, pause or transcript regression.

For blinded review, run FULL AUTO with `--ab`. Share only the files in `ab/` and the manuscript with reviewers. Keep `ab_key.json` closed until their judgments are recorded. Ask reviewers to rate lexical pronunciation, stress, tonal naturalness, rhythm and continuity, and permit a tie. `ab_metrics.json` contains descriptive signal and ASR data for the two takes. Compare repeat takes and matched references before generalizing.

## Learn

See [review record schema](correction_record.md). The SQLite store accepts corrections with source evidence, matched native and acoustic confirmation, and explicit dialect, model and optional voice. Candidate rules stay with the run artifacts until independently reviewed.

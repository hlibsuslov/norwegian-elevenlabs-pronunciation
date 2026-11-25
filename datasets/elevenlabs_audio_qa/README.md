# ElevenLabs audio QA dataset

The first item is `qa-001`. Its source copy is `raw_audio/qa-001.mp3`. Generated media, reverse transcripts and personal review state are kept locally. The manually supplied rough points and second-pass phone hypotheses can be versioned. Each case ID includes a source-specific digest.

Open `review/qa-001.html` for the clips and identity-linked cards. The reports retain the raw timestamp, candidate interval, waveform, spectra, F0, energy, acoustic contrasts, independent model evidence and any repair auditions. The separate `metadata/qa-001-phones.json` includes word-level analysis and particle-verb occurrences.

| Location | Contents |
| --- | --- |
| `raw_audio/` | Byte-identical original |
| `annotations/` | Input points, phone hypotheses and dated decision histories |
| `confirmed_artifacts/`, `confirmed_pronunciation_errors/`, `language_interference/` | Human-reviewed positive cases |
| `clean_controls/`, `hard_negatives/`, `ambiguous_cases/` | Human-reviewed controls and unresolved cases |
| `review/clips/` | Micro, local and extended context, figures, diagnostic cards and repair auditions |
| `acoustic_features/`, `embeddings/` | NumPy arrays on the decoded sample clock and local embedding vectors |
| `repaired_examples/` | Source excerpt and DSP trial, with approval in the experiment record |
| `metadata/` | Frozen baseline and checksum, experiment files, source metadata, manifests and metrics |

The first review queue includes eight unadjudicated controls. Confirm or reject each candidate in the local review interface. The server and offline import commands are in the [research protocol](../../references/real_audio_qa.md).

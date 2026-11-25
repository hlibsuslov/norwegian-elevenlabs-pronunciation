# Reviewed correction record

Pass one UTF-8 JSON object to `python -m scripts.workflow learn --record FILE --out DIR`.

| Fields | Value |
| --- | --- |
| `word`, `lemma` | Original grapheme and lemma |
| `scope`, `context` | `sentence` with the exact original sentence, or `global` with null context after checking all intended senses |
| `dialect`, `model`, `voice_id` | Tested dialect, compatible model ID and tested voice. The voice may be null for confirmed voice-independent phoneme rules |
| `ipa`, `toneme` | Source-attested standard segmental/stress IPA and separate lexical `1`, `2` or null |
| `primary_stress`, `secondary_stress` | Lists of zero-based phone indices in the documented pronunciation |
| `source`, `error_type` | Reference to the transcription, and the observed synthesis defect |
| `method`, `replacement` | `phoneme` with null replacement, or `alias` with a phonetic replacement tested with that voice |
| `confidence` | A documented reviewer confidence from 0 to 1 |
| `acoustic_review`, `native_review` | Each an object with `status` set to `confirmed` and a nonempty `evidence` reference |

Identities are scoped by word, context, dialect, model and voice. Re-learning that identity updates the stored record. JSON and CSV retain all scopes. PLS export includes only global rules for the selected dialect, model and voice, since PLS alone cannot express context. PREVENT drafts separate PLS files for eligible manuscript segments.

Mixed-language entries can additionally supply `word_language`, `context_language`, `previous_language`, `next_language`, `alternate_pronunciations` and `interference_type`. Supported locales are `nb-NO`, `uk-UA`, `en-US` and `en-GB`. Use the locale as the dialect value for Ukrainian and English, and use sentence scope. Mixed generation applies a rule only when its word, context, voice and neighbor scopes agree. A grapheme with conflicting occurrences stays unchanged.

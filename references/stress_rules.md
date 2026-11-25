# Word and sentence stress

Read hovedtrykk and bitrykk from NoFAbet nucleus codes and inspect the exact wordform. Codes 1 and 2 identify primary stress and the lexical accent category. Code 3 identifies secondary stress. Nuclei can be vowels or syllabic consonants. Inflection, loans and compound morphology can change the expected stress pattern. Check the record for the inflected form.

For homographs, start with the sentence, the spaCy lemma, POS and morphology and all matching NB Uttale IDs. POS can suggest a candidate, while same-POS senses can require lexical disambiguation. A reviewed exception can map an exact sentence and dialect to a source wordform ID in `dictionaries/exceptions.json`. Add `word`, `dialect`, `sentence`, `wordform_id` and `source`. Use a sentence-scoped correction when meaning determines pronunciation. Review repeats of the same homograph within one sentence before applying a shared grapheme rule.

Judge setningstrykk with intended contrast, given versus new information, phrase boundaries and utterance type in the same context. A longer nucleus, a pitch movement or louder speech can each arise from other causes. Use matched audio, syllable and word duration, relative intensity and normalized contour together. Review dialectal variants as valid alternatives.

[NB Uttale transcription documentation](https://www.nb.no/sbfil/uttaleleksikon/NB_Uttale_v_1_0_documentation.pdf)

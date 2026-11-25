# ElevenLabs compatibility

Checked 25 November 2026. The current run also checked Norwegian language availability against the authenticated models endpoint. The API guide gives a stricter statement about non-English phoneme rules than the Agents documentation for v3. This implementation follows the stricter API guidance.

| Model | Norwegian listed | PLS aliases | PLS phoneme rules | Inline IPA documented | Norwegian IPA enabled | Skill transport |
| --- | --- | --- | --- | --- | --- | --- |
| Eleven v4 `eleven_v4` | Yes | Yes | Yes | Yes | Yes, lexical tone reviewed separately | Dialogue HTTP with timestamps |
| Eleven v4 Turbo `eleven_v4_turbo` | Yes | Yes | Yes | Yes | Yes, lexical tone reviewed separately | Dialogue WebSocket |
| Eleven v3 `eleven_v3` | Yes | Yes | Yes | Yes | Aliases only in this skill | TTS HTTP with timestamps |
| Multilingual v2 `eleven_multilingual_v2` | Absent | Yes | No | No | No | Analysis only for Norwegian |
| Eleven Flash v2.5 `eleven_flash_v2_5` | Yes | Yes | No | No | No | TTS HTTP with timestamps |
| Eleven Flash v2 `eleven_flash_v2` | English only | Yes | Yes | No | No | Analysis only for Norwegian |

Use PLS dictionary locators over inline IPA for provenance and to preserve graphemes. Use only model- and voice-reviewed aliases. When there is no reviewed alias, use the original spelling and flag the occurrence. The skill sends one dictionary version per applicable request. The provider's matching is case-sensitive. Context-dependent rules stay on their sentence, and repeated ambiguous graphemes or competing rules can require human review.

Keep selected voice, text, settings and seed stable between HTTP A/B takes. The provider describes seeded output as best effort. WebSocket requests use the documented model, language and alignment settings. That transport lacks the HTTP seed and previous/next text controls used by this skill. Human review is especially valuable when using it for repairs. The HTTP Dialogue per-request chunk limit is the endpoint’s conservative 2,000-character reliability recommendation.

- [Model IDs, languages and interfaces](https://elevenlabs.io/docs/overview/models)
- [API pronunciation dictionary guidance](https://elevenlabs.io/docs/eleven-api/guides/how-to/text-to-speech/pronunciation-dictionaries)
- [Vendor pronunciation compatibility table](https://elevenlabs.io/blog/pronunciation-dictionary-guide)
- [Agents pronunciation dictionary comparison](https://elevenlabs.io/docs/eleven-agents/customization/voice/pronunciation-dictionary)
- [PLS file creation](https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/create-from-file)
- [Speech with timing](https://elevenlabs.io/docs/api-reference/text-to-speech/convert-with-timestamps) · [Dialogue with timing](https://elevenlabs.io/docs/api-reference/text-to-dialogue/convert-with-timestamps) · [Dialogue WebSocket](https://elevenlabs.io/docs/api-reference/text-to-dialogue/ttd-websocket)
- [Provider cost headers](https://elevenlabs.io/docs/api-reference/introduction) · [Character accounting](https://elevenlabs.io/docs/help-center/account/general/how-is-character-length-calculated)

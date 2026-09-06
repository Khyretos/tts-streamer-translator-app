# Module split + test suite (Phase 7)

## New layout

`voice_translator.py` was ~4,500 lines after phases 1–6 (it started at
~3,950 and grew with every fix). Rather than a mechanical "put everything
in its own file" split — which for the most tightly-coupled parts (the
~1,700-line `create_ui()`, the FastAPI routes, `VoiceTranslatorApp` itself)
would mean threading 100+ Gradio component variables across module
boundaries with no way for me to actually run the app here to verify it
still boots — I split out the pieces that are genuinely self-contained,
have clear inputs/outputs, and I could verify in isolation:

```
voice_translator.py    Entry point: VoiceTranslatorApp, create_ui(), FastAPI
                        routes, uvicorn.run(). Imports everything below.
session.py              Slug resolution: SessionSlugMiddleware, get_slug,
                        sanitize_slug, RESERVED_PATH_SEGMENTS.
settings_store.py       Per-slug settings persistence: load/persist,
                        PERSISTABLE_KEYS, the vad_threshold migration.
vad.py                  FastVAD — voice activity detection + preprocessing.
subtitles.py            SubtitleManager — subtitle buffering/pacing.
recognizers.py          ArgosTranslator, WhisperRecognizer,
                        MoonshineRecognizer, dots_or_stars,
                        is_whisper_hallucination.
translators.py          (already separate) TranslationService — AI /
                        LibreTranslate / Argos dispatch.
logger.py               (already separate) Logger.
```

`VoiceTranslatorApp` and `create_ui()` stay in `voice_translator.py`. They're
the orchestration layer — deeply intertwined with Gradio's request/event
model, `sounddevice`, and `vosk`, none of which I can actually instantiate
or run in this environment (no audio hardware, no Gradio server loop here).
Splitting them further would mean shipping code I could only verify with
`py_compile` and static analysis, not by actually exercising it — for the
orchestration layer specifically, that's a real risk I'd rather not take.
Everything above the line, I *did* verify by actually running it (see
below) — that's the deciding factor in what got extracted.

**If you build/run this and it comes back clean, `create_ui()` and
`VoiceTranslatorApp` are reasonable next candidates** to split further
(e.g. `ui.py`, `app_core.py`) once there's a running instance to check each
step against — happy to help with that next.

## ⚠️ If you deploy via Docker

`Dockerfile`'s `COPY` line now lists all seven files
(`voice_translator.py translators.py logger.py session.py settings_store.py
vad.py subtitles.py recognizers.py`) — it used to list only three. If
you've made local Dockerfile edits since receiving this, make sure the new
modules are included or the container will fail on import at startup.

## How each module was verified

`vad.py`, `subtitles.py`, `recognizers.py`, `session.py`, and
`settings_store.py` have no dependency on Gradio, sounddevice, or vosk being
actually running — they're plain Python (plus numpy/requests/starlette) — so
I could `import` and exercise them directly in a real Python process here,
not just read the code. Concretely:
- Ran `python3 -m py_compile` on every module.
- Ran `pyflakes` across the whole set — flags **undefined names**, not just
  style, so this is real evidence nothing references a symbol that no
  longer exists after the split (it found zero undefined-name issues).
- Wrote and ran the pytest suite below against the actual extracted code —
  87 tests, all passing.

I did not, and could not, boot the full Gradio app end-to-end here (no
display, no audio device, no installed `vosk`/`gradio`/`sounddevice` in this
sandbox) — so please do a normal smoke test (`docker compose up` and open
the UI) before relying on this in production, same as any change you'd
review from a PR.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

87 tests across 6 files, covering the concrete bugs fixed in earlier
phases as regression tests — not just "does it run":
- `test_vad.py` (10) — silence produces no segments, short bursts are
  dropped as noise, threshold/end-silence hot-reload.
- `test_subtitles.py` (13) — the three subtitle-timing bugs fixed in
  phase 5 (cross-language misalignment, hold-time-from-wrong-text,
  interim corrupting buffered pacing), each as an explicit regression test.
- `test_recognizers.py` (26) — hallucination filtering, and the
  verbose_json confidence-filtering fix from phase 4 (segments with high
  no_speech_prob / low avg_logprob / high compression_ratio are dropped
  client-side).
- `test_translators.py` (15) — the editable AI prompt/endpoint/request/
  response-path feature from phase 6, including every fallback-to-default
  path for malformed custom values.
- `test_session.py` (11) — slug resolution priority (cookie > query param >
  session_hash fallback), sanitization.
- `test_settings_store.py` (12) — per-slug persistence, the legacy
  settings.json seeding behavior, the vad_threshold migration, debouncing.

None of these need vosk models, audio hardware, or a running Gradio server
— they run in well under 5 seconds total, so there's no excuse not to run
them before a deploy once you're making further changes.

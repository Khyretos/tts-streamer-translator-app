# Editable prompts, endpoints, and request/response shapes

There are two independent places in this app that call an external
API-compatible service and previously assumed it was shaped exactly like
OpenAI's: **AI translation** (a chat-completion call) and **Whisper
recognition/translation** (an audio-transcription call). Both are covered
below — each with its own settings, but the same underlying idea: leave
everything blank and you get sensible OpenAI-compatible defaults; override
whatever you need for a server that isn't shaped that way.

## AI translation: two real bugs found and fixed along the way

1. **The default AI Host was broken out of the box.** `DEFAULT_SETTINGS["ai_host"]`
   was `""` (empty). With no override configured, the endpoint-guessing logic
   built a URL from that empty string, producing a relative path like
   `/v3/chat/completions` with no scheme or host — `requests.post()` on that
   fails immediately. AI translation could not work until a user manually
   filled in the host. Fixed: default is now `http://localhost:11434`
   (Ollama's default), and the fallback in `translators.py` additionally
   treats a stored empty string as "unset" (`self.settings.get("ai_host") or
   "http://localhost:11434"`), so existing configs with `ai_host: ""` saved
   self-heal automatically without any migration needed.
2. **The endpoint-guessing logic appended `/v3/chat/completions`.** That's
   not a real path on any standard OpenAI-compatible server (Ollama, vLLM,
   LM Studio, OpenAI itself, etc. all use `/v1/chat/completions`). Any setup
   relying on the auto-guess (rather than a host string that already
   happened to include the full working path) was silently 404ing. Fixed to
   `/v1`.


## Editable prompt

`ai_translation_prompt_template` (Settings → Translation → AI, "Advanced").
Supports three placeholders: `{source_lang}`, `{target_lang}`, `{text}`.
Leave blank to use the built-in default (same wording as before). An invalid
template (bad/unknown placeholder) logs a warning and falls back to the
default rather than breaking translation.

## Editable endpoint

`ai_endpoint_url` — if set, used **exactly as given** as the full POST URL,
bypassing the `/v1/chat/completions` auto-derivation from "API Host"
entirely. Use this for a provider whose chat-completion path isn't
`/v1/chat/completions` at all.

## Editable request/response shape

For a genuinely non-OpenAI-compatible API (different request body shape,
different response shape), two more fields:

- `ai_request_body_template` — the raw JSON body, as a template. Two
  tokens are substituted: `__MODEL__` and `__PROMPT__` (as JSON-escaped
  string *content*, no surrounding quotes — your template supplies those).
  Default:
  ```json
  {
    "model": "__MODEL__",
    "messages": [{"role": "user", "content": "__PROMPT__"}],
    "temperature": 0.3,
    "max_tokens": 500
  }
  ```
  Point this at a completely different shape if needed, e.g. an
  Anthropic-style body:
  ```json
  {
    "model": "__MODEL__",
    "max_tokens": 500,
    "messages": [{"role": "user", "content": "__PROMPT__"}]
  }
  ```
  If the substituted result isn't valid JSON, a warning is logged and the
  default shape is used instead — translation doesn't just break silently.

- `ai_response_text_path` — a dot-path into the JSON response to the
  translated text, e.g. the default `choices.0.message.content`
  (`response["choices"][0]["message"]["content"]`). For an Anthropic-style
  response that'd be `content.0.text`. Numeric segments index into lists;
  anything else is treated as a dict key. If the path doesn't resolve, it
  automatically retries the default path before giving up.

All four fields persist per-session like every other setting, and there's a
"Reset these 4 fields to default" button in the Advanced accordion.

This is intentionally an advanced/developer surface — the placeholder text
in each field shows the exact default value being used when left blank, so
you can see the working baseline before diverging from it.

---

# Whisper recognition/translation: editable endpoint & response shape

This was originally requested alongside the AI-translation editability
above, but only the AI-translation side actually got built — Whisper's own
transcription/translation call (used by the "whisper" recognition engine
and the "Whisper Translate" translation mode) was still hardcoded to
`{host}/audio/transcriptions` / `{host}/audio/translations` with no
override, and to OpenAI's `verbose_json` segment response shape. Fixed:

- **`whisper_endpoint_url`** (recognition) / **`whisper_translate_endpoint_url`**
  (translate mode) — if set, used exactly as-is as the full POST URL,
  instead of deriving `{API Host}/audio/transcriptions` or
  `{API Host}/audio/translations`.
- **`whisper_response_text_path`** / **`whisper_translate_response_text_path`**
  — a dot-path (same syntax as `ai_response_text_path` above, e.g.
  `result.text` or `alternatives.0.transcript`) into a non-standard JSON
  response. When set, this **bypasses the no_speech/logprob/
  compression_ratio confidence filtering** from RECOGNITION_QUALITY.md
  entirely, since a custom response shape can't be assumed to carry
  OpenAI's `segments[]` confidence data at all. If the custom path doesn't
  resolve, it logs a warning and falls back to the standard verbose_json
  parsing.

Both live in a "🛠️ Advanced: custom endpoint & response shape (developer)"
accordion inside the existing Advanced Whisper Parameters sections, one for
recognition and one for Whisper Translate — independent overrides, since
you might point recognition at one server and translation at another.

# Vosk model sharing across sessions

Previously every session's `start_recognition()` called `Model(model_path)`
itself. Two sessions using the *same* Vosk model (e.g. two people both using
the English model, in different named sessions) loaded that model into RAM
twice — for a large model that's a real amount of memory wasted for no
reason, since it's the identical model file both times.

Fixed with a small refcounted cache (`_acquire_vosk_model` /
`_release_vosk_model` in `voice_translator.py`): the first session to use a
given model path loads it; every subsequent session using that *same* path
gets a reference to the already-loaded `Model` object, and Vosk's own API is
explicitly designed to support many independent `KaldiRecognizer` instances
sharing one `Model` safely across threads — this isn't a workaround, it's
the documented intended usage. The underlying model is only actually freed
once every session using it has stopped.

Two different models (e.g. English + Spanish) still each load their own
copy, as they should — only the *same* model path loaded by *multiple
sessions simultaneously* is deduplicated.

**Whisper** isn't affected — it's a remote API call, no local model to
duplicate. **Moonshine** isn't included in this cache: its `Transcriber` is
inherently a stateful per-stream object (live VAD + line-buffering state
tied to one audio stream's timing), so unlike Vosk's Model/Recognizer split,
there isn't a clean "big shared weights, small per-session state" boundary
to exploit the same way. Moonshine's models are also small ONNX files
(tens of MB, not the ~1GB+ some Vosk models reach), so the RAM duplication
concern is much smaller there. If two sessions running Moonshine
simultaneously turns out to matter for your setup, worth revisiting
specifically — but it wasn't part of what you reported.

# Recognition quality (Phase 4)

## VAD / noise detection — assessment, not a rewrite

I read through `FastVAD` in detail looking for a "better way to do it."
Honest assessment: it's already well-built for what it's doing — vectorized
spectral subtraction (single FFT per block, not per-frame), a transient/
click detector, an RMS energy gate, and an optional webrtcvad confirmation
pass, all batched as numpy array ops rather than Python loops. There isn't
an obvious architecture-level win available without changing its actual
behavior (e.g. swapping in a neural VAD like Silero, which is a much bigger
dependency + latency trade-off — say if you want to explore that
specifically). So I left it as-is and focused on tuning, below, which is
where the actual "thank you"/"subscribe" symptom was coming from.

## The Whisper hallucination fix

Two real bugs, not "Whisper is just like that":

**1. The confidence thresholds were sent but never checked.**
`whisper_no_speech_threshold`, `whisper_logprob_threshold`, and
`whisper_compression_ratio_threshold` were already exposed as settings and
sent as request params to your Whisper server — but the request asked for
`response_format: "json"`, which only returns final text, not the
per-segment confidence data those thresholds apply to. So nothing ever
actually verified the server enforced them — and plenty of minimal
OpenAI-compatible Whisper servers accept those params without fully
honoring them.

Fixed: now requests `response_format: "verbose_json"` and re-checks every
returned segment's `no_speech_prob` / `avg_logprob` / `compression_ratio`
against your thresholds client-side, dropping any segment that fails —
regardless of what the server did with them. If your Whisper host doesn't
support `verbose_json` and just returns plain JSON anyway, this falls back
to the old plain-text behavior automatically (nothing to configure).

**2. The default end-of-speech pause (80ms) was shorter than a normal breath.**
The code's own comment already said "raise this if phrases are cut off
mid-sentence... recommended range 200–600ms" — but the actual default was
80ms. That's short enough to slice a sentence into fragments on a normal
pause, and short/ambiguous clips are exactly what Whisper hallucinates
fillers on. Raised the default to 300ms. This only affects Whisper
(Vosk uses its own internal endpointer; Moonshine does its own VAD) — Vosk
and Moonshine behavior is unchanged.

If you have an existing `settings/<slug>.json` with `vad_end_silence_ms: 80`
already saved, it's left alone — I didn't want to silently override a value
you may have deliberately tuned. The new 300ms default only applies to
brand-new sessions. Bump it in the UI slider if you want the same change on
an existing session.

The existing exact-match hallucination denylist (`_WHISPER_HALLUCINATIONS`)
is left as-is — it's a good conservative safety net for the odd case that
slips past the confidence filtering above. I didn't switch it to
substring/fuzzy matching, since that risks dropping real short utterances
that happen to contain phrases like "thank you" or "subscribe" — a false
negative here is much less annoying than a false positive that eats real
speech.

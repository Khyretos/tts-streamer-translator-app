"""
Speech recognizer backends and translation-quality helpers, extracted from
voice_translator.py during the module-split refactor — see REFACTOR.md.

- ArgosTranslator: offline translation via the argostranslate package
- WhisperRecognizer: OpenAI-compatible Whisper API client
- MoonshineRecognizer: wraps moonshine_voice.Transcriber
- dots_or_stars / is_whisper_hallucination: Whisper output filtering
"""

import gc
import io
import os
import re
import wave
from pathlib import Path

import numpy as np
import requests

os.environ.setdefault("ARGOS_PACKAGES_DIR", os.getcwd() + "/argos_models")

try:
    import argostranslate.package
    import argostranslate.translate

    ARGOS_AVAILABLE = True
except ImportError:
    ARGOS_AVAILABLE = False
    print("[WARNING] argostranslate not installed. Offline translation disabled.")

def dots_or_stars(input_str: str, second_arg=None) -> bool:
    if input_str == "." or re.match(r"<\|.*|>", input_str):
        return True
    return False


# Known Whisper hallucinations produced when fed noise/silence/desk taps.
# Whisper was trained on subtitles which have polite phrases at segment ends.
# Any short segment that matches these exactly (case-insensitive, stripped) is dropped.
_WHISPER_HALLUCINATIONS: set[str] = {
    "thank you",
    "thank you.",
    "thanks for watching",
    "thanks for watching.",
    "thanks for watching!",
    "thank you for watching",
    "thank you for watching.",
    "you",
    "you.",
    "bye",
    "bye.",
    "bye!",
    "goodbye",
    "goodbye.",
    "like and subscribe",
    "like and subscribe.",
    "subscribe",
    "music",
    "music.",
    "[music]",
    "(music)",
    "[applause]",
    "(applause)",
    "applause",
    "applause.",
    "[laughter]",
    "(laughter)",
    "laughter",
    "hmm",
    "hmm.",
    "hm",
    "hm.",
    "uh",
    "uh.",
    "um",
    "um.",
    "ah",
    "ah.",
    "oh",
    "oh.",
    "okay",
    "okay.",
    "ok",
    "ok.",
    ".",
    "..",
    "...",
    "…",
    "[silence]",
    "(silence)",
    "[noise]",
    "(noise)",
    "[inaudible]",
    "(inaudible)",
    "subtitles by",
    "subtitles by the amara.org community",
    "www.mooji.org",
    "www.facebook.com",
    "the end",
    "the end.",
    "end.",
}

# Used only for the short-utterance prefix check in is_whisper_hallucination
# (see below). Split into two tiers: generic openers ("thank you", "bye")
# are common in real short speech too, so they only count for very short
# utterances; phrases that are essentially never said in normal
# conversation ("thanks for watching", "like and subscribe") are safe to
# match over a longer word count.
_HALLUCINATION_PREFIXES_GENERIC: tuple[str, ...] = (
    "thank you",
    "goodbye",
    "bye",
)
_HALLUCINATION_PREFIXES_DISTINCTIVE: tuple[str, ...] = (
    "thanks for watching",
    "thank you for watching",
    "thank you so much for watching",
    "like and subscribe",
    "please subscribe",
    "subscribe",
    "see you next time",
    "see you in the next video",
)


def is_whisper_hallucination(text: str) -> bool:
    """Return True if text is a known Whisper hallucination / filler output."""
    normalized = text.strip().lower()
    if normalized in _WHISPER_HALLUCINATIONS:
        return True
    # Exact-match alone misses real-world variation ("Thank you for
    # watching, don't forget to subscribe!" vs. the exact denylist entries).
    # For SHORT utterances specifically, matching just the opening words is
    # safe: a genuine sentence that happens to start the same way but keeps
    # going ("Thank you for the detailed explanation...") is long enough to
    # skip this check entirely. Hallucinated fillers are almost always
    # short — that's what makes this a reasonable, conservative widening
    # rather than a blanket fuzzy match.
    word_count = len(normalized.split())
    if word_count <= 4:
        for prefix in _HALLUCINATION_PREFIXES_GENERIC:
            if normalized.startswith(prefix):
                return True
    if word_count <= 10:
        for prefix in _HALLUCINATION_PREFIXES_DISTINCTIVE:
            if normalized.startswith(prefix):
                return True
    return False

class ArgosTranslator:
    """Offline translation using Argos Translate."""

    def __init__(self, logger=None):
        self.logger = logger
        self.models_dir = Path("argos_models")
        self.models_dir.mkdir(exist_ok=True)
        if ARGOS_AVAILABLE:
            argostranslate.package.settings.package_data_dir = str(self.models_dir)
            try:
                argostranslate.package.update_package_index()
            except Exception as e:
                if self.logger:
                    self.logger.log(
                        f"Argos package index update failed: {e}", level="warning"
                    )

    def translate(self, text, source_lang, target_lang):
        if not ARGOS_AVAILABLE:
            return f"[Argos not available: {text}]"
        try:
            installed = argostranslate.translate.get_installed_languages()
            source = target = None
            for lang in installed:
                if lang.code == source_lang or lang.code.startswith(source_lang):
                    source = lang
                if lang.code == target_lang or lang.code.startswith(target_lang):
                    target = lang
            if not source or not target:
                return f"[Model not installed: {text}]"
            translation = source.get_translation(target)
            if not translation:
                return f"[No translation available: {text}]"
            result = translation.translate(text)
            if self.logger:
                self.logger.log(f"Argos: {text} -> {result}", level="info")
            return result
        except Exception as e:
            if self.logger:
                self.logger.log(f"Argos translation error: {e}", level="error")
            return text

    def get_available_languages(self):
        if not ARGOS_AVAILABLE:
            return []
        try:
            installed = argostranslate.translate.get_installed_languages()
            return [
                (src.code, tgt.code, f"{src.name} -> {tgt.name}")
                for src in installed
                for tgt in src.translations_from
            ]
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error getting Argos languages: {e}", level="error")
            return []


def _resolve_response_path(data, path: str):
    """
    Walk a dot-separated path through nested dicts/lists, e.g.
    "result.text" -> data["result"]["text"]. Numeric segments index into
    lists. Raises on a bad path — callers catch and fall back.
    """
    node = data
    for segment in path.split("."):
        if isinstance(node, list):
            node = node[int(segment)]
        else:
            node = node[segment]
    return node


# ── Whisper Recognizer ────────────────────────────────────────────────────────
class WhisperRecognizer:
    """Whisper API-based speech recognizer with configurable parameters."""

    def __init__(
        self,
        host,
        api_key=None,
        model="whisper-large-v3",
        logger=None,
        temperature=0.0,
        best_of=5,
        beam_size=5,
        patience=1.0,
        length_penalty=1.0,
        suppress_tokens="-1",
        initial_prompt=None,
        condition_on_previous_text=True,
        temperature_increment_on_fallback=0.2,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        endpoint_url=None,
        response_text_path=None,
    ):
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.logger = logger
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self.temperature = temperature
        self.best_of = best_of
        self.beam_size = beam_size
        self.patience = patience
        self.length_penalty = length_penalty
        self.suppress_tokens = suppress_tokens
        self.initial_prompt = initial_prompt
        self.condition_on_previous_text = condition_on_previous_text
        self.temperature_increment_on_fallback = temperature_increment_on_fallback
        self.no_speech_threshold = no_speech_threshold
        self.logprob_threshold = logprob_threshold
        self.compression_ratio_threshold = compression_ratio_threshold
        # Advanced/developer overrides for a non-OpenAI-compatible server —
        # both optional, blank/None means "use the OpenAI-compatible
        # defaults" (auto-derived endpoint path, verbose_json segment
        # parsing). See AI_TRANSLATION.md for the equivalent AI-translation
        # feature this mirrors.
        self.endpoint_url = (endpoint_url or "").strip() or None
        self.response_text_path = (response_text_path or "").strip() or None

    def _build_data(self, language=None, task="transcribe"):
        data = {
            "model": self.model,
            # verbose_json is what makes the no_speech/logprob/compression_ratio
            # thresholds below actually mean something on the *client* side (see
            # _extract_text) — with plain "json" we only get the final text, so
            # nothing here ever double-checks whether the server actually
            # enforced them. Falls back to plain text automatically if the
            # server ignores this and returns "json" anyway.
            "response_format": "verbose_json",
            "temperature": self.temperature,
            "best_of": self.best_of,
            "beam_size": self.beam_size,
            "patience": self.patience,
            "length_penalty": self.length_penalty,
            "suppress_tokens": self.suppress_tokens,
            "condition_on_previous_text": self.condition_on_previous_text,
            "temperature_increment_on_fallback": self.temperature_increment_on_fallback,
            "no_speech_threshold": self.no_speech_threshold,
            "logprob_threshold": self.logprob_threshold,
            "compression_ratio_threshold": self.compression_ratio_threshold,
        }
        if language:
            data["language"] = language
        if self.initial_prompt:
            data["prompt"] = self.initial_prompt
        return data

    def _extract_text(self, result) -> str:
        """
        Pull text out of a Whisper API response, re-checking every segment
        against no_speech_threshold / logprob_threshold / compression_ratio_
        threshold *client-side*.

        Why: those three settings were already being sent as request params,
        but with response_format="json" the client only ever saw the final
        text — never the per-segment confidence data needed to verify the
        server actually enforced them. Many minimal OpenAI-compatible Whisper
        servers accept those params but don't fully honor them, which is a
        very plausible source of hallucinated fillers ("thank you",
        "subscribe") slipping through even with sensible thresholds set: the
        settings were being sent, nothing was checking the response against
        them. This enforces them regardless of what the server actually did.
        """
        if not isinstance(result, dict):
            return str(result)
        if self.response_text_path:
            # Custom response shape — can't assume verbose_json segments
            # exist at all, so skip the confidence-filtering below entirely
            # and just extract the text at the given path.
            try:
                value = _resolve_response_path(result, self.response_text_path)
                return str(value).strip()
            except (KeyError, IndexError, TypeError, ValueError) as e:
                if self.logger:
                    self.logger.log(
                        f"Custom Whisper response path '{self.response_text_path}' "
                        f"didn't resolve ({e}) — falling back to the standard "
                        f"verbose_json segment parsing.",
                        level="warning",
                    )
        segments = result.get("segments")
        if not segments:
            # Server didn't return verbose_json segment data (older/minimal
            # server, or it ignored response_format) — nothing to re-check
            # beyond the plain text.
            return result.get("text", "")
        kept = []
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            no_speech_prob = seg.get("no_speech_prob")
            avg_logprob = seg.get("avg_logprob")
            compression_ratio = seg.get("compression_ratio")
            reason = None
            if no_speech_prob is not None and no_speech_prob > self.no_speech_threshold:
                reason = f"no_speech_prob {no_speech_prob:.2f} > {self.no_speech_threshold}"
            elif avg_logprob is not None and avg_logprob < self.logprob_threshold:
                reason = f"avg_logprob {avg_logprob:.2f} < {self.logprob_threshold}"
            elif (
                compression_ratio is not None
                and compression_ratio > self.compression_ratio_threshold
            ):
                reason = (
                    f"compression_ratio {compression_ratio:.2f} > "
                    f"{self.compression_ratio_threshold}"
                )
            if reason:
                if self.logger:
                    self.logger.log(
                        f"Dropped low-confidence Whisper segment ({reason}): {text!r}",
                        level="debug",
                    )
                continue
            kept.append(text)
        return " ".join(kept)

    def transcribe(self, audio_bytes, sample_rate=16000, language=None):
        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)
            buffer.seek(0)
            files = {"file": ("audio.wav", buffer, "audio/wav")}
            data = self._build_data(language=language, task="transcribe")
            url = self.endpoint_url or f"{self.host}/audio/transcriptions"
            response = self.session.post(url, files=files, data=data, timeout=30)
            if response.status_code == 200:
                return self._extract_text(response.json())
            else:
                if self.logger:
                    self.logger.log(
                        f"Whisper API error: {response.status_code} - {response.text}",
                        level="error",
                    )
                return ""
        except Exception as e:
            if self.logger:
                self.logger.log(f"Whisper transcription error: {e}", level="error")
            return ""

    def translate(self, audio_bytes, sample_rate=16000):
        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)
            buffer.seek(0)
            files = {"file": ("audio.wav", buffer, "audio/wav")}
            data = self._build_data(task="translate")
            url = self.endpoint_url or f"{self.host}/audio/translations"
            response = self.session.post(url, files=files, data=data, timeout=30)
            if response.status_code == 200:
                return self._extract_text(response.json())
            else:
                if self.logger:
                    self.logger.log(
                        f"Whisper translation API error: {response.status_code}",
                        level="error",
                    )
                return ""
        except Exception as e:
            if self.logger:
                self.logger.log(f"Whisper translation error: {e}", level="error")
            return ""

MOONSHINE_LANGUAGES = [
    ("English", "en"),
    ("Spanish", "es"),
    ("Mandarin Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Vietnamese", "vi"),
    ("Ukrainian", "uk"),
    ("Arabic", "ar"),
]
MOONSHINE_LANGUAGE_CODES = [code for _, code in MOONSHINE_LANGUAGES]

_MOONSHINE_AVAILABLE = False
try:
    import moonshine_voice as _mv_mod

    _MOONSHINE_AVAILABLE = True
except ImportError:
    pass


class MoonshineRecognizer:
    """
    Wraps moonshine_voice.Transcriber.
    Feed raw int16 16 kHz mono bytes via add_audio(); results arrive through
    the on_result callback as (text: str, is_final: bool).
    """

    def __init__(
        self,
        language: str = "en",
        cache_dir: str = "moonshine_models",
        on_result=None,  # callable(text: str, is_final: bool)
        logger=None,
    ):
        self.language = language
        self.cache_dir = cache_dir
        self.on_result = on_result
        self.logger = logger
        self._transcriber = None
        self._started = False

    def start(self):
        """Download model if needed, create Transcriber, start session."""
        if not _MOONSHINE_AVAILABLE:
            raise ImportError(
                "moonshine-voice is not installed. Run: pip install moonshine-voice"
            )
        try:
            # Force absolute path so the native C++ library can locate cached models
            # regardless of the process working directory.
            abs_cache = os.path.abspath(self.cache_dir)
            os.makedirs(abs_cache, exist_ok=True)
            os.environ["MOONSHINE_VOICE_CACHE"] = abs_cache

            if self.logger:
                self.logger.log(
                    f"Loading Moonshine for language '{self.language}' "
                    f"(downloads model automatically if not cached)…",
                    level="info",
                )

            model_path, model_arch = _mv_mod.get_model_for_language(self.language)

            # The C++ core resolves paths from the process working directory.
            # get_model_for_language() may return a relative path (inside the cache
            # folder), which the native library can't find unless we expand it first.
            model_path = os.path.abspath(model_path)

            if self.logger:
                self.logger.log(
                    f"Moonshine model ready: {model_path} (arch {model_arch})",
                    level="info",
                )

            # Inner listener — closes over self to push results to the app queue
            outer = self

            class _Listener(_mv_mod.TranscriptEventListener):
                def on_line_text_changed(self, event):
                    if outer.on_result and event.line.text:
                        outer.on_result(event.line.text, False)

                def on_line_completed(self, event):
                    if outer.on_result and event.line.text:
                        outer.on_result(event.line.text, True)

            self._transcriber = _mv_mod.Transcriber(
                model_path=model_path,
                model_arch=model_arch,
            )
            self._transcriber.add_listener(_Listener())
            self._transcriber.start()
            self._started = True

        except Exception as exc:
            if self.logger:
                self.logger.log(f"Moonshine start error: {exc}", level="error")
            raise

    def add_audio(self, audio_bytes: bytes, sample_rate: int = 16000):
        """Feed int16 mono bytes; the library handles VAD and segmentation."""
        if self._transcriber is None or not self._started:
            return
        audio_np = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        self._transcriber.add_audio(audio_np, sample_rate)

    def close(self):
        """Stop transcription and release resources."""
        if self._transcriber and self._started:
            try:
                self._transcriber.stop()
            except Exception:
                pass
        self._transcriber = None
        self._started = False
        gc.collect()

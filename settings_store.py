"""
Per-slug settings persistence. One JSON file per slug (see session.py) under
SETTINGS_DIR, so each named session keeps its own settings independently.
"""

import json
import threading
from pathlib import Path

from session import sanitize_slug

SETTINGS_DIR = Path("settings")
SETTINGS_FILE = Path("settings.json")  # legacy single-file location (pre-slugs)

# Keys that are safe to persist between sessions
PERSISTABLE_KEYS = [
    "audio_mode",
    "recognition_engine",
    "vosk_model",
    "enable_translation",
    "display_interim",
    "translation_mode",
    "source_language",
    "target_language",
    "font_family",
    "custom_font",
    "recognized_font_size",
    "translated_font_size",
    "recognized_color",
    "translated_color",
    "background_color",
    "text_alignment",
    "translation_position",
    "whisper_host",
    "whisper_api_key",
    "whisper_model",
    "whisper_language",
    "whisper_temperature",
    "whisper_best_of",
    "whisper_beam_size",
    "whisper_patience",
    "whisper_length_penalty",
    "whisper_suppress_tokens",
    "whisper_initial_prompt",
    "whisper_condition_on_previous_text",
    "whisper_temperature_increment_on_fallback",
    "whisper_no_speech_threshold",
    "whisper_logprob_threshold",
    "whisper_compression_ratio_threshold",
    "whisper_translate_host",
    "whisper_translate_api_key",
    "whisper_translate_model",
    "whisper_translate_temperature",
    "whisper_translate_best_of",
    "whisper_translate_beam_size",
    "whisper_translate_patience",
    "whisper_translate_length_penalty",
    "whisper_translate_suppress_tokens",
    "whisper_translate_initial_prompt",
    "whisper_translate_condition_on_previous_text",
    "whisper_translate_temperature_increment_on_fallback",
    "whisper_translate_no_speech_threshold",
    "whisper_translate_logprob_threshold",
    "whisper_translate_compression_ratio_threshold",
    "argos_source_lang",
    "argos_target_lang",
    "libretranslate_host",
    "libretranslate_api_key",
    "fade_timeout",
    "ai_host",
    "ai_api_key",
    "ai_model",
    "ai_translation_prompt_template",
    "ai_endpoint_url",
    "ai_request_body_template",
    "ai_response_text_path",
    "outline_width",
    "outline_color",
    "translated_outline_width",
    "translated_outline_color",
    "vad_threshold",
    "vad_end_silence_ms",
    "subtitle_mode",
    "subtitle_cps",
    "subtitle_max_lines",
    "moonshine_language",
    "moonshine_cache_dir",
    "noise_filter_threshold",
]

_settings_save_timers: dict[str, threading.Timer] = {}
_settings_save_lock = threading.Lock()


def _settings_path(slug: str) -> Path:
    return SETTINGS_DIR / f"{sanitize_slug(slug) or 'default'}.json"


def _migrate_vad_threshold_in_place(data: dict) -> dict:
    """Migrate old 0–1 vad_threshold values to dB if present (in-place-ish)."""
    if "vad_threshold" in data:
        v = data["vad_threshold"]
        if isinstance(v, (int, float)) and v > 0:
            import math

            rms = 0.001 + (float(v) ** 1.5) * 0.499
            data["vad_threshold"] = round(
                max(-60.0, min(0.0, 20.0 * math.log10(max(rms, 1e-9)))), 1
            )
            print(f"[SETTINGS] Migrated vad_threshold {v} → {data['vad_threshold']} dB")
    return data


def load_saved_settings(slug: str) -> dict:
    """
    Load persisted settings for this slug. Returns empty dict on failure.

    If this slug has never been saved before, fall back to the legacy
    single-file settings.json (pre-named-sessions installs) so an existing
    user's tuned settings carry over into their first named session. Once
    the slug is saved, it gets its own file and no longer touches the
    legacy one.
    """
    path = _settings_path(slug)
    try:
        if path.exists():
            with open(path, "r") as f:
                data = _migrate_vad_threshold_in_place(json.load(f))
                print(f"[SETTINGS] Loaded saved settings for '{slug}' from {path}")
                return data
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r") as f:
                data = _migrate_vad_threshold_in_place(json.load(f))
                print(
                    f"[SETTINGS] No settings file for '{slug}' yet — seeding from "
                    f"legacy {SETTINGS_FILE}"
                )
                return data
    except Exception as e:
        print(f"[WARNING] Could not load settings for '{slug}': {e}")
    return {}


def persist_settings(slug: str, settings: dict):
    """Debounced save of persistable settings to disk (1 s delay), per slug."""
    with _settings_save_lock:
        existing = _settings_save_timers.get(slug)
        if existing is not None:
            existing.cancel()
        snapshot = {k: settings[k] for k in PERSISTABLE_KEYS if k in settings}
        timer = threading.Timer(1.0, _write_settings, args=[slug, snapshot])
        timer.daemon = True
        timer.start()
        _settings_save_timers[slug] = timer


def _write_settings(slug: str, data: dict):
    """Actually write settings to disk (called from timer thread)."""
    try:
        SETTINGS_DIR.mkdir(exist_ok=True)
        with open(_settings_path(slug), "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Could not save settings for '{slug}': {e}")

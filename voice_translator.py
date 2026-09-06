import argparse
import gc
import json
import os
import queue
import re
import secrets
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
import requests
import sounddevice as sd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from vosk import KaldiRecognizer, Model

from logger import Logger
import translators as tmod
from translators import TranslationService

# ── Recognizer / translation backends (see recognizers.py) ───────────────────
from recognizers import (
    ARGOS_AVAILABLE,
    MOONSHINE_LANGUAGES,
    ArgosTranslator,
    MoonshineRecognizer,
    WhisperRecognizer,
    _MOONSHINE_AVAILABLE,
    dots_or_stars,
    is_whisper_hallucination,
)
from vad import FastVAD, _WRTCVAD_AVAILABLE
from subtitles import SubtitleManager
from session import SessionSlugMiddleware, get_slug
from settings_store import SETTINGS_DIR, load_saved_settings, persist_settings

if ARGOS_AVAILABLE:
    # Needed directly (not just via ArgosTranslator) for the Argos settings
    # panel in create_ui(), which lists installed language pairs.
    import argostranslate.translate


# ── Session storage ───────────────────────────────────────────────────────────
# Keyed by *slug* (a persistent identity — see SessionSlugMiddleware below),
# not by Gradio's ephemeral per-page-load session_hash. This is what lets a
# tab reload / reconnect / a second tab re-attach to the same running
# session instead of spawning a brand new one.
SESSION_APPS: dict = {}
SESSION_LOCK = threading.Lock()

# Session slug resolution (SessionSlugMiddleware, get_slug, sanitize_slug)
# and per-slug settings persistence (load_saved_settings, persist_settings,
# PERSISTABLE_KEYS, SETTINGS_DIR) now live in session.py / settings_store.py
# — imported above. See SESSIONS.md.

# ── Font helpers ──────────────────────────────────────────────────────────────
def get_available_fonts():
    """Scan the fonts/ directory for supported font files."""
    fonts_dir = Path("fonts")
    fonts_dir.mkdir(exist_ok=True)
    fonts = []
    for ext in (".ttf", ".otf", ".woff", ".woff2"):
        for file in fonts_dir.glob(f"*{ext}"):
            fonts.append((f"[Custom] {file.stem}", file.name))
    return fonts


SYSTEM_FONTS = [
    ("Arial", "Arial"),
    ("Helvetica", "Helvetica"),
    ("Courier New", "Courier New"),
    ("Georgia", "Georgia"),
    ("Verdana", "Verdana"),
    ("Times New Roman", "Times New Roman"),
    ("Comic Sans MS", "Comic Sans MS"),
    ("Impact", "Impact"),
]

# ── Browser-side JavaScript ───────────────────────────────────────────────────
js = """
<script>
window.__audioStreamActive = false;
window.__audioProcessor = null;
window.__audioSource = null;
window.__mediaStream = null;
window.__audioContext = null;
window.__websocket = null;
window.__volumeMeterInterval = null;
window.__hwLevelInterval = null;   // hardware mic polling
window.__micTestActive = false;    // mic test (monitor) mode

function setStatus(text) {
    var status = document.getElementById('browser-status');
    if (status) status.value = text;
}

// Peak-hold state: the highest dB seen in the last PEAK_HOLD_MS milliseconds.
var peakDb = -60;
var peakHoldTimer = null;
var PEAK_HOLD_MS = 2000;   // hold peak for 2 s before decaying

function updateVolumeMeter(rmsLevel) {
    var bar          = document.getElementById('volume-meter-bar');
    var tline        = document.getElementById('volume-meter-threshold-line');
    var peakLine     = document.getElementById('volume-meter-peak-line');
    var label        = document.getElementById('volume-meter-db-label');
    var thresholdLabel = document.getElementById('volume-meter-threshold-label');
    if (!bar) return;

    // Convert RMS → dB (dBFS), clamp to display range
    var db = rmsLevel > 0 ? 20 * Math.log10(rmsLevel) : -60;
    db = Math.max(-60, Math.min(0, db));
    var percent = (db + 60) / 60 * 100;

    // ── Bar: NO smoothing — shows exactly what the VAD sees ──────────────────
    bar.style.width = percent + '%';

    // Get threshold dB from slider (direct dB value)
    var tDb = -30;
    var slider = document.querySelector('#vad-threshold-slider input[type="range"]');
    if (slider) tDb = parseFloat(slider.value) || -30;
    tDb = Math.max(-60, Math.min(0, tDb));

    // Colour: red if above threshold (VAD is active), green if below
    var aboveThreshold = db >= tDb;
    bar.style.background = aboveThreshold ? '#ff5555' : '#44cc44';

    var led = document.getElementById('vad-led');
    if (led) {
        led.style.background = aboveThreshold ? '#44ff44' : '#888';
        led.style.boxShadow = aboveThreshold ? '0 0 5px #44ff44' : 'none';
    }

    // ── dB label ──────────────────────────────────────────────────────────────
    if (label) label.textContent = db.toFixed(1) + ' dB';

    // ── Threshold line ────────────────────────────────────────────────────────
    if (tline) {
        var tPercent = (tDb + 60) / 60 * 100;
        tline.style.left = tPercent + '%';
        if (thresholdLabel) {
            thresholdLabel.textContent = 'threshold: ' + tDb.toFixed(0) + ' dB';
        }
    }

    // ── Peak-hold line ────────────────────────────────────────────────────────
    // Stays at the maximum dB seen; resets after PEAK_HOLD_MS of no new peaks.
    if (db > peakDb) {
        peakDb = db;
        if (peakHoldTimer) clearTimeout(peakHoldTimer);
        peakHoldTimer = setTimeout(function() {
            // Decay peak over 1 s after hold period
            peakDb = -60;
            if (peakLine) peakLine.style.left = '0%';
        }, PEAK_HOLD_MS);
    }
    if (peakLine) {
        var peakPercent = (peakDb + 60) / 60 * 100;
        peakLine.style.left = Math.max(0, peakPercent - 0.5) + '%'; // centre the 2px line
    }
}

document.addEventListener('DOMContentLoaded', function() {
    var slider = document.querySelector('#vad-threshold-slider input[type="range"]');
    if (slider) {
        slider.addEventListener('input', function() {
            updateVolumeMeter(window.__lastVolume || 0);
        });
    }
});

// ── Hardware mic level polling ─────────────────────────────────────────────
window.startHwLevelPolling = function() {
    if (window.__hwLevelInterval) return;
    var sessionDiv = document.getElementById('session-data');
    if (!sessionDiv) return;
    var session = sessionDiv.dataset.session;
    window.__hwLevelInterval = setInterval(function() {
        fetch('/mic_level/' + session)
            .then(function(r) { return r.json(); })
            .then(function(d) { if (typeof d.rms === 'number') updateVolumeMeter(d.rms); })
            .catch(function() {});
    }, 50);
};

window.stopHwLevelPolling = function() {
    if (window.__hwLevelInterval) {
        clearInterval(window.__hwLevelInterval);
        window.__hwLevelInterval = null;
    }
    updateVolumeMeter(0);
};

// ── Audio source detection ─────────────────────────────────────────────────
// Mirrors whichever "Audio Source" radio option is currently checked.
window.__getAudioSourceType = function() {
    var micRadio = document.querySelector('input[value="browser_mic"]');
    var dispRadio = document.querySelector('input[value="browser_display"]');
    if (dispRadio && dispRadio.checked) return 'display';
    if (micRadio && micRadio.checked) return 'mic';
    return 'mic';
};

// ── Browser streaming (full: recognition + meter) ─────────────────────────
window.startBrowserStreaming = function() {
    var micRadio = document.querySelector('input[value="browser_mic"]');
    var dispRadio = document.querySelector('input[value="browser_display"]');
    var isBrowserMode = (micRadio && micRadio.checked) || (dispRadio && dispRadio.checked);
    if (!isBrowserMode) return;
    if (window.__audioStreamActive) return;
    window.__audioStreamActive = true;
    window.__startAudioCapture(true, window.__getAudioSourceType());
};

// ── Browser mic test (meter only, no WebSocket) ───────────────────────────
window.startBrowserMicTest = function() {
    if (window.__audioStreamActive || window.__micTestActive) return;
    window.__micTestActive = true;
    window.__startAudioCapture(false, window.__getAudioSourceType());
};

window.stopBrowserMicTest = function() {
    if (!window.__micTestActive) return;
    window.__micTestActive = false;
    window.__stopAudioCapture();
};

// ── Core audio capture (shared) ───────────────────────────────────────────
// sourceType: 'mic' (getUserMedia, this device's microphone) or 'display'
// (getDisplayMedia — a shared browser tab/window/screen, with its audio —
// e.g. a Discord web tab, or on Windows/ChromeOS, whole-system audio).
window.__startAudioCapture = function(withWs, sourceType) {
    sourceType = sourceType || 'mic';
    var sessionDiv = document.getElementById('session-data');
    if (!sessionDiv) { setStatus('Error: session data missing'); window.__audioStreamActive = false; window.__micTestActive = false; return; }

    var mediaPromise;
    if (sourceType === 'display') {
        // video:true is required by the getDisplayMedia spec even though we
        // discard the track immediately below — only the audio is used.
        mediaPromise = navigator.mediaDevices.getDisplayMedia({ video: true, audio: true })
            .then(function(stream) {
                stream.getVideoTracks().forEach(function(t) { stream.removeTrack(t); t.stop(); });
                if (stream.getAudioTracks().length === 0) {
                    stream.getTracks().forEach(function(t) { t.stop(); });
                    throw new Error("No audio shared — check 'Share audio' in the picker");
                }
                return stream;
            });
    } else {
        mediaPromise = navigator.mediaDevices.getUserMedia({ audio: true });
    }

    mediaPromise
        .then(function(stream) {
            window.__mediaStream = stream;
            var AudioContext = window.AudioContext || window.webkitAudioContext;
            window.__audioContext = new AudioContext();
            var inputSampleRate = window.__audioContext.sampleRate;
            window.__audioSource = window.__audioContext.createMediaStreamSource(stream);
            window.__audioProcessor = window.__audioContext.createScriptProcessor(4096, 1, 1);
            window.__audioSource.connect(window.__audioProcessor);
            window.__audioProcessor.connect(window.__audioContext.destination);

            if (withWs) {
                var wsPath = sessionDiv.dataset.wsPath;
                var wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                window.__websocket = new WebSocket(wsProtocol + '//' + window.location.host + wsPath);
                window.__websocket.binaryType = 'arraybuffer';
                window.__websocket.onopen  = function() { setStatus('Streaming...'); };
                window.__websocket.onerror = function() { setStatus('WebSocket error'); window.stopBrowserStreaming(); };
                window.__websocket.onclose = function() { setStatus('Stopped'); window.__audioStreamActive = false; updateVolumeMeter(0); };
            } else {
                setStatus('Mic test active...');
            }

            if (window.__volumeMeterInterval) clearInterval(window.__volumeMeterInterval);
            window.__volumeMeterInterval = setInterval(function() {
                if (window.__lastVolume !== undefined) updateVolumeMeter(window.__lastVolume);
            }, 80);

            window.__audioProcessor.onaudioprocess = function(e) {
                var inputData = e.inputBuffer.getChannelData(0);

                // Resample to 16 kHz if needed
                var outputData;
                if (inputSampleRate !== 16000) {
                    var ratio = 16000 / inputSampleRate;
                    var outLen = Math.floor(inputData.length * ratio);
                    outputData = new Float32Array(outLen);
                    for (var i = 0; i < outLen; i++) {
                        var pos = i / ratio;
                        var i1 = Math.floor(pos), i2 = Math.min(i1 + 1, inputData.length - 1);
                        outputData[i] = inputData[i1] * (1 - (pos - i1)) + inputData[i2] * (pos - i1);
                    }
                } else {
                    outputData = inputData;
                }

                // Calculate PEAK RMS over 10 ms frames — exactly what the server VAD sees.
                // (server uses blocksize=480 → 3 frames of 160 samples each at 16 kHz)
                var frameSize = 160;  // 10 ms at 16 kHz
                var peakRms = 0;
                for (var f = 0; f + frameSize <= outputData.length; f += frameSize) {
                    var s2 = 0;
                    for (var i = f; i < f + frameSize; i++) s2 += outputData[i] * outputData[i];
                    var frameRms = Math.sqrt(s2 / frameSize);
                    if (frameRms > peakRms) peakRms = frameRms;
                }
                // Fallback: whole-block RMS if block is shorter than one frame
                if (peakRms === 0) {
                    var s2 = 0;
                    for (var i = 0; i < outputData.length; i++) s2 += outputData[i] * outputData[i];
                    peakRms = Math.sqrt(s2 / outputData.length);
                }
                window.__lastVolume = peakRms;

                // If streaming, send the resampled audio (already int16 conversion)
                if (withWs && window.__websocket && window.__websocket.readyState === WebSocket.OPEN) {
                    var int16 = new Int16Array(outputData.length);
                    for (var i = 0; i < outputData.length; i++) {
                        var s = Math.max(-1, Math.min(1, outputData[i]));
                        int16[i] = Math.round(s < 0 ? s * 32768 : s * 32767);
                    }
                    window.__websocket.send(int16.buffer);
                }
            };
        })
        .catch(function(err) { setStatus((sourceType === 'display' ? 'Capture error: ' : 'Mic error: ') + err.message); window.__audioStreamActive = false; window.__micTestActive = false; });
};

window.__stopAudioCapture = function() {
    if (window.__audioProcessor)  { window.__audioProcessor.disconnect(); window.__audioProcessor = null; }
    if (window.__audioSource)     { window.__audioSource.disconnect(); window.__audioSource = null; }
    if (window.__mediaStream)     { window.__mediaStream.getTracks().forEach(function(t){t.stop();}); window.__mediaStream = null; }
    if (window.__audioContext)    { window.__audioContext.close(); window.__audioContext = null; }
    if (window.__websocket)       { window.__websocket.close(); window.__websocket = null; }
    if (window.__volumeMeterInterval) { clearInterval(window.__volumeMeterInterval); window.__volumeMeterInterval = null; }
    updateVolumeMeter(0);
};

window.stopBrowserStreaming = function() {
    if (!window.__audioStreamActive) return;
    window.__audioStreamActive = false;
    window.__stopAudioCapture();
    setStatus('Stopped');
};

// Note: tab close/reload used to call /deactivate here, which fully tore
// down the session (unloaded models, stopped recognition). Sessions are now
// permanent by default — closing this tab just stops this tab's audio
// stream; the session itself stays alive until explicitly closed from the
// "Manage Sessions" panel or reclaimed by the server's idle timeout.

// ── Display + Logs polling (bypasses Gradio SSE entirely) ────────────────
// Both fetch directly from FastAPI endpoints. Zero Gradio SSE queue usage.
// This prevents the page from ever reloading due to SSE queue pressure.
window.__logsInterval    = null;

window.startAllPolling = function() {
    var sessionDiv = document.getElementById('session-data');
    if (!sessionDiv || !sessionDiv.dataset.session) return false;
    var session = sessionDiv.dataset.session;

    // Logs polling only (display is now handled by Gradio timer)
    if (!window.__logsInterval) {
        window.__logsInterval = setInterval(function() {
            fetch('/logs_data/' + session)
                .then(function(r) { return r.json(); })
                .then(function(d) {
                    var el = document.querySelector('textarea[elem_id="log_output"], #log_output textarea');
                    if (!el) {
                        var labels = document.querySelectorAll('label');
                        for (var i = 0; i < labels.length; i++) {
                            if (labels[i].textContent.trim() === 'Log') {
                                var ta = labels[i].closest('.block') && labels[i].closest('.block').querySelector('textarea');
                                if (ta) { el = ta; break; }
                            }
                        }
                    }
                    if (el && d.logs !== undefined) el.value = d.logs;
                })
                .catch(function() {});
        }, 2000);
    }
    return true;
};

window.stopAllPolling = function() {
    if (window.__displayInterval) { clearInterval(window.__displayInterval); window.__displayInterval = null; }
    if (window.__logsInterval)    { clearInterval(window.__logsInterval);    window.__logsInterval    = null; }
};

// Start polling as soon as Gradio has injected the session-data div.
// Use a brief retry loop — Gradio renders async, typically < 500ms.
document.addEventListener('DOMContentLoaded', function() {
    var attempts = 0;
    var boot = setInterval(function() {
        attempts++;
        if (window.startAllPolling()) {
            clearInterval(boot);
        } else if (attempts > 50) {
            // Give up after 5 s — Gradio took too long to inject session div
            clearInterval(boot);
        }
    }, 100);
});
</script>
"""


# dots_or_stars, is_whisper_hallucination, ArgosTranslator, WhisperRecognizer
# now live in recognizers.py — imported above.

# FastVAD now lives in vad.py; MOONSHINE_LANGUAGES/MoonshineRecognizer now
# live in recognizers.py — both imported above.

# SubtitleManager now lives in subtitles.py — imported above.

# ── VoiceTranslatorApp ────────────────────────────────────────────────────────
class VoiceTranslatorApp:
    # Default settings – overridden by saved settings on load
    DEFAULT_SETTINGS = {
        "model_path": "",
        "microphone": 0,
        "vosk_model": "",
        "audio_mode": "hardware",
        "enable_translation": False,
        "display_interim": True,
        "translation_mode": "argos",
        "source_language": "en",
        "target_language": "es",
        "font_family": "Arial",
        "recognized_font_size": 48,
        "translated_font_size": 32,
        "recognized_color": "#FFFFFF",
        "translated_color": "#CCCCCC",
        "background_color": "#000000",
        "text_alignment": "center",
        "translation_position": "after",
        "recognition_engine": "vosk",
        "whisper_host": "http://localhost:9000",
        "whisper_api_key": "",
        "whisper_model": "whisper-large-v3",
        "whisper_language": "en",
        "whisper_translate_host": "http://localhost:9000",
        "whisper_translate_api_key": "",
        "whisper_translate_model": "whisper-large-v3",
        "argos_source_lang": "en",
        "argos_target_lang": "es",
        "libretranslate_host": "",
        "libretranslate_api_key": "",
        "fade_timeout": 5.0,
        "ai_host": "http://localhost:11434",
        "ai_api_key": "",
        "ai_model": "",
        # Advanced/developer overrides for AI-translation prompt + endpoint —
        # blank means "use the built-in default", see AI_TRANSLATION.md.
        "ai_translation_prompt_template": "",
        "ai_endpoint_url": "",
        "ai_request_body_template": "",
        "ai_response_text_path": "",
        "outline_width": 0,
        "outline_color": "#000000",
        "translated_outline_width": 0,
        "translated_outline_color": "#000000",
        "custom_font": "",
        "whisper_temperature": 0.0,
        "whisper_best_of": 5,
        "whisper_beam_size": 5,
        "whisper_patience": 1.0,
        "whisper_length_penalty": 1.0,
        "whisper_suppress_tokens": "-1",
        "whisper_initial_prompt": "",
        "whisper_condition_on_previous_text": True,
        "whisper_temperature_increment_on_fallback": 0.2,
        "whisper_no_speech_threshold": 0.6,
        "whisper_logprob_threshold": -1.0,
        "whisper_compression_ratio_threshold": 2.4,
        "whisper_translate_temperature": 0.0,
        "whisper_translate_best_of": 5,
        "whisper_translate_beam_size": 5,
        "whisper_translate_patience": 1.0,
        "whisper_translate_length_penalty": 1.0,
        "whisper_translate_suppress_tokens": "-1",
        "whisper_translate_initial_prompt": "",
        "whisper_translate_condition_on_previous_text": True,
        "whisper_translate_temperature_increment_on_fallback": 0.2,
        "whisper_translate_no_speech_threshold": 0.6,
        "whisper_translate_logprob_threshold": -1.0,
        "whisper_translate_compression_ratio_threshold": 2.4,
        # VAD — threshold + end-of-speech delay are both hot-reloadable
        "vad_threshold": -30.0,  # dB — the slider value is now in dB directly
        "vad_end_silence_ms": 300,  # ms of silence before dispatching (Whisper/Moonshine) — was 80ms, too short: natural mid-sentence pauses (breathing, thinking) routinely exceed that, so Whisper got flooded with tiny fragmented clips, which is exactly when it hallucinates fillers like "thank you"
        # Moonshine (moonshine-voice package)
        "moonshine_language": "en",
        "moonshine_cache_dir": "moonshine_models",
        # Subtitle display
        "subtitle_mode": "instant",  # "instant" | "buffered"
        "subtitle_cps": 21,  # characters per second
        "subtitle_max_lines": 2,  # sentences per chunk in buffered mode
        "noise_filter_threshold": 0.0,
    }

    def __init__(self, slug: str):
        self.session_active = True
        self.display_running = True
        self.slug = slug
        self.last_active = time.time()
        self.popout_id = secrets.token_urlsafe(16)

        self.audio_queue: queue.Queue = (
            queue.Queue()
        )  # ~450ms max backlog at 30ms blocks
        self.result_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.is_monitoring = False  # mic-test mode (level only, no recognition)

        self.recognizer = None
        self.model = None
        self._vosk_model_path: str | None = None  # which path self.model is a shared reference to — needed by _unload_vosk to release the right cache entry
        self.whisper_recognizer: WhisperRecognizer | None = None
        self.moonshine_recognizer: MoonshineRecognizer | None = None
        self.argos_translator: ArgosTranslator | None = None
        self.translation_service: TranslationService | None = None
        self.stream = None
        self._monitor_stream = None
        self.vad: FastVAD | None = None
        self.monitor_level: float = 0.0  # latest RMS from hardware mic

        self._transcribe_sem = threading.Semaphore(3)

        # Serializes start/stop/monitor transitions. Without this, Gradio's
        # concurrency settings (interface.queue(default_concurrency_limit=None))
        # allow two clicks on this session — e.g. a fast Start-then-Stop, or a
        # double-click — to run start_recognition()/stop_recognition() on two
        # different threads AT THE SAME TIME. That's a real crash vector: Vosk's
        # recognizer/model are C++ objects, and one thread deleting them (stop)
        # while another thread is mid-call into them (start's processing loop,
        # or vice versa) is undefined behavior — it can hard-crash the whole
        # Python process, not just raise a catchable exception. That matches
        # "the service restarts when I start and stop a model."
        self._control_lock = threading.RLock()

        self.vosk_audio_buffer = bytearray()
        self.last_audio_chunk: bytes | None = None

        self.logger = Logger(session_id=slug)

        # Build settings: start from defaults, overlay saved settings
        self.settings: dict = dict(self.DEFAULT_SETTINGS)
        self.settings.update(load_saved_settings(slug))
        # Migrate old "browser" audio_mode value → "browser_mic" (Phase 3
        # split the old single browser mode into mic vs tab/system audio).
        if self.settings.get("audio_mode") == "browser":
            self.settings["audio_mode"] = "browser_mic"

        # Subtitle manager — settings applied dynamically
        self.subtitles = SubtitleManager(
            mode=self.settings["subtitle_mode"],
            cps=self.settings["subtitle_cps"],
            max_lines=self.settings["subtitle_max_lines"],
            fade_timeout=self.settings["fade_timeout"],
        )

        if ARGOS_AVAILABLE:
            self.argos_translator = ArgosTranslator(logger=self.logger)

        self.display_thread = threading.Thread(
            target=self._update_display_loop, daemon=True
        )
        self.display_thread.start()

    # ── Validation helpers ────────────────────────────────────────────────────
    def is_repetitive_garbage(self, text: str) -> bool:
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) > 10:
            counts: dict = {}
            for w in words:
                counts[w] = counts.get(w, 0) + 1
            if max(counts.values()) > 5:
                self.logger.log(f"Filtered repetitive words: {text}", level="debug")
                return True
        cleaned = re.sub(r"[\s.,!?]", "", text)
        if len(cleaned) > 10:
            max_run = current_run = 1
            for i in range(1, len(cleaned)):
                if cleaned[i] == cleaned[i - 1]:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1
            if max_run > 5:
                self.logger.log(f"Filtered character run: {text}", level="debug")
                return True
        return False

    def is_valid_transcription(self, text: str) -> bool:
        if re.search(r"<\|.*?\|>", text):
            self.logger.log(f"Filtered (timestamp token): {text}", level="debug")
            return False
        if len(text) > 500:
            self.logger.log(f"Filtered (too long): {text}", level="debug")
            return False
        return not self.is_repetitive_garbage(text)

    def _moonshine_result(self, text: str, is_final: bool):
        """Called by MoonshineRecognizer's listener thread when speech is detected."""
        if not text or not self.is_running:
            return
        if is_final:
            if self.is_valid_transcription(text):
                self.result_queue.put(("final", text))
        elif self.settings.get("display_interim"):
            self.result_queue.put(("interim", text))

    # ── Dynamic settings helpers ──────────────────────────────────────────────
    def apply_vad_settings(self):
        """Push latest threshold + end_silence + noise filter into the running VAD."""
        if self.vad:
            self.vad.update_threshold(self.settings.get("vad_threshold", -30.0))
            self.vad.update_end_silence_ms(self.settings.get("vad_end_silence_ms", 300))
            thresh = self.settings.get("noise_filter_threshold", 0.0)
            self.vad.update_noise_filter_threshold(thresh)

    def apply_subtitle_settings(self):
        """Push latest subtitle settings into SubtitleManager."""
        self.subtitles.update_settings(
            mode=self.settings.get("subtitle_mode", "instant"),
            cps=self.settings.get("subtitle_cps", 21),
            max_lines=self.settings.get("subtitle_max_lines", 2),
            fade_timeout=self.settings.get("fade_timeout", 5.0),
        )

    # ── Mic monitoring (test mic, level display without recognition) ──────────
    def _monitor_callback(self, indata, frames, time_info, status):
        samples = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0
        frame_len = 160
        n_frames = len(samples) // frame_len
        if n_frames > 0:
            peak_rms = 0.0
            for i in range(n_frames):
                frame = samples[i * frame_len : (i + 1) * frame_len]
                rms = float(np.sqrt(np.mean(frame**2)))
                if rms > peak_rms:
                    peak_rms = rms
            self.monitor_level = peak_rms
        else:
            self.monitor_level = float(np.sqrt(np.mean(samples**2)))

    def start_mic_monitor(self) -> str:
        if not self._control_lock.acquire(timeout=2.0):
            return "⏳ Still starting/stopping — try again in a moment"
        try:
            if self.is_running:
                return "⚠️ Recognition is active — level already visible in the meter"
            if self.is_monitoring:
                return "Already monitoring"
            mic = self.settings.get("microphone")
            if mic is None:
                return "❌ No microphone selected"
            try:
                self._monitor_stream = sd.RawInputStream(
                    samplerate=16000,
                    blocksize=480,  # 30 ms — fast meter updates
                    device=mic,
                    dtype="int16",
                    channels=1,
                    callback=self._monitor_callback,
                )
                self._monitor_stream.start()
                self.is_monitoring = True
                return "🎙️ Mic monitor active — adjust threshold until line sits above background noise"
            except Exception as exc:
                return f"❌ Monitor error: {exc}"
        finally:
            self._control_lock.release()

    def stop_mic_monitor(self) -> str:
        if not self._control_lock.acquire(timeout=5.0):
            return "⏳ Still busy — try Stop again in a moment"
        try:
            if self._monitor_stream:
                try:
                    self._monitor_stream.stop()
                    self._monitor_stream.close()
                except Exception:
                    pass
                self._monitor_stream = None
            self.is_monitoring = False
            self.monitor_level = 0.0
            return "⏹️ Monitor stopped"
        finally:
            self._control_lock.release()

    # ── Audio processing ──────────────────────────────────────────────────────
    def audio_callback(self, indata, frames, time_info, status):
        if self.is_running:
            data = bytes(indata)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            frame_len = 160
            n_frames = len(samples) // frame_len
            if n_frames > 0:
                peak_rms = 0.0
                for i in range(n_frames):
                    frame = samples[i * frame_len : (i + 1) * frame_len]
                    rms = float(np.sqrt(np.mean(frame**2)))
                    if rms > peak_rms:
                        peak_rms = rms
                self.monitor_level = peak_rms
            else:
                self.monitor_level = float(np.sqrt(np.mean(samples**2)))

            self.audio_queue.put(data)  # <-- unbounded queue, never blocks

    def _get_or_create_vad(self) -> FastVAD:
        if self.vad is None:
            self.vad = FastVAD(
                threshold_db=self.settings.get("vad_threshold", -30.0),
                end_silence_ms=self.settings.get("vad_end_silence_ms", 300),
                noise_filter_threshold=self.settings.get("noise_filter_threshold", 0.0),
            )
        return self.vad

    def _process_vosk(self, data: bytes):
        """
        Feed audio to Vosk. Noise preprocessing applied first so Vosk receives
        cleaner audio. Vosk's own endpointer handles segmentation.
        """
        if self.recognizer is None:
            # Engine was switched away from vosk (or we're mid-stop) while a
            # block was already in flight to this thread — drop it instead of
            # crashing on a None recognizer.
            return
        vad = self._get_or_create_vad()
        clean = vad.preprocess_block(data)
        self.vosk_audio_buffer.extend(clean)
        if self.recognizer.AcceptWaveform(clean):
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                self.logger.log(f"Recognized: {text}", level="info")
                self.result_queue.put(("final", text))
                self.last_audio_chunk = bytes(self.vosk_audio_buffer)
                self.vosk_audio_buffer.clear()
            # Empty final (silence) — let the fade timer clear display naturally
        elif self.settings.get("display_interim"):
            partial = json.loads(self.recognizer.PartialResult())
            partial_text = partial.get("partial", "").strip()
            if partial_text:
                self.result_queue.put(("interim", partial_text))

    def _process_vad_engine(self, data: bytes):
        """
        VAD + preprocessing pipeline for Whisper and Moonshine.
        process_chunk() handles both noise cleaning and segmentation internally.
        Moonshine receives clean preprocessed audio; Whisper receives clean segments.
        """
        engine = self.settings["recognition_engine"]
        vad = self._get_or_create_vad()

        if engine == "moonshine":
            # Moonshine has its own VAD — feed clean audio continuously
            clean = vad.preprocess_block(data)
            if self.moonshine_recognizer:
                self.moonshine_recognizer.add_audio(clean)
            return

        # Whisper: VAD segments + cleans in one pass
        for speech_bytes in vad.process_chunk(data):
            self._transcribe_segment(engine, speech_bytes)

    def _apply_noise_filter(self, data: bytes) -> bytes:
        """Legacy shim — delegates to VAD's integrated preprocessor."""
        return self._get_or_create_vad().preprocess_block(data)

    def _transcribe_segment(self, engine: str, speech_bytes: bytes):
        """Non-blocking dispatch to a worker thread."""
        self.last_audio_chunk = speech_bytes
        if not self._transcribe_sem.acquire(blocking=False):
            self.logger.log(
                "Transcription backlog – dropping segment (recogniser is busy)",
                level="warning",
            )
            return
        t = threading.Thread(
            target=self._do_transcribe, args=(engine, speech_bytes), daemon=True
        )
        t.start()

    def _do_transcribe(self, engine: str, speech_bytes: bytes):
        """Worker: transcribe one Whisper segment and enqueue the result."""
        try:
            transcription = ""
            if engine == "whisper":
                rec = self.whisper_recognizer
                if rec is None:
                    return
                transcription = rec.transcribe(
                    speech_bytes, language=self.settings.get("whisper_language")
                )
            if transcription and not dots_or_stars(transcription):
                if is_whisper_hallucination(transcription):
                    self.logger.log(
                        f"Blocked hallucination: {repr(transcription)}", level="debug"
                    )
                elif self.is_valid_transcription(transcription):
                    self.result_queue.put(("final", transcription))
                else:
                    self.logger.log("Discarded invalid transcription", level="debug")
        except Exception as exc:
            self.logger.log(f"Transcription error: {exc}", level="error")
        finally:
            self._transcribe_sem.release()

    def process_audio_hardware(self):
        while self.is_running:
            try:
                # Timeout matches blocksize (30ms) so we never wait longer than one block
                data = self.audio_queue.get(timeout=0.03)
                engine = self.settings["recognition_engine"]
                if engine == "vosk":
                    self._process_vosk(data)
                else:
                    self._process_vad_engine(data)
            except queue.Empty:
                continue
            except Exception as exc:
                self.logger.log(f"Audio processing error: {exc}", level="error")

    # ── Recognition control ───────────────────────────────────────────────────
    def start_recognition(self, model_path: str | None, microphone_index=None) -> str:
        # Non-blocking: a rapid double-click on Start should be rejected
        # immediately rather than queue up and fire a second start while the
        # first is still loading a model.
        if not self._control_lock.acquire(timeout=0.1):
            return "⏳ Already starting or stopping — please wait a moment"
        try:
            if self.is_running:
                return "⚠️ Already running — press Stop first"

            # Defensive: release any vosk model reference from a previous run
            # that wasn't cleanly stopped. Shouldn't happen given the
            # is_running guard + stop-before-start invariant, but a leaked
            # cache refcount would slowly starve RAM over time, so make sure
            # every start begins from a clean slate regardless.
            if self._vosk_model_path:
                _release_vosk_model(self._vosk_model_path)
                self._vosk_model_path = None
            self.model = None
            self.recognizer = None

            engine = self.settings["recognition_engine"]

            if engine == "vosk":
                if not model_path or not Path(model_path).exists():
                    msg = "❌ Vosk model not found. Place model in vosk_models/"
                    self.logger.log(msg, level="error")
                    return msg
                self.model = _acquire_vosk_model(model_path)
                self._vosk_model_path = model_path
                self.recognizer = KaldiRecognizer(self.model, 16000)
                self.recognizer.SetWords(True)
                self.whisper_recognizer = None
                self.moonshine_recognizer = None

            elif engine == "whisper":
                self.whisper_recognizer = WhisperRecognizer(
                    host=self.settings["whisper_host"],
                    api_key=self.settings.get("whisper_api_key") or None,
                    model=self.settings["whisper_model"],
                    logger=self.logger,
                    temperature=self.settings["whisper_temperature"],
                    best_of=self.settings["whisper_best_of"],
                    beam_size=self.settings["whisper_beam_size"],
                    patience=self.settings["whisper_patience"],
                    length_penalty=self.settings["whisper_length_penalty"],
                    suppress_tokens=self.settings["whisper_suppress_tokens"],
                    initial_prompt=self.settings["whisper_initial_prompt"] or None,
                    condition_on_previous_text=self.settings[
                        "whisper_condition_on_previous_text"
                    ],
                    temperature_increment_on_fallback=self.settings[
                        "whisper_temperature_increment_on_fallback"
                    ],
                    no_speech_threshold=self.settings["whisper_no_speech_threshold"],
                    logprob_threshold=self.settings["whisper_logprob_threshold"],
                    compression_ratio_threshold=self.settings[
                        "whisper_compression_ratio_threshold"
                    ],
                )
                self.recognizer = None
                self.model = None
                self.moonshine_recognizer = None

            elif engine == "moonshine":
                if not _MOONSHINE_AVAILABLE:
                    msg = "❌ Moonshine is not installed. Run: pip install moonshine-voice"
                    self.logger.log(msg, level="error")
                    return msg
                self.moonshine_recognizer = MoonshineRecognizer(
                    language=self.settings["moonshine_language"],
                    cache_dir=self.settings["moonshine_cache_dir"],
                    on_result=self._moonshine_result,
                    logger=self.logger,
                )
                self.moonshine_recognizer.start()

            # Stop mic monitor if it was running — recognition provides the level now
            if self.is_monitoring:
                self.stop_mic_monitor()

            # Reset/recreate VAD with current settings
            if self.vad:
                self.vad.flush()
            self.vad = FastVAD(
                threshold_db=self.settings.get("vad_threshold", -30.0),
                end_silence_ms=self.settings.get("vad_end_silence_ms", 300),
                noise_filter_threshold=self.settings.get("noise_filter_threshold", 0.0),
            )

            # Reset subtitle buffer for new session
            self.subtitles.clear()

            # Moonshine (ONNX) is not thread-safe — serialize transcription.
            # Whisper is network-bound — allow up to 3 parallel calls.
            if engine == "moonshine":
                self._transcribe_sem = threading.Semaphore(1)
            else:
                self._transcribe_sem = threading.Semaphore(3)

            self.is_running = True

            if (
                self.settings["audio_mode"] == "hardware"
                and microphone_index is not None
            ):
                self.settings["microphone"] = microphone_index
                device_info = sd.query_devices(microphone_index, "input")
                self.logger.log(
                    f"Using hardware mic: {device_info['name']}", level="info"
                )
                self.stream = sd.RawInputStream(
                    samplerate=16000,
                    blocksize=480,  # 30 ms — Vosk gets results ~3× faster; VAD fires sooner
                    device=microphone_index,
                    dtype="int16",
                    channels=1,
                    callback=self.audio_callback,
                )
                self.stream.start()
                msg = "✅ Recognition started (Hardware)"
            else:
                msg = "✅ Recognition started (Browser)"

            self.process_thread = threading.Thread(
                target=self.process_audio_hardware, daemon=True
            )
            self.process_thread.start()

            if self.settings.get("translation_mode") == "ai":
                self.translation_service = TranslationService(
                    self.settings, self.logger
                )

            self.logger.log(msg, level="success")
            return msg

        except Exception as exc:
            msg = f"❌ Error starting recognition: {exc}"
            self.logger.log(msg, level="error")
            return msg
        finally:
            self._control_lock.release()

    def stop_recognition(self) -> str:
        """
        Stop audio capture and unload all recognition models to free memory.
        Settings are fully preserved – pressing Start again will reload models.
        """
        # Blocking with a generous timeout: if Start is mid-way through loading
        # a big model, Stop should wait for that to finish and then unload it
        # cleanly, rather than race it. Better a brief wait than a crash.
        if not self._control_lock.acquire(timeout=30.0):
            return "⏳ Still starting — press Stop again in a moment"
        try:
            if not self.is_running and self.stream is None and self.recognizer is None:
                return "Already stopped"
            self.is_running = False

            # Stop hardware stream
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None

            # Give the processing thread a moment to see is_running=False and
            # exit its loop before we start deleting the objects it's using.
            pt = getattr(self, "process_thread", None)
            if pt and pt.is_alive():
                pt.join(timeout=1.0)

            # Clear audio buffers
            self.vosk_audio_buffer.clear()

            # Flush any speech the VAD was accumulating mid-utterance (Whisper only)
            if self.vad:
                leftover = self.vad.flush()
                engine = self.settings.get("recognition_engine", "vosk")
                if leftover and engine == "whisper":
                    # Dispatch synchronously before tearing down the recognizer
                    self._do_transcribe(engine, leftover)
                self.vad = None

            # Unload Vosk model to free memory
            self._unload_vosk()

            # Unload Whisper
            self.whisper_recognizer = None

            # Stop Moonshine — its Transcriber.stop() fires on_line_completed for any open line
            if self.moonshine_recognizer:
                self.moonshine_recognizer.close()
                self.moonshine_recognizer = None

            self.translation_service = None

            gc.collect()

            msg = "⏹️ Recognition stopped (models unloaded)"
            self.logger.log(msg, level="success")
            return msg
        finally:
            self._control_lock.release()

    def _unload_vosk(self):
        """
        Release this session's KaldiRecognizer (always exclusive to this
        session — cheap) and its reference to the shared Vosk Model (only
        actually unloaded from RAM once every session using that same model
        path has stopped — see _release_vosk_model).
        """
        if self.recognizer:
            try:
                del self.recognizer
            except Exception:
                pass
            self.recognizer = None
        if self.model:
            _release_vosk_model(self._vosk_model_path)
            self.model = None
            self._vosk_model_path = None
        for _ in range(3):
            gc.collect()

    # ── Display ───────────────────────────────────────────────────────────────
    def _update_display_loop(self):
        while self.display_running:
            try:
                result_type, text = self.result_queue.get(timeout=0.02)  # 20ms max wait
                if result_type == "stop":
                    break
                if result_type == "final":
                    translated = ""
                    if self.settings["enable_translation"]:
                        translated = self._translate(text)
                    self.subtitles.add(text, translated)
                elif result_type == "interim" and self.settings.get("display_interim"):
                    self.subtitles.set_interim(text)
            except queue.Empty:
                if not self.display_running:
                    break
                time.sleep(0.005)
                # No sleep — tight loop so results appear as fast as possible.
                # CPU cost is negligible (most iterations are the 20ms queue timeout).
            except Exception as exc:
                if self.display_running:
                    self.logger.log(f"Display loop error: {exc}", level="error")

    def _translate(self, text: str) -> str:
        """Translate text using the configured translation mode."""
        mode = self.settings.get("translation_mode")
        try:
            if mode == "whisper" and self.last_audio_chunk:
                wt = WhisperRecognizer(
                    host=self.settings.get("whisper_translate_host"),
                    api_key=self.settings.get("whisper_translate_api_key") or None,
                    model=self.settings.get("whisper_translate_model"),
                    logger=self.logger,
                    temperature=self.settings["whisper_translate_temperature"],
                    best_of=self.settings["whisper_translate_best_of"],
                    beam_size=self.settings["whisper_translate_beam_size"],
                    patience=self.settings["whisper_translate_patience"],
                    length_penalty=self.settings["whisper_translate_length_penalty"],
                    suppress_tokens=self.settings["whisper_translate_suppress_tokens"],
                    initial_prompt=self.settings["whisper_translate_initial_prompt"]
                    or None,
                    condition_on_previous_text=self.settings[
                        "whisper_translate_condition_on_previous_text"
                    ],
                    temperature_increment_on_fallback=self.settings[
                        "whisper_translate_temperature_increment_on_fallback"
                    ],
                    no_speech_threshold=self.settings[
                        "whisper_translate_no_speech_threshold"
                    ],
                    logprob_threshold=self.settings[
                        "whisper_translate_logprob_threshold"
                    ],
                    compression_ratio_threshold=self.settings[
                        "whisper_translate_compression_ratio_threshold"
                    ],
                )
                return wt.translate(self.last_audio_chunk)
            elif mode == "argos" and self.argos_translator:
                return self.argos_translator.translate(
                    text,
                    self.settings.get("argos_source_lang"),
                    self.settings.get("argos_target_lang"),
                )
            elif mode == "ai" and self.translation_service:
                return self.translation_service.translate(
                    text,
                    self.settings["source_language"].split("-")[0],
                    self.settings["target_language"],
                )
            elif mode == "libretranslate":
                # Create a temporary translation service for libretranslate
                ts = TranslationService(self.settings, self.logger)
                return ts.translate(
                    text,
                    self.settings["source_language"].split("-")[0],
                    self.settings["target_language"],
                )
        except Exception as exc:
            self.logger.log(f"Translation error: {exc}", level="error")
        return ""

    def get_display_html(self, recognized_text: str, translated_text: str) -> str:
        alignment_map = {"left": "flex-start", "center": "center", "right": "flex-end"}
        base_style = f"font-family:{self._get_font_family_css()};white-space:pre-wrap;"
        rec_outline = self._get_outline_css(
            self.settings.get("outline_width", 0),
            self.settings.get("outline_color", "#000000"),
        )
        trans_outline = self._get_outline_css(
            self.settings.get("translated_outline_width", 0),
            self.settings.get("translated_outline_color", "#000000"),
        )

        # Fade: delegate entirely to SubtitleManager — if it returns empty strings we fade
        opacity = "0" if (not recognized_text and not translated_text) else "1"

        def line_div(text, size, color, outline):
            return f'<div style="font-size:{size}px;color:{color};{base_style}{outline}">{text}</div>'

        parts = []
        if (
            self.settings["enable_translation"]
            and self.settings["translation_position"] == "before"
            and translated_text
        ):
            parts.append(
                line_div(
                    translated_text,
                    self.settings["translated_font_size"],
                    self.settings["translated_color"],
                    trans_outline,
                )
            )
        if recognized_text:
            parts.append(
                line_div(
                    recognized_text,
                    self.settings["recognized_font_size"],
                    self.settings["recognized_color"],
                    rec_outline,
                )
            )
        if (
            self.settings["enable_translation"]
            and self.settings["translation_position"] == "after"
            and translated_text
        ):
            parts.append(
                line_div(
                    translated_text,
                    self.settings["translated_font_size"],
                    self.settings["translated_color"],
                    trans_outline,
                )
            )

        font_face = self._get_font_face_css()
        return (
            f"<style>{font_face}</style>"
            f'<div style="transition:opacity 0.5s;opacity:{opacity};display:flex;flex-direction:column;'
            f"align-items:{alignment_map[self.settings['text_alignment']]};justify-content:center;"
            f'padding:20px;background-color:{self.settings["background_color"]};min-height:200px;">'
            + "".join(parts)
            + "</div>"
        )

    def get_current_display(self):
        rec, trans = self.subtitles.get_display()
        if not self.settings["enable_translation"]:
            trans = ""
        return (
            self.get_display_html(rec, trans),
            rec,
            trans,
        )

    def update_logs(self):
        return self.logger.get_recent_logs(50)

    # ── Display HTML ──────────────────────────────────────────────────────────
    def _get_font_family_css(self) -> str:
        if self.settings.get("custom_font"):
            font_name = Path(self.settings["custom_font"]).stem.replace(" ", "_")
            return f"'{font_name}', {self.settings['font_family']}"
        return self.settings["font_family"]

    def _get_font_face_css(self) -> str:
        custom_font = self.settings.get("custom_font")
        if not custom_font:
            return ""
        font_name = Path(custom_font).stem.replace(" ", "_")
        return (
            f"@font-face {{font-family:'{font_name}';"
            f"src:url('/fonts/{custom_font}') format('truetype');"
            f"font-weight:normal;font-style:normal;}}"
        )

    def _get_outline_css(self, width: int, color: str) -> str:
        if width == 0:
            return ""
        offsets = [
            f"{dx}px {dy}px 0 {color}"
            for dx in range(-width, width + 1)
            for dy in range(-width, width + 1)
            if not (dx == 0 and dy == 0)
        ]
        return "text-shadow: " + ", ".join(offsets) + ";"

    def get_display(self) -> tuple[str, str]:
        """Return (recognized, translated) for the current render frame."""
        with self._lock:
            now = time.time()

            if self.mode == "instant":
                if self._last_add == 0.0:
                    return "", ""
                if self._cur_rec or self._cur_trans:
                    if now - self._last_add <= self.fade_timeout:
                        rec = self._cur_rec if self._cur_rec else self._last_final_rec
                        trans = (
                            self._cur_trans
                            if self._cur_trans
                            else self._last_final_trans
                        )
                        return rec, trans
                return "", ""

            # ========== BUFFERED MODE ==========
            # First, check if current text's hold time has expired
            if self._cur_rec or self._cur_trans:
                if now >= self._show_until:
                    # Current text has expired - clear it
                    self._cur_rec = ""
                    self._cur_trans = ""
                    # If no more queued sentences, return empty
                    if not self._rec_queue:
                        return "", ""

            # If we have queued sentences, show the next chunk (only if nothing is currently showing)
            if self._rec_queue and not self._cur_rec and not self._cur_trans:
                rec_lines, trans_lines = [], []
                for _ in range(min(self.max_lines, len(self._rec_queue))):
                    rec_lines.append(self._rec_queue.pop(0))
                    if self._trans_queue:
                        trans_lines.append(self._trans_queue.pop(0))

                self._cur_rec = ", ".join(rec_lines)
                self._cur_trans = ", ".join(trans_lines)
                text_length = len(self._cur_rec)
                hold = max(text_length / self.cps, 2.5)  # Minimum 2.5 seconds
                self._show_until = now + hold
                return self._cur_rec, self._cur_trans

            # Return current text if it exists (and not expired, because expired would have been cleared above)
            if self._cur_rec or self._cur_trans:
                return self._cur_rec, self._cur_trans

            return "", ""

    def generate_popout_html(self) -> str:
        alignment_map = {"left": "flex-start", "center": "center", "right": "flex-end"}
        fade_ms = int(self.settings.get("fade_timeout", 5.0) * 1000)
        font_face = self._get_font_face_css()
        font_family = self._get_font_family_css()
        rec_outline = self._get_outline_css(
            self.settings.get("outline_width", 0),
            self.settings.get("outline_color", "#000000"),
        )
        trans_outline = self._get_outline_css(
            self.settings.get("translated_outline_width", 0),
            self.settings.get("translated_outline_color", "#000000"),
        )
        return (
            f'<!DOCTYPE html><html><head><title>Display</title><meta charset="UTF-8">'
            f"<style>{font_face}"
            f"body,html{{margin:0;padding:0;width:100vw;height:100vh;overflow:hidden;"
            f"background:{self.settings['background_color']};display:flex;align-items:center;"
            f"justify-content:{alignment_map[self.settings['text_alignment']]}}}"
            f".container{{padding:20px;width:100%;text-align:{self.settings['text_alignment']};"
            f"transition:opacity 1s;opacity:1}}.container.fade{{opacity:0}}"
            f".rec{{font-size:{self.settings['recognized_font_size']}px;"
            f"color:{self.settings['recognized_color']};margin:10px 0;"
            f"font-family:{font_family};{rec_outline}}}"
            f".tra{{font-size:{self.settings['translated_font_size']}px;"
            f"color:{self.settings['translated_color']};margin:10px 0;"
            f"font-family:{font_family};{trans_outline}}}"
            f"</style>"
            f"<script>let t=null;"
            f'function reset(){{const c=document.getElementById("c");c.classList.remove("fade");'
            f'if(t)clearTimeout(t);t=setTimeout(()=>c.classList.add("fade"),{fade_ms})}}'
            f'async function update(){{try{{const r=await fetch("/popout_data/{self.popout_id}");'
            f'const d=await r.json();const e=document.getElementById("r");'
            f'const n=d.recognized||"";'
            f"if(n!==e.textContent){{e.textContent=n;reset()}}"
            f'document.getElementById("t").textContent=d.translated||""'
            f"}}catch(e){{}}}}"
            f"setInterval(update,500);"
            f'document.addEventListener("DOMContentLoaded",()=>{{update();reset()}});</script>'
            f"</head><body>"
            f'<div id="c" class="container">'
            f'<div id="r" class="rec"></div>'
            f'<div id="t" class="tra"></div>'
            f"</div></body></html>"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def close(self):
        """Full cleanup: stop recognition, stop monitor, stop display thread."""
        if self.is_running:
            self.stop_recognition()
        if self.is_monitoring:
            self.stop_mic_monitor()

        self.display_running = False
        if self.display_thread and self.display_thread.is_alive():
            try:
                self.result_queue.put(("stop", ""))
                self.display_thread.join(timeout=2.0)
            except Exception:
                pass

        # Drain queues
        for q in (self.result_queue, self.audio_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        self.argos_translator = None
        gc.collect()
        print(f"[SESSION CLOSED] {self.slug}")

    def deactivate_session(self):
        self.session_active = False
        self.close()

    def touch(self):
        """Mark this session as recently active (used by the idle reaper)."""
        self.last_active = time.time()


# ── Global helpers ────────────────────────────────────────────────────────────
def get_available_models() -> list[tuple[str, str]]:
    models_dir = Path("vosk_models")
    models_dir.mkdir(exist_ok=True)
    return [
        (item.name, str(item))
        for item in models_dir.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    ]


def get_microphones() -> list[tuple[str, int]]:
    try:
        devices = sd.query_devices()
        return [
            (f"[ID:{i}] {d['name']}", i)
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
    except Exception:
        return [("Error getting devices", 0)]


def _migrate_vad_threshold(v) -> float:
    """
    Migrate old 0–1 threshold values to dB.
    New slider range is -60 to 0 dB.  Old values were 0.0–1.0.
    A saved value > 0 and <= 1.0 is almost certainly the old format; convert it.
    Values already in dB are negative (or 0), so we can detect them trivially.
    """
    try:
        v = float(v)
    except (TypeError, ValueError):
        return -30.0
    if v > 0:
        # Old 0–1 value: convert via rms formula then to dB
        # rms = 0.001 + v^1.5 * 0.499  →  dB = 20*log10(rms)
        import math

        rms = 0.001 + (v**1.5) * 0.499
        return max(-60.0, min(0.0, 20.0 * math.log10(max(rms, 1e-9))))
    # Already a negative dB value — clamp to valid range
    return max(-60.0, min(0.0, v))


# ── Shared Vosk model cache ───────────────────────────────────────────────────
# Vosk's own API is explicitly designed for this: one Model (the large,
# memory-heavy object — hundreds of MB to a few GB depending on which model)
# can back many independent KaldiRecognizer instances at once, safely, across
# threads. Previously every session called Model(model_path) itself, so two
# sessions using the *same* model path loaded it into RAM twice, three
# sessions loaded it three times, etc. This cache loads each distinct model
# path once and hands out a refcounted reference; the Model is only actually
# freed once every session using it has stopped. Whisper (remote API) and
# Moonshine (small ONNX models with per-session streaming state that can't be
# shared) aren't included — see RECOGNITION_QUALITY.md / MODEL_SHARING.md.
_VOSK_MODEL_LOCK = threading.Lock()
_VOSK_MODEL_CACHE: dict[str, dict] = {}  # model_path -> {"model": Model, "refcount": int}


def _acquire_vosk_model(model_path: str) -> Model:
    with _VOSK_MODEL_LOCK:
        entry = _VOSK_MODEL_CACHE.get(model_path)
        if entry is None:
            print(f"[VOSK] Loading model '{model_path}' (first session to use it)")
            entry = {"model": Model(model_path), "refcount": 0}
            _VOSK_MODEL_CACHE[model_path] = entry
        entry["refcount"] += 1
        print(
            f"[VOSK] '{model_path}' now shared by {entry['refcount']} session(s)"
        )
        return entry["model"]


def _release_vosk_model(model_path: str | None):
    if not model_path:
        return
    with _VOSK_MODEL_LOCK:
        entry = _VOSK_MODEL_CACHE.get(model_path)
        if entry is None:
            return
        entry["refcount"] -= 1
        if entry["refcount"] <= 0:
            print(f"[VOSK] Unloading '{model_path}' — no sessions using it anymore")
            del _VOSK_MODEL_CACHE[model_path]
            # The cache's reference to entry["model"] is gone; once this
            # session's own self.model reference is cleared too (right after
            # this call, in _unload_vosk), nothing references the underlying
            # C++ object anymore and it's freed on the next gc.collect().
        else:
            print(
                f"[VOSK] '{model_path}' still shared by {entry['refcount']} "
                f"other session(s) — not unloading"
            )


def get_or_create_app(slug: str) -> VoiceTranslatorApp:
    with SESSION_LOCK:
        if slug not in SESSION_APPS:
            SESSION_APPS[slug] = VoiceTranslatorApp(slug)
            print(f"[NEW SESSION] {slug} | Total: {len(SESSION_APPS)}")
        app = SESSION_APPS[slug]
    app.touch()
    return app


def close_session(slug: str):
    with SESSION_LOCK:
        app = SESSION_APPS.pop(slug, None)
    if app:
        app.close()


# ── Idle session reaper (OFF by default) ────────────────────────────────────
# Sessions are permanent: nothing in this app ever tears down a session
# automatically. is_running/is_monitoring sessions are never touched here no
# matter how long they run or how much silence there is — silence is handled
# by the VAD producing no segments, not by touching the session. This reaper
# only ever considers sessions that are STOPPED (recognition off) and, even
# then, only if you explicitly opt in via IDLE_SESSION_TIMEOUT_SECONDS — set
# to 0 (default), it's fully disabled. Set a value only if you specifically
# want stopped-and-abandoned sessions to free their in-memory app object
# after N idle seconds (their settings.json is untouched either way).
IDLE_SESSION_TIMEOUT_SECONDS = int(os.environ.get("IDLE_SESSION_TIMEOUT_SECONDS", "0"))
_IDLE_REAPER_INTERVAL_SECONDS = 300


def _idle_reaper_loop():
    if IDLE_SESSION_TIMEOUT_SECONDS <= 0:
        return  # disabled — sessions live forever until explicitly closed
    while True:
        time.sleep(_IDLE_REAPER_INTERVAL_SECONDS)
        now = time.time()
        stale = []
        with SESSION_LOCK:
            for slug, app in SESSION_APPS.items():
                if app.is_running or app.is_monitoring:
                    continue  # never reap an actively-listening session
                if now - app.last_active > IDLE_SESSION_TIMEOUT_SECONDS:
                    stale.append(slug)
        for slug in stale:
            print(
                f"[IDLE REAPER] Closing '{slug}' after "
                f"{IDLE_SESSION_TIMEOUT_SECONDS}s stopped + idle"
            )
            close_session(slug)


threading.Thread(target=_idle_reaper_loop, daemon=True).start()


# ── UI ────────────────────────────────────────────────────────────────────────
def create_ui(args):  # noqa: C901  (complex but intentional)
    with gr.Blocks(title="Voice Translator", analytics_enabled=False) as interface:
        # ── Header ────────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                session_info = gr.Markdown("### Session initializing...")
                gr.Markdown("# 🎤 Voice Translator")
                browser_session_data = gr.HTML(visible=True)
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown(
                        "Visiting this site with no name gives you the **main** "
                        "session, always the same one. Open or create a separate, "
                        "independently-configured session below.",
                        elem_id="session-switcher-hint",
                    )
                    with gr.Row():
                        new_session_name = gr.Textbox(
                            label="Open or create a session by name",
                            placeholder="e.g. khyretos",
                            scale=3,
                        )
                        open_session_btn = gr.Button("↗️ Go", scale=1, size="sm")
                        random_session_btn = gr.Button(
                            "🎲 New", scale=1, size="sm",
                            elem_id="random-session-btn",
                        )
                    session_dropdown = gr.Dropdown(
                        label="Manage Sessions",
                        choices=[],
                        value=None,
                        interactive=True,
                        scale=4,
                    )
                    refresh_sessions_btn = gr.Button("🔄 Refresh", size="sm", scale=1)

        with gr.Row():
            # ── Left column: settings ─────────────────────────────────────────
            with gr.Column(scale=1):
                # Speech recognition
                with gr.Accordion("🗣️ Speech Recognition", open=True):
                    engine_choices = ["vosk", "whisper"]
                    if _MOONSHINE_AVAILABLE:
                        engine_choices.append("moonshine")
                    recognition_engine = gr.Radio(
                        engine_choices,
                        value="vosk",
                        label="Recognition Engine",
                        info="Vosk: offline/fast | Whisper: online/accurate | Moonshine: local/lightweight"
                        + (
                            ""
                            if _MOONSHINE_AVAILABLE
                            else " (Moonshine not installed, run `pip install moonshine-voice`)"
                        ),
                    )

                    with gr.Group(visible=True) as vosk_settings:
                        vosk_models = get_available_models()
                        vosk_choices = (
                            vosk_models
                            if vosk_models
                            else [("❌ No models detected", "")]
                        )
                        vosk_value = vosk_models[0][1] if vosk_models else ""
                        vosk_model_dropdown = gr.Dropdown(
                            choices=vosk_choices,
                            value=vosk_value,
                            label="Vosk Model",
                            info=f"Found {len(vosk_models)} models in vosk_models/"
                            if vosk_models
                            else "⚠️ No models found",
                            interactive=bool(vosk_models),
                        )
                        refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm")
                        display_interim = gr.Checkbox(
                            label="Show interim results",
                            value=True,
                            info="Display partial recognition while speaking (Vosk only)",
                        )

                    with gr.Group(visible=False) as whisper_settings:
                        whisper_host = gr.Textbox(
                            label="Whisper API Host", value="http://localhost:9000"
                        )
                        whisper_api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            placeholder="Leave empty if not required",
                        )
                        whisper_model = gr.Textbox(
                            label="Model", value="whisper-large-v3"
                        )
                        whisper_language = gr.Textbox(label="Language", value="en")
                        with gr.Accordion("Advanced Whisper Parameters", open=False):
                            whisper_temperature = gr.Slider(
                                0.0, 1.0, 0.0, step=0.1, label="Temperature"
                            )
                            whisper_best_of = gr.Slider(
                                1, 10, 5, step=1, label="Best of"
                            )
                            whisper_beam_size = gr.Slider(
                                1, 10, 5, step=1, label="Beam size"
                            )
                            whisper_patience = gr.Slider(
                                0.0, 2.0, 1.0, step=0.1, label="Patience"
                            )
                            whisper_length_penalty = gr.Slider(
                                0.0, 2.0, 1.0, step=0.1, label="Length penalty"
                            )
                            whisper_suppress_tokens = gr.Textbox(
                                value="-1", label="Suppress tokens"
                            )
                            whisper_initial_prompt = gr.Textbox(
                                label="Initial prompt", placeholder="Optional"
                            )
                            whisper_condition_on_previous_text = gr.Checkbox(
                                value=True, label="Condition on previous text"
                            )
                            whisper_temperature_increment_on_fallback = gr.Slider(
                                0.0,
                                1.0,
                                0.2,
                                step=0.1,
                                label="Temperature increment on fallback",
                            )
                            whisper_no_speech_threshold = gr.Slider(
                                0.0, 1.0, 0.6, step=0.1, label="No speech threshold"
                            )
                            whisper_logprob_threshold = gr.Slider(
                                -5.0, 0.0, -1.0, step=0.1, label="Logprob threshold"
                            )
                            whisper_compression_ratio_threshold = gr.Slider(
                                0.0,
                                5.0,
                                2.4,
                                step=0.1,
                                label="Compression ratio threshold",
                            )
                        test_whisper_btn = gr.Button("🔍 Test Connection", size="sm")

                    with gr.Group(visible=False) as moonshine_settings:
                        _moon_status = (
                            "✅ moonshine-voice installed"
                            if _MOONSHINE_AVAILABLE
                            else "⚠️ Install with: `pip install moonshine-voice`"
                        )
                        gr.Markdown(
                            f"{_moon_status}  \n"
                            "Model is **auto-downloaded** on first Start for the selected language."
                        )
                        moonshine_language = gr.Dropdown(
                            choices=MOONSHINE_LANGUAGES,
                            value="en",
                            label="Language",
                            info="Best available model for that language is downloaded automatically.",
                        )
                        moonshine_cache_dir = gr.Textbox(
                            label="Cache Directory",
                            value="moonshine_models",
                            info="Downloaded models are stored here (MOONSHINE_VOICE_CACHE).",
                        )

                # Audio
                with gr.Accordion("🎙️ Audio", open=True):
                    audio_mode = gr.Radio(
                        [
                            ("Server Audio Device", "hardware"),
                            ("Browser Microphone", "browser_mic"),
                            ("Browser Tab / System Audio", "browser_display"),
                        ],
                        value="hardware",
                        label="Audio Source",
                        info=(
                            "Server Device: any input on the host — including a virtual/"
                            "loopback device if you route another app's output through one "
                            "(see AUDIO_SOURCES.md, e.g. for Discord's desktop app) | "
                            "Browser Microphone: this device's mic | Browser Tab/System Audio: "
                            "share a tab or your screen with 'share audio' checked (e.g. "
                            "Discord in a browser tab, or Windows/ChromeOS system audio)"
                        ),
                    )

                    with gr.Group(visible=True) as hardware_group:
                        microphones = get_microphones()
                        mic_dropdown = gr.Dropdown(
                            choices=microphones,
                            label="Input Device",
                            value=microphones[0][1] if microphones else None,
                            interactive=True,
                            info=(
                                "Lists every input-capable device the server can see — "
                                "physical mics AND any virtual/loopback device configured "
                                "on the host (e.g. a PulseAudio monitor source or VB-Audio "
                                "Cable). See AUDIO_SOURCES.md to route another app's audio "
                                "into one of those."
                            ),
                        )
                        with gr.Row():
                            refresh_mic_btn = gr.Button("🔄 Refresh", size="sm")
                            test_mic_btn = gr.Button(
                                "🎙️ Test Mic", size="sm", variant="secondary"
                            )
                            stop_test_mic_btn = gr.Button(
                                "⏹ Stop Test", size="sm", visible=False
                            )
                        mic_test_status = gr.Textbox(
                            label="",
                            value="",
                            interactive=False,
                            lines=1,
                            visible=False,
                        )

                    with gr.Group(visible=False) as browser_group:
                        browser_source_hint = gr.Markdown(
                            "Uses this device's microphone.", visible=True
                        )
                        browser_status = gr.Textbox(
                            label="Browser Stream Status",
                            value="Not started",
                            interactive=False,
                            elem_id="browser-status",
                        )
                        with gr.Row():
                            browser_test_mic_btn = gr.Button(
                                "🎙️ Test Mic", size="sm", variant="secondary"
                            )
                            browser_stop_test_mic_btn = gr.Button(
                                "⏹ Stop Test", size="sm", visible=False
                            )

                    # ── VAD + Level meter (always visible, works in test mode too) ──
                    with gr.Group():
                        _vad_backend = (
                            "WebRTC VAD"
                            if _WRTCVAD_AVAILABLE
                            else "RMS energy (install webrtcvad for better accuracy)"
                        )
                        gr.Markdown(f"**Voice Detection** · {_vad_backend}")

                        vad_threshold = gr.Slider(
                            minimum=-60,
                            maximum=0,
                            value=-30,
                            step=1,
                            label="Detection threshold (dB)",
                            info="Set where the orange line sits on the meter above. "
                            "-60 = detects a whisper · -30 = normal speech · -10 = loud speech only.",
                            elem_id="vad-threshold-slider",
                        )
                        vad_end_silence_ms = gr.Slider(
                            minimum=0,
                            maximum=2000,
                            value=300,
                            step=50,
                            label="End-of-speech pause (ms)",
                            info="How long silence must last before a segment is sent to "
                            "Whisper/Moonshine. Default raised from 80ms → 300ms: 80ms is "
                            "shorter than a normal mid-sentence breath, so it was fragmenting "
                            "speech into tiny clips — and very short/ambiguous clips are "
                            "exactly what triggers Whisper to hallucinate fillers like "
                            "'thank you'. Raise further (500–800ms) if it's still cutting "
                            "off mid-sentence; lower it if replies feel laggy.",
                        )
                    # After vad_end_silence_ms slider, add:
                    with gr.Group():
                        gr.Markdown(
                            "**Noise filter · clicks, keyboard, background hum**"
                        )
                        noise_filter_threshold = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0,
                            step=0.05,
                            label="Noise filter strength",
                            info="0 = off · 0.3 = light (removes obvious clicks) · 0.7 = medium "
                            "(spectral subtraction + transient suppression) · 1.0 = aggressive "
                            "(maximum cleaning, may soften quiet speech). Changes take effect immediately.",
                        )
                        # Level meter — shows EXACTLY what the VAD sees (no smoothing).
                        # Orange vertical line = detection threshold.
                        # White vertical line  = peak hold (2 s decay).
                        # Bar turns RED when above threshold (VAD is active).
                        gr.HTML("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
  <div id="vad-led" style="width: 14px; height: 14px; border-radius: 50%; background: #888; box-shadow: 0 0 2px #000; transition: background 0.05s;"></div>
  <span style="font-size: 12px; color: #ccc;">VAD active</span>
</div>
<div style="margin:4px 0 2px; font-size:11px; color:#aaa; display:flex; justify-content:space-between; user-select:none;">
  <span>-60</span><span>-45</span><span>-30</span><span>-15</span><span>0 dB</span>
</div>
<div style="position:relative; background:#1a1a1a; height:20px; border-radius:4px;
            overflow:visible; border:1px solid #444; margin-bottom:2px;">
  <!-- Live level bar (no smoothing — matches VAD exactly) -->
  <div id="volume-meter-bar"
       style="height:100%; width:0%; border-radius:3px; background:#44cc44; transition:none;">
  </div>
  <!-- Threshold line (orange) -->
  <div id="volume-meter-threshold-line"
       style="position:absolute; top:-5px; bottom:-5px; width:3px; background:#f5a623;
              border-radius:2px; left:50%; pointer-events:none; z-index:10;
              box-shadow:0 0 5px #f5a623;" title="Detection threshold">
  </div>
  <!-- Peak-hold line (white) — holds max level for 2 s -->
  <div id="volume-meter-peak-line"
       style="position:absolute; top:2px; bottom:2px; width:2px; background:#fff;
              border-radius:1px; left:0%; pointer-events:none; z-index:9;
              opacity:0.8;" title="Peak hold">
  </div>
</div>
<div style="display:flex; justify-content:space-between; font-size:11px; color:#888; margin-top:1px;">
  <span id="volume-meter-db-label">— dB</span>
  <span style="color:#f5a623; font-weight:bold;" id="volume-meter-threshold-label">threshold: — dB</span>
</div>
<div style="font-size:10px; color:#666; margin-top:2px;">
  Bar = live level (no smoothing) · White line = peak hold (2 s) · Red bar = above threshold
</div>""")

                # Translation
                with gr.Accordion("🌐 Translation", open=True):
                    enable_translation = gr.Checkbox(
                        label="Enable Translation",
                        value=False,
                        info="Toggle translation on/off",
                    )
                    with gr.Group(visible=False) as translation_group:
                        translation_mode = gr.Radio(
                            ["argos", "ai", "libretranslate", "whisper"],
                            value="argos",
                            label="Translation Engine",
                        )
                        with gr.Row():
                            source_lang = gr.Textbox(
                                label="From", value="en", scale=1, visible=False
                            )
                            target_lang = gr.Textbox(
                                label="To", value="es", scale=1, visible=False
                            )

                        with gr.Group(visible=False) as ai_settings:
                            ai_host = gr.Textbox(
                                label="API Host",
                                value="http://localhost:11434",
                                info="Base host only — the /v1/chat/completions path is "
                                "added automatically. Was hardcoded to guess a "
                                "'/v3/...' path here before, which isn't a real endpoint "
                                "on any standard OpenAI-compatible server; fixed to /v1.",
                            )
                            ai_api_key = gr.Textbox(
                                label="API Key",
                                type="password",
                                placeholder="Leave empty for Ollama",
                            )
                            ai_model = gr.Textbox(label="Model", value="llama3.2")

                            with gr.Accordion(
                                "🛠️ Advanced: custom prompt & request shape (developer)",
                                open=False,
                            ):
                                gr.Markdown(
                                    "Full details and examples in **AI_TRANSLATION.md**. "
                                    "Leave any of these blank to use the sensible "
                                    "OpenAI/Ollama-compatible default."
                                )
                                ai_translation_prompt_template = gr.Textbox(
                                    label="Prompt template",
                                    lines=3,
                                    placeholder=tmod.DEFAULT_AI_PROMPT_TEMPLATE,
                                    info="Placeholders: {source_lang} {target_lang} {text}",
                                )
                                ai_endpoint_url = gr.Textbox(
                                    label="Full endpoint URL override",
                                    placeholder="e.g. https://api.example.com/v1/chat/completions",
                                    info="If set, used exactly as-is instead of deriving "
                                    "from API Host above — for APIs whose chat-completion "
                                    "path isn't /v1/chat/completions at all.",
                                )
                                ai_request_body_template = gr.Textbox(
                                    label="Request body template (JSON)",
                                    lines=8,
                                    placeholder=tmod.DEFAULT_AI_REQUEST_BODY_TEMPLATE,
                                    info="__MODEL__ and __PROMPT__ are substituted as "
                                    "JSON-escaped string content. Use this to match a "
                                    "non-OpenAI-compatible request shape entirely.",
                                )
                                ai_response_text_path = gr.Textbox(
                                    label="Response text path",
                                    placeholder=tmod.DEFAULT_AI_RESPONSE_TEXT_PATH,
                                    info="Dot-path to the translated text in the JSON "
                                    "response, e.g. 'choices.0.message.content' or "
                                    "'result.translation'. Numeric segments index lists.",
                                )
                                ai_advanced_reset_btn = gr.Button(
                                    "Reset these 4 fields to default", size="sm"
                                )

                        with gr.Group(visible=False) as libretranslate_settings:
                            libretranslate_host = gr.Textbox(
                                label="Host", value="http://localhost:5000"
                            )
                            libretranslate_api_key = gr.Textbox(
                                label="API Key", type="password"
                            )

                        with gr.Group(visible=False) as whisper_translate_settings:
                            whisper_trans_host = gr.Textbox(
                                label="Whisper API Host", value="http://localhost:9000"
                            )
                            whisper_trans_api_key = gr.Textbox(
                                label="API Key", type="password"
                            )
                            whisper_trans_model = gr.Textbox(
                                label="Model", value="whisper-large-v3"
                            )
                            with gr.Accordion(
                                "Advanced Whisper Translate Parameters", open=False
                            ):
                                whisper_trans_temperature = gr.Slider(
                                    0.0, 1.0, 0.0, step=0.1, label="Temperature"
                                )
                                whisper_trans_best_of = gr.Slider(
                                    1, 10, 5, step=1, label="Best of"
                                )
                                whisper_trans_beam_size = gr.Slider(
                                    1, 10, 5, step=1, label="Beam size"
                                )
                                whisper_trans_patience = gr.Slider(
                                    0.0, 2.0, 1.0, step=0.1, label="Patience"
                                )
                                whisper_trans_length_penalty = gr.Slider(
                                    0.0, 2.0, 1.0, step=0.1, label="Length penalty"
                                )
                                whisper_trans_suppress_tokens = gr.Textbox(
                                    value="-1", label="Suppress tokens"
                                )
                                whisper_trans_initial_prompt = gr.Textbox(
                                    label="Initial prompt", placeholder="Optional"
                                )
                                whisper_trans_condition_on_previous_text = gr.Checkbox(
                                    value=True, label="Condition on previous text"
                                )
                                whisper_trans_temperature_increment_on_fallback = (
                                    gr.Slider(
                                        0.0,
                                        1.0,
                                        0.2,
                                        step=0.1,
                                        label="Temperature increment on fallback",
                                    )
                                )
                                whisper_trans_no_speech_threshold = gr.Slider(
                                    0.0, 1.0, 0.6, step=0.1, label="No speech threshold"
                                )
                                whisper_trans_logprob_threshold = gr.Slider(
                                    -5.0, 0.0, -1.0, step=0.1, label="Logprob threshold"
                                )
                                whisper_trans_compression_ratio_threshold = gr.Slider(
                                    0.0,
                                    5.0,
                                    2.4,
                                    step=0.1,
                                    label="Compression ratio threshold",
                                )
                            gr.Markdown(
                                "*Translates audio directly to English using Whisper*"
                            )

                        with gr.Group(visible=True) as argos_settings:
                            argos_source = gr.Textbox(
                                label="Source Language Code",
                                value="en",
                                info="2-letter code (en, es, fr…)",
                            )
                            argos_target = gr.Textbox(
                                label="Target Language Code", value="es"
                            )
                            if not ARGOS_AVAILABLE:
                                gr.Markdown(
                                    "⚠️ *Install argostranslate: pip install argostranslate*"
                                )
                            else:
                                installed_pairs = (
                                    argostranslate.translate.get_installed_languages()
                                )
                                if len(installed_pairs) < 2:
                                    gr.Markdown(
                                        "⚠️ **No language pairs installed.** Run `download_argos_models.py`."
                                    )
                                else:
                                    gr.Markdown("✅ Argos is ready.")

            # ── Right column: display + style ─────────────────────────────────
            with gr.Column(scale=1):
                with gr.Row():
                    start_btn = gr.Button("▶️ Start", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop")

                status_text = gr.Textbox(label="Status", lines=2)

                gr.Markdown("### 📺 Display")
                popout_url = gr.Textbox(label="Popout URL", interactive=False)
                with gr.Group():
                    gr.Markdown("For a custom ID enter the string and press Enter")
                    custom_popout_id = gr.Textbox(
                        label="Custom Popout ID",
                        placeholder="Enter custom ID or leave empty for random",
                        scale=3,
                    )
                    random_btn = gr.Button("🎲 Random", scale=1, size="sm")

                display_html = gr.HTML(label="Display", value="<div>Loading...</div>")

                with gr.Accordion("Outputs", open=False):
                    recognized_output = gr.Textbox(
                        label="Recognized",
                        elem_id="recognized-output-text",
                    )
                    translated_output = gr.Textbox(
                        label="Translated",
                        elem_id="translated-output-text",
                    )

                with gr.Accordion("🎨 Display Style", open=False):
                    custom_fonts = get_available_fonts()
                    font_choices = SYSTEM_FONTS + custom_fonts
                    font_selector = gr.Dropdown(
                        choices=font_choices,
                        value="Arial",
                        label="Font Family",
                        info="Select a system font or a custom font from fonts/",
                    )
                    refresh_fonts_btn = gr.Button("🔄 Refresh Fonts", size="sm")
                    recognized_font_size = gr.Slider(
                        12, 120, 48, step=2, label="Main text size"
                    )
                    translated_font_size = gr.Slider(
                        12, 120, 32, step=2, label="Translation size"
                    )
                    with gr.Row():
                        recognized_color = gr.ColorPicker(
                            label="Main color", value="#FFFFFF"
                        )
                        translated_color = gr.ColorPicker(
                            label="Translation color", value="#CCCCCC"
                        )
                        background_color = gr.ColorPicker(
                            label="Background", value="#000000"
                        )
                    text_alignment = gr.Radio(
                        ["left", "center", "right"], value="center", label="Align"
                    )
                    translation_position = gr.Radio(
                        ["before", "after"], value="after", label="Translation position"
                    )
                    fade_timeout = gr.Slider(
                        0.1, 10.0, 5.0, step=0.1, label="Fade Timeout (seconds)"
                    )

                    with gr.Group():
                        gr.Markdown("**Recognized Text Outline**")
                        with gr.Row():
                            with gr.Column(scale=3):
                                outline_width = gr.Slider(
                                    0, 10, 0, step=1, label="Outline width (px)"
                                )
                            with gr.Column(scale=1):
                                outline_color = gr.ColorPicker(
                                    label="Outline color", value="#000000"
                                )

                    with gr.Group():
                        gr.Markdown("**Translated Text Outline**")
                        with gr.Row():
                            with gr.Column(scale=3):
                                translated_outline_width = gr.Slider(
                                    0, 10, 0, step=1, label="Outline width (px)"
                                )
                            with gr.Column(scale=1):
                                translated_outline_color = gr.ColorPicker(
                                    label="Outline color", value="#000000"
                                )

                with gr.Accordion("📺 Subtitle Timing", open=False):
                    gr.Markdown(
                        "**Instant** — subtitles appear immediately and fade after the timeout.  \n"
                        "**Buffered** — sentences queue up and are shown as chunks, "
                        "each held long enough to read at the chosen CPS speed. "
                        "Good for fast speakers or Whisper which sends whole phrases."
                    )
                    subtitle_mode = gr.Radio(
                        choices=[
                            ("Instant", "instant"),
                            ("Buffered / paced", "buffered"),
                        ],
                        value="instant",
                        label="Display mode",
                    )
                    with gr.Group(visible=False) as subtitle_buffered_group:
                        subtitle_cps = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=21,
                            step=1,
                            label="Reading speed (characters per second)",
                            info="Netflix standard ≈ 17 CPS for most languages. "
                            "Raise for faster readers, lower to give more time per chunk.",
                        )
                        subtitle_max_lines = gr.Slider(
                            minimum=1,
                            maximum=4,
                            value=2,
                            step=1,
                            label="Sentences per chunk",
                            info="How many sentences to show at once before advancing.",
                        )

                with gr.Row():
                    reset_defaults_btn = gr.Button(
                        "🔄 Reset to Defaults", variant="secondary", size="sm"
                    )

        log_output = gr.Textbox(label="Log", lines=6)

        # ── Event handlers ────────────────────────────────────────────────────

        def _set(key: str, persist: bool = True):
            """Return a function that sets app.settings[key] = value (and persists)."""

            def updater(value, request: gr.Request):
                app = get_or_create_app(get_slug(request))
                app.settings[key] = value
                if persist:
                    persist_settings(app.slug, app.settings)

            return updater

        def _set_vad(key: str):
            """Set VAD setting AND push to running VAD immediately."""

            def updater(value, request: gr.Request):
                app = get_or_create_app(get_slug(request))
                app.settings[key] = value
                app.apply_vad_settings()
                persist_settings(app.slug, app.settings)

            return updater

        def _set_subtitle(key: str):
            """Set subtitle setting AND apply to SubtitleManager immediately."""

            def updater(value, request: gr.Request):
                app = get_or_create_app(get_slug(request))
                app.settings[key] = value
                app.apply_subtitle_settings()
                persist_settings(app.slug, app.settings)

            return updater

        def _set_fade(value, request: gr.Request):
            """Fade timeout applies to both settings and subtitle manager."""
            app = get_or_create_app(get_slug(request))
            app.settings["fade_timeout"] = value
            app.apply_subtitle_settings()
            persist_settings(app.slug, app.settings)

        # Subtitle mode toggle
        def update_subtitle_mode(mode, request: gr.Request):
            app = get_or_create_app(get_slug(request))
            app.settings["subtitle_mode"] = mode
            app.apply_subtitle_settings()
            persist_settings(app.slug, app.settings)
            return gr.update(visible=(mode == "buffered"))

        # Mic test handlers
        def start_mic_test(request: gr.Request):
            app = get_or_create_app(get_slug(request))
            msg = app.start_mic_monitor()
            return (
                msg,
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
            )

        def stop_mic_test(request: gr.Request):
            app = get_or_create_app(get_slug(request))
            app.stop_mic_monitor()
            return (
                "",
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            )

        def start_browser_mic_test(request: gr.Request):
            return "", gr.update(visible=True), gr.update(visible=False)

        def stop_browser_mic_test(request: gr.Request):
            return "", gr.update(visible=False), gr.update(visible=True)

        def update_recognition_engine(engine, request: gr.Request):
            app = get_or_create_app(get_slug(request))
            app.settings["recognition_engine"] = engine
            persist_settings(app.slug, app.settings)
            return {
                vosk_settings: gr.update(visible=(engine == "vosk")),
                whisper_settings: gr.update(visible=(engine == "whisper")),
                moonshine_settings: gr.update(visible=(engine == "moonshine")),
            }

        def update_audio_mode(mode, request: gr.Request):
            app = get_or_create_app(get_slug(request))
            app.settings["audio_mode"] = mode
            persist_settings(app.slug, app.settings)
            if mode == "browser_display":
                hint = (
                    "**Tab/System Audio**: pressing Start (or Test Mic) will ask "
                    "your browser to pick a tab, window, or screen — check **'Share "
                    "audio'** in that picker. What actually gets captured (just that "
                    "tab vs. the whole system) depends on your browser/OS — see "
                    "AUDIO_SOURCES.md. This is how to translate e.g. a Discord *web* "
                    "tab, or (Windows/ChromeOS) whole-system audio."
                )
            else:
                hint = "Uses this device's microphone."
            return {
                hardware_group: gr.update(visible=(mode == "hardware")),
                browser_group: gr.update(
                    visible=(mode in ("browser_mic", "browser_display"))
                ),
                browser_source_hint: gr.update(value=hint),
            }

        def update_translation_mode(mode, request: gr.Request):
            app = get_or_create_app(get_slug(request))
            app.settings["translation_mode"] = mode
            persist_settings(app.slug, app.settings)
            return {
                ai_settings: gr.update(visible=(mode == "ai")),
                libretranslate_settings: gr.update(visible=(mode == "libretranslate")),
                whisper_translate_settings: gr.update(visible=(mode == "whisper")),
                argos_settings: gr.update(visible=(mode == "argos")),
                source_lang: gr.update(visible=(mode != "argos")),
                target_lang: gr.update(visible=(mode != "argos")),
            }

        def update_translation_toggle(enabled, request: gr.Request):
            app = get_or_create_app(get_slug(request))
            app.settings["enable_translation"] = enabled
            if not enabled:
                app.subtitles._cur_trans = ""
            persist_settings(app.slug, app.settings)
            return gr.update(visible=enabled)

        def update_font_selector(value, request: gr.Request):
            app = get_or_create_app(get_slug(request))
            custom_fonts = [f[1] for f in get_available_fonts()]
            if value in custom_fonts:
                app.settings["custom_font"] = value
            else:
                app.settings["font_family"] = value
                app.settings["custom_font"] = ""
            persist_settings(app.slug, app.settings)

        def update_custom_popout_id(value, request: gr.Request):
            app = get_or_create_app(get_slug(request))
            if value and value.strip():
                sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", value.strip())
                app.popout_id = sanitized if sanitized else secrets.token_urlsafe(16)
            else:
                app.popout_id = secrets.token_urlsafe(16)
            return (
                f"http://{args.host}:{args.port}/popout/{app.popout_id}",
                app.popout_id,
            )

        def generate_random_popout(value, request: gr.Request):
            app = get_or_create_app(get_slug(request))
            app.popout_id = secrets.token_urlsafe(16)
            return (
                f"http://{args.host}:{args.port}/popout/{app.popout_id}",
                app.popout_id,
            )

        def test_whisper_connection(request: gr.Request):
            app = get_or_create_app(get_slug(request))
            host = app.settings["whisper_host"]
            if not host:
                return "❌ Please enter a Whisper API host URL"
            headers = {}
            if app.settings.get("whisper_api_key"):
                headers["Authorization"] = f"Bearer {app.settings['whisper_api_key']}"
            try:
                r = requests.get(host, headers=headers, timeout=5)
                if r.status_code == 200:
                    return f"✅ Connected to {host}"
                for ep in ["/health", "/models", "/audio/transcriptions"]:
                    try:
                        r2 = requests.get(
                            host.rstrip("/") + ep, headers=headers, timeout=5
                        )
                        if r2.status_code == 200:
                            return f"✅ Connected to {host.rstrip('/')}{ep}"
                    except Exception:
                        continue
                return "⚠️ Whisper API responds but could not verify endpoint."
            except requests.exceptions.ConnectionError:
                return f"❌ Cannot connect to {host}"
            except Exception as exc:
                return f"❌ Error: {exc}"

        def refresh_sessions_list(request: gr.Request):
            current = get_slug(request)
            try:
                r = requests.get(
                    f"http://{args.host}:{args.port}/active_sessions", timeout=2
                )
                if r.status_code == 200:
                    sessions = r.json().get("sessions", [])
                    choices = [("All sessions", "ALL")] + [
                        (s, s) for s in sessions if s != current
                    ]
                    return (
                        gr.update(choices=choices, value=None),
                        gr.update(
                            value=f"### 🎯 Session: `{current}` | Active: {len(sessions)}"
                        ),
                    )
            except Exception:
                pass
            return gr.update(choices=[]), gr.update()

        def close_session_action(selected, request: gr.Request):
            current = get_slug(request)
            if not selected:
                return "No session selected", gr.update()
            if selected == "ALL":
                with SESSION_LOCK:
                    for h, a in list(SESSION_APPS.items()):
                        if h != current:
                            a.deactivate_session()
                            del SESSION_APPS[h]
                total = len(SESSION_APPS)
                msg = "All other sessions closed"
            else:
                with SESSION_LOCK:
                    a = SESSION_APPS.get(selected)
                    if a:
                        a.deactivate_session()
                        del SESSION_APPS[selected]
                total = len(SESSION_APPS)
                msg = f"Session {selected} closed"
            return msg, gr.update(
                value=f"### 🎯 Session: `{current}` | Active: {total}"
            )

        def start_rec(request: gr.Request):
            app = get_or_create_app(get_slug(request))
            model = (
                app.settings["vosk_model"]
                if app.settings["recognition_engine"] == "vosk"
                else None
            )
            return app.start_recognition(model, app.settings.get("microphone"))

        def stop_rec(request: gr.Request):
            return get_or_create_app(get_slug(request)).stop_recognition()

        def update_display(request: gr.Request):
            return get_or_create_app(get_slug(request)).get_current_display()

        def update_logs(request: gr.Request):
            return get_or_create_app(get_slug(request)).update_logs()

        def cleanup_user_data(request: gr.Request):
            # Tabs closing / reloading no longer destroys the session — a
            # session now lives until the user explicitly closes it (via the
            # "Manage Sessions" dropdown) or the idle reaper reclaims it after
            # IDLE_SESSION_TIMEOUT_SECONDS of inactivity with nothing running.
            # This is what makes a reload/reconnect re-attach cleanly instead
            # of losing state — which is also what fixed the "sometimes it
            # just refreshes" symptom: it used to tear the whole session down.
            slug = get_slug(request)
            app = SESSION_APPS.get(slug)
            if app:
                app.touch()

        def handle_ui_load(request: gr.Request):
            slug = get_slug(request)
            app = get_or_create_app(slug)

            # Ensure microphone and model are set to valid values
            models = get_available_models()
            mics = get_microphones()
            if models and not app.settings["vosk_model"]:
                app.settings["vosk_model"] = models[0][1]
            if mics and app.settings.get("microphone") not in [m[1] for m in mics]:
                app.settings["microphone"] = mics[0][1]

            s = app.settings
            font_value = s.get("custom_font") or s["font_family"]
            session_html = (
                f'<div id="session-data" data-session="{slug}" '
                f'data-ws-path="/ws/{slug}" '
                f'data-popout="{app.popout_id}">'
                f"<b>Session: {slug} — bookmark this URL to return to it</b></div>"
            )

            return {
                session_info: f"### 🎯 Session: `{slug}` | Active: {len(SESSION_APPS)}",
                popout_url: f"http://{args.host}:{args.port}/popout/{app.popout_id}",
                vosk_model_dropdown: s["vosk_model"],
                mic_dropdown: s.get("microphone"),
                recognition_engine: s["recognition_engine"],
                audio_mode: s["audio_mode"],
                enable_translation: s["enable_translation"],
                display_interim: s["display_interim"],
                translation_mode: s["translation_mode"],
                source_lang: s["source_language"],
                target_lang: s["target_language"],
                font_selector: font_value,
                recognized_font_size: s["recognized_font_size"],
                translated_font_size: s["translated_font_size"],
                recognized_color: s["recognized_color"],
                translated_color: s["translated_color"],
                background_color: s["background_color"],
                text_alignment: s["text_alignment"],
                translation_position: s["translation_position"],
                whisper_host: s["whisper_host"],
                whisper_api_key: s["whisper_api_key"],
                whisper_model: s["whisper_model"],
                whisper_language: s["whisper_language"],
                whisper_temperature: s["whisper_temperature"],
                whisper_best_of: s["whisper_best_of"],
                whisper_beam_size: s["whisper_beam_size"],
                whisper_patience: s["whisper_patience"],
                whisper_length_penalty: s["whisper_length_penalty"],
                whisper_suppress_tokens: s["whisper_suppress_tokens"],
                whisper_initial_prompt: s["whisper_initial_prompt"],
                whisper_condition_on_previous_text: s[
                    "whisper_condition_on_previous_text"
                ],
                whisper_temperature_increment_on_fallback: s[
                    "whisper_temperature_increment_on_fallback"
                ],
                whisper_no_speech_threshold: s["whisper_no_speech_threshold"],
                whisper_logprob_threshold: s["whisper_logprob_threshold"],
                whisper_compression_ratio_threshold: s[
                    "whisper_compression_ratio_threshold"
                ],
                whisper_trans_host: s["whisper_translate_host"],
                whisper_trans_api_key: s["whisper_translate_api_key"],
                whisper_trans_model: s["whisper_translate_model"],
                whisper_trans_temperature: s["whisper_translate_temperature"],
                whisper_trans_best_of: s["whisper_translate_best_of"],
                whisper_trans_beam_size: s["whisper_translate_beam_size"],
                whisper_trans_patience: s["whisper_translate_patience"],
                whisper_trans_length_penalty: s["whisper_translate_length_penalty"],
                whisper_trans_suppress_tokens: s["whisper_translate_suppress_tokens"],
                whisper_trans_initial_prompt: s["whisper_translate_initial_prompt"],
                whisper_trans_condition_on_previous_text: s[
                    "whisper_translate_condition_on_previous_text"
                ],
                whisper_trans_temperature_increment_on_fallback: s[
                    "whisper_translate_temperature_increment_on_fallback"
                ],
                whisper_trans_no_speech_threshold: s[
                    "whisper_translate_no_speech_threshold"
                ],
                whisper_trans_logprob_threshold: s[
                    "whisper_translate_logprob_threshold"
                ],
                whisper_trans_compression_ratio_threshold: s[
                    "whisper_translate_compression_ratio_threshold"
                ],
                argos_source: s["argos_source_lang"],
                argos_target: s["argos_target_lang"],
                fade_timeout: s["fade_timeout"],
                libretranslate_host: s["libretranslate_host"],
                libretranslate_api_key: s["libretranslate_api_key"],
                ai_host: s["ai_host"],
                ai_api_key: s["ai_api_key"],
                ai_model: s["ai_model"],
                ai_translation_prompt_template: s["ai_translation_prompt_template"],
                ai_endpoint_url: s["ai_endpoint_url"],
                ai_request_body_template: s["ai_request_body_template"],
                ai_response_text_path: s["ai_response_text_path"],
                outline_width: s["outline_width"],
                outline_color: s["outline_color"],
                translated_outline_width: s["translated_outline_width"],
                translated_outline_color: s["translated_outline_color"],
                vad_threshold: _migrate_vad_threshold(s["vad_threshold"]),
                vad_end_silence_ms: s["vad_end_silence_ms"],
                subtitle_mode: s["subtitle_mode"],
                subtitle_cps: s["subtitle_cps"],
                subtitle_max_lines: s["subtitle_max_lines"],
                moonshine_language: s["moonshine_language"],
                moonshine_cache_dir: s["moonshine_cache_dir"],
                browser_session_data: session_html,
                browser_status: "Ready",
                custom_popout_id: app.popout_id,
                session_dropdown: gr.update(choices=[]),
                noise_filter_threshold: s["noise_filter_threshold"],
            }

        def reset_to_defaults(request: gr.Request):
            app = get_or_create_app(get_slug(request))
            # Overwrite with defaults, but keep slug, popout_id, etc. – those are not in DEFAULT_SETTINGS
            app.settings.update(VoiceTranslatorApp.DEFAULT_SETTINGS)
            # Also update subtitle manager and VAD with the new values
            app.apply_subtitle_settings()
            app.apply_vad_settings()
            persist_settings(app.slug, app.settings)

            # Build the same output dictionary as handle_ui_load
            s = app.settings
            return {
                vosk_model_dropdown: s["vosk_model"],
                mic_dropdown: s.get("microphone"),
                recognition_engine: s["recognition_engine"],
                audio_mode: s["audio_mode"],
                enable_translation: s["enable_translation"],
                display_interim: s["display_interim"],
                translation_mode: s["translation_mode"],
                source_lang: s["source_language"],
                target_lang: s["target_language"],
                font_selector: s.get("custom_font") or s["font_family"],
                recognized_font_size: s["recognized_font_size"],
                translated_font_size: s["translated_font_size"],
                recognized_color: s["recognized_color"],
                translated_color: s["translated_color"],
                background_color: s["background_color"],
                text_alignment: s["text_alignment"],
                translation_position: s["translation_position"],
                whisper_host: s["whisper_host"],
                whisper_api_key: s["whisper_api_key"],
                whisper_model: s["whisper_model"],
                whisper_language: s["whisper_language"],
                whisper_temperature: s["whisper_temperature"],
                whisper_best_of: s["whisper_best_of"],
                whisper_beam_size: s["whisper_beam_size"],
                whisper_patience: s["whisper_patience"],
                whisper_length_penalty: s["whisper_length_penalty"],
                whisper_suppress_tokens: s["whisper_suppress_tokens"],
                whisper_initial_prompt: s["whisper_initial_prompt"],
                whisper_condition_on_previous_text: s[
                    "whisper_condition_on_previous_text"
                ],
                whisper_temperature_increment_on_fallback: s[
                    "whisper_temperature_increment_on_fallback"
                ],
                whisper_no_speech_threshold: s["whisper_no_speech_threshold"],
                whisper_logprob_threshold: s["whisper_logprob_threshold"],
                whisper_compression_ratio_threshold: s[
                    "whisper_compression_ratio_threshold"
                ],
                whisper_trans_host: s["whisper_translate_host"],
                whisper_trans_api_key: s["whisper_translate_api_key"],
                whisper_trans_model: s["whisper_translate_model"],
                whisper_trans_temperature: s["whisper_translate_temperature"],
                whisper_trans_best_of: s["whisper_translate_best_of"],
                whisper_trans_beam_size: s["whisper_translate_beam_size"],
                whisper_trans_patience: s["whisper_translate_patience"],
                whisper_trans_length_penalty: s["whisper_translate_length_penalty"],
                whisper_trans_suppress_tokens: s["whisper_translate_suppress_tokens"],
                whisper_trans_initial_prompt: s["whisper_translate_initial_prompt"],
                whisper_trans_condition_on_previous_text: s[
                    "whisper_translate_condition_on_previous_text"
                ],
                whisper_trans_temperature_increment_on_fallback: s[
                    "whisper_translate_temperature_increment_on_fallback"
                ],
                whisper_trans_no_speech_threshold: s[
                    "whisper_translate_no_speech_threshold"
                ],
                whisper_trans_logprob_threshold: s[
                    "whisper_translate_logprob_threshold"
                ],
                whisper_trans_compression_ratio_threshold: s[
                    "whisper_translate_compression_ratio_threshold"
                ],
                argos_source: s["argos_source_lang"],
                argos_target: s["argos_target_lang"],
                fade_timeout: s["fade_timeout"],
                libretranslate_host: s["libretranslate_host"],
                libretranslate_api_key: s["libretranslate_api_key"],
                ai_host: s["ai_host"],
                ai_api_key: s["ai_api_key"],
                ai_model: s["ai_model"],
                ai_translation_prompt_template: s["ai_translation_prompt_template"],
                ai_endpoint_url: s["ai_endpoint_url"],
                ai_request_body_template: s["ai_request_body_template"],
                ai_response_text_path: s["ai_response_text_path"],
                outline_width: s["outline_width"],
                outline_color: s["outline_color"],
                translated_outline_width: s["translated_outline_width"],
                translated_outline_color: s["translated_outline_color"],
                vad_threshold: s["vad_threshold"],
                vad_end_silence_ms: s["vad_end_silence_ms"],
                subtitle_mode: s["subtitle_mode"],
                subtitle_cps: s["subtitle_cps"],
                subtitle_max_lines: s["subtitle_max_lines"],
                moonshine_language: s["moonshine_language"],
                moonshine_cache_dir: s["moonshine_cache_dir"],
                noise_filter_threshold: s["noise_filter_threshold"],
            }

        # ── Wire events ───────────────────────────────────────────────────────

        recognition_engine.change(
            update_recognition_engine,
            [recognition_engine],
            [vosk_settings, whisper_settings, moonshine_settings],
        )
        audio_mode.change(
            update_audio_mode,
            [audio_mode],
            [hardware_group, browser_group, browser_source_hint],
        )
        translation_mode.change(
            update_translation_mode,
            [translation_mode],
            [
                ai_settings,
                libretranslate_settings,
                whisper_translate_settings,
                argos_settings,
                source_lang,
                target_lang,
            ],
        )
        enable_translation.change(
            update_translation_toggle, [enable_translation], [translation_group]
        )
        display_interim.change(_set("display_interim"), [display_interim])

        # Language / translation
        source_lang.change(_set("source_language"), [source_lang])
        target_lang.change(_set("target_language"), [target_lang])
        argos_source.change(_set("argos_source_lang"), [argos_source])
        argos_target.change(_set("argos_target_lang"), [argos_target])
        ai_host.change(_set("ai_host"), [ai_host])
        ai_api_key.change(_set("ai_api_key"), [ai_api_key])
        ai_model.change(_set("ai_model"), [ai_model])
        ai_translation_prompt_template.change(
            _set("ai_translation_prompt_template"), [ai_translation_prompt_template]
        )
        ai_endpoint_url.change(_set("ai_endpoint_url"), [ai_endpoint_url])
        ai_request_body_template.change(
            _set("ai_request_body_template"), [ai_request_body_template]
        )
        ai_response_text_path.change(
            _set("ai_response_text_path"), [ai_response_text_path]
        )

        def reset_ai_advanced(request: gr.Request):
            app = get_or_create_app(get_slug(request))
            for key in (
                "ai_translation_prompt_template",
                "ai_endpoint_url",
                "ai_request_body_template",
                "ai_response_text_path",
            ):
                app.settings[key] = ""
            persist_settings(app.slug, app.settings)
            return "", "", "", ""

        ai_advanced_reset_btn.click(
            reset_ai_advanced,
            outputs=[
                ai_translation_prompt_template,
                ai_endpoint_url,
                ai_request_body_template,
                ai_response_text_path,
            ],
        )
        libretranslate_host.change(_set("libretranslate_host"), [libretranslate_host])
        libretranslate_api_key.change(
            _set("libretranslate_api_key"), [libretranslate_api_key]
        )

        # Whisper recognition
        whisper_host.change(_set("whisper_host"), [whisper_host])
        whisper_api_key.change(_set("whisper_api_key"), [whisper_api_key])
        whisper_model.change(_set("whisper_model"), [whisper_model])
        whisper_language.change(_set("whisper_language"), [whisper_language])
        whisper_temperature.change(_set("whisper_temperature"), [whisper_temperature])
        whisper_best_of.change(_set("whisper_best_of"), [whisper_best_of])
        whisper_beam_size.change(_set("whisper_beam_size"), [whisper_beam_size])
        whisper_patience.change(_set("whisper_patience"), [whisper_patience])
        whisper_length_penalty.change(
            _set("whisper_length_penalty"), [whisper_length_penalty]
        )
        whisper_suppress_tokens.change(
            _set("whisper_suppress_tokens"), [whisper_suppress_tokens]
        )
        whisper_initial_prompt.change(
            _set("whisper_initial_prompt"), [whisper_initial_prompt]
        )
        whisper_condition_on_previous_text.change(
            _set("whisper_condition_on_previous_text"),
            [whisper_condition_on_previous_text],
        )
        whisper_temperature_increment_on_fallback.change(
            _set("whisper_temperature_increment_on_fallback"),
            [whisper_temperature_increment_on_fallback],
        )
        whisper_no_speech_threshold.change(
            _set("whisper_no_speech_threshold"), [whisper_no_speech_threshold]
        )
        whisper_logprob_threshold.change(
            _set("whisper_logprob_threshold"), [whisper_logprob_threshold]
        )
        whisper_compression_ratio_threshold.change(
            _set("whisper_compression_ratio_threshold"),
            [whisper_compression_ratio_threshold],
        )

        # Whisper translate
        whisper_trans_host.change(_set("whisper_translate_host"), [whisper_trans_host])
        whisper_trans_api_key.change(
            _set("whisper_translate_api_key"), [whisper_trans_api_key]
        )
        whisper_trans_model.change(
            _set("whisper_translate_model"), [whisper_trans_model]
        )
        whisper_trans_temperature.change(
            _set("whisper_translate_temperature"), [whisper_trans_temperature]
        )
        whisper_trans_best_of.change(
            _set("whisper_translate_best_of"), [whisper_trans_best_of]
        )
        whisper_trans_beam_size.change(
            _set("whisper_translate_beam_size"), [whisper_trans_beam_size]
        )
        whisper_trans_patience.change(
            _set("whisper_translate_patience"), [whisper_trans_patience]
        )
        whisper_trans_length_penalty.change(
            _set("whisper_translate_length_penalty"), [whisper_trans_length_penalty]
        )
        whisper_trans_suppress_tokens.change(
            _set("whisper_translate_suppress_tokens"), [whisper_trans_suppress_tokens]
        )
        whisper_trans_initial_prompt.change(
            _set("whisper_translate_initial_prompt"), [whisper_trans_initial_prompt]
        )
        whisper_trans_condition_on_previous_text.change(
            _set("whisper_translate_condition_on_previous_text"),
            [whisper_trans_condition_on_previous_text],
        )
        whisper_trans_temperature_increment_on_fallback.change(
            _set("whisper_translate_temperature_increment_on_fallback"),
            [whisper_trans_temperature_increment_on_fallback],
        )
        whisper_trans_no_speech_threshold.change(
            _set("whisper_translate_no_speech_threshold"),
            [whisper_trans_no_speech_threshold],
        )
        whisper_trans_logprob_threshold.change(
            _set("whisper_translate_logprob_threshold"),
            [whisper_trans_logprob_threshold],
        )
        whisper_trans_compression_ratio_threshold.change(
            _set("whisper_translate_compression_ratio_threshold"),
            [whisper_trans_compression_ratio_threshold],
        )

        # Display style
        font_selector.change(update_font_selector, [font_selector])
        recognized_font_size.change(
            _set("recognized_font_size"), [recognized_font_size]
        )
        translated_font_size.change(
            _set("translated_font_size"), [translated_font_size]
        )
        recognized_color.change(_set("recognized_color"), [recognized_color])
        translated_color.change(_set("translated_color"), [translated_color])
        background_color.change(_set("background_color"), [background_color])
        text_alignment.change(_set("text_alignment"), [text_alignment])
        translation_position.change(
            _set("translation_position"), [translation_position]
        )
        fade_timeout.change(_set("fade_timeout"), [fade_timeout])
        outline_width.change(_set("outline_width"), [outline_width])
        outline_color.change(_set("outline_color"), [outline_color])
        translated_outline_width.change(
            _set("translated_outline_width"), [translated_outline_width]
        )
        translated_outline_color.change(
            _set("translated_outline_color"), [translated_outline_color]
        )

        # Fade — also syncs subtitle manager
        fade_timeout.change(_set_fade, [fade_timeout])

        # Audio / mic
        mic_dropdown.change(_set("microphone"), [mic_dropdown])
        vosk_model_dropdown.change(_set("vosk_model"), [vosk_model_dropdown])

        # VAD — dynamic, no restart needed
        vad_threshold.change(_set_vad("vad_threshold"), [vad_threshold])
        vad_end_silence_ms.change(_set_vad("vad_end_silence_ms"), [vad_end_silence_ms])
        noise_filter_threshold.change(
            _set_vad("noise_filter_threshold"), [noise_filter_threshold]
        )

        # Mic test
        test_mic_btn.click(
            start_mic_test,
            outputs=[mic_test_status, stop_test_mic_btn, test_mic_btn, mic_test_status],
            js="startHwLevelPolling",
        )
        stop_test_mic_btn.click(
            stop_mic_test,
            outputs=[mic_test_status, stop_test_mic_btn, test_mic_btn, mic_test_status],
            js="stopHwLevelPolling",
        )

        browser_test_mic_btn.click(
            start_browser_mic_test,
            outputs=[browser_status, browser_stop_test_mic_btn, browser_test_mic_btn],
            js="startBrowserMicTest",
        )
        browser_stop_test_mic_btn.click(
            stop_browser_mic_test,
            outputs=[browser_status, browser_stop_test_mic_btn, browser_test_mic_btn],
            js="stopBrowserMicTest",
        )

        # Subtitle timing
        subtitle_mode.change(
            update_subtitle_mode, [subtitle_mode], [subtitle_buffered_group]
        )
        subtitle_cps.change(_set_subtitle("subtitle_cps"), [subtitle_cps])
        subtitle_max_lines.change(
            _set_subtitle("subtitle_max_lines"), [subtitle_max_lines]
        )

        # Moonshine
        moonshine_language.change(_set("moonshine_language"), [moonshine_language])
        moonshine_cache_dir.change(_set("moonshine_cache_dir"), [moonshine_cache_dir])

        # Popout
        custom_popout_id.submit(
            update_custom_popout_id, [custom_popout_id], [popout_url, custom_popout_id]
        )
        random_btn.click(
            generate_random_popout, [random_btn], [popout_url, custom_popout_id]
        )

        # Session management
        _OPEN_SESSION_JS = """
        (name) => {
            name = (name || "").trim();
            if (!name) return;
            const clean = name.replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 64);
            if (!clean) {
                alert("Session name can only contain letters, numbers, - and _");
                return;
            }
            window.location.href = "/?session=" + encodeURIComponent(clean);
        }
        """
        open_session_btn.click(fn=None, inputs=[new_session_name], js=_OPEN_SESSION_JS)
        new_session_name.submit(
            fn=None, inputs=[new_session_name], js=_OPEN_SESSION_JS
        )
        random_session_btn.click(
            fn=None,
            js="""
            () => {
                const rand = Math.random().toString(36).slice(2, 10);
                window.location.href = "/?session=" + rand;
            }
            """,
        )
        refresh_sessions_btn.click(
            refresh_sessions_list, outputs=[session_dropdown, session_info]
        )
        session_dropdown.change(
            close_session_action, [session_dropdown], [status_text, session_info]
        ).then(refresh_sessions_list, outputs=[session_dropdown, session_info])

        # Refresh buttons
        refresh_fonts_btn.click(
            lambda: gr.update(choices=SYSTEM_FONTS + get_available_fonts()),
            outputs=[font_selector],
        )
        refresh_models_btn.click(
            lambda: gr.update(choices=get_available_models()),
            outputs=[vosk_model_dropdown],
        )
        refresh_mic_btn.click(
            lambda: gr.update(choices=get_microphones()), outputs=[mic_dropdown]
        )

        # Whisper test
        test_whisper_btn.click(test_whisper_connection, outputs=[status_text])

        # Start — also starts HW level polling; stops mic test if active
        start_btn.click(
            fn=start_rec, outputs=[status_text], js="startBrowserStreaming"
        ).then(fn=lambda: "Streaming started", outputs=[browser_status]).then(
            fn=lambda: gr.update(visible=False), outputs=[stop_test_mic_btn]
        ).then(fn=None, js="startHwLevelPolling")

        # Stop — stop HW polling too
        stop_btn.click(
            fn=stop_rec, outputs=[status_text], js="stopBrowserStreaming"
        ).then(fn=lambda: "Streaming stopped", outputs=[browser_status]).then(
            fn=None, js="stopHwLevelPolling"
        )

        reset_defaults_btn.click(
            reset_to_defaults,
            outputs=[
                vosk_model_dropdown,
                mic_dropdown,
                recognition_engine,
                audio_mode,
                enable_translation,
                display_interim,
                translation_mode,
                source_lang,
                target_lang,
                font_selector,
                recognized_font_size,
                translated_font_size,
                recognized_color,
                translated_color,
                background_color,
                text_alignment,
                translation_position,
                whisper_host,
                whisper_api_key,
                whisper_model,
                whisper_language,
                whisper_temperature,
                whisper_best_of,
                whisper_beam_size,
                whisper_patience,
                whisper_length_penalty,
                whisper_suppress_tokens,
                whisper_initial_prompt,
                whisper_condition_on_previous_text,
                whisper_temperature_increment_on_fallback,
                whisper_no_speech_threshold,
                whisper_logprob_threshold,
                whisper_compression_ratio_threshold,
                whisper_trans_host,
                whisper_trans_api_key,
                whisper_trans_model,
                whisper_trans_temperature,
                whisper_trans_best_of,
                whisper_trans_beam_size,
                whisper_trans_patience,
                whisper_trans_length_penalty,
                whisper_trans_suppress_tokens,
                whisper_trans_initial_prompt,
                whisper_trans_condition_on_previous_text,
                whisper_trans_temperature_increment_on_fallback,
                whisper_trans_no_speech_threshold,
                whisper_trans_logprob_threshold,
                whisper_trans_compression_ratio_threshold,
                argos_source,
                argos_target,
                fade_timeout,
                libretranslate_host,
                libretranslate_api_key,
                ai_host,
                ai_api_key,
                ai_model,
                ai_translation_prompt_template,
                ai_endpoint_url,
                ai_request_body_template,
                ai_response_text_path,
                outline_width,
                outline_color,
                translated_outline_width,
                translated_outline_color,
                vad_threshold,
                vad_end_silence_ms,
                subtitle_mode,
                subtitle_cps,
                subtitle_max_lines,
                moonshine_language,
                moonshine_cache_dir,
                noise_filter_threshold,
            ],
        )

        # ── Polling timers (reliable Gradio method) ───────────────────────────
        gr.Timer(0.05).tick(
            update_display, outputs=[display_html, recognized_output, translated_output]
        )
        gr.Timer(1.0).tick(update_logs, outputs=[log_output])

        interface.unload(cleanup_user_data)
        interface.load(
            handle_ui_load,
            inputs=None,
            outputs=[
                session_info,
                popout_url,
                vosk_model_dropdown,
                mic_dropdown,
                recognition_engine,
                audio_mode,
                enable_translation,
                display_interim,
                translation_mode,
                source_lang,
                target_lang,
                font_selector,
                recognized_font_size,
                translated_font_size,
                recognized_color,
                translated_color,
                background_color,
                text_alignment,
                translation_position,
                whisper_host,
                whisper_api_key,
                whisper_model,
                whisper_language,
                whisper_temperature,
                whisper_best_of,
                whisper_beam_size,
                whisper_patience,
                whisper_length_penalty,
                whisper_suppress_tokens,
                whisper_initial_prompt,
                whisper_condition_on_previous_text,
                whisper_temperature_increment_on_fallback,
                whisper_no_speech_threshold,
                whisper_logprob_threshold,
                whisper_compression_ratio_threshold,
                whisper_trans_host,
                whisper_trans_api_key,
                whisper_trans_model,
                whisper_trans_temperature,
                whisper_trans_best_of,
                whisper_trans_beam_size,
                whisper_trans_patience,
                whisper_trans_length_penalty,
                whisper_trans_suppress_tokens,
                whisper_trans_initial_prompt,
                whisper_trans_condition_on_previous_text,
                whisper_trans_temperature_increment_on_fallback,
                whisper_trans_no_speech_threshold,
                whisper_trans_logprob_threshold,
                whisper_trans_compression_ratio_threshold,
                argos_source,
                argos_target,
                fade_timeout,
                libretranslate_host,
                libretranslate_api_key,
                ai_host,
                ai_api_key,
                ai_model,
                ai_translation_prompt_template,
                ai_endpoint_url,
                ai_request_body_template,
                ai_response_text_path,
                outline_width,
                outline_color,
                translated_outline_width,
                translated_outline_color,
                vad_threshold,
                vad_end_silence_ms,
                subtitle_mode,
                subtitle_cps,
                subtitle_max_lines,
                moonshine_language,
                moonshine_cache_dir,
                browser_session_data,
                browser_status,
                custom_popout_id,
                session_dropdown,
                noise_filter_threshold,
            ],
        )

    return interface


# ── Logging config ────────────────────────────────────────────────────────────
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(message)s",
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"level": "INFO", "handlers": ["default"], "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {
            "level": "WARNING",
            "handlers": ["access"],
            "propagate": False,
        },
    },
}

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    fastapi_app = FastAPI()
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Resolves the persistent session slug (see top of file) and pins it to
    # the browser via a cookie. Must be added before mount_gradio_app below.
    fastapi_app.add_middleware(SessionSlugMiddleware)

    @fastapi_app.get("/mic_level/{slug}")
    async def get_mic_level(slug: str):
        with SESSION_LOCK:
            app = SESSION_APPS.get(slug)
        if app:
            return JSONResponse({"rms": app.monitor_level})
        return JSONResponse({"rms": 0.0})

    @fastapi_app.get("/display_data/{slug}")
    async def get_display_data(slug: str):
        """
        Polled directly by JS every 50 ms — completely bypasses Gradio's SSE queue.
        This is what makes the display update without ever triggering a page reload.
        """
        with SESSION_LOCK:
            app = SESSION_APPS.get(slug)
        if app:
            html, rec, trans = app.get_current_display()
            return JSONResponse({"html": html, "recognized": rec, "translated": trans})
        return JSONResponse({"html": "", "recognized": "", "translated": ""})

    @fastapi_app.get("/logs_data/{slug}")
    async def get_logs_data(slug: str):
        """Polled by JS every 2 s — serves log text without using Gradio SSE."""
        with SESSION_LOCK:
            app = SESSION_APPS.get(slug)
        if app:
            return JSONResponse({"logs": app.update_logs()})
        return JSONResponse({"logs": ""})

    @fastapi_app.get("/active_sessions")
    async def get_active_sessions():
        with SESSION_LOCK:
            return JSONResponse({"sessions": list(SESSION_APPS.keys())})

    @fastapi_app.post("/deactivate/{slug}")
    async def deactivate_session(slug: str):
        """
        Explicit, manual full teardown of a session (stops recognition, frees
        models). Not called automatically on tab close/reload anymore — use
        the "Manage Sessions" panel in the UI, or call this directly, when you
        actually want to free a session's resources.
        """
        with SESSION_LOCK:
            app = SESSION_APPS.get(slug)
            if app:
                app.deactivate_session()
                del SESSION_APPS[slug]
                return JSONResponse({"status": "deactivated"})
        return JSONResponse({"status": "not found"}, status_code=404)

    @fastapi_app.websocket("/ws/{slug}")
    async def websocket_endpoint(websocket: WebSocket, slug: str):
        await websocket.accept()
        app = get_or_create_app(slug)
        try:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.receive" and "bytes" in msg:
                    app.audio_queue.put(msg["bytes"])
                    app.touch()
        except WebSocketDisconnect:
            # A dropped browser-audio stream (tab closed, "stop streaming"
            # clicked, network blip) no longer tears the session down — it
            # just stops feeding audio in. The session (models, settings,
            # any hardware-mic recognition) stays alive until the user
            # explicitly closes it or the idle reaper reclaims it.
            pass
        except Exception as exc:
            print(f"WebSocket error [{slug}]: {exc}")

    @fastapi_app.get("/popout/{popout_id}")
    async def get_popout(popout_id: str):
        with SESSION_LOCK:
            for app in SESSION_APPS.values():
                if app.popout_id == popout_id:
                    return HTMLResponse(content=app.generate_popout_html())
        return HTMLResponse("<h1>Not found</h1>", 404)

    @fastapi_app.get("/popout_data/{popout_id}")
    async def get_popout_data(popout_id: str):
        with SESSION_LOCK:
            for app in SESSION_APPS.values():
                if app.popout_id == popout_id:
                    _, rec, trans = app.get_current_display()
                    return JSONResponse(
                        {
                            "recognized": rec,
                            "translated": trans
                            if app.settings["enable_translation"]
                            else "",
                        }
                    )
        return JSONResponse({"error": "Not found"}, 404)

    @fastapi_app.get("/fonts/{filename}")
    async def get_font(filename: str):
        font_path = Path("fonts") / filename
        if not font_path.exists():
            return JSONResponse({"error": "Font not found"}, 404)
        return FileResponse(path=font_path, media_type="font/ttf")

    interface = create_ui(args)
    # Disable Gradio's built-in queue size limit and concurrency cap so it never
    # drops a client connection (which manifests as a page reload).
    interface.queue(
        max_size=None,  # never refuse/drop connections
        default_concurrency_limit=None,  # no cap on parallel event handlers
    )
    fastapi_app = gr.mount_gradio_app(fastapi_app, interface, path="/", head=js)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          🎤 Voice Translator - Multi-Session                 ║
╠══════════════════════════════════════════════════════════════╣
║  URL: http://{args.host}:{args.port}                                  ║
║  Named session: http://{args.host}:{args.port}/?session=<name>        ║
║                                                              ║
║  ✅ Per-session settings saved under {str(SETTINGS_DIR):<24}║
║  🎚️  VAD default: 50 ms silence timeout (all engines)      ║
║  ⏹️  Stop unloads models, settings preserved               ║
║  🌙 Moonshine: auto-download from HuggingFace               ║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        fastapi_app,
        host=args.host,
        port=args.port,
        log_config=LOG_CONFIG,
        # NOTE: this was previously `0`, which tells uvicorn to close the
        # keep-alive connection almost immediately after each response
        # instead of the "1 hour" the comment claimed. That was very likely
        # the main cause of "sessions randomly refresh" — Gradio's own SSE
        # queue connection (and any long HTTP polling) got dropped every few
        # seconds, forcing the frontend to reconnect. 3600s = 1 hour, as the
        # comment always intended.
        timeout_keep_alive=3600,  # never drop an idle SSE/WS connection
        ws_ping_interval=30,  # keep WebSocket alive with pings every 30 s
        ws_ping_timeout=120,  # give 2 minutes to respond before dropping
    )

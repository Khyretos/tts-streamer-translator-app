"""
Frame-level Voice Activity Detection with integrated vectorized preprocessing.
Extracted from voice_translator.py during the module-split refactor — see
REFACTOR.md. Only depends on numpy and (optionally) webrtcvad.
"""

import numpy as np

# ── Fast Voice Activity Detector ─────────────────────────────────────────────
# Primary backend: webrtcvad (pip install webrtcvad) — microsecond-fast per frame
# Fallback:        optimised RMS energy check
#
# Design for minimum dispatch latency:
#   • 10 ms frames at 16 kHz (160 samples = 320 bytes) — smallest webrtcvad unit
#   • 3-frame pre-roll (30 ms) so speech onset is never clipped
#   • 3 consecutive speech frames required to open a segment (30 ms min utterance)
#   • 2 consecutive silence frames trigger immediate dispatch (20 ms end-of-speech)
#   → total overhead after last spoken word: ~20 ms
#
# Only one user-tunable parameter: threshold 0.0–1.0
#   webrtcvad: maps to aggressiveness 0 (sensitive) … 3 (strict)
#   RMS fallback: maps to energy floor 0.002 … 0.05

_WRTCVAD_AVAILABLE = False
try:
    import webrtcvad as _wrtcvad_mod

    _WRTCVAD_AVAILABLE = True
except ImportError:
    pass

# Fixed frame geometry — never change these
_F_RATE = 16000
_F_MS = 10
_F_SAMP = _F_RATE * _F_MS // 1000  # 160 samples
_F_BYTES = _F_SAMP * 2  # 320 bytes (int16 mono)
_PREROLL = 3  # frames before speech onset (~30 ms)
_MIN_SPCH = 3  # frames to confirm speech (~30 ms)
# Minimum speech segment to dispatch to Whisper/Moonshine.
# Segments shorter than this are almost always desk taps / breath / noise.
# 500 ms = 50 frames.  Vosk is not affected (it handles segmentation itself).
_MIN_DISPATCH_MS = 150
_MIN_DISPATCH_FRAMES = _MIN_DISPATCH_MS // _F_MS  # 15
# End-of-speech silence (default 300 ms = 30 frames).
# ┌─ Whisper/Moonshine users: raise this if phrases are cut off mid-sentence.
# │  Each unit = 10 ms. Recommended range: 20 (200 ms) … 60 (600 ms).
# └─ Set via settings["vad_end_silence_ms"] in the UI.
_DEFAULT_END_SLNC_FRAMES = 30


class FastVAD:
    """
    Frame-level VAD with integrated, vectorized audio preprocessing.

    Pipeline (applied to every audio block as a single numpy batch operation):
      1. Transient suppression  — energy-ratio spike detector, zeroes click frames
      2. Spectral subtraction   — tracks background PSD during silence, attenuates it
      3. RMS energy gate        — user threshold in dBFS
      4. webrtcvad confirmation — optional spectral speech-shape check (webrtcvad)

    All processing is vectorized across the whole block at once (single FFT call),
    so per-callback overhead is ~0.04 ms regardless of block size.
    Works identically for Vosk, Whisper, and Moonshine (engine-agnostic).

    Hot-reloadable: update_threshold / update_end_silence_ms / update_noise_filter
    """

    # ── constants cached at class level ──────────────────────────────────────
    _WIN = np.hanning(_F_SAMP).astype(np.float32)  # shape (160,)
    _WIN_SUM = float(np.sum(_WIN**2))  # normalisation
    _N_FFT = 256  # padded FFT size
    _N_BINS = _N_FFT // 2 + 1  # 129 rfft bins

    def __init__(
        self, threshold_db=-30.0, end_silence_ms=300, noise_filter_threshold=0.0
    ):
        self._end_silence_frames = max(2, end_silence_ms // _F_MS)
        self._set_threshold(threshold_db)
        self._set_noise_filter(noise_filter_threshold)
        self._reset()

        # Spectral subtraction state (shape: (_N_BINS,))
        self._noise_psd: np.ndarray | None = None
        self._prev_clean: np.ndarray | None = None

        # Transient detector (scalar energy trackers)
        self._lt_energy = 1e-6
        self._st_energy = 1e-6

        # webrtcvad (optional)
        if _WRTCVAD_AVAILABLE:
            self._vad_obj = _wrtcvad_mod.Vad(3)
        else:
            self._vad_obj = None

    # ── hot-reload ────────────────────────────────────────────────────────────
    def _set_threshold(self, db):
        self.threshold = float(db)
        self._rms_floor = 10 ** (self.threshold / 20.0)

    def update_threshold(self, db):
        self._set_threshold(db)

    def update_end_silence_ms(self, ms):
        self._end_silence_frames = max(2, int(ms) // _F_MS)

    def _set_noise_filter(self, level):
        self._filter_level = max(0.0, min(1.0, float(level)))
        # Over-subtraction factor 1→4; spectral floor 0.05→0.001
        self._ss_alpha = 1.0 + self._filter_level * 3.0
        self._ss_floor = 0.05 * (1.0 - self._filter_level * 0.98)
        # Transient ratio: disabled at 0, tight (8×) at 1.0, loose (48×) at 0.1
        self._transient_ratio = (
            float("inf")
            if self._filter_level < 0.01
            else 8.0 + (1.0 - self._filter_level) * 40.0
        )

    def update_noise_filter(self, level):
        self._set_noise_filter(level)

    def update_noise_filter_threshold(self, level):
        self._set_noise_filter(level)

    def set_noise_filter_threshold(self, level):
        self._set_noise_filter(level)

    # ── state ─────────────────────────────────────────────────────────────────
    def _reset(self):
        self._preroll: list[bytes] = []
        self._segment: list[bytes] = []
        self._in_speech = False
        self._sil_count = 0
        self._leftover = b""

    def reset(self):
        self._reset()

    # ── vectorized block preprocessing ───────────────────────────────────────
    def _preprocess_block_array(
        self, frames: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process a batch of frames in one vectorized operation.

        Args:
            frames: float32 array, shape (N, _F_SAMP), already normalised to ±1.

        Returns:
            clean:        float32 array, shape (N, _F_SAMP) — noise-suppressed
            is_transient: bool array,    shape (N,)         — True = click/spike frame
        """
        N = frames.shape[0]
        is_transient = np.zeros(N, dtype=bool)

        if self._filter_level < 0.01:
            return frames.copy(), is_transient  # filter off — zero overhead

        # ── 1. Transient detection (per-frame energy, iterative but scalar) ──
        energies = np.mean(frames**2, axis=1) + 1e-10  # (N,)
        for i in range(N):
            e = float(energies[i])
            self._lt_energy = 0.999 * self._lt_energy + 0.001 * e
            self._st_energy = 0.85 * self._st_energy + 0.15 * e
            if (self._st_energy / self._lt_energy) > self._transient_ratio:
                is_transient[i] = True

        # Zero out transient frames before FFT so they don't corrupt the noise model
        frames_proc = frames.copy()
        frames_proc[is_transient] = 0.0

        # ── 2. Spectral subtraction (fully vectorized) ────────────────────────
        # Apply Hanning window to all frames at once: (N, _F_SAMP)
        windowed = frames_proc * self._WIN  # broadcast (N,160) * (160,)

        # Batch forward FFT: (N, _N_BINS) complex
        spectra = np.fft.rfft(windowed, n=self._N_FFT, axis=1)
        power = np.abs(spectra) ** 2  # (N, _N_BINS)

        # Update noise PSD only from frames below the RMS threshold (silence)
        rms_per_frame = np.sqrt(energies)
        silent_mask = rms_per_frame < self._rms_floor  # (N,)
        if np.any(silent_mask):
            noise_mean = np.mean(power[silent_mask], axis=0)  # (N_BINS,)
            if self._noise_psd is None:
                self._noise_psd = noise_mean
            else:
                self._noise_psd = 0.98 * self._noise_psd + 0.02 * noise_mean

        if self._noise_psd is None:
            return frames_proc, is_transient  # No noise estimate yet

        # Subtract scaled noise PSD from each frame's power
        noise = self._noise_psd[np.newaxis, :]  # (1, N_BINS) broadcast
        clean_power = power - self._ss_alpha * noise  # (N, N_BINS)

        # Half-wave rectify + spectral floor
        floor = self._ss_floor * noise
        clean_power = np.maximum(clean_power, floor)

        # Temporal smoothing to reduce musical noise: blend with previous frame's spectrum
        if self._prev_clean is not None:
            clean_power = 0.85 * self._prev_clean + 0.15 * clean_power
        self._prev_clean = clean_power[-1:].copy()  # keep last frame for next call

        # Compute gain and apply to original (unwindowed) spectra
        gain = np.sqrt(clean_power / (power + 1e-12))  # (N, N_BINS)
        gain = np.minimum(gain, 1.0)  # never amplify
        clean_spectra = spectra * gain  # (N, N_BINS)

        # Batch inverse FFT and take first _F_SAMP samples: (N, _F_SAMP)
        clean = np.fft.irfft(clean_spectra, n=self._N_FFT, axis=1)[:, :_F_SAMP]

        # Normalise windowing: scale by frame_length / window_energy
        if self._WIN_SUM > 0:
            clean = clean * (_F_SAMP / self._WIN_SUM)

        return clean.astype(np.float32), is_transient

    # ── public API ────────────────────────────────────────────────────────────
    def preprocess_block(self, audio_bytes: bytes) -> bytes:
        """
        Denoise a raw audio block (int16, 16 kHz, mono).
        Returns cleaned int16 bytes, same length as input.
        Used by Vosk and Moonshine which manage their own segmentation.
        """
        if self._filter_level < 0.01:
            return audio_bytes

        n_full = len(audio_bytes) // _F_BYTES
        tail = audio_bytes[n_full * _F_BYTES :]

        if n_full == 0:
            return audio_bytes

        # Decode entire block at once
        raw = np.frombuffer(audio_bytes[: n_full * _F_BYTES], dtype=np.int16)
        frames = raw.reshape(n_full, _F_SAMP).astype(np.float32) / 32768.0

        clean, _ = self._preprocess_block_array(frames)

        # Clip and re-encode to int16
        out = np.clip(clean * 32767.0, -32768, 32767).astype(np.int16)
        return out.tobytes() + tail  # append any sub-frame tail unchanged

    def process_chunk(self, audio_bytes: bytes) -> list[bytes]:
        """
        VAD segmentation + denoising for Whisper.
        Returns list of complete speech segments (cleaned int16 bytes).
        """
        data = self._leftover + audio_bytes
        n_full = (len(data)) // _F_BYTES
        tail = data[n_full * _F_BYTES :]

        segments: list[bytes] = []

        if n_full == 0:
            self._leftover = tail
            return segments

        # Decode and preprocess all frames in one batch
        raw = np.frombuffer(data[: n_full * _F_BYTES], dtype=np.int16)
        frames = raw.reshape(n_full, _F_SAMP).astype(np.float32) / 32768.0
        clean_f, is_transient = self._preprocess_block_array(frames)

        # Re-encode cleaned frames to int16 bytes once
        clean_int16 = np.clip(clean_f * 32767.0, -32768, 32767).astype(np.int16)
        frame_bytes = [clean_int16[i].tobytes() for i in range(n_full)]

        # Compute per-frame RMS on cleaned audio for the energy gate
        rms_per_frame = np.sqrt(np.mean(clean_f**2, axis=1))  # (N,)

        # Run VAD state machine over the preprocessed frames
        for i in range(n_full):
            fb = frame_bytes[i]
            rms = float(rms_per_frame[i])

            # A transient or below-threshold frame counts as silence
            if is_transient[i] or rms < self._rms_floor:
                is_speech = False
            elif self._vad_obj is not None:
                try:
                    is_speech = bool(self._vad_obj.is_speech(fb, _F_RATE))
                except Exception:
                    is_speech = True
            else:
                is_speech = True

            if is_speech:
                if not self._in_speech:
                    self._segment = list(self._preroll) + [fb]
                    self._in_speech = True
                    self._sil_count = 0
                else:
                    self._segment.append(fb)
                    self._sil_count = 0
                self._preroll.append(fb)
                if len(self._preroll) > _PREROLL:
                    self._preroll.pop(0)
            else:
                if self._in_speech:
                    self._segment.append(fb)
                    self._sil_count += 1
                    if self._sil_count >= self._end_silence_frames:
                        # Only dispatch segments long enough to contain real speech.
                        # Short bursts (desk tap, breath, click that slipped past) are dropped.
                        if len(self._segment) >= _MIN_DISPATCH_FRAMES:
                            segments.append(b"".join(self._segment))
                        self._segment = []
                        self._in_speech = False
                        self._sil_count = 0
                else:
                    self._preroll.append(fb)
                    if len(self._preroll) > _PREROLL:
                        self._preroll.pop(0)

        self._leftover = tail
        return segments

    def flush(self) -> bytes | None:
        seg = None
        if self._in_speech and len(self._segment) >= _MIN_SPCH:
            seg = b"".join(self._segment)
        self._reset()
        return seg

    def is_speech_rms(self, audio_bytes: bytes) -> bool:
        """Simple energy gate for Vosk (no state change)."""
        s = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(s**2))) > self._rms_floor

    # Backward-compat stubs
    def _rms(self, frame: bytes) -> float:
        s = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(s**2)))

    def is_noise_by_spectrum(self, frame: bytes) -> bool:
        """Kept for backward compat."""
        raw = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        e = float(np.mean(raw**2)) + 1e-10
        st = 0.85 * self._st_energy + 0.15 * e
        return (st / max(self._lt_energy, 1e-10)) > self._transient_ratio

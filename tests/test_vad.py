"""Tests for vad.FastVAD."""

import numpy as np
import pytest

from vad import FastVAD, _F_BYTES


def silence_bytes(n_frames: int) -> bytes:
    return (np.zeros(n_frames * (_F_BYTES // 2), dtype=np.int16)).tobytes()


def tone_bytes(n_frames: int, amplitude: int = 20000) -> bytes:
    """A block of frames well above any reasonable RMS threshold."""
    n_samples = n_frames * (_F_BYTES // 2)
    t = np.arange(n_samples)
    wave = (amplitude * np.sin(2 * np.pi * 440 * t / 16000)).astype(np.int16)
    return wave.tobytes()


class TestPreprocessBlock:
    def test_preserves_length(self):
        vad = FastVAD(noise_filter_threshold=0.5)
        data = silence_bytes(10)
        out = vad.preprocess_block(data)
        assert len(out) == len(data)

    def test_noop_when_filter_disabled(self):
        vad = FastVAD(noise_filter_threshold=0.0)
        data = tone_bytes(5)
        out = vad.preprocess_block(data)
        # Filter fully off — preprocess_block short-circuits and returns
        # the input unchanged.
        assert out == data


class TestProcessChunk:
    def test_silence_produces_no_segments(self):
        vad = FastVAD(threshold_db=-30.0, end_silence_ms=100)
        segments = vad.process_chunk(silence_bytes(50))
        assert segments == []

    def test_loud_tone_eventually_dispatches_a_segment(self):
        vad = FastVAD(threshold_db=-40.0, end_silence_ms=50)
        # Speech (350ms, comfortably over the 300ms minimum dispatch floor),
        # then enough silence frames to close the segment.
        segments = vad.process_chunk(tone_bytes(35))
        segments += vad.process_chunk(silence_bytes(20))
        assert len(segments) == 1
        # Segment bytes should be non-trivial in length.
        assert len(segments[0]) > 0

    def test_short_burst_is_dropped_as_noise(self):
        # A burst shorter than _MIN_DISPATCH_FRAMES should never produce a
        # dispatched segment, even after silence closes it.
        vad = FastVAD(threshold_db=-40.0, end_silence_ms=50)
        segments = vad.process_chunk(tone_bytes(2))  # well under min dispatch
        segments += vad.process_chunk(silence_bytes(20))
        assert segments == []


class TestHotReload:
    def test_update_threshold_changes_gate(self):
        vad = FastVAD(threshold_db=-10.0)
        rms_before = vad._rms_floor
        vad.update_threshold(-40.0)
        assert vad._rms_floor != rms_before
        assert vad.threshold == -40.0

    def test_update_end_silence_ms(self):
        vad = FastVAD(end_silence_ms=100)
        vad.update_end_silence_ms(400)
        assert vad._end_silence_frames == 40  # 400ms / 10ms per frame

    def test_end_silence_frames_has_floor_of_2(self):
        vad = FastVAD(end_silence_ms=0)
        assert vad._end_silence_frames >= 2


class TestFlushAndReset:
    def test_flush_with_no_active_segment_returns_none(self):
        vad = FastVAD()
        assert vad.flush() is None

    def test_reset_clears_state(self):
        vad = FastVAD(threshold_db=-40.0, end_silence_ms=1000)
        vad.process_chunk(tone_bytes(10))  # opens a segment, doesn't close it
        assert vad._in_speech is True
        vad.reset()
        assert vad._in_speech is False
        assert vad._segment == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

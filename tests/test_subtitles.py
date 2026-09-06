"""
Tests for subtitles.SubtitleManager — in particular the three real bugs
fixed in the timing-pass (see RECOGNITION_QUALITY.md / conversation history):

1. Cross-language sentence-count misalignment in buffered mode.
2. Hold time computed from only the recognized text's length.
3. Interim updates corrupting buffered-mode pacing.
"""

import time

import pytest

from subtitles import SubtitleManager


class TestInstantMode:
    def test_add_shows_immediately(self):
        sm = SubtitleManager(mode="instant", fade_timeout=5)
        sm.add("hello", "hola")
        assert sm.get_display() == ("hello", "hola")

    def test_fades_after_timeout(self):
        # fade_timeout has a floor of 0.5s in the constructor.
        sm = SubtitleManager(mode="instant", fade_timeout=0.5)
        sm.add("hello", "hola")
        time.sleep(0.6)
        assert sm.get_display() == ("", "")

    def test_interim_updates_current_text(self):
        sm = SubtitleManager(mode="instant", fade_timeout=5)
        sm.set_interim("partial words")
        rec, _ = sm.get_display()
        assert rec == "partial words"

    def test_empty_interim_is_ignored(self):
        sm = SubtitleManager(mode="instant", fade_timeout=5)
        sm.add("hello", "")
        sm.set_interim("")
        rec, _ = sm.get_display()
        assert rec == "hello"


class TestBufferedModeAlignment:
    """Bug 1: recognized/translated must never be paired from independent
    sentence splits — that produced mismatched or blank-translation pairs
    whenever the sentence counts didn't match across languages."""

    def test_translated_utterance_is_one_atomic_chunk(self):
        sm = SubtitleManager(mode="buffered", cps=10_000, max_lines=2)
        # 3 recognized sentences vs 1 translated sentence — would have
        # mismatched under the old independent-split-and-zip logic.
        sm.add("Hello. How are you? Great.", "Hola, ¿cómo estás?")
        rec, trans = sm.get_display()
        assert rec == "Hello. How are you? Great."
        assert trans == "Hola, ¿cómo estás?"

    def test_no_translation_still_splits_long_utterance(self):
        sm = SubtitleManager(mode="buffered", cps=10_000, max_lines=1)
        sm.add("First sentence. Second sentence. Third sentence.", "")
        rec, trans = sm.get_display()
        # With no translation to align against, splitting into sentence
        # chunks is safe and still expected.
        assert rec == "First sentence."
        assert trans == ""

    def test_multiple_translated_utterances_stay_paired_in_order(self):
        sm = SubtitleManager(mode="buffered", cps=10_000, max_lines=1)
        sm.add("one", "uno")
        sm.add("two", "dos")
        first = sm.get_display()
        assert first == ("one", "uno")
        # Force expiry. get_display() clears the expired chunk and returns
        # blank on *that* call (a deliberate one-poll-cycle gap between
        # subtitles — imperceptible at the real 50ms poll rate); the next
        # queued pair appears on the following call.
        sm._show_until = 0
        blank = sm.get_display()
        assert blank == ("", "")
        second = sm.get_display()
        assert second == ("two", "dos")


class TestBufferedModeHoldTime:
    """Bug 2: hold time must be based on whichever of rec/trans is longer,
    not just the recognized text."""

    def test_hold_time_uses_longer_translation(self):
        sm = SubtitleManager(mode="buffered", cps=10, max_lines=1)
        short_rec = "Hi."
        long_trans = "This is a much longer translated sentence than the original."
        sm.add(short_rec, long_trans)
        expected_min_hold = len(long_trans) / 10
        assert sm._show_until - time.time() >= expected_min_hold - 0.05

    def test_hold_time_uses_longer_recognized(self):
        sm = SubtitleManager(mode="buffered", cps=10, max_lines=1)
        long_rec = "This recognized sentence is much longer than its translation."
        short_trans = "Corto."
        sm.add(long_rec, short_trans)
        expected_min_hold = len(long_rec) / 10
        assert sm._show_until - time.time() >= expected_min_hold - 0.05

    def test_minimum_hold_is_2_5_seconds(self):
        sm = SubtitleManager(mode="buffered", cps=10_000, max_lines=1)
        sm.add("hi", "hola")
        assert sm._show_until - time.time() >= 2.4  # ~2.5s minimum, minus test jitter


class TestBufferedModeInterimIsolation:
    """Bug 3: interim (live-preview) updates must never disturb a chunk
    that's still being held for its paced reading time."""

    def test_interim_does_not_change_buffered_display(self):
        sm = SubtitleManager(mode="buffered", cps=10_000, max_lines=2, fade_timeout=5)
        sm.add("Final one.", "Trans one.")
        before = sm.get_display()
        sm.set_interim("some live preview text that should be ignored")
        after = sm.get_display()
        assert before == after == ("Final one.", "Trans one.")

    def test_interim_does_not_disturb_show_until(self):
        sm = SubtitleManager(mode="buffered", cps=10, max_lines=1)
        sm.add("hi", "a reasonably long translation to hold for a while")
        show_until_before = sm._show_until
        sm.set_interim("ignored")
        assert sm._show_until == show_until_before


class TestModeSwitching:
    def test_switching_to_instant_clears_buffered_queue(self):
        sm = SubtitleManager(mode="buffered", cps=10_000, max_lines=1)
        sm.add("one", "uno")
        sm.add("two", "dos")
        assert len(sm._rec_queue) >= 1
        sm.update_settings(mode="instant")
        assert sm._rec_queue == []
        assert sm._trans_queue == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

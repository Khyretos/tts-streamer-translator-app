"""Tests for recognizers.py — dots_or_stars, hallucination filtering, and
WhisperRecognizer's client-side confidence filtering (no network calls)."""

import pytest

from recognizers import (
    ArgosTranslator,
    WhisperRecognizer,
    dots_or_stars,
    is_whisper_hallucination,
)


class TestDotsOrStars:
    @pytest.mark.parametrize("text", [".", "<|something|>"])
    def test_true_cases(self, text):
        assert dots_or_stars(text) is True

    @pytest.mark.parametrize("text", ["hello", "", "..", "a normal sentence."])
    def test_false_cases(self, text):
        assert dots_or_stars(text) is False


class TestHallucinationFilter:
    @pytest.mark.parametrize(
        "text",
        [
            "Thank you.",
            "thank you",
            "SUBSCRIBE",
            "like and subscribe.",
            "  bye!  ",
            "Um.",
        ],
    )
    def test_known_hallucinations_blocked(self, text):
        assert is_whisper_hallucination(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Thank you for the detailed explanation of quantum mechanics.",
            "I need to go now, bye for now, see you tomorrow.",
            "The weather today is quite nice.",
            "",
        ],
    )
    def test_real_speech_not_blocked(self, text):
        assert is_whisper_hallucination(text) is False


class TestWhisperRecognizerRequestBuilding:
    def test_build_data_requests_verbose_json(self):
        wr = WhisperRecognizer(host="http://x", model="m")
        data = wr._build_data(language="en")
        assert data["response_format"] == "verbose_json"
        assert data["language"] == "en"

    def test_build_data_includes_confidence_thresholds(self):
        wr = WhisperRecognizer(
            host="http://x",
            model="m",
            no_speech_threshold=0.7,
            logprob_threshold=-1.2,
            compression_ratio_threshold=2.6,
        )
        data = wr._build_data()
        assert data["no_speech_threshold"] == 0.7
        assert data["logprob_threshold"] == -1.2
        assert data["compression_ratio_threshold"] == 2.6


class TestWhisperRecognizerResponseFiltering:
    """The core fix: confidence thresholds are actually enforced client-side
    now, using verbose_json segment data, instead of only being sent as
    unverified request params."""

    def make_recognizer(self, **kwargs):
        defaults = dict(
            host="http://x",
            model="m",
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        defaults.update(kwargs)
        return WhisperRecognizer(**defaults)

    def test_high_no_speech_prob_segment_is_dropped(self):
        wr = self.make_recognizer()
        result = {
            "segments": [
                {"text": "thank you", "no_speech_prob": 0.95, "avg_logprob": -0.1,
                 "compression_ratio": 1.0},
            ]
        }
        assert wr._extract_text(result) == ""

    def test_low_avg_logprob_segment_is_dropped(self):
        wr = self.make_recognizer()
        result = {
            "segments": [
                {"text": "garbled", "no_speech_prob": 0.1, "avg_logprob": -2.5,
                 "compression_ratio": 1.0},
            ]
        }
        assert wr._extract_text(result) == ""

    def test_high_compression_ratio_segment_is_dropped(self):
        wr = self.make_recognizer()
        result = {
            "segments": [
                {"text": "repeat repeat repeat repeat", "no_speech_prob": 0.1,
                 "avg_logprob": -0.1, "compression_ratio": 5.0},
            ]
        }
        assert wr._extract_text(result) == ""

    def test_confident_segment_is_kept(self):
        wr = self.make_recognizer()
        result = {
            "segments": [
                {"text": "hello there", "no_speech_prob": 0.05, "avg_logprob": -0.1,
                 "compression_ratio": 1.2},
            ]
        }
        assert wr._extract_text(result) == "hello there"

    def test_mixed_segments_keeps_only_confident_ones(self):
        wr = self.make_recognizer()
        result = {
            "segments": [
                {"text": "hello", "no_speech_prob": 0.05, "avg_logprob": -0.1,
                 "compression_ratio": 1.2},
                {"text": "thank you", "no_speech_prob": 0.9, "avg_logprob": -0.1,
                 "compression_ratio": 1.2},
                {"text": "world", "no_speech_prob": 0.05, "avg_logprob": -0.1,
                 "compression_ratio": 1.2},
            ]
        }
        assert wr._extract_text(result) == "hello world"

    def test_falls_back_to_plain_text_when_no_segments(self):
        """Server doesn't support verbose_json — nothing to filter on."""
        wr = self.make_recognizer()
        result = {"text": "plain response, no segments"}
        assert wr._extract_text(result) == "plain response, no segments"

    def test_non_dict_result_is_stringified(self):
        wr = self.make_recognizer()
        assert wr._extract_text("just a string") == "just a string"


class TestArgosTranslatorUnavailable:
    def test_translate_without_argos_returns_placeholder(self, monkeypatch):
        import recognizers

        monkeypatch.setattr(recognizers, "ARGOS_AVAILABLE", False)
        at = ArgosTranslator(logger=None)
        result = at.translate("hello", "en", "es")
        assert "not available" in result.lower()
        assert at.get_available_languages() == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

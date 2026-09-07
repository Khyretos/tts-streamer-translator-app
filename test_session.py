"""
Tests for session.py — slug sanitization and get_slug() resolution.

Rewritten for the session_hash-registry design that replaced cookies: a
cookie is shared across every tab of the same browser, which broke multi-
tab multi-session use (see the module docstring in session.py and the
regression tests below for the exact bug this replaced).
"""

import pytest

from session import (
    DEFAULT_SLUG,
    RESERVED_PATH_SEGMENTS,
    forget_session_hash,
    get_slug,
    register_slug,
    sanitize_slug,
)


class TestDefaultSlug:
    def test_default_slug_is_stable_and_readable(self):
        assert DEFAULT_SLUG == "main"
        assert DEFAULT_SLUG not in RESERVED_PATH_SEGMENTS


class TestSanitizeSlug:
    def test_strips_unsafe_characters(self):
        assert sanitize_slug("khyretos!@# $%") == "khyretos"

    def test_keeps_alphanumeric_dash_underscore(self):
        assert sanitize_slug("my-session_123") == "my-session_123"

    def test_truncates_to_64_chars(self):
        assert len(sanitize_slug("a" * 200)) == 64

    def test_none_or_empty_returns_empty(self):
        assert sanitize_slug("") == ""
        assert sanitize_slug(None) == ""


class FakeRequest:
    """Mimics the subset of gr.Request that get_slug() touches."""

    def __init__(self, session_hash="fallback-hash"):
        self.session_hash = session_hash


class TestRegisterAndGetSlug:
    def test_register_then_get_returns_registered_slug(self):
        req = FakeRequest(session_hash="hash-1")
        register_slug("hash-1", "khyretos")
        assert get_slug(req) == "khyretos"

    def test_unregistered_hash_falls_back_to_hash_itself(self):
        req = FakeRequest(session_hash="never-registered-hash")
        assert get_slug(req) == "never-registered-hash"

    def test_register_sanitizes_the_slug(self):
        register_slug("hash-2", "weird!!chars")
        req = FakeRequest(session_hash="hash-2")
        assert get_slug(req) == "weirdchars"

    def test_empty_slug_registers_as_default(self):
        register_slug("hash-3", "")
        req = FakeRequest(session_hash="hash-3")
        assert get_slug(req) == DEFAULT_SLUG

    def test_forget_removes_the_mapping(self):
        register_slug("hash-4", "temp-session")
        req = FakeRequest(session_hash="hash-4")
        assert get_slug(req) == "temp-session"
        forget_session_hash("hash-4")
        # After forgetting, falls back to the raw hash (unregistered).
        assert get_slug(req) == "hash-4"


class TestMultiTabIsolation:
    """
    Regression tests for the actual reported bug: opening two different
    named sessions in two tabs of the *same browser* must never let one
    tab's session identity leak into the other's. Cookies (the old
    mechanism) are shared across tabs and failed this; the session_hash
    registry doesn't, because Gradio gives each tab connection its own
    distinct, stable session_hash.
    """

    def test_two_tabs_with_different_hashes_stay_independent(self):
        tab_a = FakeRequest(session_hash="tab-a-hash")
        tab_b = FakeRequest(session_hash="tab-b-hash")

        register_slug("tab-a-hash", "khyretos")
        register_slug("tab-b-hash", "discord")

        # Registering tab B's slug must not affect tab A's, unlike the old
        # cookie-based approach where the second registration would
        # overwrite a single shared value.
        assert get_slug(tab_a) == "khyretos"
        assert get_slug(tab_b) == "discord"

    def test_re_registering_one_tab_does_not_affect_the_other(self):
        tab_a = FakeRequest(session_hash="tab-a-hash-2")
        tab_b = FakeRequest(session_hash="tab-b-hash-2")

        register_slug("tab-a-hash-2", "session-one")
        register_slug("tab-b-hash-2", "session-two")
        assert get_slug(tab_a) == "session-one"

        # Simulate tab B reloading and re-registering its own slug again.
        register_slug("tab-b-hash-2", "session-two")
        assert get_slug(tab_a) == "session-one"
        assert get_slug(tab_b) == "session-two"


class TestReservedPathSegments:
    def test_common_gradio_routes_are_reserved(self):
        for path in ("config", "assets", "ws", "popout", "static"):
            assert path in RESERVED_PATH_SEGMENTS

    def test_gradio_api_prefix_is_reserved(self):
        assert "gradio_api" in RESERVED_PATH_SEGMENTS

    def test_empty_string_is_reserved(self):
        assert "" in RESERVED_PATH_SEGMENTS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

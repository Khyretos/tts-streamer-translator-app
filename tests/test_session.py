"""Tests for session.py — slug sanitization and get_slug() priority order."""

import pytest

from session import DEFAULT_SLUG, RESERVED_PATH_SEGMENTS, get_slug, sanitize_slug


class TestDefaultSlug:
    def test_default_slug_is_stable_and_readable(self):
        # The base URL with no session specified should always resolve to
        # the same, human-readable slug — not a fresh random token on every
        # first visit (which made it unclear a "main" session even existed
        # to come back to).
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


class FakeQueryParams(dict):
    """Mimics starlette's QueryParams.get() interface closely enough."""

    def get(self, key, default=None):
        return dict.get(self, key, default)


class FakeRequest:
    """Mimics the subset of gr.Request that get_slug() touches."""

    def __init__(self, cookies=None, query_params=None, session_hash="fallback-hash"):
        self.cookies = cookies or {}
        self.query_params = FakeQueryParams(query_params or {})
        self.session_hash = session_hash


class TestGetSlugPriority:
    def test_cookie_takes_priority(self):
        req = FakeRequest(
            cookies={"vt_slug": "from-cookie"},
            query_params={"session": "from-query"},
        )
        assert get_slug(req) == "from-cookie"

    def test_query_param_used_when_no_cookie(self):
        req = FakeRequest(query_params={"session": "from-query"})
        assert get_slug(req) == "from-query"

    def test_falls_back_to_session_hash(self):
        req = FakeRequest(session_hash="abc123")
        assert get_slug(req) == "abc123"

    def test_empty_cookie_falls_through_to_query_param(self):
        req = FakeRequest(cookies={"vt_slug": ""}, query_params={"session": "q"})
        assert get_slug(req) == "q"

    def test_sanitizes_cookie_value(self):
        req = FakeRequest(cookies={"vt_slug": "weird!!chars"})
        assert get_slug(req) == "weirdchars"


class TestReservedPathSegments:
    def test_common_gradio_routes_are_reserved(self):
        for path in ("config", "assets", "ws", "popout", "static"):
            assert path in RESERVED_PATH_SEGMENTS

    def test_gradio_api_prefix_is_reserved(self):
        # Regression test: modern Gradio (5.x/6.x) namespaces essentially all
        # internal traffic under /gradio_api/..., and this segment being
        # missing from the reserved list was a real production bug — an
        # internal Gradio request got treated as a user-chosen slug, pinning
        # every visitor's session to the name "gradio_api".
        assert "gradio_api" in RESERVED_PATH_SEGMENTS

    def test_empty_string_is_reserved(self):
        assert "" in RESERVED_PATH_SEGMENTS


class TestGetSlugIgnoresReservedCookie:
    def test_reserved_word_as_cookie_is_treated_as_absent(self):
        # Self-healing for a browser that got poisoned by the gradio_api bug
        # before the fix: a cookie whose value is itself a reserved word
        # must never be trusted as a real slug.
        req = FakeRequest(cookies={"vt_slug": "gradio_api"})
        assert get_slug(req) != "gradio_api"

    def test_reserved_cookie_falls_through_to_query_param(self):
        req = FakeRequest(
            cookies={"vt_slug": "gradio_api"}, query_params={"session": "real-name"}
        )
        assert get_slug(req) == "real-name"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

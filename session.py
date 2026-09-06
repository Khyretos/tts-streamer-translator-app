"""
Persistent session-slug resolution.

A "slug" is the persistent, human-meaningful session identity — e.g. the
`khyretos` in `https://host/khyretos` (via reverse-proxy rewrite, see
SESSIONS.md) or `https://host/?session=khyretos`. Each slug maps to exactly
one VoiceTranslatorApp and one settings file, so re-opening the same URL —
in a new tab, after a reload, after a network blip — always re-attaches to
the same session instead of creating a new one.

See SESSIONS.md for the full design writeup.
"""

import re

from starlette.middleware.base import BaseHTTPMiddleware

SLUG_COOKIE_NAME = "vt_slug"
SLUG_COOKIE_MAX_AGE_SECONDS = 365 * 24 * 3600  # 1 year

# The slug a request gets when it names no session explicitly and has no
# existing cookie yet — i.e. a plain visit to "/". Fixed and human-readable
# rather than a random token, so "just open the site" reliably means "my
# one main session" rather than silently minting a new throwaway session
# every time (which is also what made the bug below hard to notice).
DEFAULT_SLUG = "main"

# Top-level paths Gradio/FastAPI itself serves. A bare path segment matching
# one of these is never treated as a session slug, so we never shadow the
# app's own routes when resolving "/<slug>"-style URLs.
#
# "gradio_api" is the important one: modern Gradio (5.x/6.x) namespaces
# essentially all of its internal traffic — queue, config, file serving,
# MCP, etc. — under a single /gradio_api/... prefix (see gradio.route_utils.
# API_PREFIX). Older Gradio used bare top-level paths instead (/queue,
# /config, /file=...), which is what the rest of this list still covers, for
# compatibility across versions. Missing "gradio_api" here was a real,
# confirmed bug: some internal Gradio request hit a path like
# /gradio_api/queue/join before this session's cookie was set, and the
# path-segment fallback below treated "gradio_api" as if it were a
# user-chosen slug — pinning it as that browser's session name.
RESERVED_PATH_SEGMENTS = {
    "",
    "gradio_api",
    "config",
    "info",
    "login",
    "logout",
    "reset",
    "queue",
    "static",
    "assets",
    "file",
    "upload",
    "stream",
    "proxy",
    "component_server",
    "custom_component",
    "theme.css",
    "manifest.json",
    "robots.txt",
    "startup-events",
    "heartbeat",
    "ws",
    "popout",
    "popout_data",
    "mic_level",
    "display_data",
    "logs_data",
    "deactivate",
    "fonts",
    "active_sessions",
    "favicon.ico",
}


def sanitize_slug(raw: str) -> str:
    """Keep a slug URL/filename-safe: alphanumerics, dash, underscore only."""
    return re.sub(r"[^a-zA-Z0-9_-]", "", raw or "")[:64]


# Backward-compat alias (this used to be a private module-level helper before
# the module split — kept so any external reference doesn't break).
_sanitize_slug = sanitize_slug


class SessionSlugMiddleware(BaseHTTPMiddleware):
    """
    Resolves a persistent session slug for every request and pins it to the
    browser with a long-lived cookie.

    Priority: ?session=<slug> query param > first URL path segment (if not
    reserved, e.g. reached via a reverse-proxy rewrite of /<slug>) > existing
    vt_slug cookie (unless it's a reserved word — see below) > DEFAULT_SLUG.

    The cookie is what makes this reliable: query strings aren't forwarded
    by the browser to Gradio's internal queue/websocket calls, but cookies
    are sent automatically on every request to the same origin, so every
    Python event handler can recover the same slug via get_slug(request).
    """

    async def dispatch(self, request, call_next):
        explicit = request.query_params.get("session", "")
        if not explicit:
            seg = request.url.path.strip("/").split("/")[0]
            if seg and seg not in RESERVED_PATH_SEGMENTS:
                explicit = seg
        explicit = sanitize_slug(explicit)

        cookie_slug = sanitize_slug(request.cookies.get(SLUG_COOKIE_NAME, ""))
        if cookie_slug in RESERVED_PATH_SEGMENTS:
            # Self-heal any browser that got a reserved word pinned as its
            # slug by the gradio_api bug above, before this fix shipped.
            cookie_slug = ""
        slug = explicit or cookie_slug or DEFAULT_SLUG

        request.state.vt_slug = slug
        response = await call_next(request)
        if slug != cookie_slug:
            response.set_cookie(
                SLUG_COOKIE_NAME,
                slug,
                max_age=SLUG_COOKIE_MAX_AGE_SECONDS,
                samesite="lax",
            )
        return response


def get_slug(request) -> str:
    """Resolve the persistent session slug from within a Gradio event handler."""
    try:
        slug = sanitize_slug(request.cookies.get(SLUG_COOKIE_NAME, ""))
        if slug and slug not in RESERVED_PATH_SEGMENTS:
            return slug
    except Exception:
        pass
    try:
        slug = sanitize_slug(dict(request.query_params).get("session", ""))
        if slug:
            return slug
    except Exception:
        pass
    # Last resort: Gradio's own ephemeral hash (matches pre-existing behavior
    # for a request the middleware somehow didn't see, e.g. direct queue call).
    return request.session_hash

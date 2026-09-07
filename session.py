"""
Persistent session-slug resolution.

A "slug" is the persistent, human-meaningful session identity — e.g. the
`khyretos` in `https://host/khyretos` (via reverse-proxy rewrite, see
SESSIONS.md) or `https://host/?session=khyretos`. Each slug maps to exactly
one VoiceTranslatorApp and one settings file, so re-opening the same URL —
in a new tab, after a reload, after a network blip — always re-attaches to
the same session instead of creating a new one.

See SESSIONS.md for the full design writeup.

## Why this isn't cookie-based anymore

The first version of this module used a cookie to remember the slug across
requests. That's broken for the actual use case: cookies are shared across
*every tab* of the same browser, not scoped to one tab. Opening "khyretos"
in one tab and "discord" in another meant whichever tab loaded most
recently overwrote the *one* shared cookie for the whole browser — so both
tabs' subsequent clicks (which don't carry the original page's query
string; that's a real Gradio limitation, not something we can fix here)
silently resolved to whichever slug's cookie happened to be current. That's
the "Already running" / wrong-session-name-in-title bug.

The fix: Gradio already hands every event handler a `request.session_hash`
that's stable for the lifetime of one tab's connection — that's how Gradio
itself routes queue events back to the right client, so it was never the
unreliable part. What *is* reliable per-tab is `sessionStorage` in the
browser (unlike cookies/localStorage, it's genuinely per-tab). So: resolve
the slug once, client-side, from `window.location` / `sessionStorage` at
page load (see the JS in voice_translator.py's handle_ui_load wiring),
register it against that tab's `session_hash` here, and every later
`get_slug(request)` call for the rest of that tab's life is just a dict
lookup — accurate, and it doesn't touch any of the other ~25 event handlers
that call get_slug(request) at all.
"""

import re
import threading

from starlette.middleware.base import BaseHTTPMiddleware

# The slug a request gets when it names no session explicitly. Fixed and
# human-readable rather than a random token, so "just open the site"
# reliably means "my one main session".
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
# compatibility across versions.
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
    Convenience only, no longer part of slug *resolution*: if the app is
    reached directly (no reverse proxy) via a pretty path like `/khyretos`,
    redirect to `/?session=khyretos` so the browser's URL bar ends up
    somewhere `window.location.search` can read directly. Behind nginx with
    the rewrite from SESSIONS.md, requests never reach the app as a bare
    path at all, so this rarely fires there — it's a fallback for
    direct/no-proxy access.
    """

    async def dispatch(self, request, call_next):
        if not request.query_params.get("session"):
            seg = request.url.path.strip("/").split("/")[0]
            if seg and seg not in RESERVED_PATH_SEGMENTS:
                clean = sanitize_slug(seg)
                if clean:
                    from starlette.responses import RedirectResponse

                    return RedirectResponse(url=f"/?session={clean}", status_code=307)
        return await call_next(request)


# ── session_hash -> slug registry ─────────────────────────────────────────────
# Populated once per tab, at handle_ui_load time, from a slug resolved
# client-side (see voice_translator.py). Gradio's session_hash is stable for
# the lifetime of that tab's connection, so every subsequent event handler
# in that tab gets the right slug via a plain dict lookup, without needing
# to touch every individual handler.
_HASH_TO_SLUG: dict[str, str] = {}
_HASH_TO_SLUG_LOCK = threading.Lock()


def register_slug(session_hash: str, slug: str) -> None:
    """Called once from handle_ui_load to bind this tab's session_hash to its slug."""
    slug = sanitize_slug(slug) or DEFAULT_SLUG
    with _HASH_TO_SLUG_LOCK:
        _HASH_TO_SLUG[session_hash] = slug


def forget_session_hash(session_hash: str) -> None:
    """Called from cleanup_user_data (tab close/reload) to bound the registry's size."""
    with _HASH_TO_SLUG_LOCK:
        _HASH_TO_SLUG.pop(session_hash, None)


def get_slug(request) -> str:
    """
    Resolve the persistent session slug from within a Gradio event handler.

    This is a lookup against the registry above, keyed by Gradio's own
    request.session_hash — which is reliably stable for every call from the
    same tab, unlike a cookie (shared across all tabs) or the original
    page's query string (not forwarded to Gradio's internal queue calls).
    Falls back to session_hash itself if this tab hasn't called
    handle_ui_load yet (shouldn't normally happen — it's the very first
    thing that runs — but better than crashing).
    """
    session_hash = getattr(request, "session_hash", None)
    if session_hash is None:
        return DEFAULT_SLUG
    with _HASH_TO_SLUG_LOCK:
        return _HASH_TO_SLUG.get(session_hash, session_hash)

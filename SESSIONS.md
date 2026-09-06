# Persistent named sessions (Phase 1)

## What changed

Previously each browser tab got a random, ephemeral `session_hash` from
Gradio that was thrown away (along with all running recognition state) the
moment the tab reloaded or the connection blipped — the UI even said
*"Each tab = separate session • Refresh = new session"*. That was by design,
but it meant:

- There was no way to return to "the same" session later.
- All settings were saved to a single global `settings.json` — one shared
  config for every tab/session.
- `settings.json` was never mounted as a Docker volume, so it was silently
  lost on every container restart or rebuild anyway.
- Any WebSocket hiccup, or the tab losing focus/being backgrounded, tore the
  whole session down (`close_session`), which — combined with a
  `timeout_keep_alive=0` typo in the uvicorn config (see below) — is the
  most likely explanation for "sessions randomly refresh."

## What it does now

Every request is resolved to a persistent **slug** — a stable session
identity — by `SessionSlugMiddleware`:

1. `?session=<name>` query parameter, if present.
2. The first path segment of the URL, if not one of Gradio's own reserved
   paths (`config`, `assets`, `ws`, `popout`, etc.) — see
   `RESERVED_PATH_SEGMENTS` in `voice_translator.py`.
3. An existing `vt_slug` cookie.
4. A freshly generated random slug (first-ever visit with no name given —
   still stable across reloads via the cookie, just unnamed).

The resolved slug is pinned to the browser as a 1-year cookie, so it's
recovered on *every* subsequent request/event call automatically — this is
what makes it reliable, since query strings are **not** forwarded by the
browser to Gradio's internal queue/WebSocket calls, but cookies are.

Each slug maps to exactly one `VoiceTranslatorApp` and one settings file
(`settings/<slug>.json`). Opening the same URL in a new tab, after a reload,
or from a different device re-attaches to the *same* running session —
same settings, same in-progress recognition — instead of creating a new one.

A tab closing/reloading no longer destroys anything. Sessions are permanent
by design: nothing in this app ever automatically stops or tears down a
session that's actively listening (`is_running`/`is_monitoring`), no matter
how long it runs or how much silence there is — silence is handled by the
VAD simply producing no segments, it never touches the session itself. The
only things that stop a session:
- You press Stop, or explicitly close it from the "Manage Sessions" panel.
- An optional, **off-by-default** idle reaper that only ever considers
  *stopped* sessions (recognition already off) left completely untouched
  for `IDLE_SESSION_TIMEOUT_SECONDS` — 0 means disabled. It never closes a
  running session under any circumstance. See `docker-compose.yml` to
  opt in if you want stopped-and-abandoned sessions to free memory.

## Using it

- **Query string (works everywhere, no proxy config needed):**
  `https://voice-translator.kreative-kompas.com/?session=khyretos`
- **Pretty path** (`https://.../khyretos`) — needs a rewrite at your nginx
  reverse proxy, since routing that at the Python/FastAPI layer risks
  shadowing Gradio's own top-level routes (`/config`, `/assets`, etc.) if a
  future Gradio version adds a new one we didn't know to exclude. Add this
  to your `nginx-reverse-proxy` config for the voice-translator vhost,
  *above* your default `location /` block:

  ```nginx
  # Rewrite /<name> to /?session=<name>, but only for single-segment paths
  # that aren't one of Gradio's own reserved routes.
  location ~ ^/(?!config$|info$|login$|logout$|reset$|queue|static|assets|file|upload|stream|proxy|component_server|custom_component|theme\.css$|manifest\.json$|robots\.txt$|startup-events$|heartbeat$|ws|popout|mic_level|display_data|logs_data|deactivate|fonts|active_sessions$|favicon\.ico$)([a-zA-Z0-9_-]+)/?$ {
      proxy_pass http://voice-translator:7860/?session=$1;
      proxy_set_header Host $host;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";
  }
  ```

  Keep this list in sync with `RESERVED_PATH_SEGMENTS` in
  `voice_translator.py` if you ever change it.

## Migration

Nothing to do manually. The first time a *new* slug loads with no settings
file of its own yet, it seeds itself from the legacy `settings.json` if one
exists (so your current tuned settings carry over into your first named
session). After that, each slug saves independently to
`settings/<slug>.json`.

## Other fixes bundled with this phase

- **`timeout_keep_alive=0`** in the final `uvicorn.run(...)` call was almost
  certainly a bug — the comment next to it said "1 hour" but the value was
  `0`, which tells uvicorn to close keep-alive connections almost
  immediately. This is a very plausible root cause of the random-refresh
  symptom by itself, independent of the session-teardown issue above. Fixed
  to `3600`.
- WebSocket disconnects (tab closed, "stop streaming" clicked, brief network
  drop) no longer call `close_session` — they just stop feeding audio in.

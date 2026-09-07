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

Each browser tab's slug is resolved **client-side**, once, at page load —
by JavaScript reading the `?session=` query string, or (if absent)
`sessionStorage` (a value it stores itself, purely to survive a reload of
*that specific tab* — unlike cookies or `localStorage`, `sessionStorage` is
genuinely scoped to one tab, never shared with any other tab in the same
browser), falling back to `DEFAULT_SLUG` (`"main"`) for a completely fresh
visit. That value gets registered against Gradio's own `request.session_hash`
— which is already stable for the lifetime of one tab's connection, since
that's literally how Gradio itself routes queue events back to the right
client — and every other event handler in that tab looks the slug up from
that registration for the rest of the tab's life. See `session.py`'s module
docstring for the full mechanism.

### Bug: sessions bleeding into each other across tabs (cookie-based design)

An earlier version of this resolved the slug server-side using a cookie.
That's broken for the actual use case: **cookies are shared across every
tab of the same browser**, not scoped to one tab. Opening `khyretos` in one
tab and `discord` in another meant whichever tab loaded most recently
overwrote the one shared cookie for the *entire browser* — so both tabs'
subsequent clicks (Start, Stop, settings changes — none of which carry the
original page's query string on their own; that's a real Gradio queue
limitation, not something fixable at the app layer) silently resolved to
whichever slug's cookie happened to be current at that moment. Symptoms:
"Already running — press Stop first" when starting a *different* session
that was never actually started, the session title in one tab suddenly
showing another tab's name, and closing a session from the "Manage
Sessions" dropdown not doing anything (because `current` was resolving to
the wrong tab's identity, throwing off which entries got excluded/matched).

Fixed by moving off cookies entirely, to the `sessionStorage` + `session_hash`
registry design described above — genuinely per-tab by construction, so two
tabs with two different session names can never leak into each other again.

### Bug: `Session: gradio_api`

If you saw a session literally named `gradio_api` on first load, that was a
real bug, now fixed. Modern Gradio (5.x/6.x) namespaces essentially all of
its internal traffic — the queue, config, file serving, MCP, etc. — under a
single `/gradio_api/...` path prefix. `RESERVED_PATH_SEGMENTS` was built
assuming older Gradio's bare top-level paths (`/queue`, `/config`,
`/file=...`) and didn't know about this prefix, so some internal Gradio
request to a path like `/gradio_api/queue/join` got its first path segment
("gradio_api") treated as if it were a user-chosen slug. Fixed by adding
`"gradio_api"` to the reserved list; the cookie-poisoning consequence this
originally had is moot now that slugs aren't cookie-based at all.

### Opening / creating a session from the UI

There's a "session switcher" in the header: a name field + **Go**
button (navigates to `?session=<name>`, sanitized client-side and again on
the server) and a **New** button (jumps to a fresh randomly-named session).
Previously the only way to open a named session was to already know you
could edit the URL by hand.

The resolved slug is stored in that tab's `sessionStorage`, so it survives
a reload of *that tab* without needing to retype the URL — this is also
what makes multi-tab correct: `sessionStorage` is never shared with other
tabs, unlike a cookie.

Each slug maps to exactly one `VoiceTranslatorApp` and one settings file
(`settings/<slug>.json`). Opening the same URL in a new tab (or the same
tab, after a reload) re-attaches to the *same* running session — same
settings, same in-progress recognition — instead of creating a new one.
Opening it from an entirely different browser/device also re-attaches to
the same session (settings are looked up by slug, not tied to any one
browser) — it just won't share that other browser's `sessionStorage`
convenience, so you'd need to include `?session=<name>` in the URL there.

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
  `session.py` if you ever change it.

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

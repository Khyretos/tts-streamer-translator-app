# Audio sources (Phase 3)

"Audio Mode" (hardware/browser) is now "Audio Source", with three options.
None of them are magic — each is a different, real way of getting audio
into the same recognition pipeline, with different capabilities/limits.

## 1. Server Audio Device (was "hardware")

Unchanged mechanically: the server opens an input stream on whichever
device you pick in the dropdown, via `sounddevice`/PortAudio, reading from
whatever the container can see through `/dev/snd`.

**This already supports "any source" today** — the dropdown lists *every*
input-capable device `sounddevice` finds, which includes virtual/loopback
devices, not just physical mics. If you want to translate a desktop app's
output (Discord's desktop client, Spotify, etc.), route that app's audio
into a virtual input device on the host, then pick that device here. A
couple of common ways to do that:

- **Linux (PulseAudio/PipeWire), most common for a Docker host:**
  ```bash
  # Create a virtual sink, and a virtual mic that's fed by its monitor
  pactl load-module module-null-sink sink_name=discord_capture
  pactl load-module module-remap-source \
      master=discord_capture.monitor source_name=discord_mic
  ```
  Then set Discord's output device to `discord_capture`, and this app's
  input to `discord_mic` (it'll show up in the dropdown as a normal input
  device — no code change needed). This container already mounts
  `/dev/snd:/dev/snd`; if it's not seeing PulseAudio-level devices (only
  raw ALSA hardware), you'll additionally need to share the PulseAudio
  socket into the container (bind-mount `/run/user/<uid>/pulse` and set
  `PULSE_SERVER`), which is a docker-compose change, not an app change —
  ask if you want that wired up.
- **Windows:** a virtual cable like VB-Audio Virtual Cable. Set Discord's
  output to the virtual cable, then run this app on the host (not
  necessarily in Docker, since USB/audio passthrough into Windows
  containers is a much bigger lift) and pick the virtual cable as input.
- **macOS:** BlackHole or Loopback, same idea.

This is host/OS audio routing, not something the Python app can do for you
— but once it's set up, the app sees it as just another input device.

## 2. Browser Microphone (was "browser")

Unchanged: `getUserMedia({ audio: true })` — this device's default
microphone, streamed to the server over the existing WebSocket. No device
picker (that's a Web Audio API limitation — a Chrome/Firefox permission
prompt gives you *a* mic, not a list to choose from, unless you request
`enumerateDevices()` + a `deviceId` constraint; ask if you want that added).

## 3. Browser Tab / System Audio (new)

Uses `getDisplayMedia({ video: true, audio: true })` — the same API behind
screen-sharing — and discards the video track immediately, keeping only the
audio. When you press Start (or Test Mic) in this mode, the browser shows
its native share picker; you choose a **tab**, **window**, or **entire
screen**, and must check **"Share audio"** (called "Share tab audio" for a
tab share) or the stream will have no audio track and the app will show an
error.

**What you can actually capture this way depends on your browser/OS** —
this is a browser/OS sandboxing limit, not something server code can work
around:

| Sharing... | Chrome/Edge (Win/Linux/ChromeOS) | Chrome/Edge (macOS) | Firefox | Safari |
|---|---|---|---|---|
| A browser tab's audio | ✅ | ✅ | ✅ (per-tab) | ❌ |
| Entire screen / system audio | ✅ | ❌ (OS blocks it) | ❌ | ❌ |

So: **Discord running in a browser tab** (`discord.com`) → share that tab
with audio → works everywhere Chromium does. **Discord's desktop app** is
not a browser tab, so it can only be captured this way via "entire screen"
system audio, which is Windows/ChromeOS-only in Chrome/Edge. On macOS,
route the desktop app through a virtual device instead (see option 1).

## Settings migration

Old saved `audio_mode: "browser"` is automatically migrated to
`"browser_mic"` on load — no action needed.

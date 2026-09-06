"""
Buffers and paces subtitle display. Extracted from voice_translator.py
during the module-split refactor — see REFACTOR.md. Self-contained;
only depends on the standard library.
"""

import re
import threading
import time

class SubtitleManager:
    """
    Buffers and paces subtitle display.

    instant mode  – shows text immediately, fades after timeout.
    buffered mode – queues sentences, pops max_lines at a time,
                    holds each chunk for len(chunk)/cps seconds (min 1.5 s).
    """

    def __init__(
        self,
        mode: str = "instant",
        cps: int = 21,
        max_lines: int = 2,
        fade_timeout: float = 5.0,
    ):
        self._lock = threading.Lock()
        # Initialize attributes directly
        self.mode = mode
        self.cps = max(1, cps)
        self.max_lines = max(1, max_lines)
        self.fade_timeout = max(0.5, fade_timeout)
        self._clear_state()

    def _clear_state(self):
        self._rec_queue: list[str] = []
        self._trans_queue: list[str] = []
        self._cur_rec = ""
        self._cur_trans = ""
        # Preserve the last final separately so a partial-overwrite can be recovered
        self._last_final_rec = ""
        self._last_final_trans = ""
        self._show_until = 0.0
        self._last_add = 0.0  # 0 means "nothing ever received"

    def update_settings(
        self,
        mode: str | None = None,
        cps: int | None = None,
        max_lines: int | None = None,
        fade_timeout: float | None = None,
    ):
        with self._lock:
            old_mode = self.mode
            if mode is not None:
                self.mode = mode
            if cps is not None:
                self.cps = max(1, cps)
            if max_lines is not None:
                self.max_lines = max(1, max_lines)
            if fade_timeout is not None:
                self.fade_timeout = max(0.5, fade_timeout)

            # When switching from buffered to instant, clear the queue
            if old_mode != self.mode and self.mode == "instant":
                self._rec_queue.clear()
                self._trans_queue.clear()

    def set_interim(self, text: str):
        """
        Update the live-preview text WITHOUT disturbing the subtitle queue.

        Only applies in instant mode. It used to unconditionally overwrite
        _cur_rec regardless of mode — in buffered mode that silently
        stomped on a chunk that was still being held for its paced reading
        time (_show_until was never updated to match), so the next
        get_display() call could expire the interim's leftover state early
        or late relative to what was actually being displayed. That's a
        second, independent source of the "goes fast sometimes" symptom.
        Buffered mode is specifically about pacing *finished* results, so a
        live-updating preview mid-utterance doesn't fit there anyway — it's
        simply ignored now; the queue and _show_until are untouched.
        """
        with self._lock:
            if self.mode != "instant":
                return
            stripped = (text or "").strip()
            if stripped:
                self._cur_rec = stripped
                self._last_add = time.time()
            # else: ignore empty interim – keep previous text

    def _split_into_sentences(self, text: str, max_chars: int = 100) -> list[str]:
        """Split text into sentences. If punctuation is missing, split by length."""
        if not text:
            return []
        # First try punctuation-based split
        sentences = re.split(r"(?<=[.!?;:])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # If we got only one sentence or any sentence is too long, do length-based split
        result = []
        for sent in sentences:
            if len(sent) <= max_chars:
                result.append(sent)
            else:
                # Split long sentence into chunks of max_chars at word boundaries
                words = sent.split()
                chunks = []
                current_chunk = []
                current_len = 0
                for word in words:
                    # +1 for space (except for first word in chunk)
                    if (
                        current_len + len(word) + (1 if current_chunk else 0)
                        <= max_chars
                    ):
                        current_chunk.append(word)
                        current_len += len(word) + (1 if current_chunk else 0)
                    else:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [word]
                        current_len = len(word)
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                result.extend(chunks)
        return result

    def add(self, recognized: str, translated: str = ""):
        recognized = (recognized or "").strip()
        translated = (translated or "").strip()
        if not recognized:
            return
        with self._lock:
            self._last_add = time.time()
            self._last_final_rec = recognized
            self._last_final_trans = translated

            if self.mode == "instant":
                self._cur_rec = recognized
                self._cur_trans = translated
            else:
                if translated:
                    # One utterance = one queued chunk when translating. We
                    # used to independently sentence-split *both* recognized
                    # and translated text by their own punctuation and zip
                    # them together by index — but sentence counts routinely
                    # don't match 1:1 across languages, so that produced
                    # mismatched pairs (recognized sentence N shown next to
                    # an unrelated translated sentence M) or a blank
                    # translation wherever the counts didn't line up. That's
                    # the "sometimes it just doesn't work" symptom. Keeping
                    # recognized+translated as one atomic pair makes that
                    # impossible — at the cost of not sub-splitting very long
                    # single utterances when translating, which is a fine
                    # trade for correctness.
                    self._rec_queue.append(recognized)
                    self._trans_queue.append(translated)
                else:
                    # No translation to align against, so it's safe to split
                    # a long utterance into readable sentence-sized chunks.
                    for sent in self._split_into_sentences(recognized, max_chars=100):
                        self._rec_queue.append(sent)
                        self._trans_queue.append("")

                # If nothing is currently showing, display the next chunk immediately
                if not self._cur_rec and not self._cur_trans:
                    self._advance_locked()

    def _advance_locked(self):
        """
        Pop the next chunk(s) into _cur_rec/_cur_trans and set the hold
        timer. Caller must already hold self._lock. Shared by add() (show
        immediately when nothing was queued) and get_display() (show the
        next chunk once the current one expires) — previously duplicated
        with a bug in one copy: the hold time was computed from only the
        recognized text's length, so a short recognized phrase with a much
        longer translation (or vice versa) got timed for whichever was
        SHORTER, cutting the longer one off before it could be read. That's
        the "sometimes it goes too fast" symptom.
        """
        rec_lines, trans_lines = [], []
        for _ in range(min(self.max_lines, len(self._rec_queue))):
            rec_lines.append(self._rec_queue.pop(0))
            if self._trans_queue:
                trans_lines.append(self._trans_queue.pop(0))

        self._cur_rec = " ".join(rec_lines)
        self._cur_trans = " ".join(t for t in trans_lines if t)
        # Pace by whichever text is longer, so both are readable in the same
        # hold window regardless of which language happens to be verbose.
        text_length = max(len(self._cur_rec), len(self._cur_trans))
        hold = max(text_length / self.cps, 2.5)  # minimum 2.5 s
        self._show_until = time.time() + hold

    def get_display(self) -> tuple[str, str]:
        """Return (recognized, translated) for the current render frame."""
        with self._lock:
            now = time.time()

            if self.mode == "instant":
                if self._last_add == 0.0:
                    return "", ""
                if self._cur_rec or self._cur_trans:
                    if now - self._last_add <= self.fade_timeout:
                        rec = self._cur_rec if self._cur_rec else self._last_final_rec
                        trans = (
                            self._cur_trans
                            if self._cur_trans
                            else self._last_final_trans
                        )
                        return rec, trans
                return "", ""

            # ========== BUFFERED MODE ==========
            # If we have queued sentences and nothing is currently showing, show the next chunk
            if self._rec_queue and not self._cur_rec and not self._cur_trans:
                self._advance_locked()
                return self._cur_rec, self._cur_trans

            # If current text exists, check if it has expired
            if self._cur_rec or self._cur_trans:
                if now >= self._show_until:
                    # Text expired - clear it
                    self._cur_rec = ""
                    self._cur_trans = ""
                    return "", ""
                return self._cur_rec, self._cur_trans

            return "", ""

    def clear(self):
        with self._lock:
            self._clear_state()

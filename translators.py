import json as _json

import requests

try:
    import translators as ts
except ImportError:
    ts = None


# ── Defaults for the AI-translation prompt/request/response shape ────────────
# All three are user-editable via settings (ai_translation_prompt_template,
# ai_request_body_template, ai_response_text_path) — see AI_TRANSLATION.md.
# These constants are what's used when the setting is blank/unset, and what
# a mangled custom value falls back to.
DEFAULT_AI_PROMPT_TEMPLATE = (
    "Translate the following text from {source_lang} to {target_lang}. Only "
    "provide the translation, no explanations if you cannot translate it "
    "return a single space:\n\n{text}"
)
DEFAULT_AI_REQUEST_BODY_TEMPLATE = _json.dumps(
    {
        "model": "__MODEL__",
        "messages": [{"role": "user", "content": "__PROMPT__"}],
        "temperature": 0.3,
        "max_tokens": 500,
    },
    indent=2,
)
DEFAULT_AI_RESPONSE_TEXT_PATH = "choices.0.message.content"


def _resolve_path(data, path: str):
    """
    Walk a dot-separated path through nested dicts/lists, e.g.
    "choices.0.message.content" → data["choices"][0]["message"]["content"].
    Numeric segments index into lists; anything else is a dict key. Raises
    (KeyError/IndexError/TypeError) on a bad path — callers catch and fall
    back to the default path.
    """
    node = data
    for segment in path.split("."):
        if isinstance(node, list):
            node = node[int(segment)]
        else:
            node = node[segment]
    return node


class TranslationService:
    """Handles translation using different backends"""

    def __init__(self, settings: dict, logger):
        self.settings = settings
        self.logger = logger

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using the configured translation mode"""
        if not text or not text.strip():
            return ""

        mode = self.settings.get("translation_mode", "internal")

        try:
            if mode == "ai":
                return self._translate_ai(text, source_lang, target_lang)
            elif mode == "libretranslate":
                return self._translate_libretranslate(text, source_lang, target_lang)
        except Exception as e:
            self.logger.log(f"Translation error ({mode}): {str(e)}", level="error")
            return f"[Translation error: {str(e)}]"

    def _build_ai_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """Fill in the (possibly user-edited) prompt template, falling back
        to the default template if the custom one is malformed (e.g. missing
        a required {text} placeholder, or containing an unknown one)."""
        template = (
            self.settings.get("ai_translation_prompt_template")
            or DEFAULT_AI_PROMPT_TEMPLATE
        )
        try:
            return template.format(
                source_lang=source_lang, target_lang=target_lang, text=text
            )
        except (KeyError, IndexError) as e:
            self.logger.log(
                f"Custom AI prompt template is invalid ({e}) — falling back to "
                f"the default. Check for a typo'd placeholder (only "
                f"{{source_lang}}, {{target_lang}}, {{text}} are supported).",
                level="warning",
            )
            return DEFAULT_AI_PROMPT_TEMPLATE.format(
                source_lang=source_lang, target_lang=target_lang, text=text
            )

    def _resolve_ai_endpoint_url(self, host: str) -> str:
        """
        Full endpoint URL, in priority order:
        1. ai_endpoint_url setting, if set — used verbatim, no guessing.
        2. Legacy auto-derivation from ai_host/host (kept for backward
           compatibility with existing configs) — fixed to append
           /v1/chat/completions (the actual OpenAI/Ollama-compatible path;
           this previously appended /v3/chat/completions, which is not a
           real endpoint on any standard OpenAI-compatible server and would
           silently 404 unless ai_host already happened to include the full
           path itself).
        """
        override = (self.settings.get("ai_endpoint_url") or "").strip()
        if override:
            return override
        if not host.endswith("/chat/completions"):
            if host.endswith("/v1"):
                host = f"{host}/chat/completions"
            elif host.endswith("/"):
                host = f"{host}v1/chat/completions"
            else:
                host = f"{host}/v1/chat/completions"
        return host

    def _build_ai_request_body(self, model: str, prompt: str) -> str:
        """
        Build the raw JSON request body, from the (possibly user-edited)
        template. __MODEL__ and __PROMPT__ are substituted as JSON-escaped
        string content (no surrounding quotes — the template's own quotes
        stay put), so the template controls the entire request shape, not
        just these two values. Falls back to the default OpenAI-compatible
        body if the custom template isn't valid JSON after substitution.
        """
        template = (
            self.settings.get("ai_request_body_template")
            or DEFAULT_AI_REQUEST_BODY_TEMPLATE
        )
        model_json = _json.dumps(model)[1:-1]
        prompt_json = _json.dumps(prompt)[1:-1]
        body_str = template.replace("__MODEL__", model_json).replace(
            "__PROMPT__", prompt_json
        )
        try:
            _json.loads(body_str)  # validate only — send the raw string as-is
            return body_str
        except ValueError as e:
            self.logger.log(
                f"Custom AI request body template is not valid JSON after "
                f"substitution ({e}) — falling back to the default request "
                f"shape. Check AI_TRANSLATION.md for the template format.",
                level="warning",
            )
            default_body = DEFAULT_AI_REQUEST_BODY_TEMPLATE.replace(
                "__MODEL__", model_json
            ).replace("__PROMPT__", prompt_json)
            return default_body

    def _extract_ai_response_text(self, result) -> str:
        """
        Pull the translated text out of the response JSON using the
        (possibly user-edited) dot-path, falling back to the default
        OpenAI-compatible path if the custom one doesn't resolve.
        """
        path = (
            self.settings.get("ai_response_text_path") or DEFAULT_AI_RESPONSE_TEXT_PATH
        )
        try:
            value = _resolve_path(result, path)
            return str(value).strip()
        except (KeyError, IndexError, TypeError) as e:
            if path != DEFAULT_AI_RESPONSE_TEXT_PATH:
                self.logger.log(
                    f"Custom AI response path '{path}' didn't resolve ({e}) — "
                    f"trying the default path '{DEFAULT_AI_RESPONSE_TEXT_PATH}'.",
                    level="warning",
                )
                value = _resolve_path(result, DEFAULT_AI_RESPONSE_TEXT_PATH)
                return str(value).strip()
            raise

    def _translate_ai(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate using an AI chat-completion-style service. Endpoint URL,
        request body shape, and response extraction path are all editable
        via settings for pointing this at a non-OpenAI-compatible API — see
        AI_TRANSLATION.md. Sensible OpenAI/Ollama-compatible defaults apply
        when none of the advanced fields are set.
        """
        try:
            host = self.settings.get("ai_host") or "http://localhost:11434"
            api_key = self.settings.get("ai_api_key", "")
            model = self.settings.get("ai_model", "llama3.2")

            url = self._resolve_ai_endpoint_url(host)

            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            prompt = self._build_ai_prompt(text, source_lang, target_lang)
            body_str = self._build_ai_request_body(model, prompt)

            self.logger.log(
                f"AI translation request to {url} with model {model}", level="info"
            )

            response = requests.post(
                url, headers=headers, data=body_str.encode("utf-8"), timeout=10
            )
            response.raise_for_status()
            result = response.json()

            translated_text = self._extract_ai_response_text(result)

            self.logger.log(
                f"AI translation: '{text}' -> '{translated_text}' ({source_lang}->{target_lang})",
                level="info",
            )

            return translated_text

        except requests.exceptions.RequestException as e:
            self.logger.log(f"AI API request error: {str(e)}", level="error")
            raise
        except (KeyError, IndexError) as e:
            self.logger.log(f"AI API response parsing error: {str(e)}", level="error")
            raise

    def _translate_libretranslate(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        """Translate using LibreTranslate service"""
        try:
            # Normalize language codes: "en-US" -> "en", "es-ES" -> "es", etc.
            src = source_lang.split("-")[0] if source_lang else "auto"
            tgt = target_lang.split("-")[0] if target_lang else "en"

            host = self.settings.get("libretranslate_host", "http://localhost:5000")
            api_key = self.settings.get("libretranslate_api_key", "")

            # Build URL: ensure it ends with /translate
            base = host.rstrip("/")
            if not base.endswith("/translate"):
                url = f"{base}/translate"
            else:
                url = base

            payload = {"q": text, "source": src, "target": tgt, "format": "text"}
            if api_key:
                payload["api_key"] = api_key

            self.logger.log(
                f"LibreTranslate request to {url} ({src}->{tgt})", level="info"
            )

            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            translated_text = result.get("translatedText", "")

            if not translated_text:
                self.logger.log(
                    "LibreTranslate returned empty translation", level="warning"
                )
                return ""

            self.logger.log(
                f"LibreTranslate: '{text}' -> '{translated_text}' ({src}->{tgt})",
                level="info",
            )
            return translated_text

        except requests.exceptions.ConnectionError:
            self.logger.log(
                "LibreTranslate connection error - is the server running?",
                level="error",
            )
            return f"[LibreTranslate not reachable at {host}]"
        except requests.exceptions.Timeout:
            self.logger.log("LibreTranslate request timed out", level="error")
            return "[LibreTranslate timeout]"
        except Exception as e:
            self.logger.log(f"LibreTranslate error: {str(e)}", level="error")
            raise

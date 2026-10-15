import logging

import requests
from babeldoc.utils.atomic_integer import AtomicInteger
from pdf2zh_next.config.model import SettingsModel
from pdf2zh_next.net_guard import guarded_session
from pdf2zh_next.net_guard import is_local_network_ip
from pdf2zh_next.translator.base_rate_limiter import BaseRateLimiter
from pdf2zh_next.translator.base_translator import BaseTranslator
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception
from tenacity import stop_after_attempt
from tenacity import wait_exponential

logger = logging.getLogger(__name__)


def _is_retryable(e: BaseException) -> bool:
    if isinstance(e, requests.exceptions.Timeout):
        return True
    if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
        return e.response.status_code == 429 or e.response.status_code >= 500
    return False


class OllamaTranslator(BaseTranslator):
    """Ollama on the local network, used for term extraction only.

    Every connection is checked to go to a non-global (private, loopback,
    link-local or CGNAT) address, so the document text cannot be sent to the
    internet through this translator even if the host name resolves there.
    """

    # https://github.com/ollama/ollama/blob/main/docs/api.md
    name = "ollama"

    def __init__(
        self,
        settings: SettingsModel,
        rate_limiter: BaseRateLimiter,
    ):
        super().__init__(settings, rate_limiter)
        engine_settings = settings.translate_engine_settings
        self.url = f"{engine_settings.ollama_host}/api/chat"
        self.model = engine_settings.ollama_model
        # 随机采样可能会打断公式标记
        self.temperature = 0
        self.num_predict = engine_settings.num_predict
        self.session = guarded_session(is_local_network_ip)
        self.add_cache_impact_parameters("temperature", self.temperature)
        self.add_cache_impact_parameters("num_predict", self.num_predict)
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self.prompt(""))
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()

    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        return self._chat(self.prompt(text), len(text))

    def do_llm_translate(self, text, rate_limit_params: dict = None):
        # BabelDOC calls this with None to check for LLM support.
        if text is None:
            return None
        return self._chat([{"role": "user", "content": text}], len(text))

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _chat(self, messages: list[dict], text_length: int) -> str:
        num_predict = max(self.num_predict or 0, text_length * 5)
        response = self.session.post(
            self.url,
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": num_predict,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        prompt_tokens = data.get("prompt_eval_count") or 0
        completion_tokens = data.get("eval_count") or 0
        self.token_count.inc(prompt_tokens + completion_tokens)
        self.prompt_token_count.inc(prompt_tokens)
        self.completion_token_count.inc(completion_tokens)
        message = data["message"]["content"].strip()
        return self._remove_cot_content(message)

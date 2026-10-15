"""Outbound data policy: document content only goes to the Claude Code CLI,
or, for term extraction only, to an Ollama server on the local network."""

import json
import threading
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from pathlib import Path

import pytest
import requests
from pdf2zh_next import high_level
from pdf2zh_next.config.cli_env_model import CLIEnvSettingsModel
from pdf2zh_next.config.model import SettingsModel
from pdf2zh_next.config.translate_engine_model import ClaudeCodeSettings
from pdf2zh_next.config.translate_engine_model import OllamaSettings
from pdf2zh_next.net_guard import guarded_session
from pdf2zh_next.net_guard import is_local_network_ip
from pdf2zh_next.net_guard import is_public_ip
from pdf2zh_next.translator.translator_impl.claudecode import ClaudeCodeTranslator
from pdf2zh_next.translator.translator_impl.ollama import OllamaTranslator

TEST_PDF = (
    Path(__file__).parent.parent / "test" / "file" / "translate.cli.plain.text.pdf"
)


@pytest.mark.parametrize(
    ("ip", "local"),
    [
        ("127.0.0.1", True),
        ("10.1.2.3", True),
        ("172.16.0.5", True),
        ("192.168.1.10", True),
        ("169.254.1.1", True),
        ("100.64.0.1", True),  # CGNAT, e.g. Tailscale
        ("::1", True),
        ("fd00::1", True),
        ("fe80::1%3", True),
        ("8.8.8.8", False),
        ("2606:4700:4700::1111", False),
        ("::ffff:8.8.8.8", False),
        ("0.0.0.0", False),  # noqa: S104
        ("224.0.0.1", False),
    ],
)
def test_is_local_network_ip(ip, local):
    assert is_local_network_ip(ip) is local
    if local:
        assert not is_public_ip(ip)


def test_local_network_session_refuses_public_address():
    session = guarded_session(is_local_network_ip)
    # Refused before any connection is opened (no DNS lookup for a literal IP).
    with pytest.raises(requests.exceptions.ConnectionError):
        session.post("http://8.8.8.8:11434/api/chat", json={}, timeout=5)


class _FakeOllama(BaseHTTPRequestHandler):
    requests_seen: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests_seen.append({"path": self.path, "body": body})
        content = "<think>hmm</think>" + body["messages"][-1]["content"].upper()
        payload = json.dumps(
            {
                "message": {"role": "assistant", "content": content},
                "prompt_eval_count": 3,
                "eval_count": 2,
                "done": True,
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_ollama():
    _FakeOllama.requests_seen = []
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _ollama_settings(host: str) -> SettingsModel:
    settings = SettingsModel(translate_engine_settings=ClaudeCodeSettings())
    settings.translate_engine_settings = OllamaSettings(
        ollama_host=host, ollama_model="test-model"
    )
    return settings


def test_ollama_translator_uses_chat_api(fake_ollama):
    translator = OllamaTranslator(_ollama_settings(fake_ollama), rate_limiter=None)
    assert translator.do_llm_translate(None) is None
    assert translator.do_llm_translate("extract terms") == "EXTRACT TERMS"
    request = _FakeOllama.requests_seen[-1]
    assert request["path"] == "/api/chat"
    assert request["body"]["model"] == "test-model"
    assert request["body"]["stream"] is False
    assert request["body"]["messages"] == [{"role": "user", "content": "extract terms"}]
    assert translator.token_count.value == 5


def test_claude_code_supports_llm_without_running_claude():
    translator = object.__new__(ClaudeCodeTranslator)
    # BabelDOC's LLM support check must not start the CLI.
    assert translator.do_llm_translate(None) is None


class _DummyTranslator:
    pass


def _babeldoc_config(monkeypatch, settings, term_translator):
    monkeypatch.setattr(high_level, "get_translator", lambda _: _DummyTranslator())
    monkeypatch.setattr(high_level, "get_term_translator", term_translator)
    return high_level.create_babeldoc_config(settings, TEST_PDF)


def test_unreachable_ollama_disables_term_extraction(monkeypatch):
    settings = CLIEnvSettingsModel(
        term_ollama=True,
        term_ollama_detail={"term_ollama_host": "http://192.168.1.10:11434"},
        pdf={"translate_table_text": False},
    ).to_settings_model()
    settings.validate_settings()
    assert settings.translation.no_auto_extract_glossary is False

    def unreachable(_settings):
        raise requests.exceptions.ConnectionError("refused")

    config = _babeldoc_config(monkeypatch, settings, unreachable)
    # Without a term extraction translator BabelDOC falls back to the main
    # translator (Claude Code), which it only uses for term extraction when
    # auto extraction is on, so auto extraction must be off.
    assert config.auto_extract_glossary is False


def test_ollama_term_translator_is_passed_to_babeldoc(monkeypatch):
    settings = CLIEnvSettingsModel(
        term_ollama=True,
        term_ollama_detail={"term_ollama_host": "http://192.168.1.10:11434"},
        pdf={"translate_table_text": False},
    ).to_settings_model()
    settings.validate_settings()
    term_translator = _DummyTranslator()

    config = _babeldoc_config(monkeypatch, settings, lambda _: term_translator)
    assert config.auto_extract_glossary is True
    assert config.term_extraction_translator is term_translator

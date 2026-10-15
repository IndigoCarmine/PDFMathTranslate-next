import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from pdf2zh_next.config.model import SettingsModel
from pdf2zh_next.translator.base_rate_limiter import BaseRateLimiter
from pdf2zh_next.translator.base_translator import BaseTranslator
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

logger = logging.getLogger(__name__)


class ClaudeCodeTranslator(BaseTranslator):
    name = "claudecode"

    def __init__(
        self,
        settings: SettingsModel,
        rate_limiter: BaseRateLimiter,
    ):
        super().__init__(settings, rate_limiter)
        self.claude_code_path = settings.translate_engine_settings.claude_code_path
        # Resolve now: the CLI is later started in a temporary directory, where a
        # relative path would no longer resolve.
        self.claude_code_path = shutil.which(self.claude_code_path) or str(
            Path(self.claude_code_path).resolve()
        )
        self.claude_code_model = settings.translate_engine_settings.claude_code_model
        self.add_cache_impact_parameters("model", self.claude_code_model)
        self.add_cache_impact_parameters("prompt", self.prompt(""))
        self._test_claude_code()

    def _test_claude_code(self):
        try:
            result = subprocess.run(
                [self.claude_code_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise ValueError(f"Claude Code CLI error: {result.stderr}")
        except FileNotFoundError as e:
            raise ValueError(
                f"Claude Code CLI not found at '{self.claude_code_path}'"
            ) from e

    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        return self._run_claude(self.prompt(text)[0])

    def do_llm_translate(self, text, rate_limit_params: dict = None):
        # BabelDOC calls this with None to check for LLM support.
        if text is None:
            return None
        # BabelDOC's own prompt, e.g. with glossary terms for consistency
        return self._run_claude({"role": "user", "content": text})

    @retry(
        retry=retry_if_exception_type(
            (subprocess.CalledProcessError, subprocess.TimeoutExpired)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _run_claude(self, message: dict) -> str:
        input_data = json.dumps({"type": "user", "message": message})

        # The text comes from the PDF and may contain prompt injection, so run
        # without any tools: no built-in tools (--tools "") and none of the
        # user's MCP servers (--strict-mcp-config without --mcp-config). The
        # document can then only go to the model, not to the web, files or
        # connected services. Also keep host context out of the model's view
        # (and so out of the translated output): replace the default system
        # prompt (cwd, env info, git status), load no settings/hooks
        # (--setting-sources=) or CLAUDE.md files, run in an empty directory
        # and do not save the session to disk.
        cmd = [
            self.claude_code_path,
            "-p",
            f"--model={self.claude_code_model}",
            "--max-turns",
            "1",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources=",
            "--no-session-persistence",
            "--system-prompt",
            "You are a translation engine. Follow the user's instructions.",
            "--disallowedTools",
            "Task Bash Glob Grep LS exit_plan_mode Read Edit MultiEdit Write NotebookRead NotebookEdit TodoRead TodoWrite WebFetch WebSearch",
        ]

        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

        try:
            logger.debug(f"Running Claude Code: {cmd}\n{input_data}")
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as workdir:
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    cwd=workdir,
                )

                try:
                    stdout, stderr = process.communicate(input=input_data, timeout=120)
                except subprocess.TimeoutExpired:
                    # Kill before leaving the block so the directory can be removed
                    process.kill()
                    process.communicate()
                    raise
            logger.debug(f"CC: {stdout}")

            if process.returncode != 0:
                logger.error(f"Claude Code failed: {stderr}")
                raise subprocess.CalledProcessError(process.returncode, cmd, stderr)

            translation = self._parse_output(stdout)
            return translation.strip()

        except subprocess.TimeoutExpired as e:
            raise ValueError("Claude Code translation timed out") from e

    def _parse_output(self, output: str) -> str:
        full_text = []
        for line in output.strip().split("\n"):
            if not line.strip() or not line.strip().startswith("{"):
                continue
            try:
                chunk = json.loads(line)
                if chunk.get("type") == "assistant" and "message" in chunk:
                    for content in chunk["message"].get("content", []):
                        if content.get("type") == "text":
                            full_text.append(content.get("text", ""))
                elif chunk.get("type") == "text":
                    full_text.append(chunk.get("text", ""))
            except json.JSONDecodeError:
                continue

        result = "".join(full_text).strip()
        if not result:
            raise ValueError("No translation received from Claude Code")
        return result

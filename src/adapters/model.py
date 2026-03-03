"""Model Adapters — unified interface via local CLI tools.

Uses locally installed CLI tools (claude, codex, gemini) for LLM calls.
No API keys needed — runs through the same CLIs the user already has.

Pattern: stdin pipe → subprocess → parse output.
Session-based: first call creates session, subsequent calls resume it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default models per backend
DEFAULT_MODELS = {
    "claude": "sonnet",
    "codex": "o3",
    "gemini": "gemini-2.5-pro",
}

def _clean_env() -> dict[str, str]:
    """Get environment without CLAUDECODE (prevents nested-session error)."""
    return {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}


def _detect_available_backend() -> str | None:
    """Detect which CLI backends are available on PATH."""
    for backend in ("claude", "codex", "gemini"):
        if shutil.which(backend):
            return backend
    return None


def _parse_json_response(text: str) -> dict | None:
    """Parse JSON from LLM response, handling markdown code blocks."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first { ... } block (greedy, handles nested)
    depth = 0
    start = -1
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1
    return None


# ---------------------------------------------------------------------------
# Claude CLI Session — manages a multi-turn conversation
# ---------------------------------------------------------------------------

class ClaudeSession:
    """A stateful Claude CLI session using --session-id / --resume.

    First call creates the session; subsequent calls resume it.
    Claude manages the full conversation context internally.
    """

    def __init__(self, model: str = "sonnet", system: str = "") -> None:
        self.model = model
        self.system = system
        self.session_id = str(uuid.uuid4())
        self._started = False

    def send(self, message: str, max_retries: int = 5) -> str:
        """Send a message and get a text response.

        First call uses --session-id to create session.
        Subsequent calls use --resume to continue.
        Retries on transient failures (e.g. concurrent session creation).
        """
        import time as _time
        import random as _random

        for attempt in range(max_retries):
            cmd = [
                "claude", "--print",
                "--model", self.model,
                "--output-format", "text",
                "--tools", "",
                "--dangerously-skip-permissions",
            ]

            is_new = not self._started
            if is_new:
                cmd.extend(["--session-id", self.session_id])
                if self.system:
                    cmd.extend(["--system-prompt", self.system])
            else:
                cmd.extend(["--resume", self.session_id])

            logger.info(
                "[session %s] send (%s): %d chars%s",
                self.session_id[:8],
                "new" if is_new else "resume",
                len(message),
                f" (retry {attempt})" if attempt > 0 else "",
            )

            proc = subprocess.run(
                cmd,
                input=message,
                capture_output=True,
                text=True,
                env=_clean_env(),
            )

            if proc.returncode == 0:
                self._started = True
                text = proc.stdout.strip()
                logger.info(
                    "[session %s] recv: %d chars — %s",
                    self.session_id[:8], len(text), text[:100],
                )
                return text

            stderr = proc.stderr.strip()[:500] if proc.stderr else ""
            stdout = proc.stdout.strip()[:500] if proc.stdout else ""
            error_msg = stderr or stdout or "unknown error"

            # Retry on transient session errors
            if attempt < max_retries - 1 and (
                "No conversation found" in error_msg
                or "rate" in error_msg.lower()
            ):
                delay = (2 ** attempt) + _random.random() * 2
                logger.warning(
                    "[session %s] transient error (attempt %d/%d), retrying in %.1fs: %s",
                    self.session_id[:8], attempt + 1, max_retries, delay, error_msg,
                )
                _time.sleep(delay)
                continue

            logger.error(
                "[session %s] failed (exit=%d): %s",
                self.session_id[:8], proc.returncode, error_msg,
            )
            raise RuntimeError(f"claude CLI failed: {error_msg}")

        raise RuntimeError(f"claude CLI failed after {max_retries} retries")

    def send_json(self, message: str) -> dict[str, Any]:
        """Send a message and parse the response as JSON."""
        raw = self.send(message)
        content = _parse_json_response(raw)
        if content is None:
            logger.warning("Failed to parse JSON: %s", raw[:200])
            content = {"raw_response": raw}
        return content


# ---------------------------------------------------------------------------
# Standalone CLI calls (stateless, for one-off use)
# ---------------------------------------------------------------------------

def run_claude_text(
    prompt: str,
    *,
    system: str = "",
    model: str = "haiku",
) -> str:
    """Run claude CLI with plain text output (stateless, one-off call)."""
    logger.info("[claude-text] model=%s, prompt_len=%d", model, len(prompt))

    cmd = [
        "claude", "--print",
        "--model", model,
        "--output-format", "text",
        "--tools", "",
        "--no-session-persistence",
        "--dangerously-skip-permissions",
    ]
    if system:
        cmd.extend(["--system-prompt", system])

    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        env=_clean_env(),
    )

    if proc.returncode != 0:
        error_msg = proc.stderr.strip()[:500] if proc.stderr else "unknown error"
        raise RuntimeError(f"claude CLI failed (exit={proc.returncode}): {error_msg}")

    text = proc.stdout.strip()
    logger.info("[claude-text] done: %d chars", len(text))
    return text


def run_claude_structured(
    prompt: str,
    *,
    system: str = "",
    model: str = "sonnet",
    json_schema: dict | None = None,
) -> dict[str, Any]:
    """Run claude CLI stateless, parse JSON from text response."""
    json_instruction = (
        "\n\nIMPORTANT: You MUST respond with a single valid JSON object. "
        "No markdown, no code blocks, no explanations — just raw JSON."
    )
    if json_schema:
        json_instruction += f"\n\nRequired JSON schema:\n{json.dumps(json_schema, indent=2)}"
    full_system = (system + json_instruction) if system else json_instruction

    raw = run_claude_text(prompt, system=full_system, model=model)
    content = _parse_json_response(raw)
    if content is None:
        logger.warning("Failed to parse JSON from response: %s", raw[:200])
        content = {"raw_response": raw}

    return {
        "content": content,
        "usage": {
            "input_tokens": len(prompt) // 4,
            "output_tokens": len(raw) // 4,
        },
    }


# ---------------------------------------------------------------------------
# Codex CLI backend
# ---------------------------------------------------------------------------

def _run_codex_cli(prompt: str, *, model: str = "o3") -> str:
    """Run the codex CLI and return text result."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_file = f.name
    try:
        cmd = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--json",
            "--full-auto",
            "-m", model,
            (
                f"Read the file {prompt_file} and follow the instructions in it exactly. "
                "Return ONLY the requested output format, nothing else."
            ),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"codex CLI failed (exit={proc.returncode}): {proc.stderr[:500]}"
            )
        output = proc.stdout.strip()
        if not output:
            raise RuntimeError(
                f"codex CLI returned no output (exit={proc.returncode}): {proc.stderr[:500]}"
            )
        # Parse codex JSON stream
        assistant_texts: list[str] = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type", "")
            if etype == "message" and event.get("role") == "assistant":
                content = event.get("content", "")
                if isinstance(content, str) and content:
                    assistant_texts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            assistant_texts.append(block["text"])
                        elif isinstance(block, str):
                            assistant_texts.append(block)
            elif etype in ("output", "result"):
                text = (
                    event.get("text")
                    or event.get("result")
                    or event.get("content", "")
                )
                if text:
                    assistant_texts.append(text)
        return "\n".join(assistant_texts) if assistant_texts else output
    finally:
        Path(prompt_file).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Gemini CLI backend
# ---------------------------------------------------------------------------

def _run_gemini_cli(prompt: str, *, model: str = "gemini-2.5-pro") -> str:
    """Run the gemini CLI and return text result."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_file = f.name
    try:
        cmd = [
            "gemini",
            "--prompt",
            (
                f"Read the file {prompt_file} and follow the instructions in it exactly. "
                "Return ONLY the requested output format, nothing else."
            ),
            "--model", model,
            "--output-format", "text",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"gemini CLI failed (exit={proc.returncode}): {proc.stderr[:500]}"
            )
        output = proc.stdout.strip()
        if not output:
            raise RuntimeError(
                f"gemini CLI returned no output (exit={proc.returncode}): {proc.stderr[:500]}"
            )
        return output
    finally:
        Path(prompt_file).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# JSON schema for ER agent actions
# ---------------------------------------------------------------------------

ER_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Step-by-step reasoning (ORIENT then HYPOTHESIZE)",
        },
        "policy_check": {
            "type": "string",
            "description": "Which policy applies and whether the action complies",
        },
        "action_name": {
            "type": "string",
            "description": "The tool to call, or 'respond' to reply to customer",
        },
        "action_kwargs": {
            "type": "object",
            "description": "Arguments for the action. For 'respond', must have a 'content' key with the message to the customer.",
        },
        "evidence_summary": {
            "type": "string",
            "description": "What has been learned so far",
        },
    },
    "required": ["reasoning", "action_name", "action_kwargs"],
}


# ---------------------------------------------------------------------------
# Unified CLI Model Adapter (session-based for claude)
# ---------------------------------------------------------------------------

class CLIModelAdapter:
    """Model adapter using locally installed CLI tools.

    For claude backend: uses ClaudeSession for multi-turn conversations.
    For codex/gemini: stateless calls (no session support).
    """

    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
        json_schema: dict | None = None,
    ) -> None:
        if backend is None:
            backend = _detect_available_backend()
            if backend is None:
                raise RuntimeError(
                    "No LLM CLI backend found. Install one of: "
                    "claude (Claude Code), codex (Codex CLI), gemini (Gemini CLI)"
                )
        self.backend = backend
        self.model = model or DEFAULT_MODELS.get(backend, "sonnet")
        self.json_schema = json_schema or ER_ACTION_SCHEMA
        self._session: ClaudeSession | None = None

    def new_session(self, system: str = "") -> ClaudeSession:
        """Create a new Claude session for a task."""
        self._session = ClaudeSession(model=self.model, system=system)
        return self._session

    async def generate(
        self,
        messages: list[dict[str, str]],
        system: str,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON response via CLI (stateless fallback)."""
        prompt = self._build_user_prompt(messages)
        schema = response_format or self.json_schema

        try:
            result = await asyncio.to_thread(
                self._call_backend, prompt, system, schema
            )
        except Exception as e:
            logger.error("CLI call failed (%s): %s", self.backend, e)
            return {
                "content": {"error": str(e)},
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }

        return result

    def _build_user_prompt(self, messages: list[dict[str, str]]) -> str:
        """Build user prompt from conversation messages."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                parts.append(content)
            elif role == "assistant":
                parts.append(f"[Previous response]\n{content}")
        return "\n\n".join(parts)

    def _call_backend(
        self, prompt: str, system: str, schema: dict | None
    ) -> dict[str, Any]:
        """Dispatch to the appropriate CLI backend."""
        if self.backend == "claude":
            return run_claude_structured(
                prompt, system=system, model=self.model, json_schema=schema
            )
        elif self.backend == "codex":
            full_prompt = f"# System Instructions\n{system}\n\n# Task\n{prompt}"
            full_prompt += "\n\n# Output Format\nRespond with a valid JSON object."
            raw = _run_codex_cli(full_prompt, model=self.model)
            content = _parse_json_response(raw)
            if content is None:
                content = {"raw_response": raw}
            return {
                "content": content,
                "usage": {"input_tokens": len(full_prompt) // 4, "output_tokens": len(raw) // 4},
            }
        elif self.backend == "gemini":
            full_prompt = f"# System Instructions\n{system}\n\n# Task\n{prompt}"
            full_prompt += "\n\n# Output Format\nRespond with a valid JSON object."
            raw = _run_gemini_cli(full_prompt, model=self.model)
            content = _parse_json_response(raw)
            if content is None:
                content = {"raw_response": raw}
            return {
                "content": content,
                "usage": {"input_tokens": len(full_prompt) // 4, "output_tokens": len(raw) // 4},
            }
        else:
            raise ValueError(f"Unknown backend: {self.backend}")


# ---------------------------------------------------------------------------
# Mock adapter for testing
# ---------------------------------------------------------------------------

class MockModelAdapter:
    """Mock model adapter for testing — returns predetermined responses."""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.responses = responses or []
        self._call_count = 0
        self.call_history: list[dict[str, Any]] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        system: str,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.call_history.append({
            "messages": messages,
            "system": system,
            "response_format": response_format,
        })

        if self._call_count < len(self.responses):
            response = self.responses[self._call_count]
        else:
            response = {
                "content": {},
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }

        self._call_count += 1
        return response

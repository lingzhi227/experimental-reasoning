"""CLI-based user simulator for tau-bench.

Replaces litellm API-based LLMUserSimulationEnv with one that uses
local CLI tools (claude --print) for user simulation.
No API keys needed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import tau-bench base class
try:
    from tau_bench.envs.user import BaseUserSimulationEnv
except ImportError:
    class BaseUserSimulationEnv:  # type: ignore[no-redef]
        pass

from ..adapters.model import run_claude_text, _run_codex_cli


# System prompt for user simulation (passed via --system-prompt, separate from user content)
USER_SIM_SYSTEM_PROMPT = """\
You are simulating a customer in a customer service conversation.
You will receive an instruction describing who you are and what you need.
Then you will see the conversation so far between you (Customer) and the Agent.

Rules:
- Generate ONE line at a time as the customer's next message.
- Do not give away all information at once. Only provide what's needed for the current step.
- Do not invent information not in your instruction. If asked for something you don't know, say you don't remember.
- If your goal is satisfied, respond with exactly: ###STOP###
- Use your own words, don't repeat the instruction verbatim.
- Be natural and stay in character."""


class CLIUserSimulationEnv(BaseUserSimulationEnv):
    """User simulator that uses local CLI instead of API.

    Uses claude --print with --system-prompt to generate user responses.
    Each call is stateless (full conversation history is in the prompt).
    """

    def __init__(
        self,
        backend: str = "claude",
        model: str = "haiku",
    ) -> None:
        self.backend = backend
        self.model = model
        self.instruction: Optional[str] = None
        self.conversation: List[Dict[str, str]] = []

    def _build_user_prompt(self) -> str:
        """Build the user-side prompt with instruction + conversation history."""
        parts = []

        if self.instruction:
            parts.append(f"Your instruction:\n{self.instruction}")

        if self.conversation:
            parts.append("\nConversation so far:")
            for msg in self.conversation:
                parts.append(f"{msg['role']}: {msg['content']}")

        parts.append(
            "\nGenerate the customer's next message. "
            "Output ONLY the message text, nothing else."
        )
        return "\n".join(parts)

    def _generate(self) -> str:
        """Generate next user message via CLI."""
        prompt = self._build_user_prompt()

        if self.backend == "claude":
            text = run_claude_text(
                prompt,
                system=USER_SIM_SYSTEM_PROMPT,
                model=self.model,
            )
        elif self.backend == "codex":
            full_prompt = f"{USER_SIM_SYSTEM_PROMPT}\n\n{prompt}"
            text = _run_codex_cli(full_prompt, model=self.model)
        else:
            raise ValueError(f"Unsupported backend for user sim: {self.backend}")

        # Clean up common prefixes the model might add
        text = text.strip()
        for prefix in ["Customer:", "Customer (you):", "[Customer]", "User:", "Me:"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        return text

    def reset(self, instruction: Optional[str] = None) -> str:
        self.instruction = instruction
        self.conversation = [
            {"role": "Agent", "content": "Hi! How can I help you today?"},
        ]
        logger.info("[user-sim] reset — generating initial customer message...")
        response = self._generate()
        logger.info("[user-sim] customer: %s", response[:100])
        self.conversation.append({"role": "Customer", "content": response})
        return response

    def step(self, content: str) -> str:
        self.conversation.append({"role": "Agent", "content": content})
        logger.info("[user-sim] agent said: %s", content[:100])
        logger.info("[user-sim] generating customer response...")
        response = self._generate()
        logger.info("[user-sim] customer: %s", response[:100])
        self.conversation.append({"role": "Customer", "content": response})
        return response

    def get_total_cost(self) -> float:
        return 0.0


def patch_env_with_cli_user(
    env: Any,
    backend: str = "claude",
    model: str = "haiku",
) -> Any:
    """Replace an env's LLM user simulator with a CLI-based one.

    Call this after get_env() to avoid needing API keys.

    Usage:
        env = get_env("retail", user_strategy="human", ...)
        patch_env_with_cli_user(env, backend="claude", model="haiku")
    """
    cli_user = CLIUserSimulationEnv(backend=backend, model=model)
    env.user = cli_user
    logger.info(
        "Patched env user simulator: CLI %s/%s (no API needed)",
        backend, model,
    )
    return env

"""ER Agent — the main orchestrator that ties all components together.

Usage:
    agent = ERAgent(backend="claude", model="sonnet", domain="bioinformatics")
    result = await agent.run("Analyze the RNA-seq data and identify DEGs...")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .cmm import CMMDatabase
from .hypothesis import HypothesisManager
from .loop import ERLoop, ERResult
from ..adapters.model import CLIModelAdapter, MockModelAdapter
from ..adapters.environment import (
    LocalPythonEnvironment,
    MockEnvironment,
    ToolCallingEnvironment,
)
from ..formats.engine import DefaultFormatEngine
from ..tactics.base import get_or_default_tactics

logger = logging.getLogger(__name__)


class ERAgent:
    """Experimental Reasoning Agent.

    Model-agnostic agentic scaffold that implements the ER loop:
    ORIENT → HYPOTHESIZE → EXPERIMENT → OBSERVE → EVALUATE → CONCLUDE

    Uses locally installed CLI tools (claude/codex/gemini) for LLM calls.

    Components:
      - CMM: Context Management Module (SQLite-backed)
      - HypothesisManager: Hypothesis lifecycle management
      - ERLoop: Core FSM engine
      - FormatEngine: FSM-state-aware prompt formatting
      - ModelAdapter: CLI-based LLM interface (Claude/Codex/Gemini)
      - EnvironmentAdapter: Benchmark execution interface
      - TacticCatalog: Domain-specific strategy library
    """

    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
        domain: str = "generic",
        db_path: str | Path | None = None,
        environment: Any | None = None,
        model_adapter: Any | None = None,
        max_cycles: int = 10,
        max_turns: int = 50,
    ) -> None:
        # Database
        self.cmm = CMMDatabase(db_path=db_path)
        self.cmm.initialize()

        # Hypothesis manager
        self.hyp_mgr = HypothesisManager(self.cmm)

        # Model adapter (CLI-based)
        if model_adapter is not None:
            self.model_adapter = model_adapter
        else:
            self.model_adapter = CLIModelAdapter(backend=backend, model=model)

        # Environment adapter
        if environment is not None:
            self.environment = environment
        else:
            self.environment = LocalPythonEnvironment()

        # Format engine
        self.format_engine = DefaultFormatEngine()

        # Tactics
        self.tactics = get_or_default_tactics(domain)
        self.domain = domain

        # Loop config
        self.max_cycles = max_cycles
        self.max_turns = max_turns

    async def run(self, task: str) -> ERResult:
        """Run the ER loop on a task.

        Returns an ERResult with the final answer, evidence chain,
        cycle history, and token usage.
        """
        logger.info(
            "Starting ER Agent | backend=%s | model=%s | domain=%s",
            getattr(self.model_adapter, "backend", "?"),
            getattr(self.model_adapter, "model", "?"),
            self.domain,
        )

        loop = ERLoop(
            cmm=self.cmm,
            hypothesis_manager=self.hyp_mgr,
            model=self.model_adapter,
            environment=self.environment,
            format_engine=self.format_engine,
            tactic_prompt=self.tactics.to_prompt_section(),
            max_cycles=self.max_cycles,
            max_turns=self.max_turns,
        )

        result = await loop.run(task)

        logger.info(
            "ER Agent complete | turns=%d | cycles=%d | tokens=%d | confidence=%.2f",
            result.total_turns,
            len(result.cycles),
            result.total_tokens,
            result.confidence,
        )

        return result

    def get_hypothesis_summary(self) -> dict[str, Any]:
        """Get a summary of the hypothesis state."""
        return self.hyp_mgr.summary()

    def get_cmm_stats(self) -> dict[str, Any]:
        """Get CMM database statistics."""
        return self.cmm.stats()

    def reset(self) -> None:
        """Reset the agent state for a new task."""
        self.cmm.close()
        self.cmm = CMMDatabase(db_path=self.cmm.db_path)
        self.cmm.initialize()
        self.hyp_mgr = HypothesisManager(self.cmm)

    def close(self) -> None:
        """Clean up resources."""
        self.cmm.close()


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def create_agent(
    backend: str | None = None,
    model: str | None = None,
    domain: str = "generic",
    benchmark: str | None = None,
    db_path: str | Path | None = None,
    **kwargs: Any,
) -> ERAgent:
    """Create an ER agent with sensible defaults for a given benchmark.

    Args:
        backend: CLI backend ("claude", "codex", "gemini", or None for auto-detect)
        model: Model name (e.g., "sonnet", "o3", "gemini-2.5-pro")
        domain: Tactic domain (generic, bioinformatics, ml_engineering, etc.)
        benchmark: Optional benchmark name for preset configurations
        db_path: Custom database path
        **kwargs: Additional arguments for ERAgent
    """
    if benchmark:
        presets = BENCHMARK_PRESETS.get(benchmark, {})
        domain = presets.get("domain", domain)
        if "environment" not in kwargs:
            env_class = presets.get("environment")
            if env_class:
                kwargs["environment"] = env_class()
        if "max_cycles" not in kwargs and "max_cycles" in presets:
            kwargs["max_cycles"] = presets["max_cycles"]
        if "max_turns" not in kwargs and "max_turns" in presets:
            kwargs["max_turns"] = presets["max_turns"]

    return ERAgent(
        backend=backend,
        model=model,
        domain=domain,
        db_path=db_path,
        **kwargs,
    )


# Benchmark → configuration presets
BENCHMARK_PRESETS: dict[str, dict[str, Any]] = {
    "bixbench": {
        "domain": "bioinformatics",
        "environment": LocalPythonEnvironment,
        "max_cycles": 15,
        "max_turns": 80,
    },
    "heurekabench": {
        "domain": "bioinformatics",
        "environment": LocalPythonEnvironment,
        "max_cycles": 15,
        "max_turns": 80,
    },
    "scienceagentbench": {
        "domain": "science_data",
        "environment": LocalPythonEnvironment,
        "max_cycles": 10,
        "max_turns": 60,
    },
    "mlebench": {
        "domain": "ml_engineering",
        "environment": LocalPythonEnvironment,
        "max_cycles": 12,
        "max_turns": 70,
    },
    "taubench": {
        "domain": "policy_compliance",
        "environment": ToolCallingEnvironment,
        "max_cycles": 8,
        "max_turns": 40,
    },
}

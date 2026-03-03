"""Format Engine — FSM-state-aware reasoning format selection.

Based on research showing that matching reasoning format to task type
yields Cohen's d = 1.58 improvement. Each FSM state uses the optimal
format for its cognitive demands.

| FSM State    | Format              | Rationale                          |
|------------- |---------------------|------------------------------------|
| ORIENT       | NL CoT              | Semantic understanding of task     |
| HYPOTHESIZE  | NL CoT + JSON       | NL for hypothesis, JSON for reg.   |
| EXPERIMENT   | PoT + Tool calls    | Execution needs code/tools         |
| OBSERVE      | Structured Table    | Results need structured recording  |
| EVALUATE     | Tactic Format       | Evidence evaluation needs precision|
| CONCLUDE     | NL CoT + Evidence   | Synthesis needs NL + provenance    |
"""

from __future__ import annotations

from typing import Any

from ..core.cmm import L1Context
from ..core.loop import ERState


class DefaultFormatEngine:
    """Default format engine that builds prompts based on FSM state."""

    def format_prompt(
        self,
        state: ERState,
        task: str,
        l1_context: L1Context,
        tactic_section: str = "",
        experiment_result: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        """Build the messages list for a given FSM state.

        Returns a list of message dicts for the model adapter.
        """
        context_section = l1_context.to_prompt_section()

        # State-specific formatting instructions
        format_instructions = FORMAT_INSTRUCTIONS.get(state, "")

        # Build user message with all context
        parts = [
            f"# Task\n{task}",
            context_section,
        ]

        if tactic_section:
            parts.append(f"\n## Available Tactics\n{tactic_section}")

        if experiment_result and state in (ERState.OBSERVE, ERState.EVALUATE):
            parts.append(self._format_experiment_result(experiment_result))

        if format_instructions:
            parts.append(f"\n## Output Format Instructions\n{format_instructions}")

        user_message = "\n\n".join(parts)

        return [{"role": "user", "content": user_message}]

    def _format_experiment_result(self, result: dict[str, Any]) -> str:
        """Format experiment result for observation/evaluation."""
        lines = ["\n## Experiment Result"]
        output = result.get("output", "")
        error = result.get("error")
        metrics = result.get("metrics", {})

        if error:
            lines.append(f"**Error**: {error}")
        if output:
            # Truncate large outputs
            output_str = str(output)
            if len(output_str) > 5000:
                output_str = output_str[:5000] + "\n... [truncated]"
            lines.append(f"**Output**:\n```\n{output_str}\n```")
        if metrics:
            lines.append("**Metrics**:")
            for k, v in metrics.items():
                lines.append(f"  - {k}: {v}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-state format instructions
# ---------------------------------------------------------------------------

FORMAT_INSTRUCTIONS: dict[ERState, str] = {
    ERState.ORIENT: """\
Use natural language chain-of-thought reasoning.
Think step by step about:
1. What is being asked?
2. What data/tools are available?
3. What are the key unknowns?
4. Is this simple enough to answer directly, or does it need experimentation?

Respond with a JSON object containing your analysis.""",

    ERState.HYPOTHESIZE: """\
Use natural language to formulate your hypothesis, then register it as structured JSON.
A good hypothesis is:
- Specific: precisely states what you expect
- Testable: can be confirmed or refuted with available tools
- Relevant: directly addresses the task goal

Respond with a JSON object containing your hypothesis and test plan.""",

    ERState.EXPERIMENT: """\
Use Program-of-Thought: write executable code or tool calls.
Be precise:
- Test ONE thing per experiment
- Use the smallest necessary data subset
- Include error handling for robustness
- Record what parameters/inputs you used

Respond with a JSON object containing your code/tool calls.""",

    ERState.OBSERVE: """\
Record your observations in structured format:
- Quantitative results → exact numbers, tables
- Qualitative results → categorized findings
- Errors → full error messages and likely causes
- Null results → explicitly note no signal found

Respond with a JSON object containing structured observations.""",

    ERState.EVALUATE: """\
Evaluate evidence using precise reasoning:
1. State the hypothesis prediction
2. State what was actually observed
3. Compare: does evidence support, contradict, or neither?
4. Rate evidence strength (0=weak, 1=strong)
5. Decide: conclude, test more, or revise hypothesis?

Use deductive logic. Avoid confirmation bias — contradicting evidence is informative.

Respond with a JSON object containing your evaluation.""",

    ERState.CONCLUDE: """\
Synthesize your answer with full evidence chain:
1. State your conclusion
2. List supporting evidence (with IDs)
3. Explain the reasoning chain
4. Note limitations and confidence level
5. If hypotheses were revised, explain the evolution

Your answer should be traceable: every claim backed by evidence.""",
}

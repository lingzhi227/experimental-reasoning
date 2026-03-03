"""ER Loop Engine — the core experimental reasoning cycle.

Implements the finite state machine:
  ORIENT → HYPOTHESIZE → EXPERIMENT → OBSERVE → EVALUATE → CONCLUDE

State transitions are evidence-conditioned, not free-form LLM choices.
Each state produces structured outputs that drive the next transition.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from .cmm import CMMDatabase, L1Context
from .hypothesis import HypothesisManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FSM States
# ---------------------------------------------------------------------------

class ERState(str, Enum):
    """States in the ER loop."""
    ORIENT = "orient"
    HYPOTHESIZE = "hypothesize"
    EXPERIMENT = "experiment"
    OBSERVE = "observe"
    EVALUATE = "evaluate"
    CONCLUDE = "conclude"


# ---------------------------------------------------------------------------
# State transition rules (evidence-conditioned)
# ---------------------------------------------------------------------------

# Each state maps to possible next states with conditions
TRANSITIONS: dict[ERState, dict[ERState, str]] = {
    ERState.ORIENT: {
        ERState.HYPOTHESIZE: "Task understood, ready to form hypotheses",
        ERState.CONCLUDE: "Task is trivial, can answer directly",
    },
    ERState.HYPOTHESIZE: {
        ERState.EXPERIMENT: "Hypothesis formed, ready to test",
    },
    ERState.EXPERIMENT: {
        ERState.OBSERVE: "Experiment executed, results available",
    },
    ERState.OBSERVE: {
        ERState.EVALUATE: "Observations recorded",
    },
    ERState.EVALUATE: {
        ERState.HYPOTHESIZE: "Hypothesis refuted or needs revision, form new hypothesis",
        ERState.EXPERIMENT: "Need more experiments to gather evidence",
        ERState.CONCLUDE: "Sufficient evidence gathered, ready to conclude",
    },
    ERState.CONCLUDE: {},  # Terminal state
}


# ---------------------------------------------------------------------------
# State prompts
# ---------------------------------------------------------------------------

STATE_PROMPTS: dict[ERState, str] = {
    ERState.ORIENT: """\
You are in the ORIENT phase of experimental reasoning.
Your goal: Understand the task, explore available data/tools, and identify what needs to be investigated.

Instructions:
1. Carefully read and understand the task requirements
2. Identify what data, tools, and resources are available
3. Perform initial exploratory analysis if needed
4. Identify key questions that need to be answered
5. Assess task complexity (simple → can conclude directly; complex → needs hypotheses)

Output a JSON object with:
- "understanding": your understanding of the task
- "key_questions": list of questions to investigate
- "available_resources": what data/tools are available
- "complexity": "simple" or "complex"
- "next_state": "hypothesize" (for complex) or "conclude" (for simple/trivial)
""",

    ERState.HYPOTHESIZE: """\
You are in the HYPOTHESIZE phase of experimental reasoning.
Your goal: Form or revise testable hypotheses based on current observations and evidence.

Instructions:
1. Review the active context (hypotheses, evidence, observations)
2. Either form a new hypothesis OR revise an existing one based on evidence
3. Each hypothesis must be:
   - Specific and testable
   - Connected to the task goal
   - Falsifiable through available tools/data
4. Specify what experiment would test this hypothesis

Output a JSON object with:
- "action": "propose" | "revise"
- "hypothesis_statement": the hypothesis text
- "parent_hypothesis_id": (if revising) ID of the hypothesis being revised
- "initial_confidence": float 0-1
- "test_plan": how to test this hypothesis
- "next_state": "experiment"
""",

    ERState.EXPERIMENT: """\
You are in the EXPERIMENT phase of experimental reasoning.
Your goal: Execute a specific experiment to test the current hypothesis.

Instructions:
1. Review the test plan from the hypothesis
2. Select the appropriate tactic/tool
3. Write and execute code or tool calls
4. Be precise — test ONE thing per experiment
5. Record what you did and what inputs you used

Output a JSON object with:
- "tactic": name of the tactic being used
- "code": executable code (if applicable)
- "tool_calls": list of tool calls (if applicable)
- "inputs_description": what data/parameters were used
- "expected_outcome": what you expect if hypothesis is correct
- "next_state": "observe"
""",

    ERState.OBSERVE: """\
You are in the OBSERVE phase of experimental reasoning.
Your goal: Record and structure the experiment results as evidence.

Instructions:
1. Examine the raw output from the experiment
2. Extract key findings and metrics
3. Identify any unexpected results
4. Structure the observation as evidence
5. Note any errors or issues with the experiment

Output a JSON object with:
- "raw_result_summary": summary of what was returned
- "key_findings": list of structured findings
- "metrics": any quantitative results (dict)
- "unexpected": anything surprising or unexpected
- "evidence_content": the evidence to record
- "evidence_type": "statistical" | "qualitative" | "error" | "null_result"
- "next_state": "evaluate"
""",

    ERState.EVALUATE: """\
You are in the EVALUATE phase of experimental reasoning.
Your goal: Evaluate the evidence against the current hypothesis and decide next steps.

Instructions:
1. Compare the evidence to the hypothesis prediction
2. Determine the relationship: supports / contradicts / neutral
3. Assess evidence strength (0-1)
4. Decide next action:
   - If hypothesis is well-supported AND sufficient for the task → conclude
   - If hypothesis is refuted → hypothesize (revise or new)
   - If more evidence needed → experiment (different angle)
5. Update hypothesis confidence

Output a JSON object with:
- "hypothesis_id": ID of the hypothesis being evaluated
- "evidence_relation": "supports" | "contradicts" | "neutral"
- "evidence_strength": float 0-1
- "reasoning": why this evidence supports/contradicts the hypothesis
- "hypothesis_verdict": "continue_testing" | "supported" | "refuted" | "revise"
- "sufficient_for_task": boolean - do we have enough to answer the task?
- "next_state": "conclude" | "hypothesize" | "experiment"
""",

    ERState.CONCLUDE: """\
You are in the CONCLUDE phase of experimental reasoning.
Your goal: Synthesize all evidence into a final answer with full provenance chain.

Instructions:
1. Review all hypotheses and their status
2. Review all evidence and observations
3. Construct the final answer with evidence support
4. Include the provenance chain (which evidence supports which claims)
5. Note any limitations or caveats

Output a JSON object with:
- "final_answer": the answer to the task
- "evidence_chain": list of evidence IDs supporting the answer
- "confidence": overall confidence in the answer (0-1)
- "supported_hypotheses": list of hypothesis IDs that were confirmed
- "limitations": any caveats or limitations
- "reasoning_summary": brief summary of the reasoning process
""",
}


# ---------------------------------------------------------------------------
# Protocols for pluggable components
# ---------------------------------------------------------------------------

class ModelAdapter(Protocol):
    """Protocol for LLM model adapters."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        system: str,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a structured response from the model.

        Returns a dict with at minimum:
          - "content": the parsed JSON response
          - "usage": {"input_tokens": int, "output_tokens": int}
        """
        ...


class EnvironmentAdapter(Protocol):
    """Protocol for environment adapters (benchmark-specific)."""

    async def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute an action in the environment.

        action may contain:
          - "code": Python code to execute
          - "tool_calls": list of tool invocations
          - "tactic": name of the tactic being used

        Returns a dict with:
          - "output": raw output text/data
          - "error": error message if any
          - "metrics": any extracted metrics
        """
        ...


class FormatEngine(Protocol):
    """Protocol for the reasoning format engine."""

    def format_prompt(
        self,
        state: ERState,
        task: str,
        l1_context: L1Context,
        tactic_section: str,
        experiment_result: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        """Build the messages list for a given FSM state."""
        ...


# ---------------------------------------------------------------------------
# Cycle result
# ---------------------------------------------------------------------------

@dataclass
class ERCycleResult:
    """Result of a single ER experiment cycle."""
    cycle_id: str
    hypothesis_id: str | None
    hypothesis_statement: str | None
    verdict: str | None  # "supported" | "refuted" | "revise" | "continue_testing"
    evidence_ids: list[str] = field(default_factory=list)
    state_trace: list[str] = field(default_factory=list)
    tokens_used: int = 0


@dataclass
class ERResult:
    """Final result of an ER run."""
    task: str
    final_answer: str | None = None
    confidence: float = 0.0
    evidence_chain: list[str] = field(default_factory=list)
    cycles: list[ERCycleResult] = field(default_factory=list)
    total_tokens: int = 0
    total_turns: int = 0
    state_trace: list[str] = field(default_factory=list)
    reasoning_summary: str | None = None


# ---------------------------------------------------------------------------
# ER Loop Engine
# ---------------------------------------------------------------------------

class ERLoop:
    """Core experimental reasoning loop engine.

    Orchestrates the ORIENT → HYPOTHESIZE → EXPERIMENT → OBSERVE → EVALUATE → CONCLUDE
    cycle with evidence-conditioned state transitions.
    """

    def __init__(
        self,
        cmm: CMMDatabase,
        hypothesis_manager: HypothesisManager,
        model: ModelAdapter,
        environment: EnvironmentAdapter,
        format_engine: FormatEngine,
        tactic_prompt: str = "",
        max_cycles: int = 10,
        max_turns: int = 50,
    ) -> None:
        self.cmm = cmm
        self.hyp_mgr = hypothesis_manager
        self.model = model
        self.env = environment
        self.format_engine = format_engine
        self.tactic_prompt = tactic_prompt
        self.max_cycles = max_cycles
        self.max_turns = max_turns

    async def run(self, task: str) -> ERResult:
        """Execute the full ER loop on a task."""
        result = ERResult(task=task)
        current_state = ERState.ORIENT
        cycle_id = f"C-{uuid.uuid4().hex[:8]}"
        current_cycle = ERCycleResult(cycle_id=cycle_id, hypothesis_id=None,
                                       hypothesis_statement=None, verdict=None)
        cycle_count = 0
        turn_count = 0
        total_tokens = 0

        # Conversation history for the model
        messages: list[dict[str, str]] = []

        while turn_count < self.max_turns:
            turn_count += 1
            result.state_trace.append(current_state.value)
            current_cycle.state_trace.append(current_state.value)

            logger.info(
                "Turn %d | State: %s | Cycle: %s",
                turn_count, current_state.value, cycle_id,
            )

            # Assemble L1 context
            l1 = self.cmm.assemble_l1_context(
                current_state=current_state.value,
                cycle_id=cycle_id,
            )

            # Build prompt via format engine
            messages = self.format_engine.format_prompt(
                state=current_state,
                task=task,
                l1_context=l1,
                tactic_section=self.tactic_prompt,
                experiment_result=(
                    self._last_experiment_result
                    if hasattr(self, "_last_experiment_result")
                    else None
                ),
            )

            # Get system prompt for current state
            system_prompt = STATE_PROMPTS[current_state]

            # Call model
            start_ms = time.time()
            model_response = await self.model.generate(
                messages=messages,
                system=system_prompt,
            )
            duration_ms = int((time.time() - start_ms) * 1000)

            content = model_response.get("content", {})
            usage = model_response.get("usage", {})
            tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            total_tokens += tokens

            # Log action
            action_id = self.cmm.log_action(
                state_name=current_state.value,
                cycle_id=cycle_id,
                tactic_name=content.get("tactic"),
                input_summary=json.dumps(content)[:500] if content else None,
                tokens_used=tokens,
                duration_ms=duration_ms,
            )

            # Process state-specific logic
            next_state = await self._process_state(
                state=current_state,
                content=content,
                action_id=action_id,
                cycle_id=cycle_id,
                current_cycle=current_cycle,
                result=result,
            )

            # Terminal state check
            if next_state == ERState.CONCLUDE:
                # Process conclude state
                result.state_trace.append(ERState.CONCLUDE.value)

                if current_state != ERState.CONCLUDE:
                    # Need one more model call for conclusion
                    l1 = self.cmm.assemble_l1_context(
                        current_state=ERState.CONCLUDE.value,
                        cycle_id=cycle_id,
                    )
                    messages = self.format_engine.format_prompt(
                        state=ERState.CONCLUDE,
                        task=task,
                        l1_context=l1,
                        tactic_section=self.tactic_prompt,
                    )
                    conclude_response = await self.model.generate(
                        messages=messages,
                        system=STATE_PROMPTS[ERState.CONCLUDE],
                    )
                    conclude_content = conclude_response.get("content", {})
                    conclude_usage = conclude_response.get("usage", {})
                    total_tokens += (
                        conclude_usage.get("input_tokens", 0)
                        + conclude_usage.get("output_tokens", 0)
                    )
                    result.final_answer = conclude_content.get("final_answer")
                    result.confidence = conclude_content.get("confidence", 0.0)
                    result.evidence_chain = conclude_content.get("evidence_chain", [])
                    result.reasoning_summary = conclude_content.get("reasoning_summary")
                else:
                    result.final_answer = content.get("final_answer")
                    result.confidence = content.get("confidence", 0.0)
                    result.evidence_chain = content.get("evidence_chain", [])
                    result.reasoning_summary = content.get("reasoning_summary")

                # Save cycle
                result.cycles.append(current_cycle)

                # Create L2 summary
                self.cmm.add_experiment_summary(
                    cycle_id=cycle_id,
                    summary_text=result.reasoning_summary or "Completed",
                    hypothesis_id=current_cycle.hypothesis_id,
                    outcome="concluded",
                )
                break

            # Handle cycle boundary (EVALUATE → HYPOTHESIZE means new cycle)
            if (current_state == ERState.EVALUATE
                    and next_state == ERState.HYPOTHESIZE):
                result.cycles.append(current_cycle)
                cycle_count += 1
                if cycle_count >= self.max_cycles:
                    logger.warning("Max cycles (%d) reached, forcing conclusion",
                                   self.max_cycles)
                    next_state = ERState.CONCLUDE
                    continue
                cycle_id = f"C-{uuid.uuid4().hex[:8]}"
                current_cycle = ERCycleResult(
                    cycle_id=cycle_id, hypothesis_id=None,
                    hypothesis_statement=None, verdict=None
                )

            current_state = next_state

        result.total_tokens = total_tokens
        result.total_turns = turn_count
        return result

    async def _process_state(
        self,
        state: ERState,
        content: dict[str, Any],
        action_id: str,
        cycle_id: str,
        current_cycle: ERCycleResult,
        result: ERResult,
    ) -> ERState:
        """Process state-specific logic and determine next state."""

        if state == ERState.ORIENT:
            return self._process_orient(content)

        elif state == ERState.HYPOTHESIZE:
            return self._process_hypothesize(content, current_cycle)

        elif state == ERState.EXPERIMENT:
            return await self._process_experiment(content, action_id, cycle_id)

        elif state == ERState.OBSERVE:
            return self._process_observe(content, action_id, current_cycle)

        elif state == ERState.EVALUATE:
            return self._process_evaluate(content, current_cycle)

        elif state == ERState.CONCLUDE:
            return ERState.CONCLUDE

        return ERState.CONCLUDE

    def _process_orient(self, content: dict[str, Any]) -> ERState:
        """Process ORIENT state output."""
        complexity = content.get("complexity", "complex")
        if complexity == "simple":
            return ERState.CONCLUDE
        return ERState.HYPOTHESIZE

    def _process_hypothesize(
        self, content: dict[str, Any], cycle: ERCycleResult
    ) -> ERState:
        """Process HYPOTHESIZE state output."""
        action = content.get("action", "propose")
        statement = content.get("hypothesis_statement", "")
        confidence = content.get("initial_confidence", 0.5)

        if action == "revise" and content.get("parent_hypothesis_id"):
            hid = self.hyp_mgr.revise(
                original_id=content["parent_hypothesis_id"],
                new_statement=statement,
                confidence=confidence,
            )
        else:
            hid = self.hyp_mgr.propose(
                statement=statement,
                confidence=confidence,
            )

        self.hyp_mgr.test(hid)
        cycle.hypothesis_id = hid
        cycle.hypothesis_statement = statement
        return ERState.EXPERIMENT

    async def _process_experiment(
        self,
        content: dict[str, Any],
        action_id: str,
        cycle_id: str,
    ) -> ERState:
        """Process EXPERIMENT state — execute in environment."""
        action_payload = {
            "code": content.get("code"),
            "tool_calls": content.get("tool_calls"),
            "tactic": content.get("tactic"),
        }
        env_result = await self.env.execute(action_payload)
        self._last_experiment_result = env_result

        # Record observation
        output = env_result.get("output", "")
        self.cmm.add_observation(
            action_id=action_id,
            content=str(output)[:10000],
            content_type="experiment_result",
            truncated_summary=str(output)[:500],
        )

        # Update action with output summary
        self.cmm.conn.execute(
            "UPDATE actions SET output_summary = ? WHERE id = ?",
            (str(output)[:500], action_id),
        )
        self.cmm.conn.commit()

        return ERState.OBSERVE

    def _process_observe(
        self,
        content: dict[str, Any],
        action_id: str,
        cycle: ERCycleResult,
    ) -> ERState:
        """Process OBSERVE state — record structured evidence."""
        evidence_content = content.get("evidence_content", "")
        evidence_type = content.get("evidence_type", "observation")

        if evidence_content and cycle.hypothesis_id:
            eid = self.cmm.add_evidence(
                content=evidence_content,
                evidence_type=evidence_type,
                source_action_id=action_id,
            )
            # Provenance: evidence was generated by this action
            self.cmm.add_provenance(
                subject_id=eid,
                subject_type="evidence",
                predicate="wasGeneratedBy",
                object_id=action_id,
                object_type="action",
            )
            cycle.evidence_ids.append(eid)

        return ERState.EVALUATE

    def _process_evaluate(
        self,
        content: dict[str, Any],
        cycle: ERCycleResult,
    ) -> ERState:
        """Process EVALUATE state — update hypothesis and decide next step."""
        hypothesis_id = content.get("hypothesis_id") or cycle.hypothesis_id
        relation = content.get("evidence_relation", "neutral")
        strength = content.get("evidence_strength", 0.5)
        verdict = content.get("hypothesis_verdict", "continue_testing")
        sufficient = content.get("sufficient_for_task", False)

        cycle.verdict = verdict

        # Link evidence to hypothesis with evaluation
        if hypothesis_id and cycle.evidence_ids:
            for eid in cycle.evidence_ids:
                self.cmm.link_evidence_hypothesis(
                    evidence_id=eid,
                    hypothesis_id=hypothesis_id,
                    relation=relation,
                    strength=strength,
                )

        # Update hypothesis status
        if hypothesis_id:
            if verdict == "supported":
                self.hyp_mgr.support(hypothesis_id)
            elif verdict == "refuted":
                self.hyp_mgr.refute(hypothesis_id)

        # Determine next state based on evidence conditions
        if sufficient or verdict == "supported":
            return ERState.CONCLUDE
        elif verdict in ("refuted", "revise"):
            return ERState.HYPOTHESIZE
        else:
            return ERState.EXPERIMENT

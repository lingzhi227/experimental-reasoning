"""tau-bench adapter — bridges ER system to tau-bench's agent interface.

tau-bench expects an Agent with solve(env, task_index, max_num_steps) -> SolveResult.
The ER agent operates on a different abstraction: it uses hypothesis-driven reasoning
with structured evidence tracking to decide which tools to call and when.

This adapter:
1. Implements tau-bench's Agent base class
2. Wraps the tau-bench Env as an ER EnvironmentAdapter
3. Runs the ER loop internally, translating between the two interfaces
4. Returns a SolveResult compatible with tau-bench's evaluation

Key difference from baseline agents:
- Baseline: wiki → LLM → tool call → repeat
- ER: wiki → ORIENT (understand customer need) → HYPOTHESIZE (what's the right action?)
      → EXPERIMENT (call tool) → OBSERVE (record result) → EVALUATE → conclude or revise
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# tau-bench imports (add tau-bench to PYTHONPATH or install it)
try:
    from tau_bench.agents.base import Agent as TauBenchAgent
    from tau_bench.envs.base import Env as TauBenchEnv
    from tau_bench.types import (
        Action as TauAction,
        SolveResult,
        RESPOND_ACTION_NAME,
    )
except ImportError:
    # Stub classes for when tau-bench isn't installed
    class TauBenchAgent:  # type: ignore[no-redef]
        def solve(self, env, task_index=None, max_num_steps=30):
            raise NotImplementedError

    class TauBenchEnv:  # type: ignore[no-redef]
        pass

    class TauAction:  # type: ignore[no-redef]
        def __init__(self, name="", kwargs=None):
            self.name = name
            self.kwargs = kwargs or {}

    class SolveResult:  # type: ignore[no-redef]
        def __init__(self, reward=0.0, messages=None, info=None, total_cost=None):
            self.reward = reward
            self.messages = messages or []
            self.info = info or {}
            self.total_cost = total_cost

    RESPOND_ACTION_NAME = "respond"

from ..adapters.model import CLIModelAdapter, ClaudeSession, _parse_json_response
from ..core.cmm import CMMDatabase


class TauBenchEnvironmentAdapter:
    """Wraps a tau-bench Env as an ER EnvironmentAdapter.

    Translates ER tool_calls into tau-bench Actions and returns
    structured observations.
    """

    def __init__(self, env: TauBenchEnv) -> None:
        self.env = env
        self.last_env_response: Any = None
        self.done = False

    async def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call against the tau-bench environment."""
        tool_calls = action.get("tool_calls", [])
        if not tool_calls:
            # If no tool calls, try to interpret as a respond action
            code = action.get("code", "")
            if code:
                return {"output": "", "error": "Code execution not supported in tau-bench", "metrics": {}}
            return {"output": "", "error": "No tool calls provided", "metrics": {}}

        results = []
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args", call.get("kwargs", {}))

            tau_action = TauAction(name=name, kwargs=args)
            try:
                env_response = self.env.step(tau_action)
                self.last_env_response = env_response
                self.done = env_response.done
                results.append({
                    "tool": name,
                    "observation": env_response.observation,
                    "reward": env_response.reward,
                    "done": env_response.done,
                })
            except Exception as e:
                results.append({"tool": name, "error": str(e)})

        output = json.dumps(results, indent=2) if len(results) > 1 else (
            results[0].get("observation", results[0].get("error", ""))
        )
        return {
            "output": output,
            "error": None,
            "metrics": {"done": self.done},
        }


class ERTauBenchAgent(TauBenchAgent):
    """ER-powered agent compatible with tau-bench's Agent interface.

    Instead of simple ReAct or tool-calling, uses the ER loop
    with hypothesis-driven reasoning and evidence tracking.
    """

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str = "sonnet",
        backend: str = "claude",
        temperature: float = 0.0,
    ) -> None:
        self.tools_info = tools_info
        self.wiki = wiki
        self.model_adapter = CLIModelAdapter(backend=backend, model=model)

    def solve(
        self,
        env: TauBenchEnv,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        """Solve a tau-bench task using the ER reasoning loop.

        This is a synchronous interface (required by tau-bench), so we
        implement the ER loop inline rather than using the async ERLoop.
        """
        import asyncio
        return asyncio.run(
            self._solve_async(env, task_index, max_num_steps)
        )

    async def _solve_async(
        self,
        env: TauBenchEnv,
        task_index: Optional[int],
        max_num_steps: int,
    ) -> SolveResult:
        """Async implementation of the ER-powered solve.

        Uses ClaudeSession with --session-id / --resume so Claude manages
        conversation context internally. Each step only sends NEW information
        (tool result or customer response), not the full history.
        """
        # Reset environment
        env_reset = env.reset(task_index=task_index)
        obs = env_reset.observation
        info = env_reset.info.model_dump()
        reward = 0.0

        # Build tools description and session system prompt
        tools_desc = self._format_tools_for_prompt()
        system = self._build_session_system_prompt(tools_desc)

        # Create a Claude session — context managed by Claude internally
        session = ClaudeSession(model=self.model_adapter.model, system=system)
        logger.info("Created session %s (model=%s)", session.session_id[:8], session.model)

        # Conversation messages (for tau-bench result — NOT sent to LLM)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]

        # CMM for this task
        cmm = CMMDatabase(db_path=":memory:")
        cmm.initialize()

        done = False
        step = 0
        next_prompt = ""  # built after each action for the next step

        # Compact tool name list — appended to each step to prevent hallucination
        tool_names = [t.get("function", {}).get("name", "") for t in self.tools_info if t.get("function", {}).get("name")]
        tool_names_hint = "Available tools: " + ", ".join(tool_names)

        while step < max_num_steps and not done:
            step += 1

            # Build prompt: first step gets full context, subsequent only new info
            if step == 1:
                prompt = (
                    f"Customer: {obs}\n\n"
                    f"{tool_names_hint}\n"
                    f"Step {step}/{max_num_steps}. Decide your next action. Respond in English."
                )
            else:
                prompt = (
                    f"{next_prompt}\n\n"
                    f"{tool_names_hint}\n"
                    f"Step {step}/{max_num_steps}. Decide your next action. Respond in English."
                )

            # Call LLM via session (resume handles context)
            logger.info("Step %d/%d — sending to session %s...", step, max_num_steps, session.session_id[:8])
            content = await asyncio.to_thread(session.send_json, prompt)

            # Parse action from response
            action_name, action_kwargs = self._parse_action(content)

            # Execute action
            logger.info("Step %d — action: %s(%s)", step, action_name, json.dumps(action_kwargs)[:100])
            tau_action = TauAction(name=action_name, kwargs=action_kwargs)
            env_response = env.step(tau_action)
            logger.info("Step %d — result: %s (done=%s)", step, env_response.observation[:100], env_response.done)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            done = env_response.done

            # Record in CMM
            action_id = cmm.log_action(
                state_name="experiment",
                tactic_name=action_name,
                input_summary=json.dumps(action_kwargs)[:200],
                output_summary=env_response.observation[:200],
            )
            cmm.add_observation(
                action_id=action_id,
                content=env_response.observation[:2000],
            )

            # Record evidence if the response has reasoning
            reasoning = content.get("reasoning", "")
            if reasoning:
                cmm.add_evidence(
                    content=reasoning[:500],
                    evidence_type="policy_check",
                    source_action_id=action_id,
                )

            # Update tau-bench messages and build next_prompt for session
            if action_name != RESPOND_ACTION_NAME:
                messages.append({
                    "role": "assistant",
                    "content": json.dumps({"action": action_name, "arguments": action_kwargs}),
                })
                messages.append({
                    "role": "user",
                    "content": f"API output: {env_response.observation}",
                })
                obs_text = env_response.observation
                # When tool returns error, nudge agent to analyze why
                if obs_text.startswith("Error:") or obs_text.startswith("Unknown action"):
                    next_prompt = (
                        f"[Tool Result] {action_name}({json.dumps(action_kwargs)}) returned:\n{obs_text}\n\n"
                        f"Analyze why this failed. Check parameter names, values, and formats carefully against the tool description."
                    )
                else:
                    next_prompt = f"[Tool Result] {action_name} returned:\n{obs_text}"
            else:
                messages.append({
                    "role": "assistant",
                    "content": action_kwargs.get("content", ""),
                })
                messages.append({
                    "role": "user",
                    "content": env_response.observation,
                })
                if not done:
                    next_prompt = f"Customer: {env_response.observation}"

        cmm.close()
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
        )

    def _build_session_system_prompt(self, tools_desc: str) -> str:
        """Build the system prompt for the Claude session.

        This is set ONCE at session creation. Includes wiki, tools, and
        ER reasoning instructions. Claude manages conversation context.
        """
        return f"""\
You are an Experimental Reasoning agent for customer service tasks.
You communicate with customers in English.

## Policies / Wiki
{self.wiki}

## Available Tools
{tools_desc}

## Reasoning Process
For each step, reason through:
1. ORIENT: What does the customer need right now?
2. HYPOTHESIZE: What action best serves them? Does it comply with policy?
3. Decide: Execute a tool call or respond to the customer.

## CRITICAL RULES
- Before every action, verify it complies with the policies above.
- You MUST actually call tools to perform actions — never claim you did something without calling the tool first.
- If the customer asks you to cancel/modify/return an order, call the appropriate tool BEFORE responding to confirm.
- Track what information you've gathered and what's still needed.

## Output Format
You MUST respond with a single valid JSON object. No markdown, no code blocks, no explanations — just raw JSON.

Required fields:
- "reasoning": Step-by-step reasoning (ORIENT then HYPOTHESIZE)
- "action_name": The tool to call, or "respond" to reply to customer
- "action_kwargs": Arguments for the action

When action_name is "respond", action_kwargs MUST have a "content" key with your message to the customer.
Only include the customer-facing message in action_kwargs.content — NOT your internal reasoning.

Example for respond:
{{"reasoning": "Customer wants to cancel order. I've confirmed it and executed cancellation.", "action_name": "respond", "action_kwargs": {{"content": "Your order has been cancelled successfully."}}}}

Example for tool call:
{{"reasoning": "Need to look up the customer's account first.", "action_name": "find_user_id_by_name_zip", "action_kwargs": {{"first_name": "John", "last_name": "Doe", "zip": "12345"}}}}
"""

    def _format_tools_for_prompt(self) -> str:
        """Format tools_info into a readable prompt section."""
        lines = []
        for tool in self.tools_info:
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])

            param_strs = []
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                pdesc = pinfo.get("description", "")
                ptype = pinfo.get("type", "?")
                # Include enum values if present — critical for constrained params
                enum_vals = pinfo.get("enum")
                if enum_vals:
                    enum_str = ", ".join(f'"{v}"' for v in enum_vals)
                    param_strs.append(f"    - {pname}: {ptype}{req} — {pdesc} [valid values: {enum_str}]")
                else:
                    param_strs.append(f"    - {pname}: {ptype}{req} — {pdesc}")

            lines.append(f"- **{name}**: {desc}")
            if param_strs:
                lines.extend(param_strs)
        return "\n".join(lines)

    def _parse_action(self, content: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Parse action from LLM response content."""
        action_name = content.get("action_name", RESPOND_ACTION_NAME)
        action_kwargs = content.get("action_kwargs", {})

        # Ensure kwargs is a dict
        if isinstance(action_kwargs, str):
            try:
                action_kwargs = json.loads(action_kwargs)
            except json.JSONDecodeError:
                action_kwargs = {"content": action_kwargs}

        # For respond action, ensure "content" key exists (tau-bench requires it)
        if action_name == RESPOND_ACTION_NAME:
            if "content" not in action_kwargs:
                # Try "message" key (common LLM output), then fall back to response field
                action_kwargs["content"] = (
                    action_kwargs.pop("message", None)
                    or content.get("response", "I can help you with that.")
                )

        return action_name, action_kwargs

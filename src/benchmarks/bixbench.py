"""BixBench adapter — bridges ER system to BixBench's evaluation framework.

Uses ClaudeSession with session/resume for multi-turn code generation
and execution. The agent iteratively:
1. Explores data files (EDA)
2. Forms hypotheses and writes analysis code
3. Observes execution results
4. Refines analysis based on evidence
5. Produces a final answer

Key difference from baseline agents:
- Baseline: question → write full analysis code → maybe retry on error
- ER: question → ORIENT (EDA, understand data) → HYPOTHESIZE (analysis plan)
      → EXPERIMENT (targeted code) → OBSERVE (structured results)
      → EVALUATE (does it answer the question?) → conclude or iterate
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from ..adapters.model import ClaudeSession, _parse_json_response
from ..adapters.environment import LocalPythonEnvironment, DockerPythonEnvironment
from ..core.cmm import CMMDatabase
from ..knowledge.bioinformatics import (
    get_bioinformatics_knowledge,
    get_docker_bioinformatics_knowledge,
)
from ..tactics.base import bioinformatics_tactics


class BixBenchEnvironment(LocalPythonEnvironment):
    """Extended Python environment for BixBench tasks.

    Sets up the working directory with the capsule data files
    and pre-imports common data science packages.
    """

    def __init__(
        self,
        work_dir: str | Path,
        data_files: list[str] | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds, working_dir=str(work_dir))
        self.work_dir = Path(work_dir)
        self.data_files = data_files or []

        # Pre-populate namespace with common imports
        self._namespace.update({
            "__builtins__": __builtins__,
        })
        # Set working directory for data access
        self._run_code(
            f"import os; os.chdir({str(self.work_dir)!r})\n"
            "import warnings; warnings.filterwarnings('ignore')"
        )

    async def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute code with BixBench-aware error handling."""
        code = action.get("code")
        if not code:
            return {"output": "", "error": "No code provided", "metrics": {}}

        # Add data file path hints if the code references data loading
        result = await super().execute(action)

        # Extract any _answer variable set by the code
        answer = self._namespace.get("_answer")
        if answer is not None:
            result["metrics"]["answer"] = str(answer)

        return result


class BixBenchTask:
    """Represents a single BixBench task."""

    def __init__(
        self,
        question_id: str,
        question: str,
        ideal: str,
        eval_mode: str | None = None,
        distractors: list[str] | None = None,
        data_folder: str | None = None,
        capsule_uuid: str | None = None,
    ) -> None:
        self.question_id = question_id
        self.question = question
        self.ideal = ideal
        self.eval_mode = eval_mode
        self.distractors = distractors or []
        self.data_folder = data_folder
        self.capsule_uuid = capsule_uuid


class BixBenchERRunner:
    """Session-based ER runner for BixBench tasks.

    Uses ClaudeSession with session/resume so the model maintains full
    conversation context across steps. Each step either executes code
    or produces a final answer.

    Usage:
        runner = BixBenchERRunner(model="sonnet")
        result = await runner.run_task(task, work_dir="/path/to/capsule/data")
    """

    def __init__(
        self,
        model: str = "sonnet",
        max_steps: int = 25,
        use_docker: bool = False,
        docker_image: str = "bixbench:enhanced",
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self.use_docker = use_docker
        self.docker_image = docker_image

    # JSON format reminder appended to every step prompt
    _JSON_REMINDER = (
        '\n\nREMINDER: Respond with a raw JSON object only. '
        '{"reasoning":"...","action":"code","code":"..."} or '
        '{"reasoning":"...","action":"answer","answer":"..."}'
    )

    # Recovery prompt sent when model outputs non-JSON
    _JSON_RECOVERY_PROMPT = (
        "Your previous response was not valid JSON. "
        "You MUST restate your response as a single JSON object with NO markdown formatting:\n"
        '{"reasoning": "...", "action": "code", "code": "..."}\n'
        "OR\n"
        '{"reasoning": "...", "action": "answer", "answer": "..."}\n'
        "Respond NOW with valid JSON only."
    )

    async def run_task(
        self,
        task: BixBenchTask,
        work_dir: str | Path,
    ) -> dict[str, Any]:
        """Run ER on a single BixBench task using ClaudeSession.

        The agent iteratively writes and executes code, observing results
        and refining its analysis until it produces a final answer.

        Includes:
        - JSON format recovery (up to 2 retries per step on format drift)
        - Hypothesis tracking via extended JSON fields
        - Enhanced CMM logging with hypothesis changes
        - Task-level retry on complete session failure
        """
        try:
            return await self._run_task_inner(task, work_dir)
        except Exception as e:
            logger.warning(
                "Task %s session failed: %s — retrying with new session",
                task.question_id, e,
            )
            try:
                return await self._run_task_inner(task, work_dir)
            except Exception as e2:
                logger.error("Task %s retry also failed: %s", task.question_id, e2)
                raise

    async def _run_task_inner(
        self,
        task: BixBenchTask,
        work_dir: str | Path,
    ) -> dict[str, Any]:
        """Core task execution logic."""
        work_dir = Path(work_dir)

        # Record original files (for cleanup of generated artifacts after task)
        original_files = set()
        if work_dir.exists():
            original_files = {
                f.name for f in work_dir.iterdir()
            }

        # List data files in work directory
        data_files = []
        if work_dir.exists():
            data_files = [
                f.name for f in work_dir.iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]

        # Set up code execution environment
        if self.use_docker:
            environment = DockerPythonEnvironment(
                docker_image=self.docker_image,
                work_dir=work_dir,
                timeout_seconds=300,
            )
            logger.info("Using Docker environment: %s", self.docker_image)
        else:
            environment = BixBenchEnvironment(work_dir=work_dir, data_files=data_files)

        # Set up CMM for evidence tracking
        db_path = Path(tempfile.mkdtemp()) / f"{task.question_id}.sqlite"
        cmm = CMMDatabase(db_path=db_path)
        cmm.initialize()

        # Build system prompt and create session
        system = self._build_system_prompt(task, data_files, docker=self.use_docker)
        session = ClaudeSession(model=self.model, system=system)
        logger.info(
            "Created session %s for task %s (model=%s)",
            session.session_id[:8], task.question_id, self.model,
        )

        # First prompt: the research question
        first_prompt = (
            f"Research Question: {task.question}\n\n"
            f"Available data files in working directory ({work_dir}):\n"
        )
        for f in sorted(data_files):
            size = (work_dir / f).stat().st_size
            first_prompt += f"  - {f} ({size:,} bytes)\n"
        first_prompt += (
            "\nStep 1/{max_steps}. Begin your analysis. "
            "Start by exploring the data to understand its structure."
            "{reminder}"
        ).format(max_steps=self.max_steps, reminder=self._JSON_REMINDER)

        step = 0
        answer = None
        state_trace = []
        hypothesis_trace = []
        next_prompt = first_prompt

        try:
            while step < self.max_steps and answer is None:
                step += 1

                # Send to session (resume handles context)
                logger.info(
                    "Step %d/%d — sending to session %s...",
                    step, self.max_steps, session.session_id[:8],
                )
                content = await asyncio.to_thread(session.send_json, next_prompt)

                # --- JSON format recovery ---
                if "raw_response" in content:
                    raw_text = content["raw_response"]
                    logger.warning(
                        "Step %d — JSON parse failed, attempting recovery. Raw: %s",
                        step, raw_text[:200],
                    )
                    # Try to force-extract code or answer from raw text before recovery prompt
                    content = self._force_extract_from_raw(raw_text)
                    if content is None:
                        # Send recovery prompt (up to 2 attempts, not counted as steps)
                        for recovery_attempt in range(2):
                            recovery_content = await asyncio.to_thread(
                                session.send_json, self._JSON_RECOVERY_PROMPT
                            )
                            if "raw_response" not in recovery_content:
                                content = recovery_content
                                logger.info(
                                    "Step %d — JSON recovery succeeded on attempt %d",
                                    step, recovery_attempt + 1,
                                )
                                break
                            logger.warning(
                                "Step %d — recovery attempt %d failed: %s",
                                step, recovery_attempt + 1,
                                recovery_content.get("raw_response", "")[:200],
                            )
                            # Last resort: try force-extract from recovery response
                            content = self._force_extract_from_raw(
                                recovery_content.get("raw_response", "")
                            )
                            if content is not None:
                                break
                        if content is None:
                            # All recovery failed — skip this step
                            logger.error("Step %d — all JSON recovery failed, skipping", step)
                            state_trace.append("json_error")
                            next_prompt = (
                                "I could not parse your response. Please continue your analysis. "
                                "Remember: respond with a raw JSON object ONLY.\n\n"
                                f"Step {step + 1}/{self.max_steps}."
                                f"{self._JSON_REMINDER}"
                            )
                            continue

                # Parse response
                action = content.get("action", "code")
                state_trace.append(action)

                # Track hypothesis changes
                hypothesis = content.get("current_hypothesis", "")
                h_status = content.get("hypothesis_status", "")
                if hypothesis:
                    hypothesis_trace.append({
                        "step": step,
                        "hypothesis": hypothesis,
                        "status": h_status,
                    })
                    logger.info(
                        "Step %d — hypothesis [%s]: %s",
                        step, h_status, hypothesis[:100],
                    )

                if action == "answer":
                    # Final answer — clean markdown/explanation pollution
                    raw_ans = str(content.get("answer", ""))
                    answer = self._clean_answer(raw_ans, task.eval_mode or "str_verifier")
                    reasoning = content.get("reasoning", "")
                    evidence = content.get("evidence_summary", "")

                    # Answer validation: if cleaned answer is still long text,
                    # ask for a concise re-answer (costs 1 extra API call)
                    if len(answer) > 50 and task.eval_mode != "llm_verifier":
                        logger.warning(
                            "Step %d — answer too long (%d chars), requesting concise value",
                            step, len(answer),
                        )
                        refine_content = await asyncio.to_thread(
                            session.send_json,
                            "Your answer is too verbose. Respond with ONLY the precise value "
                            "(a number, percentage, gene name, or short phrase). "
                            "No explanations, no units unless asked.\n"
                            '{"reasoning": "...", "action": "answer", "answer": "VALUE_ONLY"}'
                        )
                        if "raw_response" not in refine_content:
                            refined = str(refine_content.get("answer", ""))
                            if refined:
                                answer = self._clean_answer(refined, task.eval_mode or "str_verifier")
                                logger.info("Step %d — refined answer: %s", step, answer[:100])

                    logger.info(
                        "Step %d — ANSWER: %s (reasoning: %s)",
                        step, answer[:100], reasoning[:100],
                    )

                    # Record in CMM
                    action_id = cmm.log_action(
                        state_name="conclude",
                        tactic_name="answer",
                        input_summary=task.question[:200],
                        output_summary=answer[:200],
                    )
                    cmm.add_evidence(
                        content=f"Final answer: {answer}. Reasoning: {reasoning[:500]}. Evidence: {evidence[:500]}",
                        evidence_type="conclusion",
                        source_action_id=action_id,
                    )

                elif action == "code":
                    # Execute code
                    code = content.get("code", "")
                    reasoning = content.get("reasoning", "")
                    logger.info(
                        "Step %d — CODE (%d chars): %s",
                        step, len(code), reasoning[:100],
                    )

                    if not code:
                        next_prompt = (
                            "You returned action='code' but no code was provided. "
                            "Please provide Python code to execute, or action='answer' with your final answer.\n\n"
                            f"Step {step + 1}/{self.max_steps}."
                            f"{self._JSON_REMINDER}"
                        )
                        continue

                    # Execute
                    exec_result = await environment.execute({"code": code})
                    output = exec_result.get("output", "")
                    error = exec_result.get("error")
                    metrics = exec_result.get("metrics", {})

                    # Record in CMM with hypothesis context
                    action_id = cmm.log_action(
                        state_name="experiment",
                        tactic_name="code_execution",
                        input_summary=f"[H: {hypothesis[:80]}] {code[:120]}",
                        output_summary=(output or str(error))[:200],
                    )
                    cmm.add_observation(
                        action_id=action_id,
                        content=(output or str(error))[:2000],
                    )

                    # Check if _answer was set
                    if "answer" in metrics:
                        answer = str(metrics["answer"])
                        logger.info("Step %d — _answer variable set: %s", step, answer[:100])

                        cmm.add_evidence(
                            content=f"_answer variable set to: {answer}",
                            evidence_type="conclusion",
                            source_action_id=action_id,
                        )

                        # Inform model and let it confirm
                        next_prompt = (
                            f"[Code Execution Result]\n"
                            f"Output:\n{output[:3000]}\n"
                        )
                        if error:
                            next_prompt += f"\nStderr:\n{error[:1000]}\n"
                        next_prompt += (
                            f"\n_answer was set to: {answer}\n\n"
                            f"If this is correct, respond with action='answer' and your final answer value. "
                            f"If you need to refine, respond with action='code'.\n\n"
                            f"Step {step + 1}/{self.max_steps}."
                            f"{self._JSON_REMINDER}"
                        )
                        # Don't break yet — let model confirm
                        answer = None
                        continue

                    # Build next prompt with execution result
                    if error:
                        next_prompt = (
                            f"[Code Execution Error]\n{error[:2000]}\n\n"
                            f"Analyze why this failed and try a different approach.\n\n"
                            f"Step {step + 1}/{self.max_steps}."
                            f"{self._JSON_REMINDER}"
                        )
                    else:
                        truncated = output[:3000]
                        if len(output) > 3000:
                            truncated += "\n... [output truncated]"
                        next_prompt = (
                            f"[Code Execution Result]\n{truncated}\n\n"
                            f"Step {step + 1}/{self.max_steps}. "
                            f"Analyze results and decide next step."
                            f"{self._JSON_REMINDER}"
                        )

                else:
                    # Unknown action — nudge model
                    next_prompt = (
                        f"Unknown action '{action}'. Use action='code' to execute Python code, "
                        f"or action='answer' to provide your final answer.\n\n"
                        f"Step {step + 1}/{self.max_steps}."
                        f"{self._JSON_REMINDER}"
                    )

            # If we ran out of steps without an answer, try to extract one
            if answer is None:
                logger.warning(
                    "Task %s: reached max steps without answer, asking for forced conclusion",
                    task.question_id,
                )
                content = await asyncio.to_thread(
                    session.send_json,
                    "You have run out of steps. You MUST provide your best answer NOW. "
                    'Respond with ONLY this JSON: {"reasoning":"...","action":"answer","answer":"YOUR_VALUE"}\n'
                    "The answer field must contain ONLY the precise value — no explanations.",
                )
                answer = self._clean_answer(str(content.get("answer", content.get("final_answer", ""))), task.eval_mode or "str_verifier")
                # If still no answer from JSON, try raw extraction
                if not answer and "raw_response" in content:
                    answer = self._clean_answer(self._extract_answer_from_raw(content["raw_response"]), task.eval_mode or "str_verifier")

            cmm_stats = cmm.stats()
            cmm.close()

            return {
                "question_id": task.question_id,
                "agent_answer": answer or "",
                "ideal_answer": task.ideal,
                "eval_mode": task.eval_mode,
                "total_steps": step,
                "state_trace": state_trace,
                "hypothesis_trace": hypothesis_trace,
                "cmm_stats": cmm_stats,
            }
        finally:
            # Clean up Docker environment if used
            if hasattr(environment, "cleanup"):
                environment.cleanup()
            # Remove generated artifacts (R scripts, CSV outputs, etc.)
            # so they don't pollute the next task sharing this capsule
            if work_dir.exists():
                for f in work_dir.iterdir():
                    if f.name not in original_files:
                        try:
                            if f.is_file():
                                f.unlink()
                            elif f.is_dir():
                                import shutil
                                shutil.rmtree(f)
                        except OSError:
                            pass

    @staticmethod
    def _force_extract_from_raw(raw: str) -> dict[str, Any] | None:
        """Try to extract a usable action from raw non-JSON text.

        Looks for code blocks or answer-like patterns in free text.
        Returns a dict compatible with the action format, or None.
        """
        import re

        # Try to find a python code block
        code_match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
        if code_match:
            return {
                "reasoning": "extracted from non-JSON response",
                "action": "code",
                "code": code_match.group(1).strip(),
            }

        # Try to find an answer pattern like "answer: X" or "the answer is X"
        answer_match = re.search(
            r"(?:answer|result)\s*(?:is|=|:)\s*[\"']?([^\n\"']+)[\"']?",
            raw,
            re.IGNORECASE,
        )
        if answer_match:
            return {
                "reasoning": "extracted from non-JSON response",
                "action": "answer",
                "answer": answer_match.group(1).strip(),
            }

        return None

    @staticmethod
    def _extract_answer_from_raw(raw: str) -> str:
        """Last-resort answer extraction from raw text."""
        import re
        # Look for quoted values or "answer is X" patterns
        match = re.search(
            r"(?:answer|result)\s*(?:is|=|:)\s*[\"']?([^\n\"']+)[\"']?",
            raw,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        # Return first non-empty line as desperation fallback
        for line in raw.strip().splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "/", "```")):
                return line
        return ""

    @staticmethod
    def _clean_answer(raw_answer: str, eval_mode: str = "str_verifier") -> str:
        """Clean model's answer to extract just the precise value.

        For llm_verifier: return full text (judge LLM needs context).
        For str_verifier/range_verifier: extract precise value from full text.
        Priority: bold > "answer is X" > approx > percentage > comma-number > first line.
        """
        import re
        if not raw_answer:
            return ""

        full_text = raw_answer.strip()

        # For llm_verifier, return full text (judge LLM compares semantically)
        if eval_mode == "llm_verifier":
            return re.sub(r"\*\*(.+?)\*\*", r"\1", full_text)

        # 1. Strip markdown bold — if there's a **value**, that's the intended answer
        bold_match = re.search(r"\*\*(.+?)\*\*", full_text)
        if bold_match:
            val = bold_match.group(1).strip()
            if re.match(r'^-?\d[\d,.]*%?$', val) or len(val) <= 30:
                return val

        # 2. First line processing
        first_line = full_text.split("\n")[0].strip()
        first_line = re.sub(r"\*\*(.+?)\*\*", r"\1", first_line)
        first_line = re.sub(r"\*(.+?)\*", r"\1", first_line)
        first_line = first_line.strip("`\"'")
        if first_line.endswith(".") and not re.match(r"^\d+\.\d*$", first_line):
            first_line = first_line[:-1].strip()

        # 3. If first line is a bare value, return immediately
        if re.match(r'^-?\d[\d,.]*(?:[eE][+-]?\d+)?%?$', first_line):
            return first_line
        if len(first_line) <= 20 and ' ' not in first_line:
            return first_line

        # 4. PRIORITY: Search for "answer is X" first (highest confidence keyword)
        answer_keyed = re.search(
            r"\banswer\s*(?:is|=|:)\s*[\"'`]?(-?\d[\d,.]*(?:[eE][+-]?\d+)?%?)",
            full_text, re.IGNORECASE,
        )
        if answer_keyed:
            return answer_keyed.group(1).strip()

        # 5. Search for "result/total is/=/: X" (lower priority than "answer")
        other_keyed = re.search(
            r"(?:result|total|count)\s*(?:is|=|:)\s*[\"'`]?"
            r"(-?\d[\d,.]*(?:[eE][+-]?\d+)?%?)",
            full_text, re.IGNORECASE,
        )
        if other_keyed:
            return other_keyed.group(1).strip()

        # 6. Search for "approximately X" or "≈ X" patterns
        approx = re.search(
            r"(?:approximately|approx\.?|≈|~)\s*(-?\d[\d,.]*(?:[eE][+-]?\d+)?%?)",
            full_text, re.IGNORECASE,
        )
        if approx:
            return approx.group(1).strip()

        # 7. Search for a single percentage in full text
        pcts = re.findall(r"(\d+\.?\d*%)", full_text)
        if len(pcts) == 1:
            return pcts[0]
        # If multiple percentages, take the last one (usually the final answer)
        if pcts:
            return pcts[-1]

        # 8. Search for comma-separated numbers like "4,550" or "1,234"
        comma_nums = re.findall(r"\b(\d{1,3}(?:,\d{3})+)\b", full_text)
        if len(comma_nums) == 1:
            return comma_nums[0]

        return first_line

    def _build_system_prompt(
        self, task: BixBenchTask, data_files: list[str], docker: bool = False
    ) -> str:
        """Build system prompt. Docker mode is concise; host mode includes domain knowledge."""
        if docker:
            return self._build_docker_prompt(task)
        else:
            return self._build_host_prompt(task, data_files)

    # ── Tactic-based tool selector ──────────────────────────────────────
    # Maps question keywords → mandatory tool directives injected into prompt.
    # This prevents the agent from choosing sub-optimal tools.

    _TACTIC_RULES = [
        {
            "keywords": ["deseq2", "differential expression", "differentially expressed",
                         "de analysis", "de genes", "log2foldchange", "logfold",
                         "dispersion"],
            "tactic": "deseq2",
            "directive": (
                "## MANDATORY: R DESeq2 via subprocess\n"
                "This task requires DESeq2. You MUST use R DESeq2 via subprocess.\n"
                "PyDESeq2 is DISABLED and will error if imported.\n\n"
                "Pattern:\n"
                "```python\n"
                'r_script = """\n'
                "library(DESeq2)\n"
                'counts <- read.csv("/workspace/FILE.csv", row.names=1, check.names=FALSE)\n'
                'coldata <- data.frame(condition=c(...))\n'
                "dds <- DESeqDataSetFromMatrix(countData=counts, colData=coldata, design=~condition)\n"
                "dds <- DESeq(dds)\n"
                '# Dispersion BEFORE shrinkage: mcols(dds)$dispGeneEst\n'
                '# Dispersion AFTER shrinkage: dispersions(dds)\n'
                "res <- results(dds, contrast=c('condition','A','B'))\n"
                'write.csv(as.data.frame(res), "/workspace/deseq2_results.csv")\n'
                '"""\n'
                'with open("/workspace/analysis.R", "w") as f:\n'
                "    f.write(r_script)\n"
                "import subprocess\n"
                'result = subprocess.run(["Rscript", "/workspace/analysis.R"], capture_output=True, text=True, timeout=300)\n'
                "print(result.stdout)\n"
                'if result.returncode != 0: print("STDERR:", result.stderr)\n'
                "```\n"
            ),
        },
        {
            "keywords": ["phylogenetic", "phylogeny", "tree", "alignment", "mafft",
                         "iqtree", "newick"],
            "tactic": "phylogenetics",
            "directive": (
                "## MANDATORY: CLI bioinformatics tools\n"
                "Use mafft (alignment), clipkit (trimming), iqtree2 (tree building), "
                "phykit (tree analysis) via subprocess.\n"
            ),
        },
        {
            "keywords": ["enrichment", "go analysis", "pathway", "gsea", "gene ontology",
                         "kegg"],
            "tactic": "enrichment",
            "directive": (
                "## MANDATORY: Enrichment tools\n"
                "Use R clusterProfiler via subprocess (preferred) or Python gseapy.\n"
            ),
        },
        {
            "keywords": ["variant", "vaf", "mutation", "vcf", "allele frequency",
                         "snp", "indel"],
            "tactic": "variant_analysis",
            "directive": (
                "## MANDATORY: Variant analysis\n"
                "Use pandas for Excel/CSV variant data. For VCF files, use bcftools via subprocess.\n"
                "Watch for multi-level Excel headers (skiprows may be needed).\n"
            ),
        },
    ]

    def _detect_tactics(self, question: str) -> list[dict]:
        """Detect which tactics apply based on question keywords."""
        q_lower = question.lower()
        matched = []
        for rule in self._TACTIC_RULES:
            if any(kw in q_lower for kw in rule["keywords"]):
                matched.append(rule)
        return matched

    def _build_docker_prompt(self, task: BixBenchTask) -> str:
        """Concise Docker system prompt with tactic-driven tool selection."""
        # Detect applicable tactics from the question
        tactics = self._detect_tactics(task.question)
        tactic_names = [t["tactic"] for t in tactics]
        tactic_directives = "\n".join(t["directive"] for t in tactics)

        logger.info(
            "Task %s — detected tactics: %s",
            task.question_id, tactic_names or ["general"],
        )

        return f"""\
You are an expert bioinformatics data analyst. You iteratively write and execute Python code to answer a research question. The working directory contains the data files.

## Response Format
Respond with a single raw JSON object (no markdown, no wrapping):

Code: {{"reasoning": "...", "action": "code", "code": "..."}}
Answer: {{"reasoning": "...", "action": "answer", "answer": "PRECISE_VALUE_ONLY"}}

{tactic_directives if tactic_directives else "## Available Tools"}
{"" if tactic_directives else "pandas, numpy, scipy, statsmodels, lifelines, sklearn, biopython, R (via subprocess), CLI tools (mafft, iqtree2, blast, samtools, bcftools, bedtools, phykit)."}

## General Rules
- Explore data first: df.head(), df.shape, df.columns, df.dtypes
- Write focused code testing ONE thing at a time
- Use print() to see intermediate results
- Answer must be ONLY the precise value — no explanations, no units
- R column names: use `check.names=FALSE` in read.csv to preserve original names

## Eval mode: '{task.eval_mode or "str_verifier"}'
{"Provide an exact string/number match." if task.eval_mode == "str_verifier" else ""}\
{"Provide a number within the specified range." if task.eval_mode == "range_verifier" else ""}
"""

    def _build_host_prompt(
        self, task: BixBenchTask, data_files: list[str]
    ) -> str:
        """Host-mode prompt with full domain knowledge (no Docker tools available)."""
        domain_knowledge = get_bioinformatics_knowledge()
        tactics_section = bioinformatics_tactics().to_prompt_section()

        return f"""\
You are an expert bioinformatics data analyst. You iteratively write and execute Python code to answer a research question. The working directory contains the data files.

## Response Format
Respond with a single raw JSON object (no markdown, no wrapping):

Code: {{"reasoning": "...", "action": "code", "code": "..."}}
Answer: {{"reasoning": "...", "action": "answer", "answer": "PRECISE_VALUE_ONLY"}}

## Rules
- Explore data first: df.head(), df.shape, df.columns, df.dtypes
- Write focused code testing ONE thing at a time
- Available: pandas, numpy, scipy, sklearn, matplotlib, statsmodels, openpyxl, lifelines, pydeseq2
- pip install: subprocess.run([sys.executable, '-m', 'pip', 'install', 'pkg', '-q'])
- NEVER install R or conda — use Python alternatives
- Use print() to see intermediate results
- Answer must be ONLY the precise value — no explanations, no units

## Eval mode: '{task.eval_mode or "str_verifier"}'
{"Provide an exact string/number match." if task.eval_mode == "str_verifier" else ""}\
{"Provide a number within the specified range." if task.eval_mode == "range_verifier" else ""}

{domain_knowledge}

{tactics_section}
"""

    async def run_batch(
        self,
        tasks: list[tuple[BixBenchTask, str | Path]],
        output_dir: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Run ER on multiple BixBench tasks.

        Args:
            tasks: List of (task, work_dir) tuples
            output_dir: Optional directory to save individual results

        Returns:
            List of result dicts
        """
        results = []
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for i, (task, work_dir) in enumerate(tasks):
            logger.info(
                "Running task %d/%d: %s", i + 1, len(tasks), task.question_id
            )
            try:
                result = await self.run_task(task, work_dir)
                results.append(result)

                if output_dir:
                    out_file = output_dir / f"{task.question_id}.json"
                    with open(out_file, "w") as f:
                        json.dump(result, f, indent=2, default=str)

            except Exception as e:
                logger.error("Task %s failed: %s", task.question_id, e)
                results.append({
                    "question_id": task.question_id,
                    "agent_answer": "",
                    "ideal_answer": task.ideal,
                    "error": str(e),
                })

        return results


def load_bixbench_tasks(
    data_dir: str | Path | None = None,
    split: str = "test",
    limit: int | None = None,
) -> list[BixBenchTask]:
    """Load BixBench tasks from local data or HuggingFace.

    Args:
        data_dir: Local directory with BixBench data (capsules/)
        split: Dataset split ("test", "train")
        limit: Maximum number of tasks to load

    Returns:
        List of BixBenchTask objects
    """
    try:
        import datasets
        ds = datasets.load_dataset("futurehouse/BixBench", split=split)
        task_list = ds.to_list()
    except ImportError:
        logger.warning("datasets library not available, trying local data")
        task_list = _load_local_tasks(data_dir, split)

    if limit:
        task_list = task_list[:limit]

    tasks = []
    for item in task_list:
        tasks.append(BixBenchTask(
            question_id=item.get("question_id", f"q_{len(tasks)}"),
            question=item.get("question", ""),
            ideal=item.get("ideal", ""),
            eval_mode=item.get("eval_mode"),
            distractors=item.get("distractors", []),
            data_folder=item.get("data_folder"),
            capsule_uuid=item.get("capsule_uuid"),
        ))

    return tasks


def _load_local_tasks(
    data_dir: str | Path | None, split: str
) -> list[dict[str, Any]]:
    """Fallback: load tasks from a local JSON file."""
    if data_dir is None:
        return []
    path = Path(data_dir) / f"{split}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []

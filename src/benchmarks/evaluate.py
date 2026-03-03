"""Unified evaluation runner for ER system across benchmarks.

Provides CLI and programmatic interface to run ER agent on
BixBench and tau-bench dev sets with different backends/models.

Usage:
    # CLI
    python -m src.benchmarks.evaluate --benchmark taubench --backend claude --model sonnet --split dev

    # Programmatic
    from src.benchmarks.evaluate import run_evaluation
    results = await run_evaluation("taubench", backend="claude", model="sonnet")
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


async def run_taubench_evaluation(
    backend: str = "claude",
    model: str = "sonnet",
    env_name: str = "retail",
    split: str = "dev",
    task_ids: list[int] | None = None,
    start_index: int = 0,
    end_index: int = -1,
    max_num_steps: int = 30,
    user_backend: str = "claude",
    user_model: str = "haiku",
    output_dir: str = "results/taubench",
) -> dict[str, Any]:
    """Run ER agent on tau-bench tasks.

    Args:
        backend: CLI backend ("claude", "codex", "gemini")
        model: Model name
        env_name: "retail" or "airline"
        split: "dev", "train", or "test"
        task_ids: Specific task IDs to run (None = all)
        start_index: Start index in task list
        end_index: End index (-1 = all)
        max_num_steps: Max steps per task
        user_backend: CLI backend for user simulation
        user_model: CLI model for user simulation
        output_dir: Directory to save results

    Returns:
        Summary dict with metrics
    """
    # Add tau-bench to path
    tau_bench_path = Path.home() / "Code/7-Benchmark/data/tau-bench"
    if str(tau_bench_path) not in sys.path:
        sys.path.insert(0, str(tau_bench_path))

    from tau_bench.envs import get_env
    from .taubench import ERTauBenchAgent
    from .cli_user_sim import CLIUserSimulationEnv

    # Create environment with "human" strategy (avoids API calls)
    env = get_env(
        env_name,
        user_strategy="human",
        user_model="unused",
        task_split=split,
    )
    # Replace with CLI-based user simulator
    env.user = CLIUserSimulationEnv(backend=user_backend, model=user_model)

    # Create ER agent
    agent = ERTauBenchAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=model,
        backend=backend,
    )

    # Determine task indices
    total_tasks = len(env.tasks)
    if task_ids:
        indices = task_ids
    else:
        end = total_tasks if end_index == -1 else min(end_index, total_tasks)
        indices = list(range(start_index, end))

    logger.info(
        "Running tau-bench evaluation: %d tasks, backend=%s, model=%s, env=%s",
        len(indices), backend, model, env_name,
    )

    # Run tasks in parallel — each task gets its own env, agent session, and user sim
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    async def _run_single_task(task_idx: int) -> dict[str, Any]:
        """Run a single task with its own environment and session."""
        logger.info("Starting task %d...", task_idx)

        # Each task gets a fresh environment and user sim
        task_env = get_env(
            env_name,
            user_strategy="human",
            user_model="unused",
            task_split=split,
            task_index=task_idx,
        )
        task_env.user = CLIUserSimulationEnv(backend=user_backend, model=user_model)

        # Each task gets its own agent (= own session)
        task_agent = ERTauBenchAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=model,
            backend=backend,
        )

        start_time = time.time()
        try:
            solve_result = await task_agent._solve_async(task_env, task_idx, max_num_steps)
            duration = time.time() - start_time

            task_result = {
                "task_id": task_idx,
                "reward": solve_result.reward,
                "num_messages": len(solve_result.messages),
                "duration_s": round(duration, 1),
                "total_cost": solve_result.total_cost,
            }

            # Save individual result
            with open(output_path / f"task_{task_idx}.json", "w") as f:
                json.dump({
                    **task_result,
                    "messages": solve_result.messages,
                    "info": solve_result.info,
                }, f, indent=2, default=str)

            logger.info(
                "Task %d done: reward=%.1f | messages=%d | duration=%.1fs",
                task_idx, solve_result.reward, len(solve_result.messages), duration,
            )
            return task_result

        except Exception as e:
            duration = time.time() - start_time
            logger.error("Task %d failed (%.1fs): %s", task_idx, duration, e)
            return {
                "task_id": task_idx,
                "reward": 0.0,
                "error": str(e),
                "duration_s": round(duration, 1),
            }

    # Launch all tasks concurrently
    results = await asyncio.gather(*[_run_single_task(idx) for idx in indices])

    # Compute metrics
    rewards = [r["reward"] for r in results]
    summary = {
        "benchmark": "taubench",
        "env": env_name,
        "split": split,
        "backend": backend,
        "model": model,
        "total_tasks": len(results),
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
        "pass_rate": sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0,
        "results": results,
    }

    # Save summary
    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "tau-bench evaluation complete: avg_reward=%.3f, pass_rate=%.1f%%",
        summary["avg_reward"], summary["pass_rate"] * 100,
    )

    return summary


async def run_bixbench_evaluation(
    backend: str = "claude",
    model: str = "sonnet",
    data_dir: str | None = None,
    split: str = "test",
    limit: int | None = None,
    max_cycles: int = 10,
    max_turns: int = 50,
    output_dir: str = "results/bixbench",
) -> dict[str, Any]:
    """Run ER agent on BixBench tasks.

    Args:
        backend: CLI backend
        model: Model name
        data_dir: BixBench data directory (with capsules)
        split: Dataset split
        limit: Max tasks to run
        max_cycles: Max ER cycles per task
        max_turns: Max turns per task
        output_dir: Directory to save results

    Returns:
        Summary dict with metrics
    """
    from .bixbench import BixBenchERRunner, load_bixbench_tasks

    # Load tasks
    tasks = load_bixbench_tasks(data_dir=data_dir, split=split, limit=limit)
    if not tasks:
        logger.error("No BixBench tasks loaded")
        return {"error": "No tasks loaded"}

    logger.info(
        "Running BixBench evaluation: %d tasks, backend=%s, model=%s",
        len(tasks), backend, model,
    )

    # Create runner
    runner = BixBenchERRunner(
        backend=backend,
        model=model,
        max_cycles=max_cycles,
        max_turns=max_turns,
    )

    # Prepare task/workdir pairs
    capsule_base = Path(data_dir) / "capsules" if data_dir else Path("data/capsules")
    task_pairs = []
    for task in tasks:
        work_dir = capsule_base / (task.capsule_uuid or task.question_id)
        task_pairs.append((task, work_dir))

    # Run
    output_path = Path(output_dir)
    results = await runner.run_batch(task_pairs, output_dir=output_path)

    # Compute metrics
    correct = sum(1 for r in results if _is_correct(r))
    summary = {
        "benchmark": "bixbench",
        "split": split,
        "backend": backend,
        "model": model,
        "total_tasks": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0,
        "avg_confidence": (
            sum(r.get("confidence", 0) for r in results) / len(results)
            if results else 0
        ),
        "avg_turns": (
            sum(r.get("total_turns", 0) for r in results) / len(results)
            if results else 0
        ),
        "results": results,
    }

    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(
        "BixBench evaluation complete: accuracy=%.1f%% (%d/%d)",
        summary["accuracy"] * 100, correct, len(results),
    )

    return summary


def _is_correct(result: dict[str, Any]) -> bool:
    """Simple correctness check (exact match after normalization)."""
    agent = str(result.get("agent_answer", "")).strip().lower()
    ideal = str(result.get("ideal_answer", "")).strip().lower()
    if not agent or not ideal:
        return False
    # Normalize: remove non-alphanumeric
    import re
    agent_norm = re.sub(r"[^a-z0-9]", "", agent)
    ideal_norm = re.sub(r"[^a-z0-9]", "", ideal)
    return agent_norm == ideal_norm or ideal_norm in agent_norm


async def run_evaluation(
    benchmark: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Unified entry point for running evaluations."""
    if benchmark == "taubench":
        return await run_taubench_evaluation(**kwargs)
    elif benchmark == "bixbench":
        return await run_bixbench_evaluation(**kwargs)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="ER Benchmark Evaluation Runner")
    parser.add_argument(
        "--benchmark", required=True, choices=["taubench", "bixbench"],
        help="Which benchmark to evaluate",
    )
    parser.add_argument("--backend", default=None, help="CLI backend (claude/codex/gemini)")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--split", default="dev", help="Dataset split")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max tasks to run")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    # tau-bench specific
    parser.add_argument("--env", default="retail", help="tau-bench env (retail/airline)")
    parser.add_argument("--task-ids", type=int, nargs="+", help="Specific task IDs")
    parser.add_argument("--max-steps", type=int, default=30, help="Max steps per task")
    parser.add_argument("--user-backend", default="claude", help="CLI backend for user sim")
    parser.add_argument("--user-model", default="haiku", help="CLI model for user sim")

    # BixBench specific
    parser.add_argument("--data-dir", default=None, help="BixBench data directory")
    parser.add_argument("--max-cycles", type=int, default=10, help="Max ER cycles")
    parser.add_argument("--max-turns", type=int, default=50, help="Max turns per task")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    kwargs: dict[str, Any] = {}
    if args.backend:
        kwargs["backend"] = args.backend
    if args.model:
        kwargs["model"] = args.model
    kwargs["split"] = args.split

    if args.benchmark == "taubench":
        kwargs["env_name"] = args.env
        if args.task_ids:
            kwargs["task_ids"] = args.task_ids
        kwargs["max_num_steps"] = args.max_steps
        kwargs["user_backend"] = args.user_backend
        kwargs["user_model"] = args.user_model
        kwargs["output_dir"] = args.output_dir or f"results/taubench_{args.env}"
    elif args.benchmark == "bixbench":
        if args.data_dir:
            kwargs["data_dir"] = args.data_dir
        if args.limit:
            kwargs["limit"] = args.limit
        kwargs["max_cycles"] = args.max_cycles
        kwargs["max_turns"] = args.max_turns
        kwargs["output_dir"] = args.output_dir or "results/bixbench"

    summary = asyncio.run(run_evaluation(args.benchmark, **kwargs))

    # Print summary
    print("\n" + "=" * 60)
    print(f"Evaluation Summary: {args.benchmark}")
    print("=" * 60)
    for k, v in summary.items():
        if k != "results":
            print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()

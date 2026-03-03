"""End-to-end test: run ER agent on a single tau-bench retail dev task.

Uses CLI for BOTH the ER agent AND the user simulator (no API keys needed).

Usage:
    python scripts/test_taubench_single.py [--task-id 0] [--backend claude] [--model sonnet]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root and tau-bench to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path.home() / "Code/7-Benchmark/data/tau-bench"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--backend", default="claude")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--user-backend", default="claude",
                        help="CLI backend for user simulation")
    parser.add_argument("--user-model", default="haiku",
                        help="Model for user simulation via CLI")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ER Agent tau-bench Single Task Test (CLI-only, no API)")
    logger.info("=" * 60)
    logger.info(f"Task ID: {args.task_id}")
    logger.info(f"Agent: {args.backend}/{args.model}")
    logger.info(f"User sim: {args.user_backend}/{args.user_model}")
    logger.info(f"Max steps: {args.max_steps}")

    # Import after path setup
    from tau_bench.envs import get_env
    from src.benchmarks.taubench import ERTauBenchAgent
    from src.benchmarks.cli_user_sim import CLIUserSimulationEnv

    # Create environment with "human" strategy (avoids API call in __init__)
    logger.info("Creating tau-bench retail environment (dev split)...")
    env = get_env(
        "retail",
        user_strategy="human",
        user_model="unused",
        task_split="dev",
        task_index=args.task_id,
    )

    # Replace user simulator with CLI-based one
    env.user = CLIUserSimulationEnv(
        backend=args.user_backend,
        model=args.user_model,
    )
    logger.info("User simulator: CLI %s/%s", args.user_backend, args.user_model)

    # Show task info
    task = env.tasks[args.task_id]
    logger.info(f"Task: {task.instruction[:200]}...")
    logger.info(f"Expected actions: {len(task.actions)}")
    for a in task.actions:
        logger.info(f"  - {a.name}({json.dumps(a.kwargs)[:100]})")
    logger.info(f"Expected outputs: {task.outputs}")

    # Create ER agent
    logger.info("Creating ER agent...")
    agent = ERTauBenchAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=args.model,
        backend=args.backend,
    )

    # Run
    logger.info("Running ER agent...")
    start_time = time.time()
    result = agent.solve(env, task_index=args.task_id, max_num_steps=args.max_steps)
    duration = time.time() - start_time

    # Report
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Reward: {result.reward}")
    logger.info(f"Messages: {len(result.messages)}")
    logger.info(f"Duration: {duration:.1f}s")

    # Show conversation
    logger.info("\n--- Conversation ---")
    for i, msg in enumerate(result.messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if content:
            content_preview = str(content)[:200]
        else:
            content_preview = "(tool call)"
        logger.info(f"[{i}] {role}: {content_preview}")

    # Save result
    output_dir = project_root / "results" / "taubench_single"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"task_{args.task_id}.json"
    with open(output_file, "w") as f:
        json.dump({
            "task_id": args.task_id,
            "reward": result.reward,
            "num_messages": len(result.messages),
            "duration_s": round(duration, 1),
            "messages": result.messages,
            "info": result.info,
        }, f, indent=2, default=str)
    logger.info(f"Result saved to {output_file}")

    return result.reward


if __name__ == "__main__":
    reward = main()
    sys.exit(0 if reward > 0 else 1)

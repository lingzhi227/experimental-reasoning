#!/usr/bin/env python3
"""Test BixBench tasks in parallel with controlled concurrency.

Usage:
    python scripts/test_bixbench_parallel.py --start 0 --end 50 --concurrency 5
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def download_capsule(data_dir: Path, zip_filename: str) -> Path:
    """Download and extract a capsule, return extract dir."""
    extract_dir = data_dir / zip_filename.replace(".zip", "")

    if extract_dir.exists() and any(extract_dir.iterdir()):
        return extract_dir

    logger.info("Downloading capsule: %s", zip_filename)
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="futurehouse/BixBench",
        filename=zip_filename,
        local_dir=str(data_dir),
        repo_type="dataset",
    )

    zip_path = data_dir / zip_filename
    extract_dir.mkdir(exist_ok=True)
    shutil.unpack_archive(str(zip_path), str(extract_dir))

    # Flatten: move Data folder contents up
    data_folder = next(
        (p for p in extract_dir.rglob("*") if p.is_dir() and "Data" in p.name),
        None,
    )
    if data_folder:
        for f in data_folder.iterdir():
            dest = extract_dir / f.name
            if f.is_file():
                shutil.copy2(str(f), str(dest))
            elif f.is_dir():
                shutil.copytree(str(f), str(dest), dirs_exist_ok=True)
        shutil.rmtree(data_folder)

    # Remove notebooks and internal capsule dirs
    for nb in extract_dir.rglob("*.ipynb"):
        nb.unlink()
    for d in list(extract_dir.iterdir()):
        if d.is_dir() and d.name.startswith("Capsule"):
            shutil.rmtree(d)

    return extract_dir


async def run_one_task(idx, ds, capsules, model, max_steps, output_dir,
                       use_docker=False, docker_image="bixbench:enhanced"):
    """Run a single task (no concurrency control — caller manages scheduling)."""
    from src.benchmarks.bixbench import BixBenchTask, BixBenchERRunner

    item = ds[idx]
    task = BixBenchTask(
        question_id=item["question_id"],
        question=item["question"],
        ideal=item["ideal"],
        eval_mode=item["eval_mode"],
        distractors=item.get("distractors", []),
        data_folder=item.get("data_folder"),
        capsule_uuid=item.get("capsule_uuid"),
    )

    work_dir = capsules[item["data_folder"]]
    runner = BixBenchERRunner(
        model=model, max_steps=max_steps,
        use_docker=use_docker, docker_image=docker_image,
    )

    logger.info(
        "START [%d] %s — ideal=%s (%s) Q: %s",
        idx, task.question_id, task.ideal, task.eval_mode, task.question[:120],
    )

    start = time.time()
    try:
        result = await runner.run_task(task, work_dir)
        result["task_index"] = idx
        result["duration_s"] = round(time.time() - start, 1)
    except Exception as e:
        logger.error("Task [%d] %s failed: %s", idx, task.question_id, e)
        result = {
            "task_index": idx,
            "question_id": task.question_id,
            "agent_answer": "",
            "ideal_answer": task.ideal,
            "eval_mode": task.eval_mode,
            "error": str(e),
            "duration_s": round(time.time() - start, 1),
        }

    # Save individual result
    with open(output_dir / f"{task.question_id}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(
        "DONE [%d] %s — agent=%s ideal=%s (%s) %.0fs",
        idx, task.question_id,
        str(result.get("agent_answer", ""))[:30],
        task.ideal[:20], task.eval_mode,
        result.get("duration_s", 0),
    )
    return result


async def run_capsule_group(indices, ds, capsules, model, max_steps, output_dir,
                           stagger_s=5, use_docker=False, docker_image="bixbench:enhanced"):
    """Run all tasks for one capsule SEQUENTIALLY (avoids session race conditions)."""
    results = []
    for i, idx in enumerate(indices):
        if i > 0:
            await asyncio.sleep(stagger_s)
        result = await run_one_task(
            idx, ds, capsules, model, max_steps, output_dir,
            use_docker=use_docker, docker_image=docker_image,
        )
        results.append(result)
    return results


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start task index")
    parser.add_argument("--end", type=int, default=50, help="End task index (exclusive)")
    parser.add_argument("--concurrency", type=int, default=3, help="Max parallel tasks")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--use-docker", action="store_true", help="Run code in Docker container")
    parser.add_argument("--docker-image", default="bixbench:enhanced", help="Docker image to use")
    args = parser.parse_args()

    # Verify Docker image exists if requested
    if args.use_docker:
        import subprocess as sp
        result = sp.run(["docker", "image", "inspect", args.docker_image], capture_output=True)
        if result.returncode != 0:
            logger.error("Docker image '%s' not found. Build it first.", args.docker_image)
            sys.exit(1)
        logger.info("Using Docker image: %s", args.docker_image)

    import datasets
    ds = datasets.load_dataset("futurehouse/BixBench", split="train")

    task_indices = list(range(args.start, min(args.end, len(ds))))
    logger.info("Running %d tasks (indices %d-%d) with concurrency=%d",
                len(task_indices), args.start, task_indices[-1], args.concurrency)

    data_dir = project_root / "data" / "bixbench"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir = project_root / "results" / "bixbench_parallel"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download all needed capsules first (sequentially)
    capsules = {}
    for idx in task_indices:
        item = ds[idx]
        zf = item["data_folder"]
        if zf not in capsules:
            capsules[zf] = download_capsule(data_dir, zf)
    logger.info("All %d capsules ready", len(capsules))

    # Group tasks by capsule: tasks sharing a capsule run sequentially,
    # different capsules run in parallel (up to --concurrency capsules at once).
    from collections import defaultdict
    capsule_groups = defaultdict(list)
    for idx in task_indices:
        capsule_groups[ds[idx]["data_folder"]].append(idx)
    logger.info(
        "Grouped into %d capsules: %s",
        len(capsule_groups),
        {k: len(v) for k, v in capsule_groups.items()},
    )

    # Run capsule groups in parallel (each group is internally sequential)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_group_with_semaphore(indices):
        async with semaphore:
            return await run_capsule_group(
                indices, ds, capsules, args.model, args.max_steps, output_dir,
                use_docker=args.use_docker, docker_image=args.docker_image,
            )

    group_tasks = [
        run_group_with_semaphore(indices)
        for indices in capsule_groups.values()
    ]
    group_results = await asyncio.gather(*group_tasks, return_exceptions=True)

    # Flatten results from all groups
    results = []
    for gr in group_results:
        if isinstance(gr, Exception):
            logger.error("Capsule group raised: %s", gr)
        elif isinstance(gr, list):
            results.extend(gr)

    # Sort by task index
    results.sort(key=lambda r: r.get("task_index", 0))
    clean_results = results

    # Print summary
    print("\n" + "=" * 100)
    print(f"BATCH RESULTS — {len(clean_results)} tasks, concurrency={args.concurrency}")
    print("=" * 100)
    correct = 0
    wrong = 0
    errors = 0
    for r in clean_results:
        agent = str(r.get("agent_answer", ""))
        ideal = str(r.get("ideal_answer", ""))
        steps = r.get("total_steps", "?")
        dur = r.get("duration_s", "?")
        err = r.get("error", "")
        mode = r.get("eval_mode", "?")

        # Simple match check (not authoritative — just for quick overview)
        if err:
            status = "ERR"
            errors += 1
        elif _quick_match(agent, ideal, mode):
            status = "OK "
            correct += 1
        else:
            status = "???"
            wrong += 1

        print(f"  [{r.get('task_index', '?'):3}] {r.get('question_id', '?'):15s} | "
              f"agent={agent[:25]:25s} | ideal={ideal[:20]:20s} | "
              f"mode={mode:14s} | steps={str(steps):3s} | {str(dur):6s}s | {status}")

    print("=" * 100)
    print(f"Quick match: {correct} OK, {wrong} uncertain, {errors} errors "
          f"(out of {len(clean_results)})")
    print(f"Note: 'OK' = exact/close match; '???' needs proper eval; 'ERR' = exception")
    print("=" * 100)

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    logger.info("Summary saved to %s", output_dir / "summary.json")


def _quick_match(agent: str, ideal: str, mode: str) -> bool:
    """Quick heuristic match — not authoritative, just for overview."""
    agent = agent.strip().lower()
    ideal = ideal.strip().lower()

    if not agent:
        return False

    # Exact match
    if agent == ideal:
        return True

    # Range match: ideal like "(1.50,1.54)"
    if ideal.startswith("(") and "," in ideal:
        try:
            lo, hi = ideal.strip("()").split(",")
            val = float(agent)
            return float(lo) <= val <= float(hi)
        except (ValueError, TypeError):
            return False

    # Numeric close match
    try:
        a, b = float(agent), float(ideal)
        if b == 0:
            return abs(a) < 1e-6
        return abs(a - b) / max(abs(b), 1e-9) < 0.05  # within 5%
    except (ValueError, TypeError):
        pass

    # String containment
    return agent == ideal


if __name__ == "__main__":
    asyncio.run(main())

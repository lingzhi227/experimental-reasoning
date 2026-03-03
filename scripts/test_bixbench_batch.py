#!/usr/bin/env python3
"""Test multiple BixBench tasks sequentially to identify common failure patterns.

Usage:
    python scripts/test_bixbench_batch.py --task-indices 5 9 12 14 18
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


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-indices", type=int, nargs="+", default=[5, 9, 12, 14, 18])
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

    data_dir = project_root / "data" / "bixbench"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir = project_root / "results" / "bixbench_batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.benchmarks.bixbench import BixBenchTask, BixBenchERRunner

    # Download all needed capsules first
    capsules = {}
    for idx in args.task_indices:
        item = ds[idx]
        zf = item["data_folder"]
        if zf not in capsules:
            capsules[zf] = download_capsule(data_dir, zf)

    # Run tasks sequentially (avoid rate limiting)
    results = []
    for idx in args.task_indices:
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
            model=args.model,
            max_steps=args.max_steps,
            use_docker=args.use_docker,
            docker_image=args.docker_image,
        )

        logger.info("=" * 60)
        logger.info("TASK [%d] %s — ideal=%s (%s)", idx, task.question_id, task.ideal, task.eval_mode)
        logger.info("Q: %s", task.question[:200])
        logger.info("=" * 60)

        start = time.time()
        try:
            result = await runner.run_task(task, work_dir)
            result["task_index"] = idx
            result["duration_s"] = round(time.time() - start, 1)
        except Exception as e:
            logger.error("Task %s failed: %s", task.question_id, e, exc_info=True)
            result = {
                "task_index": idx,
                "question_id": task.question_id,
                "agent_answer": "",
                "ideal_answer": task.ideal,
                "error": str(e),
                "duration_s": round(time.time() - start, 1),
            }

        results.append(result)

        # Save individual result
        with open(output_dir / f"{task.question_id}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 80)
    print("BATCH RESULTS SUMMARY")
    print("=" * 80)
    for r in results:
        agent = r.get("agent_answer", "ERROR")
        ideal = r.get("ideal_answer", "?")
        steps = r.get("total_steps", "?")
        dur = r.get("duration_s", "?")
        err = r.get("error", "")
        status = "ERR" if err else "???"
        print(f"  [{r.get('task_index', '?'):3}] {r['question_id']:15s} | "
              f"agent={str(agent)[:25]:25s} | ideal={str(ideal)[:20]:20s} | "
              f"steps={steps} | {dur}s | {status}")
    print("=" * 80)

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    asyncio.run(main())

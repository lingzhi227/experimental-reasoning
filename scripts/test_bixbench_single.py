#!/usr/bin/env python3
"""Test a single BixBench task with the ER agent.

Usage:
    python scripts/test_bixbench_single.py [--task-index 0] [--model sonnet]
"""

import asyncio
import json
import logging
import os
import shutil
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, default=0, help="Task index in dataset")
    parser.add_argument("--model", default="sonnet", help="Claude model to use")
    parser.add_argument("--max-steps", type=int, default=25, help="Max steps")
    parser.add_argument("--use-docker", action="store_true", help="Run code in Docker container")
    parser.add_argument("--docker-image", default="bixbench:enhanced", help="Docker image to use")
    args = parser.parse_args()

    # Verify Docker image exists if requested
    if args.use_docker:
        import subprocess
        result = subprocess.run(
            ["docker", "image", "inspect", args.docker_image],
            capture_output=True,
        )
        if result.returncode != 0:
            logger.error("Docker image '%s' not found. Build it first.", args.docker_image)
            sys.exit(1)
        logger.info("Using Docker image: %s", args.docker_image)

    # Load dataset
    import datasets
    ds = datasets.load_dataset("futurehouse/BixBench", split="train")
    item = ds[args.task_index]

    logger.info("Task %d: %s", args.task_index, item["question_id"])
    logger.info("Question: %s", item["question"][:200])
    logger.info("Ideal: %s", item["ideal"])
    logger.info("Eval mode: %s", item["eval_mode"])

    # Download and extract capsule data
    data_dir = project_root / "data" / "bixbench"
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_filename = item["data_folder"]
    extract_dir = data_dir / zip_filename.replace(".zip", "")

    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        logger.info("Downloading capsule data: %s", zip_filename)
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="futurehouse/BixBench",
            filename=zip_filename,
            local_dir=str(data_dir),
            repo_type="dataset",
        )
        # Extract
        zip_path = data_dir / zip_filename
        extract_dir.mkdir(exist_ok=True)
        shutil.unpack_archive(str(zip_path), str(extract_dir))

        # Flatten Data folder
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

        # Remove notebooks
        for nb in extract_dir.rglob("*.ipynb"):
            nb.unlink()

        # Remove internal folders left over from zip structure
        for d in list(extract_dir.iterdir()):
            if d.is_dir() and d.name.startswith("Capsule"):
                shutil.rmtree(d)

    # List data files
    data_files = [f.name for f in extract_dir.iterdir() if f.is_file()]
    logger.info("Data files: %s", data_files)

    # Create task and run
    from src.benchmarks.bixbench import BixBenchTask, BixBenchERRunner

    task = BixBenchTask(
        question_id=item["question_id"],
        question=item["question"],
        ideal=item["ideal"],
        eval_mode=item["eval_mode"],
        distractors=item.get("distractors", []),
        data_folder=item.get("data_folder"),
        capsule_uuid=item.get("capsule_uuid"),
    )

    runner = BixBenchERRunner(
        model=args.model,
        max_steps=args.max_steps,
        use_docker=args.use_docker,
        docker_image=args.docker_image,
    )
    result = await runner.run_task(task, work_dir=extract_dir)

    # Print results
    print("\n" + "=" * 60)
    print(f"Task: {result['question_id']}")
    print(f"Agent answer: {result['agent_answer']}")
    print(f"Ideal answer: {result['ideal_answer']}")
    print(f"Eval mode: {result.get('eval_mode', 'unknown')}")
    print(f"Total steps: {result.get('total_steps', 0)}")
    print(f"State trace: {result.get('state_trace', [])}")
    print(f"CMM stats: {result.get('cmm_stats', {})}")
    print("=" * 60)

    # Save result
    output_dir = project_root / "results" / "bixbench_dev"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{task.question_id}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info("Result saved to %s", output_dir / f"{task.question_id}.json")


if __name__ == "__main__":
    asyncio.run(main())

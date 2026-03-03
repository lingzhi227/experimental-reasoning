# Experimental Reasoning (ER)

A model-agnostic agentic scaffold for scientific experimental reasoning. The system mimics how scientists conduct experiments: observe → hypothesize → experiment → evaluate → iterate.

## Architecture

```
ORIENT → HYPOTHESIZE → EXPERIMENT → OBSERVE → EVALUATE → CONCLUDE
   ↑                                              |
   └──────────── hypothesis refuted ───────────────┘
```

**Core components:**
- **ER Loop** — Evidence-driven state machine (not free-form LLM planning)
- **CMM (Context Management Module)** — Structured evidence store with provenance tracking
- **Hypothesis Manager** — Explicit hypothesis lifecycle (proposed → testing → supported/refuted)
- **Domain Tactics** — Pluggable strategy catalogs per domain (bioinformatics, ML, policy compliance)
- **Environment Adapters** — Local Python, Docker container, or tool-calling execution

## Benchmarks

| Benchmark | Domain | Environment | Status |
|-----------|--------|-------------|--------|
| **BixBench** | Bioinformatics | Docker (R + Python + CLI tools) | ✅ Working |
| **tau-bench** | Customer service | Tool calling | ✅ Working |
| ScienceAgentBench | Multi-discipline science | Local Python | Planned |
| MLE-bench | ML engineering | Local Python | Planned |

## Quick Start

### Prerequisites

- Python ≥ 3.11
- Docker (for BixBench with full bioinformatics toolkit)
- Anthropic API key (`ANTHROPIC_API_KEY` env var)

### Install

```bash
pip install -e ".[dev]"
```

### Build the Docker Image (for BixBench)

```bash
docker build -t bixbench:enhanced docker/
```

> **Note:** The image is ~18 GB and takes 30-60 minutes to build. It includes R, DESeq2, MAFFT, IQ-TREE, PhyKIT, ClipKIT, BLAST+, and 280+ Python/R packages.

### Run a Single BixBench Task

```bash
# With Docker (recommended — full tool access)
python scripts/test_bixbench_single.py --task-index 0 --use-docker --model sonnet

# Without Docker (host Python only, limited tools)
python scripts/test_bixbench_single.py --task-index 0 --model sonnet
```

### Run Batch / Parallel

```bash
# Sequential batch
python scripts/test_bixbench_batch.py --task-indices 5 9 12 14 18 --use-docker

# Parallel (grouped by capsule)
python scripts/test_bixbench_parallel.py --start 0 --end 50 --concurrency 3 --use-docker
```

## Project Structure

```
src/
├── core/           # ER loop, CMM, hypothesis manager
├── adapters/       # Model adapter (Claude), environment adapters (local, Docker)
├── benchmarks/     # Benchmark-specific runners (BixBench, tau-bench)
├── knowledge/      # Domain knowledge constants (bioinformatics stats, tools)
├── tactics/        # Pluggable tactic catalogs per domain
└── formats/        # Output format engine

scripts/            # CLI test runners
docker/             # Dockerfile for bixbench:enhanced
tests/              # Unit tests
```

## Docker Environment

The `bixbench:enhanced` image provides:

| Category | Tools |
|----------|-------|
| **Alignment** | MAFFT, ClipKIT |
| **Phylogenetics** | IQ-TREE, PhyKIT |
| **Sequence search** | BLAST+ |
| **NGS** | samtools, bcftools, bedtools, FastQC, Trimmomatic |
| **R packages** | DESeq2, edgeR, clusterProfiler, apeglm, org.Hs.eg.db |
| **Python** | pandas, scipy, sklearn, statsmodels, biopython, rpy2, gseapy, scanpy, lifelines |

The ER agent uses a JSON-line REPL protocol to execute code inside the container with persistent state across steps.

## License

MIT

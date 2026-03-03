# Dockerfile for bixbench:enhanced
# Full bioinformatics toolkit for BixBench evaluation
#
# Build:
#   docker build -t bixbench:enhanced docker/
#
# Size: ~18 GB (conda + R + bioinformatics tools)
# Build time: ~30-60 min depending on network

FROM continuumio/miniconda3:24.9.2-0

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        wget gpg software-properties-common build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace /envs
WORKDIR /envs

# Install mamba for faster dependency resolution
RUN conda install mamba=2.0.5 -c conda-forge -y

# ── Stage 1: R + core data science packages ──────────────────────────────
RUN mamba install -c conda-forge -y \
    r-base=4.3.3 \
    r-recommended=4.3 \
    r-irkernel=1.3.2 \
    r-factominer=2.11 \
    r-rcolorbrewer=1.1_3 \
    r-devtools=2.4.5 \
    r-broom=1.0.7 \
    r-data.table=1.15.4 \
    r-enrichr=3.2 \
    r-factoextra=1.0.7 \
    r-ggnewscale=0.5.0 \
    r-ggrepel=0.9.6 \
    r-ggpubr=0.6.0 \
    r-ggvenn=0.1.10 \
    r-janitor=2.2.1 \
    r-multcomp=1.4_26 \
    r-matrix=1.6_5 \
    r-pheatmap=1.0.12 \
    r-tidyverse=2.0.0 \
    r-readxl=1.4.3 \
    r-reshape=0.8.9 \
    r-rstatix=0.7.2 \
    r-viridis=0.6.5 \
    imbalanced-learn=0.13.0 \
    ipykernel=6.29.5

# ── Stage 2: Python data science + rpy2 ──────────────────────────────────
RUN mamba install -c conda-forge -y \
    anndata=0.11.1 \
    biopython=1.84 \
    ete3=3.1.3 \
    fcsparser=0.2.8 \
    cython=3.0.12 \
    keras=3.7.0 \
    jupyter=1.0.0 \
    matplotlib=3.10.0 \
    matplotlib-venn=1.1.1 \
    nbconvert=7.16.4 \
    "numpy=1.26.4" \
    optuna=4.1.0 \
    openpyxl=3.1.5 \
    pandas=2.2.3 \
    plotly=5.24.1 \
    rpy2=3.5.11 \
    scipy=1.14.1 \
    scanpy=1.10.4 \
    seaborn=0.13.2 \
    scikit-learn=1.6.0 \
    statsmodels=0.14.4 \
    umap-learn=0.5.7

# ── Stage 3: Bioinformatics tools (bioconda) ─────────────────────────────
RUN mamba install -c conda-forge -c bioconda -y \
    biokit=0.5.0 \
    gseapy=1.1.4 \
    blast=2.16.0 \
    clipkit=2.3.0 \
    fastqc=0.12.1 \
    iqtree=2.3.6 \
    mafft=7.526 \
    metaeuk=7.bba0d80 \
    mygene=3.2.2 \
    perl=5.32.1 \
    phykit=2.0.1 \
    pydeseq2=0.4.12 \
    spades=4.0.0 \
    trim-galore=0.6.10 \
    bioconductor-enhancedvolcano=1.20.0 \
    bioconductor-deseq2=1.42.0 \
    bioconductor-clusterprofiler=4.10.0 \
    bioconductor-org.hs.eg.db=3.18.0 \
    bioconductor-genomicranges=1.54.1 \
    bioconductor-summarizedexperiment=1.32.0 \
    bioconductor-apeglm=1.24.0

# ── Stage 4: Jupyter kernels ─────────────────────────────────────────────
RUN python -m ipykernel install --user --name python3 --display-name "Python 3 (ipykernel)"
RUN R -e 'IRkernel::installspec(name = "R", displayname = "R (4.3.3)")'

# ── Stage 5: Additional Python packages ──────────────────────────────────
RUN mamba install -c conda-forge -y --force-reinstall python sqlite
RUN pip install --no-cache-dir scikit-posthocs

# ── Cleanup ──────────────────────────────────────────────────────────────
RUN mamba clean --all -f -y && \
    conda clean --all -f -y && \
    rm -rf /root/.cache/pip

WORKDIR /workspace

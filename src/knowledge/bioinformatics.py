"""Domain knowledge for bioinformatics data analysis.

Provides structured knowledge constants that are injected into system prompts
to guide the ER agent on statistical method selection, common analysis patterns,
and known pitfalls in bioinformatics tasks (especially BixBench).
"""

# ---------------------------------------------------------------------------
# Statistical method selection guide
# ---------------------------------------------------------------------------

STATISTICAL_METHODS_GUIDE = """\
## Statistical Method Selection Guide

Choose the appropriate test based on your data and question:

### Comparing Groups (continuous outcome)
| Scenario | Parametric (normal data) | Non-parametric |
|----------|------------------------|----------------|
| 2 groups, independent | Independent t-test | Mann-Whitney U (Wilcoxon rank-sum) |
| 2 groups, paired | Paired t-test | Wilcoxon signed-rank |
| 3+ groups, independent | One-way ANOVA | Kruskal-Wallis |
| 3+ groups, paired | Repeated measures ANOVA | Friedman test |

**Post-hoc tests** (after significant ANOVA/Kruskal-Wallis):
- Parametric: Tukey HSD (all pairs), Dunnett (vs control)
- Non-parametric: Dunn's test with Bonferroni/BH correction

### Association & Correlation
| Scenario | Test |
|----------|------|
| Two categorical variables | Chi-square test (large N) / Fisher's exact (small N or expected<5) |
| Two continuous variables, linear | Pearson correlation |
| Two continuous variables, monotonic | Spearman rank correlation |
| Continuous outcome ~ categorical + continuous predictors | Linear regression / ANOVA |
| Binary outcome ~ predictors | Logistic regression |

### Survival Analysis
- **Kaplan-Meier**: estimate survival curves per group
- **Log-rank test**: compare survival curves between groups
- **Cox proportional hazards**: model survival with covariates

### Genomics-Specific
| Task | Method | Python Implementation |
|------|--------|----------------------|
| Differential expression (RNA-seq counts) | Negative binomial GLM | PyDESeq2, or manual with statsmodels |
| Differential expression (normalized/microarray) | Moderated t-test (limma-style) | scipy.stats.ttest_ind + BH correction |
| Gene set / pathway enrichment | Fisher's exact / hypergeometric | scipy.stats.hypergeom or fisher_exact |
| GO enrichment | Hypergeometric + FDR | scipy.stats + statsmodels.multipletests |

### Multiple Testing Correction
- **Bonferroni**: conservative, controls FWER — use when few tests or need strict control
- **Benjamini-Hochberg (BH)**: standard, controls FDR — use for genomics (DEGs, enrichment)
- Apply via: `from statsmodels.stats.multitest import multipletests; _, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')`
"""

# ---------------------------------------------------------------------------
# Common BixBench task patterns and analysis templates
# ---------------------------------------------------------------------------

ANALYSIS_PATTERNS = """\
## Common Analysis Patterns

### Pattern 1: Differential Expression Analysis (DEG)
```
1. Load count matrix (rows=genes, cols=samples)
2. Load sample metadata (conditions/groups)
3. Filter low-count genes (e.g., keep genes with >10 counts in ≥N samples)
4. Run DESeq2-style analysis:
   - from pydeseq2.dds import DeseqDataSet
   - from pydeseq2.ds import DeseqStats
   - OR manual: log2FC + t-test + BH correction
5. Apply significance thresholds: padj < 0.05, |log2FC| > 1
6. Count DEGs, identify top genes, or answer specific questions
```

### Pattern 2: Gene Set / Pathway Enrichment
```
1. Get DEG list (foreground) and all tested genes (background)
2. Get gene set annotations (GO terms, pathways from provided files)
3. For each gene set: Fisher's exact test or hypergeometric test
   - a = DEGs in set, b = DEGs not in set
   - c = non-DEGs in set, d = non-DEGs not in set
4. Apply FDR correction (BH method)
5. Report enriched terms with q-value < 0.05
```

### Pattern 3: Data Extraction & Aggregation
```
1. Read data file (CSV, TSV, Excel, or other format)
2. Understand structure: df.head(), df.shape, df.columns, df.dtypes
3. Filter rows based on criteria
4. Group by relevant variable
5. Aggregate (mean, median, count, sum)
6. Extract the specific value asked for
```

### Pattern 4: Statistical Comparison
```
1. Load and explore data
2. Identify groups to compare and variable of interest
3. Check normality: scipy.stats.shapiro (if n<50) or visual
4. Select appropriate test (see Statistical Methods Guide)
5. Apply test, get p-value
6. Report: test statistic, p-value, effect size if asked
```

### Pattern 5: Survival Analysis
```
1. Load survival data (time, event/status, covariates)
2. Handle censoring correctly (event=1 means occurred)
3. Kaplan-Meier curves: lifelines.KaplanMeierFitter
4. Compare groups: lifelines.statistics.logrank_test
5. Cox model if needed: lifelines.CoxPHFitter
6. Report: median survival, hazard ratios, p-values
```

### Pattern 6: Sequence / Genomic Region Analysis
```
1. Read FASTA/BED/VCF files
2. Parse with appropriate tool (BioPython, pysam, pandas)
3. Extract features: length, GC content, motifs, variants
4. Compute requested statistics
5. Handle coordinate systems: 0-based (BED) vs 1-based (VCF, GFF)
```
"""

# ---------------------------------------------------------------------------
# Common pitfalls and caveats
# ---------------------------------------------------------------------------

COMMON_PITFALLS = """\
## Common Pitfalls — Avoid These Mistakes

### Statistical Pitfalls
- **p-value type**: BixBench typically expects two-sided p-values unless stated otherwise
- **FDR vs raw p-value**: Read the question carefully — "adjusted p-value" = FDR-corrected; "p-value" may mean raw
- **Log transform confusion**: log2FoldChange is standard for DEG; check if data needs log-transform before analysis
  - RNA-seq counts: often log2(x+1) or use DESeq2's variance stabilizing transform
  - Already-normalized data (FPKM/TPM): may already be log-scale
- **Multiple testing**: ALWAYS apply FDR correction when testing many genes/features simultaneously

### Data Handling Pitfalls
- **Missing values**: Check with df.isna().sum(). Don't silently drop — understand why they're missing
- **Data types**: Gene names can look like dates in Excel (e.g., SEPT1 → Sep-1). Verify dtypes
- **File encodings**: Try 'utf-8' first, then 'latin1' or 'ISO-8859-1' for tricky files
- **Large files**: For files >100MB, read in chunks or sample first to understand structure
- **Header detection**: Some files have comment lines (#) before the header — use comment='#' or skiprows

### Answer Formatting Pitfalls
- **Precision**: Match the precision implied by the question. If asked for p-value, give enough sig figs
- **Rounding**: When the ideal answer has specific decimal places, match that format
- **Gene names**: Use official gene symbols (e.g., "TP53" not "p53", "BRCA1" not "brca1")
- **Numeric ranges**: If answer is a range like (1.50, 1.54), give a value within the range
- **Boolean/Yes-No**: Match the exact expected format ("Yes"/"No", "TRUE"/"FALSE", etc.)

### Analysis Strategy
- **Explore first, analyze second**: Always run df.head(), df.shape, df.columns before analysis
- **Verify intermediate results**: Print key values at each step, don't assume correctness
- **One thing at a time**: Test one hypothesis per code block — easier to debug
- **Don't over-engineer**: Simple correct analysis beats complex broken one
- **Read the question twice**: The exact wording matters for what statistic/value to report
"""

# ---------------------------------------------------------------------------
# Hypothesis-driven analysis instructions
# ---------------------------------------------------------------------------

HYPOTHESIS_INSTRUCTIONS = """\
## Hypothesis-Driven Analysis Protocol

You MUST maintain explicit hypothesis tracking throughout your analysis:

### For Each Step:
1. **State your current hypothesis** — what do you expect to find and why?
2. **Design a targeted experiment** — what specific code will test this hypothesis?
3. **After results, evaluate** — does the evidence support or refute your hypothesis?
4. **If refuted, revise** — form a new hypothesis before the next experiment

### Rules:
- Never repeat a failed approach without changing something
- Track what you've tried and learned — build on prior evidence
- If stuck after 3 attempts, step back and reconsider the problem from scratch
- State your confidence level: how certain are you about the current direction?
"""

# ---------------------------------------------------------------------------
# Docker container tool guidance (for bixbench:enhanced image)
# ---------------------------------------------------------------------------

DOCKER_TOOL_GUIDANCE = """\
## Docker Environment — Available Tools & Usage

You are running inside a Docker container with a full bioinformatics toolkit.
You can use R, command-line tools, and all pre-installed Python packages directly.

### Available CLI Tools (use via subprocess)
- **MAFFT**: multiple sequence alignment (`mafft --auto input.fasta > aligned.fasta`)
- **IQ-TREE**: phylogenetic inference (`iqtree2 -s aligned.fasta -m MFP -B 1000`)
- **ClipKIT**: alignment trimming (`clipkit aligned.fasta`)
- **PhyKIT**: phylogenomic analysis (CLI + Python API)
- **BLAST+**: sequence search (`blastn`, `blastp`, `makeblastdb`)
- **samtools**, **bcftools**: BAM/VCF manipulation
- **bedtools**: genomic interval operations
- **FastQC**, **Trimmomatic**: read QC and trimming

### PhyKIT Usage
```python
# CLI approach (recommended for most tasks):
import subprocess
result = subprocess.run(
    ["phykit", "saturation", "-a", "aligned.fasta", "-t", "tree.nwk"],
    capture_output=True, text=True
)
print(result.stdout)

# Python API:
from phykit.services.tree import Saturation
s = Saturation("/workspace/aligned.fasta", tree="/workspace/tree.nwk")
result = s.run()
```

### R / DESeq2 via rpy2
```python
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
deseq2 = importr("DESeq2")

# Convert pandas count matrix and metadata to R objects
r_counts = pandas2ri.py2rpy(count_df)
r_coldata = pandas2ri.py2rpy(metadata_df)

# Create DESeqDataSet and run
dds = deseq2.DESeqDataSetFromMatrix(
    countData=r_counts,
    colData=r_coldata,
    design=ro.Formula("~ condition")
)
dds = deseq2.DESeq(dds)
res = deseq2.results(dds)
res_df = pandas2ri.rpy2py(ro.r("as.data.frame")(res))
```

### Natural Splines via R's ns()
```python
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
splines = importr("splines")
# ns(x, df=4) for natural cubic splines
ns_basis = splines.ns(ro.FloatVector(x_values), df=4)
```

### Gene Set Enrichment (gseapy)
```python
import gseapy as gp
# Pre-rank GSEA
result = gp.prerank(
    rnk=ranked_genes,  # pd.Series with gene names as index, ranking metric as values
    gene_sets="KEGG_2021_Human",  # or "GO_Biological_Process_2021", "MSigDB_Hallmark_2020"
    outdir=None,
    seed=42,
)
# Enrichr (over-representation)
enr = gp.enrichr(
    gene_list=deg_list,
    gene_sets=["GO_Biological_Process_2021", "KEGG_2021_Human"],
    outdir=None,
)
```

### Excel Multi-Level Headers
Many BixBench files have merged header rows in Excel. Detect and handle:
```python
# Read first few rows to detect header structure
preview = pd.read_excel("file.xlsx", header=None, nrows=5)
print(preview)  # Check if row 0 has merged categories, row 1 has actual column names

# If multi-level: use header=[0,1] or skip the category row
df = pd.read_excel("file.xlsx", header=[0, 1])
# Or flatten:
df = pd.read_excel("file.xlsx", header=1)
```

### Variant Annotation & Filtering
```python
# For variant files, distinguish coding vs non-coding:
# - Coding: missense, nonsense, frameshift, splice site
# - Non-coding: intergenic, intronic, UTR
coding_variants = df[df["consequence"].isin([
    "missense_variant", "stop_gained", "frameshift_variant",
    "splice_donor_variant", "splice_acceptor_variant",
])]
```
"""

# ---------------------------------------------------------------------------
# Convenience: full knowledge block for system prompt injection
# ---------------------------------------------------------------------------

def get_bioinformatics_knowledge() -> str:
    """Return the full domain knowledge block for system prompt injection."""
    return "\n\n".join([
        STATISTICAL_METHODS_GUIDE,
        ANALYSIS_PATTERNS,
        COMMON_PITFALLS,
        HYPOTHESIS_INSTRUCTIONS,
    ])


def get_docker_bioinformatics_knowledge() -> str:
    """Return domain knowledge + Docker-specific tool guidance."""
    return "\n\n".join([
        STATISTICAL_METHODS_GUIDE,
        ANALYSIS_PATTERNS,
        COMMON_PITFALLS,
        HYPOTHESIS_INSTRUCTIONS,
        DOCKER_TOOL_GUIDANCE,
    ])

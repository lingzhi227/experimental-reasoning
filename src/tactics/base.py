"""Domain Tactics Library — pluggable tactic modules.

Follows the ActionCatalog pattern: each domain registers a set of tactics
that guide the agent on WHAT to do in the EXPERIMENT state.

Tactics are informational (not code) — they describe strategies, not implementations.
The LLM uses them to decide which approach to take.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Tactic:
    """A single tactic in the catalog."""
    name: str
    description: str
    when_to_use: str
    suggested_next: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    example: str | None = None


class TacticCatalog:
    """Registry of tactics with transition recommendations."""

    def __init__(self, domain: str = "generic") -> None:
        self.domain = domain
        self.tactics: dict[str, Tactic] = {}

    def add(self, tactic: Tactic) -> None:
        self.tactics[tactic.name] = tactic

    def get(self, name: str) -> Tactic | None:
        return self.tactics.get(name)

    def get_suggestions(self, current: str) -> list[str]:
        t = self.tactics.get(current)
        return list(t.suggested_next) if t else []

    def names(self) -> list[str]:
        return list(self.tactics.keys())

    def to_prompt_section(self) -> str:
        """Render tactics as a prompt section for the LLM."""
        lines = [f"## {self.domain.title()} Tactics", ""]
        for t in self.tactics.values():
            lines.append(f"### {t.name}")
            lines.append(f"**Description**: {t.description}")
            lines.append(f"**When to use**: {t.when_to_use}")
            if t.tools:
                lines.append(f"**Tools**: {', '.join(t.tools)}")
            if t.suggested_next:
                lines.append(f"**Suggested next**: {', '.join(t.suggested_next)}")
            if t.example:
                lines.append(f"**Example**: {t.example}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generic tactics (domain-agnostic, used for ALL benchmarks)
# ---------------------------------------------------------------------------

def generic_tactics() -> TacticCatalog:
    """Generic tactics applicable to any domain."""
    catalog = TacticCatalog(domain="generic")

    catalog.add(Tactic(
        name="explore_data",
        description="Perform exploratory data analysis: profiling, distributions, anomalies",
        when_to_use="At the start of a task when data is provided but not yet understood",
        suggested_next=["form_hypothesis", "statistical_test"],
        tools=["pandas", "numpy", "matplotlib"],
    ))

    catalog.add(Tactic(
        name="form_hypothesis",
        description="Formulate a specific, testable hypothesis based on observations",
        when_to_use="After initial exploration reveals patterns or questions",
        suggested_next=["execute_test", "explore_data"],
    ))

    catalog.add(Tactic(
        name="execute_test",
        description="Execute a specific test/analysis to evaluate a hypothesis",
        when_to_use="When a hypothesis is formed and needs empirical testing",
        suggested_next=["evaluate_evidence", "execute_test"],
        tools=["pandas", "scipy", "sklearn"],
    ))

    catalog.add(Tactic(
        name="evaluate_evidence",
        description="Assess whether evidence supports or contradicts the hypothesis",
        when_to_use="After an experiment produces results",
        suggested_next=["form_hypothesis", "synthesize", "execute_test"],
    ))

    catalog.add(Tactic(
        name="debug",
        description="Diagnose and fix errors in code or analysis",
        when_to_use="When an experiment fails with an error",
        suggested_next=["execute_test"],
    ))

    catalog.add(Tactic(
        name="synthesize",
        description="Combine results from multiple experiments into a conclusion",
        when_to_use="When sufficient evidence has been gathered",
        suggested_next=[],
    ))

    return catalog


# ---------------------------------------------------------------------------
# Bioinformatics tactics (BixBench, HeurekaBench)
# ---------------------------------------------------------------------------

def bioinformatics_tactics() -> TacticCatalog:
    """Bioinformatics-specific tactics."""
    catalog = TacticCatalog(domain="bioinformatics")

    catalog.add(Tactic(
        name="explore_data",
        description="Profile biological datasets: dimensions, types, missing values, distributions",
        when_to_use="When receiving a new biological dataset (expression, sequencing, etc.)",
        suggested_next=["sequence_analysis", "statistical_test", "form_hypothesis"],
        tools=["pandas", "scanpy", "anndata"],
    ))

    catalog.add(Tactic(
        name="sequence_analysis",
        description="Analyze DNA/RNA/protein sequences: BLAST, alignment, motif finding",
        when_to_use="When sequences need to be compared or characterized",
        suggested_next=["statistical_test", "database_query"],
        tools=["biopython", "blast", "mmseqs2"],
    ))

    catalog.add(Tactic(
        name="statistical_test",
        description="Select and apply appropriate statistical test for biological data",
        when_to_use="When quantitative comparison is needed (differential expression, enrichment, etc.)",
        suggested_next=["evaluate_evidence", "database_query"],
        tools=["scipy", "statsmodels", "pydeseq2"],
    ))

    catalog.add(Tactic(
        name="database_query",
        description="Query biological databases strategically (UniProt, KEGG, GO, Biomni)",
        when_to_use="When biological context is needed for interpretation",
        suggested_next=["cross_paper_synthesis", "evaluate_evidence"],
        tools=["biomni", "uniprot", "kegg"],
    ))

    catalog.add(Tactic(
        name="cross_paper_synthesis",
        description="Aggregate evidence across multiple papers/datasets",
        when_to_use="When findings from multiple sources need to be reconciled",
        suggested_next=["synthesize"],
    ))

    catalog.add(Tactic(
        name="synthesize",
        description="Combine biological findings into a coherent conclusion",
        when_to_use="When sufficient evidence has been gathered",
        suggested_next=[],
    ))

    catalog.add(Tactic(
        name="phylogenetic_analysis",
        description="Phylogenetic tree construction and analysis: alignment, trimming, tree inference, tree statistics",
        when_to_use="When analyzing evolutionary relationships, tree topologies, or sequence divergence",
        suggested_next=["evaluate_evidence", "statistical_test", "synthesize"],
        tools=["mafft", "iqtree2", "clipkit", "phykit", "biopython"],
        example="Align with MAFFT, trim with ClipKIT, build tree with IQ-TREE, analyze with PhyKIT",
    ))

    catalog.add(Tactic(
        name="deseq2_analysis",
        description="Differential expression analysis using DESeq2 via rpy2 for RNA-seq count data",
        when_to_use="When analyzing RNA-seq count matrices for differential gene expression between conditions",
        suggested_next=["enrichment_analysis", "evaluate_evidence", "synthesize"],
        tools=["rpy2", "DESeq2", "pandas"],
        example="Load counts + metadata, create DESeqDataSet via rpy2, run DESeq2, extract results with padj/log2FC",
    ))

    catalog.add(Tactic(
        name="enrichment_analysis",
        description="Gene set enrichment and over-representation analysis using gseapy or clusterProfiler via rpy2",
        when_to_use="When a list of DEGs or ranked genes needs pathway/GO/KEGG enrichment analysis",
        suggested_next=["evaluate_evidence", "synthesize"],
        tools=["gseapy", "rpy2", "clusterProfiler"],
        example="Use gseapy.prerank() for GSEA or gseapy.enrichr() for ORA against KEGG/GO databases",
    ))

    catalog.add(Tactic(
        name="variant_analysis",
        description="Variant annotation, filtering, and functional impact assessment",
        when_to_use="When analyzing VCF files, variant tables, or mutation data with coding/non-coding classification",
        suggested_next=["statistical_test", "evaluate_evidence", "synthesize"],
        tools=["pandas", "bcftools", "pysam"],
        example="Load variant table, detect multi-level Excel headers, filter by consequence type (coding vs non-coding)",
    ))

    return catalog


# ---------------------------------------------------------------------------
# ML Engineering tactics (MLE-bench)
# ---------------------------------------------------------------------------

def ml_engineering_tactics() -> TacticCatalog:
    """ML engineering / Kaggle competition tactics."""
    catalog = TacticCatalog(domain="ml_engineering")

    catalog.add(Tactic(
        name="baseline_model",
        description="Build a simple baseline model to establish performance floor",
        when_to_use="At the start of any ML task before any optimization",
        suggested_next=["feature_experiment", "model_experiment"],
        tools=["sklearn", "xgboost", "pytorch"],
    ))

    catalog.add(Tactic(
        name="feature_experiment",
        description="Hypothesis-driven feature engineering or selection experiment",
        when_to_use="When suspecting features can be improved to boost performance",
        suggested_next=["model_experiment", "ablation_analysis", "evaluate_evidence"],
        tools=["pandas", "sklearn.feature_selection"],
    ))

    catalog.add(Tactic(
        name="model_experiment",
        description="Test a model architecture or hyperparameter hypothesis",
        when_to_use="When testing whether a different model/config improves performance",
        suggested_next=["ablation_analysis", "ensemble_selection", "evaluate_evidence"],
        tools=["sklearn", "xgboost", "pytorch", "optuna"],
    ))

    catalog.add(Tactic(
        name="ablation_analysis",
        description="Systematic ablation: remove one component at a time",
        when_to_use="When understanding which components contribute to performance",
        suggested_next=["ensemble_selection", "feature_experiment"],
    ))

    catalog.add(Tactic(
        name="ensemble_selection",
        description="Evidence-based selection of complementary models for ensemble",
        when_to_use="When multiple models exist and need to be combined optimally",
        suggested_next=["synthesize"],
        tools=["sklearn.ensemble"],
    ))

    catalog.add(Tactic(
        name="synthesize",
        description="Finalize the best pipeline and produce submission",
        when_to_use="When the best approach has been identified",
        suggested_next=[],
    ))

    return catalog


# ---------------------------------------------------------------------------
# Policy compliance tactics (tau-bench)
# ---------------------------------------------------------------------------

def policy_compliance_tactics() -> TacticCatalog:
    """Policy compliance / customer service tactics."""
    catalog = TacticCatalog(domain="policy_compliance")

    catalog.add(Tactic(
        name="gather_state",
        description="Systematically collect all relevant customer/system state",
        when_to_use="At the start of any customer interaction before making decisions",
        suggested_next=["verify_policy", "execute_with_evidence"],
    ))

    catalog.add(Tactic(
        name="verify_policy",
        description="Explicitly verify which policies apply to the current situation",
        when_to_use="Before any action that might violate a policy",
        suggested_next=["execute_with_evidence"],
    ))

    catalog.add(Tactic(
        name="execute_with_evidence",
        description="Take action with explicit evidence justification",
        when_to_use="When performing any tool call or customer-facing action",
        suggested_next=["verify_policy", "error_recovery", "synthesize"],
    ))

    catalog.add(Tactic(
        name="error_recovery",
        description="Trace error to source and apply corrective action",
        when_to_use="When a tool call fails or produces unexpected results",
        suggested_next=["gather_state", "verify_policy"],
    ))

    catalog.add(Tactic(
        name="synthesize",
        description="Complete the task with full audit trail",
        when_to_use="When all customer requests have been addressed",
        suggested_next=[],
    ))

    return catalog


# ---------------------------------------------------------------------------
# Science data analysis tactics (ScienceAgentBench)
# ---------------------------------------------------------------------------

def science_data_tactics() -> TacticCatalog:
    """Science data analysis / programming tactics."""
    catalog = TacticCatalog(domain="science_data")

    catalog.add(Tactic(
        name="explore_data",
        description="EDA: understand data structure, types, distributions, edge cases",
        when_to_use="Before writing any analysis code — understand what you're working with",
        suggested_next=["hypothesis_code", "form_hypothesis"],
        tools=["pandas", "numpy", "matplotlib"],
    ))

    catalog.add(Tactic(
        name="hypothesis_code",
        description="Write code based on a specific hypothesis about data structure/approach",
        when_to_use="When you have a hypothesis about the right approach and need to implement it",
        suggested_next=["hypothesis_debug", "evaluate_evidence"],
        tools=["pandas", "scipy", "sklearn"],
    ))

    catalog.add(Tactic(
        name="hypothesis_debug",
        description="Hypothesis-driven debugging: analyze error → hypothesize cause → targeted fix",
        when_to_use="When code fails — don't blindly retry, analyze the error first",
        suggested_next=["hypothesis_code", "explore_data"],
    ))

    catalog.add(Tactic(
        name="evaluate_evidence",
        description="Check if code output matches expected results and task requirements",
        when_to_use="After code runs successfully — verify correctness",
        suggested_next=["hypothesis_code", "synthesize"],
    ))

    catalog.add(Tactic(
        name="synthesize",
        description="Finalize the solution code and produce output",
        when_to_use="When the solution is validated",
        suggested_next=[],
    ))

    return catalog


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TACTIC_REGISTRY: dict[str, TacticCatalog] = {}


def register_tactics(domain: str, catalog: TacticCatalog) -> None:
    """Register a tactic catalog for a domain."""
    TACTIC_REGISTRY[domain] = catalog


def get_tactics(domain: str) -> TacticCatalog | None:
    """Get tactics for a domain."""
    return TACTIC_REGISTRY.get(domain)


def get_or_default_tactics(domain: str) -> TacticCatalog:
    """Get domain tactics, falling back to generic."""
    return TACTIC_REGISTRY.get(domain) or generic_tactics()


# Register built-in catalogs
register_tactics("generic", generic_tactics())
register_tactics("bioinformatics", bioinformatics_tactics())
register_tactics("ml_engineering", ml_engineering_tactics())
register_tactics("policy_compliance", policy_compliance_tactics())
register_tactics("science_data", science_data_tactics())

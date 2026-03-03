"""Tests for the Domain Tactics Library."""

from src.tactics.base import (
    Tactic,
    TacticCatalog,
    generic_tactics,
    bioinformatics_tactics,
    ml_engineering_tactics,
    policy_compliance_tactics,
    science_data_tactics,
    get_or_default_tactics,
    TACTIC_REGISTRY,
)


class TestTacticCatalog:
    def test_add_and_get(self):
        catalog = TacticCatalog(domain="test")
        catalog.add(Tactic(
            name="test_tactic",
            description="A test tactic",
            when_to_use="During testing",
        ))
        t = catalog.get("test_tactic")
        assert t is not None
        assert t.name == "test_tactic"

    def test_suggestions(self):
        catalog = TacticCatalog()
        catalog.add(Tactic(
            name="a", description="A", when_to_use="...",
            suggested_next=["b", "c"],
        ))
        assert catalog.get_suggestions("a") == ["b", "c"]
        assert catalog.get_suggestions("nonexistent") == []

    def test_to_prompt_section(self):
        catalog = TacticCatalog(domain="test")
        catalog.add(Tactic(
            name="explore",
            description="Explore data",
            when_to_use="At the start",
            tools=["pandas"],
        ))
        prompt = catalog.to_prompt_section()
        assert "Test Tactics" in prompt
        assert "explore" in prompt
        assert "pandas" in prompt


class TestBuiltInCatalogs:
    def test_generic_has_core_tactics(self):
        catalog = generic_tactics()
        names = catalog.names()
        assert "explore_data" in names
        assert "form_hypothesis" in names
        assert "execute_test" in names
        assert "synthesize" in names

    def test_bioinformatics_has_domain_tactics(self):
        catalog = bioinformatics_tactics()
        names = catalog.names()
        assert "sequence_analysis" in names
        assert "database_query" in names
        assert "cross_paper_synthesis" in names

    def test_ml_engineering_has_domain_tactics(self):
        catalog = ml_engineering_tactics()
        names = catalog.names()
        assert "baseline_model" in names
        assert "feature_experiment" in names
        assert "ensemble_selection" in names

    def test_policy_compliance_has_domain_tactics(self):
        catalog = policy_compliance_tactics()
        names = catalog.names()
        assert "gather_state" in names
        assert "verify_policy" in names

    def test_science_data_has_domain_tactics(self):
        catalog = science_data_tactics()
        names = catalog.names()
        assert "hypothesis_code" in names
        assert "hypothesis_debug" in names


class TestRegistry:
    def test_all_domains_registered(self):
        assert "generic" in TACTIC_REGISTRY
        assert "bioinformatics" in TACTIC_REGISTRY
        assert "ml_engineering" in TACTIC_REGISTRY
        assert "policy_compliance" in TACTIC_REGISTRY
        assert "science_data" in TACTIC_REGISTRY

    def test_get_or_default(self):
        catalog = get_or_default_tactics("bioinformatics")
        assert catalog.domain == "bioinformatics"

        # Unknown domain falls back to generic
        catalog = get_or_default_tactics("unknown_domain")
        assert catalog.domain == "generic"

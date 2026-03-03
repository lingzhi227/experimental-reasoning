"""Tests for the Hypothesis Manager."""

import tempfile
from pathlib import Path

import pytest

from src.core.cmm import CMMDatabase
from src.core.hypothesis import HypothesisManager


@pytest.fixture
def manager():
    """Create a HypothesisManager with temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as f:
        cmm = CMMDatabase(db_path=Path(f.name))
        cmm.initialize()
        mgr = HypothesisManager(cmm)
        yield mgr
        cmm.close()


class TestLifecycle:
    def test_propose(self, manager: HypothesisManager):
        hid = manager.propose("Gene X is upregulated in disease Y", confidence=0.6)
        h = manager.get_hypothesis(hid)
        assert h.status == "proposed"
        assert h.confidence == 0.6

    def test_test(self, manager: HypothesisManager):
        hid = manager.propose("Test hypothesis")
        manager.test(hid)
        h = manager.get_hypothesis(hid)
        assert h.status == "testing"

    def test_support(self, manager: HypothesisManager):
        hid = manager.propose("Test hypothesis")
        manager.test(hid)
        manager.support(hid, confidence=0.9)
        h = manager.get_hypothesis(hid)
        assert h.status == "supported"
        assert h.confidence == 0.9

    def test_refute(self, manager: HypothesisManager):
        hid = manager.propose("Bad hypothesis")
        manager.test(hid)
        manager.refute(hid)
        h = manager.get_hypothesis(hid)
        assert h.status == "refuted"
        assert h.confidence == 0.0

    def test_revise(self, manager: HypothesisManager):
        h1 = manager.propose("Original hypothesis")
        h2 = manager.revise(h1, "Revised hypothesis", confidence=0.6)

        original = manager.get_hypothesis(h1)
        revised = manager.get_hypothesis(h2)

        assert original.status == "revised"
        assert revised.parent_id == h1
        assert revised.status == "proposed"


class TestEvidence:
    def test_add_evidence_updates_confidence(self, manager: HypothesisManager):
        hid = manager.propose("Test hypothesis", confidence=0.5)

        # Add supporting evidence
        manager.add_evidence(hid, "Strong support", relation="supports", strength=0.9)

        h = manager.get_hypothesis(hid)
        assert h.confidence > 0.5  # Should increase

    def test_contradicting_evidence_decreases_confidence(self, manager: HypothesisManager):
        hid = manager.propose("Test hypothesis", confidence=0.7)

        manager.add_evidence(hid, "Contradicts", relation="contradicts", strength=0.8)

        h = manager.get_hypothesis(hid)
        assert h.confidence < 0.7  # Should decrease

    def test_multiple_evidence_accumulation(self, manager: HypothesisManager):
        hid = manager.propose("Test hypothesis", confidence=0.5)

        manager.add_evidence(hid, "Support 1", relation="supports", strength=0.7)
        manager.add_evidence(hid, "Support 2", relation="supports", strength=0.8)
        manager.add_evidence(hid, "Contradict", relation="contradicts", strength=0.3)

        h = manager.get_hypothesis(hid)
        # Net positive evidence should increase confidence
        assert h.confidence > 0.5


class TestPriorityQueue:
    def test_next_to_test(self, manager: HypothesisManager):
        h1 = manager.propose("Low confidence", confidence=0.3)
        h2 = manager.propose("High confidence", confidence=0.8)

        next_h = manager.next_to_test()
        assert next_h is not None
        # Higher confidence should be prioritized
        assert next_h.id == h2

    def test_next_to_test_skips_resolved(self, manager: HypothesisManager):
        h1 = manager.propose("Will be supported")
        h2 = manager.propose("Still pending")
        manager.support(h1)

        next_h = manager.next_to_test()
        assert next_h is not None
        assert next_h.id == h2


class TestRevisionChain:
    def test_revision_chain(self, manager: HypothesisManager):
        h1 = manager.propose("V1")
        h2 = manager.revise(h1, "V2")
        h3 = manager.revise(h2, "V3")

        chain = manager.get_revision_chain(h3)
        assert len(chain) == 3
        assert chain[0].id == h1  # Root first
        assert chain[-1].id == h3  # Latest last


class TestSummary:
    def test_summary(self, manager: HypothesisManager):
        manager.propose("H1")
        manager.propose("H2")
        h3 = manager.propose("H3")
        manager.support(h3)

        summary = manager.summary()
        assert summary["total"] == 3
        assert summary["active_count"] == 2
        assert summary["by_status"]["proposed"] == 2
        assert summary["by_status"]["supported"] == 1

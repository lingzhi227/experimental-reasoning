"""Tests for the Context Management Module."""

import tempfile
from pathlib import Path

import pytest

from src.core.cmm import CMMDatabase, L1Context


@pytest.fixture
def cmm():
    """Create a temporary CMM database."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as f:
        db = CMMDatabase(db_path=Path(f.name))
        db.initialize()
        yield db
        db.close()


class TestHypothesisOperations:
    def test_add_and_get_hypothesis(self, cmm: CMMDatabase):
        hid = cmm.add_hypothesis("RNA expression correlates with disease", confidence=0.6)
        assert hid.startswith("H-")

        h = cmm.get_hypothesis(hid)
        assert h is not None
        assert h.statement == "RNA expression correlates with disease"
        assert h.status == "proposed"
        assert h.confidence == 0.6

    def test_update_hypothesis(self, cmm: CMMDatabase):
        hid = cmm.add_hypothesis("Test hypothesis")
        cmm.update_hypothesis(hid, status="testing", confidence=0.7)

        h = cmm.get_hypothesis(hid)
        assert h.status == "testing"
        assert h.confidence == 0.7

    def test_get_active_hypotheses(self, cmm: CMMDatabase):
        h1 = cmm.add_hypothesis("Active 1", status="proposed")
        h2 = cmm.add_hypothesis("Active 2", status="testing")
        h3 = cmm.add_hypothesis("Done", status="supported")

        active = cmm.get_active_hypotheses()
        active_ids = [h.id for h in active]
        assert h1 in active_ids
        assert h2 in active_ids
        assert h3 not in active_ids

    def test_parent_child_relationship(self, cmm: CMMDatabase):
        parent = cmm.add_hypothesis("Original hypothesis")
        child = cmm.add_hypothesis("Revised hypothesis", parent_id=parent)

        c = cmm.get_hypothesis(child)
        assert c.parent_id == parent


class TestEvidenceOperations:
    def test_add_evidence(self, cmm: CMMDatabase):
        eid = cmm.add_evidence("p-value = 0.003", evidence_type="statistical")
        assert eid.startswith("E-")

    def test_link_evidence_to_hypothesis(self, cmm: CMMDatabase):
        hid = cmm.add_hypothesis("Test hypothesis")
        eid = cmm.add_evidence("Supporting data")

        cmm.link_evidence_hypothesis(eid, hid, "supports", strength=0.8)

        evidence = cmm.get_evidence_for_hypothesis(hid)
        assert len(evidence) == 1
        assert evidence[0]["relation"] == "supports"
        assert evidence[0]["strength"] == 0.8


class TestActionAndObservation:
    def test_log_action(self, cmm: CMMDatabase):
        aid = cmm.log_action(
            state_name="experiment",
            cycle_id="C-test",
            tactic_name="statistical_test",
            tokens_used=150,
        )
        assert aid.startswith("A-")

        actions = cmm.get_cycle_actions("C-test")
        assert len(actions) == 1
        assert actions[0].tactic_name == "statistical_test"

    def test_add_observation(self, cmm: CMMDatabase):
        aid = cmm.log_action(state_name="experiment")
        oid = cmm.add_observation(
            action_id=aid,
            content="Mean: 42.5, Std: 3.2",
            content_type="statistical",
        )
        assert oid.startswith("O-")

        obs = cmm.get_action_observations(aid)
        assert len(obs) == 1
        assert "42.5" in obs[0].content


class TestProvenance:
    def test_add_and_trace_provenance(self, cmm: CMMDatabase):
        cmm.add_provenance("E-001", "evidence", "wasGeneratedBy", "A-001", "action")
        cmm.add_provenance("E-001", "evidence", "wasDerivedFrom", "O-001", "observation")

        chain = cmm.trace_provenance("E-001", direction="backward")
        assert len(chain) == 2

    def test_forward_provenance(self, cmm: CMMDatabase):
        cmm.add_provenance("H-002", "hypothesis", "wasDerivedFrom", "H-001", "hypothesis")

        derived = cmm.trace_provenance("H-001", direction="forward")
        assert len(derived) == 1
        assert derived[0]["subject_id"] == "H-002"


class TestL1Context:
    def test_assemble_l1_context(self, cmm: CMMDatabase):
        cmm.add_hypothesis("Test hyp", status="testing")
        cmm.add_evidence("Some evidence")

        l1 = cmm.assemble_l1_context(current_state="experiment", cycle_id="C-1")
        assert isinstance(l1, L1Context)
        assert l1.current_state == "experiment"

    def test_l1_to_prompt(self, cmm: CMMDatabase):
        cmm.add_hypothesis("Test hyp", status="testing", confidence=0.7)
        l1 = cmm.assemble_l1_context(current_state="evaluate")

        prompt = l1.to_prompt_section()
        assert "Active Context" in prompt
        assert "Test hyp" in prompt


class TestExperimentSummaries:
    def test_add_and_get_summaries(self, cmm: CMMDatabase):
        sid = cmm.add_experiment_summary(
            cycle_id="C-001",
            summary_text="Found significant correlation",
            key_findings="r=0.85, p<0.001",
            outcome="supported",
        )
        assert sid.startswith("S-")

        summaries = cmm.get_experiment_summaries()
        assert len(summaries) == 1
        assert summaries[0]["outcome"] == "supported"


class TestStats:
    def test_stats(self, cmm: CMMDatabase):
        cmm.add_hypothesis("H1")
        cmm.add_evidence("E1")
        cmm.log_action(state_name="test")

        stats = cmm.stats()
        assert stats["total_hypotheses"] == 1
        assert stats["total_evidence"] == 1
        assert stats["total_actions"] == 1

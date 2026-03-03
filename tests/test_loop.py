"""Tests for the ER Loop Engine."""

import tempfile
from pathlib import Path

import pytest

from src.core.cmm import CMMDatabase
from src.core.hypothesis import HypothesisManager
from src.core.loop import ERLoop, ERState, ERResult
from src.adapters.model import MockModelAdapter
from src.adapters.environment import MockEnvironment
from src.formats.engine import DefaultFormatEngine


@pytest.fixture
def components():
    """Create all ER loop components with mocks."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as f:
        cmm = CMMDatabase(db_path=Path(f.name))
        cmm.initialize()
        hyp_mgr = HypothesisManager(cmm)
        yield cmm, hyp_mgr
        cmm.close()


class TestERStateTransitions:
    def test_state_enum_values(self):
        assert ERState.ORIENT.value == "orient"
        assert ERState.CONCLUDE.value == "conclude"

    def test_all_states_have_prompts(self):
        from src.core.loop import STATE_PROMPTS
        for state in ERState:
            assert state in STATE_PROMPTS


class TestERLoopSimpleTask:
    @pytest.mark.asyncio
    async def test_simple_task_concludes_directly(self, components):
        """A task assessed as 'simple' should go ORIENT → CONCLUDE."""
        cmm, hyp_mgr = components

        model = MockModelAdapter(responses=[
            # ORIENT response: simple task
            {
                "content": {
                    "understanding": "Simple math question",
                    "key_questions": [],
                    "available_resources": [],
                    "complexity": "simple",
                    "next_state": "conclude",
                },
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
            # CONCLUDE response
            {
                "content": {
                    "final_answer": "42",
                    "evidence_chain": [],
                    "confidence": 0.95,
                    "supported_hypotheses": [],
                    "limitations": "None",
                    "reasoning_summary": "Direct calculation",
                },
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        ])
        env = MockEnvironment()
        fmt = DefaultFormatEngine()

        loop = ERLoop(
            cmm=cmm,
            hypothesis_manager=hyp_mgr,
            model=model,
            environment=env,
            format_engine=fmt,
            max_cycles=5,
            max_turns=10,
        )

        result = await loop.run("What is 6 * 7?")
        assert result.final_answer == "42"
        assert result.confidence == 0.95
        assert "orient" in result.state_trace


class TestERLoopFullCycle:
    @pytest.mark.asyncio
    async def test_full_experiment_cycle(self, components):
        """Test a full ORIENT → HYPOTHESIZE → EXPERIMENT → OBSERVE → EVALUATE → CONCLUDE cycle."""
        cmm, hyp_mgr = components

        model = MockModelAdapter(responses=[
            # ORIENT
            {
                "content": {
                    "understanding": "Need to analyze data",
                    "key_questions": ["Is X correlated with Y?"],
                    "available_resources": ["dataset.csv"],
                    "complexity": "complex",
                    "next_state": "hypothesize",
                },
                "usage": {"input_tokens": 200, "output_tokens": 100},
            },
            # HYPOTHESIZE
            {
                "content": {
                    "action": "propose",
                    "hypothesis_statement": "X is positively correlated with Y",
                    "initial_confidence": 0.5,
                    "test_plan": "Compute Pearson correlation",
                    "next_state": "experiment",
                },
                "usage": {"input_tokens": 200, "output_tokens": 100},
            },
            # EXPERIMENT
            {
                "content": {
                    "tactic": "statistical_test",
                    "code": "import scipy; print('r=0.85, p=0.001')",
                    "inputs_description": "X and Y columns",
                    "expected_outcome": "Significant positive correlation",
                    "next_state": "observe",
                },
                "usage": {"input_tokens": 200, "output_tokens": 100},
            },
            # OBSERVE
            {
                "content": {
                    "raw_result_summary": "r=0.85, p=0.001",
                    "key_findings": ["Strong positive correlation"],
                    "metrics": {"r": 0.85, "p": 0.001},
                    "unexpected": None,
                    "evidence_content": "Pearson r=0.85, p=0.001: strong positive correlation",
                    "evidence_type": "statistical",
                    "next_state": "evaluate",
                },
                "usage": {"input_tokens": 200, "output_tokens": 100},
            },
            # EVALUATE
            {
                "content": {
                    "hypothesis_id": None,  # Will use cycle's hypothesis
                    "evidence_relation": "supports",
                    "evidence_strength": 0.9,
                    "reasoning": "p < 0.05 with strong effect size",
                    "hypothesis_verdict": "supported",
                    "sufficient_for_task": True,
                    "next_state": "conclude",
                },
                "usage": {"input_tokens": 200, "output_tokens": 100},
            },
            # CONCLUDE
            {
                "content": {
                    "final_answer": "X is positively correlated with Y (r=0.85, p=0.001)",
                    "evidence_chain": ["E-001"],
                    "confidence": 0.9,
                    "supported_hypotheses": ["H-001"],
                    "limitations": "Correlation does not imply causation",
                    "reasoning_summary": "Tested correlation hypothesis, confirmed with p<0.05",
                },
                "usage": {"input_tokens": 200, "output_tokens": 100},
            },
        ])

        env = MockEnvironment(responses=[
            {"output": "r=0.85, p=0.001", "error": None, "metrics": {"r": 0.85}},
        ])
        fmt = DefaultFormatEngine()

        loop = ERLoop(
            cmm=cmm,
            hypothesis_manager=hyp_mgr,
            model=model,
            environment=env,
            format_engine=fmt,
            max_cycles=5,
            max_turns=20,
        )

        result = await loop.run("Is X correlated with Y in the dataset?")

        assert result.final_answer is not None
        assert "correlated" in result.final_answer.lower()
        assert result.confidence > 0.5
        assert len(result.cycles) >= 1
        assert result.total_turns > 0

        # Verify hypothesis was created
        all_h = hyp_mgr.get_all()
        assert len(all_h) >= 1


class TestERResult:
    def test_result_dataclass(self):
        result = ERResult(task="test task")
        assert result.final_answer is None
        assert result.confidence == 0.0
        assert result.cycles == []
        assert result.total_tokens == 0

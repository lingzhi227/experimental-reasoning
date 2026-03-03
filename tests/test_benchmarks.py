"""Tests for benchmark adapters."""

import tempfile
from pathlib import Path

import pytest

from src.benchmarks.bixbench import (
    BixBenchTask,
    BixBenchERRunner,
    BixBenchEnvironment,
)
from src.benchmarks.taubench import (
    TauBenchEnvironmentAdapter,
    ERTauBenchAgent,
)


class TestBixBenchTask:
    def test_task_creation(self):
        task = BixBenchTask(
            question_id="q_001",
            question="What is the mean expression of gene X?",
            ideal="42.5",
            eval_mode="str_verifier",
        )
        assert task.question_id == "q_001"
        assert task.ideal == "42.5"
        assert task.distractors == []

    def test_task_with_distractors(self):
        task = BixBenchTask(
            question_id="q_002",
            question="Which gene?",
            ideal="BRCA1",
            distractors=["TP53", "MYC"],
        )
        assert len(task.distractors) == 2


class TestBixBenchEnvironment:
    @pytest.mark.asyncio
    async def test_basic_code_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = BixBenchEnvironment(work_dir=tmpdir)
            result = await env.execute({"code": "print(2 + 2)"})
            assert "4" in result["output"]
            assert result["error"] is None

    @pytest.mark.asyncio
    async def test_answer_variable_extraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = BixBenchEnvironment(work_dir=tmpdir)
            result = await env.execute({"code": "_answer = 'BRCA1'"})
            assert result["metrics"].get("answer") == "BRCA1"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = BixBenchEnvironment(work_dir=tmpdir)
            result = await env.execute({"code": "raise ValueError('bad data')"})
            assert result["error"] is not None
            assert "ValueError" in result["error"]

    @pytest.mark.asyncio
    async def test_no_code_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = BixBenchEnvironment(work_dir=tmpdir)
            result = await env.execute({})
            assert result["error"] == "No code provided"


class TestBixBenchERRunner:
    def test_runner_creation(self):
        runner = BixBenchERRunner(model="sonnet")
        assert runner.model == "sonnet"
        assert runner.max_steps == 25

    def test_build_system_prompt(self):
        runner = BixBenchERRunner()
        task = BixBenchTask(
            question_id="q_001",
            question="Analyze gene expression",
            ideal="42",
            eval_mode="str_verifier",
        )
        prompt = runner._build_system_prompt(task, ["data.csv", "metadata.json"])
        assert "bioinformatics" in prompt.lower()
        assert "str_verifier" in prompt
        assert "action" in prompt


class TestERTauBenchAgent:
    def test_format_tools_for_prompt(self):
        tools_info = [
            {
                "type": "function",
                "function": {
                    "name": "get_user_details",
                    "description": "Get user details",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "The user ID",
                            }
                        },
                        "required": ["user_id"],
                    },
                },
            }
        ]
        agent = ERTauBenchAgent.__new__(ERTauBenchAgent)
        agent.tools_info = tools_info
        agent.wiki = "Test wiki"

        formatted = agent._format_tools_for_prompt()
        assert "get_user_details" in formatted
        assert "user_id" in formatted
        assert "required" in formatted

    def test_parse_action_normal(self):
        agent = ERTauBenchAgent.__new__(ERTauBenchAgent)
        agent.tools_info = []
        agent.wiki = ""

        name, kwargs = agent._parse_action({
            "action_name": "get_user_details",
            "action_kwargs": {"user_id": "user_123"},
        })
        assert name == "get_user_details"
        assert kwargs == {"user_id": "user_123"}

    def test_parse_action_respond(self):
        agent = ERTauBenchAgent.__new__(ERTauBenchAgent)
        agent.tools_info = []
        agent.wiki = ""

        name, kwargs = agent._parse_action({
            "action_name": "respond",
            "action_kwargs": {"content": "Hello!"},
        })
        assert name == "respond"
        assert kwargs["content"] == "Hello!"

    def test_parse_action_default_respond(self):
        agent = ERTauBenchAgent.__new__(ERTauBenchAgent)
        agent.tools_info = []
        agent.wiki = ""

        name, kwargs = agent._parse_action({
            "reasoning": "I should greet the user",
        })
        assert name == "respond"
        assert "content" in kwargs

    def test_parse_action_string_kwargs(self):
        agent = ERTauBenchAgent.__new__(ERTauBenchAgent)
        agent.tools_info = []
        agent.wiki = ""

        name, kwargs = agent._parse_action({
            "action_name": "get_user_details",
            "action_kwargs": '{"user_id": "user_123"}',
        })
        assert name == "get_user_details"
        assert kwargs == {"user_id": "user_123"}

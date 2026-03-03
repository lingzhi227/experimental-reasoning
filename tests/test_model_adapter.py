"""Tests for the CLI Model Adapter."""

import pytest

from src.adapters.model import (
    CLIModelAdapter,
    MockModelAdapter,
    _parse_json_response,
    _detect_available_backend,
    ER_ACTION_SCHEMA,
)


class TestParseJsonResponse:
    def test_direct_json(self):
        result = _parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = _parse_json_response(text)
        assert result == {"key": "value"}

    def test_json_in_text(self):
        text = 'Here is the result:\n{"key": "value"}\nThat is all.'
        result = _parse_json_response(text)
        assert result == {"key": "value"}

    def test_nested_json(self):
        text = '{"outer": {"inner": 42}, "list": [1, 2]}'
        result = _parse_json_response(text)
        assert result == {"outer": {"inner": 42}, "list": [1, 2]}

    def test_invalid_json(self):
        result = _parse_json_response("not json at all")
        assert result is None

    def test_empty_string(self):
        result = _parse_json_response("")
        assert result is None

    def test_code_block_no_lang(self):
        text = '```\n{"x": 1}\n```'
        result = _parse_json_response(text)
        assert result == {"x": 1}


class TestCLIModelAdapter:
    def test_init_with_explicit_backend(self):
        adapter = CLIModelAdapter.__new__(CLIModelAdapter)
        adapter.backend = "claude"
        adapter.model = "sonnet"
        adapter.json_schema = ER_ACTION_SCHEMA
        assert adapter.backend == "claude"
        assert adapter.model == "sonnet"

    def test_build_user_prompt(self):
        adapter = CLIModelAdapter.__new__(CLIModelAdapter)
        adapter.backend = "claude"
        adapter.model = "sonnet"
        adapter.json_schema = ER_ACTION_SCHEMA

        prompt = adapter._build_user_prompt(
            messages=[
                {"role": "user", "content": "What is 2+2?"},
            ],
        )
        assert "2+2" in prompt

    def test_build_user_prompt_multi_message(self):
        adapter = CLIModelAdapter.__new__(CLIModelAdapter)
        adapter.backend = "claude"
        adapter.model = "sonnet"
        adapter.json_schema = ER_ACTION_SCHEMA

        prompt = adapter._build_user_prompt(
            messages=[
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Follow up"},
            ],
        )
        assert "First question" in prompt
        assert "First answer" in prompt
        assert "Follow up" in prompt


class TestDetectBackend:
    def test_returns_string_or_none(self):
        result = _detect_available_backend()
        assert result is None or isinstance(result, str)


class TestMockModelAdapter:
    @pytest.mark.asyncio
    async def test_returns_predetermined_responses(self):
        adapter = MockModelAdapter(responses=[
            {"content": {"answer": "42"}, "usage": {"input_tokens": 10, "output_tokens": 5}},
        ])
        result = await adapter.generate(
            messages=[{"role": "user", "content": "test"}],
            system="test system",
        )
        assert result["content"] == {"answer": "42"}

    @pytest.mark.asyncio
    async def test_records_call_history(self):
        adapter = MockModelAdapter()
        await adapter.generate(
            messages=[{"role": "user", "content": "hello"}],
            system="sys",
        )
        assert len(adapter.call_history) == 1
        assert adapter.call_history[0]["system"] == "sys"

    @pytest.mark.asyncio
    async def test_default_response_when_exhausted(self):
        adapter = MockModelAdapter(responses=[])
        result = await adapter.generate(
            messages=[{"role": "user", "content": "test"}],
            system="test",
        )
        assert "content" in result
        assert "usage" in result

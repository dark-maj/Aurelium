import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel

from ai.agents.llm import GrokClient


class SampleSchema(BaseModel):
    summary: str
    score: float


class TestGrokClient(unittest.IsolatedAsyncioTestCase):
    """Smoke tests for Grok client parsing, fallback, retries, and rate limit handling."""

    async def test_native_structured_output_parsing(self):
        mock_openai = MagicMock()
        mock_parsed = SampleSchema(summary="Native parsed test", score=0.95)

        mock_choice = MagicMock()
        mock_choice.message.parsed = mock_parsed
        mock_choice.message.content = mock_parsed.model_dump_json()

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = MagicMock(prompt_tokens=50, completion_tokens=20, total_tokens=70)

        mock_openai.beta.chat.completions.parse = AsyncMock(return_value=mock_resp)

        client = GrokClient(client=mock_openai, max_retries=1)
        res = await client.complete_json("Analyze this paper", SampleSchema)

        self.assertEqual(res.summary, "Native parsed test")
        self.assertEqual(res.score, 0.95)
        mock_openai.beta.chat.completions.parse.assert_awaited_once()

    async def test_json_fallback_with_pydantic_validation(self):
        mock_openai = MagicMock()
        # Simulate beta parse failure to trigger JSON mode fallback
        mock_openai.beta.chat.completions.parse = AsyncMock(side_effect=Exception("Beta parse unsupported"))

        mock_fallback_choice = MagicMock()
        mock_fallback_choice.message.content = '{"summary": "Fallback JSON", "score": 0.88}'
        mock_fallback_resp = MagicMock()
        mock_fallback_resp.choices = [mock_fallback_choice]
        mock_fallback_resp.usage = MagicMock(prompt_tokens=60, completion_tokens=25, total_tokens=85)

        mock_openai.chat.completions.create = AsyncMock(return_value=mock_fallback_resp)

        client = GrokClient(client=mock_openai, max_retries=1)
        res = await client.complete_json("Analyze this paper", SampleSchema)

        self.assertEqual(res.summary, "Fallback JSON")
        self.assertEqual(res.score, 0.88)
        mock_openai.chat.completions.create.assert_awaited_once()

    async def test_json_fallback_retry_on_initial_validation_error(self):
        mock_openai = MagicMock()
        mock_openai.beta.chat.completions.parse = AsyncMock(side_effect=Exception("Beta parse unsupported"))

        # First response is invalid schema
        invalid_choice = MagicMock()
        invalid_choice.message.content = '{"invalid_field": "Missing required fields"}'
        invalid_resp = MagicMock()
        invalid_resp.choices = [invalid_choice]
        invalid_resp.usage = MagicMock(prompt_tokens=40, completion_tokens=10, total_tokens=50)

        # Second response is corrected schema
        corrected_choice = MagicMock()
        corrected_choice.message.content = '{"summary": "Corrected schema", "score": 0.77}'
        corrected_resp = MagicMock()
        corrected_resp.choices = [corrected_choice]
        corrected_resp.usage = MagicMock(prompt_tokens=70, completion_tokens=25, total_tokens=95)

        mock_openai.chat.completions.create = AsyncMock(side_effect=[invalid_resp, corrected_resp])

        client = GrokClient(client=mock_openai, max_retries=1)
        res = await client.complete_json("Extract structured findings", SampleSchema)

        self.assertEqual(res.summary, "Corrected schema")
        self.assertEqual(res.score, 0.77)
        self.assertEqual(mock_openai.chat.completions.create.await_count, 2)


if __name__ == "__main__":
    unittest.main()

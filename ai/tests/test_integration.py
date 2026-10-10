import os
import unittest

from ai.config import settings
from ai.agents.llm import GrokClient
from ai.retrieval.pinecone_client import PineconeVectorClient, VectorRecord


def has_real_xai_key() -> bool:
    key = settings.xai_api_key
    return bool(key and key != "xai-placeholder-key" and not key.startswith("xai-placeholder"))


def has_real_pinecone_key() -> bool:
    key = settings.pinecone_api_key
    return bool(key and key != "pc-placeholder-key" and not key.startswith("pc-placeholder"))


class TestLiveIntegration(unittest.IsolatedAsyncioTestCase):
    """
    Integration tests running against live external APIs.
    Automatically skipped during regular offline test runs unless real credentials are provided.
    """

    async def test_live_grok_completion(self):
        if not has_real_xai_key():
            self.skipTest("Skipping live Grok test: XAI_API_KEY is not set to a real key.")

        client = GrokClient()
        response = await client.complete("Reply with 'READY'.", temperature=0.0)
        self.assertTrue(len(response) > 0)

    async def test_live_pinecone_connection(self):
        if not has_real_pinecone_key():
            self.skipTest("Skipping live Pinecone test: PINECONE_API_KEY is not set to a real key.")

        client = PineconeVectorClient()
        await client.ensure_index_ready()


if __name__ == "__main__":
    unittest.main()

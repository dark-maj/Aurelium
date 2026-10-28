import os
import unittest
from ai.config import Settings


class TestConfig(unittest.TestCase):
    """Smoke test ensuring configuration loads properly from defaults and environment."""

    def test_default_config_loading(self):
        cfg = Settings(
            XAI_API_KEY="test-xai-key",
            PINECONE_API_KEY="test-pc-key",
            NEON_DATABASE_URL="postgresql+asyncpg://user:pass@ep-pooler.region.neon.tech/db?ssl=require",
            NEON_DIRECT_URL="postgresql://user:pass@ep.region.neon.tech/db?sslmode=require",
        )
        self.assertEqual(cfg.xai_api_key, "test-xai-key")
        self.assertEqual(cfg.xai_base_url, "https://api.x.ai/v1")
        self.assertEqual(cfg.llm_model, "grok-4.7")
        self.assertEqual(cfg.judge_model, "grok-4.7")
        self.assertEqual(cfg.pinecone_index, "aurelium-papers")
        self.assertEqual(cfg.pinecone_namespace, "research-papers")
        self.assertEqual(cfg.embedding_dim, 768)
        self.assertEqual(cfg.retrieval_top_k, 20)
        self.assertEqual(cfg.rerank_top_k, 5)
        self.assertTrue("ssl=require" in cfg.neon_database_url)
        self.assertTrue("sslmode=require" in cfg.neon_direct_url)

    def test_env_override(self):
        os.environ["LLM_MODEL"] = "grok-2-1212"
        os.environ["EMBEDDING_DIM"] = "1024"
        cfg = Settings()
        self.assertEqual(cfg.llm_model, "grok-2-1212")
        self.assertEqual(cfg.embedding_dim, 1024)
        del os.environ["LLM_MODEL"]
        del os.environ["EMBEDDING_DIM"]


if __name__ == "__main__":
    unittest.main()

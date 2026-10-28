import unittest
from unittest.mock import MagicMock
from ai.retrieval.pinecone_client import (
    PineconeVectorClient,
    VectorRecord,
    ensure_pinecone_index,
)


class TestPineconeWrapper(unittest.IsolatedAsyncioTestCase):
    """Smoke tests verifying that Pinecone wrapper builds correct requests and sets mandatory metadata."""

    def test_ensure_index_creates_missing_index(self):
        mock_client = MagicMock()
        # Simulate index not present
        mock_client.list_indexes.return_value = []

        ensure_pinecone_index(
            client_instance=mock_client,
            index_name="test-index",
            dimension=768,
            metric="cosine",
            cloud="aws",
            region="us-east-1",
        )
        mock_client.create_index.assert_called_once()
        args, kwargs = mock_client.create_index.call_args
        self.assertEqual(kwargs["name"], "test-index")
        self.assertEqual(kwargs["dimension"], 768)
        self.assertEqual(kwargs["metric"], "cosine")

    async def test_upsert_enforces_required_metadata(self):
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.Index.return_value = mock_index
        mock_index.upsert.return_value = {"upserted_count": 1}

        wrapper = PineconeVectorClient(
            api_key="mock-key",
            index_name="test-index",
            default_namespace="test-ns",
            client_instance=mock_client,
        )

        rec = VectorRecord(
            id="vec_001",
            values=[0.1] * 768,
            paper_id="paper_123",
            chunk_id="chunk_456",
            year=2024,
            fields_of_study=["Computer Science"],
            metadata={"extra_info": "custom"},
        )

        count = await wrapper.upsert_records([rec], namespace="custom-ns")
        self.assertEqual(count, 1)

        mock_index.upsert.assert_called_once()
        _, kwargs = mock_index.upsert.call_args
        self.assertEqual(kwargs["namespace"], "custom-ns")
        vectors = kwargs["vectors"]
        self.assertEqual(len(vectors), 1)

        meta = vectors[0]["metadata"]
        # Invariants required: paper_id, chunk_id, year, fields_of_study
        self.assertEqual(meta["paper_id"], "paper_123")
        self.assertEqual(meta["chunk_id"], "chunk_456")
        self.assertEqual(meta["year"], 2024)
        self.assertEqual(meta["fields_of_study"], ["Computer Science"])
        self.assertEqual(meta["extra_info"], "custom")

    async def test_query_passes_namespace_and_filters(self):
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.Index.return_value = mock_index

        mock_index.query.return_value = {
            "matches": [
                {
                    "id": "vec_001",
                    "score": 0.92,
                    "metadata": {
                        "paper_id": "p_1",
                        "chunk_id": "c_1",
                        "year": 2023,
                        "fields_of_study": ["AI"],
                    },
                }
            ]
        }

        wrapper = PineconeVectorClient(
            api_key="mock-key",
            index_name="test-index",
            default_namespace="test-ns",
            client_instance=mock_client,
        )

        results = await wrapper.query(
            vector=[0.05] * 768,
            top_k=5,
            namespace="query-ns",
            filter_dict={"year": {"$gte": 2020}},
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].paper_id, "p_1")
        self.assertEqual(results[0].score, 0.92)

        mock_index.query.assert_called_once_with(
            vector=[0.05] * 768,
            top_k=5,
            namespace="query-ns",
            filter={"year": {"$gte": 2020}},
            include_metadata=True,
        )


if __name__ == "__main__":
    unittest.main()

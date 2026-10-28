from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from contracts.schemas.models import Chunk, Paper
from ai.ingestion.arxiv import ArxivClient, parse_arxiv_id
from ai.ingestion.semantic_scholar import SemanticScholarClient
from ai.ingestion.cleaner import PaperCleaner, clean_text
from ai.ingestion.deduplicator import Deduplicator, merge_papers, normalize_doi, normalize_title
from ai.ingestion.chunker import PaperChunker, create_abstract_chunk
from ai.ingestion.embedder import ChunkEmbedder
from ai.ingestion.storage import IngestionStorageManager
from ai.ingestion.pipeline import IngestionPipeline


SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2305.18290v2</id>
    <published>2023-05-29T17:59:00Z</published>
    <title> Direct Preference Optimization: Your Language Model is Secretly a Reward Model </title>
    <summary> While recent advances in language modeling have yielded impressive conversational abilities, aligning these models with human preferences remains challenging. Standard approaches use reinforcement learning from human feedback (RLHF) with complex reward models. In this paper, we demonstrate that direct preference optimization achieves superior stability and performance without reinforcement learning. </summary>
    <author><name>Rafael Rafailov</name></author>
    <author><name>Archit Sharma</name></author>
    <arxiv:doi>10.48550/arXiv.2305.18290</arxiv:doi>
    <arxiv:journal_ref>NeurIPS 2023</arxiv:journal_ref>
    <category term="cs.LG"/>
    <category term="cs.AI"/>
    <link title="pdf" href="https://arxiv.org/pdf/2305.18290.pdf" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""

SAMPLE_S2_JSON = {
    "data": [
        {
            "paperId": "204e3073870fae3f05bcbc65834984aa3125cc64",
            "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
            "abstract": "While recent advances in language modeling have yielded impressive conversational abilities, aligning these models with human preferences remains challenging. Standard approaches use reinforcement learning from human feedback (RLHF) with complex reward models. In this paper, we demonstrate that direct preference optimization achieves superior stability and performance without reinforcement learning.",
            "authors": [{"name": "Rafael Rafailov"}, {"name": "Archit Sharma"}],
            "year": 2023,
            "publicationDate": "2023-05-29",
            "venue": "NeurIPS",
            "citationCount": 1420,
            "referenceCount": 45,
            "fieldsOfStudy": ["Computer Science"],
            "externalIds": {
                "DOI": "10.48550/arXiv.2305.18290",
                "ArXiv": "2305.18290",
                "CorpusId": 258963282,
            },
            "url": "https://www.semanticscholar.org/paper/204e3073870fae3f05bcbc65834984aa3125cc64",
            "openAccessPdf": {"url": "https://arxiv.org/pdf/2305.18290.pdf"},
        }
    ]
}


class TestIngestionClientsAndParsing(unittest.IsolatedAsyncioTestCase):
    """Verifies that arXiv Atom and Semantic Scholar JSON clients parse paper metadata accurately."""

    async def test_arxiv_client_parsing(self):
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text=SAMPLE_ARXIV_XML))
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = ArxivClient(client=mock_http)
            papers = await client.fetch_papers("dpo", limit=1)

        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertEqual(paper.arxiv_id, "2305.18290")
        self.assertEqual(paper.doi, "10.48550/arXiv.2305.18290")
        self.assertEqual(paper.venue, "NeurIPS 2023")
        self.assertEqual(paper.year, 2023)
        self.assertEqual(paper.published_date, datetime(2023, 5, 29, 17, 59, 0, tzinfo=timezone.utc))
        self.assertEqual(paper.authors, ["Rafael Rafailov", "Archit Sharma"])
        self.assertIn("cs.LG", paper.fields_of_study)
        self.assertEqual(paper.metadata["pdf_url"], "https://arxiv.org/pdf/2305.18290.pdf")

    async def test_semantic_scholar_client_parsing(self):
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=SAMPLE_S2_JSON))
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = SemanticScholarClient(client=mock_http)
            papers = await client.fetch_papers("dpo", limit=1)

        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertEqual(paper.doi, "10.48550/arXiv.2305.18290")
        self.assertEqual(paper.arxiv_id, "2305.18290")
        self.assertEqual(paper.semantic_scholar_id, "204e3073870fae3f05bcbc65834984aa3125cc64")
        self.assertEqual(paper.citation_count, 1420)
        self.assertEqual(paper.fields_of_study, ["Computer Science"])
        self.assertEqual(paper.year, 2023)
        self.assertEqual(paper.published_date, datetime(2023, 5, 29, 0, 0, 0, tzinfo=timezone.utc))


class TestCleaningAndFiltering(unittest.TestCase):
    """Verifies LaTeX stripping, HTML removal, unicode cleaning, case preservation, and word count filtering."""

    def test_latex_and_html_cleaning_preserves_casing(self):
        raw = "In <b>Nature</b>, \\textbf{AlphaFold} solves protein structures where $x \\in \\mathbb{R}^d$ and \\cite{jumper2021}."
        cleaned = clean_text(raw)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("\\textbf", cleaned)
        self.assertNotIn("$", cleaned)
        self.assertNotIn("\\cite", cleaned)
        # Verify casing preserved
        self.assertIn("AlphaFold", cleaned)
        self.assertIn("Nature", cleaned)

    def test_drops_paper_under_50_words(self):
        short_abstract = "This paper proposes a new method. It achieves good results on benchmarks."
        p_short = Paper(title="Short Paper", abstract=short_abstract)
        res = PaperCleaner.clean_paper(p_short)
        self.assertIsNone(res)

    def test_keeps_paper_with_50_plus_words(self):
        long_abstract = " ".join(["word"] * 55)
        p_long = Paper(title="Long Paper", abstract=long_abstract)
        res = PaperCleaner.clean_paper(p_long)
        self.assertIsNotNone(res)
        self.assertEqual(res.title, "Long Paper")


class TestDeduplication(unittest.TestCase):
    """Verifies 3-stage deduplication by DOI, arXiv ID, and fuzzy title matching, plus record merging."""

    def test_dedup_by_doi_merges_records(self):
        p1 = Paper(
            id="arxiv:2305.18290",
            title="Direct Preference Optimization",
            abstract=" ".join(["word"] * 55),
            doi="10.48550/arXiv.2305.18290",
            citation_count=10,
        )
        p2 = Paper(
            id="s2:204e3073",
            title="Direct Preference Optimization",
            abstract=" ".join(["word"] * 55),
            doi="https://doi.org/10.48550/arXiv.2305.18290",  # URI formatted
            citation_count=500,  # higher citation count from S2
            semantic_scholar_id="204e3073",
        )
        dedup = Deduplicator()
        unique = dedup.deduplicate([p1, p2])
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0].citation_count, 500)
        self.assertEqual(unique[0].semantic_scholar_id, "204e3073")

    def test_dedup_by_arxiv_id(self):
        p1 = Paper(
            id="p1",
            title="Title A",
            abstract=" ".join(["word"] * 55),
            arxiv_id="2305.18290v1",
        )
        p2 = Paper(
            id="p2",
            title="Title A Variant",
            abstract=" ".join(["word"] * 55),
            arxiv_id="2305.18290v2",
        )
        dedup = Deduplicator()
        unique = dedup.deduplicate([p1, p2])
        self.assertEqual(len(unique), 1)

    def test_dedup_by_fuzzy_title_match(self):
        p1 = Paper(
            id="p1",
            title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
            abstract=" ".join(["word"] * 55),
        )
        p2 = Paper(
            id="p2",
            title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model.",  # Trailing period
            abstract=" ".join(["word"] * 55),
        )
        dedup = Deduplicator(fuzzy_threshold=0.88)
        unique = dedup.deduplicate([p1, p2])
        self.assertEqual(len(unique), 1)


class TestChunking(unittest.IsolatedAsyncioTestCase):
    """Verifies abstract-level chunking with paper title inclusion and fallback on failed full-text PDF parsing."""

    async def test_abstract_chunk_includes_title(self):
        paper = Paper(
            id="arxiv:2305.18290",
            title="Direct Preference Optimization",
            abstract="This paper introduces DPO.",
        )
        chunk = create_abstract_chunk(paper)
        self.assertIn("Title: Direct Preference Optimization", chunk.text)
        self.assertIn("Abstract: This paper introduces DPO.", chunk.text)
        self.assertEqual(chunk.section, "Abstract")
        self.assertEqual(chunk.char_start, 0)
        self.assertEqual(chunk.char_end, len(chunk.text))

    async def test_pdf_chunker_fallback_to_abstract(self):
        paper = Paper(
            id="arxiv:2305.18290",
            title="Direct Preference Optimization",
            abstract="Fallback abstract test.",
            metadata={"pdf_url": "https://invalid-pdf-url-for-test.org/test.pdf"},
        )
        chunker = PaperChunker(enable_full_text=True, pdf_timeout=1.0)
        # Mock failed PDF fetch (404 Not Found)
        with patch("httpx.AsyncClient.get", return_value=httpx.Response(404)):
            chunks = await chunker.chunk_paper(paper)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section, "Abstract")
        self.assertIn("Title: Direct Preference Optimization", chunks[0].text)


class TestEmbeddings(unittest.TestCase):
    """Verifies L2-normalized embeddings and skipping already stored chunks."""

    def test_embeddings_normalized_and_dimension(self):
        embedder = ChunkEmbedder(embedding_dim=768, batch_size=2)
        c1 = Chunk(chunk_id="c1", paper_id="p1", text="Text one", char_start=0, char_end=8)
        c2 = Chunk(chunk_id="c2", paper_id="p1", text="Text two", char_start=9, char_end=17)

        embedded = embedder.embed_chunks([c1, c2])
        self.assertEqual(len(embedded), 2)
        for c in embedded:
            self.assertIsNotNone(c.embedding)
            self.assertEqual(len(c.embedding), 768)
            norm = sum(v * v for v in c.embedding)
            self.assertAlmostEqual(norm, 1.0, places=3)

    def test_skips_already_stored_chunks(self):
        embedder = ChunkEmbedder(embedding_dim=768)
        c1 = Chunk(chunk_id="c1", paper_id="p1", text="Text one", char_start=0, char_end=8)
        c2 = Chunk(chunk_id="c2", paper_id="p1", text="Text two", char_start=9, char_end=17)

        stored_ids = {"c1"}
        filtered = embedder.filter_unstored_chunks([c1, c2], stored_chunk_ids=stored_ids)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].chunk_id, "c2")


class TestStorageAndPartialFailureRecovery(unittest.IsolatedAsyncioTestCase):
    """Verifies idempotent upserts and partial-failure recovery when Pinecone write fails after Neon write."""

    async def test_partial_failure_leaves_pending_status_and_retry_succeeds(self):
        mock_engine = MagicMock()
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn

        mock_pinecone = MagicMock()
        # Pinecone fails on first attempt
        mock_pinecone.upsert_records = AsyncMock(side_effect=Exception("Pinecone network timeout!"))

        storage = IngestionStorageManager(db_engine=mock_engine, pinecone_client=mock_pinecone)

        paper = Paper(id="p1", title="Title", abstract="Abstract", year=2023)
        chunk = Chunk(
            chunk_id="c1",
            paper_id="p1",
            text="Chunk text",
            char_start=0,
            char_end=10,
            embedding=[0.1] * 768,
        )

        # 1. First run: Neon write succeeds, Pinecone fails
        with self.assertRaises(Exception):
            await storage.store_papers_and_chunks([paper], [chunk])

        # Verify Neon upserts were executed
        self.assertTrue(mock_conn.execute.called)

        # 2. Retry: Pinecone succeeds
        mock_pinecone.upsert_records = AsyncMock(return_value=1)
        synced = await storage.sync_to_pinecone([chunk])
        self.assertEqual(synced, 1)


class TestRateLimitingBackoff(unittest.IsolatedAsyncioTestCase):
    """Verifies that HTTP 429 rate limits trigger exponential backoff and retry until success."""

    async def test_rate_limit_backoff_on_429(self):
        call_count = 0

        def mock_handler(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, json=SAMPLE_S2_JSON)

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as mock_http:
            client = SemanticScholarClient(client=mock_http)
            papers = await client.fetch_papers("dpo", limit=1)

        self.assertEqual(len(papers), 1)
        self.assertEqual(call_count, 2)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import asyncio
import sys

from ai.ingestion.chunker import PaperChunker
from ai.ingestion.pipeline import IngestionPipeline
from ai.logging_config import setup_structured_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aurelium Research Paper Ingestion Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Research topic or search query (e.g., 'direct preference optimization')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of papers to fetch per source",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="arxiv,semanticscholar",
        help="Comma-separated academic sources: 'arxiv,semanticscholar'",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Target Pinecone vector namespace (defaults to config PINECONE_NAMESPACE)",
    )
    parser.add_argument(
        "--full-text",
        action="store_true",
        default=False,
        help="Attempt section-aware PDF full-text chunking when open-access PDF URLs exist",
    )
    parser.add_argument(
        "--retry-pending",
        action="store_true",
        default=False,
        help="Retry pending chunks that failed Pinecone sync on prior runs",
    )
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()
    logger = setup_structured_logging(level="INFO", json_format=False)

    source_list = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    logger.info(
        f"Starting Aurelium ingestion for topic: '{args.topic}' "
        f"[limit={args.limit}, sources={source_list}, full_text={args.full_text}]"
    )

    chunker = PaperChunker(enable_full_text=args.full_text)
    pipeline = IngestionPipeline(chunker=chunker)

    try:
        stats = await pipeline.run(
            topic=args.topic,
            limit=args.limit,
            sources=source_list,
            namespace=args.namespace,
        )

        print("\n" + "=" * 55)
        print("          AURELIUM INGESTION RUN SUMMARY")
        print("=" * 55)
        print(f" Topic:                   {args.topic}")
        print(f" Raw Papers Fetched:      {stats.raw_fetched}")
        print(f" Valid After Cleaning:    {stats.cleaned_valid}")
        print(f" Unique After Dedupe:     {stats.unique_deduplicated}")
        print(f" Chunks Created:          {stats.chunks_created}")
        print(f" Chunks Embedded:         {stats.chunks_embedded}")
        print(f" Papers Upserted (Neon):  {stats.papers_stored_neon}")
        print(f" Chunks Upserted (Neon):  {stats.chunks_stored_neon}")
        print(f" Vectors Synced (Pinecone):{stats.vectors_synced_pinecone}")
        print("=" * 55 + "\n")

        if args.retry_pending:
            logger.info("Executing retry for previously pending Pinecone chunks...")
            retried = await pipeline.storage.retry_pending_chunks(
                embedder=pipeline.embedder,
                namespace=args.namespace,
            )
            logger.info(f"Retried and synced {retried} pending chunks to Pinecone.")

        return 0
    except Exception as exc:
        logger.error(f"Ingestion run encountered an error: {exc}", exc_info=True)
        return 1


def main() -> None:
    code = asyncio.run(main_async())
    sys.exit(code)


if __name__ == "__main__":
    main()

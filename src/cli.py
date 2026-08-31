"""Command-line inspection tools for the local Arthurian corpus."""

import argparse
import sys
from collections.abc import Sequence
from textwrap import shorten

from src.chunking import corpus_chunk_stats, chunks_for_document, find_chunk
from src.corpus import find_document, load_documents, read_document


METADATA_LABELS = (
    ("id", "ID"),
    ("work_title", "Work"),
    ("chapter_number", "Chapter"),
    ("chapter_title", "Title"),
    ("author_adaptor", "Author/adaptor"),
    ("publisher", "Publisher"),
    ("publication_year", "Publication year"),
    ("local_path", "Local path"),
    ("source_url", "Source URL"),
    ("provenance_note", "Provenance"),
)

CHUNK_METADATA_LABELS = (
    ("chunk_id", "Chunk ID"),
    ("source_document_id", "Source document ID"),
    ("work_title", "Work"),
    ("chapter_number", "Chapter"),
    ("chapter_title", "Title"),
    ("chunk_position", "Chunk position"),
    ("word_count", "Word count"),
    ("source_url", "Source URL"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the local Arthurian corpus.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list documents in editorial order")
    show_parser = subparsers.add_parser("show", help="show one document")
    show_parser.add_argument("document_id", help="stable ID such as lancelot-01")
    chunks_parser = subparsers.add_parser(
        "chunks", help="list chunks for one document"
    )
    chunks_parser.add_argument("document_id", help="stable ID such as lancelot-01")
    chunk_parser = subparsers.add_parser("chunk", help="show one complete chunk")
    chunk_parser.add_argument(
        "chunk_id", help="stable ID such as lancelot-01-chunk-001"
    )
    subparsers.add_parser("stats", help="show corpus-level chunk statistics")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list":
        for document in load_documents():
            print(
                f'{document["chapter_number"]:02}  '
                f'{document["id"]:<11}  '
                f'{document["chapter_title"]}'
            )
        return 0

    if args.command == "stats":
        stats = corpus_chunk_stats()
        print(f'Source documents: {stats["source_documents"]}')
        print(f'Total chunks: {stats["total_chunks"]}')
        print(f'Minimum chunk words: {stats["minimum_words"]}')
        print(f'Maximum chunk words: {stats["maximum_words"]}')
        print(f'Average chunk words: {stats["average_words"]:.1f}')
        print(f'Configured maximum words: {stats["configured_maximum"]}')
        print(f'Overlap target words: {stats["overlap_target"]}')
        return 0

    if args.command == "chunk":
        try:
            chunk = find_chunk(args.chunk_id)
        except KeyError:
            print(f"error: unknown chunk ID: {args.chunk_id}", file=sys.stderr)
            print(
                "Run 'python -m src.cli chunks DOCUMENT_ID' to see valid IDs.",
                file=sys.stderr,
            )
            return 2

        for key, label in CHUNK_METADATA_LABELS:
            print(f"{label}: {getattr(chunk, key)}")
        print()
        print(chunk.text)
        return 0

    if args.command == "chunks":
        try:
            chunks = chunks_for_document(args.document_id)
        except KeyError:
            print(f"error: unknown document ID: {args.document_id}", file=sys.stderr)
            print("Run 'python -m src.cli list' to see valid IDs.", file=sys.stderr)
            return 2

        for chunk in chunks:
            preview = shorten(chunk.text.replace("\n", " "), width=72, placeholder="…")
            print(f"{chunk.chunk_id}  {chunk.word_count:3} words  {preview}")
        return 0

    try:
        document = find_document(args.document_id)
    except KeyError:
        print(f"error: unknown document ID: {args.document_id}", file=sys.stderr)
        print("Run 'python -m src.cli list' to see valid IDs.", file=sys.stderr)
        return 2

    for key, label in METADATA_LABELS:
        print(f"{label}: {document[key]}")
    print()
    print(read_document(document), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

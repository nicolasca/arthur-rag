"""Command-line inspection tools for the local Arthurian corpus."""

import argparse
import sys
from collections.abc import Sequence

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the local Arthurian corpus.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list documents in editorial order")
    show_parser = subparsers.add_parser("show", help="show one document")
    show_parser.add_argument("document_id", help="stable ID such as lancelot-01")
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

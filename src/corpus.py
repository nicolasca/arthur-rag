"""Small, explicit helpers for reading the local source corpus."""

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "corpus" / "manifest.json"


def load_documents() -> list[dict[str, Any]]:
    """Load the ordered document records from the JSON manifest."""
    with MANIFEST_PATH.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    return manifest["documents"]


def find_document(document_id: str) -> dict[str, Any]:
    """Return one manifest record, or raise KeyError for an unknown ID."""
    for document in load_documents():
        if document["id"] == document_id:
            return document
    raise KeyError(document_id)


def read_document(document: dict[str, Any]) -> str:
    """Read a document's Markdown source using its manifest path."""
    path = PROJECT_ROOT / document["local_path"]
    return path.read_text(encoding="utf-8")

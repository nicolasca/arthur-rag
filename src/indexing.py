"""Build, validate, save, load, and inspect the local embedding index."""

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from src.chunking import (
    DEFAULT_MAX_WORDS,
    DEFAULT_OVERLAP_WORDS,
    Chunk,
    load_chunks,
)
from src.corpus import PROJECT_ROOT
from src.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, embed_chunks


INDEX_PATH = PROJECT_ROOT / "data" / "index.json"
Embedder = Callable[[Sequence[Chunk]], list[list[float]]]


class IndexValidationError(ValueError):
    """The index structure does not match its declared metadata or corpus."""


def build_index(
    embedder: Embedder = embed_chunks,
    chunks: Sequence[Chunk] | None = None,
) -> dict[str, Any]:
    """Generate all vectors in memory and return a validated index."""
    source_chunks = list(load_chunks() if chunks is None else chunks)
    if not source_chunks:
        raise IndexValidationError("cannot build an empty index")

    vectors = embedder(source_chunks)
    if len(vectors) != len(source_chunks):
        raise IndexValidationError(
            f"embedder returned {len(vectors)} vectors for "
            f"{len(source_chunks)} chunks"
        )

    items = []
    for chunk, vector in zip(source_chunks, vectors):
        items.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.source_document_id,
                "work_title": chunk.work_title,
                "chapter_number": chunk.chapter_number,
                "chapter_title": chunk.chapter_title,
                "chunk_position": chunk.chunk_position,
                "source_url": chunk.source_url,
                "text": chunk.text,
                "word_count": chunk.word_count,
                "embedding": vector,
            }
        )

    index = {
        "metadata": {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "chunk_maximum_words": DEFAULT_MAX_WORDS,
            "overlap_target": DEFAULT_OVERLAP_WORDS,
            "total_document_count": len(
                {chunk.source_document_id for chunk in source_chunks}
            ),
            "total_chunk_count": len(source_chunks),
        },
        "items": items,
    }
    validate_index(index, source_chunks)
    return index


def validate_index(
    index: dict[str, Any],
    expected_chunks: Sequence[Chunk] | None = None,
) -> None:
    """Reject malformed vectors, metadata, provenance, IDs, or ordering."""
    if not isinstance(index, dict):
        raise IndexValidationError("index must be a JSON object")
    metadata = index.get("metadata")
    items = index.get("items")
    if not isinstance(metadata, dict):
        raise IndexValidationError("index metadata must be an object")
    if not isinstance(items, list) or not items:
        raise IndexValidationError("index must contain at least one item")

    required_metadata = {
        "embedding_model",
        "embedding_dimensions",
        "chunk_maximum_words",
        "overlap_target",
        "total_document_count",
        "total_chunk_count",
    }
    missing_metadata = required_metadata - metadata.keys()
    if missing_metadata:
        raise IndexValidationError(
            "index metadata is missing: " + ", ".join(sorted(missing_metadata))
        )

    dimensions = metadata["embedding_dimensions"]
    if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0:
        raise IndexValidationError("embedding_dimensions must be a positive integer")
    if metadata["total_chunk_count"] != len(items):
        raise IndexValidationError("total_chunk_count does not match item count")
    expected_metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "chunk_maximum_words": DEFAULT_MAX_WORDS,
        "overlap_target": DEFAULT_OVERLAP_WORDS,
    }
    for key, expected_value in expected_metadata.items():
        if metadata[key] != expected_value:
            raise IndexValidationError(
                f"{key} does not match the configured value {expected_value}"
            )

    required_item_fields = {
        "chunk_id",
        "document_id",
        "work_title",
        "chapter_number",
        "chapter_title",
        "chunk_position",
        "source_url",
        "text",
        "word_count",
        "embedding",
    }
    identifiers: list[str] = []
    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise IndexValidationError(f"item {position} must be an object")
        missing_fields = required_item_fields - item.keys()
        if missing_fields:
            raise IndexValidationError(
                f"item {position} is missing: " + ", ".join(sorted(missing_fields))
            )
        identifiers.append(item["chunk_id"])

        vector = item["embedding"]
        if not isinstance(vector, list):
            raise IndexValidationError(
                f'{item["chunk_id"]} embedding must be a list'
            )
        if len(vector) != dimensions:
            raise IndexValidationError(
                f'{item["chunk_id"]} has vector dimension {len(vector)}; '
                f"expected {dimensions}"
            )
        for value in vector:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise IndexValidationError(
                    f'{item["chunk_id"]} contains a missing or non-finite value'
                )

    if len(identifiers) != len(set(identifiers)):
        raise IndexValidationError("index contains duplicate chunk IDs")

    chunks = list(load_chunks() if expected_chunks is None else expected_chunks)
    expected_ids = [chunk.chunk_id for chunk in chunks]
    if identifiers != expected_ids:
        raise IndexValidationError(
            "indexed chunk IDs or ordering do not match the generated corpus"
        )

    expected_documents = {chunk.source_document_id for chunk in chunks}
    if metadata["total_document_count"] != len(expected_documents):
        raise IndexValidationError(
            "total_document_count does not match the generated corpus"
        )

    comparable_fields = {
        "document_id": "source_document_id",
        "work_title": "work_title",
        "chapter_number": "chapter_number",
        "chapter_title": "chapter_title",
        "chunk_position": "chunk_position",
        "source_url": "source_url",
        "text": "text",
        "word_count": "word_count",
    }
    for item, chunk in zip(items, chunks):
        for item_field, chunk_field in comparable_fields.items():
            if item[item_field] != getattr(chunk, chunk_field):
                raise IndexValidationError(
                    f'{item["chunk_id"]} has incorrect {item_field}'
                )


def save_index(
    index: dict[str, Any],
    path: Path = INDEX_PATH,
    expected_chunks: Sequence[Chunk] | None = None,
) -> None:
    """Validate then atomically replace the readable JSON index."""
    validate_index(index, expected_chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as index_file:
            json.dump(index, index_file, ensure_ascii=False, indent=2)
            index_file.write("\n")
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def build_and_save_index(
    embedder: Embedder = embed_chunks,
    path: Path = INDEX_PATH,
    chunks: Sequence[Chunk] | None = None,
) -> dict[str, Any]:
    """Build completely in memory, then save only after success."""
    index = build_index(embedder, chunks)
    save_index(index, path, chunks)
    return index


def load_index(path: Path = INDEX_PATH) -> dict[str, Any]:
    """Read and validate an index against the current generated chunks."""
    try:
        with path.open(encoding="utf-8") as index_file:
            index = json.load(index_file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"index not found at {path}; run 'python -m src.cli index build'"
        ) from None
    except json.JSONDecodeError as error:
        raise IndexValidationError(f"index contains invalid JSON: {error}") from error
    validate_index(index)
    return index


def find_indexed_item(index: dict[str, Any], chunk_id: str) -> dict[str, Any]:
    for item in index["items"]:
        if item["chunk_id"] == chunk_id:
            return item
    raise KeyError(chunk_id)


def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def index_stats(index: dict[str, Any]) -> dict[str, int | float | str]:
    """Return metadata plus vector dimension and norm statistics."""
    validate_index(index)
    norms = [vector_norm(item["embedding"]) for item in index["items"]]
    metadata = index["metadata"]
    return {
        **metadata,
        "item_count": len(index["items"]),
        "vector_dimensions": len(index["items"][0]["embedding"]),
        "minimum_norm": min(norms),
        "maximum_norm": max(norms),
        "average_norm": sum(norms) / len(norms),
    }

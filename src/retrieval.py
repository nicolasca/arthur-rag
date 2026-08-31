"""Transparent cosine-similarity retrieval over the local JSON index."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from src.embeddings import embed_query


DEFAULT_TOP_K = 3
QueryEmbedder = Callable[[str, str, int], list[float]]


class RetrievalError(ValueError):
    """A question or vector cannot be used for retrieval."""


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    item: dict[str, Any]


@dataclass(frozen=True)
class SearchResponse:
    query_vector: list[float]
    query_norm: float
    results: list[SearchResult]


def _validate_vector(vector: Sequence[float], label: str) -> None:
    if not vector:
        raise RetrievalError(f"{label} must not be empty")
    for value in vector:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise RetrievalError(f"{label} contains a non-finite numeric value")


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the sum of pairwise products for equal, non-empty vectors."""
    _validate_vector(left, "left vector")
    _validate_vector(right, "right vector")
    if len(left) != len(right):
        raise RetrievalError(
            f"vector dimension mismatch: {len(left)} != {len(right)}"
        )
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    )


def euclidean_norm(vector: Sequence[float]) -> float:
    """Return the square root of the vector's sum of squared values."""
    _validate_vector(vector, "vector")
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity, rejecting mismatched or zero vectors."""
    product = dot_product(left, right)
    left_norm = euclidean_norm(left)
    right_norm = euclidean_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        raise RetrievalError("cosine similarity is undefined for a zero vector")
    return product / (left_norm * right_norm)


def rank_items(
    query_vector: Sequence[float],
    items: Sequence[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
) -> list[SearchResult]:
    """Score every item and return descending results with index-order ties."""
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise RetrievalError("top-k must be an integer")
    if top_k < 1 or top_k > len(items):
        raise RetrievalError(f"top-k must be between 1 and {len(items)}")

    scored_items = []
    for index_position, item in enumerate(items):
        score = cosine_similarity(query_vector, item["embedding"])
        scored_items.append((score, index_position, item))
    scored_items.sort(key=lambda scored: (-scored[0], scored[1]))

    return [
        SearchResult(rank=rank, score=score, item=item)
        for rank, (score, _index_position, item) in enumerate(
            scored_items[:top_k], start=1
        )
    ]


def search_index(
    index: dict[str, Any],
    question: str,
    top_k: int = DEFAULT_TOP_K,
    query_embedder: QueryEmbedder = embed_query,
) -> SearchResponse:
    """Embed one question once, score all items, and return the top passages."""
    stripped_question = question.strip()
    if not stripped_question:
        raise RetrievalError("question must not be empty")

    items = index["items"]
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise RetrievalError("top-k must be an integer")
    if top_k < 1 or top_k > len(items):
        raise RetrievalError(f"top-k must be between 1 and {len(items)}")

    metadata = index["metadata"]
    model = metadata["embedding_model"]
    dimensions = metadata["embedding_dimensions"]
    query_vector = query_embedder(stripped_question, model, dimensions)
    if len(query_vector) != dimensions:
        raise RetrievalError(
            f"query vector dimension {len(query_vector)} does not match "
            f"index dimension {dimensions}"
        )

    query_norm = euclidean_norm(query_vector)
    if query_norm == 0.0:
        raise RetrievalError("query embedding must not be a zero vector")
    results = rank_items(query_vector, items, top_k)
    return SearchResponse(
        query_vector=query_vector,
        query_norm=query_norm,
        results=results,
    )

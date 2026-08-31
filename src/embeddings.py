"""The project's only Gemini-specific network boundary."""

import os
from collections.abc import Sequence

from src.chunking import Chunk


EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768


class EmbeddingError(RuntimeError):
    """A clear, API-key-safe embedding failure."""


def format_embedding_input(chunk: Chunk) -> str:
    """Format one chunk for asymmetric question-answering retrieval."""
    return f"title: {chunk.chapter_title} | text: {chunk.text}"


def format_query_input(question: str) -> str:
    """Format one non-empty question for question-answering retrieval."""
    stripped_question = question.strip()
    if not stripped_question:
        raise EmbeddingError("question must not be empty")
    return f"task: question answering | query: {stripped_question}"


def _vectors_from_response(response: object, expected_count: int) -> list[list[float]]:
    """Convert Gemini SDK objects to plain vectors and check one-to-one output."""
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None:
        raise EmbeddingError("Gemini returned a malformed response: no embeddings")
    if len(embeddings) != expected_count:
        raise EmbeddingError(
            "Gemini returned "
            f"{len(embeddings)} embeddings for {expected_count} chunks"
        )

    vectors: list[list[float]] = []
    for position, embedding in enumerate(embeddings, start=1):
        values = getattr(embedding, "values", None)
        if values is None:
            raise EmbeddingError(
                f"Gemini returned a malformed embedding at position {position}"
            )
        try:
            vectors.append([float(value) for value in values])
        except (TypeError, ValueError) as error:
            raise EmbeddingError(
                f"Gemini returned non-numeric values at position {position}"
            ) from error
    return vectors


def embed_chunks(chunks: Sequence[Chunk]) -> list[list[float]]:
    """Request one distinct Gemini embedding for every supplied chunk."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EmbeddingError("GEMINI_API_KEY is not set")
    if not chunks:
        return []

    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise EmbeddingError(
            "google-genai is not installed; install the project dependencies"
        ) from error

    contents = [
        types.Content(
            parts=[types.Part.from_text(text=format_embedding_input(chunk))]
        )
        for chunk in chunks
    ]

    try:
        with genai.Client(api_key=api_key) as client:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBEDDING_DIMENSIONS
                ),
            )
    except Exception as error:
        safe_message = str(error).replace(api_key, "[REDACTED]")
        raise EmbeddingError(
            f"Gemini embedding request failed: {safe_message}"
        ) from error

    return _vectors_from_response(response, len(chunks))


def embed_query(
    question: str,
    model: str = EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> list[float]:
    """Request exactly one query embedding using the index configuration."""
    formatted_query = format_query_input(question)
    if not model:
        raise EmbeddingError("embedding model must not be empty")
    if (
        not isinstance(dimensions, int)
        or isinstance(dimensions, bool)
        or dimensions < 1
    ):
        raise EmbeddingError("embedding dimensions must be a positive integer")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EmbeddingError("GEMINI_API_KEY is not set")

    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise EmbeddingError(
            "google-genai is not installed; install the project dependencies"
        ) from error

    content = types.Content(
        parts=[types.Part.from_text(text=formatted_query)]
    )
    try:
        with genai.Client(api_key=api_key) as client:
            response = client.models.embed_content(
                model=model,
                contents=content,
                config=types.EmbedContentConfig(
                    output_dimensionality=dimensions
                ),
            )
    except Exception as error:
        safe_message = str(error).replace(api_key, "[REDACTED]")
        raise EmbeddingError(
            f"Gemini query embedding request failed: {safe_message}"
        ) from error

    return _vectors_from_response(response, 1)[0]

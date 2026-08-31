"""Minimal synchronous HTTP boundary around the existing Arthurian RAG."""

import logging
import math
import os
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StringConstraints

from src.embeddings import EmbeddingError
from src.generation import GENERATION_MODEL, GenerationError, ask_question
from src.indexing import IndexValidationError, load_index
from src.retrieval import RetrievalError


API_TOP_K = 5
MAX_QUESTION_CHARACTERS = 500
logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=MAX_QUESTION_CHARACTERS,
        ),
    ]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    source_document_count: int
    indexed_chunk_count: int
    embedding_model: str
    embedding_dimensions: int
    generation_model: str
    gemini_api_key_configured: bool


class CitationResponse(BaseModel):
    evidence_id: str
    chunk_id: str
    quote: str
    chapter_number: int
    chapter_title: str
    source_url: str


class RetrievedPassageResponse(BaseModel):
    rank: int
    chunk_id: str
    similarity: float
    chapter_number: int
    chapter_title: str
    source_url: str
    text: str


class AskResponse(BaseModel):
    status: Literal["answered", "insufficient"]
    answer: str
    citations: list[CitationResponse]
    retrieved_passages: list[RetrievedPassageResponse]


app = FastAPI(title="Arthurian RAG API", version="0.1.0")


@lru_cache(maxsize=1)
def get_index() -> dict:
    """Load and validate the local index once per warm application process."""
    return load_index()


def clear_index_cache() -> None:
    """Allow tests and local maintenance to force the next index reload."""
    get_index.cache_clear()


def _safe_error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


@app.exception_handler(RequestValidationError)
def request_validation_error(
    _request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    error_types = {item["type"] for item in error.errors()}
    if "string_too_short" in error_types:
        return _safe_error(422, "Question must not be empty.")
    if "string_too_long" in error_types:
        return _safe_error(422, "Question must not exceed 500 characters.")
    return _safe_error(422, "Invalid request body.")


@app.exception_handler(FileNotFoundError)
@app.exception_handler(IndexValidationError)
@app.exception_handler(OSError)
def index_unavailable(_request: Request, _error: Exception) -> JSONResponse:
    return _safe_error(503, "Local index is unavailable.")


@app.exception_handler(EmbeddingError)
@app.exception_handler(GenerationError)
@app.exception_handler(RetrievalError)
def upstream_rag_failure(_request: Request, _error: Exception) -> JSONResponse:
    return _safe_error(502, "The grounded-answer service failed.")


@app.exception_handler(Exception)
def unexpected_failure(_request: Request, _error: Exception) -> JSONResponse:
    logger.error("Unexpected API failure (%s)", type(_error).__name__)
    return _safe_error(500, "Internal server error.")


@app.get("/api/health", response_model=HealthResponse)
def health(index: dict = Depends(get_index)) -> HealthResponse:
    """Report local readiness without contacting Gemini."""
    metadata = index["metadata"]
    return HealthResponse(
        status="ok",
        source_document_count=metadata["total_document_count"],
        indexed_chunk_count=metadata["total_chunk_count"],
        embedding_model=metadata["embedding_model"],
        embedding_dimensions=metadata["embedding_dimensions"],
        generation_model=GENERATION_MODEL,
        gemini_api_key_configured=bool(os.environ.get("GEMINI_API_KEY")),
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    index: dict = Depends(get_index),
) -> AskResponse:
    """Run one existing grounded-answer request with server-owned settings."""
    if not os.environ.get("GEMINI_API_KEY"):
        return _safe_error(503, "Gemini configuration is unavailable.")

    result = ask_question(index, request.question, API_TOP_K)
    citations = [
        CitationResponse(
            evidence_id=citation.evidence_id,
            chunk_id=citation.chunk_id,
            quote=citation.quote,
            chapter_number=citation.chapter_number,
            chapter_title=citation.chapter_title,
            source_url=citation.source_url,
        )
        for citation in result.grounded_answer.citations
    ]
    retrieved_passages = []
    for search_result in result.retrieval.results:
        if not math.isfinite(search_result.score):
            raise RetrievalError("retrieval returned a non-finite similarity")
        item = search_result.item
        retrieved_passages.append(
            RetrievedPassageResponse(
                rank=search_result.rank,
                chunk_id=item["chunk_id"],
                similarity=search_result.score,
                chapter_number=item["chapter_number"],
                chapter_title=item["chapter_title"],
                source_url=item["source_url"],
                text=item["text"],
            )
        )

    return AskResponse(
        status=result.grounded_answer.status,
        answer=result.grounded_answer.answer,
        citations=citations,
        retrieved_passages=retrieved_passages,
    )

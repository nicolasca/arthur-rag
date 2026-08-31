"""Small deterministic evaluation harness around the existing RAG pipeline."""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.embeddings import embed_query
from src.generation import (
    GENERATION_MODEL,
    GenerationError,
    StructuredGenerator,
    ask_question,
    generate_structured_answer,
)
from src.indexing import INDEX_PATH
from src.retrieval import QueryEmbedder, search_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "evaluation" / "cases.json"
HIT_CUTOFFS = (1, 3, 5)
EXPECTED_CASE_COUNT = 10
EXPECTED_ANSWERED_COUNT = 6
EXPECTED_INSUFFICIENT_COUNT = 4
CASE_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
CASE_FIELDS = {
    "case_id",
    "question",
    "expected_status",
    "acceptable_evidence_chunk_ids",
    "rationale",
}


class EvaluationError(ValueError):
    """Evaluation cases or arguments are malformed."""


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_status: str
    acceptable_evidence_chunk_ids: tuple[str, ...]
    rationale: str


def _non_empty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{label} must be non-empty text")
    return value.strip()


def load_evaluation_cases(
    index: dict[str, Any],
    path: Path = CASES_PATH,
) -> list[EvaluationCase]:
    """Load and fully validate the fixed ten-case evaluation dataset."""
    try:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvaluationError(
            f"evaluation cases contain invalid JSON: {error.msg}"
        ) from error
    if not isinstance(raw_cases, list):
        raise EvaluationError("evaluation cases must be a JSON array")
    if len(raw_cases) != EXPECTED_CASE_COUNT:
        raise EvaluationError(
            f"evaluation dataset must contain exactly {EXPECTED_CASE_COUNT} cases"
        )

    known_chunk_ids = {item["chunk_id"] for item in index["items"]}
    seen_case_ids: set[str] = set()
    cases = []
    for position, raw_case in enumerate(raw_cases, start=1):
        label = f"evaluation case {position}"
        if not isinstance(raw_case, dict):
            raise EvaluationError(f"{label} must be an object")
        if set(raw_case) != CASE_FIELDS:
            raise EvaluationError(f"{label} must contain exactly the required fields")

        case_id = _non_empty_text(raw_case["case_id"], f"{label} case_id")
        if not CASE_ID_PATTERN.fullmatch(case_id):
            raise EvaluationError(f"{label} case_id is not stable kebab-case")
        if case_id in seen_case_ids:
            raise EvaluationError(f"duplicate evaluation case ID: {case_id}")
        seen_case_ids.add(case_id)

        question = _non_empty_text(raw_case["question"], f"{label} question")
        rationale = _non_empty_text(raw_case["rationale"], f"{label} rationale")
        expected_status = raw_case["expected_status"]
        if expected_status not in {"answered", "insufficient"}:
            raise EvaluationError(
                f"{label} expected_status must be 'answered' or 'insufficient'"
            )

        raw_chunk_ids = raw_case["acceptable_evidence_chunk_ids"]
        if not isinstance(raw_chunk_ids, list) or any(
            not isinstance(chunk_id, str) or not chunk_id
            for chunk_id in raw_chunk_ids
        ):
            raise EvaluationError(
                f"{label} acceptable_evidence_chunk_ids must be a list of IDs"
            )
        if len(raw_chunk_ids) != len(set(raw_chunk_ids)):
            raise EvaluationError(f"{label} repeats an acceptable chunk ID")
        unknown_chunk_ids = set(raw_chunk_ids) - known_chunk_ids
        if unknown_chunk_ids:
            unknown = sorted(unknown_chunk_ids)[0]
            raise EvaluationError(f"{label} uses unknown chunk ID: {unknown}")
        if expected_status == "answered" and not raw_chunk_ids:
            raise EvaluationError(f"{label} answered case requires acceptable evidence")
        if expected_status == "insufficient" and raw_chunk_ids:
            raise EvaluationError(
                f"{label} insufficient case must not have acceptable evidence"
            )

        cases.append(
            EvaluationCase(
                case_id=case_id,
                question=question,
                expected_status=expected_status,
                acceptable_evidence_chunk_ids=tuple(raw_chunk_ids),
                rationale=rationale,
            )
        )

    answered_count = sum(case.expected_status == "answered" for case in cases)
    insufficient_count = len(cases) - answered_count
    if (
        answered_count != EXPECTED_ANSWERED_COUNT
        or insufficient_count != EXPECTED_INSUFFICIENT_COUNT
    ):
        raise EvaluationError(
            "evaluation dataset must contain exactly six answered and four "
            "insufficient cases"
        )
    return cases


def select_cases(
    cases: Sequence[EvaluationCase],
    case_id: str | None,
) -> list[EvaluationCase]:
    """Return all cases or one named case, preserving dataset order."""
    if case_id is None:
        return list(cases)
    for case in cases:
        if case.case_id == case_id:
            return [case]
    raise EvaluationError(f"unknown evaluation case ID: {case_id}")


def index_sha256(path: Path = INDEX_PATH) -> str:
    """Hash the exact saved index bytes without loading vectors into output."""
    digest = hashlib.sha256()
    with path.open("rb") as index_file:
        for block in iter(lambda: index_file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_top_k(index: dict[str, Any], top_k: int, minimum: int) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise EvaluationError("top-k must be an integer")
    if top_k < minimum or top_k > len(index["items"]):
        raise EvaluationError(
            f"top-k must be between {minimum} and {len(index['items'])}"
        )


def _retrieved_results(response: object) -> list[dict[str, Any]]:
    return [
        {
            "rank": result.rank,
            "chunk_id": result.item["chunk_id"],
            "score": result.score,
        }
        for result in response.results
    ]


def _hit_metrics(case_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scored = [
        result for result in case_results
        if result["expected_status"] == "answered"
    ]
    metrics: dict[str, Any] = {"scored_answerable_cases": len(scored)}
    for cutoff in HIT_CUTOFFS:
        hits = sum(bool(result["hits"][f"hit_at_{cutoff}"]) for result in scored)
        metrics[f"hit_at_{cutoff}"] = hits / len(scored) if scored else None
    return metrics


def run_retrieval_evaluation(
    index: dict[str, Any],
    cases: Sequence[EvaluationCase],
    top_k: int = 5,
    query_embedder: QueryEmbedder = embed_query,
    index_path: Path = INDEX_PATH,
) -> dict[str, Any]:
    """Evaluate ranked retrieval once per question, excluding insufficients."""
    _validate_top_k(index, top_k, minimum=max(HIT_CUTOFFS))
    case_results = []
    for case in cases:
        response = search_index(index, case.question, top_k, query_embedder)
        retrieved = _retrieved_results(response)
        acceptable = set(case.acceptable_evidence_chunk_ids)
        acceptable_ranks = [
            result["rank"]
            for result in retrieved
            if result["chunk_id"] in acceptable
        ]
        first_acceptable_rank = min(acceptable_ranks, default=None)
        hits = None
        if case.expected_status == "answered":
            hits = {
                f"hit_at_{cutoff}": (
                    first_acceptable_rank is not None
                    and first_acceptable_rank <= cutoff
                )
                for cutoff in HIT_CUTOFFS
            }
        case_results.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "expected_status": case.expected_status,
                "acceptable_evidence_chunk_ids": list(
                    case.acceptable_evidence_chunk_ids
                ),
                "rationale": case.rationale,
                "retrieved_results": retrieved,
                "first_acceptable_rank": first_acceptable_rank,
                "hits": hits,
            }
        )

    return {
        "evaluation_type": "retrieval",
        "embedding_model": index["metadata"]["embedding_model"],
        "index_sha256": index_sha256(index_path),
        "top_k": top_k,
        "aggregate_metrics": _hit_metrics(case_results),
        "cases": case_results,
    }


def _answer_metrics(case_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answered = [
        result for result in case_results
        if result["expected_status"] == "answered"
    ]
    status_matches = sum(
        bool(result["automatic_checks"]["status_matches_expected"])
        for result in case_results
    )
    evidence_hits = sum(
        bool(result["automatic_checks"]["acceptable_evidence_cited"])
        for result in answered
    )
    return {
        "case_count": len(case_results),
        "status_accuracy": (
            status_matches / len(case_results) if case_results else None
        ),
        "scored_answered_cases": len(answered),
        "evidence_hit_rate": (
            evidence_hits / len(answered) if answered else None
        ),
        "validation_error_count": sum(
            result["error_type"] == "validation" for result in case_results
        ),
        "generation_error_count": sum(
            result["error_type"] == "generation" for result in case_results
        ),
    }


def run_answer_evaluation(
    index: dict[str, Any],
    cases: Sequence[EvaluationCase],
    top_k: int = 5,
    query_embedder: QueryEmbedder = embed_query,
    generator: StructuredGenerator = generate_structured_answer,
    index_path: Path = INDEX_PATH,
) -> dict[str, Any]:
    """Run the existing ask pipeline once per case and check local outcomes."""
    _validate_top_k(index, top_k, minimum=1)
    case_results = []
    for case in cases:
        try:
            response = ask_question(
                index,
                case.question,
                top_k,
                query_embedder,
                generator,
            )
        except GenerationError as error:
            error_message = str(error)
            error_type = (
                "generation"
                if error_message.startswith("Gemini answer generation failed:")
                else "validation"
            )
            retrieved = (
                _retrieved_results(error.retrieval)
                if error.retrieval is not None
                else []
            )
            case_results.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "expected_status": case.expected_status,
                    "acceptable_evidence_chunk_ids": list(
                        case.acceptable_evidence_chunk_ids
                    ),
                    "rationale": case.rationale,
                    "retrieved_results": retrieved,
                    "generated_status": None,
                    "answer": None,
                    "citations": [],
                    "automatic_checks": {
                        "status_matches_expected": False,
                        "generation_completed": False,
                        "citation_validation_passed": False,
                        "acceptable_evidence_cited": False,
                        "insufficient_status_confirmed": False,
                    },
                    "error_type": error_type,
                    "error": error_message,
                }
            )
            continue

        grounded = response.grounded_answer
        citations = [
            {
                "evidence_id": citation.evidence_id,
                "chunk_id": citation.chunk_id,
                "quote": citation.quote,
                "chapter_number": citation.chapter_number,
                "chapter_title": citation.chapter_title,
                "source_url": citation.source_url,
            }
            for citation in grounded.citations
        ]
        acceptable = set(case.acceptable_evidence_chunk_ids)
        acceptable_evidence_cited = (
            case.expected_status == "answered"
            and any(citation["chunk_id"] in acceptable for citation in citations)
        )
        case_results.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "expected_status": case.expected_status,
                "acceptable_evidence_chunk_ids": list(
                    case.acceptable_evidence_chunk_ids
                ),
                "rationale": case.rationale,
                "retrieved_results": _retrieved_results(response.retrieval),
                "generated_status": grounded.status,
                "answer": grounded.answer,
                "citations": citations,
                "automatic_checks": {
                    "status_matches_expected": (
                        grounded.status == case.expected_status
                    ),
                    "generation_completed": True,
                    "citation_validation_passed": True,
                    "acceptable_evidence_cited": acceptable_evidence_cited,
                    "insufficient_status_confirmed": (
                        case.expected_status == "insufficient"
                        and grounded.status == "insufficient"
                    ),
                },
                "error_type": None,
                "error": None,
            }
        )

    return {
        "evaluation_type": "answers",
        "embedding_model": index["metadata"]["embedding_model"],
        "generation_model": GENERATION_MODEL,
        "index_sha256": index_sha256(index_path),
        "top_k": top_k,
        "aggregate_metrics": _answer_metrics(case_results),
        "cases": case_results,
    }


def write_evaluation_result(path: Path, result: dict[str, Any]) -> None:
    """Write optional readable JSON only when explicitly requested."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_retrieval_report(result: dict[str, Any]) -> str:
    """Render transparent per-case rankings and aggregate Hit@k metrics."""
    lines = [
        "Evaluation: retrieval",
        f'Embedding model: {result["embedding_model"]}',
        f'Top-k: {result["top_k"]}',
    ]
    for case in result["cases"]:
        lines.extend(("", f'=== {case["case_id"]} ===', case["question"]))
        for retrieved in case["retrieved_results"]:
            lines.append(
                f'{retrieved["rank"]}. {retrieved["chunk_id"]} '
                f'(cosine: {retrieved["score"]:.8f})'
            )
        if case["expected_status"] == "answered":
            rank = case["first_acceptable_rank"]
            lines.append(
                "First acceptable evidence rank: "
                + (str(rank) if rank is not None else "not retrieved")
            )
        else:
            lines.append(
                "Retrieval scoring: not applicable to an insufficient case."
            )

    metrics = result["aggregate_metrics"]
    lines.extend(
        (
            "",
            "Aggregate retrieval metrics:",
            f'Scored answerable cases: {metrics["scored_answerable_cases"]}',
            f'Hit@1: {_format_rate(metrics["hit_at_1"])}',
            f'Hit@3: {_format_rate(metrics["hit_at_3"])}',
            f'Hit@5: {_format_rate(metrics["hit_at_5"])}',
        )
    )
    return "\n".join(lines)


def format_answer_report(result: dict[str, Any]) -> str:
    """Render complete answers, citations, checks, and the review boundary."""
    lines = [
        "Evaluation: answers",
        f'Embedding model: {result["embedding_model"]}',
        f'Generation model: {result["generation_model"]}',
        f'Top-k: {result["top_k"]}',
    ]
    for case in result["cases"]:
        lines.extend(
            (
                "",
                f'=== {case["case_id"]} ===',
                "MANUAL REVIEW REQUIRED",
                f'Question: {case["question"]}',
                f'Expected status: {case["expected_status"]}',
                "Retrieved passages:",
            )
        )
        if not case["retrieved_results"]:
            lines.append("None (pipeline error before a validated response).")
        for retrieved in case["retrieved_results"]:
            lines.append(
                f'{retrieved["rank"]}. {retrieved["chunk_id"]} '
                f'(cosine: {retrieved["score"]:.8f})'
            )
        lines.append(f'Generated status: {case["generated_status"]}')
        lines.append("Answer:")
        lines.append(case["answer"] if case["answer"] is not None else "None")
        lines.append("Resolved citations:")
        if not case["citations"]:
            lines.append("None")
        for citation in case["citations"]:
            lines.append(
                f'- {citation["evidence_id"]} -> {citation["chunk_id"]}: '
                f'"{citation["quote"]}"'
            )
        checks = case["automatic_checks"]
        lines.extend(
            (
                "Automatic checks:",
                f'- Status matches: {checks["status_matches_expected"]}',
                f'- Generation completed: {checks["generation_completed"]}',
                f'- Citation validation passed: '
                f'{checks["citation_validation_passed"]}',
                f'- Acceptable evidence cited: '
                f'{checks["acceptable_evidence_cited"]}',
                f'- Insufficient status confirmed: '
                f'{checks["insufficient_status_confirmed"]}',
            )
        )
        if case["error"]:
            lines.append(f'Error: {case["error"]}')

    metrics = result["aggregate_metrics"]
    lines.extend(
        (
            "",
            "Aggregate answer metrics:",
            f'Cases: {metrics["case_count"]}',
            f'Status accuracy: {_format_rate(metrics["status_accuracy"])}',
            f'Answered cases scored for evidence: '
            f'{metrics["scored_answered_cases"]}',
            f'Evidence-hit rate: {_format_rate(metrics["evidence_hit_rate"])}',
            f'Validation errors: {metrics["validation_error_count"]}',
            f'Generation errors: {metrics["generation_error_count"]}',
            "",
            "Human review checklist (no automatic semantic score):",
            "- Does the answer directly address the question?",
            "- Is every factual claim supported by the cited evidence?",
            "- Does the answer avoid external Arthurian knowledge?",
            "- Is an insufficient answer appropriately cautious?",
        )
    )
    return "\n".join(lines)

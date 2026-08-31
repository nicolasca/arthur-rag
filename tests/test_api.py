import math
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as api
from src.embeddings import EmbeddingError
from src.generation import (
    AskResponse,
    GenerationError,
    GroundedAnswer,
    ValidatedCitation,
)
from src.indexing import INDEX_PATH
from src.retrieval import SearchResponse, SearchResult


def fake_index():
    return {
        "metadata": {
            "embedding_model": "gemini-embedding-2",
            "embedding_dimensions": 768,
            "total_document_count": 19,
            "total_chunk_count": 70,
        },
        "items": [],
    }


def fake_item(position):
    chunk_id = f"lancelot-07-chunk-{position:03}"
    return {
        "chunk_id": chunk_id,
        "chapter_number": 7,
        "chapter_title": "La Dame du Lac et Lancelot",
        "source_url": "https://fr.wikisource.org/wiki/Les_Enfances_de_Lancelot/07",
        "text": f"Passage français numéro {position} avec accents et typographie.",
        "embedding": [0.1, 0.2],
    }


def fake_ask_response(status="answered"):
    results = [
        SearchResult(
            rank=position,
            score=0.9 - position / 100,
            item=fake_item(position),
        )
        for position in range(1, 6)
    ]
    citations = []
    answer = "Les passages récupérés sont insuffisants."
    if status == "answered":
        citations = [
            ValidatedCitation(
                evidence_id="evidence-01",
                chunk_id="lancelot-07-chunk-001",
                quote="La Dame du Lac donna à Lancelot une bonne nourrice.",
                chapter_number=7,
                chapter_title="La Dame du Lac et Lancelot",
                source_url=(
                    "https://fr.wikisource.org/wiki/"
                    "Les_Enfances_de_Lancelot/07"
                ),
            )
        ]
        answer = "Lancelot est recueilli et élevé par la Dame du Lac."
    return AskResponse(
        retrieval=SearchResponse(
            query_vector=[0.5, 0.5],
            query_norm=math.sqrt(0.5),
            results=results,
        ),
        grounded_answer=GroundedAnswer(
            status=status,
            answer=answer,
            citations=citations,
        ),
    )


class ApiTests(unittest.TestCase):
    def setUp(self):
        api.clear_index_cache()
        api.app.dependency_overrides[api.get_index] = fake_index
        self.client = TestClient(api.app, raise_server_exceptions=False)

    def tearDown(self):
        api.app.dependency_overrides.clear()
        api.clear_index_cache()

    def post_ask(self, payload, orchestration=fake_ask_response()):
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-secret-key"}, clear=False
        ):
            with patch.object(
                api, "ask_question", return_value=orchestration
            ) as mocked:
                response = self.client.post("/api/ask", json=payload)
        return response, mocked

    def test_health_loads_local_index_without_calling_gemini(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(api, "ask_question") as orchestration:
                response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "source_document_count": 19,
                "indexed_chunk_count": 70,
                "embedding_model": "gemini-embedding-2",
                "embedding_dimensions": 768,
                "generation_model": "gemini-3.1-flash-lite",
                "gemini_api_key_configured": False,
            },
        )
        orchestration.assert_not_called()

    def test_health_reports_key_boolean_without_exposing_value(self):
        secret = "do-not-return-this-secret"
        with patch.dict(os.environ, {"GEMINI_API_KEY": secret}, clear=False):
            response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["gemini_api_key_configured"])
        self.assertNotIn(secret, response.text)
        self.assertNotIn("GEMINI_API_KEY", response.text)

    def test_answered_response_serializes_citations_and_five_passages(self):
        before_exists = INDEX_PATH.exists()
        before_bytes = INDEX_PATH.read_bytes() if before_exists else None
        response, orchestration = self.post_ask(
            {"question": "  Qui recueille et élève Lancelot ?  "}
        )

        self.assertEqual(response.status_code, 200)
        orchestration.assert_called_once_with(
            fake_index(), "Qui recueille et élève Lancelot ?", 5
        )
        body = response.json()
        self.assertEqual(body["status"], "answered")
        self.assertEqual(
            body["answer"],
            "Lancelot est recueilli et élevé par la Dame du Lac.",
        )
        self.assertEqual(body["citations"][0]["evidence_id"], "evidence-01")
        self.assertEqual(
            body["citations"][0]["chapter_title"],
            "La Dame du Lac et Lancelot",
        )
        self.assertEqual(len(body["retrieved_passages"]), 5)
        self.assertEqual(
            [passage["rank"] for passage in body["retrieved_passages"]],
            [1, 2, 3, 4, 5],
        )
        self.assertTrue(
            all(
                math.isfinite(passage["similarity"])
                for passage in body["retrieved_passages"]
            )
        )
        self.assertIn("français", body["retrieved_passages"][0]["text"])
        self.assertNotIn("embedding", response.text)
        self.assertNotIn("query_vector", response.text)
        self.assertNotIn("test-secret-key", response.text)
        self.assertNotIn(str(INDEX_PATH), response.text)
        self.assertEqual(INDEX_PATH.exists(), before_exists)
        if before_exists:
            self.assertEqual(INDEX_PATH.read_bytes(), before_bytes)

    def test_insufficient_response_has_no_citations(self):
        response, orchestration = self.post_ask(
            {"question": "Comment Arthur rencontre-t-il Lancelot ?"},
            fake_ask_response("insufficient"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "insufficient")
        self.assertEqual(response.json()["citations"], [])
        self.assertEqual(len(response.json()["retrieved_passages"]), 5)
        orchestration.assert_called_once()

    def test_blank_and_overlong_questions_use_clear_safe_validation_errors(self):
        blank_response, blank_orchestration = self.post_ask(
            {"question": " \t\n "}
        )
        long_question = "é" * 501
        long_response, long_orchestration = self.post_ask(
            {"question": long_question}
        )

        self.assertEqual(blank_response.status_code, 422)
        self.assertEqual(
            blank_response.json(), {"detail": "Question must not be empty."}
        )
        self.assertEqual(long_response.status_code, 422)
        self.assertEqual(
            long_response.json(),
            {"detail": "Question must not exceed 500 characters."},
        )
        self.assertNotIn(long_question, long_response.text)
        blank_orchestration.assert_not_called()
        long_orchestration.assert_not_called()

    def test_malformed_body_and_browser_owned_configuration_are_rejected(self):
        malformed = self.client.post(
            "/api/ask",
            content="{not-json",
            headers={"Content-Type": "application/json"},
        )
        extra_configuration, orchestration = self.post_ask(
            {"question": "Question valide ?", "top_k": 20}
        )

        self.assertEqual(malformed.status_code, 422)
        self.assertEqual(malformed.json(), {"detail": "Invalid request body."})
        self.assertEqual(extra_configuration.status_code, 422)
        self.assertEqual(
            extra_configuration.json(), {"detail": "Invalid request body."}
        )
        orchestration.assert_not_called()

    def test_missing_index_returns_service_unavailable_without_path(self):
        hidden_path = "/private/secret/index.json"

        def missing_index():
            raise FileNotFoundError(hidden_path)

        api.app.dependency_overrides[api.get_index] = missing_index
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(), {"detail": "Local index is unavailable."}
        )
        self.assertNotIn(hidden_path, response.text)

    def test_missing_gemini_configuration_returns_service_unavailable(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(api, "ask_question") as orchestration:
                response = self.client.post(
                    "/api/ask", json={"question": "Question valide ?"}
                )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {"detail": "Gemini configuration is unavailable."},
        )
        orchestration.assert_not_called()

    def test_upstream_failures_return_safe_bad_gateway_response(self):
        failures = (
            EmbeddingError("upstream response contains secret-value"),
            GenerationError("structured output failed at /private/path"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                with patch.dict(
                    os.environ,
                    {"GEMINI_API_KEY": "secret-value"},
                    clear=False,
                ):
                    with patch.object(
                        api, "ask_question", side_effect=failure
                    ):
                        response = self.client.post(
                            "/api/ask",
                            json={"question": "Question valide ?"},
                        )

                self.assertEqual(response.status_code, 502)
                self.assertEqual(
                    response.json(),
                    {"detail": "The grounded-answer service failed."},
                )
                self.assertNotIn("secret-value", response.text)
                self.assertNotIn("/private/path", response.text)

    def test_index_cache_can_be_cleared_and_replaced_safely(self):
        api.app.dependency_overrides.clear()
        first_index = fake_index()
        second_index = fake_index()
        second_index["metadata"]["total_chunk_count"] = 71

        with patch.object(
            api, "load_index", side_effect=[first_index, second_index]
        ) as loader:
            api.clear_index_cache()
            self.assertIs(api.get_index(), first_index)
            self.assertIs(api.get_index(), first_index)
            self.assertEqual(loader.call_count, 1)
            api.clear_index_cache()
            self.assertIs(api.get_index(), second_index)
            self.assertEqual(loader.call_count, 2)


if __name__ == "__main__":
    unittest.main()

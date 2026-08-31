import copy
import io
import math
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src import cli
from src.embeddings import EmbeddingError, embed_query, format_query_input
from src.retrieval import (
    RetrievalError,
    SearchResponse,
    SearchResult,
    cosine_similarity,
    dot_product,
    euclidean_norm,
    rank_items,
    search_index,
)


def indexed_item(chunk_id, embedding, chapter_number=1, chunk_position=1):
    return {
        "chunk_id": chunk_id,
        "document_id": "lancelot-01",
        "work_title": "Les Enfances de Lancelot",
        "chapter_number": chapter_number,
        "chapter_title": "Fuite du roi Ban",
        "chunk_position": chunk_position,
        "source_url": "https://fr.wikisource.org/wiki/example",
        "text": f"Complete text for {chunk_id}.",
        "word_count": 4,
        "embedding": embedding,
    }


def small_index():
    return {
        "metadata": {
            "embedding_model": "test-model",
            "embedding_dimensions": 2,
            "total_chunk_count": 3,
        },
        "items": [
            indexed_item("chunk-a", [1.0, 0.0], chunk_position=1),
            indexed_item("chunk-b", [0.8, 0.6], chunk_position=2),
            indexed_item("chunk-c", [0.0, 1.0], chunk_position=3),
        ],
    }


class RetrievalTests(unittest.TestCase):
    def test_dot_product_and_euclidean_norm_use_known_vectors(self):
        self.assertEqual(dot_product([1.0, 2.0], [3.0, 4.0]), 11.0)
        self.assertEqual(euclidean_norm([3.0, 4.0]), 5.0)
        self.assertAlmostEqual(
            cosine_similarity([1.0, 1.0], [1.0, 0.0]),
            1.0 / math.sqrt(2.0),
        )

    def test_identical_and_orthogonal_cosine_similarity(self):
        self.assertAlmostEqual(cosine_similarity([0.6, 0.8], [0.6, 0.8]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_dimension_mismatch_and_zero_vectors_are_rejected(self):
        with self.assertRaisesRegex(RetrievalError, "dimension mismatch"):
            cosine_similarity([1.0, 0.0], [1.0])
        with self.assertRaisesRegex(RetrievalError, "zero vector"):
            cosine_similarity([0.0, 0.0], [1.0, 0.0])
        with self.assertRaisesRegex(RetrievalError, "must not be empty"):
            euclidean_norm([])

    def test_malformed_numeric_values_are_rejected(self):
        with self.assertRaisesRegex(RetrievalError, "non-finite"):
            cosine_similarity([float("nan"), 0.0], [1.0, 0.0])
        with self.assertRaisesRegex(RetrievalError, "non-finite"):
            cosine_similarity([True, 0.0], [1.0, 0.0])

    def test_ranking_is_descending_with_index_order_as_tie_breaker(self):
        items = [
            indexed_item("first-tie", [1.0, 0.0]),
            indexed_item("second-tie", [1.0, 0.0]),
            indexed_item("lower", [0.0, 1.0]),
        ]
        results = rank_items([1.0, 0.0], items, top_k=3)
        self.assertEqual(
            [result.item["chunk_id"] for result in results],
            ["first-tie", "second-tie", "lower"],
        )
        self.assertEqual([result.rank for result in results], [1, 2, 3])
        self.assertGreaterEqual(results[0].score, results[1].score)
        self.assertGreaterEqual(results[1].score, results[2].score)

    def test_top_k_is_validated(self):
        items = small_index()["items"]
        for invalid_top_k in (0, -1, 4):
            with self.subTest(top_k=invalid_top_k):
                with self.assertRaisesRegex(RetrievalError, "between 1 and 3"):
                    rank_items([1.0, 0.0], items, invalid_top_k)
        with self.assertRaisesRegex(RetrievalError, "must be an integer"):
            rank_items([1.0, 0.0], items, True)

    def test_search_embeds_once_with_index_configuration_and_does_not_mutate(self):
        index = small_index()
        original = copy.deepcopy(index)
        calls = []

        def fake_query_embedder(question, model, dimensions):
            calls.append((question, model, dimensions))
            return [1.0, 0.0]

        response = search_index(
            index,
            "  Qui recueille Lancelot ?  ",
            top_k=2,
            query_embedder=fake_query_embedder,
        )
        self.assertEqual(calls, [("Qui recueille Lancelot ?", "test-model", 2)])
        self.assertEqual(index, original)
        self.assertEqual(len(response.results), 2)
        self.assertEqual(response.query_vector, [1.0, 0.0])
        self.assertAlmostEqual(response.query_norm, 1.0)

    def test_search_rejects_question_and_query_vector_before_ranking(self):
        calls = []

        def fake_query_embedder(question, model, dimensions):
            calls.append((question, model, dimensions))
            return [1.0, 0.0]

        with self.assertRaisesRegex(RetrievalError, "question must not be empty"):
            search_index(small_index(), "   ", query_embedder=fake_query_embedder)
        self.assertEqual(calls, [])

        with self.assertRaisesRegex(RetrievalError, "dimension 1.*index dimension 2"):
            search_index(
                small_index(),
                "question",
                query_embedder=lambda question, model, dimensions: [1.0],
            )
        with self.assertRaisesRegex(RetrievalError, "zero vector"):
            search_index(
                small_index(),
                "question",
                query_embedder=lambda question, model, dimensions: [0.0, 0.0],
            )

    def test_query_format_and_empty_query_embedding(self):
        self.assertEqual(
            format_query_input("  Qui recueille Lancelot ?  "),
            "task: question answering | query: Qui recueille Lancelot ?",
        )
        with self.assertRaisesRegex(EmbeddingError, "question must not be empty"):
            format_query_input(" \t ")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(EmbeddingError, "GEMINI_API_KEY is not set"):
                embed_query("Qui recueille Lancelot ?")

    def test_embed_query_makes_one_sdk_request_for_one_vector(self):
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.6, 0.8])]
        )

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            with patch("google.genai.Client", return_value=fake_client) as client_type:
                vector = embed_query(
                    "Qui recueille Lancelot ?",
                    model="model-from-index",
                    dimensions=2,
                )

        self.assertEqual(vector, [0.6, 0.8])
        client_type.assert_called_once_with(api_key="test-key")
        fake_client.models.embed_content.assert_called_once()
        request = fake_client.models.embed_content.call_args.kwargs
        self.assertEqual(request["model"], "model-from-index")
        self.assertEqual(
            request["contents"].parts[0].text,
            "task: question answering | query: Qui recueille Lancelot ?",
        )
        self.assertEqual(request["config"].output_dimensionality, 2)

    def run_cli(self, arguments, **patches):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with patch.multiple("src.cli", **patches):
                result = cli.main(arguments)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_search_cli_prints_observable_results_without_full_query_vector(self):
        index = small_index()
        response = SearchResponse(
            query_vector=[1.0, 0.0],
            query_norm=1.0,
            results=[
                SearchResult(rank=1, score=0.812345678, item=index["items"][0])
            ],
        )
        result, output, error = self.run_cli(
            ["search", "Qui recueille Lancelot ?"],
            load_index=lambda: index,
            search_index=lambda loaded, question, top_k: response,
        )
        self.assertEqual(result, 0, error)
        self.assertIn("Query vector dimensions: 2", output)
        self.assertIn("Query vector norm: 1.00000000", output)
        self.assertIn("Cosine similarity: 0.81234568", output)
        self.assertIn("Chunk ID: chunk-a", output)
        self.assertIn("Chapter: 1 — Fuite du roi Ban", output)
        self.assertIn("Complete text for chunk-a.", output)
        self.assertIn("https://fr.wikisource.org/wiki/example", output)
        self.assertNotIn(str(response.query_vector), output)

    def test_search_cli_reports_expected_errors(self):
        result, _output, error = self.run_cli(
            ["search", "question"],
            load_index=lambda: (_ for _ in ()).throw(
                FileNotFoundError("index not found")
            ),
        )
        self.assertEqual(result, 2)
        self.assertIn("index not found", error)

        result, _output, error = self.run_cli(
            ["search", "question", "--top-k", "0"],
            load_index=lambda: small_index(),
            search_index=lambda loaded, question, top_k: (_ for _ in ()).throw(
                RetrievalError("top-k must be between 1 and 3")
            ),
        )
        self.assertEqual(result, 2)
        self.assertIn("top-k must be between 1 and 3", error)

        result, _output, error = self.run_cli(
            ["search", "question"],
            load_index=lambda: small_index(),
            search_index=lambda loaded, question, top_k: (_ for _ in ()).throw(
                EmbeddingError("GEMINI_API_KEY is not set")
            ),
        )
        self.assertEqual(result, 2)
        self.assertIn("GEMINI_API_KEY is not set", error)


if __name__ == "__main__":
    unittest.main()

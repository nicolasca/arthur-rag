import copy
import io
import math
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import cli
from src.chunking import load_chunks
from src.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EmbeddingError,
    _vectors_from_response,
    embed_chunks,
    format_embedding_input,
)
from src.indexing import (
    IndexValidationError,
    build_and_save_index,
    build_index,
    find_indexed_item,
    index_stats,
    load_index,
    save_index,
    validate_index,
)


def fake_embedder(chunks):
    vectors = []
    for position, _chunk in enumerate(chunks):
        vector = [0.0] * EMBEDDING_DIMENSIONS
        vector[position % EMBEDDING_DIMENSIONS] = 1.0
        vectors.append(vector)
    return vectors


class IndexingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_chunks()
        cls.index = build_index(fake_embedder)

    def test_build_has_one_distinct_vector_per_chunk(self):
        self.assertEqual(len(self.index["items"]), 70)
        self.assertEqual(
            [item["chunk_id"] for item in self.index["items"]],
            [chunk.chunk_id for chunk in self.chunks],
        )
        vectors = [tuple(item["embedding"]) for item in self.index["items"]]
        self.assertEqual(len(vectors), len(set(vectors)))
        self.assertTrue(all(len(vector) == 768 for vector in vectors))
        self.assertTrue(
            all(math.isfinite(value) for vector in vectors for value in vector)
        )

    def test_metadata_and_items_contain_the_required_fields(self):
        self.assertEqual(
            self.index["metadata"],
            {
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimensions": 768,
                "chunk_maximum_words": 300,
                "overlap_target": 50,
                "total_document_count": 19,
                "total_chunk_count": 70,
            },
        )
        item = self.index["items"][0]
        self.assertEqual(
            set(item),
            {
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
            },
        )

    def test_fake_builds_are_structurally_deterministic(self):
        self.assertEqual(self.index, build_index(fake_embedder))

    def test_embedder_count_must_match_chunk_count(self):
        with self.assertRaisesRegex(IndexValidationError, "69 vectors for 70"):
            build_index(lambda chunks: fake_embedder(chunks)[:-1])

    def test_validation_rejects_empty_duplicate_and_reordered_items(self):
        empty = copy.deepcopy(self.index)
        empty["items"] = []
        empty["metadata"]["total_chunk_count"] = 0
        with self.assertRaisesRegex(IndexValidationError, "at least one"):
            validate_index(empty)

        duplicate = copy.deepcopy(self.index)
        duplicate["items"][1]["chunk_id"] = duplicate["items"][0]["chunk_id"]
        with self.assertRaisesRegex(IndexValidationError, "duplicate"):
            validate_index(duplicate)

        reordered = copy.deepcopy(self.index)
        reordered["items"][0], reordered["items"][1] = (
            reordered["items"][1],
            reordered["items"][0],
        )
        with self.assertRaisesRegex(IndexValidationError, "ordering"):
            validate_index(reordered)

    def test_validation_rejects_missing_nonfinite_and_wrong_size_vectors(self):
        missing = copy.deepcopy(self.index)
        missing["items"][0]["embedding"][0] = None
        with self.assertRaisesRegex(IndexValidationError, "missing or non-finite"):
            validate_index(missing)

        nonfinite = copy.deepcopy(self.index)
        nonfinite["items"][0]["embedding"][0] = float("nan")
        with self.assertRaisesRegex(IndexValidationError, "missing or non-finite"):
            validate_index(nonfinite)

        wrong_size = copy.deepcopy(self.index)
        wrong_size["items"][0]["embedding"].pop()
        with self.assertRaisesRegex(IndexValidationError, "dimension 767"):
            validate_index(wrong_size)

    def test_validation_rejects_incorrect_provenance(self):
        malformed = copy.deepcopy(self.index)
        malformed["items"][0]["source_url"] = "https://example.invalid"
        with self.assertRaisesRegex(IndexValidationError, "incorrect source_url"):
            validate_index(malformed)

    def test_save_is_readable_atomic_and_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "index.json"
            save_index(self.index, path)
            contents = path.read_text(encoding="utf-8")
            self.assertIn('\n  "metadata": {', contents)
            self.assertEqual(load_index(path), self.index)

    def test_embedding_failure_does_not_replace_existing_index(self):
        def failing_embedder(_chunks):
            raise RuntimeError("simulated failure")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text("existing index", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                build_and_save_index(failing_embedder, path)
            self.assertEqual(path.read_text(encoding="utf-8"), "existing index")

    def test_norm_statistics_are_one_for_fake_vectors(self):
        stats = index_stats(self.index)
        self.assertEqual(stats["item_count"], 70)
        self.assertEqual(stats["vector_dimensions"], 768)
        self.assertAlmostEqual(stats["minimum_norm"], 1.0)
        self.assertAlmostEqual(stats["maximum_norm"], 1.0)
        self.assertAlmostEqual(stats["average_norm"], 1.0)

    def test_embedding_input_and_response_validation(self):
        formatted = format_embedding_input(self.chunks[0])
        self.assertEqual(
            formatted,
            f"title: {self.chunks[0].chapter_title} | text: {self.chunks[0].text}",
        )
        response = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[1, 2]), SimpleNamespace(values=[3, 4])]
        )
        self.assertEqual(_vectors_from_response(response, 2), [[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaisesRegex(EmbeddingError, "1 embeddings for 2"):
            _vectors_from_response(SimpleNamespace(embeddings=response.embeddings[:1]), 2)
        with self.assertRaisesRegex(EmbeddingError, "no embeddings"):
            _vectors_from_response(SimpleNamespace(), 1)

    def test_missing_api_key_fails_before_any_network_request(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(EmbeddingError, "GEMINI_API_KEY is not set"):
                embed_chunks(self.chunks[:1])

    def test_find_indexed_item(self):
        item = find_indexed_item(self.index, "lancelot-01-chunk-001")
        self.assertEqual(item["document_id"], "lancelot-01")
        with self.assertRaises(KeyError):
            find_indexed_item(self.index, "not-a-chunk")

    def run_cli(self, arguments, **patches):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with patch.multiple("src.cli", **patches):
                result = cli.main(arguments)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_index_cli_commands_without_network(self):
        result, output, error = self.run_cli(
            ["index", "build"], build_and_save_index=lambda: self.index
        )
        self.assertEqual(result, 0, error)
        self.assertIn("Wrote 70 indexed chunks", output)

        result, output, error = self.run_cli(
            ["index", "stats"], load_index=lambda: self.index
        )
        self.assertEqual(result, 0, error)
        self.assertIn("Embedding model: gemini-embedding-2", output)
        self.assertIn("Minimum vector norm: 1.000000", output)

        result, output, error = self.run_cli(
            ["index", "show", "lancelot-01-chunk-001"],
            load_index=lambda: self.index,
        )
        self.assertEqual(result, 0, error)
        self.assertIn("Vector dimensions: 768", output)
        self.assertIn("Vector preview: ", output)
        self.assertNotIn(str(self.index["items"][0]["embedding"]), output)

    def test_index_cli_reports_expected_errors(self):
        result, _output, error = self.run_cli(
            ["index", "build"],
            build_and_save_index=lambda: (_ for _ in ()).throw(
                EmbeddingError("simulated API failure")
            ),
        )
        self.assertEqual(result, 2)
        self.assertIn("simulated API failure", error)

        result, _output, error = self.run_cli(
            ["index", "stats"],
            load_index=lambda: (_ for _ in ()).throw(
                FileNotFoundError("index not found")
            ),
        )
        self.assertEqual(result, 2)
        self.assertIn("index not found", error)

        result, _output, error = self.run_cli(
            ["index", "show", "not-a-chunk"], load_index=lambda: self.index
        )
        self.assertEqual(result, 2)
        self.assertIn("unknown indexed chunk ID", error)


if __name__ == "__main__":
    unittest.main()

import subprocess
import sys
import unittest

from src.chunking import (
    DEFAULT_MAX_WORDS,
    DEFAULT_OVERLAP_WORDS,
    chunk_document,
    chunk_text,
    corpus_chunk_stats,
    load_chunks,
)
from src.corpus import PROJECT_ROOT, load_documents, read_document


def shared_boundary_words(first: str, second: str) -> int:
    first_words = first.split()
    second_words = second.split()
    for size in range(min(len(first_words), len(second_words)), 0, -1):
        if first_words[-size:] == second_words[:size]:
            return size
    return 0


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = load_documents()
        cls.chunks = load_chunks()

    def test_generation_is_deterministic_with_stable_unique_ids(self):
        self.assertEqual(self.chunks, load_chunks())
        identifiers = [chunk.chunk_id for chunk in self.chunks]
        self.assertEqual(len(identifiers), len(set(identifiers)))

        for document in self.documents:
            chunks = chunk_document(document)
            self.assertEqual(
                [chunk.chunk_id for chunk in chunks],
                [
                    f'{document["id"]}-chunk-{position:03}'
                    for position in range(1, len(chunks) + 1)
                ],
            )

    def test_chunks_are_nonempty_bounded_source_spans(self):
        sources = {
            document["id"]: read_document(document).strip("\n")
            for document in self.documents
        }
        for chunk in self.chunks:
            self.assertTrue(chunk.text)
            self.assertEqual(chunk.word_count, len(chunk.text.split()))
            self.assertLessEqual(chunk.word_count, DEFAULT_MAX_WORDS)
            self.assertIn(chunk.text, sources[chunk.source_document_id])

    def test_every_chunk_preserves_its_document_provenance(self):
        documents = {document["id"]: document for document in self.documents}
        for chunk in self.chunks:
            document = documents[chunk.source_document_id]
            self.assertEqual(chunk.work_title, document["work_title"])
            self.assertEqual(chunk.chapter_number, document["chapter_number"])
            self.assertEqual(chunk.chapter_title, document["chapter_title"])
            self.assertEqual(chunk.source_url, document["source_url"])

    def test_consecutive_chunks_overlap_in_source_order(self):
        for document in self.documents:
            chunks = chunk_document(document)
            for first, second in zip(chunks, chunks[1:]):
                self.assertGreater(
                    shared_boundary_words(first.text, second.text),
                    0,
                    f"{first.chunk_id} -> {second.chunk_id}",
                )

    def test_oversized_paragraph_uses_sentence_boundaries(self):
        sentences = [
            "Un deux trois quatre cinq six.",
            "Sept huit neuf dix onze douze.",
            "Treize quatorze quinze seize dix-sept dix-huit.",
        ]
        chunks = chunk_text(
            self.documents[0],
            " ".join(sentences),
            max_words=12,
            overlap_words=6,
        )
        self.assertEqual([chunk.word_count for chunk in chunks], [12, 12])
        self.assertEqual(chunks[0].text, " ".join(sentences[:2]))
        self.assertEqual(chunks[1].text, " ".join(sentences[1:]))

    def test_single_oversized_sentence_uses_exact_word_boundaries(self):
        text = "un deux trois quatre cinq six sept huit neuf dix onze douze."
        chunks = chunk_text(
            self.documents[0], text, max_words=5, overlap_words=1
        )
        self.assertEqual([chunk.word_count for chunk in chunks], [5, 5, 3])
        for chunk in chunks:
            self.assertIn(chunk.text, text)
            self.assertLessEqual(chunk.word_count, 5)

    def test_statistics_describe_the_generated_chunks(self):
        stats = corpus_chunk_stats()
        sizes = [chunk.word_count for chunk in self.chunks]
        self.assertEqual(stats["source_documents"], 19)
        self.assertEqual(stats["total_chunks"], len(self.chunks))
        self.assertEqual(stats["minimum_words"], min(sizes))
        self.assertEqual(stats["maximum_words"], max(sizes))
        self.assertEqual(stats["average_words"], sum(sizes) / len(sizes))
        self.assertEqual(stats["configured_maximum"], DEFAULT_MAX_WORDS)
        self.assertEqual(stats["overlap_target"], DEFAULT_OVERLAP_WORDS)

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "src.cli", *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_chunk_cli_commands(self):
        listed = self.run_cli("chunks", "lancelot-01")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertIn("lancelot-01-chunk-001", listed.stdout)
        self.assertIn("words", listed.stdout)

        shown = self.run_cli("chunk", "lancelot-01-chunk-001")
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertIn("Chunk ID: lancelot-01-chunk-001", shown.stdout)
        self.assertIn("Source document ID: lancelot-01", shown.stdout)
        self.assertIn("# I — Fuite du roi Ban", shown.stdout)

        stats = self.run_cli("stats")
        self.assertEqual(stats.returncode, 0, stats.stderr)
        self.assertIn("Source documents: 19", stats.stdout)
        self.assertIn("Configured maximum words: 300", stats.stdout)
        self.assertIn("Overlap target words: 50", stats.stdout)

    def test_chunk_cli_rejects_unknown_ids(self):
        document = self.run_cli("chunks", "not-a-document")
        self.assertNotEqual(document.returncode, 0)
        self.assertIn("unknown document ID: not-a-document", document.stderr)

        chunk = self.run_cli("chunk", "not-a-chunk")
        self.assertNotEqual(chunk.returncode, 0)
        self.assertIn("unknown chunk ID: not-a-chunk", chunk.stderr)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text(self.documents[0], "texte", max_words=0)
        with self.assertRaises(ValueError):
            chunk_text(self.documents[0], "texte", overlap_words=300)


if __name__ == "__main__":
    unittest.main()

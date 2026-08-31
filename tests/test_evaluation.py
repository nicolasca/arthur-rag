import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation import (
    CASES_PATH,
    EvaluationCase,
    EvaluationError,
    format_answer_report,
    load_evaluation_cases,
    run_answer_evaluation,
    run_retrieval_evaluation,
    select_cases,
    write_evaluation_result,
)


def ranked_item(position):
    embeddings = (
        [1.0, 0.0],
        [0.9, 0.1],
        [0.8, 0.2],
        [0.7, 0.3],
        [0.6, 0.4],
        [0.5, 0.5],
    )
    chunk_id = f"chunk-{position}"
    return {
        "chunk_id": chunk_id,
        "document_id": "lancelot-test",
        "work_title": "Les Enfances de Lancelot",
        "chapter_number": position,
        "chapter_title": f"Chapitre {position}",
        "chunk_position": 1,
        "source_url": f"https://local.example/{chunk_id}",
        "text": f"Le passage numéro {position} contient une preuve locale exacte.",
        "word_count": 9,
        "embedding": embeddings[position - 1],
    }


def ranked_index():
    return {
        "metadata": {
            "embedding_model": "fake-embedding-model",
            "embedding_dimensions": 2,
            "total_chunk_count": 6,
        },
        "items": [ranked_item(position) for position in range(1, 7)],
    }


def metric_cases():
    cases = [
        EvaluationCase(
            case_id=f"answered-{position}",
            question=f"Question répondue {position} ?",
            expected_status="answered",
            acceptable_evidence_chunk_ids=(f"chunk-{position}",),
            rationale="Une preuve synthétique directe.",
        )
        for position in range(1, 7)
    ]
    cases.extend(
        EvaluationCase(
            case_id=f"insufficient-{position}",
            question=f"Question insuffisante {position} ?",
            expected_status="insufficient",
            acceptable_evidence_chunk_ids=(),
            rationale="Aucune preuve synthétique acceptable.",
        )
        for position in range(1, 5)
    )
    return cases


def case_file_index(raw_cases):
    chunk_ids = {
        chunk_id
        for case in raw_cases
        for chunk_id in case.get("acceptable_evidence_chunk_ids", [])
        if isinstance(chunk_id, str) and chunk_id != "unknown-chunk"
    }
    return {
        "metadata": {"embedding_model": "fake", "embedding_dimensions": 2},
        "items": [
            {"chunk_id": chunk_id}
            for chunk_id in sorted(chunk_ids)
        ],
    }


class EvaluationTests(unittest.TestCase):
    def write_json(self, directory, value):
        path = Path(directory) / "cases.json"
        path.write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def index_file(self, directory):
        path = Path(directory) / "index.json"
        path.write_text('{"test": true}\n', encoding="utf-8")
        return path

    def test_curated_dataset_has_exact_required_shape(self):
        raw_cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        cases = load_evaluation_cases(case_file_index(raw_cases))

        self.assertEqual(len(cases), 10)
        self.assertEqual(
            sum(case.expected_status == "answered" for case in cases), 6
        )
        self.assertEqual(
            sum(case.expected_status == "insufficient" for case in cases), 4
        )
        self.assertEqual(
            [case.case_id for case in cases[:3]],
            [
                "raises-lancelot",
                "ban-leaves-trebe",
                "lionel-bohor-brothers",
            ],
        )

    def test_malformed_duplicate_and_unknown_cases_are_rejected(self):
        raw_cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            malformed = json.loads(json.dumps(raw_cases))
            malformed[0]["expected_status"] = "perhaps"
            with self.assertRaisesRegex(EvaluationError, "expected_status"):
                load_evaluation_cases(
                    case_file_index(malformed),
                    self.write_json(directory, malformed),
                )

            duplicate = json.loads(json.dumps(raw_cases))
            duplicate[1]["case_id"] = duplicate[0]["case_id"]
            with self.assertRaisesRegex(EvaluationError, "duplicate"):
                load_evaluation_cases(
                    case_file_index(duplicate),
                    self.write_json(directory, duplicate),
                )

            unknown = json.loads(json.dumps(raw_cases))
            unknown[0]["acceptable_evidence_chunk_ids"] = ["unknown-chunk"]
            with self.assertRaisesRegex(EvaluationError, "unknown chunk ID"):
                load_evaluation_cases(
                    case_file_index(unknown),
                    self.write_json(directory, unknown),
                )

    def test_case_filtering_preserves_order_and_rejects_unknown_id(self):
        cases = metric_cases()
        self.assertEqual(select_cases(cases, None), cases)
        self.assertEqual(
            select_cases(cases, "answered-3"), [cases[2]]
        )
        with self.assertRaisesRegex(EvaluationError, "unknown evaluation case"):
            select_cases(cases, "missing-case")

    def test_retrieval_hit_at_1_3_and_5_and_insufficient_exclusion(self):
        embedding_calls = []

        def fake_embedder(question, model, dimensions):
            embedding_calls.append((question, model, dimensions))
            return [1.0, 0.0]

        with tempfile.TemporaryDirectory() as directory:
            result = run_retrieval_evaluation(
                ranked_index(),
                metric_cases(),
                top_k=5,
                query_embedder=fake_embedder,
                index_path=self.index_file(directory),
            )

        self.assertEqual(len(embedding_calls), 10)
        metrics = result["aggregate_metrics"]
        self.assertEqual(metrics["scored_answerable_cases"], 6)
        self.assertAlmostEqual(metrics["hit_at_1"], 1 / 6)
        self.assertAlmostEqual(metrics["hit_at_3"], 3 / 6)
        self.assertAlmostEqual(metrics["hit_at_5"], 5 / 6)
        self.assertEqual(result["cases"][1]["first_acceptable_rank"], 2)
        self.assertIsNone(result["cases"][6]["hits"])
        self.assertEqual(
            result["cases"][6]["retrieved_results"][0]["chunk_id"],
            "chunk-1",
        )

    def test_answer_status_accuracy_and_acceptable_evidence_matching(self):
        cases = [
            EvaluationCase(
                "first-answer",
                "Première question ?",
                "answered",
                ("chunk-1",),
                "Preuve au premier rang.",
            ),
            EvaluationCase(
                "second-answer",
                "Deuxième question ?",
                "answered",
                ("chunk-2",),
                "Preuve au deuxième rang.",
            ),
            EvaluationCase(
                "one-insufficient",
                "Question sans réponse ?",
                "insufficient",
                (),
                "Les passages sont insuffisants.",
            ),
        ]

        def fake_generator(prompt, allowed_evidence_ids):
            if "Question sans réponse ?" in prompt:
                return {
                    "status": "insufficient",
                    "answer": "Les passages sont insuffisants.",
                    "evidence_ids": [],
                }
            return {
                "status": "answered",
                "answer": "Réponse synthétique.",
                "evidence_ids": [allowed_evidence_ids[0]],
            }

        with tempfile.TemporaryDirectory() as directory:
            result = run_answer_evaluation(
                ranked_index(),
                cases,
                top_k=5,
                query_embedder=lambda question, model, dimensions: [1.0, 0.0],
                generator=fake_generator,
                index_path=self.index_file(directory),
            )

        metrics = result["aggregate_metrics"]
        self.assertEqual(metrics["status_accuracy"], 1.0)
        self.assertEqual(metrics["evidence_hit_rate"], 0.5)
        self.assertEqual(metrics["validation_error_count"], 0)
        self.assertEqual(metrics["generation_error_count"], 0)
        self.assertTrue(
            result["cases"][0]["automatic_checks"][
                "acceptable_evidence_cited"
            ]
        )
        self.assertFalse(
            result["cases"][1]["automatic_checks"][
                "acceptable_evidence_cited"
            ]
        )
        self.assertTrue(
            result["cases"][2]["automatic_checks"][
                "insufficient_status_confirmed"
            ]
        )
        report = format_answer_report(result)
        self.assertIn("MANUAL REVIEW REQUIRED", report)
        self.assertIn("Does the answer directly address the question?", report)
        self.assertIn("Réponse synthétique.", report)

    def test_validation_error_is_recorded_without_a_retry(self):
        case = EvaluationCase(
            "invalid-evidence",
            "Question avec preuve invalide ?",
            "answered",
            ("chunk-1",),
            "La réponse attend une preuve valide.",
        )
        generation_calls = []

        def invalid_generator(prompt, allowed_evidence_ids):
            generation_calls.append(prompt)
            return {
                "status": "answered",
                "answer": "Réponse.",
                "evidence_ids": ["unknown-evidence"],
            }

        with tempfile.TemporaryDirectory() as directory:
            result = run_answer_evaluation(
                ranked_index(),
                [case],
                top_k=5,
                query_embedder=lambda question, model, dimensions: [1.0, 0.0],
                generator=invalid_generator,
                index_path=self.index_file(directory),
            )

        self.assertEqual(len(generation_calls), 1)
        self.assertEqual(
            result["aggregate_metrics"]["validation_error_count"], 1
        )
        self.assertEqual(
            result["aggregate_metrics"]["generation_error_count"], 0
        )
        self.assertEqual(len(result["cases"][0]["retrieved_results"]), 5)
        self.assertFalse(
            result["cases"][0]["automatic_checks"][
                "citation_validation_passed"
            ]
        )

    def test_generation_request_error_is_not_counted_as_validation_error(self):
        case = EvaluationCase(
            "service-error",
            "Question interrompue ?",
            "answered",
            ("chunk-1",),
            "La génération échoue avant validation.",
        )

        def failing_generator(prompt, allowed_evidence_ids):
            from src.generation import GenerationError

            raise GenerationError(
                "Gemini answer generation failed: 503 UNAVAILABLE"
            )

        with tempfile.TemporaryDirectory() as directory:
            result = run_answer_evaluation(
                ranked_index(),
                [case],
                top_k=5,
                query_embedder=lambda question, model, dimensions: [1.0, 0.0],
                generator=failing_generator,
                index_path=self.index_file(directory),
            )

        metrics = result["aggregate_metrics"]
        self.assertEqual(metrics["validation_error_count"], 0)
        self.assertEqual(metrics["generation_error_count"], 1)
        self.assertEqual(result["cases"][0]["error_type"], "generation")
        self.assertEqual(len(result["cases"][0]["retrieved_results"]), 5)

    def test_optional_result_serialization_is_readable_and_vector_free(self):
        result = {
            "evaluation_type": "retrieval",
            "embedding_model": "fake-model",
            "index_sha256": "abc123",
            "top_k": 5,
            "aggregate_metrics": {"hit_at_1": 1.0},
            "cases": [
                {
                    "case_id": "one-case",
                    "retrieved_results": [
                        {"rank": 1, "chunk_id": "chunk-1", "score": 0.9}
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "results.json"
            self.assertFalse(output.exists())
            write_evaluation_result(output, result)
            loaded = json.loads(output.read_text(encoding="utf-8"))
            raw_text = output.read_text(encoding="utf-8")

        self.assertEqual(loaded, result)
        self.assertNotIn("embedding", loaded["cases"][0]["retrieved_results"][0])
        self.assertNotIn("GEMINI_API_KEY", raw_text)
        self.assertTrue(raw_text.endswith("\n"))


if __name__ == "__main__":
    unittest.main()

import re
import subprocess
import sys
import unittest
from pathlib import Path

from src.corpus import PROJECT_ROOT, load_documents, read_document


EXPECTED_TITLES = [
    "Fuite du roi Ban",
    "Prise de Trèbe",
    "Le roi qui mourut de deuil",
    "La reine aux grandes douleurs",
    "Les fils du roi Bohor",
    "Claudas de la Terre déserte",
    "La Dame du Lac et Lancelot",
    "Le cheval donné",
    "La venaison donnée",
    "Lancelot et son maître",
    "La pucelle Saraide",
    "Lionel",
    "Les lévriers enchantés",
    "Délivrance des enfants",
    "Prise de Claudas",
    "Pharien et Lambègue au Lac",
    "Les cousins",
    "Les mères",
    "La chevalerie",
]

EXPECTED_FILENAMES = [
    "01-fuite-du-roi-ban.md",
    "02-prise-de-trebe.md",
    "03-le-roi-qui-mourut-de-deuil.md",
    "04-la-reine-aux-grandes-douleurs.md",
    "05-les-fils-du-roi-bohor.md",
    "06-claudas-de-la-terre-deserte.md",
    "07-la-dame-du-lac-et-lancelot.md",
    "08-le-cheval-donne.md",
    "09-la-venaison-donnee.md",
    "10-lancelot-et-son-maitre.md",
    "11-la-pucelle-saraide.md",
    "12-lionel.md",
    "13-les-levriers-enchantes.md",
    "14-delivrance-des-enfants.md",
    "15-prise-de-claudas.md",
    "16-pharien-et-lambegue-au-lac.md",
    "17-les-cousins.md",
    "18-les-meres.md",
    "19-la-chevalerie.md",
]


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = load_documents()

    def test_manifest_has_the_ordered_nineteen_chapters(self):
        self.assertEqual(len(self.documents), 19)
        self.assertEqual(
            [document["chapter_number"] for document in self.documents],
            list(range(1, 20)),
        )
        self.assertEqual(
            [document["chapter_title"] for document in self.documents],
            EXPECTED_TITLES,
        )
        self.assertEqual(
            [Path(document["local_path"]).name for document in self.documents],
            EXPECTED_FILENAMES,
        )
        self.assertEqual(self.documents[0]["chapter_title"], "Fuite du roi Ban")
        self.assertEqual(self.documents[-1]["chapter_title"], "La chevalerie")

    def test_manifest_identifiers_paths_and_titles_are_unique(self):
        for key in ("id", "chapter_number", "chapter_title", "local_path"):
            values = [document[key] for document in self.documents]
            self.assertEqual(len(values), len(set(values)), key)

    def test_every_manifest_entry_has_an_exact_source_and_nonempty_file(self):
        required_fields = {
            "id",
            "work_title",
            "chapter_number",
            "chapter_title",
            "author_adaptor",
            "publisher",
            "publication_year",
            "local_path",
            "source_url",
            "provenance_note",
        }
        for number, document in enumerate(self.documents, start=1):
            self.assertTrue(required_fields <= document.keys())
            self.assertEqual(document["id"], f"lancelot-{number:02}")
            self.assertEqual(
                document["source_url"],
                "https://fr.wikisource.org/wiki/Les_Enfances_de_Lancelot/"
                f"{number:02}",
            )
            path = PROJECT_ROOT / document["local_path"]
            self.assertTrue(path.is_file(), path)
            self.assertTrue(read_document(document).strip(), path)

    def test_corpus_contains_no_wikisource_furniture_or_html(self):
        forbidden = (
            "Wikisource",
            "Modifier",
            "Récupérée de",
            "La bibliothèque libre",
            "pagenum",
            "ws-noexport",
        )
        html_tag = re.compile(r"</?[A-Za-z][^>]*>")
        for document in self.documents:
            text = read_document(document)
            for marker in forbidden:
                self.assertNotIn(marker, text, document["id"])
            self.assertIsNone(html_tag.search(text), document["id"])

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "src.cli", *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_cli_list_displays_all_chapters_in_order(self):
        result = self.run_cli("list")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 19)
        self.assertIn("lancelot-01", lines[0])
        self.assertIn("Fuite du roi Ban", lines[0])
        self.assertIn("lancelot-19", lines[-1])
        self.assertIn("La chevalerie", lines[-1])

    def test_cli_show_displays_metadata_and_text(self):
        result = self.run_cli("show", "lancelot-01")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ID: lancelot-01", result.stdout)
        self.assertIn("Source URL: https://fr.wikisource.org/", result.stdout)
        self.assertIn("# I — Fuite du roi Ban", result.stdout)
        self.assertIn("En la marche de Gaule", result.stdout)

    def test_cli_show_rejects_an_unknown_id(self):
        result = self.run_cli("show", "not-a-document")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown document ID: not-a-document", result.stderr)


if __name__ == "__main__":
    unittest.main()

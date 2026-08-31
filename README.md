# Arthur RAG — corpus foundation

This repository is the first, deliberately small step of an educational RAG project. It contains source documents and simple inspection tools only: there is no chunking, embedding model, vector search, LLM, or generation layer yet.

## Corpus

The initial work is Jacques Boulenger’s French adaptation *Les Enfances de Lancelot*, published by Librairie Plon in 1922. Its 19 editorial chapters are stored as 19 UTF-8 Markdown files under `corpus/`. Each file is one complete source document, not a RAG chunk.

`corpus/manifest.json` is the single machine-readable metadata source. Its ordered `documents` array records each stable ID, title, chapter number, author/adaptor, publication details, local path, exact Wikisource chapter URL, and a provenance note. Metadata is deliberately not duplicated as Markdown front matter.

## How the corpus was obtained

The chapter list and titles come from the [Wikisource table of contents](https://fr.wikisource.org/wiki/Les_Enfances_de_Lancelot). On 2026-08-31, chapters `01` through `19` were retrieved individually through the MediaWiki `action=parse` API for their exact Wikisource pages.

For each rendered page, the conversion kept the chapter body’s paragraphs, dialogue, accents, punctuation, and italic emphasis. It removed the Wikisource header and previous/next navigation, hidden metadata, edit and interface controls, scan page-number markers, styles, and other HTML wrappers. The editorial chapter title from the table of contents was added as the Markdown heading. The dedication and surrounding work-level furniture were not imported. No text was modernized, summarized, translated, or split into chunks.

The local files therefore derive from Wikisource’s validated transcription of the Plon edition. Consult each manifest entry for the precise source page and provenance note, and consult Wikisource for the terms attached to its transcription. Corpus provenance is separate from the source-code license: this repository does not yet assign a license to future application code.

## Inspect locally

Only Python’s standard library is required. From the repository root:

```console
python -m src.cli list
python -m src.cli show lancelot-01
```

`list` prints the 19 documents in editorial order. `show` prints one document’s manifest metadata followed by its full Markdown source. Unknown document IDs produce a clear error and a non-zero exit status.

Run the verification suite with:

```console
python -m unittest discover -s tests
```

The tests check the 19-document order, unique metadata, exact source URLs, local files, absence of Wikisource/HTML furniture, and both CLI commands.

# Arthur RAG — corpus foundation

This repository is a deliberately small educational RAG project. It contains editorial source documents, deterministic chunking, and simple inspection tools. There is still no embedding model, vector search, LLM, or generation layer.

## Corpus

The initial work is Jacques Boulenger’s French adaptation *Les Enfances de Lancelot*, published by Librairie Plon in 1922. Its 19 editorial chapters are stored as 19 UTF-8 Markdown files under `corpus/`. Each file is one complete source document, not a RAG chunk.

`corpus/manifest.json` is the single machine-readable metadata source. Its ordered `documents` array records each stable ID, title, chapter number, author/adaptor, publication details, local path, exact Wikisource chapter URL, and a provenance note. Metadata is deliberately not duplicated as Markdown front matter.

## How the corpus was obtained

The chapter list and titles come from the [Wikisource table of contents](https://fr.wikisource.org/wiki/Les_Enfances_de_Lancelot). On 2026-08-31, chapters `01` through `19` were retrieved individually through the MediaWiki `action=parse` API for their exact Wikisource pages.

For each rendered page, the conversion kept the chapter body’s paragraphs, dialogue, accents, punctuation, and italic emphasis. It removed the Wikisource header and previous/next navigation, hidden metadata, edit and interface controls, scan page-number markers, styles, and other HTML wrappers. The editorial chapter title from the table of contents was added as the Markdown heading. The dedication and surrounding work-level furniture were not imported. No text was modernized, summarized, or translated.

The local files therefore derive from Wikisource’s validated transcription of the Plon edition. Consult each manifest entry for the precise source page and provenance note, and consult Wikisource for the terms attached to its transcription. Corpus provenance is separate from the source-code license: this repository does not yet assign a license to future application code.

## Inspect locally

Only Python’s standard library is required. From the repository root:

```console
python -m src.cli list
python -m src.cli show lancelot-01
python -m src.cli chunks lancelot-01
python -m src.cli chunk lancelot-01-chunk-001
python -m src.cli stats
```

`list` and `show` inspect the 19 editorial documents. `chunks` lists one chapter’s generated chunks with sizes and previews; `chunk` prints complete chunk metadata and text; `stats` summarizes the generated corpus. Unknown IDs produce a clear error and a non-zero exit status.

## Deterministic chunking

`src/chunking.py` generates chunks in memory every time; chunks are not persisted or added to the manifest. The defaults are a maximum of 300 whitespace-delimited words and an overlap target of 50 words.

The algorithm greedily packs complete blank-line-separated Markdown paragraphs. At a boundary it carries forward a trailing source suffix close to the overlap target, preferring sentence boundaries and using an exact word boundary only when no complete trailing sentence fits beside the next paragraph. A paragraph larger than the configured maximum is split at deterministic sentence boundaries; only a single sentence larger than the maximum uses the final word-boundary fallback. Chunk text is always an unchanged contiguous span of its source document.

With the current 19 chapters and default settings, the generated corpus contains 70 chunks: 73 words minimum, 300 maximum, and 227.9 words on average. These values are derived at runtime by `stats` and may change if the corpus or configuration changes.

Smaller chunks make matches more focused but provide less context. Larger chunks preserve more narrative context but can mix several ideas and reduce retrieval precision. Overlap protects context at boundaries, at the cost of duplicated text and a larger retrieval index.

Run the verification suite with:

```console
python -m unittest discover -s tests
```

The tests check the source corpus, deterministic IDs and output, provenance, size limits, overlap, oversized-paragraph fallbacks, statistics, and all CLI commands.

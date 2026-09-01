# Arthur RAG — corpus foundation

This repository is a deliberately small educational RAG project. It contains editorial source documents, deterministic chunking, a local Gemini embedding index, transparent cosine retrieval, one grounded French answer step with locally validated citations, a minimal local HTTP API, and a small React consultation interface. It has no conversation memory, agents, or production infrastructure beyond a single preview-oriented Vercel Services configuration.

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

## Local embedding index

The index is built with the official `google-genai` SDK and stable `gemini-embedding-2` model. Every chunk is sent as its own SDK `Content` object in this retrieval-document form:

```text
title: {chapter title} | text: {chunk text}
```

The model returns one distinct 768-dimensional embedding per chunk. `src/embeddings.py` is the embedding-specific network boundary: it imports Gemini SDK types, reads `GEMINI_API_KEY`, and makes embedding requests without retries or fallback models. The separate generation request is isolated in `src/generation.py`. `src/indexing.py` joins document vectors to chunk text and provenance, validates the complete result, and only then atomically writes an indented `data/index.json`.

Install the project, provide the key through the environment, and build the index. For a local project-specific secret, put `GEMINI_API_KEY=your-key` in the Git-ignored `.env.local`, then source it into the shell:

```console
python -m pip install -e .
set -a
source .env.local
set +a
python -m src.cli index build
```

The Python code reads only the resulting environment variable; it does not parse `.env.local` or require a dotenv package. The key is never printed, copied into the index, or committed to Git. `.env.local` remains ignored.

`data/index.json` is intentionally versioned as a deployment artifact. It contains embeddings derived from the public Wikisource corpus, not credentials, and lets the read-only FastAPI service start without calling Gemini during installation or deployment. Rebuild and recommit it whenever the corpus, deterministic chunking settings, embedding model, or embedding dimensions change. A rebuild is a deliberate local operation using the command above; Vercel never rebuilds the index.

Once a build succeeds, inspect the index without another network request:

```console
python -m src.cli index stats
python -m src.cli index show lancelot-01-chunk-001
```

`index stats` reports model and corpus metadata plus minimum, maximum, and average vector norms. `index show` prints one item’s complete provenance and text, its dimension and norm, and only the first eight vector values.

Before committing a rebuilt artifact, run `python -m src.cli index stats`, review its metadata, and stage the validated file with `git add data/index.json`.

## Semantic retrieval

Search embeds one French question with the index's declared Gemini model and dimensionality, calculates cosine similarity against all 70 stored vectors in plain Python, and prints the highest-scoring passages without generating an answer:

```console
set -a
source .env.local
set +a
python -m src.cli search "Qui a recueilli Lancelot ?"
python -m src.cli search "Qui a recueilli Lancelot ?" --top-k 5
```

Every result includes its rank, cosine score, chunk ID, chapter, complete chunk text, and exact Wikisource URL. The default is three results. Search makes one query-embedding request and never changes `data/index.json`; it has no threshold, adjacent-chunk filtering, reranking, deduplication, or generated answer.

Cosine similarity measures how closely two vector directions align. Since Gemini's 768-dimensional vectors are normalized, their norms are approximately one and cosine similarity is approximately their dot product. A highest-ranked passage is merely the nearest passage in this corpus—it is not proof that the passage actually answers the question, especially for questions outside the corpus.

## Grounded answers

The `ask` command retrieves five passages by default, sends only the question and those passages to `gemini-3.1-flash-lite`, and requests one structured French answer:

```console
python -m src.cli ask "Qui recueille et élève Lancelot ?"
python -m src.cli ask "Qui recueille et élève Lancelot ?" --top-k 5
```

An `answered` result must contain at least one request-local evidence ID. Local code derives four exact citation candidates from each retrieved passage and binds every one to a stable ID such as `evidence-01`. Gemini returns only those IDs; local code resolves each ID to its one fixed chunk, verbatim quote, chapter metadata, and Wikisource URL. This prevents a quote from one passage being paired with another passage's chunk ID. Unknown IDs are rejected and repeated IDs are deterministically collapsed to their first occurrence. An `insufficient` result states that the retrieved passages do not establish an answer and may contain no evidence IDs.

`answered` also means that the response directly answers the question. For relationship or meeting questions, the supplied passages must explicitly connect the named characters; tangential facts or the separate occurrence of both names are insufficient. The generator is instructed not to infer such a connection and not to use external Arthurian knowledge.

Each `ask` execution makes exactly two independent network calls: one query embedding for retrieval, followed by one non-streaming structured generation request. The response is limited to 1,024 output tokens. There are no tools, external grounding sources, retries, follow-up retrievals, conversation memory, or generated citation URLs. Local substring validation proves where a quote occurs; it does not prove that the quote semantically supports every generated claim.

## Local HTTP API

The root-level `app.py` exposes the same synchronous pipeline through FastAPI. After dependency changes, reinstall the editable project and its test extra:

```console
python -m pip install -e ".[test]"
```

Source `.env.local` as shown above, then start the development server from the repository root:

```console
python -m uvicorn app:app --reload
```

The health endpoint validates and reports the cached local index without contacting Gemini:

```console
curl http://127.0.0.1:8000/api/health
```

Ask one question with the server-controlled `top-k=5` pipeline:

```console
curl \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"question":"Qui recueille et élève Lancelot ?"}' \
  http://127.0.0.1:8000/api/ask
```

`POST /api/ask` accepts only a trimmed, non-empty `question` of at most 500 characters. Its typed JSON response contains the existing generated answer, locally resolved citations, and all five ranked passages, but no vectors, prompts, keys, configuration values, or filesystem paths. The API has no CORS middleware, application authentication, persistence, or streaming.

## Local web interface

The `web/` directory contains a deliberately small React, TypeScript, and Vite single-page interface. It sends one trimmed question to `POST /api/ask`, displays the grounded answer and locally resolved citations, and leaves the five retrieved passages in a secondary collapsed section. It does not reproduce retrieval or generation logic in the browser.

Run the API and interface in two terminals from the repository root. In terminal 1:

```console
source .venv/bin/activate
set -a
source .env.local
set +a
python -m uvicorn app:app --reload
```

In terminal 2:

```console
cd web
npm install
npm run dev
```

Open the local URL printed by Vite. During development, Vite proxies same-origin `/api` requests to `http://127.0.0.1:8000`; the React code contains neither a backend base URL nor a Gemini key. `GEMINI_API_KEY` remains in the environment of the Python process, and browser questions are sent to Gemini by that server. Do not submit private or confidential information.

Frontend checks run independently of FastAPI and mock the browser `fetch` boundary:

```console
cd web
npm run lint
npm run typecheck
npm test
npm run build
```

## Vercel Services preview

The root `vercel.json` defines one Vercel project with two independently built services:

| Public route | Service | Build/runtime behavior |
|---|---|---|
| `/api/*` | `backend` | Vercel detects the root `app.py` export `app`, installs `pyproject.toml` dependencies, and runs FastAPI in the Python runtime. |
| All non-API paths | `frontend` | Vercel runs `npm ci` and `npm run build` in `web/`, then serves `web/dist`. A rewrite inside this service selects `index.html` for unknown browser paths, while real assets retain their paths. |

The backend service root is the repository root, so its Python bundle contains the tracked `data/index.json`, corpus metadata, and Python modules. Existing paths are anchored to source files with `Path(__file__)`, not to an assumed working directory. The index is loaded read-only on the first request in a warm function instance and retained by the existing in-process cache for later requests. No Vercel build command writes the index or contacts Gemini.

The browser continues to call the relative URL `/api/ask`. Vercel’s top-level routing sends that same-origin request to FastAPI, so no public backend URL or CORS middleware is required.

Use the current CLI without installing global tooling:

```console
npx --yes vercel@59.10.0 login
npx --yes vercel@59.10.0 link
npx --yes vercel@59.10.0 dev -L
```

The current CLI's local FastAPI Services runtime requires Python 3.12 or newer. Local mode can validate service detection, FastAPI health, and the unified routing order, but the deployed preview remains the authoritative check for the production `/assets/*` paths and SPA fallback.

Link the repository to the single `arthur-rag` project and set its framework preset to **Services**. Before creating a preview, add the environment variable named `GEMINI_API_KEY` manually in the Vercel dashboard for the Preview environment. Never paste its value into source files, `vercel.json`, a command argument, or a frontend variable. If the dashboard cannot scope variables per service, the unprefixed server variable remains unavailable to Vite client code: Vite only exposes explicitly referenced `VITE_*` variables, and this project defines none.

Enable **Vercel Authentication** with **Standard Protection** under the project’s Deployment Protection settings, then create a preview—not a production deployment—with:

```console
npx --yes vercel@59.10.0 deploy
```

Visitor questions are transmitted by the backend to Google Gemini. Do not submit personal or confidential information. Availability and quotas for both Vercel and Gemini free tiers can change and are not guaranteed; do not expose a public production endpoint without reviewing quota-consumption risk.

## Evaluation baseline

The fixed dataset in `evaluation/cases.json` contains six answerable questions and four questions expected to be insufficient. It is intentionally small and curated against the actual deterministic chunk IDs. Run retrieval and answer generation separately:

```console
python -m src.cli eval retrieval
python -m src.cli eval answers
python -m src.cli eval retrieval --case raises-lancelot
python -m src.cli eval answers --case raises-lancelot
```

Both commands accept `--top-k` (default `5`). Retrieval evaluation requires at least five results so it can report Hit@1, Hit@3, and Hit@5. A hit means that at least one acceptable evidence chunk appears by that rank; insufficient cases are displayed but excluded because cosine retrieval always returns neighbours.

Answer evaluation calls the existing `ask` pipeline once per case. It checks status agreement, local citation validation, and whether an answered result cites an acceptable chunk. These checks do not measure semantic correctness. Every complete answer is printed with a manual-review checklist, and no LLM judge or keyword-based semantic score is used.

Add `--output evaluation/results.json` to either command to save readable JSON containing models, the index SHA-256, metrics, ranks, scores, answers, and locally resolved citations. No file is written without that option; the conventional result path is ignored by Git.

The first recorded run is summarized in the versioned `evaluation/baseline.md`; unlike the ignored generated JSON, it preserves the baseline metrics and manual verdicts across future model runs.

The corpus, chunk ordering, IDs, input formatting, index metadata, and JSON structure are deterministic. The floating-point embedding values come from Gemini and may change if Google updates the stable model implementation; rebuilding never adds timestamps or reorders chunks.

See Google’s [embedding guide](https://ai.google.dev/gemini-api/docs/embeddings) and [`gemini-embedding-2` model card](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2) for the model behavior and dimensionality guidance.

Run the verification suite with:

```console
python -m unittest discover -s tests
```

The tests check the source corpus, deterministic chunks, provenance, size limits, overlap, fake-vector index construction, validation and atomic persistence, explicit vector math, stable retrieval ordering, structured generation, citation validation, statistics, and all CLI commands. Tests never make a real network request.

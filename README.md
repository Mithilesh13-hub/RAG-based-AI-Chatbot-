# Agentic AI eBook RAG Chatbot

Document-grounded chatbot: **PDF → chunks → OpenAI embeddings → Pinecone**, and per question
**LangGraph** (`retrieve → evaluate_context → generate | refuse`) behind a **FastAPI** backend with a **Streamlit** UI.
Answers use only retrieved eBook text; otherwise the bot returns exactly
`I cannot answer based on the provided document.`

## Architecture
```
Streamlit (streamlit_app.py) ──HTTP──> FastAPI (app.py) ──> LangGraph (src/graph.py) ──> Pinecone + OpenAI
src/ingestion.py: PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter(1000/200) -> embeddings -> upsert
```
Graph: `START → retrieve → evaluate_context → {generate | refuse} → END`.

## Prerequisites
Python 3.10+, an OpenAI API key, a Pinecone API key (serverless).

## Install
```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env  -> then fill in the keys
```

## Configure
Edit `.env`: `OPENAI_API_KEY`, `PINECONE_API_KEY`, optionally cloud/region/index name. Never commit `.env`.

## Get the PDF and ingest
`python -m src.ingestion` first tries to download the public Google Drive file. If Drive is unreachable or the file is
private, download it manually and save it as **`data/Ebook-Agentic-AI.pdf`**, then rerun.
```bash
python -m src.ingestion            # idempotent: unchanged chunks are skipped (deterministic ids)
python -m src.ingestion --reset    # wipe the index first (use after the PDF changes)
```
The index (1536 dims, cosine, serverless) is created automatically and awaited until ready.

## Run
```bash
uvicorn app:app --reload           # backend on :8000
streamlit run streamlit_app.py     # frontend, separate terminal
```

## API
```bash
curl -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{"query":"What is Agentic AI?"}'
curl localhost:8000/health
```
Response shape (values illustrative, not real retrieval output):
```json
{"final_answer":"Document-grounded answer (page 1).",
 "retrieved_context":[{"content":"Relevant document text","page":1,"relevance_score":0.89,
                       "source":"Ebook-Agentic-AI.pdf","chunk_id":"..."}],
 "confidence_score":0.89,
 "citations":[{"source":"Ebook-Agentic-AI.pdf","page":1}]}
```
Errors: 422 invalid/empty query, 503 missing config or Pinecone unavailable, 502 LLM unavailable, 500 other. Bodies never contain keys or stack traces.

## Relevance scoring
* **Vector similarity**: Pinecone cosine score per chunk.
* **Chunk relevance**: that score clamped to [0, 1].
* **Overall score (`confidence_score`)**: mean relevance of the chunks at or above `RELEVANCE_THRESHOLD`; `0.0` on refusal.
* **Sufficiency check**: at least one chunk passes the threshold **and** ≥ `MIN_TERM_COVERAGE` (default 0.3) of the question's
  content words occur in those chunks. This catches off-topic questions that still score moderately.
* The score measures retrieval relevance. It is **not** a calibrated probability that the answer is correct.

### Calibrating the threshold
The default (0.30) is an uncalibrated fallback. After ingesting:
```bash
python tests_sample_queries.py --calibrate   # prints top scores for in/out-of-scope questions and a suggested value
```
Set `RELEVANCE_THRESHOLD` in `.env` and restart the API.

## Tests
```bash
pytest                                     # all external services mocked; no keys needed
python tests_sample_queries.py             # LIVE: benchmark via the graph (needs keys + ingestion)
python tests_sample_queries.py --api       # LIVE: via the running backend
```
Q1–Q5 are reported as "answered / review": whether the eBook covers them must be confirmed against the PDF. Q6 (FIFA) must refuse.

## Troubleshooting
* `503 not configured`: missing `OPENAI_API_KEY`/`PINECONE_API_KEY`; fix `.env`, restart uvicorn.
* `Index ... dimension`: the index was created with another embedding size; delete it or change `PINECONE_INDEX_NAME`.
* Everything refused: not ingested yet, or threshold too high; run `--calibrate`.
* Streamlit "Cannot reach the backend": start uvicorn, check `API_BASE_URL`.
* `PDF not found`/download fails: save the file manually to `data/Ebook-Agentic-AI.pdf`.
* Dependency conflicts: use a fresh virtualenv and the pinned ranges in `requirements.txt`.

## Known limitations / ideas
* Citations list pages of all above-threshold chunks given to the model, not only those it used.
* Re-ingesting a *changed* PDF leaves stale vectors unless you pass `--reset`.
* Scanned (image-only) PDFs need OCR; not included.
* Ideas: reranker, hybrid search, LLM-based sufficiency judge, streaming, per-claim citation checking.

# Agentic AI RAG Chatbot (LangGraph + Pinecone)

A Retrieval-Augmented Generation chatbot that answers questions **only** from the
*Agentic AI eBook*. It chunks and embeds the PDF into Pinecone, runs a LangGraph workflow
(retrieve, generate, grade for groundedness, retry or refuse), and serves the result through
FastAPI (plus an optional Streamlit UI).

**Live demo:** https://rag-agentic-ai-bqguqzseduxudenrwxw4gl.streamlit.app/

Every response has this shape:

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "Agentic AI refers to ... (p. 4)",
  "retrieved_context_chunks": ["Chunk 1 text from PDF...", "Chunk 2 text from PDF..."],
  "confidence_score": 0.87
}
```

## Architecture

| Component | Technology | File |
|---|---|---|
| Data ingestion | PyPDFLoader + RecursiveCharacterTextSplitter | `src/ingestion.py` |
| Embeddings and vector DB | Gemini `gemini-embedding-001` (1536-d) or OpenAI `text-embedding-3-small` + Pinecone serverless, cosine | `src/ingestion.py`, `src/models.py` |
| Orchestration | LangGraph `StateGraph` | `src/graph.py` |
| LLM | Gemini `gemini-2.5-flash-lite` (with fallbacks) or OpenAI `gpt-4o-mini`, temperature 0, structured outputs | `src/models.py` |
| API / UI | FastAPI (`POST /chat`), Streamlit | `app.py`, `streamlit_app.py` |
| Configuration | `.env` + constants | `src/config.py` |

### Model provider

The reference suggests OpenAI; the assignment allows alternatives. The default provider is
**Google Gemini** because its free tier needs no billing. Set `LLM_PROVIDER=openai` and
`OPENAI_API_KEY` in `.env` to use OpenAI instead; nothing else changes. Both providers
produce 1536-dimensional embeddings, so the Pinecone index settings are identical
(re-run ingestion with `--reset` after switching, since vectors from different models are
not comparable).

### Ingestion pipeline

1. Download `Ebook-Agentic-AI.pdf` into `data/` (it is already committed; the script re-downloads it if missing).
2. Load it page by page with `PyPDFLoader`, normalise whitespace, drop empty pages (59 text pages).
3. Split with `RecursiveCharacterTextSplitter` (`chunk_size=800`, `chunk_overlap=100`) and drop
   fragments under 150 characters (cover page, QR-code caption), giving 130 chunks.
4. Create the Pinecone index if needed (dimension 1536, cosine, serverless).
5. Embed and upsert each chunk with metadata: chunk text, `page_number`, `source`, `chunk_index`.
   IDs are a hash of the content, so re-running ingestion overwrites instead of duplicating.

### LangGraph workflow

```
 START -> retrieve --(no chunk above MIN_RELEVANCE)--------------------> refuse -> END
              |                                                          ^
              v                                                          |
           generate --(LLM says context lacks the answer)----------------+
              ^   |                                                      |
              |   v                                                      |
              +- grade --(not grounded, attempt limit reached)-----------+
    (not grounded,  |
     retry with     +--(grounded)--> finalize -> END
     feedback)
```

State (`AgentState`): `question`, `chunks`, `context`, `answer`, `answerable`,
`groundedness`, `grounded`, `feedback`, `attempts`, `score`.

- **retrieve**: top-k (k=5) cosine search in Pinecone. Chunks below `MIN_RELEVANCE` (0.60 for Gemini,
  measured: off-topic questions top out near 0.50, eBook questions near 0.77)
  are not given to the LLM, so an unrelated question never reaches generation.
- **generate**: the LLM gets only the numbered retrieved passages and must return a structured
  `{answerable, answer}`. It is told to ignore outside knowledge and cite pages.
- **grade** (hallucination check): a second LLM call acts as a fact-checker and returns
  `{groundedness 0..1, grounded, unsupported_claims}` for the answer against the same context.
- **retry loop**: if the answer is not grounded, the graph cycles back to `generate` with the
  unsupported claims as feedback (up to `MAX_GENERATION_ATTEMPTS=2`).
- **greet**: greetings such as "hi" or "thanks" get a short help message without retrieval.
- **refuse**: returns a fixed "not available in the eBook" message with confidence 0.0.
- **finalize**: computes the confidence score.

`retrieved_context_chunks` contains exactly the chunks the answer was based on (those above
`MIN_RELEVANCE`), so it is empty for greetings and refusals.

### Confidence score

```
confidence_score = 0.7 * groundedness + 0.3 * best_retrieval_similarity
```

- `groundedness` (0 to 1) is the LLM grader's estimate of how much of the answer the retrieved text supports.
- `best_retrieval_similarity` is the highest cosine similarity from Pinecone, which shows how
  closely the eBook matches the question.
- Refusals score `0.0`.

This mixes two independent signals: whether the answer is faithful to the retrieved text, and
whether the retrieved text actually covers the question.

## Setup

Requirements: Python 3.10+, a Gemini API key (free at https://aistudio.google.com/apikey) or an
OpenAI API key, and a Pinecone API key (free tier is enough).

```bash
git clone <this-repo-url> rag-agentic-ai
cd rag-agentic-ai

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # then fill in GOOGLE_API_KEY and PINECONE_API_KEY
```

## Run

**1. Ingest the eBook into Pinecone** (once):

```bash
python -m src.ingestion           # add --reset to rebuild the index from scratch
```

**2a. Start the API:**

```bash
uvicorn app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Agentic AI?"}'
```

Interactive docs: http://127.0.0.1:8000/docs

**2b. Or use the Streamlit UI:**

```bash
streamlit run streamlit_app.py
```

The UI shows the chat on the left and the confidence score, retrieved chunks and raw JSON on the right.

## Deployment (Streamlit Community Cloud)

1. Run the ingestion once locally (`python -m src.ingestion`). The vectors live in Pinecone, so the deployed app does not need to ingest again.
2. Go to https://share.streamlit.io, click **Create app**, pick this repository, branch `main`, main file `streamlit_app.py`.
3. Under **Advanced settings -> Secrets**, add:
   ```toml
   LLM_PROVIDER = "gemini"
   GOOGLE_API_KEY = "AIza..."
   PINECONE_API_KEY = "pcsk_..."
   PINECONE_INDEX_NAME = "agentic-ai-index"
   ```
   Streamlit exposes top-level secrets as environment variables, which `src/config.py` reads.
4. Deploy and share the app URL.

## Testing

**Sample queries** (the assignment's validation set, plus two out-of-scope checks):

```bash
python tests_sample_queries.py                              # runs the graph directly
python tests_sample_queries.py --api http://127.0.0.1:8000  # or against the running API
```

The full responses are written to `sample_outputs.json`.

| # | Query | Expected |
|---|---|---|
| 1 | What is the core definition of Agentic AI as outlined in the eBook? | Grounded answer |
| 2 | What are the main architectural components required to build agentic systems? | Grounded answer |
| 3 | What real-world industry use cases for Agentic AI are discussed in the eBook? | Grounded answer |
| 4 | How does Agentic AI differ from traditional generative AI chatbots according to the text? | Grounded answer |
| 5 | What key challenges or limitations of Agentic AI are mentioned in the document? | Grounded answer |
| 6 | What role does memory play in Agentic AI workflows? | Grounded answer |
| 7 | What is the capital of France? | Refusal, confidence 0.0 |
| 8 | Who won the 2022 FIFA World Cup? | Refusal, confidence 0.0 |

**Offline unit tests** (no API keys needed). They use a fake retriever and a fake LLM to check
every graph route (grounded, greeting, low-relevance refusal, LLM refusal, retry then accept,
retry then refuse), the `/chat` payload, and the PDF chunking:

```bash
pytest -q
```

## Project structure

```
rag-agentic-ai/
├── data/
│   └── Ebook-Agentic-AI.pdf     # source document
├── src/
│   ├── __init__.py
│   ├── config.py                # env vars and constants
│   ├── models.py                # Gemini / OpenAI embedding and chat model factories
│   ├── ingestion.py             # PDF loading, chunking, Pinecone index + upsert
│   └── graph.py                 # LangGraph state, nodes, routing, confidence score
├── tests/
│   └── test_graph.py            # offline tests with fake retriever / LLM
├── app.py                       # FastAPI app (POST /chat, GET /health)
├── streamlit_app.py             # optional Streamlit UI
├── tests_sample_queries.py      # benchmark queries -> sample_outputs.json
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

All settings can be overridden in `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `PINECONE_INDEX_NAME` | `agentic-ai-index` | Pinecone index name |
| `LLM_PROVIDER` | `gemini` | `gemini` or `openai` |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` / `text-embedding-3-small` | Embedding model (1536-d) |
| `LLM_MODEL` | Gemini flash-lite / flash models, or `gpt-4o-mini` | Chat model(s) for generation and grading. Comma-separated list; later models are fallbacks (the Gemini free tier has a small daily quota per model) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | Splitter settings |
| `TOP_K` | `5` | Chunks retrieved per query |
| `MIN_RELEVANCE` | `0.60` (Gemini) / `0.25` (OpenAI) | Cosine similarity cut-off for a chunk to count as relevant |
| `MAX_GENERATION_ATTEMPTS` | `2` | Generate/grade cycles before refusing |
| `PINECONE_CLOUD` / `PINECONE_REGION` | `aws` / `us-east-1` | Serverless index location |

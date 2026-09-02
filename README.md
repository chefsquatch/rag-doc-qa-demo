# RAG Document Q&A Demo

A small, deployed **retrieval-augmented generation** app: it ingests a set of
documents, embeds them into a vector database, answers questions using the
retrieved passages as context, and **shows which passages the answer came from**.

Its one differentiator: **when the answer isn't in the documents, it says so
instead of making one up.** Most cheap RAG demos will confidently answer from the
model's general knowledge when retrieval finds nothing — the exact failure mode
that makes RAG untrustworthy. This one refuses, and the refusal is proven, not
assumed.

**Live demo:** _(add your deployed URL here)_

---

## Stack, and why

| Choice | What | Why |
| --- | --- | --- |
| **ChromaDB** | Vector database | Python-native, embedded (no server to run), and the vector DB most named in the postings this demo targets — matching them word-for-word is the point. |
| **all-MiniLM-L6-v2** | Embedding model | ChromaDB's built-in default. Runs locally via ONNX, so there is **no separate embedding API key** — one less credential, one less failure mode. |
| **Claude (Messages API)** | Answer generation | Answers are written from the retrieved passages under a strict grounding instruction. |
| **FastAPI + vanilla JS** | Web app | Minimal, readable, one-process. Deploys cleanly to Render/Railway. |

## How it stays honest (the grounding rail)

Two independent guards:

1. **Retrieval gate.** A chunk is used as context only if its cosine similarity
   to the question clears `MIN_SCORE`. If nothing clears the bar, the app returns
   the refusal **without ever calling the model** — it cannot hallucinate from
   context it never received.
2. **Instruction gate.** When context *is* passed, the system prompt tells Claude
   to answer *only* from it and to say when the answer isn't there. So a passage
   that clears the similarity bar but doesn't actually contain the answer still
   yields an honest "not in the documents."

## Run it locally

```bash
python -m venv .venv
# Windows:  .\.venv\Scripts\activate    macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then edit .env and add your ANTHROPIC_API_KEY
python -m app.ingest          # chunk + embed + store docs/ (first run downloads the embedder)
uvicorn app.main:app --reload # open http://127.0.0.1:8000
```

The server also auto-ingests `docs/` on first startup if the store is empty, so
`python -m app.ingest` is optional — but running it lets you watch the chunk count.

## Environment variables

All optional except the key. See `.env.example`.

| Var | Default | Meaning |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | _(required)_ | Your Anthropic key. Never commit it. |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Model used to write answers. |
| `TOP_K` | `4` | How many chunks to retrieve per question. |
| `MIN_SCORE` | `0.30` | Similarity a chunk must clear to count as relevant. Higher = stricter refusals. |
| `CHUNK_TARGET_CHARS` | `600` | Max characters per chunk. |
| `CHUNK_OVERLAP_CHARS` | `100` | Overlap between adjacent chunks. |

### Swapping the API key

Change `ANTHROPIC_API_KEY` in `.env` (local) or in your host's dashboard
(deployed). Nothing else changes — the key is only ever read from the environment.

## Adding your own documents

1. Drop `.md`, `.txt`, or `.pdf` files into `docs/` (and remove the sample ones).
2. Re-run `python -m app.ingest` to rebuild the store.
3. Ask away. Tune `MIN_SCORE` if refusals are too strict or too loose for your corpus.

## Deploy

A `render.yaml` blueprint and a `Procfile` are included.

- **Render:** New → Blueprint → point at this repo → set `ANTHROPIC_API_KEY` in the
  dashboard. (The store and embedder rebuild automatically on first boot.)
- **Railway:** New Project → deploy from repo (it reads the `Procfile`) → add the
  `ANTHROPIC_API_KEY` variable.

## Project layout

```
app/
  config.py    # env-driven config, one place
  store.py     # ChromaDB collection (cosine space) + embedding function
  ingest.py    # load docs/ -> paragraph-aware chunks -> embed -> store
  rag.py       # retrieve (similarity gate) + grounded answer + honest refusal
  main.py      # FastAPI: GET / , GET /health , POST /ask
static/
  index.html   # minimal ask UI, shows answer + source passages
docs/          # the sample corpus (swap for your own)
scripts/       # per-brick proof scripts (smoke_brick0..5)
```

## How it was built

Brick by brick, each proven on real data before the next
(`scripts/smoke_brick*.py`): retrieval spine → ingest+chunk → retrieve+answer →
show sources → **honest empty case (forced, watched)** → error handling → UI +
deploy. "It ran" was never treated as "it works."

## License

MIT.

"""Brick 0 proof: does the retrieval spine actually work?

Embed two clearly different strings, store them, then query with something
semantically close to ONE of them. If ChromaDB returns the nearer string first,
the embed -> store -> query round-trip is real. This proves retrieval before any
UI or Claude call exists.

Run:  ./.venv/Scripts/python.exe scripts/smoke_brick0.py
"""

import chromadb
from chromadb.utils import embedding_functions

# ChromaDB's built-in default: all-MiniLM-L6-v2 via onnxruntime. Downloads the
# model once on first use, then runs fully locally -- no embedding API key.
ef = embedding_functions.DefaultEmbeddingFunction()

client = chromadb.Client()  # in-memory; nothing persisted for this smoke test
col = client.create_collection(name="smoke", embedding_function=ef)

col.add(
    ids=["a", "b"],
    documents=[
        "The espresso machine uses a 15-bar pump to extract coffee.",
        "The hiking trail climbs 2,000 feet through an alpine forest.",
    ],
)

query = "How much pressure does the coffee maker use?"
res = col.query(query_texts=[query], n_results=2)

top_id = res["ids"][0][0]
top_doc = res["documents"][0][0]
top_dist = res["distances"][0][0]

print(f"Query:   {query}")
print(f"Nearest: [{top_id}] {top_doc}")
print(f"Distance: {top_dist:.4f}")

assert top_id == "a", f"expected the espresso doc nearest, got {top_id!r}"
print("\nBRICK 0 PASS: embed -> store -> query round-trips, nearer string returned.")

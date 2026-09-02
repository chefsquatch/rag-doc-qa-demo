"""Brick 1 proof: ingestion stores RELEVANT, retrievable chunks.

Rebuild the store from docs/, then query it directly with a question that is
answerable from the corpus. If the top chunk is the right passage (from the
right source file) with a high similarity, ingestion + retrieval is real.

Run:  ./.venv/Scripts/python.exe -m scripts.smoke_brick1   (from project root)
   or ./.venv/Scripts/python.exe scripts/smoke_brick1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import ingest
from app.store import get_collection, similarity_from_distance

count = ingest()
print(f"Ingested {count} chunks.\n")

col = get_collection()
question = "How long do I have to return unopened coffee?"
res = col.query(query_texts=[question], n_results=3)

print(f"Q: {question}\n")
for doc, meta, dist in zip(
    res["documents"][0], res["metadatas"][0], res["distances"][0]
):
    sim = similarity_from_distance(dist)
    snippet = doc.replace("\n", " ")[:90]
    print(f"  sim={sim:.3f}  [{meta['source']} #{meta['chunk_index']}]  {snippet}...")

top_meta = res["metadatas"][0][0]
top_sim = similarity_from_distance(res["distances"][0][0])
assert top_meta["source"] == "meridian_returns_policy.md", (
    f"expected returns policy on top, got {top_meta['source']}"
)
assert top_sim > 0.3, f"top similarity too low: {top_sim:.3f}"
print("\nBRICK 1 PASS: ingestion stored relevant chunks; the right source ranked first.")

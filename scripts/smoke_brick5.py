"""Brick 5 proof: every failure mode surfaces a clean message, never a stack trace.

Forces each of the three failures the SPEC names and checks the error is one of
our typed RagErrors with a readable message:
  - retrieval failure (vector store / embedding)
  - API failure (Claude call rejected)
  - config failure (missing key)

Run:  ./.venv/Scripts/python.exe scripts/smoke_brick5.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import rag
from app.ingest import ingest
from app.rag import ConfigError, GenerationError, RetrievalError, answer, retrieve

ingest()
IN_CORPUS = "How long do I have to return unopened coffee?"

# --- 1. Retrieval failure ------------------------------------------------------
_orig = rag.get_collection
rag.get_collection = lambda: (_ for _ in ()).throw(RuntimeError("store offline"))
try:
    retrieve(IN_CORPUS)
    raise SystemExit("FAIL: expected RetrievalError")
except RetrievalError as e:
    print(f"[retrieval failure] clean: {e}")
finally:
    rag.get_collection = _orig

# --- 2. API failure (bad key -> auth rejected -> GenerationError) ---------------
rag.config.ANTHROPIC_API_KEY = "sk-ant-invalid-key-for-testing"
try:
    answer(IN_CORPUS)
    raise SystemExit("FAIL: expected GenerationError")
except GenerationError as e:
    print(f"[api failure]       clean: {str(e)[:80]}...")

# --- 3. Config failure (no key) ------------------------------------------------
rag.config.ANTHROPIC_API_KEY = ""
try:
    answer(IN_CORPUS)
    raise SystemExit("FAIL: expected ConfigError")
except ConfigError as e:
    print(f"[config failure]    clean: {e}")

print("\nBRICK 5 PASS: all three failure modes surface clean, typed messages.")

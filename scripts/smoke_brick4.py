"""Brick 4 proof (the differentiator): the honest empty case holds.

Ask something clearly OUTSIDE the corpus. The retrieval gate finds nothing above
MIN_SCORE, so answer() returns the refusal WITHOUT ever calling Claude -- proven
here with no API key set. Then ask an IN-corpus question and confirm retrieval
DID find context (it fails on the missing key at the generate step, which also
proves the clean ConfigError message for Brick 5).

Run:  ./.venv/Scripts/python.exe scripts/smoke_brick4.py
"""

import os
import sys
from pathlib import Path

# Guarantee no key is visible, so we prove Guard 1 needs no model call.
os.environ.pop("ANTHROPIC_API_KEY", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import ingest
from app import rag
from app.rag import answer, retrieve, ConfigError

rag.config.ANTHROPIC_API_KEY = ""  # ensure the lazy client has no key

ingest()

# --- Out of corpus: must refuse, must NOT call the model -----------------------
oo = "What is the capital of France?"
scores = [round(p.score, 3) for p in retrieve(oo)]
result = answer(oo)
print(f"OUT-OF-CORPUS Q: {oo}")
print(f"  passages above MIN_SCORE ({rag.config.MIN_SCORE}): {scores or 'none'}")
print(f"  grounded={result.grounded}  answer={result.answer!r}")
assert result.grounded is False, "out-of-corpus question must not be grounded"
assert result.answer == rag.REFUSAL, "out-of-corpus question must return the refusal"
assert result.sources == [], "a refusal must show no sources"
print("  -> refused honestly, no model call. PASS\n")

# --- In corpus: retrieval finds context; missing key surfaces cleanly ----------
inq = "How long do I have to return unopened coffee?"
found = retrieve(inq)
print(f"IN-CORPUS Q: {inq}")
print(f"  passages above MIN_SCORE: {[round(p.score, 3) for p in found]}")
assert found, "in-corpus question should retrieve context"
try:
    answer(inq)
    raise SystemExit("expected ConfigError with no key set")
except ConfigError as e:
    print(f"  clean ConfigError: {e}")
print("  -> retrieval found context; missing key surfaced cleanly. PASS\n")

print("BRICK 4 PASS: honest empty case holds without a model call; in-corpus retrieval works.")

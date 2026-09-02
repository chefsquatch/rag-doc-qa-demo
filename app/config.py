"""Central configuration, read once from the environment.

Every tunable lives here with a sensible default so the app runs out of the box
for the demo, and every value can be overridden with an env var for a real
deployment. See .env.example for the full list.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present. In production (Render/Railway) the vars are set in the
# dashboard and there is no .env file -- load_dotenv is a no-op then.
load_dotenv()

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
CHROMA_DIR = ROOT_DIR / "chroma_db"

# Vector store
COLLECTION_NAME = "documents"

# Claude
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

# Retrieval
TOP_K = int(os.getenv("TOP_K", "4"))

# The grounding threshold: cosine SIMILARITY in [0, 1]. A retrieved chunk counts
# as "relevant" only if its similarity to the question is at least this. When no
# chunk clears the bar, the app refuses to answer instead of inventing one --
# this single number is what makes the honest-empty-case fire. Tuned against the
# sample corpus; raise it to be stricter, lower it to be more permissive.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))

# Chunking (characters). Paragraph-aware; these bound each chunk's size.
CHUNK_TARGET_CHARS = int(os.getenv("CHUNK_TARGET_CHARS", "600"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "100"))

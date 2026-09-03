"""Package init — runs before any submodule (and therefore before chromadb) is
imported, whether the entrypoint is `uvicorn app.main:app` or `python -m app.ingest`.

Render free-tier fix (the cold-start 502 on /ask):
ChromaDB's default embedder downloads all-MiniLM-L6-v2 to Path.home()/.cache/chroma
(a class attribute read from $HOME at import time). On Render the default $HOME
(/opt/render) is wiped on a free-tier cold start, so the model built at deploy time
vanished and every query 502'd with "No such file ... onnx_models/.../onnx.tar.gz".

Redirect $HOME into the deploy artifact (/opt/render/project/src/.render-home) BEFORE
chromadb is imported. That path persists across cold starts exactly like chroma_db/,
so the model — downloaded there at build time — ships and is present on every boot,
with no runtime download. The override is scoped to this Python process (it never
touches Render's buildpack, which has already run), and gated to Render (via the
RENDER env var Render sets) so local development is unchanged. On Windows this would
be a no-op anyway (Path.home() there uses USERPROFILE, not HOME).
"""

import os
from pathlib import Path

if os.getenv("RENDER"):
    _home = Path(__file__).resolve().parent.parent / ".render-home"
    _home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(_home)

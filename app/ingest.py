"""Ingest: load the document set, chunk it, embed it, store it.

Chunking is paragraph-aware. Documents are split on blank lines into paragraphs,
then paragraphs are packed into chunks up to CHUNK_TARGET_CHARS with a small
overlap so a fact that straddles a boundary is not lost. Each chunk carries its
source filename and position, which is what lets the app show sources later.

Supported files in docs/: .md, .txt, .pdf.

Run as a script to rebuild the store from docs/:
    ./.venv/Scripts/python.exe -m app.ingest
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import config
from .store import reset_collection


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def load_document(path: Path) -> str:
    """Return the plain text of a supported document."""
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return read_pdf(path)
    raise ValueError(f"Unsupported file type: {path.name}")


def chunk_text(text: str) -> list[str]:
    """Pack paragraphs into overlapping chunks bounded by CHUNK_TARGET_CHARS."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= config.CHUNK_TARGET_CHARS:
            current += "\n\n" + para
        else:
            chunks.append(current)
            # Start the next chunk with a tail of the previous one for overlap.
            overlap = current[-config.CHUNK_OVERLAP_CHARS :]
            current = (overlap + "\n\n" + para).strip()

    if current:
        chunks.append(current)
    return chunks


def collect_chunks(docs_dir: Path) -> tuple[list[str], list[dict], list[str]]:
    """Walk docs_dir, chunk every supported file, return (documents, metadatas, ids)."""
    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    supported = {".md", ".txt", ".pdf"}
    files = sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in supported)
    if not files:
        raise FileNotFoundError(f"No .md/.txt/.pdf documents found in {docs_dir}")

    for path in files:
        text = load_document(path)
        for i, chunk in enumerate(chunk_text(text)):
            documents.append(chunk)
            metadatas.append({"source": path.name, "chunk_index": i})
            ids.append(f"{path.name}::chunk-{i}")

    return documents, metadatas, ids


def ingest(docs_dir: Path | None = None) -> int:
    """Rebuild the store from docs_dir. Returns the number of chunks stored."""
    docs_dir = docs_dir or config.DOCS_DIR
    documents, metadatas, ids = collect_chunks(docs_dir)

    collection = reset_collection()
    # Embedding happens inside add() via the collection's embedding function.
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return len(documents)


def main() -> int:
    count = ingest()
    print(f"Ingested {count} chunks from {config.DOCS_DIR} into '{config.COLLECTION_NAME}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

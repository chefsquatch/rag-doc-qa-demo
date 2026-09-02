"""The vector store: a thin, honest wrapper over a persistent ChromaDB collection.

Cosine space is deliberate. ChromaDB's default is squared-L2, whose distances
are unbounded and hard to threshold. With cosine space, distance is in [0, 2] and
similarity = 1 - distance is in [-1, 1] (in practice [0, 1] for these embeddings),
which gives the grounding threshold (MIN_SCORE) a clean, interpretable meaning.

The embedding model is ChromaDB's built-in default (all-MiniLM-L6-v2, ONNX). It
downloads once and runs fully locally -- no embedding API key. The same function
is attached to the collection, so ingest and query always embed identically.
"""

from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from . import config

_embedding_fn = embedding_functions.DefaultEmbeddingFunction()


def get_collection():
    """Return the persistent collection, creating it with cosine space if needed."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    """Drop and recreate the collection. Used by ingest to rebuild from scratch."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        # Nothing to delete on a first run -- not an error.
        pass
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def similarity_from_distance(distance: float) -> float:
    """Convert a cosine distance to a cosine similarity in [0, 1]."""
    return 1.0 - distance

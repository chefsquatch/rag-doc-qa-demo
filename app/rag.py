"""Retrieval-augmented answering, with grounding as a hard rail.

Two independent guards keep answers honest:

  1. The retrieval gate. A chunk counts as context only if its cosine similarity
     to the question clears MIN_SCORE. If nothing clears the bar, we DO NOT call
     the model at all -- we return the honest refusal directly. The model cannot
     hallucinate from context it never received.

  2. The instruction gate. When context IS passed, the system prompt tells Claude
     to answer ONLY from that context and to say when the answer is not in it. So
     even a chunk that clears the similarity bar but does not actually contain the
     answer still yields an honest "not in the documents."

That belt-and-suspenders design is the demo's whole differentiator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .store import get_collection, similarity_from_distance

REFUSAL = "I couldn't find that in the provided documents."

SYSTEM_PROMPT = (
    "You are a document question-answering assistant. Answer the user's question "
    "using ONLY the provided context passages. Do not use any outside or general "
    "knowledge. If the answer is not contained in the passages, reply with exactly: "
    f'"{REFUSAL}" and nothing else. When you do answer, be concise and stay '
    "faithful to the passages."
)


# --- Errors, so failures surface as clean messages, never raw stack traces ----
class RagError(Exception):
    """Base class for expected, user-facing RAG failures."""


class RetrievalError(RagError):
    """Embedding or vector-store query failed."""


class GenerationError(RagError):
    """The Claude API call failed."""


class ConfigError(RagError):
    """Required configuration (e.g. the API key) is missing."""


@dataclass
class Passage:
    text: str
    source: str
    chunk_index: int
    score: float


@dataclass
class AnswerResult:
    answer: str
    grounded: bool  # True if answered from retrieved context; False = honest refusal
    sources: list[Passage] = field(default_factory=list)


def retrieve(question: str, top_k: int | None = None, min_score: float | None = None) -> list[Passage]:
    """Return the relevant passages for a question, filtered by the similarity gate."""
    top_k = top_k or config.TOP_K
    min_score = config.MIN_SCORE if min_score is None else min_score

    try:
        collection = get_collection()
        res = collection.query(query_texts=[question], n_results=top_k)
    except Exception as exc:  # embedding or store failure
        raise RetrievalError(f"Could not search the documents: {exc}") from exc

    passages: list[Passage] = []
    # A brand-new / empty store returns empty lists rather than raising.
    docs = res.get("documents") or [[]]
    metas = res.get("metadatas") or [[]]
    dists = res.get("distances") or [[]]
    for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
        score = similarity_from_distance(dist)
        if score >= min_score:
            passages.append(
                Passage(
                    text=doc,
                    source=str(meta.get("source", "unknown")),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    score=score,
                )
            )
    return passages


def _format_context(passages: list[Passage]) -> str:
    blocks = []
    for i, p in enumerate(passages, 1):
        blocks.append(f"[Passage {i} | source: {p.source}]\n{p.text}")
    return "\n\n".join(blocks)


def _generate(question: str, passages: list[Passage]) -> str:
    if not config.ANTHROPIC_API_KEY:
        raise ConfigError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        context = _format_context(passages)
        message = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Context passages:\n\n{context}\n\nQuestion: {question}",
                }
            ],
        )
        return "".join(block.text for block in message.content if block.type == "text").strip()
    except ConfigError:
        raise
    except Exception as exc:
        raise GenerationError(f"The answer service failed: {exc}") from exc


def answer(question: str) -> AnswerResult:
    """Answer a question, grounded in retrieved passages or honestly refusing."""
    question = (question or "").strip()
    if not question:
        raise RagError("Please enter a question.")

    passages = retrieve(question)

    # Guard 1: nothing relevant retrieved -> refuse without ever calling the model.
    if not passages:
        return AnswerResult(answer=REFUSAL, grounded=False, sources=[])

    text = _generate(question, passages)

    # Guard 2: the model itself judged the answer absent from the passages.
    grounded = text.strip().rstrip(".") != REFUSAL.rstrip(".")
    return AnswerResult(
        answer=text,
        grounded=grounded,
        sources=passages if grounded else [],
    )

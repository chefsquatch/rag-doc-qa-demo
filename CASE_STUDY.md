# Case Study — A RAG Pipeline That Refuses to Bluff

*A short write-up of what this demo is, the one decision that makes it different,
and how each claim in it was proven rather than asserted.*

---

## The problem this exists to show

Retrieval-augmented generation is easy to demo and hard to trust. The easy version
works like this: embed some documents, retrieve the nearest chunks to a question,
hand them to a language model, print the answer. It looks correct because the model
is fluent, and it stays looking correct right up until you ask it something the
documents do not contain — at which point most cheap RAG demos answer anyway, from
the model's general knowledge, in the same confident voice they use for grounded
answers.

That single behavior is the failure mode that makes RAG untrustworthy in production.
A knowledge base that will confidently answer a question its sources never covered is
worse than no knowledge base, because it launders a guess as a citation. The value of
this demo is that it does not do that.

## The one differentiator

**When retrieval finds nothing relevant, the app says so — "I couldn't find that in
the provided documents" — instead of inventing an answer.** It is built as a
first-class requirement, not a fallback, and it is enforced twice:

1. **The retrieval gate.** A chunk counts as context only if its cosine similarity to
   the question clears a threshold (`MIN_SCORE`, default 0.30). If nothing clears the
   bar, the app returns the refusal **without ever calling the language model.** The
   model cannot hallucinate from context it never received. This is the strong guard:
   it is not a prompt asking the model to behave, it is the absence of an input.

2. **The instruction gate.** When context *is* passed, the system prompt constrains
   the model to answer only from that context and to say when the answer is not in it.
   So a chunk that clears the similarity bar but does not actually contain the answer
   still yields an honest "not in the documents."

Two independent mechanisms, one of which does not depend on the model's cooperation at
all. That is the difference between hoping the model is honest and building a system
that is honest whether the model cooperates or not.

## Why cosine space, and why a threshold at all

The vector store (ChromaDB) defaults to squared-L2 distance, whose magnitudes are
unbounded and awkward to threshold — a strong match and a weak one can both be "some
largish number." The store here is built in **cosine space** on purpose, so distance
lands in [0, 2] and `similarity = 1 − distance` is an interpretable [0, 1] score. That
interpretability is what lets the grounding threshold mean something a human can
reason about and tune, rather than a magic constant.

The threshold was not guessed. On the sample corpus, an in-corpus question retrieves
its correct passage at similarity ~0.70, unrelated passages from the same corpus sit
around 0.45–0.53, and a clearly out-of-corpus question ("what is the capital of
France?" against a coffee-roaster knowledge base) retrieves **nothing** above 0.30.
The floor sits in the empty band between "relevant" and "unrelated," which is exactly
where a grounding threshold should sit.

## How each claim here was proven

The build followed one rule: *"it works" means you watched it work on real data, not
that it compiled.* Every capability was forced to demonstrate itself before being
called done — including the failures, which were provoked on purpose rather than
assumed to be handled.

| Claim | How it was proven |
|---|---|
| The retrieval spine works | Embedded two unrelated strings, queried with a paraphrase of one, confirmed the nearer one comes back. |
| Ingestion stores *relevant* chunks | Queried the store directly after ingest; the correct source passage ranks first at similarity 0.70. |
| Grounded answers cite their source | A real question returns a correct answer *with the source passages and their similarity scores shown.* |
| The honest empty case holds | Asked an out-of-corpus question and watched the app refuse **with no model call at all** — the retrieval gate fired. |
| Errors surface cleanly | Forced retrieval failure, API failure, and a missing key; each returns a clean typed message, never a stack trace or a silent hang. |

The out-of-corpus refusal is the one that matters, and it was not verified by
inference. It was forced: ask the thing the documents cannot answer, and watch the
system decline to answer it, on the live deployment, not just in a unit test.

## Honest limitations

- **It is a proof, not a product.** No auth, no multi-user, a handful of documents. It
  demonstrates a correct retrieve-then-answer-or-refuse pipeline someone can read and
  extend; it is not a production knowledge base.
- **The threshold is corpus-relative.** `MIN_SCORE = 0.30` is tuned to the sample
  corpus and the `all-MiniLM-L6-v2` embedder. A different corpus or embedder needs it
  re-checked; the honest empty case is only as good as that number, and the README
  says so where someone swapping the documents will read it.
- **Grounded ≠ correct.** The two guards ensure the answer comes *from the retrieved
  passages*. If the passages themselves are wrong, the grounded answer will be wrong —
  faithfully. Grounding is a guarantee about provenance, not about truth.

## The principle underneath

The interesting part is not the retrieval; it is the refusal. A system that answers
only from what it retrieved, and says so plainly when it retrieved nothing, is a
system that has been made **checkable** rather than merely careful — the honesty is a
property of how it is wired, not of how well the model behaves on the day. That is the
same discipline, in miniature, that makes any automated system trustworthy: attach the
constraint to the mechanism, so the failure mode cannot dress itself up as a success.

Most RAG demos are careful. This one is checkable. When it doesn't know, it tells you —
and it tells you because it cannot do otherwise.

---

*Live demo: https://rag-doc-qa-demo.onrender.com · Source and setup: see
[README.md](README.md).*

# Personal-RA

I read a lot of ML safety papers and kept losing track of what was in them. So I built the
thing I actually wanted: ask a question, get an answer, and see **exactly which page it came
from** — with every quote checked back against the PDF rather than taken on trust.

It runs on my laptop, over my own 34 papers. Nothing is uploaded anywhere.

```
"which papers study sandbagging?"
        │
        ▼
  router  ──►  one paper?  ──►  whole paper in context, no chunking
          ──►  the library? ──►  hybrid retrieval → grade → rewrite if thin
          ──►  the web?     ──►  stops and asks me first
        │
        ▼
  every quote string-matched back to the source → page number, or flagged unverified
```

## The one design decision everything follows from

A single paper is ~15–30k tokens. It **fits in the context window**, so for one-paper
questions I send the whole paper — no chunking, no retrieval. Chunking a paper to answer
"does the ablation support the main claim?" gives you four fragments of an argument instead
of the argument.

RAG is reserved for cross-paper questions, where nothing fits.

## Stack

| Layer | What I used | Why |
|---|---|---|
| **LLM** | Anthropic API — `claude-sonnet-4-5` for answers, `claude-haiku-4-5` for routing/grading, `claude-sonnet-5` as a refusal fallback | Haiku for the many cheap calls, Sonnet where quality shows |
| **Prompt caching** | Anthropic prompt caching | 7–13× cheaper on repeat questions about the same paper |
| **Vision** | Claude vision on rendered page crops | Recovers display maths and figures that text extraction destroys |
| **Orchestration** | **LangGraph** + `SqliteSaver` checkpointer | Routing, a retry loop, and a human-approval gate that survives a process restart |
| **PDF parsing** | **PyMuPDF** | Column-aware reading order; PyPDF mangles two-column layouts |
| **Vector store** | **ChromaDB** (local, on disk) | No server to run |
| **Embeddings** | **sentence-transformers** `all-MiniLM-L6-v2` | Local and free — the library re-indexes with no API spend |
| **Retrieval** | **rank-bm25** + dense, fused with Reciprocal Rank Fusion | BM25 turned out to be the strongest single signal on this corpus |
| **Reranking** | cross-encoder `ms-marco-MiniLM-L-6-v2` | Measured, then made opt-in — it won precision, not recall |
| **Citation check** | **RapidFuzz** | Exact match first, then fuzzy at 95% for ligature/hyphenation damage |
| **UI** | **Streamlit** + `streamlit-pdf-viewer` | PDF pane, click a citation and it highlights on the page |
| **API** | **FastAPI** + SSE, `uvicorn` | Streams *node transitions*, not just tokens |
| **Agent access** | **MCP** server (stdio) | The library becomes queryable from inside Claude Code |
| **Web search** | **Tavily** | Behind an approval gate — it never spends without asking |
| **Tracing** | **Langfuse** | Per-node latency, tokens, cost |
| **Automation** | **n8n** | Daily arXiv job: filter → score → ingest |
| **Testing** | **pytest** (404 tests, no network) + **ruff** | The Anthropic client is mocked everywhere |

## What it does

**Verified citations.** The model must quote verbatim; every quote is normalised and
string-matched back to the parsed PDF to recover its page. If it doesn't match, it's shown
as *unverified* instead of dressed up as a citation.

> **Q:** Which papers study reward hacking, and what did they measure?
>
> **A:** …*School of Reward Hacks* — "a dataset containing over a thousand examples of
> reward hacking on short, low-stakes, self-contained tasks" **[p. 1]** … 3/3 expected
> papers retrieved, 3 exact-match verified quotes, $0.01.

**It reads figures.** Caption-anchored detection crops each figure and asks Claude vision to
describe it. This is the part I'm most pleased with, because it turned an answer that was
*confidently wrong* into a correct one:

| | Answer to *"which attacker model's submitted backdoors are most often correct?"* |
|---|---|
| Before | **GPT-4o Mini, 0.73** — invented, with a real quote attached |
| After indexing figures | *"That isn't covered in my library"* — honest, still wrong-ish |
| After fixing retrieval | **o3-mini, 93.2%** ✅ |

The number was always in the extracted text — as an orphaned run of digits,
`93.2 | 77.0 88.7 | 47.1`, dropped into unrelated prose. What the figure supplies is *which
model each number belongs to*.

**Four ways in:** a Streamlit UI, a CLI, a FastAPI server, and an MCP server so Claude Code
can query the library directly.

## Numbers

| | |
|---|---|
| Library | 34 papers → 4,871 chunks (67 of them figure descriptions) |
| Retrieval (best config) | recall@5 **0.899**, MRR 0.898 — `section_context` + hybrid, on a 63-question golden set |
| Route accuracy | 98.4% (124/126) — though the golden set contains no `web` or `direct` questions, so that covers two routes of four |
| Cost | $0.032 median per library question, $0.201 per whole-paper question (6× — it's a cache write), $0.006 per figure |
| Tests | 404, none touching the network |

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env          # add your Anthropic key
```

Drop PDFs into `papers/`, then:

```bash
python -m personal_ra.library --rebuild        # index (offline, free)
streamlit run src/personal_ra/app.py           # the UI
python -m personal_ra.graph.run "which papers study sandbagging?"
uvicorn personal_ra.api:app --reload           # HTTP + SSE
```

## What I'd do differently

The honest list. There's a much longer one in the
[engineering notes](docs/engineering-notes.md).

- **Vision output can't be verified the way prose can.** A described figure is model
  inference, not extracted text. It's marked as such everywhere — `content_type: "figure"`
  on chunks, a `source_type` on citations — because I don't want it read as a quotation. It
  still gets details wrong: one axis labelled `ℙ[Red team wins]` came back as "Pixel
  entropy", and telling the model to say "unclear" instead of guessing did not make it do so.
- **Paper-level recall@5 counts a bibliography hit as success.** A paper scores 1.0 because
  *some* chunk came back — including its reference list. This actively misled me once.
- **A metric I trusted moved on its own.** Re-running the eval returned 0.702 where it had
  said 0.711, on byte-identical chunks. Cause: rebuilding the Chroma index perturbs the deep
  tail of approximate search, and reranking draws candidates from depth 15. The conclusion
  survived; the precision of the number didn't.
- **~12.5% of whole-paper questions get refused by safety classifiers** — they're papers
  about jailbreaks. This silently returned blank answers for two versions before an eval
  caught it.
- **Figures are indexed for 3 papers, not 34.** The rest is a ~$3 vision bill I haven't
  decided to spend.
- **The arXiv job has never run for real.** It's validated against live arXiv data and ships
  in dry-run mode, but "no duplicates over a week" is a claim I can't make yet.
- **`/ingest` has no auth and will fetch a URL you give it.** Fine bound to localhost, which
  is the only way it's meant to run.

---

Built one version at a time against a written spec: v0 single-paper Q&A → v1 cross-paper
retrieval + evaluation → v2 MCP → v3 the router → v4 figures and automation. The
[engineering notes](docs/engineering-notes.md) have the full measurements, including the
experiments that didn't work.

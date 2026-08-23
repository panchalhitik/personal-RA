# Personal-RA

Grounded Q&A over my own research-paper library, where every quote is checked back against
the PDF and shown with the page it came from.

![Personal-RA UI: cited passages highlighted in the PDF, with the router's reasoning beside it](docs/images/ui.png)

## Why I use it

I read a lot of ML safety papers and kept losing track of what was in them. Attaching a PDF
to a chat window works fine for one paper, but I wanted it over the whole shelf, without
uploading anything and without having to trust a summary I could not check.

So the point of this is not "chat with your PDFs". It is that the model has to quote
verbatim, and every quote gets string-matched back to the parsed source to recover its page.
If a quote does not match, it shows as unverified rather than being dressed up as a citation.
When you can see the sentence highlighted on the page, you stop having to take the answer on
faith.

It also decides for itself where to look. Ask about the paper you have open and it reads the
whole paper. Ask something that spans the shelf and it searches the library, grades what came
back, and retries with a better query if the results are thin. Ask about something newer than
the library and it stops and asks before spending anything on a web search.

## Design

One decision shapes everything else. A single paper is about 15 to 30k tokens, so it fits in
the context window. For one-paper questions I send the whole paper, with no chunking and no
retrieval. Chunking a paper to answer "does the ablation support the main claim?" gives you
four fragments of an argument instead of the argument.

Retrieval is reserved for cross-paper questions, where nothing fits.

```
question
   │
   ▼
 router ──► one paper?   ──► whole paper in context, prompt-cached
        ──► the library? ──► hybrid retrieval → grade chunks → rewrite if thin
        ──► the web?     ──► pauses and asks first
   │
   ▼
 answer ──► every quote matched back to source → page number, or flagged unverified
```

The graph checkpoints to SQLite, so a run paused at the approval gate survives a restart.

## Stack

| Layer | Choice | Why |
|---|---|---|
| **LLM** | Anthropic API: `claude-sonnet-4-5` for answers, `claude-haiku-4-5` for routing and grading | Haiku for the many cheap calls, Sonnet where quality shows |
| **Prompt caching** | Anthropic prompt caching | 7 to 13 times cheaper on repeat questions about one paper |
| **Vision** | Claude vision on rendered page crops | Recovers maths and figures that text extraction destroys |
| **Orchestration** | **LangGraph** with a `SqliteSaver` checkpointer | Routing, a retry loop, and an approval gate that survives a restart |
| **PDF parsing** | **PyMuPDF** | Column-aware reading order. PyPDF mangles two-column layouts |
| **Vector store** | **ChromaDB**, local and on disk | No server to run |
| **Embeddings** | **sentence-transformers** `all-MiniLM-L6-v2` | Local and free, so re-indexing costs nothing |
| **Retrieval** | **rank-bm25** plus dense, fused with Reciprocal Rank Fusion | BM25 turned out to be the strongest single signal here |
| **Reranking** | cross-encoder `ms-marco-MiniLM-L-6-v2` | Measured, then made opt-in. It won precision, not recall |
| **Citation check** | **RapidFuzz** | Exact match first, then fuzzy at 95% for ligature damage |
| **UI** | **Streamlit** with `streamlit-pdf-viewer` | Click a citation and it highlights on the page |
| **API** | **FastAPI** with SSE | Streams node transitions, not just tokens |
| **Agent access** | **MCP** server over stdio | Makes the library queryable from inside Claude Code |
| **Web search** | **Tavily**, behind an approval gate | It never spends without asking |
| **Automation** | **n8n** | Daily arXiv job: filter, score, ingest |
| **Testing** | **pytest** (404 tests, no network) and **ruff** | The Anthropic client is mocked everywhere |

## Running it

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env          # add your Anthropic key
```

Drop PDFs into `papers/`, build the index, and pick a surface:

```bash
python -m personal_ra.library --rebuild        # index locally, no API spend
streamlit run src/personal_ra/app.py           # the UI in the screenshot
python -m personal_ra.graph.run "which papers study sandbagging?"
uvicorn personal_ra.api:app --reload           # HTTP API with SSE
```

### From inside Claude Code

`.mcp.json` is committed, so restarting Claude Code in this folder exposes the library as
tools: `search_library`, `read_paper`, `verify_quote`, `list_papers`, and `ask_router`. You
can then ask about your papers in the middle of doing something else, and Claude cites pages
back at you.

The path in `.mcp.json` is absolute, so edit it if you clone this somewhere else.

Two things cost money and are off by default: figure descriptions (about $0.006 each, cached
afterwards) and web search (needs a Tavily key). Everything else, including the whole
evaluation harness, runs locally for free.

## Numbers

| | |
|---|---|
| Library | 34 papers, 4,871 chunks, 67 of them figure descriptions |
| Retrieval | recall@5 of **0.899**, MRR 0.898, on a hand-written 63-question golden set |
| Route accuracy | 98.4% (124 of 126), though the golden set has no `web` or `direct` questions, so that covers two routes of four |
| Cost | $0.032 per library question, $0.201 per whole-paper one, which is 6 times more because it is a cache write |
| Tests | 404, none touching the network |

Reading figures is the piece I am most pleased with. Asked which attacker model's submitted
backdoors are most often correct, the system used to answer "GPT-4o Mini, 0.73", which was
invented, with a real quote attached. The value only exists inside a bar chart. After
indexing figure descriptions and fixing three retrieval bugs, it answers **o3-mini, 93.2%**,
correctly.

## Future work

- **Figures for the other 31 papers.** Three are done. The rest is a $3 vision bill I have
  not decided to spend.
- **Let the arXiv job run for a week.** It is validated against live arXiv data and ships in
  dry-run mode, but "no duplicates over a week" is a claim I cannot make yet.
- **Verify quotes against the enriched paper.** Right now a figure-derived quote can never
  match, so it is flagged unverified. That is the right outcome by accident, and I would
  rather it were deliberate.
- **Auth on `/ingest` before it goes anywhere.** It has none and will fetch a URL you hand
  it, which is fine bound to localhost and nowhere else.
- **Zotero sync**, so papers arrive without me dragging files around.

The [engineering notes](docs/engineering-notes.md) have the full measurements, the failed
experiments, and a much longer list of things I would do differently.

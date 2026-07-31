# Personal-RA — Build Spec

Grounded Q&A over a personal research-paper library.

This file is the contract between me (Hitik) and Claude Code. Read it fully before writing any code.

---

## 0. Working agreement — READ THIS FIRST

You are building this project **one version at a time**. Do not skip ahead.

**Rules:**

1. **Never build more than one numbered step at a time.** After each step, stop and show me what changed.
2. **Stop at every `🛑 CHECKPOINT`.** These are things only I can do (API keys, choosing papers, judging output quality, installing external software). Print exactly what you need from me and wait.
3. **Every step ends with tests.** Write the tests, run them, show me the output. If they fail, fix them before moving on. Never report a step complete with failing tests.
4. **Ask before deciding.** If the spec is ambiguous or you think a different approach is better, say so and wait — don't silently substitute.
5. **Explain as you go.** After each step, give me 3–5 lines on what this code does and why. I'm building this to learn the stack, not just to have it exist.
6. **Small commits.** One commit per step, conventional-commit style (`feat:`, `test:`, `fix:`, `docs:`). Tell me the message you used.
7. **No premature abstraction.** Write the obvious version first. No base classes, no plugin systems, no config frameworks until something actually needs them.
8. **Do not touch later-version code.** If v0 needs something v2 will also need, build the v0 version of it and move on.

**At the end of every version:** run the full test suite, update the README, and print a short summary of what now works, what the acceptance criteria showed, and what the next version adds.

---

## 1. What we're building

I read a lot of ML/NLP research papers. I want a tool that answers questions about them and shows me exactly where in the paper the answer came from — the experience of attaching a PDF to Claude, but over my own library and without attaching anything.

**The core design insight, do not violate it:**

> A single paper is ~15k tokens and fits in the context window. For one-paper questions we send the **whole paper** — no chunking, no retrieval. RAG is only for **cross-paper** questions over the library, where nothing fits.

Chunking a single paper would make answers worse, because questions like "does the ablation support the main claim?" need the whole argument, not four fragments.

**The differentiator** is verified citations: the model must quote verbatim, and we string-match every quote back to the source text to recover its page. If a quote doesn't match, the model invented it and we flag it instead of displaying it.

### Versions

| Version | What it does |
|---|---|
| **v0** | Ask questions about one paper, get answers with verified quotes and page numbers |
| **v1** | Ask questions across the whole library, find which papers say what |
| **v2** | Do both from inside Claude Code via an MCP server |
| **v3** | A router decides: read one paper, search the library, or search the web |
| **v4** | Reads figures via vision; new arXiv papers ingest automatically |

Build v0 → v1 → v2. Then I reassess before v3.

---

## 2. Fixed technical decisions

Do not deviate from these without asking.

- **Python 3.11+**, virtualenv at `.venv`, dependencies in `pyproject.toml` (use `uv` if available, else `pip`)
- **LLM:** Anthropic API via the `anthropic` SDK. Model `claude-sonnet-4-5`, `temperature=0`
- **Prompt caching is mandatory in v0** — see §4
- **PDF parsing:** PyMuPDF (`fitz`). Not PyPDF — it mangles two-column academic layouts
- **Embeddings (v1+):** `sentence-transformers` with `all-MiniLM-L6-v2`, local and free. Keep it behind one function so I can swap to OpenAI later
- **Vector store (v1+):** Chroma, persisted to disk
- **Tests:** `pytest`. Unit tests never call the network — mock the Anthropic client
- **Secrets:** `.env` via `python-dotenv`, and `.env` in `.gitignore` from commit one
- **Formatting:** `ruff` for lint and format
- **Type hints on every function.** Docstrings only where the intent isn't obvious from the name

### Repo layout (target, built up gradually)

```
personal-ra/
├── .env.example
├── pyproject.toml
├── README.md
├── PERSONAL-RA.md         # this file
├── papers/                # my PDFs (gitignored)
├── src/personal_ra/
│   ├── parse.py           # v0  PDF → structured text
│   ├── cite.py            # v0  quote verification
│   ├── ask.py             # v0  single-paper Q&A
│   ├── app.py             # v0  Streamlit UI
│   ├── library.py         # v1  ingest, chunk, embed
│   ├── search.py          # v1  cross-paper retrieval
│   ├── eval.py            # v1  retrieval metrics
│   └── mcp_server.py      # v2
├── tests/
│   ├── fixtures/          # tiny synthetic PDFs, committed
│   └── test_*.py
└── eval/
    ├── golden_set.yaml    # v1
    └── results/           # v1  metrics output
```

---

## 3. v0 — One paper, full context, verified citations

**Goal:** I pick a paper, ask a question, and get an answer where every claim carries a verified page citation.

### Step 0.1 — Scaffold

Create the repo structure, `pyproject.toml`, `.gitignore` (include `.env`, `papers/`, `.venv/`, `chroma_db/`, `__pycache__/`), `.env.example` with `ANTHROPIC_API_KEY=`, and a stub README. Set up pytest so `pytest` runs green on zero tests.

> 🛑 **CHECKPOINT 0.1** — Tell me the exact commands to create my venv, install deps, and where to put my Anthropic API key. Then stop. I'll confirm `pytest` runs and the key is set before you continue.

### Step 0.2 — PDF parsing (`parse.py`)

Parse a PDF into a structured object:

```python
@dataclass
class Page:
    number: int          # 1-indexed, matches what I see in a PDF reader
    text: str            # normalized, layout-corrected

@dataclass
class Paper:
    path: Path
    title: str           # best-effort from page 1
    pages: list[Page]
    full_text: str       # pages joined with explicit page markers
    n_tokens: int        # rough estimate, chars / 4
```

Requirements:

- Use PyMuPDF's block-based extraction with sorted reading order so **two-column layouts don't interleave**. This is the single most important thing in v0 — if it's wrong, everything downstream is garbage.
- Normalize: collapse runs of whitespace, join words hyphenated across line breaks, strip repeated headers/footers that appear on most pages.
- `full_text` must embed page markers like `\n\n[PAGE 7]\n\n` so the model can see page boundaries.
- Add a `--debug` CLI flag that dumps parsed text per page to `debug/` so I can eyeball it.

Tests:
- Generate a tiny **two-column synthetic PDF** in `tests/fixtures/` (build it with PyMuPDF in a script, commit the PDF) and assert the extracted text reads down each column, not across.
- Assert page numbers are 1-indexed.
- Assert hyphen-joining and whitespace normalization on a fixture.
- Assert repeated headers are stripped.

> 🛑 **CHECKPOINT 0.2** — I'll drop 5 real papers into `papers/`. Run the debug dump on all 5 and show me a chunk of text from the middle of each. **I must confirm the text isn't scrambled before you build anything else.** If a paper comes out badly, we fix the parser now, not later.

### Step 0.3 — Quote verification (`cite.py`)

The heart of the project.

```python
@dataclass
class Citation:
    quote: str
    page: int | None
    char_offset: int | None
    verified: bool
    match_type: Literal["exact", "fuzzy", "failed"]
```

`verify_quote(quote: str, paper: Paper) -> Citation`:

1. Normalize both quote and source identically (collapse whitespace, unify quote marks/dashes, casefold for matching only).
2. Try exact substring match on the normalized text → `verified=True`, `match_type="exact"`.
3. Fall back to fuzzy match with `rapidfuzz.fuzz.partial_ratio` over a sliding window. **Threshold 95.** Above it → `match_type="fuzzy"`, still verified. Below → `match_type="failed"`, `verified=False`.
4. Map the match position back to a page number and character offset in that page.

Tests (this module gets the most tests in v0):
- Exact match found, correct page returned
- Match works despite differing whitespace and line breaks
- Match works despite curly vs straight quotes and en/em dashes
- A hallucinated quote returns `verified=False`
- A quote spanning a page boundary resolves to the page where it starts
- Fuzzy match just above and just below threshold behaves correctly
- Empty quote and whitespace-only quote don't crash

### Step 0.4 — Single-paper Q&A (`ask.py`)

`ask(paper: Paper, question: str, history: list[Message]) -> Answer`

The Anthropic call:

- System prompt sets the role: careful research assistant, answers only from the provided paper.
- The full paper text goes in the **system prompt** with `cache_control: {"type": "ephemeral"}` on that block. This is the prompt caching requirement — first question pays full price, every later question about the same paper is much cheaper and faster. Log cache-read vs cache-write tokens so I can see it working.
- Instruct the model: support every substantive claim with a short verbatim quote from the paper, wrapped in `<quote>` tags. Quote exactly, never paraphrase inside the tags.
- Explicit refusal string when the paper doesn't cover something: `"That isn't covered in this paper."` Do not answer from general knowledge.

Post-processing: extract every `<quote>`, run it through `verify_quote`, and return:

```python
@dataclass
class Answer:
    text: str                  # quotes replaced with inline [p. N] markers
    citations: list[Citation]
    unverified: list[Citation] # surfaced separately as a warning
    usage: dict                # tokens in/out, cache read/write, cost estimate
```

Tests (mock the Anthropic client, no network):
- Quotes are correctly extracted from a mocked response
- Verified quotes become `[p. N]` markers in the output text
- An unverified quote lands in `unverified` and is **not** silently rendered as a citation
- `cache_control` is present on the paper block in the request payload — assert on the actual kwargs passed to the client
- A response with no quotes doesn't crash

Also add a CLI: `python -m personal_ra.ask papers/foo.pdf "What dataset did they use?"`

> 🛑 **CHECKPOINT 0.4** — Run 5 real questions against one of my papers, including one whose answer definitely isn't in it. Show me the answers, the citations, the unverified list, and the token/cost log for each. **I'll judge whether the answers are actually good.** Also confirm the second question is cheaper than the first — that proves caching works.

### Step 0.5 — Streamlit UI (`app.py`)

Deliberately plain. Left sidebar: paper picker from `papers/`, plus token count and running session cost. Main pane: chat history, answer text with `[p. 7]` markers, and an expander per citation showing the quote and its page. Unverified quotes render in a warning box, clearly marked as unverified — never hidden.

Keep the parsed `Paper` in `st.session_state` so switching questions doesn't re-parse.

Tests: one smoke test that imports the module and calls the pure helper functions. Don't try to test Streamlit rendering.

> 🛑 **CHECKPOINT 0.5** — Give me the run command. I'll use it for 20 minutes on real papers and come back with what's broken or annoying. Fix that list before we call v0 done.

### v0 acceptance criteria

- [ ] Two-column PDFs parse in correct reading order (verified by me on 5 real papers)
- [ ] ≥90% of quotes across ~20 test questions verify as exact or fuzzy
- [ ] A question outside the paper returns the refusal string, not an invention
- [ ] Second question on the same paper shows cache-read tokens and lower cost
- [ ] `pytest` fully green
- [ ] README has a screenshot and a real answer example

---

## 4. v1 — The library

**Goal:** ask questions across all my papers and find which ones say what. This is where RAG actually belongs.

### Step 1.1 — Ingest and chunk (`library.py`)

Chunk every paper with **section-aware** splitting, not blind character splits:

- Detect section headers with a regex over typical academic patterns (`3. Method`, `IV. Experiments`, `Related Work`) — best-effort, don't over-engineer it.
- Never split inside a code block, table region, or mid-sentence.
- Prepend context to each chunk before embedding: `"From '{paper_title}', section '{section}': {chunk_text}"`. This is a cheap, large quality win.
- Target ~1000 characters, 200 overlap, as the fallback within sections.

Metadata per chunk — this is what makes v1 useful:

```python
{
  "paper_id": str,      # stable hash of file content
  "paper_title": str,
  "page": int,
  "section": str | None,
  "chunk_index": int,
  "year": int | None,   # best-effort from text
}
```

Ingest must be **idempotent** — deterministic chunk IDs (`f"{paper_id}:{chunk_index}"`), upsert not append. Re-running must not duplicate. Add `--rebuild` to wipe and reindex.

Tests: same paper ingested twice yields identical chunk count; metadata is complete on every chunk; section detection works on a fixture; chunk IDs are stable across runs.

### Step 1.2 — Cross-paper search (`search.py`)

- Hybrid retrieval: dense (Chroma) + BM25 (`rank_bm25`), fused with Reciprocal Rank Fusion.
- Metadata filters: by paper, by year range.
- Return chunks with scores attached — I want to see them.
- `answer_across_library(question, k=8)`: retrieve, group chunks by paper, and generate an answer that cites **paper title + page**. Same verbatim-quote discipline as v0, and quotes verify against the parsed source of their own paper.

Tests: RRF fusion ranks correctly on a synthetic case; metadata filters actually filter; retrieval returns chunks from multiple papers when the question spans them.

> 🛑 **CHECKPOINT 1.2** — I'll add 20–30 papers. Run ingest and show me: chunk count, chunks per paper, and 5 random chunks printed in full so I can check the splitting isn't cutting mid-argument.

### Step 1.3 — Eval harness (`eval.py` + `eval/golden_set.yaml`)

**This is the part that makes the project portfolio-grade. Do not skip it.**

Golden set schema:

```yaml
- id: q01
  question: "Which of my papers use contrastive loss?"
  ground_truth: "..."
  expected_paper_ids: ["a1b2c3", "d4e5f6"]
  category: cross_paper   # factual | cross_paper | comparison | unanswerable
```

You write the harness and generate ~15 **draft** questions from the corpus for me to correct. I write the rest by hand.

> 🛑 **CHECKPOINT 1.3** — This is my job, not yours. I need to write ~50 question/answer pairs against my own papers, including 5 deliberately unanswerable ones. Budget me ~6 hours. Give me the template and a script that shows me candidate papers to draw from, then wait. **Do not tune retrieval before the golden set exists** — otherwise it just mirrors whatever the retriever already does.

Metrics:
- Cheap and offline: `recall@k`, `MRR` from `expected_paper_ids`. Must run in seconds with no API calls, so I can iterate.
- Paid: RAGAS faithfulness and answer relevancy, only on the best configs. Gate it behind `--full`.
- Refusal rate on the unanswerable set — a wrong answer here is worse than no answer.

Run the config matrix: 3 chunking strategies (fixed / section-aware / section-aware + context prefix) × 3 retrieval modes (dense / BM25 / hybrid+RRF). Write results to `eval/results/{timestamp}.json` and render a **markdown table** for the README.

Tests: metric functions verified against hand-computed cases (a known ranking with a known recall@5 and MRR); the harness runs end-to-end on a 3-paper fixture library.

### v1 acceptance criteria

- [ ] 20+ papers ingested, re-running ingest changes nothing
- [ ] Golden set of ~50 questions exists and is committed
- [ ] Full 9-config matrix evaluated, results table in the README
- [ ] Best config beats the fixed-chunking dense baseline on recall@5 — and if it doesn't, the README says so honestly
- [ ] Refusal rate on unanswerable questions is reported
- [ ] `pytest` green

---

## 5. v2 — MCP server

**Goal:** I open Claude Code, ask about my papers, and it queries my library. No attaching anything.

### Step 2.1 — Server (`mcp_server.py`)

Use the official MCP Python SDK, stdio transport.

Tools:
- `search_library(query, k=8, paper_id=None, year_min=None, year_max=None)` → chunks with paper title, page, score
- `read_paper(paper_id, section=None)` → full text or one section
- `list_papers()` → id, title, year, page count
- `verify_quote(quote, paper_id)` → the v0 citation checker, exposed directly

Resources:
- `library://index` — the paper list
- `eval://latest` — the current metrics table

Tool descriptions matter more than the code here. Write them so a model knows exactly when to reach for each one — especially when to use `search_library` versus `read_paper`.

Tests: each tool called directly with valid and invalid inputs; a bad `paper_id` returns a clean error, not a stack trace; schemas validate.

> 🛑 **CHECKPOINT 2.1** — Give me the exact config block and file path to register this server with Claude Code, plus the command to verify it's connected. I'll wire it up and report back. Then we test it live together.

### Step 2.2 — Live testing

Once connected, I'll run 10 real questions through Claude Code. Log every tool call server-side so we can see which tools it picked and whether the descriptions steered it correctly. Iterate on the descriptions based on what I report.

### v2 acceptance criteria

- [ ] Server connects to Claude Code and all four tools are callable
- [ ] I successfully answer a real research question through Claude Code with no PDF attached
- [ ] Tool call logs show sensible tool selection
- [ ] README has a GIF or screenshot of it working inside Claude Code
- [ ] `pytest` green

---

## 6. v3 and v4 — hold

Do not start these. I'll reassess after v2 and update this file.

- **v3:** LangGraph router (single paper / library / web), a chunk grader that rejects weak retrievals and re-queries, grounding check before returning, Langfuse tracing with p50/p95 latency and cost per query.
- **v4:** figure extraction via Claude vision with captions indexed alongside text; n8n cron watching arXiv categories and auto-ingesting; optional Zotero sync.

---

## 7. Testing policy

- `pytest` after every step. A step is not complete until it's green.
- **Unit tests never hit the network.** Mock `anthropic.Anthropic`. If a test needs the real API, mark it `@pytest.mark.live` and exclude it by default.
- Fixtures live in `tests/fixtures/` and are committed — synthetic PDFs generated by a script, never my real papers.
- Every bug I report becomes a regression test before it gets fixed.
- Track coverage on `parse.py`, `cite.py`, and `eval.py`. Aim high on those three; don't chase coverage on UI code.

---

## 8. README — build it as you go, not at the end

Order matters. Reviewers read the first screen and decide.

1. One-line description and the architecture diagram
2. **A real before/after example** — a question, the answer, the verified citations
3. The eval results table (from v1)
4. Screenshot or GIF of the Claude Code integration (from v2)
5. `docker compose up` or venv quickstart — must work on a clean machine
6. **"What I'd do differently"** — where parsing still fails, what the eval doesn't measure, when I'd move off Chroma

Update it at the end of every version, not once at the end of the project.

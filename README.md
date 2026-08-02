# Personal-RA

Grounded Q&A over a personal research-paper library — the experience of attaching a PDF
to Claude, but over your own papers, with **verified citations**: the model must quote
verbatim, and every quote is string-matched back to the source text to recover its page.
A quote that doesn't match gets flagged as unverified instead of displayed as a citation.

```
papers/*.pdf
    │  parse.py    PyMuPDF, column-aware reading order, [PAGE N] markers
    ▼
Paper ──► vision.py    equation-heavy pages → page image → Claude vision → LaTeX
    │                  (appended to page text, disk-cached per paper)
    ▼
app.py    Streamlit: PDF pane (select/copy) + assistant panel + notes
    │  question
    ▼
ask.py    whole paper in a cached system prompt (claude-sonnet-4-5, temp 0)
    │  answer with verbatim <quote> tags
    ▼
cite.py   normalize + exact/fuzzy match → page + offset, or flagged unverified
    │
locate.py quote → rectangles on the PDF page → citation highlights in the viewer
```

The core design decision: a single paper (~15–30k tokens) fits in the context window, so
one-paper questions send the **whole paper** — no chunking, no retrieval. RAG is reserved
for cross-paper questions (v1).

**Status: v3 complete** — v0's single-paper Q&A (verified citations, vision-transcribed
math, PDF-reader UI), v1's evaluated cross-paper retrieval, v2's MCP server, and a
LangGraph router that decides *for itself* whether a question wants one paper, the whole
library, or the web — then grades what it retrieved, retries when grading comes up short,
and audits the answer before showing it to you.

## The router (v3)

Before v3 I had to decide, per question, whether to run `ask` (one paper) or `search`
(the library). v3 makes that decision itself and shows its reasoning.

```
                        route  (haiku, forced tool use — 4-way + a reason)
                          │
      ┌───────────────────┼────────────────┬──────────────┐
      ▼                   ▼                ▼              ▼
 single_paper         retrieve          approve        direct
 (whole paper,     (hybrid; split      (HALTS here,   (no retrieval)
  cached prompt)    per side for        persists,          │
      │             comparisons)        waits for you)     │
      │                   ▼                  │             │
      │                rerank  (opt-in)      ▼             │
      │                   ▼              web_search        │
      │                 grade   (haiku, all chunks         │
      │                   │      concurrently)             │
      │            ┌──────┴──────┐          │              │
      │            ▼             ▼          │              │
      │        rewrite ──►   generate ◄─────┘              │
      │      (max 2, loops        │                        │
      │       to retrieve)        ▼                        │
      └──────────────────►   grounding ◄──┐                │
                                 │   └────┘                │
                                 │  (one stricter retry    │
                                 ▼   when ungrounded)      │
                                END ◄──────────────────────┘
```

### A full node trace

One real question, every decision the graph made:

```
Q: How do STAR-1 and RealSafe-R1 differ in aligning reasoning models?

route      → library — "compares two named systems, so it spans papers rather than
                        sitting inside the one that happens to be open"
retrieve   → 2 searches, not 1:
               "STAR-1 aligning reasoning models"
               "RealSafe-R1 aligning reasoning models"
             interleaved → 8 chunks, 4 from each paper
grade      → 5 kept (4 of them partial), 3 rejected
               rejected: bibliography entry; no findings
               rejected: describes a different alignment method entirely
generate   → answer with 4 verified quotes, both papers cited
grounding  → grounded — every claim traced to a retrieved excerpt
```

The two searches are the interesting line. A single search for that question returns
eight chunks from *one* of the two papers, and no amount of grading or rewriting can
build a comparison out of one side.

### Route accuracy

63 questions × two states (a paper open, no paper open), labelled by hand:
**98.4%** (124/126) on the router alone; **100%** end-to-end over a 60-run sample.

| expected \ predicted | single_paper | library | web | direct | total |
|---|---|---|---|---|---|
| single_paper | **32** | 0 | 0 | 0 | 32 |
| library | 0 | **92** | 2 | 0 | 94 |
| web | 0 | 0 | 0 | 0 | 0 |
| direct | 0 | 0 | 0 | 0 | 0 |

**Read the two empty rows before the headline number.** The golden set was built to
evaluate retrieval, so it contains no question that should reach `web` or `direct`. That
98.4% says the router separates single-paper from library almost perfectly, and says
*nothing* about the two routes where a mistake is expensive. The labels also began as
router output that I corrected, so this is not an independent test — it measures whether
the router reproduces decisions I agreed with.

Both misses are the same failure: a no-paper question naming a specific artefact
(STAR-1, RealSafe-R1) that the router assumed was too recent to be in the library, so it
reached for the web. That is the expensive direction to be wrong in.

### Reranking: measured, then made opt-in

The v1 README named `max_per_paper` as "a heuristic, not a reranker". So I built the
reranker and measured it — 3 chunking strategies × 4 retrieval modes:

| Chunking | Retrieval | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| **section_context** | hybrid | 0.655 | 0.852 | **0.899** | 0.899 | 0.898 |
| section | rerank | 0.693 | 0.838 | 0.895 | 0.895 | 0.930 |
| section | dense | 0.614 | 0.814 | 0.882 | 0.890 | 0.870 |
| fixed | hybrid | 0.681 | 0.841 | 0.879 | 0.879 | 0.925 |
| section_context | rerank | 0.693 | 0.844 | 0.873 | 0.873 | 0.930 |
| section | hybrid | 0.652 | 0.847 | 0.870 | 0.870 | 0.901 |
| section_context | dense | 0.678 | 0.838 | 0.864 | 0.864 | 0.907 |
| section | bm25 | 0.661 | 0.800 | 0.861 | 0.864 | 0.903 |
| section_context | bm25 | 0.661 | 0.800 | 0.861 | 0.864 | 0.903 |
| fixed | rerank | **0.711** | 0.849 | 0.858 | 0.861 | **0.946** |
| fixed | bm25 | 0.705 | 0.813 | 0.844 | 0.844 | 0.938 |
| fixed | dense | 0.588 | 0.779 | 0.842 | 0.853 | 0.849 |

**Reranking improves precision at the top and does not improve recall.** recall@1 rises
in all three chunking strategies (+0.029 to +0.041 against hybrid) and MRR with it, but
recall@5 *falls* in two of three — the cross-encoder is confident enough about the wrong
chunks to push a correct paper out of the deeper ranks. `section_context + hybrid` keeps
the headline 0.899.

The spec asked specifically whether reranking beats `fixed + bm25` on recall@1 (0.705).
It gets 0.711. **I don't think that counts as beating it** — on 57 answerable questions
that gap is a third of one question. The real, reproducible finding is the +0.03–0.04
against hybrid *within* each chunking strategy, which shows up three times.

A depth sweep then found recall@1 **identical** at depths 15, 20 and 30 — the
cross-encoder's top pick is already inside hybrid's top 15, so a deeper pool gives it
nothing to promote — while latency scales linearly (1.1s / 1.5s / 2.5s). So: depth 15,
and **opt-in rather than default**, because it buys precision the generator barely
notices and costs a second of wall clock.

### Operational numbers

From a 60-run end-to-end sweep (30 questions × both paper states, $3.16):

| route | p50 | p95 | median cost |
|---|---|---|---|
| `single_paper` | 16.1s | 31.3s | **$0.2019** |
| `library` | 18.0s | 26.2s | $0.0256 |

Whole-paper questions cost **8× a library question** — 15–60k tokens of paper, mostly as
a cache *write* since no paper repeats within a sweep. That is a real argument against
the router's "prefer single_paper when torn" tiebreak, which v0 chose on quality grounds
before anyone had measured the bill.

| grounding verdict | share |
|---|---|
| grounded | 65.0% |
| partially_grounded | 31.7% |
| ungrounded | 0.0% |
| api_refused | 3.3% |

**Zero `ungrounded` in 60 runs.** The one-stricter-regeneration path has never fired
outside its tests. Sixty runs cannot distinguish "the threshold is too strict" from
"generation genuinely doesn't invent when it has excerpts in front of it".

### v3 closes four gaps v2 named

The v2 README's "what I'd do differently" list called four things out. All four are now
addressed — and one of the fixes produced a result I didn't expect:

| v2 weakness | v3 fix | outcome |
|---|---|---|
| Refusal detection is string-prefix matching | Grounding-verdict refusal scoring | **No difference** — see below |
| `max_per_paper` is a heuristic, not a reranker | Cross-encoder reranker | Built, measured, made opt-in |
| The MCP path has no automated eval | `eval --routes` harness | 98.4%, with the caveats above |
| No latency or cost numbers anywhere | Per-node cost + a Langfuse tracer | Numbers above |

**The refusal comparison is a null result, and that's worth saying plainly.** v3 replaces
prefix matching with "did the answer invent anything", which is the better-justified
metric. Measured on the unanswerable set twice, on independent samples:

| method | correct |
|---|---|
| prefix matching (v2) | 12/12 |
| grounding verdict (v3) | 12/12 |

They agree completely, because the model uses the exact refusal string every time. v2's
5/6 with a hedged miss did not recur. The new method is more defensible; it is **not**
numerically better, and reporting it as an improvement would be dressing up a tie.

## A real example

> **Q:** What is the main claim of this paper?
> *(against "Why LLM Safety Guardrails Collapse After Fine-tuning", 18 pages)*
>
> **A:** The main claim is that **the similarity between upstream safety-alignment
> datasets and downstream fine-tuning datasets is a critical factor in determining the
> durability of LLM safety guardrails.** …
>
> **Verified citations:**
> - [p. 1] (exact) *"Our experiments demonstrate that high similarity between these
>   datasets significantly weakens safety guardrails, making models more susceptible to
>   jailbreaks. Conversely, low similarity between these two types of datasets yields
>   substantially more robust models and thus reduces harmfulness score by up to 10.33%."*
> - [p. 2] (exact) *"Collectively, our results indicate that scholars' and practitioners'
>   narrow focus on downstream fine-tuning processes has led them to overlook critically
>   important upstream alignment effects."*

When the model paraphrases inside quote tags, verification catches it — the quote lands
in a clearly-marked warning box instead of being rendered as a citation. A question the
paper doesn't cover gets the fixed refusal: *"That isn't covered in this paper."*

Prompt caching makes follow-up questions ~7–13× cheaper: the first question writes the
paper into the cache (~$0.08 for an 18-page paper), later questions read it (~$0.01).

<!-- TODO: screenshot of the UI — docs/screenshot.png -->

## Quickstart

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env    # put your Anthropic API key in .env
pytest                    # 363 tests, no network needed
```

Drop PDFs into `papers/`, then:

```
streamlit run src/personal_ra/app.py         # the UI
python -m personal_ra.library --rebuild      # ingest the library (offline, no API)
python -m personal_ra.graph.run "which papers study sandbagging?"        # the v3 router
python -m personal_ra.ask papers\foo.pdf "What dataset did they use?"    # one paper
python -m personal_ra.search "which papers use contrastive loss?"        # cross-paper
python -m personal_ra.eval --matrix          # the 12-config retrieval matrix (offline)
python -m personal_ra.eval --routes          # route accuracy (~$0.22, calls Haiku)
python -m personal_ra.parse papers\foo.pdf --debug          # inspect parsing
python -m personal_ra.vision papers\foo.pdf --detect-only   # equation-page scores
uvicorn personal_ra.api:app --reload         # HTTP API, SSE node stream
```

For Claude Code, add the `.mcp.json` shown below and restart — then just ask questions.

**Two things are optional and off by default.** Web search needs `TAVILY_API_KEY` (free
tier, 1,000 credits/month); without it the router simply never offers that route. Tracing
needs `docker compose up -d` and Langfuse keys in `.env`; without them the tracer is a
null object and never loads the SDK, so cloning the repo does not require Docker.

First load of an equation-heavy paper runs a one-time vision pass (a few cents,
cached in `.cache/vision/`).

## The library (v1): cross-paper search, evaluated

34 papers → 4,807 section-aware chunks in Chroma (idempotent ingest — re-running
changes nothing). Retrieval is hybrid: dense (MiniLM embeddings, local and free) and
BM25 rankings fused with Reciprocal Rank Fusion. `answer_across_library` keeps v0's
discipline — every quote verifies against the parsed source of its own paper:

> **Q:** Which of my papers study reward hacking?
>
> **A:** All three papers … *School of Reward Hacks* ("a dataset containing over a
> thousand examples of reward hacking on short, low-stakes, self-contained tasks…"
> [p. 1]), *Inference-Time Reward Hacking* (BoN hacking on GPQA [p. 8]), and *Natural
> Hacking in Production RL* [p. 6] — 3/3 expected papers retrieved, 3 exact-match
> verified quotes, $0.01.

### Evaluation

Hand-written golden set: **63 questions** (32 factual, 16 cross-paper, 9 comparison,
6 unanswerable) covering all 34 papers, written *before* any retrieval tuning. Metrics
are paper-level recall@k and MRR, computed offline in seconds. Full 3×3 config matrix
(`python -m personal_ra.eval --matrix`):

| Chunking | Retrieval | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| **section_context** | hybrid | 0.655 | 0.852 | 0.899 | 0.899 | 0.898 |
| section | dense | 0.614 | 0.814 | 0.882 | 0.890 | 0.870 |
| fixed | hybrid | 0.681 | 0.841 | 0.879 | 0.879 | 0.925 |
| section | hybrid | 0.652 | 0.847 | 0.870 | 0.870 | 0.901 |
| section_context | dense | 0.678 | 0.838 | 0.864 | 0.864 | 0.907 |
| section | bm25 | 0.661 | 0.800 | 0.861 | 0.864 | 0.903 |
| section_context | bm25 | 0.661 | 0.800 | 0.861 | 0.864 | 0.903 |
| fixed | bm25 | 0.705 | 0.813 | 0.844 | 0.844 | 0.938 |
| fixed | dense | 0.588 | 0.779 | 0.842 | 0.853 | 0.849 |

Honest reading of the numbers:

- **The best config (section_context + hybrid, 0.899 recall@5) beats the fixed+dense
  baseline (0.842) by ~5.7 points** — chunking quality and fusion both earn their keep.
- **BM25 is embarrassingly strong on this corpus** — fixed+bm25 has the best MRR (0.938)
  and recall@1 (0.705). Research questions use distinctive vocabulary ("Best-of-Poisson",
  "sandbagging"), which is exactly BM25's home turf. Anyone shipping dense-only
  retrieval for a corpus like this is leaving accuracy on the table.
- **The context prefix is a wash for recall, a gain for fusion** — it helps hybrid
  (0.870 → 0.899) but not dense recall@5 on its own. Smaller effect than advertised.
- **section == section_context for BM25, exactly** — expected, since BM25 scores raw
  chunk text and the strategies differ only in what gets embedded. A nice built-in
  sanity check that the harness measures what it claims.
- **Refusal rate on unanswerable questions: 5/6 (83%).** The one miss hedged ("only
  found one tangential mention…") rather than fabricating an answer.

RAGAS faithfulness/relevancy (paid, LLM-judged) is stubbed behind `--full` but not yet
wired up — it needs a judge-model dependency decision.

## Inside Claude Code (v2): MCP server

`mcp_server.py` exposes the library over MCP (stdio), so Claude Code can query it
directly — no PDF attached, no copy-pasting.

```json
// .mcp.json in the project root
{
  "mcpServers": {
    "personal-ra": {
      "command": ".venv/Scripts/python.exe",
      "args": ["-m", "personal_ra.mcp_server"]
    }
  }
}
```

| Tool | What it does |
|---|---|
| `search_library` | Hybrid retrieval across all papers → excerpts with title, page, section, score |
| `read_paper` | One paper in full, or a single section, with page markers |
| `list_papers` | Inventory: id, title, year, pages, chunks |
| `verify_quote` | v0's citation checker, exposed directly |

Resources: `library://index` (the paper list) and `eval://latest` (the metrics table above,
served live from the newest results file).

<!-- TODO: screenshot/GIF of Claude Code answering a library question — docs/mcp.png -->

**Tool descriptions do the real work here.** Each leads with USE THIS WHEN, and
`search_library` carries an explicit DO NOT USE THIS WHEN pointing at `read_paper` —
because v0's core insight (a whole paper fits in context; fragments are for *locating*,
not reasoning) only holds if the model knows when to switch tools. Errors are written for
the model too: a bad `paper_id` returns *"Call list_papers to get valid ids (there are
34…)"*, not a stack trace.

### What live testing changed

Running real questions through Claude Code found two things unit tests could not:

1. **Excerpt ranking crowded out papers.** "Which of my papers study emergent
   misalignment?" returned 10 excerpts from just **3** papers — the golden set says 6 —
   because ranking is per excerpt, so one paper repeating the query's wording occupied
   every slot. Adding `max_per_paper` (cap excerpts per paper, fetch deeper to
   compensate) fixed it, measured against golden-set q33:

   | Config | Distinct papers | Expected papers found |
   |---|---|---|
   | `k=10`, uncapped | 3 | 3/6 |
   | `k=10, max_per_paper=2` | 6 | 5/6 |
   | `k=12, max_per_paper=1` | 8 | **6/6** |

   It defaults off, so the v1 eval numbers above remain valid — this is a serving-layer
   choice, not a retrieval change.

2. **A tool description was lying.** `read_paper` claimed that omitting `section` would
   error with a list of section names; it actually returns the *entire* paper — 48 pages
   for one of them. Now it warns about that, and every response carries
   `available_sections`.

The eval loop also ran backwards once, in a good way: answering a comparison question
surfaced two details missing from the golden set's own reference answer (RealSafe-R1's
format-mismatch mechanism and its over-refusal limitation), which got folded back in.

## What works in v0

- Two-column academic PDFs parse in correct reading order (block-based extraction with
  column-aware sorting); repeated headers/footers, page numbers, and rotated arXiv
  watermarks are stripped; hyphenated line-breaks are rejoined.
- Display equations — which no text extractor can linearize — are transcribed to LaTeX
  via Claude vision, spliced into the page text, and cached per paper.
- Every substantive claim carries a verbatim quote, verified by exact-then-fuzzy matching
  (rapidfuzz, threshold 95) against normalized source text with a character-level map
  back to page and offset. Unverified quotes are surfaced, never silently rendered.
- The UI shows the PDF (selectable text) beside a pop-in/out assistant; after each answer
  the viewer jumps to the first cited page and draws temporary highlight boxes over the
  cited passages. Per-paper markdown notes with a txt download.

## Testing

`pytest` — 363 tests, all green. Unit tests never call the network; the Anthropic client
is mocked, and PDF fixtures are tiny synthetic files generated by
`tests/fixtures/generate_fixtures.py`. Live checks (real questions against real papers)
were run manually at each checkpoint.

Measured on scripted checkpoint runs: 17 of 20 quotes verified as exact or fuzzy (85%);
the 3 failures were genuine paraphrases that verification correctly flagged — which is
the feature working, but the ≥90% acceptance bar is met only if flagged paraphrases are
counted as correct behavior rather than misses.

## What I'd do differently

- **Column detection is a heuristic** (left/right of the page midline). It survived all
  five test papers, but full-width figure captions inside column regions could still
  interleave; a layout model would be the robust fix.
- **Vision transcriptions can't be verified** the way prose quotes can — the LaTeX is
  model-generated, so a quote from it verifies against our own transcription, not the
  original PDF. Trust chain is one link longer there.
- **Equation-page detection favors recall**: table-heavy pages get flagged and cost a
  wasted "NONE" vision call each. Cached, so it only stings once per paper.
- **Highlight placement is best-effort** — quotes that hyphenation or ligatures mangled
  fall back to prefix search, and some can't be located at all (the citation still shows
  in the panel; only the box is missing).
- **Enrichment invalidates the prompt cache once per paper** (the paper text changes), an
  unavoidable cost of splicing transcriptions in.
- **No in-PDF annotation authoring** — Streamlit components can only display highlights,
  not create them from mouse selection. The notes panel is the substitute; a Zotero sync
  (v4 candidate) would be the real answer.
- **Section labels overreach (v1)** — the section tracker holds the last-seen header, so
  appendix content after an unrecognized header inherits the previous label (e.g. chunks
  tagged "References" containing appendix text). Metadata-only; chunk text is unaffected.
- **Years from PDF metadata are compile dates** — 6 non-arXiv papers fall back to the
  PDF creation date, which can lag the true publication year. Fine for coarse filtering.
- **Refusal detection is string-prefix matching** — a hedged non-answer that doesn't use
  the refusal string counts as a non-refusal (as happened once in 6; see the eval).
- **The MCP path has no automated eval (v2)** — tool-selection quality was judged by
  running real questions by hand. Measuring "did it pick the right tool" would need a
  trace-scoring harness, which is the natural next eval investment.
- **`max_per_paper` is a heuristic, not a reranker** — capping excerpts per paper buys
  breadth cheaply, but a cross-encoder reranker would do better on both axes at once.
  *(v3: built and measured. It wins precision@1, not recall@5, so it ships opt-in.)*

### New in v3

- **Two comparison questions still answer with no citations.** q48 and q49 in the golden
  set. Both are retrieval problems wearing grader clothing: q48 retrieves bibliography
  pages and NeurIPS checklists, which the grader is *right* to reject; q49 asks about two
  labs' approaches and even per-entity retrieval finds only one side. Per-entity search
  fixed three of five such questions; these two need something else.
- **Paper-level recall@5 counts a bibliography hit as success.** A paper scores 1.0
  because *some* chunk of it was retrieved — including its reference list. This actively
  misled me: I diagnosed a grader bug from a "perfect retrieval, everything rejected"
  reading that turned out to be junk chunks the grader correctly threw away. Any recall
  number for this corpus deserves that asterisk.
- **The rewrite loop's headline metric is confounded.** Comparing recall on questions
  where the loop fired against questions where it didn't measures *question difficulty* —
  the loop fires because retrieval was bad. Measured within-question instead (first pass
  vs final pass), the loop is roughly neutral: 4 improved, 5 unchanged, 4 worsened, and 2
  rescued from zero recall. Both numbers are reported; only one means anything.
- **~12.5% of whole-paper questions are refused by Anthropic's safety classifiers.** Six
  of 32, on papers about jailbreaks and adversarial attacks — `stop_reason: "refusal"`,
  empty content, HTTP 200. This was silently returning **blank answers since v0** until
  the v3 eval surfaced it. Now reported with its category, and retried once on a fallback
  model, which rescues two of the six. Four remain unanswerable in whole-paper mode; the
  library route answers them fine, because eight excerpts don't trip what a full attack
  paper does.
- **Route accuracy is measured on a distribution that omits two of the four routes**, and
  against labels derived from the router's own output. Both caveats are in the routing
  section; neither is fixable without writing questions specifically for `web` and
  `direct`.
- **Grading costs a model call per chunk.** Eight concurrent Haiku calls per retrieval
  pass, times up to three passes when the rewrite loop runs. It is the reason a library
  answer takes ~18s rather than ~5s.
- **The Langfuse traces have never been looked at.** The tracer, its tests and a
  `docker-compose.yml` are committed and the code path is exercised, but Docker Desktop
  on my machine has been unable to start since an auto-update, so no human has yet
  confirmed a span landing in a dashboard. The operational numbers above come from the
  eval harness, not from traces.
- **`/ingest` reindexes the whole library** to add one paper, because `library.ingest`
  scans a directory rather than accepting a file. Fine for occasional additions; the
  obvious thing for v4's daily arXiv job to make incremental.

## Roadmap

| Version | Status |
|---|---|
| v0 | ✅ Single-paper Q&A with verified citations, vision math, PDF-reader UI |
| v1 | ✅ Cross-paper search (hybrid + RRF), 63-question golden set, eval matrix |
| v2 | ✅ MCP server — the library queryable from inside Claude Code |
| v3 | ✅ LangGraph router, cross-encoder reranking, chunk grading + rewrite loop, grounding checks, HITL gate before web search, route eval, per-node tracing, FastAPI |
| v4 | Next: figure understanding via vision, arXiv auto-ingest with n8n, optional Zotero sync |

See [PERSONAL-RA.md](PERSONAL-RA.md) for the full build spec. (Spec drift note: vision
math transcription was pulled forward from v4 into v0 by agreement, and the v0 UI grew
beyond "deliberately plain" — PDF viewer, highlights, notes.)

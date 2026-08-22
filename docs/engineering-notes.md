# Personal-RA — engineering notes

The long version: what was measured, what broke, and what the numbers actually
support. [README.md](../README.md) is the short one.

These are working notes kept as the project was built, version by version. Where a
number here disagrees with something I believed earlier, the disagreement is left in
rather than tidied away — that is most of the value.

Grounded Q&A over a personal research-paper library — the experience of attaching a PDF
to Claude, but over your own papers, with **verified citations**: the model must quote
verbatim, and every quote is string-matched back to the source text to recover its page.
A quote that doesn't match gets flagged as unverified instead of displayed as a citation.

```
papers/*.pdf
    │  parse.py    PyMuPDF, column-aware reading order, [PAGE N] markers
    ▼
Paper ──► vision.py    equation-heavy pages → page image → Claude vision → LaTeX
    │                  figures → caption-anchored crop → structured description
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

**Status: v4 in progress** — v0's single-paper Q&A (verified citations, vision-transcribed
math, PDF-reader UI), v1's evaluated cross-paper retrieval, v2's MCP server, v3's
LangGraph router that decides *for itself* whether a question wants one paper, the whole
library, or the web — then grades what it retrieved, retries when grading comes up short,
and audits the answer before showing it to you — and v4's figure understanding and daily
arXiv ingest.

## Figures and auto-ingest (v4)

### Reading figures

`vision.py` already turned equation-heavy pages into LaTeX. v4 extends that module —
same disk cache, same splice convention — to figures:

```
page ──► caption regex ("Figure 3:") ──► cluster the page's graphics ──► crop
     ──► Claude vision, given the caption AND the sentence that cites it
     ──► "[FIGURE 3: <description>]" appended to the page text
```

Academic plots are vector, not raster, so PyMuPDF *image blocks* find almost nothing
here; detection anchors on captions and clusters `get_drawings()` output by vertical
gaps. Across the 34-paper library it finds **508 figures**.

A described figure is not the same kind of evidence as a quote, and is marked as such
everywhere it can be. Chunks carry `content_type: "figure"` and never mix vision output
with prose — a content-type change breaks a chunk the way a section change does.
`Citation.source_type` reports whether a verified quote landed in the paper's own text,
a transcribed equation, or a described figure, and the MCP `verify_quote` tool returns
it with an instruction not to present the last one as a quotation.

**What it cost, and what it got wrong.** 54 figures across three figure-heavy papers:

| | |
|---|---|
| Cost | $0.35 for the pass; $0.72 including a re-run after fixes |
| Predicted vs actual | $0.333 predicted, $0.347 actual |
| Structure — figure type, axes, series, legend labels | correct on every one I checked |
| Wrong crops | 3 of 54 before the fix, 0 after |
| Values judged by eye rather than read off the figure | 22 of 54, then 8 of 54 |

The part worth keeping is *how* the three wrong ones failed: **the descriptions were
faithful to the wrong picture.** Three crops paired a caption with a neighbouring
figure's artwork, and the model described exactly what it had been shown — fluently,
confidently, and about a different figure. Nothing in the text hinted at it. It surfaced
only by rendering the crops as PNGs and looking at them.

Two bugs, both mine. Figures stacked closer than the cluster gap merged into a single
blob that every caption on the page then claimed. And clustering *per caption* let a
caption match the near half of a neighbour's artwork, manufacturing a phantom cluster
2pt away that beat the real figure 10pt above. Captions are now barriers that split a
cluster, clustering runs once per page before any caption is considered, and artwork
above a caption beats artwork below it. Fixing that then dropped three real figures
whose bounding box laps a few points over their own caption, which is a second lesson:
the regression was invisible in the output too, and showed up as a count going 26 → 24.

**The limit that stays.** One Control Tax figure has `ℙ[Red team wins]` on its y-axis.
The first pass called it *"Pixel entropy"*. After the prompt was tightened — including
an explicit instruction to say "unclear" rather than guess — it called it *"Pixel team
value"*. Closer, still wrong, still not flagged. Of the handful I checked against the
source image, two were wrong in a detail — that axis label, and a figure whose series
colours were swapped. I have not sampled systematically enough to put a rate on it;
what I can say is that it happens, and that the model gives no signal when it does.
Forbidding eyeballed numbers worked better but not completely: naming the hedges
("approximately", "around", "~") as the signal to omit a number cut those from 22 of
54 to 8 of 54. That is why figure content is labelled rather than trusted.

**Indexing figures was necessary and not sufficient.** A question whose answer sat
verbatim in a figure chunk — *which attacker model has the highest probability that a
submitted backdoor is actually correct?* — went from a confident invention (*GPT-4o Mini,
0.73*) before indexing, to a correct refusal after it, and only reached the right answer
(*o3-mini, 93.2%*) after three retrieval fixes. The number itself was already in the
extracted text, as an orphaned run of digits (`93.2 | 77.0 88.7 | 47.1`) dropped into
unrelated prose; what the figure supplies is which model each number belongs to.

What had to change, in order of how much it moved the target chunk's rank:

| Fix | Effect |
|---|---|
| A plot axis tick is not a section header | removes `section 'REFERENCES'` from its embedding |
| Embed figure chunks on the paper's caption, not the generated description | dense rank 37 -> 10 |
| Put the caption in the chunk text too, so BM25 sees it | BM25 58 -> 2, fused 14 -> 4 |

The last one is the structural point: dense retrieval scores `embed_text` and BM25 scores
the chunk text, so a caption present in only one of them left the two halves of hybrid
retrieval ranking the same chunk on different content, and RRF split the difference.

The first one turned out not to be a figure problem at all. A y-axis tick followed by a
legend entry — `1.0 Poison every 5 steps` — satisfies the numbered-section pattern, so
every chunk after a figure inherited a plot label as its section. It predates figures
entirely: **54 of 131 chunks (41%) in one paper**, with no vision involved. Fixing it
removed 18 bogus labels across the library and 14 chunks — and moved paper-level recall
by exactly nothing, on all nine non-rerank configs. The metric asks whether *some* chunk
of the right paper came back, and relabelling chunks almost never changes which paper
wins. The fix is still right; this harness simply cannot see it.

### The daily arXiv job

`automation/arxiv_ingest.json` is an n8n workflow, committed as the artifact:

```
Cron 07:00 ──► arXiv Atom (cs.CL, cs.LG, cs.AI)
           ──► new-since-watermark ──► two-tier keyword filter
           ──► Haiku relevance score 1-5, keep >= 4
           ──► POST /ingest
```

`/ingest` grew the two things the workflow needs. It takes a **URL** (rewriting
`arxiv.org/abs/...` to the PDF link), refuses anything that isn't a PDF by magic bytes
rather than `Content-Type`, and only keeps plain remote filenames — anything else falls
back to the content hash, because a remote filename lands on your filesystem. And it
indexes **one paper** via `library.ingest_paper` instead of rescanning the directory;
adding today's paper used to re-embed all ~4,800 chunks. Duplicates are decided on
content hash, so the same paper re-downloaded under a different name is still a
duplicate and is skipped rather than re-embedded.

Validating the workflow against live arXiv data — rather than by reading it — found two
faults before it ever ran:

- **`max_results: 100` would have silently lost most of every day.** arXiv posted 197
  papers to those three categories on 2026-08-20 alone. Each run would have taken the
  newest 100, advanced the watermark past the rest, and never come back for them. Now
  600, and it warns when a page fails to reach the watermark at all.
- **A flat keyword list is expensive noise.** On a 60-paper sample it matched a paper on
  fraudulent Solana memecoins (`backdoor`) and one on decision-tree classifiers
  (`interpretability`) — each a billed model call. Keywords are now two tiers:
  unambiguous terms (`jailbreak`, `sandbagging`, `emergent misalignment`) match anywhere;
  borrowed ones (`interpretability`, `auditing`, `backdoor`) count only when the abstract
  is also plainly about language models. Same sample: 8 matches down to 4, LLM-relevant
  ones kept. Scoring cost lands around $0.013/day.

The watermark commits at the *end* of a run, not when papers are read, so a failure
re-scans rather than skipping a day for good; content-hash dedupe makes the re-scan
safe. An unreadable score fails closed instead of becoming an ingest.

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

route      → library — "compares two named works (STAR-1 and RealSafe-R1) across the
                        library, requiring retrieval and synthesis of fragments from
                        multiple papers"
retrieve   → 2 searches, not 1:
               "STAR-1 aligning reasoning models"
               "RealSafe-R1 aligning reasoning models"
             interleaved → 8 chunks, 4 from each paper
grade      → 5 kept, 3 rejected; no rewrite needed
               rejected: bibliography entry; no findings
               rejected: describes a different alignment method entirely
generate   → answer with 6 verified quotes, both papers cited
grounding  → partially_grounded — 2 claims flagged as not supported by the excerpts

                                              20.5s, $0.0334, recall@5 = 1.0
```

Two lines in that trace are the point of v3.

**`retrieve → 2 searches`.** A single search for that question returns eight chunks from
*one* of the two papers, and no amount of grading or rewriting can build a comparison out
of one side.

**`grounding → partially_grounded`.** The answer is good and its six quotes all verify —
and the checker still found two sentences the excerpts don't support. That's the node
earning its place: without it those two claims ship looking exactly as authoritative as
the six that are real.

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
| section | rerank | 0.693 | 0.826 | 0.884 | 0.884 | 0.930 |
| section | dense | 0.614 | 0.814 | 0.882 | 0.890 | 0.870 |
| fixed | hybrid | 0.681 | 0.841 | 0.879 | 0.879 | 0.925 |
| section_context | rerank | 0.693 | 0.832 | 0.867 | 0.867 | 0.925 |
| section | hybrid | 0.652 | 0.847 | 0.870 | 0.870 | 0.901 |
| section_context | dense | 0.678 | 0.838 | 0.864 | 0.864 | 0.907 |
| section | bm25 | 0.661 | 0.800 | 0.861 | 0.864 | 0.903 |
| section_context | bm25 | 0.661 | 0.800 | 0.861 | 0.864 | 0.903 |
| fixed | rerank | 0.702 | 0.832 | 0.861 | 0.870 | 0.927 |
| fixed | bm25 | 0.705 | 0.813 | 0.844 | 0.844 | 0.938 |
| fixed | dense | 0.588 | 0.779 | 0.842 | 0.853 | 0.849 |

**Reranking improves precision at the top and does not improve recall.** recall@1 rises
in all three chunking strategies (+0.021 to +0.041 against hybrid) and MRR with it, but
recall@5 *falls* in two of three — the cross-encoder is confident enough about the wrong
chunks to push a correct paper out of the deeper ranks. `section_context + hybrid` keeps
the headline 0.899.

The spec asked specifically whether reranking beats `fixed + bm25` on recall@1 (0.705).
**It doesn't — and the first time I ran this, the number said it nearly did.** The v3 run
put `fixed + rerank` at 0.711 and I argued a 0.006 gap on 57 questions was too small to
call a win. Re-running the whole matrix three weeks later returned **0.702** for that same
config, on byte-identical chunks.

That is worth more than the conclusion it changed. `fixed` chunking was untouched by the
v4 fix that prompted the re-run, so it was the control — and the control moved. Chasing it
down: rebuilding the Chroma index perturbs the *deep tail* of approximate (HNSW) search
while leaving the head alone. On 20 golden questions the top-5 came back identical every
time, but the **depth-15 list differed on 5 of 20** — and reranking draws its candidates
from depth 15. Dense, BM25 and hybrid at k<=10 reproduce exactly across a rebuild; the
rerank rows do not. Two consecutive rerank runs against the *same* index agreed to three
decimals, so this is rebuild sensitivity, not run-to-run noise.

**So the rerank numbers are not reproducible across index rebuilds, and the 0.711-vs-0.705
comparison sat inside that instability.** The conclusion survives — reranking does not beat
BM25 on recall@1 — but it now rests on the finding that repeats: +0.02-0.04 against hybrid
*within* each chunking strategy, which shows up in all three.

A depth sweep then found recall@1 **identical** at depths 15, 20 and 30 — the
cross-encoder's top pick is already inside hybrid's top 15, so a deeper pool gives it
nothing to promote — while latency scales linearly (1.1s / 1.5s / 2.5s). So: depth 15,
and **opt-in rather than default**, because it buys precision the generator barely
notices and costs a second of wall clock.

### Operational numbers

From a 60-run end-to-end sweep (30 questions × both paper states, $3.46):

| route | p50 | p95 | median cost |
|---|---|---|---|
| `single_paper` | 15.5s | 34.9s | **$0.2014** |
| `library` | 19.7s | 30.1s | $0.0318 |

Whole-paper questions cost **6× a library question** — 15–60k tokens of paper, mostly as
a cache *write* since no paper repeats within a sweep. That is a real argument against
the router's "prefer single_paper when torn" tiebreak, which v0 chose on quality grounds
before anyone had measured the bill.

| grounding verdict | share |
|---|---|
| grounded | 50.0% |
| partially_grounded | 46.7% |
| ungrounded | 0.0% |
| api_refused | 3.3% |

**Zero `ungrounded` in 60 runs.** The one-stricter-regeneration path has never fired
outside its tests. Sixty runs cannot distinguish "the threshold is too strict" from
"generation genuinely doesn't invent when it has excerpts in front of it".

### The eval caught a bug, and the fix is the interesting part

The first full sweep produced a number I didn't like: **10 of 40 answerable library
questions came back with no citations at all**. On a project whose whole premise is
verified citations, that is the worst possible failure.

The cause was the grader. Asked "does this excerpt help answer the question?" about a
comparison — *how do STAR-1 and RealSafe-R1 differ?* — it rejected every excerpt with
some variant of "does not compare X and Y". Which is true of every excerpt individually,
and fatal collectively. Two rounds of prompt wording barely moved it, because a **binary**
relevance judgement cannot express "this is one half of a two-part answer".

Two structural changes fixed it. The grader now reports `on_topic` **separately** from
`relevant`, so an excerpt about the right subject survives even when it doesn't answer
alone; and retrieval splits a comparison into one search per side and interleaves the
results, so both sides are in the pool to begin with.

Same 30 questions, same seed, before and after:

| | before | after |
|---|---|---|
| **Answers with zero citations** | **10 / 40** | **4 / 40** |
| Mean citations per answer | 3.25 | 3.95 |
| Mean chunks surviving grading | 3.70 | 5.50 |
| Rewrite trigger rate | 27.1% | **12.5%** |
| Rewrites fired despite perfect retrieval | 7 | 2 |
| Within-question recall delta | −0.026 | **+0.024** |
| improved / unchanged / worsened | 4 / 5 / 4 | **1 / 5 / 0** |

The rewrite column is the part I'd point at. Those questions were never hard — the loop
was firing twice on each of them to paper over chunks the grader had wrongly thrown away.
Now it fires half as often, and when it does, nothing gets worse.

**One number moved the wrong way and it isn't a regression:** `grounded` fell from 65% to
50%, with `partially_grounded` rising to match. More surviving chunks means longer answers
making more claims, and more claims means more surface for the checker to flag. The
previous run's better-looking score was partly earned by answers that said nothing —
a zero-citation non-answer asserts nothing and scores `grounded`.

Two questions still answer without citations (q48, q49) and both are retrieval problems,
not grading ones. See "what I'd do differently".

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
pytest                    # 404 tests, no network needed
```

Drop PDFs into `papers/`, then:

```
streamlit run src/personal_ra/app.py         # the UI
python -m personal_ra.library --rebuild      # ingest the library (offline, no API)
python -m personal_ra.library --figures      # ... and describe figures (costs vision calls)
python -m personal_ra.graph.run "which papers study sandbagging?"        # the v3 router
python -m personal_ra.ask papers\foo.pdf "What dataset did they use?"    # one paper
python -m personal_ra.search "which papers use contrastive loss?"        # cross-paper
python -m personal_ra.eval --matrix          # the 12-config retrieval matrix (offline)
python -m personal_ra.eval --routes          # route accuracy (~$0.22, calls Haiku)
python -m personal_ra.parse papers\foo.pdf --debug          # inspect parsing
python -m personal_ra.vision papers\foo.pdf --detect-only   # equation-page scores
python -m personal_ra.vision papers\foo.pdf --figures-only  # describe figures (~$0.006 each)
uvicorn personal_ra.api:app --reload         # HTTP API, SSE node stream
npx n8n                                      # the daily arXiv job (import automation/)
```

For Claude Code, add the `.mcp.json` shown below and restart — then just ask questions.

**Two things are optional and off by default.** Web search needs `TAVILY_API_KEY` (free
tier, 1,000 credits/month); without it the router simply never offers that route. Tracing
needs `docker compose up -d` and Langfuse keys in `.env`; without them the tracer is a
null object and never loads the SDK, so cloning the repo does not require Docker.

First load of an equation-heavy paper runs a one-time vision pass (a few cents,
cached in `.cache/vision/`). Figure description is opt-in everywhere — `--figures` on
the ingester, `figures=True` on `enrich_paper` — so no existing entry point starts
billing vision calls for figures on its own.

## The library (v1): cross-paper search, evaluated

34 papers → 4,871 section-aware chunks in Chroma, of which 67 are figure descriptions
for the three papers that have been through vision (idempotent ingest — re-running
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

`pytest` — 404 tests, all green. Unit tests never call the network; the Anthropic client
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
  tagged "References" containing appendix text). *(This entry used to end "Metadata-only;
  chunk text is unaffected" — **that was wrong**. Under `section_context` the label is part
  of `embed_text`, so a bad one costs retrieval, not just tidiness. v4 found a second source
  of bad labels and fixed it: see below.)*
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

- **Four of 40 answerable library questions still answer with no citations**, down from
  ten. The two comparison cases are q48 and q49, and both are retrieval problems wearing
  grader clothing: q48 retrieves bibliography pages and NeurIPS checklists, which the
  grader is *right* to reject; q49 asks about two labs' approaches and even per-entity
  retrieval turns up only one side. Neither is reachable from the grader, and I stopped
  tuning rather than overfit a prompt to a sample of two.
- **Paper-level recall@5 counts a bibliography hit as success.** A paper scores 1.0
  because *some* chunk of it was retrieved — including its reference list. This actively
  misled me: I diagnosed a grader bug from a "perfect retrieval, everything rejected"
  reading that turned out to be junk chunks the grader correctly threw away. Any recall
  number for this corpus deserves that asterisk.
- **The rewrite loop's headline metric is confounded.** Comparing recall on questions
  where the loop fired against questions where it didn't measures *question difficulty* —
  the loop fires because retrieval was bad, so the fired group is the hard group by
  construction. On the current run the two disagree in **sign**: −0.143 between groups,
  **+0.024** within question. Both are reported; only the second one means anything, and
  the harness prints them together so nobody quotes the flattering one alone.
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
- **`/ingest` reindexed the whole library** to add one paper, because `library.ingest`
  scans a directory rather than accepting a file. *(v4: fixed — `library.ingest_paper`
  indexes a single PDF, and duplicates are decided on content hash.)*

### New in v4

- **Figures are indexed for three papers, not 34.** 67 figure chunks are live; the other
  31 papers have none. Describing all 508 detected figures is ~$3.15 typical, ~$6.03 worst
  case, and that decision waited on evidence that figure content changes an answer at all
  (it does — see above). The library is therefore heterogeneous: a question whose answer
  lives in a figure is answerable for three papers and silently not for the rest.
- **The figure answer's quote is flagged `unverified`, by accident rather than design.**
  Library answers verify quotes by re-parsing the PDF, which has no vision output in it,
  so a figure-derived quote can never match. That reads as correct — a described figure
  genuinely is not the paper's words — but it means `Citation.source_type == "figure"`,
  built precisely to mark this case, is unreachable on the library path. Verification
  would have to run against the enriched paper for that to work.
- **Figure descriptions are model inference, and some are wrong in a detail.** Not
  "vague" — wrong and fluent, with no signal that anything is off. A `ℙ[Red team wins]`
  axis came back as "Pixel entropy", then as "Pixel team value" after the prompt was
  tightened, and never
  as "unclear" despite being told to say so. Instructing a model to admit uncertainty did
  not make it admit uncertainty. Structure is reliable; small rotated labels and series
  colours are not.
- **Detection is caption-anchored**, so a figure whose caption doesn't start `Figure N:`
  or `Fig. N` is invisible to it, and a sub-figure is only described if it happens to
  fall inside the parent's crop. The 508 figures it finds are the ones that name
  themselves conventionally.
- **The keyword filter was tuned on a single 60-paper sample.** Two tiers beat one flat
  list 8 matches to 4 on that sample, and I have no second sample. Its precision and
  recall against papers I'd actually have wanted are unmeasured, and unlike routing there
  is no golden set for "should this have been ingested" — which is the natural next eval
  investment, the way route eval was for v3.
- **The relevance threshold (≥ 4 of 5) is a guess.** It has never been calibrated against
  papers I did or didn't want, and a scorer nobody has scored is just a confident filter.
- **The workflow has never completed a real run.** It is validated against live arXiv
  data — the JS was executed against a real feed, the watermark verified to return zero
  on a second pass, the overflow warning verified to fire — but it ships with
  `dryRun: true` and no digest node, and "no duplicates over a week" is a claim I cannot
  yet make.
- **`/ingest` is unauthenticated and will fetch a URL you give it.** That is fine bound
  to localhost, which is how it is meant to run, but it means anything that can reach the
  port can make this machine download a file and index it. Exposing it to a tunnel or a
  hosted n8n would need auth and a host allowlist first; that is why the n8n instance
  runs locally rather than in the cloud.
- **Docker is gone, not broken.** v3's note said Docker Desktop couldn't start after an
  auto-update; by v4 it had been uninstalled entirely — no binary, no service, no
  uninstall entry. That is why n8n runs under `npx` rather than in a container, and why
  the Langfuse traces above are still unobserved.

## Roadmap

| Version | Status |
|---|---|
| v0 | ✅ Single-paper Q&A with verified citations, vision math, PDF-reader UI |
| v1 | ✅ Cross-paper search (hybrid + RRF), 63-question golden set, eval matrix |
| v2 | ✅ MCP server — the library queryable from inside Claude Code |
| v3 | ✅ LangGraph router, cross-encoder reranking, chunk grading + rewrite loop, grounding checks, HITL gate before web search, route eval, per-node tracing, FastAPI |
| v4 | 🚧 Figure understanding ✅, arXiv auto-ingest workflow ✅ (needs a week of real runs), library-wide figure indexing deferred, Zotero sync not started |

See [PERSONAL-RA.md](PERSONAL-RA.md) for the full build spec. (Spec drift note: vision
math transcription was pulled forward from v4 into v0 by agreement, and the v0 UI grew
beyond "deliberately plain" — PDF viewer, highlights, notes.)

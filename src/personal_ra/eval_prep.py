"""Tooling for writing the golden set by hand.

  --papers    list every ingested paper with its paper_id (for expected_paper_ids)
  --template  write a starter eval/golden_set.yaml with schema + examples
  --draft N   generate N draft questions from the corpus for you to correct
  --check     validate an existing golden set and show category counts

Drafts are a starting point, not the set: the spec asks for hand-written
questions, because auto-generated ones mirror whatever the retriever already
does well and make the eval self-congratulatory.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from personal_ra.eval import GOLDEN_SET, load_golden_set
from personal_ra.library import COLLECTION, DB_PATH

TEMPLATE = """# Golden set for retrieval evaluation.
#
# Schema (one entry per question):
#   id:                 unique short id, e.g. q01
#   question:           what you'd actually type
#   ground_truth:       the correct answer in your own words (free text)
#   expected_paper_ids: papers that MUST be retrieved to answer it
#                       (get ids from: python -m personal_ra.eval_prep --papers)
#                       leave empty ONLY for unanswerable questions
#   category:           factual | cross_paper | comparison | unanswerable
#
# Aim for ~50 questions, including 5 deliberately unanswerable ones.
# Rough target mix: 20 factual, 15 cross_paper, 10 comparison, 5 unanswerable.
#
# Category meanings:
#   factual      — answer lives in one paper ("what dataset did X use?")
#   cross_paper  — spans several ("which of my papers use contrastive loss?")
#   comparison   — contrasts papers ("how do X and Y differ on Z?")
#   unanswerable — nothing in the library covers it; the system should refuse

- id: q01
  question: "Which of my papers study emergent misalignment?"
  ground_truth: "Several papers cover emergent misalignment, including model organisms
    work and the persona-features paper."
  expected_paper_ids: ["REPLACE_ME", "REPLACE_ME"]
  category: cross_paper

- id: q02
  question: "What does the fine-tuning similarity paper claim about guardrail durability?"
  ground_truth: "High similarity between alignment and fine-tuning datasets weakens
    safety guardrails; low similarity yields more robust models."
  expected_paper_ids: ["REPLACE_ME"]
  category: factual

- id: q03
  question: "What accuracy does my library report on ImageNet classification?"
  ground_truth: "Not covered — this library is about LLM safety, not image classification."
  expected_paper_ids: []
  category: unanswerable
"""

DRAFT_PROMPT = """Below are excerpts from one research paper.

Write {n} questions a researcher might ask that this paper answers, as YAML matching
this schema exactly:

- id: dNN
  question: "..."
  ground_truth: "one or two sentences answering it"
  expected_paper_ids: ["{paper_id}"]
  category: factual

Rules:
- Questions must be answerable from these excerpts alone.
- Ask about specifics (datasets, methods, numbers, findings), not vague themes.
- Do not mention "the paper" or "this study" — write them as a researcher would
  type them into a search box over a whole library.
- Output only YAML, no commentary.

Excerpts:
{excerpts}"""


def _collection(db_path: Path):
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def list_papers(db_path: Path = DB_PATH) -> list[dict]:
    data = _collection(db_path).get(include=["metadatas"])
    papers: dict[str, dict] = {}
    for meta in data["metadatas"]:
        entry = papers.setdefault(
            meta["paper_id"],
            {
                "paper_id": meta["paper_id"],
                "title": meta["paper_title"],
                "year": meta["year"],
                "source": Path(meta["source_path"]).name,
                "chunks": 0,
            },
        )
        entry["chunks"] += 1
    return sorted(papers.values(), key=lambda p: p["title"].lower())


def draft_questions(
    n_papers: int = 5,
    per_paper: int = 3,
    db_path: Path = DB_PATH,
    client=None,
    seed: int = 0,
) -> str:
    """Generate draft questions from randomly sampled papers (one API call each)."""
    import anthropic

    from personal_ra.ask import MODEL

    client = client or anthropic.Anthropic()
    data = _collection(db_path).get(include=["documents", "metadatas"])
    by_paper: dict[str, list[str]] = defaultdict(list)
    titles: dict[str, str] = {}
    for doc, meta in zip(data["documents"], data["metadatas"]):
        by_paper[meta["paper_id"]].append(doc)
        titles[meta["paper_id"]] = meta["paper_title"]

    rng = random.Random(seed)
    chosen = rng.sample(sorted(by_paper), min(n_papers, len(by_paper)))
    blocks = []
    for pid in chosen:
        chunks = by_paper[pid]
        sample = chunks[:2] + rng.sample(chunks, min(6, len(chunks)))
        excerpts = "\n\n".join(f"- {c}" for c in sample)
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": DRAFT_PROMPT.format(
                        n=per_paper, paper_id=pid, excerpts=excerpts[:12000]
                    ),
                }
            ],
        )
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        text = re.sub(r"^```(yaml)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        blocks.append(f"# --- draft from: {titles[pid]} ({pid})\n{text}")
    return _renumber_ids("\n\n".join(blocks))


def _renumber_ids(yaml_text: str) -> str:
    """The model numbers each paper's questions from d01, so ids collide across
    blocks. Renumber globally — duplicate ids fail golden-set validation."""
    counter = 0

    def bump(match: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        return f"{match.group(1)}d{counter:02d}"

    return re.sub(r"^(\s*-\s*id:\s*)\S+", bump, yaml_text, flags=re.MULTILINE)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Golden-set authoring helpers.")
    ap.add_argument("--papers", action="store_true", help="list ingested papers + ids")
    ap.add_argument("--template", action="store_true", help="write a starter golden set")
    ap.add_argument("--draft", type=int, metavar="N", help="draft questions from N papers")
    ap.add_argument("--per-paper", type=int, default=3)
    ap.add_argument("--check", action="store_true", help="validate the golden set")
    ap.add_argument("--out", type=Path, default=GOLDEN_SET)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    args = ap.parse_args(argv)

    if args.papers:
        papers = list_papers(args.db)
        print(f"{'paper_id':14} {'year':5} {'chunks':>6}  title")
        for p in papers:
            print(f"{p['paper_id']:14} {p['year']:5} {p['chunks']:>6}  {p['title'][:60]}")
        print(f"\n{len(papers)} papers. Use paper_id values in expected_paper_ids.")
        return

    if args.template:
        if args.out.exists():
            raise SystemExit(f"{args.out} already exists — not overwriting.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(TEMPLATE, encoding="utf-8")
        print(f"Wrote {args.out}")
        print("Next: python -m personal_ra.eval_prep --papers   (to get paper ids)")
        return

    if args.draft:
        load_dotenv()
        print(draft_questions(args.draft, args.per_paper, args.db))
        return

    if args.check:
        questions = load_golden_set(args.out)  # raises on malformed entries
        counts: dict[str, int] = defaultdict(int)
        for q in questions:
            counts[q.category] += 1
        print(f"{args.out}: {len(questions)} questions, all valid")
        for category, n in sorted(counts.items()):
            print(f"  {category:14} {n}")
        if counts["unanswerable"] < 5:
            print(f"\nNote: spec asks for 5 unanswerable, you have {counts['unanswerable']}.")
        return

    ap.print_help()


if __name__ == "__main__":
    main()

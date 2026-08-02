from pathlib import Path
from types import SimpleNamespace

from personal_ra.parse import Page, Paper
from personal_ra.search import RetrievedChunk


def make_paper(page_texts: list[str]) -> Paper:
    """Build a synthetic Paper directly, no PDF involved."""
    pages = [Page(number=i + 1, text=t) for i, t in enumerate(page_texts)]
    full = "\n\n".join(f"[PAGE {p.number}]\n\n{p.text}" for p in pages)
    return Paper(
        path=Path("synthetic.pdf"),
        title="Synthetic",
        pages=pages,
        full_text=full,
        n_tokens=len(full) // 4,
    )


# --- offline graph doubles ---------------------------------------------------------
#
# Once retrieve/generate/single_paper became real, any test that streams the graph
# would otherwise open the real Chroma library and call the real API. These keep the
# suite offline, per the testing policy in the spec.


def make_chunk(text: str, cid: str = "c1", paper: str = "p1", page: int = 3) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid,
        text=text,
        metadata={
            "paper_id": paper,
            "paper_title": f"Paper {paper.upper()}",
            "page": page,
            "source_path": f"papers/{paper}.pdf",
            "section": "Method",
        },
        score=0.5,
        dense_rank=1,
        bm25_rank=2,
    )


class FakeLibrary:
    """Enough of search.Library for the graph: search() and collection.get()."""

    def __init__(self, chunks=None):
        # Two by default: MIN_GRADED_CHUNKS is 2, so a one-chunk library would
        # starve grading and send every traversal test through the rewrite loop.
        self.chunks = (
            chunks
            if chunks is not None
            else [make_chunk("a finding", "c1"), make_chunk("another finding", "c2", page=5)]
        )
        self.searches = []
        self.collection = SimpleNamespace(get=self._get)

    def search(self, query, k=8, mode="hybrid", paper_id=None, **kwargs):
        self.searches.append({"query": query, "k": k, "mode": mode, "paper_id": paper_id})
        return list(self.chunks[:k])

    def _get(self, where=None, limit=None, include=None):
        return {"metadatas": [c.metadata for c in self.chunks[: limit or len(self.chunks)]]}


def _tool_use(name: str, payload: dict, in_tokens: int = 500, out_tokens: int = 40):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name=name, input=payload)],
        usage=SimpleNamespace(
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


def _text(body: str, in_tokens: int = 900, out_tokens: int = 120):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=body)],
        usage=SimpleNamespace(
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


class FakeAnthropic:
    """One fake for every node, dispatching on which tool the caller forced.

    A fixed side_effect list cannot work any more: a single graph run now calls the
    router, the rewriter, the generator and the grounding checker, and a test that
    cares about one of them should not have to script the other three.
    """

    def __init__(
        self,
        route="library",
        rewrite="a sharper query",
        verdict_substantive=True,
        unsupported=(),
        split=None,
        answer="An answer with no quotes.",
    ):
        self.route = route
        # A list lets a test give each rewrite pass a distinct query, which is how
        # you tell "the loop fed itself forward" from "the loop ran twice".
        self.rewrites = list(rewrite) if isinstance(rewrite, (list, tuple)) else [rewrite]
        self.split = split
        self.verdict_substantive = verdict_substantive
        self.unsupported = list(unsupported)
        self.answer = answer
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        tool = (kwargs.get("tool_choice") or {}).get("name")
        if tool == "classify_route":
            return _tool_use("classify_route", {"route": self.route, "reason": "fake router"})
        if tool == "grade_excerpt":
            return _tool_use("grade_excerpt", {"relevant": True, "reason": "fake grade"})
        if tool == "split_query":
            return _tool_use("split_query", {"queries": list(self.split or [])})
        if tool == "rewrite_query":
            n = len(self.calls_for("rewrite_query"))
            query = self.rewrites[min(n - 1, len(self.rewrites) - 1)]
            return _tool_use("rewrite_query", {"query": query, "rationale": f"fake rewrite {n}"})
        if tool == "report_grounding":
            return _tool_use(
                "report_grounding",
                {
                    "makes_substantive_claims": self.verdict_substantive,
                    "unsupported_claims": self.unsupported,
                },
            )
        return _text(self.answer)

    def calls_for(self, tool_name: str) -> list[dict]:
        return [c for c in self.calls if (c.get("tool_choice") or {}).get("name") == tool_name]

    def generation_calls(self) -> list[dict]:
        return [c for c in self.calls if not c.get("tool_choice")]


class FakeAsyncAnthropic(FakeAnthropic):
    """Same dispatch, awaitable — the grader uses AsyncAnthropic."""

    def __init__(self, *args, relevant=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.relevant = relevant
        self.messages = SimpleNamespace(create=self._acreate)

    async def _acreate(self, **kwargs):
        self.calls.append(kwargs)
        return _tool_use("grade_excerpt", {"relevant": self.relevant, "reason": "fake grade"})

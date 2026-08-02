import json
from pathlib import Path

import chromadb
import pytest

from personal_ra import mcp_server
from personal_ra.library import COLLECTION

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_PDF = FIXTURES / "two_column.pdf"

# Chunk text below is real text from the fixture PDF, so quote verification works.
CHUNKS = [
    {
        "id": "aaa111:0",
        "vec": [1.0, 0.0],
        "text": "alpha bravo studies attention mechanisms in depth on page 1",
        "meta": {
            "paper_id": "aaa111",
            "paper_title": "Attention Study",
            "page": 1,
            "section": "1. Introduction",
            "chunk_index": 0,
            "year": 2024,
            "source_path": str(FIXTURE_PDF),
        },
    },
    {
        "id": "aaa111:1",
        "vec": [0.9, 0.1],
        "text": "echo foxtrot reports experimental results here on page 2",
        "meta": {
            "paper_id": "aaa111",
            "paper_title": "Attention Study",
            "page": 2,
            "section": "3. Experiments",
            "chunk_index": 1,
            "year": 2024,
            "source_path": str(FIXTURE_PDF),
        },
    },
    {
        "id": "bbb222:0",
        "vec": [0.0, 1.0],
        "text": "golf hotel concludes with future work notes on page 3",
        "meta": {
            "paper_id": "bbb222",
            "paper_title": "Future Work Paper",
            "page": 3,
            "section": "5. Conclusion",
            "chunk_index": 0,
            "year": 2026,
            "source_path": str(FIXTURE_PDF),
        },
    },
]


@pytest.fixture(autouse=True)
def server_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the server at a tiny throwaway index and results dir."""
    db = tmp_path / "db"
    client = chromadb.PersistentClient(path=str(db))
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.upsert(
        ids=[c["id"] for c in CHUNKS],
        embeddings=[c["vec"] for c in CHUNKS],
        documents=[c["text"] for c in CHUNKS],
        metadatas=[c["meta"] for c in CHUNKS],
    )
    results = tmp_path / "results"
    monkeypatch.setattr("personal_ra.search.embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])
    mcp_server.configure(db_path=db, results_dir=results)
    yield results
    mcp_server.configure(db_path=Path("chroma_db"), results_dir=Path("eval/results"))


# --- list_papers ------------------------------------------------------------


def test_list_papers_returns_inventory() -> None:
    out = mcp_server.list_papers()
    assert out["n_papers"] == 2
    titles = [p["title"] for p in out["papers"]]
    assert titles == ["Attention Study", "Future Work Paper"]  # sorted
    attention = out["papers"][0]
    assert attention["paper_id"] == "aaa111"
    assert attention["year"] == 2024
    assert attention["pages"] == 2  # highest page seen in metadata
    assert attention["chunks"] == 2


# --- search_library ---------------------------------------------------------


def test_search_library_returns_scored_excerpts() -> None:
    out = mcp_server.search_library("attention mechanisms", k=3)
    assert out["n_results"] > 0
    first = out["results"][0]
    assert set(first) == {
        "paper_id",
        "paper_title",
        "page",
        "section",
        "year",
        "score",
        "text",
    }
    assert first["score"] > 0


def test_search_library_paper_filter() -> None:
    out = mcp_server.search_library("notes", k=5, paper_id="bbb222")
    assert out["n_results"] > 0
    assert all(r["paper_id"] == "bbb222" for r in out["results"])


def test_search_library_year_filter() -> None:
    out = mcp_server.search_library("page", k=5, year_min=2025)
    assert all(r["year"] >= 2025 for r in out["results"])


def test_search_library_rejects_unknown_paper_id() -> None:
    with pytest.raises(ValueError, match="Unknown paper_id"):
        mcp_server.search_library("anything", paper_id="nope999")


def test_max_per_paper_spreads_results_across_papers() -> None:
    # Regression for the live-test finding: without a cap, one paper's excerpts
    # can occupy every slot and hide papers that use different wording.
    uncapped = mcp_server.search_library("page", k=3)
    capped = mcp_server.search_library("page", k=3, max_per_paper=1)
    counts: dict[str, int] = {}
    for r in capped["results"]:
        counts[r["paper_id"]] = counts.get(r["paper_id"], 0) + 1
    assert all(n <= 1 for n in counts.values())
    assert capped["n_papers"] >= uncapped["n_papers"]
    assert capped["n_papers"] == len(counts)


def test_max_per_paper_preserves_score_order() -> None:
    out = mcp_server.search_library("page", k=5, max_per_paper=2)
    scores = [r["score"] for r in out["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_reports_paper_count() -> None:
    out = mcp_server.search_library("page", k=5)
    assert out["n_papers"] == len({r["paper_id"] for r in out["results"]})


# --- read_paper -------------------------------------------------------------


def test_read_paper_full_text() -> None:
    out = mcp_server.read_paper("aaa111")
    assert out["section"] is None
    assert out["title"] == "Attention Study"
    assert "[PAGE 1]" in out["text"]  # page markers preserved
    assert out["pages"] == 3 and out["approx_tokens"] > 0  # real fixture has 3 pages


def test_read_paper_reports_available_sections() -> None:
    # The description promises this field on every response, so the model can
    # learn section names without reading a whole paper to find them.
    full = mcp_server.read_paper("aaa111")
    assert isinstance(full["available_sections"], list)
    assert "(no section)" not in full["available_sections"]


def test_read_paper_unknown_id_gives_actionable_error() -> None:
    with pytest.raises(ValueError) as exc:
        mcp_server.read_paper("missing")
    message = str(exc.value)
    assert "Unknown paper_id" in message
    assert "list_papers" in message  # tells the model how to recover
    assert "Traceback" not in message


def test_read_paper_unknown_section_lists_available() -> None:
    with pytest.raises(ValueError) as exc:
        mcp_server.read_paper("aaa111", section="9. Nonexistent")
    message = str(exc.value)
    assert "No section" in message
    assert "Available sections" in message
    assert "Retry with one of these" in message  # actionable, not just descriptive


def test_read_paper_missing_source_file(tmp_path: Path) -> None:
    db = tmp_path / "db2"
    client = chromadb.PersistentClient(path=str(db))
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.upsert(
        ids=["ghost:0"],
        embeddings=[[1.0, 0.0]],
        documents=["text"],
        metadatas=[
            {
                "paper_id": "ghost",
                "paper_title": "Deleted Paper",
                "page": 1,
                "section": "",
                "chunk_index": 0,
                "year": 2025,
                "source_path": str(tmp_path / "gone.pdf"),
            }
        ],
    )
    mcp_server.configure(db_path=db)
    with pytest.raises(ValueError, match="no longer at"):
        mcp_server.read_paper("ghost")


# --- verify_quote -----------------------------------------------------------


def test_verify_quote_confirms_real_quote() -> None:
    out = mcp_server.verify_quote("echo foxtrot reports experimental results", "aaa111")
    assert out["verified"] is True
    assert out["match_type"] == "exact"
    assert out["page"] == 1  # where it really sits in the fixture PDF
    assert out["paper_title"] == "Attention Study"


def test_verify_quote_rejects_invented_quote() -> None:
    out = mcp_server.verify_quote("this sentence is not in the paper at all", "aaa111")
    assert out["verified"] is False
    assert out["match_type"] == "failed"
    assert out["page"] is None


def test_verify_quote_unknown_paper() -> None:
    with pytest.raises(ValueError, match="Unknown paper_id"):
        mcp_server.verify_quote("anything", "nope999")


# --- resources --------------------------------------------------------------


def test_library_index_resource() -> None:
    text = mcp_server.library_index()
    assert "Personal-RA library (2 papers)" in text
    assert "`aaa111`" in text and "Attention Study" in text


def test_eval_latest_resource_without_results(server_env: Path) -> None:
    assert "No evaluation results yet" in mcp_server.eval_latest()


def test_eval_latest_resource_with_results(server_env: Path) -> None:
    server_env.mkdir(parents=True, exist_ok=True)
    (server_env / "20260101T000000Z.json").write_text(
        json.dumps(
            {
                "golden_set": "eval/golden_set.yaml",
                "k": 10,
                "configs": [
                    {
                        "chunking": "section_context",
                        "retrieval": "hybrid",
                        "metrics": {
                            "recall@1": 0.6,
                            "recall@3": 0.8,
                            "recall@5": 0.9,
                            "recall@10": 0.9,
                            "mrr": 0.85,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    text = mcp_server.eval_latest()
    assert "Retrieval evaluation" in text
    assert "section_context" in text and "0.900" in text


# --- protocol surface -------------------------------------------------------


@pytest.mark.anyio
async def test_all_tools_registered_with_schemas() -> None:
    tools = {t.name: t for t in await _maybe_await(mcp_server.server.list_tools())}
    # v3 adds ask_router. §3.11 is explicit that it is an addition, not a
    # replacement, so the four v2 tools must all still be here.
    assert {"search_library", "read_paper", "list_papers", "verify_quote"} <= set(tools)
    assert set(tools) == {
        "search_library",
        "read_paper",
        "list_papers",
        "verify_quote",
        "ask_router",
    }
    for tool in tools.values():
        assert tool.description and len(tool.description) > 200  # descriptions do the work
        assert tool.input_schema["type"] == "object"

    search = tools["search_library"].input_schema
    assert "query" in search["required"]
    assert set(search["properties"]) == {
        "query",
        "k",
        "paper_id",
        "year_min",
        "year_max",
        "max_per_paper",
    }
    assert tools["list_papers"].input_schema.get("properties", {}) == {}
    assert set(tools["verify_quote"].input_schema["required"]) == {"quote", "paper_id"}


@pytest.mark.anyio
async def test_resources_registered() -> None:
    uris = {str(r.uri) for r in await _maybe_await(mcp_server.server.list_resources())}
    assert uris == {"library://index", "eval://latest"}


@pytest.mark.anyio
async def test_call_tool_through_server_returns_clean_error() -> None:
    # The SDK raises ToolError here and turns it into an is_error result at the
    # protocol layer; either way what reaches the model is this message, so it
    # must be actionable and free of internals.
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError) as exc:
        await _maybe_await(mcp_server.server.call_tool("read_paper", {"paper_id": "bogus"}))
    message = str(exc.value)
    assert "Unknown paper_id" in message
    assert "list_papers" in message
    assert "Traceback" not in message and 'File "' not in message


@pytest.mark.anyio
async def test_call_tool_validates_arguments() -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError):  # missing required 'quote'
        await _maybe_await(mcp_server.server.call_tool("verify_quote", {"paper_id": "aaa111"}))


@pytest.mark.anyio
async def test_call_tool_through_server_succeeds() -> None:
    result = await _maybe_await(mcp_server.server.call_tool("list_papers", {}))
    assert not result.is_error
    text = " ".join(getattr(c, "text", "") for c in result.content)
    assert "Attention Study" in text


async def _maybe_await(value):
    """list_tools/call_tool are async in MCP 2.x; tolerate sync too."""
    if hasattr(value, "__await__"):
        return await value
    return value


@pytest.fixture
def anyio_backend():
    return "asyncio"


# --- ask_router (v3 step 3.11) ----------------------------------------------


def _graph_returning(values: dict, waiting=None):
    """A stand-in graph: invoke() does nothing, get_state() returns `values`."""
    from types import SimpleNamespace

    calls = []

    class Stub:
        def invoke(self, arg, config):
            calls.append(arg)

        def get_state(self, config):
            return SimpleNamespace(values=values, tasks=())

    return Stub(), calls


def test_ask_router_returns_route_and_grounding(monkeypatch) -> None:
    """§3.11: the payload must carry the route and the grounding verdict — that is
    the whole reason to reach for this tool over search_library."""
    stub, _ = _graph_returning(
        {
            "answer": "An answer.",
            "route": "library",
            "route_reason": "spans several papers",
            "citations": [{"quote": "q", "page": 3, "paper_title": "Paper A"}],
            "unverified": [],
            "grounding": {"verdict": "grounded", "unsupported": []},
            "rewrite_count": 1,
            "retrieval_queries": ["one", "two"],
        }
    )
    mcp_server._state["graph"] = stub
    monkeypatch.setattr(mcp_server, "pending_approval", lambda g, c: None, raising=False)

    out = mcp_server.ask_router("which papers use contrastive loss?")
    assert out["route"] == "library"
    assert out["route_reason"] == "spans several papers"
    assert out["grounding"]["verdict"] == "grounded"
    assert out["citations"][0]["page"] == 3
    assert out["rewrite_count"] == 1
    assert out["web_search_declined"] is False
    mcp_server._state["graph"] = None


def test_ask_router_declines_web_search_rather_than_approving_it(monkeypatch) -> None:
    """A tool call cannot pause for a human, and approving a paid search on the
    user's behalf is exactly what the gate exists to prevent."""
    import personal_ra.graph.build as build_mod

    stub, calls = _graph_returning(
        {"answer": "From the library only.", "route": "web", "grounding": {"verdict": "grounded"}}
    )
    mcp_server._state["graph"] = stub
    monkeypatch.setattr(build_mod, "pending_approval", lambda g, c: {"query": "q"})

    out = mcp_server.ask_router("has anyone published a follow-up?")
    assert out["web_search_declined"] is True
    # The resume that ran was a denial, not an approval.
    assert any(getattr(c, "resume", None) is False for c in calls)
    mcp_server._state["graph"] = None


def test_ask_router_rejects_an_unknown_paper_id() -> None:
    with pytest.raises(ValueError, match="Unknown paper_id"):
        mcp_server.ask_router("a question", paper_id="not-a-real-id")


@pytest.mark.anyio
async def test_ask_router_description_says_it_does_not_replace_the_others() -> None:
    """The description is what a model reads to choose. It must steer toward the
    direct tools by default, per §3.11."""
    tools = {t.name: t for t in await _maybe_await(mcp_server.server.list_tools())}
    description = tools["ask_router"].description
    assert "DOES NOT REPLACE" in description
    assert "search_library" in description and "read_paper" in description
    assert set(tools["ask_router"].input_schema["properties"]) == {"question", "paper_id"}
    assert tools["ask_router"].input_schema["required"] == ["question"]

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


# --- read_paper -------------------------------------------------------------


def test_read_paper_full_text() -> None:
    out = mcp_server.read_paper("aaa111")
    assert out["section"] is None
    assert out["title"] == "Attention Study"
    assert "[PAGE 1]" in out["text"]  # page markers preserved
    assert out["pages"] == 3 and out["approx_tokens"] > 0  # real fixture has 3 pages


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
    assert "Available sections include" in message


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
    assert set(tools) == {"search_library", "read_paper", "list_papers", "verify_quote"}
    for tool in tools.values():
        assert tool.description and len(tool.description) > 200  # descriptions do the work
        assert tool.input_schema["type"] == "object"

    search = tools["search_library"].input_schema
    assert "query" in search["required"]
    assert set(search["properties"]) == {"query", "k", "paper_id", "year_min", "year_max"}
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

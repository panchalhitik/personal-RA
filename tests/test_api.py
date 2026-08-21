"""Step 3.10 — the FastAPI wrapper. Injected graph and library; no network."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from conftest import FakeAnthropic, FakeAsyncAnthropic, FakeLibrary
from personal_ra import api
from personal_ra.api import create_app, sse
from personal_ra.graph.build import build_graph, sqlite_checkpointer


class FakeTavily:
    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(query)
        return {
            "results": [
                {
                    "title": "A follow-up",
                    "url": "https://arxiv.org/abs/2601.1",
                    "content": "text",
                    "score": 0.9,
                }
            ]
        }


@pytest.fixture
def client(tmp_path):
    graph = build_graph(
        checkpointer=sqlite_checkpointer(tmp_path / "api.db"),
        client=FakeAnthropic(),
        async_client=FakeAsyncAnthropic(),
        library=FakeLibrary(),
        web_client=FakeTavily(),
    )
    return TestClient(create_app(graph=graph, library=FakeLibrary()))


def events(response) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, payload) pairs."""
    out, name = [], None
    for line in response.text.splitlines():
        if line.startswith("event: "):
            name = line[len("event: ") :]
        elif line.startswith("data: ") and name:
            out.append((name, json.loads(line[len("data: ") :])))
    return out


# --- health and papers ------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_papers_lists_the_library(client):
    body = client.get("/papers").json()
    assert body["n_papers"] >= 1
    paper = body["papers"][0]
    assert {"paper_id", "title", "pages", "chunks"} <= set(paper)


# --- /ask streams node transitions ------------------------------------------------


def test_ask_streams_a_node_event_per_transition(client):
    response = client.post("/ask", json={"question": "what is sandbagging?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    parsed = events(response)
    assert parsed[0][0] == "start"
    nodes = [p["node"] for name, p in parsed if name == "node"]
    assert nodes == ["route", "retrieve", "rerank", "grade", "generate", "grounding"]


def test_the_stream_ends_with_the_answer_and_its_provenance(client):
    parsed = events(client.post("/ask", json={"question": "what is sandbagging?"}))
    name, payload = parsed[-1]
    assert name == "answer"
    assert payload["route"] == "library"
    assert payload["route_reason"]
    assert "citations" in payload and "grounding" in payload


def test_node_events_carry_counts_not_chunk_bodies(client):
    """A delta holds whole chunk texts; a progress event must not."""
    parsed = events(client.post("/ask", json={"question": "what is sandbagging?"}))
    retrieve = next(p for name, p in parsed if name == "node" and p["node"] == "retrieve")
    assert retrieve["n_chunks"] == 2
    assert "chunks" not in retrieve


def test_a_question_needs_content(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_thread_id_is_echoed_and_reusable(client):
    parsed = events(client.post("/ask", json={"question": "q", "thread_id": "mine"}))
    assert parsed[0][1]["thread_id"] == "mine"
    assert client.get("/ask/mine").status_code == 200


# --- status and the trace ---------------------------------------------------------


def test_status_returns_the_per_node_trace(client):
    client.post("/ask", json={"question": "what is sandbagging?", "thread_id": "t1"})
    body = client.get("/ask/t1").json()
    assert body["status"] == "complete"
    traced_nodes = {row["node"] for row in body["trace"]}
    assert {"route", "grade", "generate"} <= traced_nodes
    assert all("cost_usd" in row for row in body["trace"])


def test_an_unknown_thread_is_a_404_not_a_stack_trace(client):
    response = client.get("/ask/no-such-thread")
    assert response.status_code == 404
    assert "no-such-thread" in response.json()["detail"]


def test_approving_an_unknown_thread_is_also_404(client):
    assert client.post("/ask/nope/approve", json={"approved": True}).status_code == 404


# --- the approval gate over HTTP --------------------------------------------------


def _halt_at_approval(client, thread_id="web1"):
    return events(
        client.post(
            "/ask",
            json={"question": "has anyone published a follow-up?", "thread_id": thread_id},
        )
    )


def test_a_web_question_ends_the_stream_asking_for_approval(client, monkeypatch):
    monkeypatch.setattr("personal_ra.api.pending_approval", lambda g, c: {"query": "q"})
    parsed = _halt_at_approval(client)
    assert parsed[-1][0] == "approval_required"
    assert parsed[-1][1]["thread_id"] == "web1"


def test_approve_resumes_a_halted_thread(tmp_path):
    tavily = FakeTavily()
    graph = build_graph(
        checkpointer=sqlite_checkpointer(tmp_path / "a.db"),
        client=FakeAnthropic(route="web"),
        async_client=FakeAsyncAnthropic(),
        library=FakeLibrary(),
        web_client=tavily,
    )
    client = TestClient(create_app(graph=graph, library=FakeLibrary()))

    client.post("/ask", json={"question": "newer version?", "thread_id": "w"})
    assert client.get("/ask/w").json()["status"] == "awaiting_approval"
    assert tavily.calls == []  # nothing spent before the decision

    body = client.post("/ask/w/approve", json={"approved": True}).json()
    assert len(tavily.calls) == 1
    assert body["route"] == "web"
    assert client.get("/ask/w").json()["status"] == "complete"


def test_denying_falls_back_to_the_library(tmp_path):
    tavily = FakeTavily()
    graph = build_graph(
        checkpointer=sqlite_checkpointer(tmp_path / "d.db"),
        client=FakeAnthropic(route="web"),
        async_client=FakeAsyncAnthropic(),
        library=FakeLibrary(),
        web_client=tavily,
    )
    client = TestClient(create_app(graph=graph, library=FakeLibrary()))
    client.post("/ask", json={"question": "newer version?", "thread_id": "d"})
    client.post("/ask/d/approve", json={"approved": False})
    assert tavily.calls == []


def test_approving_a_thread_that_is_not_waiting_is_a_409(client):
    client.post("/ask", json={"question": "q", "thread_id": "done"})
    response = client.post("/ask/done/approve", json={"approved": True})
    assert response.status_code == 409
    assert "not waiting" in response.json()["detail"]


def test_a_malformed_approval_body_denies_rather_than_spends(tmp_path):
    """The default must be the cheap, safe answer: an empty body is not consent."""
    tavily = FakeTavily()
    graph = build_graph(
        checkpointer=sqlite_checkpointer(tmp_path / "m.db"),
        client=FakeAnthropic(route="web"),
        async_client=FakeAsyncAnthropic(),
        library=FakeLibrary(),
        web_client=tavily,
    )
    client = TestClient(create_app(graph=graph, library=FakeLibrary()))
    client.post("/ask", json={"question": "newer version?", "thread_id": "x"})
    client.post("/ask/x/approve", json={})
    assert tavily.calls == []


# --- ingest -----------------------------------------------------------------------


def test_ingest_rejects_a_missing_file(client):
    response = client.post("/ingest", json={"path": "no/such/file.pdf"})
    assert response.status_code == 404


def test_ingest_rejects_a_non_pdf(client, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("not a paper", encoding="utf-8")
    response = client.post("/ingest", json={"path": str(notes)})
    assert response.status_code == 400


def test_dry_run_writes_nothing(tmp_path):
    calls = []
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    client = TestClient(
        create_app(graph=object(), library=FakeLibrary(), ingest_fn=lambda p: calls.append(p) or {})
    )

    body = client.post("/ingest", json={"path": str(pdf), "dry_run": True}).json()
    assert body["dry_run"] is True and body["ingested"] is False
    assert calls == []  # v4.2 needs to exercise the workflow without writing


def test_a_real_ingest_calls_through(tmp_path):
    calls = []
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    client = TestClient(
        create_app(
            graph=object(),
            library=FakeLibrary(),
            ingest_fn=lambda p: calls.append(p) or {"n_chunks": 12},
        )
    )
    body = client.post("/ingest", json={"path": str(pdf)}).json()
    assert body["ingested"] is True and body["n_chunks"] == 12
    assert len(calls) == 1


def ingest_client(tmp_path, calls=None, fetched=None):
    """An app whose ingest is recorded and whose fetch writes a stub PDF."""
    calls = [] if calls is None else calls

    def fake_fetch(url):
        if fetched is not None:
            fetched.append(url)
        destination = tmp_path / "downloads" / Path(urlparse(url).path).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"%PDF-1.4 fake")
        return destination

    return TestClient(
        create_app(
            graph=object(),
            library=FakeLibrary(),
            ingest_fn=lambda p: calls.append(p) or {"chunks": 41, "reindexed": True},
            fetch_fn=fake_fetch,
        )
    )


def test_ingest_accepts_an_arxiv_pdf_url(tmp_path):
    calls, fetched = [], []
    client = ingest_client(tmp_path, calls, fetched)
    body = client.post("/ingest", json={"url": "https://arxiv.org/pdf/2501.01234v1.pdf"}).json()
    assert body["ingested"] is True and body["chunks"] == 41
    assert body["source_url"] == "https://arxiv.org/pdf/2501.01234v1.pdf"
    assert fetched == ["https://arxiv.org/pdf/2501.01234v1.pdf"]
    assert len(calls) == 1


def test_a_duplicate_is_reported_and_not_reindexed(tmp_path):
    client = TestClient(
        create_app(
            graph=object(),
            library=FakeLibrary(),
            ingest_fn=lambda p: {"already_present": True, "reindexed": False},
        )
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    body = client.post("/ingest", json={"path": str(pdf)}).json()
    assert body["already_present"] is True
    assert body["ingested"] is False  # the digest must not claim a paper was added


def test_ingest_rejects_malformed_requests(tmp_path, client):
    assert client.post("/ingest", json={}).status_code == 400  # neither
    both = {"path": "a.pdf", "url": "https://example.com/a.pdf"}
    assert client.post("/ingest", json=both).status_code == 400  # both


def test_ingest_rejects_a_non_http_url(client):
    body = {"url": "file:///C:/Windows/System32/drivers/etc/hosts"}
    assert client.post("/ingest", json=body).status_code == 400


def test_url_dry_run_writes_nothing(tmp_path):
    calls = []
    client = ingest_client(tmp_path, calls)
    body = client.post(
        "/ingest", json={"url": "https://arxiv.org/pdf/2501.01234v1.pdf", "dry_run": True}
    ).json()
    assert body["dry_run"] is True and body["ingested"] is False
    assert body["filename"] == "2501.01234v1.pdf"
    assert calls == []


def test_download_is_cleaned_up_either_way(tmp_path):
    client = ingest_client(tmp_path)
    client.post("/ingest", json={"url": "https://arxiv.org/pdf/2501.01234v1.pdf"})
    client.post("/ingest", json={"url": "https://arxiv.org/pdf/2501.09999v1.pdf", "dry_run": True})
    assert list((tmp_path / "downloads").glob("*.pdf")) == []


# --- _fetch_pdf, the one part that touches the network ----------------------------


def fake_response(content: bytes):
    return SimpleNamespace(content=content, raise_for_status=lambda: None)


def test_fetch_rewrites_an_arxiv_abstract_url(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(api, "DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(
        api.httpx, "get", lambda url, **kw: seen.append(url) or fake_response(b"%PDF-1.4 x")
    )
    path = api._fetch_pdf("https://arxiv.org/abs/2501.01234v1")
    assert seen == ["https://arxiv.org/pdf/2501.01234v1"]
    assert path.read_bytes().startswith(b"%PDF")


def test_fetch_rejects_a_response_that_is_not_a_pdf(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(api.httpx, "get", lambda url, **kw: fake_response(b"<html>404</html>"))
    with pytest.raises(HTTPException) as caught:
        api._fetch_pdf("https://arxiv.org/pdf/2501.01234v1.pdf")
    assert caught.value.status_code == 400


def test_fetch_falls_back_to_a_content_hash_for_an_odd_filename(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(api.httpx, "get", lambda url, **kw: fake_response(b"%PDF-1.4 x"))
    # a name with a space, and a URL with no filename at all
    for url in ("https://example.com/my paper.pdf", "https://example.com"):
        path = api._fetch_pdf(url)
        assert path.parent == tmp_path  # never escapes the download directory
        assert re.fullmatch(r"[0-9a-f]{12}\.pdf", path.name)


def test_fetch_keeps_a_plain_name_and_adds_the_suffix(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(api.httpx, "get", lambda url, **kw: fake_response(b"%PDF-1.4 x"))
    assert api._fetch_pdf("https://example.com/2501.01234v1.pdf").name == "2501.01234v1.pdf"
    assert api._fetch_pdf("https://example.com/download?id=7").name == "download.pdf"


def test_fetch_never_writes_outside_the_download_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(api.httpx, "get", lambda url, **kw: fake_response(b"%PDF-1.4 x"))
    path = api._fetch_pdf("https://example.com/%2e%2e%2f%2e%2e%2fevil.pdf")
    assert path.parent == tmp_path


# --- the SSE encoding itself ------------------------------------------------------


def test_sse_frames_are_terminated_by_a_blank_line():
    frame = sse("node", {"node": "route"})
    assert frame.startswith("event: node\ndata: ")
    assert frame.endswith("\n\n")
    assert json.loads(frame.split("data: ")[1].strip()) == {"node": "route"}


def test_the_library_is_only_touched_when_asked(tmp_path):
    """Importing the module must not open Chroma; `app = create_app()` at module
    scope would otherwise connect on import."""
    library = FakeLibrary()
    client = TestClient(create_app(graph=object(), library=library))
    assert client.get("/health").status_code == 200
    assert library.searches == []

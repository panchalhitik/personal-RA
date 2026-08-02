"""Step 3.10 — the FastAPI wrapper. Injected graph and library; no network."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from conftest import FakeAnthropic, FakeAsyncAnthropic, FakeLibrary
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

"""Step 4.2 — the committed n8n workflow.

The workflow runs inside n8n, so pytest cannot execute it. What it can do is stop
a structurally broken or secret-carrying workflow reaching the repo, which is the
failure mode that would otherwise be found by importing it into n8n by hand.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).parent.parent / "automation" / "arxiv_ingest.json"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return json.loads(WORKFLOW.read_text(encoding="utf-8"))


def test_workflow_is_valid_json_with_the_expected_shape(workflow):
    assert workflow["name"]
    assert workflow["nodes"] and isinstance(workflow["connections"], dict)


def test_every_connection_points_at_a_node_that_exists(workflow):
    names = {n["name"] for n in workflow["nodes"]}
    for source, outputs in workflow["connections"].items():
        assert source in names, f"connection from unknown node {source!r}"
        for branch in outputs["main"]:
            for link in branch:
                assert link["node"] in names, f"connection to unknown node {link['node']!r}"


def test_the_pipeline_the_spec_asks_for_is_present(workflow):
    types = [n["type"] for n in workflow["nodes"]]
    assert "n8n-nodes-base.scheduleTrigger" in types  # cron
    assert types.count("n8n-nodes-base.httpRequest") >= 3  # arXiv, score, ingest
    assert "n8n-nodes-base.if" in types  # the score threshold


def test_it_posts_to_the_ingest_endpoint_with_a_url_and_dry_run(workflow):
    ingest = next(n for n in workflow["nodes"] if n["name"] == "POST /ingest")
    body = ingest["parameters"]["jsonBody"]
    assert "pdfUrl" in body and "dry_run" in body
    assert ingest["parameters"]["url"] == "={{ $('Settings').first().json.ingestUrl }}"


def test_no_credentials_or_keys_are_committed(workflow):
    raw = WORKFLOW.read_text(encoding="utf-8")
    assert "sk-ant-" not in raw
    assert not re.search(r"\bx-api-key\b\s*[\"']?\s*:\s*[\"']\S", raw)
    for node in workflow["nodes"]:
        # A header-auth credential is selected in n8n; it must never be inlined here.
        assert "credentials" not in node, f"{node['name']} carries a credential reference"


def test_the_scorer_uses_a_cheap_model_and_the_threshold_is_configurable(workflow):
    settings = next(n for n in workflow["nodes"] if n["name"] == "Settings")
    code = settings["parameters"]["jsCode"]
    assert "claude-haiku-4-5" in code  # scoring every candidate on Opus would not be cheap
    assert "minScore" in code and "dryRun: true" in code  # ships safe


def test_the_watermark_is_committed_at_the_end_not_when_papers_are_read(workflow):
    """A failed run must re-scan, never skip a day's papers for good."""
    by_name = {n["name"]: n["parameters"].get("jsCode", "") for n in workflow["nodes"]}
    assert "state.pendingNewestId =" in by_name["New + keyword filter"]
    assert "state.lastSeenId =" not in by_name["New + keyword filter"]
    assert "state.lastSeenId =" in by_name["Record run"]

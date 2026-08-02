"""Per-node tracing to a self-hosted Langfuse.

Every node becomes a span carrying its latency, tokens, cost and the routing
decision, grouped into one trace per question so a run reads top to bottom.

Two properties matter more than the feature itself:

**Unconfigured is a no-op.** With no LANGFUSE_PUBLIC_KEY the tracer is a null
object, not a disabled client — nothing is imported, nothing is buffered, and the
graph behaves exactly as it did before this module existed. Someone cloning the
repo must not need Docker to ask a question.

**Span attributes are built by whitelist.** State is copied field by field into the
span, never wholesale. A tracer that ships whatever happens to be in scope to an
external service is one refactor away from shipping a key.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"
HOST_ENV = "LANGFUSE_HOST"
DEFAULT_HOST = "http://localhost:3000"

# Copied into spans. Everything else in State — chunk bodies, answer text, history —
# is either too large to be useful in a span or not ours to send anywhere.
TRACED_STATE_FIELDS = (
    "route",
    "route_reason",
    "rewrite_count",
    "rewrite_reason",
    "rerank",
    "approved",
    "retrieval_queries",
    "thread_id",
)
# Counted rather than copied: the numbers are the diagnostic, the contents are bulk.
COUNTED_STATE_FIELDS = ("chunks", "graded_chunks", "rejected_chunks", "web_results", "citations")


def safe_attributes(state: dict) -> dict:
    """The whitelisted view of state that may leave the process."""
    attrs = {k: state.get(k) for k in TRACED_STATE_FIELDS if state.get(k) is not None}
    for field in COUNTED_STATE_FIELDS:
        value = state.get(field)
        if value is not None:
            attrs[f"n_{field}"] = len(value)
    grounding = state.get("grounding") or {}
    if grounding.get("verdict"):
        attrs["grounding_verdict"] = grounding["verdict"]
    return attrs


def node_usage(usage: dict, node: str) -> dict:
    """The usage entry a node wrote, if it wrote one.

    Node names and usage keys mostly match; rewrite and grounding suffix theirs with
    an attempt number, so a prefix match catches those without hardcoding the count.
    """
    if node in usage and isinstance(usage[node], dict):
        return usage[node]
    for key, value in usage.items():
        if key.startswith(f"{node}_") and isinstance(value, dict):
            return value
    return {}


class SpanRecord:
    """Carries a node's result back to the span that is timing it.

    The node's delta is only available *after* the wrapped call returns, and its
    usage entry is the thing §3.9 actually wants traced — so the context manager
    hands one of these out and reads it on the way out.
    """

    def __init__(self) -> None:
        self.delta: dict = {}

    def set(self, delta) -> None:
        self.delta = delta if isinstance(delta, dict) else {}


class NullTracer:
    """What you get when Langfuse is not configured. Does nothing, cheaply."""

    enabled = False

    @contextmanager
    def span(self, node: str, state: dict):
        yield SpanRecord()

    def flush(self) -> None:
        pass


class LangfuseTracer:
    """One trace per question, one span per node."""

    enabled = True

    def __init__(self, client) -> None:
        self.client = client

    def _trace_id(self, state: dict) -> str:
        # Seeded on thread + question so a follow-up in the same conversation gets
        # its own trace rather than appending spans to the previous answer's.
        seed = f"{state.get('thread_id', 'default')}:{state.get('original_question', '')}"
        return self.client.create_trace_id(seed=seed)

    @contextmanager
    def span(self, node: str, state: dict):
        started = time.perf_counter()
        observation = self.client.start_observation(
            trace_context={"trace_id": self._trace_id(state)},
            name=node,
            as_type="span",
            input=safe_attributes(state),
            metadata={"node": node, "route": state.get("route")},
        )
        record = SpanRecord()
        try:
            yield record
        except Exception as exc:
            # GraphInterrupt is the approval gate working, not a failure — it is
            # how `interrupt` suspends the run, and marking it ERROR would paint
            # every approval request red in the dashboard.
            if type(exc).__name__ not in ("GraphInterrupt", "GraphBubbleUp"):
                observation.update(level="ERROR", status_message=type(exc).__name__)
            raise
        finally:
            usage = node_usage(record.delta.get("usage") or {}, node)
            observation.update(
                output=safe_attributes({**state, **record.delta}),
                metadata={
                    "node": node,
                    "route": (record.delta.get("route") or state.get("route")),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "cache_read_tokens": usage.get("cache_read_tokens", 0),
                    "cache_write_tokens": usage.get("cache_write_tokens", 0),
                },
                usage_details={
                    k: v
                    for k, v in (
                        ("input", usage.get("input_tokens")),
                        ("output", usage.get("output_tokens")),
                    )
                    if v
                },
                cost_details=({"total": usage["cost_usd"]} if usage.get("cost_usd") else None),
            )
            observation.end()

    def flush(self) -> None:
        self.client.flush()


def make_tracer(client=None) -> NullTracer | LangfuseTracer:
    """A real tracer when configured, a null one otherwise.

    The import is inside the branch so `langfuse` (and its OpenTelemetry stack) is
    never loaded by someone who has not asked for tracing.
    """
    if client is not None:
        return LangfuseTracer(client)
    if not os.environ.get(PUBLIC_KEY_ENV):
        return NullTracer()

    from langfuse import Langfuse

    return LangfuseTracer(
        Langfuse(
            public_key=os.environ[PUBLIC_KEY_ENV],
            secret_key=os.environ.get(SECRET_KEY_ENV),
            host=os.environ.get(HOST_ENV, DEFAULT_HOST),
        )
    )


def traced(node: str, fn, tracer):
    """Wrap a node so it emits a span. Returns `fn` untouched when tracing is off."""
    if not getattr(tracer, "enabled", False):
        return fn

    def wrapper(state, *args, **kwargs):
        with tracer.span(node, state) as record:
            delta = fn(state, *args, **kwargs)
            record.set(delta)
        return delta

    wrapper.__name__ = getattr(fn, "__name__", node)
    return wrapper

# Personal-RA

Grounded Q&A over a personal research-paper library. Ask questions about your
papers and get answers with verified, page-numbered quotes — every citation is
string-matched back to the source PDF, so invented quotes get flagged instead
of displayed.

**Status: v0 in progress** (single-paper Q&A with verified citations).

## Quickstart

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env   # then put your Anthropic API key in .env
pytest
```

See [PERSONAL-RA.md](PERSONAL-RA.md) for the full build spec.

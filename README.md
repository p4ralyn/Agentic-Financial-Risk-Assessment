# Agentic Financial Risk Assessment

An autonomous agent that assesses a public company's bankruptcy risk by
computing its financial ratios from filed statements, benchmarking them
against its industry peers, and refusing to publish a report the numbers
do not support.

Built on LangGraph. The distinguishing piece is the validation loop: a
second agent scores the analyst's report against the ratios that were
actually computed, and a report scoring below 70 is sent back for
correction rather than shown.

## How it works

1. **Input** — a ticker symbol, for example `VZ`.
2. **Fetch** — a deterministic node pulls the balance sheet and income
   statement from Yahoo Finance, computes fourteen ratios and the Altman
   Z-score, and does the same concurrently for the company's industry
   peers to produce benchmark medians.
3. **Analysis** — the analyst agent writes a risk report, arguing from the
   gap between the company's ratios and the peer medians.
4. **Validation** — the validator agent scores the report 0–100 against
   those same ratios, flagging unsupported claims and internal
   contradictions.
5. **Correction** — below 70, the report returns to the analyst with the
   specific flags to fix. At most two corrections, then the best attempt is
   shown with its objections attached.

Ratios are computed, never entered by hand, so the validator always has
real numbers to check against.

### On missing ratios

A ratio that cannot be computed is reported as null, not as zero and not as
infinity. This matters most for debt-to-equity and return on equity, which
have no meaningful value when shareholders' equity is zero or negative —
a condition indicating severe distress. The agent is instructed to read a
null as "not computable", never as a clean bill of health.

## Tech stack

- **Framework** — LangChain, LangGraph
- **Model** — Google Gemini 2.0 Flash
- **Frontend** — Streamlit
- **Data** — Yahoo Finance (`yfinance`)
- **Deployment** — Docker, Google Cloud Run

## Layout

| File | Responsibility |
|---|---|
| `ratios.py` | Ratio math, Altman Z-score, median aggregation. Pure functions |
| `financial_data.py` | Industry maps, Yahoo Finance access, peer-group fetch |
| `graph.py` | Agent state, prompts, validation schema, nodes and edges |
| `streamlit_app.py` | Dashboard |
| `main.py` | CLI |

## Setup

```bash
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_key" > .env
streamlit run streamlit_app.py
```

Or from the command line:

```bash
python main.py VZ
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs offline: Yahoo Finance is mocked with fixture frames and the
model is stubbed, so no test makes a network call or spends quota. It covers
the ratio math (including zero and negative equity), the yfinance field-name
aliases, and — driving the compiled graph — that the retry loop terminates
against a validator that always rejects.

## Docker

```bash
docker build -t risk-agent .
docker run -p 8080:8080 --env-file .env risk-agent
```

The container binds `$PORT`, defaulting to 8080, which is what Cloud Run
injects:

```bash
gcloud run deploy risk-agent-service \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```


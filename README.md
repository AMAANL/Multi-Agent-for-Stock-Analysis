# Stock Analysis Agent

A multi-agent stock research assistant. Instead of asking an LLM to
"analyze this stock" in one giant prompt, the pipeline is split into
independent, testable stages — deterministic Python does the data
fetching and math, and a Gemini-compatible model is used only as the final
**reasoning/writing layer** on top of already-computed numbers.

> **This is a research/education tool, not financial advice.** Scores
> and narratives are model-generated and should not be used as the
> sole basis for investment decisions.

## Architecture

```
Ticker input
     │
     ▼
Data Collector (price, financials, company info)
     │
     ├──────────────┬──────────────┐
     ▼               ▼               ▼
Fundamental      Financial       Technical / Chart
  Agent            Agent            Agent
(Gemini)        (pure Python)    (pure Python)
     │               │               │
     └───────────────┼───────────────┘
                      ▼
             Investment Analyst Agent (Gemini)
                      │
                      ▼
              Final structured report
```

Only two of the four agents call an LLM at all:

-- **Fundamental Agent** — Gemini interprets the raw financials + company
  profile to score business quality, moat, growth, management, and
  industry position, and lists risks.
- **Financial Agent** — pure Python. Computes revenue/profit growth,
  ROE, ROCE, debt/equity, margins, P/E, P/B, FCF growth, etc.
  The model never touches these numbers directly.
- **Technical Agent** — pure Python. Detects candlestick patterns
  (Morning Star, Evening Star, Engulfing, Doji, Hammer, Shooting Star,
  Three White Soldiers/Black Crows), chart patterns (Cup & Handle,
  Double Top/Bottom, Head & Shoulders, triangles, flag/pennant,
  rectangle), indicators (RSI, MACD, SMA/EMA, ATR, Bollinger Bands),
  and support/resistance levels — all with explicit, deterministic
  rules, not an LLM "eyeballing" a chart.
-- **Investment Analyst Agent** — Gemini combines the three structured
  outputs above into one final report (it never sees raw price data).

## Project layout

```
stock-analysis-agent/
├── app/
│   ├── main.py                  # CLI entry point
│   ├── agents/
│   │   ├── fundamental_agent.py # Gemini: business quality / moat / risks
│   │   ├── financial_agent.py   # Pure Python: ratios & growth
│   │   ├── technical_agent.py   # Pure Python: indicators & patterns
│   │   └── analyst_agent.py     # Gemini: final combined report
│   ├── data/
│   │   ├── market_data.py       # OHLCV via yfinance
│   │   ├── financial_data.py    # Income/balance/cashflow via yfinance
│   │   └── company_data.py      # Company profile via yfinance
│   ├── technical/
│   │   ├── indicators.py        # RSI, MACD, SMA/EMA, ATR, Bollinger
│   │   ├── candlestick.py       # 2-3 candle pattern detectors
│   │   ├── chart_patterns.py    # Multi-candle pattern detectors
│   │   └── support_resistance.py
│   ├── llm/
│   │   └── gemini_client.py     # Gemini-compatible API wrapper
│   ├── models/
│   │   └── schemas.py           # Pydantic schemas (the data contract)
│   └── utils/
│       └── helpers.py
├── tests/
│   ├── test_indicators.py
│   ├── test_candlestick.py
│   └── test_patterns.py
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd stock-analysis-agent
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env
# then edit .env and add your GEMINI_API_KEY (provider-specific)
```

## Usage

```bash
python -m app.main --ticker INFY.NS
python -m app.main --ticker AAPL --timeframe 1d
```

Example output:

```
INFY.NS — STOCK ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Overall Score: 8.1/10

Fundamentals       8.4/10
Financial Health   8.7/10
Technical Setup    7.6/10
Valuation          7.2/10
...
```

## Tests

The deterministic layers (indicators, candlestick and chart pattern
detection) are unit tested since they don't depend on the network or
an API key:

```bash
pytest tests/ -v
```

## Extending this project

- Swap `app/data/*.py` for a paid market-data vendor without touching
  any agent code — the Pydantic schemas are the contract.
- Add a FastAPI layer around `build_report()` in `app/main.py` and a
  React + TradingView Lightweight Charts frontend for a full web UI.
- Add more chart-pattern detectors to `app/technical/chart_patterns.py`
  following the same "return a `ChartPattern` with a confidence score"
  convention.
-- Swap providers by editing `app/llm/gemini_client.py`
  — every agent talks to it through `chat()` / `chat_json()`.

## License

MIT

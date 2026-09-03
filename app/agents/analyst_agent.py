"""
Investment Analyst Agent.

Combines Fundamental, Financial, and Technical agent outputs
into one structured research report.

Gemini is used only as the reasoning and narrative layer.
"""

from __future__ import annotations

from app.llm.gemini_client import GeminiClient
from app.models.schemas import AnalystReport, StockBundle


SYSTEM_PROMPT = """You are the final Investment Analyst in a
multi-agent stock research pipeline.

You receive already-computed fundamental, financial, and technical
data for one stock.

Rules:

1. Do not invent numbers.
2. Do not modify supplied numbers.
3. Do not fetch external information.
4. Base your analysis only on the supplied data.
5. Explain contradictions between fundamentals and technicals.
6. Identify the most important risks.
7. Give a balanced research view.
8. This is research/analysis, NOT financial advice.
9. Do not give a direct buy/sell instruction.

The valuation_score must be between 0 and 10.

Return ONLY the requested JSON structure.
"""


ANALYST_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "valuation_score": {
            "type": "NUMBER",
            "description": (
                "Valuation attractiveness score from 0 to 10, "
                "considering PE, PB, EV/EBITDA and growth."
            ),
        },
        "narrative": {
            "type": "STRING",
            "description": (
                "Concise 3 short paragraphs covering fundamentals, financial health, and technical setup"
            ),
        },
        "risks": {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
            },
            "description": "Key risks identified from the supplied data.",
        },
        "final_view": {
            "type": "STRING",
            "description": (
                "One short balanced paragraph describing the "
                "overall research view."
            ),
        },
    },
    "required": [
        "valuation_score",
        "narrative",
        "risks",
        "final_view",
    ],
}


def run_analyst_agent(
    bundle: StockBundle,
    client: GeminiClient | None = None,
) -> AnalystReport:

    client = client or GeminiClient()

    user_prompt = f"""Ticker:
{bundle.ticker}

Company:
{bundle.company.name}

Sector:
{bundle.company.sector}

Industry:
{bundle.company.industry}


FUNDAMENTAL ANALYSIS
Score: {bundle.fundamental.score}/10

{bundle.fundamental.model_dump()}


FINANCIAL ANALYSIS
Score: {bundle.financial.score}/10

{bundle.financial.model_dump()}


TECHNICAL ANALYSIS
Score: {bundle.technical.score}/10

Trend:
{bundle.technical.trend}

RSI:
{bundle.technical.rsi}

MACD:
{bundle.technical.macd}

MACD Signal:
{bundle.technical.macd_signal}

Above 50 SMA:
{bundle.technical.above_50_sma}

Above 200 SMA:
{bundle.technical.above_200_sma}

ATR:
{bundle.technical.atr}

Candlestick patterns detected:
{[p.name for p in bundle.technical.candlestick_patterns]}

Chart patterns detected:
{[p.name for p in bundle.technical.chart_patterns]}

Support levels:
{bundle.technical.support_resistance.support}

Resistance levels:
{bundle.technical.support_resistance.resistance}


Combine the supplied analyses into one structured research report.
"""


    llm_result = client.chat_json(
        SYSTEM_PROMPT,
        user_prompt,
        response_schema=ANALYST_SCHEMA,
        temperature=0.2,
        max_tokens=4000,
    )

    valuation_score = float(
        llm_result.get("valuation_score", 5.0)
    )

    valuation_score = max(
        0.0,
        min(10.0, valuation_score)
    )

    overall_score = round(
        bundle.fundamental.score * 0.30
        + bundle.financial.score * 0.30
        + bundle.technical.score * 0.25
        + valuation_score * 0.15,
        2,
    )

    return AnalystReport(
        ticker=bundle.ticker,
        overall_score=overall_score,
        fundamental_score=bundle.fundamental.score,
        financial_score=bundle.financial.score,
        technical_score=bundle.technical.score,
        valuation_score=round(valuation_score, 2),
        narrative=llm_result.get("narrative", ""),
        risks=llm_result.get("risks", [])
        or bundle.fundamental.risks,
        final_view=llm_result.get("final_view", ""),
    )
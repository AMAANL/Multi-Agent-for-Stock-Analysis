"""
CLI entry point.

Usage:
    python -m app.main --ticker INFY.NS
    python -m app.main --ticker AAPL --timeframe 1d

Pipeline:
    Data Collector -> Fundamental/Financial/Technical Agents
    -> Analyst Agent -> printed report
"""
from __future__ import annotations

import argparse
import os

from app.agents.analyst_agent import run_analyst_agent
from app.agents.financial_agent import run_financial_agent
from app.agents.fundamental_agent import run_fundamental_agent
from app.agents.technical_agent import run_technical_agent
from app.data.company_data import get_company_info
from app.data.financial_data import get_financials
from app.data.market_data import get_ohlcv
from app.llm.gemini_client import GeminiClient
from app.models.schemas import StockBundle


def build_report(ticker: str, timeframe: str = "1d") -> None:
    print(f"\nFetching data for {ticker}...")
    company = get_company_info(ticker)
    financials = get_financials(ticker)
    ohlcv = get_ohlcv(ticker, timeframe=timeframe)

    print("Running Financial Agent (pure Python)...")
    financial_result = run_financial_agent(financials)

    print("Running Technical Agent (pure Python)...")
    technical_result = run_technical_agent(ohlcv)

    client = GeminiClient()

    print("Running Fundamental Agent (Gemini)...")
    fundamental_result = run_fundamental_agent(company, financials, client=client)

    bundle = StockBundle(
        ticker=ticker,
        company=company,
        fundamental=fundamental_result,
        financial=financial_result,
        technical=technical_result,
    )

    print("Running Investment Analyst Agent (Gemini)...\n")
    report = run_analyst_agent(bundle, client=client)

    print_report(bundle, report)


def print_report(bundle: StockBundle, report) -> None:
    line = "━" * 40
    print(f"{bundle.ticker} — STOCK ANALYSIS")
    print(line)
    print(f"\nOverall Score: {report.overall_score}/10\n")
    print(f"Fundamentals       {report.fundamental_score}/10")
    print(f"Financial Health   {report.financial_score}/10")
    print(f"Technical Setup    {report.technical_score}/10")
    print(f"Valuation          {report.valuation_score}/10")

    print("\nFUNDAMENTALS")
    print(bundle.fundamental.summary)

    print("\nFINANCIAL HEALTH")
    fm = bundle.financial
    print(f"Revenue growth: {fm.revenue_growth}%")
    print(f"Profit growth: {fm.profit_growth}%")
    print(f"ROE: {fm.roe}%")
    print(f"ROCE: {fm.roce}%")
    print(f"Debt/Equity: {fm.debt_equity}")
    print(f"P/E: {fm.pe}")

    print("\nTECHNICAL ANALYSIS")
    print(f"Trend: {bundle.technical.trend}")
    print("\nPatterns detected:")
    all_patterns = bundle.technical.chart_patterns + bundle.technical.candlestick_patterns
    if all_patterns:
        for p in all_patterns:
            print(f"  \u2713 {p.name} — {int(p.confidence * 100)}%")
    else:
        print("  (none detected)")

    print("\nKEY LEVELS")
    print(f"Support: {bundle.technical.support_resistance.support}")
    print(f"Resistance: {bundle.technical.support_resistance.resistance}")

    print("\nRISKS")
    for r in report.risks:
        print(f"  - {r}")

    print("\nNARRATIVE")
    print(report.narrative)

    print("\nFINAL VIEW")
    print(report.final_view)
    print(f"\n{line}")
    print("Note: scores are model-generated for research/education purposes only. Not financial advice.\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-agent stock analysis (Grok-powered).")
    parser.add_argument("--ticker", default=os.getenv("DEFAULT_TICKER", "INFY.NS"), help="Ticker symbol, e.g. INFY.NS, AAPL")
    parser.add_argument("--timeframe", default=os.getenv("DEFAULT_TIMEFRAME", "1d"), help="1d / 1wk / 1mo")
    args = parser.parse_args()

    build_report(args.ticker, args.timeframe)


if __name__ == "__main__":
    main()

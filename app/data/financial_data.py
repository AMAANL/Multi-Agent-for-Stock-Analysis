"""
Fundamental / financial statement data access.

Pulls income statement, balance sheet, and cash flow data via yfinance
and normalizes it into the FinancialsRaw schema so the Financial Agent
only ever deals with clean, typed numbers.
"""
from __future__ import annotations

from typing import Optional, List
import math

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

from app.models.schemas import FinancialsRaw


def _row_series(df, row_name: str) -> List[float]:
    """Extract a row from a yfinance financial statement DataFrame as a
    chronological (oldest -> newest) list of floats, skipping NaNs."""
    if df is None or df.empty or row_name not in df.index:
        return []
    row = df.loc[row_name].dropna()
    row = row.sort_index()  # yfinance columns are dates; oldest first after sort
    return [float(v) for v in row.tolist() if not math.isnan(v)]


def get_financials(ticker: str) -> FinancialsRaw:
    if yf is None:
        raise ImportError("yfinance is not installed. `pip install yfinance`.")

    t = yf.Ticker(ticker)

    income = t.financials            # annual income statement
    balance = t.balance_sheet
    cashflow = t.cashflow
    info = t.info or {}

    revenue = _row_series(income, "Total Revenue")
    net_profit = _row_series(income, "Net Income")
    ebitda = _row_series(income, "EBITDA") or _row_series(income, "Ebitda")

    eps = []
    try:
        raw_eps = t.get_earnings_history() if hasattr(t, "get_earnings_history") else None
    except Exception:
        raw_eps = None
    if raw_eps is not None and not raw_eps.empty and "epsActual" in raw_eps.columns:
        eps = [float(v) for v in raw_eps["epsActual"].dropna().tolist()]

    total_debt = None
    total_equity = None
    current_assets = None
    current_liabilities = None
    if balance is not None and not balance.empty:
        for key in ("Total Debt", "Long Term Debt"):
            if key in balance.index:
                total_debt = float(balance.loc[key].dropna().iloc[-1])
                break
        for key in ("Common Stock Equity", "Stockholders Equity", "Total Equity Gross Minority Interest"):
            if key in balance.index:
                total_equity = float(balance.loc[key].dropna().iloc[-1])
                break
        for key in ("Current Assets",):
            if key in balance.index:
                current_assets = float(balance.loc[key].dropna().iloc[-1])
        for key in ("Current Liabilities",):
            if key in balance.index:
                current_liabilities = float(balance.loc[key].dropna().iloc[-1])

    free_cash_flow = _row_series(cashflow, "Free Cash Flow")
    operating_cash_flow = _row_series(cashflow, "Operating Cash Flow") or _row_series(
        cashflow, "Total Cash From Operating Activities"
    )

    return FinancialsRaw(
        revenue=revenue,
        net_profit=net_profit,
        ebitda=ebitda,
        eps=eps,
        total_debt=total_debt,
        total_equity=total_equity,
        current_assets=current_assets,
        current_liabilities=current_liabilities,
        free_cash_flow=free_cash_flow,
        operating_cash_flow=operating_cash_flow,
        price=info.get("currentPrice") or info.get("regularMarketPrice"),
        book_value_per_share=info.get("bookValue"),
        ebitda_ev=info.get("enterpriseToEbitda"),
        dividend_yield=info.get("dividendYield"),
    )

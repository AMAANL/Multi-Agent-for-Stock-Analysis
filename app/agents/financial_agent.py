"""
Financial Agent.

Deliberately does NOT call the LLM. All ratios are computed in plain
Python from the raw financial statement data — this is the piece that
makes the project's numbers reproducible and testable, and it's what
Grok will later be asked to *interpret* rather than calculate.
"""
from __future__ import annotations

from app.models.schemas import FinancialMetrics, FinancialsRaw
from app.utils.helpers import cagr, latest_growth, safe_div


def run_financial_agent(financials: FinancialsRaw) -> FinancialMetrics:
    revenue_growth = latest_growth(financials.revenue) or cagr(financials.revenue)
    profit_growth = latest_growth(financials.net_profit) or cagr(financials.net_profit)
    eps_growth = latest_growth(financials.eps) or cagr(financials.eps)
    fcf_growth = latest_growth(financials.free_cash_flow) or cagr(financials.free_cash_flow)

    latest_revenue = financials.revenue[-1] if financials.revenue else None
    latest_ebitda = financials.ebitda[-1] if financials.ebitda else None
    latest_profit = financials.net_profit[-1] if financials.net_profit else None
    latest_eps = financials.eps[-1] if financials.eps else None

    ebitda_margin = safe_div(latest_ebitda, latest_revenue)
    if ebitda_margin is not None:
        ebitda_margin = round(ebitda_margin * 100, 2)

    roe = safe_div(latest_profit, financials.total_equity)
    if roe is not None:
        roe = round(roe * 100, 2)

    # ROCE approximation: EBIT-ish / (equity + debt). We use net profit as a
    # simple proxy for EBIT when a true EBIT figure isn't available.
    capital_employed = None
    if financials.total_equity is not None and financials.total_debt is not None:
        capital_employed = financials.total_equity + financials.total_debt
    roce = safe_div(latest_ebitda or latest_profit, capital_employed)
    if roce is not None:
        roce = round(roce * 100, 2)

    debt_equity = safe_div(financials.total_debt, financials.total_equity)
    if debt_equity is not None:
        debt_equity = round(debt_equity, 2)

    current_ratio = safe_div(financials.current_assets, financials.current_liabilities)
    if current_ratio is not None:
        current_ratio = round(current_ratio, 2)

    pe = safe_div(financials.price, latest_eps)
    if pe is not None:
        pe = round(pe, 2)

    pb = safe_div(financials.price, financials.book_value_per_share)
    if pb is not None:
        pb = round(pb, 2)

    dividend_yield = financials.dividend_yield
    if dividend_yield is not None and dividend_yield < 1:
        # yfinance sometimes reports this as a fraction (e.g. 0.012 == 1.2%)
        dividend_yield = round(dividend_yield * 100, 2)

    return FinancialMetrics(
        revenue_growth=revenue_growth,
        profit_growth=profit_growth,
        ebitda_margin=ebitda_margin,
        eps_growth=eps_growth,
        roe=roe,
        roce=roce,
        debt_equity=debt_equity,
        current_ratio=current_ratio,
        fcf_growth=fcf_growth,
        pe=pe,
        pb=pb,
        ev_ebitda=financials.ebitda_ev,
        dividend_yield=dividend_yield,
    )

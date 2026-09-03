"""Company profile / qualitative info (sector, industry, description)."""
from __future__ import annotations

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

from app.models.schemas import CompanyInfo


def get_company_info(ticker: str) -> CompanyInfo:
    if yf is None:
        raise ImportError("yfinance is not installed. `pip install yfinance`.")

    info = yf.Ticker(ticker).info or {}
    return CompanyInfo(
        ticker=ticker,
        name=info.get("longName") or info.get("shortName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        country=info.get("country"),
        market_cap=info.get("marketCap"),
        description=info.get("longBusinessSummary"),
    )

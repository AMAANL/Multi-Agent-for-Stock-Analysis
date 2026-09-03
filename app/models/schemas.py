"""
Pydantic schemas shared across the whole pipeline.

Every agent takes/returns one of these models so the data contract
between "Python calculations" and "Grok reasoning" stays explicit and
testable, instead of passing loose dicts around.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Raw / intermediate data
# ---------------------------------------------------------------------------

class CompanyInfo(BaseModel):
    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    market_cap: Optional[float] = None
    description: Optional[str] = None


class FinancialsRaw(BaseModel):
    """Raw numbers pulled from the data layer, before ratio math."""
    revenue: List[float] = Field(default_factory=list)
    net_profit: List[float] = Field(default_factory=list)
    ebitda: List[float] = Field(default_factory=list)
    eps: List[float] = Field(default_factory=list)
    total_debt: Optional[float] = None
    total_equity: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    free_cash_flow: List[float] = Field(default_factory=list)
    operating_cash_flow: List[float] = Field(default_factory=list)
    price: Optional[float] = None
    book_value_per_share: Optional[float] = None
    ebitda_ev: Optional[float] = None
    dividend_yield: Optional[float] = None


# ---------------------------------------------------------------------------
# Agent outputs
# ---------------------------------------------------------------------------

class FundamentalResult(BaseModel):
    business_quality: float = Field(ge=0, le=10)
    growth: float = Field(ge=0, le=10)
    competitive_advantage: float = Field(ge=0, le=10)
    management: float = Field(ge=0, le=10)
    industry_position: float = Field(ge=0, le=10)
    risks: List[str] = Field(default_factory=list)
    summary: str = ""

    @property
    def score(self) -> float:
        vals = [
            self.business_quality,
            self.growth,
            self.competitive_advantage,
            self.management,
            self.industry_position,
        ]
        return round(sum(vals) / len(vals), 2)


class FinancialMetrics(BaseModel):
    revenue_growth: Optional[float] = None      # %
    profit_growth: Optional[float] = None        # %
    ebitda_margin: Optional[float] = None         # %
    eps_growth: Optional[float] = None            # %
    roe: Optional[float] = None                   # %
    roce: Optional[float] = None                  # %
    debt_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    fcf_growth: Optional[float] = None             # %
    pe: Optional[float] = None
    pb: Optional[float] = None
    ev_ebitda: Optional[float] = None
    dividend_yield: Optional[float] = None

    @property
    def score(self) -> float:
        """A simple heuristic 0-10 score used for the dashboard tiles.
        Grok does the real interpretation; this is just a quick gauge."""
        points, count = 0.0, 0

        def add(cond_score):
            nonlocal points, count
            if cond_score is not None:
                points += cond_score
                count += 1

        if self.revenue_growth is not None:
            add(min(10, max(0, self.revenue_growth / 2)))
        if self.roe is not None:
            add(min(10, max(0, self.roe / 3)))
        if self.roce is not None:
            add(min(10, max(0, self.roce / 3)))
        if self.debt_equity is not None:
            add(10 if self.debt_equity < 0.3 else max(0, 10 - self.debt_equity * 5))
        if self.fcf_growth is not None:
            add(min(10, max(0, self.fcf_growth / 2)))

        return round(points / count, 2) if count else 0.0


class CandlestickPattern(BaseModel):
    name: str
    detected: bool
    confidence: float = Field(ge=0, le=1)
    index: Optional[int] = None       # row index in the OHLCV frame
    timeframe: str = "1D"


class ChartPattern(BaseModel):
    name: str
    detected: bool
    confidence: float = Field(ge=0, le=1)
    stage: Optional[str] = None
    breakout_level: Optional[float] = None
    volume_confirmation: Optional[bool] = None


class SupportResistance(BaseModel):
    support: List[float] = Field(default_factory=list)
    resistance: List[float] = Field(default_factory=list)


class TechnicalResult(BaseModel):
    trend: str = "neutral"          # bullish / bearish / neutral
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    above_200_sma: Optional[bool] = None
    above_50_sma: Optional[bool] = None
    atr: Optional[float] = None
    candlestick_patterns: List[CandlestickPattern] = Field(default_factory=list)
    chart_patterns: List[ChartPattern] = Field(default_factory=list)
    support_resistance: SupportResistance = Field(default_factory=SupportResistance)

    @property
    def score(self) -> float:
        score = 5.0
        if self.trend == "bullish":
            score += 2
        elif self.trend == "bearish":
            score -= 2

        if self.rsi is not None:
            if 45 <= self.rsi <= 65:
                score += 1
            elif self.rsi > 75 or self.rsi < 25:
                score -= 1

        if self.above_200_sma:
            score += 1
        if self.above_50_sma:
            score += 0.5

        for p in self.chart_patterns + self.candlestick_patterns:
            if p.detected:
                score += p.confidence

        return round(max(0, min(10, score)), 2)


class StockBundle(BaseModel):
    """Everything the analyst agent needs, bundled together."""
    ticker: str
    company: CompanyInfo
    fundamental: FundamentalResult
    financial: FinancialMetrics
    technical: TechnicalResult


class AnalystReport(BaseModel):
    ticker: str
    overall_score: float
    fundamental_score: float
    financial_score: float
    technical_score: float
    valuation_score: float
    narrative: str
    risks: List[str] = Field(default_factory=list)
    final_view: str

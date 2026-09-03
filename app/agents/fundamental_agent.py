"""
Fundamental Agent.

Answers qualitative questions such as business quality, moat,
management, industry position, and risks.

All numerical financial data is provided to Gemini by the pipeline.
Gemini interprets the data but does not fetch or invent financial data.
"""

from __future__ import annotations

from app.llm.gemini_client import GeminiClient
from app.models.schemas import (
    CompanyInfo,
    FinancialsRaw,
    FundamentalResult,
)


SYSTEM_PROMPT = """You are an equity research analyst assistant.

You will be given a company's profile and raw financial numbers.

Score the business on:
- business_quality
- growth
- competitive_advantage
- management
- industry_position

Each score must be between 0 and 10.

Provide exactly 4 concise risks.
Each risk must be one short sentence.

Provide a concise summary of 2-3 sentences.

IMPORTANT:
- Use ONLY the information provided.
- Do not invent financial numbers.
- Do not add extra fields.
- Keep the response short.
- Respond ONLY with valid JSON.
- Do not use markdown or code fences.

Required JSON:
{
  "business_quality": 0,
  "growth": 0,
  "competitive_advantage": 0,
  "management": 0,
  "industry_position": 0,
  "risks": [
    "...",
    "...",
    "...",
    "..."
  ],
  "summary": "..."
}
"""

FUNDAMENTAL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "business_quality": {
            "type": "NUMBER",
            "description": "Business quality score from 0 to 10.",
        },
        "growth": {
            "type": "NUMBER",
            "description": "Business growth score from 0 to 10.",
        },
        "competitive_advantage": {
            "type": "NUMBER",
            "description": "Competitive advantage score from 0 to 10.",
        },
        "management": {
            "type": "NUMBER",
            "description": "Management quality score from 0 to 10.",
        },
        "industry_position": {
            "type": "NUMBER",
            "description": "Industry position score from 0 to 10.",
        },
        "risks": {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
            },
            "description": "Three to six concrete risks.",
        },
        "summary": {
            "type": "STRING",
            "description": "Two to four sentence business summary.",
        },
    },
    "required": [
        "business_quality",
        "growth",
        "competitive_advantage",
        "management",
        "industry_position",
        "risks",
        "summary",
    ],
}


def run_fundamental_agent(
    company: CompanyInfo,
    financials: FinancialsRaw,
    client: GeminiClient | None = None,
) -> FundamentalResult:

    client = client or GeminiClient()

    user_prompt = f"""Company profile:

Ticker: {company.ticker}
Name: {company.name}
Sector: {company.sector}
Industry: {company.industry}
Country: {company.country}
Market cap: {company.market_cap}

Description:
{company.description}

Raw financials (oldest -> newest, reporting currency):

Revenue:
{financials.revenue}

Net profit:
{financials.net_profit}

EBITDA:
{financials.ebitda}

EPS:
{financials.eps}

Total debt:
{financials.total_debt}

Total equity:
{financials.total_equity}

Free cash flow:
{financials.free_cash_flow}

Operating cash flow:
{financials.operating_cash_flow}

Score the business and identify the major risks based ONLY
on the information above.
"""

    result = client.chat_json(
        SYSTEM_PROMPT,
        user_prompt,
        response_schema=FUNDAMENTAL_SCHEMA,
        temperature=0.2,
        max_tokens=1200,
    )

    # Basic validation before constructing the Pydantic model.
    score_fields = [
        "business_quality",
        "growth",
        "competitive_advantage",
        "management",
        "industry_position",
    ]

    for field in score_fields:
        value = float(result[field])

        if not 0 <= value <= 10:
            raise ValueError(
                f"{field} must be between 0 and 10, got {value}"
            )

    return FundamentalResult(**result)

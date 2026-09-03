import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from app.data.market_data import get_ohlcv
from app.data.company_data import get_company_info
from app.data.financial_data import get_financials

from app.agents.financial_agent import run_financial_agent
from app.agents.technical_agent import run_technical_agent
from app.agents.fundamental_agent import run_fundamental_agent
from app.agents.analyst_agent import run_analyst_agent

from app.models.schemas import StockBundle


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Stock Analysis Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
        }

        .hero {
            padding: 1.5rem 0 1rem 0;
        }

        .hero h1 {
            font-size: 2.5rem;
            margin-bottom: 0.25rem;
        }

        .hero p {
            color: #888;
            font-size: 1.05rem;
        }

        .score-card {
            padding: 1rem;
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 12px;
            text-align: center;
        }

        .score-value {
            font-size: 2rem;
            font-weight: 700;
        }

        .score-label {
            color: #888;
            font-size: 0.9rem;
        }

        .section-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>📈 Stock Analysis Agent</h1>
        <p>
            Multi-agent stock research combining fundamentals,
            financial metrics, technical analysis and AI reasoning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("🔎 Analyze Stock")

    with st.form("analysis_form"):

        ticker = st.text_input(
            "Stock Ticker",
            value="SUZLON.NS",
            placeholder="Example: RELIANCE.NS",
        ).strip().upper()

        timeframe = st.selectbox(
            "Timeframe",
            options=["1d", "1wk", "1mo"],
            format_func=lambda x: {
                "1d": "Daily",
                "1wk": "Weekly",
                "1mo": "Monthly",
            }[x],
        )

        period_options = {
            "1d": ["6mo", "1y", "2y", "5y"],
            "1wk": ["2y", "5y", "10y", "max"],
            "1mo": ["5y", "10y", "max"],
        }

        period = st.selectbox(
            "Historical Period",
            period_options[timeframe],
        )

        submitted = st.form_submit_button(
            "🚀 Analyze Stock",
            use_container_width=True,
            type="primary",
        )

    st.divider()

    st.caption(
        "Data source: Yahoo Finance\n\n"
        "AI analysis is generated from calculated financial "
        "and technical data."
    )


# ---------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------

if submitted:

    if not ticker:
        st.error("Please enter a stock ticker.")
        st.stop()

    try:

        progress = st.progress(0)
        status = st.empty()

        # ---------------------------------------------
        # DATA
        # ---------------------------------------------

        status.write("📊 Fetching market data...")
        df = get_ohlcv(
            ticker=ticker,
            timeframe=timeframe,
            period=period,
        )

        progress.progress(20)

        status.write("🏢 Fetching company information...")
        company = get_company_info(ticker)

        progress.progress(35)

        status.write("💰 Fetching financial statements...")
        financials = get_financials(ticker)

        progress.progress(45)

        # ---------------------------------------------
        # FINANCIAL AGENT
        # ---------------------------------------------

        status.write("💰 Running Financial Agent...")
        financial_result = run_financial_agent(financials)

        progress.progress(60)

        # ---------------------------------------------
        # TECHNICAL AGENT
        # ---------------------------------------------

        status.write("📈 Running Technical Agent...")
        technical_result = run_technical_agent(df)

        progress.progress(70)

        # ---------------------------------------------
        # FUNDAMENTAL AGENT
        # ---------------------------------------------

        status.write("🧠 Running Fundamental Agent...")
        fundamental_result = run_fundamental_agent(
            company,
            financials,
        )

        progress.progress(82)

        # ---------------------------------------------
        # FINAL ANALYST
        # ---------------------------------------------

        status.write("🤖 Running Investment Analyst...")

        bundle = StockBundle(
            ticker=ticker,
            company=company,
            fundamental=fundamental_result,
            financial=financial_result,
            technical=technical_result,
        )

        analyst_result = run_analyst_agent(bundle)

        progress.progress(100)

        status.success("Analysis completed successfully.")

        # ---------------------------------------------
        # SAVE RESULTS
        # ---------------------------------------------

        st.session_state["df"] = df
        st.session_state["company"] = company
        st.session_state["financials"] = financials
        st.session_state["financial"] = financial_result
        st.session_state["technical"] = technical_result
        st.session_state["fundamental"] = fundamental_result
        st.session_state["analyst"] = analyst_result
        st.session_state["ticker"] = ticker

    except Exception as exc:

        st.error("Analysis failed.")

        with st.expander("Technical details"):
            st.exception(exc)

        st.stop()


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

if "analyst" not in st.session_state:

    st.info(
        "Enter a stock ticker from the sidebar and click "
        "**Analyze Stock** to begin."
    )

    st.markdown("### Example tickers")

    cols = st.columns(4)

    examples = [
        ("SUZLON.NS", "Suzlon Energy"),
        ("RELIANCE.NS", "Reliance Industries"),
        ("TCS.NS", "TCS"),
        ("INFY.NS", "Infosys"),
    ]

    for col, (symbol, name) in zip(cols, examples):
        with col:
            st.markdown(f"**{name}**")
            st.code(symbol)

    st.stop()


# ---------------------------------------------------------
# LOAD RESULTS
# ---------------------------------------------------------

df = st.session_state["df"]
company = st.session_state["company"]
financials = st.session_state["financials"]
financial = st.session_state["financial"]
technical = st.session_state["technical"]
fundamental = st.session_state["fundamental"]
analyst = st.session_state["analyst"]
ticker = st.session_state["ticker"]


# ---------------------------------------------------------
# COMPANY HEADER
# ---------------------------------------------------------

st.subheader(
    f"{company.name or ticker} ({ticker})"
)

company_description = []

if company.sector:
    company_description.append(company.sector)

if company.industry:
    company_description.append(company.industry)

if company.country:
    company_description.append(company.country)

if company_description:
    st.caption(" • ".join(company_description))

if company.description:
    st.write(company.description)


# ---------------------------------------------------------
# TOP SCORE CARDS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Overall Assessment</div>',
    unsafe_allow_html=True,
)

score_cols = st.columns(4)

scores = [
    ("Fundamental", fundamental.score),
    ("Financial", financial.score),
    ("Technical", technical.score),
    ("Overall", analyst.overall_score),
]

for col, (label, score) in zip(score_cols, scores):

    with col:

        if score is not None:

            st.metric(
                label,
                f"{float(score):.1f}/10",
            )

        else:
            st.metric(label, "N/A")


# ---------------------------------------------------------
# FINAL VIEW
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🤖 AI Research View</div>',
    unsafe_allow_html=True,
)

st.info(analyst.final_view)


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

(
    overview_tab,
    fundamentals_tab,
    financials_tab,
    technicals_tab,
    ai_tab,
) = st.tabs(
    [
        "📊 Overview",
        "🏢 Fundamentals",
        "💰 Financials",
        "📈 Technicals",
        "🤖 AI Research",
    ]
)


# =========================================================
# OVERVIEW
# =========================================================

with overview_tab:

    st.subheader("Price Chart")

    chart_df = df.copy()

    chart_df["SMA50"] = (
        chart_df["close"]
        .rolling(50)
        .mean()
    )

    chart_df["SMA200"] = (
        chart_df["close"]
        .rolling(200)
        .mean()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="Price",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["SMA50"],
            name="SMA 50",
            mode="lines",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["SMA200"],
            name="SMA 200",
            mode="lines",
        )
    )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Key Technical Indicators")

    cols = st.columns(5)

    indicators_data = [
        ("Trend", technical.trend),
        ("RSI", technical.rsi),
        ("MACD", technical.macd),
        ("Above SMA 50", technical.above_50_sma),
        ("Above SMA 200", technical.above_200_sma),
    ]

    for col, (label, value) in zip(cols, indicators_data):

        with col:
            st.metric(
                label,
                "N/A" if value is None else str(value),
            )


# =========================================================
# FUNDAMENTALS
# =========================================================

with fundamentals_tab:

    st.subheader("Business Quality")

    fundamental_scores = pd.DataFrame(
        {
            "Category": [
                "Business Quality",
                "Growth",
                "Competitive Advantage",
                "Management",
                "Industry Position",
            ],
            "Score": [
                fundamental.business_quality,
                fundamental.growth,
                fundamental.competitive_advantage,
                fundamental.management,
                fundamental.industry_position,
            ],
        }
    )

    fig = go.Figure(
        go.Bar(
            x=fundamental_scores["Score"],
            y=fundamental_scores["Category"],
            orientation="h",
            text=fundamental_scores["Score"],
            textposition="outside",
        )
    )

    fig.update_layout(
        xaxis_title="Score",
        xaxis_range=[0, 10],
        height=350,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Summary")

    st.write(fundamental.summary)

    st.subheader("⚠️ Fundamental Risks")

    for risk in fundamental.risks:
        st.warning(risk)


# =========================================================
# FINANCIALS
# =========================================================

with financials_tab:

    st.subheader("Financial Metrics")

    metric_cols = st.columns(4)

    metrics = [
        ("Revenue Growth", financial.revenue_growth),
        ("Profit Growth", financial.profit_growth),
        ("EPS Growth", financial.eps_growth),
        ("FCF Growth", financial.fcf_growth),
        ("EBITDA Margin", financial.ebitda_margin),
        ("ROE", financial.roe),
        ("ROCE", financial.roce),
        ("Debt / Equity", financial.debt_equity),
    ]

    for i, (label, value) in enumerate(metrics):

        with metric_cols[i % 4]:

            if value is None:
                display_value = "N/A"
            elif label in [
                "Revenue Growth",
                "Profit Growth",
                "EPS Growth",
                "FCF Growth",
                "EBITDA Margin",
                "ROE",
                "ROCE",
            ]:
                display_value = f"{value}%"
            else:
                display_value = f"{value:.2f}"

            st.metric(
                label,
                display_value,
            )

    st.subheader("Valuation")

    valuation_cols = st.columns(4)

    valuation_metrics = [
    ("P/E", financial.pe),
    ("P/B", financial.pb),
    ("EV / EBITDA", financials.ebitda_ev),
    ("Dividend Yield", financial.dividend_yield),
    ]

    for col, (label, value) in zip(
        valuation_cols,
        valuation_metrics,
    ):

        with col:

            if value is None:
                value = "N/A"

            elif label == "Dividend Yield":
                value = f"{value}%"

            else:
                value = f"{value:.2f}"

            st.metric(
                label,
                value,
            )


# =========================================================
# TECHNICALS
# =========================================================

with technicals_tab:

    st.subheader("Technical Setup")

    technical_cols = st.columns(4)

    technical_metrics = [
        ("Trend", technical.trend),
        ("RSI", technical.rsi),
        ("MACD", technical.macd),
        ("ATR", technical.atr),
    ]

    for col, (label, value) in zip(
        technical_cols,
        technical_metrics,
    ):

        with col:
            st.metric(
                label,
                "N/A" if value is None else str(value),
            )

    st.subheader("Support & Resistance")

    sr_cols = st.columns(2)

    with sr_cols[0]:

        st.markdown("### 🟢 Support")

        if technical.support_resistance.support:

            for level in technical.support_resistance.support:
                st.write(f"₹ {level:,.2f}")

        else:
            st.write("No significant support levels detected.")

    with sr_cols[1]:

        st.markdown("### 🔴 Resistance")

        if technical.support_resistance.resistance:

            for level in technical.support_resistance.resistance:
                st.write(f"₹ {level:,.2f}")

        else:
            st.write("No significant resistance levels detected.")

    st.subheader("Candlestick Patterns")

    if technical.candlestick_patterns:

        for pattern in technical.candlestick_patterns:
            st.success(pattern.name)

    else:
        st.write("No major candlestick patterns detected.")

    st.subheader("Chart Patterns")

    if technical.chart_patterns:

        for pattern in technical.chart_patterns:
            st.success(pattern.name)

    else:
        st.write("No major chart patterns detected.")


# =========================================================
# AI RESEARCH
# =========================================================

with ai_tab:

    st.subheader("Investment Analyst Report")

    st.markdown("### Valuation Score")

    st.metric(
        "AI Valuation Score",
        f"{analyst.valuation_score:.1f}/10",
    )

    st.markdown("### Research Narrative")

    st.write(analyst.narrative)

    st.markdown("### Key Risks")

    for risk in analyst.risks:
        st.warning(risk)

    st.markdown("### Final View")

    st.info(analyst.final_view)


# ---------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------

st.divider()

st.caption(
    "⚠️ This application is for educational and research purposes only. "
    "It is not financial advice and does not constitute a recommendation "
    "to buy or sell any security."
)
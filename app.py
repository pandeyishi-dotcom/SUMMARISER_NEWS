# ============================================================
# INDIA MACRO & MARKETS MONITOR — FINAL STABLE BUILD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import feedparser
from urllib.parse import quote_plus

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="India Macro Pro", layout="wide", page_icon="📈")

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = []

if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark Neon"

# ------------------------------------------------------------
# THEMES
# ------------------------------------------------------------
THEMES = {
    "Dark Neon": {
        "bg": "#0e1117", "card": "#1a1d24", "text": "#ffffff",
        "accent": "#00d4ff", "template": "plotly_dark"
    },
    "Bloomberg Terminal": {
        "bg": "#000000", "card": "#111111", "text": "#ff9900",
        "accent": "#ff9900", "template": "plotly_dark"
    },
    "Solarized Light": {
        "bg": "#FDF6E3", "card": "#EEE8D5", "text": "#657B83",
        "accent": "#B58900", "template": "plotly_white"
    }
}

theme = THEMES[st.session_state["theme"]]

# ------------------------------------------------------------
# STYLE
# ------------------------------------------------------------
st.markdown(f"""
<style>
.stApp {{ background:{theme['bg']}; color:{theme['text']}; }}
.card {{
    background:{theme['card']};
    padding:16px;
    border-radius:12px;
    margin-bottom:14px;
}}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_market_regime(df):
    if df is None or df.empty or "Close" not in df:
        return "Data Unavailable"

    close = df["Close"].dropna()
    if len(close) < 200:
        return "Neutral (Insufficient Data)"

    returns = close.pct_change().dropna()
    if len(returns) < 20:
        return "Neutral (Low Liquidity)"

    vol = returns.rolling(20).std() * np.sqrt(252)
    vol_latest = vol.dropna().iloc[-1]
    vol_mean = vol.dropna().mean()

    ma50 = close.rolling(50).mean().dropna()
    ma200 = close.rolling(200).mean().dropna()

    if ma50.empty or ma200.empty:
        return "Neutral (Trend Forming)"

    trend = ma50.iloc[-1] - ma200.iloc[-1]

    if vol_latest > vol_mean * 1.2:
        return "High Volatility ⚠️"
    elif trend > 0:
        return "Risk-On 📈"
    else:
        return "Risk-Off 📉"

def monte_carlo(start, mu, sigma, days=30, sims=50):
    paths = pd.DataFrame()
    for i in range(sims):
        prices = [start]
        for _ in range(days):
            prices.append(prices[-1] * np.exp((mu - 0.5*sigma**2) + sigma*np.random.normal()))
        paths[i] = prices
    return paths

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab_news, tab_macro, tab_tech, tab_port, tab_sim = st.tabs(
    ["☕ Finshots", "📊 Macro", "💹 Technicals", "💼 Portfolio", "🎲 Monte Carlo"]
)

# ============================================================
# TAB 1 — FINSHOTS
# ============================================================
with tab_news:
    st.subheader("Daily Market Brief ☕")
    query = st.text_input("Search News", "India stock market")

    safe_query = quote_plus(query)
    rss_url = f"https://news.google.com/rss/search?q={safe_query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries:
            st.warning("No news available.")
        else:
            for entry in feed.entries[:5]:
                st.markdown(f"""
                <div class="card">
                    <b>{entry.title}</b><br>
                    <small>{entry.get("published","")}</small><br><br>
                    Markets react to liquidity, earnings, and policy signals.<br>
                    <a href="{entry.link}" target="_blank">Read full article →</a>
                </div>
                """, unsafe_allow_html=True)
    except Exception:
        st.error("News feed unavailable.")

# ============================================================
# TAB 2 — MACRO
# ============================================================
with tab_macro:
    nifty = yf.download("^NSEI", period="1y", progress=False)
    regime = detect_market_regime(nifty)
    c1, c2, c3 = st.columns(3)
    c1.metric("Market Regime", regime)
    c2.metric("India 10Y Yield", "7.1%")
    c3.metric("USD/INR", "83.4")
    st.success("Macro → Sector View: High rates favor Banks | Pressure on Real Estate")

# ============================================================
# TAB 3 — TECHNICALS
# ============================================================
with tab_tech:
    symbol = st.text_input("Stock Symbol", "RELIANCE.NS")
    df = yf.download(symbol, period="1y", progress=False)

    if not df.empty:
        df["RSI"] = calculate_rsi(df["Close"])
        df["SMA"] = df["Close"].rolling(20).mean()
        df["Upper"] = df["SMA"] + 2 * df["Close"].rolling(20).std()
        df["Lower"] = df["SMA"] - 2 * df["Close"].rolling(20).std()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=df.index, y=df["Upper"], name="Upper BB", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["Lower"], name="Lower BB", line=dict(dash="dot")))
        fig.update_layout(template=theme["template"], height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")

# ============================================================
# TAB 4 — PORTFOLIO
# ============================================================
with tab_port:
    with st.form("add_stock"):
        s, q, a, sec = st.columns(4)
        sym = s.text_input("Symbol")
        qty = q.number_input("Qty", 1)
        avg = a.number_input("Avg Price", 1.0)
        sector = sec.selectbox("Sector", ["Banking","IT","Energy","FMCG","Auto","Other"])
        if st.form_submit_button("Add"):
            st.session_state["portfolio"].append(
                {"symbol": sym, "qty": qty, "avg": avg, "sector": sector}
            )

    if st.session_state["portfolio"]:
        pf = pd.DataFrame(st.session_state["portfolio"])
        prices = yf.download(pf["symbol"].tolist(), period="1d", progress=False)["Close"]

        pf["LTP"] = pf["symbol"].apply(lambda x: float(prices[x].iloc[-1]))
        pf["Value"] = pf["LTP"] * pf["qty"]
        pf["Invested"] = pf["avg"] * pf["qty"]
        pf["P/L"] = pf["Value"] - pf["Invested"]

        st.metric("Total P/L", f"₹{pf['P/L'].sum():,.0f}")
        st.dataframe(pf, use_container_width=True)

        fig = px.pie(pf, values="Value", names="sector", template=theme["template"])
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 5 — MONTE CARLO
# ============================================================
with tab_sim:
    if 'df' in locals() and not df.empty:
        if st.button("Run Simulation"):
            r = np.log(df["Close"]/df["Close"].shift(1)).dropna()
            sim = monte_carlo(df["Close"].iloc[-1], r.mean(), r.std())
            fig = px.line(sim, template=theme["template"])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.header("🤖 Analyst Bot")
    q = st.text_input("Ask a question")
    if q:
        if "rsi" in q.lower():
            st.info("RSI >70 Overbought | <30 Oversold")
        else:
            st.info("Ask about RSI, trends, macro, or risk.")
    st.selectbox("Theme", THEMES.keys(), key="theme")

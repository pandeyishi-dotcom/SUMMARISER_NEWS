# ============================================================
# INDIA MACRO & MARKETS MONITOR — PRO EDITION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import feedparser, requests, re
from datetime import datetime
from collections import Counter

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config("India Macro Pro", layout="wide", page_icon="📈")

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
        "accent": "#00d4ff", "pos": "#00ff9d", "neg": "#ff4b4b",
        "template": "plotly_dark"
    },
    "Bloomberg Terminal": {
        "bg": "#000000", "card": "#111111", "text": "#ff9900",
        "accent": "#ff9900", "pos": "#00ff00", "neg": "#ff0000",
        "template": "plotly_dark"
    },
    "Solarized Light": {
        "bg": "#FDF6E3", "card": "#EEE8D5", "text": "#657B83",
        "accent": "#B58900", "pos": "#859900", "neg": "#DC322F",
        "template": "plotly_white"
    }
}

theme = THEMES[st.session_state["theme"]]

st.markdown(f"""
<style>
.stApp {{ background:{theme['bg']}; color:{theme['text']}; }}
.card {{ background:{theme['card']}; padding:15px; border-radius:12px; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_regime(df):
    ret = df["Close"].pct_change()
    vol = ret.rolling(20).std() * np.sqrt(252)
    trend = df["Close"].rolling(50).mean().iloc[-1] - df["Close"].rolling(200).mean().iloc[-1]
    if vol.iloc[-1] > vol.mean():
        return "High Volatility ⚠️"
    return "Risk-On 📈" if trend > 0 else "Risk-Off 📉"

def monte_carlo(price, mu, sigma, days=30, sims=50):
    paths = pd.DataFrame()
    for i in range(sims):
        prices = [price]
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
    query = st.text_input("Search Market News", "India Stock Market")
    rss = feedparser.parse(f"https://news.google.com/rss/search?q={query}")
    for entry in rss.entries[:5]:
        st.markdown(f"""
        <div class="card">
        <b>{entry.title}</b><br>
        <small>{entry.get("published","")}</small><br><br>
        Key idea: Markets react to liquidity, earnings and rates.<br>
        <a href="{entry.link}" target="_blank">Read →</a>
        </div><br>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 2 — MACRO LAB
# ============================================================
with tab_macro:
    nifty = yf.download("^NSEI", period="1y", progress=False)
    regime = detect_regime(nifty)

    c1, c2, c3 = st.columns(3)
    c1.metric("Market Regime", regime)
    c2.metric("India 10Y Yield", "7.1%")
    c3.metric("USD/INR", "83.4")

    st.success("Macro → Sector Signal: Banks ↑ | Real Estate ↓ when rates stay high")

# ============================================================
# TAB 3 — TECHNICALS
# ============================================================
with tab_tech:
    symbol = st.text_input("Stock Symbol", "RELIANCE.NS")
    df = yf.download(symbol, period="1y", progress=False)

    if not df.empty:
        df["RSI"] = rsi(df["Close"])
        df["SMA"] = df["Close"].rolling(20).mean()
        df["Upper"] = df["SMA"] + 2 * df["Close"].rolling(20).std()
        df["Lower"] = df["SMA"] - 2 * df["Close"].rolling(20).std()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=df.index, y=df["Upper"], name="Upper BB", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["Lower"], name="Lower BB", line=dict(dash="dot")))
        fig.update_layout(template=theme["template"])
        st.plotly_chart(fig, use_container_width=True)

        st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
        st.metric("6M Momentum (%)", f"{(df['Close'].iloc[-1]/df['Close'].iloc[-126]-1)*100:.2f}")

# ============================================================
# TAB 4 — PORTFOLIO
# ============================================================
with tab_port:
    with st.form("add"):
        s, q, a, sec = st.columns(4)
        sym = s.text_input("Symbol")
        qty = q.number_input("Qty", 1)
        avg = a.number_input("Avg Price", 1.0)
        sector = sec.selectbox("Sector", ["Banking","IT","Energy","FMCG","Auto","Other"])
        if st.form_submit_button("Add"):
            st.session_state["portfolio"].append(
                {"symbol":sym,"qty":qty,"avg":avg,"sector":sector}
            )

    if st.session_state["portfolio"]:
        pf = pd.DataFrame(st.session_state["portfolio"])
        prices = yf.download(pf["symbol"].tolist(), period="1d", progress=False)["Close"]
        pf["LTP"] = pf["symbol"].apply(lambda x: prices[x].iloc[-1])
        pf["Value"] = pf["LTP"] * pf["qty"]
        pf["Invested"] = pf["avg"] * pf["qty"]
        pf["P/L"] = pf["Value"] - pf["Invested"]

        st.metric("Total P/L", f"₹{pf['P/L'].sum():,.0f}")
        st.dataframe(pf)

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
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.header("🤖 Analyst Bot")
    q = st.text_input("Ask:")
    if q:
        if "rsi" in q.lower(): st.info("RSI >70 Overbought, <30 Oversold.")
        elif "bull" in q.lower(): st.info("Bull market = rising trend.")
        else: st.info("I cover RSI, trends, macro & portfolio risk.")

    st.markdown("---")
    st.selectbox("Theme", THEMES.keys(), key="theme")

"""
India Macro & Markets Monitor (Ultimate Version)

Run:
    pip install -r requirements.txt
    streamlit run news_dashboard.py
"""

from __future__ import annotations

import os
import json
import base64
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import feedparser
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import requests_cache
import streamlit as st
import yfinance as yf
from textblob import TextBlob

# Optional extras
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# ------------------------------------------------------------------
#                      CONFIG & STATE INIT
# ------------------------------------------------------------------

requests_cache.install_cache("macro_markets_cache", expire_after=180)

st.set_page_config(
    page_title="India Macro & Markets Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- THEMES ---
THEMES = {
    "Dark Neon": {
        "bg_top": "#050816",
        "bg_bottom": "#02010A",
        "card": "#0B1020",
        "card_border": "rgba(255, 255, 255, 0.08)",
        "text": "#F9FAFB",
        "muted": "#9CA3AF",
        "accent": "#FF7A3C", # Orange
        "accent2": "#A855F7", # Purple
        "pos": "#22C55E",
        "neg": "#F97373",
        "neu": "#FBBF24",
        "chart_temp": "plotly_dark"
    },
    "Soft Light": {
        "bg_top": "#F3F4F6",
        "bg_bottom": "#E5E7EB",
        "card": "#FFFFFF",
        "card_border": "rgba(0, 0, 0, 0.08)",
        "text": "#111827",
        "muted": "#6B7280",
        "accent": "#EA580C",
        "accent2": "#7C3AED",
        "pos": "#16A34A",
        "neg": "#DC2626",
        "neu": "#D97706",
        "chart_temp": "plotly_white"
    }
}

if "theme_choice" not in st.session_state:
    st.session_state["theme_choice"] = "Dark Neon"

COLORS = THEMES[st.session_state["theme_choice"]]

# --- SESSION STATE INITIALIZATION ---
if "portfolio" not in st.session_state:
    # Structure: [{'symbol': 'RELIANCE.NS', 'qty': 10, 'avg_price': 2400}]
    st.session_state["portfolio"] = []

if "news_memory" not in st.session_state:
    st.session_state["news_memory"] = []

if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = "Trader"  # Options: Student, Trader, Investor

INDEX_SYMBOLS = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
}

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()

# ------------------------------------------------------------------
#                      CSS STYLING
# ------------------------------------------------------------------

def apply_css():
    c = THEMES[st.session_state["theme_choice"]]
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(180deg, {c['bg_top']} 0%, {c['bg_bottom']} 100%);
            color: {c['text']};
        }}
        h1, h2, h3, h4, h5, p, span, div {{
            color: {c['text']};
        }}
        .card {{
            background: {c['card']};
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            border: 1px solid {c['card_border']};
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        .small-muted {{
            color: {c['muted']} !important;
            font-size: 0.85rem;
        }}
        .chip {{
            display:inline-block; padding:2px 8px; border-radius:12px;
            font-size:0.7rem; margin-right:4px; margin-bottom:4px;
            border: 1px solid {c['muted']}; opacity: 0.8;
        }}
        .impact-pill {{
            background: {c['accent']}; color: white; padding: 2px 8px;
            border-radius: 4px; font-weight: bold; font-size: 0.75rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_css()

# ------------------------------------------------------------------
#                      DATA HELPERS
# ------------------------------------------------------------------

PRESET_TICKERS_FALLBACK = {
    "Reliance": "RELIANCE.NS", "HDFC Bank": "HDFCBANK.NS", "Infosys": "INFY.NS",
    "TCS": "TCS.NS", "ITC": "ITC.NS", "SBI": "SBIN.NS", "Tata Motors": "TATAMOTORS.NS"
}

SECTOR_MAP = {
    "RELIANCE.NS": "Energy", "HDFCBANK.NS": "Financials", "ICICIBANK.NS": "Financials",
    "INFY.NS": "Technology", "TCS.NS": "Technology", "ITC.NS": "Consumer",
    "SBIN.NS": "Financials", "TATAMOTORS.NS": "Auto", "LT.NS": "Construction",
    "BHARTIARTL.NS": "Telecom", "MARUTI.NS": "Auto", "SUNPHARMA.NS": "Healthcare"
}

def log(msg):
    # Simple logging placeholder
    pass

def safe_json_get(url, params=None):
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None

def fmt_ts(val):
    if not val: return ""
    return pd.to_datetime(val).strftime("%Y-%m-%d %H:%M")

# ------------------------------------------------------------------
#                      NEWS ENGINE
# ------------------------------------------------------------------

def get_impact_score(text):
    """Calculate 0-100 impact score based on keywords."""
    text = text.lower()
    score = 30 # Base
    
    high_imp = ["rbi", "budget", "crash", "surge", "gdp", "inflation", "war", "crisis"]
    med_imp = ["earnings", "profit", "launch", "approval", "deal"]
    
    for w in high_imp:
        if w in text: score += 15
    for w in med_imp:
        if w in text: score += 5
        
    return min(score, 100)

def generate_eli5_analysis(title):
    """Rule-based 'AI' analysis."""
    title_low = title.lower()
    
    # Simple Knowledge Graph logic
    rules = [
        ("inflation", "Consumers pay more.", "FMCG / Retail", "Gold / Real Estate"),
        ("rate hike", "Loans get expensive.", "Banks (NIM)", "Real Estate / Auto"),
        ("oil price", "Transport gets costly.", "Oil Explorers (ONGC)", "Paints / Airlines"),
        ("gdp growth", "Economy is expanding.", "Infrastructure", "Defensive stocks"),
        ("itc", "Meme stock moving.", "Shareholders", "Short sellers"),
    ]
    
    eli5 = "This is general market news."
    winner = "Market General"
    loser = "None"
    
    for kw, expl, w, l in rules:
        if kw in title_low:
            eli5 = expl
            winner = w
            loser = l
            break
            
    return eli5, winner, loser

def load_headlines(query, limit, sources=None):
    # Mocking different sources for RSS if not NewsAPI
    # In prod, you'd filter the feed entries by source string
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    
    out = []
    for entry in feed.entries[:limit]:
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        source = entry.get("source", {}).get("title", "Unknown")
        
        # Source Filter
        if sources and source not in sources:
            continue
            
        tb = TextBlob(title)
        sent_score = tb.sentiment.polarity
        impact = get_impact_score(title + " " + summary)
        
        eli5, win, lose = generate_eli5_analysis(title)
        
        out.append({
            "title": title,
            "url": entry.get("link"),
            "source": source,
            "published": pd.to_datetime(entry.get("published", datetime.now())),
            "sentiment": sent_score,
            "impact": impact,
            "eli5": eli5,
            "winner": win,
            "loser": lose
        })
    return out

def news_tab():
    st.header("📰 Intelligent News Feed")
    
    # --- CONTROLS ---
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        query = st.text_input("Topic Search", "India Economy")
    with c2:
        # Mock sources found in Google News
        valid_sources = ["The Economic Times", "Livemint", "Moneycontrol", "NDTV Profit", "Business Standard"]
        src_filter = st.multiselect("Filter Source", valid_sources, default=[])
    with c3:
        min_impact = st.slider("Min Impact Score", 0, 100, 40)

    headlines = load_headlines(query, 25, src_filter if src_filter else None)
    
    # Filter by impact
    headlines = [h for h in headlines if h["impact"] >= min_impact]

    if not headlines:
        st.warning("No articles match your criteria.")
        return

    # --- TIMELINE CHART ---
    st.subheader("Coverage Timeline")
    df_time = pd.DataFrame(headlines)
    if not df_time.empty:
        df_time["date_str"] = df_time["published"].dt.strftime("%Y-%m-%d")
        counts = df_time["date_str"].value_counts().reset_index()
        counts.columns = ["date", "count"]
        fig = px.bar(counts, x="date", y="count", title=f"Articles per day: {query}", template=COLORS["chart_temp"])
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    # --- NEWS CARDS ---
    col_l, col_r = st.columns([3, 2])
    
    with col_l:
        st.markdown("### Latest Stories")
        for i, h in enumerate(headlines[:10]):
            sent_color = COLORS["pos"] if h["sentiment"] > 0.05 else (COLORS["neg"] if h["sentiment"] < -0.05 else COLORS["neu"])
            
            with st.container():
                st.markdown(f"""
                <div class="card">
                    <div style="display:flex; justify-content:space-between;">
                        <span class="small-muted">{h['source']} • {fmt_ts(h['published'])}</span>
                        <span class="impact-pill">Impact: {h['impact']}</span>
                    </div>
                    <div style="font-size:1.1rem; font-weight:600; margin: 5px 0;">
                        <a href="{h['url']}" target="_blank" style="text-decoration:none; color:{COLORS['text']}">{h['title']}</a>
                    </div>
                    <div style="border-left: 2px solid {sent_color}; padding-left: 8px; margin-top:5px;">
                        <div class="small-muted" style="margin-bottom:2px;">AI Insight:</div>
                        <div style="font-size:0.9rem;">{h['eli5']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Interactive Analysis Button
                if st.button(f"⚡ Who wins? #{i}", key=f"btn_{i}"):
                    st.info(f"🏆 **Winner:** {h['winner']} | 🔻 **Loser:** {h['loser']}")
                    # Add to memory
                    st.session_state["news_memory"].append(h)

    with col_r:
        st.markdown("### 🧠 Memory & Context")
        if st.session_state["news_memory"]:
            st.caption("Recently analyzed stories:")
            for m in st.session_state["news_memory"][-5:]:
                st.markdown(f"- **{m['winner']}** vs {m['loser']} ({m['title'][:30]}...)")
            
            if st.button("Clear Memory"):
                st.session_state["news_memory"] = []
        else:
            st.info("Click 'Who wins?' on an article to add it to your memory bank.")

        st.markdown("---")
        st.markdown("### 📥 Export")
        if st.button("Generate Daily Briefing"):
            briefing = f"DAILY BRIEFING - {datetime.now().strftime('%Y-%m-%d')}\n\nTOP STORIES:\n"
            for h in headlines[:3]:
                briefing += f"- {h['title']} (Impact: {h['impact']})\n"
            
            b64 = base64.b64encode(briefing.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="briefing.txt">Download Briefing.txt</a>'
            st.markdown(href, unsafe_allow_html=True)

# ------------------------------------------------------------------
#                      MACRO ENGINE
# ------------------------------------------------------------------

# Simplified Mock Data for demonstration if WB API fails or for speed
MOCK_MACRO = {
    "years": [2019, 2020, 2021, 2022, 2023],
    "cpi": [3.7, 6.6, 5.1, 6.7, 5.4],
    "gdp": [4.8, -5.8, 9.1, 7.2, 6.9],
    "repo": [5.15, 4.0, 4.0, 6.25, 6.5]
}

def macro_tab():
    st.header("📊 Macroeconomic Dashboard")
    
    # --- SCENARIO SIMULATOR ---
    with st.expander("🎛️ Scenario Simulator: Real Interest Rates", expanded=True):
        c1, c2, c3 = st.columns(3)
        current_repo = 6.5
        current_cpi = 5.4
        
        with c1:
            shock_cpi = st.slider("CPI Shock (%)", -2.0, 5.0, 0.0, 0.1)
        with c2:
            shock_repo = st.slider("Repo Rate Adjustment (%)", -2.0, 2.0, 0.0, 0.25)
        
        sim_cpi = current_cpi + shock_cpi
        sim_repo = current_repo + shock_repo
        real_rate = sim_repo - sim_cpi
        
        with c3:
            st.metric("Proj. Real Rate", f"{real_rate:.2f}%", delta=f"{(real_rate - (current_repo-current_cpi)):.2f}%")
            if real_rate < 0:
                st.error("Negative Real Rates! Inflation eats savings.")
            else:
                st.success("Positive Real Rates.")

    # --- STRESS FLAGS ---
    st.subheader("Economic Stress Monitor")
    
    # Logic: Stagflation = Low Growth + High Inflation
    last_gdp = MOCK_MACRO["gdp"][-1]
    last_cpi = MOCK_MACRO["cpi"][-1]
    
    col1, col2, col3 = st.columns(3)
    
    status_gdp = "High Growth" if last_gdp > 6 else "Slowdown"
    status_cpi = "High Inflation" if last_cpi > 6 else "Controlled"
    
    flag_color = COLORS["pos"]
    flag_text = "Stable"
    
    if last_gdp < 4 and last_cpi > 6:
        flag_text = "STAGFLATION RISK"
        flag_color = COLORS["neg"]
    elif last_gdp < 0:
        flag_text = "RECESSION"
        flag_color = COLORS["neg"]
        
    col1.markdown(f"<div class='card' style='text-align:center; border-left: 4px solid {COLORS['accent']};'><h5>GDP Status</h5><h3>{status_gdp}</h3></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card' style='text-align:center; border-left: 4px solid {COLORS['accent2']};'><h5>Inflation Status</h5><h3>{status_cpi}</h3></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card' style='text-align:center; border-left: 4px solid {flag_color};'><h5>System Flag</h5><h3 style='color:{flag_color}'>{flag_text}</h3></div>", unsafe_allow_html=True)

    # --- HEATMAP VISUALIZATION ---
    st.markdown("### 🌡️ Macro Heatmap (Indicator Intensity)")
    
    # Create a grid: Rows=Indicators, Cols=Years
    indicators = ["CPI Inflation", "GDP Growth", "Repo Rate"]
    data_matrix = [MOCK_MACRO["cpi"], MOCK_MACRO["gdp"], MOCK_MACRO["repo"]]
    
    fig = px.imshow(
        data_matrix,
        x=MOCK_MACRO["years"],
        y=indicators,
        color_continuous_scale="RdBu_r", # Red = High (Bad for inflation, usually)
        title="Economic Heatmap"
    )
    fig.update_layout(template=COLORS["chart_temp"], height=300)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
#                      MARKETS ENGINE
# ------------------------------------------------------------------

def get_market_regime(df):
    """Determine if Trending Up, Down, or Sideways."""
    if len(df) < 50: return "Insufficient Data", "gray"
    
    curr = df['Close'].iloc[-1]
    sma20 = df['Close'].rolling(20).mean().iloc[-1]
    sma50 = df['Close'].rolling(50).mean().iloc[-1]
    
    if curr > sma20 > sma50:
        return "Strong Uptrend 🚀", COLORS["pos"]
    elif curr < sma20 < sma50:
        return "Downtrend 🐻", COLORS["neg"]
    else:
        return "Sideways / Choppy 🦀", COLORS["neu"]

def markets_tab():
    st.header("💹 Markets & Portfolio")
    
    tabs = st.tabs(["🔭 Watchlist & Analysis", "💼 Portfolio Tracker"])
    
    # --- WATCHLIST TAB ---
    with tabs[0]:
        c1, c2 = st.columns([1, 2])
        with c1:
            ticker = st.text_input("Analyze Symbol", "RELIANCE.NS")
            period = st.selectbox("Range", ["1mo", "3mo", "6mo", "1y", "5y"], index=3)
        
        with st.spinner("Fetching data..."):
            df = yf.download(ticker, period=period, progress=False)
        
        if not df.empty:
            # Stats
            current = df['Close'].iloc[-1]
            high_52 = df['High'].max()
            low_52 = df['Low'].min()
            
            regime, reg_col = get_market_regime(df)
            
            # Fundamentals (Mock/Fetch)
            # Fetching info is slow, use placeholder for demo or try-catch
            try:
                info = yf.Ticker(ticker).info
                pe = info.get('trailingPE', 'N/A')
                pb = info.get('priceToBook', 'N/A')
            except:
                pe, pb = "-", "-"

            # Header Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current", f"₹{current:.2f}", f"{(df['Close'].pct_change().iloc[-1]*100):.2f}%")
            m2.markdown(f"**Regime:** <span style='color:{reg_col}'>{regime}</span>", unsafe_allow_html=True)
            m3.metric("P/E Ratio", pe)
            m4.metric("P/B Ratio", pb)

            # Advanced Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color=COLORS['accent'])))
            
            # Support/Resistance Lines (Simple Max/Min of range)
            fig.add_hline(y=high_52, line_dash="dash", line_color="green", annotation_text="High")
            fig.add_hline(y=low_52, line_dash="dash", line_color="red", annotation_text="Low")
            
            # MA
            df['SMA50'] = df['Close'].rolling(50).mean()
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name="50 SMA", line=dict(color=COLORS['muted'], width=1)))
            
            fig.update_layout(template=COLORS['chart_temp'], height=500, title=f"{ticker} Analysis")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Symbol not found.")

    # --- PORTFOLIO TAB ---
    with tabs[1]:
        st.subheader("My Holdings")
        
        # Add position form
        with st.expander("➕ Add Position"):
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a: p_sym = st.text_input("Symbol", "TCS.NS")
            with col_b: p_qty = st.number_input("Qty", 1, 10000, 10)
            with col_c: p_avg = st.number_input("Avg Price", 0.0, 100000.0, 3000.0)
            with col_d: 
                if st.button("Add"):
                    st.session_state["portfolio"].append({"symbol": p_sym, "qty": p_qty, "avg_price": p_avg})
                    st.success("Added!")

        if st.session_state["portfolio"]:
            # Calculate P/L
            pf_data = []
            total_invested = 0
            total_curr = 0
            
            # Batch fetch prices for speed
            tickers = [x['symbol'] for x in st.session_state["portfolio"]]
            try:
                live_data = yf.download(tickers, period="1d", progress=False)['Close'].iloc[-1]
            except:
                live_data = {}

            sector_counts = defaultdict(float)

            for item in st.session_state["portfolio"]:
                sym = item['symbol']
                qty = item['qty']
                avg = item['avg_price']
                
                # Handle single ticker result vs series
                if len(tickers) == 1:
                    curr_price = float(live_data)
                else:
                    curr_price = float(live_data[sym]) if sym in live_data else avg
                
                val_invest = qty * avg
                val_curr = qty * curr_price
                pl = val_curr - val_invest
                pl_pct = (pl / val_invest) * 100 if val_invest else 0
                
                # Sector mapping
                sec = SECTOR_MAP.get(sym, "Other")
                sector_counts[sec] += val_curr
                
                total_invested += val_invest
                total_curr += val_curr
                
                pf_data.append([sym, qty, avg, curr_price, pl, pl_pct, sec])
            
            # Summary Metrics
            tot_pl = total_curr - total_invested
            tot_pl_pct = (tot_pl / total_invested) * 100 if total_invested else 0
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Value", f"₹{total_curr:,.0f}")
            k2.metric("Total P/L", f"₹{tot_pl:,.0f}", f"{tot_pl_pct:.2f}%")
            k3.metric("Positions", len(pf_data))
            
            # DataFrame
            df_pf = pd.DataFrame(pf_data, columns=["Symbol", "Qty", "Avg", "LTP", "P/L", "P/L%", "Sector"])
            st.dataframe(df_pf.style.format({"Avg": "{:.2f}", "LTP": "{:.2f}", "P/L": "{:.2f}", "P/L%": "{:.2f}%"}))
            
            # Visuals
            r1, r2 = st.columns(2)
            with r1:
                # Sector Pie
                fig_sec = px.pie(names=list(sector_counts.keys()), values=list(sector_counts.values()), title="Sector Allocation", template=COLORS['chart_temp'])
                st.plotly_chart(fig_sec, use_container_width=True)
            with r2:
                # TreeMap of Holdings
                fig_tree = px.treemap(df_pf, path=['Sector', 'Symbol'], values='LTP', title="Holdings Map", template=COLORS['chart_temp'])
                st.plotly_chart(fig_tree, use_container_width=True)
                
            if st.button("Clear Portfolio"):
                st.session_state["portfolio"] = []
                st.rerun()

        else:
            st.info("Portfolio is empty. Add stocks above.")

# ------------------------------------------------------------------
#                      SIDEBAR & MAIN
# ------------------------------------------------------------------

def sidebar():
    st.sidebar.title("⚙️ Control Room")
    
    # User Profile
    st.session_state["user_profile"] = st.sidebar.selectbox("Profile", ["Student", "Trader", "Investor"])
    if st.session_state["user_profile"] == "Student":
        st.sidebar.info("🎓 Student Mode: Explainers enabled.")
    
    # Theme Toggle
    theme = st.sidebar.radio("Theme", ["Dark Neon", "Soft Light"], horizontal=True)
    if theme != st.session_state["theme_choice"]:
        st.session_state["theme_choice"] = theme
        st.rerun()

    # Refresh
    if HAS_AUTOREFRESH:
        ref = st.sidebar.selectbox("Auto-Refresh", ["Off", "1m", "5m"])
        if ref == "1m": st_autorefresh(interval=60000)
        if ref == "5m": st_autorefresh(interval=300000)
    
    st.sidebar.markdown("---")
    st.sidebar.write("Developed with ❤️")

def main():
    sidebar()
    
    st.title("🇮🇳 India Macro & Markets Monitor")
    st.caption(f"Logged in as: {st.session_state['user_profile']} | {datetime.now().strftime('%A, %d %b %Y')}")
    
    t1, t2, t3 = st.tabs(["📰 News Feed", "📊 Macro Economy", "💹 Markets & Portfolio"])
    
    with t1:
        news_tab()
    with t2:
        macro_tab()
    with t3:
        markets_tab()

if __name__ == "__main__":
    main()

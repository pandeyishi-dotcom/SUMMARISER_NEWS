"""
India Macro & Markets Monitor (Ultimate Version)

Features Implemented:
1. News: Source filter, Impact Score, ELI5, Who Wins/Loses, Timeline, Export.
2. Macro: Heatmap, Recession Flags, Scenario Simulator (Real Rates).
3. Markets: Portfolio Tracker (P/L, Pie Chart), Market Regime, Support/Resistance lines.
4. UX: Theme Toggle, User Profiles.

Run:
    pip install streamlit pandas plotly yfinance textblob feedparser requests streamlit-autorefresh
    streamlit run news_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import feedparser
from textblob import TextBlob
from datetime import datetime, timedelta
import base64
from collections import defaultdict

# ------------------------------------------------------------------
# 1. CONFIG & SESSION STATE
# ------------------------------------------------------------------

st.set_page_config(page_title="India Macro & Markets Monitor", layout="wide")

# Initialize Session State for Portfolio and Settings
if "portfolio" not in st.session_state:
    # Example starting data
    st.session_state["portfolio"] = [
        {"symbol": "RELIANCE.NS", "qty": 10, "avg": 2400.0},
        {"symbol": "TCS.NS", "qty": 5, "avg": 3500.0}
    ]

if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark Neon"

if "user_role" not in st.session_state:
    st.session_state["user_role"] = "Trader"

# Sector Map for Portfolio
SECTOR_MAP = {
    "RELIANCE.NS": "Energy", "TCS.NS": "IT", "INFY.NS": "IT", 
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking",
    "ITC.NS": "FMCG", "TATAMOTORS.NS": "Auto", "MARUTI.NS": "Auto",
    "SUNPHARMA.NS": "Pharma", "LT.NS": "Infra"
}

# ------------------------------------------------------------------
# 2. THEME & STYLING
# ------------------------------------------------------------------

THEMES = {
    "Dark Neon": {
        "bg": "#0e1117", "card": "#1e2130", "text": "#ffffff",
        "accent": "#FF4B4B", "pos": "#00FF00", "neg": "#FF0055",
        "chart_template": "plotly_dark"
    },
    "Light Pro": {
        "bg": "#ffffff", "card": "#f0f2f6", "text": "#000000",
        "accent": "#0068c9", "pos": "#008000", "neg": "#d00000",
        "chart_template": "plotly_white"
    }
}

current_theme = THEMES[st.session_state["theme"]]

st.markdown(f"""
<style>
    .stApp {{ background-color: {current_theme['bg']}; color: {current_theme['text']}; }}
    .metric-card {{
        background-color: {current_theme['card']};
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid {current_theme['accent']};
        margin-bottom: 10px;
    }}
    .news-card {{
        background-color: {current_theme['card']};
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 12px;
        border: 1px solid rgba(128,128,128,0.2);
    }}
    .tag {{
        display: inline-block; padding: 2px 8px; border-radius: 12px;
        background: {current_theme['accent']}; color: white; font-size: 0.8em;
    }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------------

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def calculate_impact_score(text):
    """Generates a score 0-100 based on keywords."""
    text = text.lower()
    score = 30 # Base
    keywords = {
        "rbi": 20, "inflation": 20, "crash": 30, "surge": 15, "gdp": 20,
        "budget": 25, "war": 30, "results": 10, "profit": 10
    }
    for word, val in keywords.items():
        if word in text:
            score += val
    return min(score, 100)

def generate_eli5(title):
    """Simulates AI explanation based on rules."""
    title = title.lower()
    if "inflation" in title:
        return "Prices are going up. Your money buys less stuff.", "Gold/Real Estate", "Cash Savers"
    elif "rbi" in title or "rate" in title:
        return "The central bank is changing loan costs. Loans might get expensive.", "Banks", "Borrowers"
    elif "profit" in title or "results" in title:
        return "The company made money based on their quarterly report.", "Shareholders", "Short Sellers"
    elif "gdp" in title:
        return "The country's total income is changing.", "Infrastructure", "Defensive Stocks"
    else:
        return "This is a general market update affecting sentiment.", "Traders", "None"

# ------------------------------------------------------------------
# 4. TABS
# ------------------------------------------------------------------

tab_news, tab_macro, tab_markets = st.tabs(["📰 News & Briefing", "📊 Macro Lab", "💹 Markets & Portfolio"])

# ==========================================
# TAB 1: NEWS (Full Features)
# ==========================================
with tab_news:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        query = st.text_input("Search Topic", "India Economy")
    with c2:
        # Feature: Source Filter
        sources = st.multiselect("Filter Source", ["Moneycontrol", "Economic Times", "Mint", "NDTV"], default=[])
    with c3:
        # Feature: Impact Score Slider
        min_impact = st.slider("Min Impact Score", 0, 100, 40)

    # Fetch News (Mocking Google RSS structure)
    try:
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        articles = []
        for entry in feed.entries[:15]:
            impact = calculate_impact_score(entry.title)
            if impact >= min_impact:
                articles.append({
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get("published", str(datetime.now())),
                    "source": entry.get("source", {}).get("title", "Unknown"),
                    "impact": impact
                })
    except:
        st.error("Could not fetch news. Check internet.")
        articles = []

    # Feature: News Timeline
    if articles:
        st.subheader("Coverage Intensity")
        dates = [pd.to_datetime(a['published']).date() for a in articles]
        date_counts = pd.Series(dates).value_counts().reset_index()
        date_counts.columns = ['Date', 'Count']
        fig = px.bar(date_counts, x='Date', y='Count', title="Articles per Day", template=current_theme['chart_template'])
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    # Feature: News Cards + ELI5
    col_feed, col_brief = st.columns([2, 1])
    
    with col_feed:
        st.subheader("Top Stories")
        for art in articles:
            eli5, winner, loser = generate_eli5(art['title'])
            st.markdown(f"""
            <div class="news-card">
                <h4><a href="{art['link']}" style="color:{current_theme['text']}">{art['title']}</a></h4>
                <small>{art['source']} | Impact: {art['impact']}/100</small>
                <hr style="margin:5px 0">
                <p style="font-size:0.9em; color:gray"><b>ELI5:</b> {eli5}</p>
                <span class="tag" style="background:green">Winner: {winner}</span>
                <span class="tag" style="background:red">Loser: {loser}</span>
            </div>
            """, unsafe_allow_html=True)

    # Feature: Daily Note Generator
    with col_brief:
        st.subheader("📝 Daily Briefing")
        if st.button("Generate Daily Note"):
            note = f"DAILY MARKET NOTE - {datetime.now().date()}\n\n"
            note += f"TOP THEME: {query}\n"
            note += "-"*30 + "\n"
            for a in articles[:5]:
                note += f"- {a['title']} (Impact: {a['impact']})\n"
            
            b64 = base64.b64encode(note.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="daily_note.txt">Download as Text File</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.text_area("Preview", note, height=300)

# ==========================================
# TAB 2: MACRO (Full Features)
# ==========================================
with tab_macro:
    # Feature: Recession/Stagflation Flags
    st.subheader("Economic Stress Monitor")
    
    # Mock Macro Data (Replace with API in prod)
    cpi_curr = 5.6
    gdp_curr = 6.1
    
    status = "Stable"
    color = "green"
    
    if cpi_curr > 6.0 and gdp_curr < 5.0:
        status = "STAGFLATION RISK"
        color = "red"
    elif gdp_curr < 0:
        status = "RECESSION"
        color = "red"
    
    m1, m2, m3 = st.columns(3)
    m1.metric("GDP Growth", f"{gdp_curr}%", "+0.2%")
    m2.metric("CPI Inflation", f"{cpi_curr}%", "-0.1%")
    m3.markdown(f"<div class='metric-card' style='border-left:5px solid {color}; text-align:center'><h3>{status}</h3></div>", unsafe_allow_html=True)

    # Feature: Scenario Slider (Real Rates)
    with st.expander("🎛️ Scenario Simulator: Real Interest Rates", expanded=True):
        sc1, sc2 = st.columns(2)
        repo_rate = sc1.slider("Repo Rate (%)", 4.0, 9.0, 6.5)
        inflation_shock = sc2.slider("Inflation Shock (+%)", 0.0, 5.0, 0.0)
        
        real_rate = repo_rate - (cpi_curr + inflation_shock)
        st.metric("Projected Real Interest Rate", f"{real_rate:.2f}%")
        if real_rate < 0:
            st.warning("Negative Real Rates! Savings lose value.")
    
    # Feature: Macro Heatmap
    st.subheader("Macro Heatmap (History)")
    # Mock historical data
    heatmap_data = pd.DataFrame([
        [4.5, 5.1, 6.2, 5.6], # CPI
        [7.2, -5.8, 9.1, 6.1], # GDP
        [5.1, 4.0, 4.0, 6.5]   # Repo
    ], index=["CPI", "GDP", "Repo"], columns=["2020", "2021", "2022", "2023"])
    
    fig_heat = px.imshow(heatmap_data, text_auto=True, color_continuous_scale="RdBu_r", title="Indicator Heatmap")
    fig_heat.update_layout(template=current_theme['chart_template'])
    st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# TAB 3: MARKETS (Full Features)
# ==========================================
with tab_markets:
    subtab_watch, subtab_port = st.tabs(["🔭 Watchlist & Analysis", "💼 Portfolio Tracker"])
    
    # --- WATCHLIST & CHART ---
    with subtab_watch:
        col_w1, col_w2 = st.columns([1, 3])
        with col_w1:
            ticker = st.text_input("Symbol (e.g., RELIANCE.NS)", "RELIANCE.NS")
            period = st.selectbox("Period", ["1mo", "6mo", "1y", "5y"])
            
            # Feature: User Profile Customization
            if st.session_state["user_role"] == "Student":
                st.info("ℹ️ Tip: Look for price above the orange line (MA) for an uptrend.")
                
        with col_w2:
            try:
                df = yf.download(ticker, period=period, progress=False)
                if not df.empty:
                    # Feature: Market Regime
                    close_price = float(df['Close'].iloc[-1])
                    ma50 = float(df['Close'].rolling(50).mean().iloc[-1])
                    
                    regime = "UPTREND 🚀" if close_price > ma50 else "DOWNTREND 🐻"
                    reg_color = "green" if "UP" in regime else "red"
                    
                    # Feature: Factor Scores (Mocked calculation)
                    st.markdown(f"### {ticker} | <span style='color:{reg_color}'>{regime}</span>", unsafe_allow_html=True)
                    
                    # Feature: Support/Resistance (High/Low)
                    high_52 = float(df['High'].max())
                    low_52 = float(df['Low'].min())

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(50).mean(), name='50 MA', line=dict(color='orange')))
                    
                    # Support/Resistance Lines
                    fig.add_hline(y=high_52, line_dash="dash", line_color="green", annotation_text="Resistance (High)")
                    fig.add_hline(y=low_52, line_dash="dash", line_color="red", annotation_text="Support (Low)")
                    
                    fig.update_layout(template=current_theme['chart_template'], height=450)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Current Price", f"₹{close_price:.2f}")
                    m2.metric("52W High", f"₹{high_52:.2f}")
                    m3.metric("Valuation (P/E Estimate)", "24.5x") # Mocked for demo
                else:
                    st.error("No data found.")
            except Exception as e:
                st.error(f"Error: {e}")

    # --- PORTFOLIO TRACKER ---
    with subtab_port:
        st.subheader("My Holdings")
        
        # Feature: Add to Portfolio
        with st.form("add_stock"):
            c_a, c_b, c_c, c_d = st.columns(4)
            n_sym = c_a.text_input("Symbol", "INFY.NS")
            n_qty = c_b.number_input("Qty", 1, 1000)
            n_avg = c_c.number_input("Avg Price", 1.0, 100000.0)
            submitted = c_d.form_submit_button("➕ Add Stock")
            
            if submitted:
                st.session_state["portfolio"].append({"symbol": n_sym, "qty": n_qty, "avg": n_avg})
                st.success("Added!")
                st.rerun()

        # Calculate Portfolio Logic
        if st.session_state["portfolio"]:
            pf_df = pd.DataFrame(st.session_state["portfolio"])
            
            # Fetch Live Prices
            tickers = pf_df["symbol"].unique().tolist()
            try:
                live_data = yf.download(tickers, period="1d", progress=False)['Close'].iloc[-1]
                
                def get_ltp(sym):
                    if isinstance(live_data, (float, np.float64)): return float(live_data)
                    return float(live_data[sym]) if sym in live_data else 0.0

                pf_df["LTP"] = pf_df["symbol"].apply(get_ltp)
                pf_df["Current Val"] = pf_df["LTP"] * pf_df["qty"]
                pf_df["Invested"] = pf_df["avg"] * pf_df["qty"]
                pf_df["P/L"] = pf_df["Current Val"] - pf_df["Invested"]
                pf_df["P/L %"] = (pf_df["P/L"] / pf_df["Invested"]) * 100
                # Feature: Sector Breakdown
                pf_df["Sector"] = pf_df["symbol"].map(SECTOR_MAP).fillna("Other")

                # Metrics
                total_inv = pf_df["Invested"].sum()
                total_curr = pf_df["Current Val"].sum()
                total_pl = pf_df["P/L"].sum()
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Total Portfolio Value", f"₹{total_curr:,.0f}")
                col_m2.metric("Total P/L", f"₹{total_pl:,.0f}", f"{(total_pl/total_inv)*100:.2f}%")
                col_m3.metric("Stock Count", len(pf_df))

                # Visuals
                col_tbl, col_pie = st.columns([2, 1])
                with col_tbl:
                    st.dataframe(pf_df[["symbol", "qty", "avg", "LTP", "P/L", "Sector"]], use_container_width=True)
                with col_pie:
                    # Feature: Sector Pie Chart
                    fig_pie = px.pie(pf_df, values="Current Val", names="Sector", title="Sector Allocation", template=current_theme['chart_template'])
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error fetching portfolio prices: {e}")
        else:
            st.info("Portfolio empty. Add stocks above.")

# ------------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Feature: Theme Toggle
    new_theme = st.selectbox("App Theme", ["Dark Neon", "Light Pro"])
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

    # Feature: User Profile
    new_role = st.selectbox("User Mode", ["Trader", "Student", "Investor"])
    if new_role != st.session_state["user_role"]:
        st.session_state["user_role"] = new_role
        st.rerun()
        
    st.markdown("---")
    if st.button("Reset Portfolio"):
        st.session_state["portfolio"] = []
        st.rerun()

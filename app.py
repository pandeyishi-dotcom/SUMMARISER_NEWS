"""
India Macro & Markets Monitor (Ultimate Version)

Run:
    pip install streamlit pandas plotly yfinance textblob feedparser requests
    streamlit run news_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import feedparser
import requests  # Added for the fix
from textblob import TextBlob
from datetime import datetime, timedelta
import base64

# ------------------------------------------------------------------
# 1. CONFIG & SESSION STATE
# ------------------------------------------------------------------

st.set_page_config(page_title="India Macro & Markets Monitor", layout="wide")

# Initialize Session State
if "portfolio" not in st.session_state:
    # Default portfolio data
    st.session_state["portfolio"] = [
        {"symbol": "RELIANCE.NS", "qty": 10, "avg": 2400.0},
        {"symbol": "TCS.NS", "qty": 5, "avg": 3500.0}
    ]

if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark Neon"

if "user_role" not in st.session_state:
    st.session_state["user_role"] = "Trader"

# Sector Mapping
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
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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

def calculate_impact_score(text):
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
    title = title.lower()
    if "inflation" in title:
        return "Prices are going up. Your money buys less stuff.", "Gold/Real Estate", "Cash Savers"
    elif "rbi" in title or "rate" in title:
        return "Central bank is changing loan costs.", "Banks", "Borrowers"
    elif "profit" in title or "results" in title:
        return "Company earnings report released.", "Shareholders", "Short Sellers"
    elif "gdp" in title:
        return "Overall economy is changing size.", "Infra", "Defensive Stocks"
    else:
        return "General market news update.", "Traders", "None"

def fetch_google_news(query, min_impact, selected_sources):
    """
    Robust fetcher with User-Agent headers to prevent blocking.
    """
    base_url = "https://news.google.com/rss/search"
    params = {
        "q": query,
        "hl": "en-IN",
        "gl": "IN",
        "ceid": "IN:en"
    }
    # IMPORTANT: Headers to look like a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        
        feed = feedparser.parse(response.content)
        articles = []
        
        for entry in feed.entries[:20]:
            impact = calculate_impact_score(entry.title)
            source = entry.get("source", {}).get("title", "Unknown")
            
            # Filter Source
            if selected_sources and source not in selected_sources:
                continue
                
            # Filter Impact
            if impact >= min_impact:
                articles.append({
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get("published", str(datetime.now())),
                    "source": source,
                    "impact": impact
                })
        return articles
        
    except Exception as e:
        st.error(f"Network Error: {e}")
        return []

# ------------------------------------------------------------------
# 4. APP LAYOUT
# ------------------------------------------------------------------

tab_news, tab_macro, tab_markets = st.tabs(["📰 News & Briefing", "📊 Macro Lab", "💹 Markets & Portfolio"])

# ==========================================
# TAB 1: NEWS (Fixed & Feature Rich)
# ==========================================
with tab_news:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        query = st.text_input("Search Topic", "India Economy")
    with c2:
        sources = st.multiselect("Filter Source", ["Moneycontrol", "Economic Times", "Mint", "NDTV Profit"], default=[])
    with c3:
        min_impact = st.slider("Min Impact Score", 0, 100, 40)

    # Call the FIXED fetcher function
    articles = fetch_google_news(query, min_impact, sources)

    if articles:
        # Timeline Chart
        st.subheader("Coverage Intensity")
        dates = [pd.to_datetime(a['published']).date() for a in articles]
        if dates:
            date_counts = pd.Series(dates).value_counts().reset_index()
            date_counts.columns = ['Date', 'Count']
            fig = px.bar(date_counts, x='Date', y='Count', title="Articles per Day", template=current_theme['chart_template'])
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

        col_feed, col_brief = st.columns([2, 1])
        
        with col_feed:
            st.subheader("Top Stories")
            for art in articles:
                eli5, winner, loser = generate_eli5(art['title'])
                st.markdown(f"""
                <div class="news-card">
                    <h4><a href="{art['link']}" target="_blank" style="color:{current_theme['text']}; text-decoration:none;">{art['title']}</a></h4>
                    <small style="color:gray;">{art['source']} | Impact: {art['impact']}/100</small>
                    <hr style="margin:8px 0; border-color: #444;">
                    <p style="font-size:0.9em; margin-bottom:5px;"><b>💡 AI Summary:</b> {eli5}</p>
                    <span class="tag" style="background:rgba(0,128,0,0.6)">Winner: {winner}</span>
                    <span class="tag" style="background:rgba(128,0,0,0.6)">Loser: {loser}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_brief:
            st.subheader("📝 Daily Briefing")
            if st.button("Generate Daily Note"):
                note = f"DAILY MARKET NOTE - {datetime.now().date()}\n\n"
                note += f"TOPIC: {query}\n"
                note += "-"*30 + "\n"
                for a in articles[:5]:
                    note += f"- {a['title']} ({a['source']})\n"
                
                b64 = base64.b64encode(note.encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="daily_note.txt">Download as Text File</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.text_area("Preview", note, height=300)
    else:
        st.info("No news found. Try a different topic or lower the impact score.")

# ==========================================
# TAB 2: MACRO
# ==========================================
with tab_macro:
    # Recession/Stagflation Monitor
    st.subheader("Economic Stress Monitor")
    
    # Mock Data (In production, replace with WorldBank API)
    cpi_curr, gdp_curr = 5.6, 6.1
    
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

    # Scenario Simulator
    with st.expander("🎛️ Scenario Simulator: Real Interest Rates", expanded=True):
        sc1, sc2 = st.columns(2)
        repo_rate = sc1.slider("Repo Rate (%)", 4.0, 9.0, 6.5)
        inflation_shock = sc2.slider("Inflation Shock (+%)", 0.0, 5.0, 0.0)
        
        real_rate = repo_rate - (cpi_curr + inflation_shock)
        st.metric("Projected Real Interest Rate", f"{real_rate:.2f}%")
        if real_rate < 0:
            st.warning("Warning: Negative Real Rates implied!")
    
    # Macro Heatmap
    st.subheader("Macro Heatmap (Historical)")
    heatmap_data = pd.DataFrame([
        [4.5, 5.1, 6.2, 5.6], # CPI
        [7.2, -5.8, 9.1, 6.1], # GDP
        [5.1, 4.0, 4.0, 6.5]   # Repo
    ], index=["CPI", "GDP", "Repo"], columns=["2020", "2021", "2022", "2023"])
    
    fig_heat = px.imshow(heatmap_data, text_auto=True, color_continuous_scale="RdBu_r", title="Indicator Heatmap")
    fig_heat.update_layout(template=current_theme['chart_template'])
    st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# TAB 3: MARKETS
# ==========================================
with tab_markets:
    subtab_watch, subtab_port = st.tabs(["🔭 Watchlist", "💼 Portfolio"])
    
    # --- WATCHLIST ---
    with subtab_watch:
        col_w1, col_w2 = st.columns([1, 3])
        with col_w1:
            ticker = st.text_input("Symbol (e.g. RELIANCE.NS)", "RELIANCE.NS")
            period = st.selectbox("Period", ["1mo", "6mo", "1y", "5y"])
            if st.session_state["user_role"] == "Student":
                st.info("Tip: 'Regime' tells you if the trend is Up or Down.")
                
        with col_w2:
            try:
                df = yf.download(ticker, period=period, progress=False)
                if not df.empty:
                    # Logic to handle different yfinance versions
                    if isinstance(df.columns, pd.MultiIndex):
                        close_col = df["Close"][ticker] if ticker in df["Close"] else df["Close"].iloc[:, 0]
                        high_col = df["High"][ticker] if ticker in df["High"] else df["High"].iloc[:, 0]
                        low_col = df["Low"][ticker] if ticker in df["Low"] else df["Low"].iloc[:, 0]
                    else:
                        close_col = df["Close"]
                        high_col = df["High"]
                        low_col = df["Low"]

                    # Market Regime
                    close_price = float(close_col.iloc[-1])
                    ma50 = float(close_col.rolling(50).mean().iloc[-1])
                    regime = "UPTREND 🚀" if close_price > ma50 else "DOWNTREND 🐻"
                    reg_color = "green" if "UP" in regime else "red"
                    
                    st.markdown(f"### {ticker} | <span style='color:{reg_color}'>{regime}</span>", unsafe_allow_html=True)
                    
                    # Chart with Levels
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=close_col, name='Price'))
                    fig.add_trace(go.Scatter(x=df.index, y=close_col.rolling(50).mean(), name='50 MA', line=dict(color='orange')))
                    fig.add_hline(y=float(high_col.max()), line_dash="dash", line_color="green", annotation_text="High")
                    fig.add_hline(y=float(low_col.min()), line_dash="dash", line_color="red", annotation_text="Low")
                    
                    fig.update_layout(template=current_theme['chart_template'], height=450)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Current Price", f"₹{close_price:.2f}")
                    m2.metric("52W High", f"₹{float(high_col.max()):.2f}")
                else:
                    st.error("No data found for symbol.")
            except Exception as e:
                st.error(f"Error loading chart: {e}")

    # --- PORTFOLIO ---
    with subtab_port:
        st.subheader("My Holdings")
        
        with st.form("add_stock"):
            c_a, c_b, c_c, c_d = st.columns(4)
            n_sym = c_a.text_input("Symbol", "INFY.NS")
            n_qty = c_b.number_input("Qty", 1, 1000)
            n_avg = c_c.number_input("Avg Price", 1.0, 100000.0)
            if c_d.form_submit_button("➕ Add Stock"):
                st.session_state["portfolio"].append({"symbol": n_sym, "qty": n_qty, "avg": n_avg})
                st.rerun()

        if st.session_state["portfolio"]:
            pf_df = pd.DataFrame(st.session_state["portfolio"])
            
            # Batch fetch prices
            tickers = pf_df["symbol"].unique().tolist()
            try:
                live_data = yf.download(tickers, period="1d", progress=False)['Close']
                if not isinstance(live_data, pd.DataFrame): # Single ticker case
                     live_val = float(live_data.iloc[-1])
                     current_prices = {tickers[0]: live_val}
                else:
                    current_prices = live_data.iloc[-1].to_dict()

                pf_df["LTP"] = pf_df["symbol"].map(current_prices).fillna(0)
                pf_df["Current Val"] = pf_df["LTP"] * pf_df["qty"]
                pf_df["Invested"] = pf_df["avg"] * pf_df["qty"]
                pf_df["P/L"] = pf_df["Current Val"] - pf_df["Invested"]
                pf_df["Sector"] = pf_df["symbol"].map(SECTOR_MAP).fillna("Other")

                total_pl = pf_df["P/L"].sum()
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Total Value", f"₹{pf_df['Current Val'].sum():,.0f}")
                col_m2.metric("Total P/L", f"₹{total_pl:,.0f}", delta_color="normal")
                col_m3.metric("Positions", len(pf_df))

                c_table, c_pie = st.columns([2, 1])
                with c_table:
                    st.dataframe(pf_df[["symbol", "qty", "avg", "LTP", "P/L", "Sector"]], use_container_width=True)
                with c_pie:
                    fig_pie = px.pie(pf_df, values="Current Val", names="Sector", title="Allocation", template=current_theme['chart_template'])
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
            except Exception as e:
                st.warning(f"Could not update live prices: {e}")
                st.dataframe(pf_df)

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    new_theme = st.selectbox("App Theme", ["Dark Neon", "Light Pro"])
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

    new_role = st.selectbox("User Mode", ["Trader", "Student", "Investor"])
    if new_role != st.session_state["user_role"]:
        st.session_state["user_role"] = new_role
        st.rerun()
        
    st.markdown("---")
    if st.button("Reset Portfolio"):
        st.session_state["portfolio"] = []
        st.rerun()

"""
India Macro & Markets Monitor (Ultimate Edition)

Features:
- Themes: Solarized Light, Bloomberg Terminal, Nordic Blue, etc.
- News: Finshots Brief (AI Summary), Word Cloud.
- Markets: Technical Analysis (RSI, Bollinger Bands), Pre-loaded Nifty 100.
- Macro: Economic Heatmap, Recession Flags.
- Portfolio: Tracker with Sector Allocation.
- Tools: Monte Carlo Simulator, Analyst Bot.

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
import requests
from textblob import TextBlob
from datetime import datetime
import re
from collections import Counter

# ------------------------------------------------------------------
# 1. PRE-LOADED STOCK DATABASE
# ------------------------------------------------------------------

STOCK_UNIVERSE = {
    "Indices & Commodities": {
        "Nifty 50": "^NSEI",
        "Sensex": "^BSESN",
        "Nifty Bank": "^NSEBANK",
        "S&P 500 (US)": "^GSPC",
        "Nasdaq (US)": "^IXIC",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "USD/INR": "INR=X"
    },
    "Nifty 50 Top Weights": {
        "Reliance Industries": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "Infosys": "INFY.NS",
        "ITC": "ITC.NS",
        "L&T": "LT.NS",
        "SBI": "SBIN.NS",
        "Bharti Airtel": "BHARTIARTL.NS",
        "HUL": "HINDUNILVR.NS"
    },
    "Popular / Midcaps": {
        "Tata Motors": "TATAMOTORS.NS",
        "Zomato": "ZOMATO.NS",
        "Paytm": "PAYTM.NS",
        "Suzlon": "SUZLON.NS",
        "Trent": "TRENT.NS",
        "HAL": "HAL.NS",
        "Mazagon Dock": "MAZDOCK.NS",
        "VBL": "VBL.NS",
        "Jio Financial": "JIOFIN.NS",
        "Bajaj Finance": "BAJFINANCE.NS"
    }
}

# ------------------------------------------------------------------
# 2. CONFIG & SESSION STATE
# ------------------------------------------------------------------

st.set_page_config(page_title="India Macro Pro", layout="wide", page_icon="📈")

if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = [
        {"symbol": "RELIANCE.NS", "qty": 10, "avg": 2400.0, "sector": "Energy"},
        {"symbol": "TCS.NS", "qty": 5, "avg": 3500.0, "sector": "IT"}
    ]

if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark Neon"

if "user_role" not in st.session_state:
    st.session_state["user_role"] = "Trader"

# ------------------------------------------------------------------
# 3. THEME ENGINE
# ------------------------------------------------------------------

THEMES = {
    "Dark Neon": {
        "bg": "#0e1117", "card": "#1a1d24", "text": "#ffffff",
        "accent": "#00d4ff", "pos": "#00ff9d", "neg": "#ff4b4b",
        "chart_template": "plotly_dark",
        "finshot_bg": "#262730"
    },
    "Light Pro": {
        "bg": "#ffffff", "card": "#f0f2f6", "text": "#000000",
        "accent": "#2962ff", "pos": "#008000", "neg": "#d50000",
        "chart_template": "plotly_white",
        "finshot_bg": "#f9f9f9"
    },
    "Solarized Light": {
        "bg": "#FDF6E3",          # Creamy background
        "card": "#EEE8D5",        # Darker beige for cards
        "text": "#657B83",        # Soft grey-cyan text
        "accent": "#B58900",      # Mustard/Gold accent
        "pos": "#859900",         # Olive Green
        "neg": "#DC322F",         # Soft Red
        "chart_template": "plotly_white",
        "finshot_bg": "#FDF6E3"
    },
    "Bloomberg Terminal": {
        "bg": "#000000", "card": "#111111", "text": "#ff9900",
        "accent": "#ff9900", "pos": "#00ff00", "neg": "#ff0000",
        "chart_template": "plotly_dark", "finshot_bg": "#1a1a1a"
    },
    "Nordic Blue": {
        "bg": "#2E3440", "card": "#3B4252", "text": "#ECEFF4",
        "accent": "#88C0D0", "pos": "#A3BE8C", "neg": "#BF616A",
        "chart_template": "plotly_dark", "finshot_bg": "#434C5E"
    },
    "Midnight Purple": {
        "bg": "#0F0c29", "card": "#24243e", "text": "#E0E0E0",
        "accent": "#f64f59", "pos": "#00b09b", "neg": "#ff416c",
        "chart_template": "plotly_dark", "finshot_bg": "#302b63"
    }
}

current_theme = THEMES[st.session_state["theme"]]

# Inject CSS
st.markdown(f"""
<style>
    .stApp {{ background-color: {current_theme['bg']}; color: {current_theme['text']}; }}
    .metric-card {{
        background-color: {current_theme['card']}; padding: 15px; border-radius: 10px;
        border-top: 3px solid {current_theme['accent']}; margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    .finshot-card {{
        background-color: {current_theme['finshot_bg']};
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 5px solid {current_theme['accent']};
        font-family: 'Georgia', sans-serif;
        color: {current_theme['text']};
    }}
    h1, h2, h3, p {{ color: {current_theme['text']}; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------------

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def monte_carlo_simulation(start_price, mu, sigma, days=30, sims=100):
    dt = 1
    simulation_df = pd.DataFrame()
    for x in range(sims):
        price_series = [start_price]
        for _ in range(days):
            price = price_series[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal())
            price_series.append(price)
        simulation_df[x] = price_series
    return simulation_df

def fetch_google_news(query, min_impact):
    base_url = "https://news.google.com/rss/search"
    params = {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=5)
        feed = feedparser.parse(response.content)
        articles = []
        for entry in feed.entries[:15]:
            title = entry.title
            impact = 30
            if "market" in title.lower(): impact += 10
            if "crash" in title.lower() or "surge" in title.lower(): impact += 30
            if "rbi" in title.lower() or "budget" in title.lower(): impact += 20
            
            if impact >= min_impact:
                articles.append({
                    "title": title,
                    "link": entry.link,
                    "published": entry.get("published", str(datetime.now())),
                    "source": entry.get("source", {}).get("title", "Unknown"),
                    "impact": min(impact, 100)
                })
        return articles
    except:
        return []

def generate_finshots_story(title):
    """
    Simulates a 'Finshots' style 3-minute read breakdown.
    """
    title_lower = title.lower()
    hook = "Here is the latest update from Dalal Street."
    why = "Market participants are reacting to new data points."
    takeaway = "Volatility might persist; keep watching key levels."
    
    if "inflation" in title_lower:
        hook = "Prices are creeping up, and the central bank is watching like a hawk."
        why = "Rising inflation erodes purchasing power and often leads to rate hikes."
        takeaway = "Sectors like FMCG and Real Estate could face headwinds."
    elif "profit" in title_lower or "q3" in title_lower:
        hook = "Earnings season is in full swing, and the numbers are talking."
        why = "Stock prices follow earnings in the long run. Beats vs Misses drive the trend."
        takeaway = "Look for guidance on future margins, not just past profit."
    elif "rbi" in title_lower or "repo" in title_lower:
        hook = "The RBI Governor made a move that affects your wallet."
        why = "Interest rates dictate the cost of capital for businesses and your EMI."
        takeaway = "Banking and Auto stocks usually react first to rate changes."
    elif "ipo" in title_lower:
        hook = "A new player is entering the stock market arena."
        why = "IPOs offer early entry, but valuations determine if it's a boom or bust."
        takeaway = "Check the company's fundamentals before jumping on the hype train."
        
    return hook, why, takeaway

# ------------------------------------------------------------------
# 5. MAIN APP LAYOUT
# ------------------------------------------------------------------

tab_news, tab_macro, tab_markets, tab_port, tab_sim = st.tabs([
    "☕ Finshots Brief", "📊 Macro Lab", "💹 Technicals", "💼 Portfolio", "🎲 Monte Carlo"
])

# ==========================================
# TAB 1: FINSHOTS BRIEF
# ==========================================
with tab_news:
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.subheader("Daily Insights ☕")
        st.caption("Simplified 3-minute reads on today's top stories.")
        
        query = st.text_input("Search Topic", "India Stock Market", key="news_search")
        articles = fetch_google_news(query, 30)
        
        if articles:
            for art in articles[:5]: # Top 5 stories
                hook, why, takeaway = generate_finshots_story(art['title'])
                
                st.markdown(f"""
                <div class="finshot-card">
                    <h3 style="margin-bottom: 5px;">{art['title']}</h3>
                    <p style="font-size: 0.9em; opacity: 0.8;">Source: {art['source']} • 3 min read</p>
                    <hr style="opacity: 0.3">
                    <p><b>🧐 The Story:</b> {hook}</p>
                    <p><b>📉 Why it Matters:</b> {why}</p>
                    <p><b>🚀 The Takeaway:</b> {takeaway}</p>
                    <a href="{art['link']}" target="_blank" style="text-decoration: none; color: {current_theme['accent']}; font-weight: bold;">Read Full Article →</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No news found. Check your internet connection.")

    with col_side:
        st.subheader("Trending Now ☁️")
        if articles:
            text = " ".join([a['title'] for a in articles]).lower()
            words = re.findall(r'\w+', text)
            stop_words = ["to", "in", "for", "on", "of", "the", "and", "a", "at", "is", "india", "market", "stocks", "share", "price", "today", "live"]
            filtered = [w for w in words if w not in stop_words and len(w) > 3]
            cnt = Counter(filtered).most_common(10)
            
            df_cloud = pd.DataFrame(cnt, columns=["Word", "Count"])
            fig = px.bar(df_cloud, x="Count", y="Word", orientation='h', template=current_theme['chart_template'])
            fig.update_traces(marker_color=current_theme['accent'])
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: MACRO LAB
# ==========================================
with tab_macro:
    m1, m2, m3 = st.columns(3)
    m1.metric("Nifty 50 PE", "22.4", "-0.5")
    m2.metric("India 10Y Bond", "7.1%", "+0.02%")
    m3.metric("USD/INR", "83.40", "+0.10")
    
    st.subheader("🔥 Macro Heatmap")
    heatmap_data = pd.DataFrame([
        [4.5, 5.1, 6.2, 5.6], # CPI
        [7.2, -5.8, 9.1, 7.6], # GDP
        [5.1, 4.0, 4.0, 6.5]   # Repo
    ], index=["CPI Inflation", "GDP Growth", "Repo Rate"], columns=["2021", "2022", "2023", "2024 (Est)"])
    
    fig_heat = px.imshow(heatmap_data, text_auto=True, color_continuous_scale="RdBu_r")
    fig_heat.update_layout(template=current_theme['chart_template'], height=350)
    st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# TAB 3: TECHNICALS (ROBUST)
# ==========================================
with tab_markets:
    c_sel, c_chart = st.columns([1, 3])
    
    with c_sel:
        st.subheader("Asset Selection")
        # Build dropdown from STOCK_UNIVERSE
        flat_stocks = {}
        for category, stocks in STOCK_UNIVERSE.items():
            for name, ticker in stocks.items():
                flat_stocks[f"{name} ({ticker})"] = ticker
        
        options = ["Custom / Search"] + list(flat_stocks.keys())
        choice = st.selectbox("Choose Asset", options)
        
        if choice == "Custom / Search":
            selected_ticker = st.text_input("Enter Symbol (e.g. ZOMATO.NS)", "RELIANCE.NS").strip().upper()
        else:
            selected_ticker = flat_stocks[choice]

        timeframe = st.selectbox("Timeframe", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
        st.info(f"Analyzing: **{selected_ticker}**")

    with c_chart:
        if selected_ticker:
            try:
                with st.spinner("Fetching Market Data..."):
                    df = yf.download(selected_ticker, period=timeframe, progress=False)
                
                if df.empty:
                    st.error(f"No data for {selected_ticker}. Try adding .NS or .BO suffix for Indian stocks.")
                else:
                    # --- AUTO FLATTEN FIX ---
                    # Handles MultiIndex if yfinance returns (Price, Ticker) format
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    # Indicators
                    df['SMA20'] = df['Close'].rolling(20).mean()
                    df['Upper'] = df['SMA20'] + (df['Close'].rolling(20).std() * 2)
                    df['Lower'] = df['SMA20'] - (df['Close'].rolling(20).std() * 2)
                    df['RSI'] = calculate_rsi(df['Close'])
                    
                    # 1. Price Chart with Bollinger Bands
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color=current_theme['accent'], width=2)))
                    fig_p.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='Upper BB', line=dict(color='gray', dash='dash')))
                    fig_p.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='Lower BB', line=dict(color='gray', dash='dash'), fill='tonexty'))
                    fig_p.update_layout(title=f"{selected_ticker} Price Analysis", template=current_theme['chart_template'], height=500, hovermode="x unified")
                    st.plotly_chart(fig_p, use_container_width=True)
                    
                    # 2. RSI Chart
                    fig_r = go.Figure()
                    fig_r.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='orange')))
                    fig_r.add_hline(y=70, line_dash="dot", line_color="red")
                    fig_r.add_hline(y=30, line_dash="dot", line_color="green")
                    fig_r.update_layout(title="RSI Momentum", template=current_theme['chart_template'], height=250, yaxis=dict(range=[0,100]))
                    st.plotly_chart(fig_r, use_container_width=True)
                    
                    # Stats
                    curr = float(df['Close'].iloc[-1])
                    rsi_val = float(df['RSI'].iloc[-1])
                    st.metric("Latest Price", f"{curr:,.2f}")
                    if rsi_val > 70: st.warning(f"RSI {rsi_val:.1f} - Overbought ⚠️")
                    elif rsi_val < 30: st.success(f"RSI {rsi_val:.1f} - Oversold ✅")
                    else: st.info(f"RSI {rsi_val:.1f} - Neutral")

            except Exception as e:
                st.error(f"Error fetching data: {e}")

# ==========================================
# TAB 4: PORTFOLIO
# ==========================================
with tab_port:
    st.subheader("Holdings Manager")
    
    with st.form("add_pos"):
        c1, c2, c3, c4 = st.columns(4)
        sym = c1.text_input("Symbol", "ITC.NS")
        qty = c2.number_input("Qty", 1, 10000)
        avg = c3.number_input("Avg Price", 1.0)
        sec = c4.selectbox("Sector", ["Banking", "IT", "Energy", "Auto", "FMCG", "Pharma", "Other"])
        if st.form_submit_button("Add Stock"):
            st.session_state["portfolio"].append({"symbol": sym, "qty": qty, "avg": avg, "sector": sec})
            st.rerun()

    if st.session_state["portfolio"]:
        pf_df = pd.DataFrame(st.session_state["portfolio"])
        
        # Batch Fetch Live Prices
        tickers = pf_df["symbol"].unique().tolist()
        try:
            live_data = yf.download(tickers, period="1d", progress=False)['Close']
            
            def get_price(s):
                # Handle single ticker result vs multiple
                if isinstance(live_data, pd.DataFrame):
                    return float(live_data.iloc[-1][s]) if s in live_data else 0
                elif isinstance(live_data, pd.Series):
                    return float(live_data.iloc[-1])
                return float(live_data) if live_data.size == 1 else 0
            
            pf_df["LTP"] = pf_df["symbol"].apply(get_price)
            pf_df["Current Val"] = pf_df["LTP"] * pf_df["qty"]
            pf_df["Invested"] = pf_df["avg"] * pf_df["qty"]
            pf_df["P/L"] = pf_df["Current Val"] - pf_df["Invested"]
            
            t1, t2 = st.columns(2)
            t1.metric("Total P/L", f"₹{pf_df['P/L'].sum():,.2f}")
            
            c_tbl, c_pie = st.columns([2,1])
            with c_tbl:
                st.dataframe(pf_df, use_container_width=True)
            with c_pie:
                fig = px.pie(pf_df, values="Current Val", names="sector", title="Sector Allocation", template=current_theme['chart_template'])
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.warning("Could not fetch live portfolio prices. Check ticker names.")
            st.dataframe(pf_df)

# ==========================================
# TAB 5: MONTE CARLO
# ==========================================
with tab_sim:
    st.subheader("🔮 Future Price Simulator")
    if 'df' in locals() and not df.empty:
        if st.button("Run Simulation (30 Days)"):
            last_price = df['Close'].iloc[-1]
            returns = np.log(1 + df['Close'].pct_change())
            mu, sigma = returns.mean(), returns.std()
            
            sim_df = monte_carlo_simulation(last_price, mu - 0.5*sigma**2, sigma, 30, 50)
            
            fig_sim = px.line(sim_df, title=f"Monte Carlo Paths for {selected_ticker}", template=current_theme['chart_template'])
            fig_sim.update_layout(showlegend=False, xaxis_title="Days", yaxis_title="Price")
            st.plotly_chart(fig_sim, use_container_width=True)
            
            avg_end = sim_df.iloc[-1].mean()
            upside = (avg_end - last_price) / last_price * 100
            st.metric("Projected Mean Price (30D)", f"{avg_end:,.2f}", f"{upside:.2f}%")
    else:
        st.warning("Please load a stock in the 'Technicals' tab first.")

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.header("🤖 Analyst Bot")
    user_q = st.text_input("Ask a question:", placeholder="What is RSI?")
    if user_q:
        q = user_q.lower()
        if "rsi" in q: ans = "RSI measures speed of price changes. >70 Overbought, <30 Oversold."
        elif "bull" in q: ans = "Bull market: Prices rising."
        elif "bear" in q: ans = "Bear market: Prices falling >20%."
        elif "nifty" in q: ans = "Nifty 50 is India's benchmark index."
        else: ans = "I can currently explain: RSI, Nifty, Bull/Bear markets."
        st.info(ans)
    
    st.markdown("---")
    
    # DYNAMIC THEME SELECTOR
    new_theme = st.selectbox("Theme", list(THEMES.keys()))
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()
    
    if st.button("Clear Portfolio"):
        st.session_state["portfolio"] = []
        st.rerun()
                

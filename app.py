"""
India Macro & Markets Monitor (Pro Edition)

New Features:
1. Monte Carlo Simulator (Future Price Prediction)
2. Technical Indicators (RSI + Bollinger Bands)
3. Cross-Asset Correlation (Nifty vs Gold/USD)
4. News Topic Cloud (Visual Keyword Tracker)
5. Analyst Bot (Sidebar Assistant)

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
from datetime import datetime, timedelta
import base64
from collections import Counter
import re

# ------------------------------------------------------------------
# 1. CONFIG & SESSION STATE
# ------------------------------------------------------------------

st.set_page_config(page_title="India Macro & Markets Pro", layout="wide")

if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = [
        {"symbol": "RELIANCE.NS", "qty": 10, "avg": 2400.0},
        {"symbol": "TCS.NS", "qty": 5, "avg": 3500.0}
    ]

if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark Neon"

if "user_role" not in st.session_state:
    st.session_state["user_role"] = "Trader"

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ------------------------------------------------------------------
# 2. THEME & STYLING
# ------------------------------------------------------------------

THEMES = {
    "Dark Neon": {
        "bg": "#0e1117", "card": "#1a1d24", "text": "#ffffff",
        "accent": "#00d4ff", "pos": "#00ff9d", "neg": "#ff4b4b",
        "chart_template": "plotly_dark"
    },
    "Light Pro": {
        "bg": "#ffffff", "card": "#f0f2f6", "text": "#000000",
        "accent": "#2962ff", "pos": "#008000", "neg": "#d50000",
        "chart_template": "plotly_white"
    }
}

current_theme = THEMES[st.session_state["theme"]]

st.markdown(f"""
<style>
    .stApp {{ background-color: {current_theme['bg']}; color: {current_theme['text']}; }}
    .metric-card {{
        background-color: {current_theme['card']}; padding: 15px; border-radius: 10px;
        border-top: 3px solid {current_theme['accent']}; margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .news-card {{
        background-color: {current_theme['card']}; padding: 15px; border-radius: 8px;
        margin-bottom: 12px; border: 1px solid rgba(128,128,128,0.2);
    }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. ADVANCED ANALYTICS FUNCTIONS
# ------------------------------------------------------------------

def calculate_rsi(data, window=14):
    delta = data.diff()
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

def calculate_impact_score(text):
    text = text.lower()
    score = 30
    keywords = {"rbi": 20, "inflation": 20, "crash": 30, "surge": 15, "gdp": 20, "budget": 25, "war": 30}
    for word, val in keywords.items():
        if word in text: score += val
    return min(score, 100)

def fetch_google_news(query, min_impact):
    # Robust fetcher
    base_url = "https://news.google.com/rss/search"
    params = {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=5)
        feed = feedparser.parse(response.content)
        articles = []
        for entry in feed.entries[:20]:
            impact = calculate_impact_score(entry.title)
            if impact >= min_impact:
                articles.append({
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get("published", str(datetime.now())),
                    "source": entry.get("source", {}).get("title", "Unknown"),
                    "impact": impact
                })
        return articles
    except:
        return []

# ------------------------------------------------------------------
# 4. APP LAYOUT
# ------------------------------------------------------------------

tab_news, tab_macro, tab_markets, tab_sim = st.tabs(["📰 News Pro", "📊 Macro Lab", "💹 Technicals", "🎲 Monte Carlo"])

# ==========================================
# TAB 1: NEWS PRO
# ==========================================
with tab_news:
    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input("Topic", "India Economy")
        articles = fetch_google_news(query, 30)
        
        if articles:
            st.subheader("Latest Intelligence")
            for art in articles:
                # Simple sentiment extraction
                blob = TextBlob(art['title'])
                sent_color = current_theme['pos'] if blob.sentiment.polarity > 0 else current_theme['neg']
                
                st.markdown(f"""
                <div class="news-card" style="border-left: 4px solid {sent_color};">
                    <a href="{art['link']}" target="_blank" style="color:{current_theme['text']}; font-weight:bold; text-decoration:none;">{art['title']}</a>
                    <br><span style="font-size:0.8em; color:gray;">{art['source']} | Impact: {art['impact']}</span>
                </div>
                """, unsafe_allow_html=True)
    
    with c2:
        st.subheader("☁️ Topic Cloud")
        if articles:
            # Generate Topic Cloud Logic
            all_text = " ".join([a['title'] for a in articles]).lower()
            # Clean text
            words = re.findall(r'\w+', all_text)
            stopwords = set(["to", "in", "for", "on", "of", "the", "and", "a", "is", "at", "india", "indian", "with", "from"])
            filtered = [w for w in words if w not in stopwords and len(w) > 3]
            
            cnt = Counter(filtered).most_common(15)
            cloud_df = pd.DataFrame(cnt, columns=["Word", "Count"])
            
            fig_cloud = px.scatter(cloud_df, x="Count", y="Word", size="Count", color="Count", 
                                   title="Trending Keywords", template=current_theme['chart_template'])
            fig_cloud.update_layout(height=400)
            st.plotly_chart(fig_cloud, use_container_width=True)

# ==========================================
# TAB 2: MACRO LAB
# ==========================================
with tab_macro:
    st.subheader("🔗 Cross-Asset Correlation Matrix")
    st.caption("How does the Indian Market relate to Oil, Gold, and USD?")
    
    if st.button("Generate Correlation Matrix"):
        with st.spinner("Crunching correlation data..."):
            try:
                # Tickers: Nifty, Gold (GLD), Oil (USO), USD/INR (INR=X)
                assets = ["^NSEI", "GLD", "USO", "INR=X"]
                data = yf.download(assets, period="1y", progress=False)['Close']
                
                # Clean Columns
                data.columns = ["Gold", "USD/INR", "Nifty 50", "Oil"]
                corr_matrix = data.corr()
                
                fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1)
                fig_corr.update_layout(template=current_theme['chart_template'], height=400)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Insight
                nifty_usd_corr = corr_matrix.loc["Nifty 50", "USD/INR"]
                if nifty_usd_corr < -0.5:
                    st.info("💡 Insight: Strong Inverse relationship between Nifty and USD. Rupee weakness hurts stocks.")
                else:
                    st.info("💡 Insight: Correlation is weak currently.")
                    
            except Exception as e:
                st.error(f"Data fetch error: {e}")

# ==========================================
# TAB 3: TECHNICALS
# ==========================================
with tab_markets:
    col_t1, col_t2 = st.columns([1, 3])
    with col_t1:
        ticker = st.text_input("Symbol", "RELIANCE.NS", key="tech_ticker")
        period = st.selectbox("Timeframe", ["6mo", "1y", "2y"])
    
    with col_t2:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if not df.empty:
                # Fix MultiIndex if present
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.xs(ticker, axis=1, level=1)
                
                # --- CALCULATIONS ---
                # 1. RSI
                df['RSI'] = calculate_rsi(df['Close'])
                
                # 2. Bollinger Bands
                df['SMA20'] = df['Close'].rolling(window=20).mean()
                df['STD20'] = df['Close'].rolling(window=20).std()
                df['Upper'] = df['SMA20'] + (df['STD20'] * 2)
                df['Lower'] = df['SMA20'] - (df['STD20'] * 2)

                # --- PLOTTING ---
                # Main Price Chart with Bands
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color=current_theme['accent'])))
                fig_price.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='Upper Band', line=dict(color='gray', width=1, dash='dash')))
                fig_price.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='Lower Band', line=dict(color='gray', width=1, dash='dash'), fill='tonexty'))
                
                fig_price.update_layout(title=f"{ticker} Price + Bollinger Bands", template=current_theme['chart_template'], height=400)
                st.plotly_chart(fig_price, use_container_width=True)
                
                # RSI Chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='orange')))
                fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
                fig_rsi.update_layout(title="Relative Strength Index (RSI)", template=current_theme['chart_template'], height=200, yaxis=dict(range=[0,100]))
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                curr_rsi = df['RSI'].iloc[-1]
                if curr_rsi > 70:
                    st.warning(f"⚠️ RSI is {curr_rsi:.1f} (Overbought) - Potential Reversal Down")
                elif curr_rsi < 30:
                    st.success(f"✅ RSI is {curr_rsi:.1f} (Oversold) - Potential Reversal Up")
            else:
                st.error("No data.")
        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# TAB 4: MONTE CARLO
# ==========================================
with tab_sim:
    st.subheader("🎲 Monte Carlo Future Simulator")
    st.caption("Projects 100 possible future price paths for the next 30 days based on historical volatility.")
    
    if st.button("Run Simulation"):
        if 'df' in locals() and not df.empty:
            last_price = df['Close'].iloc[-1]
            # Calculate daily returns logic
            log_returns = np.log(1 + df['Close'].pct_change())
            mu = log_returns.mean()
            var = log_returns.var()
            drift = mu - (0.5 * var)
            stdev = log_returns.std()
            
            sim_data = monte_carlo_simulation(last_price, drift, stdev, days=30, sims=50)
            
            fig_mc = px.line(sim_data, title=f"Monte Carlo: {ticker} (Next 30 Days)", template=current_theme['chart_template'])
            fig_mc.update_layout(showlegend=False, xaxis_title="Days into Future", yaxis_title="Price")
            st.plotly_chart(fig_mc, use_container_width=True)
            
            # Stats
            end_prices = sim_data.iloc[-1]
            mean_end = end_prices.mean()
            upside = (mean_end - last_price) / last_price * 100
            
            m1, m2 = st.columns(2)
            m1.metric("Current Price", f"{last_price:.2f}")
            m2.metric("Projected Mean (30d)", f"{mean_end:.2f}", f"{upside:.2f}%")
        else:
            st.warning("Please load data in the Technicals tab first.")

# ------------------------------------------------------------------
# SIDEBAR: ANALYST BOT
# ------------------------------------------------------------------
with st.sidebar:
    st.header("🤖 Analyst Bot")
    st.caption("Ask basic questions about market concepts.")
    
    user_q = st.text_input("Ask me...", key="bot_input")
    
    if user_q:
        q = user_q.lower()
        ans = "I'm just a simple bot. I can explain concepts like RSI, GDP, or Inflation."
        
        if "rsi" in q:
            ans = "RSI (Relative Strength Index) measures momentum. Above 70 is 'Overbought' (expensive), below 30 is 'Oversold' (cheap)."
        elif "bull" in q:
            ans = "A Bull market is when prices are rising or expected to rise."
        elif "bear" in q:
            ans = "A Bear market is when prices are falling significantly (usually >20%)."
        elif "inflation" in q:
            ans = "Inflation is the rate at which prices for goods and services rise. High inflation usually leads to higher interest rates."
        elif "nifty" in q:
            ans = "Nifty 50 is the benchmark index of the National Stock Exchange of India, representing the top 50 companies."
            
        st.info(ans)

    st.markdown("---")
    # Theme Toggle
    new_theme = st.selectbox("App Theme", ["Dark Neon", "Light Pro"])
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

    st.markdown("---")
    st.caption("India Macro Monitor Pro v2.0")

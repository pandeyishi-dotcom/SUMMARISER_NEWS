"""
India Macro & Markets Monitor (Full Version)

Features:
- Dark neon dashboard for:
    1) 📰 News: Economic & policy feed
        - India / Global mode (NewsAPI or Google News RSS fallback)
        - Sentiment tags (Positive / Negative / Neutral)
        - Impact tags: Inflation, Policy, Markets, Growth, Employment, General
        - Topic chips: RBI, Inflation, Markets, Budget, Global
        - Bookmark system (session-based)
        - Keyword frequency bar chart
    2) 📊 Macro:
        - CPI / Manufacturing (IIP proxy) / GDP / Unemployment
        - Data from World Bank by default (no key) OR uploaded CSV/XLSX
        - Latest snapshot cards
        - Time-series chart
        - Change mode: Level / Change vs previous year
        - Indicator comparison: pairs normalized to 100
    3) 💹 Markets:
        - Live index snapshot (Nifty 50, Sensex, S&P 500, NASDAQ)
        - Watchlist: choose from Nifty & Sensex companies + custom tickers
        - Watchlist mini cards (1D change)
        - Correlation heatmap (6M daily returns)
        - Single-stock deep dive:
            - Range (1M–3Y), price chart
            - Moving averages
            - Risk panel: annualised volatility, max drawdown
            - Return distribution histogram

Run:
    pip install -r requirements.txt
    streamlit run news_dashboard.py
"""

from __future__ import annotations

import os
from collections import Counter
from datetime import datetime
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
#                     GLOBAL CONFIG & THEME
# ------------------------------------------------------------------

requests_cache.install_cache("macro_markets_cache", expire_after=180)

st.set_page_config(
    page_title="India Macro & Markets Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Health–dashboard style neon palette
COLORS = {
    # Background gradient
    "bg_top": "#050816",      # deep navy-purple
    "bg_bottom": "#02010A",   # almost black

    # Card surfaces
    "card": "#050816",
    "card_inner": "#0B1020",
    "border": "rgba(255, 255, 255, 0.06)",

    # Neon accents
    "accent": "#FF7A3C",      # warm orange
    "accent2": "#A855F7",     # violet / magenta

    # Text & status
    "text": "#F9FAFB",
    "muted": "#9CA3AF",
    "pos": "#22C55E",
    "neg": "#F97373",
    "neu": "#FBBF24",
}

INDEX_SYMBOLS = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
}

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()

NEWS_TTL = 120
MARKET_TTL = 30
MACRO_TTL = 1800

if "_log" not in st.session_state:
    st.session_state["_log"] = []

if "bookmarks" not in st.session_state:
    st.session_state["bookmarks"] = []

if "news_query" not in st.session_state:
    st.session_state["news_query"] = "India economy"

# --- CSS ---
st.markdown(
    f"""
<style>
/* App background: dark neon with orange / violet glow */
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(circle at 0% 0%, rgba(255,122,60,0.20), transparent 55%),
        radial-gradient(circle at 100% 0%, rgba(168,85,247,0.20), transparent 55%),
        linear-gradient(180deg, {COLORS['bg_top']} 0%, {COLORS['bg_bottom']} 100%);
    color: {COLORS['text']};
}}

h1, h2, h3, h4 {{
    color: {COLORS['text']};
    font-weight: 700;
}}

.card {{
    background: radial-gradient(circle at top left, rgba(255,255,255,0.03), transparent 55%),
                linear-gradient(145deg, #050816 0%, #0B1020 50%, #020617 100%);
    border-radius: 18px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.9rem;
    border: 1px solid {COLORS['border']};
    box-shadow:
        0 22px 55px rgba(0,0,0,0.85),
        0 0 0 1px rgba(15,23,42,0.6);
}}

.small-muted {{
    color: {COLORS['muted']};
    font-size: 0.85rem;
}}

.chip {{
    display:inline-block;
    padding:2px 10px;
    border-radius:999px;
    border:1px solid rgba(148,163,184,0.5);
    font-size:0.7rem;
    margin-right:4px;
    margin-bottom:4px;
    background:rgba(15,23,42,0.9);
}}

.sent-pill {{
    display:inline-block;
    padding:3px 12px;
    border-radius:999px;
    font-size:0.7rem;
    font-weight:600;
    color:black;
    box-shadow:0 0 16px rgba(0,0,0,0.4);
}}

.skeleton {{
    height: 16px;
    border-radius: 999px;
    background: linear-gradient(90deg, #111827, #020617, #111827);
    background-size: 200% 100%;
    animation: shimmer 1.2s infinite linear;
}}

@keyframes shimmer {{
    from {{ background-position: -200% 0; }}
    to   {{ background-position:  200% 0; }}
}}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
#                     NIFTY / SENSEX TICKERS
# ------------------------------------------------------------------

PRESET_TICKERS_FALLBACK = {
    "Reliance Industries (NIFTY, SENSEX)": "RELIANCE.NS",
    "HDFC Bank (NIFTY, SENSEX)": "HDFCBANK.NS",
    "ICICI Bank (NIFTY, SENSEX)": "ICICIBANK.NS",
    "State Bank of India (NIFTY, SENSEX)": "SBIN.NS",
    "Infosys (NIFTY)": "INFY.NS",
    "TCS (NIFTY, SENSEX)": "TCS.NS",
    "Bharti Airtel (NIFTY, SENSEX)": "BHARTIARTL.NS",
    "ITC (NIFTY, SENSEX)": "ITC.NS",
    "Larsen & Toubro (NIFTY, SENSEX)": "LT.NS",
    "Axis Bank (NIFTY, SENSEX)": "AXISBANK.NS",
    "Kotak Mahindra Bank (NIFTY, SENSEX)": "KOTAKBANK.NS",
    "Hindustan Unilever (NIFTY, SENSEX)": "HINDUNILVR.NS",
    "Maruti Suzuki (NIFTY, SENSEX)": "MARUTI.NS",
    "Mahindra & Mahindra (NIFTY, SENSEX)": "M&M.NS",
    "UltraTech Cement (NIFTY, SENSEX)": "ULTRACEMCO.NS",
    "Sun Pharma (NIFTY, SENSEX)": "SUNPHARMA.NS",
    "Power Grid (NIFTY, SENSEX)": "POWERGRID.NS",
    "NTPC (NIFTY, SENSEX)": "NTPC.NS",
    "Asian Paints (NIFTY, SENSEX)": "ASIANPAINT.NS",
    "Bajaj Finance (NIFTY, SENSEX)": "BAJFINANCE.NS",
    "Titan (NIFTY, SENSEX)": "TITAN.NS",
    "Nestle India (NIFTY, SENSEX)": "NESTLEIND.NS",
    "JSW Steel (NIFTY, SENSEX)": "JSWSTEEL.NS",
    "Tata Motors (NIFTY)": "TATAMOTORS.NS",
    "Tata Steel (NIFTY)": "TATASTEEL.NS",
    "Wipro (NIFTY)": "WIPRO.NS",
    "HCL Tech (NIFTY)": "HCLTECH.NS",
    "Adani Ports (NIFTY)": "ADANIPORTS.NS",
    "Adani Enterprises (NIFTY)": "ADANIENT.NS",
    "Coal India (NIFTY)": "COALINDIA.NS",
    "ONGC (NIFTY)": "ONGC.NS",
    "Britannia (NIFTY)": "BRITANNIA.NS",
    "SBI Life (NIFTY)": "SBILIFE.NS",
    "HDFC Life (NIFTY)": "HDFCLIFE.NS",
    "Grasim (NIFTY)": "GRASIM.NS",
    "Tech Mahindra (NIFTY)": "TECHM.NS",
    "Tata Consumer (NIFTY)": "TATACONSUM.NS",
    "Bajaj Finserv (NIFTY)": "BAJAJFINSV.NS",
}

TICKERS_CSV_PATH = "indian_index_tickers.csv"


@st.cache_data
def load_preset_tickers() -> dict:
    """
    Load full Nifty/Sensex list from CSV if available, else fallback dict.

    CSV format:
        name,symbol,index
        Reliance Industries,RELIANCE.NS,NIFTY;SENSEX
        ...
    """
    if os.path.exists(TICKERS_CSV_PATH):
        try:
            df = pd.read_csv(TICKERS_CSV_PATH)
            df = df.dropna(subset=["name", "symbol"])
            mapping: Dict[str, str] = {}
            for _, row in df.iterrows():
                label = str(row["name"]).strip()
                idx = str(row.get("index", "")).strip()
                if idx:
                    label = f"{label} ({idx})"
                mapping[label] = str(row["symbol"]).strip()
            return mapping
        except Exception as e:
            st.warning(f"Could not read {TICKERS_CSV_PATH}, using fallback list. Error: {e}")
            return PRESET_TICKERS_FALLBACK
    return PRESET_TICKERS_FALLBACK


# ------------------------------------------------------------------
#                         GENERIC HELPERS
# ------------------------------------------------------------------

def log(msg: str) -> None:
    st.session_state["_log"].append(
        f"{datetime.utcnow().isoformat(timespec='seconds')} | {msg}"
    )


def safe_json_get(url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log(f"HTTP error: {e} | {url}")
        return None


def fmt_ts(val) -> str:
    if not val:
        return ""
    try:
        return pd.to_datetime(val).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(val)


def read_table_or_text(uploaded):
    """Return DataFrame or string for CSV/XLSX/PDF; None otherwise."""
    if uploaded is None:
        return None

    name = uploaded.name.lower()

    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        if name.endswith(".xlsx"):
            return pd.read_excel(uploaded)
        if name.endswith(".pdf") and PyPDF2:
            reader = PyPDF2.PdfReader(uploaded)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        return None

    st.warning("Unsupported file format. Use CSV / XLSX / PDF.")
    return None


# ------------------------------------------------------------------
#                     SENTIMENT & IMPACT TAGS
# ------------------------------------------------------------------

def get_sentiment(text: str) -> Tuple[str, float]:
    try:
        tb = TextBlob(text or "")
        score = round(tb.sentiment.polarity, 3)
        if score >= 0.05:
            return "positive", score
        if score <= -0.05:
            return "negative", score
        return "neutral", score
    except Exception as e:
        log(f"sentiment error: {e}")
        return "neutral", 0.0


IMPACT_KEYWORDS = {
    "Inflation": ["inflation", "cpi", "price rise", "wpi"],
    "Policy": ["rbi", "repo rate", "policy", "budget", "regulation"],
    "Markets": ["sensex", "nifty", "stocks", "equity", "markets"],
    "Growth": ["gdp", "growth", "expansion", "output"],
    "Employment": ["jobs", "unemployment", "employment"],
}


def detect_impact_tags(text: str) -> List[str]:
    text_low = (text or "").lower()
    tags = []
    for label, keys in IMPACT_KEYWORDS.items():
        if any(k in text_low for k in keys):
            tags.append(label)
    return tags or ["General"]


# ------------------------------------------------------------------
#                           NEWS
# ------------------------------------------------------------------

@st.cache_data(ttl=NEWS_TTL)
def fetch_newsapi(query: str, n: int, lang: str) -> List[Dict[str, Any]] | None:
    if not NEWSAPI_KEY:
        return None
    params = {
        "q": query,
        "language": lang,
        "pageSize": n,
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY,
    }
    js = safe_json_get("https://newsapi.org/v2/everything", params)
    if not js or js.get("status") != "ok":
        return None
    out: List[Dict[str, Any]] = []
    for a in js.get("articles", [])[:n]:
        out.append(
            {
                "title": a.get("title"),
                "summary": a.get("description") or a.get("content") or "",
                "url": a.get("url"),
                "source": (a.get("source") or {}).get("name"),
                "published_raw": a.get("publishedAt"),
            }
        )
    return out


@st.cache_data(ttl=NEWS_TTL)
def fetch_google_news(query: str, n: int, country: str) -> List[Dict[str, Any]]:
    q = requests.utils.requote_uri(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-{country}&gl={country}&ceid={country}:en"
    feed = feedparser.parse(url)
    out: List[Dict[str, Any]] = []
    for entry in feed.entries[:n]:
        out.append(
            {
                "title": entry.get("title"),
                "summary": entry.get("summary") or "",
                "url": entry.get("link"),
                "source": (entry.get("source") or {}).get("title")
                if entry.get("source")
                else None,
                "published_raw": entry.get("published") or entry.get("updated") or "",
            }
        )
    return out


def _parse_utc(val):
    try:
        return pd.to_datetime(val, utc=True, errors="coerce")
    except Exception:
        return None


def load_headlines(
    query: str,
    limit: int,
    country_mode: str = "India",
) -> List[Dict[str, Any]]:
    """Get news from NewsAPI if available, else Google News RSS."""
    if country_mode == "India":
        lang = "en"
        country = "IN"
    else:
        lang = "en"
        country = "US"

    raw = fetch_newsapi(query, n=limit, lang=lang) or fetch_google_news(
        query, n=limit, country=country
    )
    if not raw:
        return []

    cleaned: List[Dict[str, Any]] = []
    for r in raw:
        ts = _parse_utc(r.get("published_raw"))
        title = r.get("title") or ""
        summary = r.get("summary") or ""
        sent_label, sent_score = get_sentiment(title + " " + summary)
        impact_tags = detect_impact_tags(title + " " + summary)

        cleaned.append(
            {
                "title": title,
                "summary": summary,
                "url": r.get("url") or "",
                "source": r.get("source"),
                "published": ts,
                "sent_label": sent_label,
                "sent_score": sent_score,
                "impact": impact_tags,
            }
        )
    return cleaned


def news_tab(headline_count: int):
    st.subheader("📰 Economic & policy news")

    # Quick topic chips
    st.markdown("**Quick topics:**", unsafe_allow_html=True)
    chip_cols = st.columns(5)
    topics = [
        ("RBI", "RBI monetary policy India"),
        ("Inflation", "India CPI inflation"),
        ("Markets", "Indian stock markets Nifty Sensex"),
        ("Budget", "India Union Budget"),
        ("Global", "global economy"),
    ]
    for (label, q), col in zip(topics, chip_cols):
        if col.button(label):
            st.session_state["news_query"] = q

    c1, c2 = st.columns([2, 1])
    with c1:
        query = st.text_input(
            "Search / topic",
            value=st.session_state["news_query"],
            key="news_query_input",
        )
        st.session_state["news_query"] = query
    with c2:
        country_mode = st.selectbox("Feed region", ["India", "Global"])

    sentiment_filter = st.radio(
        "Filter by sentiment",
        options=["All", "Positive", "Negative", "Neutral"],
        horizontal=True,
    )

    with st.spinner("Fetching headlines..."):
        headlines = load_headlines(query, headline_count, country_mode)

    if not headlines:
        st.info("No articles found. Try a broader keyword or set NEWSAPI_KEY.")
        return

    # Overall sentiment stats
    labels = [h["sent_label"] for h in headlines]
    pos_cnt = labels.count("positive")
    neg_cnt = labels.count("negative")
    neu_cnt = labels.count("neutral")

    st.markdown(
        f"""
        <div class="card" style="display:flex; gap:1.2rem; align-items:center;">
          <div>
            <div class="small-muted">Today’s sentiment mix</div>
            <div style="font-size:2.1rem; font-weight:700; color:{COLORS['accent2']};">
                {len(headlines)} articles
            </div>
          </div>
          <div>
            <div class="small-muted">Positive</div>
            <div style="color:{COLORS['pos']}; font-weight:600;">{pos_cnt}</div>
          </div>
          <div>
            <div class="small-muted">Negative</div>
            <div style="color:{COLORS['neg']}; font-weight:600;">{neg_cnt}</div>
          </div>
          <div>
            <div class="small-muted">Neutral</div>
            <div style="color:{COLORS['neu']}; font-weight:600;">{neu_cnt}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # sentiment filter
    if sentiment_filter != "All":
        label_map = {
            "Positive": "positive",
            "Negative": "negative",
            "Neutral": "neutral",
        }
        wanted = label_map[sentiment_filter]
        headlines = [h for h in headlines if h["sent_label"] == wanted]

    # trending keywords
    keywords = []
    for h in headlines:
        for w in (h["title"] or "").split():
            w = "".join(ch for ch in w.lower() if ch.isalpha())
            if len(w) > 4:
                keywords.append(w)
    common = Counter(keywords).most_common(10)

    left, right = st.columns([3, 2])

    # Right: keyword bar chart + bookmarks
    with right:
        st.markdown("#### 🔍 Frequent words")
        if common:
            kw_df = pd.DataFrame(common, columns=["word", "count"])
            fig_kw = px.bar(
                kw_df,
                x="word",
                y="count",
                title="Top headline words",
            )
            fig_kw.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.caption("–")

        st.markdown("---")
        st.markdown("#### ⭐ Bookmarks (this session)")
        if not st.session_state["bookmarks"]:
            st.caption("No bookmarked articles yet.")
        else:
            for b in st.session_state["bookmarks"]:
                st.markdown(
                    f"- [{b['title']}]({b['url']})  \n"
                    f"  <span class='small-muted'>{b.get('source','')} · {fmt_ts(b.get('published'))}</span>",
                    unsafe_allow_html=True,
                )

    with left:
        # breaking banner
        top = sorted(
            headlines,
            key=lambda x: x.get("published") or pd.Timestamp.min,
            reverse=True,
        )[0]
        st.markdown(
            f"""
            <div class="card" style="border-left:3px solid {COLORS['accent2']}; margin-bottom:0.9rem;">
              <div class="small-muted">Breaking</div>
              <a href="{top['url']}" target="_blank" style="color:{COLORS['accent2']}; text-decoration:none; font-weight:600;">
                {top['title']}
              </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for i, art in enumerate(headlines):
            label = art["sent_label"]
            score = art["sent_score"]
            if label == "positive":
                bc = COLORS["pos"]
            elif label == "negative":
                bc = COLORS["neg"]
            else:
                bc = COLORS["neu"]

            sent_html = f"<span class='sent-pill' style='background:{bc};'>{label.upper()}</span>"
            tags_html = " ".join(
                f"<span class='chip'>{t}</span>" for t in art["impact"]
            )

            st.markdown(
                f"""
                <div class="card">
                  <div style="display:flex; justify-content:space-between; gap:0.8rem;">
                    <div style="flex:1;">
                      <a href="{art['url']}" target="_blank"
                         style="color:{COLORS['text']}; font-weight:600; text-decoration:none;">
                        {art['title']}
                      </a>
                      <div class="small-muted">
                        {(art.get('source') or '')} · {fmt_ts(art.get('published'))}
                      </div>
                    </div>
                    <div style="text-align:right;">
                      {sent_html}
                      <div class="small-muted" style="margin-top:4px;">score {score:+.2f}</div>
                    </div>
                  </div>
                  <div style="margin-top:6px; font-size:0.9rem;">{art['summary']}</div>
                  <div style="margin-top:6px;">{tags_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("⭐ Bookmark", key=f"bookmark_{i}"):
                st.session_state["bookmarks"].append(
                    {
                        "title": art["title"],
                        "url": art["url"],
                        "source": art.get("source"),
                        "published": art.get("published"),
                    }
                )


# ------------------------------------------------------------------
#                       MACRO INDICATORS (WORLD BANK)
# ------------------------------------------------------------------

# World Bank India indicator codes
WB_CODES = {
    "CPI": "FP.CPI.TOTL.ZG",           # Inflation, consumer prices (annual %)
    "GDP": "NY.GDP.MKTP.KD.ZG",        # GDP growth (annual %)
    "MANUF": "NV.IND.MANF.KD.ZG",      # Manufacturing value added growth (proxy for IIP)
    "UNEMP": "SL.UEM.TOTL.ZS",         # Unemployment, total (% of labour force)
}


@st.cache_data(ttl=MACRO_TTL)
def fetch_wb_indicator(code: str, max_years: int = 60) -> pd.DataFrame:
    """Fetch a World Bank indicator for India and return df[year, value]."""
    url = f"https://api.worldbank.org/v2/country/IND/indicator/{code}"
    js = safe_json_get(url, params={"format": "json", "per_page": max_years})
    if not js or len(js) < 2 or js[1] is None:
        return pd.DataFrame(columns=["year", "value"])

    rows = []
    for item in js[1]:
        rows.append(
            {
                "year": item.get("date"),
                "value": item.get("value"),
            }
        )
    df = pd.DataFrame(rows)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year", "value"]).sort_values("year")
    return df


def latest_from_df(df: pd.DataFrame):
    """Return (latest_value, latest_year) from df[year, value]."""
    if df is None or df.empty:
        return None, None
    tmp = df.dropna(subset=["value"]).sort_values("year")
    row = tmp.iloc[-1]
    return float(row["value"]), int(row["year"])


def normalise_year_value(df: pd.DataFrame | None) -> pd.DataFrame:
    """
    For uploaded CSV/XLSX:
    - first column -> year
    - second numeric column -> value
    Returns clean df[year, value].
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["year", "value"])

    out = df.copy()
    cols = list(out.columns)

    # year column
    if "year" not in out.columns:
        year_col = cols[0]
        out.rename(columns={year_col: "year"}, inplace=True)

    # value column
    if "value" not in out.columns:
        val_col = None
        for c in cols[1:]:
            if pd.api.types.is_numeric_dtype(out[c]):
                val_col = c
                break
        if val_col is None and len(cols) > 1:
            val_col = cols[1]
        elif val_col is None:
            val_col = cols[0]
        out.rename(columns={val_col: "value"}, inplace=True)

    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["year", "value"]).sort_values("year")
    return out[["year", "value"]]


def macro_tab():
    st.subheader("📊 India macro indicators")

    # ---------- Upload fallback (optional) ----------
    with st.expander("Upload fallback files (optional)"):
        c1, c2 = st.columns(2)
        with c1:
            up_cpi = st.file_uploader("CPI (inflation)", type=["csv", "xlsx"])
            up_iip = st.file_uploader("IIP / manufacturing output", type=["csv", "xlsx"])
        with c2:
            up_gdp = st.file_uploader("GDP", type=["csv", "xlsx"])
            up_unemp = st.file_uploader("Unemployment", type=["csv", "xlsx"])

    st.caption(
        "Default data source: World Bank World Development Indicators (no API key needed). "
        "If you upload CSV/XLSX, that will override World Bank data for that indicator."
    )

    # ---------- World Bank defaults ----------
    with st.spinner("Fetching macro data from World Bank..."):
        cpi_df_wb = fetch_wb_indicator(WB_CODES["CPI"])
        gdp_df_wb = fetch_wb_indicator(WB_CODES["GDP"])
        iip_df_wb = fetch_wb_indicator(WB_CODES["MANUF"])
        unemp_df_wb = fetch_wb_indicator(WB_CODES["UNEMP"])

    def choose_df(uploaded, wb_df):
        if uploaded is None:
            return wb_df
        obj = read_table_or_text(uploaded)
        if isinstance(obj, pd.DataFrame):
            return normalise_year_value(obj)
        return wb_df

    cpi_df = choose_df(up_cpi, cpi_df_wb)
    iip_df = choose_df(up_iip, iip_df_wb)
    gdp_df = choose_df(up_gdp, gdp_df_wb)
    unemp_df = choose_df(up_unemp, unemp_df_wb)

    # ---------- Snapshot cards ----------
    cpi_val, cpi_year = latest_from_df(cpi_df)
    iip_val, iip_year = latest_from_df(iip_df)
    gdp_val, gdp_year = latest_from_df(gdp_df)
    u_val, u_year = latest_from_df(unemp_df)

    cards = st.columns(4)

    def macro_card(col, emoji, label, value, year):
        if isinstance(value, (int, float, np.floating)):
            disp = f"{value:.1f}"
        else:
            disp = "N/A"
        yr = str(year) if year is not None else "Latest"
        col.markdown(
            f"""
            <div class="card" style="text-align:center;">
              <div style="font-size:2rem;">{emoji}</div>
              <div style="font-size:1.8rem; color:{COLORS['accent']}; font-weight:700;">{disp}</div>
              <div style="font-size:0.9rem; font-weight:600;">{label}</div>
              <div class="small-muted">{yr}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    macro_card(cards[0], "📊", "CPI inflation (%)", cpi_val, cpi_year)
    macro_card(cards[1], "🏭", "Manufacturing growth (%)", iip_val, iip_year)
    macro_card(cards[2], "💹", "GDP growth (%)", gdp_val, gdp_year)
    macro_card(cards[3], "👷", "Unemployment (%)", u_val, u_year)

    st.markdown("---")

    # ---------- Detailed explorer ----------
    indicator_map = {
        "CPI": {
            "df": cpi_df,
            "desc": "Inflation, consumer prices (annual %, code FP.CPI.TOTL.ZG)",
        },
        "Manufacturing (IIP proxy)": {
            "df": iip_df,
            "desc": "Manufacturing value added, annual % growth – proxy for IIP (NV.IND.MANF.KD.ZG)",
        },
        "GDP": {
            "df": gdp_df,
            "desc": "GDP growth (annual %, constant prices – NY.GDP.MKTP.KD.ZG)",
        },
        "Unemployment": {
            "df": unemp_df,
            "desc": "Unemployment, total (% of labour force – SL.UEM.TOTL.ZS)",
        },
    }

    top_cols = st.columns([2, 2, 2])
    with top_cols[0]:
        indicator = st.selectbox("Indicator", list(indicator_map.keys()))
    with top_cols[1]:
        mode = st.selectbox(
            "Display mode",
            ["Level", "Change vs previous year"],
            index=0,
        )
    with top_cols[2]:
        compare_opt = st.selectbox(
            "Compare (normalized index)",
            ["None", "CPI vs Manufacturing", "CPI vs GDP", "Manufacturing vs GDP"],
        )

    info = indicator_map[indicator]
    df = info["df"]
    st.caption(info["desc"])

    if df is None or df.empty:
        st.info("No data available for this indicator.")
    else:
        plot_df = df.copy()
        if mode == "Level":
            ycol = "value"
            title_suffix = ""
        else:
            plot_df["change"] = plot_df["value"].diff()
            ycol = "change"
            title_suffix = " – change vs previous year"

        if plot_df.empty:
            st.info("Not enough data to plot.")
        else:
            fig = px.line(
                plot_df,
                x="year",
                y=ycol,
                markers=True,
                title=f"{indicator} {title_suffix}",
            )
            fig.update_layout(
                template="plotly_dark",
                height=420,
                xaxis_title="Year",
                yaxis_title="Value (%)",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Raw data (tail)"):
                st.dataframe(plot_df.tail(15))

    st.markdown("---")

    # ---------- Comparison chart (normalized to 100) ----------
    if compare_opt != "None":
        pairs = {
            "CPI vs Manufacturing": ("CPI", "Manufacturing (IIP proxy)"),
            "CPI vs GDP": ("CPI", "GDP"),
            "Manufacturing vs GDP": ("Manufacturing (IIP proxy)", "GDP"),
        }
        a_key, b_key = pairs[compare_opt]
        df_a = indicator_map[a_key]["df"]
        df_b = indicator_map[b_key]["df"]

        if df_a is None or df_a.empty or df_b is None or df_b.empty:
            st.info("Not enough data for comparison.")
        else:
            ja = normalise_year_value(df_a)
            jb = normalise_year_value(df_b)

            joined = ja.merge(jb, on="year", how="inner", suffixes=("_a", "_b"))
            if joined.empty:
                st.info("No overlapping years to compare.")
            else:
                base_a = joined["value_a"].iloc[0]
                base_b = joined["value_b"].iloc[0]
                joined[f"{a_key}_idx"] = joined["value_a"] / base_a * 100
                joined[f"{b_key}_idx"] = joined["value_b"] / base_b * 100

                plot_df = joined[["year", f"{a_key}_idx", f"{b_key}_idx"]]

                fig_cmp = px.line(
                    plot_df,
                    x="year",
                    y=[f"{a_key}_idx", f"{b_key}_idx"],
                    title=f"{compare_opt} (index, base year = 100)",
                )
                fig_cmp.update_layout(
                    template="plotly_dark",
                    height=420,
                    xaxis_title="Year",
                    yaxis_title="Index (100 = first year)",
                )
                st.plotly_chart(fig_cmp, use_container_width=True)


# ------------------------------------------------------------------
#                          MARKETS
# ------------------------------------------------------------------

@st.cache_data(ttl=MARKET_TTL)
def fetch_index_snapshot() -> Dict[str, Dict[str, float | None]]:
    out: Dict[str, Dict[str, float | None]] = {}
    for name, ticker in INDEX_SYMBOLS.items():
        try:
            df = yf.download(
                ticker, period="2d", interval="5m", progress=False, threads=False
            )
            if df is None or df.empty:
                out[name] = {"last": None, "pct": None}
                continue
            df = df[~df.index.duplicated(keep="last")]
            last = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
            pct = (last - prev) / prev * 100 if prev else 0.0
            out[name] = {"last": last, "pct": pct}
        except Exception as e:
            log(f"index error {name}: {e}")
            out[name] = {"last": None, "pct": None}
    return out


def markets_header():
    st.subheader("📈 Equity indices snapshot")

    cols = st.columns(len(INDEX_SYMBOLS))
    snap = fetch_index_snapshot()
    for col, (name, _) in zip(cols, INDEX_SYMBOLS.items()):
        d = snap.get(name, {})
        price = d.get("last")
        pct = d.get("pct")

        if price is None or pct is None:
            p_txt = "N/A"
            delta_txt = ""
            clr = COLORS["muted"]
        else:
            p_txt = f"{float(price):,.2f}"
            arrow = "▲" if pct >= 0 else "▼"
            clr = COLORS["pos"] if pct >= 0 else COLORS["neg"]
            delta_txt = f"{arrow} {pct:+.2f}%"

        col.markdown(
            f"""
            <div class="card" style="text-align:center;">
              <div class="small-muted">{name}</div>
              <div style="font-size:1.4rem; font-weight:700;">{p_txt}</div>
              <div style="font-size:0.85rem; font-weight:600; color:{clr}; margin-top:2px;">
                {delta_txt}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def stock_tab(default_symbol: str):
    markets_header()
    st.markdown("---")

    st.subheader("💹 Watchlist & single-stock view")

    preset_map = load_preset_tickers()

    # ------------- Nifty/Sensex picker + extra tickers -------------
    c1, c2 = st.columns([2, 1])
    with c1:
        selected_labels = st.multiselect(
            "Select from Nifty & Sensex companies",
            options=sorted(preset_map.keys()),
            default=[
                "Reliance Industries (NIFTY, SENSEX)",
                "HDFC Bank (NIFTY, SENSEX)",
                "ICICI Bank (NIFTY, SENSEX)",
            ],
        )
        preset_symbols = [preset_map[l] for l in selected_labels] if selected_labels else []
    with c2:
        extra_raw = st.text_input(
            "Any extra tickers? (comma-separated, e.g. INFY.NS, AAPL, TSLA)",
            value=default_symbol,
        )
        extra_syms = [s.strip() for s in extra_raw.split(",") if s.strip()]

    watchlist_symbols = sorted(set(preset_symbols + extra_syms))

    if not watchlist_symbols:
        st.info("Select at least one company from the list or add a manual ticker above.")
        return

    # ------------- Mini watchlist cards (1D change) -------------
    st.markdown("#### 🔭 Watchlist snapshot")
    cols = st.columns(min(len(watchlist_symbols), 4))

    for i, sym in enumerate(watchlist_symbols):
        col = cols[i % len(cols)]
        with col:
            try:
                df = yf.download(sym, period="5d", interval="1d", progress=False)
                if df.empty:
                    st.markdown(
                        f"<div class='card'><div class='small-muted'>{sym}</div><div>no data</div></div>",
                        unsafe_allow_html=True,
                    )
                    continue
                df = df.tail(5)
                last = df["Close"].iloc[-1]
                prev = df["Close"].iloc[-2] if len(df) > 1 else last
                pct = (last - prev) / prev * 100 if prev else 0.0
                clr = COLORS["pos"] if pct >= 0 else COLORS["neg"]
                st.markdown(
                    f"""
                    <div class='card'>
                      <div class='small-muted'>{sym}</div>
                      <div style='font-size:1.3rem; font-weight:600;'>{last:,.2f}</div>
                      <div style='color:{clr}; font-size:0.8rem; font-weight:600;'>{pct:+.2f}% (1D)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                log(f"watchlist error {sym}: {e}")

    st.markdown("---")

    # ------------- Correlation heatmap on watchlist -------------
    if len(watchlist_symbols) >= 2:
        st.markdown("#### 🔗 Correlation heatmap (daily returns)")
        try:
            price_df = yf.download(
                watchlist_symbols,
                period="6mo",
                interval="1d",
                progress=False,
            )["Close"]
            price_df = price_df.dropna(how="all")
            returns = price_df.pct_change().dropna()
            corr = returns.corr()

            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation of daily returns (6M)",
            )
            fig_corr.update_layout(template="plotly_dark", height=420)
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute correlation heatmap: {e}")

    st.markdown("---")

    # ------------- Single stock deep dive (use one from watchlist) -------------
    st.subheader("🔍 Single-stock analysis")

    default_choice = watchlist_symbols[0] if watchlist_symbols else default_symbol
    symbol = st.selectbox(
        "Choose symbol for detailed view",
        options=watchlist_symbols,
        index=watchlist_symbols.index(default_choice) if default_choice in watchlist_symbols else 0,
    )

    range_map = {
        "1M": ("1mo", "1d"),
        "3M": ("3mo", "1d"),
        "6M": ("6mo", "1d"),
        "1Y": ("1y", "1d"),
        "3Y": ("3y", "1wk"),
    }

    if "stock_range" not in st.session_state:
        st.session_state["stock_range"] = "6M"

    st.caption("Time range:")
    tabs = st.tabs(list(range_map.keys()))
    for label, tab in zip(range_map.keys(), tabs):
        with tab:
            if st.button(label, key=f"range_{label}"):
                st.session_state["stock_range"] = label
                st.experimental_rerun()

    selected = st.session_state["stock_range"]
    period, interval = range_map[selected]

    with st.spinner(f"Fetching {symbol}..."):
        data = yf.download(symbol, period=period, interval=interval, progress=False)

    if data.empty:
        st.error("No price data found. Try another symbol or add `.NS` for Indian stocks.")
        return

    data = data.reset_index()
    last = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else last

    price = float(last["Close"])
    prev_price = float(prev["Close"])
    diff = price - prev_price
    pct = (diff / prev_price) * 100 if prev_price else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last", f"{price:,.2f}", f"{diff:+.2f}")
    c2.metric("Change %", f"{pct:+.2f}%")
    c3.metric("Open", f"{float(last['Open']):,.2f}")
    c4.metric("High", f"{float(last['High']):,.2f}")
    c5.metric("Low", f"{float(last['Low']):,.2f}")

    st.caption(f"Last bar: {last['Date']} | Range: {selected}")

    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Close"))
    fig.update_layout(
        title=f"{symbol} – price ({selected})",
        template="plotly_dark",
        height=420,
        xaxis_title="Date",
        yaxis_title="Price",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Moving averages + basic risk metrics
    st.markdown("### Moving averages & risk")
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()

    show20 = st.checkbox("Show 20-period MA", value=True)
    show50 = st.checkbox("Show 50-period MA", value=False)

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Close"))
    if show20:
        fig_ma.add_trace(go.Scatter(x=data["Date"], y=data["MA20"], mode="lines", name="MA20"))
    if show50:
        fig_ma.add_trace(go.Scatter(x=data["Date"], y=data["MA50"], mode="lines", name="MA50"))
    fig_ma.update_layout(
        title=f"{symbol} – moving averages",
        template="plotly_dark",
        height=380,
    )
    st.plotly_chart(fig_ma, use_container_width=True)

    # Risk: daily volatility + max drawdown + return histogram
    st.markdown("### Risk profile (based on selected range)")
    data["ret"] = data["Close"].pct_change()
    ret = data["ret"].dropna()
    if not ret.empty:
        vol = ret.std() * np.sqrt(252) * 100  # annualised approx
        cum = (1 + ret).cumprod()
        peak = cum.cummax()
        dd = (cum / peak - 1).min() * 100

        r1, r2 = st.columns(2)
        r1.metric("Annualised volatility", f"{vol:.2f}%")
        r2.metric("Max drawdown", f"{dd:.2f}%")

        fig_hist = px.histogram(
            ret * 100,
            nbins=40,
            title="Distribution of daily returns (%)",
        )
        fig_hist.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Not enough data points to compute risk metrics.")


# ------------------------------------------------------------------
#                         SIDEBAR & MAIN
# ------------------------------------------------------------------

def sidebar_controls() -> Tuple[int, str]:
    st.sidebar.title("Controls")

    # auto-refresh
    choice = st.sidebar.selectbox(
        "Auto-refresh",
        ["Off", "30s", "60s", "5m"],
        index=1,
    )
    interval_map = {"Off": 0, "30s": 30, "60s": 60, "5m": 300}
    seconds = interval_map[choice]
    if HAS_AUTOREFRESH and seconds > 0:
        tick = st_autorefresh(interval=seconds * 1000, key="auto_refresh")
        st.sidebar.caption(f"Tick: {tick}")
    elif seconds > 0:
        st.sidebar.info("Install `streamlit-autorefresh` for automatic refresh.")

    headline_count = st.sidebar.slider("News items", 4, 20, value=8)

    default_symbol = st.sidebar.text_input(
        "Default stock symbol", value="RELIANCE.NS"
    )

    if st.sidebar.button("Clear caches and rerun"):
        requests_cache.clear()
        st.cache_data.clear()
        st.experimental_rerun()

    return headline_count, default_symbol


def main():
    headline_count, default_symbol = sidebar_controls()

    st.markdown(
        f"""
        <h1>India Macro & Markets Monitor</h1>
        <div class="small-muted">
          Dark neon dashboard for economic news, macro indicators and equity markets.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    tabs = st.tabs(["📰 News", "📊 Macro", "💹 Markets"])

    with tabs[0]:
        news_tab(headline_count)

    with tabs[1]:
        macro_tab()

    with tabs[2]:
        stock_tab(default_symbol)

    st.markdown("---")
    st.markdown(
        f"<div class='small-muted'>Last update: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Debug log (developer)"):
        for line in st.session_state["_log"][-200:]:
            st.text(line)


if __name__ == "__main__":
    main()
